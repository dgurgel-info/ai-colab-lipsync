"""Worker do InfiniteTalk: roda DENTRO do venv do modelo e atende por HTTP local.

    python -m lipsync_studio.engine.worker --repo <InfiniteTalk> --weights <pesos> --port 8123

Rotas:
    GET  /health    200 quando o modelo está carregado (503 enquanto carrega, com a etapa atual)
    GET  /progress  etapa e passos da geração em curso
    POST /generate  {"video", "audio", "output", "prompt", "resolution", "steps", "text_guide",
                     "audio_guide", "seed", "teacache"}
    POST /cancel    interrompe a geração em curso no próximo passo
    POST /shutdown  encerra o worker

O modelo (Wan2.1 14B + InfiniteTalk, quantizado em fp8) é carregado uma única vez. Só biblioteca padrão
no topo: torch e o código do InfiniteTalk só são importados já dentro do venv. Compatível com Python 3.10.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from lipsync_studio.engine import Cancelled, OutOfMemoryError

FPS = 25
FRAME_NUM = 81          # quadros por janela do InfiniteTalk (4n+1)
MOTION_FRAME = 9        # quadros que ligam uma janela à próxima
MIN_AUDIO_FRAMES = FRAME_NUM + 12   # o pipeline exige áudio maior que uma janela
SHIFT = {"480": 7.0, "720": 11.0}   # valores oficiais do generate_infinitetalk.py
SIZE = {"480": "infinitetalk-480", "720": "infinitetalk-720"}
ALL_ON_GPU = 30_000_000_000         # mais que os 14B parâmetros: tudo fica na VRAM


def windows_for(frames: int) -> int:
    """Quantas janelas de 81 quadros o modo streaming gera para ``frames`` quadros de áudio."""
    if frames <= FRAME_NUM:
        return 1
    return 1 + math.ceil((frames - FRAME_NUM) / (FRAME_NUM - MOTION_FRAME))


def auto_persistent(vram_gb: float) -> int:
    """Parâmetros da DiT mantidos na GPU: tudo com ~40 GB ou mais (A100, H100); senão, modo de pouca VRAM."""
    return ALL_ON_GPU if vram_gb >= 39 else 0


class Progress:
    """Estado compartilhado entre a geração e as rotas de consulta."""

    def __init__(self):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.ready = False
        self.reset("Carregando o modelo...")

    def reset(self, stage: str = "", total: int = 0) -> None:
        with self.lock:
            self.stage, self.step, self.total, self.started = stage, 0, int(total), time.time()

    def set(self, stage: Optional[str] = None) -> None:
        with self.lock:
            if stage is not None:
                self.stage = stage

    def advance(self) -> None:
        with self.lock:
            self.step += 1

    def check(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("Cancelado pelo usuário.")

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            frac = min(0.99, self.step / self.total) if self.total else 0.0
            return {"ready": self.ready, "stage": self.stage, "step": self.step, "total": self.total,
                    "fraction": frac, "elapsed": time.time() - self.started}


def write_frames(video, output: str, ffmpeg: str = "ffmpeg", chunk: int = 32) -> int:
    """Grava o tensor C×T×H×W em [-1, 1] como MP4 H.264 (25 fps, sem áudio), aos poucos."""
    import numpy as np
    import torch

    _, total, height, width = video.shape
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
           "-crf", "12", "-pix_fmt", "yuv420p", output]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for start in range(0, total, chunk):
            part = video[:, start:start + chunk].to(torch.float32)
            frames = ((part + 1) / 2).clamp(0, 1).mul(255).round().to(torch.uint8)
            proc.stdin.write(np.ascontiguousarray(frames.permute(1, 2, 3, 0).cpu().numpy()).tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait() != 0:
        raise RuntimeError(f"FFmpeg não conseguiu gravar o vídeo gerado: {err[-600:]}")
    return total


class InfiniteTalkRunner:
    """Carrega o InfiniteTalk uma vez e gera uma parte por chamada."""

    def __init__(self, repo: str, weights: str, progress: Progress, persistent: Optional[int] = None):
        self.repo, self.weights = os.path.abspath(repo), os.path.abspath(weights)
        self.progress = progress
        self._load(persistent)

    def _load(self, persistent: Optional[int]) -> None:
        os.chdir(self.repo)
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        p = self.progress
        p.set("Importando bibliotecas (torch, xformers, quanto)...")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("GPU CUDA não encontrada. O InfiniteTalk precisa de GPU (L4, A100 ou H100).")
        import wan
        from wan import multitalk as mt
        from wan.configs import WAN_CONFIGS
        from transformers import Wav2Vec2FeatureExtractor
        from src.audio_analysis.wav2vec2 import Wav2Vec2Model

        self.torch = torch
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.persistent = auto_persistent(vram_gb) if persistent is None or persistent < 0 else int(persistent)
        mode = "tudo na GPU" if self.persistent >= ALL_ON_GPU else f"{self.persistent / 1e9:.0f}B parâmetros na GPU"
        print(f"🖥️ {torch.cuda.get_device_name(0)} · {vram_gb:.0f} GB de VRAM · DiT: {mode}", flush=True)

        wan_dir = os.path.join(self.weights, "Wan2.1-I2V-14B-480P")
        quant_file = os.path.join(self.weights, "InfiniteTalk", "quant_models", "infinitetalk_single_fp8.safetensors")
        p.set("Carregando o Wan2.1 14B + InfiniteTalk (fp8) na memória... (alguns minutos)")
        self.pipeline = wan.InfiniteTalkPipeline(
            config=WAN_CONFIGS["infinitetalk-14B"], checkpoint_dir=wan_dir, quant_dir=quant_file,
            device_id=0, rank=0, t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
            lora_dir=None, lora_scales=None, quant="fp8", dit_path=None,
            infinitetalk_dir=os.path.join(self.weights, "InfiniteTalk", "single", "infinitetalk.safetensors"),
        )
        p.set("Organizando a memória da GPU...")
        self.pipeline.vram_management = True
        self.pipeline.enable_vram_management(num_persistent_param_in_dit=self.persistent)

        p.set("Carregando o codificador de áudio (chinese-wav2vec2-base)...")
        w2v = os.path.join(self.weights, "chinese-wav2vec2-base")
        self.audio_encoder = Wav2Vec2Model.from_pretrained(w2v, local_files_only=True).to("cpu")
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(w2v, local_files_only=True)

        # Etapas visíveis e cancelamento: o laço de amostragem usa o tqdm do módulo wan.multitalk.
        def tracked(iterable, total=None, **kwargs):
            for item in iterable:
                p.check()
                p.set("Gerando o vídeo (difusão)...")
                yield item
                p.advance()

        mt.tqdm = tracked
        p.set("Modelo pronto.")

    # ------------------------------------------------------------------ áudio
    def _embedding(self, audio_path: str, emb_path: str, min_frames: int) -> int:
        """wav2vec2 -> embeddings por quadro (mesmo cálculo do generate_infinitetalk.py oficial)."""
        import librosa
        import numpy as np
        import pyloudnorm as pyln
        from einops import rearrange

        torch = self.torch
        speech, sr = librosa.load(audio_path, sr=16000)
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(speech)
        if abs(loudness) <= 100:                       # silêncio puro fica como está
            speech = pyln.normalize.loudness(speech, loudness, -23)
        need = int(math.ceil(min_frames / FPS * sr))
        if len(speech) < need:                          # janela mínima do modelo: completa com silêncio
            speech = np.concatenate([speech, np.zeros(need - len(speech), dtype=speech.dtype)])
        video_length = len(speech) / sr * FPS
        values = np.squeeze(self.feature_extractor(speech, sampling_rate=sr).input_values)
        values = torch.from_numpy(values).float().unsqueeze(0)
        with torch.no_grad():
            out = self.audio_encoder(values, seq_len=int(video_length), output_hidden_states=True)
        emb = torch.stack(out.hidden_states[1:], dim=1).squeeze(0)
        emb = rearrange(emb, "b s d -> s b d").cpu().detach()
        torch.save(emb, emb_path)
        return int(emb.shape[0])

    # ------------------------------------------------------------------ geração
    def run(self, req: Dict[str, Any]) -> Dict[str, Any]:
        torch, p = self.torch, self.progress
        video, audio, output = req["video"], req["audio"], req["output"]
        for path in (video, audio):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        resolution = "720" if str(req.get("resolution", "480")).startswith("720") else "480"
        steps = int(req.get("steps", 40))
        seed = int(req.get("seed", 42))
        frames = int(req.get("frames") or 0)

        work = os.path.join(os.path.dirname(os.path.abspath(output)), "_tmp")
        os.makedirs(work, exist_ok=True)
        p.reset("Lendo o áudio (wav2vec2)...", total=0)
        t0 = time.time()
        try:
            emb_path = os.path.join(work, "audio_emb.pt")
            emb_frames = self._embedding(audio, emb_path, MIN_AUDIO_FRAMES)
            target = frames or emb_frames
            with p.lock:
                p.total = windows_for(emb_frames) * steps
            extra = SimpleNamespace(use_teacache=bool(req.get("teacache", True)), teacache_thresh=0.2,
                                    size=SIZE[resolution], use_apg=False, apg_momentum=-0.75,
                                    apg_norm_threshold=55)
            clip = {"prompt": str(req.get("prompt") or "A person is talking."), "cond_video": video,
                    "cond_audio": {"person1": emb_path}}
            generated = self.pipeline.generate_infinitetalk(
                clip, size_buckget=SIZE[resolution], motion_frame=MOTION_FRAME, frame_num=FRAME_NUM,
                shift=SHIFT[resolution], sampling_steps=steps,
                text_guide_scale=float(req.get("text_guide", 5.0)),
                audio_guide_scale=float(req.get("audio_guide", 4.0)),
                seed=seed, offload_model=True, max_frames_num=emb_frames,
                color_correction_strength=1.0, extra_args=extra,
            )
            p.check()
            p.set("Gravando o vídeo gerado...")
            generated = generated[:, :target]
            written = write_frames(generated, output)
            height, width = int(generated.shape[2]), int(generated.shape[3])
            del generated
        except RuntimeError as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower():
                raise OutOfMemoryError("A GPU ficou sem memória. Use 480p, partes menores ou uma GPU maior.") from exc
            raise
        finally:
            shutil.rmtree(work, ignore_errors=True)
            gc.collect()
            torch.cuda.empty_cache()
        return {"output": output, "seconds": time.time() - t0, "seed": seed, "frames": written,
                "width": width, "height": height}


ERRORS = {OutOfMemoryError: (507, "oom"), Cancelled: (409, "cancelled"), FileNotFoundError: (404, "not_found")}


class Service:
    def __init__(self, progress: Progress):
        self.progress = progress
        self.runner: Any = None
        self.load_error: Optional[str] = None
        self.lock = threading.Lock()

    def generate(self, req: Dict[str, Any]) -> Dict[str, Any]:
        if self.runner is None:
            raise RuntimeError(self.load_error or "O modelo ainda está carregando.")
        with self.lock:                       # uma geração por vez
            self.progress.cancel.clear()
            try:
                return self.runner.run(req)
            finally:
                self.progress.cancel.clear()


def make_handler(service: Service, shutdown: Callable[[], None]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            snap = service.progress.snapshot()
            if self.path == "/health":
                if service.load_error:
                    self._send(500, {**snap, "ready": False, "error": service.load_error})
                else:
                    self._send(200 if service.runner is not None else 503, snap)
            elif self.path == "/progress":
                self._send(200, snap)
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0) or 0)
            body = self.rfile.read(length) if length else b"{}"
            if self.path == "/shutdown":
                self._send(200, {"ok": True})
                shutdown()
                return
            if self.path == "/cancel":
                service.progress.cancel.set()
                self._send(200, {"ok": True})
                return
            if self.path != "/generate":
                self._send(404, {"error": "not_found"})
                return
            try:
                self._send(200, service.generate(json.loads(body or b"{}")))
            except BaseException as exc:
                status, code = next(((s, c) for cls, (s, c) in ERRORS.items() if isinstance(exc, cls)),
                                    (500, type(exc).__name__))
                self._send(status, {"error": code, "message": str(exc),
                                    "traceback": "" if status != 500 else traceback.format_exc()[-3000:]})

    return Handler


def serve(port: int, build_runner: Callable[[Progress], Any], host: str = "127.0.0.1",
          block: bool = True) -> ThreadingHTTPServer:
    """Sobe o servidor já (para informar a etapa de carregamento) e carrega o modelo em seguida."""
    progress = Progress()
    service = Service(progress)

    def shutdown() -> None:
        threading.Thread(target=lambda: (time.sleep(0.2), os._exit(0)), daemon=True).start()

    server = ThreadingHTTPServer((host, port), make_handler(service, shutdown))
    server.daemon_threads = True
    server.service = service  # type: ignore[attr-defined]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"🌐 Worker ouvindo em {host}:{port}", flush=True)
    try:
        service.runner = build_runner(progress)
        progress.ready = True
        progress.reset("Modelo pronto.")
        print("✅ InfiniteTalk pronto.", flush=True)
    except BaseException as exc:
        service.load_error = f"{type(exc).__name__}: {exc}"
        print(traceback.format_exc(), flush=True)
    if block:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            server.shutdown()
    return server


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="pasta do código do InfiniteTalk")
    parser.add_argument("--weights", required=True, help="pasta dos pesos baixados")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--persistent", type=int, default=-1,
                        help="parâmetros da DiT mantidos na GPU (-1 = automático pela VRAM)")
    args = parser.parse_args(argv)
    serve(args.port, lambda progress: InfiniteTalkRunner(args.repo, args.weights, progress, args.persistent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
