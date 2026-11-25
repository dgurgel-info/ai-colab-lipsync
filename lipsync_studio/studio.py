"""Fluxo completo: preparar a sessão, gerar as partes com o InfiniteTalk e montar o vídeo final.

O motor é qualquer objeto com ``ensure_running(log, progress)`` e
``generate(video, audio, output, frames, settings, on_progress, should_cancel)`` — o InfiniteTalk real
(``engine.infinitetalk``) no Colab, ou um motor falso nos testes e na prévia local.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional

from lipsync_studio import media
from lipsync_studio.engine import Cancelled, EngineError, OutOfMemoryError
from lipsync_studio.media import FPS, format_time
from lipsync_studio.session import DONE, ERROR, PENDING, RUNNING, Part, Session, Settings, list_sessions, plan_parts

MAX_INPUT_SIDE = 1080          # o modelo gera em 480p/720p; não vale a pena guardar 4K nas partes
MIN_SECONDS = 1.0

# Tamanhos de saída do InfiniteTalk (altura/largura -> [altura, largura]), copiados de
# wan/utils/multitalk_utils.py (ASPECT_RATIO_627 e ASPECT_RATIO_960). O quadro é recortado no centro.
BUCKETS = {
    "480": {0.26: (320, 1216), 0.38: (384, 1024), 0.50: (448, 896), 0.67: (512, 768), 0.82: (576, 704),
            1.00: (640, 640), 1.22: (704, 576), 1.50: (768, 512), 1.86: (832, 448), 2.00: (896, 448),
            2.50: (960, 384), 2.83: (1088, 384), 3.60: (1152, 320), 3.80: (1216, 320), 4.00: (1280, 320)},
    "720": {0.22: (448, 2048), 0.29: (512, 1792), 0.36: (576, 1600), 0.45: (640, 1408), 0.55: (704, 1280),
            0.63: (768, 1216), 0.76: (832, 1088), 0.88: (896, 1024), 1.00: (960, 960), 1.14: (1024, 896),
            1.31: (1088, 832), 1.50: (1152, 768), 1.58: (1216, 768), 1.82: (1280, 704), 1.91: (1344, 704),
            2.20: (1408, 640), 2.30: (1472, 640), 2.67: (1536, 576), 2.89: (1664, 576), 3.62: (1856, 512),
            3.75: (1920, 512)},
}


def output_size(width: int, height: int, resolution: str = "480") -> tuple:
    """(largura, altura) que o InfiniteTalk vai gerar para um quadro ``width``×``height``."""
    table = BUCKETS["720" if str(resolution).startswith("720") else "480"]
    if not width or not height:
        return 0, 0
    ratio = height / width
    h, w = table[min(table, key=lambda r: abs(r - ratio))]
    return w, h


class Context:
    """Interface mínima que o Studio usa para falar com quem o chamou (a fila de tarefas ou um teste)."""

    def log(self, message: str) -> None:
        print(message, flush=True)

    def stage(self, text: str, pct: Optional[float] = None) -> None:
        pass

    def cancelled(self) -> bool:
        return False

    def check(self) -> None:
        if self.cancelled():
            raise Cancelled("Cancelado pelo usuário.")


class Studio:
    def __init__(self, engine: Any, root: str = "."):
        self.engine = engine
        self.root = os.path.abspath(root)
        self.sessions_dir = os.path.join(self.root, "sessoes")
        self.outputs_dir = os.path.join(self.root, "outputs")
        os.makedirs(self.sessions_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    # ------------------------------------------------------------------ consultas
    def sessions(self) -> List[Session]:
        return list_sessions(self.sessions_dir)

    def load(self, session_id: str) -> Session:
        return Session.load(os.path.join(self.sessions_dir, os.path.basename(session_id)))

    def outputs(self) -> List[str]:
        files = [os.path.join(self.outputs_dir, f) for f in os.listdir(self.outputs_dir) if f.endswith(".mp4")]
        return sorted(files, key=os.path.getmtime, reverse=True)

    @staticmethod
    def target_frames(video_frames: int, audio_frames: int, settings: Settings) -> int:
        """Duração final (em quadros) conforme o que fazer quando vídeo e áudio têm tamanhos diferentes."""
        if audio_frames > video_frames:
            return video_frames if settings.audio_longer == "trim" else audio_frames
        if video_frames > audio_frames:
            return video_frames if settings.video_longer == "silence" else audio_frames
        return audio_frames

    # ------------------------------------------------------------------ 1. preparar
    def create_session(self, visual: str, audio: str, settings: Settings, ctx: Optional[Context] = None) -> Session:
        ctx = ctx or Context()
        settings = Settings.from_dict(settings.__dict__).validate()
        if not visual or not os.path.exists(visual):
            raise ValueError("Envie o vídeo (ou a foto) da pessoa.")
        if not audio or not os.path.exists(audio):
            raise ValueError("Envie o áudio com a fala.")
        kind = "image" if media.is_image(visual) else "video"
        session = Session.create(self.sessions_dir, visual, audio, settings)
        session.source_kind = kind
        ctx.log(f"🗂️ Sessão {session.id}")

        ctx.stage("Lendo o áudio...", 1)
        original_audio = session.path("src", "audio_original.wav")
        media.normalize_audio(audio, original_audio)
        audio_frames = int(round(media.probe(original_audio).duration * FPS))
        if audio_frames < MIN_SECONDS * FPS:
            raise ValueError("O áudio precisa ter pelo menos 1 segundo.")
        session.audio_frames = audio_frames

        if kind == "image":
            ctx.stage("Transformando a foto em vídeo...", 3)
            total = audio_frames
            media.image_to_video(visual, session.video_path, total, MAX_INPUT_SIDE)
            session.video_frames = total
        else:
            ctx.stage("Convertendo o vídeo para 25 fps...", 3)
            original_video = session.path("src", "video_original.mp4")
            info = media.normalize_video(visual, original_video, MAX_INPUT_SIDE)
            session.video_frames = info.frames
            total = self.target_frames(info.frames, audio_frames, settings)
            if total > info.frames:
                how = "vai e volta" if settings.audio_longer == "pingpong" else "repetindo do início"
                ctx.log(f"🔁 O áudio é {format_time((total - info.frames) / FPS)} mais longo: o vídeo segue {how}.")
            ctx.stage("Ajustando a duração do vídeo...", 5)
            media.fit_video(original_video, session.video_path, total,
                            "loop" if settings.audio_longer == "loop" else "pingpong")
            if settings.background_volume > 0 and media.probe(visual).has_audio:
                raw_bg = session.path("src", "background_raw.wav")
                media.normalize_audio(visual, raw_bg)
                media.fit_audio(raw_bg, session.background_path, total)
                os.remove(raw_bg)
                session.has_background = True

        ctx.stage("Ajustando a duração do áudio...", 6)
        media.fit_audio(original_audio, session.audio_path, total)
        media.normalize_audio(session.audio_path, session.model_audio_path, media.MODEL_SAMPLE_RATE, 1)
        info = media.probe(session.video_path)
        session.total_frames, session.width, session.height = total, info.width, info.height

        session.parts = plan_parts(total, settings.part_seconds * FPS)
        for i, part in enumerate(session.parts, start=1):
            ctx.check()
            ctx.stage(f"Separando as partes ({i}/{len(session.parts)})...", 6 + 3 * i / len(session.parts))
            os.makedirs(session.part_dir(part), exist_ok=True)
            media.cut_video(session.video_path, session.part_input_video(part), part.start_frame, part.frames)
            media.cut_audio(session.model_audio_path, session.part_input_audio(part), part.start_frame, part.frames)
        session.save()
        ctx.log(f"✂️ {len(session.parts)} parte(s) de ~{settings.part_seconds}s · duração final {format_time(session.duration)}")
        return session

    # ------------------------------------------------------------------ 2. gerar
    def engine_settings(self, session: Session, seed: Optional[int] = None, steps: Optional[int] = None,
                        audio_guide: Optional[float] = None) -> Dict[str, Any]:
        s = session.settings
        return {"prompt": s.prompt, "resolution": s.resolution, "steps": int(steps or s.steps),
                "text_guide": s.text_guide, "audio_guide": float(audio_guide or s.audio_guide),
                "seed": int(s.seed if seed is None else seed), "teacache": bool(s.teacache)}

    def generate(self, session: Session, ctx: Optional[Context] = None, indices: Optional[Iterable[int]] = None,
                 seed: Optional[int] = None, steps: Optional[int] = None,
                 audio_guide: Optional[float] = None) -> List[Part]:
        """Gera as partes pedidas (padrão: as que ainda não ficaram prontas). Devolve as que falharam."""
        ctx = ctx or Context()
        targets = [session.get_part(i) for i in indices] if indices else session.pending_parts()
        if not targets:
            ctx.log("Nenhuma parte pendente.")
            return []
        self.engine.ensure_running(ctx.log, lambda pct, text="", *a, **k: ctx.stage(text, None))
        failed: List[Part] = []
        for n, part in enumerate(targets):
            ctx.check()
            previous = part.status
            part.status = RUNNING
            session.save()
            params = self.engine_settings(session, seed, steps, audio_guide)
            ctx.log(f"🎬 {part.label()} · semente {params['seed']} · {params['steps']} passos")
            raw = os.path.join(session.part_dir(part), "raw.mp4")
            fresh = os.path.join(session.part_dir(part), "new.mp4")
            t0 = time.time()

            def on_progress(info: dict, n=n) -> None:
                frac = float(info.get("fraction") or 0.0)
                step, total = info.get("step", 0), info.get("total", 0)
                eta = ""
                if frac > 0.02:
                    remaining = (time.time() - t0) * (1 - frac) / frac
                    eta = f" · faltam ~{format_time(remaining)} nesta parte"
                detail = f" · passo {step}/{total}" if total else ""
                pct = 10 + 85 * (n + frac) / len(targets)
                ctx.stage(f"Parte {part.index}/{len(session.parts)} · {info.get('stage', '')}{detail}{eta}", pct)

            try:
                self.engine.ensure_running(ctx.log, lambda *a, **k: None)   # reinicia se o worker caiu
                result =self.engine.generate(session.part_input_video(part), session.part_input_audio(part), raw,
                                              part.frames, params, on_progress=on_progress,
                                              should_cancel=ctx.cancelled)
                media.conform_frames(raw, fresh, part.frames)
                os.replace(fresh, session.part_output(part))
                part.status, part.message = DONE, f"{result.get('width', '?')}×{result.get('height', '?')}"
                part.seed, part.steps, part.audio_guide = params["seed"], params["steps"], params["audio_guide"]
                part.seconds, part.attempts = round(time.time() - t0, 1), part.attempts + 1
                ctx.log(f"✅ Parte {part.index} pronta em {format_time(part.seconds)}")
            except Cancelled:
                part.status = DONE if previous == DONE and os.path.exists(session.part_output(part)) else PENDING
                session.save()
                raise
            except OutOfMemoryError as exc:
                part.status, part.message = ERROR, str(exc)
                session.save()
                raise
            except (EngineError, media.MediaError) as exc:
                keep = previous == DONE and os.path.exists(session.part_output(part))
                part.status = DONE if keep else ERROR
                part.message = str(exc).splitlines()[0][:300]
                failed.append(part)
                ctx.log(f"❌ Parte {part.index}: {exc}")
            finally:
                for tmp in (raw, fresh):
                    if os.path.exists(tmp):
                        os.remove(tmp)
                session.save()
        return failed

    # ------------------------------------------------------------------ 3. montar
    def assemble(self, session: Session, ctx: Optional[Context] = None) -> str:
        ctx = ctx or Context()
        if not session.ready_to_assemble():
            c = session.counts()
            raise ValueError(f"Ainda faltam partes: {c[PENDING] + c[RUNNING]} pendente(s) e {c[ERROR]} com erro. "
                             "Gere as partes que faltam na aba de revisão.")
        ctx.stage("Juntando as partes...", 96)
        joined = session.path("joined.mp4")
        media.concat_videos([session.part_output(p) for p in session.parts], joined)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(self.outputs_dir, f"{session.title}_infinitetalk_{stamp}.mp4")
        ctx.stage("Colocando o áudio e gravando o vídeo final...", 98)
        background = session.background_path if session.has_background and os.path.exists(session.background_path) else None
        media.mux(joined, session.audio_path, out, background, session.settings.background_volume)
        os.remove(joined)
        session.output = out
        session.save()
        ctx.log(f"🎉 Vídeo final: {out}")
        return out

    # ------------------------------------------------------------------ tudo de uma vez
    def run(self, visual: str, audio: str, settings: Settings, ctx: Optional[Context] = None) -> Dict[str, Any]:
        ctx = ctx or Context()
        session = self.create_session(visual, audio, settings, ctx)
        failed = self.generate(session, ctx)
        if failed:
            return {"session": session.id, "output": None,
                    "message": f"{len(failed)} parte(s) falharam. Refaça-as na aba 🧩 Revisão por partes."}
        return {"session": session.id, "output": self.assemble(session, ctx), "message": ""}
