"""Instala, baixa e controla o InfiniteTalk (MeiGen-AI) num venv próprio, com um worker HTTP local.

Tudo acontece no Colab, no Passo 2 do notebook:

1. Código oficial do InfiniteTalk (github.com/MeiGen-AI/InfiniteTalk) num commit fixo.
2. venv com Python 3.10 (via ``uv``) e as versões que o InfiniteTalk pede: torch 2.4.1 (CUDA 12.1),
   xformers 0.0.28, flash-attn 2.7.4.post1 e demais dependências. Nada disso entra no ambiente do notebook.
3. Pesos do Hugging Face no modo fp8 (~32 GB): Wan2.1-I2V-14B-480P (sem a DiT), InfiniteTalk fp8,
   T5 fp8 e chinese-wav2vec2-base.
4. Worker com o modelo carregado uma única vez; a interface conversa com ele por HTTP.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests

from lipsync_studio.engine import Cancelled, EngineError, OutOfMemoryError

Log = Callable[[str], None]
# progress(percentual, texto, até): a barra está em ``percentual`` e pode avançar devagar até ``até``
# enquanto a etapa durar (etapas longas sem medida real, como instalar pacotes).
Progress = Callable[..., None]

REPO_URL = "https://github.com/MeiGen-AI/InfiniteTalk.git"
REPO_COMMIT = "50aa0a94184315407a991ae804d9b58d6d311ba8"
PYTHON = "3.10"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
TORCH = ("torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1")
XFORMERS = "xformers>=0.0.28,<0.0.29"
FLASH_ATTN_WHEEL = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
                    "flash_attn-2.7.4.post1+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl")
# requirements.txt oficial com versões fixas compatíveis com torch 2.4.1 e numpy < 2.
PACKAGES = (
    "opencv-python==4.11.0.86", "diffusers==0.33.1", "transformers==4.51.3", "tokenizers>=0.20.3",
    "accelerate==1.6.0", "tqdm", "imageio", "imageio-ffmpeg", "easydict", "ftfy", "dashscope",
    "scikit-image==0.24.0", "loguru", "numpy==1.26.4", "xfuser==0.4.1", "pyloudnorm",
    "optimum-quanto==0.2.6", "scenedetect", "moviepy==1.0.3", "decord==0.6.0", "librosa==0.10.2.post1",
    "soundfile", "einops", "safetensors", "sentencepiece", "regex",
    "huggingface_hub[hf_transfer]>=0.30,<1.0",
)
CONSTRAINTS = (*TORCH, "numpy<2")
INSTALL_KEY = {"commit": REPO_COMMIT, "python": PYTHON, "torch": list(TORCH), "packages": list(PACKAGES), "v": 1}


def clean_env(extra: Optional[dict] = None) -> dict:
    """Ambiente sem variáveis que fariam o uv/Python usar o ambiente do Colab em vez do venv."""
    drop = ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME", "CONDA_PREFIX", "MPLBACKEND")
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_") and k not in drop}
    env.update(extra or {})
    return env


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def src_root() -> str:
    """Pasta que contém o pacote ``lipsync_studio`` (vai no PYTHONPATH do worker)."""
    import lipsync_studio
    return str(Path(lipsync_studio.__file__).resolve().parents[1])


def _noop(*args, **kwargs) -> None:
    pass


class InfiniteTalkEngine:
    """Ciclo de vida do InfiniteTalk: instalar, baixar, iniciar, gerar e encerrar."""

    label = "InfiniteTalk"

    def __init__(self, root: str = "/content/lipsync_engine", port: Optional[int] = None,
                 persistent: int = -1, command: Optional[Sequence[str]] = None,
                 startup_timeout: float = 3600.0):
        self.root = os.path.abspath(root)
        self.repo_dir = os.path.join(self.root, "InfiniteTalk")
        self.venv_dir = os.path.join(self.root, ".venv")
        self.weights_dir = os.path.join(self.root, "weights")
        self.hf_home = os.path.join(self.root, "hf_cache")
        self.install_log_path = os.path.join(self.root, "install.log")
        self.log_path = os.path.join(self.root, "worker.log")
        self.marker_path = os.path.join(self.root, "installed.json")
        self.port = port or free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.persistent = persistent
        self.command = list(command) if command else None          # injetável nos testes
        self.startup_timeout = startup_timeout
        self.proc: Optional[subprocess.Popen] = None
        self._logfile = None
        self.state, self.state_detail = "stopped", ""
        self.gpu: Dict[str, Any] = {}
        self._lock = threading.RLock()
        atexit.register(self.stop)

    # ------------------------------------------------------------------ utilidades
    @property
    def python(self) -> str:
        sub = "Scripts" if os.name == "nt" else "bin"
        return os.path.join(self.venv_dir, sub, "python.exe" if os.name == "nt" else "python")

    def _run(self, cmd: List[str], log: Log, env: Optional[dict] = None, cwd: Optional[str] = None,
             check: bool = True) -> int:
        """Executa mostrando a saída aos poucos (barras de download viram uma linha a cada 15 s)."""
        log("$ " + " ".join(cmd))
        os.makedirs(self.root, exist_ok=True)
        # Sessão própria: o "parar célula" do Colab não interrompe um download de vários GB no meio.
        popen_kw = {"start_new_session": True} if os.name != "nt" else {}
        proc = subprocess.Popen(cmd, env=env or clean_env(), cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, **popen_kw)
        tail: List[str] = []
        last_cr = 0.0
        with open(self.install_log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            buf = b""
            while True:
                chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    m = re.search(rb"\r\n|\r|\n", buf)
                    if not m or (m.group() == b"\r" and m.end() == len(buf)):
                        break                              # um "\r" no fim pode ser o início de "\r\n"
                    line, sep, buf = buf[:m.start()], m.group(), buf[m.end():]
                    text = line.decode("utf-8", "replace").rstrip()
                    if not text:
                        continue
                    fh.write(text + "\n")
                    tail = (tail + [text])[-40:]
                    if sep != b"\r":
                        log("   " + text[:220])
                    elif time.time() - last_cr > 15:      # barra de progresso (\r)
                        log("   " + text[:220])
                        last_cr = time.time()
            if buf.strip():
                text = buf.decode("utf-8", "replace").rstrip()
                fh.write(text + "\n")
                tail.append(text)
            code = proc.wait()
            fh.write(f"[exit {code}]\n")
        if check and code != 0:
            raise EngineError(f"Comando falhou ({code}): {' '.join(cmd)}\n" + "\n".join(tail[-15:]) +
                              f"\n(log completo: {self.install_log_path})")
        return code

    def _marker(self) -> dict:
        try:
            return json.loads(Path(self.marker_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_marker(self, **items) -> None:
        data = self._marker()
        data.update(items)
        Path(self.marker_path).write_text(json.dumps(data, indent=1), encoding="utf-8")

    # ------------------------------------------------------------------ 1. código
    def ensure_repo(self, log: Log = print) -> bool:
        if self._marker().get("repo") == REPO_COMMIT and os.path.isfile(os.path.join(self.repo_dir, "wan", "multitalk.py")):
            return False
        log(f"📥 Baixando o código oficial do InfiniteTalk ({REPO_COMMIT[:7]})...")
        shutil.rmtree(self.repo_dir, ignore_errors=True)
        os.makedirs(self.repo_dir)
        env = dict(os.environ)
        for cmd in (["git", "init", "-q"], ["git", "remote", "add", "origin", REPO_URL],
                    ["git", "fetch", "-q", "--depth", "1", "origin", REPO_COMMIT],
                    ["git", "checkout", "-q", "--detach", "FETCH_HEAD"]):
            self._run(cmd, log, env=env, cwd=self.repo_dir)
        self._save_marker(repo=REPO_COMMIT)
        return True

    # ------------------------------------------------------------------ 2. dependências
    def venv_ready(self) -> bool:
        return self._marker().get("venv") == INSTALL_KEY and os.path.exists(self.python)

    def ensure_venv(self, log: Log = print, progress: Progress = _noop) -> bool:
        if self.venv_ready():
            return False
        env = clean_env()
        if shutil.which("uv") is None:
            self._run([sys.executable, "-m", "pip", "install", "-q", "uv"], log, env=dict(os.environ))
        uv = shutil.which("uv") or os.path.join(os.path.dirname(sys.executable), "uv")
        progress(12, f"Criando o ambiente isolado (Python {PYTHON})...", 14)
        shutil.rmtree(self.venv_dir, ignore_errors=True)
        self._run([uv, "venv", self.venv_dir, "-p", PYTHON, "-q"], log, env=env)
        constraints = os.path.join(self.root, "constraints.txt")
        Path(constraints).write_text("\n".join(CONSTRAINTS) + "\n", encoding="utf-8")
        pip = [uv, "pip", "install", "--python", self.python]

        progress(14, "Instalando o PyTorch 2.4.1 (CUDA 12.1)...", 24)
        self._run([*pip, *TORCH, "--index-url", TORCH_INDEX], log, env=env)
        progress(24, "Instalando o xformers...", 28)
        self._run([*pip, XFORMERS, "--index-url", TORCH_INDEX, "-c", constraints], log, env=env)
        progress(28, "Instalando o flash-attn (wheel pronta)...", 31)
        if self._run([*pip, FLASH_ATTN_WHEEL, "-c", constraints], log, env=env, check=False) != 0:
            log("⚠️ flash-attn não instalou; o InfiniteTalk usa a atenção padrão do PyTorch (mais lento).")
        progress(31, "Instalando as demais dependências do InfiniteTalk...", 40)
        self._run([*pip, *PACKAGES, "-c", constraints], log, env=env)
        self._save_marker(venv=INSTALL_KEY)
        return True

    # ------------------------------------------------------------------ 3. pesos
    def weights_ready(self) -> bool:
        from lipsync_studio.engine.download import missing
        return not missing(self.weights_dir)

    def worker_env(self) -> dict:
        extra = {"PYTHONPATH": src_root(), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                 "HF_HOME": self.hf_home, "HF_HUB_ENABLE_HF_TRANSFER": "1", "TOKENIZERS_PARALLELISM": "false",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        return clean_env(extra)

    def ensure_weights(self, log: Log = print, progress: Progress = _noop) -> bool:
        if self.weights_ready():
            return False
        progress(40, "Baixando os pesos do Hugging Face (~32 GB)...", 70)
        self._run([self.python, "-m", "lipsync_studio.engine.download", "--weights", self.weights_dir],
                  log, env=self.worker_env())
        return True

    # ------------------------------------------------------------------ 4. worker
    def _command(self) -> List[str]:
        if self.command:
            return self.command
        return [self.python, "-u", "-m", "lipsync_studio.engine.worker", "--repo", self.repo_dir,
                "--weights", self.weights_dir, "--port", str(self.port), "--persistent", str(self.persistent)]

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def log_tail(self, n: int = 2000) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()[-n:]
        except OSError:
            return ""

    def _get(self, path: str, timeout: float = 5) -> Optional[requests.Response]:
        try:
            return requests.get(self.base + path, timeout=timeout)
        except requests.RequestException:
            return None

    def start(self, log: Log = print, progress: Progress = _noop) -> None:
        with self._lock:
            if self.alive() and self.ready():
                return
            self.stop()
            os.makedirs(self.root, exist_ok=True)
            self.state, self.state_detail = "loading", "Iniciando o worker..."
            progress(72, "Carregando o InfiniteTalk na memória (alguns minutos)...", 98)
            log(f"🚀 Iniciando o worker do InfiniteTalk na porta {self.port}...")
            self._logfile = open(self.log_path, "w", encoding="utf-8")
            popen_kw = {"start_new_session": True} if os.name != "nt" else {}
            cwd = self.repo_dir if os.path.isdir(self.repo_dir) else self.root
            self.proc = subprocess.Popen(self._command(), cwd=cwd, stdout=self._logfile, stderr=subprocess.STDOUT,
                                         env=self.worker_env(), **popen_kw)
            t0, last_stage = time.time(), ""
            while time.time() - t0 < self.startup_timeout:
                if not self.alive():
                    self.stop()
                    self.state, self.state_detail = "error", "O worker encerrou ao iniciar."
                    raise EngineError(f"O worker do InfiniteTalk encerrou ao iniciar.\n{self.log_tail()}")
                resp = self._get("/health")
                if resp is not None:
                    data = resp.json()
                    if resp.status_code == 200:
                        self.state, self.state_detail = "ready", "Pronto"
                        log(f"✅ InfiniteTalk pronto ({time.time() - t0:.0f}s).")
                        return
                    if resp.status_code == 500:
                        self.stop()
                        self.state, self.state_detail = "error", str(data.get("error", ""))[:300]
                        raise EngineError(f"Falha ao carregar o InfiniteTalk: {data.get('error')}\n{self.log_tail()}")
                    stage = data.get("stage", "")
                    if stage and stage != last_stage:
                        log(f"   {stage} ({time.time() - t0:.0f}s)")
                        self.state_detail = last_stage = stage
                time.sleep(1)
            self.stop()
            self.state, self.state_detail = "error", "Tempo esgotado ao carregar."
            raise EngineError(f"Tempo esgotado ao iniciar o InfiniteTalk.\n{self.log_tail()}")

    def ready(self) -> bool:
        resp = self._get("/health", timeout=3)
        return resp is not None and resp.status_code == 200

    def stop(self) -> None:
        """Encerra o worker (devolve VRAM e RAM). Seguro para chamar várias vezes."""
        proc, self.proc = self.proc, None
        if self._logfile is not None:
            self._logfile.close()
            self._logfile = None
        self.state = "stopped"
        if proc is None or proc.poll() is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            else:
                proc.terminate()
            proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception:
                pass

    # ------------------------------------------------------------------ tudo de uma vez (Passo 2)
    def prepare(self, log: Log = print, progress: Progress = _noop) -> None:
        """Instala o que faltar, baixa os pesos e deixa o modelo carregado."""
        try:
            self.state = "installing"
            progress(2, "Baixando o código do InfiniteTalk...", 10)
            self.ensure_repo(log)
            progress(10, "Preparando as dependências...", 12)
            self.ensure_venv(log, progress)
            self.ensure_weights(log, progress)
            self.start(log, progress)
            progress(100, "InfiniteTalk pronto.")
        except BaseException:
            if self.state != "ready":
                self.state = "error"
            raise

    def ensure_running(self, log: Log = print, progress: Progress = _noop) -> None:
        """Usado antes de cada geração: reinicia o worker se ele tiver caído."""
        if self.alive() and self.ready():
            return
        log("♻️ O worker do InfiniteTalk não está ativo; iniciando de novo...")
        if not self.command and not (self.venv_ready() and self.weights_ready()):
            self.prepare(log, progress)
        else:
            self.start(log, progress)

    # ------------------------------------------------------------------ geração
    def generate(self, video: str, audio: str, output: str, frames: int, settings: Dict[str, Any],
                 on_progress: Optional[Callable[[dict], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None, poll: float = 1.0) -> Dict[str, Any]:
        """Gera ``output`` (sem áudio) com a pessoa do ``video`` falando o ``audio``."""
        payload = {"video": os.path.abspath(video), "audio": os.path.abspath(audio),
                   "output": os.path.abspath(output), "frames": int(frames), **settings}
        box: Dict[str, Any] = {}

        def call():
            try:
                box["resp"] = requests.post(self.base + "/generate", json=payload, timeout=24 * 3600)
            except BaseException as exc:      # conexão caiu (worker morreu, por exemplo)
                box["exc"] = exc

        thread = threading.Thread(target=call, daemon=True)
        thread.start()
        cancel_sent = False
        while thread.is_alive():
            thread.join(poll)
            if should_cancel and should_cancel() and not cancel_sent:
                try:
                    requests.post(self.base + "/cancel", timeout=5)
                except requests.RequestException:
                    pass
                cancel_sent = True
            if on_progress and thread.is_alive():
                resp = self._get("/progress", timeout=3)
                if resp is not None and resp.status_code == 200:
                    on_progress(resp.json())
        if "exc" in box:
            raise EngineError(f"Conexão com o worker perdida: {box['exc']}\n{self.log_tail(1200)}")
        resp = box["resp"]
        try:
            data = resp.json()
        except ValueError:
            raise EngineError(f"Resposta inválida do worker ({resp.status_code}): {resp.text[:300]}")
        if resp.status_code == 200:
            return data
        code, message = data.get("error"), data.get("message", "")
        if code == "cancelled":
            raise Cancelled(message)
        if code == "oom":
            raise OutOfMemoryError(message)
        raise EngineError(f"{code}: {message}\n{data.get('traceback', '')}".strip())

    # ------------------------------------------------------------------ interface
    def status(self) -> Dict[str, Any]:
        """Estado para a interface: stopped | installing | loading | ready | busy | error."""
        if self.state in ("ready", "loading") and self.proc is not None and not self.alive():
            self.state = "error"
            self.state_detail = "O worker encerrou. Ele será iniciado de novo na próxima geração."
        return {"state": self.state, "detail": self.state_detail}
