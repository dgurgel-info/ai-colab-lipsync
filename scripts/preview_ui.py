#!/usr/bin/env python3
"""Abre a interface do LipSync Studio no seu computador, sem GPU e sem baixar o InfiniteTalk.

Usa a interface, a fila de tarefas, as partes e a montagem REAIS; só o modelo é trocado por um motor
simulado (recorta o vídeo no tamanho de saída do InfiniteTalk e pinta um retângulo sobre a boca).

Uso:
    pip install -r requirements-dev.txt
    python scripts/preview_ui.py               # http://127.0.0.1:7860
    python scripts/preview_ui.py --delay 20    # cada parte "leva" 20 s, para ver o progresso
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def ensure_ffmpeg() -> None:
    """Garante ``ffmpeg`` no PATH (usa o binário do imageio-ffmpeg se necessário)."""
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    src = imageio_ffmpeg.get_ffmpeg_exe()
    bindir = Path(tempfile.mkdtemp(prefix="lipsync_ffbin_"))
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    shutil.copyfile(src, bindir / exe)
    os.chmod(bindir / exe, 0o755)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ["PATH"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--workdir", default=None, help="pasta de trabalho (padrão: temporária)")
    ap.add_argument("--delay", type=float, default=6.0, help="segundos simulados por parte")
    args = ap.parse_args()

    ensure_ffmpeg()
    from lipsync_studio.engine.fake import FakeEngine
    from lipsync_studio.jobs import JobManager
    from lipsync_studio.studio import Studio
    from lipsync_studio.ui.app import build_app, launch

    workdir = args.workdir or tempfile.mkdtemp(prefix="lipsync_preview_")
    studio = Studio(FakeEngine(delay=args.delay), root=workdir)
    demo, style = build_app(studio, JobManager(), gpu_label="Prévia local · motor simulado")
    print(f"📁 Pasta de trabalho: {workdir}")
    launch(demo, studio, style, share=False, server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
