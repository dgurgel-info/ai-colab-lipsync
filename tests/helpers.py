"""Mídia sintética para os testes (gerada com o FFmpeg, sem arquivos no repositório)."""

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lipsync_studio import media  # noqa: E402


def has_ffmpeg() -> bool:
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg  # noqa: F401
        return True
    except ImportError:
        return False


needs_ffmpeg = unittest.skipUnless(has_ffmpeg(), "FFmpeg não disponível")


def make_video(path: str, seconds: float, width: int = 320, height: int = 240, fps: int = 30,
               with_audio: bool = False) -> str:
    args = ["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate={fps}:duration={seconds}"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}", "-c:a", "aac", "-shortest"]
    media.run_ffmpeg([*args, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", path])
    return path


def make_audio(path: str, seconds: float, freq: int = 440) -> str:
    media.run_ffmpeg(["-f", "lavfi", "-i", f"sine=frequency={freq}:duration={seconds}", "-ar", "44100", path])
    return path


def make_image(path: str, width: int = 300, height: int = 400) -> str:
    media.run_ffmpeg(["-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=1", "-frames:v", "1", path])
    return path


def tmpdir(test: unittest.TestCase) -> str:
    import tempfile
    path = tempfile.mkdtemp(prefix="lipsync_test_")
    test.addCleanup(shutil.rmtree, path, True)
    return path


def file_ok(path: str) -> bool:
    return os.path.isfile(path) and os.path.getsize(path) > 0
