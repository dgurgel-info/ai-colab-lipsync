"""Motor falso para testes e prévia local (sem GPU e sem baixar o InfiniteTalk).

Faz o que o motor real faz por fora: recorta o vídeo no tamanho de saída do InfiniteTalk, grava um quadro
colorido pela semente sobre a boca e devolve um quadro a mais ou a menos (como o modelo real pode fazer),
para que o Studio precise ajustar a contagem de quadros.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from lipsync_studio import media
from lipsync_studio.engine import Cancelled, EngineError


class FakeEngine:
    label = "InfiniteTalk (simulado)"

    def __init__(self, delay: float = 0.0, fail_parts: Optional[set] = None, frame_skew: int = -1):
        self.delay = delay
        self.fail_parts = set(fail_parts or ())
        self.frame_skew = frame_skew
        self.calls = []

    def ensure_running(self, log: Callable = print, progress: Callable = lambda *a, **k: None) -> None:
        pass

    def status(self) -> Dict[str, Any]:
        return {"state": "ready", "detail": ""}

    def generate(self, video: str, audio: str, output: str, frames: int, settings: Dict[str, Any],
                 on_progress: Optional[Callable[[dict], None]] = None,
                 should_cancel: Optional[Callable[[], bool]] = None, poll: float = 0.0) -> Dict[str, Any]:
        from lipsync_studio.studio import output_size

        self.calls.append({"video": video, "audio": audio, "frames": frames, **settings})
        if len(self.calls) in self.fail_parts:
            raise EngineError("falha simulada")
        steps = int(settings.get("steps", 20))
        for step in range(steps):
            if should_cancel and should_cancel():
                raise Cancelled("Cancelado pelo usuário.")
            if on_progress:
                on_progress({"stage": "Gerando o vídeo (simulado)...", "step": step, "total": steps,
                             "fraction": step / steps})
            if self.delay:
                time.sleep(self.delay / steps)
        info = media.probe(video)
        width, height = output_size(info.width, info.height, settings.get("resolution", "480"))
        seed = int(settings.get("seed", 0))
        color = f"0x{(seed * 2654435761) % 0xFFFFFF:06x}"
        out_frames = max(1, frames + self.frame_skew)
        vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
              f"drawbox=x=iw*0.35:y=ih*0.55:w=iw*0.3:h=ih*0.12:color={color}@0.85:t=fill,format=yuv420p")
        media.run_ffmpeg(["-i", video, "-vf", vf, "-frames:v", str(out_frames), "-an", "-c:v", "libx264",
                          "-preset", "ultrafast", "-crf", "18", output], "Motor simulado")
        return {"output": output, "seconds": 0.0, "seed": seed, "frames": out_frames,
                "width": width, "height": height}
