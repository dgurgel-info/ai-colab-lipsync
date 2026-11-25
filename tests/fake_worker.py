"""Worker de teste: o servidor HTTP real do ``engine.worker`` com um gerador falso (sem torch).

    python -m tests.fake_worker --port 8123 [--load-delay 1] [--fail-load]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from lipsync_studio.engine import OutOfMemoryError
from lipsync_studio.engine.worker import Progress, serve


class FakeRunner:
    def __init__(self, progress: Progress, load_delay: float = 0.0, fail: bool = False):
        self.progress = progress
        progress.set("Carregando modelo falso...")
        time.sleep(load_delay)
        if fail:
            raise RuntimeError("pesos corrompidos")

    def run(self, req):
        p = self.progress
        for path in (req["video"], req["audio"]):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
        if req.get("prompt") == "oom":
            raise OutOfMemoryError("sem memória (simulado)")
        steps = int(req.get("steps", 4))
        p.reset("Gerando (falso)...", total=steps)
        for _ in range(steps):
            p.check()
            time.sleep(float(req.get("step_delay", 0.01)))
            p.advance()
        with open(req["output"], "wb") as fh:
            fh.write(b"fake-mp4")
        return {"output": req["output"], "seconds": 0.1, "seed": req.get("seed", 0), "frames": req.get("frames", 0),
                "width": 896, "height": 448}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--load-delay", type=float, default=0.0)
    ap.add_argument("--fail-load", action="store_true")
    args = ap.parse_args(argv)
    serve(args.port, lambda progress: FakeRunner(progress, args.load_delay, args.fail_load))
    return 0


if __name__ == "__main__":
    sys.exit(main())
