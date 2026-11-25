"""Motor InfiniteTalk: protocolo cliente/worker (com worker falso) e comandos de instalação."""

import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from tests.helpers import tmpdir

from lipsync_studio.engine import Cancelled, EngineError, OutOfMemoryError
from lipsync_studio.engine import download, infinitetalk
from lipsync_studio.engine.infinitetalk import InfiniteTalkEngine, free_port
from lipsync_studio.engine.worker import ALL_ON_GPU, FRAME_NUM, auto_persistent, windows_for


def fake_engine(test, *extra):
    port = free_port()
    root = tmpdir(test)
    engine = InfiniteTalkEngine(root=root, port=port, startup_timeout=60,
                                command=[sys.executable, "-m", "tests.fake_worker", "--port", str(port), *extra])
    test.addCleanup(engine.stop)
    return engine, root


class WorkerMathTests(unittest.TestCase):
    def test_windows(self):
        self.assertEqual(windows_for(10), 1)
        self.assertEqual(windows_for(FRAME_NUM), 1)
        self.assertEqual(windows_for(FRAME_NUM + 1), 2)
        self.assertEqual(windows_for(FRAME_NUM + 72), 2)
        self.assertEqual(windows_for(FRAME_NUM + 73), 3)
        self.assertEqual(windows_for(750), 11)

    def test_auto_memory_profile(self):
        self.assertEqual(auto_persistent(22.5), 0)        # L4
        self.assertEqual(auto_persistent(39.6), ALL_ON_GPU)  # A100 40 GB
        self.assertEqual(auto_persistent(80), ALL_ON_GPU)


class ClientWorkerTests(unittest.TestCase):
    def test_start_generate_progress_and_stop(self):
        engine, root = fake_engine(self, "--load-delay", "0.5")
        logs = []
        engine.start(log=logs.append)
        self.assertEqual(engine.status()["state"], "ready")
        self.assertTrue(any("Carregando modelo falso" in line for line in logs))
        seen = []
        out = os.path.join(root, "out.mp4")
        result = engine.generate(__file__, __file__, out, 120, {"steps": 20, "seed": 9, "step_delay": 0.05},
                                 on_progress=seen.append, poll=0.05)
        self.assertEqual(result["seed"], 9)
        self.assertEqual(result["frames"], 120)
        self.assertTrue(os.path.exists(out))
        self.assertTrue(any(s.get("total") == 20 for s in seen))
        engine.stop()
        self.assertFalse(engine.alive())
        self.assertEqual(engine.status()["state"], "stopped")

    def test_cancel_and_errors(self):
        engine, root = fake_engine(self)
        engine.start(log=lambda m: None)
        flag = threading.Event()
        threading.Timer(0.3, flag.set).start()
        with self.assertRaises(Cancelled):
            engine.generate(__file__, __file__, os.path.join(root, "c.mp4"), 10,
                            {"steps": 500, "step_delay": 0.02}, should_cancel=flag.is_set, poll=0.05)
        with self.assertRaises(OutOfMemoryError):
            engine.generate(__file__, __file__, os.path.join(root, "o.mp4"), 10, {"prompt": "oom"}, poll=0.05)
        with self.assertRaises(EngineError):
            engine.generate("/nao/existe.mp4", __file__, os.path.join(root, "n.mp4"), 10, {}, poll=0.05)
        # o worker continua de pé depois dos erros
        self.assertTrue(engine.ready())

    def test_load_failure_is_reported(self):
        engine, _ = fake_engine(self, "--fail-load")
        with self.assertRaises(EngineError) as ctx:
            engine.start(log=lambda m: None)
        self.assertIn("pesos corrompidos", str(ctx.exception))
        self.assertEqual(engine.status()["state"], "error")

    def test_ensure_running_restarts_dead_worker(self):
        engine, _ = fake_engine(self)
        engine.start(log=lambda m: None)
        engine.proc.kill()
        engine.proc.wait()
        self.assertEqual(engine.status()["state"], "error")
        engine.ensure_running(log=lambda m: None)
        self.assertTrue(engine.ready())


class InstallTests(unittest.TestCase):
    def test_install_commands_and_markers(self):
        root = tmpdir(self)
        engine = InfiniteTalkEngine(root=root)
        commands = []

        def fake_run(cmd, log, env=None, cwd=None, check=True):
            commands.append(cmd)
            if cmd[:2] == ["git", "init"] or cmd[0] == "git":
                os.makedirs(os.path.join(engine.repo_dir, "wan"), exist_ok=True)
                open(os.path.join(engine.repo_dir, "wan", "multitalk.py"), "w").close()
            if "venv" in cmd:
                os.makedirs(os.path.dirname(engine.python), exist_ok=True)
                open(engine.python, "w").close()
            if "flash_attn" in " ".join(cmd):
                return 1                     # wheel indisponível: segue sem flash-attn
            return 0

        with mock.patch.object(engine, "_run", side_effect=fake_run), \
                mock.patch.object(infinitetalk.shutil, "which", return_value="/usr/bin/uv"):
            self.assertTrue(engine.ensure_repo(lambda m: None))
            self.assertTrue(engine.ensure_venv(lambda m: None))
            n = len(commands)
            self.assertFalse(engine.ensure_repo(lambda m: None))
            self.assertFalse(engine.ensure_venv(lambda m: None))
            self.assertEqual(len(commands), n)          # nada é refeito

        flat = [" ".join(c) for c in commands]
        self.assertTrue(any(infinitetalk.REPO_COMMIT in c for c in flat))
        torch_cmd = next(c for c in flat if "torch==2.4.1" in c and "xformers" not in c)
        self.assertIn(infinitetalk.TORCH_INDEX, torch_cmd)
        self.assertTrue(any("xformers" in c and "-c" in c for c in flat))
        self.assertTrue(any("venv" in c and "-p 3.10" in c for c in flat))
        last = flat[-1]
        for pkg in ("xfuser==0.4.1", "optimum-quanto==0.2.6", "numpy==1.26.4", "decord==0.6.0"):
            self.assertIn(pkg, last)
        constraints = open(os.path.join(root, "constraints.txt"), encoding="utf-8").read()
        self.assertIn("torch==2.4.1", constraints)
        self.assertIn("numpy<2", constraints)

    def test_required_weights(self):
        root = tmpdir(self)
        self.assertEqual(len(download.missing(root)), len(download.REQUIRED))
        for rel in download.REQUIRED:
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()
        self.assertEqual(download.missing(root), [])
        engine = InfiniteTalkEngine(root=tempfile.mkdtemp())
        engine.weights_dir = root
        self.assertTrue(engine.weights_ready())

    def test_download_plan_uses_fp8_not_full_dit(self):
        patterns = [p for _, _, pats, _ in download.DOWNLOADS for p in pats]
        self.assertIn("quant_models/infinitetalk_single_fp8.safetensors", patterns)
        self.assertIn("quant_models/t5_fp8.safetensors", patterns)
        self.assertFalse(any("diffusion_pytorch_model" in p for p in patterns))
        self.assertFalse(any(p.endswith("*") and "/" not in p for p in patterns))

    def test_run_streams_output_and_raises(self):
        engine = InfiniteTalkEngine(root=tmpdir(self))
        lines = []
        code = engine._run([sys.executable, "-c", "print('a'); print('b')"], lines.append)
        self.assertEqual(code, 0)
        self.assertIn("   b", lines)
        with self.assertRaises(EngineError):
            engine._run([sys.executable, "-c", "import sys; print('falhou'); sys.exit(3)"], lines.append)
        self.assertIn("falhou", open(engine.install_log_path, encoding="utf-8").read())


if __name__ == "__main__":
    unittest.main()
