import threading
import time
import unittest

import tests.helpers  # noqa: F401  (ajusta o sys.path)
from lipsync_studio.jobs import CANCELLED, COMPLETED, ERROR, IDLE, JobManager


class JobManagerTests(unittest.TestCase):
    def test_completes_with_result_logs_and_progress(self):
        jobs = JobManager(echo=False)
        self.assertEqual(jobs.state()["status"], IDLE)

        def work(ctx, x):
            ctx.log("olá")
            ctx.stage("meio", 50)
            return x * 2

        ok, job_id = jobs.start("run", "teste", work, 21)
        self.assertTrue(ok)
        st = jobs.wait(5)
        self.assertEqual(st["status"], COMPLETED)
        self.assertEqual(st["result"], 42)
        self.assertEqual(st["job_id"], job_id)
        self.assertIn("olá", st["logs"])
        self.assertEqual(st["pct"], 100.0)

    def test_only_one_job_at_a_time(self):
        jobs = JobManager(echo=False)
        gate = threading.Event()
        jobs.start("run", "primeira", lambda ctx: gate.wait(5))
        ok, msg = jobs.start("run", "segunda", lambda ctx: None)
        self.assertFalse(ok)
        self.assertIn("primeira", msg)
        gate.set()
        jobs.wait(5)

    def test_cancel(self):
        jobs = JobManager(echo=False)

        def work(ctx):
            while True:
                ctx.check()
                time.sleep(0.01)

        jobs.start("run", "longa", work)
        time.sleep(0.05)
        jobs.stop()
        self.assertEqual(jobs.wait(5)["status"], CANCELLED)

    def test_error_is_reported(self):
        jobs = JobManager(echo=False)

        def work(ctx):
            raise ValueError("deu ruim")

        jobs.start("run", "erro", work)
        st = jobs.wait(5)
        self.assertEqual(st["status"], ERROR)
        self.assertEqual(st["error"], "deu ruim")
        self.assertIn("ValueError", st["logs"])


if __name__ == "__main__":
    unittest.main()
