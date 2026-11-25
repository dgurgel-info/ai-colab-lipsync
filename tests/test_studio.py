"""Fluxo completo com o motor simulado: sessão, partes, revisão e montagem."""

import os
import unittest

from tests.helpers import make_audio, make_image, make_video, needs_ffmpeg, tmpdir

from lipsync_studio import media
from lipsync_studio.engine import Cancelled
from lipsync_studio.engine.fake import FakeEngine
from lipsync_studio.session import DONE, ERROR, Session, Settings
from lipsync_studio.studio import Context, Studio


class Quiet(Context):
    def __init__(self, cancel_after=None):
        self.stages, self.messages, self.cancel_after = [], [], cancel_after

    def log(self, message):
        self.messages.append(message)

    def stage(self, text, pct=None):
        self.stages.append((text, pct))

    def cancelled(self):
        return self.cancel_after is not None and len(self.stages) >= self.cancel_after


@needs_ffmpeg
class StudioTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir(self)
        self.video = make_video(os.path.join(self.dir, "pessoa.mp4"), 4.0, 640, 360, fps=30, with_audio=True)
        self.audio = make_audio(os.path.join(self.dir, "fala.mp3"), 9.0)

    def test_full_run_audio_longer_pingpong(self):
        engine = FakeEngine(frame_skew=-1)
        studio = Studio(engine, root=os.path.join(self.dir, "work"))
        ctx = Quiet()
        result = studio.run(self.video, self.audio, Settings(part_seconds=5, seed=5, steps=6,
                                                             background_volume=0.2), ctx)
        out = result["output"]
        self.assertTrue(out and os.path.exists(out))
        info = media.probe(out)
        self.assertEqual(media.count_frames(out), 225)              # 9 s exatos a 25 fps
        self.assertAlmostEqual(info.duration, 9.0, delta=0.1)
        self.assertEqual((info.width, info.height), (896, 448))      # balde oficial 480p para 16:9
        session = studio.load(result["session"])
        self.assertEqual(len(session.parts), 2)
        self.assertTrue(all(p.status == DONE and p.seed == 5 for p in session.parts))
        self.assertTrue(session.has_background)
        self.assertEqual(len(engine.calls), 2)
        self.assertEqual(engine.calls[0]["steps"], 6)
        self.assertEqual(sum(c["frames"] for c in engine.calls), 225)
        self.assertEqual(studio.outputs(), [out])
        self.assertTrue(any("vai e volta" in m for m in ctx.messages))

    def test_failed_part_then_regenerate_and_assemble(self):
        engine = FakeEngine(fail_parts={2})
        studio = Studio(engine, root=os.path.join(self.dir, "work"))
        result = studio.run(self.video, self.audio, Settings(part_seconds=5), Quiet())
        self.assertIsNone(result["output"])
        session = studio.load(result["session"])
        self.assertEqual(session.total_frames, 225)
        self.assertEqual([p.status for p in session.parts], [DONE, ERROR])
        with self.assertRaises(ValueError):
            studio.assemble(session, Quiet())
        failed = studio.generate(session, Quiet(), seed=77)
        self.assertEqual(failed, [])
        self.assertTrue(session.ready_to_assemble())
        self.assertEqual(Session.load(session.dir).get_part(2).seed, 77)
        out = studio.assemble(session, Quiet())
        self.assertEqual(media.count_frames(out), 225)

    def test_regenerate_done_part_keeps_others(self):
        engine = FakeEngine()
        studio = Studio(engine, root=os.path.join(self.dir, "work"))
        result = studio.run(self.video, self.audio, Settings(part_seconds=5, seed=1), Quiet())
        session = studio.load(result["session"])
        studio.generate(session, Quiet(), indices=[1], seed=999, steps=4, audio_guide=5)
        again = studio.load(session.id)
        self.assertEqual((again.get_part(1).seed, again.get_part(1).steps, again.get_part(1).audio_guide), (999, 4, 5))
        self.assertEqual(again.get_part(2).seed, 1)
        self.assertEqual(engine.calls[-1]["audio_guide"], 5)

    def test_video_longer_keep_with_silence(self):
        short_audio = make_audio(os.path.join(self.dir, "curta.wav"), 2.0)
        studio = Studio(FakeEngine(), root=os.path.join(self.dir, "work"))
        result = studio.run(self.video, short_audio, Settings(video_longer="silence"), Quiet())
        self.assertEqual(media.count_frames(result["output"]), 100)

    def test_photo_input(self):
        photo = make_image(os.path.join(self.dir, "rosto.png"), 300, 400)
        studio = Studio(FakeEngine(), root=os.path.join(self.dir, "work"))
        result = studio.run(photo, self.audio, Settings(part_seconds=10), Quiet())
        session = studio.load(result["session"])
        self.assertEqual(session.source_kind, "image")
        self.assertEqual(media.count_frames(result["output"]), 225)
        self.assertEqual(media.probe(result["output"]).width, 576)    # 3:4 -> balde 576×704

    def test_cancel_keeps_session_resumable(self):
        engine = FakeEngine()
        studio = Studio(engine, root=os.path.join(self.dir, "work"))
        session = studio.create_session(self.video, self.audio, Settings(part_seconds=5), Quiet())
        with self.assertRaises(Cancelled):
            studio.generate(session, Quiet(cancel_after=2))
        again = studio.load(session.id)
        self.assertFalse(again.ready_to_assemble())
        self.assertGreater(len(again.pending_parts()), 0)
        self.assertEqual(studio.generate(again, Quiet()), [])
        self.assertTrue(again.ready_to_assemble())

    def test_rejects_bad_inputs(self):
        studio = Studio(FakeEngine(), root=os.path.join(self.dir, "work"))
        with self.assertRaises(ValueError):
            studio.run("", self.audio, Settings(), Quiet())
        tiny = make_audio(os.path.join(self.dir, "tiny.wav"), 0.3)
        with self.assertRaises(ValueError):
            studio.run(self.video, tiny, Settings(), Quiet())


if __name__ == "__main__":
    unittest.main()
