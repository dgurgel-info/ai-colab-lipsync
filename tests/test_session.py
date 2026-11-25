import os
import unittest

from tests.helpers import tmpdir

from lipsync_studio.session import DONE, ERROR, PENDING, Session, Settings, list_sessions, plan_parts
from lipsync_studio.studio import Studio, output_size


class PlanPartsTests(unittest.TestCase):
    def test_parts_cover_everything_without_gaps(self):
        for total in (1, 50, 749, 750, 751, 1100, 4321, 25 * 600):
            parts = plan_parts(total, 750)
            self.assertEqual(parts[0].start_frame, 0)
            self.assertEqual(sum(p.frames for p in parts), total)
            for a, b in zip(parts, parts[1:]):
                self.assertEqual(a.start_frame + a.frames, b.start_frame)
            self.assertEqual([p.index for p in parts], list(range(1, len(parts) + 1)))

    def test_no_tiny_last_part(self):
        for total in range(800, 6000, 37):
            sizes = [p.frames for p in plan_parts(total, 750)]
            self.assertGreaterEqual(min(sizes), 750 * 0.66, (total, sizes))
            self.assertLessEqual(max(sizes), 750 * 1.5 + 1, (total, sizes))

    def test_empty(self):
        self.assertEqual(plan_parts(0, 750), [])


class SettingsTests(unittest.TestCase):
    def test_validate_clamps(self):
        s = Settings(prompt="  ", resolution="720p", steps=999, audio_guide=0, text_guide=50, seed=-5,
                     part_seconds=1, audio_longer="x", video_longer="y", background_volume=3).validate()
        self.assertTrue(s.prompt)
        self.assertEqual((s.resolution, s.steps, s.audio_guide, s.text_guide), ("720", 50, 1.0, 10.0))
        self.assertEqual((s.seed, s.part_seconds, s.audio_longer, s.video_longer), (42, 5, "pingpong", "trim"))
        self.assertEqual(s.background_volume, 1.0)

    def test_from_dict_ignores_unknown(self):
        self.assertEqual(Settings.from_dict({"steps": 12, "obsolete": 1}).steps, 12)


class SessionTests(unittest.TestCase):
    def test_roundtrip_and_listing(self):
        root = tmpdir(self)
        s = Session.create(root, "/x/Meu vídeo (final).mp4", "fala.wav", Settings(seed=7))
        s.total_frames = 1500
        s.parts = plan_parts(1500, 750)
        s.parts[0].status = DONE
        s.parts[1].status = ERROR
        s.save()
        loaded = Session.load(s.dir)
        self.assertEqual(loaded.settings.seed, 7)
        self.assertEqual(loaded.title, "Meu_v_deo_final")
        self.assertEqual(loaded.counts()[DONE], 1)
        self.assertEqual([p.index for p in loaded.pending_parts()], [2])
        self.assertFalse(loaded.ready_to_assemble())
        self.assertNotIn(" ", os.path.basename(loaded.dir))
        self.assertEqual([x.id for x in list_sessions(root)], [s.id])
        with self.assertRaises(KeyError):
            loaded.get_part(9)

    def test_pending_includes_interrupted(self):
        s = Session.create(tmpdir(self), "v.mp4", "a.wav", Settings())
        s.parts = plan_parts(100, 50)
        s.parts[0].status = "running"
        self.assertEqual(len(s.pending_parts()), 2)
        self.assertEqual(s.parts[1].status, PENDING)


class StudioPlanningTests(unittest.TestCase):
    def test_target_frames(self):
        f = Studio.target_frames
        self.assertEqual(f(100, 150, Settings(audio_longer="pingpong")), 150)
        self.assertEqual(f(100, 150, Settings(audio_longer="trim")), 100)
        self.assertEqual(f(200, 150, Settings(video_longer="trim")), 150)
        self.assertEqual(f(200, 150, Settings(video_longer="silence")), 200)
        self.assertEqual(f(150, 150, Settings()), 150)

    def test_output_size_matches_official_buckets(self):
        self.assertEqual(output_size(1920, 1080, "480"), (896, 448))
        self.assertEqual(output_size(1080, 1920, "480"), (448, 832))
        self.assertEqual(output_size(1000, 1000, "480"), (640, 640))
        self.assertEqual(output_size(1920, 1080, "720"), (1280, 704))
        self.assertEqual(output_size(0, 0), (0, 0))


if __name__ == "__main__":
    unittest.main()
