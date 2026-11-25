import os
import unittest

from tests.helpers import file_ok, make_audio, make_image, make_video, needs_ffmpeg, tmpdir

from lipsync_studio import media
from lipsync_studio.media import FPS


@needs_ffmpeg
class MediaTests(unittest.TestCase):
    def setUp(self):
        self.dir = tmpdir(self)

    def p(self, name):
        return os.path.join(self.dir, name)

    def test_normalize_video_to_25fps_even_size(self):
        src = make_video(self.p("in.mp4"), 2, 321, 241, fps=30, with_audio=True)
        info = media.normalize_video(src, self.p("n.mp4"), max_side=1080)
        self.assertEqual(info.frames, 50)
        self.assertEqual((info.width % 2, info.height % 2), (0, 0))
        self.assertFalse(media.probe(self.p("n.mp4")).has_audio)
        self.assertAlmostEqual(info.fps, 25, places=0)

    def test_max_side_limits_the_short_side(self):
        land = make_video(self.p("l.mp4"), 0.4, 1280, 720)
        port = make_video(self.p("p.mp4"), 0.4, 720, 1280)
        a = media.normalize_video(land, self.p("ln.mp4"), max_side=360)
        b = media.normalize_video(port, self.p("pn.mp4"), max_side=360)
        self.assertEqual((a.width, a.height), (640, 360))
        self.assertEqual((b.width, b.height), (360, 640))

    def test_fit_video_modes_give_exact_frames(self):
        src = media.normalize_video(make_video(self.p("s.mp4"), 1.2, fps=25), self.p("n.mp4"))
        self.assertEqual(src.frames, 30)
        for mode in ("pingpong", "loop"):
            out = self.p(f"{mode}.mp4")
            media.fit_video(self.p("n.mp4"), out, 77, mode)
            self.assertEqual(media.count_frames(out), 77, mode)
        media.fit_video(self.p("n.mp4"), self.p("cut.mp4"), 12)
        self.assertEqual(media.count_frames(self.p("cut.mp4")), 12)

    def test_audio_fit_cut_and_normalize(self):
        src = make_audio(self.p("a.mp3"), 2.0)
        media.normalize_audio(src, self.p("a.wav"))
        media.fit_audio(self.p("a.wav"), self.p("long.wav"), 75)          # 3 s: completa com silêncio
        media.fit_audio(self.p("a.wav"), self.p("short.wav"), 25)         # 1 s: corta
        self.assertAlmostEqual(media.probe(self.p("long.wav")).duration, 3.0, delta=0.03)
        self.assertAlmostEqual(media.probe(self.p("short.wav")).duration, 1.0, delta=0.03)
        media.cut_audio(self.p("long.wav"), self.p("part.wav"), 25, 40)
        self.assertAlmostEqual(media.probe(self.p("part.wav")).duration, 1.6, delta=0.03)

    def test_audio_from_video_file(self):
        src = make_video(self.p("v.mp4"), 1.0, with_audio=True)
        info = media.normalize_audio(src, self.p("x.wav"))
        self.assertTrue(info.has_audio)
        with self.assertRaises(media.MediaError):
            media.normalize_audio(make_video(self.p("mute.mp4"), 0.5), self.p("y.wav"))

    def test_cut_conform_concat_and_mux(self):
        media.normalize_video(make_video(self.p("s.mp4"), 3.0), self.p("n.mp4"))
        media.cut_video(self.p("n.mp4"), self.p("a.mp4"), 0, 40)
        media.cut_video(self.p("n.mp4"), self.p("b.mp4"), 40, 35)
        self.assertEqual(media.count_frames(self.p("a.mp4")), 40)
        self.assertEqual(media.count_frames(self.p("b.mp4")), 35)
        media.conform_frames(self.p("a.mp4"), self.p("a2.mp4"), 43)      # repete o último quadro
        media.conform_frames(self.p("b.mp4"), self.p("b2.mp4"), 32)      # corta
        self.assertEqual(media.count_frames(self.p("a2.mp4")), 43)
        self.assertEqual(media.count_frames(self.p("b2.mp4")), 32)
        media.concat_videos([self.p("a2.mp4"), self.p("b2.mp4")], self.p("j.mp4"))
        self.assertEqual(media.count_frames(self.p("j.mp4")), 75)
        audio = media.normalize_audio(make_audio(self.p("a.wav"), 3.0), self.p("m.wav"))
        self.assertTrue(audio.has_audio)
        bg = media.normalize_audio(make_audio(self.p("bg.wav"), 3.0, 200), self.p("bgm.wav"))
        self.assertTrue(bg.has_audio)
        media.mux(self.p("j.mp4"), self.p("m.wav"), self.p("final.mp4"), self.p("bgm.wav"), 0.3)
        final = media.probe(self.p("final.mp4"))
        self.assertTrue(final.has_video and final.has_audio)
        self.assertAlmostEqual(final.duration, 3.0, delta=0.1)
        media.preview_with_audio(self.p("a2.mp4"), self.p("m.wav"), self.p("prev.mp4"))
        self.assertTrue(media.probe(self.p("prev.mp4")).has_audio)

    def test_image_to_video(self):
        img = make_image(self.p("foto.png"), 301, 401)
        self.assertTrue(media.is_image(img))
        info = media.image_to_video(img, self.p("foto.mp4"), 60)
        self.assertEqual(info.frames, 60)
        self.assertEqual((info.width % 2, info.height % 2), (0, 0))

    def test_probe_fallback_without_ffprobe(self):
        src = make_video(self.p("v.mp4"), 1.0, 320, 240, with_audio=True)
        info = media._probe_ffmpeg(src)
        self.assertTrue(info.has_video and info.has_audio)
        self.assertEqual((info.width, info.height), (320, 240))
        self.assertAlmostEqual(info.duration, 1.0, delta=0.1)
        with self.assertRaises(media.MediaError):
            media.probe(self.p("nao_existe.mp4"))
        self.assertTrue(file_ok(src))

    def test_format_time(self):
        self.assertEqual(media.format_time(65.25), "01:05.2")
        self.assertEqual(media.format_time(3725), "1:02:05.0")
        self.assertEqual(FPS, 25)


if __name__ == "__main__":
    unittest.main()
