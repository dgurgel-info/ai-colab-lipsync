"""A interface monta e os textos de apoio funcionam (pula se o Gradio não estiver instalado)."""

import os
import unittest

from tests.helpers import make_audio, make_video, needs_ffmpeg, tmpdir

from lipsync_studio.engine.fake import FakeEngine
from lipsync_studio.jobs import JobManager
from lipsync_studio.studio import Studio
from lipsync_studio.ui import theme

try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None


class ThemeTests(unittest.TestCase):
    def test_html_helpers_escape_and_render(self):
        header = theme.render_header("<GPU>", "loading", "1.0.0")
        self.assertIn("&lt;GPU&gt;", header)
        self.assertIn("carregando", header)
        self.assertEqual(theme.render_engine_progress(None), "")
        self.assertIn("42%", theme.render_engine_progress(42, "x", "<detalhe>"))
        steps = theme.render_steps(2, current=3)
        self.assertEqual(steps.count("completed"), 2)
        self.assertIn("current", steps)
        running = theme.render_monitor({"status": "running", "stage": "<b>", "pct": 40, "elapsed": 5, "title": "t"})
        self.assertIn("&lt;b&gt;", running)
        self.assertIn("40%", running)
        self.assertIn("Falhou", theme.render_monitor({"status": "error", "error": "x"}))
        self.assertIn("Nenhuma sessão", theme.render_summary(None))
        self.assertIn("{", theme.load_css())


@unittest.skipIf(gr is None, "Gradio não instalado")
class AppTests(unittest.TestCase):
    def test_build_app(self):
        from lipsync_studio.ui.app import build_app

        studio = Studio(FakeEngine(), root=tmpdir(self))
        demo, launch_style = build_app(studio, JobManager(echo=False), gpu_label="Teste")
        self.assertIsInstance(demo, gr.Blocks)
        self.assertIsInstance(launch_style, dict)

    @needs_ffmpeg
    def test_describe_inputs(self):
        from lipsync_studio.ui.app import KIND_VIDEO, describe_inputs, input_steps

        d = tmpdir(self)
        video = make_video(os.path.join(d, "v.mp4"), 2.0, 640, 360)
        audio = make_audio(os.path.join(d, "a.wav"), 5.0)
        text = describe_inputs(KIND_VIDEO, video, None, audio, "480p (recomendado)",
                               "Vídeo vai e volta (recomendado)", "Cortar o vídeo no fim do áudio")
        self.assertIn("896×448", text)
        self.assertIn("vai e volta", text)
        self.assertIn("current", input_steps(KIND_VIDEO, video, None, audio))


if __name__ == "__main__":
    unittest.main()
