"""Estrutura do notebook do Colab: passos, parâmetros e código Python válido."""

import ast
import json
import re
import unittest

from tests.helpers import ROOT

NOTEBOOK = ROOT / "LipSync_Studio.ipynb"


def cells():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]


def python_source(cell) -> str:
    """Código da célula sem os comandos de shell do Colab (``!``)."""
    return "".join(line for line in cell["source"] if not line.lstrip().startswith(("!", "%")))


class NotebookTests(unittest.TestCase):
    def test_steps_in_order(self):
        code = [c for c in cells() if c["cell_type"] == "code"]
        titles = [re.search(r"# @title (Passo \d)", "".join(c["source"])).group(1) for c in code]
        self.assertEqual(titles, ["Passo 1", "Passo 2", "Passo 3", "Passo 4"])

    def test_code_cells_are_valid_python(self):
        for cell in cells():
            if cell["cell_type"] == "code":
                ast.parse(python_source(cell))

    def test_points_to_this_repository_and_infinitetalk(self):
        text = "".join("".join(c["source"]) for c in cells())
        self.assertIn("https://github.com/dgurgel-info/ai-colab-lipsync.git", text)
        self.assertIn('LIPSYNC_REF = "main"', text)
        self.assertIn("InfiniteTalkEngine", text)
        self.assertIn("MeiGen-AI/InfiniteTalk", text)
        self.assertNotIn("LatentSync", text)

    def test_colab_metadata(self):
        meta = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]
        self.assertEqual(meta["accelerator"], "GPU")
        self.assertEqual(meta["colab"]["gpuType"], "L4")

    def test_interface_cell_uses_public_link_and_allowed_paths(self):
        step3 = "".join(cells()[3]["source"])
        self.assertIn("share=True", step3)
        self.assertIn("build_app(studio, jobs", step3)


if __name__ == "__main__":
    unittest.main()
