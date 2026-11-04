"""Source publication regressions; runs without torch/transformers."""
import tempfile
import unittest
from pathlib import Path

from check_sw_token_source import check_file, check_repository

class SourceIntegrityTests(unittest.TestCase):
    def test_actual_experiment_sources(self):
        self.assertGreater(check_repository(), 0)

    def test_truncation_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.py"
            path.write_text("x = 1\n# 6444 tokens " + "truncated\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                check_file(path)

    def test_syntax_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.py"
            path.write_text("if True\n    pass\n", encoding="utf-8")
            with self.assertRaises(SyntaxError):
                check_file(path)

if __name__ == "__main__":
    unittest.main()
