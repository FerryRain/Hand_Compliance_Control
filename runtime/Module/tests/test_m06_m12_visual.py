from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Module.m06_m12_visual_demo import run_visual_demo


class M06M12VisualDemoTest(unittest.TestCase):
  def test_benchmark_backed_gallery_contains_all_modules_and_scope_boundaries(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      output = Path(temporary_directory)
      result = run_visual_demo(output, refresh_benchmark=True)
      self.assertTrue(result["passed"])
      self.assertFalse(result["dp_used"])
      self.assertFalse(result["g1_changed"])
      for module_id, path_string in result["artifacts"].items():
        path = Path(path_string)
        self.assertTrue(path.is_file(), module_id)
        self.assertGreater(path.stat().st_size, 10_000)
        with Image.open(path) as image:
          self.assertEqual(image.format, "PNG")
          self.assertGreaterEqual(image.width, 1000)

      summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
      self.assertEqual(summary["performance_verdict"], "MET")
      self.assertFalse(summary["scope"]["dp_used"])
      self.assertFalse(summary["scope"]["g1_changed"])
      self.assertTrue(summary["integration_smoke"]["completed"])
      self.assertEqual(set(summary["modules"]), {f"M{index:02d}" for index in range(6, 13)})
      self.assertTrue((output / "performance.csv").is_file())
      self.assertTrue((output / "traces.npz").is_file())
      gallery = (output / "index.html").read_text(encoding="utf-8")
      self.assertIn("目的、效果与实测性能总览", gallery)
      self.assertIn("G1 保持 No-Go", gallery)
      self.assertIn("module_12_shadow_viability.png", gallery)


if __name__ == "__main__":
  unittest.main()
