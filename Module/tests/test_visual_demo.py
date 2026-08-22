from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from Module.visual_demo import run_visual_demo


class VisualDemoTest(unittest.TestCase):
  def test_gallery_contains_three_real_renderings_and_multiframe_animation(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      output = Path(temporary_directory)
      result = run_visual_demo(output, animation_frames=6)

      self.assertEqual(result["demo"], "M0_TO_M4_E05_MCC_VISUAL_GALLERY")
      self.assertTrue(
        all(result["modules"][module_id]["passed"] for module_id in ("M01", "M02", "M03"))
      )
      # A temporary output directory intentionally lacks the separately rendered
      # formal FR3 videos, so the complete gallery readiness flag stays false.
      self.assertFalse(result["modules"]["M0_M04_FR3"]["passed"])
      self.assertFalse(result["passed"])
      for filename in (
        "module_1_surface_model.png",
        "module_2_fingertip_mcc.png",
        "module_3_runtime_guards.png",
      ):
        path = output / filename
        self.assertTrue(path.is_file())
        self.assertGreater(path.stat().st_size, 10_000)
        with Image.open(path) as image:
          self.assertEqual(image.format, "PNG")
          self.assertGreaterEqual(image.width, 1000)

      with Image.open(output / "module_2_curved_surface.gif") as animation:
        self.assertEqual(animation.format, "GIF")
        self.assertEqual(animation.n_frames, 6)

      gallery = (output / "index.html").read_text(encoding="utf-8")
      self.assertIn("Module 1 · Oracle SurfaceModel", gallery)
      self.assertIn("module_2_curved_surface.gif", gallery)
      self.assertTrue((output / "summary.json").is_file())


if __name__ == "__main__":
  unittest.main()
