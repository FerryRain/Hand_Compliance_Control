from __future__ import annotations

import sys
from pathlib import Path
import unittest

import mujoco
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from object_catalog import (  # noqa: E402
    build_object_spec,
    list_object_ids,
    load_object_config,
    object_local_aabb,
)


class ObjectCatalogTest(unittest.TestCase):
    def test_catalog_contains_baseline_and_multiple_families(self) -> None:
        object_ids = list_object_ids()
        self.assertIn("capsule_medium", object_ids)
        self.assertIn("rounded_box_medium", object_ids)
        self.assertGreaterEqual(len(object_ids), 8)
        families = {load_object_config(item).family for item in object_ids}
        self.assertGreaterEqual(len(families), 3)

    def test_every_object_resolves_and_compiles(self) -> None:
        for object_id in list_object_ids():
            with self.subTest(object_id=object_id):
                config = load_object_config(object_id)
                self.assertIn("translation", config.motion)
                self.assertIn("rotation", config.motion)
                lower, upper = object_local_aabb(config)
                self.assertTrue(np.all(np.isfinite(lower)))
                self.assertTrue(np.all(upper > lower))
                model = build_object_spec(object_id).compile()
                self.assertEqual(model.nbody, 2)
                self.assertEqual(model.ngeom, len(config.geoms))
                body_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, config.body_name
                )
                self.assertGreaterEqual(int(model.body_mocapid[body_id]), 0)

    def test_current_capsule_geometry_is_preserved(self) -> None:
        config = load_object_config("capsule_medium")
        self.assertEqual(len(config.geoms), 1)
        geom = config.geoms[0]
        self.assertEqual(geom.geom_type, "capsule")
        np.testing.assert_allclose(geom.size[:2], (0.15, 0.08))
        np.testing.assert_allclose(
            config.contact["solref"], (-20_000.0, -400.0)
        )

    def test_rounded_box_is_one_convex_mesh_geom(self) -> None:
        config = load_object_config("rounded_box_medium")
        self.assertEqual(len(config.geoms), 1)
        self.assertEqual(config.geoms[0].geom_type, "rounded_box")
        model = build_object_spec("rounded_box_medium").compile()
        self.assertEqual(model.ngeom, 1)
        self.assertEqual(model.nmesh, 1)
        self.assertEqual(
            int(model.geom_type[0]), int(mujoco.mjtGeom.mjGEOM_MESH)
        )


if __name__ == "__main__":
    unittest.main()
