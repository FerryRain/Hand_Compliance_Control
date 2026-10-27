"""Real split-tip contacts must not keep collection in precontact forever."""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_trajectories import FullHandMCCCollectionPolicy
from object_catalog import load_object_config
from surface_mcc_finger import TIP_BACK_CONTACT_X_LIMIT_M


class SplitCollectionContactGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_object_config('ycb_mustard')
        env = SimpleNamespace(num_envs=1, device='cpu',
                              scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
        base = SimpleNamespace(
            palm_controller=SimpleNamespace(fixed_target_np=np.array([.65, 0, .67, -np.pi, 0, 0]), reset=lambda: None),
            finger_controller=SimpleNamespace(reset=lambda: None))
        kwargs = dict(surface_preload_m=.002, anchor_force_threshold=.05,
            precontact_force_threshold=.10, anchor_settle_frames=50,
            surface_target_mode='nearest_surface', contact_search_step_m=.00015,
            contact_search_step_rad=.02, contact_search_limit_rad=.30,
            contact_transient_loss_frames=6, contact_recovery_confirm_frames=4,
            contact_transient_search_step_m=.0002, contact_transient_release_step_m=.0001,
            persistent_recovery_max_joint_step_rad=.04,
            nominal_grasp_q=tuple(config.collection['pregrasp_q']), differential_contact_qp=True)
        cls.legacy = FullHandMCCCollectionPolicy(env, base, config, 0, 0, [0, 1, 2, 3],
                                                 free_tip_orientation=False, **kwargs)
        cls.split = FullHandMCCCollectionPolicy(env, base, config, 0, 0, [0, 1, 2, 3],
                                                free_tip_orientation=True, **kwargs)
        cls.fixture = json.loads((Path(__file__).parent / 'fixtures/split_tip_back_contact.json').read_text())

    def test_legacy_keeps_existing_front_pad_gate(self):
        np.testing.assert_array_equal(self.legacy.controllers[0].config.back_contact_x_limit_m,
                                      TIP_BACK_CONTACT_X_LIMIT_M)

    def test_physically_detected_split_tip_back_contacts_are_accepted(self):
        f = self.fixture
        arguments = (np.asarray(f['q']), np.asarray(f['palm']), np.asarray(f['points']), np.asarray(f['found']))
        old, _ = self.legacy.controllers[0].pad_contact_validity(*arguments)
        new, _ = self.split.controllers[0].pad_contact_validity(*arguments)
        self.assertFalse(old.all())
        self.assertTrue(new.all())

    def test_missing_or_nonfinite_contact_still_rejected(self):
        f = self.fixture
        points = np.asarray(f['points']).copy()
        points[1] = np.nan
        found = np.asarray(f['found']).copy()
        found[0] = False
        valid, _ = self.split.controllers[0].pad_contact_validity(
            np.asarray(f['q']), np.asarray(f['palm']), points, found)
        self.assertFalse(valid[0])
        self.assertFalse(valid[1])


if __name__ == '__main__':
    unittest.main()
