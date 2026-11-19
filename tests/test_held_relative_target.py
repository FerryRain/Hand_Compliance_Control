import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"mcc_finger_compliance_control/scripts"))
from held_relative_target import apply_held_target, process_held_target


class HeldTargetTests(unittest.TestCase):
    def test_physics_motion_does_not_reanchor_the_command(self):
        sent = []
        data = SimpleNamespace(joint_pos=torch.tensor([[.4, .5]]), encoder_bias=torch.zeros(1, 2))
        entity = SimpleNamespace(data=data,
            set_joint_position_target=lambda target, joint_ids: sent.append(target.clone()))
        term = SimpleNamespace(_entity=entity, _target_ids=torch.arange(2))
        def process(t, actions):
            t._processed_actions = .08*actions
        process_held_target(term, torch.tensor([[.5, -.5]]), process)
        for _ in range(10):
            apply_held_target(term, lambda _: self.fail("Unexpected legacy fallback"))
            data.joint_pos += .01
        for target in sent:
            torch.testing.assert_close(target, torch.tensor([[.44, .46]]))
        process_held_target(term, torch.tensor([[.5, -.5]]), process)
        apply_held_target(term, lambda _: None)
        torch.testing.assert_close(sent[-1], torch.tensor([[.54, .56]]))

    def test_zero_delta_holds_tick_start_position(self):
        sent = []
        data = SimpleNamespace(joint_pos=torch.tensor([[.4]]), encoder_bias=torch.tensor([[.02]]))
        term = SimpleNamespace(_entity=SimpleNamespace(data=data,
            set_joint_position_target=lambda target, joint_ids: sent.append(target.clone())),
            _target_ids=torch.tensor([0]))
        def process(t, actions):
            t._processed_actions = actions
        process_held_target(term, torch.zeros(1, 1), process)
        data.joint_pos[:] = .6
        apply_held_target(term, lambda _: None)
        torch.testing.assert_close(sent[0], torch.tensor([[.42]]))


if __name__ == "__main__":
    unittest.main()
