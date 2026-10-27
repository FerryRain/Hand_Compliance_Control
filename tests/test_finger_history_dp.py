"""Run directly on CPU or with pytest; CUDA smoke uses the real trainer."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mcc_finger_compliance_control/scripts'))
from finger_history_dp import FingerHistoryConditionEncoder, FINGER_INDICES
from train_dp import build_policy


def test_feature_partition_and_temporal_gradients():
    torch.manual_seed(42)
    encoder = FingerHistoryConditionEncoder()
    values = torch.arange(2 * 16 * 96).reshape(2, 16, 96).float()
    local, palm = encoder.split_history(values)
    restored = torch.empty_like(values)
    for finger, indices in enumerate(FINGER_INDICES):
        restored[:, :, list(indices)] = local[:, finger]
    restored[..., 84:] = palm
    assert torch.equal(values, restored)
    assert sorted(sum((list(x) for x in FINGER_INDICES), []) + list(range(84, 96))) == list(range(96))
    history = torch.randn(2, 16, 96, requires_grad=True)
    encoder(history).square().sum().backward()
    assert torch.isfinite(history.grad).all()
    # Every channel at every history sample contributes to the condition.
    assert (history.grad.abs().sum(0) > 0).all()
    for parameter in encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_train_save_load_sampling_and_legacy():
    path = ROOT / 'mcc_finger_compliance_control/data/models/DP_regional_222_scratch_20261022/run_manifest.json'
    args = SimpleNamespace(**json.loads(path.read_text())['config'])
    args.conditioning_architecture = 'finger_history'
    args.down_dims = [16, 32, 64]
    args.inference_steps = 2
    policy = build_policy(args, torch.device('cpu'))
    before = policy.diffusion.condition_encoder.finger_id.weight.detach().clone()
    batch = {'observation.state': torch.randn(2, 16, 84),
             'observation.environment_state': torch.randn(2, 16, 12),
             'action': torch.randn(2, 16, 16),
             'action_is_pad': torch.zeros(2, 16, dtype=torch.bool)}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    policy.diffusion.compute_loss(batch).backward()
    optimizer.step()
    assert not torch.equal(before, policy.diffusion.condition_encoder.finger_id.weight)
    payload = io.BytesIO()
    torch.save({'config': vars(args), 'model': policy.state_dict()}, payload)
    payload.seek(0)
    checkpoint = torch.load(payload, weights_only=False)
    restored = build_policy(SimpleNamespace(**checkpoint['config']), torch.device('cpu'))
    restored.load_state_dict(checkpoint['model'], strict=True)
    outputs = []
    for model in (policy, restored):
        model.eval()
        with torch.no_grad():
            condition = model.diffusion._prepare_global_conditioning(batch)
            assert condition.shape == (2, 256)
            outputs.append(model.diffusion.conditional_sample(2, global_cond=condition,
                            generator=torch.Generator().manual_seed(123)))
    assert torch.equal(outputs[0], outputs[1])
    assert outputs[0].shape == (2, 16, 16)
    delattr(args, 'conditioning_architecture')  # old checkpoint lacks this flag
    legacy = build_policy(args, torch.device('cpu'))
    condition = legacy.diffusion._prepare_global_conditioning(batch)
    assert torch.equal(condition, torch.cat((batch['observation.state'],
                       batch['observation.environment_state']), -1).flatten(1))
    assert not any('condition_encoder' in key for key in legacy.state_dict())


if __name__ == '__main__':
    torch.set_num_threads(1)
    test_feature_partition_and_temporal_gradients()
    test_train_save_load_sampling_and_legacy()
    print('PASS: all 1536 inputs covered; gradients; training update; checkpoint sampling; legacy compatibility')
