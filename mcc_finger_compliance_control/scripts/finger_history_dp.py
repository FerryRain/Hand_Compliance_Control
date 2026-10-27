"""Per-finger temporal conditioning for the existing 96D palm-frame DP contract.

Uses the earlier Finger DP temporal encoders, but preserves *all* fields of the
regional training data instead of requiring its different force-history inputs.
The shared finger encoder sees 16 past samples, not 16 future waypoints.
"""
from pathlib import Path
import sys

import torch
from torch import nn
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionModel, DiffusionConditionalUnet1d,
)

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from Module.module_4_finger_dp.policy import TemporalVectorEncoder


# Legacy feature-major layout -> each finger's q, position, normal, validity,
# q velocity, contact-point velocity and normal angular rate (21 channels).
FINGER_INDICES = tuple(tuple(
    index
    for start, width in ((0, 4), (16, 3), (28, 3), (40, 1),
                         (44, 4), (60, 3), (72, 3))
    for index in range(start + finger * width, start + (finger + 1) * width)
) for finger in range(4))

STATE_FIELDS = (
    ('q_hand', 16), ('fingertip_contact_pos_palm', 12),
    ('fingertip_contact_normal_palm', 12), ('fingertip_contact_mask', 4),
    ('q_velocity', 16), ('fingertip_contact_point_velocity_palm', 12),
    ('fingertip_contact_normal_angular_rate_palm', 12),
    ('palm_relative_twist_palm', 6), ('planner_palm_delta_pose_palm', 6),
)


class FingerHistoryConditionEncoder(nn.Module):
    """Shared local history encoder, finger identity, palm history, joint fusion."""

    def __init__(self, condition_dim=256):
        super().__init__()
        self.register_buffer('finger_indices', torch.tensor(FINGER_INDICES))
        self.local_history = TemporalVectorEncoder(21, 64)
        self.finger_id = nn.Embedding(4, 8)
        self.local_token = nn.Sequential(nn.Linear(72, 64), nn.SiLU())
        self.palm_history = TemporalVectorEncoder(12, 64)
        self.fusion = nn.Sequential(
            nn.Linear(4 * 64 + 64, condition_dim), nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

    def split_history(self, history):
        if history.ndim != 3 or history.shape[-1] != 96:
            raise ValueError('finger history requires [batch, time, 96]')
        # [B,T,4,21] -> [B,4,T,21]; no channel or history is discarded.
        return history[:, :, self.finger_indices].permute(0, 2, 1, 3), history[..., 84:96]

    def forward(self, history):
        local, palm = self.split_history(history)
        batch, fingers, steps, channels = local.shape
        encoded = self.local_history(local.reshape(batch * fingers, steps, channels))
        encoded = encoded.reshape(batch, fingers, 64)
        identity = self.finger_id.weight.unsqueeze(0).expand(batch, -1, -1)
        tokens = self.local_token(torch.cat((encoded, identity), dim=-1))
        return self.fusion(torch.cat((tokens.flatten(1), self.palm_history(palm)), dim=-1))


class FingerHistoryDiffusionModel(DiffusionModel):
    """Keep LeRobot loss/sampler semantics and replace only conditioning + U-Net."""

    def __init__(self, original: DiffusionModel, condition_dim=256):
        nn.Module.__init__(self)
        self.config = original.config
        if (self.config.robot_state_feature.shape != (84,)
                or self.config.env_state_feature.shape != (12,)
                or self.config.image_features):
            raise ValueError('finger_history supports the 84+12 motion96 contract only')
        self.noise_scheduler = original.noise_scheduler
        self.num_inference_steps = original.num_inference_steps
        self.condition_encoder = FingerHistoryConditionEncoder(condition_dim)
        self.unet = DiffusionConditionalUnet1d(self.config, global_cond_dim=condition_dim)

    def _prepare_global_conditioning(self, batch):
        history = torch.cat((batch['observation.state'],
                             batch['observation.environment_state']), dim=-1)
        if history.shape[1] != self.config.n_obs_steps:
            raise ValueError('observation history length does not match checkpoint')
        return self.condition_encoder(history)
