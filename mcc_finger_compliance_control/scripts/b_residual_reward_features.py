"""Measured B/221 reward geometry; same strict pad rules as serial replay."""

from __future__ import annotations

import torch

from batched_fullhand_mcc import _quat_matrix
from torch_hand_fk import TorchHandFK


class BRewardGeometry:
    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self.fk = TorchHandFK().to(self.device).eval()
        self.x_limit = torch.tensor((0.012, 0.012, 0.012, 0.016), device=device)

    @torch.no_grad()
    def extract(self, q: torch.Tensor, pose: torch.Tensor, points: torch.Tensor,
                normals: torch.Tensor, found: torch.Tensor, force_n: torch.Tensor,
                *, normals_are_palm: bool = False, include_pad_pose: bool = False) -> dict[str, torch.Tensor]:
        sites, rotations, _, _, _ = self.fk.contact_kinematics(q)
        palm_inv = _quat_matrix(pose).transpose(-1, -2)
        points_palm = torch.einsum("nij,nfj->nfi", palm_inv, points - pose[:, None, :3])
        local = torch.einsum("nfij,nfj->nfi", rotations.transpose(-1, -2), points_palm - sites)
        normals_palm = normals if normals_are_palm else torch.einsum("nij,nfj->nfi", palm_inv, normals)
        angle, normal_valid = self.angle(rotations, normals_palm)
        strict = self.strict(local, found, force_n)
        result = {"strict": strict, "angle_deg": angle,
                "normal_valid": normal_valid & found & (force_n >= 0.05),
                "local_x_m": local[..., 0], "local_radius_m": torch.linalg.vector_norm(local, dim=-1),
                "points_palm": points_palm, "normals_palm": normals_palm}
        if include_pad_pose:
            result.update(pad_sites_palm=sites, pad_outward_palm=-rotations[..., :, 0])
        return result

    def strict(self, local: torch.Tensor, found: torch.Tensor, force_n: torch.Tensor) -> torch.Tensor:
        return (found & torch.isfinite(local).all(-1)
                & (torch.linalg.vector_norm(local, dim=-1) <= 0.05)
                & (local[..., 0] <= self.x_limit) & (force_n >= 0.05))

    @torch.no_grad()
    def prior_support(self, q: torch.Tensor, reference_q: torch.Tensor,
                      points_palm: torch.Tensor, normals_palm: torch.Tensor,
                      found: torch.Tensor, force_n: torch.Tensor,
                      margin_deg: torch.Tensor, scale_deg: torch.Tensor,
                      radius_margin: torch.Tensor, x_margin: torch.Tensor,
                      low_force_margin: torch.Tensor) -> dict[str, torch.Tensor]:
        """Necessary static support for an Oracle recovery prior, not safety proof."""
        delta = reference_q - q
        delta = delta * (.02 / delta.abs().amax(-1).clamp_min(1e-9)).clamp_max(1)[:, None]
        costs, geometric_costs, flags, normal_masks = [], [], [], []
        for value in (q, q + delta):
            sites, rotations, _, _, _ = self.fk.contact_kinematics(value)
            angle, valid = self.angle(rotations, normals_palm)
            valid &= found & (force_n >= .05)
            excess = ((angle - margin_deg) / scale_deg).clamp_min(0).square()
            cost = excess / (1 + excess)
            costs.append((cost * valid).sum(-1) / valid.sum(-1).clamp_min(1))
            normal_masks.append(valid)
            local = torch.einsum("nfij,nfj->nfi", rotations.transpose(-1, -2), points_palm - sites)
            flags.append(self.strict(local, found, force_n))
            radius = torch.linalg.vector_norm(local, dim=-1)
            position = torch.maximum(
                ((radius - radius_margin) / (.05 - radius_margin)).clamp(0, 1).square(),
                ((local[..., 0] - x_margin) / (self.x_limit - x_margin)).clamp(0, 1).square())
            unload = ((low_force_margin - force_n) / (low_force_margin - .05)).clamp(0, 1).square()
            geometric_costs.append(torch.maximum(cost * valid, torch.maximum(position, unload)).mean(-1))
        change = costs[1] - costs[0]
        geometric_change = geometric_costs[1] - geometric_costs[0]
        loses_contact = (flags[0] & ~flags[1]).any(-1)
        support = normal_masks[0].any(-1) & (change <= 1e-5) & (geometric_change < -1e-5) & ~loses_contact
        return {"support": support, "angle_cost_change": change,
                "geometry_cost_change": geometric_change, "loses_contact": loses_contact}

    @staticmethod
    def angle(rotations: torch.Tensor, normals_palm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm = torch.linalg.vector_norm(normals_palm, dim=-1)
        valid = torch.isfinite(normals_palm).all(-1) & (norm > 0.5)
        unit = torch.nan_to_num(normals_palm) / norm.nan_to_num().clamp_min(1e-9)[..., None]
        pad_outward = -rotations[..., :, 0]
        cosine = (pad_outward * unit).sum(-1).clamp(-1, 1)
        return torch.rad2deg(torch.acos(cosine)), valid


class HeldAngleCost:
    """Keep last measurable angle cost during missing normals.

    Missing tactile information cannot reset badness to zero and earn shaping
    reward. This memory, its validity and age must be included in reward-state
    snapshots and exposed to the critic. It is not a geometric estimate while
    contact is absent.
    """

    def __init__(self, batch: int, margin_deg: torch.Tensor, scale_deg: torch.Tensor):
        self.margin = margin_deg
        self.scale = scale_deg
        self.cost = torch.zeros(batch, 4, device=margin_deg.device)
        self.age = torch.zeros(batch, 4, dtype=torch.long, device=margin_deg.device)

    def update(self, angle_deg: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        excess = ((angle_deg - self.margin) / self.scale).clamp_min(0).square()
        measured = excess / (1 + excess)
        self.cost = torch.where(valid, measured, self.cost)
        self.age = torch.where(valid, 0, self.age + 1)
        return self.cost.clone()


class HeldContactMarginCost:
    """Continuous pad-edge and unloading warning, used only in potential.

    The low-force boundary is a healthy-reference quantile, not MCC's force
    setpoint. No task reward asks RL to servo to a particular normal force.
    Invalid/unloaded contact positions are held; force loss increases badness.
    """

    def __init__(self, batch: int, calibration: dict, device: torch.device | str):
        self.radius = torch.tensor(calibration["pad_radius_margin_m"], device=device)
        self.x = torch.tensor(calibration["pad_x_margin_m"], device=device)
        self.low = torch.tensor(calibration["loaded_force_lower_margin_n"], device=device)
        self.x_limit = torch.tensor((.012, .012, .012, .016), device=device)
        self.position_cost = torch.zeros(batch, 4, device=device)

    def update(self, local_x: torch.Tensor, radius: torch.Tensor,
               found: torch.Tensor, force_n: torch.Tensor) -> torch.Tensor:
        valid = found & (force_n >= .05) & torch.isfinite(radius) & torch.isfinite(local_x)
        radius_cost = ((radius - self.radius) / (.05 - self.radius).clamp_min(.001)).clamp(0, 1).square()
        x_cost = ((local_x - self.x) / (self.x_limit - self.x).clamp_min(.001)).clamp(0, 1).square()
        measured = torch.maximum(radius_cost, x_cost)
        self.position_cost = torch.where(valid, measured, self.position_cost)
        unload_cost = ((self.low - force_n) / (self.low - .05).clamp_min(.05)).clamp(0, 1).square()
        return torch.maximum(self.position_cost, unload_cost)
