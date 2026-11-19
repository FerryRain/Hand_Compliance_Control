"""Differentiable four-fingertip FK from the exact fixed-hand MCC model."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn

from surface_mcc_finger import HAND_JOINT_NAMES, TIP_NAMES, _fixed_hand_model


class TorchHandFK(nn.Module):
    """Map action-order 16-D joint q to four MCC fingertip sites in palm frame.

    This intentionally supports only the current Leap Hand topology: four
    independent chains, four hinge joints each, hinge pivot at body origin.
    Fail on geometry changes rather than silently training against wrong FK.
    """

    def __init__(self) -> None:
        super().__init__()
        model = _fixed_hand_model()
        body_positions = np.empty((4, 4, 3), dtype=np.float32)
        body_rotations = np.empty((4, 4, 3, 3), dtype=np.float32)
        joint_axes = np.empty((4, 4, 3), dtype=np.float32)
        site_positions = np.empty((4, 3), dtype=np.float32)
        site_rotations = np.empty((4, 3, 3), dtype=np.float32)
        palm_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "palm_lower"
        )
        for finger, site_name in enumerate(TIP_NAMES):
            parent_id = palm_id
            for joint_index in range(4):
                name = HAND_JOINT_NAMES[4 * finger + joint_index]
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                bid = int(model.jnt_bodyid[jid])
                if int(model.body_parentid[bid]) != parent_id:
                    raise ValueError(f"{name}: unexpected hand-chain parent")
                if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError(f"{name}: expected a hinge joint")
                if np.linalg.norm(model.jnt_pos[jid]) > 1.0e-8:
                    raise ValueError(f"{name}: nonzero joint pivot is unsupported")
                body_positions[finger, joint_index] = model.body_pos[bid]
                body_rotations[finger, joint_index] = Rotation.from_quat(
                    model.body_quat[bid][[1, 2, 3, 0]]
                ).as_matrix()
                joint_axes[finger, joint_index] = model.jnt_axis[jid]
                parent_id = bid
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if int(model.site_bodyid[sid]) != parent_id:
                raise ValueError(f"{site_name}: site is not on the distal body")
            site_positions[finger] = model.site_pos[sid]
            site_rotations[finger] = Rotation.from_quat(
                model.site_quat[sid][[1, 2, 3, 0]]
            ).as_matrix()
        self.register_buffer("body_positions", torch.from_numpy(body_positions))
        self.register_buffer("body_rotations", torch.from_numpy(body_rotations))
        self.register_buffer("joint_axes", torch.from_numpy(joint_axes))
        self.register_buffer("site_positions", torch.from_numpy(site_positions))
        self.register_buffer("site_rotations", torch.from_numpy(site_rotations))

    @staticmethod
    def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
        axis = axis / torch.linalg.vector_norm(axis)
        x, y, z = axis.unbind()
        zero = torch.zeros_like(x)
        skew = torch.stack(
            (
                torch.stack((zero, -z, y)),
                torch.stack((z, zero, -x)),
                torch.stack((-y, x, zero)),
            )
        )
        eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
        cosine = torch.cos(angle)[..., None, None]
        sine = torch.sin(angle)[..., None, None]
        outer = axis[:, None] * axis[None, :]
        return cosine * eye + (1.0 - cosine) * outer + sine * skew

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != 16:
            raise ValueError(f"expected q[...,16], got {tuple(q.shape)}")
        # FK remains in FP32 under mixed-precision training; gradients still
        # flow through the cast back to the denoiser's output.
        value = q.float()
        prefix = value.shape[:-1]
        batch = value.reshape(-1, 4, 4)
        result = []
        for finger in range(4):
            rotation = torch.eye(3, dtype=value.dtype, device=value.device).expand(
                len(batch), 3, 3
            )
            position = torch.zeros((len(batch), 3), dtype=value.dtype, device=value.device)
            for joint in range(4):
                offset = self.body_positions[finger, joint]
                position = position + torch.einsum("bij,j->bi", rotation, offset)
                rotation = (
                    rotation
                    @ self.body_rotations[finger, joint]
                    @ self._axis_angle(
                        self.joint_axes[finger, joint], batch[:, finger, joint]
                    )
                )
            tip = position + torch.einsum(
                "bij,j->bi", rotation, self.site_positions[finger]
            )
            result.append(tip)
        return torch.stack(result, dim=1).reshape(*prefix, 4, 3)

    def contact_kinematics(
        self,
        q: torch.Tensor,
        anchor_body_local: torch.Tensor | None = None,
        anchor_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched site/body poses and 3x4 contact-point Jacobians.

        Returns site positions, site rotations, distal-body origins, distal-body
        rotations and Jacobians in the palm frame. No environment loop or CPU
        readback is used.
        """
        if q.ndim != 2 or q.shape[1] != 16:
            raise ValueError(f"expected [N,16], got {tuple(q.shape)}")
        value = q.float().reshape(-1, 4, 4)
        batch = value.shape[0]
        sites, site_rots, origins, body_rots, jacobians = [], [], [], [], []
        for finger in range(4):
            rotation = torch.eye(3, dtype=value.dtype, device=value.device).expand(batch, 3, 3)
            position = torch.zeros((batch, 3), dtype=value.dtype, device=value.device)
            joint_origins, joint_axes = [], []
            for joint in range(4):
                position = position + torch.einsum("bij,j->bi", rotation, self.body_positions[finger, joint])
                fixed_rotation = rotation @ self.body_rotations[finger, joint]
                joint_origins.append(position)
                joint_axes.append(torch.einsum("bij,j->bi", fixed_rotation, self.joint_axes[finger, joint]))
                rotation = fixed_rotation @ self._axis_angle(
                    self.joint_axes[finger, joint], value[:, finger, joint]
                )
            site = position + torch.einsum("bij,j->bi", rotation, self.site_positions[finger])
            if anchor_body_local is not None and anchor_valid is not None:
                anchor = position + torch.einsum(
                    "bij,bj->bi", rotation, anchor_body_local[:, finger]
                )
                point = torch.where(anchor_valid[:, finger, None], anchor, site)
            else:
                point = site
            jacobian = torch.stack(
                [torch.linalg.cross(axis, point - origin) for axis, origin in zip(joint_axes, joint_origins)],
                dim=-1,
            )
            sites.append(site)
            site_rots.append(rotation @ self.site_rotations[finger])
            origins.append(position)
            body_rots.append(rotation)
            jacobians.append(jacobian)
        return (torch.stack(sites, 1), torch.stack(site_rots, 1),
                torch.stack(origins, 1), torch.stack(body_rots, 1),
                torch.stack(jacobians, 1))
