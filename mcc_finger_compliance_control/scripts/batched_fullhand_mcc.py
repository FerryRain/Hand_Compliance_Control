"""Batched CUDA port of the direct-force, contact-Jacobian MCC execution path.

The original CPU controller remains available as the reference and fallback.
This class intentionally rejects unsupported controller modes rather than
silently changing their behavior.
"""

from __future__ import annotations

import numpy as np
import torch

from surface_mcc_finger import FullHandMCCFingerConfig, FullHandMCCFingerController
from torch_hand_fk import TorchHandFK


def _quat_matrix(pose: torch.Tensor) -> torch.Tensor:
    q = pose[:, 3:7]
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


class BatchedFullHandMCC:
    def __init__(self, batch: int, device: torch.device | str,
                 config: FullHandMCCFingerConfig | None = None, *,
                 overforce_takeover: bool = True,
                 continuous_contact_regulation: bool = False):
        self.config = config or FullHandMCCFingerConfig()
        # WM collection may observe overload without threshold-based takeover.
        # Ordinary signed force-error regulation remains active. Deployment
        # keeps the original default; this is not a change to its safety policy.
        self.overforce_takeover = bool(overforce_takeover)
        self.continuous_contact_regulation = bool(continuous_contact_regulation)
        if self.continuous_contact_regulation and self.overforce_takeover:
            raise ValueError('WM continuous regulation requires threshold takeover disabled')
        cfg = self.config
        if not cfg.use_direct_force_servo:
            raise NotImplementedError("admittance/Mink IK mode is not batched")
        if cfg.project_nominal_normal_motion or cfg.normal_synergy_control:
            raise NotImplementedError("normal projection/synergy modes need separate parity validation")
        self.device = torch.device(device)
        self.batch = batch
        reference = FullHandMCCFingerController(cfg)
        self.fk = TorchHandFK().to(self.device).eval()
        self.lower = torch.as_tensor(reference.lower, device=self.device, dtype=torch.float32)
        self.upper = torch.as_tensor(reference.upper, device=self.device, dtype=torch.float32)
        self.flexion_indices = torch.tensor((2,3,6,7,10,11,14,15),device=self.device,dtype=torch.long)
        self.flexion_columns = torch.tensor((0,2,3),device=self.device,dtype=torch.long)
        self._constants = {}
        # The serial controller stores the CLAMPED closure posture
        # (surface_mcc_finger.py:1415), which bites here: DEFAULT_GRASP_CLOSURE_Q
        # has joint 9 at -0.5 rad, below natural_flexion_floor (-0.30).  Taking
        # the raw tuple would give the control-normal fallback a different grasp
        # axis than the serial run from the very first frame.
        self.grasp = self._clamp_q(
            torch.as_tensor(reference.grasp_closure_q, device=self.device,
                            dtype=torch.float32).reshape(1, 16)
        ).reshape(4, 4)
        self.open = torch.as_tensor(reference.open_grasp_q, device=self.device, dtype=torch.float32).reshape(4, 4)
        self.force_setpoint = torch.as_tensor(reference.force_setpoint, device=self.device, dtype=torch.float32).expand(batch, 4).clone()
        self.reset()

    def reset(self) -> None:
        if hasattr(self, 'force_command_diagnostic'):
            del self.force_command_diagnostic
        n, device = self.batch, self.device
        self.anchor = torch.zeros((n, 4, 3), device=device)
        self.anchor_valid = torch.zeros((n, 4), device=device, dtype=torch.bool)
        self.force_filtered = torch.zeros((n, 4), device=device)
        self.filter_valid = torch.zeros((n, 4), device=device, dtype=torch.bool)
        self.contact_active = torch.zeros((n, 4), device=device, dtype=torch.bool)
        self.force_offset = torch.zeros((n, 4), device=device)
        self.hold = torch.zeros((n, 4), device=device, dtype=torch.int32)
        self.contact_phase = torch.zeros((n, 4), device=device, dtype=torch.int32)
        self.loss_streak = torch.zeros((n, 4), device=device, dtype=torch.int32)
        self.recontact_streak = torch.zeros((n, 4), device=device, dtype=torch.int32)
        self.transient_offset = torch.zeros((n, 4), device=device)
        self.previous_command = None
        self.reference_q = None
        self.previous_nominal = None

    def _clamp_q(self, q: torch.Tensor) -> torch.Tensor:
        result = torch.maximum(torch.minimum(q, self.upper), self.lower)
        if self.config.natural_flexion_floor is not None:
            result = result.clone()
            result.index_copy_(1,self.flexion_indices,
                result.index_select(1,self.flexion_indices).clamp_min(self.config.natural_flexion_floor))
        return result

    def _constant(self,name,values):
        # Configuration values are refreshed when changed; no per-frame
        # pageable CPU->CUDA scalar/vector copy for a fixed controller.
        key=(name,values)
        if key not in self._constants:
            self._constants[key]=torch.as_tensor(values,device=self.device,dtype=torch.float32)
        return self._constants[key]

    def _finger_constant(self,name,value,thumb=None):
        return self._constant(name,(value,)*3+(value if thumb is None else thumb,))

    def _update_anchors(self, points_palm: torch.Tensor, mask: torch.Tensor,
                        sites: torch.Tensor, origins: torch.Tensor,
                        body_rots: torch.Tensor) -> None:
        plausible = torch.isfinite(points_palm).all(-1) & (
            torch.linalg.vector_norm(points_palm - sites, dim=-1)
            <= self.config.contact_anchor_max_site_distance
        )
        capture = mask & plausible
        local = torch.einsum("nfij,nfj->nfi", body_rots.transpose(-1, -2), points_palm - origins)
        self.anchor = torch.where(capture[..., None], local, self.anchor)
        self.anchor_valid |= capture

    @torch.no_grad()
    def directional_search_delta(self, *, q_live: torch.Tensor,
                                 palm_pose_world: torch.Tensor,
                                 inward_directions_world: torch.Tensor,
                                 missing: torch.Tensor,
                                 inward_step: float,
                                 max_joint_step: float) -> torch.Tensor:
        """Batched port of ``FullHandMCCFingerController.directional_search_delta``.

        For every finger flagged ``missing``, advance its control point along the
        selected inward axis by ``inward_step`` and realise that Cartesian step as
        the minimum-norm joint step ``J^T (J J^T + lambda I)^-1 target``, clipped to
        ``max_joint_step``.  Fingers that still have contact get an exact zero, so
        the recovery integrator only moves the fingers that lost contact.

        The serial routine also refreshes its contact-point anchors first; that
        update only re-anchors the Jacobian and does not change the search axis,
        so it is left to the caller's normal anchor path.
        """
        q = q_live.to(device=self.device, dtype=torch.float32)
        pose = palm_pose_world.to(device=self.device, dtype=torch.float32)
        mask = missing.to(device=self.device).bool()
        rot = _quat_matrix(pose)
        directions = torch.einsum(
            "nij,nfj->nfi", rot.transpose(-1, -2),
            inward_directions_world.to(device=self.device, dtype=torch.float32))
        directions = directions / torch.linalg.vector_norm(
            directions, dim=-1, keepdim=True).clamp_min(1.0e-12)
        _, _, _, _, jacobian = self.fk.contact_kinematics(q)          # (n,4,3,4)
        target = float(inward_step) * directions                      # (n,4,3)
        lhs = (jacobian @ jacobian.transpose(-1, -2)
               + 1.0e-5 * torch.eye(3, device=self.device, dtype=jacobian.dtype))
        solved = torch.linalg.solve(lhs, target[..., None])[..., 0]   # (n,4,3)
        correction = torch.einsum("nfji,nfj->nfi", jacobian, solved)  # (n,4,4)
        delta = correction.clamp(-float(max_joint_step), float(max_joint_step))
        return torch.where(mask[..., None], delta, torch.zeros_like(delta))

    def update(self, *, q_live: torch.Tensor, palm_pose_world: torch.Tensor,
               force_world: torch.Tensor, found: torch.Tensor,
               surface_points_world: torch.Tensor, surface_normals_world: torch.Tensor,
               nominal_posture_q: torch.Tensor,
               contact_points_world: torch.Tensor | None = None,
               collision_normals_outward_world: torch.Tensor | None = None,
               collision_points_world: torch.Tensor | None = None,
               force_magnitude_only: bool = True,
               use_contact_point_jacobian: bool = True,
               manage_contact_state: bool = True,
               contact_observed: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        if not force_magnitude_only or not use_contact_point_jacobian:
            raise NotImplementedError("only the deployed magnitude/contact-Jacobian path is batched")
        consistent_command = getattr(self, 'force_command_consistency', False)
        incremental_command = getattr(self, 'incremental_force_compliance', False)
        if consistent_command and incremental_command:
            raise ValueError('Choose one collection force-command interface')
        if (consistent_command or incremental_command) and (self.overforce_takeover or not self.continuous_contact_regulation
                                   or cfg.nominal_joint_correction_limit is not None):
            raise ValueError('Force-consistent command is a collection-only continuous mode')
        if contact_points_world is None:
            raise ValueError("contact_points_world required")
        q = q_live.to(device=self.device, dtype=torch.float32)
        pose = palm_pose_world.to(device=self.device, dtype=torch.float32)
        force = force_world.to(device=self.device, dtype=torch.float32)
        raw_found = found.to(device=self.device, dtype=torch.bool)
        nominal = nominal_posture_q.to(device=self.device, dtype=torch.float32)
        if q.shape != (self.batch, 16) or raw_found.shape != (self.batch, 4):
            raise ValueError("invalid MCC batch shapes")
        rot = _quat_matrix(pose)
        to_palm = lambda p: torch.einsum("nij,nfj->nfi", rot.transpose(-1, -2), p - pose[:, None, :3])
        vector_palm = lambda v: torch.einsum("nij,nfj->nfi", rot.transpose(-1, -2), v)
        points_palm = to_palm(contact_points_world)
        sites, site_rot, origins, body_rot, _ = self.fk.contact_kinematics(q)
        local_tip = torch.einsum("nfij,nfj->nfi", site_rot.transpose(-1, -2), points_palm - sites)
        pad_valid = (raw_found & torch.isfinite(local_tip).all(-1)
                     & (torch.linalg.vector_norm(local_tip, dim=-1) <= cfg.contact_anchor_max_site_distance)
                     & (local_tip[..., 0] <= self._constant('back_contact_x',tuple(cfg.back_contact_x_limit_m))))
        found_valid = raw_found & pad_valid
        if self.continuous_contact_regulation:
            # One bounded compliance state, across both loss and recontact.
            # The old transient search accumulator was released faster than
            # the ordinary force servo could replace it on first touch.
            self.transient_offset.zero_()
            self.loss_streak=torch.where(found_valid,0,self.loss_streak+1)
            self.recontact_streak=torch.where(found_valid,self.recontact_streak+1,0)
            self.contact_phase=torch.where(found_valid,0,1)
        elif cfg.enable_loss_state_machine and manage_contact_state:
            state_found = found_valid if contact_observed is None else (contact_observed.bool() & pad_valid)
            missing = ~state_found
            self.loss_streak = torch.where(missing, self.loss_streak + 1, 0)
            self.recontact_streak = torch.where(state_found, self.recontact_streak + 1, 0)
            self.contact_phase = torch.where(missing, torch.where(
                self.loss_streak <= cfg.transient_loss_frames, 1, 2), self.contact_phase)
            self.contact_phase = torch.where(state_found & (self.contact_phase != 0), 3, self.contact_phase)
            self.contact_phase = torch.where(state_found & (self.recontact_streak >= cfg.recovery_contact_confirm_frames), 0, self.contact_phase)
            self.transient_offset = torch.where(self.contact_phase == 1,
                self.transient_offset + cfg.transient_search_step, self.transient_offset)
            release = (self.contact_phase == 0) | state_found
            self.transient_offset = torch.where(release,
                (self.transient_offset - cfg.transient_release_step).clamp_min(0), self.transient_offset)
        normals = vector_palm(surface_normals_world)
        normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True).clamp_min(1e-12)
        raw_force = torch.linalg.vector_norm(force, dim=-1)
        hard_ratio = self._finger_constant('hard_ratio',cfg.overforce_hard_ratio,cfg.thumb_overforce_hard_ratio)
        collision_overforce = raw_found & (raw_force >= hard_ratio * self.force_setpoint)
        if not self.overforce_takeover:
            collision_overforce = torch.zeros_like(collision_overforce)
            self.hold.zero_()
        regulation_measurement=getattr(self,'regulation_measurement',None)
        measurement = torch.where(found_valid,
            raw_force if regulation_measurement is None else regulation_measurement, 0)
        alpha = float(np.clip(cfg.force_filter_alpha, 0, 1))
        self.force_filtered = torch.where(self.filter_valid,
            alpha * measurement + (1 - alpha) * self.force_filtered, measurement)
        self.filter_valid.fill_(True)
        self.contact_active = torch.where(self.contact_active,
            self.force_filtered >= cfg.contact_off_force,
            self.force_filtered >= cfg.contact_on_force)
        # Optional upper-layer load authority affects regulation only. Hard
        # force/safety thresholds continue to use the frozen nominal setpoint.
        regulation_scale = getattr(self, 'regulation_load_scale', None)
        regulation_target = (self.force_setpoint if regulation_scale is None
                             else self.force_setpoint * regulation_scale)
        force_error = regulation_target - self.force_filtered
        if cfg.force_servo_min_force is not None and cfg.force_servo_max_force is not None:
            band_min = cfg.force_servo_min_force
            band_max = cfg.force_servo_max_force
            if regulation_scale is not None:
                band_min = band_min * regulation_scale
                band_max = band_max * regulation_scale
            effective = torch.where(self.force_filtered < band_min,
                band_min - self.force_filtered,
                torch.where(self.force_filtered > band_max,
                    band_max - self.force_filtered, 0))
        else:
            effective = torch.where(force_error.abs() <= cfg.force_servo_deadband, 0, force_error)
        step = (cfg.force_servo_integral_gain * effective * cfg.control_dt).clamp(
            -cfg.force_servo_max_step, cfg.force_servo_max_step)
        missing = ~found_valid
        weak = found_valid & ~self.contact_active & (self.hold == 0)
        if self.continuous_contact_regulation:
            # Collection is not a deployment recovery controller. No contact
            # means return the bounded compliance displacement to zero and
            # track the fixed pregrasp reference. Do not treat zero load in
            # free space as an instruction to search deeper into the object.
            neutral_step=(-self.force_offset).clamp(-cfg.force_servo_max_step,
                                                   cfg.force_servo_max_step)
            step=torch.where(found_valid,step,neutral_step)
        elif cfg.enable_loss_state_machine:
            step = torch.where(missing, 0, step)
            step = torch.where(weak, torch.maximum(step, self._constant('weak_contact_step',cfg.force_servo_weak_contact_step)), step)
            self.force_filtered = torch.where(missing, 0, self.force_filtered)
            self.filter_valid = torch.where(missing, False, self.filter_valid)
        else:
            held = missing & (self.hold > 0)
            search = missing & ~held
            search_step = self._finger_constant('search_step',cfg.force_servo_search_step,cfg.thumb_force_servo_search_step)
            step = torch.where(held, 0, step)
            step = torch.where(search, torch.maximum(step, search_step), step)
            step = torch.where(weak, torch.maximum(step, self._constant('weak_contact_step',cfg.force_servo_weak_contact_step)), step)
            self.force_filtered = torch.where(search, 0, self.force_filtered)
            self.filter_valid = torch.where(search, False, self.filter_valid)
        self.hold = torch.where(collision_overforce, cfg.overforce_recontact_hold_frames,
                                (self.hold - 1).clamp_min(0))
        safety = collision_overforce | (self.hold > 0)
        if getattr(self, 'external_final_arbitration', False):
            # The opt-in unified executor owns the hold/retreat handover.
            # Retain the ordinary force integrator, including its hard-force
            # outward increment, but do not replace nominal with live q here.
            self.hold.zero_()
            safety = torch.zeros_like(safety)
        step = torch.where(safety, torch.minimum(step, torch.zeros_like(step)), step)
        hard_step = self._finger_constant('hard_step',cfg.force_servo_hard_step,cfg.thumb_force_servo_hard_step)
        step = torch.where(collision_overforce, torch.minimum(step, -hard_step), step)
        # Opt-in handover protocol. Never attenuate outward overforce relief.
        intent_gain = getattr(self, 'intent_inward_gain', None)
        if intent_gain is not None:
            step = torch.where(step > 0, step * intent_gain, step)
            unload = self.intent_unloading
            self.force_offset = torch.where(unload, self.force_offset.clamp_max(0), self.force_offset)
            self.transient_offset = torch.where(unload, 0., self.transient_offset)
        lower_offset = self._finger_constant('lower_offset',-cfg.max_normal_offset,
            None if cfg.thumb_max_outward_offset is None else -cfg.thumb_max_outward_offset)
        if self.continuous_contact_regulation:
            outward=cfg.force_servo_max_outward_offset
            outward=cfg.max_normal_offset if outward is None else outward
            lower_offset=self._finger_constant('wm_lower_offset',-outward,
                None if cfg.thumb_max_outward_offset is None else -cfg.thumb_max_outward_offset)
        upper_offset = self._finger_constant('upper_offset',cfg.max_normal_offset,cfg.thumb_max_inward_offset)
        lower_offset = torch.where(safety, -max(cfg.max_normal_offset, cfg.overforce_max_offset), lower_offset)
        offset_before = self.force_offset.clone() if consistent_command else None
        self.force_offset = torch.maximum(torch.minimum(self.force_offset + step, upper_offset), lower_offset)
        total_offset = self.force_offset + self.transient_offset + cfg.nominal_surface_preload
        if intent_gain is not None:
            total_offset = torch.where(total_offset > 0, total_offset * intent_gain, total_offset)
        self._update_anchors(points_palm, found_valid, sites, origins, body_rot)
        _, _, _, _, jac = self.fk.contact_kinematics(q, self.anchor, self.anchor_valid)
        normal_jac = torch.einsum("nfi,nfij->nfj", normals, jac)
        # Optional collection-only joint mobility: weighted minimum-norm
        # normal correction. Zero mobility excludes that joint from force
        # regulation, without constraining the physical joint or nominal DP.
        # Keep the default arithmetic path exactly as before.
        mobility = getattr(self, 'normal_joint_mobility', None)
        if mobility is None:
            normal_direction = normal_jac
            denom = normal_jac.square().sum(-1) + cfg.contact_point_jacobian_regularization
        else:
            normal_direction = normal_jac * mobility
            denom = (normal_jac * normal_direction).sum(-1) + cfg.contact_point_jacobian_regularization
        if incremental_command:
            from contact_wm_incremental_compliance import incremental_command as advance
            previous = q if self.previous_command is None else self.previous_command
            normal_step = (-cfg.force_servo_integral_gain * effective * cfg.control_dt).clamp(
                -cfg.force_servo_max_step, cfg.force_servo_max_step)
            command, self.force_command_diagnostic = advance(
                previous.reshape(-1,4,4), nominal.reshape(-1,4,4), normal_jac,
                torch.ones_like(normal_jac) if mobility is None else mobility,
                normal_step, found_valid,
                self._clamp_q(self.lower.expand(self.batch,-1)).reshape(-1,4,4),
                self.upper.reshape(1,4,4),
                cfg.action_rate_limit * float(np.clip(cfg.command_ema_alpha,0,1)),
                cfg.flexion_synergy_gain, cfg.flexion_synergy_max_step,
                -lower_offset, cfg.contact_point_jacobian_regularization)
            self.previous_command = command.reshape(-1,16).clone()
            # Diagnostic projection only; the candidate's stored control
            # state is previous_command. No accumulated scalar remapping.
            self.force_offset = -(normal_jac*(command-nominal.reshape(-1,4,4))).sum(-1)
            return self.previous_command.clone(), dict(
                pad_contact_valid=pad_valid, collision_safety_active=safety,
                normal_offset=self.force_offset, contact_phase=self.contact_phase,
                raw_force_magnitude=raw_force)
        normal_distance = -total_offset
        normal_distance = torch.where(safety, torch.where(collision_overforce, hard_step, 0), normal_distance)
        if cfg.project_nominal_normal_motion:
            if self.reference_q is None:
                self.reference_q = q.clone()
                self.previous_nominal = nominal.clone()
            increment = nominal.reshape(-1, 4, 4) - self.previous_nominal.reshape(-1, 4, 4)
            projection = (normal_jac * increment).sum(-1)
            increment = torch.where(found_valid[..., None],
                increment - projection[..., None] * normal_jac / denom[..., None], increment)
            base = self.reference_q.reshape(-1, 4, 4) + increment
        else:
            base = nominal.reshape(-1, 4, 4).clone()
        base = torch.where(safety[..., None], q.reshape(-1, 4, 4), base)
        correction = normal_distance[..., None] * normal_direction / denom[..., None]
        if cfg.normal_synergy_control:
            synergy = (self.grasp - self.open).clone()
            synergy[:, 1] = 0
            per_fraction = (normal_jac * synergy).sum(-1)
            sy_step = normal_distance / torch.where(per_fraction.abs() > 1e-5, per_fraction, 1)
            sy_correction = (sy_step[..., None] * synergy).clamp(-cfg.normal_synergy_max_step, cfg.normal_synergy_max_step)
            correction = torch.where(found_valid[..., None] & (per_fraction.abs() > 1e-5)[..., None], sy_correction, correction)
        command = self._clamp_q((base + correction).reshape(-1, 16))
        if cfg.project_nominal_normal_motion:
            reference = base + correction - normal_distance[..., None] * normal_jac / denom[..., None]
            self.reference_q = self._clamp_q(reference.reshape(-1, 16))
            self.previous_nominal = nominal.clone()
        safety_command = torch.where(safety[..., None], q.reshape(-1, 4, 4), command.reshape(-1, 4, 4))
        command_4 = command.reshape(-1, 4, 4).clone()
        lower = self.lower.reshape(4, 4).index_select(1,self.flexion_columns)
        travel = (self.upper.reshape(4, 4).index_select(1,self.flexion_columns) - lower).clamp_min(1e-6)
        ratio = (command_4.index_select(-1,self.flexion_columns) - lower) / travel
        spread = ratio.amax(-1) - ratio.amin(-1)
        gain = torch.where(spread > cfg.flexion_synergy_spread_threshold,
            cfg.flexion_synergy_hard_gain, cfg.flexion_synergy_gain)
        flex = command_4.index_select(-1,self.flexion_columns)
        flex = flex + (gain[..., None] * (nominal.reshape(-1, 4, 4).index_select(-1,self.flexion_columns) - flex)).clamp(
            -cfg.flexion_synergy_max_step, cfg.flexion_synergy_max_step)
        command_4.index_copy_(-1,self.flexion_columns,flex)
        command_4 = self._clamp_q(command_4.reshape(-1, 16)).reshape(-1, 4, 4)
        command_4 = torch.where(safety[..., None], safety_command, command_4)
        limits = None
        if cfg.nominal_joint_correction_limit is not None:
            thumb_limit = cfg.nominal_joint_correction_limit if cfg.thumb_nominal_joint_correction_limit is None else cfg.thumb_nominal_joint_correction_limit
            limits = self._constant('joint_correction_limits',(cfg.nominal_joint_correction_limit,)*12+(thumb_limit,)*4)
            bounded = nominal + (command_4.reshape(-1, 16) - nominal).clamp(-limits, limits)
            command_4 = torch.where(safety[..., None], command_4, bounded.reshape(-1, 4, 4))
        command = command_4.reshape(-1, 16)
        if self.previous_command is None:
            self.previous_command = q.clone()
        command = self.previous_command + (command - self.previous_command).clamp(-cfg.action_rate_limit, cfg.action_rate_limit)
        command = self.previous_command + float(np.clip(cfg.command_ema_alpha, 0, 1)) * (command - self.previous_command)
        command = torch.where(safety[..., None].expand(-1, -1, 4).reshape(-1, 16),
            q + (safety_command.reshape(-1, 16) - q).clamp(-cfg.action_rate_limit, cfg.action_rate_limit), command)
        if limits is not None:
            bounded = nominal + (command - nominal).clamp(-limits, limits)
            command = torch.where(safety[..., None].expand(-1, -1, 4).reshape(-1, 16), command, bounded)
        trace_before_guard = command.clone() if getattr(self, 'trace_authority', False) else None
        if (self.overforce_takeover and cfg.solver_contact_safety and collision_normals_outward_world is not None
                and not getattr(self, 'external_final_arbitration', False)):
            outward = vector_palm(collision_normals_outward_world)
            norm = torch.linalg.vector_norm(outward, dim=-1, keepdim=True)
            outward = torch.where(norm > 1e-9, outward / norm.clamp_min(1e-9), 0)
            collision_points_palm = points_palm if collision_points_world is None else to_palm(collision_points_world)
            self._update_anchors(collision_points_palm, raw_found, sites, origins, body_rot)
            _, _, _, _, jac = self.fk.contact_kinematics(q, self.anchor, self.anchor_valid)
            jn = torch.einsum("nfi,nfij->nfj", outward, jac)
            denom = jn.square().sum(-1) + cfg.contact_point_jacobian_regularization
            hard = raw_force >= hard_ratio * self.force_setpoint
            engage = raw_found & (norm[..., 0] > 1e-9) & (
                (raw_force >= cfg.solver_contact_safety_force_ratio * self.force_setpoint) | hard)
            delta = command.reshape(-1, 4, 4) - q.reshape(-1, 4, 4)
            step_n = (jn * delta).sum(-1)
            minimum = torch.where(hard, cfg.solver_contact_escape_step, 0)
            extra = torch.where(engage & (step_n < minimum), (minimum - step_n) / denom, 0)
            command = self._clamp_q((q.reshape(-1, 4, 4) + delta + extra[..., None] * jn).reshape(-1, 16))
        if consistent_command:
            from contact_wm_command_projection import project_force_increment
            previous = self.previous_command.reshape(-1, 4, 4)
            delta = command.reshape(-1, 4, 4) - previous
            lower_q = self._clamp_q(self.lower.expand(self.batch, -1)).reshape(-1, 4, 4)
            upper_q = self.upper.reshape(1, 4, 4)
            rate = cfg.action_rate_limit * float(np.clip(cfg.command_ema_alpha, 0, 1))
            lower_delta = (lower_q-previous).clamp_min(-rate)
            upper_delta = (upper_q-previous).clamp_max(rate)
            direction = torch.sign(self.force_filtered-regulation_target)
            desired_normal = offset_before-self.force_offset
            minimum = (desired_normal*direction).clamp_min(0)
            enabled = found_valid & ((self.force_filtered-regulation_target).abs() > cfg.force_servo_deadband)
            weights = torch.ones_like(normal_jac) if mobility is None else mobility
            delta, self.force_command_diagnostic = project_force_increment(
                delta, normal_jac, weights, direction, minimum,
                lower_delta, upper_delta, enabled)
            self.force_command_diagnostic['direction'] = direction
            self.force_command_diagnostic['enabled'] = enabled
            command = (previous+delta).reshape(-1,16)
        self.previous_command = command.clone()
        diagnostic = {
            "pad_contact_valid": pad_valid,
            "collision_safety_active": safety,
            "normal_offset": total_offset,
            "contact_phase": self.contact_phase,
            "raw_force_magnitude": raw_force,
        }
        if trace_before_guard is not None:
            diagnostic['intent_before_geometry_guard_q'] = trace_before_guard
            diagnostic['intent_after_geometry_guard_q'] = command.clone()
        return command, diagnostic
