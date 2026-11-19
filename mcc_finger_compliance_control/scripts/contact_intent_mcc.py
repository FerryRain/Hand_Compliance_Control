"""Explicit per-finger unload/recontact handover, opt-in over legacy MCC.

No intention is inferred from force, joint signs, or lost contact. An upper
layer supplies a Boolean unload request AND its bounded joint target. This
module arbitrates authority; it is not a learned retreat or recovery policy.
"""
from dataclasses import dataclass
import torch
from batched_fullhand_mcc import BatchedFullHandMCC, _quat_matrix
from normal_constrained_hold import project_box_halfspace


@dataclass(frozen=True)
class HandoverConfig:
    max_unload_frames: int = 50
    max_recontact_frames: int = 100
    confirm_frames: int = 4
    ramp_frames: int = 20
    contact_force_n: float = .05

    def __post_init__(self):
        if min(self.max_unload_frames, self.max_recontact_frames,
               self.confirm_frames, self.ramp_frames) < 1 or self.contact_force_n <= 0:
            raise ValueError('Handover durations and contact force must be positive')


class ContactIntentMCC(BatchedFullHandMCC):
    NORMAL, UNLOAD, RECONTACT, BLEND = range(4)

    def __init__(self, batch, device, config=None, handover=None):
        self.handover = handover or HandoverConfig()
        super().__init__(batch, device, config)

    def reset(self):
        super().reset()
        shape = (self.batch, 4)
        self.intent_phase = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.intent_age = torch.zeros_like(self.intent_phase)
        self.intent_recontact_age = torch.zeros_like(self.intent_phase)
        self.intent_loaded_streak = torch.zeros_like(self.intent_phase)
        self.intent_blocked = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.intent_timeout = torch.zeros_like(self.intent_blocked)
        self.intent_unloading = torch.zeros_like(self.intent_blocked)
        self.intent_inward_gain = torch.ones(shape, device=self.device)
        self.intent_touched = torch.zeros_like(self.intent_blocked)
        self.intent_normal_world = torch.zeros((*shape, 3), device=self.device)
        self.intent_normal_valid = torch.zeros_like(self.intent_blocked)

    def prepare_contact_intent(self, request, found, force):
        cfg = self.handover
        if request is None:
            request = torch.zeros_like(found, dtype=torch.bool)
        if request.shape != found.shape or request.dtype != torch.bool or request.device != found.device:
            raise ValueError('unload_request must be Boolean [batch,4] on controller device')
        old = self.intent_phase.clone()
        self.intent_timeout.zero_()
        self.intent_blocked &= request
        self.intent_age = torch.where(old != self.NORMAL, self.intent_age + 1, 0)
        self.intent_recontact_age = torch.where(old >= self.RECONTACT, self.intent_recontact_age + 1, 0)
        loaded = found & torch.isfinite(force).all(-1) & (force.norm(dim=-1) >= cfg.contact_force_n)
        self.intent_loaded_streak = torch.where(loaded & (old >= self.RECONTACT), self.intent_loaded_streak + 1, 0)
        start = request & ~self.intent_blocked & (old == self.NORMAL)
        self.intent_touched |= start
        self.intent_phase = torch.where(start, self.UNLOAD, self.intent_phase)
        end_unload = (old == self.UNLOAD) & (~request | (self.intent_age >= cfg.max_unload_frames))
        unload_timeout = end_unload & request
        self.intent_blocked |= unload_timeout
        self.intent_phase = torch.where(end_unload, self.RECONTACT, self.intent_phase)
        stable = (old == self.RECONTACT) & (self.intent_loaded_streak >= cfg.confirm_frames)
        self.intent_phase = torch.where(stable, self.BLEND, self.intent_phase)
        lost_again = (old == self.BLEND) & ~loaded
        self.intent_phase = torch.where(lost_again, self.RECONTACT, self.intent_phase)
        done = (old == self.BLEND) & loaded & (self.intent_age >= cfg.ramp_frames)
        failed = (old >= self.RECONTACT) & ~done & (self.intent_recontact_age >= cfg.max_recontact_frames)
        self.intent_phase = torch.where(done | failed, self.NORMAL, self.intent_phase)
        self.intent_blocked |= (done | failed) & request
        self.intent_timeout = unload_timeout | failed
        changed = self.intent_phase != old
        self.intent_age = torch.where(changed, 0, self.intent_age)
        self.intent_unloading = self.intent_phase == self.UNLOAD
        # Recontact and blending restart at zero inward servo gain. Ordinary
        # outward force relief is never scaled. Returning to NORMAL restores
        # legacy recovery; a timeout is explicitly reported, never success.
        ramp = (self.intent_recontact_age.float() / cfg.ramp_frames).clamp(0, 1)
        self.intent_inward_gain = torch.where(self.intent_phase == self.NORMAL, 1.,
            torch.where(self.intent_unloading, 0., ramp))

    def update(self, **inputs):
        previous = self.previous_command
        command, diagnostic = super().update(**inputs)
        q = inputs['q_live'].to(command)
        pose = inputs['palm_pose_world'].to(command)
        normal = inputs.get('collision_normals_outward_world')
        if normal is None:
            normal = inputs['surface_normals_world']
        normal = normal.to(command)
        norm = normal.norm(dim=-1, keepdim=True)
        valid = torch.isfinite(normal).all(-1) & (norm[..., 0] > 1e-6)
        # Remember the last measured direction through a short deliberate
        # loss; do not fabricate a new normal when the sensor has no contact.
        capture = inputs['found'].bool() & valid
        self.intent_normal_world = torch.where(capture[..., None], normal / norm.clamp_min(1e-12), self.intent_normal_world)
        self.intent_normal_valid |= capture
        local = torch.einsum('nij,nfj->nfi', _quat_matrix(pose).transpose(-1, -2), self.intent_normal_world)
        _, _, _, _, jac = self.fk.contact_kinematics(q, self.anchor, self.anchor_valid)
        jn = torch.einsum('nfi,nfij->nfj', local, jac)
        rate = self.config.action_rate_limit
        lo = (self._clamp_q(q-rate)-q).clamp(-rate, rate).reshape(-1, 4, 4)
        hi = (self._clamp_q(q+rate)-q).clamp(-rate, rate).reshape(-1, 4, 4)
        if previous is None:
            previous = q
        prev_delta = (previous-q).reshape(-1, 4, 4)
        both_lo, both_hi = torch.maximum(lo, prev_delta-rate), torch.minimum(hi, prev_delta+rate)
        incompatible = (both_lo > both_hi).any(-1)
        lo = torch.where(incompatible[..., None], lo, both_lo)
        hi = torch.where(incompatible[..., None], hi, both_hi)
        hard_ratio = self._finger_constant('hard_ratio', self.config.overforce_hard_ratio, self.config.thumb_overforce_hard_ratio)
        hard = inputs['found'].bool() & (diagnostic['raw_force_magnitude'] >= hard_ratio * self.force_setpoint)
        minimum = torch.where(hard, self.config.solver_contact_escape_step, 0.)
        desired = (inputs['nominal_posture_q'].to(q)-q).reshape(-1, 4, 4)
        projected, feasible = project_box_halfspace(desired, jn, lo, hi, minimum)
        enabled = self.intent_unloading & self.intent_normal_valid
        # Override whole-finger hold only during explicit UNLOAD, not on every
        # overforce tick as in the earlier normal_constrained_hold_v1 probe.
        command = torch.where(enabled[..., None], q.reshape(-1, 4, 4)+projected,
                              command.reshape(-1, 4, 4)).reshape(-1, 16)
        # Slew limit hand-back after an explicit intervention. Keep hard-force
        # safety authority; incompatible tracking bounds are logged above.
        handback = self.intent_touched & ~self.intent_unloading & ~diagnostic['collision_safety_active']
        limited = previous + (command-previous).clamp(-rate, rate)
        command = torch.where(handback.repeat_interleave(4, -1), limited, command)
        self.intent_touched &= self.intent_phase != self.NORMAL
        self.previous_command = command.clone()
        diagnostic.update(intent_phase=self.intent_phase.clone(), intent_timeout=self.intent_timeout.clone(),
            intent_inward_gain=self.intent_inward_gain.clone(), intent_projection_active=enabled,
            intent_projection_infeasible=enabled & ~feasible,
            intent_missing_normal=self.intent_unloading & ~self.intent_normal_valid,
            intent_slew_conflict=enabled & incompatible,
            intent_applied_normal_m=(jn*(command-q).reshape(-1, 4, 4)).sum(-1))
        return command, diagnostic
