"""Opt-in upper-layer load authority; no autonomous timeout recontact.

load_request is a force-target fraction, not a blend of competing joint
policies. A zero request prohibits inward nominal motion/search in the last
recently measured contact-normal direction. This local linear constraint is
not a guarantee about object motion, other links, or finite-step collisions.
"""
from dataclasses import dataclass
import torch
from batched_fullhand_mcc import BatchedFullHandMCC, _quat_matrix
from normal_constrained_hold import project_box_halfspace


@dataclass(frozen=True)
class LoadAuthorityConfig:
    ramp_frames: int = 20
    normal_max_age_frames: int = 20
    unload_warning_frames: int = 100

    def __post_init__(self):
        for value in (self.ramp_frames, self.normal_max_age_frames, self.unload_warning_frames):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError('Load authority durations must be positive integer control frames')


class LoadAuthorityMCC(BatchedFullHandMCC):
    def __init__(self, batch, device, config=None, authority=None):
        self.authority = authority or LoadAuthorityConfig()
        super().__init__(batch, device, config)

    def reset(self):
        super().reset()
        shape = (self.batch, 4)
        self.load_request = torch.ones(shape, device=self.device)
        self.regulation_load_scale = torch.ones_like(self.load_request)
        self.intent_inward_gain = torch.ones_like(self.load_request)
        self.intent_unloading = torch.zeros(shape, device=self.device, dtype=torch.bool)
        self.intent_touched = torch.zeros_like(self.intent_unloading)
        self.intent_request_missing = False
        self.intent_age = torch.zeros(shape, device=self.device, dtype=torch.long)
        self.intent_normal_age = torch.full_like(self.intent_age, self.authority.normal_max_age_frames + 1)
        self.intent_normal_world = torch.zeros((*shape, 3), device=self.device)
        self.measured_contact = None
        self.use_matched_contact = False
        self.matched_guard_all = False
        self.compensate_palm_motion = False
        self.palm_twist_world = None

    def set_palm_motion(self, twist):
        if twist.shape != (self.batch,6) or twist.device != self.load_request.device or not torch.isfinite(twist).all():
            raise ValueError('Published palm twist must be finite [B,6] on controller device')
        self.palm_twist_world = twist

    def set_measured_contact(self, found, point, inward):
        """Matched fingertip point + normal (e.g. tactile reconstruction).

        Never pass netforce.normal: MJWarp deliberately fills it with a fixed
        world-axis placeholder. Non-tip collision data must not enter here.
        """
        if (found.shape != (self.batch,4) or found.dtype != torch.bool
                or point.shape != (self.batch,4,3) or inward.shape != point.shape
                or any(x.device != self.load_request.device for x in (found,point,inward))
                or not point.is_floating_point() or not inward.is_floating_point()):
            raise ValueError('Matched fingertip measurement requires Boolean [B,4] and floating [B,4,3] point/normal on device')
        self.measured_contact = (found, point, inward)

    def prepare_load_authority(self, request):
        # An absent producer never silently revokes an already-issued unload.
        # A fresh/reset controller starts at full load; resuming requires an
        # explicit request. Missing input is separately recorded.
        self.intent_request_missing = request is None
        if request is not None:
            if (request.shape != self.load_request.shape or request.device != self.load_request.device
                    or not request.is_floating_point() or not torch.isfinite(request).all()
                    or bool(((request < 0) | (request > 1)).any())):
                raise ValueError('load_request must be finite floating [batch,4] in [0,1] on controller device')
            self.load_request = request.to(self.load_request).clone()
        self.intent_unloading = self.load_request == 0
        self.intent_age = torch.where(self.intent_unloading, self.intent_age + 1, 0)
        self.intent_touched |= self.load_request < 1
        rate = 1. / self.authority.ramp_frames
        self.regulation_load_scale += (self.load_request - self.regulation_load_scale).clamp(-rate, rate)
        # Suppress inward authority immediately on unloading; give it back
        # only as fast as the force-target ramp. Outward relief is unscaled.
        self.intent_inward_gain = torch.minimum(self.load_request, self.regulation_load_scale)

    def update(self, **inputs):
        previous = self.previous_command
        measured_active = torch.zeros_like(self.intent_unloading)
        guard_active = torch.zeros_like(measured_active)
        if self.use_matched_contact:
            if self.measured_contact is None:
                raise ValueError('Matched-contact mode requires an explicit fingertip measurement')
            found, point, inward = self.measured_contact
            measured_active = self.intent_touched
            guard_active = torch.ones_like(measured_active) if self.matched_guard_all else measured_active
            valid_measurement = found & torch.isfinite(point).all(-1) & torch.isfinite(inward).all(-1) & (inward.norm(dim=-1)>1e-6)
            inputs = dict(inputs)
            # Inactive fingers retain exact baseline behavior. Only the
            # authority-controlled fingers use matched geometry in the servo.
            normal = torch.where(valid_measurement[...,None], -inward, 0.)
            inputs['collision_normals_outward_world'] = torch.where(guard_active[...,None], normal,
                inputs['collision_normals_outward_world'])
            inputs['collision_points_world'] = torch.where((guard_active & valid_measurement)[...,None],
                point, inputs['contact_points_world'])
            inputs['contact_points_world'] = torch.where((measured_active & valid_measurement)[...,None],
                point, inputs['contact_points_world'])
            inputs['surface_normals_world'] = torch.where((measured_active & valid_measurement)[...,None],
                normal, inputs['surface_normals_world'])
        command, diagnostic = super().update(**inputs)
        q = inputs['q_live'].to(command)
        pose = inputs['palm_pose_world'].to(command)
        # Only measured contact normals; no mesh/privileged non-tip feedback.
        normal = inputs.get('collision_normals_outward_world')
        if normal is None:
            normal = torch.zeros_like(self.intent_normal_world)
        normal = normal.to(command)
        magnitude = normal.norm(dim=-1, keepdim=True)
        capture = inputs['found'].bool() & torch.isfinite(normal).all(-1) & (magnitude[..., 0] > 1e-6)
        if self.use_matched_contact:
            # Keep only actual measured normals in the cache, including before
            # intervention. Parent force feedback/DP observations stay intact.
            found, point, inward = self.measured_contact
            normal = -inward.to(command)
            magnitude = normal.norm(dim=-1, keepdim=True)
            capture = found & torch.isfinite(point).all(-1) & torch.isfinite(normal).all(-1) & (magnitude[...,0]>1e-6)
        self.intent_normal_world = torch.where(capture[..., None],
            normal / magnitude.clamp_min(1e-12), self.intent_normal_world)
        self.intent_normal_age = torch.where(capture, 0,
            (self.intent_normal_age + 1).clamp_max(self.authority.normal_max_age_frames + 1))
        valid = self.intent_normal_age <= self.authority.normal_max_age_frames
        local = torch.einsum('nij,nfj->nfi', _quat_matrix(pose).transpose(-1, -2), self.intent_normal_world)
        sites, _, origins, rotations, jac = self.fk.contact_kinematics(q, self.anchor, self.anchor_valid)
        jn = torch.einsum('nfi,nfij->nfj', local, jac)
        rate = self.config.action_rate_limit
        lo = (self._clamp_q(q-rate)-q).clamp(-rate, rate).reshape(-1, 4, 4)
        hi = (self._clamp_q(q+rate)-q).clamp(-rate, rate).reshape(-1, 4, 4)
        previous = q if previous is None else previous
        prev_delta = (previous-q).reshape(-1, 4, 4)
        both_lo, both_hi = torch.maximum(lo, prev_delta-rate), torch.minimum(hi, prev_delta+rate)
        incompatible = (both_lo > both_hi).any(-1)
        lo = torch.where(incompatible[..., None], lo, both_lo)
        hi = torch.where(incompatible[..., None], hi, both_hi)
        palm_normal_step = torch.zeros_like(self.load_request)
        guard_feasible = torch.ones_like(self.intent_unloading)
        guard_enabled = torch.zeros_like(self.intent_unloading)
        if self.compensate_palm_motion:
            if not self.use_matched_contact or self.palm_twist_world is None:
                raise ValueError('Palm compensation requires matched normal and explicit published twist')
            point_palm = torch.where(self.anchor_valid[...,None],
                origins+torch.einsum('nfij,nfj->nfi',rotations,self.anchor), sites)
            lever = torch.einsum('nij,nfj->nfi',_quat_matrix(pose),point_palm)
            velocity = self.palm_twist_world[:,None,:3]+torch.linalg.cross(
                self.palm_twist_world[:,None,3:].expand_as(lever),lever,dim=-1)
            palm_normal_step=(velocity*self.intent_normal_world).sum(-1)*self.config.control_dt
            hard_ratio=self._finger_constant('hard_ratio',self.config.overforce_hard_ratio,self.config.thumb_overforce_hard_ratio)
            raw_force=diagnostic['raw_force_magnitude']
            hard=raw_force>=hard_ratio*self.force_setpoint
            guard_enabled=capture & ((raw_force>=self.config.solver_contact_safety_force_ratio*self.force_setpoint)|hard)
            clearance=torch.where(hard,self.config.solver_contact_escape_step,0.)
            target=(command-q).reshape(-1,4,4)
            # Absolute/world normal motion = palm contribution + J*dq.
            # Never increase the actuator bound to make a constraint feasible.
            safe,guard_feasible=project_box_halfspace(target,jn,lo,hi,clearance-palm_normal_step)
            command=torch.where(guard_enabled[...,None],q.reshape(-1,4,4)+safe,
                                command.reshape(-1,4,4)).reshape(-1,16)
        # Preserve the parent's outward response (including normal force
        # regulation), while allowing the requested tangential/posture motion.
        parent_normal = (jn * (command-q).reshape(-1, 4, 4)).sum(-1)
        if getattr(self, 'trace_authority', False):
            diagnostic.update(intent_after_motion_guard_q=command.clone(),
                              intent_normal_jacobian=jn.clone(),
                              intent_cached_outward_world=self.intent_normal_world.clone(),
                              intent_raw_tip_force_n=diagnostic['raw_force_magnitude'].clone(),
                              intent_lower_delta_q=lo.clone(), intent_upper_delta_q=hi.clone())
        minimum = parent_normal.clamp_min(0)
        minimum = torch.where(diagnostic['collision_safety_active'],
            torch.maximum(minimum, torch.full_like(minimum, self.config.solver_contact_escape_step)), minimum)
        if self.compensate_palm_motion:
            minimum=torch.maximum(parent_normal, -palm_normal_step)
        desired = (inputs['nominal_posture_q'].to(q)-q).reshape(-1, 4, 4)
        projected, feasible = project_box_halfspace(desired, jn, lo, hi, minimum)
        enabled = self.intent_unloading & valid
        missing = self.intent_unloading & ~valid
        command = torch.where(enabled[..., None], q.reshape(-1, 4, 4)+projected,
                              command.reshape(-1, 4, 4))
        # Unknown surface direction: bounded measured-position hold, explicitly
        # marked unavailable, rather than fabricate a retreat or resume search.
        hold_delta = torch.maximum(torch.minimum(torch.zeros_like(lo), hi), lo)
        unknown_hold = missing & ~diagnostic['collision_safety_active']
        command = torch.where(unknown_hold[..., None], q.reshape(-1, 4, 4)+hold_delta, command).reshape(-1, 16)
        handback = self.intent_touched & ~self.intent_unloading & ~diagnostic['collision_safety_active'] & ~guard_enabled
        limited = previous + (command-previous).clamp(-rate, rate)
        # Keep both actuator tracking and command slew bounds when compatible.
        limited = torch.maximum(torch.minimum(limited-q, hi.reshape(-1,16)), lo.reshape(-1,16)) + q
        command = torch.where(handback.repeat_interleave(4, -1), limited, command)
        self.intent_touched &= (self.load_request < 1) | (self.regulation_load_scale < 1)
        self.previous_command = command.clone()
        diagnostic.update(
            intent_load_request=self.load_request.clone(),
            intent_load_applied=self.regulation_load_scale.clone(),
            intent_force_target=self.force_setpoint*self.regulation_load_scale,
            intent_inward_gain=self.intent_inward_gain.clone(),
            intent_unloading=self.intent_unloading.clone(),
            intent_unload_warning=self.intent_age >= self.authority.unload_warning_frames,
            intent_request_missing=torch.full_like(self.intent_unloading,self.intent_request_missing),
            intent_normal_age=self.intent_normal_age.clone(),
            intent_missing_normal=missing,
            intent_projection_active=enabled,
            intent_projection_infeasible=enabled & ~feasible,
            intent_slew_conflict=self.intent_touched & incompatible,
            intent_applied_normal_m=(jn*(command-q).reshape(-1,4,4)).sum(-1),
            intent_parent_normal_m=parent_normal,
            intent_matched_geometry_active=measured_active,
            intent_matched_guard_active=guard_active,
            intent_palm_normal_step_m=palm_normal_step,
            intent_motion_guard_active=guard_enabled,
            intent_motion_guard_infeasible=guard_enabled & ~guard_feasible,
            intent_world_normal_step_m=palm_normal_step+(jn*(command-q).reshape(-1,4,4)).sum(-1),
        )
        return command, diagnostic
