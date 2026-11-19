"""Frozen DP / smooth exploration action sources for the WM collector.

This module owns pre-MCC references only. It does not instantiate a second MCC,
search for missing contacts, reset physics, or provide recovery supervision.
The physical object still moves; the DP sees the equivalent object-frame palm
motion, including an already-published future command (never future measurements).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat, quat_apply_inverse, subtract_frame_transforms,
)


def relative_pose(parent, child):
    p, q = subtract_frame_transforms(parent[:, :3], parent[:, 3:],
                                     child[:, :3], child[:, 3:])
    return torch.cat((p, q), -1)


def relative_dp_inputs(palm, object_actual, object_future_command, points, normals,
                       previous_virtual_palm=None):
    """Virtual stationary object frame; wxyz poses and spatial twist.

    Current pose/contact use measured state. Future pose uses a published
    object command and the fixed palm, so a future measurement cannot leak in.
    Stored contact memory must also live in this virtual frame on contact loss.
    """
    from palm_support import measured_pose_difference_twist
    virtual = relative_pose(object_actual, palm)
    future = relative_pose(object_future_command, palm)
    delta = relative_pose(virtual, future)
    command = torch.cat((delta[:, :3], axis_angle_from_quat(delta[:, 3:])), -1)
    rotation = object_actual[:, None, 3:].expand(-1, points.shape[1], -1)
    point = quat_apply_inverse(rotation, points - object_actual[:, None, :3])
    normal = quat_apply_inverse(rotation, normals)
    twist = measured_pose_difference_twist(virtual, previous_virtual_palm, .01)
    return virtual, point, normal, twist, command


class SmoothPerturbation:
    """Independent seeded bounded joint offsets; smooth endpoints every .5s.

    Directions alternate single-finger and all-active-finger excitation. This
    is exploration, not a claim that a random direction is a recovery action.
    Zero amplitude is exact identity and consumes no policy sampling noise.
    """
    def __init__(self, jobs, active, frames, amplitude, device):
        if not np.isfinite(amplitude) or not 0 <= amplitude <= .08:
            raise ValueError('exploration amplitude must be in [0,.08] rad')
        values = np.zeros((frames, len(jobs), 16), np.float32)
        for lane, job in enumerate(jobs):
            rng = np.random.default_rng(job['seed'] + 59021)
            fingers = np.flatnonzero(active[lane].cpu().numpy())
            previous = np.zeros(16)
            for segment, start in enumerate(range(0, frames, 50)):
                target = np.zeros(16)
                selected = [rng.choice(fingers)] if segment % 2 == 0 else fingers
                for finger in selected:
                    target[4*finger:4*finger+4] = rng.uniform(-amplitude, amplitude, 4)
                count = min(50, frames-start)
                u = np.arange(1, count+1) / 50
                blend = u**3 * (10 - 15*u + 6*u*u)
                values[start:start+count, lane] = previous + blend[:, None]*(target-previous)
                previous = target
        self.values = torch.as_tensor(values, device=device)


class FrozenDPNominal:
    def __init__(self, checkpoint, q_reference, seed):
        from batched_b_control_chain import BatchedBObservation, BatchedDPChunkScheduler
        from deploy_dp_inverse import DPRuntime
        self.device = q_reference.device
        self.batch = len(q_reference)
        self.runtime = r = DPRuntime(Path(checkpoint), self.device, None, seed)
        if (r.state_schema != 'contact_geometry_planner_motion' or r.state_dim != 96
                or r.robot_state_dim != 84 or r.action_dim != 16
                or r.action_representation != 'anchored_delta_q' or r.input_frame != 'palm'
                or r.obs_horizon != 16 or r.stride != 5 or r.pred_horizon not in (8,16)
                or r.planner_waypoints != 1 or r.planner_step_frames != 20
                or r.contact_normal_polarity != 'primary_fingertip_to_object'):
            raise ValueError('Frozen collector requires verified palm-frame 96-D / anchored-delta-q contract')
        r.policy.eval().requires_grad_(False)
        self.observation = BatchedBObservation(self.batch, self.device)
        self.scheduler = BatchedDPChunkScheduler(self.batch, torch.as_tensor(r.state_std[:16], device=self.device))
        self.reference = q_reference.clone()
        self.previous_nominal = q_reference.clone()
        for _ in range(30): self.scheduler.observe(q_reference)
        self.state_mean = torch.as_tensor(r.state_mean, device=self.device)
        self.state_std = torch.as_tensor(r.state_std, device=self.device)
        self.state_mask = torch.as_tensor(r.state_input_mask, device=self.device)
        self.action_mean = torch.as_tensor(r.action_mean, device=self.device)
        self.action_std = torch.as_tensor(r.action_std, device=self.device)
        self.previous_virtual_palm = None
        self.plans = []
        self.metadata = dict(checkpoint=str(Path(checkpoint).resolve()),
            checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
            trainable_parameters=0, inference_steps=r.policy.diffusion.num_inference_steps,
            scheduler=asdict(self.scheduler.config), observation_horizon=16, observation_stride=5,
            bootstrap_frames=76, planner_horizon_frames=20,
            action_representation=r.action_representation, action_sample_clip=r.config.action_sample_clip,
            normals='matched physical contact hand-to-object; no source-mesh normal oracle',
            coordinates='virtual stationary object frame from measured current pose; future from published command',
            limitation='Old policy used only as frozen interaction source; no equivalence of moving-object dynamics to deployment claimed')

    @torch.no_grad()
    def decode(self, history, q):
        normalized = (history-self.state_mean)/self.state_std*self.state_mask
        cond = self.runtime.policy.diffusion._prepare_global_conditioning({
            'observation.state': normalized[:, :, :self.runtime.robot_state_dim],
            'observation.environment_state': normalized[:, :, self.runtime.robot_state_dim:]})
        prediction = self.runtime.policy.diffusion.conditional_sample(
            self.batch, global_cond=cond, generator=self.runtime.generator)
        chunk = prediction*self.action_std + self.action_mean + q[:, None]
        return chunk, prediction, normalized

    @torch.no_grad()
    def step(self, frame, q, palm, object_actual, future_command, points, normals, found):
        virtual, point, normal, twist, command = relative_dp_inputs(
            palm, object_actual, future_command, points, normals, self.previous_virtual_palm)
        self.previous_virtual_palm = virtual.clone()
        state, _ = self.observation.push(q=q, palm_pose_world=virtual,
            contact_points_world=point, contact_normals_inward_world=normal, found=found,
            palm_twist_world=twist, planner_command_palm=command)
        self.scheduler.observe(self.previous_nominal)
        replan = frame >= 75 and (frame-75) % 10 == 0 and self.observation.history_ready
        if replan:
            history = self.observation.history()
            chunk, prediction, normalized = self.decode(history, q)
            if not torch.isfinite(chunk).all():
                raise RuntimeError('nonfinite_physics_or_measurement: frozen DP prediction')
            self.scheduler.install(chunk)
            self.plans.append(dict(frame=np.asarray(frame), history=history.cpu().numpy(),
                normalized=normalized.cpu().numpy(), prediction=prediction.cpu().numpy(),
                chunk=chunk.cpu().numpy(), anchor=q.cpu().numpy(),
                drop=self.scheduler.last_drop_index.cpu().numpy()))
        nominal = self.reference.clone() if frame <= 75 else self.scheduler.next_command()
        self.previous_nominal = nominal.clone()
        return nominal, dict(dp_input_state=state, dp_input_valid=torch.ones(self.batch, dtype=torch.bool, device=self.device),
            dp_virtual_palm_pose=virtual, dp_virtual_palm_twist=twist, dp_planner_command=command,
            dp_replanned=torch.full((self.batch,), replan, device=self.device),
            dp_active=torch.full((self.batch,), frame > 75, device=self.device),
            dp_control_frame=torch.full((self.batch,), frame, device=self.device))

    def save(self, folder):
        if self.plans:
            np.savez_compressed(folder/'dp_plans.npz',
                **{k:np.stack([r[k] for r in self.plans]) for k in self.plans[0]})

