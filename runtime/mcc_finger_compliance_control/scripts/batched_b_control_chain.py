"""Reusable batched B -> chunk scheduler -> direct-force MCC GPU runtime.

The caller owns MJWarp stepping and supplies the already-published palm
command for each environment. The original single-environment implementation
is preserved for parity checks and unsupported controller modes.
"""

from __future__ import annotations
from pathlib import Path
import torch
from batched_fullhand_mcc import BatchedFullHandMCC, _quat_matrix
from deploy_dp_inverse import DPRuntime
from surface_mcc_finger import FullHandMCCFingerConfig
from dp_chunk_scheduler import DPChunkSchedulerConfig


class ZeroResidualPolicy(torch.nn.Module):
    """Phase-3 interface probe: preserve the scheduler command exactly."""

    def forward(self, q_live: torch.Tensor, nominal_q: torch.Tensor,
                observation_state: torch.Tensor) -> torch.Tensor:
        del q_live, observation_state
        return torch.zeros_like(nominal_q)







class BatchedBObservation:
    """Accept current sensors and published palm command for every environment.

    Layout is q16, contact point12, inward normal12, found4, q velocity16,
    point velocity12, normal rate12, palm twist6, planner command6. The first
    84 dimensions are robot state. History is sampled every five control frames.
    """

    def __init__(self, batch: int, device: torch.device | str, *, stride: int = 5,
                 horizon: int = 16, control_dt: float = 0.01):
        self.batch = batch
        self.device = torch.device(device)
        self.stride = stride
        self.horizon = horizon
        self.control_dt = control_dt
        self.capacity = (horizon - 1) * stride + 1
        self.states = torch.zeros((batch, self.capacity, 96), device=self.device)
        self.motion_q = torch.zeros((batch, stride + 1, 16), device=self.device)
        self.motion_point = torch.zeros((batch, stride + 1, 4, 3), device=self.device)
        self.motion_normal = torch.zeros_like(self.motion_point)
        self.motion_mask = torch.zeros((batch, stride + 1, 4), device=self.device, dtype=torch.bool)
        self.last_point = torch.zeros((batch, 4, 3), device=self.device)
        self.last_normal = torch.zeros_like(self.last_point)
        self.point_valid = torch.zeros((batch, 4), device=self.device, dtype=torch.bool)
        self.normal_valid = torch.zeros_like(self.point_valid)
        self.count = 0
        # Frames pushed since the last clear_history().  `count` drives the
        # velocity features and the buffer roll and must survive a warm-up
        # clear; this counter drives DP-history readiness and must not.
        self.frames_since_clear = 0

    @property
    def history_ready(self) -> bool:
        """True once a full `obs_horizon`-sample history exists again.

        The serial controller gates DP on `len(live_history) == obs_horizon`
        (deploy_dp_inverse.py:2383); after `live_history.clear()` that takes
        (obs_horizon-1)*stride+1 frames to refill.
        """
        return self.frames_since_clear >= self.capacity

    def clear_history(self) -> None:
        """Drop the DP observation history, keeping every contact memory.

        Mirrors `self.live_history.clear()` at the end of the serial warm-up
        (deploy_dp_inverse.py:3471).  That call empties ONLY the deque of
        sampled states: the held point/normal memory, its validity flags, the
        stride-5 motion buffers that feed the velocity features, and the MCC
        direction EMA all survive the warm-up and are inherited by frame 0.
        Clearing them here would hand the first prediction a contact-free
        observation the serial run never sees.
        """
        self.states.zero_()
        self.frames_since_clear = 0

    def push(self, *, q: torch.Tensor, palm_pose_world: torch.Tensor,
             contact_points_world: torch.Tensor, contact_normals_inward_world: torch.Tensor,
             found: torch.Tensor, palm_twist_world: torch.Tensor,
             planner_command_palm: torch.Tensor,
             mesh_oracle=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Push one frame.  Returns ``(state, corrected_inward_normals)``.

        When a source-mesh oracle is supplied the contact normal is corrected
        exactly as the serial `_live_state` does (deploy_dp_inverse.py:1774+):
        the collision point anchors a query onto the undecomposed source mesh,
        the interpolated source normal is taken as the inward direction, and a
        branch is rejected when the proxy point sits more than 8 mm off the
        source surface or when the candidate flips more than 60 deg from the
        remembered branch.  The corrected normal feeds BOTH the DP observation
        and the MCC hybrid direction; the raw sensor normal is left untouched
        for the collision-safety argument.
        """
        q = q.to(self.device)
        pose = palm_pose_world.to(self.device)
        points = contact_points_world.to(self.device)
        normals = contact_normals_inward_world.to(self.device)
        found = found.to(self.device).bool()
        point_now = found & (torch.linalg.vector_norm(points, dim=-1) > 1e-9)
        if mesh_oracle is not None and bool(found.any()):
            # `query_surface_object_frame` returns the OUTWARD source normal;
            # the checkpoint convention is sensor-primary -> object, i.e. inward.
            # the oracle takes (M,3); points arrive as (batch, 4, 3)
            shape = points.shape
            _, outward, lift = mesh_oracle.query_surface_object_frame(
                points.reshape(-1, 3))
            outward = outward.reshape(shape)
            lift = lift.reshape(shape[:-1])
            candidate = -outward
            distance_invalid = lift > 0.008
            branch_invalid = torch.zeros_like(found)
            if bool(self.normal_valid.any()):
                aligned = torch.einsum("nfi,nfi->nf", candidate, self.last_normal)
                branch_invalid = self.normal_valid & (aligned < 0.5)
            rejected = found & (distance_invalid | branch_invalid)
            corrected = torch.where(
                rejected[..., None] & self.normal_valid[..., None],
                self.last_normal,
                torch.where(rejected[..., None],
                            torch.zeros_like(candidate), candidate))
            normals = torch.where(found[..., None], corrected, normals)
        corrected_inward = normals
        normal_now = found & (torch.linalg.vector_norm(normals, dim=-1) > 1e-6)
        self.last_point = torch.where(point_now[..., None], points, self.last_point)
        self.last_normal = torch.where(normal_now[..., None], normals, self.last_normal)
        self.point_valid |= point_now
        self.normal_valid |= normal_now
        points_held = torch.where((~point_now & self.point_valid)[..., None], self.last_point, points)
        normals_held = torch.where((~normal_now & self.normal_valid)[..., None], self.last_normal, normals)
        rotation_inv = _quat_matrix(pose).transpose(-1, -2)
        points_palm = torch.einsum("nij,nfj->nfi", rotation_inv, points_held - pose[:, None, :3])
        normals_palm = torch.einsum("nij,nfj->nfi", rotation_inv, normals_held)
        twist = torch.cat((
            torch.einsum("nij,nj->ni", rotation_inv, palm_twist_world[:, :3]),
            torch.einsum("nij,nj->ni", rotation_inv, palm_twist_world[:, 3:]),
        ), dim=-1)
        self.motion_q[:, :-1] = self.motion_q[:, 1:].clone()
        self.motion_point[:, :-1] = self.motion_point[:, 1:].clone()
        self.motion_normal[:, :-1] = self.motion_normal[:, 1:].clone()
        self.motion_mask[:, :-1] = self.motion_mask[:, 1:].clone()
        self.motion_q[:, -1] = q
        self.motion_point[:, -1] = points_palm
        self.motion_normal[:, -1] = normals_palm
        self.motion_mask[:, -1] = found
        if self.count >= self.stride:
            duration = self.stride * self.control_dt
            q_velocity = (q - self.motion_q[:, 0]) / duration
            valid_motion = found & self.motion_mask[:, 0]
            point_velocity = torch.where(valid_motion[..., None],
                (points_palm - self.motion_point[:, 0]) / duration, 0)
            normal_rate = torch.where(valid_motion[..., None],
                torch.linalg.cross(self.motion_normal[:, 0], normals_palm) / duration, 0)
        else:
            q_velocity = torch.zeros_like(q)
            point_velocity = torch.zeros_like(points_palm)
            normal_rate = torch.zeros_like(normals_palm)
        state = torch.cat((
            q, points_palm.flatten(1), normals_palm.flatten(1), found.float(),
            q_velocity, point_velocity.flatten(1), normal_rate.flatten(1),
            twist, planner_command_palm,
        ), dim=-1)
        if state.shape != (self.batch, 96):
            raise ValueError(f"invalid B state shape {tuple(state.shape)}")
        if self.count == 0:
            self.states[:] = state[:, None]
        else:
            self.states[:, :-1] = self.states[:, 1:].clone()
            self.states[:, -1] = state
        self.count += 1
        self.frames_since_clear += 1
        return state, corrected_inward

    def history(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("push one observation before reading history")
        indices = torch.arange(self.horizon - 1, -1, -1, device=self.device) * self.stride
        return self.states[:, self.capacity - 1 - indices]


class BatchedHybridMCCDirection:
    def __init__(self, batch: int, device: torch.device | str, fk,
                 grasp_q: torch.Tensor, *, memory_frames: int = 20,
                 blend_alpha: float = 0.35):
        self.batch = batch
        self.device = torch.device(device)
        self.fk = fk
        self.grasp_q = grasp_q.to(device=self.device, dtype=torch.float32).reshape(1, 4, 4)
        self.memory_frames = memory_frames
        self.blend_alpha = blend_alpha
        self.reset()

    def reset(self) -> None:
        self.sensor_normal = torch.zeros((self.batch, 4, 3), device=self.device)
        self.sensor_valid = torch.zeros((self.batch, 4), device=self.device, dtype=torch.bool)
        self.sensor_age = torch.full((self.batch, 4), self.memory_frames + 1,
                                     device=self.device, dtype=torch.int32)
        self.previous = torch.zeros((self.batch, 4, 3), device=self.device)
        self.previous_valid = False

    @torch.no_grad()
    def update(self, q: torch.Tensor, palm_pose_world: torch.Tensor,
               found: torch.Tensor, normal_inward_world: torch.Tensor) -> torch.Tensor:
        """Return object-outward vectors in world frame, matching hybrid MCC."""
        q = q.reshape(self.batch, 16)
        _, _, _, _, jac = self.fk.contact_kinematics(q)
        direction = torch.einsum(
            "nfij,nfj->nfi", jac,
            self.grasp_q - q.reshape(self.batch, 4, 4))
        fallback = torch.as_tensor((0.20, 0.0, 0.12, 0.12),
                                   device=self.device, dtype=q.dtype)
        fallback_direction = torch.einsum("nfij,j->nfi", jac, fallback)
        direction = torch.where(
            (torch.linalg.vector_norm(direction, dim=-1) < 1e-7)[..., None],
            fallback_direction, direction)
        direction = direction / torch.linalg.vector_norm(
            direction, dim=-1, keepdim=True).clamp_min(1e-9)
        rotation = _quat_matrix(palm_pose_world)
        closure_inward_world = torch.einsum("nij,nfj->nfi", rotation, direction)
        closure_outward_world = -closure_inward_world
        normal_valid = found.bool() & (
            torch.linalg.vector_norm(normal_inward_world, dim=-1) > 1e-6)
        measured_outward = -normal_inward_world
        self.sensor_normal = torch.where(normal_valid[..., None],
                                         measured_outward, self.sensor_normal)
        self.sensor_valid |= normal_valid
        self.sensor_age += 1
        self.sensor_age = torch.where(normal_valid, 0, self.sensor_age)
        memory = self.sensor_valid & (self.sensor_age <= self.memory_frames)
        raw = torch.where(memory[..., None], self.sensor_normal,
                          closure_outward_world)
        if self.previous_valid:
            raw = (1 - self.blend_alpha) * self.previous + self.blend_alpha * raw
        result = raw / torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-8)
        self.previous = result
        self.previous_valid = True
        return result


class BatchedBControlChain:
    def __init__(self, checkpoint: str | Path, batch: int, device: str | torch.device,
                 *, seed: int = 42, mcc_config: FullHandMCCFingerConfig | None = None,
                 max_safe_force_n: float = 100.0):
        self.device = torch.device(device)
        self.batch = batch
        self.runtime = DPRuntime(Path(checkpoint), self.device, None, seed)
        runtime = self.runtime
        if (runtime.state_schema != "contact_geometry_planner_motion"
                or runtime.state_dim != 96 or runtime.robot_state_dim != 84
                or runtime.action_representation != "anchored_delta_q"
                or runtime.action_dim != 16 or runtime.pred_horizon not in (8, 16)
                or runtime.obs_horizon != 16 or runtime.stride != 5
                or runtime.planner_waypoints != 1 or runtime.input_frame != "palm"):
            raise ValueError("checkpoint must use the verified B 96-D contract with8 or16 future waypoints")
        self.observation = BatchedBObservation(batch, self.device,
            stride=runtime.stride, horizon=runtime.obs_horizon,
            control_dt=float(getattr(runtime.config, "control_dt", 0.01)))
        self.scheduler = BatchedDPChunkScheduler(batch,
            torch.as_tensor(runtime.state_std[:16], device=self.device))
        self.mcc = BatchedFullHandMCC(batch, self.device, mcc_config)
        self.direction = BatchedHybridMCCDirection(
            batch, self.device, self.mcc.fk, self.mcc.grasp)
        self.state_mean = torch.as_tensor(runtime.state_mean, device=self.device)
        self.state_std = torch.as_tensor(runtime.state_std, device=self.device)
        self.state_mask = torch.as_tensor(runtime.state_input_mask, device=self.device)
        self.action_mean = torch.as_tensor(runtime.action_mean, device=self.device)
        self.action_std = torch.as_tensor(runtime.action_std, device=self.device)
        # Values are the serial `collection_matched_sensor` preset plus the
        # MCCPrecontactConfig defaults it does not override.
        cfg = self.mcc.config
        self.runtime_loss_frames = int(getattr(cfg, "runtime_loss_frames", 2))
        self.recovery_confirm_frames = int(getattr(cfg, "recovery_confirm_frames", 4))
        self.runtime_recovery_limit_rad = float(
            getattr(cfg, "runtime_recovery_limit_rad", 0.04))
        self.recovery_offset_decay = float(getattr(cfg, "recovery_offset_decay", 0.999))
        self.recovery_decay_force_ratio = float(
            getattr(cfg, "recovery_decay_force_ratio", 1.5))
        self.search_cartesian_step_m = float(getattr(cfg, "cartesian_step_m", 0.00015))
        self.search_joint_step_rad = float(getattr(cfg, "joint_step_rad", 0.02))
        self.settle_force_ratio = float(getattr(cfg, "settle_force_ratio", 0.80))
        self.force_threshold = float(getattr(cfg, "force_threshold", 0.10))
        self.max_safe_force_n = max_safe_force_n
        # Serial deploy_dp_inverse.py runs ONE loop; `t <= bootstrap_end` is a
        # branch inside it, not a separate phase.  During bootstrap the nominal
        # comes from the caller's smoothstep blend and the chunk scheduler is
        # only observed, never asked for a command.
        self.bootstrap_end = (runtime.obs_horizon - 1) * runtime.stride
        # Serial warm-up: `--initial-settle-steps` control frames are run
        # through the SAME loop with the posture command pinned to q_hand[0];
        # only the frame counter, the DP history and the metrics are held back.
        self.settle_remaining = 0
        self.settle_nominal = None
        self.settle_steps = 0
        self.control_frame = 0
        self.primed = False
        self.previous_nominal = None
        # Optional BatchedDPTrace.  Left at None for every ordinary run so the
        # deployed command path stays bit-identical; the two recording calls
        # below are the only places the hot loop can observe the DP interface.
        self.trace = None
        # Optional TorchMeshNormalOracle.  With one attached the DP observation
        # and the MCC hybrid direction share the source-mesh corrected normal,
        # while the collision-safety argument keeps the raw sensor normal.
        self.mesh_oracle = None
        # A residual changes only the command entering MCC. The scheduler
        # history continues to observe its own unmodified nominal command.
        self.residual_policy: torch.nn.Module | None = None
        # Optional absolute-q chunk source for Oracle calibration. The default
        # B sampler and all scheduler/MCC timing remain unchanged.
        self.chunk_provider = None
        # Opt-in asynchronous inference; default preserves the accepted chain.
        self.async_planner = None
        # --- recovery orchestrator state (port of deploy_dp_inverse.py:2751-2876) ---
        self.loss_streak = torch.zeros((batch, 4), device=self.device, dtype=torch.long)
        self.recovery_active = torch.zeros((batch, 4), device=self.device, dtype=torch.bool)
        self.recovery_confirm_streak = torch.zeros((batch, 4), device=self.device, dtype=torch.long)
        self.runtime_recovery_offset = torch.zeros((batch, 16), device=self.device)

    def begin_settle(self, q_start: torch.Tensor, steps: int) -> None:
        """Arm the warm-up phase the serial `--initial-settle-steps` provides."""
        self.settle_steps = int(steps)
        self.settle_remaining = int(steps)
        self.settle_nominal = q_start.to(device=self.device, dtype=torch.float32).clone()

    def prime(self, q: torch.Tensor, repeats: int = 30) -> None:
        """Populate scheduler history from a settled measured posture."""
        if repeats < 1:
            raise ValueError("repeats must be positive")
        for _ in range(repeats):
            self.scheduler.observe(q)
        self.previous_nominal = q.clone()
        self.primed = True

    @torch.no_grad()
    def _orchestrate(self, *, q: torch.Tensor, pose: torch.Tensor,
                     force: torch.Tensor, found: torch.Tensor,
                     control_outward_world: torch.Tensor,
                     nominal: torch.Tensor) -> torch.Tensor:
        """Loss/recovery state machine, ported from the serial live_dp loop.

        The serial deployment does not hand the scheduler's nominal straight to
        MCC.  It tracks a per-finger loss streak, and once a finger has been
        missing for ``runtime_loss_frames`` it starts a Cartesian re-contact
        search along the inward axis whose result is integrated into
        ``runtime_recovery_offset`` (clamped to +-``runtime_recovery_limit_rad``)
        and added to the nominal posture.  A finger leaves recovery only after
        ``recovery_confirm_frames`` consecutive *loaded* contacts, and the
        integrator is slowly bled off while a finger carries its target force.

        Without this layer a lost contact is never actively re-established,
        which is why the batched rollout keeps far less contact than the serial
        one even when both run identical DP output.
        """
        cfg = self.mcc.config
        force_mag = torch.linalg.vector_norm(force, dim=-1)
        force_setpoint = self.mcc.force_setpoint
        # hybrid direction source (the deployed default) counts only a force at
        # or above settle_force_ratio * setpoint as a re-established contact.
        settle_force = torch.maximum(
            torch.full_like(force_setpoint, self.force_threshold),
            self.settle_force_ratio * force_setpoint)
        real_contact = found & (force_mag >= settle_force)

        self.loss_streak = torch.where(
            found, torch.zeros_like(self.loss_streak), self.loss_streak + 1)
        newly_lost = (~self.recovery_active) & (self.loss_streak >= self.runtime_loss_frames)
        self.recovery_active = self.recovery_active | newly_lost
        self.recovery_confirm_streak = torch.where(
            self.recovery_active & real_contact,
            self.recovery_confirm_streak + 1,
            torch.zeros_like(self.recovery_confirm_streak))
        recovered = self.recovery_active & (
            self.recovery_confirm_streak >= self.recovery_confirm_frames)
        self.recovery_active = self.recovery_active & ~recovered
        self.loss_streak = torch.where(
            recovered, torch.zeros_like(self.loss_streak), self.loss_streak)

        force_supported = ((~self.recovery_active) & found
                           & (force_mag >= self.recovery_decay_force_ratio * force_setpoint))
        self.runtime_recovery_offset = torch.where(
            force_supported.repeat_interleave(4, dim=-1),
            self.runtime_recovery_offset * self.recovery_offset_decay,
            self.runtime_recovery_offset)

        base_nominal = self.mcc._clamp_q(nominal)
        if bool(self.recovery_active.any()):
            search = self.mcc.directional_search_delta(
                q_live=q, palm_pose_world=pose,
                inward_directions_world=-control_outward_world,
                missing=self.recovery_active,
                inward_step=self.search_cartesian_step_m,
                max_joint_step=self.search_joint_step_rad)
            # search is (batch,4,4) per finger; the integrator is (batch,16)
            self.runtime_recovery_offset = (
                self.runtime_recovery_offset + search.reshape(self.batch, 16)).clamp(
                    -self.runtime_recovery_limit_rad, self.runtime_recovery_limit_rad)
        return self.mcc._clamp_q(base_nominal + self.runtime_recovery_offset)

    @torch.no_grad()
    def step(self, *, q: torch.Tensor, palm_pose_world: torch.Tensor,
             contact_force_world: torch.Tensor, contact_found: torch.Tensor,
             contact_points_world: torch.Tensor,
             contact_normals_inward_world: torch.Tensor,
             palm_twist_world: torch.Tensor,
             planner_command_palm: torch.Tensor,
             contact_observed: torch.Tensor | None = None,
             bootstrap_nominal: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Return the `[N,16]` actuator action for one 10 ms control frame."""
        if not self.primed:
            raise RuntimeError("prime with the settled q before deployment")
        q = q.to(self.device)
        if hasattr(self.mcc.fk,'begin_frame'):
            self.mcc.fk.begin_frame()
        self.projection_palm_twist = palm_twist_world.to(self.device)
        pose = palm_pose_world.to(self.device)
        force = contact_force_world.to(self.device)
        found = contact_found.to(self.device).bool()
        points = contact_points_world.to(self.device)
        inward = contact_normals_inward_world.to(self.device)
        if (q.shape != (self.batch, 16) or pose.shape != (self.batch, 7)
                or force.shape != (self.batch, 4, 3) or found.shape != (self.batch, 4)
                or points.shape != (self.batch, 4, 3) or inward.shape != (self.batch, 4, 3)
                or palm_twist_world.shape != (self.batch, 6)
                or planner_command_palm.shape != (self.batch, 6)):
            raise ValueError("invalid batched B control input shapes")
        force_max = torch.linalg.vector_norm(force, dim=-1).max()
        # The serial deployment gates this on `not settling`
        # (deploy_dp_inverse.py:2282): the warm-up step-0 interpenetration
        # transient is a legitimate ~976 N reading that the settle frames
        # absorb, so it must not trip the safety stop.
        if (self.max_safe_force_n > 0 and self.settle_remaining == 0
                and float(force_max.item()) > self.max_safe_force_n):
            raise RuntimeError(f"contact force safety stop: {force_max.item():.2f} N")
        _, corrected_inward = self.observation.push(
            q=q, palm_pose_world=pose,
            contact_points_world=points, contact_normals_inward_world=inward,
            found=found, palm_twist_world=palm_twist_world.to(self.device),
            planner_command_palm=planner_command_palm.to(self.device),
            mesh_oracle=self.mesh_oracle)
        # The accepted deployment observes the previously issued nominal
        # trajectory before installing a new chunk, not the MCC-corrected q.
        self.scheduler.observe(self.previous_nominal)
        settling = self.settle_remaining > 0
        if settling:
            self.settle_remaining -= 1
            if self.settle_remaining == 0:
                # Serial: the warm-up frames are excluded from DP history and
                # metrics, and the transient filter/recovery state is dropped
                # while the offset that established the loaded grasp is kept
                # (deploy_dp_inverse.py:3457-3476).
                self.observation.clear_history()
                self.runtime_recovery_offset.zero_()
                self.recovery_active.zero_()
                self.loss_streak.zero_()
                self.recovery_confirm_streak.zero_()
                # Serial sets this to the measured posture, not None
                # (deploy_dp_inverse.py:3473).
                self.mcc.previous_command = q.clone()
                print(f"[INIT] contact settle complete; starting trajectory "
                      f"frame 0 after {self.settle_steps} fixed steps", flush=True)
        bootstrapping = self.control_frame <= self.bootstrap_end
        # Serial plans AND installs on frame bootstrap_end itself while still
        # executing the bootstrap nominal (deploy_dp_inverse.py:2378-2390), so
        # the gate is `>=`, and planning is decoupled from the nominal source.
        replanned = (
            not settling
            and self.control_frame >= self.bootstrap_end
            and (self.control_frame - self.bootstrap_end)
            % self.scheduler.config.replan_interval == 0
            and self.observation.history_ready)
        if settling:
            # Serial pins the posture command to clamp(bootstrap_q), which is
            # clamp(q_hand[0]) for every warm-up frame (t stays 0).
            nominal = self.mcc._clamp_q(self.settle_nominal)
        else:
            if self.async_planner is not None:
                if self.chunk_provider is not None:
                    raise ValueError('Async DP cannot be combined with an alternate chunk provider')
                requested = replanned
                # Consume a completed result before trying to submit another
                # request. A result may arrive on ANY control frame.
                plan = self.async_planner.poll(self.control_frame)
                replanned = plan is not None
                if plan is not None:
                    drop = self.scheduler.install(plan.chunk,
                        execution_frames=self.async_planner.buffer_frames)
                    self.async_planner.installed(self.control_frame,plan,drop)
                    if self.trace is not None:
                        self.trace.record_plan(frame=self.control_frame,
                            history=plan.history,normalized=plan.normalized,
                            prediction=plan.prediction,chunk=plan.chunk,anchor=plan.anchor,
                            drop_index=drop,path=self.scheduler.path,observation=plan.observation)
                if requested:
                    history = self.observation.history()
                    normalized = (history-self.state_mean)/self.state_std*self.state_mask
                    self.async_planner.submit(self.control_frame,q,history,normalized,
                        self.observation.states if self.trace is not None else None)
                    # Only the explicit inline parity probe can install a
                    # newly submitted plan on this same frame.
                    if self.async_planner.inline:
                        plan = self.async_planner.poll(self.control_frame)
                        replanned = plan is not None
                        if plan is not None:
                            drop = self.scheduler.install(plan.chunk,
                                execution_frames=self.async_planner.buffer_frames)
                            self.async_planner.installed(self.control_frame,plan,drop)
                            if self.trace is not None:
                                self.trace.record_plan(frame=self.control_frame,
                                    history=plan.history,normalized=plan.normalized,
                                    prediction=plan.prediction,chunk=plan.chunk,anchor=plan.anchor,
                                    drop_index=drop,path=self.scheduler.path,observation=plan.observation)
            elif replanned:
                history = self.observation.history()
                normalized = (history - self.state_mean) / self.state_std * self.state_mask
                if self.chunk_provider is None:
                    condition = self.runtime.policy.diffusion._prepare_global_conditioning({
                        "observation.state": normalized[:, :, :self.runtime.robot_state_dim],
                        "observation.environment_state": normalized[:, :, self.runtime.robot_state_dim:],
                    })
                    prediction = self.runtime.policy.diffusion.conditional_sample(
                        self.batch, global_cond=condition, generator=self.runtime.generator)
                    chunk = prediction * self.action_std + self.action_mean + q[:, None]
                else:
                    chunk = self.chunk_provider(self.control_frame, q, history)
                    if chunk.shape != (self.batch, self.runtime.pred_horizon, 16) or not torch.isfinite(chunk).all():
                        raise ValueError("chunk provider must return finite [batch,H,16] absolute q")
                    prediction = (chunk - q[:, None] - self.action_mean) / self.action_std
                self.scheduler.install(chunk)
                if self.trace is not None:
                    self.trace.record_plan(
                        frame=self.control_frame, history=history, normalized=normalized,
                        prediction=prediction, chunk=chunk, anchor=q,
                        drop_index=self.scheduler.last_drop_index, path=self.scheduler.path,
                        observation=self.observation.states.clone())
            if bootstrapping:
                nominal = (bootstrap_nominal.to(self.device).to(torch.float32)
                           if bootstrap_nominal is not None else q.clone())
            elif self.async_planner is not None and self.scheduler.path is None:
                # First request is in flight: continue the last bootstrap
                # nominal, with normal live MCC feedback and moving physics.
                nominal = self.previous_nominal.clone()
            else:
                nominal = self.scheduler.next_command()
        self.previous_nominal = nominal.clone()
        if not settling and not bootstrapping and self.residual_policy is not None:
            residual = self.residual_policy(q, nominal, self.observation.states[:, -1])
            if residual.shape != nominal.shape or not torch.isfinite(residual).all():
                raise ValueError("residual policy must return finite [batch,16] joint offsets")
            nominal = nominal + residual
        # Serial feeds the hybrid direction the CORRECTED normal, not the raw
        # sensor one (deploy_dp_inverse.py:2485 uses `live_normals`, which is
        # the mesh-oracle output).
        independent_admittance = getattr(self.mcc, 'independent_normal_admittance', False)
        control_normal = (-corrected_inward if independent_admittance else
                          self.direction.update(q, pose, found, corrected_inward))
        retain_recovery=(independent_admittance and self.mcc.admittance.retain_legacy_recovery)
        recovery_normal=(self.direction.update(q,pose,found,corrected_inward)
                         if retain_recovery else control_normal)
        if not settling and not bootstrapping and (not independent_admittance or retain_recovery):
            # Serial gates the recovery orchestrator on `t > bootstrap_end`.
            nominal = self._orchestrate(q=q, pose=pose, force=force, found=found,
                                        control_outward_world=recovery_normal,
                                        nominal=nominal)
        if independent_admittance:
            surface_points = points  # the admittance uses the measured contact Jacobian
        else:
            tip = self.mcc.fk(nominal)
            rotation = _quat_matrix(pose)
            surface_points = pose[:, None, :3] + torch.einsum("nij,nfj->nfi", rotation, tip)
        outward = control_normal if independent_admittance else -inward
        command, diagnostic = self.mcc.update(
            q_live=q, palm_pose_world=pose, force_world=force, found=found,
            surface_points_world=surface_points,
            surface_normals_world=control_normal,
            nominal_posture_q=nominal, contact_points_world=points,
            collision_normals_outward_world=outward,
            force_magnitude_only=True, use_contact_point_jacobian=True,
            contact_observed=contact_observed,
        )
        action = ((command - q) / 0.08).clamp(-2, 2)
        if self.trace is not None:
            self.trace.record_frame(
                frame=self.control_frame, replanned=replanned, q_live=q,
                nominal_q=nominal, command_q=command, action=action,
                palm_pose_world=pose, palm_twist_world=palm_twist_world,
                planner_command_palm=planner_command_palm, force_world=force,
                found=found, contact_points_world=points,
                contact_normals_inward_world=inward,
                state=self.observation.states[:, -1],
                control_normal_world=control_normal)
        if not settling:
            self.control_frame += 1
        return action, {"replanned": replanned, "settling": settling, "nominal_q": nominal,
                        "command_q": command, "force_max_n": force_max,
                        **diagnostic}

    def close(self):
        if self.async_planner is not None:
            self.async_planner.close()


# --------------------------------------------------------------------------
# merged from batched_dp_chunk_scheduler.py
# """CUDA batch of the accepted DTW/quintic chunk scheduler.
#
# Only the next replan interval is materialized. This is the complete executed
# part of each installed chunk when a fresh chunk arrives every 10 frames.
# --------------------------------------------------------------------------
class BatchedDPChunkScheduler:
    def __init__(self, batch: int, action_scale: torch.Tensor,
                 config: DPChunkSchedulerConfig | None = None):
        self.config = config or DPChunkSchedulerConfig()
        if self.config.replan_interval < 1 or self.config.waypoint_dt < self.config.control_dt:
            raise ValueError("invalid scheduler timing")
        self.scale = action_scale.to(dtype=torch.float64).clamp_min(0.02)
        self.batch = batch
        self.device = self.scale.device
        self.step = max(1, round(self.config.waypoint_dt / self.config.control_dt))
        self.capacity = max(3, self.config.history_points * self.step)
        self.history = torch.zeros((batch, self.capacity, self.scale.numel()),
                                   device=self.device, dtype=torch.float64)
        self.count = 0
        self.path = None
        self.path_index = 0
        self.last_drop_index = torch.zeros(batch, device=self.device, dtype=torch.long)
        self._history_index_key = None
        self._history_indices = None

    def observe(self, q_actual: torch.Tensor) -> None:
        q = q_actual.to(device=self.device, dtype=torch.float64)
        if q.shape != (self.batch, self.scale.numel()):
            raise ValueError(f"invalid q shape {tuple(q.shape)}")
        self.history[:, :-1].copy_(self.history[:, 1:].clone())
        self.history[:, -1].copy_(q)
        self.count = min(self.count + 1, self.capacity)

    @staticmethod
    def _velocity(path: torch.Tensor, dt: float) -> torch.Tensor:
        result = torch.zeros_like(path)
        if path.shape[1] >= 2:
            result[:, 0] = (path[:, 1] - path[:, 0]) / dt
            result[:, -1] = (path[:, -1] - path[:, -2]) / dt
        if path.shape[1] > 2:
            result[:, 1:-1] = (path[:, 2:] - path[:, :-2]) / (2 * dt)
        return result

    def _drop(self, prediction: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        if self.count < self.step + 1 or prediction.shape[1] < 2 or cfg.max_drop <= 0:
            return torch.zeros(self.batch, device=self.device, dtype=torch.long)
        offsets = list(range(self.count - 1, -1, -self.step))[:cfg.history_points]
        key=(self.count,cfg.history_points,self.step)
        if key!=self._history_index_key:
            self._history_indices=torch.tensor(
                [self.capacity-self.count+k for k in reversed(offsets)],device=self.device,dtype=torch.long)
            self._history_index_key=key
        history = self.history.index_select(1,self._history_indices)
        predicted = prediction[:, :min(prediction.shape[1], cfg.max_drop + 1)]
        hv = self._velocity(history, cfg.waypoint_dt)
        pv = self._velocity(predicted, cfg.waypoint_dt)
        position = (history[:, :, None] - predicted[:, None]) / self.scale
        velocity = (hv[:, :, None] - pv[:, None]) / (self.scale / cfg.waypoint_dt)
        cost = cfg.position_weight * position.square().mean(-1) + cfg.velocity_weight * velocity.square().mean(-1)
        rows, cols = cost.shape[1:]
        acc = torch.full_like(cost, float("inf"))
        length = torch.ones((self.batch, rows, cols), device=self.device, dtype=torch.long)
        acc[:, 0, 0] = cost[:, 0, 0]
        # Tiny 6x5 DP. All environment lanes execute together on the GPU.
        for i in range(rows):
            for j in range(cols):
                if i == 0 and j == 0:
                    continue
                candidates = []
                if i:
                    candidates.append((acc[:, i - 1, j], length[:, i - 1, j]))
                if j:
                    candidates.append((acc[:, i, j - 1], length[:, i, j - 1]))
                if i and j:
                    candidates.append((acc[:, i - 1, j - 1], length[:, i - 1, j - 1]))
                best_cost, best_len = candidates[0]
                for next_cost, next_len in candidates[1:]:
                    better = next_cost < best_cost
                    best_cost = torch.where(better, next_cost, best_cost)
                    best_len = torch.where(better, next_len, best_len)
                acc[:, i, j] = best_cost + cost[:, i, j]
                length[:, i, j] = best_len + 1
        endpoint = ((predicted - history[:, -1:]) / self.scale).square().mean(-1)
        return torch.argmin(acc[:, -1] / length[:, -1] + endpoint, dim=-1)

    def install(self, predicted_absolute: torch.Tensor, *, execution_frames: int | None = None) -> torch.Tensor:
        prediction = predicted_absolute.to(device=self.device, dtype=torch.float64)
        if prediction.ndim != 3 or prediction.shape[0] != self.batch or prediction.shape[2] != self.scale.numel():
            raise ValueError(f"invalid prediction shape {tuple(prediction.shape)}")
        if self.count == 0:
            raise RuntimeError("No executed trajectory is available")
        cfg = self.config
        drop = self._drop(prediction)
        self.last_drop_index = drop
        requested_frames = cfg.replan_interval if execution_frames is None else int(execution_frames)
        if requested_frames < 1:
            raise ValueError('execution_frames must be positive')
        segments = (requested_frames + self.step - 1) // self.step
        if prediction.shape[1] - cfg.max_drop < segments + 1:
            raise ValueError("prediction too short for requested execution buffer")
        # The following waypoint determines velocity/acceleration at the last
        # segment boundary even though it is not itself executed this cycle.
        waypoint_indices = drop[:, None] + torch.arange(segments + 1, device=self.device)[None]
        retained = prediction.gather(1, waypoint_indices[:, :, None].expand(-1, -1, prediction.shape[-1]))
        current = self.history[:, -1:]
        waypoints = torch.cat((current, retained), dim=1)
        velocity = self._velocity(waypoints, cfg.waypoint_dt)
        acceleration = torch.zeros_like(waypoints)
        if waypoints.shape[1] > 2:
            acceleration[:, 1:-1] = (waypoints[:, 2:] - 2 * waypoints[:, 1:-1] + waypoints[:, :-2]) / cfg.waypoint_dt**2
        if self.count >= 2:
            velocity[:, 0] = (self.history[:, -1] - self.history[:, -2]) / cfg.control_dt
        else:
            velocity[:, 0] = 0
        if self.count >= 3:
            acceleration[:, 0] = (self.history[:, -1] - 2 * self.history[:, -2] + self.history[:, -3]) / cfg.control_dt**2
        else:
            acceleration[:, 0] = 0
        duration = cfg.waypoint_dt
        p0, p1 = waypoints[:, :-1], waypoints[:, 1:]
        v0, v1 = velocity[:, :-1] * duration, velocity[:, 1:] * duration
        a0, a1 = acceleration[:, :-1] * duration**2, acceleration[:, 1:] * duration**2
        delta = p1 - p0
        c0, c1, c2 = p0, v0, 0.5 * a0
        c3 = 10 * delta - 6 * v0 - 4 * v1 - 1.5 * a0 + 0.5 * a1
        c4 = -15 * delta + 8 * v0 + 7 * v1 + 1.5 * a0 - a1
        c5 = 6 * delta - 3 * v0 - 3 * v1 - 0.5 * a0 + 0.5 * a1
        s = (torch.arange(1, self.step + 1, device=self.device, dtype=torch.float64) / self.step)[None, None, :, None]
        raw = (c0[:, :, None] + c1[:, :, None] * s + c2[:, :, None] * s**2
               + c3[:, :, None] * s**3 + c4[:, :, None] * s**4 + c5[:, :, None] * s**5)
        raw = raw.reshape(self.batch, -1, self.scale.numel())[:, :requested_frames]
        limited = []
        previous = current[:, 0]
        for command in raw.unbind(1):
            previous = torch.maximum(torch.minimum(command, previous + cfg.max_joint_step), previous - cfg.max_joint_step)
            limited.append(previous)
        self.path = torch.stack(limited, dim=1)
        self.path_index = 0
        return drop

    def next_command(self) -> torch.Tensor:
        if self.path is None:
            return self.history[:, -1].float()
        if self.path_index >= self.path.shape[1]:
            raise RuntimeError("chunk expired: install the next batch before requesting another command")
        command = self.path[:, self.path_index].float()
        self.path_index += 1
        return command
