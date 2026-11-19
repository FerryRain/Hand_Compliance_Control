"""Batched residual-RL reward contract for the frozen B/221 execution chain.

This module deliberately accepts measured health and contact features instead
of choosing their margins from the incomplete randomness audit.  It runs on
CPU or CUDA without copying tensors between devices.  One call scores one
residual decision and the physics/control frames it covered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RewardConfig:
    gamma_decision: float
    decision_frames: int
    contact_weight: float
    potential_weight: float
    force_weight: float
    residual_weight: float
    force_soft_n: float
    force_scale_n: float
    safety_failure_penalty: float

    def __post_init__(self) -> None:
        if not 0 < self.gamma_decision <= 1:
            raise ValueError("gamma_decision must be in (0, 1]")
        if self.decision_frames < 1:
            raise ValueError("decision_frames must be positive")
        for name in ("contact_weight", "potential_weight", "force_weight",
                     "residual_weight", "force_soft_n", "safety_failure_penalty"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.force_scale_n <= 0:
            raise ValueError("force_scale_n must be positive")

    @property
    def gamma_frame(self) -> float:
        return self.gamma_decision ** (1.0 / self.decision_frames)


@dataclass(frozen=True)
class HealthComponents:
    """Nonnegative, already calibrated [batch] early-warning costs.

    q_cost is a distance outside a healthy configuration region.  angle_cost
    is pad-orientation badness.  Their feature definitions and soft margins
    must be calibrated from the completed audit before training.
    """

    q_cost: torch.Tensor
    angle_cost: torch.Tensor


@dataclass(frozen=True)
class HealthWeights:
    q: float
    angle: float

    def __post_init__(self) -> None:
        if self.q < 0 or self.angle < 0:
            raise ValueError("health weights must be nonnegative")


def health_badness(components: HealthComponents, weights: HealthWeights) -> torch.Tensor:
    if components.q_cost.shape != components.angle_cost.shape or components.q_cost.ndim != 1:
        raise ValueError("health components must have matching [batch] shapes")
    if not torch.isfinite(components.q_cost).all() or not torch.isfinite(components.angle_cost).all():
        raise ValueError("health components must be finite")
    if (components.q_cost < 0).any() or (components.angle_cost < 0).any():
        raise ValueError("health components must be nonnegative")
    return weights.q * components.q_cost + weights.angle * components.angle_cost


@dataclass(frozen=True)
class RewardTransition:
    # [frames, batch, 4]. Contact quality is the continuous version of the
    # strict pad filter, calibrated separately; `found` alone is insufficient.
    contact_quality: torch.Tensor
    force_n: torch.Tensor
    # [frames, batch], a contiguous prefix of True for each environment.
    frame_mask: torch.Tensor
    health_before: HealthComponents
    health_after: HealthComponents
    # [batch, 16], actual residual divided by its calibrated per-joint bound.
    normalized_residual: torch.Tensor
    # [batch]. Completed 2500-frame episodes and safety failures are terminal;
    # a training-horizon truncation is not terminal and must bootstrap.
    terminated: torch.Tensor
    safety_failure: torch.Tensor


@dataclass(frozen=True)
class RewardTerms:
    contact: torch.Tensor
    potential: torch.Tensor
    force: torch.Tensor
    residual: torch.Tensor
    safety_failure: torch.Tensor
    total: torch.Tensor
    bootstrap_discount: torch.Tensor


def compute_reward(transition: RewardTransition, config: RewardConfig,
                   health_weights: HealthWeights) -> RewardTerms:
    """Return per-environment reward components for one residual decision.

    Per-frame task/force rewards use the control-frame discount.  Potential
    shaping uses the actual number of valid frames, so early safety stops and
    full decisions share the same time base.  The terminal potential is zero;
    truncations retain the observed next-state potential for bootstrapping.
    """
    quality = transition.contact_quality
    force = transition.force_n
    mask = transition.frame_mask
    if quality.ndim != 3 or quality.shape[-1] != 4 or force.shape != quality.shape:
        raise ValueError("contact_quality and force_n must have shape [frames,batch,4]")
    frames, batch, _ = quality.shape
    if frames > config.decision_frames or frames == 0 or mask.shape != (frames, batch):
        raise ValueError("invalid frame window or frame_mask shape")
    if mask.dtype != torch.bool or torch.any(mask[1:] & ~mask[:-1]):
        raise ValueError("frame_mask must be a contiguous True prefix per environment")
    if torch.any(mask.sum(dim=0) == 0):
        raise ValueError("each environment needs at least one observed frame")
    if quality.device != force.device or quality.device != mask.device:
        raise ValueError("reward tensors must share a device")
    if not torch.isfinite(quality).all() or not torch.isfinite(force).all():
        raise ValueError("contact quality and force must be finite")
    if torch.any((quality < 0) | (quality > 1)) or torch.any(force < 0):
        raise ValueError("contact quality must be in [0,1] and force nonnegative")
    residual = transition.normalized_residual
    terminated = transition.terminated
    failure = transition.safety_failure
    if residual.shape != (batch, 16) or terminated.shape != (batch,) or failure.shape != (batch,):
        raise ValueError("residual and terminal flags have invalid batch shape")
    if terminated.dtype != torch.bool or failure.dtype != torch.bool or torch.any(failure & ~terminated):
        raise ValueError("safety failures must be terminal")
    if not torch.isfinite(residual).all():
        raise ValueError("residual must be finite")
    if any(value.device != quality.device for value in (
            residual, terminated, failure, transition.health_before.q_cost,
            transition.health_before.angle_cost, transition.health_after.q_cost,
            transition.health_after.angle_cost)):
        raise ValueError("reward tensors must share a device")

    bad_before = health_badness(transition.health_before, health_weights)
    bad_after = health_badness(transition.health_after, health_weights)
    if bad_before.shape != (batch,) or bad_after.shape != (batch,):
        raise ValueError("health components must have shape [batch]")
    discount = torch.pow(quality.new_tensor(config.gamma_frame),
                         torch.arange(frames, device=quality.device, dtype=quality.dtype))[:, None]
    weighted_mask = discount * mask.to(quality.dtype)
    contact = config.contact_weight * (quality.amin(dim=-1) * weighted_mask).sum(dim=0)
    excess_force = torch.relu((force - config.force_soft_n) / config.force_scale_n)
    force_cost = config.force_weight * (excess_force.square().mean(dim=-1) * weighted_mask).sum(dim=0)
    elapsed_discount = torch.pow(quality.new_tensor(config.gamma_frame),
                                 mask.sum(dim=0).to(quality.dtype))
    # Phi = -badness.  For true terminal states Phi_terminal = 0.
    next_potential = torch.where(terminated, torch.zeros_like(bad_after), -bad_after)
    potential = config.potential_weight * (elapsed_discount * next_potential + bad_before)
    residual_cost = config.residual_weight * residual.square().mean(dim=-1)
    failure_cost = config.safety_failure_penalty * failure.to(quality.dtype)
    total = contact + potential - force_cost - residual_cost - failure_cost
    bootstrap_discount = torch.where(terminated, torch.zeros_like(elapsed_discount),
                                     elapsed_discount)
    return RewardTerms(contact, potential, force_cost, residual_cost, failure_cost,
                       total, bootstrap_discount)


@dataclass(frozen=True)
class FullRewardConfig:
    """Provisional Full reward; amounts refer to one five-frame decision.

    Calibration supplies force_warn_n. Oracle q is a privileged training-only
    prior; reference_valid must reject unsafe or invalid reference states.
    Recovery is smaller than loss so deliberate lose/recover cycles cost reward.
    """

    force_warn_n: float
    gamma_decision: float = 0.997
    decision_frames: int = 5
    loss_frames: int = 11
    recovery_frames: int = 5
    contact_weight: float = 0.2
    contact_four_fraction: float = 0.5
    loss_weight: float = 2.0
    recovery_weight: float = 1.0
    persistent_loss_weight: float = 0.0
    force_weight: float = 1.0
    geometry_weight: float = 0.3
    q_prior_weight: float = 0.05
    effort_weight: float = 0.01
    smooth_weight: float = 0.01
    failure_penalty: float = 100.0
    stop_force_n: float = 100.0

    def __post_init__(self) -> None:
        if not 0 <= self.force_warn_n < self.stop_force_n:
            raise ValueError("force warning must be below the independent stop line")
        if not 0 < self.gamma_decision <= 1 or min(self.decision_frames, self.loss_frames, self.recovery_frames) < 1:
            raise ValueError("invalid discount or event confirmation duration")
        weights = (self.contact_weight, self.loss_weight, self.recovery_weight, self.persistent_loss_weight,
                   self.force_weight, self.geometry_weight, self.q_prior_weight,
                   self.effort_weight, self.smooth_weight, self.failure_penalty)
        if any(value < 0 for value in weights):
            raise ValueError("reward weights must be nonnegative")
        if self.recovery_weight > self.loss_weight:
            raise ValueError("recovery bonus must not exceed the corresponding loss cost")
        if not 0 <= self.contact_four_fraction <= 1:
            raise ValueError("contact_four_fraction must be in [0,1]")

    @property
    def gamma_frame(self) -> float:
        return self.gamma_decision ** (1 / self.decision_frames)


class FullResidualReward:
    """Stateful, vectorized control-frame scorer; independent of MCC.

    Call after observing a new physical state, starting from reset()'s initial
    potential. Sum frame rewards with gamma_frame inside each residual decision.
    Expose loss/recovery counters and previous residual to the critic/actor so
    the history-dependent event reward has an explicit state representation.
    """

    def __init__(self, batch: int, device: torch.device | str, config: FullRewardConfig):
        self.config = config
        self.device = torch.device(device)
        self.batch = batch
        self.loss_count = torch.zeros(batch, 4, dtype=torch.long, device=device)
        self.recovery_count = torch.zeros_like(self.loss_count)
        self.lost = torch.zeros(batch, 4, dtype=torch.bool, device=device)
        self.active = torch.ones(batch, dtype=torch.bool, device=device)
        self.previous_phi = torch.zeros(batch, device=device)
        self.previous_geometry_phi = torch.zeros(batch, device=device)
        self.previous_residual = torch.zeros(batch, 16, device=device)

    @torch.no_grad()
    def potential(self, strict: torch.Tensor, force_n: torch.Tensor,
                  geometry_cost: torch.Tensor, q_prior_cost: torch.Tensor,
                  reference_valid: torch.Tensor,
                  expected_contact: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        risk = ((force_n.amax(-1) - cfg.force_warn_n)
                / (cfg.stop_force_n - cfg.force_warn_n)).clamp(0, 1).square()
        expected = torch.ones_like(strict) if expected_contact is None else expected_contact
        count = expected.sum(-1).clamp_min(1)
        geometry = (geometry_cost.clamp(0, 1) * expected).sum(-1) / count
        # Geometry already has its own potential. A geometry-only departure
        # must not enable an Oracle posture prior on stable-contact episodes.
        # Persistent loss is explicit reward state, not an incidental one-frame
        # collision miss. reference_valid also requires local geometry support.
        loss = (self.lost & expected).float().sum(-1) / count
        gate = torch.maximum(loss, risk) * reference_valid
        phi = (-cfg.geometry_weight * geometry
               - cfg.q_prior_weight * gate * q_prior_cost.clamp(0, 1))
        return phi, gate, risk

    @torch.no_grad()
    def reset(self, initial_phi: torch.Tensor, initial_geometry_phi: torch.Tensor | None = None) -> None:
        if initial_phi.shape != (self.batch,) or not torch.isfinite(initial_phi).all():
            raise ValueError("initial potential must be finite [batch]")
        self.loss_count.zero_()
        self.recovery_count.zero_()
        self.lost.zero_()
        self.active.fill_(True)
        self.previous_phi.copy_(initial_phi)
        self.previous_geometry_phi.copy_(initial_phi if initial_geometry_phi is None else initial_geometry_phi)
        self.previous_residual.zero_()

    @torch.no_grad()
    def step(self, *, strict: torch.Tensor, force_n: torch.Tensor,
             geometry_cost: torch.Tensor, q_prior_cost: torch.Tensor,
             reference_valid: torch.Tensor, normalized_residual: torch.Tensor,
             completed: torch.Tensor, failure: torch.Tensor,
             decision_boundary: bool, expected_contact: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        cfg = self.config
        if strict.shape != (self.batch, 4) or force_n.shape != strict.shape or geometry_cost.shape != strict.shape:
            raise ValueError("strict, force and geometric features must have shape [batch,4]")
        if (q_prior_cost.shape != (self.batch,) or reference_valid.shape != (self.batch,)
                or completed.shape != (self.batch,) or failure.shape != (self.batch,)
                or normalized_residual.shape != (self.batch, 16)):
            raise ValueError("invalid prior, terminal flag or residual shape")
        expected = torch.ones_like(strict) if expected_contact is None else expected_contact
        if expected.shape != strict.shape:
            raise ValueError("expected contact must have shape [batch,4]")
        finite = (torch.isfinite(force_n).all(-1) & torch.isfinite(geometry_cost).all(-1)
                  & torch.isfinite(q_prior_cost) & torch.isfinite(normalized_residual).all(-1))
        failure = failure | ~finite
        # Nonfinite measurements terminate their lane, without poisoning others.
        force_n = torch.nan_to_num(force_n, nan=cfg.stop_force_n, posinf=cfg.stop_force_n, neginf=0).clamp_min(0)
        geometry_cost = torch.nan_to_num(geometry_cost, nan=1).clamp(0, 1)
        q_prior_cost = torch.nan_to_num(q_prior_cost, nan=1).clamp(0, 1)
        normalized_residual = torch.nan_to_num(normalized_residual, nan=0, posinf=0, neginf=0)
        active = self.active.clone()
        failure = failure | (force_n.amax(-1) > cfg.stop_force_n)
        done = (completed | failure) & active
        enabled = expected & active[:, None]
        missing = enabled & ~strict
        self.loss_count = torch.where(missing, self.loss_count + 1, 0)
        new_loss = enabled & ~self.lost & (self.loss_count >= cfg.loss_frames)
        self.lost |= new_loss
        # A contact recovered by pressing toward the stop line is not healthy
        # recovery. Require the entire hand below the calibrated warning level
        # throughout the confirmation window before paying any event bonus.
        safe_recovery = force_n.amax(-1) <= cfg.force_warn_n
        self.recovery_count = torch.where(enabled & self.lost & strict & safe_recovery[:, None],
                                          self.recovery_count + 1, 0)
        recovered = enabled & self.lost & (self.recovery_count >= cfg.recovery_frames)
        self.lost &= ~recovered & expected
        self.recovery_count = torch.where(recovered, 0, self.recovery_count)

        phi, gate, risk = self.potential(strict, force_n, geometry_cost, q_prior_cost, reference_valid, expected)
        terminal_phi = torch.where(done, 0, phi)
        count = expected.sum(-1).clamp_min(1)
        fraction = (strict & expected).float().sum(-1) / count
        fraction = torch.where(expected.any(-1), fraction, 1)
        quality = ((1 - cfg.contact_four_fraction) * fraction
                   + cfg.contact_four_fraction * (strict | ~expected).all(-1).float())
        contact = cfg.contact_weight / cfg.decision_frames * quality
        loss = cfg.loss_weight * new_loss.float().sum(-1)
        # Amount per decision; count only currently missing, confirmed fingers.
        # Preserve the legacy component name and exact default arithmetic.
        if cfg.persistent_loss_weight:
            persistent = ((self.loss_count >= cfg.loss_frames) & enabled).float().sum(-1) / count
            loss = loss + cfg.persistent_loss_weight / cfg.decision_frames * persistent
        recovery = cfg.recovery_weight * recovered.float().sum(-1)
        safety = cfg.force_weight / cfg.decision_frames * risk
        shaping = cfg.gamma_frame * terminal_phi - self.previous_phi
        geometry_phi = -cfg.geometry_weight * (geometry_cost * expected).sum(-1) / count
        geometry_shaping = cfg.gamma_frame * torch.where(done, 0, geometry_phi) - self.previous_geometry_phi
        effort = cfg.effort_weight / cfg.decision_frames * normalized_residual.square().mean(-1)
        smooth = cfg.smooth_weight * (normalized_residual - self.previous_residual).square().mean(-1)
        if not decision_boundary:
            smooth = torch.zeros_like(smooth)
        else:
            # Compare actual corrections at successive residual decisions,
            # not the immediately preceding physical/control frame.
            self.previous_residual.copy_(normalized_residual)
        terminal = cfg.failure_penalty * failure.float()
        components = {"contact": contact, "loss": -loss, "recovery": recovery,
                      "force": -safety, "potential": shaping, "effort": -effort,
                      "smooth": -smooth, "terminal": -terminal}
        components = {key: torch.where(active, value, 0) for key, value in components.items()}
        components["total"] = sum(components.values())
        components["potential_geometry"] = torch.where(active, geometry_shaping, 0)
        components["potential_q"] = components["potential"] - components["potential_geometry"]
        components["gate"] = torch.where(active, gate, 0)
        components["phi"] = torch.where(active, phi, 0)
        components["loss_events"] = new_loss.sum(-1)
        components["recovery_events"] = recovered.sum(-1)
        components["done"] = done
        self.previous_phi.copy_(terminal_phi)
        self.previous_geometry_phi.copy_(torch.where(done, 0, geometry_phi))
        self.active &= ~done
        return components
