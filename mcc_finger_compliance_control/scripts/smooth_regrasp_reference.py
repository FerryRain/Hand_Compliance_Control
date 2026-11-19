"""Opt-in nominal shaping and measured regrasp; never filters MCC output.

Joint bounds outrank acceleration in the reference filter. This is not a
collision or torque feasibility certificate. MCC retains final authority.
"""
import math
import torch


class ReferenceMotionFilter:
    def __init__(self, lower, upper, *, velocity=2., acceleration=40., dt=.01):
        if not all(math.isfinite(x) and x > 0 for x in (velocity, acceleration, dt)):
            raise ValueError('Positive finite motion bounds required')
        if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or not (upper > lower).all():
            raise ValueError('Invalid joint bounds')
        self.lower, self.upper = lower, upper
        self.vmax, self.amax, self.dt = velocity, acceleration, dt
        self.position = self.velocity = None
        self.diagnostic = {}

    def __call__(self, target, anchor):
        if not torch.isfinite(target).all() or not torch.isfinite(anchor).all():
            raise ValueError('Nonfinite reference/state: no command issued')
        if self.position is None:
            self.position = anchor.clamp(self.lower, self.upper).clone()
            self.velocity = torch.zeros_like(target)
        bounded = target.clamp(self.lower, self.upper)
        error = bounded - self.position
        # Braking-distance governor, followed by a discrete acceleration bound.
        speed = (torch.sqrt((self.amax*self.dt)**2 + 2*self.amax*error.abs())
                 - self.amax*self.dt).clamp_max(self.vmax)
        desired_v = error.sign() * speed
        velocity = self.velocity + (desired_v-self.velocity).clamp(-self.amax*self.dt, self.amax*self.dt)
        proposed = self.position + self.dt*velocity
        position = proposed.clamp(self.lower, self.upper)
        actual_v = (position-self.position)/self.dt
        self.diagnostic = dict(raw=target.clone(), filtered=position.clone(),
            velocity=actual_v.clone(), acceleration=(actual_v-self.velocity)/self.dt,
            bound_override=position!=proposed, lag=(bounded-position).abs())
        self.position, self.velocity = position, actual_v
        return position.clone()


def regrasp_class():
    # Lazy imports keep filter checks independent of simulation initialization.
    from local_contact_search import LocalContactSearch
    from palm_posture_reset import TRACK, UNLOAD, joint_reserve

    class SmoothRegrasp(LocalContactSearch):
        """Reuse the existing transaction, but reset to a qualified initial grasp.

        A persistent loss of at least three fingers requests a reset as well as
        the existing complete-loss / joint-reserve trigger. Transient single
        finger loss remains the fingertip controller's responsibility.
        """
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.poor_count = torch.zeros_like(self.count)
            self.fixed_healthy = self.pregrasp.clone()
            self.fixed_confirmed = torch.zeros_like(self.healthy_valid)

        def observe(self, q, pose, tactile):
            strict = tactile['pad_found'] & (tactile['pad_force'].norm(dim=-1)>=.05)
            ready = (self.phase==TRACK) & self.initial_ready & (self.cooldown==0)
            self.poor_count = torch.where(ready & (strict.sum(-1)<2),self.poor_count+1,0)
            self.external_reset_pending |= self.poor_count>=100
            old_count = self.count.clone()
            super().observe(q, pose, tactile)
            # Freeze the first sustained healthy measured grasp, not a late
            # drifting pose. It is a reusable posture, not an object pose.
            capture = self.healthy_valid & ~self.fixed_confirmed & (self.phase==TRACK)
            self.fixed_healthy = torch.where(capture[:,None],self.healthy_q,self.fixed_healthy)
            self.fixed_confirmed |= capture
            new = self.count!=old_count
            self.reset_q = torch.where(new[:,None],self.fixed_healthy,self.reset_q)
            self.chain.mcc.reset_posture_target = self.reset_q
            # Clear old search integration on transaction entry. Nothing from
            # the failed posture may be added to the reset target later.
            self.chain.runtime_recovery_offset[new] = 0
            self.chain.recovery_active[new] = False
            self.chain.loss_streak[new] = 0

        def diagnostics(self):
            return dict(super().diagnostics(), regrasp_initial_target=self.fixed_healthy.clone(),
                        regrasp_initial_confirmed=self.fixed_confirmed.clone(),
                        regrasp_poor_contact_frames=self.poor_count.clone())

        def report(self):
            return dict(super().report(), regrasp_version='initial_healthy_v1',
                        regrasp_target='first sustained healthy measured grasp, else bounded supplied pregrasp',
                        additional_trigger='fewer than two true fingertips for 100 control frames')
    return SmoothRegrasp
