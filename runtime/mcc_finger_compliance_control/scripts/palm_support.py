"""Opt-in kinematic and finite-weld palm supports for transfer diagnostics.

Kinematic support removes the free joint; finite support retains its dynamics.
Neither modifies the sixteen finger actuators or collision assets, and neither
is an identified physical robot-arm impedance model.
"""
from __future__ import annotations

import mujoco


def extrapolate_published_root_state(state, elapsed_s):
    """Advance a published [p, quaternion(wxyz), world linear/angular twist].

    Used only to smooth a finite support's reference, never to overwrite the
    measured floating-base state. No future trajectory or object geometry.
    """
    import torch
    result=state.clone()
    result[:,:3]+=elapsed_s*state[:,7:10]
    omega=state[:,10:13];speed=omega.norm(dim=-1,keepdim=True)
    half_angle=.5*elapsed_s*speed
    v=omega*(torch.sin(half_angle)/speed.clamp_min(1e-12));w=torch.cos(half_angle)
    qw=state[:,3:4];qv=state[:,4:7]
    result[:,3:7]=torch.cat((w*qw-(v*qv).sum(-1,keepdim=True),
        w*qv+qw*v+torch.linalg.cross(v,qv)),dim=-1)
    return result


def measured_pose_difference_twist(current, previous, dt):
    """Causal spatial twist, matching inverse/export's 10ms pose difference."""
    import torch
    if previous is None:return current.new_zeros((len(current),6))
    a=current[:,3:7];b=previous[:,3:7]
    a=a/a.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    b=b/b.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    aw=a[:,:1];av=a[:,1:];bw=b[:,:1];bv=-b[:,1:]
    w=aw*bw-(av*bv).sum(-1,keepdim=True)
    v=aw*bv+bw*av+torch.linalg.cross(av,bv)
    sign=torch.where(w<0,-1.,1.);w=w*sign;v=v*sign
    norm=v.norm(dim=-1,keepdim=True)
    angle=2*torch.atan2(norm,w)
    rotvec=v*torch.where(norm>1e-8,angle/norm.clamp_min(1e-12),2.)
    return torch.cat(((current[:,:3]-previous[:,:3])/dt,rotvec/dt),dim=-1)


class KinematicPalmSupport:
    recenter_substeps = False

    def configure(self, cfg):
        original = cfg.scene.entities['robot'].spec_fn

        def supported_spec():
            spec = original()
            root = spec.body('palm_lower')
            joints = list(root.joints)
            if len(joints) != 1 or joints[0].type != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError('Expected exactly one free joint at palm_lower')
            spec.delete(joints[0])
            root.mocap = True
            return spec

        cfg.scene.entities['robot'].spec_fn = supported_spec

    def bind(self, env):
        self.robot = env.scene['robot']
        if not self.robot.is_fixed_base or not self.robot.is_mocap:
            raise ValueError('Kinematic support did not compile into a mocap base')
        self.data = env.sim.data
        self.mocap_id = int(self.robot.data.indexing.mocap_id)

    def write_root_state(self, state):
        # Pose is prescribed at the original 100Hz timing. Native mocap bodies
        # have no dynamic root velocity; do not silently overwrite finger qvel.
        # This mjlab revision's entity helper adds an extra singleton axis
        # for an integer mocap index, which does not broadcast for batch>1.
        self.data.mocap_pos[:, self.mocap_id, :] = state[:, :3]
        self.data.mocap_quat[:, self.mocap_id, :] = state[:, 3:7]

    def report(self):
        return dict(kind='kinematic', root='mocap palm with no freejoint',
                    support='ideal prescribed base; not finite arm impedance',
                    timing='original100Hz zero-order-hold poses; no interpolation',
                    velocity='native mocap has no integrated root DOFs; requested twist remains available to the controller',
                    recenter_after_substep=False)


class WeldPalmSupport:
    """Finite soft weld to a prescribed anchor; dynamics advance without resets.

    The time constant is a diagnostic setting, not identified arm stiffness.
    MeasuredWeldPalmSupport supplies measured base feedback for policy tests;
    this base class preserves the earlier recorded-command replay interface.
    """
    recenter_substeps = False

    def __init__(self, time_constant_s=.01):
        if not .002 <= time_constant_s <= .05:
            raise ValueError('weld time constant must be2..50ms')
        self.time_constant_s = float(time_constant_s)

    def configure(self, cfg):
        original = cfg.scene.entities['robot'].spec_fn

        def supported_spec():
            spec = original()
            root = spec.body('palm_lower')
            anchor = spec.worldbody.add_body(name='palm_support_anchor', mocap=True,
                                            pos=root.pos, quat=root.quat)
            spec.add_equality(name='palm_support_weld', type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY, name1=root.name, name2=anchor.name,
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                solref=[self.time_constant_s, 1.], solimp=[.95, .99, .001, .5, 2.])
            return spec

        cfg.scene.entities['robot'].spec_fn = supported_spec

    def bind(self, env):
        self.robot = env.scene['robot']
        self.data = env.sim.data
        model = env.sim.mj_model
        bids = [b for b in range(model.nbody)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or '')
                .endswith('/palm_support_anchor')]
        if len(bids) != 1:
            raise ValueError('Expected one palm support mocap anchor')
        self.mocap_id = int(model.body_mocapid[bids[0]])
        self.initialized = False
        self.last_position = None

    def write_root_state(self, state):
        import torch
        self.data.mocap_pos[:, self.mocap_id, :] = state[:, :3]
        self.data.mocap_quat[:, self.mocap_id, :] = state[:, 3:7]
        if not self.initialized:
            self.robot.write_root_state_to_sim(state)
            self.initialized = True
        else:
            # Preserve the accepted termination parking operation, but never
            # teleport a live palm along ordinary task reference increments.
            parked = (state[:, :3]-self.last_position).norm(dim=-1) > 1.
            ids = torch.nonzero(parked, as_tuple=False).flatten()
            if ids.numel():
                self.robot.write_root_state_to_sim(state[ids], env_ids=ids)
        self.last_position = state[:, :3].clone()

    def report(self):
        return dict(kind='weld', root='original freejoint with soft6D support constraint',
                    support='finite numerical weld; not calibrated arm impedance',
                    time_constant_s=self.time_constant_s, damping_ratio=1.,
                    timing='original100Hz anchor poses; no interpolation',
                    state_reset='initialization and stopped-lane parking only',
                    recenter_after_substep=False)


class MeasuredWeldPalmSupport(WeldPalmSupport):
    """Opt-in finite support with measured feedback and optional 1ms reference.

    The original weld remains available unchanged for old replay diagnostics.
    Pose differences match the DP training exporter; instantaneous solver
    velocity is retained solely as an explicit ablation.
    """
    def __init__(self,time_constant_s=.005,*,interpolate=True,twist_mode='pose_difference'):
        super().__init__(time_constant_s)
        if twist_mode not in ('pose_difference','instantaneous'):
            raise ValueError('Unknown measured palm twist mode')
        self.interpolate=bool(interpolate);self.twist_mode=twist_mode

    def bind(self,env):
        super().bind(env);self.env=env
        self.reference_state=None;self.interpolated_target=None;self.substep=0
        self.previous_feedback_pose=None
        if self.interpolate:
            original_step=env.sim.step
            def smooth_step():
                self.advance_reference()
                return original_step()
            env.sim.step=smooth_step

    def write_root_state(self,state):
        super().write_root_state(state)
        self.reference_state=state.clone();self.interpolated_target=state.clone();self.substep=0

    def advance_reference(self):
        if self.reference_state is None:return
        self.substep+=1
        dt=self.substep*float(self.env.sim.mj_model.opt.timestep)
        state=extrapolate_published_root_state(self.reference_state,dt)
        self.data.mocap_pos[:,self.mocap_id,:]=state[:,:3]
        self.data.mocap_quat[:,self.mocap_id,:]=state[:,3:7]
        self.interpolated_target=state

    def measured_feedback(self,control_dt=.01):
        """Call once per control tick, after forward refreshed body kinematics."""
        pose=self.robot.data.root_link_pose_w.clone()
        if self.twist_mode=='pose_difference':
            twist=measured_pose_difference_twist(pose,self.previous_feedback_pose,control_dt)
        else:twist=self.robot.data.root_link_vel_w.clone()
        self.previous_feedback_pose=pose.clone()
        return pose,twist

    def report(self):
        result=super().report()
        result.update(measured_base_feedback=True,measured_twist_mode=self.twist_mode,
                      interpolate_support=self.interpolate)
        if self.interpolate:
            result['timing']='1ms reference extrapolation from published pose/world twist; refreshed every10ms; no future teacher joints'
        return result
