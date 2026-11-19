"""Independent normal admittance around a moving DP reference.

Experimental opt-in controller, not the 2024 contact-implicit MPC algorithm.
Positive displacement is OUTWARD. The tactile adapter supplies force exerted
by the hand on the object; sensor_force_sign=-1 converts to force on the hand.
No connector/object-pose truth is consumed by this controller.
"""
from dataclasses import dataclass
import math
import torch
from batched_fullhand_mcc import BatchedFullHandMCC, _quat_matrix


@dataclass(frozen=True)
class FingerAdmittanceConfig:
    mass_kg: float = 1.0
    damping_ns_m: float = 80.0
    stiffness_n_m: float = 400.0
    filter_tau_s: float = .015
    max_inward_m: float = .003
    max_outward_m: float = .008
    max_speed_m_s: float = .02
    lost_search_speed_m_s: float = .002
    normal_memory_s: float = .25
    max_correction_rad: float = .04
    max_joint_speed_rad_s: float = 2.0
    dls_regularization_m2: float = 1.e-6
    sensor_force_sign: float = -1.0
    force_priority_ratio: float = 0.0  # 0 disables; >=1 enables tactile-only normal guard
    normal_jacobian_only: bool = False
    tangential_task_weight: float = 0.0  # soft preservation, normal mapping only
    retain_legacy_recovery: bool = False
    contact_handover_revision: int = 1

    def __post_init__(self):
        for key, value in vars(self).items():
            if key == 'contact_handover_revision':
                if value not in (1,2,3,4,5): raise ValueError('contact_handover_revision must be 1 through 5')
                continue
            if key in ('normal_jacobian_only','retain_legacy_recovery'):
                if not isinstance(value,bool):
                    raise ValueError(f'{key} must be bool')
                continue
            if not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
            if key == 'tangential_task_weight':
                if value < 0: raise ValueError('tangential_task_weight must be nonnegative')
                continue
            if key not in ('sensor_force_sign','force_priority_ratio') and value <= 0:
                raise ValueError(f'{key} must be positive')
        if self.sensor_force_sign not in (-1., 1.):
            raise ValueError('sensor_force_sign must be +1 or -1')
        if self.force_priority_ratio != 0 and self.force_priority_ratio < 1:
            raise ValueError('force_priority_ratio must be 0 or >=1')
        if self.tangential_task_weight and not self.normal_jacobian_only:
            raise ValueError('tangential_task_weight requires normal_jacobian_only')


def normal_priority_projection(command, lower, upper, jn, minimum_outward):
    """Project a joint step into a normal halfspace and joint box.

    Four active-set iterations suffice for four joints. Infeasible constraints
    return the maximum feasible outward motion, with explicit shortfall.
    All fingers and environments execute together, with no host QP calls.
    """
    result=torch.maximum(torch.minimum(command,upper),lower)
    for _ in range(4):
        gap=(minimum_outward-(jn*result).sum(-1)).clamp_min(0)
        free=torch.where(jn>=0,result<upper-1e-8,result>lower+1e-8)
        direction=jn*free
        delta=gap[...,None]*direction/direction.square().sum(-1,keepdim=True).clamp_min(1e-12)
        result=torch.maximum(torch.minimum(result+delta,upper),lower)
    return result,(minimum_outward-(jn*result).sum(-1)).clamp_min(0)


def integrate_normal(offset, velocity, force_error, dt, cfg):
    """Backward Euler for M*xdd+B*xd+K*x=Fn-Fdesired (moving frame)."""
    velocity = (cfg.mass_kg * velocity + dt * (force_error-cfg.stiffness_n_m*offset)) / (
        cfg.mass_kg + dt*cfg.damping_ns_m + dt*dt*cfg.stiffness_n_m)
    velocity = velocity.clamp(-cfg.max_speed_m_s, cfg.max_speed_m_s)
    candidate = offset + dt*velocity
    bounded = candidate.clamp(-cfg.max_inward_m, cfg.max_outward_m)
    velocity = torch.where(candidate != bounded, torch.zeros_like(velocity), velocity)
    return bounded, velocity


class BatchedFingerAdmittanceMCC(BatchedFullHandMCC):
    independent_normal_admittance = True

    def __init__(self, batch, device, config=None, admittance=None):
        self.admittance = admittance or FingerAdmittanceConfig()
        super().__init__(batch, device, config)
        axes = self.fk.joint_axes
        x,y,z = axes.unbind(-1); zero = torch.zeros_like(x)
        self.axis_skew = torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(4,4,3,3)
        self.axis_outer = axes[..., :, None]*axes[..., None, :]
        self.eye = torch.eye(3,device=self.device)

    def reset(self):
        super().reset()
        self.displacement = torch.zeros_like(self.force_offset)
        self.velocity = torch.zeros_like(self.force_offset)
        self.normal_world = torch.zeros(self.batch,4,3,device=self.device)
        self.normal_age = torch.full_like(self.force_offset,float('inf'))
        self.handover_command = None
        self.handover_nominal = None

    def geometry(self, q, points_palm=None, contact=None, use_cached_anchor=False):
        """Four serial joints, all fingers/environments vectorized together."""
        value=q.reshape(self.batch,4,4)
        cosine=value.cos()[...,None,None]; sine=value.sin()[...,None,None]
        moving=cosine*self.eye+(1-cosine)*self.axis_outer+sine*self.axis_skew
        rot=self.eye.expand(self.batch,4,3,3)
        pos=q.new_zeros(self.batch,4,3); origins=[]; axes=[]
        for j in range(4):
            pos=pos+(rot@self.fk.body_positions[:,j,:,None])[...,0]
            fixed=rot@self.fk.body_rotations[:,j]
            origins.append(pos)
            axes.append((fixed@self.fk.joint_axes[:,j,:,None])[...,0])
            rot=fixed@moving[:,:,j]
        site=pos+(rot@self.fk.site_positions[...,None])[...,0]
        point=site if points_palm is None else torch.where(contact[...,None],points_palm,site)
        if use_cached_anchor:
            if points_palm is not None:
                self._update_anchors(points_palm,contact,site,pos,rot)
            anchored=pos+(rot@self.anchor[...,None])[...,0]
            point=torch.where(self.anchor_valid[...,None],anchored,site)
        jac=torch.linalg.cross(torch.stack(axes,-2),point[...,None,:]-torch.stack(origins,-2),dim=-1).transpose(-1,-2)
        return site,jac

    @torch.no_grad()
    def update(self, *, q_live, palm_pose_world, force_world, found,
               surface_normals_world, nominal_posture_q, contact_points_world,
               **unused):
        a=self.admittance; dt=self.config.control_dt
        q=q_live.float(); nominal=self._clamp_q(nominal_posture_q.float())
        rot=_quat_matrix(palm_pose_world.float())
        normal=surface_normals_world.float()
        length=normal.norm(dim=-1)
        valid=found.bool() & torch.isfinite(normal).all(-1) & (length>1e-6)
        normal=torch.nan_to_num(normal)/length.nan_to_num().clamp_min(1e-6)[...,None]
        new_normal=torch.where(valid[...,None],normal,self.normal_world)
        transport=(new_normal*self.normal_world).sum(-1).clamp(0,1)
        transport=torch.where(torch.isfinite(self.normal_age),transport,torch.ones_like(transport))
        self.displacement*=transport; self.velocity*=transport
        self.normal_world=new_normal
        self.normal_age=torch.where(valid,0.,self.normal_age+dt)
        reliable=self.normal_age<=a.normal_memory_s
        raw=a.sensor_force_sign*(force_world*new_normal).sum(-1)
        finite_force=torch.isfinite(force_world).all(-1)
        loaded=found.bool() & finite_force & reliable
        measured=torch.where(loaded,raw.nan_to_num().clamp_min(0),0.)
        alpha=1-math.exp(-dt/a.filter_tau_s)
        self.force_filtered=torch.where(self.filter_valid,
            self.force_filtered+alpha*(measured-self.force_filtered),measured)
        self.filter_valid.fill_(True)
        # Rising force bypasses filter delay; falling force releases smoothly.
        feedback=torch.maximum(measured,self.force_filtered)
        normal_palm=(rot.transpose(-1,-2)[:,None]@new_normal[...,None])[...,0]
        if a.contact_handover_revision>=5:
            points=(rot.transpose(-1,-2)[:,None]@(contact_points_world-palm_pose_world[:,None,:3])[...,None])[...,0]
            # Keep the last tactile material point attached to the fingertip
            # when contact vanishes. Switching to the arbitrary site center
            # at that instant changes the very Jacobian used to re-contact.
            site,jac=self.geometry(q,points,loaded,use_cached_anchor=True)
        if self.handover_command is not None:
            if a.contact_handover_revision>=5:handover_jac=jac
            else:_,handover_jac=self.geometry(q)
            previous_nominal=(self.handover_nominal if a.contact_handover_revision>=2
                              and self.handover_nominal is not None else nominal)
            shift=(handover_jac@(self.handover_command-previous_nominal).reshape(self.batch,4,4)[...,None])[...,0]
            self.displacement=(shift*normal_palm).sum(-1).clamp(-a.max_inward_m,a.max_outward_m)
            self.handover_command=None
        old=self.displacement.clone()
        candidate,velocity=integrate_normal(old,self.velocity,feedback-self.force_setpoint,dt,a)
        integrated=candidate
        velocity=torch.where(~loaded,velocity.clamp_min(-a.lost_search_speed_m_s),velocity)
        candidate=(old+dt*velocity).clamp(-a.max_inward_m,a.max_outward_m)
        if a.contact_handover_revision>=4:
            # Undoing our own outward offset is not new inward exploration.
            # Use normal admittance speed up to zero; only the displacement
            # beyond zero is capped by the slow missing-contact search rate.
            returning=old.clamp_min(0)+dt*a.lost_search_speed_m_s
            candidate=torch.where(~loaded,torch.maximum(integrated,old-returning),candidate)
            velocity=torch.where(~loaded,(candidate-old)/dt,velocity)
        if a.contact_handover_revision>=2:
            # Preserve the integrator's bounded position even when it zeros
            # velocity on hitting the bound. Previously that step was undone.
            candidate=torch.where(loaded,integrated,candidate)
            # An expired normal disallows further search, but must not latch
            # an old outward correction forever. Return the existing offset
            # to neutral with the same bounded speed, without crossing zero.
            release_speed=a.max_speed_m_s if a.contact_handover_revision>=4 else a.lost_search_speed_m_s
            neutral=old+(-old).clamp(-dt*release_speed,dt*release_speed)
            candidate=torch.where(~reliable & finite_force,neutral,candidate)
        velocity=torch.where(reliable & finite_force,velocity,0.)
        candidate=torch.where((reliable | (a.contact_handover_revision>=2)) & finite_force,candidate,old)
        if a.contact_handover_revision<5:
            points=(rot.transpose(-1,-2)[:,None]@(contact_points_world-palm_pose_world[:,None,:3])[...,None])[...,0]
            site,_=self.geometry(q)
            point_valid=loaded & torch.isfinite(points).all(-1) & ((points-site).norm(dim=-1)<self.config.contact_anchor_max_site_distance)
            _,jac=self.geometry(q,points,point_valid)
        target=candidate[...,None]*normal_palm
        if a.normal_jacobian_only:
            # One normal task / four joint variables: minimum joint motion.
            # Tangential displacement is not constrained by this scalar task.
            jn=(normal_palm[...,None]*jac).sum(-2)
            correction=candidate[...,None]*jn/(jn.square().sum(-1,keepdim=True)+a.dls_regularization_m2)
            if a.tangential_task_weight:
                # Minimum joint motion can produce a large tangential tip
                # displacement. Penalize that displacement softly while
                # retaining the normal force task and existing joint bounds.
                jt=jac-normal_palm[...,None]*jn[...,None,:]
                hessian=(jn[..., :,None]*jn[...,None,:]
                         +a.tangential_task_weight*(jt.transpose(-1,-2)@jt)
                         +a.dls_regularization_m2*torch.eye(4,device=q.device))
                correction=torch.linalg.solve(hessian,(candidate[...,None]*jn)[...,None])[...,0]
        else:
            solved=torch.linalg.solve(jac@jac.transpose(-1,-2)+a.dls_regularization_m2*self.eye,target[...,None])
            correction=(jac.transpose(-1,-2)@solved)[...,0]
        requested=nominal+correction.clamp(-a.max_correction_rad,a.max_correction_rad).reshape(self.batch,16)
        previous=q if self.previous_command is None else self.previous_command
        command=previous+(requested-previous).clamp(-dt*a.max_joint_speed_rad_s,dt*a.max_joint_speed_rad_s)
        # Match the existing relative action encoder's physical range.
        command=self._clamp_q(torch.maximum(torch.minimum(command,q+.16),q-.16))
        priority=torch.zeros_like(loaded)
        shortfall=torch.zeros_like(measured)
        if a.force_priority_ratio:
            # The nominal trajectory must not cancel retreat at excessive tip
            # force. Preserve tangent motion, joint/rate bounds and diagnose
            # cases where geometry/authority prevents the requested unloading.
            priority=loaded & (feedback>a.force_priority_ratio*self.force_setpoint)
            jn=(normal_palm[...,None]*jac).sum(-2)
            lower=self._clamp_q(torch.maximum(previous-dt*a.max_joint_speed_rad_s,q-.16))
            upper=self._clamp_q(torch.minimum(previous+dt*a.max_joint_speed_rad_s,q+.16))
            minimum=dt*((feedback-self.force_setpoint)/a.damping_ns_m).clamp(0,a.max_speed_m_s)
            projected,shortfall=normal_priority_projection(
                (command-q).reshape(self.batch,4,4),
                (lower-q).reshape(self.batch,4,4),(upper-q).reshape(self.batch,4,4),jn,minimum)
            command=torch.where(priority[...,None],q.reshape(self.batch,4,4)+projected,
                                command.reshape(self.batch,4,4)).reshape(self.batch,16)
            shortfall=torch.where(priority,shortfall,0.)
        realized=(jac@(command-nominal).reshape(self.batch,4,4)[...,None])[...,0]
        achieved=(realized*normal_palm).sum(-1)
        clipped=(correction.abs()>a.max_correction_rad).any(-1) | ((command-requested).reshape(self.batch,4,4).abs().amax(-1)>1e-7)
        # Back-calculate only toward realizable compliance; do not integrate
        # hidden displacement while the joint/rate constraints block motion.
        feasible=achieved.clamp(-a.max_inward_m,a.max_outward_m)
        if a.contact_handover_revision>=3:
            # Nominal tracking lag is not a force-controller displacement.
            # Measure only the compliance change relative to the command the
            # same actuator bounds would permit without compliance.
            baseline=previous+(nominal-previous).clamp(-dt*a.max_joint_speed_rad_s,dt*a.max_joint_speed_rad_s)
            baseline=self._clamp_q(torch.maximum(torch.minimum(baseline,q+.16),q-.16))
            applied=(jac@(command-baseline).reshape(self.batch,4,4)[...,None])[...,0]
            feasible=(applied*normal_palm).sum(-1).clamp(-a.max_inward_m,a.max_outward_m)
            feasible=old+(feasible-old).clamp(-dt*a.max_speed_m_s,dt*a.max_speed_m_s)
        self.displacement=torch.where(clipped,feasible,candidate)
        self.velocity=torch.where(clipped & ((candidate-feasible)*velocity>0),0.,velocity)
        self.force_offset=-self.displacement  # compatibility: legacy inward sign
        self.previous_command=command.clone()
        return command,dict(normal_offset=self.force_offset,
            admittance_outward_m=self.displacement,admittance_velocity_m_s=self.velocity,
            admittance_normal_force_n=measured,admittance_signed_force_n=raw,
            admittance_saturated=clipped,admittance_normal_valid=reliable,
            admittance_priority=priority,admittance_retreat_shortfall_m=shortfall,
            admittance_normal_world=new_normal,
            admittance_input_force_filtered_n=self.force_filtered,
            admittance_command_normal_step_m=((jac@(command-q).reshape(self.batch,4,4)[...,None])[...,0]*normal_palm).sum(-1),
            admittance_normal_error_m=candidate-achieved)
