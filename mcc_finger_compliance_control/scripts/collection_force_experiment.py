"""Explicit motion-phase interventions for the force diagnostic suite."""
from dataclasses import replace
import math
import mujoco
import numpy as np


def expanded_outward_authority(config):
    # Inward limits and contact-force targets stay fixed. Increase the joint
    # correction available for outward synergy, as well as its Cartesian budget.
    return replace(config, force_servo_max_outward_offset=.010,
                   normal_synergy_max_outward_step=.080)


def scale_arm_support(env, scale):
    """Change PD gains in place, preserving CUDA graph buffer addresses."""
    if not .05 <= scale <= 1.:
        raise ValueError('Arm stiffness scale must be in [.05,1]')
    import torch
    import warp as wp
    m = env.sim.mj_model
    ids = [a for a in range(m.nu)
           if int(m.actuator_trntype[a]) == int(mujoco.mjtTrn.mjTRN_JOINT)
           and m.joint(int(m.actuator_trnid[a,0])).name.split('/')[-1]
           in {f'joint{i}' for i in range(1,7)}]
    if len(ids) != 6:
        raise ValueError(f'Expected six arm actuators, found {len(ids)}')
    kp = m.actuator_gainprm[ids,0].copy()*scale
    kd = -m.actuator_biasprm[ids,2].copy()*math.sqrt(scale)
    if scale != 1.:
        stream = torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream)
        with torch.cuda.stream(stream):
            gain = torch.as_tensor(kp, device=env.device, dtype=torch.float32)
            damping = torch.as_tensor(kd, device=env.device, dtype=torch.float32)
            env.sim.model.actuator_gainprm[:,ids,0] = gain
            env.sim.model.actuator_biasprm[:,ids,1] = -gain
            env.sim.model.actuator_biasprm[:,ids,2] = -damping
        m.actuator_gainprm[ids,0] = kp
        m.actuator_biasprm[ids,1] = -kp
        m.actuator_biasprm[ids,2] = -kd
    # Check the actual arrays consumed by Warp, including the captured graph.
    wp.synchronize_stream(wp.get_stream(env.sim.wp_device))
    actual_kp = env.sim.wp_model.actuator_gainprm.numpy()[:,ids,0]
    actual_kd = -env.sim.wp_model.actuator_biasprm.numpy()[:,ids,2]
    if not (np.allclose(actual_kp,kp) and np.allclose(actual_kd,kd)):
        raise RuntimeError('Arm gain update did not reach Warp model buffers')
    return dict(stiffness_scale=scale,damping_scale=math.sqrt(scale),
                actuator_ids=ids,kp=actual_kp.tolist(),kd=actual_kd.tolist(),
                effort_limits=m.actuator_forcerange[ids].tolist(),
                phase='motion start; same legacy settling gains; no state reset')
