"""Ideal support preserves finger dynamics/FK and constrains the palm in solve."""
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from mjlab.entity import Entity
from palm_support import KinematicPalmSupport, WeldPalmSupport
from replay_inverted import replay_env_cfg


def models():
    cfg = replay_env_cfg(hand_stiffness=35., hand_damping=2.5, hand_effort_limit=35.)
    free = Entity(cfg.scene.entities['robot']).spec.compile()
    KinematicPalmSupport().configure(cfg)
    supported = Entity(cfg.scene.entities['robot']).spec.compile()
    return free, supported


def test_support_preserves_finger_servo_mass_collision_and_joint_order():
    free, supported = models()
    assert free.nv == supported.nv + 6 == 22
    assert supported.nmocap == 1
    names = lambda m: [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                       for j in range(m.njnt) if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    assert names(free) == names(supported)
    for field in ('actuator_gainprm', 'actuator_biasprm', 'actuator_gear', 'actuator_forcerange',
                  'body_mass', 'body_inertia', 'geom_contype', 'geom_conaffinity',
                  'geom_friction', 'geom_solref', 'geom_solimp'):
        np.testing.assert_array_equal(getattr(free, field), getattr(supported, field))
    np.testing.assert_array_equal(free.dof_damping[6:], supported.dof_damping)
    np.testing.assert_array_equal(free.dof_frictionloss[6:], supported.dof_frictionloss)


def test_prescribed_pose_keeps_all_finger_geometry_identical():
    free, supported = models()
    a, b = mujoco.MjData(free), mujoco.MjData(supported)
    pose = np.array([.1, -.2, .3, .9238795325, 0., .3826834324, 0.])
    pose[3:] /= np.linalg.norm(pose[3:])
    q = np.linspace(.1, .7, 16)
    a.qpos[:7] = pose
    a.qpos[7:] = q
    b.mocap_pos[0] = pose[:3]
    b.mocap_quat[0] = pose[3:]
    b.qpos[:] = q
    mujoco.mj_forward(free, a)
    mujoco.mj_forward(supported, b)
    np.testing.assert_allclose(a.geom_xpos, b.geom_xpos, atol=1e-12)
    np.testing.assert_allclose(a.geom_xmat, b.geom_xmat, atol=1e-12)


def test_support_absorbs_base_load_during_physics_instead_of_after_reset():
    free, supported = models()
    for m in (free, supported):
        m.opt.gravity[:] = 0
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 0
    a, b = mujoco.MjData(free), mujoco.MjData(supported)
    a.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    b.mocap_pos[0] = 0
    b.mocap_quat[0] = [1, 0, 0, 0]
    for m, d in ((free, a), (supported, b)):
        root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'palm_lower')
        d.xfrc_applied[root, 0] = 100.
        mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
    assert abs(a.qpos[0]) > 1e-6
    np.testing.assert_allclose(b.xpos[1], [0, 0, 0], atol=1e-12)


def test_batched_mocap_write_changes_only_the_robot_index():
    data = SimpleNamespace(mocap_pos=torch.zeros(5, 3, 3),
                           mocap_quat=torch.zeros(5, 3, 4))
    robot = SimpleNamespace(is_fixed_base=True, is_mocap=True,
                            data=SimpleNamespace(indexing=SimpleNamespace(mocap_id=2)))
    env = SimpleNamespace(scene={'robot':robot}, sim=SimpleNamespace(data=data))
    driver = KinematicPalmSupport()
    driver.bind(env)
    state = torch.arange(65, dtype=torch.float32).reshape(5, 13)
    driver.write_root_state(state)
    assert torch.equal(data.mocap_pos[:, 2], state[:, :3])
    assert torch.equal(data.mocap_quat[:, 2], state[:, 3:7])
    assert not bool(data.mocap_pos[:, :2].any())
    assert not bool(data.mocap_quat[:, :2].any())


def test_soft_weld_reacts_to_load_with_finite_compliance():
    cfg = replay_env_cfg(hand_stiffness=35., hand_damping=2.5, hand_effort_limit=35.)
    WeldPalmSupport().configure(cfg)
    m = Entity(cfg.scene.entities['robot']).spec.compile()
    m.opt.timestep = .001
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.gravity[:] = 0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    d = mujoco.MjData(m)
    d.qpos[:7] = [0, 0, 0, 1, 0, 0, 0]
    d.qpos[7:] = .3
    d.ctrl[:] = .3
    d.mocap_pos[0] = 0
    d.mocap_quat[0] = [1, 0, 0, 0]
    root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'palm_lower')
    d.xfrc_applied[root, 0] = 100.
    mujoco.mj_forward(m, d)
    assert m.neq == 1 and m.nv == 22
    assert np.max(abs(d.efc_pos[:6])) < 1e-10
    for _ in range(30):
        mujoco.mj_step(m, d)
    assert 1e-6 < d.qpos[0] < .02
    assert np.isfinite(d.qpos).all()


def test_soft_support_does_not_teleport_a_live_base_each_control_frame():
    writes = []
    robot = SimpleNamespace(write_root_state_to_sim=lambda state, **kw: writes.append((state.clone(),kw)))
    driver = WeldPalmSupport()
    driver.robot = robot
    driver.data = SimpleNamespace(mocap_pos=torch.zeros(5, 3, 3),
                                  mocap_quat=torch.zeros(5, 3, 4))
    driver.mocap_id = 2
    driver.initialized = False
    driver.last_position = None
    state = torch.zeros(5, 13)
    state[:, 3] = 1
    driver.write_root_state(state)
    moving = state.clone()
    moving[:, 0] = .001
    driver.write_root_state(moving)
    assert len(writes) == 1
    parked = moving.clone()
    parked[2, 2] = 2
    driver.write_root_state(parked)
    assert len(writes) == 2
    assert torch.equal(writes[1][1]['env_ids'], torch.tensor([2]))
