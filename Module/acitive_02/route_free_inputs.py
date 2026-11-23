"""Load a matched fixture, keeping the physical plant separate from planning."""

from pathlib import Path

import mujoco
import numpy as np

from . import active_a as execution
from .experiment import digest, read_json
from ..i04_geometric_traversal.fingertip_hand import FingertipHand, HEAD_FRACTION

OBJECTS = ('pear','mustard','cracker','bleach','coffee')


def load_trial(folder, obj):
    folder = Path(folder)
    spec = read_json(folder/'fixtures.json')
    paired = spec['objects'][obj]
    case = folder/obj
    if digest(folder/'hand.mjb')!=spec['hand_sha256'] or digest(case/'initial.npz')!=paired['initial.npz']:
        raise ValueError('Hand or initial-state file does not match the fixture manifest')

    def load_model(path):
        return mujoco.MjModel.from_binary_path('scene.mjb',{'scene.mjb':path.read_bytes()})

    model = load_model(case/'model.mjb')
    data = mujoco.MjData(model)
    with np.load(case/'initial.npz',allow_pickle=False) as initial:
        state = initial['integration_state']
        warm = initial['warm_policy_observations'] if 'warm_policy_observations' in initial else None
    mujoco.mj_setState(model,data,state,execution.STATE)
    mujoco.mj_forward(model,data)
    grasp = (data.qpos[:3].copy(),execution.rotation(data.qpos[3:7]),data.qpos[7:].copy())
    plant = FingertipHand(model,object_name=obj,grasp=grasp,surface_queries=False,
                          min_head_fraction=None if obj=='pear' else HEAD_FRACTION)
    robot = FingertipHand(load_model(folder/'hand.mjb'),object_name='fixture',grasp=grasp,surface_queries=False)
    plant.inward_in_palm[:]=robot.inward_in_palm[:]=(0.,0.,-1.)
    sensed = mujoco.MjData(robot.model)
    execution.robot_state(plant,robot,data,sensed)
    np.testing.assert_allclose(data.geom_xpos[plant.pads],sensed.geom_xpos[robot.pads],atol=1e-7,rtol=0.)
    touch = plant.measure(data)
    if not touch.measured_heads.all() or touch.loads.max()>=8:
        raise ValueError('Expected the registered four-contact start')
    return paired,(plant,data,robot,sensed,touch,state,warm),case,case/'surface.obj'
