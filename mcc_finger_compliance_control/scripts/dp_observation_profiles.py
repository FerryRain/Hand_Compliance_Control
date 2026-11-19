"""Explicit opt-in observation ablations preserving the exported state contract."""
import numpy as np


def without_explicit_q_velocity(field_slices, state_dim, schema, action_representation):
    if (state_dim != 96 or schema != 'contact_geometry_planner_motion'
            or action_representation != 'anchored_delta_q'):
        raise ValueError('Velocity ablation requires motion96 anchored-delta targets')
    if field_slices.get('q_hand') != slice(0,16) or field_slices.get('q_velocity') != slice(44,60):
        raise ValueError('Unexpected hand position/velocity field layout')
    mask=np.ones(96,dtype=np.float32)
    mask[44:60]=0
    return mask
