"""Palm/object collision geometry and positive clearance observations.

The historical E05 fixture disabled the detailed hand meshes. New I04 runs
must explicitly enable the rigid palm, four palm sensor housings and mount.
Finger-belly contact credit remains tied to the four original pad geoms.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
import mujoco
import numpy as np


PALM_MASK = 8
QUERY_MARGIN_M = .006
PALM_NAMES = ('fr3_leap_mount_adapter', 'palm_lower_collision',
              'palm_1_fsr_geom', 'palm_2_fsr_geom', 'palm_3_fsr_geom', 'palm_4_fsr_geom')


def enable_palm_collisions(xml):
    """Change only new-run XML in memory; compile to update broadphase masks.

    margin == gap gives positive-distance observations without applying a
    physical force at that distance. Actual overlap remains a real contact.
    MuJoCo uses conservative convex mesh hulls for these six rigid parts.
    """
    root = ET.fromstring(xml)
    for name in PALM_NAMES:
        geom = root.find(f".//worldbody//geom[@name='{name}']")
        if geom is None:
            raise ValueError(f'Missing palm part: {name}')
        if int(geom.get('contype', '0')) or int(geom.get('conaffinity', '0')):
            raise ValueError(f'Unexpected pre-existing palm mask: {name}')
        geom.set('contype', str(PALM_MASK))
        geom.set('conaffinity', '2')
        geom.set('margin', str(QUERY_MARGIN_M))
        geom.set('gap', str(QUERY_MARGIN_M))
    root.find('compiler').set('usethread', 'false')
    return ET.tostring(root, encoding='unicode')


class PalmMonitor:
    def __init__(self, model):
        self.model = model
        self.ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                             for name in PALM_NAMES], dtype=int)
        if np.any(self.ids < 0):
            raise ValueError('Palm geometry is incomplete')
        self.palm = set(map(int, self.ids))
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
        self.objects = set(map(int, np.flatnonzero((model.geom_bodyid == body)
                               & (model.geom_contype != 0))))

    def observe(self, data):
        minimum = QUERY_MARGIN_M
        nearest = ''
        count = 0
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in self.palm and b in self.objects:
                palm = a
            elif b in self.palm and a in self.objects:
                palm = b
            else:
                continue
            count += 1
            if contact.dist < minimum:
                minimum = float(contact.dist)
                nearest = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, palm)
        return dict(clearance_m=minimum, nearest_geom=nearest, query_contacts=count,
                    upper_censored=not bool(nearest))
