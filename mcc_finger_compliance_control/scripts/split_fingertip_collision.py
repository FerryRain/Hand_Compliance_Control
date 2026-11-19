"""Accepted two-part fingertip collision geometry, without material changes."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT/'mcc_finger_compliance_control/assets/true_fingertip_contact_v1'
GEOMS = ('fingertip_geom', 'fingertip_2_geom', 'fingertip_3_geom', 'thumb_fingertip_geom')
SITES = ('if_tip', 'mf_tip', 'rf_tip', 'th_tip')
BODIES = ('fingertip', 'fingertip_2', 'fingertip_3', 'thumb_fingertip')


def replace_tip_collision(spec):
    """Original mesh remains visual/inertial; only two accepted hulls collide."""
    manifest = json.loads((ASSETS/'asset_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ASSETS/name).read_bytes()).hexdigest() != expected:
            raise ValueError('accepted split asset changed')
    meshes = {}
    for kind in ('fingertip_two_part', 'thumb_fingertip_two_part'):
        split = json.loads((ASSETS/kind/'split_manifest.json').read_text())
        if not split['accepted_interactively']:
            raise ValueError('unapproved anatomical split')
        if hashlib.sha256((ROOT/split['source']).read_bytes()).hexdigest() != split['source_sha256']:
            raise ValueError('source fingertip STL changed')
        for role in ('pad', 'connector'):
            mesh = trimesh.load(ASSETS/kind/f'{role}_convex.obj', force='mesh', process=False)
            if not mesh.is_watertight:
                raise ValueError('accepted collision hull is not watertight')
            name = f'true_tip_{kind}_{role}'
            spec.add_mesh(name=name, uservert=np.asarray(mesh.vertices).ravel(),
                          userface=np.asarray(mesh.faces, dtype=np.int32).ravel())
            meshes[kind, role] = name
    for body_name, geom_name in zip(BODIES, GEOMS):
        source = spec.geom(geom_name)
        if source is None or not (source.contype or source.conaffinity):
            raise ValueError(f'missing/disabled merged source geom: {geom_name}')
        kind = 'thumb_fingertip_two_part' if body_name == 'thumb_fingertip' else 'fingertip_two_part'
        contype, affinity = int(source.contype), int(source.conaffinity)
        for role in ('pad', 'connector'):
            part = spec.body(body_name).add_geom(name=f'{geom_name}_{role}_collision',
                type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshes[kind, role],
                pos=tuple(source.pos), quat=tuple(source.quat), mass=0.,
                rgba=(0., 0., 0., 0.), contype=contype, conaffinity=affinity,
                condim=int(source.condim))
            # This experiment changes collision geometry, not compliance.
            for attr in ('friction', 'solref', 'solimp'):
                getattr(part, attr)[:] = getattr(source, attr)
            for attr in ('priority', 'solmix', 'margin', 'gap'):
                setattr(part, attr, getattr(source, attr))
        source.contype = 0
        source.conaffinity = 0
    return spec


def configure_split_collision(cfg, controller_scope='body'):
    if controller_scope not in ('body', 'pad'):
        raise ValueError('controller scope must be body or pad')
    original = cfg.scene.entities['robot'].spec_fn
    cfg.scene.entities['robot'].spec_fn = lambda: replace_tip_collision(original())
    sensors = []
    connector_pattern = r'^(?:fingertip_geom|fingertip_2_geom|fingertip_3_geom|thumb_fingertip_geom)_connector_collision$'
    for sensor in cfg.scene.sensors:
        name = sensor.name
        tip_index = next((i for i, site in enumerate(SITES)
            if name in (f'{site}_contact', f'{site}_geometry_contact')), None)
        if tip_index is not None:
            i = tip_index
            # Preserve the legacy controller observation over the distal
            # body. Separate pad sensors provide the new evaluation metric.
            sensor = replace(sensor, primary=replace(sensor.primary,
                mode='body' if controller_scope == 'body' else 'geom',
                pattern=f'^{BODIES[i]}$' if controller_scope == 'body' else f'^{GEOMS[i]}_pad_collision$'))
        elif name.startswith('incidental_hand_object'):
            # The old non-tip mask did not include a separate connector geom.
            old = sensor.primary.pattern
            patterns = (old,) if isinstance(old, str) else tuple(old)
            sensor = replace(sensor, primary=replace(sensor.primary,
                pattern=(*patterns, connector_pattern)))
        sensors.append(sensor)
    cfg.scene.sensors = tuple(sensors)
