"""Semantic geometry and frame tests for the third contact metric."""
import sys
from pathlib import Path

import numpy as np
import unittest
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from true_fingertip_contact import ASSETS, TrueFingertipContact, ContactSlotObserver


def surface_points(metric, kind, role):
    mesh = trimesh.load(ASSETS/kind/f'{role}_convex.obj', force='mesh', process=False)
    centers = np.asarray(mesh.triangles_center)
    split = metric.split_manifests[0 if kind == 'fingertip_two_part' else 1]
    signed = centers @ np.asarray(split['plane_normal_source_mesh'])-split['plane_offset_m']
    sign = 1 if split['pad_side'] == 'positive' else -1
    # Exclude artificial cap triangles; they are tested separately below.
    good = sign*signed > 1e-5 if role == 'pad' else sign*signed < -1e-5
    return torch.tensor(centers[good], dtype=torch.float32)


def pack(ordinary, thumb):
    n = max(len(ordinary), len(thumb))
    o, t = ordinary[torch.arange(n) % len(ordinary)], thumb[torch.arange(n) % len(thumb)]
    return torch.stack((o, o, o, t), 1)


def inputs(points):
    return torch.ones(points.shape[:-1], dtype=torch.bool), torch.ones(points.shape[:-1])


def check_entire_exterior_tip_is_allowed(metric):
    points = pack(surface_points(metric, 'fingertip_two_part', 'pad'),
                  surface_points(metric, 'thumb_fingertip_two_part', 'pad'))
    result = metric.classify_source(points, *inputs(points))
    assert result['true_fingertip'].all()
    assert not result['connector'].any()


def check_connector_is_rejected_in_both_mesh_types(metric):
    points = pack(surface_points(metric, 'fingertip_two_part', 'connector'),
                  surface_points(metric, 'thumb_fingertip_two_part', 'connector'))
    result = metric.classify_source(points, *inputs(points))
    assert not result['true_fingertip'].any()
    assert result['connector'].all()


def check_cut_face_and_far_halfspace_points_are_rejected(metric):
    ordinary = trimesh.load(ASSETS/'fingertip_two_part/pad_convex.obj', force='mesh')
    thumb = trimesh.load(ASSETS/'thumb_fingertip_two_part/pad_convex.obj', force='mesh')
    points = pack(torch.tensor(ordinary.vertices.mean(0)[None], dtype=torch.float32),
                  torch.tensor(thumb.vertices.mean(0)[None], dtype=torch.float32))
    for f in range(4):
        split = metric.split_manifests[1 if f == 3 else 0]
        points[:, f, 1] = split['plane_offset_m']
    assert not metric.classify_source(points, *inputs(points))['true_fingertip'].any()
    points[:, :, 0] += .2
    points[:, :3, 1] += .02
    points[:, 3, 1] -= .02
    assert not metric.classify_source(points, *inputs(points))['true_fingertip'].any()


def check_tolerance_never_admits_connector_side(metric):
    points = pack(surface_points(metric, 'fingertip_two_part', 'connector')[:1],
                  surface_points(metric, 'thumb_fingertip_two_part', 'connector')[:1])
    for f in range(4):
        part = metric.parts[1 if f == 3 else 0]
        signed = (points[:, f]*part['normal']).sum(-1)-part['offset']
        points[:, f] -= (signed+.00005)[:, None]*part['normal']
    assert not metric.classify_source(points, *inputs(points))['true_fingertip'].any()


def check_source_site_palm_roundtrip_with_rotating_joints(metric):
    source = pack(surface_points(metric, 'fingertip_two_part', 'pad')[:8],
                  surface_points(metric, 'thumb_fingertip_two_part', 'pad')[:8])
    body = torch.einsum('fij,bfj->bfi', metric.geom_rotation, source)+metric.geom_position
    local = torch.einsum('fji,bfj->bfi', metric.fk.site_rotations, body-metric.fk.site_positions)
    torch.testing.assert_close(metric.source_from_tip_local(local), source, atol=3e-8, rtol=1e-6)
    q = torch.linspace(-.3, 1.4, len(source)*16).reshape(-1, 16)
    _, _, origins, rotations, _ = metric.fk.contact_kinematics(q)
    palm = torch.einsum('bfij,bfj->bfi', rotations, body)+origins
    recovered = metric.source_from_palm(q, palm)
    torch.testing.assert_close(recovered, source, atol=8e-8, rtol=1e-6)
    assert metric.classify_source(recovered, *inputs(source))['true_fingertip'].all()


def check_missing_nan_and_underloaded_contacts_do_not_count(metric):
    points = pack(surface_points(metric, 'fingertip_two_part', 'pad')[:1],
                  surface_points(metric, 'thumb_fingertip_two_part', 'pad')[:1])
    found, force = inputs(points)
    found[:, 0] = False
    force[:, 1] = .049
    points[:, 2] = float('nan')
    result = metric.classify_source(points, found, force)['true_fingertip']
    assert result.tolist() == [[False, False, False, True]]


def check_multi_slot_counts_true_tip_after_connector_without_force_borrowing(metric):
    tip = pack(surface_points(metric, 'fingertip_two_part', 'pad')[:1],
               surface_points(metric, 'thumb_fingertip_two_part', 'pad')[:1])
    connector = pack(surface_points(metric, 'fingertip_two_part', 'connector')[:1],
                     surface_points(metric, 'thumb_fingertip_two_part', 'connector')[:1])
    points = torch.stack((connector, tip), 2)
    found, force = inputs(points)
    result = metric.classify_source(points, found, force)['true_fingertip']
    assert not result[:, :, 0].any() and result[:, :, 1].all()
    force[:, :, 1] = .01  # Loaded connector cannot supply force for the tip.
    assert not metric.classify_source(points, found, force)['true_fingertip'].any()


def check_legacy_back_x_exclusion_is_not_used_by_new_metric(metric):
    source = pack(surface_points(metric, 'fingertip_two_part', 'pad'),
                  surface_points(metric, 'thumb_fingertip_two_part', 'pad'))
    body = torch.einsum('fij,bfj->bfi', metric.geom_rotation, source)+metric.geom_position
    local = torch.einsum('fji,bfj->bfi', metric.fk.site_rotations, body-metric.fk.site_positions)
    excluded_by_old_x = local[:, :, 0] > torch.tensor([.012, .012, .012, .016])
    assert excluded_by_old_x.any()
    assert metric.classify_source(source, *inputs(source))['true_fingertip'][excluded_by_old_x].all()


def check_penetrating_solver_midpoints_remain_in_true_tip_region(metric):
    points = []
    for kind in ('fingertip_two_part', 'thumb_fingertip_two_part'):
        mesh = trimesh.load(ASSETS/kind/'pad_convex.obj', force='mesh')
        points.append(torch.tensor(mesh.vertices.mean(0)[None], dtype=torch.float32))
    point = pack(*points)
    assert metric.classify_source(point, *inputs(point))['true_fingertip'].all()


class TrueFingertipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metric = TrueFingertipContact('cpu')

    def test_exterior(self):
        check_entire_exterior_tip_is_allowed(self.metric)

    def test_connector(self):
        check_connector_is_rejected_in_both_mesh_types(self.metric)

    def test_cut_and_far_points(self):
        check_cut_face_and_far_halfspace_points_are_rejected(self.metric)

    def test_no_tolerance_leak(self):
        check_tolerance_never_admits_connector_side(self.metric)

    def test_frames(self):
        check_source_site_palm_roundtrip_with_rotating_joints(self.metric)

    def test_invalid_measurements(self):
        check_missing_nan_and_underloaded_contacts_do_not_count(self.metric)

    def test_slots(self):
        check_multi_slot_counts_true_tip_after_connector_without_force_borrowing(self.metric)

    def test_back_surface(self):
        check_legacy_back_x_exclusion_is_not_used_by_new_metric(self.metric)

    def test_solver_penetration(self):
        check_penetrating_solver_midpoints_remain_in_true_tip_region(self.metric)

    def test_split_sensors_preserve_force_aggregation(self):
        from replay_inverted import replay_env_cfg
        cfg = replay_env_cfg()
        original = cfg.scene.sensors
        before = [(s.name, s.reduce, s.num_slots, s.fields) for s in original]
        observer = ContactSlotObserver()
        observer.configure(cfg)
        self.assertEqual(before, [(s.name, s.reduce, s.num_slots, s.fields) for s in original])
        for name in ('if_tip', 'mf_tip', 'rf_tip', 'th_tip'):
            old = next(s for s in original if s.name == f'{name}_contact')
            new = next(s for s in cfg.scene.sensors if s.name == f'{name}_contact')
            self.assertEqual(new.reduce, old.reduce)
            self.assertEqual(new.num_slots, old.num_slots)
            self.assertEqual(new.primary.mode, 'body')
        added = cfg.scene.sensors[len(original):]
        self.assertEqual(len(added), 8)
        self.assertTrue(all(s.reduce == 'netforce' and s.num_slots == 1 for s in added))
        self.assertEqual(sum(s.primary.pattern.endswith('_connector_collision$') for s in added), 4)
        self.assertEqual(sum(s.primary.pattern.endswith('_pad_collision$') for s in added), 4)

    def test_only_accepted_parts_collide_and_inertia_is_preserved(self):
        import mujoco
        from replay_inverted import replay_env_cfg
        from split_fingertip_collision import GEOMS
        cfg = replay_env_cfg()
        old_spec = cfg.scene.entities['robot'].spec_fn()
        before = old_spec.compile()
        observer = ContactSlotObserver(); observer.configure(cfg)
        new_spec = cfg.scene.entities['robot'].spec_fn()
        after = new_spec.compile()
        for name in GEOMS:
            source = old_spec.geom(name)
            original = new_spec.geom(name)
            self.assertEqual((original.contype, original.conaffinity), (0, 0))
            for role in ('pad', 'connector'):
                part = new_spec.geom(f'{name}_{role}_collision')
                self.assertEqual((part.contype, part.conaffinity), (source.contype, source.conaffinity))
                for field in ('pos', 'quat', 'friction', 'solref', 'solimp'):
                    np.testing.assert_array_equal(getattr(part, field), getattr(source, field))
                self.assertEqual(part.priority, source.priority)
                self.assertEqual(part.mass, 0)
        np.testing.assert_allclose(after.body_mass, before.body_mass, atol=1e-10)
        np.testing.assert_allclose(after.body_inertia, before.body_inertia, atol=1e-10)
        np.testing.assert_allclose(after.body_ipos, before.body_ipos, atol=1e-10)
        self.assertEqual(after.nq, before.nq)
        self.assertEqual(after.nv, before.nv)


if __name__ == '__main__':
    unittest.main()
