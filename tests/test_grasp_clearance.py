"""Full-hand clearance must exclude visuals/pads and reject non-tip contact."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from object_catalog import load_object_config
from true_tip_waypoint_optimizer import SplitHullGeometry, TrueTipWaypointController


class GraspClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry = SplitHullGeometry(load_object_config('ycb_mustard'))

    def test_collision_scope_includes_dip_and_excludes_true_tips(self):
        g = self.geometry
        self.assertEqual(len(g.non_tip_geoms),17)
        self.assertIn('dip_geom',g.non_tip_names)
        self.assertIn('thumb_dip_geom',g.non_tip_names)
        self.assertIn('palm_lower_collision',g.non_tip_names)
        self.assertFalse(set(g.pads)&set(g.non_tip_geoms))
        self.assertTrue(set(g.connectors)<=set(g.non_tip_geoms))

    def test_all_link_queries_match_independent_native_queries(self):
        import mujoco
        g = self.geometry
        x = g.evaluate(np.full(16,.6),np.array([.03,.08,.07,1.,0.,0.,0.]),
                       non_tip_clearance=True)
        exact = []
        for geom in g.non_tip_geoms:
            values = []
            for target in g.objects:
                line = np.zeros(6)
                d = mujoco.mj_geomDistance(g.model,g.data,int(geom),int(target),.2,line)
                if d == 0 and not np.any(line): continue
                if d == 0 and np.linalg.norm(line[:3]-line[3:]) > 1e-6:
                    d = np.linalg.norm(line[:3]-line[3:])
                values.append(d)
            exact.append(min([.2]+values))
        np.testing.assert_allclose(x['non_tip_distance'],exact,atol=1e-10)

    def test_good_tip_contact_does_not_override_bad_link_clearance(self):
        c = TrueTipWaypointController.__new__(TrueTipWaypointController)
        c.non_tip_clearance_m = .002
        c.max_cross_finger_penetration_m = .0001
        g = dict(distance=np.full(4,-.001),connector_distance=np.full(4,.005),
                 cut_cap=np.zeros(4,dtype=bool),self_penetration=0.,
                 non_tip_distance=np.full(17,.003))
        self.assertTrue(c.geometry_valid(g))
        c.preferred_non_tip_clearance_m = .005
        # A reserve target must not silently replace the explicit hard gate.
        self.assertTrue(c.geometry_valid(g))
        g['non_tip_distance'][3] = -.001  # DIP collision, pads stay unchanged.
        self.assertFalse(c.geometry_valid(g))
        c.non_tip_clearance_m = None  # Legacy acceptance remains unchanged.
        self.assertTrue(c.geometry_valid(g))
        c.non_tip_clearance_m = .002
        g['non_tip_distance'][3] = .003
        g['self_penetration'] = .002
        self.assertFalse(c.geometry_valid(g))

    def test_mocap_quaternion_normalization_matches_rotation_transform(self):
        g = self.geometry
        q = np.full(16,.6)
        pose = np.array([.03,.08,.07,.9238795325,0.,.3826834324,0.])
        x = g.evaluate(q,pose,self_collision=True,non_tip_clearance=True)
        scaled = pose.copy(); scaled[3:] *= 1.000001
        y = g.evaluate(q,scaled,self_collision=True,non_tip_clearance=True)
        np.testing.assert_allclose(x['non_tip_distance'],y['non_tip_distance'],atol=1e-8)
        self.assertAlmostEqual(float(np.linalg.norm(g.data.mocap_quat[g.mocap])),1.,places=14)


if __name__=='__main__': unittest.main()
