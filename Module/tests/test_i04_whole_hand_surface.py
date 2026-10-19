"""Cooperative continuation and the exact ellipsoid orientation derivative."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from Module.i04_geometric_traversal.whole_hand_surface_traversal import support_ready
from Module.i04_geometric_traversal.whole_hand_kinematics import support_derivative
from Module.i04_geometric_traversal.finger_reconfiguration_planner import ellipsoid_support_derivative
from Module.i04_geometric_traversal.whole_hand_step_solver import WholeHandStepSolver
from Module.i04_geometric_traversal.belly_bounded_step import bounded_belly_step


class WholeHandSurfaceChecks(unittest.TestCase):
    def test_strict_initial_handoff_rejects_a_permanently_passive_fourth(self):
        safe=np.ones((150,4),bool);safe[:,3]=False
        self.assertFalse(support_ready(safe,np.ones(4,bool),strict_four=True))

    def test_narrow_patch_allows_two_contacts_but_a_broad_patch_requires_three(self):
        safe=np.tile([True,True,False,False],(150,1))
        self.assertTrue(support_ready(safe,[True,True,False,False]))
        self.assertFalse(support_ready(safe,[True,True,True,True]))

    def test_window_cannot_be_satisfied_by_a_single_good_sample(self):
        safe=np.zeros((150,4),bool);safe[-1]=True
        self.assertFalse(support_ready(safe,[True,True,False,False]))

    def test_vectorized_ellipsoid_derivative_matches_scalar_reference(self):
        rng=np.random.default_rng(20261018)
        for _ in range(100):
            frame=Rotation.random(random_state=rng).as_matrix()
            sizes=rng.uniform(.001,.020,3);normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
            angular=rng.normal(size=(3,23))
            np.testing.assert_allclose(support_derivative(frame,sizes,normal,angular),
                ellipsoid_support_derivative(frame,sizes,normal,angular),atol=1e-14,rtol=1e-12)

    def test_derivative_includes_support_shift_when_pad_rotates(self):
        frame=Rotation.from_euler('xyz',[.4,-.7,.2]).as_matrix();sizes=np.array([.012,.008,.002])
        normal=np.array([.3,.4,.5]);normal/=np.linalg.norm(normal);axis=np.array([.2,-.7,.3])
        def support(r):
            shape=(r*sizes**2)@r.T
            return shape@normal/np.sqrt(normal@shape@normal)
        eps=1e-6
        fd=(support(Rotation.from_rotvec(eps*axis).as_matrix()@frame)-
            support(Rotation.from_rotvec(-eps*axis).as_matrix()@frame))/(2*eps)
        np.testing.assert_allclose(support_derivative(frame,sizes,normal,axis[:,None])[:,0],fd,atol=1e-11,rtol=1e-8)

    def test_warm_solve_preserves_the_original_box_objective(self):
        rng=np.random.default_rng(19);solver=WholeHandStepSolver();q=np.zeros(23)
        for _ in range(100):
            J=rng.normal(size=(32,23))*.1;e=rng.normal(size=32)*.003
            old,od=bounded_belly_step(J,e,q,np.full(23,-.002),np.full(23,.002),gain=.55,maximum_joint_step=.0008)
            new,nd=solver.solve(J,e,q,np.full(23,-.002),np.full(23,.002),gain=.55,maximum_joint_step=.0008)
            self.assertTrue(od['feasible'] and nd['feasible'])
            objective=lambda x:np.linalg.norm(J@x+.55*e)**2+1e-6*np.linalg.norm(x)**2
            self.assertLessEqual(objective(new),objective(old)+1e-13)
            self.assertTrue(np.all(q+new>=-.002) and np.all(q+new<=.002))
            q+=new

    def test_fixed_and_saturated_joints_do_not_freeze_other_fingers(self):
        J=np.eye(4);q=np.zeros(4);lower=np.array([0.,-.1,-.1,-.1]);upper=np.array([0.,0.,.1,.1])
        delta,diag=WholeHandStepSolver().solve(J,np.full(4,-.01),q,lower,upper,maximum_joint_step=.001)
        self.assertTrue(diag['feasible']);self.assertEqual(delta[0],0.);self.assertLessEqual(delta[1],0.)
        self.assertGreater(delta[2],.0009);self.assertGreater(delta[3],.0009)


if __name__=='__main__':unittest.main()
