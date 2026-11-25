"""Regression tests for the nominal-start deadlock found in the pear recording."""
from pathlib import Path as File
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = File(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"PARK-3"))
from frontier_planner import FrontierPlanner, WatchedWorker
from measured_start import MeasuredStartGlobalWorker, MeasuredStartPlanner, MeasuredStartWorker, measured_prefix
from park3.common import configuration, planning
from park3.geometry import Clearance
from park3.trajectory import Path, extrema, limits
from planner_adapter import PalmWorker


class MeasuredStartTest(unittest.TestCase):
    def setUp(self):
        self.actual = np.array([0.,0.,.002,1.,0.,0.,0.])
        self.cfg = planning(configuration())
        self.points = np.array([[0.,0.,0.]])
        self.normals = np.array([[0.,0.,1.]])
        self.hull = np.array([[0.,0.,0.]])
        self.clearance = Clearance(self.points,self.normals,self.hull,self.actual,self.cfg)
        self.planner = MeasuredStartPlanner(SimpleNamespace(cfg=self.cfg,origin=self.actual,variant="D"),
                                            SimpleNamespace(support_sites=32,surface_count=112))

    def test_invalid_start_reranks_without_stale_incumbent(self):
        bad = self.actual.copy()
        bad[2] = -.005
        invalid_path = Path(0,np.tile(bad,(101,1)))
        def solve(tick,splice,path,actual,pads,hull,points,normals,incumbent):
            self.assertIsNone(incumbent)
            np.testing.assert_allclose(path.at(splice),actual)
            return dict(reason="new_target"), Path(splice,np.tile(actual,(3,1)))
        with patch.object(FrontierPlanner,"plan",side_effect=solve) as solve_call:
            row,tail = self.planner.plan(10,44,invalid_path,self.actual,self.hull,self.hull,
                                         self.points,self.normals,{"point":self.points[0]})
        self.assertEqual(solve_call.call_count,1)
        self.assertTrue(row["replanned_from_measured"])
        self.assertEqual(tail.start,24)
        self.assertTrue(self.clearance.valid(tail.samples).all())

    def test_valid_start_keeps_path_and_incumbent(self):
        path = Path(0,self.actual)
        incumbent = {"point":self.points[0]}
        with patch.object(FrontierPlanner,"plan",return_value=({"reason":"keep_current_target"},None)) as solve:
            row,_ = self.planner.plan(10,44,path,self.actual,self.hull,self.hull,self.points,self.normals,incumbent)
        self.assertFalse(row["replanned_from_measured"])
        self.assertIs(solve.call_args.args[2],path)
        self.assertIs(solve.call_args.args[-1],incumbent)

    def test_rejected_replan_clears_target_and_holds_measured_wrist(self):
        wrist = SimpleNamespace(path=Path(0,self.actual))
        worker = MeasuredStartGlobalWorker.__new__(MeasuredStartGlobalWorker)
        worker.wrist, worker.measured, worker.target = wrist, self.actual, {"point":self.points[0]}
        row = dict(reason="no_clear_trajectory",replanned_from_measured=True)
        with patch.object(WatchedWorker,"install",return_value=row):
            event = worker.install(24)
        self.assertIsNone(worker.target)
        self.assertEqual(wrist.path.start,24)
        np.testing.assert_allclose(wrist.path.at(40),self.actual)
        follower = MeasuredStartWorker.__new__(MeasuredStartWorker)
        follower.reference = object()
        with patch.object(PalmWorker,"receive"):
            follower.receive(event)
        self.assertIsNone(follower.reference)

    def test_prefix_checks_every_sample_and_respects_motion_limits(self):
        for dz,expected_end in ((.01,44),(-.01,10)):
            end = self.actual.copy()
            end[2] += dz
            path = measured_prefix(10,44,self.actual,self.actual,(self.actual,end),
                                   self.clearance,self.actual,self.cfg)
            np.testing.assert_allclose(path.at(10),self.actual)
            self.assertEqual(path.end,expected_end)
            self.assertTrue(self.clearance.valid(path.samples).all())
            self.assertTrue(np.all(extrema(path.samples)<=limits(self.cfg)*(1+1e-8)))


if __name__ == "__main__":
    unittest.main()
