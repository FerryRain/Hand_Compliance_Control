"""Collection must preserve initialization and report physical region coverage."""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS=Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'
sys.path.insert(0,str(SCRIPTS))
from collect_regional_healthy_reference import jacobian_force_config
from regional_contact_coverage import SurfaceRegions,regional_priority


class RegionalCollectionTests(unittest.TestCase):
    def regions(self):
        return SurfaceRegions(np.array([[x,y,z] for x in (-.03,.03)
            for y in (-.02,.02) for z in (-.10,0.,.10)]))

    def test_only_motion_normal_mapping_changes(self):
        @dataclass
        class Config:
            normal_synergy_control:bool=True
            force:float=3.
            limit:float=.025
        original=lambda cfg,profile:replace(cfg,force=6.) if profile=='regulated' else cfg
        config=Config()
        self.assertIs(jacobian_force_config(config,'legacy',original),config)
        expected=asdict(original(config,'regulated'));expected['normal_synergy_control']=False
        self.assertEqual(asdict(jacobian_force_config(config,'regulated',original)),expected)

    def test_region_coverage_counts_distinct_frames_not_fingers(self):
        regions=self.regions()
        points=np.tile([.03,.02,0.],(50,4,1))
        short=regions.coverage(points[:49],np.ones((49,4),bool))
        self.assertEqual(short['visited_cells'],[])
        full=regions.coverage(points,np.ones((50,4),bool))
        self.assertEqual(len(full['visited_cells']),1)
        self.assertEqual(np.asarray(full['cell_distinct_frames']).max(),50)

    def test_measured_contacts_use_inverse_object_pose(self):
        regions=self.regions();n=60
        points=np.tile([.03,.02,0.],(n,4,1))
        rotation=Rotation.from_euler('xyz',[.7,.2,-.8])
        translation=np.array([.4,-.3,.8])
        world=rotation.apply(points.reshape(-1,3)).reshape(n,4,3)+translation
        quat=rotation.as_quat();pose=np.r_[translation,quat[3],quat[:3]]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'collected.h5'
            with h5py.File(path,'w') as h:
                h['fingertip_contact_pos_world']=world[:,None]
                h['object_pose_world']=np.tile(pose,(n,1,1))
                h['true_fingertip_contact_valid']=np.ones((n,1,4))
                h['fingertip_collision_found']=np.ones((n,1,4))
            measured=regions.measured(path)
        expected=regions.coverage(points,np.ones((n,4),bool))
        self.assertEqual(measured['strict'],expected)
        self.assertEqual(measured['found'],expected)

    def test_priority_prefers_underrepresented_regions(self):
        heights=[5,0,3,4,2];cells=np.ones((5,12),int)
        old=dict(visited_height_bands=[0],visited_cells=[[0,0]])
        new=dict(visited_height_bands=[1],visited_cells=[[1,0]])
        self.assertLess(regional_priority(new,heights,cells),regional_priority(old,heights,cells))


if __name__=='__main__':unittest.main()
