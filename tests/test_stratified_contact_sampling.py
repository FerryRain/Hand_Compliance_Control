"""Region credit must use measured contact and independent parent paths."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from prepare_variable_collection_references import write_reference
from stratified_contact_sampling import (parent_coverage,region_order,region_reached,
    target_windows,targeted_spec)


class StratifiedSamplingTests(unittest.TestCase):
    def test_children_and_paired_speeds_do_not_multiply_region_credit(self):
        contact=dict(visited_cells=[[4,2]],visited_height_bands=[4])
        def row(parent):
            return dict(selected=True,related_plan_group_id=parent,
                        measured_coverage=dict(strict=contact,found=contact))
        coverage=parent_coverage([row('A'),row('A'),row('B')])
        self.assertEqual(coverage['cells'][4][2],2)
        self.assertEqual(coverage['heights'][4],2)

    def test_priority_and_reservations_prefer_missing_top_regions(self):
        coverage=dict(cells=np.full((5,12),10).tolist(),heights=[15]*5)
        coverage['cells'][4]=[0]*12;coverage['heights'][4]=1
        self.assertEqual(region_order(coverage,{})[0],(4,0))
        self.assertEqual(region_order(coverage,{},[(4,0)])[0],(4,1))
        self.assertEqual(region_order(coverage,{(4,0):1})[0],(4,1))

    def test_found_or_nominal_contact_does_not_credit_strict_target(self):
        measured=dict(strict=dict(visited_cells=[[2,3]]),found=dict(visited_cells=[[2,3],[4,2]]))
        self.assertFalse(region_reached(measured,[4,2]))
        self.assertTrue(region_reached(measured,[4,2],'found'))

    def test_long_reference_first_then_long_valid_children(self):
        band=np.full((5000,4),2);band[3000:3100]=4
        nominal=dict(band=band,sector=np.full((5000,4),3),valid=np.ones((5000,4),bool))
        full=target_windows(nominal,(4,3),800,4000,False)
        self.assertEqual(full[0],(0,5000))
        children=target_windows(nominal,(4,3),800,4000,True)
        self.assertNotIn((0,5000),children)
        self.assertEqual(children[0][1]-children[0][0],4000)
        self.assertTrue(all(b-a>=800 for a,b in children))
        nominal['band'][:]=2
        self.assertEqual(target_windows(nominal,(4,3)),[])

    def test_regional_proposals_explicitly_reach_beyond_old_midbody_range(self):
        cap=targeted_spec(1,20261015,(4,5),'meridian_arc')
        bottom=targeted_spec(2,20261015,(0,7),'meridian_arc')
        upper_ring=targeted_spec(3,20261015,(4,2),'oblique_orbit')
        self.assertEqual(cap['arc_region'],'cap')
        self.assertEqual(bottom['arc_region'],'bottom')
        self.assertGreater(upper_ring['section_fraction'],.9)
        self.assertEqual(cap['target_region'],[4,5])

    def test_repeated_reference_splits_preserve_original_frame_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.h5';child=Path(directory)/'child.h5';grand=Path(directory)/'grand.h5'
            with h5py.File(source,'w') as h:
                h.attrs['healthy_reference_admissible']=True
                h['palm_pose_object']=np.arange(140,dtype=np.float32).reshape(20,1,7)
                h['palm_twist_object']=np.zeros((20,1,6),np.float32)
                for name in ('q_hand','qvel','grasp_keyframe_q'):h[name]=np.arange(320,dtype=np.float32).reshape(20,1,16)
                for name in ('record_step','grasp_keyframe_frame_index'):h[name]=np.arange(20)[:,None]
                for name in ('grasp_keyframe_valid','grasp_keyframe_true_tip_geometry_valid'):h[name]=np.ones((20,1),bool)
                h['grasp_keyframe_non_tip_distance']=np.ones((20,1,17))
                h['grasp_keyframe_local_motion_capacity_m']=np.ones((20,1,4,5))
            write_reference(source,child,4,16,'A')
            write_reference(child,grand,2,8,'A')
            with h5py.File(source) as h,h5py.File(grand) as g:
                np.testing.assert_array_equal(g['palm_pose_object'][:],h['palm_pose_object'][6:12])
                np.testing.assert_array_equal(g['subtrajectory_source_frame'][:],np.arange(6,12))
                np.testing.assert_array_equal(g['grasp_keyframe_frame_index'][:,0],np.arange(6))


if __name__=='__main__':unittest.main()
