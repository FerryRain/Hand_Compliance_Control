"""Execute prechecked object-pose waypoints with unchanged fixed-reference MCC."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
import collect_contact_wm as collection
import collect_contact_wm_interior as interior
from collect_contact_wm_motion import MotionProbe

class ObjectWaypointProbe(interior.InteriorProbe):
    def prepare_motion(self,meta):
        path=Path(self.args.object_waypoints)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=self.args.object_waypoints_sha256:
            raise ValueError('Object waypoint hash mismatch')
        with np.load(path) as h:plan={k:h[k] for k in h.files}
        if plan['commanded'].shape!=(self.args.frames,self.n,7):raise ValueError('Wrong waypoint shape')
        command=plan['commanded'].copy();command[:,:,:3]+=self.origins.cpu().numpy()[None]
        if abs(command[0]-self.driver_pose.cpu().numpy()).max()>1e-6:raise ValueError('Initial waypoint handoff mismatch')
        plan['commanded']=command
        self.object_path=torch.tensor(command,dtype=torch.float32,device=self.device)
        folder=collection.episode_folder(self.args,self.row)
        np.savez_compressed(folder/'object_path_plan.npz',**plan)
        (folder/'object_waypoint_design.json').write_text(path.with_suffix('.json').read_text())
        MotionProbe.prepare_motion(self,meta)
        meta.update(nominal_reference='fixed virtual pregrasp; object waypoints planned offline',
            no_future_waypoint_planning=False,interior_initialization=self.interior_init_report,
            object_waypoints_sha256=self.args.object_waypoints_sha256,
            mcc_candidate_overrides=dict(incremental_force_compliance=self.mcc.incremental_force_compliance,
                side_mobility=self.args.side_mobility,force_target_n=self.mcc.force_setpoint.cpu().tolist()),
            training_admissible=False)
        self.row['object_motion_protocol'].update(object_pose_waypoints=True,no_finger_waypoints=True)

if __name__=='__main__':
    interior.InteriorProbe=ObjectWaypointProbe
    interior.main()
