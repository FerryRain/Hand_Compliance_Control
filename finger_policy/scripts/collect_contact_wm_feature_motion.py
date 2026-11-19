"""Isolated feature-crossing pilot; inherited MCC and actuator unchanged."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
import collect_contact_wm as collection
import collect_contact_wm_interior as interior
from collect_contact_wm_motion import MotionProbe
from contact_wm_feature_motion import feature_path


class FeatureProbe(interior.InteriorProbe):
    def prepare_motion(self,meta):
        model=self.env.sim.mj_model;data=mujoco.MjData(model)
        gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/palm_lower_collision')
        mesh=int(model.geom_dataid[gid]);verts=model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
        paths=[];reports=[]
        state=self.sample(1)
        for lane,job in enumerate(self.jobs):
            palm=self.palm[lane].cpu().numpy();mid=self.driver.mocap_id
            data.mocap_pos[mid]=palm[:3];data.mocap_quat[mid]=palm[3:];mujoco.mj_kinematics(model,data)
            pv=verts@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
            grid=np.array([(x,y) for x in np.linspace(pv[:,0].min(),pv[:,0].max(),5)
                                  for y in np.linspace(pv[:,1].min(),pv[:,1].max(),5)])
            home=self.driver_pose[lane].cpu().numpy()
            anchor=state['pad_representative_point_world'][lane,job['finger']]
            path,report=feature_path(self.args.root,self.row,home,anchor,grid,pv[:,2].min(),self.args.frames,
                                     half_band=self.args.feature_half_band_m,
                                     edge_sign=getattr(self.args,'feature_edge_sign',1))
            paths.append(path);reports.append(dict(lane=lane,finger=job['finger'],**report))
            job.update(motion_rate=1.,motion_randomized=False,physical_motion_period_s=None,
                       motion_program='one-way material-feature crossing with deadband height')
        folder=collection.episode_folder(self.args,self.row)
        np.savez_compressed(folder/'object_path_plan.npz',**{k:np.stack([p[k] for p in paths],1) for k in paths[0]})
        (folder/'feature_motion.json').write_text(json.dumps(dict(rows=reports,mcc_unchanged=True,
            no_finger_waypoints=True,training_admissible=False),indent=2))
        for r in reports:
            if r['max_translation_m_s']>.08 or r['max_angular_rad_s']>.4:
                raise ValueError('Feature motion too fast; extend physical time before collection')
            if r['incompatible_frames'] or r['remaining_band_violation_m']>.006:
                raise ValueError('Palm-clearance / reach band incompatible; needs revised motion or waypoints')
        self.object_path=torch.tensor(np.stack([p['commanded'] for p in paths],1),device=self.device,dtype=torch.float32)
        if float(abs(self.object_path[0]-self.driver_pose).max())>1e-6:raise ValueError('Initial command discontinuity')
        MotionProbe.prepare_motion(self,meta)
        meta.update(nominal_reference='fixed virtual interior pregrasp',no_future_waypoint_planning=True,
            object_path_reference='feature crossing with bounded distance-band height',
            feature_motion=reports,interior_initialization=self.interior_init_report,
            mcc_candidate_overrides=dict(incremental_force_compliance=self.mcc.incremental_force_compliance,
                side_mobility=self.args.side_mobility,force_target_n=self.mcc.force_setpoint.cpu().tolist()),
            training_admissible=False)


if __name__=='__main__':
    interior.InteriorProbe=FeatureProbe
    interior.main()
