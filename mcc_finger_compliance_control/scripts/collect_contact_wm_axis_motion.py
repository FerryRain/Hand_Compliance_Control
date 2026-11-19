"""Cylinder-axis motion with a fixed virtual grasp and unchanged MCC.

Axes come from explicit object geometry metadata. This is a prepublished rigid
motion program, not force feedback, height compensation or fingertip tracking.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import torch

import collect_contact_wm as collection
import collect_contact_wm_rooted_interaction as rooted
from collect_contact_wm_virtual_root import VirtualRootProbe
from collect_contact_wm_motion import MotionProbe
from contact_wm_motion_profiles import normal_excursion_m


def axis_path(home, axis, job, frames, dt, period, slide_amplitude):
    home, axis = np.asarray(home, float), np.asarray(axis, float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('Invalid explicit cylinder axis')
    axis = axis/np.linalg.norm(axis)
    t = np.arange(frames)*dt*job['motion_rate']
    phase = 2*np.pi*t/period
    u = np.minimum(t/.3,1.)
    ramp = u*u*(3-2*u)
    signal = np.sin(phase)*ramp*job.get('motion_amplitude_scale',1.)
    initial_rotation = Rotation.from_quat(home[[4,5,6,3]])
    path = np.tile(home,(frames,1))
    if job['behavior'] in ('slide','slide_release'):
        sign = 1 if job['seed']%2 else -1
        path[:,:3] += signal[:,None]*sign*slide_amplitude*initial_rotation.apply(axis)[None]
        label = 'axial_translation' if job['behavior']=='slide' else 'axial_translation_with_release'
    elif job['behavior']=='rotate':
        path[:,3:] = (initial_rotation*Rotation.from_rotvec(signal[:,None]*.12*axis[None])).as_quat()[:,[3,0,1,2]]
        label = 'rotation_about_own_axis'
    else:
        raise ValueError('Axis motion only supports slide/rotate/slide_release')
    path[0] = home
    return path,label


class AxisMotionProbe(VirtualRootProbe):
    def prepare_motion(self, meta):
        if self.args.motion_owner!='object' or self.args.object_surface_follow:
            raise ValueError('Axis experiment requires independent object motion')
        if float((self.driver_pose-self.published_home).abs().max())>1e-6:
            raise ValueError('Loading handoff discontinuity')
        paths=[]
        for lane,job in enumerate(self.jobs):
            path,label=axis_path(self.published_home[lane].cpu().numpy(),self.args.motion_axis_object,
                                 job,self.args.frames,.01,self.args.motion_period_s,self.args.slide_amplitude_m)
            if job['behavior']=='slide_release':
                excursion=np.array([normal_excursion_m(job,f,self.args.motion_period_s) for f in range(self.args.frames)])
                path[:,:3]+=excursion[:,None]*self.inward[lane].cpu().numpy()[None]
            paths.append(path)
            job.update(motion_geometry=label,motion_axis_object=list(self.args.motion_axis_object),
                       force_target_n=self.args.force_target_n)
        self.object_path=torch.tensor(np.stack(paths,axis=1),device=self.device,dtype=torch.float32)
        if float((self.object_path[0]-self.published_home).abs().max())>1e-6:
            raise ValueError('Motion starts away from loading endpoint')
        speed=(self.object_path[1:,:,:3]-self.object_path[:-1,:,:3]).norm(dim=-1)/.01
        if float(speed.max())>self.args.max_object_command_speed:
            raise ValueError('Object command exceeds original speed bound')
        MotionProbe.prepare_motion(self,meta)
        meta.update(no_future_waypoint_planning=True,
            nominal_reference='fixed virtual inward grasp, healthy separated physical initial q',
            virtual_root_reference=self.virtual_reference_report,
            object_path_reference='explicit cylinder-axis translation/rotation; no runtime height compensation',
            motion_axis_object=list(self.args.motion_axis_object),
            collection_purpose='WM axis motion interactions; failures retained',
            interior_initialization=self.interior_init_report,
            mcc_candidate_overrides=dict(incremental_force_compliance=True,side_mobility=0.,
                force_target_n=self.args.force_target_n,outward_travel_m=self.args.outward_travel_m,
                normal_step_m=self.args.normal_step_m),
            motion_command_max_speed_m_s=float(speed.max()),training_admissible=False)
        folder=collection.episode_folder(self.args,self.row)
        np.savez_compressed(folder/'object_path_plan.npz',commanded=self.object_path.cpu().numpy())
        (folder/'axis_motion.json').write_text(json.dumps(dict(axis_object=self.args.motion_axis_object,
            source=self.args.motion_axis_source,control_dt=.01,geometry_feedback=False,
            constant_height_enforced=False,physical_motion_time_s=self.args.frames*.01),indent=2))


if __name__=='__main__':
    rooted.RootedInteractionProbe=AxisMotionProbe
    rooted.main()
