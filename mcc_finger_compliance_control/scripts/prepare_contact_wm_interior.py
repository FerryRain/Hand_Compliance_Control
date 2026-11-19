"""Choose ONE object height for a published path and a fixed virtual grasp.

Geometry is used offline to place the object and verify the virtual reference,
not to change runtime height or finger targets. Physical initialization remains
separated and is independently checked by the collector.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from collect_contact_wm_prescribed import PrescribedSweep
from contact_wm_geometry import collision_planes, surface_anchor


def collision_top(folder, parts, xy):
    top=np.full(len(xy),-np.inf)
    for e in collision_planes(str(folder),tuple(parts)):
        value=xy@e[:,:2].T+e[:,3]
        up=e[:,2]>1.e-9;down=e[:,2]<-1.e-9;side=~(up|down)
        if not up.any():continue
        high=(-value[:,up]/e[up,2]).min(-1)
        low=(-value[:,down]/e[down,2]).max(-1) if down.any() else -np.inf
        valid=(high>=low-1.e-8)&(value[:,side]<=1.e-8).all(-1)
        top=np.maximum(top,np.where(valid,high,-np.inf))
    if not np.isfinite(top).all():raise ValueError('Published path leaves collision top-surface domain')
    return top


def prepare(root,row,template,seed,frames,depth,height_policy='path_min'):
    if row['backend']!='convex_parts':raise ValueError('Complete collision object required')
    folder=root/'objects'/row['object_id'];mesh=trimesh.load(folder/'surface.obj',process=False)
    if not mesh.is_watertight:raise ValueError('Virtual interior requires a watertight envelope')
    rows=[]
    for old in template['rows']:
        f=old['finger'];job_seed=seed+f*101
        rate=float(np.random.default_rng(job_seed+47021).uniform(2,4))
        values=PrescribedSweep(job_seed).evaluate(np.arange(frames)*.01,rate)
        rotation=Rotation.from_euler('z',values[:,2])
        xy=rotation.inv().apply(np.c_[-values[:,:2],np.zeros(frames)])[:,:2]
        top=collision_top(folder,row['collision_parts'],xy)
        # Use the exact collision-union upper envelope. A ray through a mesh
        # pole/triangle vertex can miss the top face and hit the bottom; do not
        # interpret such a hit as the upper surface and place the object high.
        if height_policy not in ('path_min','initial'):
            raise ValueError('Unknown static height policy')
        # Complex objects can lose contact later. Do not push their high
        # regions deeply into the hand merely to cover a later low region.
        local_z=float((top.min() if height_policy=='path_min' else top[0])-depth)
        virtual=np.c_[xy,np.full(frames,local_z)]
        inside=mesh.contains(virtual)
        if not inside[0] or (height_policy=='path_min' and not inside.all()):
            raise ValueError('Virtual target does not satisfy the requested interior policy')
        # Check the vectorized collision union against the existing scalar
        # initializer before publishing a new static placement.
        for i in np.linspace(0,frames-1,12).astype(int):
            point,_=surface_anchor(root,row,xy[i])
            if abs(point[2]-top[i])>1.e-8:raise ValueError('Collision-height implementations disagree')
        home=np.array(old['home_pose_local'],float)
        home[:3]=old['point_world'];home[2]-=local_z
        depths=top-local_z
        rows.append(dict(finger=f,q=old['q'],point_world=old['point_world'],
            home_pose_local=home.tolist(),home_pose_world=home.tolist(),virtual_only=True,
            local_anchor_xy=[0.,0.],motion_frames=frames,seed=job_seed,motion_rate=rate,
            full_path_point_inside_visual_mesh=bool(inside.all()),
            virtual_point_inside_fraction=float(inside.mean()),
            static_height_policy=height_policy,
            collision_surface_virtual_depth_range_m=[float(depths.min()),float(depths.max())],
            method='One fixed pregrasp and rigid placement; constant Z during prescribed XY/yaw, no runtime geometry feedback',
            reference_definition='Virtual equilibrium only; actual initialized q/object must separately pass no-penetration checks'))
    return dict(object_id=row['object_id'],mesh_sha256=row['mesh_sha256'],seed=seed,rows=rows,
                runtime_height_fitting=False,physical_teacher_admission=False,
                whole_hand_admission=False,scope='Four independent single fingers')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--objects',nargs='+',required=True)
    p.add_argument('--seeds',nargs='+',type=int,required=True)
    p.add_argument('--frames',type=int,default=1500)
    p.add_argument('--depth-m',type=float,default=.005)
    p.add_argument('--height-policy',choices=['path_min','initial'],default='path_min',
                   help='initial permits later loss instead of fitting height to the lowest future surface')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--heldout-evaluation',action='store_true',
                   help='Use unchanged placement on a frozen held-out split; never admit these references to training')
    a=p.parse_args();a.root=a.root.resolve();a.output.mkdir(exist_ok=False,parents=True)
    manifest=json.loads((a.root/'object_manifest.json').read_text());template=json.loads(a.template.read_text())
    records=[]
    for name in a.objects:
        row=next(r for r in manifest['objects'] if r['object_id']==name)
        if row['split']!='train' and not a.heldout_evaluation:
            raise ValueError('Development placement uses training objects only; held-out evaluation must be explicit')
        if row['split']=='train' and a.heldout_evaluation:
            raise ValueError('Held-out evaluation requires a non-training object')
        for seed in a.seeds:
            try:
                result=prepare(a.root,row,template,seed,a.frames,a.depth_m,a.height_policy)
                result.update(object_split=row['split'],heldout_evaluation=a.heldout_evaluation,
                              training_admissible=False)
                path=a.output/f'{name}_seed{seed}.json';path.write_text(json.dumps(result,indent=2))
                records.append(dict(object=name,seed=seed,status='prepared',reference=str(path.resolve()),
                                    sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            except ValueError as error:
                records.append(dict(object=name,seed=seed,status='preflight_rejected',reason=str(error)))
            (a.output/'manifest.json').write_text(json.dumps(dict(rows=records,training_admissible=False),indent=2))
            print(records[-1],flush=True)


if __name__=='__main__':main()
