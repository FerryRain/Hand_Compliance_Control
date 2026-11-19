"""Offline 3-D banana sweeps; fixed finger reference and unchanged MCC.

Transport the curved body longitudinally while changing roll/pitch/yaw and
sampling across its crest. Only XY is centered; surface Z is deliberately not
tracked. The existing 40 mm distance band and palm clearance remain active.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from contact_wm_clearance_waypoints import Geometry
from contact_wm_distance_band import surface_heights, smooth_band
from contact_wm_feature_motion import blend
from contact_wm_geometry import collision_planes, collision_line_exit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def smooth_knots(values, n):
    u = np.linspace(0., len(values)-1, n)
    left = np.minimum(u.astype(int), len(values)-2)
    w = blend(u-left)
    values = np.asarray(values)
    return values[left] + (values[left+1]-values[left])*w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--roll-deg', nargs='+', type=float, default=[30., 50.])
    ap.add_argument('--pivot-z', type=float, default=.08,
                    help='Material-frame rotation pivot height; not a surface-height servo')
    a = ap.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    reports, jobs = [], []
    for amplitude in a.roll_deg:
        name = f'banana_3d_roll{amplitude:g}'
        paths, nominal_paths, offsets, material, roots, lane_reports = [], [], [], [], [], []
        for finger in [0, 3]:
            p = json.loads((a.source/f'banana_f{finger}_pos.protocol.json').read_text())
            root = Path(p['collector']['root'])
            obj = next(r for r in json.loads((root/'object_manifest.json').read_text())['objects']
                       if r['object_id'] == p['object'])
            source = root/'episodes/single_finger'/p['collector']['run_tag']/(p['object']+'_mu1_k1')
            ref = json.loads(Path(p['interior_reference']).read_text())['rows'][0]
            roots.append(ref)
            with np.load(p['collector']['object_waypoints']) as h:
                old = h['commanded'][:, 0].copy()
                points = h['material_feature_point'][:, 0].copy()
            n = len(old)
            g = Geometry(source, 0)
            g.q0 = np.asarray(ref['q'], float)
            g.cache = None
            g.evaluate(g.q0[g.idx])
            with h5py.File(source/'rollout.h5') as h:
                origin = np.asarray(json.loads(h.attrs['metadata'])['env_origins'][0])
            roll = amplitude*smooth_knots([0., 1., -.8, .7, 0.], n)
            pitch = amplitude*.30*smooth_knots([0., -.5, .7, -.6, 0.], n)
            yaw = amplitude*.40*smooth_knots([0., .4, -.3, .5, 0.], n)
            rotations = Rotation.from_euler('xyz', np.c_[roll, pitch, yaw], degrees=True)
            matrices = rotations.as_matrix()
            # Walk around each body cross-section as well as along its length.
            # A lateral XY offset alone would still ride the same flat crest.
            # Use the real collision union, never a full-object convex hull.
            axial_centers = points.copy()
            axial_centers[:, 2] = 0.
            surface_ids = np.unique(np.r_[np.arange(0,n,10),n-1])
            surface_points = []
            for t in surface_ids:
                phi = np.deg2rad(roll[t])
                direction = np.array([0., np.sin(phi), np.cos(phi)])
                distance,_ = collision_line_exit(root,obj,axial_centers[t],direction)
                surface_points.append(axial_centers[t]+distance*direction)
            surface_points = np.asarray(surface_points)
            points = np.stack([np.interp(np.arange(n),surface_ids,surface_points[:,k]) for k in range(3)],axis=1)
            world_anchor = old[0, :3]+points[0]
            poses = old.copy()
            transformed = np.einsum('tij,tj->ti', matrices, points)
            poses[:, :2] = world_anchor[:2]-transformed[:, :2]
            pivot = axial_centers.copy()
            pivot[:, 2] = a.pivot_z
            # Rotate around a local body-section pivot, not the mesh origin.
            # This removes the large lever arm of the banana's curved axis.
            # The surface still rises/falls relative to the fixed hand.
            rotated_pivot = np.einsum('tij,tj->ti', matrices, pivot)
            poses[:, 2] = old[0, 2]+pivot[:, 2]-rotated_pivot[:, 2]
            poses[:, 3:] = rotations.as_quat()[:, [3, 0, 1, 2]]
            prescribed = poses.copy()
            gid = g.m.geom('robot/palm_lower_collision').id
            mid = g.m.geom_dataid[gid]
            vertices = g.m.mesh_vert[g.m.mesh_vertadr[mid]:g.m.mesh_vertadr[mid]+g.m.mesh_vertnum[mid]]
            pv = vertices@g.d.geom_xmat[gid].reshape(3,3).T+g.d.geom_xpos[gid]-origin
            grid = np.array([(x,y) for x in np.linspace(pv[:,0].min(),pv[:,0].max(),5)
                             for y in np.linspace(pv[:,1].min(),pv[:,1].max(),5)])
            patch = world_anchor[:2]+np.array([[0,0],[.004,0],[-.004,0],[0,.004],[0,-.004]])
            planes = collision_planes(str(root/'objects'/p['object']), tuple(obj['collision_parts']))
            ids = np.unique(np.r_[np.arange(0,n,10),n-1])
            heights = np.concatenate([surface_heights(planes,poses[ii],np.vstack([patch,grid]))
                                      for ii in np.array_split(ids,20)])
            high = np.max(np.where(np.isfinite(heights[:,:5]),heights[:,:5],-np.inf),axis=1)
            if not np.isfinite(high).all():
                raise ValueError(f'{name}/f{finger}: path outside actual collision surface')
            lower = np.interp(np.arange(n),ids,high[0]-.04-high)
            upper = np.interp(np.arange(n),ids,high[0]+.04-high)
            palm_high = np.max(np.where(np.isfinite(heights[:,5:]),heights[:,5:],-np.inf),axis=1)
            upper = np.minimum(upper,np.interp(np.arange(n),ids,pv[:,2].min()-.008-palm_high))
            if np.any(lower>upper):
                raise ValueError(f'{name}/f{finger}: incompatible distance/palm band')
            z,_,triggers = smooth_band(lower,upper,vmax=.025,amax=.12)
            poses[:,2] += z
            checks, hits, normals = [], [], []
            for t in np.unique(np.r_[np.arange(0,n,50),n-1]):
                pose = poses[t].copy();pose[:3] += origin
                g.place(pose)
                pad,c,sg,_ = g.evaluate(g.q0[g.idx])
                checks.append(dict(frame=int(t),pad_gap_m=float(pad),nonpad_min_m=float(c.min()),self_min_m=float(sg.min())))
                rot = matrices[t]
                ray_start = rot.T@(np.r_[world_anchor[:2],0.]-poses[t,:3])
                ray_dir = rot.T@np.array([0.,0.,1.])
                distance,normal = collision_line_exit(root,obj,ray_start,ray_dir)
                hits.append(ray_start+distance*ray_dir)
                normals.append(rot@normal)
            speed = np.linalg.norm(np.diff(poses[:,:3],axis=0),axis=1).max()/.01
            angular = (rotations[:-1].inv()*rotations[1:]).magnitude().max()/.01
            nonpad = min(x['nonpad_min_m'] for x in checks)
            lag = np.maximum(np.maximum(lower-z,z-upper),0).max()
            valid = nonpad>=.004 and speed<=.08 and angular<=.4 and lag<=.006
            normals = np.asarray(normals)
            angle = np.degrees(np.arccos(np.clip(normals@normals[0],-1,1)))
            lane_reports.append(dict(finger=finger,ready=bool(valid),max_speed_m_s=float(speed),max_angular_rad_s=float(angular),nonpad_min_m=nonpad,
                distance_band_lag_m=float(lag),height_correction_span_m=float(np.ptp(z)),height_trigger_fraction=float(np.mean(triggers!=0)),
                surface_height_span_m=float(np.ptp(high)),world_normal_change_max_deg=float(angle.max()),
                material_contact_query_span_m=np.ptp(hits,axis=0).tolist(),checks=checks))
            paths.append(poses);nominal_paths.append(prescribed);offsets.append(z);material.append(points)
        report = dict(id=name,lanes=lane_reports,ready=all(x['ready'] for x in lane_reports),half_band_m=.04,roll_amplitude_deg=amplitude,pivot_z_m=a.pivot_z,
            scope='3-D prescribed object sweep, no surface-Z tracking inside band; unchanged fixed virtual reference/MCC. Sampled nominal geometry is not physical acceptance.',training_admissible=False)
        reports.append(report)
        (out/f'{name}.json').write_text(json.dumps(report,indent=2))
        if report['ready']:
            path=out/f'{name}.npz'
            np.savez_compressed(path,commanded=np.stack(paths,1),prescribed=np.stack(nominal_paths,1),height_offset=np.stack(offsets,1),material_feature_point=np.stack(material,1))
            rp=out/f'{name}.roots.json';rp.write_text(json.dumps(dict(rows=roots,virtual_only=True),indent=2))
            protocol=copy.deepcopy(p);tag=out.name+'_'+name
            protocol.update(id=name,interior_reference=str(rp),interior_reference_sha256=sha(rp),scope=report['scope'])
            protocol['collector'].update(fingers=[0,3],run_tag=tag,object_waypoints=str(path),object_waypoints_sha256=sha(path))
            protocol['source_hashes']['prepare_contact_wm_banana_3d.py']=sha(Path(__file__))
            pp=out/f'{name}.protocol.json';pp.write_text(json.dumps(protocol,indent=2))
            jobs.append(dict(id=name,protocol=str(pp),collector='collect_contact_wm_object_waypoints.py',folder=str(root/'episodes/single_finger'/tag/(p['object']+'_mu1_k1')),expected_frames=n,expected_lanes=2))
        print(name,json.dumps([{k:v for k,v in x.items() if k!='checks'} for x in lane_reports]),flush=True)
    (out/'preflight.json').write_text(json.dumps(reports,indent=2))
    (out/'queue.json').write_text(json.dumps(dict(jobs=jobs,planned_lanes=len(jobs)*2,training_admissible=False),indent=2))


if __name__ == '__main__':
    main()
