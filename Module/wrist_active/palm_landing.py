"""Solve for a palm ray landing and a finger frontier on the same observed object."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import NOISE_VARIANCE
from ..acitive_02.surface_scan import align,limit
from ..i04_geometric_traversal.physics import quaternion
from . import palm_facing as first,tactile_field as field
from .multirate_wrist import SPEED_M_S
from .palm_acquisition import PalmGoal,PalmGuide,score_poses
from .palm_surface import OUT,footprint,normalize,on_plane,palm_sites
from .sweep import DT,PREVIEW_S,LOAD_LIMIT
from .ycb_fixtures import OBJECTS


@dataclass
class Landing(PalmGoal):
    site: np.ndarray
    height: float


def solve_pair(point,normal,frontier,lever,site,side):
    """Exact SE(3) poses with p+h*u at the hand site and the selected pad at the frontier."""
    delta=frontier-point
    distance=np.linalg.norm(delta,axis=1)
    radial=np.linalg.norm(lever[:2])
    if radial<.005:
        return None
    direction=normalize(delta)
    along=side*np.sqrt(np.maximum(0.,distance**2-radial**2))
    h=along-lever[2]
    nz=along/np.maximum(distance,1.e-8)
    perp=on_plane(normal,direction)
    outward=normalize(nz[:,None]*direction+(radial/np.maximum(distance,1.e-8))[:,None]*normalize(perp))
    cosine=np.einsum('ai,ai->a',normal,outward)
    ok=(distance>=radial)&(h>=.024)&(h<=.158)&(cosine>.64)&(np.linalg.norm(perp,axis=1)>.12)
    idx=np.flatnonzero(ok)
    if not len(idx):
        return None
    ex=normalize(np.r_[lever[:2],0.]);ey=np.cross(OUT,ex)
    world_x=normalize(on_plane(delta[idx],outward[idx]))
    axes=np.stack((world_x,np.cross(outward[idx],world_x),outward[idx]),axis=-1)
    rot=Rotation.from_matrix(axes@np.stack((ex,ey,OUT)))
    pos=point[idx]+h[idx,None]*outward[idx]-rot.apply(site)
    return idx,h[idx],rot,pos,cosine[idx]


class LandingGuide(PalmGuide):
    def bank(self,model,surface):
        mesh=model.mesh
        if mesh is not None and len(mesh.faces):
            if getattr(self,'bank_mesh',None) is not mesh:
                n=mesh.normals
                p=mesh.centers-model.band.half_width*n
                _,first_idx=np.unique(np.c_[np.floor(p/.014),np.argmax(abs(n),axis=1),np.sign(n[np.arange(len(n)),np.argmax(abs(n),axis=1)])],
                                      axis=0,return_index=True)
                self.bank_points,self.bank_normals=p[first_idx],n[first_idx]
                self.bank_mesh=mesh
            return self.bank_points,self.bank_normals
        return surface.points,surface.normals

    def choose(self,surface,pads,ids,normal,event):
        self.palm_goal=None
        state=self.state
        wrist,att=state['wrist_position'],state['wrist_rotation']
        hand=att.inv().apply(pads-wrist)
        points,normals=self.bank(self.model,surface)
        regions=field.mesh_regions(self.model,pads,ids,state['normals'])
        regions+=field.gradient_regions(self.model,surface,pads,ids,state['normals'],self.rng)
        rows=[]
        center=self.sites[0]
        # The center ray is preferred. Near an incompletely observed edge we
        # also allow two points inside the physical plate, toward the long
        # fingers or thumb. Their locations come from the actual hand geometry.
        for i,r in enumerate(regions):
            distance=np.linalg.norm(points-r.point,axis=1)
            keep=(distance>.028)&(distance<.20)
            p,n=points[keep],normals[keep]
            if not len(p):
                continue
            finger=r.finger
            ahead=normalize(on_plane(hand[finger]-center,OUT))
            edge=self.sites[np.argmax((self.sites-center)@ahead)]
            candidates=[]
            for site in (center,center+.68*(edge-center)):
                for sign in (-1.,1.):
                    solutions=solve_pair(p,n,r.point,hand[finger]-site,site,sign)
                    if solutions is None:
                        continue
                    ind,h,rr,pos,cosine=solutions
                    angular=(rr*att.inv()).magnitude()
                    transfer=np.maximum(np.linalg.norm(pos-wrist,axis=1)/.13,angular/1.0)
                    skin=pos[:,None]+np.einsum('nij,fj->nfi',rr.as_matrix(),hand)
                    d,near=surface.tree.query(skin)
                    delta=skin-surface.points[near]
                    gaps=np.sum(delta*surface.normals[near],axis=2)
                    support=np.exp(-.5*(gaps/.011)**2-.5*(d/.028)**2)
                    support[:,finger]=0.
                    h0=.50*np.ptp(self.sites[:,1])
                    short=np.exp(-.5*(np.maximum(h-h0,0.)/.050)**2)
                    score=(.25+.75*support.max(1))*(.45+.55*short)*cosine/(.60+transfer)
                    score*=np.exp(-np.linalg.norm(site-center)/.070)
                    feasible=np.flatnonzero((np.linalg.norm(pos-state['workspace_center'],axis=1)<.445)&(angular<2.65))
                    for k in feasible[np.argsort(-score[feasible])[:2]]:
                        k=int(k)
                        candidates.append((score[k],i,site,p[ind[k]],n[ind[k]],float(h[k]),rr[k],pos[k],float(transfer[k])))
            retained=[]
            for trial in sorted(candidates,key=lambda v:-v[0]):
                if any(np.linalg.norm(trial[7]-old[7])<.010 and (trial[6]*old[6].inv()).magnitude()<.14 for old in retained):
                    continue
                retained.append(trial)
                if len(retained)==3:
                    break
            rows.extend(retained)
        if not rows:
            super().choose(surface,pads,ids,normal,event)
            event['palm_pair_solved']=False
            return
        owners=np.array([regions[a[1]].finger for a in rows])
        ends=np.array([regions[a[1]].point for a in rows])
        poses=Rotation.from_quat(np.stack([a[6].as_quat() for a in rows]))
        positions=np.stack([a[7] for a in rows])
        paths=np.array([regions[a[1]].distance for a in rows])
        durations=np.maximum(np.array([a[8] for a in rows]),paths/SPEED_M_S)
        gain,pair,three,receiver,legal=score_poses(state,self.model,self.history,owners,ends,poses,positions,durations)
        desired=poses.apply(OUT)
        target=np.stack([a[3] for a in rows])
        ns=np.stack([a[4] for a in rows])
        surface_sites=positions[:,None]+np.einsum('aij,sj->asi',poses.as_matrix(),self.sites)
        broad,rays,_,_,_,_=footprint(self.patches,self.model,surface,surface_sites,target,ns,desired,np.full(len(rows),-1))
        if not np.any(legal&(gain*receiver>1.e-8)):
            prior,conditional=self.model.observation_covariances(ends)
            gain=.5*np.log((NOISE_VARIANCE+prior.diagonal())/(NOISE_VARIANCE+conditional.diagonal()))
            gain*=np.clip(surface.tree.query(ends)[0]/.024,0.,1.)
            receiver=np.maximum(receiver,.20)
        heights=np.array([a[5] for a in rows])
        short=1./(1.+(np.maximum(heights-.065,0.)/.045)**2)
        score=gain*(.20+.80*receiver)*(.24+.76*pair)*(.65+.35*three)*(.24+.76*broad)*(.42+.58*short)/(.5+durations)
        if getattr(self,'recovery_observation',None) is not None:
            score*=self.after_loss_weights(regions)[[a[1] for a in rows]]
        possible=np.flatnonzero(legal&(score>0))
        if not len(possible):
            super().choose(surface,pads,ids,normal,event)
            event['palm_pair_solved']=False
            return
        k=possible[np.argmax(score[possible])]
        _,i,site,p,n,h,rot,w,_=rows[k]
        self.region=r=regions[i]
        self.palm_goal=Landing(r,w,rot,p,n,broad[k],site,h)
        event.update(source=r.source,target_m=r.point.tolist(),target_normal=r.normal.tolist(),target_finger=r.finger,
            pivot_m=pads[r.finger].tolist(),mesh_path_m=r.distance,candidates=len(rows),acquisition_nat=float(gain[k]),
            selected_rate_nat_s=float(score[k]),predicted_light3=float(three[k]),predicted_light2=float(pair[k]),
            predicted_owner_support=float(receiver[k]),palm_goal_position_m=w.tolist(),palm_goal_xyzw=rot.as_quat().tolist(),
            palm_endpoint_patch_m=p.tolist(),palm_endpoint_outward=n.tolist(),palm_endpoint_footprint=float(broad[k]),
            palm_endpoint_rays=int(rays[k].sum()),palm_endpoint_height_m=float(h),palm_pair_solved=True,
            palmar_site_local_m=site.tolist())


class LandingFollower(first.PalmFollower):
    def face_surface(self,state,row,result):
        super().face_surface(state,row,result)
        if not row['palm_reference_valid']:
            return
        target=self.guide.palm_goal
        if not isinstance(target,Landing) or target.region is not self.guide.region:
            return
        wrist,att=state['wrist_position'],state['wrist_rotation']
        site=wrist+att.apply(target.site)
        toward=normalize(site-target.patch)
        # Close the loop on both endpoints. The final orientation satisfies
        # the ray and pad constraints exactly for the hand shape used at issue.
        # At intervening states, the observed line of sight supplies feedback.
        axis=target.rotation.apply(OUT)
        usable=np.clip((toward@target.normal-.15)/.6,0.,1.)
        out=normalize((1.-.43*usable)*axis+.43*usable*toward)
        pose=align(axis,out)*target.rotation
        drift=target.patch+target.height*axis-site
        heading=result['direction']
        if np.linalg.norm(heading)>.5:
            drift-=.72*(drift@heading)*heading
        trim=limit(.55*on_plane(drift,target.normal),.025)
        vertical=np.clip(.48*(drift@target.normal),-.021,.024)
        if (max(state['loads'])>2.8 or state['loads'][-1]>.28):
            vertical=max(0.,vertical)
        trim+=vertical*target.normal
        actual=att.apply(OUT)
        row.update(palm_patch_m=target.patch.tolist(),palm_patch_outward=target.normal.tolist(),
            palm_patch_source='gpis_palm_finger_pair' if self.global_model.mesh is not None and len(self.global_model.mesh.faces) else 'measured_palm_finger_pair',
            palm_target_xyzw=pose.as_quat().tolist(),palm_trim_velocity=trim.tolist(),palm_height_m=float((site-target.patch)@target.normal),
            palm_reach_height_m=target.height,palm_facing_deg=float(np.degrees(np.arccos(np.clip(actual@target.normal,-1,1)))),
            palm_view_deg=float(np.degrees(np.arccos(np.clip(normalize(np.asarray(row['palm_center_m'])-target.patch)@actual,-1,1)))),
            palmar_site_local_m=target.site.tolist(),palm_pair_active=True)
        pivot=np.asarray(row['palm_pivot_m'])
        delta=Rotation.from_rotvec(limit((pose*att.inv()).as_rotvec(),.048))
        result.update(position=pivot+delta.apply(wrist-pivot)+.05*(trim+row['tangent_speed_m_s']*heading),rotation=delta*att)


class LandingWrist(first.PalmWrist):
    def __init__(self,*args,**opts):
        super().__init__(*args,**opts)
        base=self.selector.base
        base.global_model=self.selector.global_model.base
        sites=self.selector.sites
        base.guide=LandingGuide(base.global_model,base.guide.history,opts['seed'],sites)
        self.selector=LandingFollower(base,sites)
        self.low_load_history=[]
        self.retreat_to=None
        self.retreat_tick=-1000

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        if tick%5==0 and touch.loads.max()<2.8 and touch.loads[-1]<.30:
            self.low_load_history.append((tick,self.measured_position.copy(),self.live_attitude))
            if len(self.low_load_history)>100:
                self.low_load_history.pop(0)
        if touch.loads.max()>=LOAD_LIMIT and self.retreat_to is None:
            self.retreat_to=next(((p,r) for t,p,r in reversed(self.low_load_history)
                             if 12<=tick-t<=250 and np.linalg.norm(self.measured_position-p)>.003),None)
            self.retreat_tick=tick
        if touch.loads.max()<3.5:
            self.retreat_to=None

    def command(self,touch):
        if self.unloading and self.retreat_to is not None and self.tick-self.retreat_tick>=12:
            # A jammed finger link can have no reliable pad normal. Retreat to
            # a measured low-load wrist pose instead of pushing along that pad.
            p,r=self.retreat_to
            self.position=self.workspace_center+limit(self.position+limit(p-self.position,.0011)-self.workspace_center,.45)
            self.attitude=Rotation.from_rotvec(limit((r*self.attitude.inv()).as_rotvec(),.009))*self.attitude
            cmd=np.r_[self.position,quaternion(self.attitude)]
            return cmd,cmd.copy()
        return super().command(touch)


def run(folder,obj,seconds=240,device='cuda',prior=None):
    prev=first.previous
    return field.run(folder,obj,seconds,device,prior,wrist_cls=LandingWrist,worker_cls=prev.ReturnWorker,
        source_files=(__file__,first.__file__,first.palm_acquisition.__file__,first.palm_surface.__file__,
                      prev.__file__,prev.ct.__file__,prev.ct.entry_frontier.__file__,prev.ct.transition_compute.__file__,
                      prev.ct.delivery.__file__,prev.ct.tactile_reentry.__file__,prev.ct.tactile_stall.__file__),
        release='mesh-palm-finger-pair')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--object',choices=OBJECTS,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=240)
    p.add_argument('--prior',type=Path)
    a=p.parse_args()
    print(run(a.out,a.object,a.seconds,prior=a.prior),flush=True)
