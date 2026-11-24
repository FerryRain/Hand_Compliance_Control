"""Score physical multi-pad strokes on a convex envelope inferred only from the cumulative tactile map."""
import numpy as np

from ..acitive_02.aude import tangent, unit
from .multirate_wrist import GLOBAL_TICKS
from .posterior_route import TactileEnvelope, normalized
from .strip_information import batch_information


class MeasuredSupports:
    def fit(self, surface, center):
        pts = surface.points-center
        _, indices = np.unique(np.round(surface.normals/.16),axis=0,return_index=True)
        self.n = surface.normals[indices]
        self.h = np.maximum(.020,(pts@self.n.T).max(0))
        self.outer = max(.052,1.18*np.max(np.linalg.norm(pts,axis=1)))
        self.center = center.copy()

    def predict(self, rays):
        dot = rays@self.n.T
        front = np.where(dot>.25,self.h/np.maximum(dot,.25),np.inf)
        back = np.where(dot<-.40,(self.h+.012)/np.maximum(-dot,.40),np.inf)
        j = np.argmin(back,axis=-1)
        reverse = np.min(back,axis=-1)
        radius = np.minimum(self.outer,np.minimum(np.min(front,axis=-1),reverse))
        w = np.maximum(0.,1.-(front-radius[...,None])/.018)**2
        n = np.where((w.sum(-1)>.02)[...,None], normalized(w@self.n),
                     np.where((reverse<=self.outer)[...,None],-self.n[j],rays))
        return self.center+radius[...,None]*rays,n


class AtlasRoute:
    def __init__(self, model, points, normal, orient):
        self.model = model
        self.estimate = TactileEnvelope(points,normal,orient)
        self.envelope = MeasuredSupports()
        self.ray = self.path_dir = self.last_good = None
        self.batch,self.last_tick = 0,-GLOBAL_TICKS
        self.failed = []

    def update(self,surface,tick,pads,hits,n,orient,light):
        if light and hits.sum()>=2:
            self.last_good = (pads.copy(),hits.copy(),n.copy(),orient)
        if surface.tree is None or tick-self.last_tick<GLOBAL_TICKS:
            return None
        old = self.model.count
        self.model.update(surface,tick)
        self.estimate.fit(surface)
        center=self.estimate.center
        self.envelope.fit(surface,center)
        self.batch+=1
        self.last_tick=tick
        e=dict(batch=self.batch,observed_tick=tick,old_cells=old,new_cells=self.model.count-old,
               total_cells=self.model.count,mesh_version=self.model.mesh_version,mesh_faces=len(self.model.mesh.faces),
               direction_chosen=False)
        if not light and self.last_good is not None:
            pads,hits,n,orient=self.last_good
        if not hits.any():
            return e
        pivot=pads.mean(0)
        if self.ray is None or self.model.count-old<10:
            start=unit(pivot-center)
        else:
            # Rebase the old physical setpoint in the newly estimated frame.
            start=unit(self.anchor-center)
            actual=unit(pivot-center)
            if actual@start<.64:
                start=unit(.60*actual+.40*start)
        ref=tangent(self.path_dir,start) if self.path_dir is not None else np.zeros(3)
        if np.linalg.norm(ref)<.25:
            ref=max((tangent(x,start) for x in orient.as_matrix().T),key=np.linalg.norm)
        ref=unit(ref)
        angle=np.arange(24)*np.pi/12
        headings=np.cos(angle[:,None])*ref+np.sin(angle[:,None])*np.cross(start,ref)
        stations=np.array([.12,.25,.42,.64,.91,1.24,1.65])
        rays=np.cos(stations[None,:,None])*start+np.sin(stations[None,:,None])*headings[:,None]
        pos,ns=self.envelope.predict(rays)
        cross=np.cross(n,ns)
        def rotate_arm(arm):
            return arm+np.cross(cross,arm)+np.cross(cross,np.cross(cross,arm))/np.maximum(.03,1.+(ns@n)[...,None])
        proposed=np.stack([pos+rotate_arm(x-pivot) for x in pads],axis=2)
        projected,pn=self.envelope.predict(normalized(proposed-center))
        gap=np.linalg.norm(projected-proposed,axis=-1)
        dist,idx=surface.tree.query(projected)
        near_n=surface.normals[idx]
        plane_gap=abs(np.sum((projected-surface.points[idx])*near_n,axis=-1))
        align=(pn*near_n).sum(-1)
        evidence=np.exp(-np.maximum(dist-.014,0.)/.052)*np.maximum(.05,align)**2*np.exp(-plane_gap/.012)
        prob=(.21+.76*evidence)*np.exp(-np.maximum(gap-.005,0.)/.019)
        # Reject an apparent zero crossing that disagrees with a measured tangent.
        xyz=projected.reshape(-1,3)
        near=(dist<.025)&(align>.55)
        probe=xyz[near.ravel()]
        if len(probe):
            normal=near_n[near]
            a,b=np.split(self.model.metric_distance(np.vstack((probe-.004*normal,probe+.004*normal))),2)
            slope=(b-a)/.008
            dz=-.5*(b+a)/np.maximum(slope,.15)
            valid=(slope>.22)&(abs(dz)<.010)
            loc=np.flatnonzero(near.ravel())
            xyz[loc[valid]]+=dz[valid,None]*normal[valid]
            prob.ravel()[loc[~valid]]*=.5
        new=np.clip(surface.tree.query(xyz)[0].reshape(dist.shape)/.014,0.,1.)
        # A useful strip needs several reachable pads along its early stations.
        weights=prob*new**2
        gain=batch_information(self.model,xyz.reshape(len(headings),-1,3),weights.reshape(len(headings),-1))
        quorum=np.sort(prob[:,:4],axis=-1)[:,:,-2].mean(1)
        breadth=np.clip(np.linalg.norm(np.cross(headings,pads[-1]-pads[0]),axis=1)/.085,.40,1.)
        # Penalize a long wrist reorientation relative to its useful footprint advance.
        wrist_travel=np.linalg.norm(np.diff(pos+.115*ns,axis=1),axis=-1).sum(1)
        reach=.65*quorum+.35*prob.mean((1,2))
        score=gain*(.25+.75*reach)*(.70+.30*breadth)/(1.+2.5*wrist_travel)
        if self.path_dir is not None:
            productive=self.model.count-old>=35
            score*=1.+(.33 if productive else -.16)*(headings@ref)
            if self.model.count-old<15:
                self.failed=(self.failed+[(self.anchor.copy(),self.path_dir.copy(),tick)])[-16:]
        for p,d,t in self.failed:
            score*=1.-.8*np.exp(-np.linalg.norm(p-pivot)/.050-(tick-t)*.01/35.)*np.maximum(0.,headings@d)**2
        pick=int(np.argmax(score))
        self.ray,self.path_dir=start.copy(),headings[pick].copy()
        self.anchor=self.envelope.predict(self.ray)[0]
        e.update(direction_chosen=True,source='reachable_contact_strip',direction=self.path_dir.tolist(),
                 pivot_m=pivot.tolist(),normal=n.tolist(),acquisition_nat=float(gain[pick]),quorum=float(quorum[pick]),
                 estimated_center_m=center.tolist(),target_m=pos[pick,-1].tolist())
        return e

    def advance(self,angle):
        r,d=self.ray,self.path_dir
        self.ray=np.cos(angle)*r+np.sin(angle)*d
        self.path_dir=np.cos(angle)*d-np.sin(angle)*r
        self.anchor,n=self.envelope.predict(self.ray)
        return self.anchor,n
