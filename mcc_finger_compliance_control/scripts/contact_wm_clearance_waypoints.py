"""Offline contact/clearance planning with explicit separated transit phases.

Uses known collision geometry only for collection. All physical state changes
remain under the unmodified collector and MCC; no reset or contact spoofing.
"""
import argparse,json,hashlib,time
from pathlib import Path
import h5py,mujoco,numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
from contact_wm_initialization import hand_qpos_indices,set_object_static
from split_fingertip_collision import GEOMS

class Geometry:
    def __init__(self,folder,lane,initial_state=None,trace_name='rollout.h5'):
        self.m=m=mujoco.MjModel.from_binary_path(str(folder/'physics_model.mjb'));self.d=mujoco.MjData(m)
        with h5py.File(folder/trace_name) as h:
            g=h['raw_100hz'];self.q0=g['pre_mcc_q'][0,lane].astype(float);self.qa=g['q'][0,lane].astype(float);self.palm=g['palm_pose'][0,lane];initial_object=g['object_command_pose'][0,lane];meta=json.loads(h.attrs['metadata']);self.f=meta['jobs'][lane]['finger']
        if initial_state is not None:
            self.q0=np.asarray(initial_state['reference_q'],float)
            self.qa=np.asarray(initial_state['healthy_q'],float)
            self.palm=np.asarray(initial_state['palm_pose'],float)
            initial_object=np.asarray(initial_state['object_pose'],float)
        self.idx=np.arange(self.f*4,self.f*4+4);self.qids=hand_qpos_indices(m);self.jids=[int(np.flatnonzero(m.jnt_qposadr==a)[0]) for a in self.qids];self.limits=m.jnt_range[self.jids]
        self.names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
        self.pad=next(g for g,n in enumerate(self.names) if n.endswith(GEOMS[self.f]+'_pad_collision'))
        self.targets=[g for g,n in enumerate(self.names) if n.startswith('target/') and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.hand=[g for g,n in enumerate(self.names) if n.startswith('robot/') and (m.geom_contype[g] or m.geom_conaffinity[g])]
        roots=['robot/mcp_joint','robot/mcp_joint_2','robot/mcp_joint_3','robot/pip_4'];root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,roots[self.f])
        self.active=[]
        for g in self.hand:
            b=int(m.geom_bodyid[g])
            while b and b!=root:b=int(m.body_parentid[b])
            if b==root:self.active.append(g)
        self.nonpad=[g for g in self.active if g!=self.pad]
        self.pairs=[]
        for a in self.active:
            for b in self.hand:
                if b==a or (b in self.active and b<a):continue
                x,y=int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
                if x==y or m.body_parentid[x]==y or m.body_parentid[y]==x:continue
                if ((min(x,y)<<16)+max(x,y)) in m.exclude_signature:continue
                self.pairs.append((a,b))
        palm=m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/palm_lower')];self.driver=m.body_mocapid[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'target/object_driver')]
        self.d.mocap_pos[palm]=self.palm[:3];self.d.mocap_quat[palm]=self.palm[3:]
        self.cache=None
        self.world_hulls={}
        self.hulls={}
        for geom in self.hand+self.targets:
            mid=int(m.geom_dataid[geom])
            v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].astype(float)
            hull=ConvexHull(v)
            self.hulls[geom]=(v[hull.vertices],hull.equations[:,:3])
        self.place(initial_object)
        native=self.evaluate(self.qa[self.idx])[2]
        if (native < -5e-5).any():raise ValueError('Initial active self collision')
        self.self_required=np.minimum(.001,np.maximum(0,native*.7))
    def place(self,pose):
        set_object_static(self.m,self.d,self.driver,pose[:3],pose[3:]);self.cache=None
    def distance(self,a,b,cutoff):
        value=mujoco.mj_geomDistance(self.m,self.d,a,b,cutoff,None)
        if value != 0.:return value
        # Native CCD occasionally reports zero with witnesses far outside both
        # meshes in this saved model. A positive separating-plane gap is an
        # independent conservative certificate, never a guessed penetration.
        return min(cutoff,self.separation(a,b))
    def separation(self,a,b):
        for geom in (a,b):
            if geom not in self.world_hulls:
                vertices,normals=self.hulls[geom];r=self.d.geom_xmat[geom].reshape(3,3)
                vertices=vertices@r.T+self.d.geom_xpos[geom]
                self.world_hulls[geom]=(vertices,normals@r.T,vertices.min(0),vertices.max(0))
        va,na,la,ha=self.world_hulls[a];vb,nb,lb,hb=self.world_hulls[b]
        broad=float(np.maximum(la-hb,lb-ha).max())
        if broad>.002:return broad
        axes=np.r_[na,nb]
        pa=va@axes.T;pb=vb@axes.T
        gap=float(np.maximum(pa.min(0)-pb.max(0),pb.min(0)-pa.max(0)).max())
        return gap
    def evaluate(self,x):
        if self.cache is not None and np.array_equal(x,self.cache[0]):return self.cache[1]
        q=self.q0.copy();q[self.idx]=x;self.d.qpos[self.qids]=q;mujoco.mj_kinematics(self.m,self.d);mujoco.mj_comPos(self.m,self.d)
        self.world_hulls={}
        pad=min(self.distance(self.pad,g,.1) for g in self.targets)
        clearance=np.array([min(self.distance(a,g,.1) for g in self.targets) for a in self.nonpad])
        # Use the SAME conservative metric throughout self-clearance planning.
        # Alternating native exact distance and a lower bound introduces fake
        # clearance jumps at native zero-query transitions.
        selfgap=np.array([min(.05,self.separation(a,b)) for a,b in self.pairs])
        mid=int(self.m.geom_dataid[self.pad]);vertices=self.m.mesh_vert[self.m.mesh_vertadr[mid]:self.m.mesh_vertadr[mid]+self.m.mesh_vertnum[mid]]
        center=self.d.geom_xpos[self.pad]+self.d.geom_xmat[self.pad].reshape(3,3)@vertices.mean(0)
        result=(pad,clearance,selfgap,center);self.cache=(x.copy(),result);return result
    def solve(self,pose,seed,xy,free=False):
        self.place(pose);lo=self.limits[self.idx,0]+.30;hi=self.limits[self.idx,1]-.30
        # Keep ab/adduction in a moderate neighborhood; do not use splay as escape.
        lo[1]=max(lo[1],self.qa[self.idx[1]]-.20);hi[1]=min(hi[1],self.qa[self.idx[1]]+.20)
        lo=np.maximum(lo,seed-.15);hi=np.minimum(hi,seed+.15)
        neutral=self.qa[self.idx];scale=np.ones(4)
        def obj(x):
            gap,clear,sg,center=self.evaluate(x)
            contact=(max(0,.015-gap) if free else gap-.0003)/.001
            return contact**2+float(np.sum(((center[:2]-xy)/.025)**2))*.12+float(np.sum(((x-neutral)/scale)**2))*.12+float(np.sum((x-seed)**2))*.08
        def constraints(x):
            pad,c,sg,_=self.evaluate(x)
            return np.r_[(c-.004)/.01,(sg-self.self_required)/.01,
                         ([pad-.012] if free else [(pad+.0001)/.005,(.0006-pad)/.005])]
        best=None
        rng=np.random.default_rng(17+self.f)
        starts=[seed,neutral]+[neutral+rng.uniform(-.28,.28,4) for _ in range(6)]
        for start in starts:
            result=minimize(obj,np.clip(start,lo,hi),method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=90,ftol=1e-7,eps=1e-4))
            quality=float(np.min(constraints(result.x)))
            if best is None or (quality>=-1e-3 and best[1]<-1e-3) or (quality>=-1e-3 and best[1]>=-1e-3 and result.fun<best[0].fun) or (quality<0 and best[1]<0 and quality>best[1]):best=(result,quality)
            if result.success and quality>=-1e-3 and result.fun<.2:break
        result,quality=best;gap,c,sg,center=self.evaluate(result.x)
        return result.x,dict(success=bool(result.success),constraint_min=quality,pad_gap_m=float(gap),nonpad_min_m=float(c.min()),self_min_m=float(sg.min()),joint_margin_rad=float(np.minimum(result.x-self.limits[self.idx,0],self.limits[self.idx,1]-result.x).min()),objective=float(result.fun))

def prepare(folder,output):
    with np.load(folder/'object_path_plan.npz') as a:old={k:a[k] for k in a.files}
    poses=old['commanded'];n,b=poses.shape[:2];knots=np.unique(np.r_[0,np.arange(100,n,100),n-1]);out=[];reports=[]
    for lane in range(b):
        geo=Geometry(folder,lane);geo.place(poses[0,lane]);xy=geo.evaluate(geo.qa[geo.idx])[3][:2];q=geo.qa[geo.idx].copy();qs=[];rows=[]
        for k in knots:
            q,row=geo.solve(poses[k,lane],q,xy)
            full=geo.q0.copy();full[geo.idx]=q;qs.append(full);rows.append(dict(frame=int(k),**row))
            if k%1000==0:print(lane,k,row,flush=True)
        out.append(np.asarray(qs));reports.append(dict(lane=lane,finger=geo.f,rows=rows))
    output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,knots=knots,q_knots=np.stack(out,1),object_command=poses)
    output.with_suffix('.json').write_text(json.dumps(dict(source=str(folder.resolve()),rows=reports,geometry_only=True,not_yet_admitted=True),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();prepare(a.folder,a.output)
