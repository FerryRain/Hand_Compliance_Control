"""Cooperative fitting against finite mesh patches, including narrow bottle ends."""
from time import perf_counter
import numpy as np
import trimesh
from .whole_hand_surface_planner import WholeHandSurfacePlanner
from .whole_hand_belly import cooperative_hand_reference
from .natural_belly_traversal import set_kinematics
from .belly_contact import belly_ik_terms
from .oracle import align_vector


class FiniteSurfacePlanner(WholeHandSurfacePlanner):
    def support_projection(self,q,normals):
        set_kinematics(self.h,self.data,q)
        support=[]
        for finger,geom in enumerate(self.h.tip_geom_ids):
            r=self.data.geom_xmat[geom].reshape(3,3);a=self.h.model.geom_size[geom]
            local=r.T@normals[finger]
            support.append(self.data.geom_xpos[geom]-r@(a*a*local/np.linalg.norm(a*local)))
        points,distance,_=trimesh.proximity.closest_point(self.graph.mesh,np.asarray(support)-self.h.object_position_m)
        return points+self.h.object_position_m,np.array([self.smooth_normal(p) for p in points]),distance

    def fit_patch(self,root,points,normals,*,anchor=1,clearance=.0002,iterations=55,
                  reproject=0,anchor_weight=1.,flexible=True):
        q=root.copy();points=np.asarray(points).copy();normals=np.asarray(normals).copy()
        # Converge each other finger to an actual finite patch. A low tangent
        # residual against a plane alone can put a finger beyond a narrow cap.
        for outer in range(1+(6 if reproject else 0)):
            patch=super().fit_patch(q,points,normals,anchor=anchor,clearance=clearance,
                iterations=iterations,reproject=0,anchor_weight=anchor_weight,flexible=flexible)
            q=patch['q'];projected,ns,gaps=self.support_projection(q,normals)
            other=np.arange(4)!=anchor
            moved=np.linalg.norm(projected[other]-points[other],axis=1).max()
            turned=np.arccos(np.clip(np.sum(ns[other]*normals[other],axis=1),-1.,1.)).max()
            points[other]=projected[other];normals[other]=ns[other]
            if moved<.0008 and turned<np.radians(3.):break
        set_kinematics(self.h,self.data,q);details=[]
        _,_,gaps=self.support_projection(q,normals)
        for finger,geom in enumerate(self.h.tip_geom_ids):
            e,_,d=belly_ik_terms(self.h.model,self.data,int(geom),points[finger],normals[finger],
                clearance_m=clearance,dof_indices=self.order)
            details.append(dict(normal_error_m=float(e[:3]@normals[finger]),position_error_m=float(np.linalg.norm(e[:3])),
                angle_deg=d['alignment_angle_deg'],qualified_patch=d['geometry_qualified'],finite_surface_gap_m=float(gaps[finger])))
        eligible=np.array([abs(d['normal_error_m'])<.0025 and d['angle_deg']<15. and d['qualified_patch']
            and d['finite_surface_gap_m']<clearance+.0025 for d in details])
        reason=self.safety.collision_reason(q)
        if reason is None and (details[anchor]['position_error_m']>.004 or not eligible[anchor]):reason='ANCHOR_UNREACHED'
        if reason is None and eligible.sum()<2:reason='INSUFFICIENT_FINITE_SURFACE_SUPPORT'
        return dict(q=q,points=points,normals=normals,anchor=anchor,expected_contacts=eligible,
            reason=reason,details=details,clearance=clearance)

    def find_patch(self,root,p,n,*,clearance=.0002,budget_s=8.,preferred=None):
        begin=perf_counter();seeds=[];attempts=[];best=None;best_score=None
        if preferred is not None:seeds.append((np.asarray(root),int(preferred),'continuation'))
        if self.seed_tree is not None:
            _,indices=self.seed_tree.query(p,k=min(64,len(self.seed_q)))
            for k in np.atleast_1d(indices):
                seed=np.clip(self.seed_q[k],self.lower+1e-9,self.upper-1e-9);finger=int(self.seed_finger[k])
                if any(finger==f and np.linalg.norm(seed[:7]-a[:7])<.5 for a,f,_ in seeds):continue
                seeds.append((seed,finger,'previous_geometry_seed'))
                if len(seeds)>=5:break
        for roll in (0.,np.pi/4,-np.pi/4,np.pi/2,-np.pi/2,np.pi):
            for finger in (1,0,2,3):
                target,hand=self.target(p,n,np.array([0.,1.,0.]),finger=finger,clearance=clearance,
                    roll=roll,root_hand=cooperative_hand_reference())
                seed=np.asarray(root).copy();seed[7:]=hand
                q,ep,er,_=self.safety.solve(seed,target,iterations=50)
                if ep<.008 and er<.1:seeds.append((q,finger,'cooperative_template'))
                if len(seeds)>=12:break
            if len(seeds)>=12:break
        for seed,finger,kind in seeds:
            points,normals=self.project_pads(seed);points[finger]=p;normals[finger]=n
            patch=self.fit_patch(seed,points,normals,anchor=finger,clearance=clearance,reproject=2,anchor_weight=1.8)
            count=int(patch['expected_contacts'].sum())
            attempts.append(dict(seed=kind,anchor=finger,reason=patch['reason'],contacts=count))
            if patch['reason'] is None:
                motion=float(np.linalg.norm(patch['q'][:7]-root[:7])+.15*np.linalg.norm(patch['q'][7:]-root[7:]))
                score=(-count,motion)
                if best_score is None or score<best_score:best=patch;best_score=score
                if count==4:break
            if perf_counter()-begin>budget_s and attempts:break
        if best is not None:best['attempts']=attempts
        return best,attempts

    def group_route(self,root,initial_points,initial_normals,anchor,goal,*,maximum_m=.12):
        start=self.graph.nearest_vertex(initial_points[anchor]-self.h.object_position_m)[0]
        route,route_normals=self.surface_route(start,int(goal),spacing=.003)
        if len(route)<2:return None
        route+=self.h.object_position_m;route[0]=initial_points[anchor];route_normals[0]=initial_normals[anchor]
        points=np.asarray(initial_points).copy();normals=np.asarray(initial_normals).copy()
        qs=[root.copy()];pps=[points.copy()];nns=[normals.copy()];distances=[0.];expected=[]
        for k in range(1,len(route)):
            turn=align_vector(route_normals[k-1],route_normals[k])
            guesses=route[k]+turn.apply(points-route[k-1])
            projected,_,_=trimesh.proximity.closest_point(self.graph.mesh,guesses-self.h.object_position_m)
            points=projected+self.h.object_position_m
            normals=np.array([self.smooth_normal(p) for p in projected]);points[anchor]=route[k];normals[anchor]=route_normals[k]
            patch=self.fit_patch(qs[-1],points,normals,anchor=anchor,iterations=35,flexible=False)
            if patch['reason'] is not None or not self.safety.edge_free(qs[-1],patch['q']):break
            points=patch['points'];normals=patch['normals']
            length=distances[-1]+float(np.linalg.norm(route[k]-route[k-1]))
            qs.append(patch['q']);pps.append(points.copy());nns.append(normals.copy());distances.append(length)
            expected.append(patch['expected_contacts'])
            if length>=maximum_m:break
        if len(qs)<3 or distances[-1]<.006:return None
        return dict(q=np.array(qs),points=np.array(pps),normals=np.array(nns),distance=np.array(distances),
            anchor=int(anchor),expected_contacts=np.array(expected),goal=int(goal),complete=len(qs)==len(route))
