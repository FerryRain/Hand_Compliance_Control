"""Rank tactile frontiers with the wrist attitude and palm placement that would execute them."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.active_gpis import NOISE_VARIANCE
from . import measured_target_yield, tactile_field as field
from .contact_first import palm_clear
from .history import normal_spread
from .informative_actions import FRACTIONS,HORIZON,MASK_THRESHOLDS,joint_gain
from .multirate_wrist import SPEED_M_S
from .palm_surface import OUT, TactilePatches, absolute_attitudes, clamp_length, footprint, normalize, on_plane


@dataclass
class PalmGoal:
    region: object
    position: np.ndarray
    rotation: Rotation
    patch: np.ndarray
    normal: np.ndarray
    footprint: float


def score_poses(state, model, history, owner, ends, poses, positions, duration):
    """Evaluate the same paired hand responses at the actual proposed wrist transforms."""
    wrist, current = state['wrist_position'], state['wrist_rotation']
    pads, surface = state['pads'], state['surface']
    rel = current.inv().apply(pads-wrist)
    turn = (poses*current.inv()).as_rotvec()
    last = np.minimum(1., HORIZON/np.maximum(duration,.001))
    fractions = 1.-last[:,None]*(1.-FRACTIONS)
    def location(fraction):
        rot = Rotation.from_rotvec((fraction[:,:,None]*turn[:,None]).reshape(-1,3))*current
        mat = rot.as_matrix().reshape(len(owner),-1,3,3)
        anchor = pads[owner,None]+fraction[:,:,None]*(ends-pads[owner])[:,None]
        correction = positions-ends+poses.apply(rel[owner])
        origin = anchor-np.einsum('atij,aj->ati',mat,rel[owner])+fraction[:,:,None]*correction[:,None]
        return mat,origin
    mats,wrist_knots = location(fractions)
    begin, start_wrist = location(1.-last[:,None])
    start_rot=Rotation.from_matrix(begin[:,0])
    actions=np.c_[start_rot.inv().apply(positions-start_wrist[:,0]),start_rot.inv().apply(last[:,None]*turn)]
    delta,low,weights=history.predict(state['q'],state['hits'],normal_spread(state['normals'],state['hits']),actions)
    predicted=wrist_knots[:,None,:,None]+np.einsum('atij,ahtfj->ahtfi',mats,rel+delta)
    d,idx=surface.tree.query(predicted)
    n=surface.normals[idx]
    gap=np.sum((predicted-surface.points[idx])*n,axis=-1)
    prob=low*np.exp(-.5*((gap+.0007)/.007)**2)*(d<.038)
    gain,_=joint_gain(predicted-gap[...,None]*n,prob,weights,surface,model)
    mask=prob[...,None,:]>MASK_THRESHOLDS[:,:4]
    pair=(mask.sum(axis=-1)>=2).mean(axis=(2,3))
    three=(mask.sum(axis=-1)>=3).mean(axis=(2,3))
    owner_p=(weights*prob[np.arange(len(owner)),:,-1,owner]).sum(1)
    # Check the sweep as well as the end pose around the real support pivot.
    geometry = current.inv().apply(state['palm_vertices']-wrist)
    sweep_mats,sweep_origins=location(np.broadcast_to([.25,.5,.75,1.],(len(owner),4)))
    palm=sweep_origins[:,:,None]+np.einsum('atij,vj->atvi',sweep_mats,geometry)
    legal=palm_clear(surface,state['palm_vertices'],palm)
    legal &= np.max(np.linalg.norm(sweep_origins-state['workspace_center'],axis=2),axis=1)<=.45
    return gain,(weights*pair).sum(1),(weights*three).sum(1),owner_p,legal


class PalmGuide(measured_target_yield.DeliveryGuide):
    def __init__(self,model,history,seed,sites):
        super().__init__(model,history,seed)
        self.sites=sites
        self.patches=TactilePatches()
        self.palm_goal=None

    def choose(self,surface,pads,ids,normal,event):
        self.palm_goal=None
        state=self.state
        model=self.model
        regions=field.mesh_regions(model,pads,ids,state['normals'])
        regions+=field.gradient_regions(model,surface,pads,ids,state['normals'],self.rng)
        if not regions:
            return
        wrist,current=state['wrist_position'],state['wrist_rotation']
        owner=np.array([r.finger for r in regions])
        ends=np.array([r.point for r in regions])
        hints=np.array([r.normal for r in regions])
        anchor=self.patches.query(model,surface,ends,hints,.036)
        base_n=normalize(.55*hints+.45*anchor.normals)
        base=absolute_attitudes(current,base_n)
        hand=current.inv().apply(pads-wrist)
        arm=self.sites[0]-hand[owner]
        centers=ends+base.apply(arm)
        projected=ends+on_plane(centers-ends,base_n)
        under=self.patches.query(model,surface,projected,base_n,.11,anchor.components)
        patch=np.where(under.valid[:,None],under.points,ends)
        broad_n=normalize(.56*base_n+.44*np.where(under.valid[:,None],under.normals,base_n))
        view=normalize(centers-patch)
        blend=.32*np.clip(np.sum(view*broad_n,axis=1),0.,1.)
        desired=normalize((1-blend[:,None])*broad_n+blend[:,None]*view)
        minimal=absolute_attitudes(current,desired)
        lateral=on_plane(minimal.apply(arm),desired)
        supported=on_plane(patch-ends,desired)
        yaw=np.arctan2(np.einsum('ai,ai->a',desired,np.cross(lateral,supported)),
                       np.einsum('ai,ai->a',lateral,supported))
        yaw=np.where(under.valid & (np.linalg.norm(supported,axis=1)>.014),np.clip(yaw,-1.0,1.0),0.)
        twisted=Rotation.from_rotvec(yaw[:,None]*desired)*minimal
        indices=np.r_[np.arange(len(owner)),np.flatnonzero(abs(yaw)>.09)]
        poses=Rotation.from_quat(np.vstack((minimal.as_quat(),twisted.as_quat()[abs(yaw)>.09])))
        anchor_points=ends[indices]
        digit=owner[indices]
        goal=anchor_points-poses.apply(hand[digit])
        center=goal+poses.apply(np.repeat(self.sites[:1],len(goal),axis=0))
        side_error=on_plane(patch[indices]-center,desired[indices])
        shift=clamp_length(.18*side_error,.007)*under.valid[indices,None]
        goal+=shift
        sites=goal[:,None]+np.einsum('aij,vj->avi',poses.as_matrix(),self.sites)
        broad,ray,matched,matched_n,clearance,mesh=footprint(self.patches,model,surface,sites,
                                 anchor_points,base_n[indices],desired[indices],anchor.components[indices])
        path=np.array([regions[k].distance for k in indices])
        angles=(poses*current.inv()).magnitude()
        duration=np.maximum(path/SPEED_M_S,angles/.95)
        gain,pair,three,receiver,legal=score_poses(state,model,self.history,digit,anchor_points,poses,goal,duration)
        heads=np.array([field.tangent(next((pt for pt in r.path if np.linalg.norm(pt-pads[r.finger])>=.012),
                                 r.point)-pads[r.finger],normal) for r in regions])
        novelty=field.unobserved_side(surface,pads,ids,state['normals'],heads)[indices]
        legal&=np.linalg.norm(heads[indices],axis=1)>.002
        # When an ordinary, skin-supported region cannot supply a predicted new
        # observation, its single-contact hypothetical gain remains admissible.
        if not np.any(legal & (gain*receiver>1.e-9)):
            prior,condition=model.observation_covariances(ends)
            gain=(.5*np.log((NOISE_VARIANCE+prior.diagonal())/(NOISE_VARIANCE+condition.diagonal())))[indices]
            gain*=np.clip(surface.tree.query(anchor_points)[0]/.024,0.,1.)
            receiver=np.maximum(receiver,.26)
        score=gain*novelty*(.15+.85*receiver)*(.42+.58*pair)*(.21+.79*broad)/(HORIZON+duration)
        if getattr(self,'recovery_observation',None) is not None:
            score*=self.after_loss_weights(regions)[indices]
        eligible=np.flatnonzero(legal & (score>0))
        if not len(eligible):
            return
        k=eligible[np.argmax(score[eligible])]
        self.region=r=regions[indices[k]]
        chosen_patch=matched[k] if ray[k,0] else patch[indices[k]]
        chosen_n=matched_n[k] if ray[k,0] else broad_n[indices[k]]
        self.palm_goal=PalmGoal(r,goal[k],poses[k],chosen_patch,chosen_n,float(broad[k]))
        event.update(source=r.source,target_m=r.point.tolist(),target_normal=r.normal.tolist(),target_finger=r.finger,
            pivot_m=pads[r.finger].tolist(),mesh_path_m=r.distance,candidates=len(goal),acquisition_nat=float(gain[k]),
            selected_rate_nat_s=float(score[k]),predicted_light3=float(three[k]),predicted_owner_support=float(receiver[k]),
            predicted_light2=float(pair[k]),palm_goal_position_m=goal[k].tolist(),palm_goal_xyzw=poses[k].as_quat().tolist(),
            palm_endpoint_patch_m=chosen_patch.tolist(),palm_endpoint_outward=chosen_n.tolist(),
            palm_endpoint_footprint=float(broad[k]),palm_endpoint_rays=int(ray[k].sum()),palm_endpoint_height_m=float(clearance[k]))
