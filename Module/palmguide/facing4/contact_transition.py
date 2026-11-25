"""Hand off a measured supporting pad or select a tactile re-entry with a continuation."""

import argparse
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial.transform import Rotation

from ..acitive_02.aude import tangent, unit
from ..acitive_02.surface_scan import limit
from ..i04_geometric_traversal.physics import quaternion,rotation
from . import measured_target_yield as delivery, tactile_field as field, tactile_reentry, tactile_stall
from .contact_first import palm_clear, signed_gap
from .history import normal_spread
from .informative_wrist import InformativeWrist
from .multirate_wrist import LOCAL_S, transport
from .sweep import DT, LOAD_LIMIT, PLAN_TICKS, PREVIEW_S
from . import entry_frontier,transition_compute
from .link_clearance import RobotLinkHull
from .ycb_fixtures import OBJECTS


MODES = ('r10', 'handoff', 'reacquire', 'both')
LOOKAHEAD = .24
KNOTS = np.array([.33,.67,1.])


@dataclass
class ContactPose:
    tick: int
    position: np.ndarray
    attitude: Rotation
    fingers: np.ndarray
    points: np.ndarray
    normals: np.ndarray
    material: np.ndarray
    dwell: int


@dataclass
class Entry:
    station: ContactPose
    finger: int
    landing: np.ndarray
    normal: np.ndarray
    next_point: np.ndarray
    next_normal: np.ndarray
    path: np.ndarray
    length: float
    wrist: np.ndarray
    attitude: Rotation
    information: float
    support: float
    rate: float
    issued: int
    deadline: int


def handoff_action(state, initial, history, observations):
    """Compare a continued step with rotations fitted to the other measured patches."""
    hits, pads = state['hits'],state['pads']
    anchor = int(np.flatnonzero(hits)[0])
    surface = state['surface']
    n = state['normals'][anchor]
    pivot,wrist,orient = pads[anchor],state['wrist_position'],state['wrist_rotation']
    info = initial['diagnostics']
    direction = np.asarray(initial['direction'])
    speed = info.get('tangent_speed_m_s',0.)
    shift = LOOKAHEAD*speed*direction
    dwells,forces,slope,pair_age,onset_age = observations
    fresh = surface.last_observation_tick is not None and state['tick']-surface.last_observation_tick<=24
    secure = dwells[anchor]>=12 and forces[anchor]>=.55 and slope[anchor]>-.10 and fresh
    # A pad that just returned has not yet taken the load. A rising load, or
    # fresh productive single support, also argues against a corrective twist.
    risk = ((slope[anchor]<-.042 and forces[anchor]<1.45) or
            (not fresh and pair_age>=35 and forces[anchor]<1.6))
    if (onset_age<24 or dwells[anchor]<7 or not risk or secure or
            forces[anchor]>2.2 or state['loads'].max()>3.0 or info['mode']!='explore'):
        return initial
    info['nominal_flow']=dict(tangent_speed_m_s=speed,height_speed_m_s=info['height_speed_m_s'],
                               angular_velocity=info['angular_velocity'],selected_direction=direction.tolist())
    def guard():
        # Conserve the unloading anchor, even if a second pad lacks a reliable patch.
        if slope[anchor]<-.042:
            info.update(handoff_choice='guard',handoff_anchor=anchor,tangent_speed_m_s=.58*speed)
        return initial
    patches = [surface.patch(pad,preferred_normal=n if j==anchor else None) for j,pad in enumerate(pads)]
    if patches[anchor] is None:
        return guard()
    # Each trial twist is derived from a particular recipient's observed surface
    # error. The anchor is the pivot and the tangent comes from the global GPIS.
    steps = [shift]
    turns = [limit(LOOKAHEAD*np.asarray(info['angular_velocity']),.14)]
    owners = [-1]
    initial_gap = np.full(4,np.inf)
    for j,patch in enumerate(patches):
        if j==anchor or patch is None or surface.tree.query(pads[j])[0]>.044:
            continue
        gap,nn = signed_gap(patch,pads[j])
        lever = pads[j]-pivot
        jac = np.cross(lever,nn)
        if not -.003<=gap<=.014 or nn@n<.48 or jac@jac<.008**2:
            continue
        initial_gap[j] = float(gap)
        for fraction in ((.85,.4) if secure else (.42,.12)):
            d = fraction*shift
            target = -float(np.clip(gap+d@nn+.0008,-.004,.014))
            turn = limit(target*jac/(jac@jac+.013**2),.080)
            # Fit the normal rise of the supporting point on its own patch.
            d = d-float(signed_gap(patches[anchor],pivot+d,patches[anchor].predict(pivot)[0][0])[0])*n
            steps.append(d)
            turns.append(turn)
            owners.append(j)
    if len(steps)==1:
        if secure:
            return initial
        # No missing pad has a credible surface. Unload a falling anchor more
        # slowly and keep sensing, without claiming a successful handoff.
        info.update(single_support_uncertain=True)
        return guard()
    steps,turns=np.asarray(steps),np.asarray(turns)
    mats=Rotation.from_rotvec((turns[:,None]*KNOTS[None,:,None]).reshape(-1,3)).as_matrix().reshape(-1,3,3,3)
    centers=pivot+steps[:,None]*KNOTS[None,:,None]
    predicted=centers[:,:,None]+np.einsum('atij,fj->atfi',mats,pads-pivot)
    hull=centers[:,:,None]+np.einsum('atij,vj->atvi',mats,state['palm_vertices']-pivot)
    future_w=centers+np.einsum('atij,j->ati',mats,wrist-pivot)
    allowed=palm_clear(surface,state['palm_vertices'],hull)
    allowed &= np.linalg.norm(future_w-state['workspace_center'],axis=2).max(1)<=.45
    # Query the empirical joint response at the speed of each proposed wrist.
    actions=np.c_[orient.inv().apply(future_w[:,-1]-wrist),orient.inv().apply(turns)]*(.5/LOOKAHEAD)
    delta,low,weights=history.predict(state['q'],hits,normal_spread(state['normals'],hits),actions)
    response=orient.apply(delta[:,:,0].reshape(-1,3)).reshape(len(steps),-1,4,3)
    gap=np.full(predicted.shape[:-1],np.inf)
    normals=np.zeros_like(predicted)
    for j,patch in enumerate(patches):
        if patch is None:
            continue
        base=patch.predict(pivot)[0][0] if j==anchor else 0.
        gap[:,:,j],normals[:,:,j]=signed_gap(patch,predicted[:,:,j],base)
    drift=np.einsum('atfi,atij,ahfj->ahtf',normals,mats,response)
    alpha=KNOTS*LOOKAHEAD/.25
    actual=gap[:,None]+alpha[None,None,:,None]*drift
    observed=(surface.tree.query(predicted)[0]<.043)
    prior=hits[None,None,None,:]+alpha[None,None,:,None]*(low[:,:,0,None,:]-hits.astype(float))
    support=np.exp(-.5*((actual+.0006)/.0048)**2)*(.32+.68*prior)*observed[:,None]
    keeps=np.minimum.accumulate(support[:,:,:,anchor],axis=2)
    overlap=(keeps[:,:,:,None]*support).max(axis=2)
    joint=np.sum(weights[:,:,None]*overlap,axis=1)
    joint[:,anchor]=0.
    joint[:,~np.isfinite(initial_gap)]=0.
    retention=(weights*keeps[:,:,-1]).sum(1)
    recipient=np.argmax(joint,axis=1)
    for i,j in enumerate(owners[1:],1):
        recipient[i]=j
    overlap=joint[np.arange(len(steps)),recipient]
    progress=steps@direction/max(np.linalg.norm(shift),.001)
    score=overlap+(.36 if secure else .20)*retention+(.30 if secure else .07)*progress
    score-=.07*np.linalg.norm(turns,axis=1)/.19
    valid=np.flatnonzero(allowed & np.isfinite(score))
    if not len(valid):
        return guard()
    k=int(valid[np.argmax(score[valid])])
    j=int(recipient[k])
    info.update(handoff_anchor=anchor,handoff_recipient=j,predicted_overlap=float(overlap[k]),
                nominal_overlap=float(overlap[0]),anchor_retention=float(retention[k]),single_productive=bool(secure))
    if k==0 or overlap[k]<.14 or overlap[k]<overlap[0]+.055 or retention[k]<.55*retention[0]:
        info['handoff_choice']='continue'
        return initial
    turn=Rotation.from_rotvec(turns[k]*LOCAL_S/LOOKAHEAD)
    small=steps[k]*LOCAL_S/LOOKAHEAD
    tang=tangent(steps[k],n)
    moving=bool(speed and np.linalg.norm(tang)>.001)
    info.update(handoff_choice='recruit',mode='explore' if moving else 'recontact',
                tangent_speed_m_s=float(np.linalg.norm(tang)/LOOKAHEAD),height_speed_m_s=float(steps[k]@n/LOOKAHEAD),
                angular_velocity=(turns[k]/LOOKAHEAD).tolist())
    initial.update(position=pivot+small+turn.apply(wrist-pivot),rotation=turn*orient,
                   normal=n,direction=unit(tang) if moving else np.zeros(3))
    return initial


def select_entry(state, model, stations, failed, origins, matrices, empty):
    """Rank a measured landing pose together with a reachable posterior frontier."""
    mesh=model.mesh
    if mesh is None or not model.count or not stations:
        return None
    wrist,orient=state['wrist_position'],state['wrist_rotation']
    valid=np.flatnonzero(mesh.value>0) if len(mesh.nominees) else []
    if len(valid):
        ns=mesh.normals[mesh.nominees[valid]]
        points=mesh.goal_points[valid]-model.band.half_width*ns
    local=entry_frontier.chart_options(state['surface'],model,stations,wrist,orient,empty)
    for station in stations if len(valid) else ():
        d=np.linalg.norm(station.position-wrist)
        angle=(station.attitude*orient.inv()).magnitude()
        # A brief loss still has a nearby, measured support neighbourhood.
        # Expand the reachable set as that evidence grows stale.
        if d>min(.18,.042+.0011*empty) or angle>min(1.1,.40+.006*empty):
            continue
        diff=points[None]-station.points[:,None]
        dd=np.linalg.norm(diff,axis=-1)
        use=(dd>.008)&(dd<.085)&(station.normals@ns.T>.40)
        if not use.any():
            continue
        merits=np.where(use,mesh.value[valid][None,:]/(.32+dd/.045),0.)
        f,k=np.unravel_index(merits.argmax(),merits.shape)
        # Distant successes must repay their transfer time as well as the scan.
        quick=merits[f,k]*min(1.,.60+.14*len(station.fingers))/(.4+d/.12+angle/1.2)
        for at,normal,stamp in failed:
            if state['tick']-stamp<2000 and station.normals[f]@normal>.65:
                quick*=1.-.93*np.exp(-.5*(np.linalg.norm(station.points[f]-at)/.020)**2)
        local.append((float(quick),station,int(f),None))
    best=None
    considered=[]
    for coarse,station,f,chart in sorted(local,key=lambda x:-x[0]):
        at=station.points[f]
        if any(np.linalg.norm(at-pt)<.010 for pt in considered):
            continue
        considered.append(at)
        ids=station.fingers
        current=origins[ids]+np.einsum('fij,fj->fi',matrices[ids],station.material)
        relative=station.attitude.apply(orient.inv().apply(current-wrist))
        estimated=station.points-relative
        goal=(estimated.mean(0)+estimated[f])/2
        mismatch=np.linalg.norm(goal+relative-station.points,axis=1)
        n=station.normals[f]
        if (np.linalg.norm(goal-station.position)>.055 or mismatch[f]>.020 or
                np.linalg.norm(goal+.012*n-state['workspace_center'])>.445):
            if len(considered)>=6:
                break
            continue
        rr=station.attitude*orient.inv()
        reach=goal+.010*n
        palm=state['palm_vertices']
        hull=np.stack((.5*(wrist+reach)+Rotation.from_rotvec(.5*rr.as_rotvec()).apply(palm-wrist),
                       reach+rr.apply(palm-wrist),goal+rr.apply(palm-wrist)))
        if palm_clear(state['surface'],palm,hull[None])[0]:
            probability=np.mean(np.exp(-.5*(mismatch/.011)**2))*min(1.,.60+.13*len(ids)+.002*min(station.dwell,32))
            transfer=np.linalg.norm(reach-wrist)/.11+rr.magnitude()/1.1
            options=[]
            if chart is not None:
                result=entry_frontier.verify_chart(state['surface'],at,n,chart)
                if result is not None:
                    options.append(result)
            if len(valid):
                for route in mesh.routes(at,n):
                    target=route.center-model.band.half_width*route.normal
                    way=route.waypoint-model.band.half_width*route.waypoint_normal
                    face_row=np.flatnonzero(mesh.nominees==route.face)
                    if len(face_row):
                        options.append((target,route.normal,np.stack((way,target)),route.distance,float(mesh.value[face_row[0]])))
            for target,next_normal,path,length,info in options:
                repeats=sum(np.linalg.norm(at-p)<.026 and n@nn>.65 and state['tick']-stamp<1600
                            for p,nn,stamp in failed)
                rate=probability*info/((.34+transfer+length/.045)*(1.+3.*repeats))
                if best is None or rate>best.rate:
                    best=Entry(station,int(ids[f]),at,n,target,next_normal,path,length,goal,
                               station.attitude,info,float(probability),float(rate),state['tick'],
                               state['tick']+int(100*min(3.8,1.+1.4*transfer)))
        if len(considered)>=6:
            break
    return best


class TransitionFollower:
    def __init__(self, base):
        self.base=base
        self.guide,self.global_model=base.guide,base.global_model
        self.global_model=entry_frontier.BatchPosterior(self.global_model)
        self.base.global_model=self.guide.model=self.global_model
        self.tip_forces=np.zeros(4)
        self.entry=None
        self.failed=deque(maxlen=12)
        self.retry_tick=0
        self.visit=None

    def select(self,**state):
        self.base.tip_forces=self.tip_forces
        tick=state['tick']
        mode,ready=self.settings
        self.global_model.enabled=ready and mode=='both' and self.recovery_needed
        if not ready or mode=='r10':
            return self.base.select(**state)
        single=mode in ('handoff','both')
        zero=mode in ('reacquire','both')
        ids=np.flatnonzero(state['hits'])
        low=state['loads'].max()<LOAD_LIMIT and state['loads'][-1]<.48
        began=perf_counter()
        continued=None
        if self.visit is not None:
            start,cells,prev=self.visit
            if tick-start>=100 or (not len(ids) and self.empty>=14):
                if len(state['surface'].samples)-cells<4:
                    self.failed.append((prev.landing,prev.normal,tick))
                self.visit=None
        if zero and self.entry is not None:
            entry=self.entry
            if len(ids) and low:
                f=int(ids[np.argmin(np.linalg.norm(state['pads'][ids]-entry.landing,axis=1))])
                if np.linalg.norm(state['pads'][f]-entry.landing)<.030 and state['normals'][f]@entry.normal>.45:
                    continued=field.Region(f,entry.next_point,entry.next_normal,entry.path,entry.length,'reentry_frontier')
                    self.guide.region=continued
                    self.guide.cursor=0
                    self.guide.refresh=False
                    self.guide.last_tick=tick
                    self.guide.owner_lost_since=None
                    self.visit=tick,len(state['surface'].samples),entry
                else:
                    self.failed.append((entry.landing,entry.normal,tick))
                self.entry=None
                self.retry_tick=tick+45
            elif tick>=entry.deadline or not low or state['observed_phase']=='withdraw':
                self.failed.append((entry.landing,entry.normal,tick))
                self.entry=None
                self.retry_tick=tick+15
        continuation_ms=1000*(perf_counter()-began) if continued is not None else 0.
        result=self.base.select(**state)
        row=result['diagnostics']
        if continued is not None and row['mode']=='explore' and self.guide.direction is not None:
            row['global_ms']+=continuation_ms
            row['global_event']=dict(batch=self.guide.batch,observed_tick=tick,direction_chosen=True,
                    source='reentry_frontier',target_m=continued.point.tolist(),target_normal=continued.normal.tolist(),
                    target_finger=continued.finger,pivot_m=state['pads'][continued.finger].tolist(),mesh_path_m=continued.distance,
                    direction=result['direction'].tolist(),normal=result['normal'].tolist(),
                    acquisition_nat=entry.information,selected_rate_nat_s=entry.rate,predicted_light3=0.,
                    predicted_owner_support=entry.support)
        if (zero and not len(ids) and low and self.empty>=17 and state['observed_phase']!='withdraw' and
                (self.entry is not None or self.recovery_needed)):
            selected=False
            if self.entry is None and tick>=self.retry_tick:
                began=perf_counter()
                self.entry=select_entry(state,self.global_model,self.stations,self.failed,self.origins,self.matrices,self.empty)
                selected=self.entry is not None
                self.retry_tick=tick+(30 if selected else 55)
                elapsed=1000*(perf_counter()-began)
                row['zero_selection_ms']=elapsed
                row['global_ms']+=elapsed
                if row['global_event'] is None:
                    row['global_event']=dict(batch=self.guide.batch,observed_tick=tick,direction_chosen=False,
                                             source='recontact_and_frontier',entry_available=selected)
            if self.entry is not None:
                e=self.entry
                material=e.station.material[np.flatnonzero(e.station.fingers==e.finger)[0]]
                row.update(mode='recontact',entry_tick=e.issued,entry_finger=e.finger,entry_point=e.landing.tolist(),
                           entry_normal=e.normal.tolist(),entry_wrist=e.wrist.tolist(),entry_xyzw=e.attitude.as_quat().tolist(),
                           entry_material=material.tolist(),entry_prior_fingers=len(e.station.fingers),
                           entry_continuation=e.next_point.tolist(),entry_rate=e.rate,entry_probability=e.support,
                           entry_new=selected)
                result.update(position=e.wrist,rotation=e.attitude,normal=e.normal)
        elif single and self.recovery_needed and len(ids)==1 and low and state['observed_phase']=='scan':
            began=perf_counter()
            result=handoff_action(state,result,self.guide.history,(self.dwell,self.tip_forces,self.slope,
                                  self.pair_age,self.onset_age))
            if row['mode']=='explore' and self.side_load>1.55 and self.onset_age>=8 and self.dwell[ids[0]]>=4:
                state.update(bearing=result['direction'],side_load=self.side_load)
                entry_frontier.unload_link(state['surface'],self.link_vertices,row,state,int(ids[0]))
            row['handoff_ms']=1000*(perf_counter()-began)
        row['measured_stall']=self.recovery_needed
        return result


class TransitionWrist(delivery.DeliveryWrist):
    def __init__(self,hand,data,touch,*,device,seed,mode='both',enable_from=0,**options):
        super().__init__(hand,data,touch,device=device,seed=seed,**options)
        self.mode,self.enable_from=mode,enable_from
        self.selector=TransitionFollower(self.selector)
        self.stations=deque(maxlen=100)
        self.single_stations=deque(maxlen=60)
        self.link_hull=RobotLinkHull(hand)
        self.slope=np.zeros(4)
        self.previous_force=touch.forces.copy()
        self.transition=None
        self.last_empty=-1000
        self.safe_wrist=deque(maxlen=80)
        self.high_since=None
        self.escape_wrist=None
        self.activity=deque(maxlen=101)
        self.recovery_needed=False
        self.recovery_cells=0
        self.recovery_start=0

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        self.slope=.75*self.slope+.25*(touch.forces-self.previous_force)
        self.previous_force=touch.forces.copy()
        if tick%5==0:
            cells=len(self.surface.samples)
            hits=int(np.count_nonzero(touch.measured_heads)) if touch.loads.max()<LOAD_LIMIT else 0
            self.activity.append((cells,hits))
            if len(self.activity)==self.activity.maxlen:
                growth=cells-self.activity[0][0]
                if self.recovery_needed:
                    paired=np.count_nonzero(self.dwell>=12)>=2
                    self.recovery_needed=not ((cells-self.recovery_cells>=8 and paired) or
                                              (growth>=14 and tick-self.recovery_start>=70))
                elif growth<=4 and self.last_pair<tick-55:
                    contacts=np.array([count for _,count in self.activity])
                    if np.mean(contacts<=1)>.65:
                        self.recovery_needed=True
                        self.recovery_cells,self.recovery_start=cells,tick
        if not touch.measured_heads.any():
            self.last_empty=tick
        if tick%5==0 and touch.loads.max()<3. and touch.loads[-1]<.45:
            self.safe_wrist.append((tick,self.measured_position.copy(),rotation(data.qpos[3:7])))
        if touch.loads.max()>=LOAD_LIMIT and self.high_since is None:
            self.high_since=tick
            self.escape_wrist=next(((xyz,att) for when,xyz,att in reversed(self.safe_wrist)
                    if when<=tick-15 and np.linalg.norm(xyz-self.measured_position)>.005),None)
        if touch.loads.max()<4.:
            self.high_since=self.escape_wrist=None
        self.origins=data.geom_xpos[self.hand.pads]
        self.matrices=data.geom_xmat[self.hand.pads].reshape(-1,3,3)
        self.live_attitude=rotation(data.qpos[3:7])
        ids=np.flatnonzero(touch.measured_heads & ((self.dwell>=4)|(tick==0)))
        if len(ids)>=2 and max(touch.loads)<4.5 and touch.loads[-1]<.45:
            pos=self.measured_position
            att=rotation(data.qpos[3:7])
            prev=self.stations[-1] if self.stations else None
            if (prev is None or np.linalg.norm(pos-prev.position)>.012 or (prev.attitude*att.inv()).magnitude()>.15 or
                    len(ids)>len(prev.fingers)):
                self.stations.append(ContactPose(tick,pos.copy(),att,ids,touch.points[ids].copy(),touch.normals[ids].copy(),
                                            self.pad_offsets(data,touch)[ids],int(self.dwell[ids].min())))
        elif len(ids)==1 and self.dwell[ids[0]]>=10 and max(touch.loads)<4. and touch.loads[-1]<.30:
            previous=self.single_stations[-1] if self.single_stations else None
            pos=self.measured_position
            if (previous is None or ids[0]!=previous.fingers[0] or np.linalg.norm(pos-previous.position)>.016):
                self.single_stations.append(ContactPose(tick,pos.copy(),rotation(data.qpos[3:7]),ids,
                    touch.points[ids].copy(),touch.normals[ids].copy(),self.pad_offsets(data,touch)[ids],
                    int(self.dwell[ids[0]])))
        if self.transition and 'entry_tick' in self.transition and touch.measured_heads.any() and not self.unloading:
            self.rescue=None
            if self.anchor is not None:
                self.leave_return(tick)

    def command(self,touch):
        r=self.transition
        # If withdrawal has failed to unload a region, reverse toward an
        # actually observed low-load wrist configuration, including its attitude.
        if (self.mode=='both' and self.tick>=self.enable_from and self.recovery_needed and self.unloading and self.high_since is not None
                and self.tick-self.high_since>=24 and self.escape_wrist is not None):
            xyz,att=self.escape_wrist
            self.position+=limit(xyz-self.position,.0013)
            self.attitude=Rotation.from_rotvec(limit((att*self.attitude.inv()).as_rotvec(),.012))*self.attitude
            cmd=np.r_[self.position,quaternion(self.attitude)]
            return cmd,cmd.copy()
        if r is None or self.unloading or self.tick<self.enable_from:
            return super().command(touch)
        ids=np.flatnonzero(touch.measured_heads)
        if 'entry_tick' in r and not len(ids) and max(touch.loads)<4. and touch.loads[-1]<.48:
            n=np.asarray(r['entry_normal'])
            end=np.asarray(r['entry_wrist'])
            desired=Rotation.from_quat(r['entry_xyzw'])
            finger=r['entry_finger']
            material=self.origins[finger]+self.matrices[finger]@r['entry_material']
            fit=np.asarray(r['entry_point'])-desired.apply(self.live_attitude.inv().apply(material-self.measured_position))
            end=end+limit(fit-end,.014)
            error=self.measured_position-end
            lateral=np.linalg.norm(tangent(error,n))
            clearance=.009 if lateral>.012 or (desired*self.attitude.inv()).magnitude()>.14 else -min(.006,.001+.00007*self.empty)
            # Reuse the measured-plane collision and empty-hand wrist speed limits.
            saved=self.rescue,self.rescue_stage,self.rescue_tick
            self.rescue,self.rescue_stage,self.rescue_tick=(end+clearance*n,desired,n),'near',self.tick
            start,att=self.position,self.attitude
            try:
                command,_=InformativeWrist.command(self,touch)
                ratio=min(PREVIEW_S/DT,np.linalg.norm(end+clearance*n-self.position)/max(1e-8,np.linalg.norm(self.position-start)))
                ahead=self.position+ratio*(self.position-start)
                future=Rotation.from_rotvec(ratio*(self.attitude*att.inv()).as_rotvec())*self.attitude
                return command,np.r_[ahead,quaternion(future)]
            finally:
                self.rescue,self.rescue_stage,self.rescue_tick=saved
        if r.get('handoff_choice') in ('recruit','guard','support_roll'):
            roll=r['handoff_choice']=='support_roll'
            expected=(len(ids)==1 and ids[0]==r['handoff_anchor'] and self.anchor is None and
                      touch.loads.max()<(6.5 if roll else 3.) and touch.loads[-1]<.35 and
                      (roll or self.slope[r['handoff_anchor']]<.10))
            if not expected and self.flow is not None:
                flow,self.flow=self.flow,(r['nominal_flow'],self.flow[1])
                try:
                    return super().command(touch)
                finally:
                    self.flow=flow
            if not expected or r['handoff_choice']=='guard':
                return super().command(touch)
            if roll:
                return super().command(touch)
            self.rescue=None
            i=ids[0]
            n=touch.normals[i]
            speed=r['tangent_speed_m_s']
            direction=transport(np.asarray(r['selected_direction']),self.normal,n) if speed else np.zeros(3)
            omega=np.asarray(r['angular_velocity'])
            arm=self.measured_position-touch.points[i]
            peak=touch.loads.max()
            old=min(.12,.013*(peak-3.)) if peak>4. else (-.060 if peak<1.3 else -.014)
            desired_force=np.clip(.010*(touch.forces[i]-1.),-.009,.023)
            v=speed*direction+(r['height_speed_m_s']+desired_force-old)*n
            v+=(Rotation.from_rotvec(DT*omega).apply(arm)-arm)/DT
            remain=max(1,self.apply_tick+PLAN_TICKS-self.tick)*DT
            self.goal=self.position+remain*v
            self.goal_attitude=Rotation.from_rotvec(remain*omega)*self.attitude
            self.normal=n
            return InformativeWrist.command(self,touch)
        return super().command(touch)


class TransitionWorker(delivery.DeliveryWorker):
    def __init__(self,servo,paced):
        super().__init__(servo,paced)
        self.nominal_kernel=self.map.kernel
        self.dense_kernel=transition_compute.EagerMatern32()

    def submit(self,data,touch,tick,lead):
        s=self.servo
        follower=s.selector
        follower.settings=(s.mode,tick>=s.enable_from)
        follower.dwell,follower.slope=s.dwell.copy(),s.slope.copy()
        follower.empty=s.empty
        follower.recovery_needed=s.recovery_needed
        follower.pair_age=tick-s.last_pair
        follower.onset_age=tick-s.last_empty
        self.map.kernel=self.dense_kernel if tick>=s.enable_from and s.mode=='both' else self.nominal_kernel
        follower.stations=(*s.stations,*s.single_stations)
        extra=touch.loads[:4]-touch.forces
        region=int(np.argmax(extra))
        follower.side_load=float(extra[region])
        follower.link_vertices=None
        if (tick>=s.enable_from and s.mode in ('handoff','both') and s.recovery_needed and
                touch.measured_heads.sum()==1 and extra[region]>1.55):
            select=s.link_hull.regions==region
            geoms=s.link_hull.ids[select]
            follower.link_vertices=(data.geom_xpos[geoms]+np.einsum('vij,vj->vi',
                data.geom_xmat[geoms].reshape(-1,3,3),s.link_hull.vertices[select]))
        follower.origins,follower.matrices=s.origins.copy(),s.matrices.copy()
        return super().submit(data,touch,tick,lead)

    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        if row is not None:
            self.servo.transition=row
        return row


def run(folder,obj,seconds=240,device='cuda',mode='both',enable_from=0,prior=None):
    if mode not in MODES:
        raise ValueError(f'Unknown transition mode {mode}')
    wrist=partial(TransitionWrist,mode=mode,enable_from=enable_from)
    label=f'tactile-transition-{mode}-from{enable_from}'
    return field.run(folder,obj,seconds,device,prior,wrist_cls=wrist,worker_cls=TransitionWorker,
                     source_files=(__file__,entry_frontier.__file__,transition_compute.__file__,delivery.__file__,tactile_reentry.__file__,tactile_stall.__file__),release=label)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',required=True,choices=OBJECTS)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seconds',default=240,type=float)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--mode',choices=MODES,default='both')
    parser.add_argument('--enable-from',type=int,default=0)
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,args.device,args.mode,args.enable_from,args.prior),flush=True)
