"""Online reward and five-frame transition collection on the accepted B chain."""
import torch
from torch import nn
from bounded_residual import BoundedResidualPolicy
from b_residual_reward_features import BRewardGeometry,HeldAngleCost,HeldContactMarginCost
from residual_reward import FullResidualReward,FullRewardConfig

ACTOR_FIELDS=[('q',16),('q_velocity',16),('contact_points_palm',12),('contact_normals_palm',12),('found',4),
    ('palm_twist',6),('published_palm_command',6),('nominal_delta',16),('candidate_future_delta',48),
    ('previous_applied_residual',16),('pad_sites',12),('pad_outward',12),('strict_contact',4),('force',4),
    ('loss_count',4),('recovery_count',4),('lost',4),('angle_cost',4),('angle_age',4),('pad_margin_cost',4),
    ('previous_decision_residual',16),('scheduler_phase',1),('previous_mcc_delta',16),('recontact_offset',16),('recontact_active',4)]
OBS_DIM=sum(width for _,width in ACTOR_FIELDS)
CRITIC_EXTRA_FIELDS=['previous_phi','previous_geometry_phi','reference_q_cost','reference_valid','palm_phase']

class DecisionActor(nn.Module):
    def __init__(self,owner):
        super().__init__();object.__setattr__(self,'owner',owner)
    def forward(self,*args):return self.owner.next_action


class TrainingResidual(nn.Module):
    def __init__(self,chain,agent,calibration,bounds,reference,frames,*,zero=False,evaluate=False):
        super().__init__();self.chain=chain;self.agent=agent;self.cal=calibration;self.zero=zero;self.evaluate=evaluate
        self.frames=frames;self.context=None;self.latest_chunk=None;self.plan_source=None
        B=chain.batch;device=chain.device
        self.amplitude=torch.tensor(bounds['amplitude_rad'],device=device).expand(B,-1).clone()
        # Frozen-actor range probes retain the training input/reward units.
        self.normalization_amplitude=torch.tensor(bounds.get('normalization_amplitude_rad',bounds['amplitude_rad']),device=device).expand(B,-1).clone()
        if not torch.isfinite(self.normalization_amplitude).all() or (self.normalization_amplitude<=0).any():
            raise ValueError('normalization amplitude must be finite and positive')
        slew=torch.tensor(bounds['decision_slew_rad'],device=device).expand(B,-1).clone()
        self.limiter=BoundedResidualPolicy(DecisionActor(self),self.amplitude,slew,5)
        self.geom=BRewardGeometry(device)
        self.angle=HeldAngleCost(B,torch.tensor(calibration['angle_margin_deg'],device=device),torch.tensor(calibration['angle_scale_deg'],device=device))
        self.margin=HeldContactMarginCost(B,calibration,device)
        self.reward=FullResidualReward(B,device,FullRewardConfig(**calibration['config']))
        self.reference=reference
        self.pending=torch.zeros(B,dtype=torch.bool,device=device)
        self.elapsed=torch.zeros(B,dtype=torch.long,device=device)
        self.accumulated=torch.zeros(B,device=device)
        self.previous_obs=None;self.previous_critic=None;self.previous_action=None
        self.next_action=torch.zeros(B,16,device=device)
        self.started=False;self.component_sums={};self.decision_count=0
        self.strict_count=torch.zeros(B,4,device=device);self.strict_all_count=torch.zeros(B,device=device)
        self.found_count=torch.zeros(B,4,device=device);self.found_all_count=torch.zeros(B,device=device)
        self.observed_count=torch.zeros(B,device=device);self.peak=torch.zeros(B,device=device)
        self.force_history=torch.zeros(frames,B,device=device)
        self.observed_history=torch.zeros(frames,B,device=device,dtype=torch.bool)
        self.current_loss=torch.zeros(B,dtype=torch.long,device=device);self.longest_loss=torch.zeros_like(self.current_loss)
        self.max_offset_ratio=0.;self.max_slew_ratio=0.

    @property
    def actor_decisions(self):return self.limiter.actor_decisions
    def set_context(self,context):self.context=context

    def observations(self,q,nominal,state,features,geometry,qcost,valid):
        B=self.chain.batch;chunk=self.latest_chunk
        if self.plan_source is not None and self.plan_source.plan:
            chunk=self.plan_source.plan['chunk'];self.latest_chunk=chunk
        if chunk is None:future=nominal[:,None].expand(-1,3,-1)
        else:
            ids=(self.chain.scheduler.last_drop_index[:,None]+torch.tensor([0,1,2],device=q.device)[None]).clamp_max(chunk.shape[1]-1)
            future=chunk.gather(1,ids[:,:,None].expand(-1,-1,16))
        phase=torch.full((B,1),self.chain.scheduler.path_index/10,device=q.device)
        values=[q,self.context['q_velocity']/10,features['points_palm'].flatten(1)*10,
            features['normals_palm'].flatten(1),state[:,40:44],state[:,84:90],state[:,90:96],
            (nominal-q)/.5,(future-q[:,None]).flatten(1)/.5,self.limiter.offset/self.normalization_amplitude,
            features['pad_sites_palm'].flatten(1)*10,features['pad_outward_palm'].flatten(1),
            features['strict'].float(),torch.linalg.vector_norm(self.context['force'],dim=-1)/self.cal['config']['force_warn_n'],
            self.reward.loss_count.float()/11,self.reward.recovery_count.float()/5,self.reward.lost.float(),
            self.angle.cost,self.angle.age.float()/100,self.margin.position_cost,self.reward.previous_residual,
            phase,(self.chain.mcc.previous_command-q)/.5,self.chain.runtime_recovery_offset/.05,self.chain.recovery_active.float()]
        obs=torch.nan_to_num(torch.cat(values,-1),nan=0,posinf=10,neginf=-10).clamp(-10,10)
        if obs.shape[-1]!=OBS_DIM:raise AssertionError('actor observation layout mismatch')
        progress=torch.full((B,),self.context['frame']/2500,device=q.device)
        privileged=torch.stack((self.reward.previous_phi,self.reward.previous_geometry_phi,qcost,valid.float(),progress),-1)
        critic=torch.cat((obs,privileged),-1)
        return obs,critic

    def flush(self,mask,obs,critic,terminal):
        mask=mask & self.pending
        if not bool(mask.any()):return
        discount=self.reward.config.gamma_frame**self.elapsed[mask]
        discount=torch.where(terminal[mask],0,discount)
        if not self.evaluate and not self.zero:
            ids=mask.nonzero().flatten()
            if self.agent.cfg.get('exact_transition_budget',False):
                remaining=max(0,self.agent.cfg['total_transitions']-self.agent.transitions)
                ids=ids[:remaining]
                discount=discount[:remaining]
            if len(ids):
                self.agent.replay.add(obs=self.previous_obs[ids],critic=self.previous_critic[ids],action=self.previous_action[ids],
                    next_obs=obs[ids],next_critic=critic[ids],reward=self.accumulated[ids],discount=discount)
                self.agent.transitions+=len(ids)
        self.pending[mask]=False;self.elapsed[mask]=0;self.accumulated[mask]=0

    @torch.no_grad()
    def forward(self,q,nominal,state):
        ctx=self.context;t=ctx['frame'];B=len(q);device=q.device
        force=torch.linalg.vector_norm(ctx['force'],dim=-1)
        f=self.geom.extract(q,ctx['palm_pose'],ctx['points'],state[:,28:40].reshape(B,4,3),ctx['found'],force,
                            normals_are_palm=True,include_pad_pose=True)
        angle=self.angle.update(f['angle_deg'],f['normal_valid'])
        margin=self.margin.update(f['local_x_m'],f['local_radius_m'],ctx['found'],force)
        geometry=torch.maximum(angle,margin)
        stabilizer=self.agent.cfg.get('stabilizer_prototype',False)
        if stabilizer:
            qref=q;qcost=torch.zeros(B,device=device);valid=torch.zeros(B,dtype=torch.bool,device=device)
        else:
            qref=self.reference['q'][t];qcost=((q-qref).square().mean(-1).sqrt()/self.cal['q_prior_rms_scale_rad']).clamp(0,1)
            valid=self.reference['strict'][t].all(-1) & (self.reference['force'][t].amax(-1)<=self.cal['config']['force_warn_n'])
        # Privileged teacher q is only used in this reward qualifier, never in actor features.
        if not stabilizer and (bool((valid & (self.reward.lost.any(-1) | (self.reward.loss_count>=10).any(-1))).any()) or bool((force.amax(-1)>self.cal['config']['force_warn_n']).any())):
            support=self.geom.prior_support(q,qref,f['points_palm'],f['normals_palm'],ctx['found'],force,
                self.angle.margin,self.angle.scale,self.margin.radius,self.margin.x,self.margin.low)['support']
            valid &= support
        else:valid.zero_()
        boundary=self.limiter.control_steps%5==0
        last=bool(ctx['last_observation'])
        completed=torch.full((B,),last and self.frames==2500,device=device,dtype=torch.bool)
        failure=ctx['newly_stopped'] | ~torch.isfinite(q).all(-1) | ~torch.isfinite(force).all(-1)
        if not self.started:
            phi=self.reward.potential(f['strict'],force,geometry,qcost,valid)[0]
            self.reward.reset(phi,-self.reward.config.geometry_weight*geometry.mean(-1))
            self.reward.active.copy_(ctx['active']);self.started=True
        else:
            r=self.reward.step(strict=f['strict'],force_n=force,geometry_cost=geometry,q_prior_cost=qcost,
                reference_valid=valid,normalized_residual=self.limiter.offset/self.normalization_amplitude,
                completed=completed,failure=failure,decision_boundary=boundary)
            self.accumulated+=torch.where(self.pending,(self.reward.config.gamma_frame**self.elapsed)*r['total'],0)
            self.elapsed+=self.pending.long()
            for k in ('contact','loss','recovery','force','potential','potential_geometry','potential_q',
                      'effort','smooth','terminal','total','loss_events','recovery_events'):
                self.component_sums[k]=self.component_sums.get(k,0.)+float(r[k].sum())
        obs,critic=self.observations(q,nominal,state,f,geometry,qcost,valid)
        self.flush(failure | completed,obs,critic,failure | completed)
        if boundary or last:
            self.flush(self.pending,obs,critic,failure | completed)
            if not self.zero and not self.evaluate and (boundary or ('updates_per_transition' in self.agent.cfg and last)):
                if 'updates_per_transition' in self.agent.cfg:
                    target=int(max(0,self.agent.transitions-self.agent.cfg['learning_starts'])*self.agent.cfg['updates_per_transition'])
                    count=max(0,target-self.agent.updates)
                else:count=self.agent.cfg['updates_per_decision']
                for _ in range(count):self.agent.update()
            if not last:
                action=self.agent.act(obs,zero=self.zero,evaluate=self.evaluate)
                self.next_action=torch.where(ctx['active'][:,None],action,torch.zeros_like(action))
                self.previous_obs=obs.clone();self.previous_critic=critic.clone();self.previous_action=self.next_action.clone()
                self.pending.copy_(ctx['active']);self.decision_count+=1
            else:self.next_action.zero_()
        alive=ctx['active'] | ctx['newly_stopped']
        found=ctx['found'].bool()
        self.found_count+=(found & alive[:,None]).float()
        self.found_all_count+=(found.all(-1) & alive).float()
        self.strict_count+=(f['strict'] & alive[:,None]).float();self.strict_all_count+=(f['strict'].all(-1) & alive).float()
        self.observed_count+=alive.float();self.peak=torch.maximum(self.peak,force.amax(-1))
        self.force_history[t]=force.amax(-1);self.observed_history[t]=alive
        self.current_loss=torch.where(alive & ~f['strict'].all(-1),self.current_loss+1,0)
        self.longest_loss=torch.maximum(self.longest_loss,self.current_loss)
        before=self.limiter.offset.clone();offset=self.limiter(q,nominal,state)
        self.max_offset_ratio=max(self.max_offset_ratio,float((offset.abs()/self.amplitude).max()))
        frame_slew=self.limiter.decision_slew/5
        self.max_slew_ratio=max(self.max_slew_ratio,float(((offset-before).abs()/frame_slew).max()))
        return offset

    def summary(self):
        # Failed/parked tails count as missed contact over the requested deployment horizon.
        denominator=max(1,self.frames-76)
        return dict(actor_decisions=self.actor_decisions,components=self.component_sums,
            metrics_scope='control observations from frame 76; stopped tails count as failed contact',
            strict_four_rate=(self.strict_all_count/denominator).cpu().tolist(),
            strict_per_finger=(self.strict_count/denominator).cpu().tolist(),
            healthy_four_rate=(self.strict_all_count/denominator).cpu().tolist(),
            healthy_per_finger=(self.strict_count/denominator).cpu().tolist(),
            found_four_rate=(self.found_all_count/denominator).cpu().tolist(),
            found_per_finger=(self.found_count/denominator).cpu().tolist(),
            contact_metric_definitions=dict(found='sensor found, no pad-region or force filtering',
                strict='original collection standard: found, finite local pad position, radius<=0.05m, local x<=0.012m (thumb 0.016m), force>=0.05N; healthy aliases use this unchanged rule'),
            peak_force_n=self.peak.cpu().tolist(),
            force_p95_observed=[float(torch.quantile(self.force_history[:,i][self.observed_history[:,i]],.95))
                if bool(self.observed_history[:,i].any()) else None for i in range(self.chain.batch)],
            longest_observed_loss=self.longest_loss.cpu().tolist(),
            max_amplitude_ratio=self.max_offset_ratio,max_frame_slew_ratio=self.max_slew_ratio,
            pending_transitions=int(self.pending.sum()))


class PlanTrace:
    def __init__(self,policy):self.policy=policy
    def record_plan(self,**values):self.policy.latest_chunk=values['chunk'].detach()
    def record_frame(self,**values):pass
