"""Lane-local DP memory; live tactile/velocity sensing remains uninterrupted."""
import torch

class ResetPolicyHistory:
    def __init__(self, supervisor, observation):
        self.supervisor=supervisor;self.capacity=observation.capacity;self.stride=observation.stride;self.horizon=observation.horizon
        self.states=torch.zeros_like(observation.states)
        self.counts=torch.zeros(observation.batch,device=observation.device,dtype=torch.long)
        self.initialized=False
        self.frozen_ticks=torch.zeros_like(self.counts)
        self.clear_count=torch.zeros_like(self.counts)
        self.pending_restart=torch.zeros_like(self.counts,dtype=torch.bool)

    def active_mask(self):
        # Inactive batch slots may still be sampled to keep a fixed RNG/batch
        # contract, but their prediction never owns an executable command.
        p=self.supervisor.phase
        return (p==0)|(p==6)

    def clear_lanes(self,mask):
        self.states[mask]=0;self.counts=torch.where(mask,0,self.counts)
        self.clear_count+=mask.long()

    def push(self,observation):
        if not self.initialized:
            self.states.copy_(observation.states);self.counts.fill_(observation.frames_since_clear)
            self.initialized=True
            return
        p=self.supervisor.phase
        # Bootstrap/settling precedes the first reset. Fresh warm-up after a
        # long interruption records real states instead of inventing history.
        commit=(p==0)|(p==6)|(p==9)
        rolled=torch.cat((self.states[:,1:],observation.states[:,-1:]),dim=1)
        self.states=torch.where(commit[:,None,None],rolled,self.states)
        self.counts+=commit.long();self.frozen_ticks+=(~commit).long()

    def history(self):
        indices=torch.arange(self.horizon-1,-1,-1,device=self.states.device)*self.stride
        return self.states[:,self.capacity-1-indices]

    def before_scheduler_observe(self,scheduler,nominal):
        if bool(self.pending_restart.any()):
            scheduler.restart_lanes(self.pending_restart,nominal)
            self.pending_restart.zero_()
