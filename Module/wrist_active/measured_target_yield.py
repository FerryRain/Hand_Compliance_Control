"""Reweight global tactile information by recently observed pad tracking and support."""

import argparse
from pathlib import Path

import numpy as np

from . import tactile_field as field, tactile_reentry as reentry, tactile_stall as stall
from .sweep import LOAD_LIMIT
from .ycb_fixtures import OBJECTS


def group(source, distance, dwell, support):
    return (int(source=='mesh_frontier'),int(np.searchsorted((.022,.043),distance)),
            int(dwell>=10),int(support>=2))


class DeliveryGuide(reentry.ReentryGuide):
    def after_loss_weights(self,regions):
        current = self.delivery
        weights = super().after_loss_weights(regions) if self.recovery_observation[0] is not None else np.ones(len(regions))
        if current is None:
            return weights
        counts,quality,dwell,n=current
        pooled=(quality.sum()+3.)/(counts.sum()+6.)
        values=[]
        for r in regions:
            key=group(r.source,r.distance,dwell[r.finger],n)
            # Sparse contexts borrow tracking outcomes at similar distances.
            neighbours=(slice(None),key[1],slice(None),key[3])
            rate=(quality[neighbours].sum()+8.*pooled)/(counts[neighbours].sum()+8.)
            local=(quality[key]+4.*rate)/(counts[key]+4.)
            values.append(.25+.75*local)
        return weights*np.asarray(values)


class DeliveryWrist(stall.StallWrist):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        old=self.selector.guide
        self.selector.guide=DeliveryGuide(old.model,old.history,kwargs['seed'])
        self.counts=np.zeros((2,3,2,2))
        self.quality=np.zeros_like(self.counts)
        self.visit=None
        self.touch=None

    def observe(self,data,touch,tick):
        super().observe(data,touch,tick)
        self.touch=touch
        v=self.visit
        if v is None:
            return
        v['frames']+=1
        ids=touch.measured_heads
        if touch.loads.max()>=LOAD_LIMIT:
            return
        v['pair']+=int(ids.sum()>=2)
        f=v['finger']
        if ids[f] and touch.normals[f]@v['normal']>=.5:
            v['closest']=min(v['closest'],np.linalg.norm(touch.points[f]-v['point']))

    def close_visit(self):
        v=self.visit
        if v and v['frames']:
            progress=np.clip(1.-v['closest']/v['initial'],0.,1.)
            value=progress*(.45+.55*v['pair']/v['frames'])
            self.counts[v['key']]+=1
            self.quality[v['key']]+=value
        self.visit=None


class DeliveryWorker(stall.StallWorker):
    def submit(self,data,touch,tick,lead):
        from .multirate_wrist import MultirateWorker
        s=self.servo
        g=s.selector.guide
        g.delivery=(s.counts.copy(),s.quality.copy(),s.dwell.copy(),int(np.count_nonzero(s.dwell>=8))) if s.counts.sum()>=8 else None
        g.recovery_observation=s.failure if s.failure is not None else ((None,0) if g.delivery else None)
        return MultirateWorker.submit(self,data,touch,tick,lead)

    def install(self,tick,*,initial=False):
        row=super().install(tick,initial=initial)
        if row is None:
            return
        s=self.servo
        event=row['global_event']
        if event or row['mode']!='explore':
            s.close_visit()
        if event and event.get('direction_chosen') and row['mode']=='explore':
            index=event['target_finger']
            if s.touch.measured_heads[index]:
                point=np.asarray(event['target_m'])
                dist=np.linalg.norm(point-s.touch.points[index])
                if dist>.006:
                    s.visit=dict(key=group(event['source'],event['mesh_path_m'],s.dwell[index],np.count_nonzero(s.dwell>=8)),
                                 normal=np.asarray(event['target_normal']),finger=index,point=point,initial=dist,
                                 closest=dist,frames=0,pair=0)
        row['previous_tracking_observations']=int(s.counts.sum())
        return row


def run(folder,obj,seconds=240,device='cuda',prior=None):
    return field.run(folder,obj,seconds,device,prior,wrist_cls=DeliveryWrist,worker_cls=DeliveryWorker,
                     source_files=(__file__,stall.__file__,reentry.__file__),release='measured-target-yield')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',required=True,choices=OBJECTS)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--seconds',default=240,type=int)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--prior',type=Path)
    args=parser.parse_args()
    print(run(args.out,args.object,args.seconds,args.device,args.prior),flush=True)
