"""Plot the complete time series and render synchronized measured MuJoCo states."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .experiment import ARMS, CASES, ROOT, scenario
from .history import read_model


LABEL={'history-dp':'History + DP + MCC','history-mcc':'History + MCC','rigid-dp':'Rigid + DP + MCC',
       'no-gain-dp':'No local gain + DP + MCC','aude-mcc':'Aude + MCC','aude-single-all-mcc':'Active-2 Aude + DP + MCC'}
COLOR=dict(zip(ARMS,('#007968','#d27e14','#315eae','#9270a8','#933b39')))


def plots(root):
    fig,axes=plt.subplots(2,2,figsize=(10.2,7),sharex=True,sharey='row',layout='constrained')
    for col,obj in enumerate(('ellipsoid','meat-can')):
        for arm in ARMS:
            curves=[]; qualities=[]
            for start in ('start-01','start-02'):
                folder=root/'runs'/obj/start/arm
                with np.load(folder/'coverage-curve.npz') as d:
                    curves.append(100*d['coverage10'][::100])
                    t=d['time_s'][::100]
                scores=json.loads((folder/'score.json').read_bytes())['snapshots']
                qualities.append([100*s['signed']['fscore'] for s in scores])
            axes[0,col].plot(t,np.mean(curves,axis=0),c=COLOR[arm],label=LABEL[arm],lw=2)
            axes[1,col].plot([s['seconds'] for s in scores],np.mean(qualities,axis=0),c=COLOR[arm],marker='o',lw=2,ms=4)
        axes[0,col].set_title('Ellipsoid' if col==0 else 'Meat can',fontweight='bold')
        for ax in axes[:,col]:
            ax.set_ylim(0,104);ax.set_xlim(0,240);ax.grid(axis='y',alpha=.22)
            ax.spines[['top','right']].set_visible(False)
        axes[1,col].set_xlabel('Simulated time (s)')
    axes[0,0].set_ylabel('Low-load 10 mm coverage (%)')
    axes[1,0].set_ylabel('Signed GPIS F-score at 5 mm (%)')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.savefig(root/'time-curves.png',dpi=180,bbox_inches='tight');plt.close(fig)


def pear_plot(root):
    from .evaluate import read_trace, reference
    from .experiment import ACTIVE2, CONTROL_DT
    spec=json.loads((root/'pear-full/long-protocol.json').read_bytes())
    ticks=round(spec['seconds']/CONTROL_DT)
    fig,axes=plt.subplots(2,1,figsize=(9.0,6.8),sharex=True,layout='constrained')
    legend=(('history-dp','Wrist history + DP + MCC','#007968'),('history-mcc','Wrist history + MCC','#d27e14'),
            ('aude-mcc','Active-2 Aude + MCC','#933b39'),('aude-single-all-mcc','Active-2 Aude + DP + MCC','#315eae'))
    cache=root/'pear-full/plot-series.npz'
    signals={}
    if cache.is_file():
        with np.load(cache) as d:
            signals={k:d[k] for k in d.files}
    else:
        ref=reference('pear')
        for arm,_,_ in legend:
            case=ACTIVE2/arm if arm.startswith('aude') else root/'pear-full/runs/pear/active-02'/arm
            trace=read_trace(case/'trace.npz',ticks)
            valid=trace['pad_valid']&(trace['region_load_n'].max(1)<8.)[:,None]
            t,f=np.nonzero(valid)
            coverage,_=ref.area(trace['points_m'][t,f],trace['normals_outward'][t,f],t,ticks)
            signals[arm+'_area']=coverage[0][::100]
            # A trailing window represents only already executed times.
            last=np.arange(1,ticks+1,100);first=np.maximum(0,last-3000)
            count=np.r_[0,np.cumsum(valid[1:].sum(1)>=3)]
            signals[arm+'_three']=(count[last]-count[first])/(last-first)
        np.savez_compressed(cache,**signals)
    for arm,label,color in legend:
        for ax,key in zip(axes,('area','three')):
            values=100*signals[arm+'_'+key]
            ax.plot(np.arange(len(values)),values,label=label,color=color,lw=1.8)
    for ax in axes:
        ax.set_ylim(-1,103);ax.set_xlim(0,spec['seconds']);ax.grid(axis='y',alpha=.22)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Low-load 10 mm coverage (%)')
    axes[1].set_ylabel('Low-load three tips (%)\nTrailing 30 s')
    axes[1].set_xlabel('Simulated time (s)')
    axes[0].set_title('Pear, matched Active-2 initial state',fontweight='bold')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(root/'pear-full-comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)


def replay(root):
    arms=('history-dp','history-mcc','aude-mcc')
    width,height=384,304
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    bold=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',18)
    options=mujoco.MjvOption();options.geomgroup[3:]=0
    times=np.linspace(0,240,145)
    output=root/'matched-replay.mp4'
    with imageio.get_writer(output,fps=12,codec='libx264',quality=8,macro_block_size=1,
                            ffmpeg_params=['-preset','veryfast']) as writer:
        for obj,start in CASES:
            source,_=scenario(obj,start)
            model=read_model(source/'model.mjb')
            model.vis.global_.offwidth=width;model.vis.global_.offheight=height
            model.vis.headlight.ambient[:]=.55
            data=mujoco.MjData(model)
            traces=[];curves=[]
            for arm in arms:
                directory=root/'runs'/obj/start/arm
                with np.load(directory/'trace.npz') as d:
                    traces.append({k:d[k] for k in ('qpos','region_load_n','pad_valid','points_m')})
                with np.load(directory/'coverage-curve.npz') as d:
                    curves.append(d['coverage10'])
            visual=model.geom('bottle_visual').id
            geom=model.geom_dataid[visual]
            offset=model.mesh_vertadr[geom]
            local=model.mesh_vert[offset:offset+model.mesh_vertnum[geom]]
            with mujoco.Renderer(model,height=height,width=width) as renderer:
                for t in times:
                    tick=round(t*100)
                    canvas=Image.new('RGB',(len(arms)*width,height+76),'#17212a')
                    pen=ImageDraw.Draw(canvas)
                    pen.text((12,6),f'{"Ellipsoid" if obj=="ellipsoid" else "Meat can"}  /  {"A" if start.endswith("01") else "B"}     {t:5.1f} s',fill='white',font=bold)
                    for column,arm in enumerate(arms):
                        tr=traces[column]
                        data.qpos[:]=tr['qpos'][tick]
                        mujoco.mj_forward(model,data)
                        center=data.geom_xpos[visual]+local.mean(0)@data.geom_xmat[visual].reshape(3,3).T
                        radial=data.qpos[:3]-center
                        cam=mujoco.MjvCamera()
                        cam.lookat[:]=.78*center+.22*data.qpos[:3]
                        cam.distance=max(.48,2.2*np.ptp(local,axis=0).max())
                        cam.azimuth=np.rad2deg(np.arctan2(radial[1],radial[0]))+112.
                        cam.elevation=np.clip(-12.-25.*radial[2]/max(np.linalg.norm(radial),.01),-48,18)
                        renderer.update_scene(data,camera=cam,scene_option=options)
                        clean=tr['region_load_n'][tick].max()<8.
                        for p in tr['points_m'][tick,tr['pad_valid'][tick]]:
                            scene=renderer.scene
                            marker=scene.geoms[scene.ngeom]
                            mujoco.mjv_initGeom(marker,mujoco.mjtGeom.mjGEOM_SPHERE,[.004]*3,p,np.eye(3).ravel(),
                                                [0.,.9,.55,1.] if clean else [1.,.25,.18,1.])
                            scene.ngeom+=1
                        canvas.paste(Image.fromarray(renderer.render()),(column*width,32))
                        pen.text((column*width+8,height+34),LABEL[arm],font=bold,fill='white')
                        pen.text((column*width+8,height+56),f'10 mm coverage {curves[column][tick]:.1%}     load {tr["region_load_n"][tick].max():.1f} N',font=font,fill='#cbdce4')
                    writer.append_data(np.asarray(canvas))
    print(output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('plot','video'));p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args()
    (plots if args.action=='plot' else replay)(args.root)


if __name__=='__main__':
    main()
