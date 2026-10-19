"""Clearly labelled actual-state progress stills; not a completed experiment report."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import mujoco
import numpy as np
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .sliding_audit import _native_context
from .pad_contact_visual import camera_from_direction,marker
from .natural_belly_report import coverage_figure


def run(directory,clip_seconds=0.):
    case=Path(directory).resolve();catalog=json.loads((case/'trace_chunks.json').read_text())
    item=catalog[-1];path=case/item['file'];assert sha256(path.read_bytes()).hexdigest()==item['sha256']
    with np.load(path) as z:
        if clip_seconds:
            first=np.searchsorted(z['time_s'],z['time_s'][-1]-min(20.,clip_seconds))
            ids=np.arange(first,len(z['time_s']),25).tolist()
            if ids[-1]!=len(z['time_s'])-1:ids.append(len(z['time_s'])-1)
        else:ids=sorted(set([0,len(z['time_s'])//2,len(z['time_s'])-1]))
        data={k:z[k][ids] for k in ('time_s','q_rad','forces_n','safe_belly','belly_contact_positions_world_m','coverage')}
    c=_native_context(case);m,d=c.model,c.data;pads=c.handles.tip_geom_ids
    option=mujoco.MjvOption();option.geomgroup[3:]=0;option.sitegroup[:]=0
    renderer=mujoco.Renderer(m,height=540,width=800);camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    m.vis.headlight.ambient[:]=[.25]*3;m.vis.headlight.diffuse[:]=[.45]*3
    rgba=m.geom_rgba.copy();matids=m.geom_matid.copy()
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
    outputs=[]
    video=case/'progress_actual_1x.mp4'
    writer=imageio.get_writer(video,fps=20,codec='libx264',quality=7,macro_block_size=2,
        output_params=['-threads','2']) if clip_seconds else None
    for k,t in enumerate(data['time_s']):
        d.qpos[c.qpos]=data['q_rad'][k];mujoco.mj_forward(m,d)
        centers=d.geom_xpos[pads].copy();frames=d.geom_xmat[pads].reshape(4,3,3)
        normal=frames[:3,:,2].mean(axis=0);normal/=np.linalg.norm(normal)
        canvas=Image.new('RGB',(2400,630),'#122238')
        for panel in range(3):
            m.geom_rgba[:]=rgba;m.geom_matid[:]=matids
            for geom,name in enumerate(names):
                if 'normal_marker' in name:m.geom_rgba[geom,3]=0.
                if name=='mustard_visual':m.geom_matid[geom]=-1;m.geom_rgba[geom]=[.96,.72,.08,.12 if panel else 1.]
            if panel==2:camera_from_direction(camera,frames[3,:,2]+.2*frames[3,:,1],centers[3],.27)
            elif panel:camera_from_direction(camera,normal+.65*frames[1,:,1],centers.mean(axis=0),.44)
            else:camera.lookat[:]=[.3,0.,.49];camera.distance=1.05;camera.azimuth=130.;camera.elevation=-25.
            renderer.update_scene(d,camera=camera,scene_option=option)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=not bool(panel)
            for finger in np.flatnonzero(data['safe_belly'][k]):marker(renderer.scene,data['belly_contact_positions_world_m'][k,finger],.002,[1.,.1,.55,1.])
            canvas.paste(Image.fromarray(renderer.render()),(800*panel,0))
        draw=ImageDraw.Draw(canvas);draw.rectangle((0,0,2400,42),fill='#122238')
        draw.text((15,8),f'整瓶运行中 · 原生实际姿态 t={t:.3f}s · 阶段节选，尚未宣称完整覆盖',font=font,fill='white')
        draw.text((15,550),f'实测覆盖 {data["coverage"][k]*100:.2f}% | 当前安全指腹 {data["safe_belly"][k].sum()}/4 | 粉点为原始记录中的实际合格接触',font=font,fill='white')
        draw.text((15,586),'左：整臂与原物体  中：三根长指协同屈伸（仅物体透明）  右：拇指主动对掌',font=font,fill='#a7dbf0')
        if writer:writer.append_data(np.asarray(canvas))
        if not writer or k in (0,len(ids)//2,len(ids)-1):
            name=f'progress_actual_{k}.png';canvas.save(case/name);outputs.append(name)
    if writer:
        writer.close()
        meta=dict(scope='Running experiment excerpt, not whole-object completion',speed=1.,fps=20,frames=len(ids),
            time_range_s=[float(data['time_s'][0]),float(data['time_s'][-1])],native_chunk=item,
            video=video.name,video_sha256=sha256(video.read_bytes()).hexdigest())
        (case/'progress_actual_1x_render.json').write_text(json.dumps(meta,indent=2))
    renderer.close();coverage=coverage_figure(case,in_progress=True)
    page='<!doctype html><meta charset="utf-8"><title>Mustard 整手遍历运行进度</title><style>body{font:18px/1.7 system-ui;background:#eef2f6;color:#17283b;margin:24px}img{width:100%;margin:14px 0}main{max-width:1500px;margin:auto}</style><main><h1>四指协同整瓶遍历 · 进行中</h1><p>以下是最新原生记录的实际姿态，不是规划动画。完整轨迹和最终独立核验尚未完成。</p>'
    if video.exists():page+='<p>正常速度原生实际运动节选：</p><video controls style="width:100%" src="progress_actual_1x.mp4"></video>'
    page+=''.join(f'<img src="{name}" alt="原生实际手形阶段截图">' for name in outputs)
    page+='<h2>当前覆盖与缺口</h2><img src="coverage_running.png" alt="运行中原网格覆盖"><p>绿色为实际合格接触的25mm测地邻域；灰色仍未覆盖。完成后另提供最终核验及完整视频。</p></main>'
    (case/'progress_review.html').write_text(page,encoding='utf-8')
    print(json.dumps(dict(stills=outputs,time_s=float(data['time_s'][-1]),coverage=coverage['area_fraction'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--clip-seconds',type=float,default=0.)
    a=p.parse_args();run(a.case,a.clip_seconds)
