"""Uncut, audited native-q playback for local belly-contact experiments.

Only the auxiliary bottle is transparent. Hands remain opaque in both views;
registered normal-marker geoms are hidden in the renderer, not in the model.
The gold cap is the declared classifier patch, never a pressure footprint.
"""
from __future__ import annotations
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from Module.fr3_leap.model import compile_model_xml, ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.e05_physics.scene import FINGERS
from .pad_contact_visual import camera_from_direction, connector, marker


def render(case, *, filename='belly_actual_uncut_1x.mp4', start=0., end=None, qa_times=()):
    case=Path(case).resolve()
    if Path(filename).name!=filename or not filename.endswith('.mp4'):
        raise ValueError('filename must be a local mp4 basename')
    raw=(case/'trace.npz').read_bytes();trace_hash=sha256(raw).hexdigest()
    model_bytes=(case/'model.xml').read_bytes();model_hash=sha256(model_bytes).hexdigest()
    audit_path=case/'independent_native_audit'/'audit.json'
    audit_bytes=audit_path.read_bytes();audit=json.loads(audit_bytes)
    if audit.get('status')!='PASSED_NATIVE_EVIDENCE_AUDIT':
        raise ValueError('Independent native audit has not passed')
    if audit['source_files']['trace.npz']!=trace_hash or audit['source_files']['model.xml']!=model_hash:
        raise ValueError('Independent audit does not bind these exact trace/model bytes')
    keys=('time_s','q_rad','forces_n','active','belly_forces_n','belly_active',
          'belly_contact_positions_world_m','phase')
    with np.load(BytesIO(raw),allow_pickle=False) as z:trace={k:z[k].copy() for k in keys}
    del raw
    t=trace['time_s'];dt=float(np.median(np.diff(t)))
    if len(t)<2 or not np.allclose(np.diff(t),dt,atol=1e-9,rtol=0):
        raise ValueError('The playback must use a continuous saved native trace')
    stop=float(t[-1] if end is None else end)
    if not float(t[0])-1e-9<=start<stop<=float(t[-1])+1e-9:
        raise ValueError('Requested interval must be within the saved trace')
    manifest=json.loads((case/'manifest.json').read_bytes())
    cfg=manifest['belly_contact'];rho=float(cfg['maximum_patch_radius'])
    model=compile_model_xml(model_bytes.decode('utf-8'));data=mujoco.MjData(model)
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in (*ARM_JOINT_NAMES,*HAND_JOINT_NAMES)]
    addresses=model.jnt_qposadr[joints]
    pads=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,f.proxy_geom.replace('e05_','fr3_')) for f in FINGERS]
    names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(model.ngeom)]
    base_rgba=model.geom_rgba.copy();base_matid=model.geom_matid.copy()
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    options=mujoco.MjvOption();options.geomgroup[3:]=0;options.sitegroup[:]=0
    renderer=mujoco.Renderer(model,height=540,width=840)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    title=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',22)
    mono=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',19)
    frames=np.r_[np.arange(start,stop,.05),stop]
    qa_indices={0,len(frames)//2,len(frames)-1}
    qa_indices.update(int(np.argmin(abs(frames-x))) for x in qa_times if start<=x<=stop)
    path=case/filename;temp=case/(path.stem+'.rendering.mp4')
    initial_q=trace['q_rad'][int(np.searchsorted(t,start))]
    saved_frames=[]
    with imageio.get_writer(temp,fps=20,codec='libx264',quality=7,macro_block_size=2,
                            output_params=['-threads','2']) as writer:
        for frame,at in enumerate(frames):
            k=min(int(np.searchsorted(t,at)),len(t)-1);actual=float(t[k])
            data.qpos[addresses]=trace['q_rad'][k];mujoco.mj_forward(model,data)
            centers=data.geom_xpos[pads];belly_dir=np.mean(data.geom_xmat[pads[:3]].reshape(3,3,3)[:,:,2],axis=0)
            panels=[]
            for panel in range(2):
                model.geom_rgba[:]=base_rgba;model.geom_matid[:]=base_matid
                for g,name in enumerate(names):
                    if name=='mustard_visual':
                        model.geom_matid[g]=-1;model.geom_rgba[g]=[.96,.72,.08,.14 if panel else 1]
                    if 'normal_marker' in name:model.geom_rgba[g,3]=0
                if panel:
                    camera_from_direction(camera,belly_dir,centers[:3].mean(axis=0),.25)
                else:
                    camera.lookat[:]=centers[:3].mean(axis=0)
                    camera.azimuth=130.;camera.elevation=-30.;camera.distance=.36
                renderer.update_scene(data,camera=camera,scene_option=options)
                for finger,pad in enumerate(pads):
                    center=data.geom_xpos[pad];R=data.geom_xmat[pad].reshape(3,3);size=model.geom_size[pad]
                    theta=np.linspace(0,2*np.pi,33)
                    local=np.column_stack([rho*size[0]*np.cos(theta),rho*size[1]*np.sin(theta),
                                           np.full_like(theta,np.sqrt(1-rho*rho)*size[2]+.00012)])
                    world=local@R.T+center
                    for a,b in zip(world[:-1],world[1:]):
                        connector(renderer.scene,a,b,[1.,.72,.12,1.],width=.00018)
                    if trace['belly_active'][k,finger]:
                        marker(renderer.scene,trace['belly_contact_positions_world_m'][k,finger],.0009,[1.,.12,.6,1.])
                im=Image.fromarray(renderer.render());d=ImageDraw.Draw(im)
                d.rectangle((0,0,840,62),fill='#101a2d')
                d.text((12,6),'局部指腹试验 · 实心手/物体' if panel==0 else '腹面辅助视角 · 仅物体透明，手仍实心',font=title,fill='white')
                phase_label={'UNLOAD_WRONG_FACE':'短暂卸载边缘接触','BELLY_ALIGN':'手指与腕联合调姿',
                             'BELLY_APPROACH':'指腹接近并主动寻触','BELLY_SLIDE':'指腹沿表面滑动'}.get(str(trace['phase'][k]),'实际运动')
                d.text((12,35),f'真实保存时刻 {actual:.3f}s  |  {phase_label}  |  连续 1×',font=font,fill='#c3eaff')
                panels.append(im)
            combined=Image.new('RGB',(1680,744),'#101a2d');combined.paste(panels[0],(0,0));combined.paste(panels[1],(840,0))
            d=ImageDraw.Draw(combined)
            for x,text in ((12,'手指'),(112,'实际四关节角度 / °'),(500,'相对片段开始变化 / °'),(930,'全部代理接触力'),(1150,'合格腹面力'),(1365,'实际腹面接触')):
                d.text((x,548),text,font=small,fill='#d2e8ff')
            for finger,name in enumerate(('食指','中指','无名指','拇指')):
                q=trace['q_rad'][k,7+finger*4:11+finger*4]
                delta=q-initial_q[7+finger*4:11+finger*4];y=579+29*finger
                d.text((12,y),name,font=font,fill='white')
                d.text((112,y),' '.join(f'{v:7.1f}' for v in np.rad2deg(q)),font=mono,fill='white')
                d.text((500,y),' '.join(f'{v:+7.1f}' for v in np.rad2deg(delta)),font=mono,fill='#d2e8ff')
                d.text((930,y),f'{trace["forces_n"][k,finger]:.3f} N',font=mono,fill='white')
                d.text((1150,y),f'{trace["belly_forces_n"][k,finger]:.3f} N',font=mono,fill='#ffde91')
                active=bool(trace['belly_active'][k,finger])
                d.text((1365,y),'有（实测）' if active else '无',font=font,fill='#74e39d' if active else '#bec9d9')
            d.text((12,709),'金环=腹面判定范围；粉点=实测合格腹面接触中心。姿态/指令本身不算接触；本视频不声称整瓶完成。',font=small,fill='#ffde91')
            writer.append_data(np.asarray(combined))
            if frame in qa_indices:
                output=case/f'{path.stem}_frame_{frame:04d}.png';temporary=output.with_name(output.stem+'.rendering.png')
                combined.save(temporary);saved_frames.append((temporary,output,actual))
            if frame%200==0:print(f'BELLY_VIDEO {frame}/{len(frames)} actual={actual:.3f}',flush=True)
    renderer.close()
    reader=imageio.get_reader(temp)
    try:
        if reader.count_frames()!=len(frames) or reader.get_data(len(frames)-1).shape!=(744,1680,3):
            raise RuntimeError('Video verification failed; previous output preserved')
    finally:reader.close()
    metadata={'source_case':str(case),'trace_sha256':trace_hash,'model_xml_sha256':model_hash,
              'audit_path':str(audit_path),'audit_sha256':sha256(audit_bytes).hexdigest(),
              'renderer_source_sha256':sha256(Path(__file__).read_bytes()).hexdigest(),
              'start_s':start,'end_s':stop,'source_saved_sample_interval_s':dt,'playback_speed':1.,
              'fps':20,'frames':len(frames),'encoded_duration_s':len(frames)/20.,'uncut':True,
              'source_scope':'Completed local development trial; not whole surface success',
              'no_mj_step':True,'actual_q_only':True,'no_model_file_changes':True,
              'display_changes':['Yellow mustard display material','Right bottle only alpha=.14',
                                 'Normal-marker geoms hidden','Declared belly patch outline and measured belly-contact centroids'],
              'belly_criteria':cfg,'left_camera':{'azimuth':130.,'elevation':-30.,'distance_m':.36},
              'right_camera':'Eye on mean actual first-three-pad +Z, distance .25m, lookat mean pad centers',
              'scope_limit':'Rendered frame rate cannot prove 150 ms stability; use the independent 2 ms measurements. Gold region is a classifier definition, not pressure area.',
              'qa_frames':[{'path':str(p),'time_s':at} for _,p,at in saved_frames],
              'video_sha256':sha256(temp.read_bytes()).hexdigest()}
    meta=case/(path.stem+'_render.json');meta_temp=meta.with_name(meta.stem+'.rendering.json')
    meta_temp.write_bytes(json.dumps(metadata,ensure_ascii=False,indent=2).encode('utf-8'))
    temp.replace(path);meta_temp.replace(meta)
    for a,b,_ in saved_frames:a.replace(b)
    print('BELLY_VIDEO_DONE',path,flush=True)
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('case',type=Path)
    parser.add_argument('--filename',default='belly_actual_uncut_1x.mp4')
    parser.add_argument('--start',type=float,default=0.);parser.add_argument('--end',type=float)
    parser.add_argument('--qa-time',type=float,action='append',default=[])
    args=parser.parse_args();render(args.case,filename=args.filename,start=args.start,end=args.end,qa_times=args.qa_time)
