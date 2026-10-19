"""Native saved-q visualization; no invented motion or geometric coverage."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from Module.fr3_leap.model import compile_model_xml,ARM_JOINT_NAMES,HAND_JOINT_NAMES
from Module.e05_physics.scene import FINGERS
from .pad_contact_visual import camera_from_direction,marker


def load_preview(case,allow_in_progress=False):
    audit=case/'independent_native_audit/audit.json'
    if audit.exists():
        result=json.loads(audit.read_text());assert result['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
        assert sha256((case/'model.xml').read_bytes()).hexdigest()==result['source_files']['model.xml']
        source=case/'independent_native_audit/verified_preview.npz'
        assert sha256(source.read_bytes()).hexdigest()==result['preview_sha256']
        status='原生连续重放已核验'+(' · 本次整瓶验收通过' if result['full_surface_complete'] else ' · 本次未完成整瓶')
        if not result.get('fresh_single_version',True):status+=' · 多版本开发续接'
        with np.load(source) as z:return {k:z[k].copy() for k in z.files},status
    if not allow_in_progress:raise ValueError('A passed native replay audit is required for a finished video')
    chunks=json.loads((case/'trace_chunks.json').read_text());arrays={}
    keys=('time_s','q_rad','forces_n','belly_active','belly_contact_positions_world_m','phase','coverage','leader_finger')
    for chunk in chunks:
        with np.load(case/chunk['file']) as z:
            for k in keys:arrays.setdefault(k,[]).append(z[k][::25])
    return {k:np.concatenate(v) for k,v in arrays.items()},'运行中保存状态 · 尚未完成全程重放核验'


def regional_contact_samples(case,trace):
    """Select real safe contact frames on six sides from audited raw chunks."""
    audit=json.loads((case/'independent_native_audit/audit.json').read_text())
    assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
    assert sha256((case/'summary.json').read_bytes()).hexdigest()==audit['source_files']['summary.json']
    summary=json.loads((case/'summary.json').read_text())
    body=ET.fromstring((case/'model.xml').read_text()).find(".//body[@name='fr3_e05_object']")
    origin=np.fromstring(body.get('pos'),sep=' ')
    names=['宽面 +Z','侧面 +X','宽面 −Z','侧面 −X','瓶盖/瓶肩端 −Y','瓶底端 +Y']
    axes=[2,0,2,0,1,1];signs=[1,1,-1,-1,-1,1]
    candidates=[[] for _ in names];offset=0
    for chunk in summary['trace_chunks']:
        source=case/chunk['file'];assert sha256(source.read_bytes()).hexdigest()==chunk['sha256']
        with np.load(source) as z:
            count=len(z['time_s'])-int(chunk.get('skip_last',0))
            # Match the global 20 Hz grid of the independently verified preview.
            ids=np.arange((-offset)%25,count,25);offset+=count
            safe=z['safe_belly'][ids];n=z['belly_normals_world'][ids]
            p=z['belly_contact_positions_world_m'][ids]-origin
            for r,(axis,sign) in enumerate(zip(axes,signs)):
                qualifies=safe&(n[:,:,axis]*sign>.6)
                if r==4:qualifies&=p[:,:,1]<-.20
                if r==5:qualifies&=p[:,:,1]>.20
                rows=np.flatnonzero(qualifies.any(axis=1))
                candidates[r].extend(z['time_s'][ids[rows]].tolist())
    result=[]
    for name,values in zip(names,candidates):
        times=np.asarray(values)
        if not len(times):
            result.append(dict(region=name,available=False));continue
        breaks=np.r_[0,np.flatnonzero(np.diff(times)>.051)+1,len(times)]
        starts,ends=breaks[:-1],breaks[1:]
        best=int(np.argmax(ends-starts));a,b=int(starts[best]),int(ends[best])
        t=float(times[(a+b-1)//2]);k=int(np.searchsorted(trace['time_s'],t-1e-8))
        assert abs(float(trace['time_s'][k])-t)<1e-7
        result.append(dict(region=name,available=True,time_s=t,
            preview_contact_window_s=float(times[b-1]-times[a]),preview_index=k))
    return result


def render(case,speed=12.,stills_only=False,allow_in_progress=False,contact_stills=False,contact_details=False):
    if not np.isfinite(speed) or speed<=0:raise ValueError('Playback speed must be positive')
    case=Path(case).resolve();trace,status=load_preview(case,allow_in_progress)
    m=compile_model_xml((case/'model.xml').read_text());d=mujoco.MjData(m)
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in (*ARM_JOINT_NAMES,*HAND_JOINT_NAMES)]
    pads=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f.proxy_geom.replace('e05_','fr3_')) for f in FINGERS]
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
    rgba=m.geom_rgba.copy();matid=m.geom_matid.copy()
    options=mujoco.MjvOption();options.geomgroup[3:]=0;options.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    renderer=mujoco.Renderer(m,height=540,width=800)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    title=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',24)
    end=float(trace['time_s'][-1]);regions=None;selected=[]
    segments=[]
    if contact_stills or contact_details:
        if allow_in_progress:raise ValueError('Regional contact frames require a completed native audit')
        regions=regional_contact_samples(case,trace);selected=[r for r in regions if r['available']]
        if contact_details:
            if contact_stills:raise ValueError('Choose regional stills or the normal-speed detail video')
            speed=1.;frames=[]
            for r in selected:
                start=max(0.,r['time_s']-1.5);finish=min(end,r['time_s']+1.5)
                segments.append(dict(region=r['region'],start_s=start,end_s=finish))
                for t in np.arange(start,finish,.05):
                    k=min(int(np.searchsorted(trace['time_s'],t-1e-8)),len(trace['time_s'])-1)
                    frames.append(dict(region=r['region'],time_s=float(trace['time_s'][k]),preview_index=k))
            selected=frames;stills_only=False
        else:stills_only=True
        times=np.array([r['time_s'] for r in selected])
    else:times=np.linspace(0,end,6) if stills_only else np.r_[np.arange(0,end,speed/20.),end]
    prefix=('natural_details' if contact_details else 'natural_contact') if (contact_stills or contact_details) else ('natural_running' if allow_in_progress else 'natural_complete')
    visual_source=case/'visualization_source';visual_source.mkdir(exist_ok=True)
    renderer_source=visual_source/f'{prefix}_renderer.py';renderer_source.write_bytes(Path(__file__).read_bytes())
    video=case/f'{prefix}_{speed:g}x.mp4'
    writer=None if stills_only else imageio.get_writer(video,fps=20,codec='libx264',quality=7,macro_block_size=2,output_params=['-threads','2'])
    snapshots=[]
    for frame,t in enumerate(times):
        k=selected[frame]['preview_index'] if (contact_stills or contact_details) else min(int(np.searchsorted(trace['time_s'],t)),len(trace['time_s'])-1)
        d.qpos[m.jnt_qposadr[ids]]=trace['q_rad'][k];mujoco.mj_forward(m,d)
        lead=int(trace['leader_finger'][k]);center=d.geom_xpos[pads[lead]].copy();frameR=d.geom_xmat[pads[lead]].reshape(3,3)
        panels=[]
        for panel in range(2):
            m.geom_rgba[:]=rgba;m.geom_matid[:]=matid
            for g,name in enumerate(names):
                if 'normal_marker' in name:m.geom_rgba[g,3]=0
                if name=='mustard_visual':m.geom_matid[g]=-1;m.geom_rgba[g]=[.96,.72,.08,.12 if panel else 1.]
            if panel:
                camera_from_direction(camera,.85*frameR[:,2]+.4*frameR[:,1]-.15*frameR[:,0],center,.35)
            else:
                camera.lookat[:]=[.3,0.,.49];camera.distance=.95;camera.azimuth=130.;camera.elevation=-25.
            renderer.update_scene(d,camera=camera,scene_option=options)
            for f in np.flatnonzero(trace['belly_active'][k]):marker(renderer.scene,trace['belly_contact_positions_world_m'][k,f],.002,[1.,.1,.55,1.])
            im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
            draw.rectangle((0,0,800,70),fill='#122238')
            headline=(selected[frame]['region']+(' · 局部正常速度回放' if contact_details else ' · 实测指肚接触')) if (contact_stills or contact_details) else '整瓶遍历 · 物体与手保持实心'
            draw.text((12,7),headline if panel==0 else '手形近景 · 仅物体透明辅助观察指肚',font=title,fill='white')
            draw.text((12,42),f'实际 t={trace["time_s"][k]:.2f}s   指肚面积覆盖 {trace["coverage"][k]*100:.2f}%',font=font,fill='#9ddbf9')
            panels.append(im)
        canvas=Image.new('RGB',(1600,690),'#122238');canvas.paste(panels[0],(0,0));canvas.paste(panels[1],(800,0));draw=ImageDraw.Draw(canvas)
        phase={'SLIDE':'贴面滑动','APPROACH':'指肚主动接近','ACTIVE_RECONTACT':'暂停前进并主动恢复接触',
               'NECESSARY_REORIENTATION':'必要的离面调姿','CONTACT_PIVOT':'贴面转腕与寻触','FINAL_SAFE_HOLD':'最终保持',
               'CONTACT_CONTEXT':'读取实际接触并继续规划','CLEARANCE_EXIT':'主动退出几何间隙边界'}.get(str(trace['phase'][k]),str(trace['phase'][k]))
        if phase=='FORCE_RELEASE':phase='主动卸力，随后恢复接触'
        if not np.any(trace['belly_active'][k]) and str(trace['phase'][k]) in ('SLIDE','CONTACT_PIVOT','FINAL_SAFE_HOLD'):
            phase+=' · 当前无合格指肚接触'
        playback='实测接触静帧' if contact_stills else f'播放 {speed:g}×'
        draw.text((15,549),f'{phase}  |  {playback}  |  {status}',font=font,fill='white')
        for f,label in enumerate(('食指','中指','无名指','拇指')):
            q=np.degrees(trace['q_rad'][k,7+4*f:11+4*f]);y=582+(f//2)*32;x=15+(f%2)*790
            text=f'{label}  关节° [{q[0]:5.1f}, {q[1]:5.1f}, {q[2]:5.1f}, {q[3]:5.1f}]   {trace["forces_n"][k,f]:.2f} N   '+('指肚接触' if trace['belly_active'][k,f] else '无合格接触')
            draw.text((x,y),text,font=small,fill='#a7edba' if trace['belly_active'][k,f] else '#d7e1ea')
        footer=('局部接触片段，切换时实际时间跳变；完整遍历见单独的全程视频。粉点为实际合格指肚接触。' if contact_details else
            '粉点为实际合格指肚接触。覆盖按原网格 25 mm 测地邻域计算；姿态或规划路径本身不计覆盖。')
        draw.text((15,655),footer,font=small,fill='#f3d38d')
        if writer:writer.append_data(np.asarray(canvas))
        if stills_only or frame in {0,len(times)//3,2*len(times)//3,len(times)-1}:
            file=case/f'{prefix}_frame_{frame:04d}.png';canvas.save(file)
            snapshots.append(dict(file=str(file),time_s=float(trace['time_s'][k]),**({'region':selected[frame]['region']} if (contact_stills or contact_details) else {})))
        if frame%200==0:print('RENDER',frame,len(times),flush=True)
    if writer:writer.close()
    renderer.close()
    meta=dict(actual_saved_q_only=True,no_physics_step=True,no_geometry_change=True,status=status,speed=speed,
        uncut=not stills_only and not contact_details,start_s=0.,end_s=end,frames=len(times),fps=20,stills=snapshots,
        source_manifest_sha256=sha256((case/'manifest.json').read_bytes()).hexdigest())
    meta.update(renderer_source=str(renderer_source),renderer_source_sha256=sha256(renderer_source.read_bytes()).hexdigest())
    if regions is not None:meta['regions']=regions
    if contact_details:meta.update(segments=segments,scope='Six local 1x contact excerpts with explicit source-time jumps; not the full traversal video')
    if writer:meta.update(video=str(video),video_sha256=sha256(video.read_bytes()).hexdigest())
    (case/f'{prefix}_render.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--speed',type=float,default=12.)
    p.add_argument('--stills',action='store_true');p.add_argument('--in-progress',action='store_true')
    p.add_argument('--contact-stills',action='store_true')
    p.add_argument('--contact-details',action='store_true')
    a=p.parse_args();render(a.case,a.speed,a.stills,a.in_progress,a.contact_stills,a.contact_details)
