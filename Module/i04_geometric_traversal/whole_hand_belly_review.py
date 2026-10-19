"""Review actual contact participation and whole-hand shape from native audit."""
from __future__ import annotations
import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sliding_audit import _native_context
from .pad_contact_visual import camera_from_direction, marker


def verified_data(case):
    case=Path(case).resolve()
    audit=json.loads((case/'independent_native_audit/audit.json').read_text())
    assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
    for name,digest in audit['source_files'].items():
        assert sha256((case/name).read_bytes()).hexdigest()==digest
    summary=json.loads((case/'summary.json').read_text());arrays={}
    keys=('time_s','q_rad','forces_n','belly_forces_n','safe_belly','belly_contact_positions_world_m','phase','stroke_distance_m')
    for chunk in summary['trace_chunks']:
        source=case/chunk['file'];assert sha256(source.read_bytes()).hexdigest()==chunk['sha256']
        with np.load(source) as z:
            count=len(z['time_s'])-int(chunk.get('skip_last',0))
            for key in keys:arrays.setdefault(key,[]).append(z[key][:count])
    data={key:np.concatenate(value) for key,value in arrays.items()}
    assert len(data['time_s'])==audit['native_states']
    return case,audit,summary,data


def metrics(case):
    case,audit,summary,data=verified_data(case)
    begin=summary.get('cooperative_slide_start_s')
    mask=data['time_s']>=begin if begin is not None else np.zeros(len(data['time_s']),bool)
    count=data['safe_belly'].sum(axis=1);qdeg=np.degrees(data['q_rad'][:,7:]).reshape(-1,4,4)
    manifest=json.loads((case/'manifest.json').read_text())
    completion_check=None
    if summary['cooperative_local_complete']:
        assert summary['stroke_completed_m']>=manifest['stroke_m']-1e-10
        if 'cooperative_window_s' in manifest:
            window=data['safe_belly'][-150:]
            assert window[-1].all() and np.mean(window.sum(axis=1)>=3)>=.9 and np.all(window.mean(axis=0)>=.6)
            completion_check=dict(last_observed_four_contacts=True,window_s=.3,
                three_or_more_fraction=float(np.mean(window.sum(axis=1)>=3)),each_finger_fraction=window.mean(axis=0).tolist())
    result=dict(scope='Local cooperative contact and sliding; not whole-object completion',
        local_complete=summary['cooperative_local_complete'],native_audit_passed=True,
        slide_start_s=begin,slide_duration_s=float(mask.sum()*.002),
        contact_count_fraction_during_slide=(np.bincount(count[mask],minlength=5)/max(1,mask.sum())).tolist(),
        finger_contact_fraction_during_slide=data['safe_belly'][mask].mean(axis=0).tolist() if mask.any() else [0.]*4,
        at_least_three_fraction_during_slide=float(np.mean(count[mask]>=3)) if mask.any() else 0.,
        four_fraction_during_slide=float(np.mean(count[mask]==4)) if mask.any() else 0.,
        actual_joint_min_deg=qdeg.min(axis=0).tolist(),actual_joint_max_deg=qdeg.max(axis=0).tolist(),
        middle_motion_range_deg=np.ptp(qdeg[mask,1],axis=0).tolist() if mask.any() else [0.]*4,
        measured_tangential_material_motion_m=audit['actual_tangential_pad_material_motion_m'],
        peak_raw_force_n=max(audit['raw_force_peak_n']),shared_reference_progress_m=summary['stroke_completed_m'],
        measured_contact_displacement_along_world_y_m=[],completion_window_independently_checked=completion_check,
        actual_joint_slide_min_deg=qdeg[mask].min(axis=0).tolist() if mask.any() else None,
        actual_joint_slide_max_deg=qdeg[mask].max(axis=0).tolist() if mask.any() else None)
    for finger in range(4):
        indices=np.flatnonzero(mask&data['safe_belly'][:,finger])
        net=float(data['belly_contact_positions_world_m'][indices[-1],finger,1]-
            data['belly_contact_positions_world_m'][indices[0],finger,1]) if len(indices)>1 else None
        result['measured_contact_displacement_along_world_y_m'].append(net)
    (case/'cooperative_metrics.json').write_text(json.dumps(result,indent=2))
    t=data['time_s'];fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True)
    labels=('Index','Middle','Ring','Thumb');colors=('#357ba5','#3a9662','#d49430','#b65772')
    axes[0].step(t[::10],count[::10],where='post',color='#233b61');axes[0].set_ylim(-.2,4.2)
    axes[0].set_yticks(range(5));axes[0].set_ylabel('Simultaneous safe contacts')
    for finger in range(4):axes[1].plot(t,data['forces_n'][:,finger],label=labels[finger],color=colors[finger],lw=.8)
    axes[1].set_ylabel('Raw contact force (N)');axes[1].legend(ncol=4,loc='upper right')
    for joint,label in enumerate(('Abduction','MCP','PIP','DIP')):
        axes[2].plot(t[::10],qdeg[::10,1,joint],label=label,lw=1.3)
    axes[2].set_ylabel('Actual middle finger (deg)');axes[2].legend(ncol=4)
    for ax in axes:
        ax.grid(alpha=.2)
        if begin is not None:ax.axvline(begin,color='#888',ls='--',lw=1.)
    axes[2].set_xlabel('Actual simulation time (s)')
    fig.suptitle('Whole-hand contact participation, force sharing and actual middle-finger motion')
    fig.tight_layout();fig.savefig(case/'cooperative_metrics.png',dpi=150);plt.close(fig)
    return result


def render(case,stills=False):
    case,audit,summary,data=verified_data(case)
    context=_native_context(case);m,d=context.model,context.data
    pads=context.handles.tip_geom_ids;rgba=m.geom_rgba.copy();matid=m.geom_matid.copy()
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
    options=mujoco.MjvOption();options.geomgroup[3:]=0;options.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    renderer=mujoco.Renderer(m,height=540,width=800)
    m.vis.headlight.ambient[:]=[.25,.25,.25]
    m.vis.headlight.diffuse[:]=[.45,.45,.45]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    title=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',24)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    all_four=np.flatnonzero(data['safe_belly'].all(axis=1))
    selected=np.unique(np.r_[0,all_four[len(all_four)//2] if len(all_four) else 0,
        np.linspace(0,len(data['time_s'])-1,4).astype(int)]) if stills else np.unique(np.r_[np.arange(0,len(data['time_s']),25),len(data['time_s'])-1])
    prefix='cooperative_stills' if stills else 'cooperative_full';video=case/'cooperative_full_1x.mp4'
    writer=None if stills else imageio.get_writer(video,fps=20,codec='libx264',quality=7,macro_block_size=2,output_params=['-threads','2'])
    captures=[]
    for frame,k in enumerate(selected):
        d.qpos[context.qpos]=data['q_rad'][k];mujoco.mj_forward(m,d)
        centers=d.geom_xpos[pads].copy();normal=np.mean(d.geom_xmat[pads].reshape(4,3,3)[:3,:,2],axis=0)
        normal/=np.linalg.norm(normal);middle_frame=d.geom_xmat[pads[1]].reshape(3,3)
        panels=[]
        for panel in range(3):
            m.geom_rgba[:]=rgba;m.geom_matid[:]=matid
            for geom,name in enumerate(names):
                if 'normal_marker' in name:m.geom_rgba[geom,3]=0.
                if name=='mustard_visual':m.geom_matid[geom]=-1;m.geom_rgba[geom]=[.96,.72,.08,.12 if panel else 1.]
            if panel==2:
                thumb_frame=d.geom_xmat[pads[3]].reshape(3,3)
                camera_from_direction(camera,thumb_frame[:,2]+.2*thumb_frame[:,1],centers[3],.27)
            elif panel:camera_from_direction(camera,normal+.65*middle_frame[:,1],centers.mean(axis=0),.42)
            else:camera.lookat[:]=[.3,0.,.49];camera.distance=.95;camera.azimuth=130.;camera.elevation=-25.
            renderer.update_scene(d,camera=camera,scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=not bool(panel)
            for finger in np.flatnonzero(data['safe_belly'][k]):marker(renderer.scene,data['belly_contact_positions_world_m'][k,finger],.002,[1.,.1,.55,1.])
            image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
            draw.rectangle((0,0,800,70),fill='#122238')
            draw.text((12,7),('整手协同贴合 · 原生实际运动','长指与整手姿态 · 仅物体透明','拇指对掌贴合 · 原生实际接触')[panel],font=title,fill='white')
            draw.text((12,42),f'实际 t={data["time_s"][k]:.2f}s · 安全指腹接触 {int(data["safe_belly"][k].sum())}/4',font=font,fill='#a7dbf0')
            panels.append(image)
        canvas=Image.new('RGB',(2400,700),'#122238')
        for panel,image in enumerate(panels):canvas.paste(image,(800*panel,0))
        draw=ImageDraw.Draw(canvas)
        phase={'COOPERATIVE_APPROACH':'四指分别主动接近','COOPERATIVE_SLIDE':'整手共同贴面滑动','FORCE_RELEASE':'独立卸力、其余手指继续贴合','CLEARANCE_EXIT':'调整整手避让近节'}.get(str(data['phase'][k]),str(data['phase'][k]))
        draw.text((15,548),phase+'  |  正常速度 1×  |  原生连续重放已核验',font=font,fill='white')
        for finger,label in enumerate(('食指','中指','无名指','拇指')):
            q=np.degrees(data['q_rad'][k,7+4*finger:11+4*finger]);x=15+1190*(finger%2);y=582+32*(finger//2)
            text=f'{label} 关节° [{q[0]:5.1f}, {q[1]:5.1f}, {q[2]:5.1f}, {q[3]:5.1f}]  {data["forces_n"][k,finger]:.2f} N'
            text+='  指腹贴合' if data['safe_belly'][k,finger] else '  正在调整'
            draw.text((x,y),text,font=small,fill='#a7edba' if data['safe_belly'][k,finger] else '#d7e1ea')
        draw.text((15,660),'本段为局部整手滑动试验，非整瓶完成结果。粉点均为实际安全指腹接触；四指目标共同求解。',font=small,fill='#f3d38d')
        if writer:writer.append_data(np.asarray(canvas))
        if stills or frame in {0,len(selected)//3,2*len(selected)//3,len(selected)-1}:
            path=case/f'{prefix}_{frame:04d}.png';canvas.save(path);captures.append(dict(file=str(path),time_s=float(data['time_s'][k])))
        if frame%200==0:print('RENDER',frame,len(selected),flush=True)
    if writer:writer.close()
    renderer.close();source=case/'visualization_source';source.mkdir(exist_ok=True)
    (source/'whole_hand_review.py').write_bytes(Path(__file__).read_bytes())
    meta=dict(actual_saved_q_only=True,no_physics_step=True,no_geometry_change=True,full_local_time_axis=not stills,
        fresh_single_version=audit['fresh_single_version'],fps=20,speed=1.,frames=len(selected),stills=captures,
        source_manifest_sha256=audit['source_files']['manifest.json'])
    if writer:meta.update(video=str(video),video_sha256=sha256(video.read_bytes()).hexdigest())
    (case/(prefix+'_render.json')).write_text(json.dumps(meta,indent=2))
    return meta


def report(case):
    case,audit,summary,data=verified_data(case);result=metrics(case)
    render_meta=json.loads((case/'cooperative_full_render.json').read_text())
    video=Path(render_meta['video']);assert sha256(video.read_bytes()).hexdigest()==render_meta['video_sha256']
    assert render_meta['source_manifest_sha256']==audit['source_files']['manifest.json']
    rows=''
    for i,name in enumerate(('食指','中指','无名指','拇指')):
        net=result['measured_contact_displacement_along_world_y_m'][i]
        net_text=f'{net*1000:.1f} mm' if net is not None else '未形成有效滑动'
        rows+=f'<tr><td>{name}</td><td>{100*result["finger_contact_fraction_during_slide"][i]:.1f}%</td><td>{net_text}</td><td>{1000*result["measured_tangential_material_motion_m"][i]:.1f} mm</td></tr>'
    lo=result['actual_joint_slide_min_deg'];hi=result['actual_joint_slide_max_deg']
    joints=''
    if lo:
        for j,name in enumerate(('侧摆','MCP屈伸','PIP屈伸','DIP屈伸')):
            joints+=f'<tr><td>{name}</td><td>{lo[1][j]:.1f}° — {hi[1][j]:.1f}°</td><td>{hi[1][j]-lo[1][j]:.1f}°</td></tr>'
    images=''.join(f'<img src="{html.escape(Path(item["file"]).name)}" alt="实际整手姿态 t={item["time_s"]:.2f}s">' for item in render_meta['stills'][1:])
    status='已完成局部整手滑动' if result['local_complete'] else '局部整手试验未完成'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mustard · 整手协同与自然屈指</title><style>body{{margin:0;background:#eef2f6;color:#17283b;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1420px;margin:auto;padding:30px}}.card{{background:white;padding:24px;border-radius:14px;margin:22px 0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}}strong{{display:block;font-size:30px}}video,img{{width:100%;border-radius:9px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dbe3ec}}.note{{color:#526779;font-size:15px}}a{{color:#1768a8}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}main{{padding:12px}}}}</style>
<main><h1>Mustard · 整手协同与自然屈指</h1><p>{status}。本轮重点是四指主动参与、共同贴合，以及中指屈伸姿态。</p>
<p>腕部与四指共同求解；四指各自调整法向接触力和附近落点。三根长指采用弯曲先验及PIP/DIP软协同，拇指主动对掌。接触人数只统计真实、安全、符合掌侧指腹判据的接触。</p>
<div class="card metrics"><div>滑动阶段 ≥3指同时接触<strong>{100*result['at_least_three_fraction_during_slide']:.1f}%</strong></div><div>滑动阶段 4指同时接触<strong>{100*result['four_fraction_during_slide']:.1f}%</strong></div><div>原始接触力峰<strong>{result['peak_raw_force_n']:.3f} N</strong></div><div>共同扫描参考前进<strong>{1000*result['shared_reference_progress_m']:.0f} mm</strong></div></div>
<section class="card"><h2>完整局部过程 · 正常速度</h2><video controls preload="metadata" src="{html.escape(video.name)}"></video><p>保留从初始悬停到结束的完整 {summary['executed_s']:.3f} 秒，1倍速、无时间剪接。中间看三根长指与手形，右侧单独看拇指对掌；只将物体透明，手部保持实心。粉点是合格的实际指腹接触。</p><p class="note">这是新控制器的局部整手试验，整瓶遍历尚未重新验证。没有继承旧单指版本的整瓶覆盖。</p></section>
<section class="card"><h2>四指是否真正参与</h2><table><tr><th>手指</th><th>滑动阶段安全指腹接触率</th><th>首末接触点沿扫描方向净位移</th><th>原生核验的切向材料运动</th></tr>{rows}</table><p class="note">接触率分母从实际进入共同滑动开始，包含其间卸力和调整。净位移与累计材料运动分别列出，参考路径长度不冒充实际触面距离。</p></section>
<section class="card"><h2>中指如何随表面调整</h2><table><tr><th>关节</th><th>滑动阶段实际范围</th><th>实际变化幅度</th></tr>{joints}</table><p>中指持续参与接触目标求解，PIP与DIP保持正向弯曲，并允许按表面曲率调整；不会被拉回旧版固定伸直等待姿态。</p>{images}</section>
<section class="card"><h2>接触人数、力分配与实际手指运动</h2><img src="cooperative_metrics.png" alt="实际多指接触人数与中指运动"><p class="note">力曲线使用全部2ms原生观测。竖虚线是开始共同滑动的时刻。</p></section>
<section class="card"><h2>验证范围与证据</h2><p>{audit['native_states']}个原生状态从唯一初态连续重放通过；分块不重置，原始力、指腹分类和覆盖均独立重算。所有实际命令再次通过原严格几何与参考守护。</p><p>共同滑动的交接允许短暂个别失联：300ms内至少三指接触占90%以上、每根手指参与60%以上，并在当前四指同时贴合时交接。不会把三指一直接触、第四指一直不动判成整手就绪。</p><p>该结果仍有个别手指接触间断；不宣称四指始终贴合、一般多物体性能或整瓶验证完成。DP继续按要求暂缓。</p><p><a href="cooperative_metrics.json">整手指标</a> · <a href="independent_native_audit/audit.json">独立原生核验</a> · <a href="summary.json">运行结果</a> · <a href="manifest.json">控制器与模型清单</a> · <a href="events.json">实际事件</a> · <a href="trace_chunks.json">2ms原始轨迹</a></p></section></main></html>'''
    (case/'review.html').write_text(page,encoding='utf-8')
    return case/'review.html'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case');parser.add_argument('--stills',action='store_true')
    args=parser.parse_args();metrics(args.case);render(args.case,args.stills)
    if not args.stills:print(report(args.case))
