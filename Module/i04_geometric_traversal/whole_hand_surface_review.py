"""Audited full-time-axis whole-hand coverage and posture review."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sliding_audit import _native_context
from .pad_contact_visual import camera_from_direction,marker
from .natural_belly_report import coverage_figure,contact_path_figure,verified_force_excursions,timeline_figure


def controller_changes(audit):
    result=[];resume=audit.get('resume')
    while resume:
        result.append(float(resume['join_time_s']));resume=resume.get('parent_resume')
    return sorted(result)


def load_verified(case):
    case=Path(case).resolve();audit=json.loads((case/'independent_native_audit/audit.json').read_text())
    assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
    for name,digest in audit['source_files'].items():assert sha256((case/name).read_bytes()).hexdigest()==digest
    summary=json.loads((case/'summary.json').read_text());arrays={};phase_counts={};phase_contacts={};phase_fingers={}
    total=0;contact_hist=np.zeros(5,int);contact_counts=np.zeros(4,int);lo=np.full((4,4),np.inf);hi=-lo
    gap=np.zeros(5,int);maxgap=np.zeros(5,int);all_tasks=True;phase_bins=[]
    keys=('time_s','q_rad','forces_n','safe_belly','belly_contact_positions_world_m','belly_normals_world','phase','coverage')
    for chunk in summary['trace_chunks']:
        path=case/chunk['file'];assert sha256(path.read_bytes()).hexdigest()==chunk['sha256']
        with np.load(path) as z:
            end=len(z['time_s'])-int(chunk.get('skip_last',0));ids=np.arange((-total)%25,end,25)
            safe=z['safe_belly'][:end];count=safe.sum(axis=1)
            contact_hist+=np.bincount(count,minlength=5);contact_counts+=safe.sum(axis=0)
            all_tasks=all_tasks and bool(z['controlled_fingers'][:end].all())
            q=np.degrees(z['q_rad'][:end,7:]).reshape(-1,4,4);lo=np.minimum(lo,q.min(axis=0));hi=np.maximum(hi,q.max(axis=0))
            for phase in np.unique(z['phase'][:end]):
                mask=z['phase'][:end]==phase
                phase_counts[str(phase)]=phase_counts.get(str(phase),0)+int(mask.sum())
                phase_contacts.setdefault(str(phase),np.zeros(5,int));phase_contacts[str(phase)]+=np.bincount(count[mask],minlength=5)
                phase_fingers.setdefault(str(phase),np.zeros(4,int));phase_fingers[str(phase)]+=safe[mask].sum(axis=0)
            live=np.column_stack((safe,safe.any(axis=1)))
            for finger in range(5):
                missing=~live[:,finger];edges=np.diff(np.r_[False,missing,False].astype(int))
                starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
                if len(starts):
                    lengths=ends-starts
                    if starts[0]==0:lengths[0]+=gap[finger]
                    maxgap[finger]=max(maxgap[finger],int(lengths.max()))
                    gap[finger]=int(lengths[-1]) if ends[-1]==end else 0
                else:gap[finger]=0
            for key in keys:arrays.setdefault(key,[]).append(z[key][ids].copy())
            total+=end
    assert total==audit['native_states'] and all_tasks
    data={k:np.concatenate(v) for k,v in arrays.items()}
    # Include the final actual observation for the rendered endpoint.
    with np.load(case/summary['trace_chunks'][-1]['file']) as z:
        last=len(z['time_s'])-1-int(summary['trace_chunks'][-1].get('skip_last',0))
        if data['time_s'][-1]<z['time_s'][last]-1e-9:
            for key in keys:data[key]=np.concatenate((data[key],z[key][last:last+1]))
    result=dict(native_states=total,full_surface_complete=audit['full_surface_complete'],
        fresh_single_version=audit['fresh_single_version'],resume=audit.get('resume'),
        all_four_control_flags=True,all_time_contact_count_fraction=(contact_hist/total).tolist(),
        all_time_finger_contact_fraction=(contact_counts/total).tolist(),longest_safe_gap_s=(maxgap*.002).tolist(),
        actual_joint_min_deg=lo.tolist(),actual_joint_max_deg=hi.tolist(),
        actual_tangential_material_motion_m=audit['actual_tangential_pad_material_motion_m'],
        phases={phase:dict(duration_s=count*.002,contact_count_fraction=(phase_contacts[phase]/count).tolist(),
            finger_contact_fraction=(phase_fingers[phase]/count).tolist()) for phase,count in phase_counts.items()})
    (case/'whole_hand_metrics.json').write_text(json.dumps(result,indent=2))
    return case,audit,summary,data,result


def region_samples(data,origin):
    names=('宽面 +Z','侧面 +X','宽面 −Z','侧面 −X','瓶盖／瓶肩端 −Y','瓶底端 +Y')
    samples=[]
    for label,axis,sign in zip(names,(2,0,2,0,1,1),(1,1,-1,-1,-1,1)):
        live=data['safe_belly']&(sign*data['belly_normals_world'][:,:,axis]>.65)
        if axis==1:
            local_y=data['belly_contact_positions_world_m'][:,:,1]-origin[1]
            live&=(sign*local_y>.20)
        regional=live.any(axis=1);count=data['safe_belly'].sum(axis=1)
        # Prefer a readable cooperative example in the clearly labelled detail
        # excerpts. The complete video and statistics still include all states.
        quorum=next((q for q in (3,2,1) if np.any(regional&(count>=q))),1)
        rows=np.flatnonzero(regional&(count>=quorum))
        if not len(rows):samples.append(dict(region=label,available=False));continue
        breaks=np.r_[0,np.flatnonzero(np.diff(data['time_s'][rows])>.051)+1,len(rows)]
        a,b=breaks[:-1],breaks[1:];best=int(np.argmax(b-a));k=int(rows[(a[best]+b[best]-1)//2])
        samples.append(dict(region=label,available=True,index=k,time_s=float(data['time_s'][k]),
            selected_minimum_simultaneous_contacts=quorum))
    return samples


def render(case,speed=12.,stills=False,details=False):
    case,audit,summary,data,metrics=load_verified(case)
    c=_native_context(case);m,d=c.model,c.data;pads=c.handles.tip_geom_ids
    options=mujoco.MjvOption();options.geomgroup[3:]=0;options.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    renderer=mujoco.Renderer(m,height=540,width=800)
    m.vis.headlight.ambient[:]=[.25,.25,.25];m.vis.headlight.diffuse[:]=[.45,.45,.45]
    rgba=m.geom_rgba.copy();matid=m.geom_matid.copy()
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or '' for g in range(m.ngeom)]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    title=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',24)
    body=ET.fromstring((case/'model.xml').read_text()).find(".//body[@name='fr3_e05_object']")
    regions=region_samples(data,np.fromstring(body.get('pos'),sep=' '));segments=[];frame_regions=[]
    if stills:
        selected=sorted(set([0,len(data['time_s'])-1,*[r['index'] for r in regions if r['available']]]))
        prefix='whole_hand_regions';writer=None
    elif details:
        speed=1.;selected=[];prefix='whole_hand_details_1x'
        for region in regions:
            if not region['available']:continue
            begin=max(0.,region['time_s']-2.);end=min(summary['executed_s'],region['time_s']+2.)
            ids=np.flatnonzero((data['time_s']>=begin)&(data['time_s']<=end));selected.extend(ids.tolist())
            frame_regions.extend([region['region']]*len(ids))
            segments.append(dict(region=region['region'],start_s=begin,end_s=end))
        writer=imageio.get_writer(case/(prefix+'.mp4'),fps=20,codec='libx264',quality=7,macro_block_size=2,output_params=['-threads','2'])
    else:
        requested=np.arange(0.,summary['executed_s'],speed/20)
        selected=np.clip(np.searchsorted(data['time_s'],requested),0,len(data['time_s'])-1).tolist()
        if selected[-1]!=len(data['time_s'])-1:selected.append(len(data['time_s'])-1)
        prefix=f'whole_hand_complete_{speed:g}x'
        writer=imageio.get_writer(case/(prefix+'.mp4'),fps=20,codec='libx264',quality=7,macro_block_size=2,output_params=['-threads','2'])
    captures=[]
    for frame,k in enumerate(selected):
        d.qpos[c.qpos]=data['q_rad'][k];mujoco.mj_forward(m,d)
        centers=d.geom_xpos[pads].copy();frames=d.geom_xmat[pads].reshape(4,3,3)
        normal=frames[:3,:,2].mean(axis=0);normal/=np.linalg.norm(normal);panels=[]
        for panel in range(3):
            m.geom_rgba[:]=rgba;m.geom_matid[:]=matid
            for geom,name in enumerate(names):
                if 'normal_marker' in name:m.geom_rgba[geom,3]=0.
                if name=='mustard_visual':m.geom_matid[geom]=-1;m.geom_rgba[geom]=[.96,.72,.08,.12 if panel else 1.]
            if panel==2:camera_from_direction(camera,frames[3,:,2]+.2*frames[3,:,1],centers[3],.27)
            elif panel:camera_from_direction(camera,normal+.65*frames[1,:,1],centers.mean(axis=0),.44)
            else:camera.lookat[:]=[.3,0.,.49];camera.distance=1.05;camera.azimuth=130.;camera.elevation=-25.
            renderer.update_scene(d,camera=camera,scene_option=options);renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=not bool(panel)
            for finger in np.flatnonzero(data['safe_belly'][k]):marker(renderer.scene,data['belly_contact_positions_world_m'][k,finger],.002,[1.,.1,.55,1.])
            picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
            draw.rectangle((0,0,800,70),fill='#122238')
            first_title=(frame_regions[frame]+' · 原生实际运动') if details else 'Mustard 整手遍历 · 原生实际运动'
            draw.text((12,7),(first_title,'长指协同屈伸 · 仅物体透明','拇指主动对掌 · 原生实际接触')[panel],font=title,fill='white')
            draw.text((12,42),f'实际 t={data["time_s"][k]:.2f}s · 安全指腹 {int(data["safe_belly"][k].sum())}/4',font=font,fill='#a7dbf0')
            panels.append(picture)
        canvas=Image.new('RGB',(2400,700),'#122238')
        for panel,picture in enumerate(panels):canvas.paste(picture,(800*panel,0))
        draw=ImageDraw.Draw(canvas)
        labels={'COOPERATIVE_APPROACH':'四指主动接近','COOPERATIVE_SLIDE':'整手连续贴面滑动','FORCE_RELEASE':'主动卸力',
            'COOPERATIVE_CONTACT_PIVOT':'保持接触调整方位','COOPERATIVE_WITHDRAWAL':'必要撤离：四指协调让位',
            'NECESSARY_REORIENTATION':'必要离面调姿','COOPERATIVE_RECONTACT':'整手主动重接触','CLEARANCE_EXIT':'调整姿态保持间隙',
            'FINAL_SAFE_HOLD':'覆盖完成后的实际接触保持'}
        phase=labels.get(str(data['phase'][k]),str(data['phase'][k]))
        draw.text((15,548),f'{phase}  |  回放 {speed:g}×  |  实测测地覆盖 {100*data["coverage"][k]:.2f}%  |  原生连续重放已核验',font=font,fill='white')
        for i,name in enumerate(('食指','中指','无名指','拇指')):
            q=np.degrees(data['q_rad'][k,7+4*i:11+4*i]);x=15+1190*(i%2);y=582+32*(i//2)
            text=f'{name} 关节° [{q[0]:5.1f}, {q[1]:5.1f}, {q[2]:5.1f}, {q[3]:5.1f}]  {data["forces_n"][k,i]:.2f} N'
            text+='  指腹贴合' if data['safe_belly'][k,i] else '  调整／转移中'
            draw.text((x,y),text,font=small,fill='#a7edba' if data['safe_belly'][k,i] else '#d7e1ea')
        scope=('本轮整瓶覆盖已独立核验。' if audit['full_surface_complete'] else '本轮尚未完成整瓶覆盖。')
        scope+=(' 此为六方向正常速度节选，区域之间有时间跳转。' if details else ' 本视频保留完整实际时间轴，含必要离面与调整。')
        if audit.get('resume'):scope+=' 开发续接轨迹，控制器切换t='+'、'.join(f'{t:.0f}' for t in controller_changes(audit))+'s。'
        draw.text((15,660),scope+' 粉点仅表示实际安全指腹接触。',font=small,fill='#f3d38d')
        if writer:writer.append_data(np.asarray(canvas))
        if stills or frame in {0,len(selected)//3,2*len(selected)//3,len(selected)-1}:
            path=case/f'{prefix}_{frame:04d}.png';canvas.save(path);captures.append(dict(file=path.name,time_s=float(data['time_s'][k])))
        if frame%250==0:print('RENDER',prefix,frame,len(selected),flush=True)
    if writer:writer.close()
    renderer.close();source=case/'visualization_source';source.mkdir(exist_ok=True)
    (source/'whole_hand_surface_review.py').write_bytes(Path(__file__).read_bytes())
    meta=dict(actual_saved_q_only=True,no_physics_step=True,full_time_axis=not(details or stills),speed=speed,fps=20,
        frames=len(selected),stills=captures,segments=segments,regions=regions,
        source_manifest_sha256=audit['source_files']['manifest.json'],whole_object_complete=audit['full_surface_complete'])
    if writer:
        video=case/(prefix+'.mp4');meta.update(video=video.name,video_sha256=sha256(video.read_bytes()).hexdigest())
    (case/(prefix+'_render.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


def report(case,speed=12.):
    case,audit,summary,data,metrics=load_verified(case)
    covered=coverage_figure(case);contact_path_figure(case);timeline_figure(case,audit,summary)
    excursions=verified_force_excursions(case,audit,summary)
    force_note=(f'原始力达到8N以上累计{excursions["total_s"]:.3f}s，最长单段{excursions["longest_s"]:.3f}s；这些状态全部排除覆盖计分，并保留在完整回放中。'
        if excursions['episodes'] else '全程没有原始力达到8N的状态。')
    videos=[]
    for name in (f'whole_hand_complete_{speed:g}x','whole_hand_details_1x'):
        path=case/(name+'_render.json')
        if not path.exists():continue
        meta=json.loads(path.read_text(encoding='utf-8'));assert sha256((case/meta['video']).read_bytes()).hexdigest()==meta['video_sha256']
        title='完整实际时间轴' if meta['full_time_axis'] else '六方向正常速度节选（有时间跳转）'
        videos.append(f'<section><h2>{title} · {meta["speed"]:g}倍速</h2><video controls preload="metadata" src="{meta["video"]}"></video></section>')
    regions=''.join(f'<tr><td>{r["region"]}</td><td>{r["covered"]}/{r["vertices"]}</td><td>{100*r["area_fraction"]:.2f}%</td></tr>' for r in covered['regions'])
    contacts=''
    for i,name in enumerate(('食指','中指','无名指','拇指')):
        contacts+=f'<tr><td>{name}</td><td>{100*metrics["all_time_finger_contact_fraction"][i]:.1f}%</td><td>{metrics["actual_tangential_material_motion_m"][i]:.3f}m</td><td>{metrics["actual_joint_min_deg"][i][2]:.1f}°–{metrics["actual_joint_max_deg"][i][2]:.1f}°</td><td>{metrics["actual_joint_min_deg"][i][3]:.1f}°–{metrics["actual_joint_max_deg"][i][3]:.1f}°</td></tr>'
    phases=''
    phase_names={'COOPERATIVE_APPROACH':'四指主动接近','COOPERATIVE_SLIDE':'整手贴面滑动','FORCE_RELEASE':'主动卸力',
        'COOPERATIVE_CONTACT_PIVOT':'贴面调整方位','COOPERATIVE_WITHDRAWAL':'必要撤离','NECESSARY_REORIENTATION':'必要离面调姿',
        'COOPERATIVE_RECONTACT':'主动重新接触','CLEARANCE_EXIT':'间隙调整','FINAL_SAFE_HOLD':'最终接触保持'}
    for phase,values in metrics['phases'].items():
        fractions=values['contact_count_fraction']
        phases+=f'<tr><td>{phase_names.get(phase,phase)}</td><td>{values["duration_s"]:.3f}s</td><td>{100*sum(fractions[3:]):.1f}%</td><td>{100*fractions[4]:.1f}%</td></tr>'
    status='整瓶实测覆盖完成' if audit['full_surface_complete'] else '尚未完成整瓶实测覆盖'
    lineage='从唯一悬停初态、零覆盖开始，采用冻结单版本控制器。'
    if audit.get('resume'):
        lineage=('保留原始零覆盖初态及完整接触记录，在'+'、'.join(f'{t:.3f}' for t in controller_changes(audit))+'s更新控制器；'
                 '物理状态准确接续，整条轨迹从头连续独立核验。这是开发续接结果，未拼合独立轨迹，不作为单版本性能对比。')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mustard · 四指协同整瓶遍历</title><style>body{{margin:0;background:#eef2f6;color:#17283b;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1420px;margin:auto;padding:30px}}section{{background:white;padding:24px;border-radius:14px;margin:22px 0}}video,img{{width:100%;border-radius:9px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #dbe3ec}}.note{{color:#526779;font-size:15px}}a{{color:#1768a8}}@media(max-width:700px){{main{{padding:12px}}section{{padding:12px}}}}</style>
<main><h1>Mustard · 四指协同整瓶遍历</h1><p><b>{status}</b>：{100*audit['area_fraction']:.3f}%，{audit['covered_vertices']}/8192顶点、{audit['covered_goals']}/388目标；实际运行{summary['executed_s']:.3f}s。</p>
<p>腕部与四指共同求解，每根手指主动寻找掌侧指腹接触。沿表面优先连续滑动和转向，必要时才撤离调姿。</p><p class="note">{lineage}</p>
{''.join(videos)}
<section><h2>整个物体是否真正覆盖</h2><img src="coverage_verified.png" alt="六方向独立核验的覆盖与缺口"><table><tr><th>区域</th><th>已覆盖顶点</th><th>面积指标</th></tr>{regions}</table><p class="note">绿色为原生安全指腹接触的25mm网格边测地邻域，灰色为未覆盖；不是皮肤压力足迹面积。原始网格分母保持，不用规划点授予覆盖。</p><img src="contact_paths_verified.png" alt="六方向实际指腹接触点"></section>
<section><h2>四指参与和自然屈伸</h2><table><tr><th>手指</th><th>完整时间轴接触率</th><th>合格接触时切向材料运动</th><th>实际PIP范围</th><th>实际DIP范围</th></tr>{contacts}</table><p class="note">完整时间轴包含初始接近、必要离面和恢复。各指存在短暂失联，四指任务持续参与，不能把四指率解释成始终四指同时接触。</p><table><tr><th>阶段</th><th>时长</th><th>≥3指同时接触</th><th>4指同时接触</th></tr>{phases}</table></section>
<section><h2>完整轨迹、力与接触间断</h2><img src="timeline_verified.png" alt="全程实测覆盖、力峰和四指接触"><p>换姿计划{summary['necessary_lifts']}次（含中断后的重新规划）；最长全手安全指腹接触中断{metrics['longest_safe_gap_s'][-1]:.3f}s；原始力峰{max(audit['raw_force_peak_n']):.4f}N。</p><p class="note">{force_note}</p></section>
<section><h2>原生验证</h2><p>{audit['native_states']}个2ms状态从唯一初态连续独立重放通过，分块不重置；已执行命令全部通过原严格几何与上一参考守护。四指接触、力和覆盖重新测量。当前仅验证固定Mustard、零重力、MCC条件，DP继续暂缓。</p><p>终止原因：{summary['stop_reason']}。旧单领触整瓶和旧局部试验均未继承到本轮覆盖。</p><p><a href="whole_hand_metrics.json">整手指标</a> · <a href="independent_native_audit/audit.json">独立原生核验</a> · <a href="coverage_verified.json">分区覆盖</a> · <a href="summary.json">运行结果</a> · <a href="manifest.json">冻结清单</a> · <a href="events.json">路线与换面事件</a></p></section></main></html>'''
    (case/'review.html').write_text(page,encoding='utf-8');return case/'review.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--speed',type=float,default=12.)
    p.add_argument('--stills',action='store_true');p.add_argument('--details',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    if not a.report_only:render(a.case,a.speed,a.stills,a.details)
    if not a.stills:print(report(a.case,a.speed))
