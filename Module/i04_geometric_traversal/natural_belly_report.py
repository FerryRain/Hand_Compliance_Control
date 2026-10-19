"""Audited whole-surface coverage, posture and contact report."""
from __future__ import annotations
import argparse
from hashlib import sha256
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .full_review import task_mesh


def verified_force_excursions(case,audit,summary):
    """Summarize high-force observations from the replay-verified raw log."""
    count=episodes=longest=tail=states=applied_high=0;peak=0.;previous_high=False
    hard_count=hard_longest=hard_tail=0;peak_time=0.
    for chunk in summary['trace_chunks']:
        source=case/chunk['file'];assert sha256(source.read_bytes()).hexdigest()==chunk['sha256']
        with np.load(source) as z:
            end=len(z['time_s'])-int(chunk.get('skip_last',0));forces=z['forces_n'][:end]
            raw_peak=forces.max(axis=1);high=raw_peak>=8.;states+=end;count+=int(high.sum())
            if float(raw_peak.max())>peak:
                peak=float(raw_peak.max());peak_time=float(z['time_s'][int(raw_peak.argmax())])
            assert not z['safe_belly'][:end][high].any()
            episodes+=int(np.count_nonzero(high&~np.r_[previous_high,high[:-1]]));previous_high=bool(high[-1])
            # A terminal observation is evidence of a peak, but has no
            # subsequent 2 ms interval when execution stopped at that state.
            live=high&(z['time_s'][:end]<summary['executed_s']-.001);applied_high+=int(live.sum())
            edges=np.diff(np.r_[False,live,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            if len(starts):
                lengths=ends-starts
                if starts[0]==0:lengths[0]+=tail
                longest=max(longest,int(lengths.max()))
                tail=int(lengths[-1]) if ends[-1]==len(high) else 0
            else:tail=0
            hard=(raw_peak>12.)&(z['time_s'][:end]<summary['executed_s']-.001)
            hard_count+=int(hard.sum())
            edges=np.diff(np.r_[False,hard,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
            if len(starts):
                lengths=ends-starts
                if starts[0]==0:lengths[0]+=hard_tail
                hard_longest=max(hard_longest,int(lengths.max()))
                hard_tail=int(lengths[-1]) if ends[-1]==len(hard) else 0
            else:hard_tail=0
    assert states==audit['native_states']
    assert abs(peak-max(audit['raw_force_peak_n']))<1e-9
    result=dict(threshold_n=8.,episodes=episodes,observations=count,total_s=applied_high*.002,longest_s=longest*.002,
        high_force_observations_grant_no_coverage=True,peak_n=peak,peak_time_s=peak_time,
        hard_stop_threshold_n=12.,hard_stop_sustained_s=.020,
        above_hard_threshold_total_s=hard_count*.002,above_hard_threshold_longest_s=hard_longest*.002)
    (case/'force_excursions_verified.json').write_text(json.dumps(result,indent=2))
    return result


def timeline_figure(case,audit,summary):
    """100 ms bins preserve every native force peak, including brief spikes."""
    width=50;bins=(audit['native_states']+width-1)//width
    force=np.zeros(bins);coverage=np.zeros(bins);safe=np.zeros((bins,4));counts=np.zeros(bins);offset=0
    for chunk in summary['trace_chunks']:
        source=case/chunk['file'];assert sha256(source.read_bytes()).hexdigest()==chunk['sha256']
        with np.load(source) as z:
            end=len(z['time_s'])-int(chunk.get('skip_last',0));index=(offset+np.arange(end))//width;offset+=end
            np.maximum.at(force,index,z['forces_n'][:end].max(axis=1))
            np.maximum.at(coverage,index,z['coverage'][:end])
            np.add.at(counts,index,1)
            for finger in range(4):np.add.at(safe[:,finger],index,z['safe_belly'][:end,finger])
    assert offset==audit['native_states'];assert abs(force.max()-max(audit['raw_force_peak_n']))<1e-9
    t=np.minimum((np.arange(bins)+.5)*.1,summary['executed_s']);safe/=counts[:,None]
    fig,axes=plt.subplots(3,1,figsize=(14,8),sharex=True,gridspec_kw={'height_ratios':[1,1,1.2]},facecolor='#f3f6fa')
    axes[0].plot(t,coverage*100,color='#18835e',lw=1.5);axes[0].set_ylabel('Measured coverage (%)');axes[0].set_ylim(0,102)
    axes[1].plot(t,force,color='#344f85',lw=.8,label='Maximum raw force in each 100 ms bin')
    axes[1].axhline(8,color='#d29a1b',ls='--',lw=1,label='8 N: no coverage credit')
    axes[1].axhline(12,color='#c65353',ls='--',lw=1,label='12 N: 20 ms sustained stop threshold')
    k=int(force.argmax());axes[1].annotate(f'{force[k]:.3f} N',(t[k],force[k]),xytext=(5,7),textcoords='offset points',fontsize=9)
    axes[1].set_ylim(0,max(14,force.max()*1.15));axes[1].set_ylabel('Raw force (N)');axes[1].legend(loc='upper left',fontsize=8)
    im=axes[2].imshow(safe.T,aspect='auto',origin='lower',interpolation='nearest',cmap='Blues',vmin=0,vmax=1,
        extent=(0,summary['executed_s'],-.5,3.5))
    axes[2].set_yticks(range(4),['Index','Middle','Ring','Thumb']);axes[2].set_ylabel('Safe belly contact')
    axes[2].set_xlabel('Elapsed physical simulation time (s)')
    for ax in axes:ax.set_xlim(0,summary['executed_s']);ax.grid(axis='x',alpha=.15)
    colour_axis=axes[2].inset_axes([1.01,0,.012,1])
    fig.colorbar(im,cax=colour_axis,label='Contact fraction per 100 ms bin')
    fig.suptitle('REPLAY-VERIFIED WHOLE TRAJECTORY\nForce peaks are retained; the contact raster includes approach, transfers and recovery.',fontsize=14)
    fig.subplots_adjust(left=.09,right=.92,top=.88,bottom=.08,hspace=.2)
    fig.savefig(case/'timeline_verified.png',dpi=150);plt.close(fig)
    return case/'timeline_verified.png'


def contact_path_figure(case):
    """Plot verified contact samples themselves, without filling gaps."""
    case=Path(case).resolve();audit=json.loads((case/'independent_native_audit/audit.json').read_text())
    assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
    assert sha256((case/'summary.json').read_bytes()).hexdigest()==audit['source_files']['summary.json']
    assert sha256((case/'model.xml').read_bytes()).hexdigest()==audit['source_files']['model.xml']
    summary=json.loads((case/'summary.json').read_text());mesh=task_mesh();points=[];normals=[];times=[]
    body=ET.fromstring((case/'model.xml').read_text()).find(".//body[@name='fr3_e05_object']")
    origin=np.fromstring(body.get('pos'),sep=' ')
    for chunk in summary['trace_chunks']:
        source=case/chunk['file'];assert sha256(source.read_bytes()).hexdigest()==chunk['sha256']
        with np.load(source) as z:
            end=len(z['time_s'])-int(chunk.get('skip_last',0));ids=np.arange(0,end,25)
            safe=z['safe_belly'][ids]
            points.extend((z['belly_contact_positions_world_m'][ids]-origin)[safe])
            normals.extend(z['belly_normals_world'][ids][safe])
            times.extend(np.broadcast_to(z['time_s'][ids,None],safe.shape)[safe])
    points=np.asarray(points);normals=np.asarray(normals);times=np.asarray(times)
    fig=plt.figure(figsize=(15,9),facecolor='#f3f6fa')
    views=[('Broad face +Z',(0,1),(0,0,1)),('Side +X',(1,2),(1,0,0)),('Broad face -Z',(0,1),(0,0,-1)),
           ('Side -X',(1,2),(-1,0,0)),('Cap / shoulder -Y',(0,2),(0,-1,0)),('Base +Y',(0,2),(0,1,0))]
    for k,(label,axes,direction) in enumerate(views):
        ax=fig.add_subplot(2,3,k+1)
        vertices=np.asarray(mesh.vertices)[:,axes]
        ax.add_collection(PolyCollection(vertices[mesh.faces],facecolor='#d7dee6',edgecolors='none',linewidths=0,zorder=1))
        direction=np.asarray(direction)
        visible=normals@direction>.05
        xy=points[visible][:,axes]
        ax.scatter(xy[:,0],xy[:,1],c=times[visible],s=1.7,cmap='viridis',vmin=0,vmax=summary['executed_s'],zorder=2)
        lo,hi=vertices.min(axis=0),vertices.max(axis=0);padding=.08*max(hi-lo)
        ax.set_xlim(lo[0]-padding,hi[0]+padding);ax.set_ylim(lo[1]-padding,hi[1]+padding)
        ax.set_aspect('equal');ax.set_axis_off();ax.set_title(label)
    fig.suptitle('ACTUAL ACCEPTED BELLY CONTACT SAMPLES\nDots are measured contact points; no line connects intervals without contact.',fontsize=15)
    fig.text(.5,.025,'Original whole object. Contact samples are shown at 20 Hz, coloured by elapsed simulation time; measured normals filter the hidden side.',ha='center',fontsize=10)
    sm=plt.cm.ScalarMappable(cmap='viridis',norm=plt.Normalize(0,summary['executed_s']));sm.set_array([])
    bar=fig.colorbar(sm,ax=fig.axes,orientation='horizontal',fraction=.025,pad=.02,shrink=.6);bar.set_label('Elapsed simulation time (s)')
    fig.savefig(case/'contact_paths_verified.png',dpi=160);plt.close(fig)
    return case/'contact_paths_verified.png'


def coverage_figure(case,*,in_progress=False):
    case=Path(case).resolve();mesh=task_mesh();vertices=np.asarray(mesh.vertices);faces=np.asarray(mesh.faces)
    if in_progress:
        with np.load(case/'checkpoint.npz') as z:mask=z['visited_vertices'];time_s=float(z['time'])
        title='RUNNING NATIVE CONTACT LEDGER / NOT YET REPLAY-AUDITED'
    else:
        verified=case/'independent_native_audit/verified_preview.npz'
        audit=json.loads((case/'independent_native_audit/audit.json').read_text())
        assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
        assert sha256(verified.read_bytes()).hexdigest()==audit['preview_sha256']
        with np.load(verified) as z:mask=z['covered_vertices'];time_s=float(z['time_s'][-1])
        title='REPLAY-VERIFIED NATIVE BELLY CONTACTS'
    assert len(mask)==len(vertices)==8192
    plt.rcParams.update({'font.size':11})
    fig=plt.figure(figsize=(15,9),facecolor='#f3f6fa')
    panels=[('Broad face +Z',90,-90),('Side +X',0,0),('Broad face -Z',-90,-90),('Side -X',0,180),('Cap / shoulder end -Y',0,-90),('Base end +Y',0,90)]
    face_fraction=mask[faces].mean(axis=1)
    colors=np.array([[.79,.82,.86,1.]]*len(faces));colors[face_fraction>0]=[.56,.77,.67,1.];colors[face_fraction==1]=[.12,.58,.39,1.]
    for k,(label,elevation,azimuth) in enumerate(panels):
        ax=fig.add_subplot(2,3,k+1,projection='3d');ax.set_proj_type('ortho')
        ax.add_collection3d(Poly3DCollection(vertices[faces],facecolors=colors,edgecolors='none',linewidths=0))
        lo,hi=mesh.bounds;center=(lo+hi)/2;extent=max(hi-lo)*.52
        for setter,c in zip((ax.set_xlim,ax.set_ylim,ax.set_zlim),center):setter(c-extent,c+extent)
        ax.set_box_aspect((1,1,1));ax.view_init(elev=elevation,azim=azimuth);ax.set_axis_off();ax.set_title(label)
    fig.suptitle(f'{title}\nt = {time_s:.2f} s | covered vertices {mask.sum()} / 8192 | green: covered, grey: missing',fontsize=15)
    fig.text(.5,.025,'Original complete mesh. Coverage uses a 25 mm mesh-edge geodesic neighbourhood of qualified native belly contact; it is not literal pressure footprint.',ha='center',fontsize=10)
    fig.subplots_adjust(left=.02,right=.98,bottom=.065,top=.88,hspace=.02,wspace=.01)
    filename='coverage_running.png' if in_progress else 'coverage_verified.png';fig.savefig(case/filename,dpi=160);plt.close(fig)
    area=np.zeros(len(vertices));np.add.at(area,faces.ravel(),np.repeat(mesh.area_faces/3,3))
    normals=np.asarray(mesh.vertex_normals);axis=np.argmax(abs(normals),axis=1);sign=np.sign(normals[np.arange(len(normals)),axis])
    regions=[]
    for label,a,s in [('宽面 +Z',2,1),('侧面 +X',0,1),('宽面 −Z',2,-1),('侧面 −X',0,-1),('瓶盖/瓶肩端 −Y',1,-1),('瓶底端 +Y',1,1)]:
        region=(axis==a)&(sign==s);regions.append(dict(region=label,vertices=int(region.sum()),covered=int(mask[region].sum()),area_fraction=float(area[region]@mask[region]/area[region].sum())))
    result=dict(file=filename,vertices=8192,covered_vertices=int(mask.sum()),area_fraction=float(area@mask/area.sum()),regions=regions,
        scope='Dominant-normal regions are reporting partitions only; no original surface is removed.',time_s=time_s)
    (case/('coverage_running.json' if in_progress else 'coverage_verified.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def report(case):
    case=Path(case).resolve();audit=json.loads((case/'independent_native_audit/audit.json').read_text())
    summary=json.loads((case/'summary.json').read_text());assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
    visual_source=case/'visualization_source';visual_source.mkdir(exist_ok=True)
    (visual_source/'report.py').write_bytes(Path(__file__).read_bytes())
    assert sha256((case/'summary.json').read_bytes()).hexdigest()==audit['source_files']['summary.json']
    geometry=coverage_figure(case);contact_path_figure(case);video=json.loads((case/'natural_complete_render.json').read_text())
    assert video['actual_saved_q_only'] and video['uncut']
    assert sha256(Path(video['video']).read_bytes()).hexdigest()==video['video_sha256']
    assert video['source_manifest_sha256']==audit['source_files']['manifest.json']
    excursions=verified_force_excursions(case,audit,summary)
    timeline_figure(case,audit,summary)
    force_note=(f'记录到{excursions["episodes"]}段原始力≥8 N的片段，累计{excursions["total_s"]:.3f}秒，最长{excursions["longest_s"]:.3f}秒；这些状态均未计入覆盖。瞬时力峰仍有优化空间。'
        if excursions['episodes'] else '全程未记录到原始力≥8 N的状态。')
    complete=audit['full_surface_complete'];result='按25mm测地邻域判据，原网格整瓶指肚覆盖已完成' if complete else '仍未完成整瓶指肚覆盖'
    lineage=('本结果是一条保持物理状态连续的开发续接轨迹，期间更新过控制器；不是最终版本从零启动的单次实验。接续点、旧版区段和源码均在清单中保留。'
             if not audit.get('fresh_single_version',True) else '本结果从声明的初态开始，由同一控制版本完整执行。')
    rows=''.join(f'<tr><td>{r["region"]}</td><td>{r["covered"]} / {r["vertices"]}</td><td>{r["area_fraction"]*100:.2f}%</td></tr>' for r in geometry['regions'])
    qmin=np.array(audit['actual_joint_min_deg'][7:19]).reshape(3,4);qmax=np.array(audit['actual_joint_max_deg'][7:19]).reshape(3,4)
    jointrows=''.join(f'<tr><td>{name}</td>'+''.join(f'<td>{qmin[k,j]:.2f} — {qmax[k,j]:.2f}°</td>' for j in range(4))+'</tr>' for k,name in enumerate(('食指','中指','无名指')))
    actualrows=''.join(f'<tr><td>{name}</td><td>{audit["actual_tangential_pad_material_motion_m"][k]:.3f} m</td><td>{100*audit["safe_belly_fraction"][k]:.2f}%</td><td>{audit["longest_safe_belly_gap_s"][k]:.3f} s</td></tr>' for k,name in enumerate(('食指','中指','无名指','拇指')))
    duration=sum(audit['phase_duration_s'].values())
    any_fraction=sum(t*audit['phase_any_belly_fraction'][p] for p,t in audit['phase_duration_s'].items())/duration
    slide_fraction=audit['phase_any_belly_fraction'].get('SLIDE',0.)
    contact_gallery=''
    detail_video=''
    detail_meta=case/'natural_details_render.json'
    if detail_meta.exists():
        detail=json.loads(detail_meta.read_text())
        assert detail['source_manifest_sha256']==audit['source_files']['manifest.json']
        assert not detail['uncut'] and detail['speed']==1.
        assert sha256(Path(detail['video']).read_bytes()).hexdigest()==detail['video_sha256']
        detail_video=f'<section class="card"><h2>正常速度查看手指怎样贴面</h2><video controls preload="metadata" src="{html.escape(Path(detail["video"]).name)}"></video><p class="note">这段由六个方向的局部真实接触片段组成，正常速度播放，切换时实际时间会跳变；用于看清手指动作，不替代上方完整遍历视频。</p></section>'
    contact_meta=case/'natural_contact_render.json'
    if contact_meta.exists():
        contact=json.loads(contact_meta.read_text())
        assert contact['source_manifest_sha256']==audit['source_files']['manifest.json']
        frames=''.join(f'<figure><img src="{html.escape(Path(s["file"]).name)}" alt="{html.escape(s["region"])}实际指肚接触"><figcaption>{html.escape(s["region"])} · 实际 t={s["time_s"]:.2f}s</figcaption></figure>' for s in contact['stills'])
        missing='、'.join(r['region'] for r in contact['regions'] if not r['available'])
        note=('以下方向未找到符合分区条件的实测指肚截图：'+missing+'。') if missing else '六个方向均选自已核验的真实安全指肚接触时刻。'
        contact_gallery=f'<section class="card"><h2>六方向接触与手形近景</h2><p>{note}这些截图用于查看局部手形，完整遍历仍以上方全程视频和原网格覆盖为准。</p>{frames}</section>'
    file=Path(video['video']).name
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mustard 自然手形与整瓶指肚遍历</title>
<style>body{{margin:0;background:#eef2f6;color:#16273b;font:17px/1.65 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1280px;margin:auto;padding:32px}}h1{{font-size:32px}}h2{{margin-top:36px}}.card{{background:white;border-radius:15px;padding:24px;margin:20px 0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:18px}}strong{{font-size:25px;display:block}}video,img{{max-width:100%;border-radius:10px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #dde4eb;padding:10px}}.note{{color:#586b7d;font-size:15px}}a{{color:#1567ad}}@media(max-width:700px){{.metrics{{grid-template-columns:1fr 1fr}}main{{padding:16px}}}}</style>
<main><h1>Mustard · 自然手形与整瓶指肚遍历</h1><p>{result}。只运行 MCC；物体、碰撞模型和全部原网格范围保持。</p><p class="note">{lineage}</p>
<div class="card metrics"><div>实测指肚覆盖<strong>{audit['area_fraction']*100:.3f}%</strong>{audit['covered_vertices']} / 8192 顶点</div><div>原覆盖目标<strong>{audit['covered_goals']} / 388</strong>几何规划不计接触</div><div>全部接触力峰值<strong>{max(audit['raw_force_peak_n']):.3f} N</strong>臂自碰峰 {audit['arm_self_peak_n']:.3f} N</div><div>最长全手有效接触中断<strong>{audit['longest_safe_belly_gap_s'][-1]:.3f} s</strong>包含调姿和重新接近</div></div>
<p>整瓶覆盖与手形限制已经验证；换面离面时间仍偏长，不能称为不间断贴面遍历。本结果仅限当前单个Mustard仿真案例。</p>
<p>{force_note} 超过12 N的累计时长为{excursions['above_hard_threshold_total_s']*1000:.0f}毫秒、最长连续{excursions['above_hard_threshold_longest_s']*1000:.0f}毫秒。本开发试验的硬停止条件为12 N持续20毫秒；独立重放通过代表记录与原生执行一致，不代表力峰已充分优化。</p>
<section class="card"><h2>完整实际过程</h2><video controls preload="metadata" src="{html.escape(file)}"></video><p>视频由连续保存的真实关节状态生成，时间轴不剪接，{video['speed']:g}倍速。右侧只将物体透明，用于查看手指肚与表面的关系。</p><p class="note">实际运行 {summary['executed_s']:.3f} 秒；{summary['routes']} 段表面路径，规划器执行了{summary['necessary_lifts']}次离面重定位。停止原因：{html.escape(summary['stop_reason'])}。</p></section>
<section class="card"><h2>整瓶覆盖检查</h2><img src="coverage_verified.png" alt="六方向原网格实测覆盖"><table><tr><th>区域</th><th>覆盖顶点</th><th>面积覆盖</th></tr>{rows}</table><p class="note">绿色是合格原生指肚接触的25mm网格测地邻域；灰色是尚未覆盖。该指标不是实际皮肤压力面积。分区仅用于检查，未删除任何原物体区域。</p></section>
<section class="card"><h2>实际接触点经过了哪里</h2><img src="contact_paths_verified.png" alt="六方向实际指肚接触点轨迹"><p class="note">每个点来自已独立核验的真实安全指肚接触，颜色表示时间。不跨失去接触的区间连线，也不将25mm覆盖邻域画成真实接触点。</p></section>
{contact_gallery}
{detail_video}
<section class="card"><h2>手形是否改正</h2><p>上一版为了固定三个旧接触点，长指侧摆达到59.16°，中指PIP曾到−28.19°。新版限制侧摆、禁止末节命令反折，并根据当前可达姿态选择领触指。当前仍以单指领触为主，其他手指轻展；不声称四指持续协同。</p><table><tr><th>手指</th><th>实际侧摆范围</th><th>实际MCP范围</th><th>实际PIP范围</th><th>实际DIP范围</th></tr>{jointrows}</table><p class="note">表中是实际关节范围，可能包含执行器跟踪的微小偏差；不以指令代替运动。拇指采用独立的对掌姿态，其运动与接触在视频中逐帧显示。</p></section>
<section class="card"><h2>主动接触与连续性</h2><p>至少一根手指具有有效安全指肚接触：占全程 {any_fraction*100:.2f}%，占滑动控制阶段 {slide_fraction*100:.2f}%。全程统计包含接近、换面和寻触。</p><table><tr><th>手指</th><th>实际切向材料运动</th><th>安全指肚接触占全程</th><th>最长有效接触中断</th></tr>{actualrows}</table><p>短时接触中断会暂停扫描并主动寻触；贴面转腕失败后才尝试离面重定位。上述中断按0.2N及指肚几何资格统计，并不等同于始终完全没有物理接触。</p></section>
<section class="card"><h2>覆盖、力峰与接触时间轴</h2><img src="timeline_verified.png" alt="全程覆盖、原始力峰和四指安全接触时间图"><p class="note">以100毫秒为一格显示：力曲线保留格内全部原生观测的最大值，不平滑掉短时峰值；接触图颜色深度是该格内安全指肚接触比例，空白表示没有有效安全接触。</p></section>
<section class="card"><h2>验证与原始证据</h2><p>从唯一初态连续重放 {audit['native_states']} 个原生2ms状态；分块边界不重置物理状态。全程独立重算指肚接触和原网格覆盖。{'全部已执行命令再次通过原严格碰撞检查。' if audit['all_applied_commands_original_geometry_checked'] else '命令几何未在本审计重复，见运行时守护。'}</p><p><a href="independent_native_audit/audit.json">独立核验</a> · <a href="summary.json">运行结果</a> · <a href="manifest.json">模型与源码清单</a> · <a href="events.json">换面与恢复事件</a> · <a href="trace_chunks.json">2ms连续原始轨迹分块</a></p></section></main></html>'''
    (case/'review.html').write_text(page,encoding='utf-8')
    return case/'review.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');p.add_argument('--in-progress',action='store_true');a=p.parse_args()
    print(coverage_figure(a.case,in_progress=True) if a.in_progress else report(a.case))
