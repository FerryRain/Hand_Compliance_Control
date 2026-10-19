"""Build the local belly-contact review from one independently audited trial."""
from __future__ import annotations
import argparse
from hashlib import sha256
from html import escape
import json
import os
from pathlib import Path
from urllib.parse import quote
import numpy as np


def runs(mask):
    edges=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))


def build(case,output,video_name='belly_actual_uncut_1x.mp4'):
    case,output=Path(case).resolve(),Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    ap=case/'independent_native_audit'/'audit.json';ab=ap.read_bytes();audit=json.loads(ab)
    if audit.get('status')!='PASSED_NATIVE_EVIDENCE_AUDIT':raise ValueError('Independent audit is required')
    for name in ('trace.npz','model.xml','summary.json','manifest.json'):
        if sha256((case/name).read_bytes()).hexdigest()!=audit['source_files'][name]:
            raise ValueError('Audited source changed: '+name)
    summary=json.loads((case/'summary.json').read_bytes());manifest=json.loads((case/'manifest.json').read_bytes())
    local_metrics_path=case/'independent_native_audit'/'local_slide_metrics.json'
    local_metrics=json.loads(local_metrics_path.read_bytes()) if local_metrics_path.exists() else None
    if local_metrics is not None:
        if local_metrics['source_native_audit_sha256']!=sha256(ab).hexdigest() or local_metrics['trace_sha256']!=audit['source_files']['trace.npz']:
            raise ValueError('Local sliding metrics do not bind the selected native audit')
    with np.load(case/'trace.npz',allow_pickle=False) as z:
        trace={k:z[k].copy() for k in ('time_s','phase','forces_n','belly_forces_n','belly_contact_positions_world_m')}
    measure=case/'independent_native_audit'/'independent_measurements.npz'
    if sha256(measure.read_bytes()).hexdigest()!=audit['artifacts'][measure.name]:raise ValueError('Measurement artifact changed')
    with np.load(measure) as z:
        safe=z['belly_safe'].copy();speeds=z['belly_tangential_material_speed_m_s'].copy()
        assert np.array_equal(z['time_s'],trace['time_s'])
    t=trace['time_s'];phase=trace['phase'];slide=phase=='BELLY_SLIDE';dt=np.diff(t)
    names=('食指','中指','无名指','拇指');colors=('#1675b7','#c53080','#219264','#b95a16')
    cumulative=np.zeros((len(t),4));rows=[]
    for f,stat in enumerate(audit['finger_statistics']):
        keep=safe[:-1,f]&safe[1:,f]&slide[:-1]&slide[1:]
        cumulative[1:,f]=np.cumsum(.5*(speeds[:-1,f]+speeds[1:,f])*dt*keep)*1000
        expected=stat['actual_motion_by_phase'].get('BELLY_SLIDE',{}).get('tangential_material_motion_m',0)*1000
        assert abs(cumulative[-1,f]-expected)<1e-8
        losses=[]
        for a,b in runs(slide&~safe[:,f]):
            end=float(t[b]) if b<len(t) and slide[b] else float(t[b-1])
            losses.append({'start_s':float(t[a]),'end_s':end,'duration_s':end-float(t[a]),
                           'censored':bool(b==len(t) or not slide[b])})
        rows.append({**stat,'finger_index':f,'finger':names[f],'slide_max_gap_s':max([x['duration_s'] for x in losses]+[0.]),
                     'slide_gaps':losses,'peak_raw_force_n':audit['raw_force_peak_n'][f],
                     'slide_material_motion_mm':expected})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,gridspec_kw={'height_ratios':[1.2,1,.7]})
    for f in range(4):
        axes[0].plot(t,trace['forces_n'][:,f],color=colors[f],lw=1.25,label=names[f])
        axes[1].plot(t,cumulative[:,f],color=colors[f],lw=1.8)
    axes[0].set_ylabel('全部代理接触力 / N');axes[0].legend(ncol=4,loc='upper right');axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel('滑动阶段实测切向运动\n累计 / mm')
    for f in range(4):
        for a,b in runs(safe[:,f]):
            axes[2].broken_barh([(float(t[a]),max(.002,float(t[b-1]-t[a])))],(f-.3,.6),facecolors=colors[f])
    axes[2].set_yticks(range(4),names);axes[2].set_ylim(3.6,-.6);axes[2].set_ylabel('实际合格腹面接触');axes[2].set_xlabel('原生记录时间 / s')
    phase_labels={'UNLOAD_WRONG_FACE':'卸载边缘接触','BELLY_ALIGN':'实际调姿','BELLY_APPROACH':'接近并寻触','BELLY_SLIDE':'腹面滑动'}
    changes=np.r_[0,np.flatnonzero(phase[1:]!=phase[:-1])+1,len(t)]
    for a,b in zip(changes[:-1],changes[1:]):
        for ax in axes:ax.axvline(t[a],color='#718096',lw=.65,alpha=.5)
        axes[0].text(float((t[a]+t[b-1])/2),1.03,phase_labels.get(str(phase[a]),str(phase[a])),
                     ha='center',va='bottom',transform=axes[0].get_xaxis_transform(),fontsize=9)
    for ax in axes:ax.grid(alpha=.18);ax.set_xlim(float(t[0]),float(t[-1]))
    fig.suptitle('同一局部试验：实际力、切向材料运动与腹面接触连续性',y=.995)
    fig.tight_layout();figure=output/'actual_force_motion_contact.png';fig.savefig(figure,dpi=170);plt.close(fig)
    # Contact centroid trajectories are separate from material motion and are
    # never used as the distance displayed in the main result table.
    points=trace['belly_contact_positions_world_m'];all_points=points[safe]
    origin=all_points.mean(axis=0) if len(all_points) else np.zeros(3)
    fig=plt.figure(figsize=(8,6));ax=fig.add_subplot(111,projection='3d')
    for f in range(4):
        first=True
        for a,b in runs(safe[:,f]&slide):
            p=(points[a:b,f]-origin)*1000
            ax.plot(p[:,0],p[:,1],p[:,2],color=colors[f],lw=1.4,label=names[f] if first else None);first=False
    ax.set(xlabel='世界X（相对固定中心）/ mm',ylabel='世界Y / mm',zlabel='世界Z / mm',
           title='滑动阶段实测接触中心轨迹：失联不连线')
    if len(all_points):ax.legend()
    fig.text(.04,.02,'接触中心会在指垫上迁移；此轨迹长度不等于材料滑动距离。',fontsize=10)
    fig.tight_layout();trajectory=output/'actual_contact_centroid_paths.png';fig.savefig(trajectory,dpi=170);plt.close(fig)
    def url(p):return quote(os.path.relpath(Path(p),output).replace('\\','/'),safe='/._-')
    def link(p,text):return f'<a href="{url(p)}">{escape(text)}</a>'
    local_complete=summary['stop_reason']=='LOCAL_SLIDE_FINISHED'
    headline='已实现一段真实腹面贴滑' if local_complete else '已测到局部腹面接触，试验提前停止'
    slide_t=t[slide];slide_span=float(slide_t[-1]-slide_t[0]) if len(slide_t) else 0.
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mustard 指腹贴滑局部试验</title><style>
body{{margin:0;background:#f2f5fa;color:#182538;font:17px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1150px;margin:auto;padding:36px 24px 70px}}h1{{font-size:34px;line-height:1.3;margin:8px 0 14px}}h2{{font-size:23px;margin:32px 0 10px}}p{{margin:10px 0}}.eyebrow{{color:#55708f;font-size:14px}}.notice{{padding:14px 18px;background:#fff3d8;border-left:4px solid #c48819;border-radius:5px}}.card{{background:white;padding:22px;border-radius:12px;margin:20px 0;box-shadow:0 2px 14px #1d35570b}}video,img{{width:100%;border-radius:7px;background:#101a2d}}table{{width:100%;border-collapse:collapse;font-size:15px}}td,th{{padding:10px 9px;border-bottom:1px solid #dce3ed;text-align:left}}th{{background:#edf3fa}}a{{color:#1465a1}}.muted,small{{color:#5a6e85;font-size:14px}}.scroll{{overflow:auto}}summary{{cursor:pointer;font-weight:600}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.stat{{background:#edf3fa;border-radius:8px;padding:14px}}.stat b{{display:block;font-size:26px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}h1{{font-size:28px}}}}</style><main>
<div class="eyebrow">新 I04 · MCC · mustard · 局部三指指腹示范 · 2026-10-17</div>
<h1>{headline}</h1><p>先短暂卸载旧的边缘接触，调整手指与腕姿态，让末节肚面朝向物体，再主动寻触并沿表面移动。下方展示真实原生记录，不用目标姿态替代实际运动。</p>
<div class="notice"><b>范围：</b>这是局部试验。拇指尚未参与，整瓶遍历未完成。{'本次完成预设局部滑动段。' if local_complete else '终止原因：'+escape(summary['stop_reason'])+'。'}腹面判定仍基于注册的椭球碰撞代理；真实细节 mesh 用于显示。</div>
<div class="grid"><div class="stat">滑动阶段时长<b>{slide_span:.3f} s</b></div><div class="stat">全段最大实测力<b>{max(audit['raw_force_peak_n']):.3f} N</b></div><div class="stat">至少两指保持腹面接触<b>{100*np.mean(safe[slide,:3].sum(axis=1)>=2) if slide.any() else 0:.2f}%</b></div></div>
<h2>完整过程，连续 1×</h2>'''
    video=case/video_name;vm=video.with_name(video.stem+'_render.json')
    if video.exists() and vm.exists():
        meta=json.loads(vm.read_bytes())
        if meta['trace_sha256']!=audit['source_files']['trace.npz']:raise ValueError('Video belongs to another trace')
        if sha256(video.read_bytes()).hexdigest()!=meta['video_sha256']:raise ValueError('Video hash changed')
        poster=min(meta['qa_frames'],key=lambda x:abs(x['time_s']-20.))['path']
        jump=float(slide_t[0]+.038) if len(slide_t) else 0.
        body+=f'<video id="mainVideo" controls preload="metadata" poster="{url(poster)}" src="{url(video)}#t={jump:.3f}"></video><p><button onclick="document.getElementById(\'mainVideo\').currentTime=0;document.getElementById(\'mainVideo\').play()">从头看完整过程</button> <button onclick="document.getElementById(\'mainVideo\').currentTime={jump:.3f};document.getElementById(\'mainVideo\').play()">从 {jump:.1f}s 看贴滑</button></p><p class="muted">完整 {meta["start_s"]:.3f}–{meta["end_s"]:.3f} s，连续1×，默认定位贴滑开始。卸载、调姿、离面和失联均保留。左侧手和物体不透明；右侧仅物体透明。金环是腹面区域，粉点是实测合格接触中心，关节角和力取实际保存状态。</p>'
    else:body+='<p class="notice">主录像待生成；当前仅展示已经核验的数值。</p>'
    body+='<h2>各指实际做到什么</h2>'
    if local_metrics is not None:
        body+=f'<p>12秒滑动期间始终至少两指保持安全腹面接触，三指同时接触占 {100*local_metrics["three_safe_belly_fraction"]:.2f}%。有效接触短暂中断后恢复；少量弱的错面接触已从合格腹面统计中排除，不代表每个碰撞点都在肚面。</p>'
    body+='<div class="card scroll"><table><tr><th>手指</th><th>滑动期合格腹面占比</th><th>实测切向材料运动</th><th>有效腹面接触最长中断</th><th>全段峰值力</th></tr>'
    for row in rows:
        fraction=row['slide_safe_belly_fraction'];pc='—' if fraction is None else f'{fraction*100:.3f}%'
        body+=f'<tr><td>{row["finger"]}</td><td>{pc}</td><td>{row["slide_material_motion_mm"]:.2f} mm</td><td>{row["slide_max_gap_s"]:.3f} s</td><td>{row["peak_raw_force_n"]:.3f} N</td></tr>'
    body+='</table><p class="muted">占比只针对腹面滑动阶段；有效接触中断按腹面判据与≥0.2 N力阈值计算，低于阈值不一定代表物理完全离面。卸载与调姿时允许离面。切向运动由真实关节速度和接触点 Jacobian 积分，只累计相邻原生状态都保持安全腹面接触的区间。它反映接触处材料运动，不能单独区分滑动与滚动；不把请求路径长度当实测距离。</p></div>'
    body+=f'<details class="card"><summary>查看实际力、连续性与轨迹</summary><img src="{url(figure)}" alt="实际接触力和连续腹面接触图"><img src="{url(trajectory)}" alt="实测接触中心轨迹"><p class="muted">世界坐标减去同一个固定中心，失联区间不连线。接触中心迁移与真实材料运动分别记录。</p></details>'
    cfg=manifest['belly_contact']
    body+=f'<details class="card"><summary>怎样判定“肚子贴上了”</summary><p>固定腹面方向是末节 body −X，也就是注册指垫的局部 +Z。必须是正面中央区域（归一化半径 ≤ {cfg["maximum_patch_radius"]:g}）、接触法向与腹面方向夹角 ≤ {cfg["maximum_alignment_angle_deg"]:g}°，并有真实接触力；安全连续性要求合格腹面力 ≥0.2 N、该指全部接触力 &lt;8 N。只碰到同一个指垫代理，不再自动算腹面接触。</p></details>'
    old=case.parent/'pad_belly_visual_diagnostic_20261017'/'old_ring_recontact_not_belly.png'
    if old.exists():body+=f'<details class="card"><summary>旧接触为什么不算腹面成功</summary><p>旧例 t=1249.900 s 无名指已有 1.971 N，但原生接触点在背侧边缘：法向夹角119.80°、归一化半径0.9946。旧录像中的“恢复”只代表碰撞代理重新接触。</p><img src="{url(old)}" alt="旧接触在指垫边缘和背侧的原生诊断"></details>'
    body+='<details class="card"><summary>保留开发过程中的失败与间断</summary><div class="scroll"><table><tr><th>试验</th><th>真实终态</th><th>滑动期食/中/无名指腹面占比</th><th>审计</th></tr>'
    history=[]
    for prior in sorted(case.parent.glob('mustard_belly_probe*_20261017')):
        if not (prior/'summary.json').exists():continue
        s=json.loads((prior/'summary.json').read_bytes());pa=prior/'independent_native_audit'/'audit.json'
        audited=pa.exists() and json.loads(pa.read_bytes()).get('status')=='PASSED_NATIVE_EVIDENCE_AUDIT'
        v=s.get('slide_belly_contact_fraction_by_finger');text='未进入腹面滑动' if v is None else ' / '.join(f'{x*100:.2f}%' for x in v[:3])
        body+=f'<tr><td>{escape(prior.name)}</td><td>{escape(s["stop_reason"])}</td><td>{text}</td><td>{link(pa,"原生证据核验") if audited else "待核验"}</td></tr>'
        history.append({'case':str(prior),'stop_reason':s['stop_reason'],'independently_audited':audited})
    body+='</table></div><p class="muted">这是逐步开发过程，控制参数与数值实现有变化；不作为单因素对照。没有修改旧试验的结果或覆盖账本。</p></details>'
    more_evidence=''
    for file,label in (('command_geometry_audit.json','全部下发命令几何核验'),('local_slide_metrics.json','局部滑动与中断细节')):
        ep=case/'independent_native_audit'/file
        if ep.exists():more_evidence+=' · '+link(ep,label)
    body+=f'<h2>可复查证据</h2><p>{link(ap,"完整原生核验与各指指标")}{more_evidence} · {link(case/"manifest.json","本次协议和源快照")} · {link(case/"summary.json","本次真实终态")} · {link(output/"review_evidence.json","报告来源清单")}</p><p class="muted">独立连续重放 {audit["native_states"]:,} 个 2 ms 原生状态，核对实际关节、速度、控制、接触传感与覆盖。证据核验通过不等于整瓶完成；本次覆盖仍是 {audit["covered_vertices"]}/{audit["total_vertices"]} 个采样点、{audit["measured_belly_goals"]}/{audit["total_goals"]} 个目标，25 mm 测地邻域覆盖不等于每个顶点都直接触碰。</p></main></html>'
    (output/'index.html').write_bytes(body.encode('utf-8'))
    evidence={'selected_case':str(case),'behavior_stop_reason':summary['stop_reason'],'local_slide_complete':local_complete,
              'full_surface_complete':False,'source_files':audit['source_files'],'independent_audit_sha256':sha256(ab).hexdigest(),
              'slide_endpoint_span_s':slide_span,'rows':rows,'history':history,
              'coverage_area_fraction':audit['measured_belly_coverage_fraction'],'plot_measurements_sha256':sha256(measure.read_bytes()).hexdigest(),
              'plot_scope':'All actual native force; BELLY_SLIDE adjacent-safe material movement; native safe-belly intervals',
              'local_slide_metrics':local_metrics,'local_slide_metrics_sha256':sha256(local_metrics_path.read_bytes()).hexdigest() if local_metrics else None,
              'report_generator_sha256':sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'review_evidence.json').write_bytes(json.dumps(evidence,ensure_ascii=False,indent=2).encode('utf-8'))
    print(output/'index.html');return evidence


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case',type=Path)
    p.add_argument('--output',type=Path,default=Path('Module/generated/new_i04_mustard_belly_review_20261017'))
    p.add_argument('--video-name',default='belly_actual_uncut_1x.mp4')
    a=p.parse_args();build(a.case,a.output,a.video_name)
