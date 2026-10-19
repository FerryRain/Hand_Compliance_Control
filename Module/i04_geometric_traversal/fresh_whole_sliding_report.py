"""Separate actual-data report for one fresh whole-mustard coordinated run.

Only this run supplies its coverage, motion, figures and videos. Historical
versions appear in a matched-prefix comparison and an explicitly separate link.
An ongoing run, a passed integrity audit, or a local recovery never establishes
whole-surface completion. No simulation state, controller or run file is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html import escape
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from .coordinated_prefix_observer import DEFAULT_CASE, GENERATED, _case_evidence, read_completed_prefix
from .coordinated_recovery_report import (
    completed_run_evidence, evidence_file, link, old_checkpoint, read_json, relative_url,
)


NAMES = ('食指','中指','无名指','拇指')
COLORS = ('#176f91','#e18b2e','#8562b1','#38905f')
DEFAULT_OUTPUT = GENERATED/'mustard_coordinated_prepared_review_20261016'


def available_horizon(case):
    case = Path(case)
    paths = [case/'trace.npz'] if (case/'trace.npz').exists() else sorted((case/'trace_chunks').glob('*.npz'))
    for path in reversed(paths):
        try:
            with np.load(BytesIO(path.read_bytes()),allow_pickle=False) as saved:
                if len(saved['time_s']) > 1:
                    return float(saved['time_s'][-1])
        except (OSError,ValueError,EOFError):
            continue
    return None


def source_status(case,manifest):
    repository = Path(__file__).resolve().parents[2]
    result = []
    for name,expected in (manifest or {}).get('source_hashes',{}).items():
        snapshot = evidence_file(Path(case)/'source_snapshot'/name)
        workspace = evidence_file(repository/name)
        result.append(dict(path=name,manifest_sha256=expected,
            snapshot_sha256=snapshot['sha256'] if snapshot else None,
            workspace_sha256=workspace['sha256'] if workspace else None,
            snapshot_matches=bool(snapshot and snapshot['sha256']==expected),
            workspace_matches=bool(workspace and workspace['sha256']==expected)))
    return dict(files=result,count=len(result),all_frozen_snapshots_match=bool(result and all(s['snapshot_matches'] for s in result)),
                all_current_workspace_matches=bool(result and all(s['workspace_matches'] for s in result)))


def plot_measurements(trace,output,scope):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt, font_manager
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']='Microsoft YaHei'
    plt.rcParams['axes.unicode_minus']=False
    t,force,active,q = (trace[k] for k in ('time_s','forces_n','active','q_rad'))
    contact = active & (force>=.2)
    fig,ax=plt.subplots(3,1,figsize=(13,8),sharex=True,constrained_layout=True)
    ax[0].plot(t,100*trace['coverage'],color='#176f91');ax[0].set_ylabel('实测面积覆盖 / %');ax[0].set_ylim(0,100)
    for finger in range(4):
        ax[1].plot(t,force[:,finger],color=COLORS[finger],label=NAMES[finger],lw=.8)
        x=np.where(contact[:,finger],finger+1,np.nan)
        ax[2].plot(t,x,color=COLORS[finger],lw=1.5)
    ax[1].axhline(8,color='#a33',ls='--',lw=1,label='8 N 控制边界')
    ax[1].set_ylabel('实测接触力 / N');ax[1].legend(ncol=5,fontsize=9)
    ax[2].set(yticks=np.arange(1,5),yticklabels=NAMES,ylabel='实际接触',xlabel='本轮从初态开始的时间 / s')
    for item in ax:
        item.set_xlim(t[0],t[-1]);item.grid(alpha=.18)
    fig.suptitle(f'{scope}：本轮覆盖、实测力与四指参与（0–{t[-1]:.2f} s）',fontsize=14)
    timeline=output/'fresh_measured_timeline.png';fig.savefig(timeline,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,1,figsize=(13,8.5),sharex=True,constrained_layout=True)
    for finger,axis in enumerate(axes):
        for joint in range(4):
            column=7+4*finger+joint
            axis.plot(t,np.rad2deg(q[:,column]-q[0,column]),lw=.8,label=f'关节 {joint+1}')
        axis.set_ylabel(f'{NAMES[finger]} / °');axis.grid(alpha=.18);axis.set_xlim(t[0],t[-1])
    axes[0].legend(ncol=4,fontsize=9)
    axes[-1].set_xlabel('本轮从初态开始的时间 / s')
    fig.suptitle(f'{scope}：实际四指关节相对初态的变化，未使用控制指令或偏置',fontsize=14)
    joints=output/'fresh_actual_joint_motion.png';fig.savefig(joints,dpi=150);plt.close(fig)
    return dict(timeline=str(timeline),actual_joints=str(joints))


def prefix_video(case,output,filename):
    path=case/filename;meta=read_json(case/(path.stem+'_render.json'))
    if not path.exists() or not meta:
        return '',None
    video=evidence_file(path)
    checks=[]
    for source in meta.get('sources',[]):
        source_path=Path(source.get('file',source.get('path','')))
        actual=evidence_file(source_path)
        in_case=case.resolve() in source_path.resolve().parents
        checks.append(bool(in_case and actual and actual['sha256']==source.get('sha256')))
    if meta.get('trace_sha256'):
        actual=evidence_file(case/'trace.npz');checks.append(bool(actual and actual['sha256']==meta['trace_sha256']))
    valid=bool(checks and all(checks) and video['sha256']==meta.get('video_sha256') and meta.get('uncut') is True)
    info=dict(metadata=meta,metadata_file=evidence_file(case/(path.stem+'_render.json')),video=video,source_matches=valid)
    if not valid:
        return '<p>录像源与哈希待核验，暂不展示。</p>',info
    caption=(f'{meta["start_s"]:.2f}–{meta["end_s"]:.2f} s，{meta["playback_speed"]:g}×，连续未剪辑。'
        '由保存的实际 q 做 FK 回放；关节角、Δ、接触力读取实测。CONTROL INTENT 只表示指令模式。'
        '这段前缀录像不构成整瓶完成证据；100 Hz 记录与视频不能排除 2 ms 级短间断。')
    return f'<video controls preload="metadata" src="{relative_url(path,output)}"></video><p class="caption">{caption}</p>',info


def comparison_table(comparison,case):
    if not comparison or comparison.get('status')!='READY' or Path(comparison['current_case']).resolve()!=case.resolve():
        return '<p>尚无本轮完整 300 秒同初态对照。</p>'
    rows=[]
    for label,record in list(comparison['comparisons'].items())+[('本轮 v3.6',comparison['current'])]:
        m=record['metrics'];finger=' / '.join(f'{100*f["contact_fraction"]:.1f}' for f in m['per_finger'])
        rows.append(f'<tr><td>{escape(label)}</td><td>{100*record["coverage"]:.2f}%</td>'
            f'<td>{100*m["safe_contact_continuity_fraction"]:.2f}%</td><td>{100*m["safe_moving_contact_fraction"]:.2f}%</td>'
            f'<td>{record["loss"]["longest_any_contact_gap_s"]:.2f} s</td><td>{finger}</td></tr>')
    return ('<p>逐值核对初始 qpos、qvel、ctrl；所有版本采用相同 100 Hz 网格、同一切向移动判据。'
        '每列均为 0–300 s，保留不利指标。跨版本是完整策略的描述性比较。</p>'
        '<div class="scroll"><table><tr><th>版本</th><th>面积</th><th>安全接触</th><th>安全移动接触</th>'
        '<th>最长全手失联</th><th>食/中/无名/拇 接触率 %</th></tr>'+''.join(rows)+'</table></div>')


def independent_followup_evidence(case,output,horizon_s):
    """Keep later adverse evidence visible without altering the fixed prefix."""
    peak_path=GENERATED/'fresh_v36_force_peak_927_20261016/independent_peak_audit.json'
    middle_path=GENERATED/'sliding_audit_v3_checks_20261016/middle_native_recovery_windows.json'
    peak=read_json(peak_path);middle=read_json(middle_path)
    evidence={};warning='';middle_html=''
    if (peak and Path(peak.get('source_run','')).resolve()==case
            and peak.get('status')=='CONFIRMED_NATIVE_FORCE_OVERSHOOT'):
        frozen=output/'frozen_force_peak_927_audit.json';frozen.write_bytes(peak_path.read_bytes())
        evidence['force_overshoot']=dict(audit=peak,audit_source=evidence_file(peak_path),
            frozen_audit=evidence_file(frozen),excluded_from_fixed_prefix_metrics=peak['first_native_over_12_s']>horizon_s,
            event_after_prefix=peak['first_native_over_12_s']>horizon_s)
        when=max(peak['peak_events'],key=lambda item:max(item['forces_n']))['time_s']
        heading=('快照后已知异常' if when>horizon_s else '已知力峰异常')
        exclusion=(f'这次事件发生在本页 0–{horizon_s:g}s 固定窗口之后，未混入其统计。'
                   if when>horizon_s else '本页指标不替代对此异常的单独审查。')
        warning=(f'<section class="critical"><h2>{heading}：拇指 {peak["peak_force_n"]:.6f} N</h2>'
            f'<p>{when:.3f}s 实际拇指力达到 <strong>{peak["peak_force_n"]:.6f} N</strong>。'
            f'{peak["release_started_s"]:.3f}s 已触发 8N 卸力；独立连续原生重放确认，'
            f'大于12N持续 {1000*peak["hardstop_accumulator_max_s"]:.0f}ms（{peak["consecutive_native_ticks_over_12"]} 个2ms时步）。'
            f'未达到原 {1000*peak["hardstop_requires_s"]:.0f}ms 硬停条件，<strong>不能据此称安全通过或平稳完成</strong>。</p>'
            '<p>原生峰值及卸力连续重放与已保存状态、力和命令锚点逐元素误差均为0；'
            f'本段未见臂/自碰力或非指腹异常接触。{exclusion}</p>'
            f'<p>{link(frozen,"本页冻结的独立峰值审计",output)} · '
            f'{link(peak_path,"审计原始结果",output)}</p></section>')
    if middle and middle.get('status')=='PASSED':
        windows=middle.get('windows',[])
        valid=bool(windows)
        for window in windows:
            path=Path(window.get('source_window',''));actual=evidence_file(path)
            valid=valid and case in path.resolve().parents and bool(actual and actual['sha256']==window.get('source_window_sha256'))
        if valid:
            frozen=output/'frozen_middle_native_windows_audit.json';frozen.write_bytes(middle_path.read_bytes())
            evidence['middle_dense_windows']=dict(audit=middle,audit_source=evidence_file(middle_path),frozen_audit=evidence_file(frozen))
            middle_html=('<p>另有后续密集原生状态的独立证明：847.868–848.018s、855.400–855.550s，'
                '各76帧连续重放均无状态/传感器误差，中指全程安全接触，力分别1.822–2.464N和1.981–2.052N。'
                '因此最早可独立确认“到848.018s已连续接触150ms”；828.92s仍是运行中观察器记录，'
                '没有保存完整2ms窗口，不能冒充独立验证的首次稳定时刻。'
                f'{link(frozen,"后续中指150ms连续窗口审计",output)}</p>')
    return warning,middle_html,evidence


def generate(case=DEFAULT_CASE,output=DEFAULT_OUTPUT,*,end_s=None):
    case,output=Path(case).resolve(),Path(output).resolve()
    if case in output.parents or case==output:
        raise ValueError('Report output must be outside the experimental directory')
    output.mkdir(parents=True,exist_ok=True)
    horizon=available_horizon(case) if end_s is None else float(end_s)
    frozen=read_completed_prefix(case,horizon) if horizon is not None else None
    if not frozen or frozen['status']!='READY':
        raise ValueError('No complete saved prefix is available for this report')
    trace=frozen['trace'];record=_case_evidence(case,frozen)
    state=old_checkpoint(case,output/'frozen_current_coverage_checkpoint.npz',version_scope='THIS_FRESH_RUN_ONLY')
    completed=completed_run_evidence(case,state['summary'])
    full=bool(completed['full_surface_success'])
    status=('已完成整瓶，完整记录与审计匹配' if full else
            '运行已结束，整瓶未完成' if state['summary'] and not state['summary'].get('full_surface_complete') else
            '整瓶完成标志已记录，仍待审计匹配' if state['summary'] else '运行中：以下为已落盘前缀')
    summary=state['summary'] or {};checkpoint=state['checkpoint']
    saved_end=available_horizon(case)
    scope=('完整运行记录' if state['summary'] and saved_end is not None and horizon>=saved_end-1e-9
           else '指定前缀' if state['summary'] else '运行中前缀')
    plots=plot_measurements(trace,output,scope)
    sources=source_status(case,state['manifest'])
    comparison=read_json(output/'prefix_300s_comparison.json')
    videos=[];video_html=''
    for path in sorted(case.glob('fresh_*.mp4'),key=lambda p:('full_overview' not in p.name,p.name)):
        if '.rendering.' in path.name:
            continue
        if path.name=='fresh_early_recovery_1x.mp4' and (case/'fresh_early_recovery_v2_1x_render.json').exists():
            continue
        html,info=prefix_video(case,output,path.name)
        if info and info['source_matches']:
            label=('全程概览：保留全部抬离和失联，20×播放' if 'full_overview' in path.name else
                   '腕调整与调形的失败尝试：1×完整保留' if 'failed_combo' in path.name else
                   '无名指尝试失败后，中指随腕滑动与局部寻触回触：1×' if 'middle_return805' in path.name else
                   '目标拇指实际调形回触及随后间断：1×' if 'thumb_target679' in path.name else
                   '目标无名指短调形与随后接触保持：1×' if 'ring_target530' in path.name else
                   '换面后的实际贴滑：1×' if 'post_facechange' in path.name else '实际手指主动回触：1×')
            video_html+=f'<h3>{label}</h3>'+html
            if 'failed_combo' in path.name:
                video_html+=('<p class="warning">本片段内中指腕调整＋调形未获得实际接触，食指腕调整后的手指路径重规划失败。'
                    '动作确实执行过，但不能计作回触成功；完整失败过程和随后继续滑动均保留。</p>')
            if 'middle_return805' in path.name:
                video_html+=('<p class="caption">完整路径的目标是无名指，它未恢复。中指是先随腕姿态改变接触，再失联，'
                    '随后在腕继续滑动与中指局部寻触共同进行时重新接触；不能称固定腕单指调形成功。</p>')
            if 'thumb_target679' in path.name:
                video_html+=('<p class="warning">拇指达到原生执行器150ms稳定判据，但随后30s采样安全接触仅21.24%，'
                    '最长失联5.70s。达到首次稳定恢复不等于持续参与已解决。</p>')
            if 'ring_target530' in path.name:
                video_html+=('<p class="caption">无名指532.768s首触、532.922s执行器记录稳定；实际关节运动范围最大1.56°，'
                    '之后29.99s保存样本全部安全接触。微小实际调整也可能恢复接触，不能夸大为大幅关节运动。</p>')
        if info:videos.append(info)
    if not video_html:video_html='<p>本轮实际回触片段待渲染；此处不使用旧版本录像。</p>'
    m=record['metrics'];rows=[]
    for finger in m['per_finger']:
        rows.append(f'<tr><td>{NAMES[finger["finger"]]}</td><td>{100*finger["contact_fraction"]:.2f}%</td>'
            f'<td>{100*finger["safe_moving_contact_fraction"]:.2f}%</td><td>{finger["longest_absence_s"]:.2f} s</td>'
            f'<td>{finger["stable_safe_return_count"]}</td></tr>')
    episodes=[]
    for item in record['recovery']['episodes']:
        angles=' / '.join(f'{v:.1f}' for v in item['actual_joint_range_deg'] or [])
        follow=item.get('after_terminal_contact',item['followup_contact'])
        retained=(f'{100*follow["safe_contact_fraction"]:.1f}% / {follow["longest_absence_s"]:.2f}s失联'
                  if follow else '尚无结束后的完整观测窗口')
        first=(f'{item["first_native_contact_event_s"]:.3f}' if item['first_native_contact_event_s'] is not None else '未发生')
        terminal_label=str(item['terminal_status'])
        if item.get('terminal_reason'):
            terminal_label+=' / '+str(item['terminal_reason'])
        episodes.append(f'<tr><td>{NAMES[item["finger"]]}</td><td>{item["begin_s"]:.3f}–{item["end_s"]:.3f}</td>'
            f'<td>{angles}</td><td>{first}</td><td>{escape(terminal_label)}</td><td>{retained}</td></tr>')
    gaps=record['phase_gap_attribution']['longest_whole_hand_absence_intervals']
    gap_rows=''.join(f'<tr><td>{g["start_s"]:.2f}–{g["end_s"]:.2f}</td><td>{g["duration_s"]:.2f}s</td>'
        f'<td>{escape(" / ".join(f"{p} {d:.2f}s" for p,d in g["phase_durations_s"].items()))}</td></tr>' for g in gaps[:5])
    checkpoints=(f'固定 checkpoint：{checkpoint["time_s"]:.2f} s，面积 {100*checkpoint["measured_coverage_fraction"]:.2f}%，'
        f'{checkpoint["covered_vertices"]}/{checkpoint["full_vertex_count"]} 表面点，'
        f'{checkpoint["visited_required_goals"]}/{checkpoint["full_required_goal_count"]} 必访点。' if checkpoint else '尚无覆盖 checkpoint。')
    if checkpoint and abs(checkpoint['time_s']-float(trace['time_s'][-1]))>1e-9:
        checkpoints=('另行读取的覆盖 checkpoint 时间与本页轨迹窗口不同，以下数值不计入本页 '
                     f'0–{trace["time_s"][-1]:.2f}s 的统计：'+checkpoints)
    progress=state['progress'] or {}
    progress_text=(f'另列最新进度：{progress.get("time_s",0):.2f} s，面积 {100*progress.get("coverage",0):.2f}%，'
        f'阶段 {escape(str(progress.get("phase","未知")))}。它不替换上面的固定记录。')
    terminal=(f'最终停止：{escape(str(summary.get("stop_reason")))}，执行 {summary.get("executed_s",0):.3f}s；'
        f'最终面积 {100*summary.get("final_covered_area_fraction",0):.5f}%，'
        f'{summary.get("covered_vertex_count")}/{summary.get("vertex_count")} 表面点，'
        f'{summary.get("visited_required_goals")}/{summary.get("required_goal_count")} 必访点。'
        if summary else '尚无最终 summary，不能推断完整运行结果。')
    adverse_html,middle_audit_html,independent_evidence=independent_followup_evidence(case,output,float(trace['time_s'][-1]))
    behavior=read_json(output/'snapshot900_behavior.json')
    behavior_html=''
    if (behavior and Path(behavior.get('case','')).resolve()==case
            and float(trace['time_s'][-1])>=behavior['horizon_s']):
        counts=behavior['attempt_counts'];focus_rows=[]
        for index,item in enumerate(behavior['middle_focus']):
            following=item['following_30s_contact']
            focus_rows.append(f'<tr><td>{"随无名指腕调整接触" if index==0 else "腕滑动＋中指局部寻触"}</td>'
                f'<td>{item["first_saved_actual_contact_s"]:.2f}s</td>'
                f'<td>{item["first_saved_native_stable_age_at_least_150ms_s"]:.2f}s / {item["native_stable_age_at_confirmation_s"]:.3f}s</td>'
                f'<td>{100*following["safe_contact_fraction"]:.2f}%</td><td>{following["longest_absence_s"]:.2f}s</td></tr>')
        behavior_html=('<section><h2>固定900秒：恢复成功与失败分别记账</h2>'
            f'<p>面积 {100*behavior["coverage"]:.5f}%；已执行完 {counts["completed"]} 个完整恢复尝试，'
            f'其中执行器记录目标指稳定恢复 {counts["native_stable_event_success"]} 次，失败 {counts["failed"]} 次。'
            '这些计数只来自真实执行后的事件，静态路径accepted不计成功。900s是运行中快照，不是完整实验。</p>'
            '<p>810–830s中指2N接触不是“无名指恢复成功”。先是非目标中指随无名指腕调整进入接触，'
            '之后再次失联，再由继续腕滑动与中指局部寻触共同回触。第二段中指实际关节净变化 '
            '−0.38 / −7.98 / +4.25 / +7.51°，腕也明显运动。</p>'
            '<table><tr><th>中指过程</th><th>首个保存接触</th><th>原生观察器达到稳定的保存时刻 / 年龄</th>'
            '<th>随后30s安全接触</th><th>最长失联</th></tr>'+''.join(focus_rows)+'</table>'
            '<p class="caption">稳定年龄是运行中的2ms观察器日志；此离线分析没有独立重放首次150ms窗口，'
            '因此它与独立原生审计分开表述。后续30s占比来自100Hz保存样本，可能遗漏短间断。</p>'
            +middle_audit_html+
            f'<p>{link(output/"snapshot900_behavior.json","900秒全部尝试、两次中指回触与源哈希",output)} · '
            f'{link(output/"snapshot900_behavior.py","可重跑分析脚本",output)}</p></section>')
    figure=output/'fresh_actual_contact_three_views.png'
    figure_source=read_json(figure.with_suffix('.json'))
    figure_matches=bool(figure_source and Path(figure_source.get('source_case','')).resolve()==case
        and figure_source.get('trace_last_time_s',float('inf'))<=float(trace['time_s'][-1])+1e-9)
    surface_html=(f'<img src="{relative_url(figure,output)}" alt="本轮实际接触轨迹"/>'
        '<p class="caption">实际接触按同指连续段绘制，失联不连线；图的精确前缀与来源以图注为准。'
        f'{link(figure.with_suffix(".json"),"绘图来源与显示规则",output)} · {link(figure.with_suffix(".pdf"),"下载PDF",output)}</p>'
        if figure.exists() and figure_matches else '<p>本轮接触分布图等待具有匹配审计与覆盖 mask 的冻结记录。</p>')
    evidence=dict(created_utc=datetime.now(timezone.utc).isoformat(),case=str(case),scope='ONE_FRESH_RUN_ONLY',
        full_surface_claim=full,trace_horizon_s=float(trace['time_s'][-1]),record=record,
        coverage_state=state,source_integrity=sources,completed=completed,videos=videos,
        independent_followup_evidence=independent_evidence,
        historical_coverage_credit=False,comparison_source=evidence_file(output/'prefix_300s_comparison.json'))
    (output/'fresh_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>mustard 新版完整运行观察</title>
<style>body{{margin:auto;max-width:1160px;padding:30px 24px;font:16px/1.65 system-ui;color:#203542;background:#f4f7f9}}h1,h2{{line-height:1.3}}section{{background:white;border:1px solid #dce4e9;border-radius:14px;padding:22px;margin:20px 0}}.lead{{font-size:21px}}.warning{{border-left:4px solid #d18a38;padding:10px;background:#fff8ec}}.critical{{border:2px solid #b44138;background:#fff2ed}}.critical h2{{color:#982d26}}.caption{{font-size:14px;color:#566879}}video,img{{width:100%;border-radius:8px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{border-bottom:1px solid #dce4e9;padding:9px;text-align:left}}th{{background:#eaf0f4}}.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}a{{color:#176f91}}</style>
<h1>mustard：连续滑动与失联手指主动恢复</h1><p class="lead">{status}</p>
<p>本页只使用本轮 <code>{escape(str((state['manifest'] or {}).get('protocol','未知')))}</code> 的实际记录。覆盖分母保留原始 8192 表面点、388 必访点；不拼接旧运行或专项试验覆盖。</p>
{adverse_html}
<section><h2>本轮看到什么</h2><p>固定轨迹 0–{trace['time_s'][-1]:.2f}s：面积 {100*record['coverage']:.2f}%，安全接触 {100*m['safe_contact_continuity_fraction']:.2f}%，安全移动接触 {100*m['safe_moving_contact_fraction']:.2f}%。最长同指移动接触段 {m['longest_sliding_bout_s']:.2f}s。</p><p>{terminal}</p><p>{checkpoints}</p><p class="caption">{progress_text}</p><p>完整运行审计匹配：{completed['audit_verified']}；冻结源快照 {sources['count']} 项全部匹配：{sources['all_frozen_snapshots_match']}。</p></section>
{behavior_html}
<section><h2>实际手指运动与回触</h2>{video_html}<p>记录了 {record['recovery']['native_stable_event_count']} 次原生稳定回触事件，当前完整运行审计已匹配的事件为 {record['recovery']['audited_native_stable_event_count']} 次。两者是证据状态的区分；仅有路径指令不计回触。</p><div class="scroll"><table><tr><th>手指</th><th>实际恢复阶段 / s</th><th>四关节实际范围 / °</th><th>首次接触 / s</th><th>终态及原因</th><th>结束后最多30s安全接触</th></tr>{''.join(episodes)}</table></div><p class="caption">事件采用原生 2 ms 时钟；后续参与和运动曲线来自 100 Hz 记录，可能遗漏短于10ms的失联。失败后的接触也照实记录，不回填为本次恢复成功。</p><img src="fresh_actual_joint_motion.png" alt="四指实测关节角度变化"/></section>
<section><h2>连续参与与必要换区</h2><table><tr><th>手指</th><th>实际接触率</th><th>安全移动接触率</th><th>最长失联</th><th>采样稳定重触</th></tr>{''.join(rows)}</table><p>采样稳定重触计数与原生执行器事件采用不同观察窗口，不能混为一次恢复的两种总数。移动判据使用实测接触点切向位移并结合指尖位移，仍不能区分滚动与滑移。</p><img src="fresh_measured_timeline.png" alt="本轮覆盖、接触力与接触参与"/><table><tr><th>最长全手失联窗口 / s</th><th>时长</th><th>实际阶段归因</th></tr>{gap_rows}</table><p class="caption">短暂失联和必要抬离不自动判失败；所有离面时间保留，继续核对如何恢复。</p></section>
<section><h2>本轮实际接触在物体上的分布</h2>{surface_html}<p>面积覆盖按实测接触在原 mesh 上的 25 mm 测地邻域记账，不代表8192个顶点都被指尖直接触碰。</p></section>
<section><h2>同初态 0–300 s 比较</h2>{comparison_table(comparison,case)}<p>{link(output/'prefix_300s_comparison.json','精确窗口全部指标与初态核验',output) if comparison else ''}</p></section>
<section><h2>证据与范围</h2><p>{link(output/'fresh_evidence.json','本页全部源哈希、实测指标与状态',output)} · {link(case/'manifest.json','本轮冻结协议',output)} · {link(case/'events.json','本轮实际事件',output)}</p><p>{link(GENERATED/'mustard_coordinated_review_20261016/index.html','历史v3.5报告与旧终态续接，单独查看',output)}。历史六段链的完整覆盖不计入本轮。</p></section></html>'''
    temporary=output/'index.html.tmp';temporary.write_text(html,encoding='utf-8');temporary.replace(output/'index.html')
    return dict(report=str(output/'index.html'),horizon_s=float(trace['time_s'][-1]),full_surface_claim=full,audit_verified=completed['audit_verified'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case',type=Path,nargs='?',default=DEFAULT_CASE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--end',type=float)
    args=parser.parse_args()
    print(json.dumps(generate(args.case,args.output,end_s=args.end),ensure_ascii=False))
