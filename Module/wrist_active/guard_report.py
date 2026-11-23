"""Final comparison including the paired force-priority controller revision."""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from .evaluate import Reference,movement,reconstruct
from .experiment import ACTIVE2,ROOT,digest,save_json
from .sweep_experiment import OUT
from .sweep_report import NAMES,OBJECTS,draw,pc,read,seconds,snap
from .sweep_score import ref_mesh
from .sweep_suite import jobs


def same_three_seconds(path, end=24000):
    with np.load(path/'trace.npz') as tr:
        q=tr['qpos'][:end+1]
        moving=movement(q)[1:]
        light=tr['region_load_n'][1:end+1].max(1)<8.
        pads=tr['pad_valid'][1:end+1]
    retained=np.zeros(end,dtype=bool)
    for trio in combinations(range(4),3):
        active=moving & light & np.all(pads[:,trio],axis=1)
        edges=np.flatnonzero(np.diff(np.r_[False,active,False])).reshape(-1,2)
        for start,stop in edges[edges[:,1]-edges[:,0]>=100]:
            retained[start:stop]=True
    return float(retained.sum()*.01)


def report(root):
    guard=root/'force-priority'
    main=guard/'runs/pear/active-02/sweep-dp'
    initial=root/'runs/pear/active-02/sweep-dp'
    archive=root/'references/pear/active-02'
    candidates={'sweep-dp':main,'history-dp':archive/'history-dp',
                'aude-mcc':archive/'aude-mcc','aude-single-all-mcc':archive/'aude-single-all-mcc'}
    data={a:read(p/'scores.json') for a,p in candidates.items()}
    draw(guard,candidates,data,wrist_label='Wrist GPIS + DP (force priority)')
    unguarded=read(initial/'scores.json')
    record=read(main/'run.json')
    timed=read(guard/'paced/pear/active-02/sweep-dp/scores.json')
    paced=read(guard/'paced/pear/active-02/sweep-dp/run.json')
    causal=read(guard/'causal-replay.json')
    seed=read(guard/'new-seed/pear/active-02/sweep-dp/scores.json')
    u=data['sweep-dp']
    v=data['aude-mcc']
    assert read(ACTIVE2/'aude-mcc/run.json')['initial_state_sha256']==record['initial_sha256']
    assert read(initial/'run.json')['initial_sha256']==record['initial_sha256']
    assert causal['trace_sha256']==record['trace_sha256'] and causal['packets']==record['steps']
    source=main/'trace.npz'
    assert record['trace_sha256']==digest(source)
    timelines=read(main/'recoveries.json')
    with np.load(source) as trace:
        peak=trace['region_load_n']
        idx=np.unravel_index(peak.argmax(),peak.shape)
        k,f=np.nonzero(trace['pad_valid']&(peak.max(1)<8.)[:,None])
        pts=trace['points_m'][k,f]
        _,ids=np.unique(np.floor(pts/.0045).astype(int),axis=0,return_index=True)
        first=np.sort(k[ids])
        select=k<=24000
        ridge_path=guard/'pear-ridge-240.json'
        if ridge_path.exists() and read(ridge_path)['trace_sha256']==record['trace_sha256']:
            ridge=read(ridge_path)['signed_ridge']
        else:
            ridge=reconstruct(pts[select],trace['normals_outward'][k[select],f[select]],
                              Reference(ref_mesh('pear','active-02')),guard/'ridge-240s.ply','ridge')
            save_json(ridge_path,dict(trace_sha256=record['trace_sha256'],signed_ridge=ridge))
    plans=[json.loads(x) for x in (main/'plans.jsonl').read_text(encoding='utf-8').splitlines()]
    hardware=['食指','中指','无名指','拇指','手掌']
    post=0
    for i,e in enumerate(timelines):
        if e['restored_two_s'] is None:
            continue
        at=round(e['restored_two_s']*100)
        bound=min(at+501,record['steps']+1,round(timelines[i+1]['start_s']*100) if i+1<len(timelines) else record['steps']+1)
        post+=int(np.searchsorted(first,bound)>np.searchsorted(first,at))
    attempts=u['recovery']['attempts']
    restored=u['recovery']['two']
    t=u['thresholds_s']
    s=snap(u,240)
    branch=read(guard/'brake-branch/result.json')
    assert branch['source_trace_sha256']==unguarded['spec']['trace_sha256']
    finish=timed['timing']['finish_from_release']
    due=1-finish['over']/finish['n']
    sub=[('旧局部腕轨＋DP',data['history-dp']),('全局腕轨＋DP，常规卸载',unguarded),
         ('全局腕轨＋DP，过载优先',u),('Active-2 Aude＋MCC',v),
         ('Active-2 Aude＋DP',data['aude-single-all-mcc'])]
    def link(path,label):
        return f'[{label}]({path.relative_to(root).as_posix()})'
    def area_row(label,r):
        a=snap(r,240)
        return (f"| {label} | {pc(a['coverage'])} | {pc(a['mean_coverage'])} | {pc(a['signed']['fscore'])} | "
                f"{a['signed']['mean_error_mm']:.2f} | {pc(a['low2'])} | {pc(a['low3'])}（{a['low3_s']:.1f} s） | "
                f"{pc(a['sustained3'])} | {a['over40_s']:.2f} |")
    tab=['| 系统 | 10 mm 覆盖 | 0–240 秒平均覆盖 | 有符号 F@5 | 对称误差 mm | 轻载≥2指 | 轻载≥3指 | 持续≥3指 | >40 N 秒 |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    lines=['# 仅规划手腕的 GPIS 主动探索','',
           f"在 Active-2 的同一只梨、同一初态上，过载优先的手腕规划完成 1,129.32 秒闭环，最终覆盖 **{pc(snap(u,1129.32)['coverage'])}**，"
           f"旧腕轨为 **{pc(snap(data['history-dp'],1129.32)['coverage'])}**。前 240 秒，新腕轨达到 **{pc(s['coverage'])} 覆盖、{pc(s['signed']['fscore'])} 有符号重建 F@5、{pc(s['low3'])} 轻载三指时间**。"
           f"完整 Active-2 Aude＋MCC 在同一个窗口为 {pc(snap(v,240)['coverage'])}、{pc(snap(v,240)['signed']['fscore'])}、{pc(snap(v,240)['low3'])}。",'',
           '![同一只梨的实测覆盖、重建及接触曲线](force-priority/pear-comparison.png)','',
           '## 两处会让手腕停下来或继续顶压的控制逻辑','',
           '原局部规划器在双指支撑不足时返回上一个支撑点。梨在第 256.25 秒后便一直重复压入与卸载，触摸面积停在 41.74%。现在以已知夹具原点安排连续的全局扫线。掉到一根有效指尖时仍向下一段推进。达到持续失触、地图停更或长时间过载条件时，安排径向退出、横向转移和带载荷反馈的重新接近。下一遍扫线根据已测触点方向的空缺选方位。','',
           '第一轮修正解决了覆盖停滞，却在梨的第 235.16 秒产生 693.2 N 的掌部采样峰值。该时刻处于退出阶段，切向移动仍在继续。第二轮加入独立的 100 Hz 过载优先级。任一分区达到 16 N 或手掌达到 6 N 时保持切向位置和旋转，沿夹具径向外退，速度上限 0.16 m/s。连续五帧满足分区小于 5 N 且手掌小于 2 N 后，腕轨从当前姿态重新规划。指端 MCC 和 DP 继续运行。',
           '',f"完整梨轨迹的五区实测峰值从 {snap(unguarded,1129.32)['peak_n']:.1f} N 变为 **{snap(u,1129.32)['peak_n']:.1f} N**，"
           f"新峰值位于 {idx[0]*.01:.2f} 秒的{hardware[idx[1]]}分区。累计超过 40 N 的时间从 {snap(unguarded,1129.32)['over40_s']:.2f} 秒变为 {snap(u,1129.32)['over40_s']:.2f} 秒。"
           '制动阈值决定何时响应。实测峰值还取决于接触刚度、当前速度和采样周期。 [高载优先逻辑](../../wrist_active/load_guard.py) · [初版完整对照](SWEEP_REPORT.md)','',
           f"对第 235 秒的掌部事件另做同状态分叉。先重放 {branch['identical_prefix_frames']:,} 帧，腕令和全部关节位置最大分量误差为 {branch['max_prefix_error']:.2g}。"
           f"在第 234.00 秒启用腕部制动，保留相同的名义指令和实时指端 MCC。到 237.50 秒，两支的掌部峰值分别为 "
           f"{branch['original']['peak_palm_n']:.1f} N 和 {branch['guarded']['peak_palm_n']:.1f} N。分区超过 40 N 的累计时间分别为 "
           f"{branch['original']['over40_s']:.2f} 秒和 {branch['guarded']['over40_s']:.2f} 秒。 [事件分叉原始数据](force-priority/brake-branch/result.json)",'',
           '## 梨的配对结果','',
           '初态、仿真时钟、指尖有效力下限 0.2 N、五区均低于 8 N 的评分门槛，以及重建器全部对齐。覆盖按实触点的 10 mm 测地邻域并集计算。每次重建最多使用 1,000 个实测代表点。F@5 取整物体表面和重建面各 50,000 个采样点，在 5 mm 距离内计算精确率和召回率的调和平均。平均覆盖为曲线下面积除以时间，持续三指要求至少 1 秒连续轻载接触。','',*tab]
    for name,r in sub:
        lines.append(area_row(name,r))
    lines+=['',f"单看手腕运动的样本，240 秒内轻载三指的占比为新腕轨 {pc(s['moving_low3'])}、Active-2 Aude＋MCC {pc(snap(v,240)['moving_low3'])}。"
            '运动判定使用实际腕位，线速度达到 1 mm/s 或角速度达到 0.02 rad/s 即计入。'
            f"若严格要求连续至少 1 秒，其间每个 10 ms 采样均满足轻载三指，新腕轨前 240 秒的合格时长为 {s['sustained3']*240:.1f} 秒，"
            f"全程为 {snap(u,1129.32)['sustained3']*1129.32:.1f} 秒。前 240 秒的优势体现为较多次短时接触的累计。",'',
            '| 系统 | 到50% 秒 | 到70% 秒 | 到90% 秒 | 到95% 秒 | 到99% 秒 | 600 秒覆盖 | 最终覆盖 | 最终 F@5 | 全程轻载三指 | 全程峰值 N |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in sub:
        final=snap(r,1129.32)
        h=' | '.join(seconds(r['thresholds_s'][str(i)]) for i in (50,70,90,95,99))
        lines.append(f"| {name} | {h} | {pc(snap(r,600)['coverage'])} | {pc(final['coverage'])} | {pc(final['signed']['fscore'])} | {pc(final['low3'])} | {final['peak_n']:.1f} |")
    oldridge=read(root/'pear-ridge-240.json')['scores']
    timing_archive=read(ROOT.parent/'active-02-timing_20261121/module-timing.json')
    aude_timing=next(z for z in timing_archive['runs'] if z['experiment']=='active-02-a' and z['arm']=='aude-mcc')
    fits=aude_timing['exclusive_functions']['joint_planning/JointIK.fit']
    projection=aude_timing['exclusive_functions']['joint_planning/TactileProjection.solve']
    ss=snap(seed,240)
    lines+=['',f"保留初态并把策略采样种子换为 20261122，240 秒覆盖、F@5、轻载三指为 {pc(ss['coverage'])}、{pc(ss['signed']['fscore'])}、{pc(ss['low3'])}。"
            f"第二种子首达 70%／95% 为 {seconds(seed['thresholds_s']['70'])}／{seconds(seed['thresholds_s']['95'])} 秒，600 秒覆盖 {pc(snap(seed,600)['coverage'])}，该轮五区采样峰值 {snap(seed,600)['peak_n']:.1f} N。"
            ' [第二种子](force-priority/new-seed/pear/active-02/sweep-dp/scores.json)','',
            f"备用 I01-A R-GPIS 重建器以相同 240 秒触点重新评分。过载优先的腕轨 F@5 为 {pc(ridge['fscore'])}，"
            f"常规卸载腕轨 {pc(oldridge['sweep-dp']['fscore'])}，旧局部腕轨 {pc(oldridge['history-dp']['fscore'])}，"
            f"完整 Aude＋MCC {pc(oldridge['aude-mcc']['fscore'])}。主表使用有符号 GPIS。 [备用重建结果](force-priority/pear-ridge-240.json)",'',
            '## 手腕和手指各负责什么','',
            '手腕规划器在无物体碰撞几何的手模型内，根据当前实测关节计算四个指尖的位置。用累计轻载触点及法向拟合全局二次隐式面和局部 Matérn GPIS。每 200 ms，取四个三指组合，各做一次 3×3 线性求解，修正腕高和两个倾角，再与原腕姿比较。候选依据第二、第三指的接触带、GPIS 方差、离既有触点的距离和预计压入深度评分。可触概率加权的空间新颖度和后验标准差用作信息增益的近似。最终输出 SE(3) 腕段与约 200 ms 的运动前瞻。','',
            'single_all 使用发布权重、10 Hz 请求频率和已有指端 MCC。手指名义角限于初始四指握姿 ±0.48 rad 及物理范围，变化率限于 1.8 rad/s。手腕候选中需要在线优化的手指关节变量为零。局部可达性使用当前实测手型外推。首轮历史手型、刚性外推和 I01-A 的原始对照均保留在先前报告。 [腕部求解及高度控制](../../wrist_active/sweep.py) · [手指动作缓冲](../../wrist_active/buffered_fingers.py) · [前序实验](../gpis_wrist_active_20261121/REPORT.md)','',
            f"1,129.32 秒的原始 Active-2 Aude＋MCC 计时回放记录到 {fits['calls']:,} 次 JointIK.fit 和 {projection['calls']:,} 次联合触点投影。"
            f"前者在该归档计时条件下单次 P99 为 {fits['p99_ms']:.1f} ms。新腕部规划路径这两类调用均为零。"
            '各系统的采集循环和测时环境有差异。 [Active-2 的分模块调用计数](../active-02-timing_20261121/module-timing.json)','',
            '### 240 秒消融','',*tab,area_row('完整过载优先版本',u)]
    for arm in ('route-dp','no-recontact-dp','sweep-mcc'):
        f=main.parent/arm/'scores.json'
        lines.append(area_row(link(f,NAMES[arm]),read(f)))
    no_recovery=snap(read(main.parent/'no-recontact-dp/scores.json'),240)
    lines+=['','去局部 GPIS 组保留全局地图、扫线、载荷反馈、复触和 DP。关闭显式复触组保留全局推进及快速过载制动。纯 MCC 组保持初始名义指角，腕轨由本组真实触觉重新规划。','',
            f"梨全程触发 {attempts} 次显式复触，{restored} 次在六秒观察窗及下一次重试前恢复至少 200 ms 连续轻载双指，其中 {post} 次在恢复后五秒内且下次重试前摸到新单元。"
            f"快环制动另触发 {plans[-1]['brake_events']} 次（截至最后一次规划采样）。在相同的 240 秒窗口，关闭显式复触仍达到 {pc(no_recovery['coverage'])} 覆盖，打开后为 {pc(s['coverage'])}。"
            '两组都能靠全局扫线越过旧的停滞点。 [复触事件](force-priority/runs/pear/active-02/sweep-dp/recoveries.json)','',
            '| 同一腕令回放，240 秒 | DP 轻载三指 | 纯 MCC 轻载三指 | 纯 MCC 两指且两组移动 | DP 同一对指再加指 | 条件占比 | 只遮去这些额外触点后的覆盖 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for obj,start,label in [('pear','active-02','梨'),('ovoid','start-03','卵形体 C')]:
        p=guard/'paired-wrist'/obj/start/'result.json'
        x=read(p)
        lines.append(f"| {link(p,label)} | {pc(x['dp_low3'])} | {pc(x['mcc_low3'])} | {x['mcc_exact2_s']:.2f} s | "
                     f"{x['retained_plus_s']:.2f} s | {pc(x['conditional']) if x['conditional'] is not None else '分母为零'} | {pc(x['masked_cover'])} |")
    lines+=['','同腕令回放用固定的初始名义指角执行纯 MCC。条件占比的分母是两组手腕都在动且纯 MCC 恰好有两指轻载接触的时长。分子还要求 DP 保留相同两指并增加至少一指。表内同时列出分母秒数，便于判断这个条件在实际腕轨上有多常见。','',
            '## 七组 240 秒的完整 Active-2 Aude 对照','',
            'Aude 基线从同一模型和四指初态启动，使用 Active-2 的联合腕指求解、全局扫查、手型约束和反馈。移植后先在梨回放 20 秒，2,001 帧的六组关键量和原始归档逐值相等。各组按相同 <8 N 规则评分，原 Active-2 基线在线可吸收至 12 N 的样本。','',
            '| 物体／起点 | 新腕覆盖 | Aude 覆盖 | 新腕 F@5 | Aude F@5 | 新腕轻载三指 | Aude 轻载三指 | 新腕峰值 N | Aude 峰值 N |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for trial in jobs():
        obj,start=trial['obj'],trial['start']
        if obj=='pear':
            continue
        key=Path(obj)/start
        p=guard/'runs'/key/'sweep-dp'
        b=root/'strong-baseline'/key/'aude-mcc'
        assert read(p/'run.json')['initial_sha256']==read(b/'run.json')['initial_sha256']
        a,c=snap(read(p/'scores.json'),240),snap(read(b/'scores.json'),240)
        previous=(ROOT/'smooth-transfer' if trial['group']=='smooth' else ROOT)/'runs'/key/'rigid-dp/score.json'
        old_cover=read(previous)['snapshots'][-1]['coverage10']
        same=[same_three_seconds(path) for path in (p,b)]
        rows.append(dict(obj=obj,start=start,coverage=[a['coverage'],c['coverage']],
                         fscore=[a['signed']['fscore'],c['signed']['fscore']],low3=[a['low3'],c['low3']],
                         sustained_three_s=[240*a['sustained3'],240*c['sustained3']],
                         moving_same_three_s=same,
                         previous_rigid_coverage=old_cover))
        lines.append(f"| {OBJECTS[obj]} {start[-2:]} | {link(p/'scores.json',pc(a['coverage']))} | "
                     f"{link(b/'scores.json',pc(c['coverage']))} | {pc(a['signed']['fscore'])} | {pc(c['signed']['fscore'])} | "
                     f"{pc(a['low3'])} | {pc(c['low3'])} | {a['peak_n']:.1f} | {c['peak_n']:.1f} |")
    means={k:np.mean([r[k] for r in rows],axis=0).tolist() for k in ('coverage','fscore','low3')}
    smooth=[r for r in rows if r['obj']!='meat-can']
    carry=np.mean([r['sustained_three_s'] for r in smooth],axis=0)
    same_avg=np.mean([r['moving_same_three_s'] for r in smooth],axis=0)
    lines+=['| **七组等权平均** | '+' | '.join(pc(n) for k in means for n in means[k])+' | — | — |','',
            f"相同七组中，上一轮主场景表现最好的实测手型腕规划 rigid-dp 平均覆盖为 {pc(np.mean([r['previous_rigid_coverage'] for r in rows]))}，"
            f"改用本轮带制动的腕轨后为 {pc(means['coverage'][0])}，完整 Aude＋MCC 为 {pc(means['coverage'][1])}。",'',
            f"七组中，新腕轨的覆盖、F@5、轻载三指分别有 "
            + '、'.join(f"{sum(r[k][0]>r[k][1] for r in rows)}/7" for k in means)
            +' 组超过完整 Aude＋MCC。梨和午餐肉罐 A 进入过开发试跑。过载优先修订由主梨载荷曲线触发，所有修订后结果共用同一组参数。按物体和起点重复使用相同初态，因而不能视为随机独立物体总体的统计估计。','',
            '### 连续轻载三指的时长','',
            '每段至少连续 1 秒，每个 10 ms 样本均有至少三根有效指尖且全部分区低于 8 N。后两列还要求整个片段是相同的三根手指且每帧腕部在移动。如有四指同时达标，重叠秒数只计一次。以下数字为每组 240 秒内的累计时长。','',
            '| 物体／起点 | 新腕轨，秒 | Aude＋MCC，秒 | 新腕轨移动且保持同三指，秒 | Aude 移动且保持同三指，秒 |', '|---|---:|---:|---:|---:|',
            *[f"| {OBJECTS[r['obj']]} {r['start'][-2:]} | {r['sustained_three_s'][0]:.2f} | {r['sustained_three_s'][1]:.2f} | "
              f"{r['moving_same_three_s'][0]:.2f} | {r['moving_same_three_s'][1]:.2f} |" for r in rows], '',
            f"两个椭球起点、扁椭球、卵形体和超椭球共五组平滑曲面上，新腕轨的连续三指接触时长有 {sum(r['sustained_three_s'][0]>r['sustained_three_s'][1] for r in smooth)}/5 组更长，"
            f"每组平均 {carry[0]:.2f} 秒，Aude＋MCC 为 {carry[1]:.2f} 秒。移动中保持相同三指分别为 {same_avg[0]:.2f} 秒和 {same_avg[1]:.2f} 秒。"
            '午餐肉罐的两个起点两种腕轨均未出现一整秒连续轻载三指。','',
            '## 真实时钟下的闭环与部署边界','',
            '另行在没有其他本实验仿真或评分进程同时运行的情况下，按真实时钟闭环仿真 120 秒。Windows 版使用进程自己的高分辨率等待定时器。腕部工作线程按 5 Hz 规划，GPU 工作进程按 10 Hz 生成动作块，均提前 50 ms 提交。100 Hz 伺服可使用上一块指令，腕部计算迟到时保持切向端点并继续响应载荷。计时含触觉、调度、MCC、每帧 5 个 MuJoCo 物理子步和记录。','',
            '| 阶段 | 样本 | P50 ms | P95 ms | P99 ms | 最大 ms | 超时 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for label,key in [('腕部规划','plan'),('DP 往返','dp_rpc'),('整帧工作时间','loop_service'),('预定释放至完成','finish_from_release')]:
        q=timed['timing'][key]
        lines.append(f"| {label} | {q['n']:,} | {q['p50']:.2f} | {q['p95']:.2f} | {q['p99']:.2f} | "
                     f"{q['maximum']:.2f} | {q['over']} / {q['n']}（{q['deadline_ms']} ms） |")
    late=[r['planning_ms'] for r in plans if r['tick']>=82932]
    logical=snap(u,120)
    lines+=['',f"使用 {paced['policy']['device']}，冷启动 {paced['setup_s']:.2f} 秒，采集阶段墙钟 {paced['acquire_after_setup_s']:.2f} 秒。"
            f"10 ms 期限按时完成 {100*due:.2f}%（超时 {finish['over']}/{finish['n']}），腕段交接迟到 {paced['late_wrist_handoffs']} 次，DP 交接迟到 {paced['missed_handoffs']} 次，跳过请求 {paced['skipped_queries']} 次，缓冲耗尽 {paced['underrun_ticks']} 帧。"
            f"实际安装了 {paced['installs']:,} 个 DP 动作块，按采集墙钟折算为 {paced['installs']/paced['acquire_after_setup_s']:.2f} Hz。"
            f"该实时时钟轨迹 120 秒覆盖 {pc(snap(timed,120)['coverage'])}、F@5 {pc(snap(timed,120)['signed']['fscore'])}、轻载三指 {pc(snap(timed,120)['low3'])}。"
            f"相同种子、按固定逻辑延迟交接的主实验在 120 秒对应为 {pc(logical['coverage'])}、{pc(logical['signed']['fscore'])}、{pc(logical['low3'])}。"
            ' [逐帧计时](force-priority/paced/pear/active-02/sweep-dp/control-timing.npz) · [系统回执](force-priority/paced/pear/active-02/sweep-dp/run.json)','',
            f"另计完整仿真最后 300 秒，成熟地图阶段的腕部规划 P99 为 {np.percentile(late,99):.2f} ms，末次规划已有 {plans[-1]['cells']:,} 个有效触觉单元。"
            f"从零使用无物体的手模型，只读实测关节和触觉，独立复现 {causal['decisions']:,} 次决策与 {causal['packets']:,} 条实际腕令和前瞻，最大腕令分量误差 {causal['max_pose_component_error']:.2g}。"
            ' [完整因果复核](force-priority/causal-replay.json)','',
            '在线使用稀疏触摸地图和局部 GPIS，最后的高分辨率网格在采集后生成。实际验证范围为固定物体、已知夹具原点和给定四指起始接触。硬件腕力控、传感延迟、机械臂运动学、通信及载荷限制尚待上机测量。','',
            '[完整梨原始分时结果](force-priority/runs/pear/active-02/sweep-dp/scores.json) · [优化前的消融及曲线](SWEEP_REPORT.md) · [冻结修订协议](force-priority/protocol.json) · [机器可读摘要](guard-summary.json) · [复现脚本](../../wrist_active/run_sweep.ps1)','',
            '```powershell','.\\Module\\wrist_active\\run_sweep.ps1 -Output Module/generated/gpis_wrist_sweep_repeat','```']
    stress=read(root/'paced/pear/active-02/sweep-dp/scores.json')['timing']['finish_from_release']
    lines[-5:-5]=[f"常规卸载版本另做过有竞争负载的压力试次。当时同一块 GPU 上同时运行多组推理，普通 Windows 等待下的 10 ms 达限率为 {pc(1-stress['over']/stress['n'])}。"
                 '两轮的计算负载、等待实现和控制版本均有变化。 [有竞争负载的逐帧结果](paced/pear/active-02/sweep-dp/scores.json)','']
    telemetry=guard/'gpu-context.json'
    if telemetry.exists():
        values=read(telemetry)
        before=[r['util_percent'] for r in values if r['phase']=='before']
        during=[r['util_percent'] for r in values if r['phase']=='closed_loop']
        if before and during:
            lines[-5:-5]=[f"硬件计数器每约三秒采样一次。本实验物理任务结束、因果复核期间的 GPU 利用率中位数为 {np.median(before):.0f}%（{len(before)} 个样本），"
                         f"随后带时钟的闭环期间为 {np.median(during):.0f}%（{len(during)} 个样本）。该指标包含设备上所有进程。"
                         ' [整卡负载采样](force-priority/gpu-context.json)','']
    dispatch=guard/'paced-dispatch/pear/active-02/sweep-dp'
    dispatch_summary=None
    if (dispatch/'scores.json').exists():
        ds=read(dispatch/'scores.json')
        dr=read(dispatch/'run.json')
        dc=read(dispatch/'dispatch.json')
        assert dc['comparison_trace_sha256']==paced['trace_sha256'] and dc['trace_sha256']==dr['trace_sha256']
        assert dc['deferred']==ds['timing']['plan']['n']-1
        with np.load(guard/'paced/pear/active-02/sweep-dp/control-timing.npz') as samples:
            phase=samples['finish_from_release_ms'][15::20]
            old_collisions=int((phase>10).sum())
        ds120=snap(ds,120)
        dd=ds['timing']['finish_from_release']
        dispatch_summary=dict(timing=ds['timing'],deadline_fraction=1-dd['over']/dd['n'],
                              installed_hz=dr['installs']/dr['acquire_after_setup_s'],
                              wrist_late=dr['late_wrist_handoffs'],finger_late=dr['missed_handoffs'],
                              skipped=dr['skipped_queries'],underrun_ticks=dr['underrun_ticks'],s120=ds120)
        lines[-5:-5]=['### 错开腕部计算与动作块交接','',
            f"上面的 120 秒试次中，{finish['over']} 个超时帧有 {old_collisions} 个发生在每 20 帧的同一相位。"
            '该帧既提交腕部规划，也可能安装新的手指动作块。补充试次在同一帧保存原触觉快照，等该帧完成物理和传感更新后才唤醒腕部计算线程。'
            'Python 线程切换间隔设为 0.5 ms。腕段的预定执行时间、指令缓冲、控制参数和冻结权重均与上表相同。','',
            '| 120 秒实际时钟 | 10 ms 内完成 | 整帧 P99／最大 ms | 腕计划 P99 ms | 腕交接迟到 | 实际 DP Hz | DP 交接迟到／跳过 | 指令缓冲耗尽帧 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']+[
            f"| {name} | {100*(1-t['finish_from_release']['over']/t['finish_from_release']['n']):.2f}% | "
            f"{t['finish_from_release']['p99']:.2f}／{t['finish_from_release']['maximum']:.2f} | {t['plan']['p99']:.2f} | "
            f"{rec['late_wrist_handoffs']} | {rec['installs']/rec['acquire_after_setup_s']:.2f} | "
            f"{rec['missed_handoffs']}／{rec['skipped_queries']} | {rec['underrun_ticks']} |"
            for name,t,rec in [('原异步调度',timed['timing'],paced),('错开后',ds['timing'],dr)]
        ]+['',f"错开调度的腕部后台任务有 {dc['deferred']} 次等待了当帧传感更新。相同初态、相同种子下，"
           f"该次按墙钟运行的仿真在 120 秒获得 {pc(ds120['coverage'])} 覆盖、{pc(ds120['signed']['fscore'])} 有符号 F@5 和 {pc(ds120['low3'])} 轻载三指时间。"
           f"同一仿真时间的 Active-2 Aude＋MCC 为 {pc(snap(v,120)['coverage'])}、{pc(snap(v,120)['signed']['fscore'])} 和 {pc(snap(v,120)['low3'])}。"
           '两次墙钟运行的动作块实际到达时间不同，触觉反馈和后续轨迹也发生了变化。 [补充闭环计时与评分](force-priority/paced-dispatch/pear/active-02/sweep-dp/scores.json) · [错开调度实现](../../wrist_active/sweep_dispatch.py)','']
    save_json(root/'guard-summary.json',dict(pear={'thresholds':t,'s240':s,'final':snap(u,1129.32)},
              seed={'thresholds':seed['thresholds_s'],'s240':ss},transfers=rows,transfer_means=means,
              timings=timed['timing'],deadline_fraction=due,buffer_underruns=paced['underrun_ticks'],dispatch=dispatch_summary,
              clock_source_sha256=digest(Path(__file__).with_name('sweep_paced.py')),
              recoveries={'triggered':attempts,'restored':restored,'new_after_restore':post},causal=causal))
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    torch.set_num_threads(1)
    report(p.parse_args().root)
