"""Assemble the matched wrist, reconstruction, recovery and deadline results."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate import Reference,reconstruct
from .experiment import ACTIVE2,ROOT,digest,save_json
from .sweep_experiment import OUT,SMOOTH
from .sweep_score import ref_mesh
from .sweep_suite import jobs


NAMES={'sweep-dp':'新腕规划＋single_all＋MCC','history-dp':'上一版历史手型＋single_all＋MCC',
       'aude-mcc':'Active-2 Aude＋MCC','aude-single-all-mcc':'Active-2 Aude＋single_all＋MCC',
       'route-dp':'去局部 GPIS 手腕修正＋single_all','sweep-mcc':'新腕规划＋纯 MCC',
       'no-recontact-dp':'关闭显式复触＋single_all'}
OBJECTS={'ellipsoid':'椭球','meat-can':'午餐肉罐','oblate':'扁椭球','ovoid':'卵形体','superellipsoid':'超椭球'}
def read(p):
    return json.loads(p.read_bytes())

def snap(r,t):
    return next(s for s in r['snapshots'] if s['s']==t)

def pc(x):
    return f'{100*x:.1f}%'

def seconds(t):
    return '未达' if t is None else f'{t:.2f}'

def ridge_scores(root):
    path=root/'pear-ridge-240.json'
    folders={'sweep-dp':root/'runs/pear/active-02/sweep-dp',
             'history-dp':ROOT/'pear-full/runs/pear/active-02/history-dp',
             **{a:ACTIVE2/a for a in ('aude-mcc','aude-single-all-mcc')}}
    checks={a:digest(p/'trace.npz') for a,p in folders.items()}
    if path.exists() and read(path)['trace_sha256']==checks:
        return read(path)['scores']
    ref=Reference(ref_mesh('pear','active-02'))
    result={}
    for a,p in folders.items():
        with np.load(p/'trace.npz') as data:
            keep=data['pad_valid'][:24001]&(data['region_load_n'][:24001].max(1)<8.)[:,None]
            t,f=np.nonzero(keep)
            points=data['points_m'][:24001][t,f]
            normals=data['normals_outward'][:24001][t,f]
        result[a]=reconstruct(points,normals,ref,root/f'pear-ridge-{a}-240s.ply','ridge')
    save_json(path,dict(trace_sha256=checks,scores=result))
    return result


def draw(root,folders,records,wrist_label='Wrist GPIS + DP'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    fig,axes=plt.subplots(1,3,figsize=(13,4.35),layout='constrained')
    colors=['#117770','#707785','#c05c24','#7653a0']
    labels=[wrist_label,'Previous wrist + DP','Active-2 Aude + MCC','Active-2 Aude + DP']
    for (arm,folder),color,label in zip(folders.items(),colors,labels):
        r=records[arm]
        with np.load(folder/'coverage.npz') as curve:
            axes[0].plot(curve['time_s'][::50],curve['c10'][::50],label=label,color=color,lw=2.)
        s=r['snapshots']
        axes[1].plot([z['s'] for z in s],[z['signed']['fscore'] for z in s],'o-',color=color,lw=1.8,ms=4)
        source=(folder if arm=='sweep-dp' else ROOT/'pear-full/runs/pear/active-02'/arm
                if arm=='history-dp' else ACTIVE2/arm)
        with np.load(source/'trace.npz') as data:
            end=112932
            mask=(data['pad_valid'][1:end+1].sum(1)>=3)&(data['region_load_n'][1:end+1].max(1)<8.)
            t=np.arange(1,end+1)
            axes[2].plot(t[99::100]*.01,(np.cumsum(mask)/t)[99::100],color=color,lw=1.8)
    for ax,title in zip(axes,['Surface coverage · 10 mm','Signed GPIS F-score · 5 mm','Time with ≥3 light fingertips']):
        ax.set(title=title,xlabel='Simulated exploration time (s)',ylim=(0,1.025),xlim=(0,1130))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
    axes[0].axhline(.95,color='#61746c',ls=':',lw=.8)
    axes[0].legend(loc='lower right',frameon=True,fontsize=8)
    fig.savefig(root/'pear-comparison.png',dpi=175)
    plt.close(fig)


def create(root):
    primary=root/'runs/pear/active-02/sweep-dp'
    ref=root/'references/pear/active-02'
    paths={'sweep-dp':primary,**{a:ref/a for a in ('history-dp','aude-mcc','aude-single-all-mcc')}}
    pear={a:read(p/'scores.json') for a,p in paths.items()}
    seed=read(root/'new-seed/pear/active-02/sweep-dp/scores.json')
    rr=read(primary/'run.json')
    ridge=ridge_scores(root)
    audit=read(root/'causal-replay.json')
    draw(root,paths,pear)
    table=['| 方法 | 10 mm 覆盖 | 平均覆盖¹ | 有符号 F@5 | 对称误差 mm | 轻载≥2指 | 轻载≥3指 | 持续≥3指² | >40 N 秒 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    def row(r,t,label):
        s=snap(r,t)
        return (f"| {label} | {pc(s['coverage'])} | {pc(s['mean_coverage'])} | {pc(s['signed']['fscore'])} | "
                f"{s['signed']['mean_error_mm']:.2f} | {pc(s['low2'])} | {pc(s['low3'])} | "
                f"{pc(s['sustained3'])} | {s['over40_s']:.2f} |")
    def link(p,label):
        return f"[{label}]({p.relative_to(root).as_posix()})"
    ancillary={}
    for a in ('route-dp','no-recontact-dp','sweep-mcc'):
        ancillary[a]=read(primary.parent/a/'scores.json')
    lines=['# GPIS 手腕重规划、复触与实时控制实验','',
           '对齐 Active-2 的固定梨、初始四指接触、物理模型和评分器。主比较完整运行 1,129.32 秒。覆盖只用各分区都低于 8 N、指尖力至少 0.2 N 的物理触点。','',
           '![同起点覆盖、重建和接触](pear-comparison.png)','',
           '## 梨的等时结果','',*table]
    for a,r in pear.items():
        lines.append(row(r,240,link(paths[a]/'scores.json',NAMES[a])))
    lines+=['','上表为前 240 秒。¹平均覆盖是从开始到该时刻覆盖曲线下的面积除以时长，取值越高表示越早探索到表面。²持续三指统计属于至少 1 秒连续轻载三指区间的时间。','',
            '| 方法 | 首达50% 秒 | 首达70% 秒 | 首达90% 秒 | 首达95% 秒 | 首达99% 秒 | 600 秒覆盖 | 1,129.32 秒覆盖 | 末时 F@5 | 全程轻载≥3指 | 全程 >40 N 秒 | 峰值 N |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,r in pear.items():
        end=snap(r,1129.32)
        values=' | '.join(seconds(r['thresholds_s'][str(k)]) for k in (50,70,90,95,99))
        lines.append(f"| {NAMES[a]} | {values} | {pc(snap(r,600)['coverage'])} | {pc(end['coverage'])} | "
                     f"{pc(end['signed']['fscore'])} | {pc(end['low3'])} | {end['over40_s']:.2f} | {end['peak_n']:.1f} |")
    lines+=['','独立换用预留的 DP 噪声种子 20261122，保持梨与初态不变。'
            f"240 秒覆盖、F@5、轻载三指分别为 {pc(snap(seed,240)['coverage'])}、{pc(snap(seed,240)['signed']['fscore'])}、{pc(snap(seed,240)['low3'])}。"
            f"600 秒覆盖 {pc(snap(seed,600)['coverage'])}，首达 95% 用时 {seconds(seed['thresholds_s']['95'])} 秒。"
            " [第二种子原始结果](new-seed/pear/active-02/sweep-dp/scores.json)",'',
            '在同一批 240 秒触点上另用 I01-A 的 R-GPIS 复算。四组 F@5 为 '
            + '、'.join(f'{NAMES[a]} {pc(ridge[a]["fscore"])}' for a in paths)
            +'。两种重建器的模型假设和输出面都不同。 [R-GPIS 网格指标](pear-ridge-240.json)','',
            '## 腕部规划与失触恢复','',
            '旧方法在第 256.25 秒后没有再选择推进。失触后压回最后的双指腕位，局部非指尖受力又触发卸载。实测覆盖从 600 秒到终点都为 41.74%。','',
            '新方法在已知夹具原点周围生成连续腕轨，相邻扫线从粗到细。每圈扫向另一极后，根据实测触点方向的空缺选择下一遍的方位。单指仍可沿当前腕轨推进。每轮只发布腕位和腕姿，四指的同步位置由当前关节正运动学给出。','',
            '实测触点和法向用来拟合正则化二次隐式面。在已摸到的邻域，用局部 Matérn GPIS 修正距离、法向和不确定度。对四种三指组合，各解一个 3×3 线性方程，得到腕高和两个倾角的修正，再和原候选一起比较。评分考虑第二、第三根手指落入接触带的概率、GPIS 方差、离已触区域的距离、预测的过深接触。腕半径限制在初始值 −100 至 +130 mm，局部倾角上限 0.76 rad。','',
            '100 Hz 载荷反馈修正腕高。连续过载 0.65 秒、失去双指 2.6 秒且至少 0.95 秒未新增单元，或地图停更 6.5 秒时，经过冷却即可启动复触。约 0.28 秒径向退出、0.32 秒转移，再用约 0.40 秒带载荷反馈接近。两次启动至少相隔 4.8 秒，每次转移继续走向下一段。','',
            '手指使用发布的 single_all 权重和现有指端 MCC。执行端把策略名义角约束在初始握姿 ±0.48 rad 与物理关节范围内，以 1.8 rad/s 限速。手腕规划器的候选变量均为腕部量。物体三角网格只进入碰撞仿真和离线精度评价。200 ms 腕部前瞻是当前段的预测，后续触觉反馈可以改变实际命令。','',
            '### 梨的分项对照，240 秒','',*table,row(pear['sweep-dp'],240,NAMES['sweep-dp'])]
    for a,r in ancillary.items():
        lines.append(row(r,240,link(primary.parent/a/'scores.json',NAMES[a])))
    events=read(primary/'recoveries.json')
    both=sum(e['restored_two_s'] is not None for e in events)
    median=np.median([e['restored_two_s']-e['start_s'] for e in events if e['restored_two_s'] is not None]) if both else None
    with np.load(primary/'trace.npz') as data:
        k,f=np.nonzero(data['pad_valid']&(data['region_load_n'].max(1)<8.)[:,None])
        points=data['points_m'][k,f]
        _,idx=np.unique(np.floor(points/.0045).astype(int),axis=0,return_index=True)
        first=np.sort(k[idx])
    post=0
    for i,e in enumerate(events):
        if e['restored_two_s'] is not None:
            after=round(e['restored_two_s']*100)
            bound=min(after+501,112933,round(events[i+1]['start_s']*100) if i+1<len(events) else 112933)
            post+=int(np.searchsorted(first,bound)>np.searchsorted(first,after))
    rec=pear['sweep-dp']['recovery']
    lines+=['','去局部 GPIS 组保留同一全局扫线、地图方向空缺选择、力反馈、复触状态机和 DP，只去掉局部倾角、腕高与采样收益修正。关闭复触组仍保留持续推进、载荷反馈和局部 GPIS。纯 MCC 组冻结名义手型，其腕轨依本组实际触觉重新规划。','',
            f"梨主试次共触发 {len(events)} 次显式复触，其中 {both} 次在达到下一次尝试或六秒观察窗前恢复连续 200 ms 的轻载双指。"
            f"成功事件从开始卸载到恢复的中位数为 {seconds(median)} 秒，{post} 次在恢复后五秒内且在下一次尝试前采到新单元。"
            f"整轮目标腕位的前瞻平移误差中位数／P95 为 {pear['sweep-dp']['preview_error_p50_p95']['mm'][0]:.1f}／{pear['sweep-dp']['preview_error_p50_p95']['mm'][1]:.1f} mm。"
            ' [逐次复触](runs/pear/active-02/sweep-dp/recoveries.json)','',
            '### 相同腕令下的第三指','',
            '| 240 秒腕轨 | DP 轻载≥3指 | 纯 MCC 轻载≥3指 | MCC 恰两指且两组腕部均在动 | DP 保留同两指再加指 | 条件占比 | 遮去这些额外触点后覆盖 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for obj,start,label in [('pear','active-02','梨'),('ovoid','start-03','卵形体 C')]:
        path=root/'paired-wrist'/obj/start/'result.json'
        z=read(path)
        lines.append(f"| {link(path,label)} | {pc(z['dp_low3'])} | {pc(z['mcc_low3'])} | {z['mcc_exact2_s']:.2f} s | "
                     f"{z['retained_plus_s']:.2f} s | {pc(z['conditional']) if z['conditional'] is not None else '分母为零'} | {pc(z['masked_cover'])} |")
    lines+=['','## 七组等时迁移与强基线','',
            '两个常规物体各用 A、B 起点，三个平滑物体各用侧面 C。每组执行 240 秒。强基线使用 Active-2 完整的全局扫查、联合腕指规划、约束手型、腕部反馈及 MCC。适配后先在原梨初态复跑 20 秒，2,001 帧的腕指状态、执行角、触点、法向、有效掩码和五区载荷均与归档逐值相同。 [基线移植复核](baseline-check/pear/active-02/aude-mcc/run.json)','',
            '| 物体／起点 | 新腕覆盖 | Aude 覆盖 | 新腕 F@5 | Aude F@5 | 新腕轻载≥3指 | Aude 轻载≥3指 | 旧腕覆盖³ |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for r in jobs():
        obj,start=r['obj'],r['start']
        if obj=='pear':
            continue
        key=Path(obj)/start
        files=[root/'runs'/key/'sweep-dp/scores.json',root/'strong-baseline'/key/'aude-mcc/scores.json']
        dst=[snap(read(p),240) for p in files]
        u,v=dst
        old=read((ROOT/'smooth-transfer' if obj in SMOOTH else ROOT)/'runs'/key/'rigid-dp/score.json')['snapshots'][-1]['coverage10']
        rows.append(dict(obj=obj,start=start,coverage=[u['coverage'],v['coverage']],fscore=[u['signed']['fscore'],v['signed']['fscore']],
                         low3=[u['low3'],v['low3']],old_rigid=old))
        lines.append(f"| {OBJECTS[obj]} {start[-2:]} | {link(files[0],pc(u['coverage']))} | {link(files[1],pc(v['coverage']))} | "
                     f"{pc(u['signed']['fscore'])} | {pc(v['signed']['fscore'])} | {pc(u['low3'])} | {pc(v['low3'])} | {pc(old)} |")
    summary={key:np.mean([r[key] for r in rows],axis=0).tolist() for key in ('coverage','fscore','low3')}
    lines+=['| **七组等权平均** | '+' | '.join(pc(v) for k in ('coverage','fscore','low3') for v in summary[k])+
            ' | '+pc(np.mean([r['old_rigid'] for r in rows]))+' |','',
            '³旧腕列为上一轮在主场景表现最佳的当前实测手型 rigid-dp。午餐肉罐 A 和梨进入过开发试跑，其余六个迁移起点在冻结后运行。梨的重复种子检验同初始状态下的 DP 采样变化。这里的均值对所列起点等权。','',
            '## 实际时钟下的闭环','']
    paced=root/'paced/pear/active-02/sweep-dp'
    z=read(paced/'run.json')
    timed=read(paced/'scores.json')
    ti=timed['timing']
    lines+=['在 120 秒 MuJoCo 梨试次中按 100 Hz 墙钟启动每个控制周期。该初版的计时与后续改进版仿真并行，反映有竞争负载的情况。腕部规划和 GPU 推理从各自的工作线程提前 50 ms 提交。DP 动作块尚未返回时，伺服继续使用上一块已发布的指令。腕规划迟到时，保持切向端点并继续响应载荷。实测时钟包含传感、规划调度、DP 接口、MCC、五个物理子步及记录。','',
            '| 工作段 | 样本 | P50 ms | P95 ms | P99 ms | 最大 ms | 超过比较预算 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for title,key,deadline in [('腕规划计算','plan',200),('DP 请求与返回','dp_rpc',50),('100 Hz 周期内的工作','loop_service',10),
                               ('从预定释放到该周期完成','finish_from_release',10)]:
        q=ti[key]
        lines.append(f"| {title} | {q['n']} | {q['p50']:.2f} | {q['p95']:.2f} | {q['p99']:.2f} | {q['maximum']:.2f} | {q['over']} / {q['n']}（{deadline} ms） |")
    actual=ti['finish_from_release']
    lines+=['',f"实测 10 ms 期限完成率为 {pc(1-actual['over']/actual['n'])}。腕段交接迟到 {z['late_wrist_handoffs']} 次，"
            f"DP 交接迟到 {z['missed_handoffs']} 次，跳过请求 {z['skipped_queries']} 次，指令缓冲耗尽 {z['underrun_ticks']} 帧。"
            f"设备为 {z['policy']['device']}，场景和模型准备耗时 {z['setup_s']:.2f} 秒，随后闭环墙钟 {z['acquire_after_setup_s']:.2f} 秒。"
            f"这条真时钟轨迹的 120 秒覆盖为 {pc(snap(timed,120)['coverage'])}，轻载三指 {pc(snap(timed,120)['low3'])}。"
            ' [逐周期计时](paced/pear/active-02/sweep-dp/control-timing.npz) · [控制器回执](paced/pear/active-02/sweep-dp/run.json)','',
            f"独立回放从空地图开始，只读取关节和触觉，用无物体的手模型重算 {audit['decisions']:,} 次腕部决策、{audit['packets']:,} 条腕令及相应前瞻。"
            f"腕令、前瞻最大分量误差分别为 {audit['max_pose_component_error']:.2g} 和 {audit['max_preview_component_error']:.2g}。 [因果复核](causal-replay.json)",'',
            '## 复现和适用范围','',
            '全部主结果来自固定物体、已知夹具原点、从四指轻载接触开始的仿真。非凸遮挡、移动物体及真机传感通信需要单独验证。异步伺服的时限反映本机 Windows、CPU 与 GPU 条件。最终高分辨率网格在采集后重建，在线维护触摸地图、局部有符号场和不确定度。','',
            '在实验仓库根目录运行下列命令会生成新的结果目录。','',
            '```powershell','.\\Module\\wrist_active\\run_sweep.ps1 -Output Module/generated/gpis_wrist_sweep_repeat','```','',
            '[冻结协议](protocol.json) · [机器可读摘要](summary.json) · [手腕规划实现](../../wrist_active/sweep.py) · [完整复现入口](../../wrist_active/run_sweep.ps1)']
    save_json(root/'summary.json',dict(primary={a:{'t95':r['thresholds_s']['95'],'s240':snap(r,240),'final':snap(r,1129.32)}
          for a,r in pear.items()},alternate_seed=seed['thresholds_s'],transfer=rows,transfer_mean=summary,
          paced=ti,deadline_hits=1-actual['over']/actual['n'],dp_underrun=z['underrun_ticks'],
          recontact=dict(attempts=rec['attempts'],regained_two=both,new_after_regain=post),causal=audit))
    u,v=snap(pear['sweep-dp'],240),snap(pear['aude-mcc'],240)
    lines[4:4]=[f"同一只梨在 240 秒的覆盖由旧腕规划的 {pc(snap(pear['history-dp'],240)['coverage'])} 提升到 **{pc(u['coverage'])}**。"
        f"新腕规划的有符号 F@5 为 **{pc(u['signed']['fscore'])}**，轻载三指占 **{pc(u['low3'])}**。"
        f"同预算完整 Active-2 Aude＋MCC 的三项分别为 {pc(v['coverage'])}、{pc(v['signed']['fscore'])} 和 {pc(v['low3'])}。",'',
        f"把相同设置放到七个迁移起点，新腕规划有 {sum(r['coverage'][0]>r['coverage'][1] for r in rows)}/7 组覆盖高于完整 Aude＋MCC，"
        f"{sum(r['fscore'][0]>r['fscore'][1] for r in rows)}/7 组有符号 F@5 更高，"
        f"{sum(r['low3'][0]>r['low3'][1] for r in rows)}/7 组轻载三指时间更长。实际墙钟的 100 Hz 试次中，{pc(1-actual['over']/actual['n'])} 的周期按时完成。",'']
    (root/'SWEEP_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=OUT)
    p.add_argument('--ridge-only',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(1)
    (ridge_scores if args.ridge_only else create)(args.root)
