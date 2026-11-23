"""Assemble the complete measured experiment and its reproducibility links."""
import argparse
import json
from pathlib import Path

import numpy as np

from .experiment import ARMS, CASES, ROOT, save_json
from .smooth_transfer import OBJECTS


NAMES={'history-dp':'历史手型＋DP＋MCC','history-mcc':'历史手型＋纯 MCC','rigid-dp':'当前手型＋DP＋MCC',
       'no-gain-dp':'去局部增益＋DP＋MCC','aude-mcc':'Aude＋MCC','aude-single-all-mcc':'Active-2 Aude＋DP＋MCC'}
OBJECT={'ellipsoid':'椭球','meat-can':'午餐肉罐','pear':'梨','oblate':'扁椭球','ovoid':'不对称卵形体','superellipsoid':'超椭球'}


def read(p):
    return json.loads(Path(p).read_bytes())


def avg(rows,f):
    return float(np.mean([f(r) for r in rows]))


def pct(v):
    return f'{v*100:.1f}%'


def metrics(r):
    last=r['snapshots'][-1]
    return (last['coverage10'],last['coverage25'],last['signed']['fscore'],r['ridge']['fscore'],
            r['light_two'],r['light_three'],r['sustained_light_three'])


def result_row(name,rows):
    means=np.mean([metrics(r) for r in rows],axis=0)
    return '| '+' | '.join([name,*map(pct,means),f'{sum(r["first_70pct10_sim_s"] is not None for r in rows)}/{len(rows)}'])+' |'


def preview_range(root):
    from .history import GENERATED,palm_rot
    from ..i04_geometric_traversal.i01_a_wrist import MOTION_STD_WIDTH
    cal=read(GENERATED/'i01_a_20261120/wrist_calibration.json')
    mu=np.asarray(cal['mean'])[6:]
    sd=np.asarray(cal['std'])[6:]
    lower=np.minimum(mu-MOTION_STD_WIDTH*sd,0.)
    upper=np.maximum(mu+MOTION_STD_WIDTH*sd,0.)
    rows=[]
    for obj,s in [*CASES,('pear','active-02')]:
        with np.load(root/'runs'/obj/s/'history-dp/trace.npz') as trace:
            q=trace['qpos'][:-1,:7]
            ahead=trace['published_preview'][1:]
        inverse=palm_rot(q).inv()
        delta=np.c_[inverse.apply(ahead[:,:3]-q[:,:3]),(inverse*palm_rot(ahead)).as_rotvec()]
        outside=(delta<lower)|(delta>upper)
        rows.append(dict(obj=obj,start=s,any_fraction=float(outside.any(1).mean()),
                         component_fractions=outside.mean(0).tolist()))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    root=p.parse_args().root.resolve()
    mainrows=[read(root/'runs'/o/s/a/'score.json') for o,s in CASES for a in ARMS]
    pear=[read(root/d/'pear/active-02'/a/'score.json') for d,a in (
        ('runs','history-dp'),('runs','history-mcc'),('archive','aude-mcc'),('archive','aude-single-all-mcc'))]
    smooth=[read(root/'smooth-transfer/runs'/o/'start-03'/a/'score.json')
            for o in OBJECTS for a in ('history-dp','rigid-dp','history-mcc')]
    paired=[read(root/'paired-replay'/o/s/'paired.json') for o,s in CASES]
    probes=read(root/'probe-results.json')
    causal=read(root/'causality-audit.json')
    ranges=preview_range(root)
    long_pear=read(root/'pear-full/long-results.json')
    from .media import pear_plot
    pear_plot(root)
    elapsed=[read(root/'serial-60s/runs'/o/'start-01'/a/'elapsed.json') for o in ('ellipsoid','meat-can')
             for a in ('history-dp','history-mcc','aude-mcc')]
    index={(r['obj'],r['start'],r['arm']):r for r in mainrows}
    group=lambda a:[r for r in mainrows if r['arm']==a]
    best=max(ARMS,key=lambda a:avg(group(a),lambda r:metrics(r)[0]))
    best_model=max(ARMS,key=lambda a:avg(group(a),lambda r:metrics(r)[2]))
    main=group('history-dp')
    ranking=(f'**{NAMES[best]}** 在两项上均最高，分别为 {pct(avg(group(best),lambda r:metrics(r)[0]))} 和 {pct(avg(group(best),lambda r:metrics(r)[2]))}。'
             if best==best_model else f'覆盖最高的方法为 **{NAMES[best]}**，有符号重建最高的方法为 **{NAMES[best_model]}**。')
    rows=['# GPIS 手腕主动触觉探索','',
        f'主试验中，历史手型腕部 Active＋single_all＋MCC 的平均轻载 10 mm 覆盖为 **{pct(avg(main,lambda r:metrics(r)[0]))}**，有符号 GPIS F-score@5 mm 为 **{pct(avg(main,lambda r:metrics(r)[2]))}**。{ranking}每个物体两个指定起点，所有数字均按实际执行结果计入。','',
        '[等时覆盖与重建曲线](time-curves.png) · [四个起点的同步实测回放](matched-replay.mp4) · [逐轮指标](results.json) · [冻结配置](protocol.json)','',
        '## 方法与对照','',
        '手腕每 250 ms 选择未来 0.5 s 的平滑运动，位置与姿态共六个自由度。控制端保留已发布的 200 ms 前瞻。single_all 每 100 ms 推理，MCC 每 10 ms 更新，MuJoCo 每 2 ms 积分。记录、规划观测与接触评分按 10 ms 采样。规划输入含手自身几何、关节、本轮实测触点与法向。真实物体几何仅供物理碰撞、既有初态及离线评分。','',
        '手型库取开发物体芥末瓶和饼干盒各 1,400 个同步整手短片段，保留轻载、失触和过载结果。按当前关节、指尖接触掩码、法向分散程度及候选腕部位移/转角取六个邻近片段，预测四指在腕坐标系内的共同位移和接触可信度。在线局部有符号 GPIS 用实测点及法向建模，摸到新区域可沿可信表面外推。地图只接受指尖达到 0.2 N 且五个分区均低于 8 N 的样本。','',
        '每轮规划从近期稳定的支撑指中选一对，在局部切平面采八个推进方向，每个方向配零转动及两个绕支撑连线的转动。以局部场调整掌距，检查支撑残差和掌部间隙。对候选中段与末端，各指预期落点与 GPIS 表面带的交集，按接触可信度、场的后验标准差、离已测点的距离加权。同一历史片段中落入同一 4.5 mm 单元的观测按联合命中近似去重。该分数是预期采样价值的工程近似。低载荷两指不足时尝试复触，载荷达到门槛时沿外法向卸载。','',
        '历史手型＋纯 MCC 使用同一份手型库、候选和评分，手指的名义角固定在初态。当前手型消融用眼前的指腹几何刚性外推。去局部增益消融保留同样的历史手型、可行性与前沿导航，拿掉候选腕轨的局部增益项。Aude 组在相同物理初态上运行项目中的触觉 AudeExplorer，以联合腕指 IK 产生名义目标，再执行同款指端 MCC。梨另使用已有 Active-2-A 强化版本做同状态复核。','',
        '## 主试验','',
        '椭球与午餐肉罐，各两个起点、五组控制、每轮 240 秒。以下先逐轮计算，再等权平均。覆盖取低载荷实际指尖触点在参考网格上的测地邻域面积并集。两种 GPIS 都以整物体为参考，所有方法使用相同采样参数。持续三指列为属于至少 1 秒连续低载荷三指区间的全程时间占比。','']
    header=['| 方法 | 10 mm 覆盖 | 25 mm 覆盖 | 有符号 F@5 | R-GPIS F@5 | 轻载≥2指 | 轻载≥3指 | 持续≥3指 | 达70%覆盖 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    rows+=header+[result_row(NAMES[a],group(a)) for a in ARMS]
    h=np.mean([metrics(r) for r in main],axis=0)
    rigid=np.mean([metrics(r) for r in group('rigid-dp')],axis=0)
    nogain=np.mean([metrics(r) for r in group('no-gain-dp')],axis=0)
    mcc=np.mean([metrics(r) for r in group('history-mcc')],axis=0)
    rows+=['',
        f'相同历史手型规划下，加入 single_all 的覆盖与有符号 F@5 分别变化 **{100*(h[0]-mcc[0]):+.1f}、{100*(h[2]-mcc[2]):+.1f} 个百分点**。保持 DP 和 GPIS 评分，改为当前实测手型外推后，两项分别变化 **{100*(rigid[0]-h[0]):+.1f}、{100*(rigid[2]-h[2]):+.1f} 个百分点**。历史手型方案中，开启局部增益相对去掉该项的变化为 **{100*(h[0]-nogain[0]):+.1f}、{100*(h[2]-nogain[2]):+.1f} 个百分点**。本轮更支持使用实测手型作为腕轨预测的起点。']
    rows+=['','### 每个起点','',
           '| 物体与起点 | 方法与原始记录 | 10 mm 覆盖 | 有符号 F@5 | R-GPIS F@5 | 轻载≥3指 | 运动中轻载≥3指 | >40 N 累计 | 峰值 | 首达70%覆盖 仿真／采集墙钟 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in mainrows:
        obj,s,a=r['obj'],r['start'],r['arm'];v=metrics(r)
        arrival='未达' if r['first_70pct10_sim_s'] is None else f'{r["first_70pct10_sim_s"]:.2f} / {r["first_70pct10_acquisition_wall_s"]:.2f} s'
        rows.append(f'| {OBJECT[obj]} {s[-1]} | [{NAMES[a]}](runs/{obj}/{s}/{a}/score.json) | {pct(v[0])} | {pct(v[2])} | {pct(v[3])} | {pct(v[5])} | {pct(r["moving_light_three"] or 0)} | {r["over40_s"]:.2f} s | {r["peak_n"]:.1f} N | {arrival} |')
    rows+=['','### 实际花在扫查与复触上的时间','',
           '以下按 250 ms 的决策周期统计历史手型＋DP 主方法。连续卸载或复触时长包含一次扫查决策至下一次扫查决策之间连续用于恢复的周期。','',
           '| 物体与起点 | 推进扫查 | 复触 | 卸载 | 最长连续卸载或复触 |','| --- | ---: | ---: | ---: | ---: |']
    for obj,s in CASES:
        plans=[json.loads(l)['mode'] for l in (root/'runs'/obj/s/'history-dp/plans.jsonl').read_text(encoding='utf-8').splitlines()]
        recovery=np.array(plans)!='scan'
        edges=np.flatnonzero(np.diff(np.r_[False,recovery,False]))
        longest=.25*max((edges[1::2]-edges[::2]),default=0)
        rows.append(f'| {OBJECT[obj]} {s[-1]} | '+ ' | '.join(f'{.25*plans.count(m):.2f} s' for m in ('scan','recontact','relief'))+f' | {longest:.2f} s |')
    rows+=['','几何评分把触点按 5 mm 注册误差、法向余弦至少 0.5 投到参考表面。每次重建最多取 1,000 个实测代表点，以同一随机种子的 50,000 个表面样本计算精确率和召回率。有符号 GPIS 在 30、60、120、240 秒各重建一次，R-GPIS 在末时重建。两种重建器能从同一批触点给出不同几何质量。每个 score.json 同目录保存实测 trace、逐次候选评分、逐时覆盖和 PLY 网格。','',
           '## 同腕轨时第三指的独有贡献','',
           '将主方法在每个初态实际下发的 240 秒腕轨原样交给纯 MCC。分母只含两组手腕都在移动且纯 MCC 恰有两指轻载接触的时刻。分子要求 DP 在同一时刻保留同一对手指、另有指尖接触且五区轻载。随后只遮去这些时刻额外指尖的数据，按同一流程重算整轮覆盖与重建。持续时间很短的分母应结合原始秒数解读。','',
           '| 配对回放 | 分母 | 同两指再多一指 | 独有单元 | 遮去额外指后，10 mm 覆盖变化 | 遮去额外指后，有符号 F@5 变化 |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in paired:
        obj,s=r['obj'],r['start'];base=metrics(index[obj,s,'history-dp'])
        conditional=pct(r['same_two_plus_given_mcc_exactly_two']) if r['pure_mcc_exactly_two_s'] else '分母为零'
        rows.append(f'| [{OBJECT[obj]} {s[-1]}](paired-replay/{obj}/{s}/paired.json) | {r["pure_mcc_exactly_two_s"]:.2f} s | {conditional} | {r["exclusive_cells"]} | {(r["masked_coverage10"]-base[0])*100:+.2f} 个百分点 | {(r["masked_signed"]["fscore"]-base[2])*100:+.2f} 个百分点 |')
    bad=[r for r in [*mainrows,*pear,*smooth] if r['ridge']['status']!='ok']
    if bad:
        rows+=['','R-GPIS 未能提取等值面的试次以 F@5＝0 计入，并保留其有符号 GPIS 结果。涉及'+
               '、'.join(f'{OBJECT[r["obj"]]} {r["start"]} 的{NAMES[r["arm"]]}' for r in bad)+'。']
    rows+=['','## 腕部前瞻与策略训练范围','',
           '从实际发给 DP 的 200 ms 前瞻减去当时实测腕姿，在掌坐标系计算三维位移与旋转向量，按 I01-A 的训练均值 ±2.5 个标准差并包含零值检查。主规划使用了曲面支撑与掌距筛选，候选绕支撑轴转动时并未把这个统计范围设为硬约束。以下报告落在范围外的时间占比，属于分布诊断。','',
           '| 主方法试次 | 任一前瞻分量超出范围 | 绕掌 z 轴的旋转分量超出范围 |','| --- | ---: | ---: |']
    for r in ranges:
        rows.append(f'| {OBJECT[r["obj"]]} {r["start"]} | {pct(r["any_fraction"])} | {pct(r["component_fractions"][5])} |')
    rows+=['','## 规划因果复核','',
           f'另外载入只有手的运动学模型，从零重建 GPIS 与腕部命令缓存。输入仅有每个时刻的实测关节、指尖触觉和载荷，重算椭球、午餐肉罐四轮和梨一轮主方法。逐项复现 **{sum(r["decisions"] for r in causal):,} 次决策、{sum(r["commands"] for r in causal):,} 条实际腕令及其前瞻**。候选评分最大绝对差为 {max(r["max_score_abs_error"] for r in causal):.2g}。仿真物体几何未进入复算器。 [复核数据](causality-audit.json)','',
           '## 同状态的候选动作分叉','',
           '四组状态各执行四个通过可行性检查的动作，按预测总评分从高到低选等距名次。每条分叉都重新执行同一短前缀，校验物理状态与已发布腕轨哈希一致，再执行完整 0.5 秒候选。新增单元按分叉腕轨开始前的地图去重。下列相关系数比较预测局部增益与实测新增单元；每组只有四个动作。','',
           '| 分叉起点 | 从最高到最低评分，实测新增单元 | 第三指等额外观测独有单元 | 预测增益与新增单元的秩相关 |',
           '| --- | --- | --- | ---: |']
    for g in probes:
        xs=g['rows']; rho=g['rank_correlation']
        rows.append(f'| {OBJECT[g["obj"]]} {g["start"][-1]} | '+', '.join(str(r['realized_new_cells']) for r in xs)+' | '+', '.join(str(r['realized_extra_cells']) for r in xs)+f' | {rho if rho is None else round(rho,2)} |'.replace('None','无可辨排序'))
    rows+=['',f'最高总评分的动作在 {sum(g["top_rank_tied_or_best"] for g in probes)}/{len(probes)} 个状态得到最多新增单元（含并列）。几何交集给出可触表面的候选范围，新增面积和后验改进仍必须由实触验证。']
    rows+=['','[全部分叉评分、支撑时间和实测载荷](probe-results.json)','','## 平滑物体侧面起点迁移','',
           '使用新版 I01-A 的三个独立物体，每个物体取侧面起点 C，重复 240 秒。此轮在主试验开始后另行登记，保留同一份冻结代码和手型库。','',
           '| 物体 | 方法 | 10 mm 覆盖 | 有符号 F@5 | R-GPIS F@5 | 轻载≥2指 | 轻载≥3指 | 持续≥3指 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in smooth:
        o,a=r['obj'],r['arm'];v=metrics(r)
        rows.append(f'| {OBJECT[o]} | [{NAMES[a]}](smooth-transfer/runs/{o}/start-03/{a}/score.json) | '+' | '.join(pct(v[i]) for i in (0,2,3,4,5,6))+' |')
    rows+=['','三种新物体等权平均，'+'。'.join(f'{NAMES[a]} 的覆盖／有符号 F@5 为 '+ '／'.join(pct(avg([r for r in smooth if r["arm"]==a],lambda r:metrics(r)[i])) for i in (0,2)) for a in ('history-dp','rigid-dp','history-mcc'))+'。']
    rows+=['','[独立迁移协议](smooth-transfer/protocol.json) · [I01-A 原始平滑曲面对照](../i01_a_smooth_20261121/REPORT.md)','','## Active-2 梨的等时复核','',
           '新方法从 Active-2-A 保存的相同 MuJoCo 积分状态及 75 帧热启动历史运行。两条既有轨迹截取前 240 秒，重新按五分区全部 <8 N 取点，再用同一离线评价器评分。Active-2-A 原有手型包络、腕部反馈和联合规划均保留，因而这里比较的是整套系统。归档原文另以最高 12 N 采样并报告超过千秒的完成结果。','']
    rows+=header+[result_row(('归档 ' if r['archived'] else '新跑 ')+NAMES[r['arm']],[r]) for r in pear]
    rows+=['','[Active-2-A 原始逐帧审计](../active-02-a_20261121/active-02-comparison.json) · [原始重建时间序列](../active-02-a_20261121/signed-results.json)','','### 对齐 Aude 完成三轴扫描的完整时长','',
           '看到 240 秒结果后，另登记长时复核。按两条归档轨迹中较早的结束时间，四组统一评分到 1,129.32 秒。两条新方法从原始状态重新执行，首 240 秒逐帧对比原试次。阈值首次到达时间由整条 10 ms 实测覆盖序列计算。','',
           '| 方法 | 最终10 mm覆盖 | 最终25 mm覆盖 | 有符号 F@5 | 轻载≥2指 | 轻载≥3指 | >40 N累计 | 首达70% | 首达95% | 前240秒复核 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for r in long_pear:
        s=r['snapshots'][-1]
        targets=[f'{r["targets"][k]["sim_s"]:.2f} s' if r['targets'][k] else '未达' for k in ('0.7','0.95')]
        check='同一归档' if r['archived'] else '逐帧一致' if r['matches_240s_prefix'] else '长试次单独执行，前缀有差异'
        rows.append('| '+' | '.join([('归档 ' if r['archived'] else '新跑 ')+NAMES[r['arm']],
            pct(s['coverage10']),pct(s['coverage25']),pct(s['signed']['fscore']),pct(r['low_two']),pct(r['low_three']),
            f'{r["over40_s"]:.2f} s',*targets,check])+' |')
    pear_plans=(root/'pear-full/runs/pear/active-02/history-dp/plans.jsonl').read_text(encoding='utf-8').splitlines()
    last_scan=max(p['tick'] for p in map(json.loads,pear_plans) if p['mode']=='scan')*.01
    long_dp=next(r for r in long_pear if r['arm']=='history-dp')
    l240,l600,lend=long_dp['snapshots']
    rows+=['',f'历史手型＋DP 在 {last_scan:.2f} 秒最后一次选择推进扫查，之后的腕令都用于复触或卸载。它在 240、600 秒及终点的覆盖分别为 {pct(l240["coverage10"])}、{pct(l600["coverage10"])}、{pct(lend["coverage10"])}。这一轨迹显示长时间重复恢复限制了探索范围。']
    rows+=['','[覆盖与三指时间曲线](pear-full-comparison.png) · [长时协议](pear-full/long-protocol.json) · [240、600、1,129.32 秒各组重建、覆盖与阈值时间](pear-full/long-results.json)','','## 本实验逐组顺序计时','',
           '相同起点 A、60 秒仿真、每次只运行一种方法，从本轮场景和策略加载到实测轨迹与有符号最终网格写完计时。运行器导入和离线真值评分另计。主试验批量并行运行，以此六轮补充比较墙钟。','',
           '| 物体 | 方法 | 加载 | 在线规划累计 | 采集到最后一帧 | 最终网格计算与写出 | 总墙钟 |',
           '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in elapsed:
        rows.append(f'| {OBJECT[r["obj"]]} | {NAMES[r["arm"]]} | '+' | '.join(f'{r[k]:.2f} s' for k in ('setup_s','planning_s','acquisition_wall_s','mesh_wall_s','total_wall_s'))+' |')
    rows+=['','两个起点 A 的 60 秒试次等权平均，'+
           '，'.join(f'{NAMES[a]} 从加载到网格写完为 {avg([r for r in elapsed if r["arm"]==a],lambda r:r["total_wall_s"]):.2f} 秒' for a in ('history-dp','history-mcc','aude-mcc'))+
           '。耗时包含发布版权重的加载与在线推理，规划函数本身的时间另列。']
    rows+=['','## 复现与解释范围','',
        '在已有项目环境的仓库根目录运行下列命令。脚本建独立的输出目录，依次执行物理试验、归档复算、消融与报告。所有新试次均可从原始 trace 与逐次 GPIS 记录复查。','',
        '```powershell', '.\\Module\\wrist_active\\run.ps1 -Output Module/generated/gpis_wrist_active_repeat','```','',
        '这些是固定物体、给定四指接触起点下的 MuJoCo 结果。物体、起点与策略种子按组配对，每个起点每种方法一次。结果描述所测条件，包含未达到覆盖目标和进入复触停滞的试次。','']
    save_json(root/'results.json',dict(main=mainrows,pear=pear,pear_full=long_pear,smooth=smooth,matched_replay=paired,probes=probes,causality_audit=causal,preview_training_range=ranges,serial_timing=elapsed))
    (root/'REPORT.md').write_text('\n'.join(rows),encoding='utf-8')


if __name__=='__main__':
    main()
