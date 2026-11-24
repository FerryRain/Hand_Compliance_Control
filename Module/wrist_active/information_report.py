"""Score the transfer extension and assemble the paired information-gain results."""

import argparse
from pathlib import Path

import numpy as np

from .experiment import digest, save_json, seed_for
from .information_score import EVAL_POINTS, TIMES, evaluate, wrist_travel
from .matched_information import CASES, METHODS, PEAR_SEEDS, STEP_M, STEP_RAD
from .mcc_trials import RESULTS, read
from .sweep import DT
from .sweep_experiment import OUT
from .sweep_score import score


OBJECTS = {"ellipsoid":"椭球", "meat-can":"午餐肉罐", "oblate":"扁椭球",
           "ovoid":"卵形体", "superellipsoid":"超椭球"}
PEAR = ("pear", "active-02")
NATIVE_ARMS = ("wrist-dp", "wrist-mcc", "aude-mcc")
NAMES = {"wrist-dp":"新腕轨＋single_all＋MCC", "wrist-mcc":"新腕轨＋纯 MCC",
         "aude-mcc":"Active-2 Aude＋MCC", "aude-single-all-mcc":"Active-2 Aude＋single_all＋MCC"}


def transfer_scores(root, available):
    extension = root/"transfer-extension"
    for obj, start in CASES:
        if obj == "pear":
            continue
        seed = seed_for(obj, start)
        for arm in ("wrist-dp","aude-dp"):
            path = Path(obj)/start/f"seed-{seed}"/arm
            source = extension/"matched"/path
            if available and not (source/"run.json").exists():
                continue
            dest = extension/"evaluation/matched"/path
            dest.mkdir(exist_ok=True, parents=True)
            score(source, dest, obj, start, list(TIMES))
            evaluate(extension, "matched", obj, start, arm, source, dest/"scores.json", seed)


def metric(folder):
    r = read(folder/"information.json")
    return dict(trace_sha256=r["spec"]["trace_sha256"], snapshots=r["snapshots"],
                sensitivity=r["sensitivity"], first_new_cell_budget_s=r["first_new_cell_budget_s"])


def compare(folder, first, second):
    """Truncate both acquired curves at the same translational and rotational budgets."""
    curves = []
    for arm in (first, second):
        with np.load(folder/arm/"information-curve.npz", allow_pickle=False) as f:
            curves.append({k:f[k] for k in ("translation_m","rotation_rad","cumulative_nats")})
    distance = min(c["translation_m"][-1] for c in curves)
    angle = min(c["rotation_rad"][-1] for c in curves)
    values = {}
    for arm, c in zip((first,second),curves):
        tick = min(np.searchsorted(c["translation_m"],distance,side="right"),
                   np.searchsorted(c["rotation_rad"],angle,side="right"))-1
        values[arm] = dict(seconds=float(tick*DT), nats=float(c["cumulative_nats"][tick]),
                           translation_m=float(c["translation_m"][tick]), rotation_rad=float(c["rotation_rad"][tick]))
    return dict(translation_cap_m=float(distance), rotation_cap_rad=float(angle), arms=values)


def check_pair(root, obj, start, seed, methods):
    path = Path(obj)/start/f"seed-{seed}"
    receipts, first_points = {}, None
    initial, model, checkpoint = set(), set(), set()
    for arm in methods:
        folder = root/"matched"/path/arm
        r = read(folder/"run.json")
        assert r["steps"] == 24000 and r["segments"] == 1200 and r["protocol_sha256"] == digest(root/"protocol.json")
        initial.add(r["initial_sha256"])
        model.add(r["model_sha256"])
        if arm.endswith("dp"):
            info = r["policy"]
            assert info["inference_seed"] == seed and info["lead_s"] == .05
            checkpoint.add(info["checkpoint_sha256"])
        else:
            assert r["policy"] is None
        with np.load(folder/"trace.npz", allow_pickle=False) as f:
            poses = f["palm_command"]
            d, a = wrist_travel(poses)
            assert len(d) == 24001 and max(np.diff(d)) <= STEP_M+1.e-9 and max(np.diff(a)) <= STEP_RAD+1.e-8
            assert np.max(np.linalg.norm(poses[:,:3],axis=1)) <= r["maximum_radius_m"]+1e-9
            if arm.endswith("mcc"):
                nominal = f["finger_nominal"]
                assert np.all(nominal == nominal[0])
            if first_points is None:
                first_points = f["points_m"][0]
            else:
                np.testing.assert_array_equal(first_points, f["points_m"][0])
        receipts[arm] = dict(steps=r["steps"], segments=r["segments"], clipped_steps=r["clipped_wrist_steps"],
                              initial_sha256=r["initial_sha256"], model_sha256=r["model_sha256"],
                              checkpoint_sha256=r["policy"]["checkpoint_sha256"] if r["policy"] else None)
    assert len(initial) == len(model) == 1 and len(checkpoint) <= 1
    return receipts


def primary_replication(root, extension):
    comparisons = {}
    for arm in ("wrist-dp","aude-dp"):
        path = Path("matched/pear/active-02")/f"seed-{PEAR_SEEDS[0]}"/arm/"trace.npz"
        with np.load(root/path,allow_pickle=False) as a, np.load(extension/path,allow_pickle=False) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key],err_msg=f"{arm}/{key}")
            comparisons[arm] = a.files
    return comparisons


def make_summary(root):
    extension = root/"transfer-extension"
    pear, transfers = {}, []
    for obj,start in CASES:
        path = Path(obj)/start
        native = root/"evaluation/native"/path
        native_arms = (*NATIVE_ARMS, "aude-single-all-mcc") if obj=="pear" else NATIVE_ARMS
        initial_points = read(native/native_arms[0]/"information.json")["initial_points"]
        for arm in native_arms[1:]:
            np.testing.assert_allclose(read(native/arm/"information.json")["initial_points"],initial_points,rtol=0.,atol=1e-12)
        item = dict(obj=obj, start=start, native={arm:metric(native/arm) for arm in native_arms}, matched={})
        target = root if obj=="pear" else extension
        seeds = PEAR_SEEDS if obj=="pear" else (seed_for(obj,start),)
        for seed in seeds:
            folder = target/"evaluation/matched"/path/f"seed-{seed}"
            methods = METHODS if obj=="pear" and seed==PEAR_SEEDS[0] else ("wrist-dp","aude-dp")
            checks = check_pair(target,obj,start,seed,methods)
            same_budget = {kind:compare(folder,"wrist-"+kind,"aude-"+kind)
                           for kind in ("dp","mcc") if "wrist-"+kind in methods}
            item["matched"][str(seed)] = dict(arms={arm:metric(folder/arm) for arm in methods},
                                                receipt=checks, same_motion_budget=same_budget)
        orig = read(root/"runs"/path/"sweep-mcc/scores.json")
        item["mcc_thresholds_s"] = orig["thresholds_s"]
        if obj == "pear":
            item["mcc_final"] = orig["snapshots"][-1]
            item["mcc_prefix_equal_frames"] = 24001
            pear = item
        else:
            item["label"] = OBJECTS[obj]+" "+start[-2:]
            transfers.append(item)
    arms = ("wrist-dp","aude-dp")
    original = [x["native"] for x in transfers]
    controlled = [next(iter(x["matched"].values()))["arms"] for x in transfers]
    def mean_at(rows, arms, marker, field):
        return {a:(float(np.mean([r[a]["snapshots"][marker][field] for r in rows])) if rows else None) for a in arms}
    def wins(rows, first, second, marker):
        return sum(r[first]["snapshots"][marker]["nats"] > r[second]["snapshots"][marker]["nats"] for r in rows)
    with_1000 = [r for r in controlled if all("n1000" in r[a]["snapshots"] for a in arms)]
    transfer = dict(native_gain=mean_at(original,NATIVE_ARMS,"t240","nats"),
                    native_coverage=mean_at(original,NATIVE_ARMS,"t240","coverage"),
                    native_dp_wins=wins(original,"wrist-dp","aude-mcc","t240"),
                    native_mcc_wins=wins(original,"wrist-mcc","aude-mcc","t240"),
                    matched_gain=mean_at(controlled,arms,"t240","nats"),
                    matched_coverage=mean_at(controlled,arms,"t240","coverage"),
                    matched_new_cells=mean_at(controlled,arms,"t240","new_cells"),
                    matched_surface_variance_drop=mean_at(controlled,arms,"t240","variance_reduction_fraction"),
                    matched_first_1000=mean_at(with_1000,arms,"n1000","nats"),
                    matched_first_1000_pairs=len(with_1000),
                    matched_dp_wins=wins(controlled,*arms,"t240"),
                    matched_first_1000_wins=wins(with_1000,*arms,"n1000"))
    budgets = [next(iter(x["matched"].values()))["same_motion_budget"]["dp"]["arms"] for x in transfers]
    transfer["same_motion_gain"] = {a:float(np.mean([b[a]["nats"] for b in budgets])) for a in arms}
    transfer["same_motion_wins"] = sum(b[arms[0]]["nats"] > b[arms[1]]["nats"] for b in budgets)
    transfer["sensitivity_wins"] = {scale:sum(r[arms[0]]["sensitivity"][scale]["t240"] >
                                             r[arms[1]]["sensitivity"][scale]["t240"] for r in controlled)
                                   for scale in ("0.018","0.039")}
    return dict(pear=pear, transfers=transfers, transfer_means=transfer,
                frozen_protocols=dict(primary=digest(root/"protocol.json"),extension=digest(extension/"protocol.json")),
                primary_replication_equal_fields=primary_replication(root,extension),
                scoring=dict(kernel="Matérn 3/2",signal_variance=1.,noise_variance=.008,length_mm=25,
                             sensitivity_mm=[18,39], voxel_mm=4.5, query_points=EVAL_POINTS,
                             valid_tip_force_min_n=.2, all_five_region_loads_below_n=8,
                             entropy_units="nats, start-conditioned, one actual zero-level observation per occupied voxel"))


def plot(root, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if any(f.name=="Microsoft YaHei" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False
    fig, (left,right) = plt.subplots(1,2,figsize=(12.3,4.8),layout="constrained")
    colors = {"wrist-dp":"#17685f","aude-dp":"#b7612a"}
    for i,seed in enumerate(PEAR_SEEDS):
        for arm in ("wrist-dp","aude-dp"):
            folder=root/"evaluation/matched/pear/active-02"/f"seed-{seed}"/arm
            with np.load(folder/"information-curve.npz") as f:
                val=f["cumulative_nats"][::100]
            lab=("新腕轨" if arm=="wrist-dp" else "Aude")+" · "+str(seed)
            left.plot(np.arange(len(val)),val,color=colors[arm],ls=("-","--")[i],lw=2.1,label=lab)
    left.set(title="梨 · 同一手指策略与动作预算",xlabel="仿真时间（秒）",ylabel="累计信息增益（nat）",xlim=(0,240),ylim=(0,None))
    left.legend(frameon=False,fontsize=9,loc="upper left")
    cases=summary["transfers"]
    for i,(color,label) in enumerate((("#17685f","同一 single_all 执行"),("#8a9190","各自原系统"))):
        delta=[]
        for case in cases:
            group=next(iter(case["matched"].values()))["arms"] if i==0 else case["native"]
            other="aude-dp" if i==0 else "aude-mcc"
            delta.append(group["wrist-dp"]["snapshots"]["t240"]["nats"]-group[other]["snapshots"]["t240"]["nats"])
        right.scatter(delta,np.arange(len(cases))+(.12 if i else -.12),s=42,color=color,label=label,marker=("o","s")[i],zorder=3)
    right.axvline(0,color="#343b39",lw=1)
    right.set_yticks(np.arange(len(cases)),[c["label"] for c in cases])
    right.invert_yaxis()
    right.set(title="七个迁移起点 · 同为 240 秒",xlabel="新腕轨减去 Aude 的累计信息增益（nat）")
    right.legend(frameon=False,fontsize=9,loc="upper center",bbox_to_anchor=(.5,-.16),ncol=2)
    for ax in (left,right):
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.17)
    fig.savefig(root/"information-gain.png",dpi=160)
    plt.close(fig)


def write_report(root, summary):
    pear, transfers, means = summary["pear"], summary["transfers"], summary["transfer_means"]
    primary = pear["native"]
    time = "t240"
    primary_scores = read(root/"runs/pear/active-02/sweep-mcc/scores.json")
    final = pear["mcc_final"]
    lines = ["# 手腕 Active 的纯 MCC 基线与信息增益","",
             "## 梨的纯 MCC 闭环","",
             "冻结过载优先的腕规划和复触逻辑，手指保持四指初态的名义角，由触觉 MCC 逐物理步修正。新采集的 112,933 帧包含初始帧，策略安装数为零。前 24,001 帧的关节状态、腕令、名义指角、实触位置、有效指尖和五区载荷与归档消融逐值相等。","",
             "| 原系统 | 240 s 覆盖 | 有符号 F@5 | 轻载 ≥3 指累计 | 首达 70% | 首达 95% |","|---|---:|---:|---:|---:|---:|"]
    for arm in ("wrist-dp","wrist-mcc","aude-mcc","aude-single-all-mcc"):
        s=primary[arm]["snapshots"][time]
        if arm=="wrist-mcc":
            thresholds=primary_scores["thresholds_s"]
        else:
            src=OUT/"force-priority/runs/pear/active-02/sweep-dp" if arm=="wrist-dp" else OUT/"references/pear/active-02"/arm
            thresholds=read(src/"scores.json")["thresholds_s"]
        lines.append(f"| {NAMES[arm]} | {s['coverage']:.1%} | {s['fscore']:.1%} | {s['light_three_s']:.2f} s | {thresholds['70']:.2f} s | {thresholds['95']:.2f} s |")
    lines += ["",f"纯 MCC 在 1,129.32 秒的最终覆盖为 {final['coverage']:.2%}，有符号 F@5 为 {final['signed']['fscore']:.2%}。所有覆盖均按有效实触的 10 mm 测地邻域计分。","",
              "## 信息增益的口径","",
              "在各方法实际产生的触点位置上，用同一个单位信号方差的 Matérn 3/2 GP 评分。主长度尺度为 25 mm，观测噪声方差为 0.008。指尖力 ≥0.2 N 且五个分区均 <8 N 时，只登记每个 4.5 mm 方格的首次实触，作为隐式面的一个零值观测。条件化相同的初始四指，累计高斯信息量为新增观测前后 ½ log det(I + K/σ²) 的差，自然对数单位为 nat。整条实测采样路径确定后用时序 Cholesky 逐前缀计分。规划器的内部效用、虚拟法向点与真值网格均不进入该分数。 [GP 信息增益定义](https://icml.cc/Conferences/2010/papers/422.pdf)","",
              "表面后验方差另在 512 个固定、面积均匀的真值表面采样点评估，相对四指初态归一化。真实网格只参与这项离线验证、覆盖和 F@5。","",
              "## 梨的原系统信息增益","",
              "| 方法 | 60 s（nat） | 120 s | 240 s | 新触点格 | 前 1,000 格（nat） | 表面平均方差下降 |","|---|---:|---:|---:|---:|---:|---:|"]
    for arm in primary:
        v=primary[arm]["snapshots"]
        lines.append(f"| {NAMES[arm]} | {v['t60']['nats']:.1f} | {v['t120']['nats']:.1f} | {v[time]['nats']:.1f} | {v[time]['new_cells']:,} | {v['n1000']['nats']:.1f} | {v[time]['variance_reduction_fraction']:.1%} |")
    lines += ["","## 梨上固定执行条件，只更换腕规划","",
              "两组共用初态、MuJoCo、100 Hz 腕执行接口、逐物理步 ForceFeedback、相同频率的触觉、同一冻结 single_all 和 50 ms 逻辑推理交接。每组发出 24,000 条 SE(3) 腕令，规划 1,200 个腕段。每个 10 ms 命令限幅 2 mm、0.020 rad，活动半径上限为初始腕半径加 0.13 m。Aude 保留内部腕指联合求解、全局探索和触觉投影，仅把腕令送入共同执行器。其原版 18 帧腕段在此对照中统一为 20 帧，腕部执行器为相同的位置驱动。两个规划器保留各自建图的载荷门槛，离线信息评分统一要求五区均低于 8 N。纯 MCC 对照两组均固定起始名义指角。","",
              "![相同执行条件下的梨与七个迁移起点](information-gain.png)","",
              "| 手指／噪声种子 | 新腕轨 240 s | Aude 240 s | 相对增益 | 前 1,000 新格（新腕／Aude） | 相同总运动预算上限（新腕／Aude） |","|---|---:|---:|---:|---|---|"]
    def budget_value(snaps):
        return f"{snaps['n1000']['nats']:.1f}" if "n1000" in snaps else "未达"
    for seed, pair in pear["matched"].items():
        for kind,budget in pair["same_motion_budget"].items():
            a,b="wrist-"+kind,"aude-"+kind
            u,v=[pair["arms"][arm]["snapshots"] for arm in (a,b)]
            x,y=u[time]["nats"],v[time]["nats"]
            lines.append(f"| {'single_all' if kind=='dp' else '纯 MCC'} · {seed} | {x:.1f} | {y:.1f} | {x/y-1:+.1%} | {budget_value(u)}／{budget_value(v)} | {budget['arms'][a]['nats']:.1f}／{budget['arms'][b]['nats']:.1f} |")
    lines += ["","相同总运动预算取两条实际腕轨各自累计平移和累计转角终值的较小值，分别截到任一上限前最后一帧。两个 DP 种子复用同一物理初态，仅改变策略的采样噪声。",""]
    for seed,pair in pear["matched"].items():
        u,v=(pair["arms"][arm]["snapshots"] for arm in ("wrist-dp","aude-dp"))
        lines.append(f"种子 {seed} 下，240 秒内新腕轨／Aude 分别采到 {u[time]['new_cells']:,}／{v[time]['new_cells']:,} 个新格，"
                     f"在表面查询点上的平均后验方差较初态下降 {u[time]['variance_reduction_fraction']:.1%}／{v[time]['variance_reduction_fraction']:.1%}。"
                     f"实际累计腕平移为 {u[time]['wrist_translation_m']:.2f}／{v[time]['wrist_translation_m']:.2f} m。")
    lines += ["",
              "## 七个迁移起点","",
              "| 物体／起点 | 新腕＋纯 MCC 覆盖 | 新腕＋纯 MCC 信息 | 原系统新腕＋DP／Aude＋MCC（nat） | 同一 DP 下新腕／Aude（nat） | 同一 DP 下前 1,000 格（nat） |","|---|---:|---:|---|---|---|"]
    for case in transfers:
        n=case["native"]
        m=next(iter(case["matched"].values()))["arms"]
        u,v=m["wrist-dp"]["snapshots"],m["aude-dp"]["snapshots"]
        lines.append(f"| {case['label']} | {n['wrist-mcc']['snapshots'][time]['coverage']:.1%} | {n['wrist-mcc']['snapshots'][time]['nats']:.1f} | {n['wrist-dp']['snapshots'][time]['nats']:.1f}／{n['aude-mcc']['snapshots'][time]['nats']:.1f} | {u[time]['nats']:.1f}／{v[time]['nats']:.1f} | {budget_value(u)}／{budget_value(v)} |")
    a,b=means["matched_gain"].values()
    lines += ["",f"对七个起点等权取平均，同一 DP 的新腕轨／Aude 为 {a:.1f}／{b:.1f} nat，新腕轨在 {means['matched_dp_wins']}/7 个起点领先。保留各自原执行系统时，新腕轨＋DP 对完整 Aude＋MCC 有 {means['native_dp_wins']}/7 个起点领先。均值仅描述这些固定初态。","",
              f"两个规划器均取得 1,000 个新格的起点有 {means['matched_first_1000_pairs']} 个，固定这些观测数后新腕轨有 {means['matched_first_1000_wins']} 个领先。"
              f"统一每对的总平移和总转角预算上限后，新腕轨有 {means['same_motion_wins']}/7 个起点领先。"
              f"把评估核长度改为 18／39 mm，240 秒信息增益领先的起点数分别为 {means['sensitivity_wins']['0.018']}/7 和 {means['sensitivity_wins']['0.039']}/7。","",
              "机器可读结果包括 18、25、39 mm 三个核尺度的各项对照、逐组触点数、腕部平移与转动、等运动预算截点和原始轨迹指纹。扩展脚本在主梨与主种子上重跑两臂，逐帧核对全部采集数组。","",
              "[配对结果与审计数据](summary.json) · [冻结主协议](protocol.json) · [冻结迁移协议](transfer-extension/protocol.json) · [纯 MCC 的完整仿真记录](runs/pear/active-02/sweep-mcc/run.json)","",
              "在实验仓库根目录执行下列脚本以复跑全部物理试次、评分和图表。脚本复用原 Active-2-A 的完整基线轨迹。","",
              "```powershell", ".\\Module\\wrist_active\\run_information.ps1", "```"]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("score-transfers","report"))
    parser.add_argument("--root",type=Path,default=RESULTS)
    parser.add_argument("--available",action="store_true")
    args=parser.parse_args()
    if args.action=="score-transfers":
        import torch
        torch.set_num_threads(1)
        transfer_scores(args.root,args.available)
    else:
        result=make_summary(args.root)
        save_json(args.root/"summary.json",result)
        plot(args.root,result)
        write_report(args.root,result)
