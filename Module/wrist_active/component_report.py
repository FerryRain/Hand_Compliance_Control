"""Compare the two components within genuinely matched trials and publish the local study report."""

from statistics import mean

import numpy as np

from ..i04_geometric_traversal.fingertip_hand import DT as I01_DT
from ..i04_geometric_traversal.i01_a_wrist import STARTUP_S
from .component_study import ARCHIVE, NEW_STARTS, OUT, RESULTS, matched_path, study_jobs
from .experiment import digest, seed_for
from .face_regression_suite import PEAR, REPO, TRANSFERS, read, write
from .guard_report import same_three_seconds
from .information_report import OBJECTS, check_pair, compare
from .matched_information import METHODS
from .sweep import DT, LOAD_LIMIT

NAMES = {**OBJECTS, "pear": "梨"}
ARM_NAMES = {
  "wrist-dp": "腕部 Active＋single_all＋MCC",
  "wrist-mcc": "腕部 Active＋纯 MCC",
  "aude-dp": "Aude 腕目标＋single_all＋MCC",
  "aude-mcc": "Aude 腕目标＋纯 MCC",
  "native-aude": "完整 Aude＋MCC",
}
COMPARISONS = {
  "fingers_at_wrist": ("wrist-dp", "wrist-mcc"),
  "fingers_at_aude": ("aude-dp", "aude-mcc"),
  "wrist_at_dp": ("wrist-dp", "aude-dp"),
  "wrist_at_mcc": ("wrist-mcc", "aude-mcc"),
}
METRICS = (
  "coverage",
  "mean_coverage",
  "f5",
  "light3",
  "continuous",
  "nats",
  "over8",
  "peak",
  "rm70",
  "rm90",
)


def title(case):
  obj, start = case.split("/")
  return NAMES[obj] + " " + start[-2:]


def trial_values(source, quality, nats):
  r = read(source / "run.json")
  h = digest(source / "trace.npz")
  scores = read(quality / "scores.json")
  if h != scores["spec"]["trace_sha256"] or r.get("trace_sha256", h) != h:
    raise ValueError(f"Score and rollout differ for {source}")
  s = next(x for x in scores["snapshots"] if x["s"] == 240)
  thresholds = scores["thresholds_s"]

  def restricted(key):
    t = thresholds[key]
    return 240.0 if t is None else min(240.0, t)

  return dict(
    coverage=s["coverage"],
    mean_coverage=s["mean_coverage"],
    f5=s["signed"]["fscore"],
    light3=s["low3_s"],
    continuous=same_three_seconds(source),
    nats=nats,
    over8=s["over8_s"],
    peak=s["peak_n"],
    rm70=restricted("70"),
    rm90=restricted("90"),
    t70=thresholds["70"],
    t90=thresholds["90"],
    t95=thresholds["95"],
    trace_sha256=h,
    initial_sha256=r.get("initial_sha256", r.get("initial_state_sha256")),
    mesh_sha256=scores["spec"]["mesh_sha256"],
    source=source.relative_to(REPO).as_posix(),
    scored_at=quality.relative_to(REPO).as_posix(),
  )


def assert_paired(rows):
  if (
    any(not r["initial_sha256"] for r in rows.values())
    or len({r["initial_sha256"] for r in rows.values()}) != 1
    or len({r["mesh_sha256"] for r in rows.values()}) != 1
  ):
    raise ValueError("Paired rows do not share initial conditions and evaluation mesh")


def factorial(case):
  obj, start = case.split("/")
  seed = seed_for(obj, start)
  arms, common_motion, initial_points = {}, {}, None
  for fingers in ("dp", "mcc"):
    base = ARCHIVE if case == PEAR or fingers == "dp" else OUT / "controlled"
    names = ("wrist-" + fingers, "aude-" + fingers)
    check_pair(base, obj, start, seed, names)
    for name in names:
      folder = matched_path(base, case, name)
      scored = base / "evaluation" / folder.relative_to(base)
      info = read(scored / "information.json")
      if info["spec"]["study_protocol_sha256"] != digest(ARCHIVE / "protocol.json"):
        raise ValueError("Different matched protocol")
      pts = np.array(info["initial_points"])
      if initial_points is None:
        initial_points = pts
      np.testing.assert_allclose(pts, initial_points, atol=1.0e-12, rtol=0.0)
      final = info["snapshots"]["t240"]
      arm = trial_values(folder, scored, final["nats"])
      if arm["trace_sha256"] != info["spec"]["trace_sha256"] or not np.isclose(
        arm["coverage"], final["coverage"]
      ):
        raise ValueError(f"Information not scored on {folder}")
      arm.update(
        new_cells=final["new_cells"],
        fixed_1000=info["snapshots"].get("n1000", {}).get("nats"),
        translation=final["wrist_translation_m"],
        rotation=final["wrist_rotation_rad"],
      )
      arms[name] = arm
    common_motion[fingers] = compare(scored.parent, *names)
  assert_paired(arms)
  return dict(arms=arms, common_motion=common_motion)


def native_archives():
  old = read(RESULTS / "results.json")["cases"]
  pure = read(RESULTS / "regression/results.json")["native_mcc"]
  result = {}
  for row in old:
    case = row["obj"] + "/" + row["start"]
    src = RESULTS / "regression/native/runs" if case == PEAR else RESULTS / "runs"
    aude = (
      RESULTS / "regression/active-02-a"
      if case == PEAR
      else RESULTS / "regression/strong-baseline" / case
    ) / "aude-mcc"
    paths = {
      "wrist-dp": (src / case / "sweep-dp", row["arms"]["updated"]["information_nats"]),
      "wrist-mcc": (
        RESULTS / "regression/native/runs" / case / "sweep-mcc",
        pure[case]["nats"],
      ),
      "native-aude": (aude, row["arms"]["aude"]["information_nats"]),
    }
    arms = {a: trial_values(p, p, n) for a, (p, n) in paths.items()}
    assert_paired(arms)
    result[case] = dict(arms=arms)
  return result


def native_validation():
  cases = {}
  for case in NEW_STARTS:
    arms = {}
    for fingers in ("dp", "mcc"):
      p = OUT / "native" / case / ("sweep-" + fingers)
      evidence = read(p / "verification.json")
      row = trial_values(p, p, evidence["nats"])
      if (
        row["trace_sha256"] != evidence["trace_sha256"]
        or row["continuous"] != evidence["moving_same_three_s"]
      ):
        raise ValueError(f"Inconsistent native score for {p}")
      arms["wrist-" + fingers] = row
    assert_paired(arms)
    cases[case] = dict(arms=arms)
  return cases


def can_contact_breakdown(native):
  """Recompute the screenshot's contact columns from the two original closed loops."""
  diagnostic = read(RESULTS / "diagnostics/can-finger-commands/results.json")
  idx = diagnostic["finger_order"].index("thumb")
  result = {}
  for start in diagnostic["starts"]:
    case = "meat-can/" + start
    result[case] = {}
    for finger in ("dp", "mcc"):
      arm = native[case]["arms"]["wrist-" + finger]
      path = REPO / arm["source"]
      receipt = read(path / "run.json")
      with np.load(path / "trace.npz", allow_pickle=False) as trace:
        loads = trace["region_load_n"]
        pads = trace["pad_valid"]
        forces = trace["pad_force_n"]
        nominals = trace["finger_nominal"]
      if not (len(pads) == 24001 and pads[0].all() and loads[0].max() < LOAD_LIMIT):
        raise ValueError(f"Unexpected start or rollout length in {path}")
      if np.any(forces[pads] < 0.199999):
        raise ValueError("Valid fingertip has insufficient measured force")
      if finger == "mcc" and (
        receipt["installs"] != 0
        or receipt["policy"] is not None
        or (nominals != nominals[0]).any()
      ):
        raise ValueError("Pure MCC received an additional nominal joint command")
      count = pads[1:].sum(axis=1)
      light = loads[1:].max(axis=1) < LOAD_LIMIT

      def seconds(mask):
        return float(mask.sum() * DT)

      stats = dict(
        one=seconds(light & (count == 1)),
        two_or_more=seconds(light & (count >= 2)),
        exactly_three=seconds(light & (count == 3)),
        four=seconds(light & (count == 4)),
        thumb_light=seconds(light & pads[1:, idx]),
        no_valid_tip=seconds(count == 0),
      )
      if not np.isclose(arm["light3"], stats["exactly_three"] + stats["four"]):
        raise ValueError(
          "Raw three-finger contact disagrees with the independent scorer"
        )
      archived = diagnostic["original_closed_loop_breakdown"][start]["fingers"][finger]
      for key, old_key in [
        ("four", "four_light_s"),
        ("thumb_light", "thumb_light_s"),
        ("no_valid_tip", "all_loads_no_measured_head_s"),
      ]:
        np.testing.assert_allclose(stats[key], archived[old_key], rtol=0, atol=1.0e-8)
      result[case][finger] = dict(
        **stats,
        **{
          k: arm[k] for k in ("coverage", "f5", "light3", "continuous", "nats", "over8")
        },
      )
  averages = {
    f: {
      k: mean(result[c][f][k] for c in result) for k in next(iter(result.values()))[f]
    }
    for f in ("dp", "mcc")
  }
  gap = averages["mcc"]["light3"] - averages["dp"]["light3"]
  return dict(
    by_start=result,
    means=averages,
    four_share_of_gap=(averages["mcc"]["four"] - averages["dp"]["four"]) / gap,
    finger_command_ablation=diagnostic["online_mean"],
    still_wrist_mcc=diagnostic["stationary_mean"],
  )


def neutral_wrist():
  root = RESULTS.parent / "i01_a_smooth_20261121"
  fresh = RESULTS / "regression/i01-a/smooth"
  metrics = read(root / "metrics.json")
  groups = []
  for g in metrics["groups"]:
    pair = [
      next(r for r in g["trials"] if r["method"] == method)
      for method in ("fixed-dp", "fixed-mcc")
    ]
    if not all(
      pair[0][k] == pair[1][k]
      for k in (
        "states",
        "scan_steps",
        "wrist_sha256",
        "initial_sha256",
        "seed",
        "plan",
      )
    ):
      raise ValueError("Different wrist tape, seed or initial state in I01-A")
    steps = np.arange(pair[0]["states"])
    scan = (steps * I01_DT >= STARTUP_S) & (
      steps < read(root / pair[0]["plan"] / "wrist_plan.json")["scan_end_step"]
    )
    masks = []
    three = []
    two = []
    for r in pair:
      with np.load(
        fresh / r["trial"] / "live_head_contacts.npz", allow_pickle=False
      ) as f:
        h = np.zeros((len(steps), 4), dtype=bool)
        h[f["step"] - 1, f["finger"]] = True
      with np.load(root / r["trial"] / "moving_contact.npz", allow_pickle=False) as f:
        np.testing.assert_array_equal(h.sum(1), f["head_count"])
        light = f["region_load_n"] < 8.0
      masks.append(h)
      three.append(scan & light & (h.sum(1) >= 3))
      two.append(scan & light & (h.sum(1) == 2))
    dp, mcc = three
    gain_separate = dp & ~mcc
    loss_separate = mcc & ~dp
    gain_same = dp & two[1] & np.all(~masks[1] | masks[0], axis=1)
    loss_same = mcc & two[0] & np.all(~masks[0] | masks[1], axis=1)
    assert np.isclose(dp.sum() / scan.sum(), pair[0]["low_three_scan"])
    assert np.isclose(mcc.sum() / scan.sum(), pair[1]["low_three_scan"])
    groups.append(
      dict(
        case=g["object"] + "/" + g["start"],
        scan_s=float(scan.sum() * I01_DT),
        dp_only_light3_s=float(gain_separate.sum() * I01_DT),
        mcc_only_light3_s=float(loss_separate.sum() * I01_DT),
        mcc_exact_two_s=float(two[1].sum() * I01_DT),
        dp_exact_two_s=float(two[0].sum() * I01_DT),
        dp_retains_same_two_and_adds_s=float(gain_same.sum() * I01_DT),
        mcc_retains_same_two_and_adds_s=float(loss_same.sum() * I01_DT),
        dp_three_rate=pair[0]["low_three_scan"],
        mcc_three_rate=pair[1]["low_three_scan"],
        dp_cont_rate=pair[0]["sustained_three_scan"],
        mcc_cont_rate=pair[1]["sustained_three_scan"],
        wrist_sha256=pair[0]["wrist_sha256"],
      )
    )
  totals = {k: sum(r[k] for r in groups) for k in groups[0] if k.endswith("_s")}
  return dict(
    groups=groups,
    totals=totals,
    dp_retention_fraction=totals["dp_retains_same_two_and_adds_s"]
    / totals["mcc_exact_two_s"],
    mcc_retention_fraction=totals["mcc_retains_same_two_and_adds_s"]
    / totals["dp_exact_two_s"],
    archived_overall=metrics["means"]["overall"],
    archived_paired=metrics["paired"],
  )


def means(cases):
  labels = next(iter(cases.values()))["arms"]
  out = {}
  for arm in labels:
    rows = [v["arms"][arm] for v in cases.values()]
    out[arm] = {m: mean(v[m] for v in rows) for m in METRICS}
    for k in ("t70", "t90"):
      out[arm][k + "_reached"] = sum(r[k] is not None and r[k] <= 240 for r in rows)
  return out


def differences(cases, left, right):
  out = {}
  for m in METRICS:
    ds = [v["arms"][left][m] - v["arms"][right][m] for v in cases.values()]
    out[m] = dict(
      mean=mean(ds),
      higher=sum(d > 1.0e-9 for d in ds),
      lower=sum(d < -1.0e-9 for d in ds),
    )
  return out


def make_results():
  spec = read(OUT / "protocol.json")
  native = native_archives()
  newer = native_validation()
  matched = {case: factorial(case) for case in TRANSFERS}
  for family, case, method in study_jobs():
    receipt = read(OUT / "receipts" / family / case / (method + ".json"))
    arm = (
      matched[case]["arms"][method]
      if family == "matched"
      else newer[case]["arms"][method.replace("sweep-", "wrist-")]
    )
    if receipt["scored"] != 240 or receipt["trace_sha256"] != arm["trace_sha256"]:
      raise ValueError(
        f"Receipt differs from scored trace for {family} {case} {method}"
      )
  actual = {case: native[case] for case in TRANSFERS}
  matched_means = means(matched)
  contrast = {key: differences(matched, *pair) for key, pair in COMPARISONS.items()}
  interaction = {
    k: contrast["fingers_at_wrist"][k]["mean"] - contrast["fingers_at_aude"][k]["mean"]
    for k in METRICS
  }
  motion = {}
  for fingers in ("dp", "mcc"):
    per = [row["common_motion"][fingers]["arms"] for row in matched.values()]
    left, right = "wrist-" + fingers, "aude-" + fingers
    motion[fingers] = dict(
      wrist_nats=mean(v[left]["nats"] for v in per),
      aude_nats=mean(v[right]["nats"] for v in per),
      wrist_higher=sum(v[left]["nats"] > v[right]["nats"] for v in per),
    )
  return dict(
    protocol_sha256=digest(OUT / "protocol.json"),
    checkpoint_sha256=spec["checkpoint"],
    matched_transfer=matched,
    matched_means=matched_means,
    matched_contrasts=contrast,
    matched_interaction=interaction,
    matched_common_motion=motion,
    matched_pear=factorial(PEAR),
    native_transfer=actual,
    native_transfer_means=means(actual),
    native_transfer_contrast=differences(actual, "wrist-dp", "wrist-mcc"),
    native_validation=newer,
    native_validation_means=means(newer),
    native_validation_contrast=differences(newer, "wrist-dp", "wrist-mcc"),
    native_pear=native[PEAR],
    fixed_wrist_i01=neutral_wrist(),
    can=can_contact_breakdown(native),
  )


def figure(data, filename="finger-effects.png"):
  import matplotlib

  matplotlib.use("Agg")
  from matplotlib import pyplot as plt

  plt.rcParams.update(
    {
      "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
      "axes.unicode_minus": False,
      "font.size": 10,
    }
  )
  fig, axes = plt.subplots(1, 2, figsize=(10.5, 6.2), sharey=True)
  original, validation = data["native_transfer"], data["native_validation"]
  native = {**original, **validation}
  items = list(native.items())
  y = np.r_[np.arange(len(original)), np.arange(len(validation)) + len(original) + 0.65]
  for ax, metric, scale, label in zip(
    axes,
    ("coverage", "light3"),
    (100, 1),
    ("240 秒实触覆盖差（百分点）", "轻载至少三指累计时长差（秒）"),
    strict=True,
  ):
    d = [
      (row["arms"]["wrist-dp"][metric] - row["arms"]["wrist-mcc"][metric]) * scale
      for _, row in items
    ]
    for idx, ((case, _), value) in enumerate(zip(items, d, strict=True)):
      color = "#126b74" if case in original else "#9a4d30"
      ax.hlines(
        y[idx], min(0, value), max(0, value), color=color, alpha=0.6, linewidth=2
      )
      ax.scatter(
        value,
        y[idx],
        color=color,
        marker="o" if case in original else "D",
        s=36,
        zorder=3,
      )
    ax.axvline(0, color="#343b3d", linewidth=1)
    ax.axhline(len(original) - 0.16, color="#a5aaa8", linestyle=":", linewidth=0.8)
    ax.set_xlabel(label)
    ax.grid(axis="x", color="#d9dfdf", alpha=0.6)
    for k in ("top", "right"):
      ax.spines[k].set_visible(False)
  axes[0].set_yticks(y, [title(k) for k, _ in items])
  axes[0].invert_yaxis()
  fig.suptitle("同一腕部 Active 下，加入 single_all 后的逐起点变化", fontsize=14)
  fig.text(
    0.52,
    0.905,
    f"圆点为原来的 {len(original)} 个迁移起点，菱形为新增的 {len(validation)} 个曲面起点。正值表示 single_all＋MCC 较高。",
    ha="center",
    fontsize=10,
  )
  fig.tight_layout(rect=(0, 0, 1, 0.89))
  fig.savefig(OUT / filename, dpi=165, bbox_inches="tight")
  plt.close(fig)


def report(d):
  can = d["can"]
  a, b = can["means"]["dp"], can["means"]["mcc"]
  matched_can = means({case: d["matched_transfer"][case] for case in can["by_start"]})
  lines = [
    "# 腕部 Active 与 single_all 的独立贡献",
    "",
    "## 午餐肉罐上的直接对照",
    "",
    "两个起点都从四指轻载接触开始。修正后的腕规划器在两组使用同一份代码，并按各组实测触觉在线重规划。纯 MCC 组的 16 个名义关节角在各自 24,000 个控制步内保持初值，两组使用相同的指端力反馈。有效指尖的法向力至少为 0.2 N，轻载要求五个分区的载荷均低于 8 N。",
    "",
    "| 240 秒内两个起点等权平均 | 实触覆盖 | F@5 mm | 恰好一指 | 至少两指 | 至少三指 | 累计信息 | 分区载荷 ≥8 N |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for name, row in [("修正后＋single_all", a), ("修正后＋纯 MCC", b)]:
    lines.append(
      f"| {name} | {row['coverage']:.2%} | {row['f5']:.2%} | {row['one']:.2f} s | {row['two_or_more']:.2f} s | {row['light3']:.2f} s | {row['nats']:,.1f} nat | {row['over8']:.2f} s |"
    )
  lines += [
    "",
    f"加入 single_all 后，罐体上轻载至少三指少了 {b['light3'] - a['light3']:.2f} 秒。其中 {b['four'] - a['four']:.2f} 秒来自四指同时轻载的差额，占 {can['four_share_of_gap']:.1%}。两组拇指轻载时间为 {a['thumb_light']:.2f} 与 {b['thumb_light']:.2f} 秒。四指已经包含在至少三指内。两组随腕运动时连续保留同三指达 1 秒的累计均为 0 秒。",
    "",
    f"只让 single_all 调整拇指并固定其余三指的名义角时，轻载至少三指回升至 {can['finger_command_ablation']['thumb-only-dp']['at_least_three_s']:.2f} 秒。只让策略调整其余三指时为 {can['finger_command_ablation']['long-only-dp']['at_least_three_s']:.2f} 秒。这两条命令消融帮助定位手型变化的影响，完整策略仍按原有四指输出评分。固定手腕时，纯 MCC 从初始握姿可维持四指轻载 240 秒，而实际表面覆盖仅 {can['still_wrist_mcc']['coverage']:.2%}。",
    "",
    f"新补齐的四组对照给每种方法相同的腕执行器和限幅。在这两个罐体起点，腕部 Active 配 single_all 的轻载至少三指累计为 {matched_can['wrist-dp']['light3']:.2f} 秒，配纯 MCC 为 {matched_can['wrist-mcc']['light3']:.2f} 秒。换用 Aude 腕目标后，同两组分别为 {matched_can['aude-dp']['light3']:.2f} 和 {matched_can['aude-mcc']['light3']:.2f} 秒。手指策略的增量随腕部指令来源变化。 [罐体逐指消融与交叉回放](../gpis_wrist_face_transfer_20261121/diagnostics/can-finger-commands/results.json)",
    "",
    "## 试验范围",
    "",
    "闭环分量对照每条运行 240 仿真秒。原七个迁移起点以相同初态、手指力反馈、腕令限幅、步数和 GP 评分完成四格对照。沿用已核验的原始 DP 轨迹，补跑其中缺少的 14 条固定名义手型试次。梨在其独立初态上单列。另对三个曲面剩余的六个起点各新跑两条原生闭环，检查手指收益能否迁移到不同起始抓姿。",
    "",
    "## 相同腕部执行接口下的四组结果",
    "",
    "| 七个起点等权 | GP 信息（nat） | 10 mm 实触覆盖 | 轻载至少三指（秒） | 同三指随腕连续至少 1 秒的累计（秒） | F@5 mm | 达到 90% 的起点 |",
    "|---|---:|---:|---:|---:|---:|---:|",
  ]
  for arm in METHODS:
    x = d["matched_means"][arm]
    lines.append(
      f"| {ARM_NAMES[arm]} | {x['nats']:,.1f} | {x['coverage']:.2%} | {x['light3']:.2f} | {x['continuous']:.2f} | {x['f5']:.2%} | {x['t90_reached']}/7 |"
    )
  lines += [
    "",
    "Aude 在这个试验中提供腕目标。两种腕规划器都接同一个腕执行器，单步限幅为 2 mm 和 0.020 rad。纯 MCC 固定初始名义关节角，DP 组使用同一冻结权重，所有组继续执行相同的指端 MCC。完整 Aude 保留原腕指联合控制，其表现列在原生系统表。",
    "",
    "| 单因素配对差 | 覆盖差（百分点） | 轻载至少三指差（秒） | GP 信息差（nat） |",
    "|---|---:|---:|---:|",
  ]
  for key, label in [
    ("fingers_at_wrist", "腕部 Active 上加入 single_all"),
    ("fingers_at_aude", "Aude 腕目标上加入 single_all"),
    ("wrist_at_dp", "固定 single_all，将 Aude 腕目标换为腕部 Active"),
    ("wrist_at_mcc", "固定纯 MCC，将 Aude 腕目标换为腕部 Active"),
  ]:
    row = d["matched_contrasts"][key]
    lines.append(
      f"| {label} | {100 * row['coverage']['mean']:+.2f} | {row['light3']['mean']:+.2f} | {row['nats']['mean']:+,.1f} |"
    )
  inter = d["matched_interaction"]
  lines += [
    "",
    f"四格交互项按两种腕规划器上加入 single_all 所得增量之差计算。覆盖为 {100 * inter['coverage']:+.2f} 个百分点，三指累计为 {inter['light3']:+.2f} 秒，GP 信息为 {inter['nats']:+,.1f} nat。以上数值描述这批配对试次。",
  ]
  for fingers, label in [("dp", "single_all"), ("mcc", "纯 MCC")]:
    x = d["matched_common_motion"][fingers]
    pairs = [
      (
        v["arms"]["wrist-" + fingers]["fixed_1000"],
        v["arms"]["aude-" + fingers]["fixed_1000"],
      )
      for v in d["matched_transfer"].values()
    ]
    eligible = [
      (ours, aude) for ours, aude in pairs if ours is not None and aude is not None
    ]
    aude_ahead = sum(aude > ours for ours, aude in eligible)
    lines += [
      "",
      f"手指控制采用 {label} 时，按每对实际总腕平移与转角的共同上限截断，腕部 Active 和 Aude 腕目标的平均信息为 {x['wrist_nats']:,.1f} 与 {x['aude_nats']:,.1f} nat。腕部 Active 在 {x['wrist_higher']}/7 个起点较高。改以各自取得的前 1,000 个新接触体素为预算，双方均达到该预算的 {len(eligible)} 个起点中，Aude 腕目标在 {aude_ahead} 个上的信息较高。",
    ]
  lines += [
    "",
    "## 原生闭环和新增起点",
    "",
    "![原生腕部 Active 上逐起点的覆盖和三指时长差](finger-effects.png)",
    "",
    "| 试验 | 控制 | 实触覆盖 | 全程平均覆盖 | F@5 mm | 轻载至少三指 | 同三指随腕连续至少 1 秒的累计 | 达到 90% | 到 90% 的限时均值 | 分区载荷 ≥8 N |",
    "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for label, group in [
    ("原七个迁移起点", "native_transfer"),
    ("新增六个曲面起点", "native_validation"),
  ]:
    size = len(d[group])
    for arm, x in d[group + "_means"].items():
      lines.append(
        f"| {label} | {ARM_NAMES[arm]} | {x['coverage']:.2%} | {x['mean_coverage']:.2%} | {x['f5']:.2%} | {x['light3']:.2f} s | {x['continuous']:.2f} s | {x['t90_reached']}/{size} | {x['rm90']:.2f} s | {x['over8']:.2f} s |"
      )
  for label, group in [
    ("原七个迁移起点", "native_transfer"),
    ("新增六个曲面起点", "native_validation"),
  ]:
    size = len(d[group])
    change = d[group + "_contrast"]
    lines += [
      "",
      f"{label}中，加入 single_all 后，轻载至少三指平均改变 {change['light3']['mean']:+.2f} 秒，{change['light3']['higher']}/{size} 个起点增加。覆盖平均改变 {100 * change['coverage']['mean']:+.2f} 个百分点，{change['coverage']['higher']}/{size} 个起点提高。",
    ]
  lines += [
    "",
    "到 90% 的限时均值为每条试次达到阈值或运行至 240 秒的较早时间，再对同组取均值。全程平均覆盖为每个 10 ms 帧的累计覆盖率取平均。原生两组的腕令由各自触觉闭环产生。新增六个起点曾用于 I01-A 固定腕轨测试，未用于前一轮腕部 Active 对照，包含同一物体的不同起姿。",
    "",
    "| 原生试次 | 实触覆盖 · DP／纯 MCC | 轻载至少三指（秒）· DP／纯 MCC | 连续同三指 ≥1 s（秒）· DP／纯 MCC |",
    "|---|---:|---:|---:|",
  ]
  for case, row in {**d["native_transfer"], **d["native_validation"]}.items():
    a, b = (row["arms"][k] for k in ("wrist-dp", "wrist-mcc"))
    lines.append(
      f"| {title(case)} | {a['coverage']:.2%}／{b['coverage']:.2%} | {a['light3']:.2f}／{b['light3']:.2f} | {a['continuous']:.2f}／{b['continuous']:.2f} |"
    )
  p = d["native_pear"]["arms"]
  a, b = p["wrist-dp"], p["wrist-mcc"]
  lines += [
    "",
    f"独立的梨起点在 240 秒时，腕部 Active 配 single_all／纯 MCC 的覆盖为 {a['coverage']:.2%}／{b['coverage']:.2%}，F@5 mm 为 {a['f5']:.2%}／{b['f5']:.2%}，轻载至少三指累计为 {a['light3']:.2f}／{b['light3']:.2f} 秒。长程试次首次达到 95% 覆盖分别在 {a['t95']:.2f}／{b['t95']:.2f} 秒。",
    "",
    "## 固定中性腕令时的手指贡献",
    "",
  ]
  fixed = d["fixed_wrist_i01"]
  m = fixed["archived_overall"]
  a, b = m["fixed-dp"], m["fixed-mcc"]
  t = fixed["totals"]
  lines += [
    f"I01-A 平滑曲面的九条腕令在仿真前依共同初始手型生成。single_all／纯 MCC 的轻载至少三指率分别为 {a['low_three_scan']:.2%}／{b['low_three_scan']:.2%}。按至少三指状态连续 1 秒的原 I01-A 指标分别为 {a['sustained_three_scan']:.2%}／{b['sustained_three_scan']:.2%}，这里的三指组合允许变化。",
    "",
    f"在纯 MCC 恰有两指轻载的 {t['mcc_exact_two_s']:.2f} 秒中，DP 在 {t['dp_retains_same_two_and_adds_s']:.2f} 秒里保留两根相同指尖并至少增加一根，占 {fixed['dp_retention_fraction']:.2%}。反方向，DP 恰有两指轻载的 {t['dp_exact_two_s']:.2f} 秒中，纯 MCC 保留这两指并增加一指的比例为 {fixed['mcc_retention_fraction']:.2%}。两个条件的分母各自列出。",
    "",
    f"按共同时间轴，只有 DP 达到轻载至少三指的累计时间为 {t['dp_only_light3_s']:.2f} 秒，只有纯 MCC 达到的为 {t['mcc_only_light3_s']:.2f} 秒。原 I01-A 同腕轨报告还包括覆盖、重建和过载数据。",
    "",
    "各物体的不同起姿共享几何，配对单位为起点。腕部实时测量来自另做的实时时钟仿真。",
    "",
    "[冻结协议](protocol.json) · [逐试次指标与轨迹校验](results.json) · [原始复验](../gpis_wrist_face_transfer_20261121/regression/REPORT.md)",
  ]
  (OUT / "ALL_OBJECTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
  data = make_results()
  write(OUT / "results.json", data)
  figure(data)
  report(data)


if __name__ == "__main__":
  main()
