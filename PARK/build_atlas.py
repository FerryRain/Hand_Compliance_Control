"""Build the PARK experiment sections in the existing, buildless experiment atlas."""
import argparse
from html import escape
from hashlib import sha256
import json
from pathlib import Path
import shutil

from atlas_plots import render_charts


METHODS = ("PARK", "PalmReach-6", "AUDE")
NAMES = {"coffee":"咖啡罐", "cracker":"饼干盒", "mustard":"芥末瓶", "bleach":"漂白剂瓶", "pear":"梨"}
BUDGETS = (600, 240)
EVIDENCE = "data/park/"


def table(caption, columns, rows):
    header = "".join(f'<th scope="col">{escape(c)}</th>' for c in columns)
    body = []
    for label, values in rows:
        row_class = ' class="pg-mean"' if "平均" in label else ""
        cells = "".join(f"<td>{escape(str(value))}</td>" for value in values)
        body.append(f'<tr{row_class}><th scope="row">{escape(label)}</th>{cells}</tr>')
    return (f'<div class="pg-table"><table><caption>{escape(caption)}</caption><thead><tr>{header}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def links(items):
    return '<p class="pg-sources pg-links">' + "".join(
        f'<a href="{EVIDENCE}{name}" download>{title}</a>' for name, title in items) + '</p>'


def details(summary, content):
    return f'<details class="pg-details"><summary>{summary}</summary><div class="pg-detail-body">{content}</div></details>'


def options(selected, choices):
    return "".join(f'<option value="{v}"' + (' selected' if v == selected else '') + f'>{s}</option>'
                   for v,s in choices.items())


def charts(title, metrics, note):
    figures = []
    for metric, label in metrics.items():
        path = f"images/park/{metric}-mean.png"
        figures.append(f'<figure class="pg-chart" data-metric="{metric}"><a href="{path}">'
                       f'<img src="{path}" width="1020" height="562" loading="lazy" alt="五物体平均的{label}，PARK、PalmReach-6 与 AUDE">'
                       '</a></figure>')
    choices = options("mean", {"mean":"五物体等权平均", **NAMES})
    return (f'<div class="pg-section" data-chart-group="park" data-methods="PARK、PalmReach-6、AUDE">'
            f'<div class="pg-row"><h4>{title}</h4><label>曲线范围 <select>{choices}</select></label></div>'
            f'<div class="pg-grid">{"".join(figures)}</div><p class="pg-note">{note}</p></div>')


def replays():
    obj = "mustard"
    stem = "media/park/600s/mustard"
    mesh = "images/park/600s/mustard.webp"
    return f'''<div class="pg-section" data-park-film>
  <div class="pg-row"><h4>同步探索与终点网格</h4><div class="park-controls">
    <label>物体 <select data-film-object>{options(obj,NAMES)}</select></label>
    <label>探索时长 <select data-film-budget>{options(600,{600:"600 秒",240:"240 秒"})}</select></label></div></div>
  <figure class="pg-film"><video controls playsinline preload="none" poster="{stem}.webp" src="{stem}.mp4" width="1536" height="752" aria-label="芥末瓶的 600 秒同步探索回放"></video>
    <figcaption>左栏为 PARK，中栏为 PalmReach-6，右栏为 AUDE。三栏对齐到同一仿真时刻，上方为手部运动，下方为累计触点及当时最近一次评分使用的网格。<span data-film-speed>600 秒按 20 倍速回放。</span> <a href="{stem}.mp4" data-film-link>打开芥末瓶 600 秒视频</a></figcaption></figure>
  <figure class="pg-mesh"><a data-mesh-link href="{mesh}"><img src="{mesh}" width="1512" height="469" loading="lazy" alt="芥末瓶 600 秒的真值、PARK、PalmReach-6、AUDE 终点重建"></a>
    <figcaption>从左至右为参考真值及三种方法实际评分的终点网格。各列采用相同视角和比例，点击可放大。</figcaption></figure>
  <noscript><p>按物体打开视频。</p>{' '.join(f'<a href="media/park/{s}s/{o}.mp4">{label} {s} 秒</a>' for s in BUDGETS for o,label in NAMES.items())}</noscript>
</div>'''


def method_cards():
    return '''<div class="pg-methods pg-three">
  <div id="active-1-a"><h3><span class="pg-swatch"></span>PARK</h3><p>依据 GPIS 表面不确定度和四指概率探针选择腕部目标，降低重复触达、持续停留区域的评分。轨迹起点未通过间隙检查时，从实测腕姿重新规划。局部掌面支撑、<code>single_all</code> 手指策略与 MCC 执行接触控制。</p></div>
  <div><h3><span class="pg-swatch park-six"></span>PalmReach-6</h3><p>历史对照。依据整手可达域与预测接触收益，在六个切向动作及保持动作中选择。它与 PARK 共用局部调腕、手指策略与 MCC。</p></div>
  <div id="active-1-b"><h3><span class="pg-swatch pg-aude"></span>AUDE</h3><p>Khadivar 等人方法的 LEAP 移植。各指用独立高斯过程生成探索目标，并联合求解腕指动作。</p><p class="pg-sources"><a href="https://doi.org/10.1016/j.robot.2023.104461">原论文</a> · <a href="https://github.com/epfl-lasa/object_exploration/tree/d89a6941fec699768e64fb6acc07439ab822e59f">作者代码</a> · <a href="./#active-1-b-records">早期移植记录</a></p></div>
</div>'''


def score_section(results):
    rows = {(r["object"], r["seconds"], r["method"]):r for r in results["trials"]}
    means = {(r["cohort"], r["seconds"], r["method"]):r for r in results["means"]}
    def pair(r):
        return f'{r["c25_pct"]:.2f} / {r["f5_pct"]:.2f}'
    summaries, detailed = [], []
    for seconds in BUDGETS:
        cells = [(name,[pair(rows[obj,seconds,m]) for m in METHODS]) for obj,name in NAMES.items()]
        cells += [(label,[pair(means[cohort,seconds,m]) for m in METHODS])
                  for cohort,label in (("four","前四物体平均"),("five","五物体平均"))]
        summaries.append(table(f"{seconds} 秒 · C25 / F@5（%）", ("物体",*METHODS), cells))
        values = [(f"{name} · {m}",[f"{rows[obj,seconds,m][k]:.2f}" for k in
                                    ("c10_pct","precision5_pct","recall5_pct","surface_error_mm")])
                  for obj,name in NAMES.items() for m in METHODS]
        detailed.append(table(f"{seconds} 秒", ("物体 · 方法","C10（%）","精确率 @5（%）","召回率 @5（%）","双向表面误差（mm）"), values))
    final = means["five",600,"PARK"]
    coverage_chart = charts("重建与实触覆盖随时间的变化", {"coverage":"实触覆盖","f5":"重建 F@5"},
                            "覆盖率每秒按合格触点计算。F@5 取自 600 秒试验的实际重建检查点，图中的点为评分时刻，连线用于读图。")
    return (f'<div class="pg-section"><h4>逐物体覆盖率与重建结果</h4><p class="pg-result">600 秒时，PARK 的五物体平均覆盖率为 '
            f'<strong>{final["c25_pct"]:.2f}%</strong>，平均重建 F@5 为 <strong>{final["f5_pct"]:.2f}%</strong>。</p>'
            f'<div class="pg-grid">{"".join(summaries)}</div><p class="pg-note">C25 为参考表面上沿表面距离合格触点 25 mm 以内的面积占比。'
            'F@5 为 5 mm 容差下重建精确率和召回率的调和平均。前四物体指咖啡罐、饼干盒、芥末瓶和漂白剂瓶，五物体平均另含梨。每个物体权重相同。</p>'
            + details("查看 C10 覆盖率、精确率、召回率和表面误差", "".join(detailed)) + '</div>'
            + coverage_chart + links([("scores.csv","逐轮终点评分 CSV"),("curves.csv","逐时评分 CSV"),
                                       ("scored-meshes.zip","下载评分网格 PLY"),("wrist-trajectories.zip","下载实测腕姿轨迹"),
                                       ("provenance.json","轨迹、评分与网格校验值"),("release.json","PARK 定稿清单")]))


def info_section(information):
    fields = ("new_info_nat", "issued_decisions", "nat_per_issued_decision", "two_pad_pct", "no_qualified_s", "overload_s")
    cols = ("累计信息（nat）","下发目标数","每次收益（nat）","双指及以上轻触（%）","无合格接触（秒）","过载（秒）")
    def values(row):
        return [f"{row[key]:,.2f}" if not isinstance(row[key], int) else str(row[key]) for key in fields]
    means = {(r["seconds"],r["method"]):r for r in information["means"] if r["cohort"]=="five"}
    runs = {(r["seconds"],r["method"],r["object"]):r for r in information["trials"]}
    blocks, individual, timing = [], [], []
    for s in BUDGETS:
        blocks.append(table(f"{s} 秒 · 五物体平均",("方法",*cols),[(m,values(means[s,m])) for m in METHODS]))
        individual.append(table(f"{s} 秒",("物体 · 方法",*cols),
                                [(f"{name} · {m}",values(runs[s,m,o])) for o,name in NAMES.items() for m in METHODS]))
        timing.append(table(f"{s} 秒 · 五物体平均",("方法","全局更新数","单次中位（ms）","累计（秒）","信息率（nat/s）"),
                            [(m,[f"{means[s,m][k]:,.2f}" for k in
                                 ("high_level_updates","planning_median_ms","planning_total_s","nat_per_s")]) for m in METHODS]))
    current = means[600,"PARK"]
    return (f'<div class="pg-section pg-first"><p class="pg-result">600 秒时，PARK 在五个物体上平均获得 '
            f'<strong>{current["new_info_nat"]:,.2f} nat</strong> 实触信息，至少两指同时合格轻触占运动时间的 '
            f'<strong>{current["two_pad_pct"]:.2f}%</strong>。</p>'+ "".join(blocks)
            + '<p class="pg-note">同时满足有效指腹法向、指腹力至少 0.2 N、整手各区载荷低于 8 N 的实触才参与统计。使用 25 mm 长度尺度的 Matérn 核，以初始实触为条件，在首次触达 4.5 mm 格时计算新增信息。每次收益按目标实际下发至下一个目标或预算终点的增量核算。无合格接触与过载时段可重叠。</p>'
            + details("各物体的信息收益和接触保持", "".join(individual))
            + details("高层更新与规划用时", "".join(timing)
                       + '<p class="pg-note">用时来自原生实验的墙钟记录，会受并发负载影响。PARK 统计全局规划更新，PalmReach-6 统计原始全局事件，AUDE 计入联合腕指求解。每轮先取单次用时中位数，再对五个物体等权平均。</p>') + '</div>'
            + charts("前 240 秒的实触信息收益", {"cumulative":"累计信息","rate":"10 秒信息获取率","decisions":"已下发目标收益"},
                     "速率取每 10 秒的实测增量。目标收益按下发时刻分窗，逐物体曲线的散点表示单次已下发的决策。")
            + links([("information-summary.csv","逐轮信息与接触 CSV"),("decisions.csv","逐决策收益 CSV"),
                     ("information.json","逐秒信息与统计口径 JSON")]))


PROTOCOL = ('本页采用 PARK 的定稿版本 v3。每个物体使用一个四指轻触起点。三种方法共享相同的仿真初态、模型、种子和评分器。'
            '梨的种子为 20261120，其余四个物体为 20261122。240 秒和 600 秒分别启动，每项分数来自对应试验。')


def replace_section(page, start, end, replacement):
    if page.count(start) != 1 or page.count(end) != 1:
        raise ValueError(f"Ambiguous atlas section: {start}")
    left, old = page.split(start)
    _, right = old.split(end)
    return left + start + "\n" + replacement + "\n" + end + right


def build(dist, assets):
    data = assets / "data/park"
    results, curves, info = [json.loads((data / n).read_bytes()) for n in ("results.json","curves.json","information.json")]
    assert all(tuple(v["methods"]) == METHODS for v in (results, curves, info))
    score, information, films = score_section(results), info_section(info), replays()
    render_charts(assets / "images/park", NAMES, curves, info)
    for group in ("data","images","media"):
        shutil.copytree(assets/group/"park", dist/group/"park", dirs_exist_ok=True)
    source = Path(__file__).with_name("release.json").read_bytes()
    lock = json.loads(source)
    scored = {(r["object"],r["seconds"]):r for r in json.loads((data/"provenance.json").read_bytes())["trials"]
              if r["method"] == lock["name"]}
    published = []
    for row in lock["published_runs"]:
        obj, seconds = row["object"], row["seconds"]
        ref = scored[obj,seconds]
        files = row["files"]
        if (ref["seed"] != row["seed"] or ref["trace_sha256"] != files["native/trace.npz"]
                or ref["endpoint_mesh_sha256"] != files[f"score/signed-{seconds}s.ply"]):
            raise ValueError(f"Website inputs disagree with the PARK release: {obj}, {seconds}s")
        published.append(dict(object=obj, seconds=seconds, seed=row["seed"], trace_sha256=ref["trace_sha256"],
                              endpoint_mesh_sha256=ref["endpoint_mesh_sha256"], score_sha256=files["score/metrics.json"]))
    manifest = dict(name=lock["name"], revision=lock["revision"], release_manifest_sha256=sha256(source).hexdigest(),
                    measured_start_sha256=lock["planner_inputs"]["PalmReach-3/measured_start.py"],
                    native_git_commit=lock["native_git_commit"], trials=published)
    (dist/"data/park/release.json").write_text(json.dumps(manifest,indent=2)+"\n", encoding="utf-8")
    home = dist / "index.html"
    page = home.read_text(encoding="utf-8")
    start = '<section class="pg-study" id="active-1" aria-labelledby="active-one-title">'
    history = '<details id="active-1-b-records"'
    intro = (f'<span id="active-one" class="pg-anchor"></span><header class="pg-intro"><p class="small-heading">Active-1 · 重建性能</p>'
             f'<h3 id="active-one-title"><span id="park">PARK 主动触觉探索</span></h3><p>{PROTOCOL}</p></header>')
    page = replace_section(page,start,history,intro+method_cards()+score+films)
    info_start = '<section class="pg-study" id="active-2" aria-labelledby="pg-information-title">'
    page = replace_section(page,info_start,'<!-- active-studies:end -->',
                           '<header class="pg-intro"><p class="small-heading">Active-2 · 信息获取效率</p><h3 id="pg-information-title">每次下发目标，手获得多少新信息？</h3></header>'+information+'</section>')
    page = page.replace("五个 YCB 物体上比较 PalmReach-6 与 Aude LEAP 基线", "五个 YCB 物体上比较 PARK、PalmReach-6 与 AUDE")
    page = page.replace("PalmReach-6、Facing-4 与 Aude 对比", "PARK、PalmReach-6 与 AUDE 对比")
    page = page.replace('href="#active-1">Active-1 · 重建性能', 'href="#active-1">Active-1 · PARK 重建')
    page = page.replace('<time datetime="2026-11-25">2026.11.25</time>', '<time datetime="2026-11-26">2026.11.26</time>')
    home.write_text(page, encoding="utf-8", newline="\n")
    full = dist / "active-methods.html"
    page = full.read_text(encoding="utf-8").replace("PalmReach-6、Facing-4 与 Aude ·", "PARK、PalmReach-6 与 AUDE ·")
    page = page.replace("三种主动探索方法对比</h1>", "PARK 主动探索对比</h1>")
    content = (f'<section id="coverage" class="pg-study" aria-labelledby="coverage-title"><header class="pg-intro">'
               f'<h2 id="coverage-title">PARK、PalmReach-6 与 AUDE</h2><p>{PROTOCOL}</p></header>{method_cards()}{score}</section>'
               '<section id="efficiency" class="pg-study" aria-labelledby="efficiency-title"><header class="pg-intro">'
               f'<h2 id="efficiency-title">信息收益与接触保持</h2></header>{information}</section>'
               '<section id="replays" class="pg-study" aria-labelledby="replays-title"><header class="pg-intro">'
               f'<h2 id="replays-title">五个物体的同步探索回放</h2></header>{films}'
               '<p class="pg-links"><a href="data/park/media.json" download>回放与图像校验值</a><a href="./#park">返回 PARK 主实验</a></p></section>')
    page = replace_section(page, '<div class="wrap active-content">', '</div></main>', content)
    full.write_text(page, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dist", required=True, type=Path)
    cli.add_argument("--assets", required=True, type=Path)
    args = cli.parse_args()
    build(args.dist, args.assets)
