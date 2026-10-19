"""Scientific three-view figure from ONE independently audited fresh trace.

This report helper is not a controller dependency. It never reads live chunks,
steps physics, changes a mesh, or grants coverage. A checkpoint must first be
frozen and audited as its own complete trace/mask bundle. Supplying end_s labels
that bundle CURRENT PREFIX; it cannot apply a later mask to an earlier trace.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

FINGERS = ('食指 index', '中指 middle', '无名指 ring', '拇指 thumb')
COLORS = ('#176f91', '#4b8c3b', '#8b5aa8', '#d88725')


def _read(path, expected=None):
    raw = Path(path).read_bytes()
    actual = sha256(raw).hexdigest()
    if expected is not None and expected != actual:
        raise ValueError(f'Input changed or audit mismatch: {path}')
    return raw, dict(path=str(Path(path).resolve()), sha256=actual, bytes=len(raw))


def _runs(points, active, times, finger, spacing=.0002):
    """Indices stay in one physical finger's contiguous measured-active run."""
    valid = active[:, finger] & np.isfinite(points[:, finger]).all(axis=1)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    jump = np.linalg.norm(np.diff(points[indices, finger], axis=0), axis=1)
    dt = float(np.median(np.diff(times)))
    breaks = np.flatnonzero((np.diff(indices) != 1) | (jump > .005)
                            | (np.diff(times[indices]) > 1.5*dt)) + 1
    result = []
    for group in np.split(indices, breaks):
        keep = [int(group[0])]
        previous = points[keep[0], finger]
        for i in group[1:]:
            point = points[i, finger]
            if np.dot(point-previous, point-previous) >= spacing**2:
                keep.append(int(i)); previous = point
        if keep[-1] != group[-1]:
            keep.append(int(group[-1]))
        result.append(np.asarray(keep, dtype=np.int64))
    return result


def _display(points):
    # Display-only axis permutation, identical geometry and physical lengths.
    return np.asarray(points)[..., [0, 2, 1]]*1000.


def _object_transform(model_bytes):
    tree = ET.fromstring(model_bytes)
    body = next(b for b in tree.iter('body') if b.find("geom[@name='mustard_visual']") is not None)
    pos = np.fromstring(body.attrib['pos'], sep=' ')
    q = np.fromstring(body.attrib.get('quat', '1 0 0 0'), sep=' ')
    if pos.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[pos,q]).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('Invalid original object transform')
    q = q/np.linalg.norm(q)
    w,x,y,z = q
    rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return pos,rotation


def render_contact_figure(case, output, end_s=None):
    """Render final or explicitly marked verified checkpoint, never a live run.

    Required case files: trace.npz, audit.json, coverage_independent.npz,
    manifest.json, summary.json, model.xml, surface_graph.npz. The audit must
    identify this exact trace/model/manifest/summary and confirm acquisition
    continuity with no inherited development ledger. A caller preparing a live
    prefix must freeze and independently audit those files before calling.
    """
    case, output = Path(case).resolve(), Path(output).resolve()
    if end_s is not None and (not np.isfinite(end_s) or end_s <= 0):
        raise ValueError('Positive finite verified checkpoint time required')
    raw, audit_source = _read(case/'audit.json')
    audit = json.loads(raw)
    if (audit.get('status') != 'PASSED'
            or audit.get('continuous_from_acquisition_initial_state') is not True
            or audit.get('inherited_development_ledger') is not False):
        raise ValueError('Figure requires independently audited fresh acquisition continuity')
    sources = {'audit.json': audit_source}
    small_bytes = {'audit.json': raw}
    for filename, field in (('manifest.json','manifest_sha256'), ('summary.json','summary_sha256'),
                             ('model.xml','model_xml_sha256')):
        expected = audit.get(field)
        if not isinstance(expected, str):
            raise ValueError(f'Audit does not bind {filename}')
        small_bytes[filename], sources[filename] = _read(case/filename, expected)
    manifest = json.loads(small_bytes['manifest.json'])
    summary = json.loads(small_bytes['summary.json'])
    if manifest.get('initial_state', {}).get('integration_only'):
        raise ValueError('Injected terminal integrations cannot be labeled fresh')
    for filename in ('surface_graph.npz','coverage_independent.npz'):
        small_bytes[filename], sources[filename] = _read(case/filename)
    with np.load(BytesIO(small_bytes['surface_graph.npz']), allow_pickle=False) as z:
        vertices,faces,required = z['vertices'],z['faces'],z['required_vertices']
    with np.load(BytesIO(small_bytes['coverage_independent.npz']), allow_pickle=False) as z:
        covered = z['covered_vertices'].astype(bool)
    expected = audit.get('trace_sha256')
    if not isinstance(expected, str):
        raise ValueError('Audit does not bind one frozen trace')
    raw,sources['trace.npz'] = _read(case/'trace.npz',expected)
    with np.load(BytesIO(raw), allow_pickle=False) as z:
        times,points,active = z['time_s'],z['contact_positions_world_m'],z['active'].astype(bool)
        recorded_mask = z['covered_vertices'].astype(bool)
    del raw
    if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
            or points.shape != (len(times),4,3) or active.shape != (len(times),4)):
        raise ValueError('Expected one chronological four-finger measured trace')
    if (covered.shape != (len(vertices),) or not np.array_equal(covered,recorded_mask)
            or int(covered.sum()) != audit['vertices']
            or int(covered[required].sum()) != audit['required_goals']
            or len(vertices) != audit['total_vertices'] or len(required) != audit['total_required_goals']):
        raise ValueError('Independent mask, recorded final mask, graph and audit disagree')
    if (summary['covered_vertex_count'] != int(covered.sum())
            or summary['visited_required_goals'] != int(covered[required].sum())):
        raise ValueError('Summary and independent audited mask disagree')
    dt = float(np.median(np.diff(times)))
    if end_s is not None and (abs(float(times[-1])-end_s) > dt+1e-8
                             or abs(float(summary['executed_s'])-end_s) > dt+1e-8):
        raise ValueError('Requested prefix must match the frozen audited trace/mask endpoint')
    if abs(float(audit['coverage_radius_m'])-.025) > 1e-12:
        raise ValueError('This mustard figure labels the original 25 mm coverage rule only')
    checkpoint = end_s is not None
    final_complete = (bool(covered.all()) and audit.get('completion_hold_verified') is True
                      and audit.get('full_surface_complete') is True)
    if checkpoint:
        status = 'CURRENT PREFIX / 已核验当前前缀，非最终完成结论'
    else:
        status = '全部原始顶点与必访点已覆盖，稳定保持已核验' if final_complete else '实验停止状态：尚未获得全表面完成结论'
    translation,rotation = _object_transform(small_bytes['model.xml'])
    local = (points-translation)@rotation
    runs = [_runs(local,active,times,finger) for finger in range(4)]
    output.mkdir(parents=True,exist_ok=True)
    stem = 'fresh_actual_contact_three_views'
    png,pdf,provenance_file = (output/(stem+suffix) for suffix in ('.png','.pdf','.json'))
    if any(p.exists() for p in (png,pdf,provenance_file)):
        raise FileExistsError('Choose a fresh output directory; prior figure evidence is preserved')
    source_dir = output/(stem+'_sources'); source_dir.mkdir(exist_ok=False)
    for name,content in small_bytes.items():(source_dir/name).write_bytes(content)
    compact = {}
    for finger,groups in enumerate(runs):
        indices = np.concatenate(groups) if groups else np.zeros(0,dtype=np.int64)
        compact[f'finger_{finger}_source_indices'] = indices
        compact[f'finger_{finger}_source_time_s'] = times[indices]
        compact[f'finger_{finger}_point_local_m'] = local[indices,finger]
        compact[f'finger_{finger}_run_offsets'] = np.r_[0,np.cumsum([len(g) for g in groups])]
    display_file = output/(stem+'_display_points.npz')
    np.savez_compressed(display_file,**compact)

    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False})
    fig = plt.figure(figsize=(18,10.7),facecolor='white')
    version = manifest['protocol'].split('.development.')[-1]
    fig.suptitle(f'Fresh {version} 单次实验：mustard 原始表面与实测接触轨迹',
                 fontsize=20,fontweight='bold',y=.975,color='#172936')
    area = 100*float(audit['area_fraction'])
    fig.text(.5,.926,f'{summary["executed_s"]:.3f} s  |  顶点 {covered.sum()} / {len(vertices)}'
             f'  |  必访点 {covered[required].sum()} / {len(required)}  |  面积覆盖 {area:.4f}%',
             ha='center',fontsize=13,color='#273d47')
    fig.text(.5,.889,status,ha='center',fontsize=11,color='#56656b' if final_complete else '#9b432b')
    display = _display(vertices);triangles = display[faces]
    missing_faces = ~covered[faces].all(axis=1)
    missing_required = required[~covered[required]]
    views = [(18,-80,'视角 1 / 原坐标 −Z 一侧'),(18,100,'视角 2 / 原坐标 +Z 一侧'),
             (64,-32,'视角 3 / 原坐标 +Y 端部斜视')]
    for col,(elev,azim,title) in enumerate(views):
        ax = fig.add_axes([.012+col*.329,.245,.321,.60],projection='3d',computed_zorder=False)
        ax.set_proj_type('ortho')
        ax.add_collection3d(Poly3DCollection(triangles,facecolors='#e1d8b6',edgecolors='#c4bda6',
                                             linewidths=.035,alpha=.14,zorder=1))
        if missing_faces.any():
            ax.add_collection3d(Poly3DCollection(triangles[missing_faces],facecolors='#d56b74',
                                                 edgecolors='none',alpha=.24,zorder=2))
        for finger,groups in enumerate(runs):
            lines = [_display(local[group,finger]) for group in groups if len(group)>1]
            if lines:
                ax.add_collection3d(Line3DCollection(lines,colors=COLORS[finger],linewidths=.58,
                                                     alpha=.65,zorder=3+finger),autolim=False)
            single = [local[group[0],finger] for group in groups if len(group)==1]
            if single:
                ax.scatter(*_display(single).T,c=COLORS[finger],s=1.3,alpha=.6,depthshade=False,zorder=3+finger)
        if len(missing_required):
            ax.scatter(*display[missing_required].T,marker='o',s=15,facecolors='none',edgecolors='#b91f3a',
                       linewidths=.7,depthshade=False,zorder=9)
        ax.set_xlim(-145,145);ax.set_ylim(-8,195);ax.set_zlim(-278,280)
        ax.set_box_aspect((290,203,558));ax.view_init(elev=elev,azim=azim)
        ax.set_xlabel('原 X / mm',labelpad=4,fontsize=9)
        ax.set_ylabel('原 Z / mm',labelpad=4,fontsize=9)
        ax.set_zlabel('原 Y / mm',labelpad=4,fontsize=9)
        ax.set_xticks([-100,0,100]);ax.set_yticks([0,100]);ax.set_zticks([-200,0,200])
        ax.tick_params(labelsize=8,pad=1)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.fill=False;axis.pane.set_edgecolor('#ededed')
            axis._axinfo['grid']['color']=(.83,.85,.86,.35)
        ax.set_title(title,pad=12,fontsize=12,color='#334d5a')
    handles = [Line2D([0],[0],color=color,lw=2,label=name) for name,color in zip(FINGERS,COLORS)]
    handles += [Patch(facecolor='#d56b74',alpha=.45,label=f'含未覆盖顶点的面（未覆盖顶点 {np.count_nonzero(~covered)}）'),
                Line2D([0],[0],color='#b91f3a',marker='o',markerfacecolor='none',ls='',markersize=6,
                       label=f'尚未覆盖的必访点：{len(missing_required)}')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.163),ncol=3,frameon=False,
               fontsize=10.1,columnspacing=2.,handlelength=2.6,labelspacing=.9)
    fig.text(.5,.120,'覆盖定义：实测接触点的 25 mm 原始 mesh 测地邻域；不代表每个顶点都被指尖直接碰到。',
             ha='center',fontsize=12,color='#153f51',weight='bold')
    fig.text(.5,.080,'只画同指真实 active 接触段；失联、缺采样、接触点跳变 >5 mm 均断线，不画离面运动。',
             ha='center',fontsize=10,color='#53626a')
    fig.text(.5,.051,f'记录间隔约 {1000*dt:g} ms；显示点压缩阈值 0.20 mm，保留各段端点。透明网格展示背面轨迹；红色面保守标出未覆盖边界。',
             ha='center',fontsize=9.5,color='#53626a')
    fig.text(.5,.024,'本图仅含此单次 fresh 实验；未合并旧实验或续接。原始几何、实际接触判据和独立覆盖账本均未改动。',
             ha='center',fontsize=9.5,color='#53626a')
    fig.savefig(png,dpi=180);fig.savefig(pdf,dpi=180);plt.close(fig)
    provenance = dict(scope='SINGLE_FRESH_VERIFIED_CHECKPOINT' if checkpoint else 'SINGLE_FRESH_FINAL',
        source_case=str(case),sources=sources,requested_end_s=end_s,trace_last_time_s=float(times[-1]),
        summary_executed_s=summary['executed_s'],status_label=status,full_completion_claim=final_complete and not checkpoint,
        inherited_coverage=False,merged_traces=False,vertices=int(len(vertices)),faces=int(len(faces)),
        covered_vertices=int(covered.sum()),required_goals=int(len(required)),covered_required_goals=int(covered[required].sum()),
        area_fraction=float(audit['area_fraction']),uncovered_vertex_ids=np.flatnonzero(~covered).tolist(),
        uncovered_required_vertex_ids=missing_required.tolist(),
        coverage_definition='Audited 25 mm original-mesh geodesic neighborhoods of actual contacts, not direct touch of every vertex',
        uncovered_face_display='Face red if any original vertex is uncovered; color does not define measured area',
        geometry_modified=False,physics_simulated=False,coverage_recomputed=False,
        points='Measured contact_positions_world_m transformed to original object coordinates; no projection',
        per_finger_contact_runs=[len(groups) for groups in runs],actual_active_samples_per_finger=active.sum(axis=0).tolist(),
        recording_interval_s=dt,display_spacing_m=.0002,display_axes=['original_X_mm','original_Z_mm','original_Y_mm'],
        object_translation_world_m=translation.tolist(),object_rotation=rotation.tolist(),
        line_breaks=['inactive sample','missing temporal sample','adjacent contact jump above 5 mm','finger boundary'],
        sampling_limit='Losses shorter than saved sampling interval cannot be resolved by this figure.',
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),display_data_sha256=sha256(display_file.read_bytes()).hexdigest(),
        png=str(png),pdf=str(pdf),png_sha256=sha256(png.read_bytes()).hexdigest(),pdf_sha256=sha256(pdf.read_bytes()).hexdigest())
    provenance_file.write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding='utf-8')
    return dict(png=str(png),pdf=str(pdf),provenance=str(provenance_file),status=status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--end-s',type=float,help='Explicit verified checkpoint endpoint; always marks CURRENT PREFIX')
    args=parser.parse_args()
    print(json.dumps(render_contact_figure(args.case,args.output,args.end_s),ensure_ascii=False,indent=2))
