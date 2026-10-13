"""Review whole-object physical coverage on every side of mustard."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import trimesh
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .fixtures import MUSTARD
from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml


def task_mesh():
    mesh = trimesh.load(MUSTARD/'visual_scaled.obj', force='mesh', process=True)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    mesh.apply_transform(transform)
    bounds = mesh.bounds
    mesh.apply_translation(np.r_[-.5*(bounds[0, :2]+bounds[1, :2]), -bounds[0, 2]])
    return max(mesh.split(only_watertight=False), key=lambda m: abs(m.volume))


def detail_segments(trace, object_position):
    """Six stable real contact windows, classified from recorded sensors."""
    normal = trace['normals_world']
    points = trace['contact_positions_world_m']-np.array(object_position)
    safe = trace['active'] & (trace['forces_n'] <= 8.)
    any_point = np.ones(safe.shape, bool)
    regions = [('RIGHT SIDE (+X)', 0, 1, any_point), ('LEFT SIDE (-X)', 0, -1, any_point),
        ('BOTTLE BASE (+Y)', 1, 1, points[:, :, 1] > .22),
        ('NECK / CAP (-Y)', 1, -1, points[:, :, 1] < -.22),
        ('UPPER BROAD FACE (+Z)', 2, 1, (points[:, :, 1] > -.10) & (points[:, :, 1] < .20)),
        ('LOWER BROAD FACE (-Z)', 2, -1, (points[:, :, 1] > -.10) & (points[:, :, 1] < .20))]
    result = []
    for label, axis, sign, restriction in regions:
        mask = (safe & (normal[:, :, axis]*sign > .6) & restriction).any(axis=1)
        changes = np.diff(np.r_[False, mask, False].astype(int))
        starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        durations = trace['time_s'][ends-1]-trace['time_s'][starts]
        candidates = np.flatnonzero(durations >= .15-1e-9)
        if not len(candidates): raise ValueError(f'No stable recorded touch in {label}')
        selected = int(candidates[np.argmax(durations[candidates])])
        onset = float(trace['time_s'][starts[selected]])
        center = onset+min(.3, durations[selected]/2)
        result.append(dict(label=label, real_contact_onset_s=onset,
            continuous_region_contact_s=float(durations[selected]),
            start_s=max(0., center-.8), end_s=min(float(trace['time_s'][-1]), center+1.2)))
    return result


def replay(directory, *, speed=40., fps=15, contact_details=False):
    """Two views of the same recorded physical state; playback adds no credit."""
    directory = Path(directory)
    with np.load(directory/'trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    model = compile_model_xml((directory/'model.xml').read_text(encoding='utf-8'))
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mustard_visual')
    model.geom_matid[geom] = -1
    model.geom_rgba[geom] = [1., .72, .08, 1.]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
    addresses = model.jnt_qposadr[joints]
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    object_center = model.body_pos[object_id]+np.array([0., 0., .095])
    camera.distance = 1.15
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    path = directory/('contact_details.mp4' if contact_details else 'full_traversal.mp4')
    colors = [(0.2, .8, 1., 1.), (.9, .3, .9, 1.), (.3, 1., .3, 1.), (1., .3, .3, 1.)]
    segments = []
    if contact_details:
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
        segments = detail_segments(trace, manifest['object_position_m'])
        times = [np.arange(s['start_s'], s['end_s']+1e-9, speed/fps) for s in segments]
        labels = [s['label'] for s, values in zip(segments, times) for _ in values]
        frames = np.concatenate(times)
    else:
        frames = np.arange(0., float(trace['time_s'][-1])+1e-9, speed/fps)
        frames = np.r_[frames, float(trace['time_s'][-1])]
        labels = ['WHOLE CONTINUOUS ROUTE']*len(frames)
    with imageio.get_writer(path, fps=fps, codec='libx264', quality=7, macro_block_size=16) as writer:
        for frame, time in enumerate(frames):
            row = min(int(np.searchsorted(trace['time_s'], time)), len(trace['time_s'])-1)
            data.qpos[addresses] = trace['q_rad'][row]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = .8*object_center+.2*trace['palm_pose_world'][row, :3]
            panels = []
            for azimuth, elevation, label in ((130., -30., 'UPPER / SIDE VIEW'), (-50., 24., 'LOWER / REVERSE VIEW')):
                camera.azimuth, camera.elevation = azimuth, elevation
                renderer.update_scene(data, camera=camera, scene_option=option)
                for finger in np.flatnonzero(trace['active'][row]):
                    slot = renderer.scene.ngeom
                    if slot >= renderer.scene.maxgeom: break
                    mujoco.mjv_initGeom(renderer.scene.geoms[slot], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.006, 0., 0.]), trace['contact_positions_world_m'][row, finger],
                        np.eye(3).ravel(), np.array(colors[finger]))
                    renderer.scene.ngeom += 1
                panel = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 640, 90), fill=(15, 23, 42))
                display_label = labels[frame] if contact_details else label
                draw.text((12, 7), f'{display_label} | saved t={trace["time_s"][row]:.2f}s', font=font, fill='white')
                draw.text((12, 32), f'measured coverage={100*trace["coverage"][row]:.3f}%  '
                    f'goals={trace["covered_goals"][row]}/388  contacts={trace["active"][row].sum()}', font=font, fill='#9ee9bc')
                draw.text((12, 57), f'{trace["phase"][row]} | peak pad force={trace["forces_n"][row].max():.2f}N '
                    f'| playback {speed:g}x', font=font, fill='#bce7ff')
                panels.append(np.asarray(panel))
            combined = np.concatenate(panels, axis=1)
            writer.append_data(combined)
            if frame in (0, len(frames)//3, 2*len(frames)//3, len(frames)-1):
                prefix = 'detail' if contact_details else 'replay'
                Image.fromarray(combined).save(directory/f'{prefix}_frame_{frame:04d}.png')
            if frame % 300 == 0: print(f'RENDER {frame}/{len(frames)}', flush=True)
        # Keep the last measured state readable for two seconds.
        for _ in range(2*fps): writer.append_data(combined)
    from OpenGL import GL
    backend = {key: GL.glGetString(value).decode() for key, value in (
        ('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION))}
    backend.update(display_style='Saved joint states; solid yellow mustard; dots are measured contacts',
                   playback_speed=speed, fps=fps, physical_duration_s=float(trace['time_s'][-1]),
                   contact_detail_segments=segments,
                   video_source='trace.npz; no trajectory resimulation or inferred contact credit')
    name = 'render_backend_details.json' if contact_details else 'render_backend.json'
    (directory/name).write_text(json.dumps(backend, indent=2), encoding='utf-8')
    renderer.close()
    print('VIDEO', path, flush=True)


def review(directory):
    directory = Path(directory)
    summary = json.loads((directory/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    with np.load(directory/'trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    mesh = task_mesh()
    if len(mesh.vertices) != len(trace['covered_vertices']):
        raise RuntimeError('Coverage mask and complete task surface differ')
    covered = trace['covered_vertices']
    colors = LinearSegmentedColormap.from_list('touch', ['#dc4038', '#f9cf61', '#169d67'])
    face_covered = covered[mesh.faces].mean(axis=1)
    fig = plt.figure(figsize=(15, 8), layout='constrained')
    views = [(15, 0, 'Right side (+X)'), (15, 180, 'Left side (-X)'),
             (15, 90, 'Bottle base (+Y)'), (15, -90, 'Neck and cap (-Y)'),
             (90, -90, 'Upper broad face (+Z)'), (-90, -90, 'Lower broad face (-Z)')]
    for i, (elevation, azimuth, title) in enumerate(views):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        direction = np.array([np.cos(np.deg2rad(azimuth))*np.cos(np.deg2rad(elevation)),
            np.sin(np.deg2rad(azimuth))*np.cos(np.deg2rad(elevation)), np.sin(np.deg2rad(elevation))])
        shade = .45+.55*np.clip(mesh.face_normals@direction, 0., 1.)
        face_colors = colors(face_covered); face_colors[:, :3] *= shade[:, None]
        collection = Poly3DCollection(mesh.triangles, facecolors=face_colors,
                                      edgecolors='none', linewidth=0.)
        ax.add_collection3d(collection)
        low, high = mesh.bounds
        ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_zlim(low[2], high[2])
        ax.set_box_aspect(np.array(mesh.extents, copy=True))
        ax.view_init(elev=elevation, azim=azimuth)
        ax.set(title=title); ax.set_axis_off()
    fig.suptitle(f"Mustard measured touch coverage: {100*summary['final_covered_area_fraction']:.3f}% | "
                 f"{summary['visited_required_goals']}/{summary['required_goal_count']} goals", fontsize=16)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap=colors), ax=fig.axes,
                      shrink=.6, pad=.02)
    bar.set_label('Measured covered vertex fraction per triangle (red = untouched)')
    fig.savefig(directory/'surface_coverage.png', dpi=150)
    plt.close(fig)
    first_touch_picture = ''
    if (directory/'coverage_independent.npz').exists() and not manifest.get('resumed_development_run'):
        with np.load(directory/'coverage_independent.npz') as ledger:
            first_touch = ledger['first_touch_time_s']
        if np.all(np.isfinite(first_touch)):
            face_times = first_touch[mesh.faces].mean(axis=1)
            norm = Normalize(0., float(trace['time_s'][-1]))
            fig = plt.figure(figsize=(15, 8), layout='constrained')
            for i, (elevation, azimuth, title) in enumerate(views):
                ax = fig.add_subplot(2, 3, i+1, projection='3d')
                direction = np.array([np.cos(np.deg2rad(azimuth))*np.cos(np.deg2rad(elevation)),
                    np.sin(np.deg2rad(azimuth))*np.cos(np.deg2rad(elevation)), np.sin(np.deg2rad(elevation))])
                shade = .45+.55*np.clip(mesh.face_normals@direction, 0., 1.)
                face_colors = plt.cm.viridis(norm(face_times)); face_colors[:, :3] *= shade[:, None]
                ax.add_collection3d(Poly3DCollection(mesh.triangles, facecolors=face_colors, edgecolors='none', linewidth=0.))
                low, high = mesh.bounds
                ax.set_xlim(low[0], high[0]); ax.set_ylim(low[1], high[1]); ax.set_zlim(low[2], high[2])
                ax.set_box_aspect(np.array(mesh.extents, copy=True)); ax.view_init(elev=elevation, azim=azimuth)
                ax.set(title=title); ax.set_axis_off()
            fig.suptitle('When each part first received measured touch coverage', fontsize=16)
            bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=fig.axes, shrink=.6, pad=.02)
            bar.set_label('First actual touch-neighborhood coverage time (s)')
            fig.savefig(directory/'first_touch_time.png', dpi=150); plt.close(fig)
            first_touch_picture = '<div class="card"><h2>全瓶各处第一次实测覆盖的时间</h2><img src="first_touch_time.png" alt="first measured touch time"><p>时间来自独立重建的实测接触账本，展示各面被逐步触摸的顺序。</p></div>'
    vertex_area = np.zeros(len(mesh.vertices))
    np.add.at(vertex_area, mesh.faces.ravel(), np.repeat(mesh.area_faces/3, 3))
    normals = mesh.vertex_normals
    sectors = np.argmax(abs(normals), axis=1)*2+(normals[np.arange(len(normals)), np.argmax(abs(normals), axis=1)] < 0)
    sector_metrics = {}
    for i, name in enumerate(('+X', '-X', '+Y', '-Y', '+Z', '-Z')):
        mask = sectors == i
        sector_metrics[name] = {'area_m2': float(vertex_area[mask].sum()),
            'covered_fraction': float(vertex_area[mask]@covered[mask]/vertex_area[mask].sum())}
    (directory/'surface_sectors.json').write_text(json.dumps(sector_metrics, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), constrained_layout=True)
    axes[0].plot(trace['time_s'], 100*trace['coverage'], color='#169d67')
    axes[0].set(ylabel='Actual measured surface coverage (%)', ylim=(0, 100))
    axes[1].plot(trace['time_s'], trace['active'].sum(axis=1), lw=.7, color='#2563eb')
    axes[1].set(xlabel='Physics execution time (s)', ylabel='Real pad contacts', ylim=(-.1, 4.1))
    for ax in axes: ax.grid(alpha=.15)
    fig.savefig(directory/'coverage_time.png', dpi=150); plt.close(fig)
    complete = summary['full_surface_complete']
    resume = manifest.get('resumed_development_run')
    scope = ('development continuation/restart; prior and current contact ledgers combined' if resume
             else 'one continuous physical run from common acquisition')
    video = ('<div class="card"><h2>整段实际状态回放 · 正反两视角</h2>'
        '<video controls preload="metadata" src="full_traversal.mp4"></video>'
        '<p>同一时间的记录状态，两侧相机展示上下与背面。画面中的彩点来自实测接触；'
        '黄色是便于识别的显示颜色。回放本身不增加覆盖。</p></div>') if (directory/'full_traversal.mp4').exists() else ''
    if (directory/'contact_details.mp4').exists():
        video = ('<div class="card"><h2>各面实际接触特写 · 4 倍慢放</h2><video controls preload="metadata" src="contact_details.mp4"></video>'
            '<p>从同次实测触点、力和法向中选取六个稳定接触片段：左右侧、底面、盖/颈、上下大面。'
            '片段以实际仿真时刻标注；下面的完整连续回放保留全部换面运动。</p></div>')+video
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>mustard 全表面遍历</title>
    <style>body{{max-width:1280px;margin:30px auto;padding:0 24px;background:#f6f8fb;font-family:system-ui,"Microsoft YaHei",sans-serif;color:#182235}}.card{{background:white;padding:22px;margin:20px 0;border:1px solid #dde3ec;border-radius:12px}}p{{line-height:1.7}}img,video{{width:100%}}a{{color:#2563eb}}</style>
    <h1>mustard 全表面实际触摸 · {summary['cell']}</h1><p>状态：{'全表面实测覆盖完成' if complete else '完整覆盖尚未完成'}。
    面积覆盖 {100*summary['final_covered_area_fraction']:.4f}%，必访点 {summary['visited_required_goals']}/{summary['required_goal_count']}，
    vertices {summary['covered_vertex_count']}/{summary['vertex_count']}。停止原因：{summary['stop_reason']}。</p>
    <div class="card"><p>运行范围：{scope}。DP 未运行。物体固定，原始 2.8 倍模型尺寸保留；
    放在 FR3 全方向可达的位置。覆盖是 25 mm mesh-geodesic 实测接触并集，红色区域仍未触摸覆盖。</p>
    <p>清理的两个零体积退化三角形分量只占 raw area 的 0.00000632%；完整闭合瓶体没有因不可达而删减。
    只有整个瓶体 vertices 和 required goals 全部实际覆盖才判完成。</p></div>
    {video}
    <div class="card"><img src="surface_coverage.png" alt="actual measured coverage from six sides"></div>
    {first_touch_picture}
    <div class="card"><img src="coverage_time.png" alt="measured physical coverage over time"></div>
    <div class="card"><p><a href="summary.json">运行指标</a> · <a href="surface_sectors.json">各面覆盖</a> ·
    <a href="manifest.json">配置与来源</a> · <a href="audit.json">独立接触账本核验</a> ·
    <a href="events.json">路径、局部恢复与未完成目标</a></p></div></html>'''
    (directory/'index.html').write_text(page, encoding='utf-8')
    print('Review', directory/'index.html')
    print(json.dumps(sector_metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--speed', type=float, default=40.)
    args = parser.parse_args()
    if args.video: replay(args.directory, speed=args.speed)
    if args.details: replay(args.directory, speed=.25, contact_details=True)
    review(args.directory)


if __name__ == '__main__': main()
