"""Saved-state base view; hide obstructing arm/floor only in close-up cameras."""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Module.i04_geometric_traversal.whole_hand_surface_review import load_verified
from Module.i04_geometric_traversal.sliding_audit import _native_context
from Module.i04_geometric_traversal.pad_contact_visual import camera_from_direction, marker


def main(case):
    case, audit, summary, data, metrics = load_verified(case)
    regions = json.loads((case / 'whole_hand_regions_render.json').read_text(encoding='utf-8'))['regions']
    region = next(r for r in regions if r['region'] == '瓶底端 +Y')
    assert region['available']
    center = region['time_s']
    ids = np.flatnonzero((data['time_s'] >= center - 2.) & (data['time_s'] <= center + 2.))
    c = _native_context(case)
    m, d = c.model, c.data
    pads = c.handles.tip_geom_ids
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    renderer = mujoco.Renderer(m, height=540, width=800)
    rgba, matid, groups = m.geom_rgba.copy(), m.geom_matid.copy(), m.geom_group.copy()
    planes = np.flatnonzero(m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in range(m.ngeom)]
    arm_geoms = [g for g, name in enumerate(names) if name.startswith('fr3v2_link') or name == 'fr3_leap_mount_adapter']
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    title = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 24)
    prefix = 'whole_hand_base_clear_1x'
    video = case / (prefix + '.mp4')
    captures = []
    with imageio.get_writer(video, fps=20, codec='libx264', quality=7, macro_block_size=2,
                            output_params=['-threads', '2']) as writer:
        for frame, k in enumerate(ids):
            d.qpos[c.qpos] = data['q_rad'][k]
            mujoco.mj_forward(m, d)
            centers = d.geom_xpos[pads].copy()
            frames = d.geom_xmat[pads].reshape(4, 3, 3)
            normal = frames[:3, :, 2].mean(axis=0)
            normal /= np.linalg.norm(normal)
            canvas = Image.new('RGB', (2400, 700), '#122238')
            for panel in range(3):
                m.geom_rgba[:] = rgba
                m.geom_matid[:] = matid
                m.geom_group[:] = groups
                if panel:
                    m.geom_group[planes] = 5
                    m.geom_group[arm_geoms] = 5
                for geom, name in enumerate(names):
                    if 'normal_marker' in name:
                        m.geom_rgba[geom, 3] = 0.
                    if name == 'mustard_visual':
                        m.geom_matid[geom] = -1
                        m.geom_rgba[geom] = [.96, .72, .08, .12 if panel else 1.]
                m.vis.headlight.ambient[:] = [.4, .4, .4] if panel else [.25, .25, .25]
                m.vis.headlight.diffuse[:] = [.55, .55, .55] if panel else [.45, .45, .45]
                if panel == 2:
                    camera_from_direction(camera, frames[3, :, 2] + .2 * frames[3, :, 1], centers[3], .27)
                elif panel:
                    camera_from_direction(camera, normal + .65 * frames[1, :, 1], centers.mean(axis=0), .44)
                else:
                    camera.lookat[:] = [.3, 0., .49]
                    camera.distance, camera.azimuth, camera.elevation = 1.05, 130., -25.
                renderer.update_scene(d, camera=camera, scene_option=options)
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = not bool(panel)
                for finger in np.flatnonzero(data['safe_belly'][k]):
                    marker(renderer.scene, data['belly_contact_positions_world_m'][k, finger], .002, [1., .1, .55, 1.])
                picture = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, 800, 70), fill='#122238')
                draw.text((12, 7), ('瓶底端 +Y · 实际运动 1倍速', '长指局部 · 隐藏机械臂／地面', '拇指局部 · 隐藏机械臂／地面')[panel], font=title, fill='white')
                draw.text((12, 42), f'实际 t={data["time_s"][k]:.2f}s · 安全指腹 {int(data["safe_belly"][k].sum())}/4', font=font, fill='#a7dbf0')
                canvas.paste(picture, (panel * 800, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 548), f'瓶底接触细节 · 原生保存状态  |  实测测地覆盖 {100*data["coverage"][k]:.2f}%  |  时间节选，非完整回放', font=font, fill='white')
            for i, name in enumerate(('食指', '中指', '无名指', '拇指')):
                q = np.degrees(data['q_rad'][k, 7+4*i:11+4*i])
                label = '指腹贴合' if data['safe_belly'][k, i] else '调整／转移中'
                line = f'{name} 关节° [{q[0]:.1f}, {q[1]:.1f}, {q[2]:.1f}, {q[3]:.1f}]  {data["forces_n"][k,i]:.2f} N  {label}'
                draw.text((15 + 1190*(i % 2), 582+32*(i//2)), line, font=font, fill='#a7edba' if data['safe_belly'][k, i] else '#d7e1ea')
            draw.text((15, 660), '局部相机隐藏机械臂和地面，物体透明；原始全景在左。没有改变原生运动、接触或覆盖。粉点是实际安全指腹接触。', font=font, fill='#f3d38d')
            writer.append_data(np.asarray(canvas))
            if frame in {0, len(ids)//2, len(ids)-1}:
                path = case / f'{prefix}_{frame:04d}.png'
                canvas.save(path)
                captures.append(dict(file=path.name, time_s=float(data['time_s'][k])))
    renderer.close()
    source = case / 'visualization_source' / Path(__file__).name
    source.write_bytes(Path(__file__).read_bytes())
    meta = dict(actual_saved_q_only=True, no_physics_step=True, full_time_axis=False, speed=1., fps=20,
                frames=len(ids), stills=captures, segments=[dict(region=region['region'], start_s=float(data['time_s'][ids[0]]), end_s=float(data['time_s'][ids[-1]]))],
                regions=[region], source_manifest_sha256=audit['source_files']['manifest.json'], whole_object_complete=audit['full_surface_complete'],
                video=video.name, video_sha256=sha256(video.read_bytes()).hexdigest(),
                visualization_source=source.relative_to(case).as_posix(), visualization_source_sha256=sha256(source.read_bytes()).hexdigest(),
                visual_only_changes=dict(closeup_hidden_plane_geoms=[dict(id=int(g), name=names[g]) for g in planes], closeup_hidden_arm_geoms=[dict(id=int(g), name=names[g]) for g in arm_geoms], mustard_alpha=.12, physical_model_changed=False))
    (case / (prefix + '_render.json')).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    page = case / 'review.html'
    html = page.read_text(encoding='utf-8')
    section = '<section><h2>瓶底补充特写 · 1倍速</h2><p class="note">局部相机隐藏机械臂和地面、将物体透明显示，以便看清三根长指。左侧保留原始全景；关节运动和接触来自同一条原生记录，这是约4秒节选。</p><video controls preload="metadata" src="whole_hand_base_clear_1x.mp4"></video></section>'
    if 'whole_hand_base_clear_1x.mp4' in html:
        import re
        html, changed = re.subn(r'<section><h2>瓶底补充特写.*?</section>', section, html, count=1)
        assert changed == 1
    else:
        html = html.replace('<section><h2>整个物体是否真正覆盖</h2>', section + '<section><h2>整个物体是否真正覆盖</h2>')
    html = html.replace('各指存在短暂失联，', '各指存在失联和恢复，')
    if '覆盖采用既定25mm' not in html:
        html = html.replace('<p>腕部与四指共同求解，', '<p class="note">覆盖采用既定25mm网格边测地接触邻域，并非皮肤压力足迹面积。全程仍有离面调姿，最长全手有效接触中断146.156秒，尚未达到无间断贴面遍历。</p><p>腕部与四指共同求解，')
    page.write_text(html, encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('case')
    main(parser.parse_args().case)
