"""Uncut saved-state replay of surface strokes, with measured-contact trails."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Module.fr3_leap import ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.fr3_leap.model import compile_model_xml


def render(directory, *, speed=2., start=0., end=None, closeup=False):
    directory = Path(directory)
    with np.load(directory/'trace.npz') as saved:
        trace = {key: saved[key] for key in saved.files}
    model = compile_model_xml((directory/'model.xml').read_text(encoding='utf-8'))
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'mustard_visual')
    model.geom_matid[geom] = -1; model.geom_rgba[geom] = [1., .72, .08, 1.]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
              for name in (*ARM_JOINT_NAMES, *HAND_JOINT_NAMES)]
    addresses = model.jnt_qposadr[joints]
    object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'fr3_e05_object')
    center = model.body_pos[object_id]+np.array([0., 0., .095])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=540, width=720)
    camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    options = mujoco.MjvOption(); options.geomgroup[3:] = 0
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    colors = [(0.15, .65, 1., 1.), (.95, .25, .85, 1.), (.12, .85, .35, 1.), (1., .25, .25, 1.)]
    stop = min(float(trace['time_s'][-1]), end if end is not None else float('inf'))
    window = (trace['time_s'] >= start) & (trace['time_s'] <= stop)
    focus = trace['tip_positions_world_m'][window].mean(axis=(0, 1))
    frames = np.r_[np.arange(start, stop, speed/20.), stop]
    path = directory/('sliding_closeup.mp4' if closeup else 'full_traversal.mp4')
    with imageio.get_writer(path, fps=20, codec='libx264', quality=7, macro_block_size=2,
                           output_params=['-threads', '2']) as writer:
        for frame, t in enumerate(frames):
            row = min(int(np.searchsorted(trace['time_s'], t)), len(trace['time_s'])-1)
            data.qpos[addresses] = trace['q_rad'][row]; mujoco.mj_forward(model, data)
            panels = []
            for azimuth, elevation, label in ((130., -30., 'CONTACT / SIDE'), (-50., 24., 'REVERSE')):
                camera.azimuth, camera.elevation = azimuth, elevation
                camera.distance = .72 if closeup else .95
                camera.lookat[:] = .35*center+.65*focus if closeup else center
                renderer.update_scene(data, camera=camera, scene_option=options)
                # The trails contain only recorded actual contact positions,
                # never planned points or inferred coverage locations.
                first = int(np.searchsorted(trace['time_s'], max(start, t-4.)))
                for previous in range(first, row+1, 15):
                    for finger in np.flatnonzero(trace['active'][previous]):
                        slot = renderer.scene.ngeom
                        if slot >= renderer.scene.maxgeom:
                            break
                        mujoco.mjv_initGeom(renderer.scene.geoms[slot], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([.0013, 0., 0.]), trace['contact_positions_world_m'][previous, finger],
                            np.eye(3).ravel(), np.array(colors[finger]))
                        renderer.scene.ngeom += 1
                for finger in np.flatnonzero(trace['active'][row]):
                    slot = renderer.scene.ngeom
                    if slot >= renderer.scene.maxgeom:
                        break
                    mujoco.mjv_initGeom(renderer.scene.geoms[slot], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([.004, 0., 0.]), trace['contact_positions_world_m'][row, finger],
                        np.eye(3).ravel(), np.array(colors[finger]))
                    renderer.scene.ngeom += 1
                panel = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(panel)
                draw.rectangle((0, 0, 720, 88), fill=(15, 23, 42))
                draw.text((12, 7), f'{label} | saved time {t:.2f}s | uncut {speed:g}x', font=font, fill='white')
                draw.text((12, 33), f'{trace["phase"][row]} | real contacts {trace["active"][row].sum()} | '
                    f'pad force {trace["forces_n"][row].max():.2f} N', font=font, fill='#9ee9bc')
                draw.text((12, 59), 'Colored dots / trails = actual recorded contacts (last 4 s)', font=font, fill='#bce7ff')
                panels.append(np.asarray(panel))
            combined = np.concatenate(panels, axis=1); writer.append_data(combined)
            if frame in (0, len(frames)//3, 2*len(frames)//3, len(frames)-1):
                Image.fromarray(combined).save(directory/f'{path.stem}_frame_{frame:04d}.png')
            if frame % 400 == 0:
                print(f'RENDER {path.name} {frame}/{len(frames)}', flush=True)
    from OpenGL import GL
    meta = dict(source='trace.npz; FK playback only; no resimulation or added contact',
        start_s=start, end_s=stop, playback_speed=speed, fps=20, uncut=True,
        trail='actual same-run contacts in the preceding 4 seconds',
        renderer=GL.glGetString(GL.GL_RENDERER).decode())
    (directory/(path.stem+'_render.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    renderer.close()
    print('VIDEO', path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory', type=Path)
    parser.add_argument('--closeup', action='store_true')
    parser.add_argument('--speed', type=float, default=2.)
    parser.add_argument('--start', type=float, default=0.)
    parser.add_argument('--end', type=float)
    args = parser.parse_args()
    render(args.directory, speed=args.speed, start=args.start, end=args.end, closeup=args.closeup)
