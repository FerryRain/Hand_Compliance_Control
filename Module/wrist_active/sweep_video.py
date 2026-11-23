"""Render the paired force-priority experiment directly from its scored sensor traces."""
import argparse
from bisect import bisect_right
from contextlib import ExitStack
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import trimesh

from ..i04_geometric_traversal.pad_contact_visual import camera_from_direction, marker
from .evaluate import movement
from .experiment import ACTIVE2, digest, scenario
from .history import read_model
from .sweep_experiment import OUT


FPS = 24
SIZE = (1280, 720)
VIEW = (600, 278)
THUMB = (276, 148)
BG, PANEL, INK, MUTED, HAIR = '#101922', '#17242e', '#f2f5f6', '#b0c1c8', '#354550'
ACCENTS = ('#62d7c2', '#f4b374')
FINGERS = ('食', '中', '环', '拇')
TIP_NAMES = ('fingertip_geom', 'fingertip_2_geom', 'fingertip_3_geom', 'thumb_fingertip_geom')
STAGES = ('扫查', '退腕', '转移', '重新接近')
EYE = np.array([.40, -1., .52])


@lru_cache(None)
def font(size, bold=False, mono=False):
    name = 'consolab.ttf' if mono and bold else 'consola.ttf' if mono else 'msyhbd.ttc' if bold else 'msyh.ttc'
    return ImageFont.truetype(str(Path('C:/Windows/Fonts') / name), size)


def label(draw, xy, value, size=18, color=INK, *, bold=False, mono=False, anchor='lm'):
    draw.text(xy, value, font=font(size, bold, mono), fill=color, anchor=anchor)


def read(path):
    return json.loads(path.read_bytes())


class Projection:
    """A shared orthographic camera and physical scale for the two tactile views."""

    def __init__(self, reference):
        eye = EYE / np.linalg.norm(EYE)
        right = np.cross([0., 0., 1.], eye)
        right /= np.linalg.norm(right)
        self.basis = np.stack([right, -np.cross(eye, right), eye], axis=1)
        points = reference @ self.basis
        self.center = (points.min(0) + points.max(0)) * .5
        self.extent = np.maximum(np.ptp(points, axis=0), .001)
        self.scale = min((THUMB[0]-20) / self.extent[0], (THUMB[1]-13) / self.extent[1])

    def coordinates(self, points):
        view = points @ self.basis - self.center
        return view[:, :2] * self.scale + np.asarray(THUMB)*.5, view[:, 2]

    def mesh(self, path, tint):
        mesh = trimesh.load_mesh(path, process=False)
        position, depth = self.coordinates(mesh.vertices)
        triangles = mesh.faces
        order = np.argsort(depth[triangles].mean(1), kind='stable')
        normals = mesh.face_normals @ self.basis
        normals = np.where(normals[:, 2, None] < 0, -normals, normals)
        light = np.array([-.35, -.58, .74])
        shade = .34 + .64*np.maximum(normals @ light, 0.)
        colors = np.clip(shade[:, None] * np.array(ImageColor.getrgb(tint)) + 5, 0, 255).astype('uint8')
        aa = 2
        pts = np.rint(position * aa).astype(int)[triangles].reshape(-1, 6).tolist()
        colors = [tuple(c) for c in colors.tolist()]
        image = Image.new('RGB', (THUMB[0]*aa, THUMB[1]*aa), PANEL)
        pen = ImageDraw.Draw(image)
        for i in order:
            pen.polygon(pts[i], fill=colors[i])
        return image.resize(THUMB, Image.Resampling.LANCZOS)


class TouchCloud:
    def __init__(self, ticks, points, camera, tint):
        self.ticks = ticks
        pix, depth = camera.coordinates(points)
        self.pixels = np.round(pix).astype(int)
        self.shades = np.uint8(52 + 203*np.clip(depth/camera.extent[2]+.5, 0, 1))
        self.depth = np.zeros(THUMB[::-1], dtype=np.uint8)
        self.last = 0
        self.palette = np.uint8(np.array(ImageColor.getrgb(tint))[None, :]
                                * (.48+.52*np.arange(256)[:, None]/255))
        self.palette[0] = ImageColor.getrgb(PANEL)

    def render(self, tick):
        count = np.searchsorted(self.ticks, tick, side='right')
        if count < self.last:
            self.depth.fill(0)
            self.last = 0
        x, y = self.pixels[self.last:count].T
        value = self.shades[self.last:count]
        for dx, dy in ((0,0),(1,0),(0,1)):
            ix, iy = x+dx, y+dy
            inside = (ix>=0) & (ix<THUMB[0]) & (iy>=0) & (iy<THUMB[1])
            np.maximum.at(self.depth, (iy[inside], ix[inside]), value[inside])
        self.last = count
        return Image.fromarray(self.palette[self.depth]), count


class Arm:
    def __init__(self, raw, scored, camera, color):
        self.raw = raw
        self.run = read(raw/'run.json')
        self.scores = read(scored/'scores.json')
        if digest(raw/'trace.npz') != self.scores['spec']['trace_sha256']:
            raise ValueError(f'Trace and scores differ in {scored}')
        with np.load(raw/'trace.npz', allow_pickle=False) as trace:
            self.q = trace['qpos']
            self.points, self.pads = trace['points_m'], trace['pad_valid']
            self.peak = trace['region_load_n'].max(1)
            self.phase = trace['phase'] if 'phase' in trace else None
        with np.load(scored/'coverage.npz') as curve:
            self.coverage = curve['c10']
        self.valid = self.pads & (self.peak < 8.)[:, None]
        self.three = np.r_[0, np.cumsum(self.valid[1:].sum(1)>=3)]*.01
        radial = self.q[:,:3] / np.linalg.norm(self.q[:,:3],axis=1,keepdims=True)
        side = np.cross([0.,0.,1.],radial[0])
        if np.linalg.norm(side)<.05:
            side = np.cross([1.,0.,0.],radial[0])
        rotations = Rotation.from_quat(self.q[:,[4,5,6,3]])
        tangent = rotations.apply(rotations[0].inv().apply(side/np.linalg.norm(side)))
        tangent -= (tangent*radial).sum(1,keepdims=True)*radial
        self.camera_directions = .68*radial+tangent/np.maximum(np.linalg.norm(tangent,axis=1,keepdims=True),.01)
        k, f = np.nonzero(self.valid)
        pts = self.points[k, f]
        _, first = np.unique(np.floor(pts/.0045).astype(int), axis=0, return_index=True)
        first.sort()
        self.cloud = TouchCloud(k[first], pts[first], camera, color)
        self.snapshots, self.snap_ticks = [], []
        for row in self.scores['snapshots']:
            t = round(row['s']*100)
            np.testing.assert_allclose([self.coverage[t], self.three[t]], [row['coverage'], row['low3_s']])
            if np.searchsorted(k[first], t, side='right') != row['cells']:
                raise ValueError('Displayed touch cells and evaluated cells differ')
            self.snap_ticks.append(t)
            self.snapshots.append((camera.mesh(scored/f"signed-{row['s']:g}s.ply", color), row))


class Replay:
    def __init__(self, stack, root, obj, start, *, first_run=None, second_run=None,
                 first_scores=None, second_scores=None, fixture_paths=None):
        case, truth = fixture_paths or scenario(obj,start)
        vertices = trimesh.load_mesh(truth,process=False).vertices
        camera = Projection(vertices)
        self.object_points = vertices[::max(1, len(vertices)//600)]
        ours = first_run or root/'force-priority/runs'/obj/start/'sweep-dp'
        other = second_run or (ACTIVE2/'aude-mcc' if obj=='pear' else root/'strong-baseline'/obj/start/'aude-mcc')
        other_scores = root/'references/pear/active-02/aude-mcc' if obj=='pear' and second_run is None else other
        self.arms = [Arm(ours,first_scores or ours,camera,ACCENTS[0]),
                     Arm(other,second_scores or other_scores,camera,ACCENTS[1])]
        first, second = self.arms
        np.testing.assert_array_equal(first.q[0], second.q[0])
        if first.run['model_sha256'] != digest(case/'model.mjb') or len(first.q)!=len(second.q):
            raise ValueError('Unmatched replay model or duration')
        self.model = model = read_model(case/'model.mjb')
        self.obj = obj
        self.data = mujoco.MjData(model)
        self.pads = []
        for name in TIP_NAMES:
            original = model.geom(name)
            connector = model.geom(name+'_connector_collision')
            connector.rgba[:] = original.rgba
            original.rgba[3] = 0.
            self.pads.append(model.geom(name+'_pad_collision').id)
        model.geom('bottle_visual').rgba[:] = (.88,.60,.23,.48)
        object_body=model.geom_bodyid[model.geom('bottle_visual').id]
        hand_ids=[i for i in range(model.ngeom) if model.geom_bodyid[i]!=object_body
                  and model.geom_group[i]==0 and model.geom_rgba[i,3]>0]
        self.hand_ids=np.asarray(hand_ids)
        box=model.geom_aabb[self.hand_ids]
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        self.corners=box[:,None,:3]+box[:,None,3:]*signs
        model.vis.headlight.ambient[:] = .50
        model.vis.headlight.diffuse[:] = .82
        model.vis.global_.offwidth, model.vis.global_.offheight = VIEW
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3:] = 0
        self.options.sitegroup[:] = 0
        self.cam = mujoco.MjvCamera()
        self.renderer = stack.enter_context(mujoco.Renderer(model, height=VIEW[1], width=VIEW[0]))

    def physical(self, arm, tick, focus_tick):
        self.data.qpos[:] = arm.q[tick]
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_camlight(self.model,self.data)
        for f, pad in enumerate(self.pads):
            self.model.geom_rgba[pad] = ((.20,1.,.62,1.) if arm.valid[tick,f] else
                                         (1.,.46,.24,1.) if arm.pads[tick,f] else (.34,.56,.64,1.))
        eye=arm.camera_directions[tick if focus_tick is None else focus_tick]
        eye=eye/np.linalg.norm(eye)
        right=np.cross([0.,0.,1.],eye)
        right/=max(np.linalg.norm(right),1e-6)
        up=np.cross(eye,right)
        basis=np.stack((right,up,eye),axis=1)
        hand=self.data.geom_xpos[self.hand_ids,None,:]+np.einsum('gij,gkj->gki',
                         self.data.geom_xmat[self.hand_ids].reshape(-1,3,3),self.corners)
        frame=np.vstack((self.object_points,hand.reshape(-1,3)))@basis
        center=.5*(frame.max(0)+frame.min(0))
        x,y,z=(frame-center).T
        tangent=np.tan(np.deg2rad(self.model.vis.global_.fovy*.5))
        distance=np.max(z+1.16*np.maximum(abs(y),abs(x)*VIEW[1]/VIEW[0])/tangent)
        camera_from_direction(self.cam,eye,basis@center,max(distance,.42))
        self.renderer.update_scene(self.data,camera=self.cam,scene_option=self.options)
        for f in np.flatnonzero(arm.pads[tick]):
            color=(.20,1.,.62,1.) if arm.valid[tick,f] else (1.,.46,.24,1.)
            marker(self.renderer.scene,arm.points[tick,f],.0046,color)
        pixels = self.renderer.render()
        return Image.fromarray(pixels)


def longest_three(arm):
    moving = movement(arm.q)
    moving[0] = False
    bouts = []
    for trio in combinations(range(4),3):
        active = moving & arm.valid[:,trio].all(1)
        edges = np.flatnonzero(np.diff(np.r_[False,active,False])).reshape(-1,2)
        bouts.extend((stop-start, start, stop, trio) for start,stop in edges if stop-start>=100)
    return max(bouts)


def draw_frame(pair, tick, title, subhead, rate, note, focus_tick=None):
    canvas = Image.new('RGB', SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    label(draw,(30,29),title,30,bold=True)
    label(draw,(31,62),subhead,16,MUTED)
    label(draw,(1250,29),f'{tick/100:,.2f} s',32,bold=True,mono=True,anchor='rm')
    label(draw,(1250,62),f'仿真时间   /   {rate}',16,MUTED,anchor='rm')
    draw.line((28,81,1252,81),fill=HAIR)
    for j,(arm,x,color) in enumerate(zip(pair.arms,(28,652),ACCENTS)):
        draw.rectangle((x,95,x+4,117),fill=color)
        label(draw,(x+14,106),('GPIS 手腕规划','Aude 腕指联合规划')[j],22,bold=True)
        label(draw,(x+599,107),('single_all + MCC','MCC')[j],17,color,anchor='rm')
        canvas.paste(pair.physical(arm,tick,focus_tick),(x,127))
        for f,name in enumerate(FINGERS):
            left=x+10+35*f
            fill=('#26705a' if arm.valid[tick,f] else '#9d5038' if arm.pads[tick,f] else '#26313a')
            draw.rounded_rectangle((left,137,left+29,164),radius=5,fill=fill)
            label(draw,(left+14,150),name,16,'#ffffff' if arm.pads[tick,f] else '#b6c4cb',anchor='mm')
        label(draw,(x+589,148),f'{arm.peak[tick]:.1f} N',17,'#f4b799' if arm.peak[tick]>=8 else '#dbe9e9',mono=True,anchor='rm')
        label(draw,(x+2,429),'实触覆盖',17,MUTED)
        coverage = format(arm.coverage[tick],'.2%' if tick==len(arm.q)-1 else '.1%')
        label(draw,(x+84,428),coverage,29,color,mono=True,bold=True)
        label(draw,(x+281,429),'轻载三指累计',17,MUTED)
        label(draw,(x+596,428),f'{arm.three[tick]:.1f} s',26,mono=True,bold=True,anchor='rm')
        draw.rectangle((x,454,x+599,457),fill=HAIR)
        covered=round(599*arm.coverage[tick])
        if covered: draw.rectangle((x,454,x+covered,457),fill=color)
        draw.rounded_rectangle((x,470,x+599,661),radius=8,fill=PANEL)
        draw.line((x+299,484,x+299,649),fill=HAIR)
        cloud,count = arm.cloud.render(tick)
        label(draw,(x+13,490),'累计实触点',16,bold=True)
        label(draw,(x+282,490),f'{count:,} 格',15,MUTED,anchor='rm')
        canvas.paste(cloud,(x+10,505))
        label(draw,(x+312,490),'有符号 GPIS',16,bold=True)
        idx = bisect_right(arm.snap_ticks,tick)-1
        if idx>=0:
            image,score=arm.snapshots[idx]
            label(draw,(x+587,490),f"{score['s']:g} s · F@5 {score['signed']['fscore']:.1%}",15,color,anchor='rm')
            canvas.paste(image,(x+310,505))
        else:
            label(draw,(x+450,578),'首个网格在 60 s 评估',16,MUTED,anchor='mm')
    label(draw,(30,690),note,18,INK)
    label(draw,(1252,690),'轻载 < 8 N    /    覆盖 10 mm    /    F@5  5 mm',15,MUTED,anchor='rm')
    return canvas


def span(first,last,speed):
    frames=round((last-first)*FPS/(100*speed))
    return np.rint(np.linspace(first,last,frames,endpoint=False)).astype(int)


def conclusion(summary):
    canvas=Image.new('RGB',SIZE,BG)
    d=ImageDraw.Draw(canvas)
    label(d,(46,56),'实验结果',37,bold=True)
    label(d,(47,103),'固定物体的 MuJoCo 闭环仿真',20,MUTED)
    d.line((46,140,1234,140),fill=HAIR)
    label(d,(50,178),'同等物理时间或同一覆盖阈值',21,MUTED)
    label(d,(858,178),'GPIS 手腕规划',22,ACCENTS[0],bold=True,anchor='rm')
    label(d,(1218,178),'Aude + MCC',22,ACCENTS[1],bold=True,anchor='rm')
    smooth=[r for r in summary['transfers'] if r['obj']!='meat-can']
    fixed=np.mean([r['moving_same_three_s'] for r in smooth],axis=0)
    cov=summary['transfer_means']['coverage']
    measured=summary['pear']['s240']
    base=summary['baseline_240']
    rows=[('梨达到 70% 实触覆盖',f"{summary['pear']['thresholds']['70']:.2f} s",f"{summary['baseline_t70']:.2f} s"),
          ('梨在 240 s 的有符号重建 F@5',f"{measured['signed']['fscore']:.1%}",f"{base['signed']['fscore']:.1%}"),
          ('五组平滑曲面，移动中持续同三指¹',f'{fixed[0]:.2f} s',f'{fixed[1]:.2f} s'),
          ('七组迁移试验，240 s 平均实触覆盖',f'{cov[0]:.1%}',f'{cov[1]:.1%}')]
    for i,(name,ours,aude) in enumerate(rows):
        y=238+77*i
        label(d,(50,y),name,23)
        label(d,(858,y),ours,31,ACCENTS[0],mono=True,bold=True,anchor='rm')
        label(d,(1218,y),aude,31,ACCENTS[1],mono=True,bold=True,anchor='rm')
        d.line((48,y+37,1232,y+37),fill=HAIR)
    label(d,(50,558),'¹ 每段至少连续 1 s，五组各 240 s。表中为每组累计时长的平均值。',18,MUTED)
    clock=summary['dispatch']
    label(d,(50,617),f"独立 120 s 墙钟试验    腕部规划 P99 {clock['timing']['plan']['p99']:.2f} ms    ·    100 Hz 整帧 P99 {clock['timing']['finish_from_release']['p99']:.2f} ms",21,bold=True)
    label(d,(50,662),f"梨长程仿真的分区载荷峰值为 {summary['pear']['final']['peak_n']:.1f} N。真机载荷与通信时延尚待验证。",18,MUTED)
    return canvas


def render(root, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary=read(root/'guard-summary.json')
    total=0
    chapters=[]
    with TemporaryDirectory(dir=destination.parent,prefix='sweep-video-') as tmp, ExitStack() as stack:
        work=Path(tmp)
        pear=Replay(stack,root,'pear','active-02')
        ellipse=Replay(stack,root,'ellipsoid','start-01')
        events=read(pear.arms[0].raw/'recoveries.json')
        event=next(e for e in events if e['restored_two_s'] is not None and e['new_cells']>0)
        trigger, restored=[round(100*event[k]) for k in ('start_s','restored_two_s')]
        a,b=trigger-200,restored+250
        hold,start,end,trio=longest_three(ellipse.arms[0])
        baseline=pear.arms[1].scores
        summary['baseline_240']=next(s for s in baseline['snapshots'] if s['s']==240)
        summary['baseline_t70']=baseline['thresholds_s']['70']
        old=read(root/'references/pear/active-02/history-dp/scores.json')['snapshots'][-1]['coverage']
        last=len(pear.arms[0].q)-1
        title='梨的主动触觉探索'
        common='同一初态，视角随腕。绿色指尖为轻载接触，橙色为达到载荷门槛时的接触。'
        background='点云累积已发生的轻载触点。网格使用角标时刻以前的数据。'

        with imageio.get_writer(work/'frames.mp4', fps=FPS, codec='libx264', quality=None, macro_block_size=2,
                                ffmpeg_params=['-preset','fast','-crf','19','-pix_fmt','yuv420p']) as writer:
            def append(image, repeats=1):
                nonlocal total
                pixels=np.asarray(image)
                for _ in range(repeats): writer.append_data(pixels)
                total+=repeats

            def chapter(text):
                chapters.append((total,text))
                print(f'{total/FPS:.2f}s  {text}',flush=True)

            chapter('梨的配对探索')
            for rate,ticks in [(1,span(0,200,1)),(12,span(200,a,12))]:
                for t in ticks:
                    append(draw_frame(pear,t,title,common,f'{rate}×',background))
            chapter('原速回放：失触与重接触')
            for t in span(a,b,1):
                if t<trigger: state='双指支撑不足'
                elif t>=restored: state='已连续 0.20 s 恢复轻载双指，继续扫查'
                else: state=STAGES[int(pear.arms[0].phase[t])] if pear.arms[0].phase[t] else '继续扫查，等待稳定双指'
                append(draw_frame(pear,t,'梨的失触与重接触','第一个恢复双指、并采到新触点的复触事件','原速 1×',
                                  f'新腕轨   {state}',trigger))
            chapter('240 秒与完整覆盖')
            for t in span(b,24000,6): append(draw_frame(pear,t,title,common,'6×',background))
            append(draw_frame(pear,24000,'梨 · 240 秒对照','两侧网格均使用本组前 240 秒内的轻载实触点','定格',
                              f'上一版局部腕轨的全程最终覆盖为 {old:.2%}。'),FPS*3)
            for t in span(24000,last,36): append(draw_frame(pear,t,title,common,'36×',background))
            append(draw_frame(pear,last,'梨 · 全程结束','两侧采用相同的仿真时间预算','定格',
                              f"全程 {len(events)} 次显式复触，其中 {summary['recoveries']['restored']} 次恢复轻载双指。"),FPS*3)
            chapter('原速回放：平滑曲面连续三指')
            names='、'.join(('食指','中指','无名指','拇指')[i] for i in trio)
            for t in span(start-125,end+145,1):
                elapsed=np.clip((t-start+1)/100,0,hold/100)
                note=(f'观察指尖   {names}' if t<start else
                      f'新腕轨   {names}连续轻载 {elapsed:.2f} s，腕部持续移动' if t<end else
                      f'这段连续接触共 {hold/100:.2f} s，按 100 Hz 原始记录统计。')
                append(draw_frame(ellipse,t,'椭球上的连续三指接触','I01-A 椭球起点 A，选取新腕轨最长的移动中同三指片段',
                                  '原速 1×',note,start))
            chapter('结果汇总')
            append(conclusion(summary),FPS*7)

        meta=[';FFMETADATA1','title=GPIS 手腕主动探索与 Aude 对照']
        for i,(frame,name) in enumerate(chapters):
            stop=chapters[i+1][0] if i+1<len(chapters) else total
            meta+=['[CHAPTER]',f'TIMEBASE=1/{FPS}',f'START={frame}',f'END={stop}',f'title={name}']
        (work/'chapters.ffmeta').write_text('\n'.join(meta)+'\n',encoding='utf-8')
        staged=work/'paired.mp4'
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(work/'frames.mp4'),'-f','ffmetadata',
                        '-i',str(work/'chapters.ffmeta'),'-map','0:v:0','-map_metadata','1','-map_chapters','1',
                        '-c','copy','-movflags','+faststart',str(staged)],check=True)
        frames,_=imageio_ffmpeg.count_frames_and_secs(str(staged))
        if frames!=total: raise ValueError(f'Encoded {frames} frames, expected {total}')
        staged.replace(destination)
    print(f'{destination} | {total} frames | {total/FPS:.2f} s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=OUT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    render(args.root,args.output or args.root/'force-priority/wrist-active-result.mp4')
