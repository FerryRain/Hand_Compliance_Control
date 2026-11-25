"""Render two saved 100 Hz wrist explorations under one fixed camera; no physics rollout."""

import argparse
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
import trimesh

from ..active_1_b import film_support as film
from ..active_1_b.video import HandView, view_axes
from ..i04_geometric_traversal.pad_contact_visual import camera_from_direction
from .experiment import digest
from .ycb_fixtures import OBJECTS, fixture


GENERATED = Path(__file__).resolve().parents[1]/'generated'
OLD = GENERATED/'wrist_tactile_field_20261124'/'runs'
NEW = GENERATED/'wrist_supported_field_20261124'
OBJECT_NAMES = dict(pear='梨', mustard='芥末瓶', cracker='饼干盒', bleach='清洁剂瓶', coffee='咖啡罐')
FINGERS = ('食指','中指','环指','拇指')
STAGES = dict(explore='扫查', stabilize='稳住贴合', recontact='重新贴合', withdraw='卸载退腕',
              transfer='转移', approach='重新靠近', scan='扫查', hold='保持', recover='恢复贴合')
SIZE, VIEW = (1280,720), (604,390)
FPS, SPEED = 20, 4
END = 24000
TICKS = np.arange(0, END+1, 100*SPEED//FPS)
WITNESS = (0, 9000, 12000, 15000, 18000, 21000, 24000)
DETAIL_REASON = 'Preset two-second margins about the 120 s witness inside the previous 90–240 s stagnation.'
BG, LINE, INK, MUTED = '#111b24', '#354653', '#eff5f7', '#b2c3ca'
TARGET, EXPIRED, WRIST, CONTACT, HIGH = '#ffce68', '#9aabae', '#72bdff', '#56edb5', '#fe8d75'


def read(path):
    return json.loads(path.read_bytes())


class Recorded:
    def __init__(self, folder, obj, expected_steps=END):
        self.folder=folder
        self.run=read(folder/'run.json')
        self.clock=read(folder/'clock.json')
        self.end_tick=expected_steps
        if self.run['steps']!=expected_steps or self.run['signature']['obj']!=obj:
            raise ValueError(f'Expected the complete {expected_steps/100:g} s {obj} run at {folder}')
        for filename, expected in (('trace.npz',self.run['trace_sha256']),('plans.jsonl',self.run['plan_sha256'])):
            if digest(folder/filename)!=expected:
                raise ValueError(f'Acquisition SHA-256 mismatch for {folder/filename}')
        with np.load(folder/'trace.npz',allow_pickle=False) as t:
            self.q=t['qpos']; self.points=t['points_m']; self.normals=t['normals_outward']
            self.live=t['pad_valid']; self.loads=t['region_load_n']; self.force=t['pad_force_n']
        np.testing.assert_array_equal(self.live,(self.force>=.2)&(np.linalg.norm(self.normals,axis=2)>.99))
        if self.q.shape!=(expected_steps+1,23) or not np.isfinite(self.q).all():
            raise ValueError('Incomplete saved physical posture')
        self.peak=self.loads.max(1)
        self.light=self.live&(self.peak<8)[:,None]
        self.count=self.light.sum(1)
        self.three_s=np.r_[0,np.cumsum(self.count[1:]>=3)*.01]
        self.zero_s=np.r_[0,np.cumsum(~self.live[1:].any(1))*.01]
        self.wrist_path=np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.q[:,:3],axis=0),axis=1))]
        with (folder/'plans.jsonl').open(encoding='utf8') as plans:
            self.rows=[json.loads(line) for line in plans]
        self.applied=np.array([row['apply_tick'] for row in self.rows])
        if self.applied[0]!=0 or (np.diff(self.applied)<=0).any():
            raise ValueError('Missing initial action or nonmonotonic action application')
        self.events=[]
        self.endpoints={}
        for i,row in enumerate(self.rows):
            event=row['global_event']
            if event is None:
                continue
            if not 0<=event['observed_tick']<=row['apply_tick']:
                raise ValueError('A target was shown before it was available to the controller')
            selected=bool(event['direction_chosen'])
            point=np.asarray(event['target_m'],float) if selected else None
            if selected and (point.shape!=(3,) or not np.isfinite(point).all()):
                raise ValueError('Invalid recorded physical surface target')
            issued=selected and row['mode']=='explore'
            end=reason=None
            if issued:
                # Match audit_targets: an uninterrupted stretch of explore actions from this issue.
                j=i+1
                while j<len(self.rows) and self.rows[j]['mode']=='explore' and self.rows[j]['global_event'] is None:
                    j+=1
                end=int(self.rows[j]['apply_tick']) if j<len(self.rows) else expected_steps
                reason=(self.rows[j]['mode'] if self.rows[j]['mode']!='explore' else 'next_global') if j<len(self.rows) else 'end'
            recorded=dict(applied=int(row['apply_tick']),observed=int(event['observed_tick']),point=point,
                          finger=event.get('target_finger'),source=event.get('source'),issued_in_explore=issued,
                          inclusive_end_tick=end,end_reason=reason)
            self.events.append(recorded)
            if issued:
                self.endpoints.setdefault(end,[]).append(recorded)
        self.event_ticks=np.array([e['applied'] for e in self.events],int)

    def state(self,tick):
        row=self.rows[int(np.searchsorted(self.applied,tick,side='right'))-1]
        i=int(np.searchsorted(self.event_ticks,tick,side='right'))-1
        return row,self.events[i] if i>=0 else None

    def goal_status(self,event,tick):
        if event is None or event['point'] is None:
            return 'none'
        if not event['issued_in_explore']:
            return 'unissued'
        if tick<event['inclusive_end_tick'] or (tick==self.end_tick and event['end_reason']=='end'):
            return 'active'
        return 'endpoint' if tick==event['inclusive_end_tick'] else 'expired'

    def wrist_motion(self,start,end):
        net=float(np.linalg.norm(self.q[end,:3]-self.q[start,:3]))
        return net,float(self.wrist_path[end]-self.wrist_path[start])

    def interval(self,start,end):
        net,path=self.wrist_motion(start,end)
        steps=slice(start+1,end+1)
        return dict(start_s=.01*start,end_s=.01*end,wrist_net_m=net,wrist_path_m=path,
                    wrist_axis_spans_m=np.ptp(self.q[start:end+1,:3],axis=0).tolist(),
                    measured_contact_per_pad_s=(self.live[steps].sum(0)*.01).tolist(),
                    zero_tip_s=float(self.zero_s[end]-self.zero_s[start]),
                    light2_s=float((self.count[steps]>=2).sum()*.01),
                    light3_s=float(self.three_s[end]-self.three_s[start]))

    def witness(self,tick):
        row,event=self.state(tick)
        return dict(tick=int(tick),mode=row['mode'],local_apply_tick=row['apply_tick'],
                    displayed_target_status=self.goal_status(event,tick),
                    final_scoring_sample_of_goals_issued_at=[e['applied'] for e in self.endpoints.get(tick,())],
                    selected_event=None if event is None else {**event,'point':None if event['point'] is None else event['point'].tolist()},
                    actual_wrist_m=self.q[tick,:3].tolist(), measured_valid_mask=self.live[tick].tolist(),
                    measured_points_m=[self.points[tick,f].tolist() if ok else None for f,ok in enumerate(self.live[tick])],
                    light_heads=int(self.count[tick]),peak_five_zones_n=float(self.peak[tick]),
                    cumulative_light3_s=float(self.three_s[tick]),cumulative_zero_s=float(self.zero_s[tick]))


def shared_direction(vertices,pair):
    """Choose a fixed view that sees the occupied sides of both trials and the object's length."""
    tree=cKDTree(vertices)
    long_axis=np.linalg.svd(vertices-vertices.mean(0),full_matrices=False)[2][0]
    outward=[]
    for trial in pair:
        wrist=trial.q[::100,:3]
        ray=wrist-vertices[tree.query(wrist)[1]]
        outward.append(ray/np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1e-9))
    el,az=np.meshgrid(np.deg2rad(np.arange(-60,61,20)),np.deg2rad(np.arange(0,360,15)))
    directions=np.column_stack(((np.cos(el)*np.cos(az)).ravel(),(np.cos(el)*np.sin(az)).ravel(),np.sin(el).ravel()))
    visibility=np.array([np.clip(v@directions.T,-.4,.7).mean(0) for v in outward])
    score=.65*visibility.mean(0)+.35*visibility.min(0)-.25*(directions@long_axis)**2
    return directions[int(np.argmax(score))]


class FixedView:
    def __init__(self,model,mesh,pair,sample_ticks=None):
        vertices=np.asarray(mesh.vertices)
        pose=pair[0].q[0]
        self.hand=HandView(model,vertices,pose)
        self.model=model
        self.data=self.hand.data
        model.vis.global_.offwidth,model.vis.global_.offheight=VIEW
        model.geom('bottle_visual').rgba[:]=(.78,.65,.37,.28)
        self.eye=shared_direction(vertices,pair)
        self.basis=view_axes(self.eye)
        tan=np.tan(np.deg2rad(model.vis.global_.fovy*.5))
        xk=1.075*VIEW[1]/(tan*VIEW[0]);yk=1.075/tan
        # Each camera-plane bound is linear in projected position. Taking the
        # four extrema fits all poses without adding unrelated x/y/depth maxima.
        projection_limits=np.array([[xk,0,1],[-xk,0,1],[0,yk,1],[0,-yk,1]]).T
        projected=vertices@self.basis
        low=projected.min(0);high=projected.max(0)
        bounds=(projected@projection_limits).max(0)
        for trial in pair:
            # Use every rendered posture of both hands to set a common field of view, once.
            for qpos in trial.q[TICKS if sample_ticks is None else sample_ticks]:
                self.data.qpos[:]=qpos
                mujoco.mj_kinematics(model,self.data)
                ids=self.hand.ids
                bodies=self.data.geom_xpos[ids,None,:]+np.einsum(
                    'gij,gkj->gki',self.data.geom_xmat[ids].reshape(-1,3,3),self.hand.corners)
                xy=bodies.reshape(-1,3)@self.basis
                low=np.minimum(low,xy.min(0));high=np.maximum(high,xy.max(0))
                bounds=np.maximum(bounds,(xy@projection_limits).max(0))
            for event in trial.events:
                if event['point'] is not None:
                    xyz=event['point']@self.basis
                    low=np.minimum(low,xyz);high=np.maximum(high,xyz)
                    bounds=np.maximum(bounds,xyz@projection_limits)
        center=(high+low)*.5
        self.lookat=self.basis@center
        self.distance=max(.30,float(np.max(bounds-center@projection_limits)))
        self.focal=VIEW[1]/(2*tan)
        camera_from_direction(self.hand.camera,self.eye,self.lookat,self.distance)

    def pixels(self,points):
        view=(np.asarray(points)-self.lookat)@self.basis
        xy=view[...,:2]*(self.focal/(self.distance-view[...,2]))[...,None]
        return xy*[1,-1]+np.asarray(VIEW)*.5

    def image(self,renderer,trial,t):
        self.data.qpos[:]=trial.q[t]
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_camlight(self.model,self.data)
        for f,g in enumerate(self.hand.pads):
            self.model.geom_rgba[g]=((.22,.94,.62,1.) if trial.light[t,f] else
                                     (1.,.38,.23,1.) if trial.live[t,f] else (.43,.60,.69,1.))
        renderer.update_scene(self.data,camera=self.hand.camera,scene_option=self.hand.options)
        # Native stereo cameras are averaged for the rendered monoscopic image.
        camera=renderer.scene.camera
        midpoint=.5*(camera[0].pos+camera[1].pos)
        np.testing.assert_allclose(midpoint,self.lookat+self.distance*self.eye,atol=3e-6,rtol=0)
        np.testing.assert_allclose(camera[0].forward,-self.eye,atol=3e-6,rtol=0)
        np.testing.assert_allclose(camera[0].up,self.basis[:,1],atol=3e-6,rtol=0)
        np.testing.assert_allclose(camera[0].frustum_top/camera[0].frustum_near,VIEW[1]/(2*self.focal),atol=2e-7,rtol=0)
        image=Image.fromarray(renderer.render())
        pen=ImageDraw.Draw(image)
        _,event=trial.state(t)
        path=trial.q[max(0,t-100):t+1:5,:3]
        if len(path)>1:
            pen.line([tuple(p) for p in self.pixels(path)],fill='#5485a3',width=2)
        symbol(pen,'wrist',self.pixels(trial.q[t,:3]),WRIST)
        status=trial.goal_status(event,t)
        # The measured endpoint is scored from the outgoing command. The new command is active at the same tick.
        for ended in trial.endpoints.get(t,()):
            if ended is not event:
                symbol(pen,'expired',self.pixels(ended['point']),EXPIRED)
        if status!='none':
            symbol(pen,'target' if status=='active' else 'expired',self.pixels(event['point']),
                   TARGET if status=='active' else EXPIRED)
        for f in np.flatnonzero(trial.live[t]):
            symbol(pen,'contact',self.pixels(trial.points[t,f]),CONTACT if trial.light[t,f] else HIGH)
        return image


def symbol(draw,kind,pixel,color):
    x,y=map(int,np.rint(pixel));dark='#13202b'
    if kind=='target':
        path=[(x,y-10),(x+10,y),(x,y+10),(x-10,y),(x,y-10)]
        draw.line(path,fill=dark,width=6);draw.line(path,fill=color,width=3)
    elif kind=='expired':
        path=[(x,y-7),(x+7,y),(x,y+7),(x-7,y),(x,y-7)]
        draw.line(path,fill=dark,width=4);draw.line(path,fill=color,width=2)
        draw.line((x-6,y+6,x+6,y-6),fill=color,width=2)
    elif kind=='wrist':
        draw.rectangle((x-6,y-6,x+6,y+6),outline=dark,width=4)
        draw.rectangle((x-5,y-5,x+5,y+5),outline=color,width=2)
    else:
        draw.ellipse((x-5,y-5,x+5,y+5),fill=color,outline=dark,width=2)


def short_history(draw,x,y,trial,t):
    width,height=172,23
    draw.rectangle((x,y,x+width,y+height),fill='#1d303a')
    # The plot covers the previous ten seconds of all 100 Hz samples, including short real contacts.
    first=max(0,t-999); data=trial.count[first:t+1]
    xx=x+np.arange(1000-len(data),1000)*(width-1)/999
    yy=y+height-2-data*(height-4)/4
    if len(data)>1:
        draw.line(list(zip(xx.tolist(),yy.tolist())),fill=CONTACT,width=2)
    else:
        draw.point((xx[0],yy[0]),fill=CONTACT)


def panel(draw,x,t,trial,title,tag):
    label=film.label
    row,event=trial.state(t)
    label(draw,(x,91),title,22,bold=True)
    mode=STAGES.get(row['mode'],row['mode'])
    label(draw,(x+VIEW[0],92),mode,17,tag,anchor='rm')
    label(draw,(x,526),f'轻载 {trial.count[t]} / 4 指',22,CONTACT if trial.count[t] else INK,bold=True)
    for f,name in enumerate(('食','中','环','拇')):
        left=x+150+35*f
        fill='#245c4b' if trial.light[t,f] else '#7d4235' if trial.live[t,f] else '#2b3a45'
        draw.rounded_rectangle((left,512,left+28,538),radius=5,fill=fill)
        label(draw,(left+14,525),name,15,INK,anchor='mm')
    label(draw,(x+VIEW[0],526),f'五区峰值 {trial.peak[t]:.2f} N',18,HIGH if trial.peak[t]>=8 else MUTED,anchor='rm')
    status=trial.goal_status(event,t)
    if status=='none':
        target='当前无执行中的物面目标'
    else:
        who=FINGERS[event['finger']] if event['finger'] in range(4) else '多指'
        desc={'active':'执行中物面目标','unissued':'已选点，未开始探索','endpoint':'探索中断 · 上段评分末帧',
              'expired':'已中断的旧目标'}[status]
        target=f"{desc}   {who} · 选于 {event['applied']*.01:.2f} s"
    if any(e is not event for e in trial.endpoints.get(t,())):
        target+=' · 上段评分末帧'
    label(draw,(x,564),target,17,TARGET if status=='active' else MUTED)
    label(draw,(x,600),'近 10 s 轻载指腹数',16,MUTED)
    short_history(draw,x+169,588,trial,t)
    label(draw,(x+157,590),'4',11,MUTED,mono=True)
    label(draw,(x+157,609),'0',11,MUTED,mono=True)
    label(draw,(x+VIEW[0],600),f'三指累计 {trial.three_s[t]:.2f} s',17,INK,anchor='rm')
    net,path=trial.wrist_motion(max(0,t-1000),t)
    label(draw,(x,633),f'近 10 s 腕净移 {net*1000:.0f} mm / 行程 {path*1000:.0f} mm',16,MUTED)
    label(draw,(x+VIEW[0],633),f'零指累计 {trial.zero_s[t]:.2f} s',17,MUTED,anchor='rm')


def frame(obj,t,pair,view,renderer,preview=False,playback_speed=SPEED,timeline_note=None):
    canvas=Image.new('RGB',SIZE,BG);draw=ImageDraw.Draw(canvas);label=film.label
    label(draw,(24,35),f'{OBJECT_NAMES[obj]} · 腕部探索对照',30,bold=True)
    label(draw,(1256,34),f'{t*.01:06.2f} / 240.00 s',28,INK,bold=True,mono=True,anchor='rm')
    caption='排版核验 · 两侧均使用上一版的同一条轨迹' if preview else '相同初始姿态与相机  ·  single_all + MCC'
    label(draw,(24,63),caption,15,MUTED)
    if timeline_note is not None:
        label(draw,(466,63),f"本轮 {timeline_note['tick']*.01:.2f} s · {timeline_note['label']}",16,TARGET)
    playback=f'{playback_speed:g}× 回放'
    if playback_speed<1:
        playback+=' · 逐物理帧'
    label(draw,(1256,63),f'仿真时间   /   {playback}',16,MUTED,anchor='rm')
    draw.line((24,74,1256,74),fill=LINE)
    titles=('上一版腕部 Active','上一版轨迹 · 布局测试') if preview else ('上一版腕部 Active','本轮腕部 Active')
    for i,(trial,title,tag) in enumerate(zip(pair,titles,('#abc4d7',CONTACT))):
        x=(24,652)[i]
        canvas.paste(view.image(renderer,trial,t),(x,109))
        panel(draw,x,t,trial,title,tag)
    draw.line((24,658,1256,658),fill=LINE)
    for kind,color,x,text in (('target',TARGET,33,'执行中物面目标'),('expired',EXPIRED,241,'已中断或替换的目标'),
                              ('wrist',WRIST,468,'实际腕部与近 1 s 轨迹'),('contact',CONTACT,711,'轻载实触指腹'),
                              ('contact',HIGH,877,'超载时实触指腹')):
        symbol(draw,kind,(x,687),color);label(draw,(x+17,687),text,16,MUTED)
    label(draw,(1256,687),'五区均 < 8 N 为轻载',16,MUTED,anchor='rm')
    return canvas


def verify_movie(output,witnesses,expected=None,fps=FPS):
    # Decode the complete H.264 stream, then compare selected timestamps against
    # reduced pre-encode images (including the physical view, markers and clock).
    expected=len(TICKS) if expected is None else expected
    small=(320,180); checked={}
    stream=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24',output_params=['-vf','scale=320:180:flags=area','-threads','1'])
    n=0
    with closing(stream):
        meta=next(stream)
        if tuple(meta['source_size'])!=SIZE or tuple(meta['size'])!=small or abs(meta['fps']-fps)>.001:
            raise ValueError(f'Unexpected video metadata {meta}')
        for n,raw in enumerate(stream,1):
            i=n-1
            if i in witnesses:
                actual=np.frombuffer(raw,np.uint8).reshape((small[1],small[0],3))
                error=float(np.abs(actual.astype(float)-witnesses[i]).mean())
                if error>4:
                    raise ValueError(f'Video frame {i} differs from its source timestamp: MAE {error:.2f}')
                checked[str(i)]=error
    if n!=expected or len(checked)!=len(witnesses) or abs(meta['duration']-n/fps)>.015:
        raise ValueError(f'Video length or witness mismatch at {output}: {n} frames')
    return dict(checked_all_decoded_frames=n,fps=meta['fps'],duration_s=meta['duration'],
                frame_size=SIZE,codec=meta['codec'],witness_mean_abs_pixel_error=checked)


def encode(obj,pair,view,renderer,out,name,ticks,fps,witness_ticks,poster_tick=12000,
           selection_reason='Complete paired trial',timeline_events=()):
    old,fresh=pair
    speed=fps*int(ticks[1]-ticks[0])/100
    with TemporaryDirectory(dir=out,prefix=obj+'-') as temp:
        output=Path(temp)/(name+'.mp4'); saved={};rows=[]
        with imageio.get_writer(output,fps=fps,codec='libx264',quality=8,macro_block_size=1,
                                pixelformat='yuv420p',output_params=['-movflags','+faststart','-threads','2']) as writer:
            for i,t in enumerate(ticks):
                t=int(t)
                note=next((event for event in reversed(timeline_events) if event['tick']<=t),None)
                picture=frame(obj,t,pair,view,renderer,playback_speed=speed,timeline_note=note)
                writer.append_data(np.asarray(picture))
                if t in witness_ticks:
                    saved[i]=np.array(picture.resize((320,180),Image.Resampling.BOX),dtype=float)
                    rows.append(dict(video_frame=i,video_time_s=i/fps,simulation_s=t*.01,
                                     previous=old.witness(t),current=fresh.witness(t)))
                    if t==poster_tick:
                        picture.save(output.with_suffix('.webp'),quality=94)
        decoded=verify_movie(output,saved,len(ticks),fps)
        source=[]
        for trial in pair:
            source.append(dict(run=str(trial.folder.resolve()),release=trial.clock['release'],
                               trace_sha256=trial.run['trace_sha256'],plans_sha256=trial.run['plan_sha256'],
                               intervals=[trial.interval(start,end) for start,end in ((0,9000),(9000,24000))]))
        start=int(ticks[0]);end=int(ticks[-1])
        meta=dict(object=obj,simulation_s=(end-start)*.01,simulation_window_s=[start*.01,end*.01],
                  selection_reason=selection_reason,timeline_events=timeline_events,
                  playback_speed=speed,physical_sample_hz=100,video_fps=fps,
                  tick_stride=int(ticks[1]-ticks[0]),source=source,model_sha256=old.run['model_sha256'],
                  initial_sha256=old.run['initial_sha256'],mesh_sha256=old.run['offline_mesh_sha256'],
                  camera=dict(eye_direction=view.eye.tolist(),lookat_m=view.lookat.tolist(),distance_m=view.distance,
                              fovy_deg=float(view.model.vis.global_.fovy),fixed_for_both_views=True),
                  timeline='video time (s) = (simulation time - simulation start) / playback speed; both endpoints included',
                  target='Gold requires an issued target and uninterrupted exploration. Grey marks a retired target; its final physical sample at the interruption/replacement tick is included in arrival scoring. Later resumption does not reactivate it.',
                  witnesses=rows,decode=decoded,video_sha256=digest(output))
        output.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
        for f in (output,output.with_suffix('.webp'),output.with_suffix('.json')):
            f.replace(out/f.name)
        print(f'{name} source {start*.01:g}–{end*.01:g}s; verified {decoded["checked_all_decoded_frames"]} frames, {decoded["duration_s"]}s',flush=True)


def render(obj,old_runs,new_runs,out,preview=False,detail_center_s=120.,detail_reason=DETAIL_REASON,
           detail_events=(),detail_only=False):
    center=int(round(detail_center_s*100));start,end=center-200,center+200
    if obj=='mustard' and (start<0 or end>END):
        raise ValueError('Four-second slow-motion window must fit inside the recorded trial')
    case,surface=fixture(obj)
    old=Recorded(old_runs/obj/'ours',obj)
    fresh=old if preview else Recorded(new_runs/obj/'ours',obj)
    for key in ('seed','model_sha256','initial_sha256','offline_mesh_sha256'):
        if old.run[key]!=fresh.run[key]:
            raise ValueError(f'Unpaired runs for {obj} and {key}')
    for key in ('q','points','normals','loads','force','live'):
        np.testing.assert_array_equal(getattr(old,key)[0],getattr(fresh,key)[0])
    if digest(case/'model.mjb')!=old.run['model_sha256'] or digest(surface)!=old.run['offline_mesh_sha256']:
        raise ValueError('Replay model or evaluation surface differs from the frozen trial')
    pair=(old,fresh); model=film.read_model(case/'model.mjb')
    view=FixedView(model,trimesh.load_mesh(surface,process=False),pair)
    out.mkdir(parents=True,exist_ok=True)
    with mujoco.Renderer(model,height=VIEW[1],width=VIEW[0]) as renderer:
        if preview:
            old_frame=frame(obj,12000,pair,view,renderer,preview=True)
            old_frame.save(out/f'{obj}-layout.webp',quality=95)
            return
        if not detail_only:
            encode(obj,pair,view,renderer,out,obj+'-paired-240s',TICKS,FPS,WITNESS)
        if obj=='mustard':
            # At 0.25×, every saved 100 Hz wrist pose and short contact in the window is visible.
            name=f'mustard-detail-{start/100:g}-{end/100:g}s'
            witness_ticks=set(range(start,end+1,100))|{event['tick'] for event in detail_events}
            if not all(start<=t<=end for t in witness_ticks):
                raise ValueError('A detail event lies outside the requested four-second interval')
            encode(obj,pair,view,renderer,out,name,np.arange(start,end+1),25,witness_ticks,
                   poster_tick=center,selection_reason=detail_reason,timeline_events=detail_events)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--old-runs',type=Path,default=OLD)
    ap.add_argument('--new-runs',type=Path,default=NEW/'runs')
    ap.add_argument('--out',type=Path,default=NEW/'videos')
    ap.add_argument('--objects',nargs='+',choices=tuple(OBJECTS),default=['mustard','pear','cracker','bleach','coffee'])
    ap.add_argument('--layout-only',action='store_true',help='Preview one old physical frame before a new run exists')
    ap.add_argument('--mustard-detail-at',type=float,default=120.,help='Simulation time centered in the four-second 100 Hz detail')
    ap.add_argument('--detail-reason',default=DETAIL_REASON)
    ap.add_argument('--detail-events',type=Path,help='JSON array of independently verified {tick,label} events for the slow replay')
    ap.add_argument('--detail-only',action='store_true',help='Rebuild the mustard detail after independent event verification')
    args=ap.parse_args()
    if args.detail_only and args.objects!=['mustard']:
        ap.error('--detail-only requires --objects mustard')
    detail_events=sorted(read(args.detail_events),key=lambda r:r['tick']) if args.detail_events else []
    for obj in args.objects:
        render(obj,args.old_runs,args.new_runs,args.out,args.layout_only,args.mustard_detail_at,args.detail_reason,
               detail_events,args.detail_only)
