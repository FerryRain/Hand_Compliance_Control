"""Interactive extension of view_inverse_palm_plans: planned poses, no dynamics."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import threading
import time

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from view_inverse_palm_plans import (
    HAND_JOINT_NAMES, PALM_CONTROL_POINT_LOCAL, _add_cylinder,
    _fixed_hand_spec, _scene_spec)

RAW = '全部规划节点'
VALID = '仅有效节点'


@dataclass
class PlanReference:
    path: Path
    object_id: str
    scale: float
    dt: float
    poses: np.ndarray
    control: np.ndarray
    frames: np.ndarray
    q: np.ndarray
    valid: np.ndarray

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, path):
        path=Path(path).resolve()
        with h5py.File(path) as h:
            pose=np.asarray(h['palm_pose_object'],float)
            pose=pose[:,0] if pose.ndim==3 else pose
            if pose.ndim!=2 or pose.shape[1]!=7 or len(pose)<2 or not np.isfinite(pose).all():
                raise ValueError(f'Invalid palm reference: {path}')
            norm=np.linalg.norm(pose[:,3:],axis=-1)
            if (norm<1e-8).any():raise ValueError('Zero palm quaternion')
            pose[:,3:]/=norm[:,None]
            if 'grasp_keyframe_q' in h:
                q=np.asarray(h['grasp_keyframe_q'],float)
                q=q[:,0] if q.ndim==3 else q
                frames=np.asarray(h['grasp_keyframe_frame_index']).reshape(len(q),-1)[:,0]
                valid=np.asarray(h['grasp_keyframe_valid'],bool).reshape(len(q),-1)[:,0]
            else:
                q=np.asarray(h['q_hand'],float)
                q=q[:,0] if q.ndim==3 else q
                frames=np.arange(len(q));valid=np.ones(len(q),bool)
            if q.shape!=(len(frames),16) or not np.isfinite(q).all():
                raise ValueError('Viewer requires finite 16D planned hand waypoints')
            if (frames<0).any() or (frames>=len(pose)).any():
                raise ValueError('Waypoint frame falls outside the palm reference')
            order=np.argsort(frames,kind='stable')
            frames=frames[order].astype(int);q=q[order];valid=valid[order]
            unique=np.r_[True,np.diff(frames)>0]
            frames,q,valid=frames[unique],q[unique],valid[unique]
            local=np.asarray(h.attrs.get('planner_palm_control_point_local',PALM_CONTROL_POINT_LOCAL))
            control=pose[:,:3]+Rotation.from_quat(pose[:,[4,5,6,3]]).apply(local)
            return cls(path,str(h.attrs['object_id']),float(h.attrs.get('object_scale',1.)),
                float(h.attrs.get('control_dt',.01)),pose,control,frames,q,valid)

    def indices(self, valid_only=False):
        indices=np.flatnonzero(self.valid) if valid_only else np.arange(len(self.frames))
        if not len(indices):raise ValueError('This plan has no valid hand waypoint')
        return indices

    def sample(self, frame, valid_only=False):
        frame=int(np.clip(frame,0,len(self.poses)-1))
        indices=self.indices(valid_only)
        times=self.frames[indices]
        right=int(np.clip(np.searchsorted(times,frame,side='right'),1,max(1,len(times)-1)))
        left=right-1
        if len(times)==1:left=right=0
        phase=float(np.clip((frame-times[left])/max(int(times[right]-times[left]),1),0.,1.))
        phase=phase*phase*(3-2*phase)
        q=(1-phase)*self.q[indices[left]]+phase*self.q[indices[right]]
        nearest=int(np.argmin(abs(times-frame)))
        return self.poses[frame],q,int(indices[left]),int(indices[right]),nearest


def catalog(args):
    if args.plans:
        return [dict(label=f'{i+1:03d} | {p.stem}',path=str(p.resolve()),candidate_id=None)
                for i,p in enumerate(args.plans)]
    manifest=args.trajectory_root/'collection_manifest.json'
    rows=json.loads(manifest.read_text())['rows']
    entries=[]
    for row in sorted(rows,key=lambda r:r['candidate_id']):
        if row.get('attempt_kind')!='physical':continue
        summary=json.loads((Path(row['source_output'])/'summary.json').read_text())
        path=Path(summary['plan'])
        if not path.is_file():raise FileNotFoundError(path)
        entries.append(dict(label=f"{row['candidate_id']:04d} | {row['family']} | "
                f"{summary['requested_task_frames']*.01:.1f}s",path=str(path),
            candidate_id=row['candidate_id']))
    if not entries:raise ValueError('No executable references in this catalog')
    return entries


class ReferenceScene:
    """Same object/hand assets as the existing viewer, with movable ghost roots."""
    def __init__(self, reference, ghost_count=8, alpha=.5, ghost_alpha=.12):
        spec=_scene_spec(reference.object_id,reference.scale,[],False)
        spec.visual.global_.offwidth=1600
        spec.visual.global_.offheight=1200
        for i in range(ghost_count+1):
            anchor=spec.worldbody.add_body(name=f'pose_{i}',mocap=True)
            hand=_fixed_hand_spec((1.,.65,.1,1.) if i==0 else (.2,.65,1.,1.))
            for geom in hand.geoms:geom.rgba[3]=alpha if i==0 else ghost_alpha
            spec.attach(hand,prefix=f'hand_{i}/',frame=anchor.add_frame())
        self.model=spec.compile();self.data=mujoco.MjData(self.model)
        self.ghost_count=ghost_count
        self.addresses=np.asarray([[self.model.jnt_qposadr[mujoco.mj_name2id(
            self.model,mujoco.mjtObj.mjOBJ_JOINT,f'hand_{i}/{name}')]
            for name in HAND_JOINT_NAMES] for i in range(ghost_count+1)])
        self.mocap=np.asarray([self.model.body(f'pose_{i}').mocapid[0]
                              for i in range(ghost_count+1)],int)
        self.reference=reference
        self.set_reference(reference)

    def set_hand(self,index,pose,q):
        self.data.mocap_pos[self.mocap[index]]=pose[:3]
        self.data.mocap_quat[self.mocap[index]]=pose[3:]
        self.data.qpos[self.addresses[index]]=q

    def set_reference(self,reference,valid_only=False):
        if reference.object_id!=self.reference.object_id or reference.scale!=self.reference.scale:
            raise ValueError('A viewer catalog must use the same object and scale')
        self.reference=reference
        indices=reference.indices(valid_only)
        self.ghost_indices=indices[np.unique(np.linspace(0,len(indices)-1,
            min(self.ghost_count,len(indices))).round().astype(int))] if self.ghost_count else []
        for i in range(self.ghost_count):
            k=self.ghost_indices[min(i,len(self.ghost_indices)-1)]
            self.set_hand(i+1,reference.poses[reference.frames[k]],reference.q[k])
        self.seek(0,valid_only)

    def seek(self,frame,valid_only=False):
        pose,q,*_=self.reference.sample(frame,valid_only)
        self.set_hand(0,pose,q)
        # FK only. No mj_step, controller, integration, or CUDA allocation.
        mujoco.mj_kinematics(self.model,self.data)

    def check(self):
        errors=[]
        for frame in [0,len(self.reference.poses)//2,len(self.reference.poses)-1]:
            self.seek(frame)
            palm=self.model.body('hand_0/palm_lower').id
            errors.append(float(np.max(abs(self.data.xpos[palm]-self.reference.poses[frame,:3]))))
            np.testing.assert_allclose(self.data.xquat[palm],self.reference.poses[frame,3:],atol=1e-7)
            np.testing.assert_allclose(self.data.qpos[self.addresses[0]],self.reference.sample(frame)[1])
        return dict(max_palm_position_error_m=max(errors),waypoints=len(self.reference.frames),
                    valid_waypoints=int(self.reference.valid.sum()),frames=len(self.reference.poses))


def screenshot(scene,path,frame,stride):
    import imageio.v2 as imageio
    scene.seek(frame)
    camera=mujoco.MjvCamera();points=scene.reference.control
    camera.lookat[:]=(points.min(0)+points.max(0))*.5
    camera.distance=max(.45,float(np.linalg.norm(np.ptp(points,axis=0)))*1.4)
    camera.azimuth=135;camera.elevation=-18
    with mujoco.Renderer(scene.model,height=900,width=1200) as renderer:
        renderer.update_scene(scene.data,camera=camera)
        curve=points[np.unique(np.r_[np.arange(0,len(points),stride),len(points)-1])]
        for a,b in zip(curve[:-1],curve[1:]):
            _add_cylinder(renderer.scene,a,b,.001,(.12,.65,1.,1.))
        path.parent.mkdir(parents=True,exist_ok=True)
        imageio.imwrite(path,renderer.render())


class WaypointBrowser:
    def __init__(self,args,entries,start,server):
        from mjlab.viewer.viser.scene import ViserMujocoScene
        self.scene_type=ViserMujocoScene
        self.args=args;self.entries=entries;self.server=server
        self.lock=threading.RLock();self.updating=False
        self.scene=None;self.drawing=None;self.path_handles=[]
        self.current=start;self.frame=0;self.playhead=0.
        server.scene.set_up_direction('+z')
        server.gui.configure_theme(control_layout='floating',control_width='medium')
        with server.gui.add_folder('轨迹'):
            self.selector=server.gui.add_dropdown('当前轨迹',[r['label'] for r in entries],
                                                  initial_value=entries[start]['label'])
            nav=server.gui.add_button_group('切换轨迹',('上一条','下一条'))
            self.mode=server.gui.add_dropdown('参考内容',[RAW,VALID],initial_value=RAW)
            self.ghosts=server.gui.add_checkbox('沿途透明手型',initial_value=True)
            self.markers=server.gui.add_checkbox('节点标记',initial_value=True)
            reset=server.gui.add_button('重置相机')
        with server.gui.add_folder('时间 / waypoint'):
            self.timeline=server.gui.add_slider('时间帧',min=0,max=1,step=1,initial_value=0)
            self.waypoint=server.gui.add_slider('参考节点',min=0,max=1,step=1,initial_value=0)
            step=server.gui.add_button_group('逐节点查看',('上一节点','下一节点'))
            self.play=server.gui.add_checkbox('播放',initial_value=False)
            self.speed=server.gui.add_slider('播放倍速',min=.1,max=4.,step=.1,initial_value=1.)
            self.info=server.gui.add_markdown('')
        server.gui.add_markdown('**蓝线**：完整手掌控制点参考。**浅蓝透明手**：抽样规划节点。'
            '**黄色透明手**：当前时间。节点：绿色通过，红色未通过。\n\n'
            '显示文件中的规划；不包含在线起点校准、DP、MCC或实际接触。')

        @self.selector.on_update
        def _(_event):
            with self.lock:
                self.switch(next(i for i,r in enumerate(entries) if r['label']==self.selector.value))
        @nav.on_click
        def _(event):
            delta=-1 if event.target.value=='上一条' else 1
            self.selector.value=entries[(self.current+delta)%len(entries)]['label']
        @self.timeline.on_update
        def _(event):
            if not self.updating:
                with self.lock:
                    if event.client_id is not None:self.play.value=False
                    self.seek(int(self.timeline.value))
        @self.waypoint.on_update
        def _(_event):
            if not self.updating:
                with self.lock:
                    self.play.value=False
                    k=self.reference.indices(self.valid_only)[int(self.waypoint.value)]
                    self.seek(int(self.reference.frames[k]))
        @step.on_click
        def _(event):
            with self.lock:
                self.play.value=False
                frames=self.reference.frames[self.reference.indices(self.valid_only)]
                target=(frames[frames<self.frame][-1] if (frames<self.frame).any() else frames[0]) if (
                    event.target.value=='上一节点') else (
                    frames[frames>self.frame][0] if (frames>self.frame).any() else frames[-1])
                self.seek(int(target))
        @self.mode.on_update
        def _(_event):
            with self.lock:self.switch(self.current,keep_frame=True)
        @self.ghosts.on_update
        def _(_event):
            with self.lock:
                if self.drawing is not None:
                    self.visibility()
                    self.drawing.update_from_mjdata(self.scene.data)
        @self.markers.on_update
        def _(_event):
            with self.lock:self.marker_handle.visible=self.markers.value
        @reset.on_click
        def _(_event):self.reset_camera()
        @server.on_client_connect
        def _(_event):self.reset_camera()
        self.switch(start)
        self.seek(args.frame)
        self.reset_camera()

    @property
    def valid_only(self):return self.mode.value==VALID

    def switch(self,index,keep_frame=False):
        self.play.value=False
        reference=PlanReference.load(Path(self.entries[index]['path']))
        if self.valid_only and not reference.valid.any():self.mode.value=RAW
        old_frame=self.frame if keep_frame else 0
        self.current=index;self.reference=reference
        with self.server.atomic():
            if self.scene is None:
                self.scene=ReferenceScene(reference,self.args.ghost_count,self.args.hand_alpha,self.args.ghost_alpha)
                self.drawing=self.scene_type.create(self.server,self.scene.model,1)
                self.drawing.camera_tracking_enabled=False
                self.drawing.site_groups_visible=[False]*6
                self.apply_hand_opacity()
            self.scene.set_reference(reference,self.valid_only)
            for handle in self.path_handles:handle.remove()
            self.path_handles=[]
            indices=np.unique(np.r_[np.arange(0,len(reference.control),self.args.stride),len(reference.control)-1])
            points=reference.control[indices]
            self.path_handles.append(self.server.scene.add_line_segments('/reference/path',
                np.stack((points[:-1],points[1:]),axis=1),colors=(40,160,255),line_width=3.))
            colors=np.where(reference.valid[:,None],np.array([45,220,100]),np.array([245,65,65]))
            self.marker_handle=self.server.scene.add_point_cloud('/reference/waypoints',
                points=reference.control[reference.frames],colors=colors.astype(np.uint8),point_size=.0035,
                visible=self.markers.value)
            self.path_handles.append(self.marker_handle)
            self.path_handles.append(self.server.scene.add_label('/reference/start','起点',position=reference.control[0]))
            self.path_handles.append(self.server.scene.add_label('/reference/end','终点',position=reference.control[-1]))
            self.pointer=self.server.scene.add_frame('/reference/current',show_axes=True,axes_length=.025,axes_radius=.001)
            self.path_handles.append(self.pointer)
            self.updating=True
            self.timeline.value=0;self.timeline.max=len(reference.poses)-1
            self.waypoint.value=0;self.waypoint.max=max(1,len(reference.indices(self.valid_only))-1)
            self.waypoint.disabled=len(reference.indices(self.valid_only))==1
            self.updating=False
            self.seek(old_frame)

    def apply_hand_opacity(self):
        # GLB vertex alpha is not reliably rendered as blending by browsers.
        # Reuse mjlab's body-local mesh conversion and explicit ghost opacity.
        from mjlab.viewer.viser.conversions import merge_geoms,group_geoms_by_visual_compat
        model=self.scene.model
        for key,old in list(self.drawing.mesh_handles_by_group.items()):
            body,group,sub=key
            name=model.body(body).name or ''
            if not name.startswith('hand_'):continue
            hand=int(name.split('/')[0].removeprefix('hand_'))
            ids=np.flatnonzero((model.geom_bodyid==body)&(model.geom_group==group)).tolist()
            mesh=merge_geoms(model,group_geoms_by_visual_compat(model,ids)[sub])
            old.remove()
            self.drawing.mesh_handles_by_group[key]=self.server.scene.add_batched_meshes_simple(
                f'/reference_hands/{name}/group{group}_{sub}',mesh.vertices,mesh.faces,
                batched_wxyzs=np.array([[1.,0.,0.,0.]]),batched_positions=np.zeros((1,3)),
                batched_colors=(255,170,30) if hand==0 else (50,155,255),
                opacity=self.args.hand_alpha if hand==0 else self.args.ghost_alpha,
                cast_shadow=False,receive_shadow=False)

    def visibility(self):
        for (body,group,_),handle in self.drawing.mesh_handles_by_group.items():
            name=self.scene.model.body(body).name or ''
            if name.startswith('hand_'):
                index=int(name.split('/')[0].removeprefix('hand_'))
                handle.visible=(index==0 or (self.ghosts.value and index<=len(self.scene.ghost_indices)))

    def seek(self,frame,playhead=None):
        self.frame=int(np.clip(frame,0,len(self.reference.poses)-1))
        self.playhead=float(self.frame) if playhead is None else playhead
        self.scene.seek(self.frame,self.valid_only)
        pose,q,left,right,nearest=self.reference.sample(self.frame,self.valid_only)
        with self.server.atomic():
            self.visibility()
            self.drawing.update_from_mjdata(self.scene.data)
            self.visibility()
            self.pointer.position=self.reference.control[self.frame]
            self.pointer.wxyz=pose[3:]
            self.updating=True
            self.timeline.value=self.frame;self.waypoint.value=nearest
            self.updating=False
            exact=np.flatnonzero(self.reference.frames==self.frame)
            state=('通过筛选' if self.reference.valid[exact[0]] else '**未通过筛选**') if len(exact) else '节点之间的 smoothstep 插值'
            first=int(self.reference.frames[self.reference.indices(self.valid_only)[0]])
            self.info.content=(f"**{self.entries[self.current]['label']}**\n\n"
                f"时间 **{self.frame*self.reference.dt:.2f} / {(len(self.reference.poses)-1)*self.reference.dt:.2f} s**"
                f"　帧 **{self.frame}/{len(self.reference.poses)-1}**\n\n"
                f"waypoint 有效 **{self.reference.valid.sum()}/{len(self.reference.frames)}**；当前：{state}\n\n"
                f"关节参考区间：原始节点 {left}→{right}（帧 {self.reference.frames[left]}→{self.reference.frames[right]}）"
                +(f"\n\n首个有效节点在第{first}帧；此前为边界保持。" if self.valid_only and first else '')
                +f"\n\n文件：`{self.reference.path.name}`")

    def reset_camera(self):
        if not hasattr(self,'reference'):return
        points=self.reference.control
        center=(points.min(0)+points.max(0))*.5
        span=max(.35,float(np.linalg.norm(np.ptp(points,axis=0))))
        position=center+span*np.array([1.1,-1.1,.7])
        self.server.initial_camera.position=position
        self.server.initial_camera.look_at=center
        self.server.initial_camera.up=(0.,0.,1.)
        for client in self.server.get_clients().values():
            # Viser translates look_at when position changes. Set position
            # first, then restore the intended fixed object look-at point.
            client.camera.position=position
            client.camera.look_at=center
            client.camera.up_direction=(0.,0.,1.)

    def tick(self,elapsed):
        with self.lock:
            if self.play.value:
                position=min(len(self.reference.poses)-1,self.playhead+elapsed*self.speed.value/self.reference.dt)
                self.seek(int(position),playhead=position)
                if position>=len(self.reference.poses)-1:self.play.value=False


def run(args):
    if (args.stride<1 or not 0<=args.ghost_count<=20 or not 0<args.hand_alpha<=1
            or not 0<args.ghost_alpha<=1 or args.duration_s<0):
        raise ValueError('Invalid stride, hand count, opacity, or duration')
    entries=catalog(args)
    start=0 if args.plans else next((i for i,r in enumerate(entries) if r['candidate_id']==args.candidate_id),None)
    if start is None:raise ValueError(f'No executable reference for candidate {args.candidate_id}')
    reference=PlanReference.load(Path(entries[start]['path']))
    scene=ReferenceScene(reference,args.ghost_count,args.hand_alpha,args.ghost_alpha)
    print('[PLAN-VIEWER]',json.dumps(scene.check()),flush=True)
    if args.snapshot:screenshot(scene,args.snapshot,args.frame,args.stride)
    if args.check_only:return
    del scene
    import viser
    server=viser.ViserServer(host=args.host,port=args.port,label='规划轨迹 / Hand waypoint')
    try:
        browser=WaypointBrowser(args,entries,start,server)
        print(f'[PLAN-VIEWER] http://{args.host}:{server.get_port()} — {len(entries)} trajectories; no dynamics',flush=True)
        started=previous=time.monotonic()
        while not args.duration_s or time.monotonic()-started<args.duration_s:
            now=time.monotonic();browser.tick(now-previous);previous=now
            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
