"""Film an entire recorded rollout with GPIS meshes built from each touch prefix."""
import argparse
from bisect import bisect_right
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageColor, ImageDraw
from scipy.spatial.transform import Rotation
import torch
import trimesh

from ..acitive_02.experiment import digest, read_json, write_json
from ..i04_geometric_traversal.pad_contact_visual import camera_from_direction, marker
from . import film_support as film
from .film_support import prefix_mesh, read_model, reference_samples


FPS = 25
SIZE = (1280, 720)
VIEW = (646, 422)
TINT = '#6cd7bd'
FINGER_NAMES = ('食指', '中指', '无名指', '拇指')


def view_axes(eye):
    eye = np.asarray(eye)/np.linalg.norm(eye)
    across = np.cross([0.,0.,1.],eye)
    across /= np.linalg.norm(across)
    return np.stack([across,np.cross(eye,across),eye],axis=1)


class HandView:
    def __init__(self, model, vertices, initial):
        self.model = model
        self.data = mujoco.MjData(model)
        self.center = (vertices.min(0)+vertices.max(0))*.5
        self.reference = vertices[::max(1,len(vertices)//1500)]
        radial = initial[:3]-self.center
        radial /= np.linalg.norm(radial)
        side = np.cross([0.,0.,1.],radial)
        if np.linalg.norm(side)<.05:
            side = np.cross([1.,0.,0.],radial)
        self.local_side = Rotation.from_quat(initial[[4,5,6,3]]).inv().apply(side/np.linalg.norm(side))
        bottle = model.geom('bottle_visual')
        bottle.rgba[:] = (.98,.62,.16,.49)
        self.pads = []
        for name in film.TIP_NAMES:
            model.geom(name+'_connector_collision').rgba[:] = model.geom(name).rgba
            model.geom(name).rgba[3] = 0.
            self.pads.append(model.geom(name+'_pad_collision').id)
        self.ids = np.flatnonzero((model.geom_bodyid!=model.geom_bodyid[bottle.id]) &
                                  (model.geom_group==0) & (model.geom_rgba[:,3]>0))
        box = model.geom_aabb[self.ids]
        signs = np.array([[i,j,k] for i in (-1,1) for j in (-1,1) for k in (-1,1)])
        self.corners = box[:,None,:3]+box[:,None,3:]*signs
        model.vis.headlight.ambient[:] = .49
        model.vis.headlight.diffuse[:] = .83
        model.vis.global_.offwidth,model.vis.global_.offheight = VIEW
        self.camera = mujoco.MjvCamera()
        self.options = mujoco.MjvOption()
        self.options.geomgroup[3:] = 0
        self.options.sitegroup[:] = 0

    def render(self,renderer,qpos,live,qualified,points):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_camlight(self.model,self.data)
        colors = [(0.26,.99,.66,1.) if ok else (1.,.43,.25,1.) if hit else (.36,.63,.74,1.)
                  for hit,ok in zip(live,qualified)]
        for pad,color in zip(self.pads,colors):
            self.model.geom_rgba[pad] = color
        axes = self.data.geom_xmat[self.ids].reshape(-1,3,3)
        hand = self.data.geom_xpos[self.ids,None,:]+np.einsum('gij,gkj->gki',axes,self.corners)
        radial = qpos[:3]-self.center
        radial /= max(np.linalg.norm(radial),1e-8)
        side = Rotation.from_quat(qpos[[4,5,6,3]]).apply(self.local_side)
        side -= np.dot(side,radial)*radial
        basis = view_axes(.92*radial+side/max(np.linalg.norm(side),.01))
        cloud = np.vstack((hand.reshape(-1,3),self.reference))@basis
        center = (cloud.max(0)+cloud.min(0))*.5
        x,y,z = (cloud-center).T
        half_angle = np.tan(np.deg2rad(self.model.vis.global_.fovy*.5))
        distance = np.max(z+1.13*np.maximum(np.abs(y),np.abs(x)*VIEW[1]/VIEW[0])/half_angle)
        camera_from_direction(self.camera,basis[:,2],basis@center,max(.28,distance))
        renderer.update_scene(self.data,camera=self.camera,scene_option=self.options)
        for i in np.flatnonzero(live):
            marker(renderer.scene,points[i],.0045,colors[i])
        return Image.fromarray(renderer.render())


def surface_view(vertices,mesh,eye):
    width,height = 270,300
    scale_aa = 2
    result = Image.new('RGB',(width*scale_aa,height*scale_aa),film.PANEL)
    if mesh is None:
        return result.resize((width,height),Image.Resampling.LANCZOS)
    basis = view_axes(eye)
    bounds = vertices @ basis
    ctr = (bounds.max(0)+bounds.min(0))*.5
    scale = min((width-24)/np.ptp(bounds[:,0]),(height-30)/np.ptp(bounds[:,1]))
    view = mesh.vertices@basis-ctr
    xy = np.array([width/2,height/2])+view[:,:2]*np.array([scale,-scale])
    faces = mesh.faces
    order = np.argsort(view[faces,2].mean(1),kind='stable')
    normals = mesh.face_normals@basis
    normals = np.where(normals[:,2,None]<0,-normals,normals)
    shade = .45+.55*np.maximum(0.,normals@np.array([-.35,.57,.74]))
    rgb = np.uint8(np.clip(shade[:,None]*np.array(ImageColor.getrgb(TINT)),0,255)).tolist()
    triangles = np.rint(xy*scale_aa).astype(int)[faces].reshape(-1,6).tolist()
    pen = ImageDraw.Draw(result)
    for i in order:
        pen.polygon(triangles[i],fill=tuple(rgb[i]))
    return result.resize((width,height),Image.Resampling.LANCZOS)


def build_checkpoints(run,folder,reference,qualified,points,normals,interval,end):
    ticks,fingers = np.nonzero(qualified)
    points = points[ticks,fingers]
    normals = normals[ticks,fingers]
    selected = np.unique(np.floor(points/.0045).astype(int),axis=0,return_index=True)[1]
    first = np.sort(ticks[selected])
    rows=[]
    for t in sorted({*range(0,end+1,interval),end}):
        k = t*100
        count = int(np.searchsorted(ticks,k,side='right'))
        if t==end:
            path = run/f'signed-{t}s.ply'
            score = read_json(run/'portable-scores.json')['f5']
        else:
            path = folder/f'signed-{t}s.ply'
            score = prefix_mesh(points[:count], normals[:count], reference, path)
            if score is None:
                path = None
        rows.append(dict(s=t, samples=count, cells=int(np.searchsorted(first,k,side='right')),
                         f5=score,mesh=str(path.resolve()) if path else None,
                         mesh_sha256=digest(path) if path else None))
    return rows


def draw(tick,end,speed,title,physical,views,snap,coverage,live,qualified):
    canvas = Image.new('RGB',SIZE,film.BG)
    pen = ImageDraw.Draw(canvas)
    label=film.label
    label(pen,(32,35),title,29,bold=True)
    label(pen,(1248,35),f'{tick/100:05.1f} / {end:.0f} s',25,mono=True,anchor='rm')
    label(pen,(32,82),'手部探索',21,bold=True)
    label(pen,(700,82),'触觉 GPIS 重建',21,bold=True)
    label(pen,(1246,82),f"截至 {snap['s']} s 的合格触点",16,film.MUTED,anchor='rm')
    canvas.paste(physical,(28,110))
    pen.rounded_rectangle((690,110,1252,532),radius=10,fill=film.PANEL)
    for j,(name,img) in enumerate(zip(('正面','背面'),views)):
        x=702+274*j
        canvas.paste(img,(x,162))
        label(pen,(x+135,480),name,16,film.MUTED,anchor='mm')
    if not snap['mesh']:
        label(pen,(972,298),'积累触点中',19,film.MUTED,anchor='mm')
    for i,name in enumerate(FINGER_NAMES):
        x=42+i*67
        fill='#1b6951' if qualified[i] else '#8a492e' if live[i] else '#28343e'
        pen.rounded_rectangle((x,123,x+60,151),radius=6,fill=fill)
        label(pen,(x+30,137),name,15,'#ffffff',anchor='mm')
    pen.line((29,555,1251,555),fill=film.HAIR)
    label(pen,(32,590),'25 mm 实触覆盖率',18,film.MUTED)
    label(pen,(32,631),f'{coverage:.2%}',39,TINT,bold=True,mono=True)
    label(pen,(453,590),f"重建 F@5 mm  ·  {snap['s']} s",18,film.MUTED)
    label(pen,(453,631),f"{snap['f5']:.2%}" if snap['f5'] is not None else '—',38,bold=True,mono=True)
    label(pen,(895,590),'当前指腹接触',18,film.MUTED)
    label(pen,(895,630),f'合格 {qualified.sum()} 指   受力 {live.sum()} 指',25,bold=True)
    pen.rectangle((30,671,1250,675),fill=film.HAIR)
    if tick: pen.rectangle((30,671,30+round(1220*tick/(100*end)),675),fill=TINT)
    label(pen,(30,696),'绿色计入覆盖　 橙色有压力但未计入　 灰色无压力',15,film.MUTED)
    label(pen,(1248,696),f'{speed}× 回放',16,film.MUTED,anchor='rm')
    return canvas


def render(run,case,output,title,speed,interval):
    receipt = read_json(run/'run.json')
    trace_hash = digest(run/'trace.npz')
    if trace_hash!=receipt['trace_sha256'] or digest(case/'model.mjb')!=receipt['model_sha256']:
        raise ValueError('Replay and physical model must match the scored rollout')
    reference_file=case/'surface.obj'
    if digest(reference_file)!=receipt['offline_mesh_sha256']:
        raise ValueError('Reference differs from the scored surface')
    scores=read_json(run/'portable-scores.json')
    with np.load(run/'trace.npz') as tr, np.load(run/'coverage.npz') as curve:
        q,p,n=(tr[k] for k in ('qpos','points_m','normals_outward'))
        active,loads=tr['pad_valid'],tr['region_load_n']
        live=(tr['pad_force_n']>0)&(np.linalg.norm(n,axis=2)>.5)
        coverage=curve['c25']
        qualified=active&(loads.max(1)<8)[:,None]
        assert len(coverage)==len(q) and np.isclose(coverage[-1],scores['coverage25'])
    end=(len(q)-1)//100
    assert 100*end==len(q)-1
    output.parent.mkdir(parents=True,exist_ok=True)
    folder=output.parent/(output.stem+'-checkpoints')
    folder.mkdir(exist_ok=True)
    rows=build_checkpoints(run,folder,reference_samples(reference_file),qualified,p,n,interval,end)
    vertices=trimesh.load_mesh(reference_file,process=False).vertices
    images=[]
    for row in rows:
        m=trimesh.load_mesh(row['mesh'],process=False) if row['mesh'] else None
        images.append([surface_view(vertices,m,eye) for eye in ([.56,-1.,.62],[-.56,1.,.45])])
    times=[row['s']*100 for row in rows]
    model=read_model(case/'model.mjb')
    scene=HandView(model,vertices,q[0])
    ticks=np.rint(np.linspace(0,100*end,round(end*FPS/speed)+1)).astype(int)
    with mujoco.Renderer(model,height=VIEW[1],width=VIEW[0]) as renderer, imageio.get_writer(
            output,fps=FPS,codec='libx264',quality=8,macro_block_size=1,pixelformat='yuv420p',
            output_params=['-movflags','+faststart','-threads','2']) as writer:
        for i,t in enumerate(ticks):
            b=bisect_right(times,t)-1
            physical=scene.render(renderer,q[t],live[t],qualified[t],p[t])
            frame=draw(t,end,speed,title,physical,images[b],rows[b],coverage[t],live[t],qualified[t])
            writer.append_data(np.asarray(frame))
            if i==len(ticks)//2:
                frame.save(output.with_suffix('.webp'),quality=91)
            if i and i%(FPS*10)==0:
                print(output.stem,t/100,'s rendered',flush=True)
        for _ in range(FPS-1):
            writer.append_data(np.asarray(frame))
    write_json(output.with_suffix('.json'),dict(title=title,trace_sha256=trace_hash,
        source_run=str(run.resolve()),speed=speed,simulation_s=end,frame_count=len(ticks)+FPS-1,
        fps=FPS,final_still_s=(FPS-1)/FPS,sha256=digest(output),checkpoints=rows,final_metrics=scores))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('run','case','output'):
        ap.add_argument('--'+key,type=Path,required=True)
    ap.add_argument('--title',required=True)
    ap.add_argument('--speed',type=int,default=4,choices=(1,2,4,8))
    ap.add_argument('--interval',type=int,default=15,choices=(10,15,30,60))
    args=ap.parse_args()
    torch.set_num_threads(1)
    render(args.run,args.case,args.output,args.title,args.speed,args.interval)
