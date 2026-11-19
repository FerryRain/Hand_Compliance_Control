"""Kinematic visualization of recorded deployment states; never rerun physics."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from mjlab.scene import Scene
from replay_inverted import replay_env_cfg
from object_catalog import load_object_config
from collection_force_regulation import CollectionForceObserver
from surface_mcc_finger import HAND_JOINT_NAMES
p=argparse.ArgumentParser()
p.add_argument('--result',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--preview-only',action='store_true')
p.add_argument('--title')
a=p.parse_args()
result_path=a.result.resolve();result=json.loads(result_path.read_text());task=json.loads(result_path.with_suffix('.request.json').read_text());n=result['lanes'][0]['frames'];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
with np.load(result['trace']) as source:
 z={k:source[k][:n].copy() for k in ['q','strict','found','force','self_force','measured_palm_pose_world','measured_object_pose_world','reset_reference_frame']}
f=result_path
from surface_mcc_finger import FullHandMCCFingerConfig,FullHandMCCFingerController
ctl=FullHandMCCFingerController(FullHandMCCFingerConfig(**result['source_config']))
margin=np.minimum(z['q'][:,0]-ctl.lower,ctl.upper-z['q'][:,0]).min(-1)
label=a.title or ('Pear' if 'pear' in f.stem else 'Bleach cleanser')+' | expanded single-finger policy'
planned=task['rows'][0]['frames']
complete=bool(result['lanes'][0]['complete'])
status='Full reference completed' if complete else 'Run ended early'

cfg=replay_env_cfg(num_envs=1,hand_stiffness=35.,hand_damping=.2,hand_effort_limit=.8,object_config=load_object_config(task['rows'][0]['object_id']))
CollectionForceObserver(controller_scope='pad',capture_collision=True).configure(cfg)
m=Scene(cfg.scene,device='cpu').compile();d=mujoco.MjData(m)
qids=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+name)] for name in HAND_JOINT_NAMES]
palm=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/palm_lower');free=next(m.jnt_qposadr[i] for i in range(m.njnt) if m.jnt_bodyid[i]==palm and m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
obj=next(i for i in range(m.nbody) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,i) or '').startswith('target/') and m.body_mocapid[i]>=0);mid=m.body_mocapid[obj]
for i in range(m.ngeom):
 name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or ''
 if name.startswith('target/'):
  if '_collision_' in name:m.geom_group[i]=5
  else:
   m.geom_rgba[i,3]=.30
   if m.geom_matid[i]>=0:m.mat_rgba[m.geom_matid[i],3]=.30
m.vis.global_.offwidth=960;m.vis.global_.offheight=540;m.vis.headlight.ambient[:]=.5
renderer=mujoco.Renderer(m,height=410,width=480);opt=mujoco.MjvOption();opt.geomgroup[:]=[1,1,0,0,0,0];opt.sitegroup[:]=0
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE
points=np.concatenate([z['measured_object_pose_world'][:,0,:3],z['measured_palm_pose_world'][:,0,:3]])
fixed=(points.min(0)+points.max(0))/2
fixed_distance=max(.8,float(np.linalg.norm(np.ptp(points,axis=0)))*1.5+.28)
indices=np.arange(0,n,5)
# Keep the last measured state, with the normal 20 fps playback duration.
indices[-1]=n-1
if a.preview_only:indices=np.array([0,n//2,n-1])
segments=[('Full evaluation / 1x simulation speed',indices,1)]
writer=None if a.preview_only else imageio_ffmpeg.write_frames(str(out/'deployment_review.mp4'),(960,540),fps=20,codec='libx264',quality=7,output_params=['-movflags','+faststart'])
if writer is not None:writer.send(None)
try:
 for title,indices,speed in segments:
  for j,t in enumerate(indices):
   d.qpos[qids]=z['q'][t,0];d.qpos[free:free+7]=z['measured_palm_pose_world'][t,0];d.mocap_pos[mid]=z['measured_object_pose_world'][t,0,:3];d.mocap_quat[mid]=z['measured_object_pose_world'][t,0,3:]
   mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
   image=Image.new('RGB',(960,540),(18,22,30));draw=ImageDraw.Draw(image)
   for side in range(2):
    cam.lookat[:]=fixed if side==0 else d.xpos[palm]+d.xmat[palm].reshape(3,3)@np.array([.03,0,-.035]);cam.distance=fixed_distance if side==0 else .45;cam.azimuth=45 if side==0 else 95;cam.elevation=-20
    renderer.update_scene(d,camera=cam,scene_option=opt);image.paste(Image.fromarray(renderer.render()),(480*side,55))
   draw.text((10,7),f'{label} | measured replay',font=font,fill='white')
   cursor=int(z['reset_reference_frame'][t,0])+1
   draw.text((10,30),f't={t/100:.2f}s | path {cursor}/{planned} | 1x sim time | fixed / palm-follow',font=font,fill='white')
   strict=z['strict'][t,0].astype(int);found=z['found'][t,0].astype(int)
   draw.text((10,470),f'Strict I/M/R/T: {strict.tolist()}   Found: {found.tolist()}',font=font,fill=(170,240,190))
   draw.text((10,500),f'Object {z["force"][t,0].max():.1f} N | Self {float(np.max(z["self_force"][t,0])):.1f} N | joint margin {margin[t]:.3f} rad | {status}',font=font,fill=(255,190,150))
   if writer is not None:writer.send(np.asarray(image))
   if j in [0,len(indices)//2]:image.save(out/(title.split()[0]+f'_{t}.png'))
   if j%250==0:print(title,j,len(indices),flush=True)
finally:
 if writer is not None:writer.close()
 renderer.close()

if not a.preview_only:
 import hashlib
 def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 (out/'metadata.json').write_text(json.dumps(dict(source=result['trace'],source_sha256=sha(result['trace']),result=str(result_path),checkpoint=result['single_finger_policy'],source_frames=n,video_frames=len(indices),fps=20,playback_speed=1,final_source_frame=int(indices[-1]),planned_reference_frames=planned,last_reference_cursor=int(z['reset_reference_frame'][-1,0]),opacity=.30,object_id=task['rows'][0]['object_id'],views=['fixed','palm-follow'],replay='Recorded measured q and palm/object poses, kinematics only; forces from stored physical substeps. No policy or physics rerun.',complete_path=complete),indent=2))
