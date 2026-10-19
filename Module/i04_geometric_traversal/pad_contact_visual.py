"""Native FK diagnostic: registered belly face versus actual saved contact.

Pure offline display. Never steps physics, changes the recorded model, grants
coverage or substitutes the drawn candidate patch for a measured contact.
The isolated views deliberately hide object/other fingers to expose the pad.
"""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Module.fr3_leap.model import compile_model_xml, ARM_JOINT_NAMES, HAND_JOINT_NAMES
from Module.e05_physics.scene import FINGERS


def marker(scene, point, radius, color):
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([radius,0,0]),
                      np.asarray(point),np.eye(3).ravel(),np.asarray(color))
    scene.ngeom+=1


def connector(scene, start, end, color, *, arrow=False, width=.00045):
    kind=mujoco.mjtGeom.mjGEOM_ARROW if arrow else mujoco.mjtGeom.mjGEOM_CAPSULE
    geom=scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom,kind,np.ones(3),np.zeros(3),np.eye(3).ravel(),np.asarray(color))
    mujoco.mjv_connector(geom,kind,width,np.asarray(start),np.asarray(end))
    scene.ngeom+=1


def camera_from_direction(camera, direction, lookat, distance):
    direction=np.asarray(direction)/np.linalg.norm(direction)
    # MuJoCo azimuth/elevation specify the forward vector, not eye-lookat.
    forward=-direction
    camera.azimuth=float(np.degrees(np.arctan2(forward[1],forward[0])))
    camera.elevation=float(np.degrees(np.arcsin(forward[2])))
    camera.lookat[:]=lookat;camera.distance=distance


def render(case, time_s, finger, output, native_contacts=None):
    case,output=Path(case),Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    raw=(case/'trace.npz').read_bytes()
    source_sha=sha256(raw).hexdigest()
    with np.load(BytesIO(raw),allow_pickle=False) as trace:
        times=trace['time_s'];row=int(np.argmin(abs(times-time_s)))
        q=trace['q_rad'][row];active=bool(trace['active'][row,finger])
        point=trace['contact_positions_world_m'][row,finger].copy()
        normal=trace['normals_world'][row,finger].copy()
        force=float(trace['forces_n'][row,finger]);actual_time=float(times[row])
    assert active,'A measured active contact is required for this diagnostic'
    del raw
    model_bytes=(case/'model.xml').read_bytes();model=compile_model_xml(model_bytes.decode('utf-8'))
    data=mujoco.MjData(model)
    joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in (*ARM_JOINT_NAMES,*HAND_JOINT_NAMES)]
    data.qpos[model.jnt_qposadr[joints]]=q;mujoco.mj_forward(model,data)
    spec=FINGERS[finger]
    pad=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,spec.proxy_geom.replace('e05_','fr3_'))
    body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,spec.tip_body)
    center=data.geom_xpos[pad].copy();frame=data.geom_xmat[pad].reshape(3,3).copy();size=model.geom_size[pad].copy()
    local=frame.T@(point-center);rho=float(np.linalg.norm(local[:2]/size[:2]))
    towards=-normal/np.linalg.norm(normal);angle=float(np.degrees(np.arccos(np.clip(frame[:,2]@towards,-1,1))))
    witnesses=[];contact_source=None
    if native_contacts is not None:
        native_contacts=Path(native_contacts)
        with np.load(native_contacts,allow_pickle=False) as contact:
            rows=np.flatnonzero(np.isclose(contact['time_s'],actual_time,rtol=0,atol=1e-6)&
                               (contact['finger']==finger)&(contact['force_n']>0))
            assert len(rows),'No native contact evidence matches the selected state'
            for k in rows:
                witnesses.append({'pad_point_local_m':contact['pad_point_local_m'][k].tolist(),
                                  'midpoint_local_m':contact['midpoint_local_m'][k].tolist(),
                                  'force_n':float(contact['force_n'][k]),
                                  'rho':float(contact['projected_patch_radius'][k]),
                                  'angle_deg':float(contact['angle_deg'][k])})
        contact_source={'path':str(native_contacts.resolve()),'sha256':sha256(native_contacts.read_bytes()).hexdigest()}
    displayed_points=[center+frame@np.asarray(w['pad_point_local_m']) for w in witnesses] if witnesses else [point]
    displayed_local=np.asarray(max(witnesses,key=lambda w:w['force_n'])['pad_point_local_m']) if witnesses else local
    displayed_rho=float(np.linalg.norm(displayed_local[:2]/size[:2]))
    descendants={body}
    for candidate in range(model.nbody):
        if int(model.body_parentid[candidate]) in descendants:descendants.add(candidate)
    keep={g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants}
    base_rgba=model.geom_rgba.copy();base_matid=model.geom_matid.copy()
    options=mujoco.MjvOption();options.geomgroup[3:]=0;options.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    renderer=mujoco.Renderer(model,height=560,width=720)
    panels=[]
    for panel in range(3):
        model.geom_rgba[:]=base_rgba;model.geom_matid[:]=base_matid
        if panel:
            for geom in range(model.ngeom):
                model.geom_matid[geom]=-1
                model.geom_rgba[geom]=[.76,.79,.82,.30 if panel==1 else 1] if geom in keep else [0,0,0,0]
            model.geom_rgba[pad]=[.18,.35,.44,.35 if panel==1 else .65]
            direction=(frame[:,1]+.3*frame[:,2]+.15*frame[:,0]) if panel==1 else (frame[:,2]+.05*frame[:,0])
            camera_from_direction(camera,direction,center,.115)
        else:
            camera.azimuth=130.;camera.elevation=-30.;camera.distance=.48
            camera.lookat[:]=center
        renderer.update_scene(data,camera=camera,scene_option=options)
        if panel:
            # The gold region is a candidate geometrical definition, not a
            # measured pressure footprint: local+Z cap with rho<=0.8.
            for radial in (.2,.4,.6,.8):
                theta=np.linspace(0,2*np.pi,65)
                local_curve=np.column_stack([size[0]*radial*np.cos(theta),size[1]*radial*np.sin(theta),
                                              np.full_like(theta,size[2]*np.sqrt(1-radial**2)+.00012)])
                world=local_curve@frame.T+center
                for a,b in zip(world[:-1],world[1:]):connector(renderer.scene,a,b,(1.,.70,.08,1),width=.00017)
            connector(renderer.scene,center,center+.020*frame[:,2],(.15,.95,.25,1),arrow=True,width=.0007)
        for displayed_point in displayed_points:
            marker(renderer.scene,displayed_point,.0010 if panel else .0022,(1.,.12,.65,1))
        panels.append(Image.fromarray(renderer.render()))
    renderer.close()
    image=Image.new('RGB',(2160,780),'#101a2d');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    title=ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc',28)
    for i,panel in enumerate(panels):image.paste(panel,(720*i,100))
    titles=('真实现场：实心原模型','局部侧视：半透明辅助看法向','固定腹面正视：定位实际接触')
    for i,text in enumerate(titles):draw.text((720*i+18,56),text,font=font,fill='#d2e8ff')
    draw.text((18,12),f'旧v3.8 实测接触诊断 | {spec.name} | t={actual_time:.3f}s | F={force:.3f}N',font=title,fill='white')
    point_label='原生pad侧接触点（非碰撞中点）' if witnesses else '保存的接触中心（非逐接触点）'
    draw.text((18,670),f'绿箭头=固定腹面 +Z；与实测接触方向(对象法向反向)夹角 {angle:.2f}°。粉点={point_label}。',font=font,fill='white')
    draw.text((18,710),f'粉点局部坐标(mm): [{displayed_local[0]*1000:.3f}, {displayed_local[1]*1000:.3f}, {displayed_local[2]*1000:.3f}]；rho={displayed_rho:.4f}。金网=候选腹面区(+Z且rho≤0.8)，不是实测覆盖。',font=font,fill='#ffdf96')
    draw.text((18,748),'右两视图只隐藏遮挡以检查注册几何；所有关节q来自同一保存状态，未推进物理，也未把代理接触改写为腹面成功。',font=font,fill='#cbd5e8')
    image.save(output)
    audit=json.loads((case/'audit.json').read_bytes()) if (case/'audit.json').exists() else {}
    metadata={'source_case':str(case.resolve()),'trace_sha256':source_sha,'audit_trace_hash_matches':audit.get('trace_sha256')==source_sha,
        'model_xml_sha256':sha256(model_bytes).hexdigest(),'time_s':actual_time,'finger':spec.name,'force_n':force,
        'saved_contact_centroid_local_mm':(local*1000).tolist(),'centroid_rho':rho,'belly_to_contact_direction_angle_deg':angle,
        'pad_half_axes_mm':(size*1000).tolist(),'belly_axis':'pad local+Z, fixed distal body-X',
        'contact_is_saved_weighted_sensor_centroid_not_all_native_manifold_points':True,
        'displayed_contact_points':'native pad-side witness' if witnesses else 'saved weighted centroid',
        'native_contact_evidence':contact_source,'native_contacts':witnesses,
        'candidate_patch':{'front_z_positive':True,'rho_max':.8,'definition_only_not_accepted_performance_criterion':True},
        'view_scope':['Opaque original scene','Isolated actual distal mesh+proxy,transparent side','Isolated actual distal mesh+proxy,opaque bellyface'],
        'no_mj_step':True,'no_physics_model_file_edit':True,'no_contact_or_coverage_credit':True,
        'renderer':'Native MuJoCo FK from actual saved q; user geoms are diagnostic overlays',
        'png_sha256':sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_bytes(json.dumps(metadata,ensure_ascii=False,indent=2).encode('utf-8'))
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('case',type=Path);parser.add_argument('--time',type=float,required=True)
    parser.add_argument('--finger',type=int,choices=range(4),required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--native-contacts',type=Path)
    args=parser.parse_args();print(json.dumps(render(args.case,args.time,args.finger,args.output,args.native_contacts),ensure_ascii=False))
