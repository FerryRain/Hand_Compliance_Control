"""Viewer statistics, contact markers and orderly render-thread shutdown."""
import sys,threading,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import mujoco
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from visualize_dp0_gpu import ContactDisplay,DrivenNativeViewer
from mjlab.viewer.native import NativeMujocoViewer
from mjlab.viewer.viewer_config import ViewerConfig
from unittest.mock import MagicMock


def test_readonly_multienv_viewer_adds_native_geometries(monkeypatch):
    """Exercise setup and actual mjv_addGeoms without needing an X server."""
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".05"/></body>
    </worldbody></mujoco>''')
    env=MagicMock()
    env.cfg.viewer=ViewerConfig()
    env.unwrapped.num_envs=2
    env.unwrapped.sim.mj_model=model
    env.unwrapped.reward_manager.get_active_iterable_terms.return_value=[]
    handle=SimpleNamespace(user_scn=mujoco.MjvScene(model,maxgeom=100),opt=mujoco.MjvOption())
    monkeypatch.setattr(mujoco.viewer,'launch_passive',lambda *a,**kw:handle)
    viewer=NativeMujocoViewer(env,lambda obs:None,enable_perturbations=False)
    monkeypatch.setattr(viewer,'_setup_camera',lambda:None)
    viewer.setup()
    def copy_state(data,sim_data,lane):
        data.qpos[:]=model.qpos0
        data.qpos[0]=lane
    synced=[]
    monkeypatch.setattr(viewer,'_sync_env_state_to_mjdata',copy_state)
    monkeypatch.setattr(viewer,'_sync_model_fields',lambda sim,lane:synced.append(lane))
    viewer._render_other_env_geoms(handle,env.unwrapped.sim,None)
    assert handle.user_scn.ngeom>0
    assert not viewer.enable_perturbations
    assert viewer.pert.active==0
    assert synced==[1,0]  # Main environment's visual model is restored.


def test_closed_window_still_waits_for_context_cleanup():
    cleaned=threading.Event()
    thread=threading.Thread(target=lambda:(time.sleep(.03),cleaned.set()));thread.start()
    class Handle:
        def is_running(self):return False
        def close(self):raise AssertionError('Already closing window must not be closed twice')
    viewer=NativeMujocoViewer.__new__(NativeMujocoViewer)
    viewer.viewer=Handle();viewer._render_threads=(thread,);viewer.verbosity=0
    viewer.close();assert cleaned.is_set();assert not thread.is_alive()
    viewer.close()  # Idempotent after the rendering thread has ended.


def test_running_window_requests_exit_once_and_joins():
    requested=threading.Event();closed=[]
    thread=threading.Thread(target=lambda:requested.wait(1));thread.start()
    class Handle:
        def is_running(self):return True
        def close(self):closed.append(1);requested.set()
    viewer=NativeMujocoViewer.__new__(NativeMujocoViewer)
    viewer.viewer=Handle();viewer._render_threads=(thread,);viewer.verbosity=0
    viewer.close();viewer.close();assert len(closed)==1;assert not thread.is_alive()


def test_render_and_control_rates_have_distinct_counters():
    viewer=DrivenNativeViewer.__new__(DrivenNativeViewer)
    viewer._stats_last_time=time.perf_counter()-1
    viewer._stats_frames=30;viewer._stats_steps=100
    viewer._update_stats()
    assert 25<viewer._fps<31;assert 85<viewer._sps<101
    viewer.env=SimpleNamespace(unwrapped=SimpleNamespace(step_dt=.01))
    assert .85<viewer.actual_realtime<1.01


def test_correct_tip_connector_and_loss_markers():
    display=ContactDisplay()
    display.chain=SimpleNamespace(mcc=SimpleNamespace(fk=lambda q:torch.zeros(1,4,3)))
    display.q=display.nominal=torch.zeros(1,16)
    display.pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
    display.normal=torch.tensor([[[1.,0.,0.]]*4])
    display.contacts=dict(pad_found=torch.tensor([[1,1,0,0]],dtype=torch.bool),
        pad_points=torch.zeros(1,4,3),pad_force=torch.tensor([[[1.,0.,0.],[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]]]),
        connector_found=torch.tensor([[0,0,1,0]],dtype=torch.bool),connector_points=torch.zeros(1,4,3))
    class Drawing:
        def __init__(self):self.colors=[];self.arrows=0
        def add_sphere(self,p,radius,color):self.colors.append(color)
        def add_arrow(self,*args,**kwargs):self.arrows+=1
    drawing=Drawing();display.draw(drawing)
    assert (.15,1.,.2,1.) in drawing.colors
    assert (1.,.65,.05,1.) in drawing.colors
    assert drawing.colors.count((1.,.05,.05,1.))==2
    _,strict,found,_=display.labels(1500)
    assert strict=='1/0/0/0';assert found=='1/1/1/0'
