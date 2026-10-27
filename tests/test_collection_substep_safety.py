"""The recorded-window safety stop must retain short physical force peaks."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from collect_trajectories import install_collection_substep_safety


def test_peak_retained_even_when_post_step_force_is_safe():
    forces=[torch.tensor([120.,30.]),torch.tensor([70.,25.])]
    executed=[]
    observer=SimpleNamespace(safety_peak_force_n=None)
    def physics():executed.append(1)
    def read(env,unused):observer.safety_peak_force_n=forces.pop(0)
    observer.read=read
    env=SimpleNamespace(num_envs=2,device='cpu',sim=SimpleNamespace(step=physics,sense=lambda:None))
    state=install_collection_substep_safety(env,observer)
    state['enabled']=True
    env.sim.step();env.sim.step()
    assert len(executed)==2
    assert torch.equal(state['peak'],torch.tensor([120.,30.]))
    assert torch.equal(state['peak']>100,torch.tensor([True,False]))
    assert not bool((observer.safety_peak_force_n>100).any())


def test_settle_ignored_and_peaks_reset_between_control_ticks():
    observer=SimpleNamespace(safety_peak_force_n=torch.tensor([976.]))
    reads=[]
    def read(env,unused):reads.append(1)
    observer.read=read
    env=SimpleNamespace(num_envs=1,device='cpu',sim=SimpleNamespace(step=lambda:None,sense=lambda:None))
    state=install_collection_substep_safety(env,observer)
    env.sim.step()
    assert not reads and float(state['peak'][0])==0
    state['enabled']=True
    observer.safety_peak_force_n=torch.tensor([90.])
    env.sim.step()
    assert float(state['peak'][0])==90
    state['peak'].zero_()
    observer.safety_peak_force_n=torch.tensor([20.])
    env.sim.step()
    assert float(state['peak'][0])==20
