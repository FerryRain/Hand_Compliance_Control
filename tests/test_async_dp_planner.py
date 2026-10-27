"""Snapshot, stream ownership, delayed delivery and execution-buffer contracts."""
import sys
from pathlib import Path
from types import SimpleNamespace
import threading
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
from async_dp_planner import AsyncDPPlanner
from batched_b_control_chain import BatchedDPChunkScheduler


def fake_chain(block=None):
    def sample(batch,global_cond,generator):
        if block is not None:block.wait(timeout=5)
        return torch.randn(batch,16,16,generator=generator)+global_cond[:,:1,None]*.001
    diffusion=SimpleNamespace(_prepare_global_conditioning=lambda x:x['observation.state'][:,-1],conditional_sample=sample)
    scheduler=BatchedDPChunkScheduler(1,torch.ones(16))
    return SimpleNamespace(device=torch.device('cpu'),batch=1,scheduler=scheduler,
        runtime=SimpleNamespace(pred_horizon=16,robot_state_dim=84,policy=SimpleNamespace(diffusion=diffusion),generator=torch.Generator().manual_seed(4)),
        action_std=torch.ones(16),action_mean=torch.zeros(16))


def inputs():
    return torch.full((1,16),.2),torch.zeros(1,16,96),torch.zeros(1,16,96)


def test_inline_keeps_the_snapshot_anchor():
    chain=fake_chain();planner=AsyncDPPlanner(chain,inline=True)
    q,history,normalized=inputs();planner.submit(75,q,history,normalized)
    q.add_(10);history.add_(3);normalized.add_(4)
    plan=planner.poll(75)
    assert torch.equal(plan.anchor,torch.full((1,16),.2))
    assert not plan.history.any() and not plan.normalized.any()
    assert torch.allclose(plan.chunk-plan.prediction,torch.full((1,16,16),.2),atol=2e-7)
    assert not planner.busy
    planner.close();planner.close()


def test_thread_does_not_block_control_or_queue_old_requests():
    release=threading.Event();chain=fake_chain(release)
    planner=AsyncDPPlanner(chain,minimum_latency_frames=8)
    try:
        assert planner.submit(75,*inputs())
        assert planner.poll(76) is None
        assert not planner.submit(85,*inputs())
        assert planner.busy_requests==1
        release.set()
        planner.pending.result(timeout=5)
        assert planner.poll(82) is None
        result=planner.poll(83)
        assert result.request_frame==75
        assert planner.results[0]['age_frames']==8
    finally:
        release.set();planner.close()


def test_expired_result_is_not_installed():
    planner=AsyncDPPlanner(fake_chain(),inline=True,max_age_frames=20)
    planner.submit(75,*inputs())
    assert planner.poll(96) is None
    assert planner.expired==1 and not planner.busy
    planner.close()


def test_extended_buffer_keeps_the_original_executed_prefix():
    left=fake_chain().scheduler;right=fake_chain().scheduler
    for i in range(30):
        value=torch.full((1,16),i*.0004)
        left.observe(value);right.observe(value)
    chunk=torch.linspace(.012,.10,16)[None,:,None].expand(1,16,16)
    old_drop=left.install(chunk);new_drop=right.install(chunk,execution_frames=30)
    assert torch.equal(old_drop,new_drop)
    assert torch.equal(left.path,right.path[:,:10])
    assert right.path.shape==(1,30,16)
    for _ in range(30):assert torch.isfinite(right.next_command()).all()
    try:right.next_command()
    except RuntimeError as exc:assert 'expired' in str(exc)
    else:raise AssertionError('Exhausted buffers must not silently hold or wrap')
