import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'mcc_finger_compliance_control/scripts'))
import torch
from paired_contact_projection import project_halfspace_box,measured_step_bounds


def test_bounded_projection_matches_analytic_optimum():
    x=torch.zeros(1,4);j=torch.tensor([[1.,2.,0.,0.]])
    low=torch.full_like(x,-1.);high=torch.tensor([[.1,.6,1.,1.]])
    y,margin=project_halfspace_box(x,x,j,torch.tensor([1.]),low,high)
    torch.testing.assert_close(y,torch.tensor([[.1,.45,0.,0.]]))
    assert margin.item()>=-1e-6


def test_outward_palm_motion_permits_compensating_joint_motion():
    q=torch.zeros(1,4);candidate=torch.tensor([[-.5,0.,0.,0.]])
    j=torch.tensor([[1.,0.,0.,0.]])
    # Palm moves outward by1; an inward joint step of-.5 still leaves+.5.
    y,margin=project_halfspace_box(candidate,q,j,torch.tensor([-1.]),
                                  torch.full_like(q,-2.),torch.full_like(q,2.))
    torch.testing.assert_close(y,candidate)
    assert margin.item()>0


def test_infeasible_constraint_respects_bounds_and_reports_deficit():
    q=torch.zeros(1,4);j=torch.tensor([[1.,2.,0.,0.]])
    high=torch.tensor([[.1,.6,1.,1.]])
    y,margin=project_halfspace_box(q,q,j,torch.tensor([2.]),-high,high)
    assert bool((y<=high).all())
    torch.testing.assert_close(margin,torch.tensor([-.7]))


def test_stale_goal_does_not_allow_large_singular_escape():
    q=torch.zeros(1,4)
    low=torch.zeros_like(q);high=torch.full_like(q,.3)
    low,high,empty=measured_step_bounds(low,high,q,-torch.ones_like(q),torch.ones_like(q),.02)
    y,margin=project_halfspace_box(q,q,torch.tensor([[.001,0.,0.,0.]]),torch.tensor([.0002]),low,high)
    assert float(y.abs().max())<=.020001
    assert float(margin)<0
    assert not bool(empty.any())


def test_incompatible_old_goal_withdraws_to_measured_joint_band():
    q=torch.zeros(1,4)
    low=torch.full_like(q,.15);high=torch.full_like(q,.25)
    low,high,empty=measured_step_bounds(low,high,q,-torch.ones_like(q),torch.ones_like(q),.02)
    assert bool(empty.all())
    torch.testing.assert_close(low,torch.full_like(q,-.02))
    torch.testing.assert_close(high,torch.full_like(q,.02))


def test_disabled_measured_band_preserves_original_bounds():
    low=torch.randn(1,4);high=low+1;q=torch.zeros_like(low)
    a,b,empty=measured_step_bounds(low,high,q,-torch.ones_like(q),torch.ones_like(q),None)
    assert a is low and b is high and empty is None


def test_final_command_bound_applies_without_an_engaged_contact():
    from types import SimpleNamespace
    from batched_fullhand_mcc import BatchedFullHandMCC
    from deploy_b_gpu_serial_parity import mcc_config
    from paired_contact_projection import PairedContactProjection
    mcc=BatchedFullHandMCC(1,'cpu',mcc_config())
    q=mcc._clamp_q(((mcc.lower+mcc.upper)/2)[None]).clone()
    point,_,_,_,_=mcc.fk.contact_kinematics(q)
    x=dict(physical_contact_point=point,physical_normal_inward=torch.zeros_like(point),
           physical_normal_force=torch.zeros(1,4),physical_normal_found=torch.zeros(1,4,dtype=torch.bool))
    projection=PairedContactProjection(SimpleNamespace(mcc=mcc,device='cpu'),SimpleNamespace(latest=x),False,.04,True)
    pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
    previous=mcc._clamp_q(q+.3)
    result,diag=projection.project(q,previous,pose,torch.zeros(1,6),previous)
    assert not bool(diag['paired_engaged'].any())
    assert bool(diag['bounded_command_changed'].any())
    assert float((result-q).abs().max())<=.040001


def test_physical_projection_is_invariant_to_world_rotation_and_translation():
    from types import SimpleNamespace
    from batched_fullhand_mcc import BatchedFullHandMCC
    from deploy_b_gpu_serial_parity import mcc_config
    from paired_contact_projection import PairedContactProjection
    mcc=BatchedFullHandMCC(2,'cpu',mcc_config())
    q=mcc._clamp_q(((mcc.lower+mcc.upper)/2)[None].expand(2,-1)).clone()
    point,_,_,_,jac=mcc.fk.contact_kinematics(q)
    normal=jac[...,0];normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    x=dict(physical_contact_point=point,physical_normal_inward=-normal,
           physical_normal_force=torch.full((2,4),60.),physical_normal_found=torch.ones(2,4,dtype=torch.bool))
    projection=PairedContactProjection(SimpleNamespace(mcc=mcc,device='cpu'),SimpleNamespace(latest=x),False)
    pose=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]]).expand(2,-1)
    a,diag=projection.project(q,q,pose,torch.zeros(2,6))
    assert bool(diag['paired_engaged'].all())
    assert float((a-q).abs().max())>0
    rotation=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    translation=torch.tensor([.3,-.2,.5])
    x['physical_contact_point']=point@rotation.T+translation
    x['physical_normal_inward']=(-normal)@rotation.T
    shifted=torch.tensor([[.3,-.2,.5,2**-.5,0.,0.,2**-.5]]).expand(2,-1)
    b,_=projection.project(q,q,shifted,torch.zeros(2,6))
    torch.testing.assert_close(a,b,rtol=0,atol=2e-6)
    assert float((a-q).abs().max())<=mcc.config.action_rate_limit+1e-6
