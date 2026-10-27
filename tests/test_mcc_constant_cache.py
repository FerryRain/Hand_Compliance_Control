"""Compare cached MCC constants with the independent pre-optimization code.

Cover both contact-state-machine modes and a live configuration-value change.
No CUDA required; end-to-end CUDA parity is audited separately.
"""
import importlib.util
import sys
from pathlib import Path
from dataclasses import replace
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'mcc_finger_compliance_control/scripts'))
from batched_fullhand_mcc import BatchedFullHandMCC
from deploy_b_gpu_serial_parity import mcc_config


def compare_mode(enabled):
    path=ROOT/'mcc_finger_compliance_control/backups/control_graph_opt_20261012/batched_fullhand_mcc.py'
    spec=importlib.util.spec_from_file_location('legacy_mcc_constant_test',path)
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    cfg=replace(mcc_config(),enable_loss_state_machine=enabled,back_contact_x_limit_m=(1.,)*4)
    left=legacy.BatchedFullHandMCC(4,'cpu',cfg);right=BatchedFullHandMCC(4,'cpu',cfg)
    generator=torch.Generator().manual_seed(81)
    q=left.grasp.reshape(1,16).expand(4,-1).clone()
    pose=torch.zeros(4,7);pose[:,3]=1
    normal=torch.zeros(4,4,3);normal[:,:,2]=1
    for frame in range(24):
        if frame==12:
            cfg=replace(cfg,overforce_hard_ratio=2.5,thumb_overforce_hard_ratio=3.,
                thumb_max_inward_offset=.0015,thumb_nominal_joint_correction_limit=.03,
                force_servo_weak_contact_step=.00013)
            left.config=cfg;right.config=cfg
        points=left.fk(q)
        nominal=q+.004*torch.randn(q.shape,generator=generator)
        force=normal*torch.tensor([0.,5.,28.,75.])[:,None,None]
        found=torch.rand(4,4,generator=generator)>.3
        values=dict(q_live=q,palm_pose_world=pose,force_world=force,found=found,
            surface_points_world=left.fk(nominal),surface_normals_world=normal,
            nominal_posture_q=nominal,contact_points_world=points,
            collision_normals_outward_world=normal,contact_observed=found)
        a,da=left.update(**values);b,db=right.update(**values)
        assert torch.equal(a,b),(enabled,frame,float((a-b).abs().max()))
        for name in da:assert torch.equal(da[name],db[name]),(enabled,frame,name)
        for name in ('anchor','force_filtered','force_offset','hold','contact_phase','previous_command'):
            assert torch.equal(getattr(left,name),getattr(right,name)),(enabled,frame,name)
        q=q+.2*(a-q)


def test_cache_preserves_state_machine_control():compare_mode(True)


def test_cache_preserves_search_control():compare_mode(False)
