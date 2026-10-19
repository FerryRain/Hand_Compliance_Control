"""B-cell geometric fallback: withdraw, prepare fingers safely, then transfer.

This helper never advances physics or edits the supplied native state. A known
pose witness provides a geometric hand configuration, not a contact answer.
Every preparation/transfer edge retains the caller's strict safety planner.
"""
from __future__ import annotations

from time import perf_counter
import mujoco
import numpy as np

from .contracts import WholeHandTarget,WristTarget
from .full_surface import JointPath
from .physics import joint_state,pose,tip_positions,set_joint_state,position_jacobian,rotation


def _timing(planner,waypoints):
    """Rate-check the final clipped quintic reference with a 1% time reserve."""
    times=[];h=planner.h;cfg=planner.config
    for a,b in zip(waypoints[:-1],waypoints[1:]):
        linear=angular=joint=0.
        for u in np.linspace(0,1,401):
            s=u**3*(10-15*u+6*u*u);ds=30*u*u*(1-u)*(1-u)
            raw=a+s*(b-a);velocity=ds*(b-a)
            velocity[(raw<planner.lower)|(raw>planner.upper)]=0.
            set_joint_state(h,planner.scratch,np.clip(raw,planner.lower,planner.upper))
            jp,jr=position_jacobian(h,planner.scratch,h.palm_site_id)
            linear=max(linear,float(np.linalg.norm(jp@velocity)))
            angular=max(angular,float(np.linalg.norm(jr@velocity)))
            joint=max(joint,float(np.max(abs(velocity))))
        times.append(1.01*max(.1,linear/cfg.wrist_speed_m_s,angular/cfg.wrist_rotation_speed_rad_s,
                             joint/cfg.trajectory_joint_speed_rad_s))
    return np.asarray(times)


def plan_prepared_hand_transition(planner,data,target,goal_q,*,outward,distances=(.004,.008,.012,.025,.04)):
    """Return ``(JointPath | None, diagnostics)`` with explicit segment phases.

    The input target and q must describe the same validated Cartesian goal.
    Only the hand part of q is prepared outside the object; the normal planner
    then re-solves the target. A strict direct known-q edge is a last fallback.
    """
    started=perf_counter();h=planner.h;root=joint_state(h,data);current=pose(h,data)
    goal_q=np.asarray(goal_q,dtype=float);outward=np.asarray(outward,dtype=float)
    if goal_q.shape!=(23,) or not np.all(np.isfinite(goal_q)) or not isinstance(target,WholeHandTarget):
        raise ValueError('B-cell full geometric goal and finite 23-joint witness required')
    if outward.shape!=(3,) or not np.all(np.isfinite(outward)) or np.linalg.norm(outward)<1e-10:
        raise ValueError('A finite nonzero three-vector outward direction is required')
    outward=outward/np.linalg.norm(outward)
    if (np.any(goal_q<planner.lower) or np.any(goal_q>planner.upper) or planner.collision_reason(goal_q)):
        return None,dict(reason='INVALID_GEOMETRIC_WITNESS')
    set_joint_state(h,planner.scratch,goal_q);actual=pose(h,planner.scratch)
    residuals=[float(np.linalg.norm(actual[:3]-target.wrist.pose_world[:3])),
        float((rotation(actual[3:])*rotation(target.wrist.pose_world[3:]).inv()).magnitude()),
        float(np.max(np.linalg.norm(tip_positions(h,planner.scratch)-target.fingertip_positions_world_m,axis=1)))]
    if any(a>b for a,b in zip(residuals,(.002,.04,.003))):
        return None,dict(reason='WITNESS_TARGET_MISMATCH',residuals=residuals)
    distances=tuple(float(d) for d in distances)
    if not distances or any(not np.isfinite(d) or not 0<d<=.08 for d in distances):
        raise ValueError('Explicit retreat candidates must lie in (0,80] mm')
    rows=[];directions=[('measured_outward',outward)]+[(f'axis{i}_{sign:+g}',np.eye(3)[i]*sign) for i in (2,1,0) for sign in (1.,-1.)]
    for distance in distances:
        candidates=[]
        for label,direction in directions:
            delta=float(distance)*direction
            retreat_target=WholeHandTarget(WristTarget(np.r_[current[:3]+delta,current[3:]]),tip_positions(h,data)+delta)
            retreat,failure=planner.plan(data,retreat_target)
            row=dict(distance_m=float(distance),direction=label,retreat_failure=failure)
            rows.append(row)
            if retreat is None:continue
            end=retreat.waypoints[-1];prepared=end.copy();prepared[7:]=goal_q[7:]
            if planner.collision_reason(prepared) or not planner.edge_free(end,prepared):
                row['reason']='HAND_PREPARATION_EDGE_BLOCKED';continue
            scratch=mujoco.MjData(h.model);set_joint_state(h,scratch,prepared)
            transfer,failure=planner.plan(scratch,target,allow_rrt=True)
            row['transfer_failure']=failure
            if transfer is None:
                if not planner.edge_free(prepared,goal_q):continue
                transfer_q=np.vstack([prepared,goal_q]);row['transfer_policy']='DIRECT_KNOWN_GOAL_Q'
            else:
                transfer_q=transfer.waypoints;row['transfer_policy']='NORMAL_CARTESIAN_GOAL_PLAN'
            whole=np.concatenate([retreat.waypoints,prepared[None],transfer_q[1:]])
            phases=['LIFT']*(len(retreat.waypoints)-1)+['HAND_PREPARATION']+['TRANSFER']*(len(transfer_q)-1)
            dense_failure=None
            for a,b in zip(whole[:-1],whole[1:]):
                for u in np.linspace(0,1,max(2,int(np.ceil(np.max(abs(b-a))/.005))+1)):
                    q=a+u*(b-a)
                    if planner.collision_reason(q) or planner.collision_reason(np.clip(q,planner.lower,planner.upper)):
                        dense_failure='DENSE_RAW_OR_COMMAND_GEOMETRY';break
                if dense_failure:break
            if dense_failure:row['reason']=dense_failure;continue
            durations=_timing(planner,whole)
            candidate=JointPath(whole,durations,current.copy(),target.wrist.pose_world.copy(),perf_counter()-started)
            row['total_duration_s']=candidate.duration_s;candidates.append((candidate,phases,row))
        # Prefer the shortest fully checked route at the smallest retreat that
        # admits one. Failed hand paths are never replaced with endpoint tests.
        if candidates:
            path,phases,chosen=min(candidates,key=lambda item:item[0].duration_s)
            return path,dict(reason=None,policy='OUTSIDE_HAND_PREPARATION',segment_phases=phases,
                chosen=chosen,attempts=rows,retreat_candidate_distances_m=distances,
                witness_residuals=residuals,latency_s=perf_counter()-started)
    return None,dict(reason='NO_PREPARED_HAND_CONNECTION_IN_FINITE_CANDIDATES',attempts=rows,
        retreat_candidate_distances_m=distances,witness_residuals=residuals,latency_s=perf_counter()-started)
