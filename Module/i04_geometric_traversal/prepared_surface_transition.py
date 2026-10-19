"""Preserve the surface-first policy, then try explicit safe hand preparation.

Only measured coverage selects the remaining goals. Geometry never updates the
ledger. This is a B-cell fallback and returns the existing transition contract.
"""
from __future__ import annotations

from collections import Counter
from time import perf_counter
import numpy as np

from .contracts import WholeHandTarget,WristTarget
from .physics import joint_state,pose,tip_positions,set_joint_state,rotation
from .prepared_hand_transition import plan_prepared_hand_transition,_timing
from .full_surface import JointPath
from .surface_slide import plan_surface_transition


def _witness_target(planner,witness):
    """A cached joint vector cannot silently stand in for a different target."""
    try:
        q=np.asarray(witness['q'],dtype=float)
        target=WholeHandTarget(WristTarget(witness['pose']),witness['tips'])
    except (KeyError,TypeError,ValueError):return None,'INVALID_WITNESS_FIELDS'
    if q.shape!=(23,) or not np.all(np.isfinite(q)) or np.any(q<planner.lower) or np.any(q>planner.upper):
        return None,'WITNESS_JOINT_BOUNDS'
    reason=planner.collision_reason(q)
    if reason:return None,reason
    set_joint_state(planner.h,planner.scratch,q);actual=pose(planner.h,planner.scratch)
    errors=(np.linalg.norm(actual[:3]-target.wrist.pose_world[:3]),
        (rotation(actual[3:])*rotation(target.wrist.pose_world[3:]).inv()).magnitude(),
        np.max(np.linalg.norm(tip_positions(planner.h,planner.scratch)-target.fingertip_positions_world_m,axis=1)))
    if any(value>bound for value,bound in zip(errors,(.002,.04,.003))):return None,'WITNESS_TARGET_MISMATCH'
    return target,None


def plan_surface_transition_with_prepared(sp,oracle,data,coverage,attempts,outward,witnesses=None,*,
        maximum_goals=8,variants=(0,1,2,3),retreat_distances=(.004,.008,.012,.025,.04),short_route_s=25.):
    """Original planner first; finite B-only fallback if it returns ``None``.

    Return ``(segments, normal, metadata)`` or ``None`` exactly as before.
    ``sp.last_prepared_transition_diagnostics`` records finite failed searches.
    The fallback tries variant 0 regardless of previous attempt counters.
    A composite TRANSFER keeps the complete retreat/preparation/transfer in the
    intentional-air controller mode and avoids unverified segment-root jumps.
    """
    before_vertices=np.asarray(coverage.visited_vertices).copy()
    before_goals=np.asarray(coverage.visited_goals).copy()
    started=perf_counter();diagnostics=dict(fallback_invoked=False,candidates=[],finite_search=True)
    sp.last_prepared_transition_diagnostics=diagnostics
    try:
        original=plan_surface_transition(sp,oracle,data,coverage,attempts,outward,witnesses)
        if original is not None:
            diagnostics['policy']='ORIGINAL_SURFACE_TRANSITION';return original
        if sp.cell!='B-MCC':
            diagnostics['reason']='B_GEOMETRY_FALLBACK_NOT_EXPOSED_TO_A';return None
        if maximum_goals<1 or not np.isfinite(short_route_s) or short_route_s<=0:
            raise ValueError('Positive goal count and short-route duration required')
        variants=tuple(dict.fromkeys((0,)+tuple(int(v) for v in variants)))
        if any(v<0 or v>7 for v in variants):raise ValueError('Pose variants must lie in 0..7')
        distances=tuple(float(d) for d in retreat_distances)
        if not distances or any(not np.isfinite(d) or not 0<d<=.08 for d in distances):
            raise ValueError('Explicit retreat candidates must lie in (0,80] mm')
        outward=np.asarray(outward,dtype=float)
        if outward.shape!=(3,) or not np.all(np.isfinite(outward)) or np.linalg.norm(outward)<1e-10:
            raise ValueError('Finite nonzero outward three-vector required')
        outward=outward/np.linalg.norm(outward)
        diagnostics.update(fallback_invoked=True,variants=list(variants),maximum_goals=int(maximum_goals),
            retreat_candidate_distances_m=list(distances),short_route_s=float(short_route_s))
        h,graph,planner=sp.h,sp.graph,sp.planner
        required=graph.required_vertices[~coverage.visited_goals]
        remaining=required if len(required) else np.flatnonzero(~coverage.visited_vertices)
        if not len(remaining):diagnostics['reason']='NO_UNCOVERED_ORIGINAL_VERTEX';return None
        # Failures affect ordering only; original vertices remain in the set.
        tips=tip_positions(h,data)-h.object_position_m
        metric=np.min(np.linalg.norm(graph.vertices_m[remaining,None,:]-tips[None,:,:],axis=2),axis=1)
        ordered=remaining[np.argsort(metric+.02*np.minimum(np.asarray(attempts)[remaining],10))][:maximum_goals]
        root=joint_state(h,data)
        cache={} if witnesses is None else {int(v):i for i,v in enumerate(witnesses['vertex'])}
        feasible=[]
        def package(item):
            duration,path,vertex,target,row=item
            normal=(rotation(target.wrist.pose_world[3:])*oracle.initial_rotation.inv()).apply(oracle.initial_normal)
            normal/=max(float(np.linalg.norm(normal)),1e-12)
            diagnostics.update(reason=None,selected_vertex=vertex,selected_duration_s=duration,
                successful_routes=len(feasible),latency_s=perf_counter()-started)
            meta=dict(vertex=vertex,duration_s=duration,pose_policy='PREPARED_HAND_GEOMETRIC_FALLBACK',
                retreat_offset_m=None,hand_preparation_policy=row['policy'],pose_source=row['source'],
                pose_variant=row.get('variant'),fallback_candidate_count=len(diagnostics['candidates']),
                retreat_candidate_distances_m=list(distances),finite_candidate_search=True,
                internal_segment_phases=row.get('segment_phases',['TRANSFER']),
                physical_contact_or_coverage_claim=False)
            return [('TRANSFER',path)],normal,meta
        for value in ordered:
            vertex=int(value);attempts[vertex]+=1
            goals=[]
            if vertex in cache and 'q' in witnesses:
                i=cache[vertex]
                goals.append(('FROZEN_CACHE',None,dict(q=witnesses['q'][i],pose=witnesses['pose'][i],tips=witnesses['tips'][i]),{}))
            # Synthesis is lazy so a short safe cached route need not wait for
            # every IK branch. Do not seed the list with the failed counter.
            sources=goals+[('ORACLE',variant,None,None) for variant in variants]
            for source,variant,witness,failures in sources:
                row=dict(vertex=vertex,source=source,variant=variant);diagnostics['candidates'].append(row)
                if witness is None:
                    witness,failures=oracle.find_pose(vertex,root,extended=True,variant=variant)
                    row['pose_failures']=failures
                if witness is None:row['reason']='NO_WITNESS_IN_FINITE_POSE_CANDIDATES';continue
                target,reason=_witness_target(planner,witness)
                if reason:row['reason']=reason;continue
                path,failure=planner.plan(data,target,allow_rrt=True)
                row['ordinary_plan_failure']=failure
                if path is not None:
                    path=JointPath(path.waypoints,_timing(planner,path.waypoints),path.root_pose,path.goal_pose,path.latency_s)
                    row.update(policy='ALTERNATE_VALIDATED_POSE',segment_phases=['TRANSFER']*len(path.segment_durations))
                else:
                    path,detail=plan_prepared_hand_transition(planner,data,target,witness['q'],outward=outward,distances=distances)
                    row.update(policy='OUTSIDE_HAND_PREPARATION',reason=detail.get('reason'),
                        segment_phases=detail.get('segment_phases'),retreat_attempts=len(detail.get('attempts',[])),
                        failure_reason_counts=dict(Counter(item.get('reason') or str(item.get('transfer_failure') or item.get('retreat_failure'))
                            for item in detail.get('attempts',[]) if item.get('reason') or item.get('transfer_failure') or item.get('retreat_failure'))))
                if path is None:continue
                row['duration_s']=path.duration_s
                feasible.append((path.duration_s,path,vertex,target,row))
                if path.duration_s<=short_route_s:return package(min(feasible,key=lambda item:item[0]))
        if feasible:return package(min(feasible,key=lambda item:item[0]))
        diagnostics.update(reason='NO_SAFE_PREPARED_ROUTE_IN_FINITE_CANDIDATES',latency_s=perf_counter()-started)
        return None
    finally:
        # This is an invariant check, never a rollback that could conceal
        # geometry incorrectly awarding coverage in any called component.
        if not np.array_equal(coverage.visited_vertices,before_vertices) or not np.array_equal(coverage.visited_goals,before_goals):
            raise AssertionError('Geometric transition planning modified measured coverage')
