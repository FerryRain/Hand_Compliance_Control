"""Freeze dual/triple-finger WM interaction trials from separated static roots.

This prepares experiments, not physics admission. Object motion is independent
of contact. Failed-contact lanes are retained after a finite fixed loading ramp.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(base, roots, pool, output, tag, frames=1500):
    template=json.loads(base.read_text());data=json.loads(roots.read_text())
    manifest=pool/'object_manifest.json'
    objects={r['object_id']:r for r in json.loads(manifest.read_text())['objects']}
    scripts=Path(__file__).resolve().parent
    source_names=set(template['source_hashes'])|{
        'collect_contact_wm_rooted_interaction.py','prepare_contact_wm_rooted_protocols.py',
        'contact_wm_initialization.py','contact_wm_nominal.py','contact_wm_motion_profiles.py'}
    source_names-={'collect_contact_wm_planned.py','collect_contact_wm_planned_exploration.py'}
    frozen_sources={n:sha(scripts/n) for n in sorted(source_names)}
    groups=sorted({(r['object_id'],tuple(r['active_fingers'])) for r in data['roots']})
    output.mkdir(parents=True,exist_ok=True);queue=[]
    for object_id,active in groups:
        if len(active) not in (2,3):raise ValueError('This protocol is dual/triple only')
        selected=[r for r in data['roots'] if r['object_id']==object_id and tuple(r['active_fingers'])==active]
        if objects[object_id]['split']!='train':raise ValueError('Do not tune on held-out objects')
        if any(r['mesh_sha256']!=objects[object_id]['mesh_sha256'] for r in selected):
            raise ValueError('Root/mesh mismatch')
        if any(abs(r['collection_home_object_pose'][2]-r['object_pose'][2]-.1)>1e-8 for r in selected):
            raise ValueError('Root does not have the specified 100mm loading program')
        p={k:copy.deepcopy(template[k]) for k in ('collector','controller','collision_provenance','backend')}
        a=p['collector']
        for k in ('virtual_reference','virtual_depths_m','episode_seeds','force_targets_n'):
            a.pop(k,None)
        a.update(root=str(pool),initial_roots=str(roots),objects=[object_id],active_fingers=list(active),
                 fingers=list(active),behaviors=['slide','rotate','slide_release'],frames=frames,
                 dp_checkpoint=None,perturbation_rad=0.,branches=False,branch_diagnostic_only=False,
                 fixed_reference=True,object_surface_follow=False,surface_follow=False,
                 motion_owner='object',slide_amplitude_m=.05,motion_period_s=10.,motion_rate=1.,
                 motion_rate_range=[2.,4.],randomize_motion=True,max_object_command_speed=.3,
                 run_tag=tag)
        p['controller'].update(force_target_n=5.,outward_travel_m=.08,normal_step_m=.0008,
                               approach_distance_m=.1,approach_frames=1000,home_hold_frames=100,
                               incremental_force_compliance=True,force_command_consistency=False,
                               side_mobility=0.)
        p.update(id=f'{tag}_{object_id}_active'+''.join(map(str,active)),object=object_id,
                 source_hashes=frozen_sources,input_hashes={str(roots):sha(roots),str(manifest):sha(manifest)},
                 scope='multi-finger WM interaction, not offline-planned whole-hand teacher',
                 initialization_policy='finite_completed_loading_not_contact_success',
                 expected_lanes=len(selected)*len(a['behaviors']),expected_transitions_20hz=len(selected)*len(a['behaviors'])*(frames//5),
                 training_admissible=False,timeout_s=1200,
                 physics_review=dict(contact_rate_cutoff=None,retain_contact_loss=True,retain_overload=True,
                     no_hidden_state_reset=True,normal_support_tolerance_m=1e-4,
                     finite=True,exact_five_command_alignment=True,
                     separate_self_contact_and_chatter_review=True),
                 motion='10cm nominal slide span, 2–4x episode clock, randomized direction and amplitude; centre rotation or prescribed release',
                 initial_geometry_evidence=str(roots),counterfactual_branching_validated=False)
        path=output/(p['id']+'_protocol.json')
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(p,indent=2))
        queue.append(dict(protocol=str(path),object=object_id,active_fingers=list(active),
                          lanes=p['expected_lanes'],transitions_20hz=p['expected_transitions_20hz']))
    result=dict(schema='rooted_interaction_queue_v1',status='prepared_not_collected',jobs=queue,
                expected_transitions_20hz=sum(x['transitions_20hz'] for x in queue),
                teacher_admissible=False,training_admissible=False)
    path=output/(tag+'_queue.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','roots','pool','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--tag',required=True);p.add_argument('--frames',type=int,default=1500)
    a=p.parse_args()
    print(json.dumps(prepare(a.base.resolve(),a.roots.resolve(),a.pool.resolve(),a.output.resolve(),a.tag,a.frames),indent=2))
