"""Check the completed native whole-hand result and its review media."""
import argparse
from hashlib import sha256
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote,urlsplit
import imageio.v2 as imageio
import imageio_ffmpeg


class References(HTMLParser):
    def __init__(self):super().__init__();self.references=set()
    def handle_starttag(self,tag,attrs):
        for name,value in attrs:
            if name in ('href','src') and value:self.references.add(value)


def check(directory):
    case=Path(directory).resolve();root=Path(__file__).resolve().parents[2]
    read=lambda name:json.loads((case/name).read_text(encoding='utf-8'))
    digest=lambda path:sha256(path.read_bytes()).hexdigest()
    manifest=read('manifest.json');summary=read('summary.json');audit=read('independent_native_audit/audit.json')
    metrics=read('whole_hand_metrics.json')
    assert audit['status']=='PASSED_NATIVE_EVIDENCE_AUDIT'
    assert audit['full_surface_complete'] and summary['full_surface_complete'] and metrics['full_surface_complete']
    assert (audit['covered_vertices'],audit['covered_goals'])==(8192,388)
    assert audit['continuous_replay'] and audit['physical_initializations']==1
    resume=manifest.get('resume');assert resume==summary.get('resume')==audit.get('resume')
    if resume:
        assert not audit['fresh_single_version'] and not summary['fresh_single_version'] and summary['coverage_inherited']
        assert not resume['physical_state_reset'] and not resume['independent_trajectories_merged']
        parent=case/'parent_provenance'
        pm=json.loads((parent/'manifest.json').read_text());ps=json.loads((parent/'summary.json').read_text())
        pa=json.loads((parent/'audit.json').read_text())
        for name in ('manifest','summary','audit'):
            assert digest(parent/(name+'.json'))==resume['parent_'+name+'_sha256']
        assert pa['status']=='PASSED_NATIVE_EVIDENCE_AUDIT' and pa['continuous_replay']
        assert digest(parent/'initial.npz')==digest(case/'initial.npz')==pa['source_files']['initial.npz']
        assert digest(parent/'final_state.npz')==pa['source_files']['final_state.npz']
        assert ps['trace_chunks']==summary['trace_chunks'][:resume['prefix_chunks']]
        assert ps['native_states']==resume['prefix_native_states'] and abs(ps['executed_s']-resume['join_time_s'])<1e-9
        if audit.get('reference_guard_scope')=='whole_trajectory':
            assert audit['current_version_reference_guard_steps']==audit['native_states']
        else:
            assert pa['fresh_single_version']
            assert pa['current_version_reference_guard_steps']+audit['current_version_reference_guard_steps']==audit['native_states']
        for name,expected in pm['source_hashes'].items():assert digest(parent/'source_snapshot'/name)==expected
        ancestor=resume.get('parent_resume');child_summary=ps
        while ancestor:
            parent=parent/'parent_provenance'
            for name in ('manifest','summary','audit'):
                assert digest(parent/(name+'.json'))==ancestor['parent_'+name+'_sha256']
            pm=json.loads((parent/'manifest.json').read_text());ps=json.loads((parent/'summary.json').read_text())
            pa=json.loads((parent/'audit.json').read_text())
            assert pa['status']=='PASSED_NATIVE_EVIDENCE_AUDIT' and pa['continuous_replay']
            assert ps['trace_chunks']==child_summary['trace_chunks'][:ancestor['prefix_chunks']]
            assert ps['native_states']==ancestor['prefix_native_states']
            for name in ('manifest.json','summary.json','initial.npz','final_state.npz'):
                assert digest(parent/name)==pa['source_files'][name]
            for name,expected in pm['source_hashes'].items():assert digest(parent/'source_snapshot'/name)==expected
            child_summary=ps;ancestor=ancestor.get('parent_resume')
    else:assert audit['fresh_single_version'] and summary['fresh_single_version'] and not summary['coverage_inherited']
    assert metrics['all_four_control_flags'] and audit['all_applied_commands_original_geometry_checked']
    for name,expected in manifest['source_hashes'].items():
        assert digest(root/name)==digest(case/'source_snapshot'/name)==expected,name
    for name,expected in audit['source_files'].items():assert digest(case/name)==expected,name
    for chunk in summary['trace_chunks']:assert digest(case/chunk['file'])==chunk['sha256']
    media=[]
    for meta_path in sorted(case.glob('whole_hand_*x_render.json')):
        meta=json.loads(meta_path.read_text(encoding='utf-8'));path=case/meta['video']
        assert meta['whole_object_complete'] and meta['actual_saved_q_only'] and meta['no_physics_step']
        assert meta['source_manifest_sha256']==audit['source_files']['manifest.json']
        assert digest(path)==meta['video_sha256']
        frames,duration=imageio_ffmpeg.count_frames_and_secs(str(path));reader=imageio.get_reader(path)
        encoded=reader.get_meta_data();assert frames==meta['frames'] and encoded['fps']==meta['fps']==20
        assert encoded['size']==(2400,700) and abs(duration-frames/20)<.002
        if meta['full_time_axis']:
            assert abs(duration*meta['speed']-summary['executed_s'])<=2.*meta['speed']/20
            assert meta['stills'][0]['time_s']==0.
            assert abs(meta['stills'][-1]['time_s']-(summary['executed_s']-.002))<1e-8
        else:assert meta['speed']==1. and meta['segments']
        chosen=sorted(set([frames//4,frames//2,3*frames//4,frames-1]))
        for frame in chosen:imageio.imwrite(case/f'{path.stem}_encoded_{frame:05d}.png',reader.get_data(frame))
        reader.close();media.append(dict(video=path.name,sha256=meta['video_sha256'],frames=frames,
            fps=20,encoded_duration_s=duration,full_time_axis=meta['full_time_axis'],speed=meta['speed'],
            available_regions=[r['region'] for r in meta['regions'] if r['available']],review_frames=chosen))
    assert any(m['full_time_axis'] for m in media) and any(not m['full_time_axis'] for m in media)
    parser=References();parser.feed((case/'review.html').read_text(encoding='utf-8'))
    for ref in parser.references:
        url=urlsplit(ref);assert not url.scheme and (case/unquote(url.path)).is_file(),ref
    helper=Path(__file__).with_name('whole_hand_surface_review.py')
    assert helper.read_bytes()==(case/'visualization_source/whole_hand_surface_review.py').read_bytes()
    result=dict(status='PASSED',scope=('Continuously replayed development continuation' if resume else 'Fresh single-version')+' whole-object cooperative MCC, fixed Mustard simulation',
        controller_version_boundary=resume,
        native_states=audit['native_states'],covered_vertices=8192,covered_goals=388,
        current_and_frozen_runtime_sources_match=len(manifest['source_hashes']),
        native_chunks_match=len(summary['trace_chunks']),html_references=sorted(parser.references),media=media,
        visualization_source_sha256=digest(helper))
    (case/'delivery_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();check(a.case)
