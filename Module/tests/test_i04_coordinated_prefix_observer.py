"""Meaningful evidence-boundary tests; no runtime controller is exercised."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Module.i04_geometric_traversal.coordinated_prefix_observer import (
    observe, phase_gap_attribution, read_completed_prefix, recovery_evidence,
)


def trace(end=2.):
    t=np.arange(int(round(end*100))+1)/100.
    n=len(t)
    positions=np.zeros((n,4,3)); positions[:,:,0]=t[:,None]*.002
    normals=np.zeros_like(positions); normals[:,:,2]=1.
    return dict(time_s=t,active=np.ones((n,4),dtype=bool),forces_n=np.ones((n,4)),
        q_rad=np.zeros((n,23)),phase=np.full(n,'SLIDE',dtype='U24'),
        coverage=t/10,covered_goals=np.floor(t*10).astype(int),
        contact_positions_world_m=positions,tip_positions_world_m=positions.copy(),normals_world=normals)


def save_case(directory,record):
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(directory/'trace.npz',**record)
    np.savez(directory/'initial.npz',qpos=np.zeros(23),qvel=np.zeros(23),ctrl=np.zeros(23))
    (directory/'events.json').write_text('[]',encoding='utf-8')
    (directory/'manifest.json').write_text('{"protocol":"test-observer"}',encoding='utf-8')


class PrefixObserverTests(unittest.TestCase):
    def test_missing_case_waits_without_creating_experiment_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'not_started'
            result=observe(path,300,comparisons={})
            self.assertEqual(result['status'],'WAITING_FOR_COMPLETED_PREFIX')
            self.assertFalse(result['full_surface_claim'])
            self.assertFalse(path.exists())

    def test_partial_last_chunk_waits_without_borrowing_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); (path/'trace_chunks').mkdir()
            np.savez(path/'trace_chunks/0001.npz',**trace(1.))
            (path/'trace_chunks/0002.npz').write_bytes(b'PK\x03\x04partial')
            np.savez(path/'coverage_checkpoint.npz',elapsed=1000.,covered_vertices=np.ones(8192,bool))
            result=read_completed_prefix(path,2.)
            self.assertEqual(result['status'],'WAITING_FOR_COMPLETED_PREFIX')
            self.assertEqual(result['available_end_s'],1.)
            self.assertEqual(len(result['sources']),1)
            self.assertIsNone(result['trace'])

    def test_exact_prefix_excludes_future_coverage_and_recovery_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp); save_case(path,trace())
            (path/'events.json').write_text(json.dumps([
                dict(time_s=1.2,kind='COORDINATED_RECOVERY_BEGIN',sequence=1,selected_finger=0,duration_s=.4),
                dict(time_s=1.8,kind='COORDINATED_RECOVERY_END',sequence=1,measured_recovery_success=True)]),encoding='utf-8')
            result=observe(path,1.,comparisons={})
            self.assertEqual(result['status'],'READY')
            self.assertAlmostEqual(result['current']['coverage'],.1)
            self.assertEqual(result['current']['covered_goals'],10)
            self.assertEqual(result['current']['metrics']['sample_count'],101)
            self.assertEqual(result['current']['recovery']['episodes'],[])
            self.assertEqual(result['current']['trace_sources'][0]['sha256'],
                             hashlib.sha256((path/'trace.npz').read_bytes()).hexdigest())
            self.assertFalse(result['full_surface_claim'])

    def test_gap_or_duplicate_timestamp_is_rejected_instead_of_interpolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            for malformed in ('gap','duplicate'):
                record=trace()
                if malformed=='gap':
                    record={key:np.delete(value,50,axis=0) for key,value in record.items()}
                else:
                    record['time_s'][50]=record['time_s'][49]
                save_case(path,record)
                with self.assertRaisesRegex(ValueError,'missing row|strictly increasing'):
                    read_completed_prefix(path,1.)

    def test_initial_state_mismatch_prevents_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            new,old=Path(tmp)/'new',Path(tmp)/'old'
            save_case(new,trace());save_case(old,trace())
            np.savez(old/'initial.npz',qpos=np.ones(23),qvel=np.zeros(23),ctrl=np.zeros(23))
            with self.assertRaisesRegex(ValueError,'initial qpos/qvel/ctrl mismatch'):
                observe(new,1.,comparisons={'old':old})

    def test_claimed_event_without_actual_contact_is_not_measured_or_audited_success(self):
        record=trace();record['active'][:,0]=False;record['forces_n'][:,0]=0.
        events=[dict(time_s=.2,kind='COORDINATED_RECOVERY_BEGIN',sequence=1,selected_finger=0,duration_s=.4),
                dict(time_s=.8,kind='COORDINATED_RECOVERY_END',sequence=1,measured_recovery_success=True,status='RECOVERED')]
        result=recovery_evidence(record,events)
        episode=result['episodes'][0]
        self.assertEqual(result['native_stable_event_count'],1)
        self.assertEqual(result['audited_native_stable_event_count'],0)
        self.assertFalse(episode['sampled_terminal_safe_span_at_least_150ms'])
        self.assertEqual(episode['followup_contact']['contact_fraction'],0.)
        self.assertEqual(episode['actual_joint_range_deg'],[0.,0.,0.,0.])

    def test_phase_gap_preserves_air_and_surface_attribution(self):
        record=trace()
        record['active'][50:121]=False
        record['phase'][50:71]='LIFT'
        record['phase'][71:101]='TRANSFER'
        result=phase_gap_attribution(record)
        gap=result['longest_whole_hand_absence_intervals'][0]
        self.assertAlmostEqual(gap['duration_s'],.71)
        self.assertAlmostEqual(gap['phase_durations_s']['LIFT'],.21)
        self.assertAlmostEqual(gap['phase_durations_s']['TRANSFER'],.30)
        self.assertAlmostEqual(gap['phase_durations_s']['SLIDE'],.20)
        self.assertAlmostEqual(result['phases']['SLIDE']['longest_absence_within_phase_s'],.20)

    def test_audit_hash_mismatch_is_not_audited_even_when_status_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);save_case(path,trace())
            (path/'summary.json').write_text(json.dumps(dict(trace_sha256='stale',full_surface_complete=True)),encoding='utf-8')
            (path/'audit.json').write_text(json.dumps(dict(status='PASSED',trace_sha256='stale')),encoding='utf-8')
            result=observe(path,1.,comparisons={})
            self.assertFalse(result['current']['source_run_audit_verified'])
            self.assertFalse(result['current']['full_surface_claim'])


if __name__=='__main__':
    unittest.main()
