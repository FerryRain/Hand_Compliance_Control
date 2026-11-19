"""Predictive palm retreat during transfers, with exact audited continuation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import mujoco
import numpy as np

from .palm_safe_mcc import PalmSafeTraversal
from .physics import joint_state
from .whole_hand_belly_probe import snapshot_sources
from .whole_hand_completion import CompletionWholeHandTraversal


PROTOCOL = 'i04.b.palm-collision-recovery.mcc.v2'


class PalmTransferReplan(RuntimeError):
    pass


class PalmRecoveryTraversal(PalmSafeTraversal):
    def __init__(self, output, seconds=9000., *, parent=None, join_time=None):
        super().__init__(output, seconds)
        self.palm_retreats = 0
        if parent is not None:
            self.restore_audited_branch(Path(parent).resolve(), join_time)
        manifest = json.loads((self.out/'manifest.json').read_text())
        manifest.update(protocol=PROTOCOL, runtime_entrypoint='Module/i04_geometric_traversal/palm_safe_recovery.py',
            palm_transfer_lookahead_s=.12, palm_transfer_trigger_m=.004,
            palm_retreat_target_m=.008, palm_retreat_release_m=.0058,
            source_hashes=snapshot_sources(self.out), resume=self.resume_info,
            coverage_inherited=bool(self.resume_info), fresh_single_version=self.resume_info is None,
            scope='Predictive transfer retreat; historical B and the first palm-safe trial remain immutable')
        (self.out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def transfer_risk(self, obs):
        """Distance Jacobians use native actual velocities, not a target speed."""
        m, d = self.h.model, self.data
        projected = .006
        for ct in d.contact:
            a, b = int(ct.geom1), int(ct.geom2)
            if a in self.palm_monitor.palm and b in self.palm_monitor.objects:
                part, normal = a, -np.asarray(ct.frame[:3])
            elif b in self.palm_monitor.palm and a in self.palm_monitor.objects:
                part, normal = b, np.asarray(ct.frame[:3])
            else:
                continue
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, np.asarray(ct.pos), int(m.geom_bodyid[part]))
            rate = float(normal@jp@d.qvel)
            projected = min(projected, float(ct.dist)+.12*min(0., rate))
        return obs['palm']['clearance_m'] < .004 or projected < .004, projected

    def retreat_palm(self):
        start = float(self.data.time)
        before = self.measure()['palm']['clearance_m']
        self.palm_retreats += 1
        while self.data.time-start < 2. and self.data.time < self.duration:
            obs = self.measure()
            risk, predicted = self.transfer_risk(obs)
            if obs['palm']['clearance_m'] >= .0058 and predicted >= .0055:
                self.events.append(dict(kind='PALM_ACTIVE_RETREAT', time_s=float(self.data.time),
                    start_s=start, before_m=before, after_m=obs['palm']['clearance_m'], success=True))
                self.refresh_surface()
                return
            command = self.planner.clearance_exit(joint_state(self.h, self.data), self.last,
                arm_target_m=.004, object_target_m=.008)
            if command is None:
                # The reference can already be outside the retreat buffer
                # while the actual servo still catches up. Hold it only if
                # both original command guards accept that unchanged target.
                safety = self.planner.safety
                if (safety.command_collision_reason(joint_state(self.h, self.data), self.last)
                        or safety.command_collision_reason(self.last, self.last)):
                    raise RuntimeError('NO_PALM_SEPARATING_COMMAND')
                command = self.last.copy()
            super().tick(command, 'PALM_CLEARANCE_RECOVERY', obs)
        raise RuntimeError('PALM_RETREAT_TIMEOUT')

    def tick(self, command, phase, obs=None):
        obs = self.measure() if obs is None else obs
        if phase in ('NECESSARY_REORIENTATION', 'COOPERATIVE_WITHDRAWAL'):
            risk, predicted = self.transfer_risk(obs)
            if risk:
                self.events.append(dict(kind='PALM_PREDICTIVE_TRANSFER_INTERRUPT', time_s=float(self.data.time),
                    phase=phase, clearance_m=obs['palm']['clearance_m'], projected_clearance_m=predicted))
                self.retreat_palm()
                raise PalmTransferReplan('PALM_RETREAT_REPLAN')
        return super().tick(command, phase, obs)

    def air_move(self, path, phase):
        try:
            return super().air_move(path, phase)
        except PalmTransferReplan:
            return False

    def restore_audited_branch(self, parent, join_time):
        """Branch at a saved pre-step state; retain every preceding native tick.

        Only a constant-coverage final suffix can be replaced, so restoring the
        parent's monotone visited mask does not import future coverage credit.
        The new combined case must undergo its own continuous native audit.
        """
        ps = json.loads((parent/'summary.json').read_text())
        pa = json.loads((parent/'independent_native_audit/audit.json').read_text())
        assert pa['status'] == 'PASSED_NATIVE_EVIDENCE_AUDIT' and pa['continuous_replay']
        assert ps['protocol'] == 'i04.b.palm-collision.mcc.v1'
        requested = float(join_time)
        selected = None
        for index, chunk in enumerate(ps['trace_chunks']):
            if chunk['start_s'] <= requested <= chunk['end_s']+.002:
                with np.load(parent/chunk['file'], allow_pickle=False) as z:
                    k = int(np.searchsorted(z['time_s'], requested))
                    if k < len(z['time_s']):
                        selected = index, k, {key:z[key].copy() for key in z.files}
                        break
        assert selected is not None
        index, k, rows = selected
        assert rows['palm_clearance_m'][k] >= .004
        assert int(rows['visited_vertex_count'][k]) == ps['visited_vertices']
        assert abs(float(rows['coverage'][k])-ps['coverage_fraction']) < 1e-12
        # The physical model changes only generated asset paths between cases.
        import xml.etree.ElementTree as ET
        trees = [ET.fromstring((p/'model.xml').read_text()) for p in (parent, self.out)]
        for tree in trees:
            for asset in tree.findall('./asset/*[@file]'):
                asset.set('file', Path(asset.get('file')).name)
        assert ET.tostring(trees[0]) == ET.tostring(trees[1])
        shutil.copy2(parent/'model.xml', self.out/'model.xml')
        provenance = self.out/'parent_provenance'
        provenance.mkdir()
        for name in ('manifest.json', 'summary.json', 'initial.npz', 'final_state.npz'):
            assert sha256((parent/name).read_bytes()).hexdigest() == pa['source_files'][name]
            shutil.copy2(parent/name, provenance/name)
        shutil.copy2(parent/'independent_native_audit/audit.json', provenance/'audit.json')
        shutil.copytree(parent/'source_snapshot', provenance/'source_snapshot')
        self.chunks = []
        for chunk in ps['trace_chunks'][:index]:
            source = parent/chunk['file']
            assert sha256(source.read_bytes()).hexdigest() == chunk['sha256']
            os.link(source, self.out/chunk['file'])
            self.chunks.append(dict(chunk))
        if k:
            chunk = ps['trace_chunks'][index]
            path = self.out/chunk['file']
            np.savez_compressed(path, **{key:value[:k] for key, value in rows.items()})
            self.chunks.append(dict(file=path.name, states=k, start_s=float(rows['time_s'][0]),
                end_s=float(rows['time_s'][k-1]), sha256=sha256(path.read_bytes()).hexdigest()))
        self.states = sum(c['states'] for c in self.chunks)
        shutil.copy2(parent/'initial.npz', self.out/'initial.npz')
        d = self.data
        qpos = np.r_[self.h.arm_qpos_adrs, self.h.hand_qpos_adrs]
        actuators = np.r_[self.h.arm_actuator_ids, self.h.hand_actuator_ids]
        d.qpos[qpos] = rows['q_rad'][k]
        d.qvel[:] = rows['qvel_rad_s'][k]
        d.ctrl[actuators] = rows['sensor_command_rad'][k]
        d.qacc_warmstart[:] = rows['qacc_warmstart'][k]
        d.time = float(rows['time_s'][k])
        with np.load(parent/'final_state.npz') as z:
            self.coverage.visited_vertices[:] = z['visited_vertices']
            self.coverage.visited_goals[:] = z['visited_goals']
        mujoco.mj_forward(self.h.model, d)
        self.last = rows['sensor_command_rad'][k].copy()
        self.anchor = int(rows['leader_finger'][k])
        self.goal = int(rows['goal_vertex'][k])
        self.refresh_surface()
        self.routes, self.lifts = int(ps['routes']), int(ps['necessary_lifts'])
        self.peak = float(ps['peak_force_n'])
        self.geometry_backtracks = int(ps['geometry_backtracks'])
        self.minimum_palm_clearance = .006
        for chunk in self.chunks:
            with np.load(self.out/chunk['file'], allow_pickle=False) as z:
                self.minimum_palm_clearance = min(self.minimum_palm_clearance, float(z['palm_clearance_m'].min()))
        self.events = [e for e in json.loads((parent/'events.json').read_text()) if e.get('time_s', -1.) <= d.time]
        self.resume_info = dict(parent_case=str(parent), join_time_s=float(d.time),
            parent_summary_sha256=sha256((parent/'summary.json').read_bytes()).hexdigest(),
            parent_audit_sha256=sha256((provenance/'audit.json').read_bytes()).hexdigest(),
            prefix_native_states=self.states, prefix_chunks=len(self.chunks), physical_state_reset=False,
            independent_trajectories_merged=False, controller_state='Refreshed from actual state; physical q/velocity/ctrl/warmstart unchanged',
            suffix_replaced_s=ps['executed_s']-float(d.time), future_coverage_inherited=False)
        self.events.append(dict(kind='AUDITED_NATIVE_BRANCH', time_s=float(d.time), **self.resume_info))
        self.stage = 'residual_completion'
        np.savez_compressed(self.out/'checkpoint.npz', qpos=d.qpos, qvel=d.qvel, ctrl=d.ctrl,
            qacc_warmstart=d.qacc_warmstart, time=d.time, visited_vertices=self.coverage.visited_vertices,
            visited_goals=self.coverage.visited_goals)
        (self.out/'trace_chunks.json').write_text(json.dumps(self.chunks, indent=2))
        (self.out/'events.json').write_text(json.dumps(self.events, indent=2))

    def run(self):
        if self.resume_info is None:
            result = super().run()
        else:
            if self.transfer_risk(self.measure())[0]:
                self.retreat_palm()
            result = CompletionWholeHandTraversal.run(self)
        result.update(protocol=PROTOCOL, minimum_palm_clearance_m=self.minimum_palm_clearance,
            palm_retreats=self.palm_retreats, historical_b_modified=False,
            coverage_inherited=bool(self.resume_info), fresh_single_version=self.resume_info is None,
            resume=self.resume_info, strict_original_coverage=True)
        (self.out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (self.out/'run_complete.json').write_text(json.dumps(dict(protocol=PROTOCOL,
            stop_reason=self.stop, final_s=float(self.data.time))), encoding='utf-8')
        print('PALM_RECOVERY_RESULT', json.dumps({k:v for k,v in result.items() if k != 'trace_chunks'}), flush=True)
        return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seconds', type=float, default=9000.)
    p.add_argument('--parent', type=Path)
    p.add_argument('--join-time', type=float)
    a = p.parse_args()
    PalmRecoveryTraversal(a.output, a.seconds, parent=a.parent, join_time=a.join_time).run()
