"""Observation-only planning instrumentation for the frozen-B comparison.

Never installed in the published B experiment. Wall and process CPU times are
separate from simulated motion duration. Nested calls must not be summed with
their parents. Optimizer costs retain their own objective and units.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import json
from pathlib import Path
from time import perf_counter, process_time

import numpy as np


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


class PlanningRecorder:
    def __init__(self, path, clock=lambda: None, *, scope='ONLINE_A'):
        self.path = Path(path)
        self.stream = self.path.open('x', encoding='utf-8')
        self.clock = clock
        self.scope = scope
        self.stack = []
        self.rows = []
        self.next_id = 0
        self.restore = []

    @contextmanager
    def call(self, name, **metadata):
        row = dict(id=self.next_id, parent_id=self.stack[-1]['id'] if self.stack else None,
                   scope=self.scope, name=name, simulation_time_s=self.clock(), **metadata)
        self.next_id += 1
        self.stack.append(row)
        wall, cpu = perf_counter(), process_time()
        try:
            yield row
            row['status'] = 'RETURNED'
        except BaseException as error:
            row.update(status='RAISED', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row.update(wall_s=perf_counter()-wall, cpu_s=process_time()-cpu)
            assert self.stack.pop() is row
            self.rows.append(row)
            if not self.stack:
                for item in self.rows:
                    self.stream.write(json.dumps(plain(item), separators=(',', ':'), allow_nan=False)+'\n')
                self.stream.flush()
                self.rows.clear()

    def wrap(self, obj, name, *, label=None):
        original = getattr(obj, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            with self.call(label or name) as row:
                result = original(*args, **kwargs)
                row['success'] = result is not None
                patch = result[0] if name == 'find_patch' else result
                if isinstance(patch, dict):
                    for key in ('reason', 'anchor', 'goal', 'expected_contacts', 'clearance', 'complete'):
                        if key in patch:
                            row[key] = plain(patch[key])
                    if 'distance' in patch:
                        row['route_length_m'] = float(patch['distance'][-1])
                    if 'q' in patch:
                        row['solution_count'] = len(patch['q']) if np.ndim(patch['q']) == 2 else 1
                    row['success'] = patch.get('reason') is None
                if name == 'find_patch':
                    row['attempts'] = plain(result[1])
                    row['requested_search_budget_s'] = kwargs.get('budget_s', 8.)
                if hasattr(result, 'duration_s'):
                    row['planned_motion_s'] = float(result.duration_s)
                if name == 'prepare_transfer' and result is not None:
                    row['planned_motion_s'] = float(result[0].duration_s+result[1].duration_s)
                    row['clearance_m'] = float(result[3])
                return result
        setattr(obj, name, observed)
        self.restore.append((obj, name, original))

    def observe_least_squares(self):
        from . import natural_belly_traversal, whole_hand_surface_planner, whole_hand_belly
        for module, objective in (
            (natural_belly_traversal, 'WRIST: 0.5*(position^2 + (0.1*rotation)^2 + arm_clearance_residual^2)'),
            (whole_hand_surface_planner, 'JOINT: 0.5*sum(weighted_four_pad_pose_posture_and_clearance_residual^2)'),
            (whole_hand_belly, 'INITIAL_JOINT: 0.5*sum(four_pad_pose_and_posture_residual^2)'),
        ):
            original = module.least_squares
            def observed(*args, _original=original, _objective=objective, **kwargs):
                with self.call('least_squares', objective=_objective,
                               variables=int(np.size(args[1])), max_nfev=kwargs.get('max_nfev')) as row:
                    result = _original(*args, **kwargs)
                    row.update(cost=float(result.cost), nfev=int(result.nfev),
                               njev=None if result.njev is None else int(result.njev),
                               optimality=float(result.optimality), solver_status=int(result.status),
                               solver_success=bool(result.success), residual_count=int(len(result.fun)))
                    return result
            module.least_squares = observed
            self.restore.append((module, 'least_squares', original))

    def close(self):
        for obj, name, original in reversed(self.restore):
            setattr(obj, name, original)
        self.restore.clear()
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()
