"""Audit the applied-reference relief intervention, independently of search."""
from __future__ import annotations

import ast
import copy
from hashlib import sha256
from pathlib import Path

import mujoco
import numpy as np


NORMAL_PROTOCOL = 'i04-mustard-coordinated-finger-sliding.development.v3.7-normal'
RELEASE_PROTOCOL = 'i04-mustard-coordinated-finger-sliding.development.v3.8-release'
HELPER = 'Module/i04_geometric_traversal/reference_force_release.py'
VALIDATED_HELPER_SHA256 = '00c21a2f44a7d8064570e67a79313959fe6a0159cd7fe83c16d6f9ecf0011fd5'
PARENT = 'Module/i04_geometric_traversal/coordinated_surface_slide.py'
PARENT_SHA256 = 'e1f2b0bfc875f24637e67b2e354e21bebb6c92a67b783cc1168e87b45a4407e3'


def _run(tree):
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run')


def _loop(tree):
    return next(node for node in _run(tree).body if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'step')


def relief_entry_ast_checks(entry_source, parent_source=None):
    """A free/incorrect reference argument must fail before lineage normalization."""
    tree = ast.parse(entry_source)
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.level == 1 and node.module == 'reference_force_release']
    assert len(imports) == 1 and len(imports[0].names) == 1
    assert imports[0].names[0].name == 'applied_reference_force_release'
    assert imports[0].names[0].asname is None
    calls = [node for node in ast.walk(_run(tree)) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'applied_reference_force_release']
    assert len(calls) == 1
    expected = ast.parse('applied_reference_force_release(h, data, command, last, forces, normals, config)', mode='eval').body
    assert ast.dump(calls[0]) == ast.dump(expected), 'Relief must use the actual prior guarded last command'
    branches = [node for node in ast.walk(_loop(tree)) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == 'in_relief'
                and any(candidate is calls[0] for candidate in ast.walk(node))]
    assert len(branches) == 1
    first = branches[0].body[0]
    assert isinstance(first, ast.Assign) and first.value is calls[0]
    assert [ast.unparse(target) for target in first.targets] == ['command']
    # The next plant command is the same value saved as the next tick's prior.
    loop = _loop(tree)
    assert any(isinstance(node, ast.Assign) and ast.unparse(node) == 'last = command.copy()' for node in loop.body)
    assert any(isinstance(node, ast.Expr) and ast.unparse(node) == 'set_control(h, data, command)' for node in loop.body)
    result = dict(relief_prior_applied_argument_verified=True, relief_runtime_call_count=1)
    if parent_source is not None:
        parent = ast.parse(parent_source)
        normalized = copy.deepcopy(loop)
        class RestoreOriginalCall(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'applied_reference_force_release':
                    return ast.parse('relief.relieve_force(h, data, command, forces, normals)', mode='eval').body
                return self.generic_visit(node)
        normalized = RestoreOriginalCall().visit(normalized)
        assert ast.dump(normalized) == ast.dump(_loop(parent)), 'Single-factor entry changes other native-loop behavior'
        for name in ('checked_preload_path',):
            current_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
            parent_function = next(node for node in parent.body if isinstance(node, ast.FunctionDef) and node.name == name)
            assert ast.dump(current_function) == ast.dump(parent_function)
        search_imports = [node for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == 'active_fingers_v3']
        assert len(search_imports) == 1 and ast.unparse(search_imports[0]) == 'from .active_fingers_v3 import ActiveFingerSearchV3 as ActiveFingerSearch'
        result.update(release_only_native_loop_ast_verified=True,
                      release_only_ast_scope='Entire native step loop and preload validator; only the relief-reference call differs')
    return result


def applied_reference_relief_checks(directory, manifest, trace=None, events=()):
    directory = Path(directory)
    assert manifest['protocol'] in (NORMAL_PROTOCOL, RELEASE_PROTOCOL)
    assert manifest['force_release_source'] == HELPER
    assert manifest['source_hashes'][HELPER] == VALIDATED_HELPER_SHA256, 'Relief helper changed since independent native validation'
    assert sha256((directory/'source_snapshot'/HELPER).read_bytes()).hexdigest() == VALIDATED_HELPER_SHA256
    config = manifest['applied_reference_relief']
    assert config['reference'] == 'Actual prior guarded command'
    for field in ('original_site_jacobian_and_damping', 'original_slew_range_force_guards',
                  'arm_handoff_unchanged', 'loss_after_withdrawal_requires_measured_recovery'):
        assert config[field] is True, field
    parent = None
    if manifest['protocol'] == RELEASE_PROTOCOL:
        assert manifest['active_finger_source'] == 'Module/i04_geometric_traversal/active_fingers_v3.py'
        assert 'normal_priority_search' not in manifest, 'Release-only ablation must retain V3 search'
        assert manifest['source_lineage'] == dict(parent_entry=PARENT, parent_entry_sha256=PARENT_SHA256)
        assert manifest['source_hashes'][PARENT] == PARENT_SHA256
        parent_bytes = (directory/'source_snapshot'/PARENT).read_bytes()
        assert sha256(parent_bytes).hexdigest() == PARENT_SHA256
        parent = parent_bytes.decode('utf-8')
    entry = (directory/'source_snapshot'/manifest['runtime_entrypoint']).read_text(encoding='utf-8')
    result = relief_entry_ast_checks(entry, parent)
    result.update(applied_reference_relief_audited=True,
                  validated_relief_helper_sha256=VALIDATED_HELPER_SHA256,
                  relief_auditor_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    if trace is None:
        return result
    # Loaded finger values are independent of the preceding MCC proposal.
    # Rebuild those coordinates directly from native site Jacobians, then apply
    # the same frozen outer slew/range guards. Geometry deferrals are explicit.
    from .sliding_audit import _native_context, _restore_native, FINGERS
    ctx = None
    deferred = {round(float(e['time_s']), 9) for e in events if e['kind'] == 'COMMAND_GEOMETRY_DEFER'}
    candidates = np.flatnonzero(trace['phase'] == 'FORCE_RELIEF')
    checked, skipped, error = 0, 0, 0.
    source = ast.parse((directory/'source_snapshot/Module/i04_geometric_traversal/contracts.py').read_text())
    definition = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'ExperimentConfig')
    defaults = {node.target.id: ast.literal_eval(node.value) for node in definition.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None}
    assert defaults['dt_s'] == .002 and defaults['force_relief_speed_m_s'] == .03 and defaults['joint_margin_rad'] == .015
    for row in candidates:
        if round(float(trace['time_s'][row]), 9) in deferred:
            skipped += 1
            continue
        force, normal = trace['forces_n'][row], trace['normals_world'][row]
        loaded = (force >= 4.) & (np.linalg.norm(normal, axis=1) >= .5)
        if not loaded.any():
            continue
        if ctx is None:
            ctx = _native_context(directory)
        _restore_native(ctx, {key:trace[key][row] for key in ('q_rad','qvel_rad_s','sensor_command_rad','qacc_warmstart')})
        previous = trace['sensor_command_rad'][row]
        for finger in np.flatnonzero(loaded):
            columns = 7 + np.asarray([int(name) for name in FINGERS[finger].joint_names])
            jp, jr = np.zeros((3, ctx.model.nv)), np.zeros((3, ctx.model.nv))
            mujoco.mj_jacSite(ctx.model, ctx.data, jp, jr, int(ctx.sites[finger]))
            jacobian = jp[:,ctx.dof[columns]]
            delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-6*np.eye(3), normal[finger]*.03*.002)
            proposal = np.clip(previous[columns]+delta, previous[columns]-.0024, previous[columns]+.0024)
            joint_ids = ctx.model.dof_jntid[ctx.dof[columns]]
            proposal = np.clip(proposal, ctx.model.jnt_range[joint_ids,0]+.015, ctx.model.jnt_range[joint_ids,1]-.015)
            error = max(error, float(np.max(abs(proposal-trace['command_rad'][row,columns]))))
            checked += 1
    assert error < 1e-12, ('Applied-reference relief command mismatch', error)
    result.update(relief_loaded_finger_commands_reconstructed=checked,
                  relief_geometry_deferred_rows_skipped=skipped,
                  relief_command_reconstruction_max_error_rad=error,
                  relief_reconstruction_scope='Every saved loaded finger during FORCE_RELIEF, excluding explicit geometry deferrals; 100Hz logs, not full2ms intervention replay')
    return result
