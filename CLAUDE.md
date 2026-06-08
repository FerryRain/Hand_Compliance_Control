# Project Context: Dexterous Hand Compliance Control (MJLab)

## 1. Project Mission
Developing a **Compliance Control** algorithm for a dexterous hand (LEAP hand) to achieve stable, full-contact surface stroking. The priority is to maintain delicate contact and adapt to surface geometry without excessive force.

## 2. Scope & Boundaries
- **Primary Working Directory**: `src/mjlab/tasks/leaphand` — Task definition and compliance controller implementation.
- **Scripts**: `palm_compliance_control/` — Running scripts for palm compliance controller (run_test.py).
- **Core Framework**: `src/mjlab/` (Treat as READ-ONLY; do not modify unless explicitly instructed).
- **MCC Reference**: `minimalist_compliance_control/` — Original MCC implementation (wrench estimation, compliance reference, IK solvers).
- **Ignore**: Do not index `tests/`, `docs/`, or `typings/` unless requested.

## 3. Architecture Overview

### 3.1 Palm MCC Compliance Controller (`leaphand_palm_mcc_env_cfg.py`)
- **`MCCPalmComplianceController`**: Main controller using MCC-style admittance control for xarm6 palm.
- **Pipeline** (per control step):
  1. `_sync_observer` — Forward observer model at actual joint positions (qvel=0 for pure gravity bias)
  2. Preparation phase — Linear interpolation from initial to target posture
  3. Alignment — Initialize x_des, x_ref, q_ref from current state
  3.5 External x_des override — Accept (B,3) position or (B,6) position+orientation from external surface tracking
  4. Torque EMA filtering — Smooth motor torques and bias
  5. Wrench estimation — Regularized least-squares from joint torques (MCC estimate_wrench)
  6. **Anisotropic Admittance** — Core compliance equation with dynamic stiffness:
     - Contact normal `n` estimated from f_ext (when in contact) or palm orientation `palm_rotmat[:, normal_axis_idx]` (when free)
     - Stiffness: `K = K_position * I + (K_force - K_position) * n·n^T` (normal=soft for force control, tangent=stiff for position tracking)
     - f_cmd = -f_desired_normal * n (push into surface along normal)
     - Damping: critical damping via `get_damping_matrix`
  7. **DLS-IK** — Track x_ref using **actual palm position feedback** (not virtual FK), with nullspace posture pull (currently disabled: k_posture=0 due to DLS leakage)
  8. Output — q_ref for arm joints (6 DOF), actual finger joints held constant

- **`MCCPalmControlCfg`**: RL config dataclass, passes controller params via kwargs. Key fields:
  - `K_force` (default 20): Normal direction stiffness (low → force control dominant)
  - `K_position` (default 200): Tangent direction stiffness (high → position tracking)
  - `f_desired_normal` (default 0): Desired normal force in N (positive = into surface, 0 = pure position servo)
  - `mass_trans`, `inertia_diag`: Admittance mass/inertia parameters
  - `Kp_task`, `dls_lambda`, `k_posture`: IK tracking parameters
  - `alpha_tau`: Torque EMA smoothing factor
  - `alpha_normal`: Contact normal EMA smoothing factor
  - `contact_threshold`: Min |f_ext| to use as contact normal estimate
  - `normal_axis`: Palm local axis used for default contact normal ("x"/"y"/"z")

### 3.2 Surface Tracking (External, in `run_test.py`)
- **`--surface-track` flag** enables real-time x_des computation outside the controller
- **`capsule_surface_intersection()`**: Standalone function computing ray-capsule intersection (cylinder body + hemispherical caps)
- **`PolicyWithActionAdapter._compute_surface_x_des()`**: Reads palm_pos (obs[76:79]) and target_pos/rot (obs[82:88]), computes 6D x_des = [surface_point(3), target_orientation(3)]
- x_des passed to controller via `__call__(obs, x_des=x_des)` — controller remains generic
- Palm orientation target: local Z axis aligned with surface normal direction

### 3.3 Observation Layout (88-D)
```
0:22   joint_pos          (22,)
22:28  joint_vel_arm       (6,)
28:34  qfrc_actuator_arm   (6,)
34:40  qfrc_bias_arm       (6,)
40:58  palm_jacobian       (18,) → reshape (3,6)
58:76  palm_jacobian_rot   (18,) → reshape (3,6)
76:79  palm_pos            (3,)
79:82  palm_rot            (3,)
82:85  target_pos          (3,)
85:88  target_rot          (3,)
```

### 3.4 Task Registration (`__init__.py`)
```python
register_mjlab_task(
  task_id="Leaphand-Palm-MCC-Compliance-Control",
  rl_cfg=MCCPalmControlCfg(amplitude=0.8, K_force=20.0, K_position=200.0, ...),
)
```

## 4. Key Design Decisions & Lessons Learned

### 4.1 IK Must Use Real Palm Feedback (Not Virtual FK)
- **Problem**: Original IK used `_virtual_palm_fk(q_ref)` which converges in virtual space immediately, but actual arm never catches up → Cart_Err stuck at 40cm+, arm frozen.
- **Fix**: IK feedback now uses `palm_pos` from `_sync_observer(qpos_full)` (actual joint positions) and Jacobian from actual configuration.

### 4.2 Nullspace Posture Pull Disabled
- **Problem**: `dq_null = N_space @ (k_posture * (q_posture - q_ref))` exactly canceled dq_primary due to DLS regularization leakage (dls_lambda=0.1). Total dq ≈ 0, q_ref frozen.
- **Fix**: `k_posture = 0` — disable nullspace posture pull.

### 4.3 Contact Normal from Palm Orientation
- **Problem**: Default contact normal was hardcoded world +Z, wrong when palm approaches surface from different directions.
- **Fix**: Default normal = `palm_rotmat[:, normal_axis_idx]` — dynamically tracks palm's actual facing direction.

### 4.4 Anisotropic Stiffness Aligns with MCC
- Normal direction: K_force ≈ 10-20 (soft, force-controlled, large f_cmd offset)
- Tangent direction: K_position ≈ 100-200 (stiff, position-controlled)
- Prevents "hard position push" causing bouncing/loss of contact

## 5. Known Issues & Next Steps
- [ ] **Bouncing on contact**: f_cmd creates 5cm+ x_ref offset; large impact velocity causes oscillation. Tune `mass_trans` (increase) and `K_force` (moderate increase).
- [ ] **No force integral**: Pure feedforward f_cmd leaves steady-state force error. Need PI force control — either integral in controller or external x_des offset accumulation in run_test.py.
- [ ] **Posture tracking (k_posture)**: Need to re-enable nullspace posture pull without DLS leakage (e.g., separate task-space and nullspace gains, or higher dls_lambda).
- [ ] Combine with finger compliance control.
- [ ] Train RL/Diffusion Policy for surface stroking.

## 6. Technical Constraints & Style Guide
- **Compliance Paradigm**: Main ideas in `HandContactRe-location.md`.
- **Key Concepts**:
  - **Force Regulation**: Smooth force application; prevent contact instability or bouncing.
  - **Contact Estimation**: Use joint torque → wrench estimation for contact state.
  - **Surface Adaptation**: Palm posture adapts to surface normal vector.
- **Interaction Protocol**:
  - Focus on stability analysis and smooth force transition.
  - Use `grep` to find definitions in `mjlab/core/` rather than reading whole files.
  - Explain before showing code changes.

## 7. Prohibited Actions
- Do not implement basic tool functions that already exist.
- Do not refactor `mjlab` core physics modules.
- Do not read unrelated simulation environment setup files unless issue is explicitly about world contact settings.
- Controller should remain generic; surface-specific logic belongs in external scripts.
