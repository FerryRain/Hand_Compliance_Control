# Hand Compliance Control

This repository separates the proposed research method from its explicit
optimization baseline:

- [`PROPOSAL.md`](PROPOSAL.md) defines the main method: scalable inverse
  demonstration generation, a wrist-conditioned finger Diffusion Policy, and
  wrist-only ER-GPIS active exploration.
- [`full_hand_mcc/`](full_hand_mcc/) implements **Baseline 2**, the explicit
  FR3 + LEAP whole-hand optimizer and its low-level MCC controller.
- [`minimalist_compliance_control/`](minimalist_compliance_control/) is retained
  only as the upstream MCC reference from which the virtual
  mass-damping-stiffness interpretation was derived.

`full_hand_mcc/` is not the online architecture of the proposed main method.
It receives an optimized wrist trajectory and four optimized Cartesian
fingertip trajectories, then executes them with Wrist MCC and four independent
fingertip-force MCC loops. The exact split between the main method and
Baseline 2 is fixed in
[`CONTROL_STRATEGIES.md`](CONTROL_STRATEGIES.md).

## Active repository structure

| Path | Purpose |
| --- | --- |
| `PROPOSAL.md` | Persistent research question, contribution, experiment plan, and development order |
| `CONTROL_STRATEGIES.md` | Authoritative sensor semantics and low-level control laws for the main method and Baseline 2 |
| `full_hand_mcc/` | Active FR3 + LEAP Baseline-2 implementation, tests, experiment entry point, and subproject process log |
| `full_hand_mcc/BASELINE2_ACCEPTANCE.md` | Frozen Level 1–5 numerical, timing, generalization, and visual acceptance protocol |
| `minimalist_compliance_control/` | Read-only upstream/reference MCC implementation; not an active demo entry point |
| `PROCESS.md` | Repository-level decisions, current checkpoint, completed work, and unresolved work |

Earlier superseded robot/controller experiments and rejected media are not
active project entry points. Their historical reasoning is kept in the process
logs and Git history, not in the current run instructions.

## Baseline-2 controller in one view

```text
upper whole-hand optimizer
  -> planned wrist pose trajectory
     + measured/estimated wrist wrench
     -> lower-bandwidth Wrist MCC -> FR3 command

  -> four planned fingertip Cartesian trajectories
     + four direct physical fingertip-force measurements
     -> four normal-only Finger MCC loops -> fingertip IK -> LEAP commands
```

Motor loads remain diagnostic and safety signals. They are not reconstructed
into the four primary fingertip-force feedback channels. The palm-root point is
a kinematic guide and need not contact the object. LEAP Hand contact outside
the four pads may occur within its force and penetration limits but does not
count toward fingertip contact retention. Any FR3/object contact is a hard
failure.

The direct-force runtime is now self-contained. Superseded task
configurations, motor-force compatibility APIs, and the five historical
controller variants have been removed; the active demo has no `--variant`
switch. Read-only motor-load diagnostics remain available.

## Windows environment

Continue using the already validated project runtime until the current demo is
complete:

```powershell
.\.venv\Scripts\python.exe
```

Migration to the `handcomp` Conda environment remains deferred. Do not change
the active simulation environment midway through an acceptance sequence.

From the repository root, check the active demo and run its regression suite:

```powershell
.\.venv\Scripts\python.exe -B `
  full_hand_mcc\scripts\demo_surface_slide.py --help

.\.venv\Scripts\python.exe -B -m unittest discover `
  -s full_hand_mcc\tests -v
```

The run commands and controller parameters are in
[`full_hand_mcc/README.md`](full_hand_mcc/README.md). The project-level hard
gates and object curriculum are frozen in
[`full_hand_mcc/BASELINE2_ACCEPTANCE.md`](full_hand_mcc/BASELINE2_ACCEPTANCE.md).

## Current validation boundary

The cleaned direct-force working tree is `PASS-NUMERICAL-L1`: 17/17 unittests
pass; demo, grasp-search, and grasp-optimization CLI checks exit 0; and the
5 mm/750-step CUDA headless smoke exits 0. It records contact ratios
`[0.9975,1.0,1.0,0.99]`, majority ratio `1.0`, average `3.9875/4`, minimum
`3/4`, terminal `4/4=65` frames, filtered force peaks below 25 N, raw peaks
below 40 N, and zero audited collision/penetration/incidental contact. No video
was generated. Level 2–5 remain `NOT RUN`: this does not prove the 0.48 m
bottom-to-top route, top-surface contact, timing variants, generalization, or a
visually accepted delivery demo.

## Continuing the work

At the start of a new session, read these files in order:

1. [`PROPOSAL.md`](PROPOSAL.md)
2. [`CONTROL_STRATEGIES.md`](CONTROL_STRATEGIES.md)
3. [`full_hand_mcc/BASELINE2_ACCEPTANCE.md`](full_hand_mcc/BASELINE2_ACCEPTANCE.md)
4. [`PROCESS.md`](PROCESS.md)
5. [`full_hand_mcc/PROCESS.md`](full_hand_mcc/PROCESS.md)
6. [`full_hand_mcc/README.md`](full_hand_mcc/README.md)

The process documents intentionally retain failed experiments and superseded
decisions as historical evidence. Their newest checkpoint, rather than an old
command embedded in the history, defines what should happen next.
