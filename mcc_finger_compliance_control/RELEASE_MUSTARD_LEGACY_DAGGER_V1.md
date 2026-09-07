# Mustard legacy DAgger pilot assets v1

This release freezes the reproducible 96-D mustard q-policy pilot used to
test whether local corrective/DAgger data reduces closed-loop autoregressive
drift. It is a diagnostic baseline, not the final policy architecture.

Assets:

- `best.pt`: 25k kinematic-residual-q checkpoint, 16-step observation and
  8-step prediction horizons, stride 5.
- `mustard_v1_239_mesh_normal_inward_inverted.h5`: 239-episode inverse replay
  dataset used for physical deployment and teacher comparison.
- `mustard_v1_239_motion96_kinematic_palm_dp.h5`: 96-D clean training dataset
  and the schema/normalization reference for corrective data.

SHA-256:

```text
1542b01abc8090f54f4fb54e06e19d90f84964d05860a1d90648648b15a3b1c7  best.pt
34ea1a8181af1d3bc5cef29351b217390825976200235343aeeb8c43bc2d9f57  mustard_v1_239_mesh_normal_inward_inverted.h5
1beb6406a99661d987ec68475a4a4b3af802a78dae8f75eff5a5d69e33ebc18a  mustard_v1_239_motion96_kinematic_palm_dp.h5
```

The tested workflow and exact download, rollout, relabeling, fine-tuning and
evaluation commands are in `mcc_finger_compliance_control/DAgger_Recovery_Data_Guide.md`.
