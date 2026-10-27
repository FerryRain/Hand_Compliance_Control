# YCB Mustard Bottle V-HACD Asset

This directory contains a simulation derivative of object `006_mustard_bottle`
from the YCB Object and Model Set.

- Source: YCB Object and Model Set, Google 16k mesh
- Dataset website: https://www.ycbbenchmarks.com/
- Dataset license: CC BY 4.0
- Processing: center and scale by 2.8, export OBJ, then decompose into 256
  convex collision parts with V-HACD
- Exact preprocessing parameters and source checksum: `manifest.json`

Please cite the YCB Object and Model Set when redistributing or using this
asset. The generated collision parts remain subject to the source dataset's
attribution requirements.

`preview.xml` is intentionally not tracked because the generated file contains
machine-local absolute paths. Runtime loading uses `visual_scaled.obj` and the
`collision_part_*.obj` files through
`mcc_finger_compliance_control/configs/objects/ycb_mustard.yaml`.
