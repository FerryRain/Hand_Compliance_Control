"""Fixed virtual inward reference with unchanged rooted collector and MCC.

Physical initialization still uses the separated healthy root. Only the
pre-MCC command is changed; no state writes or runtime geometry fitting.
"""
import json
from pathlib import Path

import torch
import collect_contact_wm_rooted_interaction as rooted


class VirtualRootProbe(rooted.RootedInteractionProbe):
    def __init__(self, args, row):
        super().__init__(args, row)
        reference = json.loads(Path(args.virtual_root_reference).read_text())
        if reference['object'] != row['object_id']:
            raise ValueError('Virtual reference object mismatch')
        for job in self.jobs:
            root = self.initial_roots[job['root_index']]
            if root['q'] != reference['physical_root_q']:
                raise ValueError('Virtual reference does not match physical initial q')
            if root['collection_home_object_pose'] != reference['home_object_pose']:
                raise ValueError('Virtual reference home mismatch')
        candidate = torch.tensor(reference['virtual_q'], device=self.device, dtype=self.q0.dtype)
        self.fixed_reference_q = candidate[None].expand_as(self.q0).clone()
        if not torch.equal(self.fixed_reference_q, self.fixed_reference_q.clamp(self.mcc.lower, self.mcc.upper)):
            raise ValueError('Virtual reference outside joint limits')
        self.virtual_reference_report = reference

    def prepare_motion(self, meta):
        super().prepare_motion(meta)
        meta.update(nominal_reference='fixed virtual inward grasp, healthy separated physical initial q',
                    virtual_root_reference=self.virtual_reference_report)


if __name__ == '__main__':
    rooted.RootedInteractionProbe = VirtualRootProbe
    rooted.main()
