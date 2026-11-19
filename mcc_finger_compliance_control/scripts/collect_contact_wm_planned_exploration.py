"""Frozen planned object motion with DP or bounded action exploration.

Keeps the verified loading, incremental MCC and physical plant. Planned healthy
q remains diagnostic context, never a label for an action perturbed away from
it. Exploration is pre-MCC and uses the existing smooth action generator.
"""
import numpy as np

import collect_contact_wm as collection
import collect_contact_wm_planned as planned
from collect_contact_wm_motion import MotionProbe


class ExploratoryPlannedProbe(planned.PlannedProbe):
    def initialize(self):
        records=super().initialize()
        # InteriorProbe already persists the exact physics trace/model. Add
        # raw DP plans without replacing that close path.
        original_close=self.env.close
        def close():
            try:
                if getattr(self,'dp_source',None) is not None:
                    self.dp_source.save(collection.episode_folder(self.args,self.row))
            finally:
                original_close()
        self.env.close=close
        return records

    def prepare_motion(self,meta):
        amplitude=float(getattr(self.args,'perturbation_rad',0.))
        if not np.isfinite(amplitude) or not 0<=amplitude<=.08:
            raise ValueError('Smooth exploration must be within [0,.08] rad')
        super().prepare_motion(meta)
        MotionProbe.prepare_motion(self,meta)
        using_dp=self.dp_source is not None
        meta.update(
            nominal_reference=('frozen_DP_after_76_fixed_bootstrap_frames' if using_dp
                               else 'offline_planned_virtual_q'),
            object_path_reference='frozen offline object commands, independent of measured q or contact',
            action_owner=('frozen_DP -> scheduler' if using_dp else 'planned_virtual_q')
                         +' -> bounded_smooth_exploration -> incremental_MCC -> servo',
            labels='Only measured future q/contact/force are transition labels; planned_healthy_q is context, not a corrected target',
            collection_purpose='action_conditioned_WM_interaction_not_healthy_demonstration',
            same_state_branching_supported=False)

    def reference_and_object(self,frame,extra=None):
        if extra is not None:
            raise ValueError('Exploration branches require controller/DP state validation')
        return self.virtual[frame],self.object_path[frame]

    def desired(self,frame,extra=None):
        self.reference_index=frame
        return MotionProbe.desired(self,frame,extra)


if __name__=='__main__':
    planned.PlannedProbe=ExploratoryPlannedProbe
    planned.main()
