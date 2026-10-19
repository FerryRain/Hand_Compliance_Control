"""Reuse physical stage-boundary cases against the separate normal entry.

These cases control finite geometric proposals only. V6 search, native 2ms
physics, actual sensors, guards, source snapshots and new telemetry are real.
Aliases are scoped per class so the frozen coordinated entry's tests retain
their own controller. No surface-completion claim follows from held paths.
"""
from contextlib import ExitStack
from unittest.mock import patch

from Module.i04_geometric_traversal import normal_surface_slide
from Module.i04_geometric_traversal.active_fingers_v6 import ActiveFingerSearchV6
from Module.tests import test_i04_coordinated_combo_integration as combo
from Module.tests import test_i04_coordinated_boundary_integration as boundary
from Module.tests import test_i04_coordinated_prepared_integration as prepared


class _NormalEntryAliases:
    @classmethod
    def setUpClass(cls):
        cls.entry_aliases=ExitStack()
        for module in (combo,boundary,prepared):
            cls.entry_aliases.enter_context(patch.object(module,'runner',normal_surface_slide))
            if hasattr(module,'ActiveFingerSearchV3'):
                cls.entry_aliases.enter_context(patch.object(module,'ActiveFingerSearchV3',ActiveFingerSearchV6))
        try:super().setUpClass()
        except BaseException:
            cls.entry_aliases.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:super().tearDownClass()
        finally:cls.entry_aliases.close()


class NormalComboIntegrationTests(_NormalEntryAliases,combo.CoordinatedComboRunnerTests):
    pass


class NormalBoundaryIntegrationTests(_NormalEntryAliases,boundary.CoordinatedBoundaryIntegrationTests):
    pass


class NormalPreparedIntegrationTests(_NormalEntryAliases,prepared.CoordinatedPreparedIntegrationTests):
    pass
