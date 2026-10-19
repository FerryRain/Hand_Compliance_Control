"""Original recovery boundaries with the single-change force-release entry."""
from contextlib import ExitStack
from unittest.mock import patch

from Module.i04_geometric_traversal import release_surface_slide
from Module.tests import test_i04_coordinated_combo_integration as combo
from Module.tests import test_i04_coordinated_boundary_integration as boundary
from Module.tests import test_i04_coordinated_prepared_integration as prepared


class _ReleaseEntryAliases:
    @classmethod
    def setUpClass(cls):
        cls.entry_aliases = ExitStack()
        for module in (combo, boundary, prepared):
            cls.entry_aliases.enter_context(patch.object(module, 'runner', release_surface_slide))
        try:
            super().setUpClass()
        except BaseException:
            cls.entry_aliases.close()
            raise

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            cls.entry_aliases.close()


class ReleaseComboIntegrationTests(_ReleaseEntryAliases, combo.CoordinatedComboRunnerTests):
    pass


class ReleaseBoundaryIntegrationTests(_ReleaseEntryAliases, boundary.CoordinatedBoundaryIntegrationTests):
    pass


class ReleasePreparedIntegrationTests(_ReleaseEntryAliases, prepared.CoordinatedPreparedIntegrationTests):
    pass
