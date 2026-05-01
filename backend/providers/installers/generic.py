from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import InstallPlanStep, InstallerContext, generic_install_plan


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    return generic_install_plan(entry, context)
