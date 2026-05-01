from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import InstallPlanStep, InstallerContext
from . import generic, hunyuan_2, hunyuan_21, shap_e, stable_fast_3d, trellis, trellis_2, triposr, zero123plus


INSTALLER_MAP = {
    "shap-e": shap_e.build_plan,
    "hunyuan3d-2": hunyuan_2.build_plan,
    "hunyuan3d-2.1": hunyuan_21.build_plan,
    "trellis": trellis.build_plan,
    "trellis.2": trellis_2.build_plan,
    "stable-fast-3d": stable_fast_3d.build_plan,
    "triposr": triposr.build_plan,
    "zero123++": zero123plus.build_plan,
}


def build_install_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    builder = INSTALLER_MAP.get(entry.id, generic.build_plan)
    return builder(entry, context)
