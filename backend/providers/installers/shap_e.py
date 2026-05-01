from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import (
    InstallPlanStep,
    InstallerContext,
    PackageGroup,
    clone_step,
    keep_existing_torch_note,
    note,
    pip_install_command,
    pip_install_group,
)


TORCH_FALLBACK = PackageGroup(
    name="Shap-E compatible PyTorch runtime",
    packages=("torch", "torchvision", "torchaudio"),
    index_url="https://download.pytorch.org/whl/cu130",
)

RUNTIME_HELPERS = PackageGroup(
    name="Shap-E runtime helpers",
    packages=(
        "blobfile",
        "fire",
        "filelock",
        "humanize",
        "Pillow",
        "numpy",
        "scipy",
        "scikit-image",
        "tqdm",
        "trimesh",
        "matplotlib",
    ),
)

EXPORT_HELPERS = PackageGroup(
    name="Shap-E export helpers",
    packages=(
        "pygltflib",
        "networkx",
    ),
)

CLIP_HELPER = PackageGroup(
    name="Shap-E CLIP helper",
    packages=("git+https://github.com/openai/CLIP.git",),
    optional=True,
)


def _torch_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    if context.torch_stack_ready:
        return [
            keep_existing_torch_note(
                entry.name,
                context,
                "Shap-E does not need a stricter torch pin in Velocity3D.",
            )
        ]
    return [pip_install_group(TORCH_FALLBACK)]


def _dependency_phase() -> list[InstallPlanStep]:
    return [
        pip_install_group(RUNTIME_HELPERS),
        pip_install_group(EXPORT_HELPERS),
        pip_install_group(CLIP_HELPER),
    ]


def _repo_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan = clone_step(entry, context)
    plan.append(
        InstallPlanStep(
            label="Install Shap-E package from the cloned repo",
            command=pip_install_command(("-e", ".")),
        )
    )
    return plan


def _validation_notes() -> list[InstallPlanStep]:
    return [
        note(
            "Velocity3D validates Shap-E by checking the shap_e Python module plus torch and trimesh. "
            "Weights are downloaded lazily by the provider when generation runs."
        ),
        note(
            "If Shap-E fails at generation time, the likely causes are missing model weights, insufficient VRAM, "
            "or a torch/CUDA mismatch in the shared backend environment."
        ),
    ]


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = [
        note(
            "Velocity3D installs Shap-E as the default text-to-3D baseline. "
            "It is kept separate from image providers because it is repo-editable code plus lazy checkpoint downloads."
        )
    ]
    plan.extend(_torch_phase(entry, context))
    plan.extend(_dependency_phase())
    plan.extend(_repo_phase(entry, context))
    plan.extend(_validation_notes())
    return plan
