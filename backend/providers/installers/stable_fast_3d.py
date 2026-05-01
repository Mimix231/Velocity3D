from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import (
    InstallPlanStep,
    InstallerContext,
    PackageGroup,
    clone_step,
    huggingface_download_step,
    note,
    pip_install_group,
    requirements_or_curated,
    torch_or_reuse_step,
)


TORCH_FALLBACK = PackageGroup(
    name="Stable Fast 3D compatible PyTorch runtime",
    packages=("torch", "torchvision", "torchaudio"),
    index_url="https://download.pytorch.org/whl/cu130",
)

PRE_REQS = PackageGroup(
    name="Stable Fast 3D preflight packages",
    packages=(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
    ),
)

EXPORT_REQS = PackageGroup(
    name="Stable Fast 3D import/export helpers",
    packages=(
        "trimesh",
        "pygltflib",
        "Pillow",
    ),
)

CURATED_RUNTIME_REQS = PackageGroup(
    name="Stable Fast 3D curated runtime",
    packages=(
        "accelerate",
        "diffusers",
        "einops",
        "huggingface-hub",
        "imageio",
        "jaxtyping",
        "numpy",
        "opencv-python",
        "open_clip_torch",
        "rembg",
        "safetensors",
        "scipy",
        "transformers",
        "tqdm",
    ),
)

CURATED_MESH_REQS = PackageGroup(
    name="Stable Fast 3D mesh utilities",
    packages=(
        "gpytoolbox",
        "pynanoinstantmeshes",
        "xatlas",
        "pymeshlab",
        "omegaconf",
    ),
    optional=True,
)

LOCAL_EXTENSION_REQS = PackageGroup(
    name="Stable Fast 3D repo-local texture baker and UV unwrapper",
    packages=("./texture_baker/", "./uv_unwrapper/"),
    optional=True,
)


def _torch_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    return [
        torch_or_reuse_step(
            entry.name,
            context,
            packages=TORCH_FALLBACK.packages,
            index_url=TORCH_FALLBACK.index_url or "https://download.pytorch.org/whl/cu130",
            reuse_reason="Stable Fast 3D is invoked as an external CLI and can reuse the backend torch stack.",
        )
    ]


def _runtime_phase(context: InstallerContext) -> list[InstallPlanStep]:
    plan = [
        note(
            "Stable Fast 3D runs as a repo-local CLI. Velocity3D prepares a wheel-friendly runtime before touching "
            "upstream dependencies so one stale package pin does not break the whole model library."
        ),
        pip_install_group(PRE_REQS),
        pip_install_group(EXPORT_REQS),
    ]
    plan.extend(
        requirements_or_curated(
            "Install Stable Fast 3D upstream requirements",
            context,
            (CURATED_RUNTIME_REQS, CURATED_MESH_REQS, LOCAL_EXTENSION_REQS),
            python313_note=(
                "Python 3.13 uses the curated Stable Fast 3D dependency set instead of upstream requirements.txt. "
                "This avoids forcing old package pins into source builds while keeping the CLI, texture, and UV dependencies present."
            ),
        )
    )
    return plan


def _validation_notes() -> list[InstallPlanStep]:
    return [
        huggingface_download_step(
            "Download Stable Fast 3D checkpoint to BASE_DIR/HuggingFace",
            "stabilityai/stable-fast-3d",
        ),
        note(
            "Velocity3D validates Stable Fast 3D by checking for run.py or the sf3d package in the cloned repo, "
            "plus the shared torch/trimesh runtime."
        ),
        note(
            "Stable Fast 3D may download weights during first inference. That is expected and should happen inside "
            "the provider runtime, not the UI thread."
        ),
    ]


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []
    plan.extend(clone_step(entry, context))
    plan.extend(_torch_phase(entry, context))
    plan.extend(_runtime_phase(context))
    plan.extend(_validation_notes())
    return plan
