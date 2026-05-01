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
    name="TripoSR compatible PyTorch runtime",
    packages=("torch", "torchvision", "torchaudio"),
    index_url="https://download.pytorch.org/whl/cu130",
)

PRE_REQS = PackageGroup(
    name="TripoSR preflight packages",
    packages=(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
    ),
)

MESH_EXPORT_REQS = PackageGroup(
    name="TripoSR mesh export helpers",
    packages=(
        "trimesh",
        "pygltflib",
        "Pillow",
        "imageio",
        "xatlas",
        "moderngl",
    ),
)

CURATED_MODEL_REQS = PackageGroup(
    name="TripoSR curated model runtime",
    packages=(
        "accelerate",
        "einops",
        "huggingface-hub",
        "omegaconf",
        "safetensors",
        "transformers",
        "tqdm",
    ),
)

CURATED_IMAGE_REQS = PackageGroup(
    name="TripoSR curated image runtime",
    packages=(
        "numpy",
        "opencv-python",
        "rembg",
        "scikit-image",
    ),
)

TORCHMCUBES_REQ = PackageGroup(
    name="TripoSR torch marching cubes extension",
    packages=("git+https://github.com/tatsy/torchmcubes.git",),
    optional=True,
)


def _torch_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    return [
        torch_or_reuse_step(
            entry.name,
            context,
            packages=TORCH_FALLBACK.packages,
            index_url=TORCH_FALLBACK.index_url or "https://download.pytorch.org/whl/cu130",
            reuse_reason="TripoSR runs as an external CLI and can reuse the backend torch stack.",
        )
    ]


def _runtime_phase(context: InstallerContext) -> list[InstallPlanStep]:
    plan = [
        note(
            "TripoSR is installed as a repo-local CLI pipeline. Velocity3D keeps it separate because its image "
            "preprocessing and mesh extraction dependencies can fail independently from other providers."
        ),
        pip_install_group(PRE_REQS),
        pip_install_group(MESH_EXPORT_REQS),
    ]
    plan.extend(
        requirements_or_curated(
            "Install TripoSR upstream requirements",
            context,
            (CURATED_MODEL_REQS, CURATED_IMAGE_REQS, TORCHMCUBES_REQ),
            python313_note=(
                "Python 3.13 uses the curated TripoSR dependency set instead of upstream requirements.txt. "
                "Velocity3D still attempts the torchmcubes extension as an optional step because upstream TripoSR uses it for mesh extraction."
            ),
        )
    )
    return plan


def _validation_notes() -> list[InstallPlanStep]:
    return [
        huggingface_download_step(
            "Download TripoSR checkpoint to BASE_DIR/HuggingFace",
            "stabilityai/TripoSR",
        ),
        note("Velocity3D validates TripoSR by checking for run.py or the tsr package in the cloned repo."),
        note("Generated OBJ or PLY assets are converted back to GLB by the Velocity3D provider adapter."),
    ]


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []
    plan.extend(clone_step(entry, context))
    plan.extend(_torch_phase(entry, context))
    plan.extend(_runtime_phase(context))
    plan.extend(_validation_notes())
    return plan
