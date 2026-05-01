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
    name="Zero123++ compatible PyTorch runtime",
    packages=("torch", "torchvision", "torchaudio"),
    index_url="https://download.pytorch.org/whl/cu130",
)

DIFFUSION_REQS = PackageGroup(
    name="Zero123++ diffusion runtime",
    packages=(
        "diffusers",
        "transformers",
        "accelerate",
        "huggingface-hub",
        "safetensors",
    ),
)

IMAGE_REQS = PackageGroup(
    name="Zero123++ image runtime",
    packages=(
        "Pillow",
        "numpy",
        "opencv-contrib-python",
        "einops",
        "tqdm",
    ),
)

MULTIVIEW_REQS = PackageGroup(
    name="Zero123++ multiview helpers",
    packages=(
        "omegaconf",
        "scipy",
        "scikit-image",
        "imageio",
        "fire",
        '"altair<5"',
        "streamlit",
        "gradio",
    ),
)

SEGMENT_ANYTHING_REQ = PackageGroup(
    name="Zero123++ Segment Anything helper",
    packages=("git+https://github.com/facebookresearch/segment-anything.git",),
    optional=True,
)


def _torch_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    return [
        torch_or_reuse_step(
            entry.name,
            context,
            packages=TORCH_FALLBACK.packages,
            index_url=TORCH_FALLBACK.index_url or "https://download.pytorch.org/whl/cu130",
            reuse_reason="Zero123++ is a multiview assistant and can reuse the backend torch stack.",
        )
    ]


def _runtime_phase(context: InstallerContext) -> list[InstallPlanStep]:
    plan = [
        note(
            "Zero123++ is installed as a multiview assistant, not a final mesh generator. "
            "Velocity3D keeps it out of the generator selector until a mesh provider consumes its views."
        ),
        pip_install_group(DIFFUSION_REQS),
        pip_install_group(IMAGE_REQS),
    ]
    plan.extend(
        requirements_or_curated(
            "Install Zero123++ upstream requirements",
            context,
            (MULTIVIEW_REQS, SEGMENT_ANYTHING_REQ),
            python313_note=(
                "Python 3.13 uses the curated Zero123++ dependency set instead of upstream requirements.txt. "
                "The assistant needs multiview diffusion plus optional mask helpers, so Velocity3D avoids legacy pins while still attempting Segment Anything."
            ),
        )
    )
    return plan


def _validation_notes() -> list[InstallPlanStep]:
    return [
        huggingface_download_step(
            "Download Zero123++ multiview checkpoint to BASE_DIR/HuggingFace",
            "sudo-ai/zero123plus-v1.2",
        ),
        note("Velocity3D validates Zero123++ as library-only when the repo exists with diffusers-support and examples."),
        note("This installer does not make Zero123++ a mesh generator. It prepares it for a later multiview preview stage."),
    ]


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []
    plan.extend(clone_step(entry, context))
    plan.extend(_torch_phase(entry, context))
    plan.extend(_runtime_phase(context))
    plan.extend(_validation_notes())
    return plan
