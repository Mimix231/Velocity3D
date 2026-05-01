from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import (
    InstallPlanStep,
    InstallerContext,
    PackageGroup,
    clone_step,
    huggingface_download_step,
    manual,
    note,
    pip_install_group,
    torch_or_reuse_step,
)


TORCH_RUNTIME = PackageGroup(
    name="TRELLIS.2 CUDA 12.4 PyTorch runtime",
    packages=("torch", "torchvision", "torchaudio"),
    index_url="https://download.pytorch.org/whl/cu130",
)

BOOTSTRAP_TOOLS = PackageGroup(
    name="TRELLIS.2 bootstrap tools",
    packages=(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
        "pybind11",
    ),
)

IMAGE_RUNTIME = PackageGroup(
    name="TRELLIS.2 image runtime",
    packages=(
        "accelerate",
        "easydict",
        "einops",
        "huggingface-hub",
        "imageio",
        "imageio-ffmpeg",
        "kornia",
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "safetensors",
        "scipy",
        "timm",
        "tqdm",
        "transformers",
        "zstandard",
    ),
)

GEOMETRY_RUNTIME = PackageGroup(
    name="TRELLIS.2 geometry helpers",
    packages=(
        "lpips",
        "pandas",
        "trimesh",
        "pygltflib",
        "xatlas",
        "omegaconf",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    ),
    optional=True,
)


def _intro_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan = [
        note(
            "TRELLIS.2 is a high-end image-to-3D stack with CUDA 12.4 native components and the o-voxel submodule. "
            "Velocity3D can prepare the repo and pure Python layers, but native compilation remains explicit."
        )
    ]
    if context.current_python_series == "3.13":
        plan.append(
            note(
                "Python 3.13 is useful for the main Velocity3D backend, but TRELLIS.2 upstream native extensions "
                "are safer in a Python 3.10-3.12 WSL2/Linux environment."
            )
        )
    return plan


def _torch_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    return [
        torch_or_reuse_step(
            entry.name,
            context,
            packages=TORCH_RUNTIME.packages,
            index_url=TORCH_RUNTIME.index_url or "https://download.pytorch.org/whl/cu130",
            reuse_reason="TRELLIS.2 can reuse the backend torch stack for catalog probing before native setup.",
        )
    ]


def _safe_dependency_phase() -> list[InstallPlanStep]:
    return [
        pip_install_group(BOOTSTRAP_TOOLS),
        pip_install_group(IMAGE_RUNTIME),
        pip_install_group(GEOMETRY_RUNTIME),
    ]


def _native_setup_phase() -> list[InstallPlanStep]:
    return [
        huggingface_download_step(
            "Download TRELLIS.2 checkpoint to BASE_DIR/HuggingFace",
            "microsoft/TRELLIS.2-4B",
        ),
        manual("Prepare WSL2/Linux with CUDA 12.4, compiler tooling, and an NVIDIA driver visible to PyTorch"),
        manual("Initialize and verify the o-voxel submodule under BASE_DIR/models/trellis2/o-voxel"),
        manual("Run the upstream TRELLIS.2 setup flow inside BASE_DIR/models/trellis2"),
        manual("Build or install o-voxel and any custom voxel/rasterization CUDA extensions required by TRELLIS.2"),
        manual("Download the TRELLIS.2-4B checkpoint and verify the provider can resolve it before generation"),
    ]


def _validation_phase() -> list[InstallPlanStep]:
    return [
        note(
            "Velocity3D validates TRELLIS.2 by checking the trellis2 package, o-voxel repo path, cv2, torch, and "
            "trimesh before allowing image generation."
        ),
        note(
            "If TRELLIS.2 is still not ready after this automated phase, the unresolved work is native CUDA setup "
            "or checkpoint placement, not the model browser state."
        ),
    ]


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []
    plan.extend(clone_step(entry, context))
    plan.extend(_intro_phase(entry, context))
    plan.extend(_torch_phase(entry, context))
    plan.extend(_safe_dependency_phase())
    plan.extend(_native_setup_phase())
    plan.extend(_validation_phase())
    return plan
