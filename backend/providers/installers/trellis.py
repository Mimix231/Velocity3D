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
    name="TRELLIS compatible PyTorch runtime",
    packages=("torch", "torchvision", "torchaudio"),
    index_url="https://download.pytorch.org/whl/cu130",
)

BOOTSTRAP_TOOLS = PackageGroup(
    name="TRELLIS bootstrap tools",
    packages=(
        "wheel",
        "setuptools",
        "packaging",
        "ninja",
    ),
)

PURE_PYTHON_RUNTIME = PackageGroup(
    name="TRELLIS pure Python runtime",
    packages=(
        "accelerate",
        "easydict",
        "einops",
        "huggingface-hub",
        "imageio",
        "imageio-ffmpeg",
        "numpy",
        "omegaconf",
        "onnxruntime",
        "open3d",
        "opencv-python-headless",
        "Pillow",
        "pyvista",
        "pymeshfix",
        "safetensors",
        "scipy",
        "tqdm",
        "trimesh",
        "transformers",
    ),
)

EXPORT_RUNTIME = PackageGroup(
    name="TRELLIS export helpers",
    packages=(
        "igraph",
        "pygltflib",
        "xatlas",
        "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
    ),
    optional=True,
)


def _intro_phase(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan = [
        note(
            "TRELLIS is a Linux-first CUDA stack with submodules and native extensions. Velocity3D handles the "
            "repo bootstrap and safe Python packages here, then marks the remaining native setup explicitly."
        )
    ]
    if context.current_python_series == "3.13":
        plan.append(
            note(
                "Python 3.13 can run Velocity3D, but upstream TRELLIS native packages are usually curated for "
                "older Python/CUDA combinations. Keep TRELLIS in WSL2/Linux if native extension builds fail."
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
            reuse_reason="TRELLIS can reuse the backend torch stack for catalog probing and pure-Python imports.",
        )
    ]


def _safe_dependency_phase() -> list[InstallPlanStep]:
    return [
        pip_install_group(BOOTSTRAP_TOOLS),
        pip_install_group(PURE_PYTHON_RUNTIME),
        pip_install_group(EXPORT_RUNTIME),
    ]


def _native_setup_phase() -> list[InstallPlanStep]:
    return [
        huggingface_download_step(
            "Download TRELLIS image checkpoint to BASE_DIR/HuggingFace",
            "microsoft/TRELLIS-image-large",
        ),
        huggingface_download_step(
            "Download TRELLIS text checkpoint to BASE_DIR/HuggingFace",
            "microsoft/TRELLIS-text-xlarge",
        ),
        manual(
            "Prepare a Linux or WSL2 environment with an NVIDIA driver, matching CUDA toolkit, and compiler toolchain"
        ),
        manual(
            "Run the upstream TRELLIS setup flow inside BASE_DIR/models/trellis after reviewing the CUDA extension list"
        ),
        manual(
            "Build or install TRELLIS native dependencies such as sparse convolution, rendering, and custom geometry kernels"
        ),
        manual(
            "Download the TRELLIS image/text checkpoints and confirm they are reachable by the backend runtime"
        ),
    ]


def _validation_phase() -> list[InstallPlanStep]:
    return [
        note(
            "Velocity3D only marks TRELLIS generation-ready after the repo exists, the trellis package is present, "
            "and the shared imageio/trimesh runtime imports."
        ),
        note(
            "If TRELLIS import fails after the automated phase, the remaining failure is expected to be a native "
            "CUDA extension or checkpoint path problem, not a general Velocity3D installer failure."
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
