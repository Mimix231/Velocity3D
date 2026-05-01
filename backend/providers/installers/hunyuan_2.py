from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import (
    InstallPlanStep,
    InstallerContext,
    PackageGroup,
    clone_step,
    download_file_step,
    huggingface_download_step,
    note,
    pip_install_group,
    torch_step_for_runtime,
)
from backend.providers.helpers import huggingface_root


BUILD_TOOLS = PackageGroup(
    name="Hunyuan 2.0 build tools",
    packages=("ninja==1.11.1.1", "pybind11==2.13.4"),
)

CORE_RUNTIME = PackageGroup(
    name="Hunyuan 2.0 shape runtime",
    packages=(
        "transformers",
        "diffusers",
        "accelerate",
        "pytorch-lightning",
        "huggingface-hub",
        "torchmetrics",
    ),
)

PY313_CORE_RUNTIME = PackageGroup(
    name="Hunyuan 2.0 Python 3.13 shape runtime",
    packages=(
        '"transformers>=4.46.0"',
        '"diffusers>=0.30.0"',
        '"accelerate>=1.1.1"',
        '"pytorch-lightning>=1.9.5"',
        '"huggingface-hub>=0.30.2"',
        '"torchmetrics>=1.6.0"',
    ),
)

SCIENTIFIC_RUNTIME = PackageGroup(
    name="Hunyuan 2.0 scientific runtime",
    packages=(
        '"numpy>=2.1,<2.2"',
        "scipy",
        "einops",
        "opencv-python",
        "scikit-image",
        "omegaconf",
        "pyyaml",
        "tqdm",
    ),
)

PY313_SCIENTIFIC_RUNTIME = PackageGroup(
    name="Hunyuan 2.0 Python 3.13 scientific runtime",
    packages=(
        '"numpy>=2.1"',
        '"scipy>=1.14.1"',
        '"einops>=0.8.0"',
        '"opencv-python>=4.10.0.84"',
        '"scikit-image>=0.24.0"',
        '"omegaconf>=2.3.0"',
        '"pyyaml>=6.0.2"',
        '"tqdm>=4.66.5"',
    ),
)

SHAPE_EXTRAS = PackageGroup(
    name="Hunyuan 2.0 shape inference extras",
    packages=(
        '"safetensors>=0.5,<0.6"',
        "trimesh",
        "timm",
        "torchdiffeq",
    ),
)

PY313_SHAPE_EXTRAS = PackageGroup(
    name="Hunyuan 2.0 Python 3.13 shape inference extras",
    packages=(
        '"safetensors>=0.5"',
        '"trimesh>=4.4.7"',
        "timm",
        "torchdiffeq",
    ),
)

OPTIONAL_TEXTURE_AND_EXPORT_EXTRAS = PackageGroup(
    name="Hunyuan 2.0 optional texture/export extras",
    packages=(
        '"pygltflib>=1.16.3"',
        '"xatlas>=0.0.9"',
        '"rembg>=2.0.65"',
        '"onnxruntime>=1.16.3"',
    ),
    optional=True,
)

def _torch_phase(context: InstallerContext) -> list[InstallPlanStep]:
    torch_packages = ("torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0")
    torch_index = "https://download.pytorch.org/whl/cu130"
    return [
        torch_step_for_runtime(
            "Install Velocity3D Hunyuan 2.0 PyTorch 2.10 CUDA 13.0 stack",
            context,
            supported_series={"3.10", "3.11", "3.12", "3.13"},
            upstream_packages=torch_packages,
            upstream_index_url=torch_index,
            fallback_packages=torch_packages,
            fallback_index_url=torch_index,
        )
    ]


def _shape_runtime_phase(context: InstallerContext) -> list[InstallPlanStep]:
    if context.current_python_series == "3.13":
        return [
            note(
                "Velocity3D uses the Hunyuan 2.0 core runtime with Python 3.13-safe minimum constraints. "
                "This avoids uninstalling compiled packages already loaded by the running backend, such as numpy."
            ),
            pip_install_group(BUILD_TOOLS),
            pip_install_group(PY313_CORE_RUNTIME),
            pip_install_group(PY313_SCIENTIFIC_RUNTIME),
            pip_install_group(PY313_SHAPE_EXTRAS),
            pip_install_group(OPTIONAL_TEXTURE_AND_EXPORT_EXTRAS),
        ]

    return [
        note(
            "Velocity3D uses the Hunyuan 2.0 generation and paint runtime while avoiding demo and legacy pins "
            "that are not required by the current provider path."
        ),
        pip_install_group(BUILD_TOOLS),
        pip_install_group(CORE_RUNTIME),
        pip_install_group(SCIENTIFIC_RUNTIME),
        pip_install_group(SHAPE_EXTRAS),
        pip_install_group(OPTIONAL_TEXTURE_AND_EXPORT_EXTRAS),
    ]


def _paint_runtime_phase() -> list[InstallPlanStep]:
    return [
        note(
            "Velocity3D now uses the real Hunyuan3D-Paint texture pipeline for Hunyuan image generations. "
            "The CUDA rasterizer is required; RealESRGAN is optional because the paint pass can run without the upscale stage."
        ),
        InstallPlanStep(
            label="Build Hunyuan 2.0 paint CUDA rasterizer",
            action="build_hunyuan20_custom_rasterizer",
        ),
        InstallPlanStep(
            label="Build Hunyuan 2.0 differentiable renderer",
            action="build_hunyuan20_differentiable_renderer",
            optional=True,
        ),
    ]


def _validation_notes(context: InstallerContext) -> list[InstallPlanStep]:
    notes = [
        huggingface_download_step(
            "Download Hunyuan 2.0 shape weights to BASE_DIR/HuggingFace",
            "tencent/Hunyuan3D-2",
            allow_patterns=("hunyuan3d-dit-v2-0/*", "hunyuan3d-vae-v2-0/*"),
            local_dir=str(huggingface_root() / "hy3dgen" / "tencent" / "Hunyuan3D-2"),
        ),
        huggingface_download_step(
            "Download Hunyuan 2.0 DINO image encoder cache to BASE_DIR/HuggingFace",
            "facebook/dinov2-large",
        ),
        huggingface_download_step(
            "Download Hunyuan 2.0 paint weights to BASE_DIR/HuggingFace",
            "tencent/Hunyuan3D-2",
            allow_patterns=("hunyuan3d-paint-v2-0/*", "hunyuan3d-delight-v2-0/*"),
        ),
        download_file_step(
            "Download RealESRGAN optional paint upscaler weight",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            str(huggingface_root().parent / "models" / "hunyuan3d-2" / "hy3dgen" / "texgen" / "ckpt" / "RealESRGAN_x4plus.pth"),
            optional=True,
        ),
        note(
            "Velocity3D considers Hunyuan 2.0 shape generation ready when the repo exists, hy3dgen is present, "
            "and core modules import from the backend runtime. The texture pass additionally requires the paint CUDA rasterizer."
        )
    ]
    if context.current_python_series == "3.13":
        notes.append(
            note(
                "Python 3.13 is newer than the Hunyuan 2.0 upstream-tested environment. "
                "This installer keeps modern wheel-compatible pins and avoids legacy numpy pins."
            )
        )
    return notes


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []
    plan.extend(clone_step(entry, context))
    plan.extend(_torch_phase(context))
    plan.extend(_shape_runtime_phase(context))
    plan.extend(_paint_runtime_phase())
    plan.extend(_validation_notes(context))
    return plan
