from __future__ import annotations

from backend.providers.catalog import CatalogEntry
from backend.providers.installers.common import (
    InstallPlanStep,
    InstallerContext,
    PackageGroup,
    action,
    clone_step,
    download_file_step,
    huggingface_download_step,
    note,
    pip_install_group,
    torch_step_for_runtime,
)
from backend.providers.helpers import huggingface_root


BUILD_TOOLS = PackageGroup(
    name="Hunyuan 2.1 build tools",
    packages=("ninja==1.11.1.1", "pybind11==2.13.4"),
)

CORE_RUNTIME = PackageGroup(
    name="Hunyuan 2.1 core runtime",
    packages=(
        "transformers==4.46.0",
        "diffusers==0.30.0",
        "accelerate==1.1.1",
        "pytorch-lightning==1.9.5",
        "huggingface-hub==0.30.2",
        "torchmetrics==1.6.0",
    ),
)

PY313_CORE_RUNTIME = PackageGroup(
    name="Hunyuan 2.1 Python 3.13 core runtime",
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
    name="Hunyuan 2.1 scientific and image runtime",
    packages=(
        '"numpy>=2.1,<2.2"',
        "scipy==1.14.1",
        "einops==0.8.0",
        "opencv-python==4.10.0.84",
        "scikit-image==0.24.0",
        "omegaconf==2.3.0",
        "pyyaml==6.0.2",
        "tqdm==4.66.5",
    ),
)

PY313_SCIENTIFIC_RUNTIME = PackageGroup(
    name="Hunyuan 2.1 Python 3.13 scientific and image runtime",
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
    name="Hunyuan 2.1 shape inference extras",
    packages=(
        '"safetensors>=0.5,<0.6"',
        "trimesh==4.4.7",
        "timm",
        "torchdiffeq",
    ),
)

PY313_SHAPE_EXTRAS = PackageGroup(
    name="Hunyuan 2.1 Python 3.13 shape inference extras",
    packages=(
        '"safetensors>=0.5"',
        '"trimesh>=4.4.7"',
        "timm",
        "torchdiffeq",
    ),
)

PAINT_UV_RUNTIME = PackageGroup(
    name="Hunyuan 2.1 PBR paint UV/export runtime",
    packages=(
        '"pygltflib>=1.16.3"',
        '"xatlas>=0.0.9"',
        '"pymeshlab>=2025.7.post1"',
        '"fast-simplification>=0.1.13"',
    ),
)

OPTIONAL_TEXTURE_AND_EXPORT_EXTRAS = PackageGroup(
    name="Hunyuan 2.1 optional texture/export extras",
    packages=(
        '"imageio>=2.36.0"',
        '"pandas>=2.2.2"',
        '"configargparse>=1.7"',
        '"psutil>=6.0.0"',
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
            "Install Velocity3D Hunyuan 2.1 PyTorch 2.10 CUDA 13.0 stack",
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
                "Velocity3D installs the Hunyuan 2.1 core runtime with Python 3.13-safe minimum constraints. "
                "This avoids uninstalling compiled packages already loaded by the running backend, such as numpy."
            ),
            pip_install_group(BUILD_TOOLS),
            pip_install_group(PY313_CORE_RUNTIME),
            pip_install_group(PY313_SCIENTIFIC_RUNTIME),
            pip_install_group(PY313_SHAPE_EXTRAS),
            pip_install_group(PAINT_UV_RUNTIME),
            pip_install_group(OPTIONAL_TEXTURE_AND_EXPORT_EXTRAS),
        ]

    return [
        note(
            "Velocity3D installs the Hunyuan 2.1 generation and paint runtime instead of the full upstream "
            "requirements.txt stack. This skips demo and legacy scientific pins that are not required by the "
            "current provider path."
        ),
        pip_install_group(BUILD_TOOLS),
        pip_install_group(CORE_RUNTIME),
        pip_install_group(SCIENTIFIC_RUNTIME),
        pip_install_group(SHAPE_EXTRAS),
        pip_install_group(PAINT_UV_RUNTIME),
        pip_install_group(OPTIONAL_TEXTURE_AND_EXPORT_EXTRAS),
    ]


def _paint_runtime_phase() -> list[InstallPlanStep]:
    return [
        note(
            "Velocity3D now uses the real Hunyuan3D-Paint 2.1 texture pipeline for Hunyuan image generations. "
            "The native CUDA rasterizer must be built against the same CUDA major/minor as the backend PyTorch wheel. "
            "RealESRGAN is treated as optional because Velocity3D can run the paint pass without super-resolution."
        ),
        action(
            "Patch Hunyuan 2.1 paint, DifferentiableRenderer, and custom_rasterizer sources",
            "patch_hunyuan21_paint_sources",
        ),
        InstallPlanStep(
            label="Build Hunyuan 2.1 paint CUDA rasterizer",
            action="build_hunyuan21_custom_rasterizer",
        ),
        InstallPlanStep(
            label="Build Hunyuan 2.1 mesh inpaint processor",
            action="build_hunyuan21_mesh_inpaint_processor",
        ),
        action(
            "Materialize Velocity3D-owned Hunyuan 2.1 paint runtime",
            "materialize_hunyuan21_paint_runtime",
        ),
    ]


def _post_install_phase() -> list[InstallPlanStep]:
    return [
        huggingface_download_step(
            "Download Hunyuan 2.1 shape weights to BASE_DIR/HuggingFace",
            "tencent/Hunyuan3D-2.1",
            allow_patterns=("hunyuan3d-dit-v2-1/*", "hunyuan3d-vae-v2-1/*"),
            local_dir=str(huggingface_root() / "hy3dgen" / "tencent" / "Hunyuan3D-2.1"),
        ),
        huggingface_download_step(
            "Download Hunyuan 2.1 DINO image encoder cache to BASE_DIR/HuggingFace",
            "facebook/dinov2-large",
        ),
        huggingface_download_step(
            "Download Hunyuan 2.1 PBR paint weights to BASE_DIR/HuggingFace",
            "tencent/Hunyuan3D-2.1",
            allow_patterns=("hunyuan3d-paintpbr-v2-1/*",),
        ),
        huggingface_download_step(
            "Download Hunyuan 2.1 paint DINO giant cache to BASE_DIR/HuggingFace",
            "facebook/dinov2-giant",
        ),
        download_file_step(
            "Download RealESRGAN optional paint upscaler weight",
            "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
            str(huggingface_root().parent / "models" / "hunyuan3d-2.1" / "hy3dpaint" / "ckpt" / "RealESRGAN_x4plus.pth"),
            optional=True,
        ),
        action(
            "Patch the cloned Hunyuan 2.1 package for Velocity3D shape-only imports",
            "patch_hunyuan21_shape_init",
        ),
        note(
            "Velocity3D considers Hunyuan 2.1 shape generation ready when the repo exists, hy3dshape is present, "
            "and the shared torch/diffusers/transformers/trimesh modules import from the backend runtime. "
            "The texture pass additionally requires pymeshlab, fast-simplification, xatlas, pygltflib, "
            "and the paint CUDA rasterizer."
        ),
    ]


def build_plan(entry: CatalogEntry, context: InstallerContext) -> list[InstallPlanStep]:
    plan: list[InstallPlanStep] = []
    if context.current_python_series not in {"3.10", "3.11", "3.12", "3.13"}:
        supported = ", ".join(entry.supported_python or ("3.10", "3.11", "3.12", "3.13"))
        raise ValueError(
            f"{entry.name} is not curated for backend Python {context.current_python_version}. "
            f"Use Python {supported}, or point Velocity3D to a different backend venv."
        )

    plan.extend(clone_step(entry, context))
    plan.extend(_torch_phase(context))
    plan.extend(_shape_runtime_phase(context))
    plan.extend(_paint_runtime_phase())
    plan.extend(_post_install_phase())
    return plan
