from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

LibraryMode = Literal["text", "image", "multiview"]
Role = Literal["generator", "assistant"]


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    name: str
    family: str
    role: Role
    summary: str
    description: str
    selection_modes: tuple[Literal["text", "image"], ...]
    library_modes: tuple[LibraryMode, ...]
    recommended: bool
    repo_url: str | None
    docs_url: str | None
    huggingface_url: str | None
    license_name: str | None
    vram_hint: str | None
    size_hint: str | None
    platform_note: str | None
    install_steps: tuple[str, ...]
    vendor_dir_name: str | None
    ready_paths: tuple[str, ...]
    core_deps: tuple[str, ...]
    provider_key: str | None
    preferred_python: str | None = None
    supported_python: tuple[str, ...] = ()
    python_note: str | None = None


DEFAULT_TEXT_MODEL_ID = "shap-e"
DEFAULT_IMAGE_MODEL_ID = "hunyuan3d-2.1"


MODEL_CATALOG: tuple[CatalogEntry, ...] = (
    CatalogEntry(
        id="shap-e",
        name="Shap-E",
        family="shap-e",
        role="generator",
        summary="OpenAI text-to-3D diffusion baseline.",
        description="Generates meshes from text prompts. Best kept as the default text-only provider in this app.",
        selection_modes=("text",),
        library_modes=("text",),
        recommended=True,
        repo_url="https://github.com/openai/shap-e",
        docs_url="https://github.com/openai/shap-e",
        huggingface_url=None,
        license_name="MIT",
        vram_hint="6 GB+ recommended",
        size_hint="~1.2 GB weights",
        platform_note="Works in the current Python backend when torch and shap-e are installed.",
        install_steps=(
            "git clone https://github.com/openai/shap-e BASE_DIR\\models\\shap-e",
            "cd BASE_DIR\\models\\shap-e",
            "python -m pip install -e .",
        ),
        vendor_dir_name="shap-e",
        ready_paths=(),
        core_deps=("torch", "shap_e", "trimesh"),
        provider_key="shap-e",
    ),
    CatalogEntry(
        id="hunyuan3d-2",
        name="Hunyuan3D-2",
        family="hunyuan3d",
        role="generator",
        summary="Tencent image-to-3D shape generation.",
        description="Image-conditioned mesh generation from the Hunyuan 2.0 stack with optional Hunyuan3D-Paint texturing.",
        selection_modes=("image",),
        library_modes=("image",),
        recommended=False,
        repo_url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
        docs_url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2",
        huggingface_url="https://huggingface.co/tencent/Hunyuan3D-2",
        license_name="Tencent Hunyuan Community",
        vram_hint="10 GB+ recommended",
        size_hint="7-8 GB shape files, plus paint weights when texture pass is enabled",
        platform_note="Windows is supported by upstream, but the paint path requires a compiled CUDA rasterizer built against CUDA Toolkit 13.0 for torch+cu130.",
        install_steps=(
            "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2 BASE_DIR\\models\\hunyuan3d-2",
            "cd BASE_DIR\\models\\hunyuan3d-2",
            "python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130",
            "python -m pip install -r requirements.txt",
            "python -m pip install -e hy3dgen\\texgen\\custom_rasterizer",
        ),
        vendor_dir_name="hunyuan3d-2",
        ready_paths=("hy3dgen",),
        core_deps=("torch", "transformers", "diffusers", "trimesh"),
        provider_key="hunyuan3d-2",
        preferred_python="3.10",
        supported_python=("3.10", "3.11", "3.12", "3.13"),
        python_note=(
            "Velocity3D targets the shared torch 2.10 CUDA 13.0 runtime for native extension builds."
        ),
    ),
    CatalogEntry(
        id="hunyuan3d-2.1",
        name="Hunyuan3D-2.1",
        family="hunyuan3d",
        role="generator",
        summary="Tencent production-ready image-to-3D shape model.",
        description="Recommended image provider for this app. Uses Hunyuan 2.1 shape generation and the real Hunyuan3D-Paint PBR texture stage when AI textures are enabled.",
        selection_modes=("image",),
        library_modes=("image",),
        recommended=True,
        repo_url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
        docs_url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
        huggingface_url="https://huggingface.co/tencent/Hunyuan3D-2.1",
        license_name="Tencent Hunyuan Community",
        vram_hint="10 GB+ for shape, 21 GB+ for paint, ~29 GB for shape plus texture",
        size_hint="~7.4 GB shape weights, plus ~2B paint weights",
        platform_note="The paint path requires a compiled CUDA rasterizer built with CUDA Toolkit 13.0 to match torch+cu130. Velocity3D skips remesh and can skip RealESRGAN to reduce Windows setup friction.",
        install_steps=(
            "git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 BASE_DIR\\models\\hunyuan3d-2.1",
            "cd BASE_DIR\\models\\hunyuan3d-2.1",
            "Velocity3D installs torch==2.10.0, torchvision==0.25.0, torchaudio==2.10.0 from the CUDA 13.0 wheel index.",
            "python -m pip install -r requirements.txt",
            "Velocity3D installs pymeshlab/fast-simplification/xatlas/pygltflib for Hunyuan3D-Paint remesh, UV wrapping, and export.",
            "Velocity3D builds hy3dpaint\\custom_rasterizer and DifferentiableRenderer\\mesh_inpaint_processor in the active backend env.",
        ),
        vendor_dir_name="hunyuan3d-2.1",
        ready_paths=("hy3dshape",),
        core_deps=("torch", "transformers", "diffusers", "trimesh"),
        provider_key="hunyuan3d-2.1",
        preferred_python="3.10",
        supported_python=("3.10", "3.11", "3.12", "3.13"),
        python_note=(
            "Velocity3D targets torch 2.10.0+cu130 so native paint extensions build against CUDA Toolkit 13.0."
        ),
    ),
    CatalogEntry(
        id="trellis",
        name="TRELLIS",
        family="trellis",
        role="generator",
        summary="Microsoft structured latent image-to-3D.",
        description="Official TRELLIS image and text generation adapters with GLB export. Image-conditioned generation is still the stronger path.",
        selection_modes=("text", "image"),
        library_modes=("image", "text"),
        recommended=False,
        repo_url="https://github.com/microsoft/TRELLIS",
        docs_url="https://github.com/microsoft/TRELLIS",
        huggingface_url="https://huggingface.co/microsoft/TRELLIS-image-large",
        license_name="MIT",
        vram_hint="16 GB+ NVIDIA GPU",
        size_hint="Large checkpoints + compiled deps",
        platform_note="Upstream documents Linux first. Windows is not a reliable target for the full stack.",
        install_steps=(
            "git clone --recurse-submodules https://github.com/microsoft/TRELLIS BASE_DIR\\models\\trellis",
            "cd BASE_DIR\\models\\trellis",
            "Follow setup.sh from the upstream README for your CUDA toolchain.",
        ),
        vendor_dir_name="trellis",
        ready_paths=("trellis",),
        core_deps=("torch", "imageio", "trimesh"),
        provider_key="trellis",
    ),
    CatalogEntry(
        id="trellis.2",
        name="TRELLIS.2",
        family="trellis",
        role="generator",
        summary="Microsoft high-fidelity image-to-3D with PBR output.",
        description="The newer TRELLIS.2 image pipeline. This app exposes the official image-to-GLB path when its native dependencies are available.",
        selection_modes=("image",),
        library_modes=("image",),
        recommended=False,
        repo_url="https://github.com/microsoft/TRELLIS.2",
        docs_url="https://github.com/microsoft/TRELLIS.2",
        huggingface_url="https://huggingface.co/microsoft/TRELLIS.2-4B",
        license_name="MIT",
        vram_hint="24 GB+ NVIDIA GPU",
        size_hint="4B checkpoint plus native deps",
        platform_note="Upstream is Linux-only at the moment.",
        install_steps=(
            "git clone -b main --recursive https://github.com/microsoft/TRELLIS.2 BASE_DIR\\models\\trellis2",
            "cd BASE_DIR\\models\\trellis2",
            "Follow setup.sh from the upstream README for CUDA 12.4 and the required native packages.",
        ),
        vendor_dir_name="trellis2",
        ready_paths=("trellis2", "o-voxel"),
        core_deps=("torch", "imageio", "cv2", "trimesh"),
        provider_key="trellis.2",
    ),
    CatalogEntry(
        id="stable-fast-3d",
        name="Stable Fast 3D",
        family="sf3d",
        role="generator",
        summary="Fast single-image mesh reconstruction.",
        description="Runs the official SF3D CLI on a locally cloned repo and imports the generated asset back into Velocity3D.",
        selection_modes=("image",),
        library_modes=("image",),
        recommended=False,
        repo_url="https://github.com/Stability-AI/stable-fast-3d",
        docs_url="https://github.com/Stability-AI/stable-fast-3d",
        huggingface_url=None,
        license_name="Stability AI license",
        vram_hint="~6 GB for single-image inference",
        size_hint="Repo clone plus weights on first run",
        platform_note="Windows is experimental upstream.",
        install_steps=(
            "git clone https://github.com/Stability-AI/stable-fast-3d BASE_DIR\\models\\stable-fast-3d",
            "cd BASE_DIR\\models\\stable-fast-3d",
            "python -m pip install -r requirements.txt",
        ),
        vendor_dir_name="stable-fast-3d",
        ready_paths=("run.py", "sf3d"),
        core_deps=("torch", "trimesh"),
        provider_key="stable-fast-3d",
    ),
    CatalogEntry(
        id="triposr",
        name="TripoSR",
        family="triposr",
        role="generator",
        summary="Fast feedforward image-to-3D reconstruction.",
        description="Runs the official TripoSR CLI from a locally cloned repo and converts the generated asset to GLB if needed.",
        selection_modes=("image",),
        library_modes=("image",),
        recommended=False,
        repo_url="https://github.com/VAST-AI-Research/TripoSR",
        docs_url="https://github.com/VAST-AI-Research/TripoSR",
        huggingface_url=None,
        license_name="MIT",
        vram_hint="~6 GB for single-image inference",
        size_hint="Repo clone plus weights on first run",
        platform_note="Watch the upstream torchmcubes notes if CUDA is mis-matched.",
        install_steps=(
            "git clone https://github.com/VAST-AI-Research/TripoSR BASE_DIR\\models\\triposr",
            "cd BASE_DIR\\models\\triposr",
            "python -m pip install -r requirements.txt",
        ),
        vendor_dir_name="triposr",
        ready_paths=("run.py", "tsr"),
        core_deps=("torch", "trimesh"),
        provider_key="triposr",
    ),
    CatalogEntry(
        id="zero123++",
        name="Zero123++",
        family="zero123++",
        role="assistant",
        summary="Consistent single-image multiview synthesis.",
        description="Useful as a multiview preprocessor, but not a final mesh generator on its own. The app exposes it in the library rather than the generator selector.",
        selection_modes=(),
        library_modes=("multiview",),
        recommended=False,
        repo_url="https://github.com/SUDO-AI-3D/zero123plus",
        docs_url="https://github.com/SUDO-AI-3D/zero123plus",
        huggingface_url="https://huggingface.co/sudo-ai/zero123plus-v1.2",
        license_name="Apache-2.0 code, CC-BY-NC 4.0 weights",
        vram_hint="~5 GB for view synthesis",
        size_hint="Model auto-download on first run",
        platform_note="Best treated as a view-generation assistant in a larger pipeline.",
        install_steps=(
            "git clone https://github.com/SUDO-AI-3D/zero123plus BASE_DIR\\models\\zero123plus",
            "cd BASE_DIR\\models\\zero123plus",
            "python -m pip install -r requirements.txt",
        ),
        vendor_dir_name="zero123plus",
        ready_paths=("diffusers-support", "examples"),
        core_deps=("torch", "diffusers", "transformers"),
        provider_key=None,
    ),
)


MODEL_CATALOG_BY_ID = {entry.id: entry for entry in MODEL_CATALOG}
