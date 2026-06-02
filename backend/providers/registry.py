from __future__ import annotations

import importlib
import importlib.util
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from backend.models import (
    GenerationRequest,
    ModelCatalogItem,
    ModelCatalogResponse,
    ModelInstallStartResponse,
    ModelInstallStatusResponse,
)
from backend.providers.installers import build_install_plan as build_model_install_plan
from backend.providers.installers.common import InstallPlanStep, InstallerContext
from backend.providers.base import (
    ProviderCapabilityError,
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderExecutionError,
)
from backend.providers.catalog import (
    DEFAULT_IMAGE_MODEL_ID,
    DEFAULT_TEXT_MODEL_ID,
    MODEL_CATALOG,
    MODEL_CATALOG_BY_ID,
    CatalogEntry,
)
from backend.providers.external_cli_provider import StableFast3DProvider, TripoSRProvider
from backend.providers.helpers import (
    current_python_series,
    current_python_version,
    ensure_vendor_root,
    configure_huggingface_cache,
    huggingface_hub_cache,
    installed_package_version,
    python_executable,
    repo_path,
    run_subprocess,
)
from backend.providers.hunyuan_provider import Hunyuan20Provider, Hunyuan21Provider
from backend.providers.shap_e_provider import ShapEProvider
from backend.providers.trellis_provider import Trellis2Provider, TrellisProvider
from backend.runtimes.hunyuan21_paint import materialize_hunyuan21_paint_runtime


PROVIDER_MAP = {
    "shap-e": ShapEProvider,
    "hunyuan3d-2": Hunyuan20Provider,
    "hunyuan3d-2.1": Hunyuan21Provider,
    "trellis": TrellisProvider,
    "trellis.2": Trellis2Provider,
    "stable-fast-3d": StableFast3DProvider,
    "triposr": TripoSRProvider,
}


@dataclass
class InstallJob:
    job_id: str
    model_id: str
    model_name: str
    status: str = "running"
    current_step: int = 0
    step_count: int = 0
    active_step: str | None = None
    logs: list[str] = field(default_factory=list)
    status_detail: str = "Preparing install plan..."
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def append_log(self, line: str) -> None:
        cleaned = line.rstrip()
        if not cleaned:
            return
        with self._lock:
            self.logs.append(cleaned)
            if len(self.logs) > 2000:
                self.logs = self.logs[-2000:]

    def set_plan(self, step_count: int) -> None:
        with self._lock:
            self.step_count = step_count
            if step_count == 0:
                self.status_detail = "No automated install commands were scheduled."

    def start_step(self, index: int, label: str) -> None:
        with self._lock:
            self.current_step = index
            self.active_step = label
            self.status = "running"
            self.status_detail = label

    def finish(self, status: str, detail: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.status_detail = detail
            self.active_step = None
            self.error = error
            if self.step_count and self.current_step < self.step_count:
                self.current_step = self.step_count

    def snapshot(self) -> ModelInstallStatusResponse:
        with self._lock:
            return ModelInstallStatusResponse(
                job_id=self.job_id,
                model_id=self.model_id,
                model_name=self.model_name,
                status=self.status,
                current_step=self.current_step,
                step_count=self.step_count,
                active_step=self.active_step,
                logs=list(self.logs),
                status_detail=self.status_detail,
                error=self.error,
            )


_INSTALL_JOBS: dict[str, InstallJob] = {}
_INSTALL_JOBS_LOCK = threading.Lock()


def _has_module(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _entry_downloaded(entry: CatalogEntry) -> bool:
    if entry.vendor_dir_name:
        return repo_path(entry.vendor_dir_name).exists()
    return all(_has_module(module_name) for module_name in entry.core_deps)


def _entry_generation_ready(entry: CatalogEntry) -> bool:
    if entry.provider_key is None:
        return False

    if entry.id == "shap-e":
        return all(_has_module(module_name) for module_name in entry.core_deps)

    if not entry.vendor_dir_name:
        return all(_has_module(module_name) for module_name in entry.core_deps)

    repo_dir = repo_path(entry.vendor_dir_name)
    if not repo_dir.exists():
        return False

    for relative_path in entry.ready_paths:
        if not (repo_dir / relative_path).exists():
            return False

    return all(_has_module(module_name) for module_name in entry.core_deps)


def _entry_status(entry: CatalogEntry) -> tuple[str, bool, bool, str]:
    downloaded = _entry_downloaded(entry)
    ready = _entry_generation_ready(entry)

    if entry.provider_key is None:
        return (
            "library_only",
            downloaded,
            False,
            "Exposed as a downloadable assistant, not as a final mesh generator in this app.",
        )

    if ready:
        return ("ready", True, True, "Ready for generation.")

    if downloaded:
        return (
            "downloaded",
            True,
            False,
            "Repo is present. Run Install / Download to finish the dependency setup before using it for generation.",
        )

    return (
        "setup_required",
        False,
        False,
        "Run Install / Download to clone the repo and prepare this model for generation.",
    )


def _entry_python_status(entry: CatalogEntry) -> tuple[str | None, bool | None, str | None]:
    if not entry.supported_python:
        return (current_python_version(), None, entry.python_note)

    current_series = current_python_series()
    current_version = current_python_version()
    compatible = current_series in entry.supported_python
    supported = ", ".join(entry.supported_python)

    if compatible:
        detail = f"Current backend runtime is Python {current_version}. This model is curated for Python {supported}."
    else:
        detail = (
            f"Current backend runtime is Python {current_version}. "
            f"{entry.name} is curated for Python {supported}"
        )
        if entry.preferred_python:
            detail += f" and prefers Python {entry.preferred_python}"
        detail += "."

    if entry.python_note:
        detail = f"{detail} {entry.python_note}"

    return (current_version, compatible, detail)


def get_catalog_response() -> ModelCatalogResponse:
    items: list[ModelCatalogItem] = []
    for entry in MODEL_CATALOG:
        status, downloaded, generation_ready, status_detail = _entry_status(entry)
        current_python, python_compatible, python_status_detail = _entry_python_status(entry)
        items.append(
            ModelCatalogItem(
                id=entry.id,
                name=entry.name,
                family=entry.family,
                role=entry.role,
                summary=entry.summary,
                description=entry.description,
                selection_modes=list(entry.selection_modes),
                library_modes=list(entry.library_modes),
                recommended=entry.recommended,
                repo_url=entry.repo_url,
                docs_url=entry.docs_url,
                huggingface_url=entry.huggingface_url,
                license_name=entry.license_name,
                vram_hint=entry.vram_hint,
                size_hint=entry.size_hint,
                platform_note=entry.platform_note,
                preferred_python=entry.preferred_python,
                supported_python=list(entry.supported_python),
                current_python=current_python,
                python_compatible=python_compatible,
                python_status_detail=python_status_detail,
                install_steps=list(entry.install_steps),
                downloaded=downloaded,
                generation_ready=generation_ready,
                status=status,
                status_detail=status_detail,
            )
        )
    return ModelCatalogResponse(models=items)


def _default_model_for_mode(mode: str) -> str:
    if mode == "text":
        return DEFAULT_TEXT_MODEL_ID
    if mode == "image":
        return DEFAULT_IMAGE_MODEL_ID
    raise ProviderCapabilityError(f"Unsupported generation mode: {mode}")


def get_provider_for_request(req: GenerationRequest):
    model_id = req.model_id or _default_model_for_mode(req.type)
    entry = MODEL_CATALOG_BY_ID.get(model_id)
    if entry is None:
        raise ProviderConfigurationError(f"Unknown model_id: {model_id}")

    if req.type not in entry.selection_modes:
        raise ProviderCapabilityError(f"{entry.name} does not support {req.type}-to-3D generation")

    if not _entry_generation_ready(entry):
        raise ProviderDependencyError(
            f"{entry.name} is not ready yet. Run Install / Download in the dock or Model Browser first."
        )

    provider_cls = PROVIDER_MAP.get(model_id)
    if provider_cls is None:
        raise ProviderConfigurationError(f"{entry.name} has no provider adapter in this build")

    return entry, provider_cls(model_id)


def download_model_repo(model_id: str) -> Path:
    entry = MODEL_CATALOG_BY_ID.get(model_id)
    if entry is None:
        raise ProviderConfigurationError(f"Unknown model_id: {model_id}")
    if not entry.repo_url or not entry.vendor_dir_name:
        raise ProviderCapabilityError(f"{entry.name} does not expose a curated repo download in this app")

    ensure_vendor_root()
    target = repo_path(entry.vendor_dir_name)
    if target.exists():
        return target

    clone_args = ["git", "clone", "--depth", "1"]
    if model_id in {"trellis", "trellis.2"}:
        clone_args.append("--recurse-submodules")
    clone_args.extend([entry.repo_url, str(target)])

    result = run_subprocess(clone_args, cwd=target.parent, timeout=3600)
    if result.returncode != 0:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise ProviderConfigurationError(
            f"Failed to download {entry.name}\n{(result.stderr or result.stdout).strip()[:2000]}"
        )

    return target


def _tokenize_command(command: str) -> list[str]:
    tokens = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', command.strip())
    parsed = [token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"} else token for token in tokens]
    if not parsed:
        raise ProviderConfigurationError("Install command was empty after tokenization.")
    return parsed


def _stream_process(args: list[str], cwd: Path, on_line, timeout: int = 3600) -> int:
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except FileNotFoundError as exc:
        raise ProviderExecutionError(f"Command not found: {args[0]}") from exc

    try:
        if process.stdout is not None:
            for line in process.stdout:
                on_line(line.rstrip())
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise ProviderExecutionError(f"Command timed out after {timeout}s: {' '.join(args)}") from exc


def _build_install_plan(entry: CatalogEntry) -> list[InstallPlanStep]:
    current_torch = installed_package_version("torch")
    current_torchvision = installed_package_version("torchvision")
    current_torchaudio = installed_package_version("torchaudio")
    current_torch_stack_ready = all(
        (
            current_torch,
            current_torchvision,
            current_torchaudio,
            _has_module("torch"),
            _has_module("torchvision"),
            _has_module("torchaudio"),
        )
    )
    context = InstallerContext(
        current_python_series=current_python_series(),
        current_python_version=current_python_version(),
        installed_packages={
            "torch": current_torch,
            "torchvision": current_torchvision,
            "torchaudio": current_torchaudio,
        },
        torch_stack_ready=current_torch_stack_ready,
        repo_exists=bool(entry.vendor_dir_name and repo_path(entry.vendor_dir_name).exists()),
    )
    try:
        return build_model_install_plan(entry, context)
    except ValueError as exc:
        raise ProviderConfigurationError(str(exc)) from exc


def _clone_repo_with_logs(entry: CatalogEntry, job: InstallJob) -> Path:
    if not entry.repo_url or not entry.vendor_dir_name:
        raise ProviderCapabilityError(f"{entry.name} does not expose a curated repo download in this app")

    ensure_vendor_root()
    target = repo_path(entry.vendor_dir_name)
    if target.exists():
        job.append_log(f"Repository already present: {target}")
        return target

    clone_args = ["git", "clone", "--depth", "1"]
    if entry.id in {"trellis", "trellis.2"}:
        clone_args.append("--recurse-submodules")
    clone_args.extend([entry.repo_url, str(target)])

    job.append_log(f"$ {' '.join(clone_args)}")
    code = _stream_process(clone_args, cwd=target.parent, on_line=job.append_log, timeout=3600)
    if code != 0:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        raise ProviderConfigurationError(f"Repository clone failed with exit code {code}.")

    job.append_log(f"Repository ready: {target}")
    return target


def _job_repo_dir(entry: CatalogEntry) -> Path:
    if entry.vendor_dir_name:
        return repo_path(entry.vendor_dir_name)
    return ensure_vendor_root()


def _patch_hunyuan21_shape_init(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory to patch.")

    init_path = repo_path(entry.vendor_dir_name) / "hy3dshape" / "hy3dshape" / "__init__.py"
    if not init_path.exists():
        raise ProviderConfigurationError(f"Expected Hunyuan 2.1 package file was not found: {init_path}")

    content = (
        "from .pipelines import Hunyuan3DDiTPipeline, Hunyuan3DDiTFlowMatchingPipeline\n\n"
        "__all__ = [\n"
        '    "Hunyuan3DDiTPipeline",\n'
        '    "Hunyuan3DDiTFlowMatchingPipeline",\n'
        "]\n"
    )
    init_path.write_text(content, encoding="utf-8")
    job.append_log(f"Patched shape-only package exports: {init_path}")


def _patch_hunyuan21_paint_sources(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory to patch.")

    repo_dir = repo_path(entry.vendor_dir_name)
    paint_dir = repo_dir / "hy3dpaint"
    if not paint_dir.exists():
        raise ProviderConfigurationError(f"Expected Hunyuan 2.1 paint directory was not found: {paint_dir}")

    paint_pipeline_path = repo_path(entry.vendor_dir_name) / "hy3dpaint" / "textureGenPipeline.py"
    if paint_pipeline_path.exists():
        paint_content = paint_pipeline_path.read_text(encoding="utf-8")
        patched_content = paint_content.replace(
            "for i in range(len(enhance_images)):",
            "for i in range(len(enhance_images[\"albedo\"])):",
        )
        if patched_content != paint_content:
            paint_pipeline_path.write_text(patched_content, encoding="utf-8")
            job.append_log(
                "Patched Hunyuan3D-Paint view resize loop so every generated view is resized before UV baking."
            )
        else:
            job.append_log("Hunyuan3D-Paint view resize loop patch is already applied.")

    paint_pbr_pipeline_path = paint_dir / "hunyuanpaintpbr" / "pipeline.py"
    if paint_pbr_pipeline_path.exists():
        pbr_content = paint_pbr_pipeline_path.read_text(encoding="utf-8")
        patched_pbr = pbr_content
        if "vae_param = next(self.vae.parameters())" not in patched_pbr:
            patched_pbr = patched_pbr.replace(
                "        dtype = next(self.vae.parameters()).dtype\n"
                "        images = (images - 0.5) * 2.0\n"
                "        posterior = self.vae.encode(images.to(dtype)).latent_dist\n",
                "        vae_param = next(self.vae.parameters())\n"
                "        dtype = vae_param.dtype\n"
                "        images = (images - 0.5) * 2.0\n"
                "        images = images.to(device=vae_param.device, dtype=dtype)\n"
                "        posterior = self.vae.encode(images).latent_dist\n",
            )
        patched_pbr = patched_pbr.replace(
            "        images_vae = images_vae.to(device=self.vae.device, dtype=self.unet.dtype)\n",
            "        images_vae = images_vae.to(device=next(self.vae.parameters()).device, dtype=self.unet.dtype)\n",
        )
        if "target_device = next(self.vae.parameters()).device" not in patched_pbr:
            patched_pbr = patched_pbr.replace(
                "        def convert_pil_list_to_tensor(images):\n"
                "            bg_c = [1.0, 1.0, 1.0]\n",
                "        def convert_pil_list_to_tensor(images):\n"
                "            target_device = next(self.vae.parameters()).device\n"
                "            bg_c = [1.0, 1.0, 1.0]\n",
            )
        patched_pbr = patched_pbr.replace(
            '                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().half().to("cuda")\n',
            "                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()\n"
            "                    img = img.to(device=target_device, dtype=self.unet.dtype)\n",
        )
        if "target_device = next(self.unet.parameters()).device" not in patched_pbr:
            patched_pbr = patched_pbr.replace(
                "        if guidance_scale > 1:\n",
                "        target_device = next(self.unet.parameters()).device\n"
                "        target_dtype = next(self.unet.parameters()).dtype\n"
                "        prompt_embeds = prompt_embeds.to(device=target_device, dtype=target_dtype)\n"
                "        negative_prompt_embeds = negative_prompt_embeds.to(device=target_device, dtype=target_dtype)\n"
                "\n"
                "        if guidance_scale > 1:\n",
            )
        if patched_pbr != pbr_content:
            paint_pbr_pipeline_path.write_text(patched_pbr, encoding="utf-8")
            job.append_log(
                "Patched Hunyuan3D-Paint PBR tensor conversion to use the active CUDA devices for VAE, UNet, and prompt embeds."
            )
        else:
            job.append_log("Hunyuan3D-Paint PBR tensor-device patch is already applied.")

    dino_modules_path = paint_dir / "hunyuanpaintpbr" / "unet" / "modules.py"
    if dino_modules_path.exists():
        dino_content = dino_modules_path.read_text(encoding="utf-8")
        patched_dino = dino_content.replace(
            "        else:\n"
            "            batch_size = 1\n"
            "            dino_proceesed_images = self.dino_processor(images=images, return_tensors=\"pt\").pixel_values\n"
            "            dino_proceesed_images = torch.stack(\n"
            "                [torch.from_numpy(np.array(image)) for image in dino_proceesed_images], dim=0\n"
            "            )\n"
            "        dino_param = next(self.dino_v2.parameters())\n"
            "        dino_proceesed_images = dino_proceesed_images.to(dino_param)\n",
            "        else:\n"
            "            batch_size = 1\n"
            "            dino_proceesed_images = self.dino_processor(images=images, return_tensors=\"pt\").pixel_values\n"
            "        dino_param = next(self.dino_v2.parameters())\n"
            "        dino_proceesed_images = dino_proceesed_images.to(device=dino_param.device, dtype=dino_param.dtype)\n",
        ).replace(
            "        dino_hidden_states = rearrange(dino_hidden_states.to(dino_param), \"(b n) l c -> b (n l) c\", b=batch_size)\n",
            "        dino_hidden_states = rearrange(\n"
            "            dino_hidden_states.to(device=dino_param.device, dtype=dino_param.dtype),\n"
            "            \"(b n) l c -> b (n l) c\",\n"
            "            b=batch_size,\n"
            "        )\n",
        )
        if patched_dino != dino_content:
            dino_modules_path.write_text(patched_dino, encoding="utf-8")
            job.append_log("Patched Hunyuan3D-Paint DINO wrapper to keep processed image tensors on the DINO CUDA device.")
        else:
            job.append_log("Hunyuan3D-Paint DINO device patch is already applied.")

        mda_content = dino_modules_path.read_text(encoding="utf-8")
        patched_mda = mda_content
        if "def _velocity3d_ensure_mda_processor" not in patched_mda:
            patched_mda = patched_mda.replace(
                "    def __getattr__(self, name: str):\n",
                "    def _velocity3d_ensure_mda_processor(self):\n"
                "        if not self.use_mda:\n"
                "            return\n"
                "        if self.attn1.processor.__class__.__name__ == \"SelfAttnProcessor2_0\":\n"
                "            return\n"
                "\n"
                "        self.attn1.set_processor(\n"
                "            SelfAttnProcessor2_0(\n"
                "                query_dim=self.dim,\n"
                "                heads=self.num_attention_heads,\n"
                "                dim_head=self.attention_head_dim,\n"
                "                dropout=self.dropout,\n"
                "                bias=self.attention_bias,\n"
                "                cross_attention_dim=None,\n"
                "                upcast_attention=self.attn1.upcast_attention,\n"
                "                out_bias=True,\n"
                "                pbr_setting=self.pbr_setting,\n"
                "            )\n"
                "        )\n"
                "        for token in self.pbr_setting:\n"
                "            if token == \"albedo\":\n"
                "                continue\n"
                "            getattr(self.attn1.processor, f\"to_q_{token}\").load_state_dict(self.attn1.to_q.state_dict())\n"
                "            getattr(self.attn1.processor, f\"to_k_{token}\").load_state_dict(self.attn1.to_k.state_dict())\n"
                "            getattr(self.attn1.processor, f\"to_v_{token}\").load_state_dict(self.attn1.to_v.state_dict())\n"
                "            getattr(self.attn1.processor, f\"to_out_{token}\").load_state_dict(self.attn1.to_out.state_dict())\n"
                "        self.attn1.processor.to(device=self.attn1.to_q.weight.device, dtype=self.attn1.to_q.weight.dtype)\n"
                "\n"
                "    def __getattr__(self, name: str):\n",
            )
        if "def _velocity3d_flatten_pbr_attention" not in patched_mda:
            patched_mda = patched_mda.replace(
                "    def __getattr__(self, name: str):\n",
                "    def _velocity3d_flatten_pbr_attention(self, attn_output, num_in_batch, n_pbr, flat_batch_size, label):\n"
                "        if attn_output.ndim == 5:\n"
                "            return rearrange(attn_output, \"b n_pbr n l c -> (b n_pbr n) l c\", n=num_in_batch, n_pbr=n_pbr)\n"
                "\n"
                "        if attn_output.ndim == 4:\n"
                "            if attn_output.shape[1] == n_pbr:\n"
                "                return rearrange(attn_output, \"b n_pbr (n l) c -> (b n_pbr n) l c\", n=num_in_batch, n_pbr=n_pbr)\n"
                "            if attn_output.shape[1] == num_in_batch:\n"
                "                attn_output = attn_output.unsqueeze(1).repeat(1, n_pbr, 1, 1, 1)\n"
                "                return rearrange(attn_output, \"b n_pbr n l c -> (b n_pbr n) l c\")\n"
                "            if attn_output.shape[0] % n_pbr == 0:\n"
                "                return rearrange(attn_output, \"(b n_pbr) n l c -> (b n_pbr n) l c\", n_pbr=n_pbr)\n"
                "\n"
                "        if attn_output.ndim == 3:\n"
                "            if attn_output.shape[0] == flat_batch_size:\n"
                "                return attn_output\n"
                "            base_batch = max(1, flat_batch_size // max(1, n_pbr * num_in_batch))\n"
                "            if attn_output.shape[0] == base_batch and attn_output.shape[1] % num_in_batch == 0:\n"
                "                attn_output = rearrange(attn_output, \"b (n l) c -> b n l c\", n=num_in_batch)\n"
                "                attn_output = attn_output.unsqueeze(1).repeat(1, n_pbr, 1, 1, 1)\n"
                "                return rearrange(attn_output, \"b n_pbr n l c -> (b n_pbr n) l c\")\n"
                "            if attn_output.shape[0] == base_batch * n_pbr and attn_output.shape[1] % num_in_batch == 0:\n"
                "                return rearrange(attn_output, \"(b n_pbr) (n l) c -> (b n_pbr n) l c\", n_pbr=n_pbr, n=num_in_batch)\n"
                "\n"
                "        raise ValueError(\n"
                "            f\"Velocity3D could not normalize {label} attention output with shape {tuple(attn_output.shape)} \"\n"
                "            f\"for num_in_batch={num_in_batch}, n_pbr={n_pbr}, flat_batch_size={flat_batch_size}.\"\n"
                "        )\n"
                "\n"
                "    def __getattr__(self, name: str):\n",
            )
        if "self._velocity3d_ensure_mda_processor()" not in patched_mda:
            patched_mda = patched_mda.replace(
                "        if self.use_mda:\n"
                "            mda_norm_hidden_states = rearrange(\n",
                "        if self.use_mda:\n"
                "            self._velocity3d_ensure_mda_processor()\n"
                "            mda_norm_hidden_states = rearrange(\n",
            )
        if '"material self"' not in patched_mda:
            patched_mda = patched_mda.replace(
                '            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "material self"\n'
                '            )\n',
                1,
            )
        if '"reference"' not in patched_mda:
            patched_mda = patched_mda.replace(
                '            if attn_output.ndim == 3:\n'
                '                attn_output = rearrange(attn_output, "b (n l) c -> b n l c", n=num_in_batch)\n'
                '                attn_output = attn_output.unsqueeze(1).repeat(1, N_pbr, 1, 1, 1)\n'
                '            elif attn_output.ndim == 4:\n'
                '                attn_output = rearrange(attn_output, "b n_pbr (n l) c -> b n_pbr n l c", n=num_in_batch, n_pbr=N_pbr)\n'
                '            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "reference"\n'
                '            )\n',
            )
            patched_mda = patched_mda.replace(
                '            attn_output = rearrange(attn_output, "b n_pbr (n l) c -> (b n_pbr n) l c", n=num_in_batch, n_pbr=N_pbr)\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "reference"\n'
                '            )\n',
            )
        if '"multiview"' not in patched_mda:
            patched_mda = patched_mda.replace(
                '            attn_output = rearrange(attn_output, "(b n_pbr) (n l) c -> (b n_pbr n) l c", n_pbr=N_pbr, n=num_in_batch)\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "multiview"\n'
                '            )\n',
            )
        if patched_mda != mda_content:
            dino_modules_path.write_text(patched_mda, encoding="utf-8")
            job.append_log(
                "Patched Hunyuan3D-Paint material attention to restore the PBR-aware processor and normalize attention branch shapes."
            )
        else:
            job.append_log("Hunyuan3D-Paint material attention processor patch is already applied.")

    attn_processor_path = paint_dir / "hunyuanpaintpbr" / "unet" / "attn_processor.py"
    if attn_processor_path.exists():
        attn_content = attn_processor_path.read_text(encoding="utf-8")
        attn_marker = "target_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states"
        patched_attn = attn_content
        if attn_marker not in patched_attn:
            patched_attn = patched_attn.replace(
                "        batch_size, sequence_length, _ = (\n"
                "            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape\n"
                "        )\n",
                "        target_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states\n"
                "        if target_states.ndim > 3:\n"
                "            target_states = target_states.reshape(-1, target_states.shape[-2], target_states.shape[-1])\n"
                "            if encoder_hidden_states is None:\n"
                "                hidden_states = target_states\n"
                "            else:\n"
                "                encoder_hidden_states = target_states\n"
                "        batch_size, sequence_length, _ = target_states.shape\n",
            )
        if patched_attn != attn_content:
            attn_processor_path.write_text(patched_attn, encoding="utf-8")
            job.append_log("Patched Hunyuan3D-Paint attention processor to flatten 4D states before QKV unpacking.")
        else:
            job.append_log("Hunyuan3D-Paint attention shape patch is already applied.")

    uvwrap_path = paint_dir / "utils" / "uvwrap_utils.py"
    if uvwrap_path.exists():
        uvwrap_content = uvwrap_path.read_text(encoding="utf-8")
        uvwrap_marker = "atlas_result = xatlas.parametrize(mesh.vertices, mesh.faces)"
        patched_uvwrap = uvwrap_content
        if uvwrap_marker not in patched_uvwrap:
            patched_uvwrap = patched_uvwrap.replace(
                "    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)\n",
                "    atlas_result = xatlas.parametrize(mesh.vertices, mesh.faces)\n"
                "    if len(atlas_result) < 3:\n"
                "        raise ValueError(f\"xatlas.parametrize returned {len(atlas_result)} values; expected at least 3.\")\n"
                "    vmapping, indices, uvs = atlas_result[:3]\n",
            )
        if patched_uvwrap != uvwrap_content:
            uvwrap_path.write_text(patched_uvwrap, encoding="utf-8")
            job.append_log("Patched Hunyuan3D-Paint UV unwrap for xatlas versions that return extra values.")
        else:
            job.append_log("Hunyuan3D-Paint xatlas return-shape patch is already applied.")

    simplify_path = repo_path(entry.vendor_dir_name) / "hy3dpaint" / "utils" / "simplify_mesh_utils.py"
    if simplify_path.exists():
        simplify_content = simplify_path.read_text(encoding="utf-8")
        patched_simplify = simplify_content.replace(
            "courent = courent.simplify_quadric_decimation(target_count)",
            "courent = courent.simplify_quadric_decimation(face_count=target_count)",
        )
        if patched_simplify != simplify_content:
            simplify_path.write_text(patched_simplify, encoding="utf-8")
            job.append_log("Patched Hunyuan3D-Paint remesh helper for the current trimesh decimation API.")
        else:
            job.append_log("Hunyuan3D-Paint remesh API patch is already applied.")

    paint_mesh_utils_path = repo_path(entry.vendor_dir_name) / "hy3dpaint" / "DifferentiableRenderer" / "mesh_utils.py"
    if paint_mesh_utils_path.exists():
        mesh_utils_content = paint_mesh_utils_path.read_text(encoding="utf-8")
        patched_mesh_utils = mesh_utils_content
        if "_BPY_IMPORT_ERROR" not in patched_mesh_utils:
            patched_mesh_utils = patched_mesh_utils.replace(
                "import bpy\n",
                "try:\n"
                "    import bpy\n"
                "    _BPY_IMPORT_ERROR = None\n"
                "except Exception as exc:\n"
                "    bpy = None\n"
                "    _BPY_IMPORT_ERROR = exc\n",
            )
        if "Blender bpy module is unavailable" not in patched_mesh_utils:
            patched_mesh_utils = patched_mesh_utils.replace(
                "def convert_obj_to_glb(\n"
                "    obj_path: str,\n"
                "    glb_path: str,\n"
                "    shade_type: str = \"SMOOTH\",\n"
                "    auto_smooth_angle: float = 60,\n"
                "    merge_vertices: bool = False,\n"
                ") -> bool:\n"
                "    \"\"\"Convert OBJ file to GLB format using Blender.\"\"\"\n"
                "    try:\n",
                "def convert_obj_to_glb(\n"
                "    obj_path: str,\n"
                "    glb_path: str,\n"
                "    shade_type: str = \"SMOOTH\",\n"
                "    auto_smooth_angle: float = 60,\n"
                "    merge_vertices: bool = False,\n"
                ") -> bool:\n"
                "    \"\"\"Convert OBJ file to GLB format using Blender.\"\"\"\n"
                "    if bpy is None:\n"
                "        raise ImportError(f\"Blender bpy module is unavailable: {_BPY_IMPORT_ERROR}\")\n"
                "    try:\n",
            )
        if patched_mesh_utils != mesh_utils_content:
            paint_mesh_utils_path.write_text(patched_mesh_utils, encoding="utf-8")
            job.append_log(
                "Patched Hunyuan3D-Paint mesh_utils so unavailable Blender bpy does not block OBJ texture export."
            )
        else:
            job.append_log("Hunyuan3D-Paint DifferentiableRenderer mesh_utils patch is already applied.")

    custom_rasterizer_dir = paint_dir / "custom_rasterizer"
    render_wrapper_path = custom_rasterizer_dir / "custom_rasterizer" / "render.py"
    if render_wrapper_path.exists():
        render_content = render_wrapper_path.read_text(encoding="utf-8")
        patched_render = render_content
        if "tri = tri.to(device=pos.device, dtype=torch.int32).contiguous()" not in patched_render:
            patched_render = patched_render.replace(
                "def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):\n"
                "    assert pos.device == tri.device\n"
                "    findices, barycentric = custom_rasterizer_kernel.rasterize_image(\n",
                "def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):\n"
                "    assert pos.device == tri.device\n"
                "    tri = tri.to(device=pos.device, dtype=torch.int32).contiguous()\n"
                "    if clamp_depth.numel() > 0:\n"
                "        clamp_depth = clamp_depth.to(device=pos.device, dtype=torch.float32).contiguous()\n"
                "    findices, barycentric = custom_rasterizer_kernel.rasterize_image(\n",
            )
        if patched_render != render_content:
            render_wrapper_path.write_text(patched_render, encoding="utf-8")
            job.append_log("Patched custom_rasterizer Python wrapper to pass int32 triangle indices into the CUDA kernel.")
        else:
            job.append_log("custom_rasterizer Python index-dtype patch is already applied.")

    for rasterizer_source in (
        custom_rasterizer_dir / "lib" / "custom_rasterizer_kernel_for_windows" / "rasterizer_gpu.cu",
        custom_rasterizer_dir / "lib" / "custom_rasterizer_kernel" / "rasterizer_gpu.cu",
    ):
        if not rasterizer_source.exists():
            continue
        rasterizer_content = rasterizer_source.read_text(encoding="utf-8")
        patched_rasterizer = rasterizer_content.replace(
            "uint64_t maxint = (uint64_t)MAXINT * (uint64_t)MAXINT + (MAXINT - 1);",
            "int64_t maxint = (int64_t)MAXINT * (int64_t)MAXINT + (MAXINT - 1);",
        ).replace(
            "auto z_min = torch::ones({height, width}, INT64_options) * (uint64_t)maxint;",
            "auto z_min = torch::ones({height, width}, INT64_options) * maxint;",
        ).replace(
            "auto z_min = torch::ones({height, width}, INT64_options) * (long)maxint;",
            "auto z_min = torch::ones({height, width}, INT64_options) * maxint;",
        ).replace(
            "auto z_min = torch::ones({height, width}, UINT64_options) * (uint64_t)maxint;",
            "auto z_min = torch::ones({height, width}, INT64_options) * maxint;",
        ).replace(
            "auto UINT64_options = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA, device_id).requires_grad(false);",
            "auto INT64_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id).requires_grad(false);",
        ).replace(
            "z_min.data_ptr<uint64_t>()",
            "z_min.data_ptr<int64_t>()",
        )
        if patched_rasterizer != rasterizer_content:
            rasterizer_source.write_text(patched_rasterizer, encoding="utf-8")
            job.append_log(f"Patched custom_rasterizer z-buffer pointer dtype for PyTorch 2.10: {rasterizer_source}")
        elif "z_min.data_ptr<int64_t>()" in rasterizer_content:
            job.append_log(f"custom_rasterizer z-buffer pointer dtype patch is already applied: {rasterizer_source}")

    obsolete_shim = custom_rasterizer_dir / "custom_rasterizer_kernel.py"
    if obsolete_shim.exists():
        obsolete_shim.unlink()
        job.append_log(f"Removed obsolete Python rasterizer shim before native build: {obsolete_shim}")
    elif custom_rasterizer_dir.exists():
        job.append_log("Hunyuan3D-Paint custom_rasterizer source tree is ready for current-env native build.")


def _materialize_hunyuan21_paint_runtime(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory to materialize.")
    runtime = materialize_hunyuan21_paint_runtime(source_repo_dir=repo_path(entry.vendor_dir_name), force=True)
    job.append_log(f"Velocity3D-owned Hunyuan 2.1 paint runtime ready: {runtime.paint_dir}")
    job.append_log(
        "The downloaded Hunyuan repository is now treated as model/assets input; Velocity3D executes the private runtime copy."
    )


def _download_huggingface_snapshot(step: InstallPlanStep, job: InstallJob) -> None:
    if not step.hf_repo_id:
        raise ProviderConfigurationError("Hugging Face download step is missing a repo id.")

    configure_huggingface_cache()

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise ProviderDependencyError(
            "huggingface-hub is required before model weights can be downloaded."
        ) from exc

    repo_id = step.hf_repo_id
    local_dir = Path(step.hf_local_dir) if step.hf_local_dir else None
    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)

    job.append_log(f"Hugging Face cache root: {configure_huggingface_cache()}")
    job.append_log(f"Resolving Hugging Face files for {repo_id}")

    api = HfApi()
    files = list(api.list_repo_files(repo_id=repo_id))
    if step.hf_allow_patterns:
        files = [
            file_name
            for file_name in files
            if any(fnmatch.fnmatch(file_name, pattern) for pattern in step.hf_allow_patterns)
        ]

    if not files:
        raise ProviderConfigurationError(f"No Hugging Face files matched for {repo_id}.")

    total = len(files)
    for index, file_name in enumerate(files, start=1):
        job.append_log(f"[{index}/{total}] Downloading {repo_id}/{file_name}")
        if local_dir is not None:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=str(local_dir),
            )
        else:
            hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                cache_dir=str(huggingface_hub_cache()),
            )

    destination = local_dir if local_dir is not None else huggingface_hub_cache()
    job.append_log(f"Downloaded {repo_id} to {destination}")


def _download_file(step: InstallPlanStep, job: InstallJob) -> None:
    if not step.download_url or not step.download_path:
        raise ProviderConfigurationError("Download step is missing a URL or destination path.")

    destination = Path(step.download_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        job.append_log(f"File already present: {destination}")
        return

    job.append_log(f"Downloading {step.download_url}")
    job.append_log(f"Destination: {destination}")
    try:
        with urllib.request.urlopen(step.download_url, timeout=60) as response:
            with destination.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise ProviderExecutionError(f"Failed to download {step.download_url}: {exc}") from exc

    if not destination.exists() or destination.stat().st_size == 0:
        raise ProviderExecutionError(f"Downloaded file is empty: {destination}")
    job.append_log(f"Downloaded file to {destination}")


def _prepend_path(env: dict[str, str], *paths: Path | str | None) -> None:
    existing = env.get("PATH") or env.get("Path", "")
    additions = [str(path) for path in paths if path and str(path)]
    env["PATH"] = os.pathsep.join(additions + ([existing] if existing else []))
    if sys.platform == "win32":
        env["Path"] = env["PATH"]


def _ninja_bin_dir() -> Path | None:
    try:
        ninja_module = importlib.import_module("ninja")
    except Exception:
        return None
    bin_dir = getattr(ninja_module, "BIN_DIR", None)
    if not bin_dir:
        return None
    path = Path(str(bin_dir))
    return path if path.exists() else None


def _parse_cuda_release(output: str) -> str | None:
    match = re.search(r"release\s+(\d+\.\d+)", output)
    return match.group(1) if match else None


def _torch_cuda_version() -> str | None:
    try:
        import torch
    except Exception:
        return None
    return getattr(getattr(torch, "version", None), "cuda", None)


def _torch_lib_dir() -> Path | None:
    try:
        import torch
    except Exception:
        return None
    torch_file = getattr(torch, "__file__", None)
    if not torch_file:
        return None
    lib_dir = Path(torch_file).parent / "lib"
    return lib_dir if lib_dir.exists() else None


def _nvcc_version(env: dict[str, str]) -> tuple[str | None, str | None]:
    nvcc_path = shutil.which("nvcc", path=env.get("PATH"))
    if not nvcc_path:
        return None, None
    try:
        completed = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
    except Exception:
        return nvcc_path, None
    return nvcc_path, _parse_cuda_release((completed.stdout or "") + (completed.stderr or ""))


def _candidate_cuda_toolkits(torch_cuda: str | None) -> list[Path]:
    candidates: list[Path] = []
    if torch_cuda:
        cuda_key = torch_cuda.replace(".", "_")
        for name in (f"CUDA_PATH_V{cuda_key}", f"CUDA_PATH_V{cuda_key}_0"):
            value = os.environ.get(name)
            if value:
                candidates.append(Path(value))

    for name in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(name)
        if value:
            candidates.append(Path(value))

    if sys.platform == "win32":
        root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if torch_cuda:
            candidates.append(root / f"v{torch_cuda}")
        if root.exists():
            candidates.extend(sorted(root.glob("v*"), reverse=True))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        if resolved.lower() not in seen:
            seen.add(resolved.lower())
            deduped.append(candidate)
    return deduped


def _find_cuda_toolkit(torch_cuda: str | None) -> Path | None:
    if not torch_cuda:
        return None
    for toolkit in _candidate_cuda_toolkits(torch_cuda):
        nvcc = toolkit / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
        if not nvcc.exists():
            continue
        env = {**os.environ, "PATH": str(nvcc.parent) + os.pathsep + os.environ.get("PATH", "")}
        _, version = _nvcc_version(env)
        if version == torch_cuda:
            return toolkit
    return None


def _vswhere_paths() -> list[Path]:
    paths = []
    for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles")):
        if root:
            paths.append(Path(root) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    return [path for path in paths if path.exists()]


def _visual_studio_install_path() -> Path | None:
    for vswhere in _vswhere_paths():
        try:
            completed = subprocess.run(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-requires",
                    "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-property",
                    "installationPath",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except Exception:
            continue
        install_path = completed.stdout.strip()
        if install_path:
            path = Path(install_path)
            if path.exists():
                return path

    root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft Visual Studio" / "2022"
    if root.exists():
        for child in sorted(root.iterdir(), reverse=True):
            vcvars = child / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.exists():
                return child
    return None


def _load_vcvars_env() -> dict[str, str]:
    if sys.platform != "win32":
        return {}
    install_path = _visual_studio_install_path()
    if install_path is None:
        return {}

    batch = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
    args = ""
    if not batch.exists():
        batch = install_path / "Common7" / "Tools" / "VsDevCmd.bat"
        args = " -arch=x64 -host_arch=x64"
    if not batch.exists():
        return {}

    command = f'cmd.exe /d /s /c ""{batch}"{args} >nul && set"'
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            shell=True,
        )
    except Exception:
        return {}
    env: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key] = value
    return env


def _native_extension_env(source_dir: Path, *, require_cuda: bool = False) -> dict[str, str]:
    env = {**os.environ}
    if sys.platform == "win32":
        env.update(_load_vcvars_env())
        if "Path" in env and "PATH" in env and env["Path"] != env["PATH"]:
            env["PATH"] = env["Path"] + os.pathsep + env["PATH"]
        elif "Path" in env:
            env["PATH"] = env["Path"]
        if env.get("VSCMD_ARG_TGT_ARCH") or env.get("VCToolsInstallDir"):
            env["DISTUTILS_USE_SDK"] = "1"
            env["MSSdk"] = "1"

    _prepend_path(env, Path(python_executable()).parent, _ninja_bin_dir(), _torch_lib_dir())

    if require_cuda:
        torch_cuda = _torch_cuda_version()
        toolkit = _find_cuda_toolkit(torch_cuda)
        if toolkit is not None:
            env["CUDA_HOME"] = str(toolkit)
            env["CUDA_PATH"] = str(toolkit)
            _prepend_path(env, toolkit / "bin")

    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(source_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    return env


def _validate_native_build_env(env: dict[str, str], job: InstallJob, *, require_cuda: bool) -> None:
    if sys.platform == "win32":
        cl_path = shutil.which("cl", path=env.get("PATH"))
        if not cl_path:
            raise ProviderConfigurationError(
                "MSVC cl.exe is not available to the native extension build. Install Visual Studio 2022 "
                "with the 'Desktop development with C++' workload, then rerun Install / Download."
            )
        job.append_log(f"Using MSVC compiler: {cl_path}")

    ninja_path = shutil.which("ninja", path=env.get("PATH"))
    if ninja_path:
        job.append_log(f"Using ninja build tool: {ninja_path}")

    if not require_cuda:
        return

    torch_cuda = _torch_cuda_version()
    if not torch_cuda:
        raise ProviderConfigurationError("PyTorch is installed without CUDA support; Hunyuan paint rasterizer needs CUDA.")

    nvcc_path, nvcc_cuda = _nvcc_version(env)
    if not nvcc_path:
        raise ProviderConfigurationError(
            f"CUDA Toolkit {torch_cuda} nvcc was not found. Install CUDA Toolkit {torch_cuda} and rerun Install / Download."
        )

    if nvcc_cuda != torch_cuda:
        raise ProviderConfigurationError(
            f"CUDA Toolkit mismatch for native build: backend PyTorch is cu{torch_cuda.replace('.', '')}, "
            f"but nvcc is CUDA {nvcc_cuda or 'unknown'} at {nvcc_path}. Install CUDA Toolkit {torch_cuda} "
            "or switch the backend PyTorch stack to the installed toolkit version."
        )

    job.append_log(f"Using CUDA Toolkit {nvcc_cuda}: {nvcc_path}")


def _stream_current_env_process(
    args: list[str],
    cwd: Path,
    on_line,
    timeout: int = 5400,
    env: dict[str, str] | None = None,
) -> int:
    try:
        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env or _native_extension_env(cwd),
        )
    except FileNotFoundError as exc:
        raise ProviderExecutionError(f"Command not found: {args[0]}") from exc

    try:
        if process.stdout is not None:
            for line in process.stdout:
                on_line(line.rstrip())
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        raise ProviderExecutionError(f"Command timed out after {timeout}s: {' '.join(args)}") from exc


def _write_custom_rasterizer_build_script(source_dir: Path) -> Path:
    kernel_dir = "lib/custom_rasterizer_kernel_for_windows" if sys.platform == "win32" else "lib/custom_rasterizer_kernel"
    script_path = source_dir / "velocity3d_build_setup.py"
    script_path.write_text(
        (
            "from setuptools import setup, find_packages\n"
            "from torch.utils.cpp_extension import BuildExtension, CUDAExtension\n\n"
            f"KERNEL_DIR = {kernel_dir!r}\n"
            "custom_rasterizer_module = CUDAExtension(\n"
            "    'custom_rasterizer_kernel',\n"
            "    [\n"
            "        f'{KERNEL_DIR}/rasterizer.cpp',\n"
            "        f'{KERNEL_DIR}/grid_neighbor.cpp',\n"
            "        f'{KERNEL_DIR}/rasterizer_gpu.cu',\n"
            "    ],\n"
            "    define_macros=[('USE_CUDA', None)],\n"
            ")\n\n"
            "setup(\n"
            "    packages=find_packages(),\n"
            "    version='0.1',\n"
            "    name='custom_rasterizer',\n"
            "    include_package_data=True,\n"
            "    package_dir={'': '.'},\n"
            "    ext_modules=[custom_rasterizer_module],\n"
            "    cmdclass={'build_ext': BuildExtension},\n"
            ")\n"
        ),
        encoding="utf-8",
    )
    return script_path


def _verify_custom_rasterizer(source_dir: Path, job: InstallJob, env: dict[str, str] | None = None) -> bool:
    args = [
        python_executable(),
        "-c",
        (
            "try:\n"
            "    import torch\n"
            "    import custom_rasterizer, custom_rasterizer_kernel\n"
            "    print('custom_rasterizer import OK')\n"
            "except Exception as exc:\n"
            "    print(f'custom_rasterizer import not ready: {exc}')\n"
            "    raise SystemExit(1)\n"
        ),
    ]
    code = _stream_current_env_process(args, cwd=source_dir, on_line=job.append_log, timeout=60, env=env)
    return code == 0


def _custom_rasterizer_needs_rebuild(source_dir: Path) -> bool:
    outputs = list(source_dir.glob("custom_rasterizer_kernel*.pyd")) + list(source_dir.glob("custom_rasterizer_kernel*.so"))
    if not outputs:
        return True

    source_patterns = ("*.cpp", "*.cu", "*.h", "*.hpp", "velocity3d_build_setup.py")
    source_files: list[Path] = []
    for pattern in source_patterns:
        source_files.extend(source_dir.rglob(pattern))
    if not source_files:
        return False

    latest_source = max(path.stat().st_mtime for path in source_files)
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return latest_source > oldest_output


def _build_custom_rasterizer(source_dir: Path, job: InstallJob) -> None:
    if not source_dir.exists():
        raise ProviderConfigurationError(f"Custom rasterizer source directory was not found: {source_dir}")

    job.append_log(f"Building custom_rasterizer in current backend env: {source_dir}")
    obsolete_shim = source_dir / "custom_rasterizer_kernel.py"
    if obsolete_shim.exists():
        obsolete_shim.unlink()
        job.append_log(f"Removed obsolete Python rasterizer shim: {obsolete_shim}")

    env = _native_extension_env(source_dir, require_cuda=True)
    needs_rebuild = _custom_rasterizer_needs_rebuild(source_dir)
    if not needs_rebuild and _verify_custom_rasterizer(source_dir, job, env=env):
        job.append_log("custom_rasterizer is already importable from the source tree.")
        return
    if needs_rebuild:
        job.append_log("custom_rasterizer native sources changed; rebuilding the extension in the current backend env.")

    _validate_native_build_env(env, job, require_cuda=True)
    build_script = _write_custom_rasterizer_build_script(source_dir)
    args = [python_executable(), build_script.name, "build_ext", "--inplace"]
    job.append_log(f"$ {' '.join(args)}")
    code = _stream_current_env_process(args, cwd=source_dir, on_line=job.append_log, timeout=5400, env=env)
    if code != 0:
        raise ProviderConfigurationError(
            f"Current-env custom_rasterizer native build failed with exit code {code}. "
            "Install a CUDA Toolkit matching the backend PyTorch wheel and Visual Studio C++ build tools, "
            "then rerun Install / Download."
        )

    if not _verify_custom_rasterizer(source_dir, job, env=env):
        raise ProviderConfigurationError("custom_rasterizer built but could not be imported from the source tree.")


def _build_hunyuan21_custom_rasterizer(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory.")
    _build_custom_rasterizer(repo_path(entry.vendor_dir_name) / "hy3dpaint" / "custom_rasterizer", job)


def _write_mesh_inpaint_build_script(source_dir: Path) -> Path:
    script_path = source_dir / "velocity3d_mesh_inpaint_setup.py"
    if sys.platform == "win32":
        compile_args = ["/O2", "/std:c++17"]
    else:
        compile_args = ["-O3", "-std=c++11", "-fPIC"]
    script_path.write_text(
        (
            "from setuptools import Extension, setup\n"
            "import pybind11\n\n"
            "mesh_inpaint_module = Extension(\n"
            "    'mesh_inpaint_processor',\n"
            "    ['mesh_inpaint_processor.cpp'],\n"
            "    include_dirs=[pybind11.get_include()],\n"
            f"    extra_compile_args={compile_args!r},\n"
            "    language='c++',\n"
            ")\n\n"
            "setup(\n"
            "    name='mesh_inpaint_processor',\n"
            "    version='0.1',\n"
            "    ext_modules=[mesh_inpaint_module],\n"
            ")\n"
        ),
        encoding="utf-8",
    )
    return script_path


def _verify_mesh_inpaint_processor(source_dir: Path, job: InstallJob, env: dict[str, str] | None = None) -> bool:
    args = [
        python_executable(),
        "-c",
        (
            "try:\n"
            "    import mesh_inpaint_processor\n"
            "    print('mesh_inpaint_processor import OK')\n"
            "except Exception as exc:\n"
            "    print(f'mesh_inpaint_processor import not ready: {exc}')\n"
            "    raise SystemExit(1)\n"
        ),
    ]
    code = _stream_current_env_process(args, cwd=source_dir, on_line=job.append_log, timeout=60, env=env)
    return code == 0


def _build_mesh_inpaint_processor(source_dir: Path, job: InstallJob) -> None:
    if not source_dir.exists():
        raise ProviderConfigurationError(f"Mesh inpaint processor source directory was not found: {source_dir}")
    env = _native_extension_env(source_dir)
    job.append_log(f"Building mesh_inpaint_processor in current backend env: {source_dir}")
    if _verify_mesh_inpaint_processor(source_dir, job, env=env):
        job.append_log("mesh_inpaint_processor is already importable from the source tree.")
        return

    _validate_native_build_env(env, job, require_cuda=False)
    build_script = _write_mesh_inpaint_build_script(source_dir)
    args = [python_executable(), build_script.name, "build_ext", "--inplace"]
    job.append_log(f"$ {' '.join(args)}")
    code = _stream_current_env_process(args, cwd=source_dir, on_line=job.append_log, timeout=5400, env=env)
    if code != 0:
        raise ProviderConfigurationError(
            f"Current-env mesh_inpaint_processor native build failed with exit code {code}. "
            "Install Visual Studio C++ build tools, then rerun Install / Download."
        )
    if not _verify_mesh_inpaint_processor(source_dir, job, env=env):
        raise ProviderConfigurationError("mesh_inpaint_processor built but could not be imported from the source tree.")


def _build_hunyuan21_mesh_inpaint_processor(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory.")
    _build_mesh_inpaint_processor(repo_path(entry.vendor_dir_name) / "hy3dpaint" / "DifferentiableRenderer", job)


def _build_hunyuan20_custom_rasterizer(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory.")
    _build_custom_rasterizer(repo_path(entry.vendor_dir_name) / "hy3dgen" / "texgen" / "custom_rasterizer", job)


def _build_hunyuan20_differentiable_renderer(entry: CatalogEntry, job: InstallJob) -> None:
    if not entry.vendor_dir_name:
        raise ProviderConfigurationError(f"{entry.name} has no repo directory.")
    renderer_dir = repo_path(entry.vendor_dir_name) / "hy3dgen" / "texgen" / "differentiable_renderer"
    if not renderer_dir.exists():
        raise ProviderConfigurationError(f"Differentiable renderer source directory was not found: {renderer_dir}")
    setup_path = renderer_dir / "setup.py"
    if not setup_path.exists():
        raise ProviderConfigurationError(
            f"Differentiable renderer has no setup.py, so Velocity3D cannot build the native renderer from this source tree: {renderer_dir}"
        )
    env = _native_extension_env(renderer_dir)
    _validate_native_build_env(env, job, require_cuda=False)
    args = [python_executable(), "setup.py", "build_ext", "--inplace"]
    job.append_log(f"$ {' '.join(args)}")
    code = _stream_current_env_process(args, cwd=renderer_dir, on_line=job.append_log, timeout=5400, env=env)
    if code != 0:
        raise ProviderConfigurationError(f"Differentiable renderer build failed with exit code {code}.")


def _run_install_job(job: InstallJob, entry: CatalogEntry) -> None:
    manual_required = False
    plan = _build_install_plan(entry)
    job.set_plan(len(plan))
    job.append_log(f"Starting Install / Download for {entry.name}")

    try:
        if not plan:
            job.append_log("No curated install commands were scheduled for this model.")

        for index, step in enumerate(plan, start=1):
            job.start_step(index, step.label)

            if step.label.startswith("Clone ") and step.command is None and not step.manual:
                _clone_repo_with_logs(entry, job)
                continue

            if step.manual:
                manual_required = True
                job.append_log(f"! Manual step required: {step.label}")
                continue

            if step.note:
                job.append_log(step.label)
                continue

            if step.action:
                try:
                    if step.action == "patch_hunyuan21_shape_init":
                        _patch_hunyuan21_shape_init(entry, job)
                        continue
                    if step.action == "patch_hunyuan21_paint_sources":
                        _patch_hunyuan21_paint_sources(entry, job)
                        continue
                    if step.action == "materialize_hunyuan21_paint_runtime":
                        _materialize_hunyuan21_paint_runtime(entry, job)
                        continue
                    if step.action == "download_huggingface_snapshot":
                        _download_huggingface_snapshot(step, job)
                        continue
                    if step.action == "download_file":
                        _download_file(step, job)
                        continue
                    if step.action == "build_hunyuan21_custom_rasterizer":
                        _build_hunyuan21_custom_rasterizer(entry, job)
                        continue
                    if step.action == "build_hunyuan21_mesh_inpaint_processor":
                        _build_hunyuan21_mesh_inpaint_processor(entry, job)
                        continue
                    if step.action == "build_hunyuan20_custom_rasterizer":
                        _build_hunyuan20_custom_rasterizer(entry, job)
                        continue
                    if step.action == "build_hunyuan20_differentiable_renderer":
                        _build_hunyuan20_differentiable_renderer(entry, job)
                        continue
                    raise ProviderConfigurationError(f"Unknown installer action: {step.action}")
                except Exception as exc:
                    if step.optional:
                        job.append_log(
                            f"! Optional step failed: {step.label}. Continuing because this asset is not required for the current generation path. {exc}"
                        )
                        continue
                    raise

            command = step.command or step.label
            job.append_log(f"$ {command}")
            code = _stream_process(_tokenize_command(command), cwd=_job_repo_dir(entry), on_line=job.append_log, timeout=5400)
            if code != 0:
                if step.optional:
                    job.append_log(
                        f"! Optional step failed with exit code {code}; continuing because this dependency is not required for the current generation path."
                    )
                    continue
                recent_logs = "\n".join(job.snapshot().logs[-30:])
                if "Access is denied" in recent_logs or "Toegang geweigerd" in recent_logs or "WinError 5" in recent_logs:
                    job.append_log(
                        "! Windows blocked pip from replacing a file currently loaded by the backend. "
                        "Restart Velocity3D, then rerun Install / Download so the backend uses the updated installer plan."
                    )
                raise ProviderConfigurationError(f"Command failed with exit code {code}: {command}")

        status, _, ready, status_detail = _entry_status(entry)
        if ready:
            job.append_log(f"{entry.name} is now ready for generation.")
            job.finish("complete", status_detail)
            return

        if manual_required:
            job.finish("manual_required", status_detail)
            return

        if status == "downloaded":
            detail = (
                "Install commands finished, but the model is still not ready. "
                "Review the remaining setup requirements or restart the backend after the environment changes."
            )
            job.append_log(detail)
            job.finish("manual_required", detail)
            return

        job.finish("complete", status_detail)
    except Exception as exc:
        message = str(exc)
        job.append_log(f"! {message}")
        job.finish("error", message, error=message)


def start_model_install(model_id: str) -> ModelInstallStartResponse:
    entry = MODEL_CATALOG_BY_ID.get(model_id)
    if entry is None:
        raise ProviderConfigurationError(f"Unknown model_id: {model_id}")

    job = InstallJob(
        job_id=uuid.uuid4().hex,
        model_id=model_id,
        model_name=entry.name,
    )
    with _INSTALL_JOBS_LOCK:
        _INSTALL_JOBS[job.job_id] = job

    thread = threading.Thread(target=_run_install_job, args=(job, entry), daemon=True)
    thread.start()

    return ModelInstallStartResponse(job_id=job.job_id, model_id=model_id)


def get_model_install_status(job_id: str) -> ModelInstallStatusResponse:
    with _INSTALL_JOBS_LOCK:
        job = _INSTALL_JOBS.get(job_id)
    if job is None:
        raise ProviderConfigurationError(f"Unknown install job: {job_id}")
    return job.snapshot()
