from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import logging
import os
import shutil
import sys
import threading
import types
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from backend.pipelines.generation_presets import GenerationPreset
from backend.providers import ProviderDependencyError, ProviderExecutionError
from backend.providers.helpers import configure_huggingface_cache, maybe_add_to_syspath, repo_path
from backend.runtimes.hunyuan21_paint import materialize_hunyuan21_paint_runtime
from backend.runtimes.hunyuan21_paint.runtime import evict_hunyuan21_runtime_modules

logger = logging.getLogger("velocity3d.hunyuan_paint")

HUNYUAN_PAINT_MODEL_IDS = {"hunyuan3d-2", "hunyuan3d-2.1"}


@dataclass
class HunyuanPaintResult:
    output_path: str
    texture_dir: str
    texture_paths: list[str] = field(default_factory=list)


def supports_hunyuan_paint(model_id: str | None) -> bool:
    return model_id in HUNYUAN_PAINT_MODEL_IDS


class _IdentitySuperResolution:
    def __init__(self, _config) -> None:
        pass

    def __call__(self, image):
        return image


class _MissingPymeshlab(types.ModuleType):
    class MeshSet:
        def __init__(self) -> None:
            raise ImportError(
                "pymeshlab is not installed. Hunyuan3D-Paint needs it for the upstream remesh step "
                "that keeps UV wrapping and texture baking bounded."
            )


class HunyuanPaintPipeline:
    """
    Adapter around Velocity3D-owned Hunyuan3D-Paint runtimes.

    Hunyuan3D-2.1 downloads Tencent's repo and Hugging Face weights as assets,
    then materializes a private Velocity3D runtime under BASE_DIR/Runtimes. That
    runtime owns the paint Python modules, DifferentiableRenderer wrapper, and
    custom_rasterizer wrapper so generation no longer executes directly from the
    downloaded repository or from stale Diffusers dynamic-module cache entries.
    """

    _load_lock = threading.Lock()

    def apply(
        self,
        *,
        model_id: str,
        source_glb: str,
        reference_image_bytes: bytes,
        output_glb: Path,
        work_dir: Path,
        cancellation_event: threading.Event,
        max_num_view: int = 6,
        resolution: int = 512,
        pipeline_preset: GenerationPreset | None = None,
    ) -> HunyuanPaintResult:
        if model_id == "hunyuan3d-2.1":
            return self._apply_hunyuan21(
                source_glb=source_glb,
                reference_image_bytes=reference_image_bytes,
                output_glb=output_glb,
                work_dir=work_dir,
                cancellation_event=cancellation_event,
                max_num_view=max_num_view,
                resolution=resolution,
                pipeline_preset=pipeline_preset,
            )
        if model_id == "hunyuan3d-2":
            return self._apply_hunyuan20(
                source_glb=source_glb,
                reference_image_bytes=reference_image_bytes,
                output_glb=output_glb,
                work_dir=work_dir,
                cancellation_event=cancellation_event,
            )
        raise ProviderDependencyError(f"{model_id} does not expose a Hunyuan3D-Paint adapter.")

    def _apply_hunyuan21(
        self,
        *,
        source_glb: str,
        reference_image_bytes: bytes,
        output_glb: Path,
        work_dir: Path,
        cancellation_event: threading.Event,
        max_num_view: int,
        resolution: int,
        pipeline_preset: GenerationPreset | None = None,
    ) -> HunyuanPaintResult:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before Hunyuan3D-Paint texture synthesis")

        repo_dir = repo_path("hunyuan3d-2.1")
        source_paint_dir = repo_dir / "hy3dpaint"
        if not source_paint_dir.exists():
            raise ProviderDependencyError(
                "Hunyuan3D-2.1 paint sources are missing. Run Install / Download for Hunyuan3D-2.1."
            )

        try:
            runtime = materialize_hunyuan21_paint_runtime(source_repo_dir=repo_dir)
        except Exception as exc:
            raise ProviderDependencyError(f"Could not prepare Velocity3D Hunyuan3D-Paint runtime: {exc}") from exc
        paint_dir = runtime.paint_dir
        evict_hunyuan21_runtime_modules()
        self._prepare_runtime_paths(repo_dir, paint_dir)
        self._ensure_required_import("custom_rasterizer", "Hunyuan3D-Paint CUDA rasterizer")
        self._ensure_required_import("mesh_inpaint_processor", "Hunyuan3D-Paint mesh inpaint processor")
        self._ensure_required_import("pymeshlab", "Hunyuan3D-Paint remesh/UV runtime")

        work_dir.mkdir(parents=True, exist_ok=True)
        local_mesh_path = work_dir / "base_mesh.glb"
        image_path = work_dir / "reference_cutout.png"
        output_obj = work_dir / "hunyuan_paint.obj"
        shutil.copy2(source_glb, local_mesh_path)
        self._write_reference_image(reference_image_bytes, image_path)

        try:
            self._apply_torchvision_fix(repo_dir, paint_dir)
            module = importlib.import_module("textureGenPipeline")
            self._patch_optional_super_resolution(module, paint_dir)
            self._patch_hunyuan21_stage_logging(module)
            config_cls = getattr(module, "Hunyuan3DPaintConfig")
            pipeline_cls = getattr(module, "Hunyuan3DPaintPipeline")
        except ImportError as exc:
            raise ProviderDependencyError(
                f"Hunyuan3D-2.1 paint dependencies are missing: {exc}. "
                "Run Install / Download, then restart Velocity3D if pip changed loaded packages."
            ) from exc
        except Exception as exc:
            raise ProviderExecutionError(f"Hunyuan3D-2.1 paint import failed: {exc}") from exc

        config = config_cls(max_num_view=max_num_view, resolution=resolution)
        config.multiview_cfg_path = str(paint_dir / "cfgs" / "hunyuan-paint-pbr.yaml")
        config.custom_pipeline = str(paint_dir / "hunyuanpaintpbr")
        config.realesrgan_ckpt_path = str(runtime.real_esrgan_path)
        config.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        config.dino_ckpt_path = "facebook/dinov2-giant"
        self._configure_hunyuan21_profile(config, pipeline_preset=pipeline_preset)
        config.device = self._hunyuan21_cuda_device_name()
        self._patch_hunyuan21_dynamic_module_cache()
        self._evict_hunyuan21_dynamic_modules()

        try:
            with self._load_lock:
                self._release_cuda_cache()
                logger.info("Loading Velocity3D Hunyuan3D-Paint 2.1 runtime from %s", paint_dir)
                pipeline = pipeline_cls(config)
                module._velocity3d_remesh_target_count = int(config.velocity3d_remesh_target)
                self._configure_hunyuan21_runtime(pipeline)
            if cancellation_event.is_set():
                raise ProviderExecutionError("Generation cancelled after Hunyuan3D-Paint loaded")

            logger.info(
                "Running Hunyuan3D-Paint 2.1 texture synthesis with upstream remesh enabled "
                "(render=%s, texture=%s, views=%s)",
                config.render_size,
                config.texture_size,
                config.max_selected_view_num,
            )
            painted_obj = pipeline(
                mesh_path=str(local_mesh_path),
                image_path=str(image_path),
                output_mesh_path=str(output_obj),
                use_remesh=True,
                save_glb=False,
            )
            if cancellation_event.is_set():
                raise ProviderExecutionError("Generation cancelled after Hunyuan3D-Paint texture synthesis")
        except ProviderExecutionError:
            raise
        except Exception as exc:
            logger.exception("Hunyuan3D-Paint 2.1 texture synthesis crashed")
            raise ProviderExecutionError(f"Hunyuan3D-Paint 2.1 texture synthesis failed: {exc}") from exc
        finally:
            self._release_cuda_cache()

        painted_obj_path = Path(painted_obj)
        texture_paths = self._collect_hunyuan_obj_textures(painted_obj_path)
        self._pack_obj_with_pbr_textures(painted_obj_path, texture_paths, output_glb)
        self._assert_glb(output_glb)
        logger.info("Hunyuan3D-Paint 2.1 textured GLB ready: %s", output_glb)
        return HunyuanPaintResult(
            output_path=str(output_glb),
            texture_dir=str(work_dir),
            texture_paths=[str(path) for path in texture_paths.values() if path and path.exists()],
        )

    def _apply_hunyuan20(
        self,
        *,
        source_glb: str,
        reference_image_bytes: bytes,
        output_glb: Path,
        work_dir: Path,
        cancellation_event: threading.Event,
    ) -> HunyuanPaintResult:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before Hunyuan3D-Paint texture synthesis")

        repo_dir = repo_path("hunyuan3d-2")
        if not repo_dir.exists():
            raise ProviderDependencyError(
                "Hunyuan3D-2 paint sources are missing. Run Install / Download for Hunyuan3D-2."
            )

        self._prepare_runtime_paths(repo_dir, repo_dir / "hy3dgen" / "texgen")
        self._ensure_required_import("custom_rasterizer", "Hunyuan3D-Paint CUDA rasterizer")
        work_dir.mkdir(parents=True, exist_ok=True)
        image_path = work_dir / "reference_cutout.png"
        self._write_reference_image(reference_image_bytes, image_path)

        try:
            import trimesh
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except ImportError as exc:
            raise ProviderDependencyError(
                f"Hunyuan3D-2 paint dependencies are missing: {exc}. Run Install / Download for Hunyuan3D-2."
            ) from exc

        try:
            mesh = trimesh.load(source_glb, force="mesh")
            with self._load_lock:
                logger.info("Loading Hunyuan3D-Paint 2.0 pipeline")
                pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
            logger.info("Running Hunyuan3D-Paint 2.0 texture synthesis")
            textured_mesh = pipeline(mesh, image=str(image_path))
            textured_mesh.export(str(output_glb))
        except Exception as exc:
            raise ProviderExecutionError(f"Hunyuan3D-Paint 2.0 texture synthesis failed: {exc}") from exc
        finally:
            self._release_cuda_cache()

        self._assert_glb(output_glb)
        return HunyuanPaintResult(output_path=str(output_glb), texture_dir=str(work_dir))

    def _prepare_runtime_paths(self, repo_dir: Path, paint_dir: Path) -> None:
        configure_huggingface_cache()
        self._prepare_windows_native_dlls()
        maybe_add_to_syspath(
            (
                repo_dir,
                repo_dir / "hy3dshape",
                paint_dir,
                paint_dir / "utils",
                paint_dir / "custom_rasterizer",
                paint_dir / "DifferentiableRenderer",
            )
        )

    def _patch_hunyuan21_dynamic_module_cache(self) -> None:
        cache_root = configure_huggingface_cache() / "modules" / "diffusers_modules" / "local"
        module_paths = (
            cache_root / "modules.py",
            cache_root / "unet" / "modules.py",
        )
        for module_path in module_paths:
            if not module_path.exists():
                continue
            source = module_path.read_text(encoding="utf-8")
            patched = self._patch_hunyuan21_pbr_module_source(source)
            if patched != source:
                module_path.write_text(patched, encoding="utf-8")
                logger.info("Patched cached Hunyuan3D-Paint dynamic module: %s", module_path)

    def _patch_hunyuan21_pbr_module_source(self, source: str) -> str:
        patched = source
        if "def _velocity3d_ensure_mda_processor" not in patched:
            patched = patched.replace(
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
        if "def _velocity3d_flatten_pbr_attention" not in patched:
            patched = patched.replace(
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
        if "self._velocity3d_ensure_mda_processor()" not in patched:
            patched = patched.replace(
                "        if self.use_mda:\n"
                "            mda_norm_hidden_states = rearrange(\n",
                "        if self.use_mda:\n"
                "            self._velocity3d_ensure_mda_processor()\n"
                "            mda_norm_hidden_states = rearrange(\n",
            )
        if '"material self"' not in patched:
            patched = patched.replace(
                '            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "material self"\n'
                '            )\n',
                1,
            )
        if '"reference"' not in patched:
            patched = patched.replace(
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
            patched = patched.replace(
                '            attn_output = rearrange(attn_output, "b n_pbr (n l) c -> (b n_pbr n) l c", n=num_in_batch, n_pbr=N_pbr)\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "reference"\n'
                '            )\n',
            )
        if '"multiview"' not in patched:
            patched = patched.replace(
                '            attn_output = rearrange(attn_output, "(b n_pbr) (n l) c -> (b n_pbr n) l c", n_pbr=N_pbr, n=num_in_batch)\n',
                '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
                '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "multiview"\n'
                '            )\n',
            )
        return patched

    def _evict_hunyuan21_dynamic_modules(self) -> None:
        evicted = [
            module_name
            for module_name, module in list(sys.modules.items())
            if module_name.startswith("diffusers_modules.local")
            and "diffusers_modules" in str(getattr(module, "__file__", ""))
        ]
        for module_name in evicted:
            sys.modules.pop(module_name, None)
        if evicted:
            importlib.invalidate_caches()
            logger.info("Reloading patched Hunyuan3D-Paint dynamic modules: %s", ", ".join(sorted(evicted)))

    def _prepare_windows_native_dlls(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import torch
        except Exception:
            return

        dll_dirs: list[Path] = []
        torch_file = getattr(torch, "__file__", None)
        if torch_file:
            dll_dirs.append(Path(torch_file).parent / "lib")

        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_root = None
        if cuda_version:
            cuda_root = os.environ.get(f"CUDA_PATH_V{cuda_version.replace('.', '_')}")
        cuda_root = cuda_root or os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_root:
            dll_dirs.append(Path(cuda_root) / "bin")

        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        for dll_dir in dll_dirs:
            if not dll_dir.exists():
                continue
            try:
                os.add_dll_directory(str(dll_dir))
            except (FileNotFoundError, OSError):
                logger.debug("Could not add DLL directory: %s", dll_dir, exc_info=True)
            if str(dll_dir) not in path_entries:
                path_entries.insert(0, str(dll_dir))
        os.environ["PATH"] = os.pathsep.join(path_entries)

    def _ensure_required_import(self, module_name: str, label: str) -> None:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise ProviderDependencyError(
                f"{label} is not installed. Run Install / Download for the selected Hunyuan model. "
                "The native Hunyuan CUDA rasterizer must be built against the backend PyTorch CUDA runtime."
            ) from exc

    def _apply_torchvision_fix(self, repo_dir: Path, paint_dir: Path) -> None:
        for module_name in ("torchvision_fix", "utils.torchvision_fix"):
            try:
                module = importlib.import_module(module_name)
                apply_fix = getattr(module, "apply_fix", None)
                if callable(apply_fix):
                    apply_fix()
                    return
            except Exception:
                logger.debug("Could not apply %s from %s", module_name, paint_dir, exc_info=True)

        fix_path = repo_dir / "torchvision_fix.py"
        if fix_path.exists():
            logger.debug("Torchvision fix file exists but did not import cleanly: %s", fix_path)

    def _patch_optional_super_resolution(self, texture_module, paint_dir: Path) -> None:
        checkpoint = paint_dir / "ckpt" / "RealESRGAN_x4plus.pth"
        missing_upscaler = importlib.util.find_spec("realesrgan") is None or importlib.util.find_spec("basicsr") is None
        if checkpoint.exists() and not missing_upscaler:
            return
        logger.warning(
            "RealESRGAN is not fully available for Hunyuan3D-Paint; using identity upscale for this pass."
        )
        texture_module.imageSuperNet = _IdentitySuperResolution

    def _configure_hunyuan21_profile(
        self,
        config,
        *,
        pipeline_preset: GenerationPreset | None = None,
    ) -> None:
        requested_profile = os.environ.get("VELOCITY3D_HUNYUAN_PAINT_PROFILE")
        default_profile = pipeline_preset.hunyuan_paint_profile if pipeline_preset else "balanced"
        profile_name = (requested_profile or default_profile).strip().lower()
        profiles = {
            "fast": {
                "render_size": 768,
                "texture_size": 1024,
                "max_selected_view_num": 6,
                "resolution": 512,
                "remesh_target": 25000,
            },
            "balanced": {
                "render_size": 1024,
                "texture_size": 2048,
                "max_selected_view_num": 6,
                "resolution": 512,
                "remesh_target": 50000,
            },
            "quality": {
                "render_size": 1536,
                "texture_size": 3072,
                "max_selected_view_num": 6,
                "resolution": 640,
                "remesh_target": 120000,
            },
            "production": {
                "render_size": 2048,
                "texture_size": 4096,
                "max_selected_view_num": 8,
                "resolution": 768,
                "remesh_target": 200000,
            },
        }
        defaults = profiles.get(profile_name)
        if defaults is None:
            logger.warning(
                "Ignoring unknown VELOCITY3D_HUNYUAN_PAINT_PROFILE=%r; using balanced.",
                profile_name,
            )
            profile_name = "balanced"
            defaults = profiles[profile_name]

        config.render_size = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_RENDER_SIZE",
            default=defaults["render_size"],
            minimum=512,
            maximum=4096,
        )
        config.texture_size = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_TEXTURE_SIZE",
            default=defaults["texture_size"],
            minimum=1024,
            maximum=4096,
        )
        config.max_selected_view_num = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_MAX_VIEWS",
            default=defaults["max_selected_view_num"],
            minimum=6,
            maximum=18,
        )
        config.resolution = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_DIFFUSION_RESOLUTION",
            default=defaults["resolution"],
            minimum=384,
            maximum=1024,
        )
        config.velocity3d_remesh_target = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_REMESH_TARGET",
            default=pipeline_preset.hunyuan_remesh_target if pipeline_preset else defaults["remesh_target"],
            minimum=1000,
            maximum=2_000_000,
        )
        logger.info(
            "Configured Hunyuan3D-Paint profile=%s render=%s texture=%s views=%s diffusion=%s remesh=%s",
            profile_name,
            config.render_size,
            config.texture_size,
            config.max_selected_view_num,
            config.resolution,
            config.velocity3d_remesh_target,
        )

    def _hunyuan21_cuda_device_name(self) -> str:
        try:
            import torch
        except ImportError as exc:
            raise ProviderDependencyError("Hunyuan3D-Paint 2.1 requires torch with CUDA support.") from exc

        if not torch.cuda.is_available():
            raise ProviderDependencyError(
                "Hunyuan3D-Paint 2.1 requires a CUDA PyTorch runtime. "
                "Install the selected Hunyuan model with the torch+cu130 backend runtime."
            )

        requested = os.environ.get("VELOCITY3D_HUNYUAN_PAINT_DEVICE", "cuda:0").strip() or "cuda:0"
        if requested == "cuda":
            requested = "cuda:0"
        device = torch.device(requested)
        if device.type != "cuda":
            raise ProviderDependencyError(
                f"VELOCITY3D_HUNYUAN_PAINT_DEVICE must be a CUDA device, got {requested!r}."
            )
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ProviderDependencyError(
                f"Requested Hunyuan3D-Paint device {requested!r}, but only "
                f"{torch.cuda.device_count()} CUDA device(s) are visible."
            )
        try:
            torch.cuda.set_device(device)
        except Exception as exc:
            raise ProviderDependencyError(f"Could not activate CUDA device {requested!r}: {exc}") from exc
        return str(device)

    def _configure_hunyuan21_runtime(self, pipeline) -> None:
        models = getattr(pipeline, "models", {})
        multiview_model = models.get("multiview_model") if isinstance(models, dict) else None
        paint_pipeline = getattr(multiview_model, "pipeline", None)
        if paint_pipeline is None:
            logger.warning("Hunyuan3D-Paint multiview diffusion pipeline was not exposed for runtime tuning.")
            return

        self._set_progress_bars(paint_pipeline)
        self._enable_if_available(paint_pipeline, "enable_vae_slicing")
        self._enable_if_available(paint_pipeline, "enable_vae_tiling")
        self._enable_if_available(paint_pipeline, "enable_attention_slicing", "max")

        if self._env_bool("VELOCITY3D_HUNYUAN_PAINT_LOW_VRAM", default=True):
            logger.info(
                "Hunyuan3D-Paint low-VRAM mode uses VAE/attention slicing only. "
                "Diffusers CPU offload is disabled because the upstream paint pipeline "
                "mixes DINO and diffusion tensors manually and must stay on one CUDA device."
            )

        self._place_hunyuan21_multiview_on_device(multiview_model, stage="load")
        self._patch_hunyuan21_pipeline_device_guards(multiview_model)

    def _place_hunyuan21_multiview_on_device(self, multiview_model, *, stage: str) -> None:
        if multiview_model is None:
            return
        try:
            import torch
        except ImportError as exc:
            raise ProviderDependencyError("Hunyuan3D-Paint 2.1 requires torch with CUDA support.") from exc

        device = torch.device(self._hunyuan21_cuda_device_name())
        device_name = str(device)
        setattr(multiview_model, "device", device_name)

        paint_pipeline = getattr(multiview_model, "pipeline", None)
        if paint_pipeline is not None:
            pipeline_to = getattr(paint_pipeline, "to", None)
            if callable(pipeline_to):
                pipeline_to(device)
            pipeline_eval = getattr(paint_pipeline, "eval", None)
            if callable(pipeline_eval):
                pipeline_eval()

        dino_model = getattr(multiview_model, "dino_v2", None)
        dino_dtype = self._module_dtype(getattr(paint_pipeline, "unet", None)) or torch.float16
        if dino_model is not None:
            dino_to = getattr(dino_model, "to", None)
            if callable(dino_to):
                try:
                    dino_to(device=device, dtype=dino_dtype)
                except TypeError:
                    dino_to(device)
            dino_eval = getattr(dino_model, "eval", None)
            if callable(dino_eval):
                dino_eval()

        summary = self._hunyuan21_device_summary(multiview_model)
        wrong_devices = [
            f"{name}={value}"
            for name, value in summary.items()
            if value != "unavailable" and not value.startswith("cuda")
        ]
        if wrong_devices:
            raise ProviderExecutionError(
                "Hunyuan3D-Paint device placement failed before "
                f"{stage}: {', '.join(wrong_devices)}. Restart Velocity3D and rerun Install / Download "
                "so the paint runtime is loaded without Diffusers CPU offload hooks."
            )
        logger.info(
            "Hunyuan3D-Paint device check before %s: %s",
            stage,
            ", ".join(f"{name}={value}" for name, value in summary.items()),
        )

    def _hunyuan21_device_summary(self, multiview_model) -> dict[str, str]:
        paint_pipeline = getattr(multiview_model, "pipeline", None)
        summary = {
            "pipeline": self._device_name(getattr(paint_pipeline, "device", None)),
            "execution": self._device_name(getattr(paint_pipeline, "_execution_device", None)),
            "unet": self._device_name(self._module_device(getattr(paint_pipeline, "unet", None))),
            "vae": self._device_name(self._module_device(getattr(paint_pipeline, "vae", None))),
            "text_encoder": self._device_name(self._module_device(getattr(paint_pipeline, "text_encoder", None))),
            "dino": self._device_name(self._module_device(getattr(multiview_model, "dino_v2", None))),
        }
        return summary

    def _module_device(self, module):
        if module is None:
            return None
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return None
        try:
            parameter = next(parameters())
        except StopIteration:
            return None
        except Exception:
            logger.debug("Could not inspect module device for %r", module, exc_info=True)
            return None
        return getattr(parameter, "device", None)

    def _module_dtype(self, module):
        if module is None:
            return None
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            return None
        try:
            parameter = next(parameters())
        except StopIteration:
            return None
        except Exception:
            logger.debug("Could not inspect module dtype for %r", module, exc_info=True)
            return None
        return getattr(parameter, "dtype", None)

    def _device_name(self, device) -> str:
        if device is None:
            return "unavailable"
        return str(device)

    def _patch_hunyuan21_pipeline_device_guards(self, multiview_model) -> None:
        paint_pipeline = getattr(multiview_model, "pipeline", None)
        if paint_pipeline is None:
            return

        unet = getattr(paint_pipeline, "unet", None)
        self._patch_hunyuan21_live_attention_blocks(unet)
        self._patch_hunyuan21_mda_attention_processors(unet)

        if getattr(paint_pipeline, "_velocity3d_device_guards", False):
            return

        self._wrap_module_forward_on_own_device(getattr(paint_pipeline, "unet", None), "paint_unet")
        self._wrap_module_forward_on_own_device(getattr(paint_pipeline, "text_encoder", None), "paint_text_encoder")
        self._wrap_module_forward_on_own_device(getattr(paint_pipeline, "vae", None), "paint_vae")
        self._wrap_module_method_on_own_device(getattr(paint_pipeline, "vae", None), "encode", "paint_vae.encode")
        self._wrap_module_method_on_own_device(getattr(paint_pipeline, "vae", None), "decode", "paint_vae.decode")
        self._wrap_hunyuan21_encode_images(paint_pipeline)

        paint_pipeline._velocity3d_device_guards = True
        logger.info("Installed Hunyuan3D-Paint runtime tensor-device guards.")

    def _patch_hunyuan21_live_attention_blocks(self, unet) -> None:
        if unet is None:
            return

        try:
            runtime_modules = importlib.import_module("hunyuanpaintpbr.unet.modules")
            runtime_block_cls = getattr(runtime_modules, "Basic2p5DTransformerBlock")
        except Exception as exc:
            raise ProviderExecutionError(
                "Velocity3D Hunyuan3D-Paint runtime module did not import. "
                "Run Install / Download so BASE_DIR/Runtimes/hunyuan3d-2.1-paint is materialized."
            ) from exc

        patched_classes: set[type] = set()
        block_count = 0
        modules = getattr(unet, "modules", None)
        if not callable(modules):
            return

        for module in modules():
            if module.__class__.__name__ != "Basic2p5DTransformerBlock":
                continue
            block_count += 1
            target_cls = module.__class__
            if target_cls in patched_classes or getattr(target_cls, "_velocity3d_live_runtime_forward", False):
                continue
            target_cls._velocity3d_ensure_mda_processor = runtime_block_cls._velocity3d_ensure_mda_processor
            target_cls._velocity3d_flatten_pbr_attention = runtime_block_cls._velocity3d_flatten_pbr_attention
            target_cls.forward = runtime_block_cls.forward
            target_cls._velocity3d_live_runtime_forward = True
            patched_classes.add(target_cls)

        if block_count and patched_classes:
            logger.info(
                "Installed Velocity3D-owned Hunyuan3D-Paint attention forward on %s live block class(es), %s block(s).",
                len(patched_classes),
                block_count,
            )
        elif block_count:
            logger.info("Velocity3D-owned Hunyuan3D-Paint attention forward was already installed on %s block(s).", block_count)

    def _patch_hunyuan21_mda_attention_processors(self, unet) -> None:
        if unet is None:
            return

        repaired = 0
        checked = 0
        skipped = 0
        modules = getattr(unet, "modules", None)
        if not callable(modules):
            return

        for module in modules():
            if not getattr(module, "use_mda", False):
                continue
            attn1 = getattr(module, "attn1", None)
            if attn1 is None:
                continue

            checked += 1
            processor = getattr(attn1, "processor", None)
            if processor.__class__.__name__ != "SelfAttnProcessor2_0":
                processor_cls = self._resolve_hunyuan21_self_attn_processor(module)
                set_processor = getattr(attn1, "set_processor", None)
                if processor_cls is None or not callable(set_processor):
                    skipped += 1
                    continue

                try:
                    processor = processor_cls(
                        query_dim=getattr(module, "dim"),
                        heads=getattr(module, "num_attention_heads"),
                        dim_head=getattr(module, "attention_head_dim"),
                        dropout=getattr(module, "dropout", 0.0),
                        bias=getattr(module, "attention_bias", False),
                        cross_attention_dim=None,
                        upcast_attention=getattr(attn1, "upcast_attention", False),
                        out_bias=True,
                        pbr_setting=getattr(module, "pbr_setting", None) or ["albedo", "mr"],
                    )
                    set_processor(processor)
                    self._copy_hunyuan21_mda_projection_weights(module)
                    repaired += 1
                except Exception:
                    skipped += 1
                    logger.debug("Could not reinstall Hunyuan3D-Paint MDA attention processor.", exc_info=True)
                    continue

            self._move_hunyuan21_attention_processor_to_attn_device(attn1)

        if repaired:
            logger.info(
                "Reinstalled Hunyuan3D-Paint PBR attention processors on %s/%s material-aware blocks.",
                repaired,
                checked,
            )
        elif skipped:
            logger.warning(
                "Skipped %s Hunyuan3D-Paint material-aware attention blocks because their processor could not be repaired.",
                skipped,
            )

    def _resolve_hunyuan21_self_attn_processor(self, module):
        module_name = getattr(module.__class__, "__module__", "")
        source_module = sys.modules.get(module_name)
        processor_cls = getattr(source_module, "SelfAttnProcessor2_0", None)
        if isinstance(processor_cls, type):
            return processor_cls

        if module_name.endswith(".modules"):
            try:
                attn_module = importlib.import_module(f"{module_name[:-len('.modules')]}.attn_processor")
            except Exception:
                logger.debug("Could not import Hunyuan3D-Paint attention processor module for %s.", module_name, exc_info=True)
                return None
            processor_cls = getattr(attn_module, "SelfAttnProcessor2_0", None)
            if isinstance(processor_cls, type):
                return processor_cls
        return None

    def _copy_hunyuan21_mda_projection_weights(self, module) -> None:
        attn1 = getattr(module, "attn1", None)
        processor = getattr(attn1, "processor", None)
        if attn1 is None or processor is None:
            return

        for token in getattr(module, "pbr_setting", None) or ["albedo", "mr"]:
            if token == "albedo":
                continue
            for projection in ("to_q", "to_k", "to_v", "to_out"):
                source = getattr(attn1, projection, None)
                target = getattr(processor, f"{projection}_{token}", None)
                if source is not None and target is not None:
                    target.load_state_dict(source.state_dict())

    def _move_hunyuan21_attention_processor_to_attn_device(self, attn) -> None:
        processor = getattr(attn, "processor", None)
        move = getattr(processor, "to", None)
        if not callable(move):
            return

        device = self._module_device(getattr(attn, "to_q", None))
        dtype = self._module_dtype(getattr(attn, "to_q", None))
        if device is None:
            return

        try:
            if dtype is not None:
                move(device=device, dtype=dtype)
            else:
                move(device=device)
        except Exception:
            logger.debug("Could not move Hunyuan3D-Paint attention processor to the attention device.", exc_info=True)

    def _wrap_hunyuan21_encode_images(self, paint_pipeline) -> None:
        if getattr(paint_pipeline, "_velocity3d_encode_images_guard", False):
            return
        original_encode_images = getattr(paint_pipeline, "encode_images", None)
        if not callable(original_encode_images):
            return

        def encode_images_with_device(images, *args, **kwargs):
            vae = getattr(paint_pipeline, "vae", None)
            device = self._module_device(vae)
            dtype = self._module_dtype(vae)
            if device is not None:
                images = self._move_tensors_to_device(images, device=device, dtype=dtype)
            return original_encode_images(images, *args, **kwargs)

        paint_pipeline.encode_images = encode_images_with_device
        paint_pipeline._velocity3d_encode_images_guard = True

    def _wrap_module_forward_on_own_device(self, module, label: str) -> None:
        self._wrap_module_method_on_own_device(module, "forward", label)

    def _wrap_module_method_on_own_device(self, module, method_name: str, label: str) -> None:
        if module is None:
            return
        marker = f"_velocity3d_{method_name}_device_guard"
        if getattr(module, marker, False):
            return
        original_method = getattr(module, method_name, None)
        if not callable(original_method):
            return

        def guarded_method(*args, **kwargs):
            device = self._module_device(module)
            dtype = self._module_dtype(module)
            if device is not None:
                args = self._move_tensors_to_device(args, device=device, dtype=dtype)
                kwargs = self._move_tensors_to_device(kwargs, device=device, dtype=dtype)
            return original_method(*args, **kwargs)

        setattr(module, method_name, guarded_method)
        setattr(module, marker, True)
        logger.debug("Installed Hunyuan3D-Paint device guard on %s.%s", label, method_name)

    def _move_tensors_to_device(self, value, *, device, dtype=None):
        try:
            import torch
        except ImportError:
            return value

        if isinstance(value, torch.Tensor):
            if value.is_floating_point() and dtype is not None:
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if isinstance(value, dict):
            for key, item in list(value.items()):
                value[key] = self._move_tensors_to_device(item, device=device, dtype=dtype)
            return value
        if isinstance(value, list):
            for index, item in enumerate(value):
                value[index] = self._move_tensors_to_device(item, device=device, dtype=dtype)
            return value
        if isinstance(value, tuple):
            return tuple(self._move_tensors_to_device(item, device=device, dtype=dtype) for item in value)
        return value

    def _patch_hunyuan21_dino_forward(self) -> None:
        try:
            import torch
            from hunyuanpaintpbr.unet import modules as modules_module
        except Exception:
            logger.debug("Could not patch Hunyuan3D-Paint DINO forward device handling.", exc_info=True)
            return

        dino_class = getattr(modules_module, "Dino_v2", None)
        if not isinstance(dino_class, type) or getattr(dino_class, "_velocity3d_device_forward", False):
            return

        rearrange = getattr(modules_module, "rearrange")

        def forward_with_device(self, images):
            dino_param = next(self.dino_v2.parameters())
            if isinstance(images, torch.Tensor):
                batch_size = images.shape[0]
                processed_images = self.dino_processor(
                    images=rearrange(images, "b n c h w -> (b n) c h w"),
                    return_tensors="pt",
                    do_rescale=False,
                ).pixel_values
            else:
                batch_size = 1
                processed_images = self.dino_processor(images=images, return_tensors="pt").pixel_values

            processed_images = processed_images.to(device=dino_param.device, dtype=dino_param.dtype)
            dino_hidden_states = self.dino_v2(processed_images)[0]
            dino_hidden_states = rearrange(
                dino_hidden_states.to(device=dino_param.device, dtype=dino_param.dtype),
                "(b n) l c -> b (n l) c",
                b=batch_size,
            )
            return dino_hidden_states

        dino_class.forward = forward_with_device
        dino_class._velocity3d_device_forward = True

    def _patch_hunyuan21_attention_shapes(self) -> None:
        try:
            attn_module = importlib.import_module("hunyuanpaintpbr.unet.attn_processor")
        except Exception:
            logger.debug("Could not patch Hunyuan3D-Paint attention shape handling.", exc_info=True)
            return

        attn_core = getattr(attn_module, "AttnCore", None)
        if not isinstance(attn_core, type) or getattr(attn_core, "_velocity3d_shape_patch", False):
            return

        attn_utils = getattr(attn_module, "AttnUtils")
        functional = getattr(attn_module, "F")

        def process_attention_base(
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            get_qkv_fn=None,
            apply_rope_fn=None,
            **kwargs,
        ):
            hidden_states, residual, input_ndim, shape_info = attn_utils.prepare_hidden_states(
                hidden_states,
                attn,
                temb,
            )

            target_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states
            if target_states.ndim > 3:
                target_states = target_states.reshape(-1, target_states.shape[-2], target_states.shape[-1])
                if encoder_hidden_states is None:
                    hidden_states = target_states
                else:
                    encoder_hidden_states = target_states
            batch_size, sequence_length, _ = target_states.shape

            attention_mask = attn_utils.prepare_attention_mask(attention_mask, attn, sequence_length, batch_size)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

            query, key, value = get_qkv_fn(attn, hidden_states, encoder_hidden_states, **kwargs)
            inner_dim = key.shape[-1]
            head_dim = inner_dim // attn.heads

            query = attn_utils.reshape_qkv_for_attention(query, batch_size, attn.heads, head_dim)
            key = attn_utils.reshape_qkv_for_attention(key, batch_size, attn.heads, head_dim)
            value = attn_utils.reshape_qkv_for_attention(value, batch_size, attn.heads, value.shape[-1] // attn.heads)

            query, key = attn_utils.apply_norms(
                query,
                key,
                getattr(attn, "norm_q", None),
                getattr(attn, "norm_k", None),
            )

            if apply_rope_fn is not None:
                query, key = apply_rope_fn(query, key, head_dim, **kwargs)

            hidden_states = functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            return hidden_states, residual, input_ndim, shape_info, batch_size, attn.heads, head_dim

        attn_core.process_attention_base = staticmethod(process_attention_base)
        attn_core._velocity3d_shape_patch = True

    def _set_progress_bars(self, pipeline) -> None:
        set_progress = getattr(pipeline, "set_progress_bar_config", None)
        if callable(set_progress):
            try:
                set_progress(disable=True)
            except Exception:
                logger.debug("Could not configure Hunyuan3D-Paint progress bars.", exc_info=True)

    def _enable_if_available(self, target, method_name: str, *args) -> None:
        method = getattr(target, method_name, None)
        if not callable(method):
            return
        try:
            method(*args)
            logger.info("Enabled Hunyuan3D-Paint runtime optimization: %s", method_name)
        except Exception:
            logger.debug("Could not enable Hunyuan3D-Paint optimization: %s", method_name, exc_info=True)

    def _patch_hunyuan21_stage_logging(self, texture_module) -> None:
        self._patch_hunyuan21_dino_forward()
        self._patch_hunyuan21_attention_shapes()

        if getattr(texture_module, "_velocity3d_stage_logging", False):
            return

        original_remesh_mesh = getattr(texture_module, "remesh_mesh", None)
        if callable(original_remesh_mesh):
            def logged_remesh_mesh(mesh_path, remesh_path, *args, **kwargs):
                started = time.time()
                target_count = getattr(texture_module, "_velocity3d_remesh_target_count", None)
                logger.info(
                    "Hunyuan3D-Paint remesh started: %s target_faces=%s",
                    mesh_path,
                    target_count or "upstream",
                )
                if target_count:
                    simplify_module = importlib.import_module("utils.simplify_mesh_utils")
                    result = simplify_module.mesh_simplify_trimesh(
                        mesh_path,
                        remesh_path,
                        target_count=int(target_count),
                    )
                else:
                    result = original_remesh_mesh(mesh_path, remesh_path, *args, **kwargs)
                logger.info(
                    "Hunyuan3D-Paint remesh finished in %.1fs: %s",
                    time.time() - started,
                    remesh_path,
                )
                return result

            texture_module.remesh_mesh = logged_remesh_mesh

        original_mesh_uv_wrap = getattr(texture_module, "mesh_uv_wrap", None)
        if callable(original_mesh_uv_wrap):
            def logged_mesh_uv_wrap(mesh, *args, **kwargs):
                face_count = self._mesh_face_count(mesh)
                started = time.time()
                logger.info("Hunyuan3D-Paint UV unwrap started on %s faces", face_count)
                result = original_mesh_uv_wrap(mesh, *args, **kwargs)
                result_face_count = self._mesh_face_count(result)
                logger.info(
                    "Hunyuan3D-Paint UV unwrap finished in %.1fs on %s faces",
                    time.time() - started,
                    result_face_count,
                )
                return result

            texture_module.mesh_uv_wrap = logged_mesh_uv_wrap

        original_view_processor = getattr(texture_module, "ViewProcessor", None)
        if isinstance(original_view_processor, type):
            class LoggedViewProcessor(original_view_processor):
                def render_normal_multiview(self, *args, **kwargs):
                    started = time.time()
                    logger.info("Hunyuan3D-Paint normal-view render started")
                    result = super().render_normal_multiview(*args, **kwargs)
                    logger.info("Hunyuan3D-Paint normal-view render finished in %.1fs", time.time() - started)
                    return result

                def render_position_multiview(self, *args, **kwargs):
                    started = time.time()
                    logger.info("Hunyuan3D-Paint position-view render started")
                    result = super().render_position_multiview(*args, **kwargs)
                    logger.info("Hunyuan3D-Paint position-view render finished in %.1fs", time.time() - started)
                    return result

                def bake_view_selection(self, *args, **kwargs):
                    started = time.time()
                    logger.info("Hunyuan3D-Paint bake view selection started")
                    result = super().bake_view_selection(*args, **kwargs)
                    logger.info(
                        "Hunyuan3D-Paint bake view selection finished in %.1fs with %s views",
                        time.time() - started,
                        len(result[0]) if result else 0,
                    )
                    return result

                def bake_from_multiview(self, views, camera_elevs, camera_azims, view_weights):
                    started = time.time()
                    logger.info("Hunyuan3D-Paint UV bake started for %s views", len(views))
                    result = super().bake_from_multiview(views, camera_elevs, camera_azims, view_weights)
                    logger.info("Hunyuan3D-Paint UV bake finished in %.1fs", time.time() - started)
                    return result

                def texture_inpaint(self, *args, **kwargs):
                    started = time.time()
                    logger.info("Hunyuan3D-Paint texture inpaint started")
                    result = super().texture_inpaint(*args, **kwargs)
                    logger.info("Hunyuan3D-Paint texture inpaint finished in %.1fs", time.time() - started)
                    return result

            LoggedViewProcessor.__name__ = original_view_processor.__name__
            texture_module.ViewProcessor = LoggedViewProcessor

        multiview_class = getattr(texture_module, "multiviewDiffusionNet", None)
        if isinstance(multiview_class, type) and not getattr(multiview_class, "_velocity3d_forward_logging", False):
            original_forward_one = multiview_class.forward_one

            def logged_forward_one(multiview_self, input_images, control_images, *args, **kwargs):
                self._place_hunyuan21_multiview_on_device(multiview_self, stage="multiview diffusion")
                self._patch_hunyuan21_pipeline_device_guards(multiview_self)
                paint_pipeline = getattr(multiview_self, "pipeline", None)
                original_denoise = getattr(paint_pipeline, "denoise", None)
                custom_view_size = kwargs.get("custom_view_size") or getattr(paint_pipeline, "view_size", "unknown")
                num_views = len(control_images) // 2 if hasattr(control_images, "__len__") else "unknown"
                started = time.time()
                logger.info(
                    "Hunyuan3D-Paint multiview diffusion started: views=%s resolution=%s",
                    num_views,
                    custom_view_size,
                )

                if callable(original_denoise):
                    def denoise_with_logging(*denoise_args, **denoise_kwargs):
                        total_steps = denoise_kwargs.get("num_inference_steps")
                        user_callback = denoise_kwargs.get("callback_on_step_end")
                        last_log = {"step": -1}

                        def callback_with_logging(pipe, step, timestep, callback_kwargs):
                            if step != last_log["step"]:
                                logger.info(
                                    "Hunyuan3D-Paint multiview diffusion step %s/%s",
                                    step + 1,
                                    total_steps or "?",
                                )
                                last_log["step"] = step
                            if callable(user_callback):
                                callback_result = user_callback(pipe, step, timestep, callback_kwargs)
                                return callback_result if callback_result is not None else callback_kwargs
                            return callback_kwargs

                        denoise_kwargs["callback_on_step_end"] = callback_with_logging
                        return original_denoise(*denoise_args, **denoise_kwargs)

                    paint_pipeline.denoise = denoise_with_logging

                try:
                    result = original_forward_one(multiview_self, input_images, control_images, *args, **kwargs)
                    logger.info(
                        "Hunyuan3D-Paint multiview diffusion finished in %.1fs",
                        time.time() - started,
                    )
                    return result
                finally:
                    if callable(original_denoise):
                        paint_pipeline.denoise = original_denoise

            multiview_class.forward_one = logged_forward_one
            multiview_class._velocity3d_forward_logging = True

        texture_module._velocity3d_stage_logging = True

    def _env_int(self, name: str, *, default: int, minimum: int, maximum: int) -> int:
        raw = os.environ.get(name)
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; expected an integer.", name, raw)
            return default
        return max(minimum, min(maximum, value))

    def _env_bool(self, name: str, *, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def _mesh_face_count(self, mesh) -> int:
        faces = getattr(mesh, "faces", None)
        if faces is not None:
            try:
                return len(faces)
            except TypeError:
                pass
        geometry = getattr(mesh, "geometry", None)
        if geometry:
            total = 0
            for item in geometry.values():
                total += self._mesh_face_count(item)
            return total
        return 0

    def _write_reference_image(self, image_bytes: bytes, image_path: Path) -> None:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
        except Exception as exc:
            raise ProviderExecutionError(f"Could not decode Hunyuan paint reference image: {exc}") from exc
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

    def _collect_hunyuan_obj_textures(self, obj_path: Path) -> dict[str, Path | None]:
        base = obj_path.with_suffix("")
        paths = {
            "albedo": base.with_suffix(".jpg"),
            "metallic": Path(f"{base}_metallic.jpg"),
            "roughness": Path(f"{base}_roughness.jpg"),
            "normal": Path(f"{base}_normal.jpg"),
        }
        if not paths["albedo"].exists():
            raise ProviderExecutionError(f"Hunyuan3D-Paint did not write an albedo texture: {paths['albedo']}")
        return paths

    def _pack_obj_with_pbr_textures(
        self,
        obj_path: Path,
        texture_paths: dict[str, Path | None],
        output_glb: Path,
    ) -> None:
        try:
            import pygltflib
            import trimesh
        except ImportError as exc:
            raise ProviderDependencyError(
                "Packing Hunyuan3D-Paint output requires trimesh and pygltflib."
            ) from exc

        output_glb.parent.mkdir(parents=True, exist_ok=True)
        temp_glb = obj_path.with_name(f"{obj_path.stem}_packed_source.glb")
        mr_path = obj_path.with_name(f"{obj_path.stem}_metallic_roughness.png")

        mesh = trimesh.load(str(obj_path), force="scene")
        mesh.export(str(temp_glb))

        gltf = pygltflib.GLTF2().load(str(temp_glb))
        images: list = []
        textures: list = []

        def add_texture(path: Path, mime_type: str) -> int:
            with path.open("rb") as handle:
                encoded = base64.b64encode(handle.read()).decode("ascii")
            images.append(pygltflib.Image(uri=f"data:{mime_type};base64,{encoded}"))
            textures.append(pygltflib.Texture(source=len(images) - 1))
            return len(textures) - 1

        albedo_index = add_texture(texture_paths["albedo"], "image/jpeg")  # type: ignore[arg-type]
        mr_index = None
        metallic_path = texture_paths.get("metallic")
        roughness_path = texture_paths.get("roughness")
        if metallic_path and roughness_path and metallic_path.exists() and roughness_path.exists():
            self._write_metallic_roughness_texture(metallic_path, roughness_path, mr_path)
            mr_index = add_texture(mr_path, "image/png")

        normal_index = None
        normal_path = texture_paths.get("normal")
        if normal_path and normal_path.exists():
            normal_index = add_texture(normal_path, "image/jpeg")

        material = pygltflib.Material(
            name="Velocity3D Hunyuan Paint PBR",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
                metallicFactor=1.0,
                roughnessFactor=1.0,
                baseColorTexture=pygltflib.TextureInfo(index=albedo_index),
            ),
        )
        if mr_index is not None:
            material.pbrMetallicRoughness.metallicRoughnessTexture = pygltflib.TextureInfo(index=mr_index)
        if normal_index is not None:
            material.normalTexture = pygltflib.NormalTextureInfo(index=normal_index)

        gltf.images = images
        gltf.textures = textures
        gltf.materials = [material]
        for mesh_item in gltf.meshes or []:
            for primitive in mesh_item.primitives or []:
                primitive.material = 0

        gltf.save_binary(str(output_glb))

    def _write_metallic_roughness_texture(self, metallic_path: Path, roughness_path: Path, output_path: Path) -> None:
        metallic = Image.open(metallic_path).convert("L")
        roughness = Image.open(roughness_path).convert("L")
        if roughness.size != metallic.size:
            roughness = roughness.resize(metallic.size, Image.Resampling.LANCZOS)

        output = Image.new("RGB", metallic.size)
        output.putchannel("R", Image.new("L", metallic.size, 255))
        output.putchannel("G", roughness)
        output.putchannel("B", metallic)
        output.save(output_path)

    def _assert_glb(self, path: Path) -> None:
        try:
            with path.open("rb") as handle:
                if handle.read(4) == b"glTF":
                    return
        except OSError:
            pass
        raise ProviderExecutionError(f"Hunyuan3D-Paint output is not a valid GLB: {path}")

    def _release_cuda_cache(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Could not release CUDA cache after Hunyuan paint", exc_info=True)
