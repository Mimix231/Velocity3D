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

from backend.providers import ProviderDependencyError, ProviderExecutionError
from backend.providers.helpers import configure_huggingface_cache, maybe_add_to_syspath, repo_path

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
    Thin adapter around Tencent's real Hunyuan3D-Paint implementations.

    Hunyuan3D-2.1 paints a mesh by rendering geometry-aware normal/position
    views, synthesizing PBR maps, baking those maps to UVs, and exporting an OBJ
    plus texture files. Velocity3D wraps that output into a packed GLB for the
    viewport and history system.
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
    ) -> HunyuanPaintResult:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before Hunyuan3D-Paint texture synthesis")

        repo_dir = repo_path("hunyuan3d-2.1")
        paint_dir = repo_dir / "hy3dpaint"
        if not paint_dir.exists():
            raise ProviderDependencyError(
                "Hunyuan3D-2.1 paint sources are missing. Run Install / Download for Hunyuan3D-2.1."
            )

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
        config.realesrgan_ckpt_path = str(paint_dir / "ckpt" / "RealESRGAN_x4plus.pth")
        config.multiview_pretrained_path = "tencent/Hunyuan3D-2.1"
        config.dino_ckpt_path = "facebook/dinov2-giant"
        self._configure_hunyuan21_profile(config)

        try:
            with self._load_lock:
                logger.info("Loading Hunyuan3D-Paint 2.1 pipeline")
                pipeline = pipeline_cls(config)
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

    def _configure_hunyuan21_profile(self, config) -> None:
        config.render_size = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_RENDER_SIZE",
            default=int(getattr(config, "render_size", 2048)),
            minimum=512,
            maximum=4096,
        )
        config.texture_size = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_TEXTURE_SIZE",
            default=int(getattr(config, "texture_size", 4096)),
            minimum=1024,
            maximum=4096,
        )
        config.max_selected_view_num = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_MAX_VIEWS",
            default=int(getattr(config, "max_selected_view_num", 6)),
            minimum=6,
            maximum=18,
        )
        config.resolution = self._env_int(
            "VELOCITY3D_HUNYUAN_PAINT_DIFFUSION_RESOLUTION",
            default=int(getattr(config, "resolution", 512)),
            minimum=384,
            maximum=1024,
        )

    def _patch_hunyuan21_stage_logging(self, texture_module) -> None:
        if getattr(texture_module, "_velocity3d_stage_logging", False):
            return

        original_remesh_mesh = getattr(texture_module, "remesh_mesh", None)
        if callable(original_remesh_mesh):
            def logged_remesh_mesh(mesh_path, remesh_path, *args, **kwargs):
                started = time.time()
                logger.info("Hunyuan3D-Paint remesh started: %s", mesh_path)
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
