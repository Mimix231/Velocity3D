"""
POST /generate - runs the selected provider in a thread pool and returns the GLB path.
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from backend.models import GenerationMetadata, GenerationRequest, GenerationResponse
from backend.pipelines.bpy_post import BpyPostProcessor, BpyPostProcessorError
from backend.pipelines.hunyuan_paint import HunyuanPaintPipeline, supports_hunyuan_paint
from backend.pipelines.texture_to_image import (
    DEFAULT_TEXTURE_CHECKPOINT,
    TextureToImagePipeline,
    build_projected_texture_set,
)
from backend.providers import (
    ProviderCapabilityError,
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderExecutionError,
    get_provider_for_request,
)
from backend.providers.helpers import base_dir

logger = logging.getLogger("velocity3d.routes.generate")

router = APIRouter()

_active_events: Dict[str, threading.Event] = {}
_active_events_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)

OUTPUT_DIR = base_dir() / "outputs"
TEXTURE_DIR = OUTPUT_DIR / "textures"
_texture_pipeline = TextureToImagePipeline()
_hunyuan_paint_pipeline = HunyuanPaintPipeline()
_bpy_processor = BpyPostProcessor()


@dataclass
class TextureApplicationResult:
    output_path: str
    material_texture_dir: str | None = None
    material_textures: list[str] = field(default_factory=list)


def _register_event(request_id: str) -> threading.Event:
    event = threading.Event()
    with _active_events_lock:
        _active_events[request_id] = event
    return event


def _unregister_event(request_id: str) -> None:
    with _active_events_lock:
        _active_events.pop(request_id, None)


def get_cancellation_event(request_id: str) -> threading.Event | None:
    with _active_events_lock:
        return _active_events.get(request_id)


@router.post("/generate")
async def generate(req: GenerationRequest):
    start = time.time()
    logger.info("POST /generate type=%s model=%s request_id=%s", req.type, req.model_id, req.request_id)

    cancellation_event = _register_event(req.request_id)

    try:
        entry, provider = get_provider_for_request(req)
        texture_checkpoint = None
        texture_applied = False
        material_texture_dir = None
        material_textures: list[str] = []
        image_reference_bytes: bytes | None = None
        if req.type == "image" and req.reference_image_base64:
            image_reference_bytes = base64.b64decode(req.reference_image_base64 or "")

        needs_texture_pipeline = bool(
            req.texture_options
            and req.texture_options.enabled
            and not (req.type == "image" and image_reference_bytes is not None)
        )
        if needs_texture_pipeline:
            texture_checkpoint = req.texture_options.checkpoint or DEFAULT_TEXTURE_CHECKPOINT
            texture_pipeline_mode = "img2img" if image_reference_bytes is not None else "text2img"
            logger.info("Preparing Stable Diffusion texture checkpoint before provider load: %s", texture_checkpoint)
            texture_checkpoint_path = await _run_in_executor(
                lambda: _prepare_texture_checkpoint(texture_checkpoint, mode=texture_pipeline_mode)
            )
            logger.info("Stable Diffusion texture checkpoint loaded from %s", texture_checkpoint_path)
            logger.info("Stable Diffusion texture pipeline released before 3D provider execution")

        if req.type == "text":
            output_path = await _run_in_executor(
                lambda: provider.generate_text(req.prompt or "", OUTPUT_DIR, cancellation_event)
            )
        else:
            image_bytes = base64.b64decode(req.image_base64 or "")
            if image_reference_bytes is None:
                image_reference_bytes = image_bytes
            output_path = await _run_in_executor(
                lambda: provider.generate_image(image_bytes, OUTPUT_DIR, cancellation_event, prompt=req.prompt)
            )

        if req.texture_options and req.texture_options.enabled:
            use_hunyuan_paint = bool(image_reference_bytes and supports_hunyuan_paint(entry.id))
            texture_checkpoint = (
                "Hunyuan3D-Paint"
                if use_hunyuan_paint
                else texture_checkpoint or req.texture_options.checkpoint or DEFAULT_TEXTURE_CHECKPOINT
            )
            if use_hunyuan_paint and hasattr(provider, "release_pipeline"):
                provider.release_pipeline()
            texture_result = await _run_in_executor(
                lambda: _apply_ai_texture(
                    source_glb=output_path,
                    prompt=req.prompt,
                    checkpoint=texture_checkpoint,
                    request_id=req.request_id,
                    cancellation_event=cancellation_event,
                    reference_image_bytes=image_reference_bytes,
                    model_id=entry.id,
                )
            )
            output_path = texture_result.output_path
            material_texture_dir = texture_result.material_texture_dir
            material_textures = texture_result.material_textures
            texture_applied = True

        output_path, vertex_count, face_count = _get_mesh_stats(output_path)

    except (ProviderCapabilityError, ProviderConfigurationError) as exc:
        logger.warning("Provider validation error for %s: %s", req.request_id, exc)
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_provider", "details": str(exc)},
        )
    except ProviderDependencyError as exc:
        logger.warning("Provider dependency error for %s: %s", req.request_id, exc)
        return JSONResponse(
            status_code=409,
            content={"error": "provider_not_ready", "details": str(exc)},
        )
    except (ProviderExecutionError, BpyPostProcessorError) as exc:
        logger.error("Provider execution error for %s: %s", req.request_id, exc)
        return JSONResponse(
            status_code=500,
            content={"error": "provider_execution_failed", "details": str(exc)},
        )
    except Exception as exc:
        logger.error("Unexpected error for %s: %s", req.request_id, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "internal_error", "details": str(exc)},
        )
    finally:
        _unregister_event(req.request_id)

    elapsed_ms = int((time.time() - start) * 1000)
    response = GenerationResponse(
        request_id=req.request_id,
        model_path=output_path,
        metadata=GenerationMetadata(
            vertex_count=vertex_count,
            face_count=face_count,
            generation_time_ms=elapsed_ms,
            pipeline=entry.family,
            model_id=entry.id,
            model_name=entry.name,
            texture_applied=texture_applied,
            texture_checkpoint=texture_checkpoint,
            material_texture_dir=material_texture_dir,
            material_textures=material_textures,
        ),
    )
    logger.info("POST /generate done in %dms -> %s", elapsed_ms, output_path)
    return response


@router.get("/outputs/latest")
async def latest_generated_output(
    after_ms: int = Query(0, ge=0),
    stable_ms: int = Query(1800, ge=0),
):
    candidate = _find_latest_stable_glb(after_ms=after_ms, stable_ms=stable_ms)
    if candidate is None:
        return JSONResponse(
            status_code=404,
            content={"error": "no_output", "details": "No stable generated GLB is available yet."},
        )

    stat = candidate.stat()
    vertex_count, face_count = _get_glb_mesh_stats(candidate) or (0, 0)
    return {
        "model_path": str(candidate),
        "size": stat.st_size,
        "modified_ms": int(stat.st_mtime * 1000),
        "vertex_count": vertex_count,
        "face_count": face_count,
    }


async def _run_in_executor(fn):
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn)


def _get_mesh_stats(output_path: str):
    path = Path(output_path)
    if path.suffix.lower() == ".glb":
        stats = _get_glb_mesh_stats(path)
        if stats is not None:
            vertex_count, face_count = stats
            return output_path, vertex_count, face_count

    try:
        import trimesh

        mesh = trimesh.load(output_path)
        if hasattr(mesh, "vertices"):
            return output_path, len(mesh.vertices), len(mesh.faces)
        if hasattr(mesh, "geometry"):
            total_v = sum(len(g.vertices) for g in mesh.geometry.values())
            total_f = sum(len(g.faces) for g in mesh.geometry.values())
            return output_path, total_v, total_f
    except Exception:
        pass
    return output_path, 0, 0


def _find_latest_stable_glb(after_ms: int, stable_ms: int) -> Path | None:
    if not OUTPUT_DIR.exists():
        return None

    now_ms = int(time.time() * 1000)
    cutoff_ms = max(0, after_ms - 500)
    candidates: list[tuple[float, Path]] = []

    for path in OUTPUT_DIR.glob("*.glb"):
        try:
            stat = path.stat()
        except OSError:
            continue

        modified_ms = int(stat.st_mtime * 1000)
        if modified_ms < cutoff_ms:
            continue
        if now_ms - modified_ms < stable_ms:
            continue
        if stat.st_size < 20:
            continue
        if not _looks_like_glb(path):
            continue

        candidates.append((stat.st_mtime, path))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _looks_like_glb(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"glTF"
    except OSError:
        return False


def _get_glb_mesh_stats(path: Path) -> tuple[int, int] | None:
    """
    Read vertex/face counts from the GLB JSON chunk without loading mesh buffers.
    This keeps /generate responsive after exporters have already written the file.
    """
    try:
        with path.open("rb") as handle:
            header = handle.read(12)
            if len(header) != 12 or header[:4] != b"glTF":
                return None

            total_length = int.from_bytes(header[8:12], "little")
            bytes_read = 12

            while bytes_read + 8 <= total_length:
                chunk_header = handle.read(8)
                bytes_read += len(chunk_header)
                if len(chunk_header) != 8:
                    return None

                chunk_length = int.from_bytes(chunk_header[:4], "little")
                chunk_type = chunk_header[4:8]
                chunk_data = handle.read(chunk_length)
                bytes_read += len(chunk_data)
                if len(chunk_data) != chunk_length:
                    return None

                if chunk_type != b"JSON":
                    continue

                document = json.loads(chunk_data.rstrip(b" \t\r\n\0").decode("utf-8"))
                accessors = document.get("accessors", [])
                vertex_count = 0
                face_count = 0

                for mesh in document.get("meshes", []):
                    for primitive in mesh.get("primitives", []):
                        attributes = primitive.get("attributes", {})
                        position_accessor = attributes.get("POSITION")
                        if isinstance(position_accessor, int) and position_accessor < len(accessors):
                            vertex_count += int(accessors[position_accessor].get("count", 0))

                        indices_accessor = primitive.get("indices")
                        if isinstance(indices_accessor, int) and indices_accessor < len(accessors):
                            face_count += int(accessors[indices_accessor].get("count", 0)) // 3
                        elif isinstance(position_accessor, int) and position_accessor < len(accessors):
                            face_count += int(accessors[position_accessor].get("count", 0)) // 3

                return vertex_count, face_count
    except Exception:
        logger.debug("Could not read GLB stats from %s", path, exc_info=True)
    return None


def _apply_ai_texture(
    source_glb: str,
    prompt: str | None,
    checkpoint: str,
    request_id: str,
    cancellation_event: threading.Event,
    reference_image_bytes: bytes | None = None,
    model_id: str | None = None,
) -> TextureApplicationResult:
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    texture_path = TEXTURE_DIR / f"{request_id}_albedo.png"
    uv_layout_path = TEXTURE_DIR / f"{request_id}_uv_layout.png"
    uv_ready_path = OUTPUT_DIR / f"{Path(source_glb).stem}_uv.glb"
    material_texture_dir = TEXTURE_DIR / f"{request_id}_materials"
    material_output_path = OUTPUT_DIR / f"{Path(source_glb).stem}_material_textured.glb"
    textured_output_path = OUTPUT_DIR / f"{Path(source_glb).stem}_textured.glb"
    uv_albedo_path = material_texture_dir / "uv_albedo.png"
    uv_roughness_path = material_texture_dir / "uv_roughness.png"
    uv_normal_path = material_texture_dir / "uv_normal.png"

    if reference_image_bytes and supports_hunyuan_paint(model_id):
        logger.info("Running Hunyuan3D-Paint texture synthesis for %s", request_id)
        paint_work_dir = material_texture_dir / "hunyuan_paint"
        output_path = _hunyuan_paint_pipeline.apply(
            model_id=model_id or "",
            source_glb=source_glb,
            reference_image_bytes=reference_image_bytes,
            output_glb=OUTPUT_DIR / f"{Path(source_glb).stem}_hunyuan_paint_textured.glb",
            work_dir=paint_work_dir,
            cancellation_event=cancellation_event,
        )
        logger.info("Hunyuan3D-Paint textured GLB export complete for %s: %s", request_id, output_path.output_path)
        return TextureApplicationResult(
            output_path=output_path.output_path,
            material_texture_dir=output_path.texture_dir,
            material_textures=output_path.texture_paths,
        )

    if reference_image_bytes:
        logger.info("Preparing projected reference texture set for %s", request_id)
        projected_textures = build_projected_texture_set(
            reference_image_bytes=reference_image_bytes,
            output_dir=material_texture_dir,
        )
        logger.info("Exporting source-projected textured GLB for %s", request_id)
        projected_output_path = OUTPUT_DIR / f"{Path(source_glb).stem}_projected_textured.glb"
        output_path = _bpy_processor.project_reference_texture_baked(
            source_glb=source_glb,
            albedo_image=str(projected_textures["projected_albedo"]),
            roughness_image=str(projected_textures["projected_roughness"]),
            output_path=str(projected_output_path),
        )
        logger.info("Projected textured GLB export complete for %s: %s", request_id, output_path)
        return TextureApplicationResult(
            output_path=output_path,
            material_texture_dir=str(material_texture_dir),
            material_textures=[str(path) for path in projected_textures.values()],
        )

    uv_source_glb = _bpy_processor.prepare_texture_target(
        source_glb=source_glb,
        output_path=str(uv_ready_path),
        uv_layout_path=str(uv_layout_path),
    )

    try:
        generated_texture = _texture_pipeline.generate(
            prompt=prompt,
            output_path=texture_path,
            cancellation_event=cancellation_event,
            checkpoint=checkpoint,
            uv_layout_path=uv_layout_path,
        )
    finally:
        _texture_pipeline.release()

    output_path = _bpy_processor.apply_texture(
        source_glb=uv_source_glb,
        texture_image=str(generated_texture),
        output_path=str(textured_output_path),
    )
    return TextureApplicationResult(output_path=output_path)


def _prepare_texture_checkpoint(checkpoint: str, mode: str = "text2img") -> Path:
    try:
        return _texture_pipeline.prepare(checkpoint, mode=mode)
    finally:
        _texture_pipeline.release()
