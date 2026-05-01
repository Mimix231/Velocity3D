from __future__ import annotations

import base64
import binascii
import io
import logging
import os
import threading

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from backend.providers.helpers import base_dir

logger = logging.getLogger("velocity3d.routes.images")

router = APIRouter()

_rembg_session = None
_rembg_session_lock = threading.Lock()
_rembg_model_dir = base_dir() / "models" / "rembg"


class BackgroundRemovalRequest(BaseModel):
    image_base64: str = Field(..., min_length=16)


class BackgroundRemovalResponse(BaseModel):
    image_base64: str
    mime_type: str = "image/png"


class BackgroundRemovalDependencyError(RuntimeError):
    pass


def _strip_data_url(value: str) -> str:
    raw = value.strip()
    if raw.startswith("data:") and "," in raw:
        return raw.split(",", 1)[1]
    return raw


def _decode_base64_image(value: str) -> bytes:
    cleaned = "".join(_strip_data_url(value).split())
    try:
        image_bytes = base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64 image data") from exc

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:
        raise ValueError("image_base64 does not contain a readable image") from exc

    return image_bytes


def _get_rembg_session():
    global _rembg_session

    _rembg_model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("U2NET_HOME", str(_rembg_model_dir))

    try:
        from rembg import new_session
    except Exception as exc:  # pragma: no cover - depends on local install
        raise BackgroundRemovalDependencyError(
            "rembg is not available in the backend environment. Install rembg and onnxruntime in the active backend venv."
        ) from exc

    with _rembg_session_lock:
        if _rembg_session is not None:
            return _rembg_session

        # ISNet gives cleaner hard-surface/object masks than the old border-color
        # canvas fallback. If that model is not available locally, use rembg's
        # default session so the feature still works after a plain rembg install.
        try:
            _rembg_session = new_session("isnet-general-use")
        except Exception as exc:
            logger.warning("Could not initialize isnet-general-use rembg session, falling back to default: %s", exc)
            _rembg_session = new_session()
        return _rembg_session


def _rembg_remove(image_bytes: bytes) -> bytes:
    try:
        from rembg import remove
    except Exception as exc:  # pragma: no cover - depends on local install
        raise BackgroundRemovalDependencyError(
            "rembg is not available in the backend environment. Install rembg and onnxruntime in the active backend venv."
        ) from exc

    session = _get_rembg_session()
    try:
        result = remove(
            image_bytes,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=12,
            alpha_matting_erode_size=8,
            post_process_mask=True,
        )
    except TypeError:
        result = remove(image_bytes, session=session)

    if isinstance(result, Image.Image):
        out = io.BytesIO()
        result.save(out, format="PNG")
        return out.getvalue()
    return bytes(result)


def _postprocess_cutout(png_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(png_bytes)) as image:
        rgba = image.convert("RGBA")

    alpha = rgba.getchannel("A")

    # Rembg can leave low-alpha haze around dark studio backgrounds. The hard
    # threshold removes background contamination while keeping antialiased edges.
    alpha = alpha.point(lambda value: 0 if value < 24 else 255 if value > 246 else value, "L")

    cutout = Image.new("RGBA", rgba.size, (0, 0, 0, 0))
    cutout.paste(rgba, (0, 0), alpha)
    cutout.putalpha(alpha)

    out = io.BytesIO()
    cutout.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _remove_background_base64(image_base64: str) -> str:
    source_bytes = _decode_base64_image(image_base64)
    removed_bytes = _rembg_remove(source_bytes)
    cleaned_bytes = _postprocess_cutout(removed_bytes)
    return base64.b64encode(cleaned_bytes).decode("ascii")


@router.post("/images/remove-background")
async def remove_background(req: BackgroundRemovalRequest):
    try:
        image_base64 = await run_in_threadpool(_remove_background_base64, req.image_base64)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_image", "details": str(exc)},
        )
    except BackgroundRemovalDependencyError as exc:
        logger.warning("Background removal dependency error: %s", exc)
        return JSONResponse(
            status_code=409,
            content={"error": "background_removal_unavailable", "details": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        logger.error("Background removal failed: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "background_removal_failed", "details": str(exc)},
        )

    return BackgroundRemovalResponse(image_base64=image_base64)
