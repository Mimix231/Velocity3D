import logging
import logging.handlers
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

# ── Logging setup ──────────────────────────────────────────────────────────────
log_dir = Path.home() / ".velocity3d"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "velocity3d.log"

handler = logging.handlers.RotatingFileHandler(
    log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[handler, logging.StreamHandler()])
logger = logging.getLogger("velocity3d")

from backend.providers.helpers import configure_huggingface_cache

hf_cache_root = configure_huggingface_cache()
logger.info("Hugging Face cache root: %s", hf_cache_root)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Velocity3D Backend", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_request", "details": str(exc.errors())},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "details": str(exc)},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Route registration ─────────────────────────────────────────────────────────
from backend.routes import cancel, export, generate, images, models
app.include_router(generate.router)
app.include_router(cancel.router)
app.include_router(export.router)
app.include_router(images.router)
app.include_router(models.router)
