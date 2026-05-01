from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models import (
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelInstallRequest,
    ModelInstallStartResponse,
    ModelInstallStatusResponse,
)
from backend.providers import (
    ProviderCapabilityError,
    ProviderConfigurationError,
    get_catalog_response,
)
from backend.providers.registry import (
    download_model_repo,
    get_model_install_status,
    start_model_install,
)

logger = logging.getLogger("velocity3d.routes.models")

router = APIRouter()


@router.get("/models")
async def list_models():
    return get_catalog_response()


@router.post("/models/download")
async def download_model(req: ModelDownloadRequest):
    try:
        destination = download_model_repo(req.model_id)
    except (ProviderCapabilityError, ProviderConfigurationError) as exc:
        logger.warning("Model download rejected for %s: %s", req.model_id, exc)
        return JSONResponse(
            status_code=400,
            content={"error": "download_unavailable", "details": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        logger.error("Model download failed for %s: %s", req.model_id, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "download_failed", "details": str(exc)},
        )

    return ModelDownloadResponse(model_id=req.model_id, destination=str(destination))


@router.post("/models/install")
async def install_model(req: ModelInstallRequest):
    try:
        job = start_model_install(req.model_id)
    except (ProviderCapabilityError, ProviderConfigurationError) as exc:
        logger.warning("Model install rejected for %s: %s", req.model_id, exc)
        return JSONResponse(
            status_code=400,
            content={"error": "install_unavailable", "details": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        logger.error("Model install failed to start for %s: %s", req.model_id, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "install_start_failed", "details": str(exc)},
        )

    return ModelInstallStartResponse(job_id=job.job_id, model_id=job.model_id)


@router.get("/models/install/{job_id}")
async def install_model_status(job_id: str):
    try:
        status = get_model_install_status(job_id)
    except ProviderConfigurationError as exc:
        logger.warning("Unknown install job requested: %s", job_id)
        return JSONResponse(
            status_code=404,
            content={"error": "install_not_found", "details": str(exc)},
        )
    except Exception as exc:  # pragma: no cover - defensive wrapper
        logger.error("Install status lookup failed for %s: %s", job_id, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "install_status_failed", "details": str(exc)},
        )

    return ModelInstallStatusResponse(**status.model_dump())
