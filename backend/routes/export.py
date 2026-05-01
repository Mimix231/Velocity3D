"""
POST /export — re-exports a GLB to obj/fbx/glb using bpy.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models import ExportRequest, ExportResponse
from backend.pipelines.bpy_post import BpyPostProcessor, BpyPostProcessorError

logger = logging.getLogger("velocity3d.routes.export")

router = APIRouter()


@router.post("/export")
async def export(req: ExportRequest):
    logger.info("POST /export format=%s -> %s", req.format, req.output_path)
    processor = BpyPostProcessor()
    try:
        output_path = processor.export(req.model_path, req.output_path, req.format)
    except BpyPostProcessorError as exc:
        logger.error("Export failed: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "export_error", "details": str(exc)},
        )
    except OSError as exc:
        logger.error("Export OS error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "export_error", "details": str(exc)},
        )
    return ExportResponse(output_path=output_path, format=req.format)
