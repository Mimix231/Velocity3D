"""
POST /cancel — signals an active generation to stop.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models import CancelRequest
from backend.routes.generate import get_cancellation_event

logger = logging.getLogger("velocity3d.routes.cancel")

router = APIRouter()


@router.post("/cancel")
async def cancel(req: CancelRequest):
    event = get_cancellation_event(req.request_id)
    if event is None:
        logger.warning("Cancel: no active generation for request_id=%s", req.request_id)
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "details": f"No active generation for {req.request_id}"},
        )

    event.set()
    logger.info("Cancel: signalled request_id=%s", req.request_id)
    return {"status": "cancelled", "request_id": req.request_id}
