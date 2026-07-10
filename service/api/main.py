"""FastAPI app factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from service.core.auth import AuthError
from service.core.config import get_settings
from service.core.errors import INTERNAL
from service.core.ids import new_request_id
from service.core.logging_config import setup_logging
from service.api.routes.recognize import router as recognize_router
from service.api.routes.system import health_router, tasks_router
from service.api.routes.dashboard import router as dashboard_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from service.core.metrics import QUEUE_DEPTH
    from service.core.queue import TaskQueue

    q = TaskQueue()
    if q.ping():
        QUEUE_DEPTH.set(q.queue_depth())
    logger.info("API started")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level, settings.json_logs)

    app = FastAPI(
        title="Passport Recognition Service",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.exception_handler(AuthError)
    async def auth_error_handler(_request: Request, exc: AuthError):
        rid = new_request_id()
        return JSONResponse(
            status_code=exc.error.http_status,
            content=exc.error.to_body(rid),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(_request: Request, exc: Exception):
        rid = new_request_id()
        logger.exception("Unhandled error request_id=%s", rid)
        return JSONResponse(
            status_code=INTERNAL.http_status,
            content=INTERNAL.to_body(rid),
        )

    app.include_router(recognize_router)
    app.include_router(health_router)
    app.include_router(tasks_router)
    app.include_router(dashboard_router)

    return app


app = create_app()
