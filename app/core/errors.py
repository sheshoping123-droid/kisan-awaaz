from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.logging import get_logger

logger = get_logger(__name__)


class KisanAwaazError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AdapterError(KisanAwaazError):
    def __init__(self, adapter: str, message: str):
        super().__init__(f"[{adapter}] {message}", status_code=502)
        self.adapter = adapter


class PipelineError(KisanAwaazError):
    def __init__(self, pipeline: str, message: str):
        super().__init__(f"[{pipeline}] {message}", status_code=500)
        self.pipeline = pipeline


class ValidationError(KisanAwaazError):
    def __init__(self, message: str):
        super().__init__(message, status_code=422)


async def kisanaawaaz_error_handler(request: Request, exc: KisanAwaazError) -> JSONResponse:
    logger.error("Application error: %s", exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "type": type(exc).__name__},
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "type": "InternalServerError"},
    )
