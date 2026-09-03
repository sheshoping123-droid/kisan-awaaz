from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.core.errors import (
    KisanAwaazError,
    kisanaawaaz_error_handler,
    unhandled_error_handler,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Kisan Awaaz starting up | env=%s | log_level=%s", settings.app_env, settings.log_level)
    yield
    logger.info("Kisan Awaaz shutting down")


app = FastAPI(
    title="Kisan Awaaz",
    description="WhatsApp-based AI agricultural advisory for Pakistani farmers",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_exception_handler(KisanAwaazError, kisanaawaaz_error_handler)  # type: ignore[arg-type]
app.add_exception_handler(Exception, unhandled_error_handler)  # type: ignore[arg-type]


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "kisan-awaaz",
        "version": app.version,
        "env": settings.app_env,
    }
