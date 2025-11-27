from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.v1.router import api_router
from src.core.config import settings
from src.core.database import init_db
from src.middleware.rate_limit import RateLimitMiddleware, RateLimitConfig
from src.core.exceptions import register_exception_handlers

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan events."""
    # Startup
    logger.info("Starting InvestAI Platform API", environment=settings.ENVIRONMENT)

    if settings.ENVIRONMENT == "development":
        await init_db()
        logger.info("Database tables initialized")

    yield

    # Shutdown
    logger.info("Shutting down InvestAI Platform API")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-powered investment analysis platform",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Rate Limiting Middleware
    rate_limit_config = RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        requests_per_day=10000,
        burst_limit=10,
        enabled=settings.ENVIRONMENT != "development",  # Disabled in dev for easier testing
    )
    app.add_middleware(
        RateLimitMiddleware,
        config=rate_limit_config,
        exclude_paths=[
            "/health",
            f"{settings.API_V1_PREFIX}/docs",
            f"{settings.API_V1_PREFIX}/openapi.json",
            f"{settings.API_V1_PREFIX}/redoc",
            f"{settings.API_V1_PREFIX}/health",
        ],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Register exception handlers
    register_exception_handlers(app)

    return app


app = create_application()


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }
