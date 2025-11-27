"""Health check endpoints for monitoring and deployment."""
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.core.deps import get_db
from src.core.config import settings

router = APIRouter()
logger = structlog.get_logger()


@router.get("/")
async def health_check() -> dict:
    """
    Basic health check endpoint.

    Returns basic application status. Use this for load balancer health checks.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_db)) -> dict:
    """
    Readiness check endpoint.

    Checks if the application is ready to receive traffic by verifying
    database connectivity.
    """
    checks = {
        "database": False,
        "redis": False,
    }

    # Check database
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as e:
        logger.error("Database health check failed", error=str(e))

    # Check Redis
    try:
        from src.core.security import get_redis
        redis = await get_redis()
        await redis.ping()
        checks["redis"] = True
    except Exception as e:
        logger.error("Redis health check failed", error=str(e))

    all_healthy = all(checks.values())

    return {
        "status": "ready" if all_healthy else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness check endpoint.

    Simple check to verify the application is running.
    Use this for Kubernetes liveness probes.
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/info")
async def app_info() -> dict:
    """
    Application information endpoint.

    Returns basic application metadata.
    """
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "api_version": "v1",
        "documentation": "/api/v1/docs",
    }
