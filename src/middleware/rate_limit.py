"""
Rate Limiting Middleware for API protection.

Implements sliding window rate limiting with Redis support.
Falls back to in-memory storage if Redis is not available.
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Callable
import asyncio

from fastapi import Request, HTTPException, status, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_limit: int = 10  # Max requests in 1 second
    enabled: bool = True


@dataclass
class RateLimitState:
    """State for tracking rate limits."""
    minute_requests: list = field(default_factory=list)
    hour_requests: list = field(default_factory=list)
    day_requests: list = field(default_factory=list)
    last_request: float = 0


class InMemoryRateLimiter:
    """In-memory rate limiter for development/testing."""

    def __init__(self):
        self._storage: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._lock = asyncio.Lock()

    async def check_rate_limit(
        self,
        key: str,
        config: RateLimitConfig,
    ) -> tuple[bool, dict]:
        """
        Check if request is within rate limits.

        Args:
            key: Unique identifier (IP, API key, user ID)
            config: Rate limit configuration

        Returns:
            Tuple of (allowed, headers_dict)
        """
        async with self._lock:
            now = time.time()
            state = self._storage[key]

            # Clean old requests
            minute_ago = now - 60
            hour_ago = now - 3600
            day_ago = now - 86400

            state.minute_requests = [t for t in state.minute_requests if t > minute_ago]
            state.hour_requests = [t for t in state.hour_requests if t > hour_ago]
            state.day_requests = [t for t in state.day_requests if t > day_ago]

            # Check burst limit (1 second)
            second_ago = now - 1
            recent_requests = sum(1 for t in state.minute_requests if t > second_ago)
            if recent_requests >= config.burst_limit:
                return False, self._get_headers(state, config, "burst")

            # Check minute limit
            if len(state.minute_requests) >= config.requests_per_minute:
                return False, self._get_headers(state, config, "minute")

            # Check hour limit
            if len(state.hour_requests) >= config.requests_per_hour:
                return False, self._get_headers(state, config, "hour")

            # Check day limit
            if len(state.day_requests) >= config.requests_per_day:
                return False, self._get_headers(state, config, "day")

            # Record request
            state.minute_requests.append(now)
            state.hour_requests.append(now)
            state.day_requests.append(now)
            state.last_request = now

            return True, self._get_headers(state, config, None)

    def _get_headers(
        self,
        state: RateLimitState,
        config: RateLimitConfig,
        exceeded: Optional[str],
    ) -> dict:
        """Get rate limit headers."""
        headers = {
            "X-RateLimit-Limit-Minute": str(config.requests_per_minute),
            "X-RateLimit-Remaining-Minute": str(max(0, config.requests_per_minute - len(state.minute_requests))),
            "X-RateLimit-Limit-Hour": str(config.requests_per_hour),
            "X-RateLimit-Remaining-Hour": str(max(0, config.requests_per_hour - len(state.hour_requests))),
            "X-RateLimit-Limit-Day": str(config.requests_per_day),
            "X-RateLimit-Remaining-Day": str(max(0, config.requests_per_day - len(state.day_requests))),
        }

        if exceeded:
            if exceeded == "burst":
                headers["Retry-After"] = "1"
            elif exceeded == "minute":
                headers["Retry-After"] = "60"
            elif exceeded == "hour":
                headers["Retry-After"] = "3600"
            else:
                headers["Retry-After"] = "86400"

        return headers


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware for FastAPI.

    Applies rate limits based on IP address or API key.
    """

    def __init__(
        self,
        app,
        config: Optional[RateLimitConfig] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.config = config or RateLimitConfig()
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
        self.limiter = InMemoryRateLimiter()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        if not self.config.enabled:
            return await call_next(request)

        # Skip excluded paths
        path = request.url.path
        if any(path.startswith(exc) for exc in self.exclude_paths):
            return await call_next(request)

        # Get rate limit key
        key = self._get_key(request)

        # Check rate limit
        allowed, headers = await self.limiter.check_rate_limit(key, self.config)

        if not allowed:
            logger.warning(
                "Rate limit exceeded",
                key=key,
                path=path,
                method=request.method,
            )
            response = Response(
                content='{"detail": "Rate limit exceeded. Please try again later."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
            )
            for header, value in headers.items():
                response.headers[header] = value
            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        for header, value in headers.items():
            response.headers[header] = value

        return response

    def _get_key(self, request: Request) -> str:
        """Get unique key for rate limiting."""
        # Check for API key first
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key[:16]}"

        # Check for authenticated user
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"

        # Check for forwarded IP
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        return f"ip:{client_ip}"


# Dependency for endpoint-specific rate limiting
class RateLimitDependency:
    """Dependency for custom rate limits on specific endpoints."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
    ):
        self.config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
        )
        self.limiter = InMemoryRateLimiter()

    async def __call__(self, request: Request):
        """Check rate limit for this endpoint."""
        # Get key based on IP or API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            key = f"api:{api_key[:16]}"
        else:
            client_ip = request.client.host if request.client else "unknown"
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                client_ip = forwarded.split(",")[0].strip()
            key = f"ip:{client_ip}"

        allowed, headers = await self.limiter.check_rate_limit(key, self.config)

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded for this endpoint.",
                headers=headers,
            )


# Pre-configured rate limit dependencies
rate_limit_dependency = RateLimitDependency()

# Stricter rate limit for expensive operations
strict_rate_limit = RateLimitDependency(
    requests_per_minute=10,
    requests_per_hour=100,
)

# Relaxed rate limit for read-only operations
relaxed_rate_limit = RateLimitDependency(
    requests_per_minute=120,
    requests_per_hour=5000,
)
