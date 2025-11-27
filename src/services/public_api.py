"""
Public API Service for third-party integrations.

Manages API keys and provides secure access to platform data.
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional
import structlog
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.api_key import APIKey, APIKeyPermission, APIRequestLog
from src.models.user import User

logger = structlog.get_logger()


class PublicAPIService:
    """Service for public API management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_api_key(
        self,
        user: User,
        name: str,
        description: Optional[str] = None,
        permission: APIKeyPermission = APIKeyPermission.READ_ONLY,
        rate_limit_per_minute: int = 60,
        rate_limit_per_day: int = 10000,
        allowed_ips: Optional[list[str]] = None,
        expires_in_days: Optional[int] = None,
        is_test_mode: bool = False,
    ) -> tuple[APIKey, str]:
        """
        Create a new API key for a user.

        Args:
            user: User creating the key
            name: Name for the API key
            description: Optional description
            permission: Permission level
            rate_limit_per_minute: Rate limit per minute
            rate_limit_per_day: Rate limit per day
            allowed_ips: List of allowed IP addresses
            expires_in_days: Days until expiration
            is_test_mode: Whether this is a test key

        Returns:
            Tuple of (APIKey, raw_key) - raw_key is only shown once!
        """
        # Check user's key limit based on subscription
        existing_keys = await self.db.execute(
            select(APIKey).where(
                and_(
                    APIKey.user_id == user.id,
                    APIKey.is_active == True,
                )
            )
        )
        keys = existing_keys.scalars().all()

        # Subscription limits
        from src.schemas.user import SubscriptionPlan
        limits = {
            SubscriptionPlan.STARTER: 1,
            SubscriptionPlan.INVESTOR: 5,
            SubscriptionPlan.PREMIUM: 20,
        }
        limit = limits.get(user.subscription_plan, 1)

        if len(keys) >= limit:
            raise ValueError(
                f"Limite de {limit} chaves de API atingido para seu plano."
            )

        # Generate key
        raw_key = self._generate_api_key()
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:8]

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create API key
        api_key = APIKey(
            user_id=user.id,
            name=name,
            description=description,
            key_hash=key_hash,
            key_prefix=key_prefix,
            permission=permission.value,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_day=rate_limit_per_day,
            allowed_ips=allowed_ips,
            expires_at=expires_at,
            is_test_mode=is_test_mode,
        )

        self.db.add(api_key)
        await self.db.commit()
        await self.db.refresh(api_key)

        logger.info(
            "API key created",
            user_id=user.id,
            key_id=api_key.id,
            key_prefix=key_prefix,
        )

        return api_key, raw_key

    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        # Format: inv_live_XXXXX or inv_test_XXXXX
        return f"inv_{'test' if False else 'live'}_{secrets.token_urlsafe(32)}"

    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    async def validate_api_key(
        self,
        raw_key: str,
        required_permission: Optional[APIKeyPermission] = None,
        endpoint: Optional[str] = None,
        client_ip: Optional[str] = None,
    ) -> Optional[APIKey]:
        """
        Validate an API key and return the key object if valid.

        Args:
            raw_key: The raw API key to validate
            required_permission: Required permission level
            endpoint: Endpoint being accessed
            client_ip: Client IP address

        Returns:
            APIKey if valid, None otherwise
        """
        key_hash = self._hash_key(raw_key)

        query = select(APIKey).where(APIKey.key_hash == key_hash)
        result = await self.db.execute(query)
        api_key = result.scalar_one_or_none()

        if not api_key:
            return None

        # Check if active
        if not api_key.is_active:
            logger.warning("Inactive API key used", key_id=api_key.id)
            return None

        # Check expiration
        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            logger.warning("Expired API key used", key_id=api_key.id)
            return None

        # Check IP restrictions
        if api_key.allowed_ips and client_ip:
            if client_ip not in api_key.allowed_ips:
                logger.warning(
                    "API key used from unauthorized IP",
                    key_id=api_key.id,
                    ip=client_ip,
                )
                return None

        # Check permission level
        if required_permission:
            permission_levels = {
                APIKeyPermission.READ_ONLY.value: 1,
                APIKeyPermission.READ_WRITE.value: 2,
                APIKeyPermission.FULL_ACCESS.value: 3,
            }
            if permission_levels.get(api_key.permission, 0) < permission_levels.get(required_permission.value, 0):
                logger.warning(
                    "Insufficient API key permissions",
                    key_id=api_key.id,
                    required=required_permission.value,
                    actual=api_key.permission,
                )
                return None

        # Check endpoint restrictions
        if api_key.allowed_endpoints and endpoint:
            if endpoint not in api_key.allowed_endpoints:
                logger.warning(
                    "API key endpoint not allowed",
                    key_id=api_key.id,
                    endpoint=endpoint,
                )
                return None

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        api_key.total_requests += 1
        await self.db.commit()

        return api_key

    async def revoke_api_key(self, key_id: int, user: User) -> bool:
        """Revoke an API key."""
        query = select(APIKey).where(
            and_(
                APIKey.id == key_id,
                APIKey.user_id == user.id,
            )
        )
        result = await self.db.execute(query)
        api_key = result.scalar_one_or_none()

        if not api_key:
            return False

        api_key.is_active = False
        api_key.revoked_at = datetime.utcnow()
        await self.db.commit()

        logger.info("API key revoked", key_id=key_id, user_id=user.id)
        return True

    async def list_user_keys(self, user: User) -> list[APIKey]:
        """List all API keys for a user."""
        query = select(APIKey).where(
            APIKey.user_id == user.id
        ).order_by(APIKey.created_at.desc())

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def log_request(
        self,
        api_key: APIKey,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: int,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> APIRequestLog:
        """Log an API request."""
        log = APIRequestLog(
            api_key_id=api_key.id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time_ms,
            ip_address=ip_address,
            user_agent=user_agent,
            error_message=error_message,
        )

        self.db.add(log)
        await self.db.commit()

        return log

    async def get_key_stats(self, key_id: int, user: User) -> dict:
        """Get usage statistics for an API key."""
        # Verify ownership
        query = select(APIKey).where(
            and_(
                APIKey.id == key_id,
                APIKey.user_id == user.id,
            )
        )
        result = await self.db.execute(query)
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise ValueError("API key not found")

        # Get request logs for last 30 days
        from sqlalchemy import func
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)

        # Total requests
        total_query = select(func.count(APIRequestLog.id)).where(
            and_(
                APIRequestLog.api_key_id == key_id,
                APIRequestLog.created_at >= thirty_days_ago,
            )
        )
        total_result = await self.db.execute(total_query)
        total_requests = total_result.scalar() or 0

        # Successful requests
        success_query = select(func.count(APIRequestLog.id)).where(
            and_(
                APIRequestLog.api_key_id == key_id,
                APIRequestLog.created_at >= thirty_days_ago,
                APIRequestLog.status_code < 400,
            )
        )
        success_result = await self.db.execute(success_query)
        successful_requests = success_result.scalar() or 0

        # Average response time
        avg_time_query = select(func.avg(APIRequestLog.response_time_ms)).where(
            and_(
                APIRequestLog.api_key_id == key_id,
                APIRequestLog.created_at >= thirty_days_ago,
            )
        )
        avg_time_result = await self.db.execute(avg_time_query)
        avg_response_time = avg_time_result.scalar()

        return {
            "key_id": key_id,
            "key_name": api_key.name,
            "total_requests_30d": total_requests,
            "successful_requests_30d": successful_requests,
            "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 100,
            "avg_response_time_ms": float(avg_response_time) if avg_response_time else None,
            "total_requests_all_time": api_key.total_requests,
            "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
        }


def get_public_api_service(db: AsyncSession) -> PublicAPIService:
    return PublicAPIService(db)
