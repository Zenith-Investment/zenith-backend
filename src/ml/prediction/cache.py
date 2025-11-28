"""
Redis cache for ML predictions.

Implements caching strategy to reduce redundant predictions.
"""

import redis
import json
from typing import Optional, Any, Dict
import logging
from datetime import timedelta
import hashlib

logger = logging.getLogger(__name__)


class PredictionCache:
    """Cache for ML predictions using Redis."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "ml_pred:"
    ):
        """
        Initialize prediction cache.

        Args:
            redis_url: Redis connection URL
            default_ttl: Default time-to-live in seconds (default: 1 hour)
            key_prefix: Prefix for cache keys
        """
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix

        try:
            self.client = redis.from_url(redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except redis.RedisError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def _generate_key(self, ticker: str, model_type: str, **params) -> str:
        """
        Generate cache key from parameters.

        Args:
            ticker: Stock ticker
            model_type: Type of model (lstm, ensemble, etc.)
            **params: Additional parameters to include in key

        Returns:
            Cache key string
        """
        # Create deterministic key from params
        key_parts = [ticker, model_type]

        # Sort params for consistent hashing
        for k, v in sorted(params.items()):
            key_parts.append(f"{k}={v}")

        key_string = ":".join(key_parts)

        # Hash for shorter keys
        key_hash = hashlib.md5(key_string.encode()).hexdigest()

        return f"{self.key_prefix}{key_hash}"

    def get(
        self,
        ticker: str,
        model_type: str,
        **params
    ) -> Optional[Dict[str, Any]]:
        """
        Get prediction from cache.

        Args:
            ticker: Stock ticker
            model_type: Type of model
            **params: Additional parameters

        Returns:
            Cached prediction or None if not found
        """
        if self.client is None:
            return None

        try:
            key = self._generate_key(ticker, model_type, **params)
            cached_value = self.client.get(key)

            if cached_value:
                logger.info(f"Cache hit for {ticker} ({model_type})")
                return json.loads(cached_value)

            logger.debug(f"Cache miss for {ticker} ({model_type})")
            return None

        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(
        self,
        ticker: str,
        model_type: str,
        prediction: Dict[str, Any],
        ttl: Optional[int] = None,
        **params
    ) -> bool:
        """
        Store prediction in cache.

        Args:
            ticker: Stock ticker
            model_type: Type of model
            prediction: Prediction data to cache
            ttl: Time-to-live in seconds (uses default if None)
            **params: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            return False

        try:
            key = self._generate_key(ticker, model_type, **params)
            value = json.dumps(prediction)
            ttl = ttl or self.default_ttl

            self.client.setex(key, ttl, value)
            logger.info(f"Cached prediction for {ticker} ({model_type}) with TTL={ttl}s")

            return True

        except (redis.RedisError, TypeError) as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def delete(self, ticker: str, model_type: str, **params) -> bool:
        """
        Delete prediction from cache.

        Args:
            ticker: Stock ticker
            model_type: Type of model
            **params: Additional parameters

        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            return False

        try:
            key = self._generate_key(ticker, model_type, **params)
            self.client.delete(key)
            logger.info(f"Deleted cache for {ticker} ({model_type})")

            return True

        except redis.RedisError as e:
            logger.error(f"Error deleting from cache: {e}")
            return False

    def invalidate_ticker(self, ticker: str) -> int:
        """
        Invalidate all cache entries for a ticker.

        Args:
            ticker: Stock ticker

        Returns:
            Number of keys deleted
        """
        if self.client is None:
            return 0

        try:
            # Find all keys matching the ticker
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)

            deleted = 0
            for key in keys:
                # Check if key contains ticker
                # This is a simple approach; for production, you might want
                # to store a ticker index
                try:
                    value = self.client.get(key)
                    if value and ticker in value:
                        self.client.delete(key)
                        deleted += 1
                except Exception:
                    continue

            logger.info(f"Invalidated {deleted} cache entries for {ticker}")
            return deleted

        except redis.RedisError as e:
            logger.error(f"Error invalidating cache: {e}")
            return 0

    def clear_all(self) -> bool:
        """
        Clear all prediction cache entries.

        Returns:
            True if successful, False otherwise
        """
        if self.client is None:
            return False

        try:
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)

            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache entries")

            return True

        except redis.RedisError as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if self.client is None:
            return {"status": "disconnected"}

        try:
            info = self.client.info()
            pattern = f"{self.key_prefix}*"
            keys = self.client.keys(pattern)

            return {
                "status": "connected",
                "total_keys": len(keys),
                "used_memory": info.get("used_memory_human", "N/A"),
                "connected_clients": info.get("connected_clients", 0),
                "uptime_days": info.get("uptime_in_days", 0)
            }

        except redis.RedisError as e:
            logger.error(f"Error getting stats: {e}")
            return {"status": "error", "error": str(e)}

    def health_check(self) -> bool:
        """
        Check if Redis is healthy.

        Returns:
            True if healthy, False otherwise
        """
        if self.client is None:
            return False

        try:
            return self.client.ping()
        except redis.RedisError:
            return False
