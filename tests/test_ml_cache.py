"""
Unit tests for ML Prediction Cache.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from src.ml.prediction.cache import PredictionCache


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = Mock()
    mock.ping.return_value = True
    mock.get.return_value = None
    mock.set.return_value = True
    mock.delete.return_value = 1
    mock.keys.return_value = []
    return mock


@pytest.fixture
def cache(mock_redis):
    """Create cache instance with mocked Redis."""
    with patch('src.ml.prediction.cache.redis.Redis', return_value=mock_redis):
        return PredictionCache(redis_url="redis://localhost:6379/0")


def test_cache_initialization(cache):
    """Test cache initialization."""
    assert cache is not None
    assert cache.default_ttl == 3600
    assert cache.key_prefix == "ml_prediction:"


def test_cache_set(cache, mock_redis):
    """Test setting cache value."""
    data = {"prediction": [1.0, 2.0, 3.0]}

    cache.set("PETR4.SA", "lstm", data, ttl=1800)

    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    assert call_args[1]["ex"] == 1800


def test_cache_get_hit(mock_redis):
    """Test cache hit."""
    cached_data = {"prediction": [1.0, 2.0, 3.0]}
    mock_redis.get.return_value = json.dumps(cached_data)

    with patch('src.ml.prediction.cache.redis.Redis', return_value=mock_redis):
        cache = PredictionCache()
        result = cache.get("PETR4.SA", "lstm")

        assert result == cached_data


def test_cache_get_miss(cache, mock_redis):
    """Test cache miss."""
    mock_redis.get.return_value = None

    result = cache.get("PETR4.SA", "lstm")

    assert result is None


def test_cache_invalidate_ticker(cache, mock_redis):
    """Test invalidating cache for a ticker."""
    mock_redis.keys.return_value = [
        b"ml_prediction:abc123",
        b"ml_prediction:def456"
    ]

    cache.invalidate_ticker("PETR4.SA")

    mock_redis.keys.assert_called_once()
    assert mock_redis.delete.call_count == 2


def test_cache_clear_all(cache, mock_redis):
    """Test clearing all cache."""
    mock_redis.keys.return_value = [
        b"ml_prediction:abc123",
        b"ml_prediction:def456",
        b"ml_prediction:ghi789"
    ]

    cache.clear_all()

    mock_redis.keys.assert_called_once()
    assert mock_redis.delete.call_count == 3


def test_cache_get_stats(cache, mock_redis):
    """Test getting cache statistics."""
    mock_redis.keys.return_value = [
        b"ml_prediction:abc123",
        b"ml_prediction:def456"
    ]

    stats = cache.get_stats()

    assert stats["total_keys"] == 2
    assert "ttl" in stats


def test_cache_key_generation(cache):
    """Test cache key generation."""
    key1 = cache._generate_key("PETR4.SA", "lstm", horizon=5)
    key2 = cache._generate_key("PETR4.SA", "lstm", horizon=10)
    key3 = cache._generate_key("PETR4.SA", "ensemble")

    assert key1 != key2  # Different params
    assert key1 != key3  # Different model type
    assert key1.startswith("ml_prediction:")
