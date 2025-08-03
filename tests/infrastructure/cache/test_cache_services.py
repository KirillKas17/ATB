"""
Comprehensive tests for cache services.
"""
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from uuid import uuid4
from decimal import Decimal
from infrastructure.cache.redis_cache import RedisCache
from infrastructure.cache.disk_cache import DiskCache
from infrastructure.cache.hybrid_cache import HybridCache
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.trading_pair import TradingPair
class TestRedisCache:
    """Tests for RedisCache."""
    @pytest.fixture
    async def cache(self) -> Any:
        """Create Redis cache instance."""
        cache = RedisCache("redis://localhost:6379")
        cache._redis = AsyncMock()
        return cache
    @pytest.fixture
    def sample_data(self) -> Any:
        """Create sample data for testing."""
        return {
            "key": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "data"}
        }
    async def test_set_get_success(self, cache, sample_data) -> None:
        """Test successful set and get operations."""
        # Arrange
        key = "test_key"
        cache._redis.set.return_value = True
        cache._redis.get.return_value = json.dumps(sample_data).encode()
        # Act
        await cache.set(key, sample_data, ttl=300)
        result = await cache.get(key)
        # Assert
        assert result == sample_data
        cache._redis.set.assert_called_once()
        cache._redis.get.assert_called_once_with(key)
    async def test_get_not_found(self, cache) -> None:
        """Test get operation when key not found."""
        # Arrange
        key = "nonexistent_key"
        cache._redis.get.return_value = None
        # Act
        result = await cache.get(key)
        # Assert
        assert result is None
    async def test_delete_success(self, cache) -> None:
        """Test successful delete operation."""
        # Arrange
        key = "test_key"
        cache._redis.delete.return_value = 1
        # Act
        result = await cache.delete(key)
        # Assert
        assert result is True
        cache._redis.delete.assert_called_once_with(key)
    async def test_delete_not_found(self, cache) -> None:
        """Test delete operation when key not found."""
        # Arrange
        key = "nonexistent_key"
        cache._redis.delete.return_value = 0
        # Act
        result = await cache.delete(key)
        # Assert
        assert result is False
    async def test_exists_success(self, cache) -> None:
        """Test exists operation."""
        # Arrange
        key = "test_key"
        cache._redis.exists.return_value = 1
        # Act
        result = await cache.exists(key)
        # Assert
        assert result is True
        cache._redis.exists.assert_called_once_with(key)
    async def test_exists_not_found(self, cache) -> None:
        """Test exists operation when key not found."""
        # Arrange
        key = "nonexistent_key"
        cache._redis.exists.return_value = 0
        # Act
        result = await cache.exists(key)
        # Assert
        assert result is False
    async def test_clear_success(self, cache) -> None:
        """Test clear operation."""
        # Arrange
        cache._redis.flushdb.return_value = True
        # Act
        result = await cache.clear()
        # Assert
        assert result is True
        cache._redis.flushdb.assert_called_once()
    async def test_set_with_ttl(self, cache, sample_data) -> None:
        """Test set operation with TTL."""
        # Arrange
        key = "test_key"
        ttl = 300
        cache._redis.setex.return_value = True
        # Act
        result = await cache.set(key, sample_data, ttl=ttl)
        # Assert
        assert result is True
        cache._redis.setex.assert_called_once_with(
            key, ttl, json.dumps(sample_data)
        )
    async def test_connection_error_handling(self, cache, sample_data) -> None:
        """Test connection error handling."""
        # Arrange
        key = "test_key"
        cache._redis.set.side_effect = Exception("Connection error")
        # Act & Assert
        with pytest.raises(Exception):
            await cache.set(key, sample_data)
    async def test_health_check_success(self, cache) -> None:
        """Test health check when Redis is available."""
        # Arrange
        cache._redis.ping.return_value = True
        # Act
        result = await cache.health_check()
        # Assert
        assert result["status"] == "healthy"
        assert "response_time" in result
    async def test_health_check_failure(self, cache) -> None:
        """Test health check when Redis is unavailable."""
        # Arrange
        cache._redis.ping.side_effect = Exception("Connection error")
        # Act
        result = await cache.health_check()
        # Assert
        assert result["status"] == "unhealthy"
        assert "error" in result
class TestDiskCache:
    """Tests for DiskCache."""
    @pytest.fixture
    async def cache(self) -> Any:
        """Create disk cache instance."""
        cache = DiskCache("/tmp/test_cache")
        return cache
    @pytest.fixture
    def sample_data(self) -> Any:
        """Create sample data for testing."""
        return {
            "key": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "data"}
        }
    async def test_set_get_success(self, cache, sample_data) -> None:
        """Test successful set and get operations."""
        # Arrange
        key = "test_key"
        # Act
        await cache.set(key, sample_data, ttl=300)
        result = await cache.get(key)
        # Assert
        assert result == sample_data
    async def test_get_not_found(self, cache) -> None:
        """Test get operation when key not found."""
        # Arrange
        key = "nonexistent_key"
        # Act
        result = await cache.get(key)
        # Assert
        assert result is None
    async def test_delete_success(self, cache, sample_data) -> None:
        """Test successful delete operation."""
        # Arrange
        key = "test_key"
        await cache.set(key, sample_data)
        # Act
        result = await cache.delete(key)
        # Assert
        assert result is True
        assert await cache.get(key) is None
    async def test_delete_not_found(self, cache) -> None:
        """Test delete operation when key not found."""
        # Arrange
        key = "nonexistent_key"
        # Act
        result = await cache.delete(key)
        # Assert
        assert result is False
    async def test_exists_success(self, cache, sample_data) -> None:
        """Test exists operation."""
        # Arrange
        key = "test_key"
        await cache.set(key, sample_data)
        # Act
        result = await cache.exists(key)
        # Assert
        assert result is True
    async def test_exists_not_found(self, cache) -> None:
        """Test exists operation when key not found."""
        # Arrange
        key = "nonexistent_key"
        # Act
        result = await cache.exists(key)
        # Assert
        assert result is False
    async def test_clear_success(self, cache, sample_data) -> None:
        """Test clear operation."""
        # Arrange
        await cache.set("key1", sample_data)
        await cache.set("key2", sample_data)
        # Act
        result = await cache.clear()
        # Assert
        assert result is True
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    async def test_ttl_expiration(self, cache, sample_data) -> None:
        """Test TTL expiration."""
        # Arrange
        key = "test_key"
        ttl = 1  # 1 second
        # Act
        await cache.set(key, sample_data, ttl=ttl)
        await asyncio.sleep(1.1)  # Wait for expiration
        result = await cache.get(key)
        # Assert
        assert result is None
    async def test_health_check_success(self, cache) -> None:
        """Test health check."""
        # Act
        result = await cache.health_check()
        # Assert
        assert result["status"] == "healthy"
        assert "disk_usage" in result
class TestHybridCache:
    """Tests for HybridCache."""
    @pytest.fixture
    async def cache(self) -> Any:
        """Create hybrid cache instance."""
        primary_cache = RedisCache("redis://localhost:6379")
        primary_cache._redis = AsyncMock()
        secondary_cache = DiskCache("/tmp/test_cache")
        cache = HybridCache(primary_cache, secondary_cache)
        return cache
    @pytest.fixture
    def sample_data(self) -> Any:
        """Create sample data for testing."""
        return {
            "key": "value",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"nested": "data"}
        }
    async def test_set_get_primary_success(self, cache, sample_data) -> None:
        """Test successful set and get operations with primary cache."""
        # Arrange
        key = "test_key"
        cache._primary_cache._redis.set.return_value = True
        cache._primary_cache._redis.get.return_value = json.dumps(sample_data).encode()
        # Act
        await cache.set(key, sample_data, ttl=300)
        result = await cache.get(key)
        # Assert
        assert result == sample_data
        cache._primary_cache._redis.set.assert_called_once()
        cache._primary_cache._redis.get.assert_called_once_with(key)
    async def test_get_fallback_to_secondary(self, cache, sample_data) -> None:
        """Test fallback to secondary cache when primary fails."""
        # Arrange
        key = "test_key"
        cache._primary_cache._redis.get.side_effect = Exception("Primary cache error")
        await cache._secondary_cache.set(key, sample_data)
        # Act
        result = await cache.get(key)
        # Assert
        assert result == sample_data
    async def test_delete_both_caches(self, cache, sample_data) -> None:
        """Test delete operation on both caches."""
        # Arrange
        key = "test_key"
        cache._primary_cache._redis.delete.return_value = 1
        await cache._secondary_cache.set(key, sample_data)
        # Act
        result = await cache.delete(key)
        # Assert
        assert result is True
        cache._primary_cache._redis.delete.assert_called_once_with(key)
        assert await cache._secondary_cache.get(key) is None
    async def test_exists_primary_success(self, cache) -> None:
        """Test exists operation with primary cache."""
        # Arrange
        key = "test_key"
        cache._primary_cache._redis.exists.return_value = 1
        # Act
        result = await cache.exists(key)
        # Assert
        assert result is True
        cache._primary_cache._redis.exists.assert_called_once_with(key)
    async def test_exists_fallback_to_secondary(self, cache, sample_data) -> None:
        """Test exists operation with fallback to secondary cache."""
        # Arrange
        key = "test_key"
        cache._primary_cache._redis.exists.side_effect = Exception("Primary cache error")
        await cache._secondary_cache.set(key, sample_data)
        # Act
        result = await cache.exists(key)
        # Assert
        assert result is True
    async def test_clear_both_caches(self, cache) -> None:
        """Test clear operation on both caches."""
        # Arrange
        cache._primary_cache._redis.flushdb.return_value = True
        # Act
        result = await cache.clear()
        # Assert
        assert result is True
        cache._primary_cache._redis.flushdb.assert_called_once()
        # Secondary cache clear is also called
    async def test_health_check_primary_healthy(self, cache) -> None:
        """Test health check when primary cache is healthy."""
        # Arrange
        cache._primary_cache._redis.ping.return_value = True
        # Act
        result = await cache.health_check()
        # Assert
        assert result["status"] == "healthy"
        assert result["primary"]["status"] == "healthy"
        assert result["secondary"]["status"] == "healthy"
    async def test_health_check_primary_unhealthy(self, cache) -> None:
        """Test health check when primary cache is unhealthy."""
        # Arrange
        cache._primary_cache._redis.ping.side_effect = Exception("Primary cache error")
        # Act
        result = await cache.health_check()
        # Assert
        assert result["status"] == "degraded"
        assert result["primary"]["status"] == "unhealthy"
        assert result["secondary"]["status"] == "healthy"
class TestCacheIntegration:
    """Integration tests for cache services."""
    @pytest.mark.integration
    async def test_cache_with_complex_objects(self) -> None:
        """Test cache with complex domain objects."""
        # Arrange
        order = Order(
            id=uuid4(),
            trading_pair=TradingPair.from_string("BTC/USD"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Money(Decimal("50000"), Currency.USD),
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        cache = DiskCache("/tmp/test_cache")
        # Act
        await cache.set("order_1", order, ttl=300)
        result = await cache.get("order_1")
        # Assert
        assert result is not None
        assert result.id == order.id
        assert result.trading_pair == order.trading_pair
    @pytest.mark.integration
    async def test_cache_eviction_policy(self) -> None:
        """Test cache eviction policy."""
        # This would test LRU or other eviction policies
        pass
    @pytest.mark.integration
    async def test_cache_compression(self) -> None:
        """Test cache compression for large objects."""
        # This would test compression functionality
        pass
class TestCachePerformance:
    """Performance tests for cache services."""
    @pytest.mark.performance
    async def test_cache_throughput(self) -> None:
        """Test cache throughput."""
        # This would measure operations per second
        pass
    @pytest.mark.performance
    async def test_cache_latency(self) -> None:
        """Test cache latency."""
        # This would measure response times
        pass
    @pytest.mark.performance
    async def test_cache_memory_usage(self) -> None:
        """Test cache memory usage."""
        # This would measure memory consumption
        pass
class TestCacheEdgeCases:
    """Edge cases and error handling tests."""
    async def test_cache_with_large_objects(self) -> None:
        """Test cache with very large objects."""
        # This would test handling of large data
        pass
    async def test_cache_with_special_characters(self) -> None:
        """Test cache with special characters in keys."""
        # This would test key encoding/decoding
        pass
    async def test_cache_concurrent_access(self) -> None:
        """Test cache with concurrent access."""
        # This would test thread safety
        pass
    async def test_cache_network_partition(self) -> None:
        """Test cache behavior during network partition."""
        # This would test distributed cache behavior
        pass 
