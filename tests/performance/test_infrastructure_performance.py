"""
Performance benchmarks for Infrastructure layer.
"""
import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime, timedelta
from uuid import uuid4
from decimal import Decimal
import psutil
import gc
from infrastructure.repositories.order_repository import PostgresOrderRepository
from infrastructure.cache.redis_cache import RedisCache
from infrastructure.cache.disk_cache import DiskCache
from infrastructure.cache.hybrid_cache import HybridCache
from infrastructure.agents.agent_context_refactored import AgentContext
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.market import MarketData
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.trading_pair import TradingPair
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.type_definitions import OrderId, VolumeValue, create_trading_pair

class PerformanceMetrics:
    """Класс для сбора метрик производительности."""
    def __init__(self) -> None:
        self.operation_times: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        self.start_time = time.time()
    def record_operation(self, duration: float) -> None:
        """Записывает время операции."""
        self.operation_times.append(duration)
    def record_memory_usage(self) -> None:
        """Записывает использование памяти."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
    def record_cpu_usage(self) -> None:
        """Записывает использование CPU."""
        process = psutil.Process()
        cpu_percent = process.cpu_percent()
        self.cpu_usage.append(cpu_percent)
    def get_statistics(self) -> Dict[str, Any]:
        """Получает статистику производительности."""
        if not self.operation_times:
            return {}
        return {
            "total_operations": len(self.operation_times),
            "total_time": time.time() - self.start_time,
            "avg_operation_time": statistics.mean(self.operation_times),
            "median_operation_time": statistics.median(self.operation_times),
            "min_operation_time": min(self.operation_times),
            "max_operation_time": max(self.operation_times),
            "std_deviation": statistics.stdev(self.operation_times) if len(self.operation_times) > 1 else 0,
            "operations_per_second": len(self.operation_times) / (time.time() - self.start_time),
            "avg_memory_usage_mb": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            "max_memory_usage_mb": max(self.memory_usage) if self.memory_usage else 0,
            "avg_cpu_usage": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            "max_cpu_usage": max(self.cpu_usage) if self.cpu_usage else 0
        }
class TestRepositoryPerformance:
    """Performance tests for repositories."""
    @pytest.fixture
    async def order_repo(self) -> Any:
        """Create order repository for testing."""
        repo = PostgresOrderRepository("postgresql://test:test@localhost/test")
        repo._pool = None  # Disable actual DB connection for benchmarks
        return repo
    @pytest.fixture
    def sample_orders(self) -> Any:
        """Create sample orders for testing."""
        orders = []
        for i in range(1000):
            order = Order(
                id=OrderId(uuid4()),
                trading_pair="BTC/USD",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=VolumeValue(Decimal("1.0")),
                price=Price(Decimal("50000"), Currency.USD),
                status=OrderStatus.PENDING,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now()
            )
            orders.append(order)
        return orders
    @pytest.mark.performance
    async def test_order_repository_bulk_save_performance(self, order_repo, sample_orders) -> None:
        """Test bulk save performance for orders."""
        metrics = PerformanceMetrics()
        # Warm up
        await order_repo.bulk_save(sample_orders[:10])
        # Benchmark
        for batch_size in [10, 50, 100, 500, 1000]:
            for i in range(0, len(sample_orders), batch_size):
                batch = sample_orders[i:i + batch_size]
                start_time = time.time()
                await order_repo.bulk_save(batch)
                duration = time.time() - start_time
                metrics.record_operation(duration)
                metrics.record_memory_usage()
                metrics.record_cpu_usage()
        stats = metrics.get_statistics()
        print(f"Order Repository Bulk Save Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        print(f"  Max memory usage: {stats['max_memory_usage_mb']:.2f}MB")
        # Assertions for performance requirements
        assert stats['operations_per_second'] > 10  # At least 10 ops/sec
        assert stats['avg_operation_time'] < 1.0    # Less than 1 second per operation
        assert stats['max_memory_usage_mb'] < 1000  # Less than 1GB memory usage
    @pytest.mark.performance
    async def test_order_repository_single_operations_performance(self, order_repo, sample_orders) -> None:
        """Test single operation performance for orders."""
        metrics = PerformanceMetrics()
        # Benchmark single operations
        for order in sample_orders[:100]:  # Test with 100 orders
            start_time = time.time()
            await order_repo.save_order(order)
            duration = time.time() - start_time
            metrics.record_operation(duration)
            metrics.record_memory_usage()
        stats = metrics.get_statistics()
        print(f"Order Repository Single Operations Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 50   # At least 50 ops/sec
        assert stats['avg_operation_time'] < 0.1     # Less than 100ms per operation
class TestCachePerformance:
    """Performance tests for cache services."""
    @pytest.fixture
    async def redis_cache(self) -> Any:
        """Create Redis cache for testing."""
        cache = RedisCache("redis://localhost:6379")
        cache._redis = None  # Disable actual Redis connection for benchmarks
        return cache
    @pytest.fixture
    async def disk_cache(self) -> Any:
        """Create disk cache for testing."""
        cache = DiskCache("/tmp/benchmark_cache")
        return cache
    @pytest.fixture
    async def hybrid_cache(self) -> Any:
        """Create hybrid cache for testing."""
        primary = RedisCache("redis://localhost:6379")
        primary._redis = None
        secondary = DiskCache("/tmp/benchmark_cache")
        cache = HybridCache(primary, secondary)
        return cache
    @pytest.fixture
    def sample_data(self) -> Any:
        """Create sample data for testing."""
        data = {}
        for i in range(1000):
            data[f"key_{i}"] = {
                "id": str(uuid4()),
                "value": f"value_{i}",
                "timestamp": datetime.now().isoformat(),
                "data": [i] * 100  # Some bulk data
            }
        return data
    @pytest.mark.performance
    async def test_redis_cache_performance(self, redis_cache, sample_data) -> None:
        """Test Redis cache performance."""
        metrics = PerformanceMetrics()
        # Benchmark set operations
        for key, value in sample_data.items():
            start_time = time.time()
            await redis_cache.set(key, value)
            duration = time.time() - start_time
            metrics.record_operation(duration)
            metrics.record_memory_usage()
        # Benchmark get operations
        for key in sample_data.keys():
            start_time = time.time()
            await redis_cache.get(key)
            duration = time.time() - start_time
            metrics.record_operation(duration)
        stats = metrics.get_statistics()
        print(f"Redis Cache Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 1000  # At least 1000 ops/sec
        assert stats['avg_operation_time'] < 0.01     # Less than 10ms per operation
    @pytest.mark.performance
    async def test_disk_cache_performance(self, disk_cache, sample_data) -> None:
        """Test disk cache performance."""
        metrics = PerformanceMetrics()
        # Benchmark set operations
        for key, value in sample_data.items():
            start_time = time.time()
            await disk_cache.set(key, value)
            duration = time.time() - start_time
            metrics.record_operation(duration)
            metrics.record_memory_usage()
        # Benchmark get operations
        for key in sample_data.keys():
            start_time = time.time()
            await disk_cache.get(key)
            duration = time.time() - start_time
            metrics.record_operation(duration)
        stats = metrics.get_statistics()
        print(f"Disk Cache Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 100   # At least 100 ops/sec
        assert stats['avg_operation_time'] < 0.1      # Less than 100ms per operation
    @pytest.mark.performance
    async def test_hybrid_cache_performance(self, hybrid_cache, sample_data) -> None:
        """Test hybrid cache performance."""
        metrics = PerformanceMetrics()
        # Benchmark set operations
        for key, value in sample_data.items():
            start_time = time.time()
            await hybrid_cache.set(key, value)
            duration = time.time() - start_time
            metrics.record_operation(duration)
            metrics.record_memory_usage()
        # Benchmark get operations
        for key in sample_data.keys():
            start_time = time.time()
            await hybrid_cache.get(key)
            duration = time.time() - start_time
            metrics.record_operation(duration)
        stats = metrics.get_statistics()
        print(f"Hybrid Cache Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 500   # At least 500 ops/sec
        assert stats['avg_operation_time'] < 0.05     # Less than 50ms per operation
class TestAgentContextPerformance:
    """Performance tests for AgentContext."""
    @pytest.fixture
    async def agent_context(self) -> Any:
        """Create agent context for testing."""
        context = AgentContext(symbol="BTC/USDT")
        return context
    @pytest.fixture
    def sample_market_data(self) -> Any:
        """Create sample market data for testing."""
        data = []
        for i in range(1000):
            data.append({
                "symbol": "BTC/USD",
                "timestamp": datetime.now() + timedelta(seconds=i),
                "open": 50000.0 + i,
                "high": 50100.0 + i,
                "low": 49900.0 + i,
                "close": 50050.0 + i,
                "volume": 1000000.0 + i * 1000
            })
        return data
    @pytest.mark.performance
    async def test_agent_context_data_insertion_performance(self, agent_context, sample_market_data) -> None:
        """Test data insertion performance for AgentContext."""
        metrics = PerformanceMetrics()
        # Benchmark data insertion
        for data in sample_market_data:
            start_time = time.time()
            agent_context.insert_market_data(data)
            duration = time.time() - start_time
            metrics.record_operation(duration)
            metrics.record_memory_usage()
        stats = metrics.get_statistics()
        print(f"Agent Context Data Insertion Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 10000  # At least 10000 ops/sec
        assert stats['avg_operation_time'] < 0.001     # Less than 1ms per operation
    @pytest.mark.performance
    async def test_agent_context_query_performance(self, agent_context, sample_market_data) -> None:
        """Test query performance for AgentContext."""
        # Insert data first
        for data in sample_market_data:
            agent_context.insert_market_data(data)
        metrics = PerformanceMetrics()
        # Benchmark queries
        for _ in range(1000):
            start_time = time.time()
            result = agent_context.get_latest_market_data("BTC/USD")
            duration = time.time() - start_time
            metrics.record_operation(duration)
            metrics.record_memory_usage()
        stats = metrics.get_statistics()
        print(f"Agent Context Query Performance:")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 10000  # At least 10000 ops/sec
        assert stats['avg_operation_time'] < 0.001     # Less than 1ms per operation
class TestConcurrentPerformance:
    """Performance tests for concurrent operations."""
    @pytest.mark.performance
    async def test_concurrent_cache_operations(self) -> None:
        """Test concurrent cache operations."""
        cache = DiskCache("/tmp/concurrent_cache")
        metrics = PerformanceMetrics()
        async def cache_operation(operation_id: int) -> None:
            """Single cache operation."""
            for i in range(100):
                key = f"key_{operation_id}_{i}"
                value = f"value_{operation_id}_{i}"
                start_time = time.time()
                await cache.set(key, value)
                await cache.get(key)
                duration = time.time() - start_time
                metrics.record_operation(duration)
        # Run concurrent operations
        tasks = [cache_operation(i) for i in range(10)]
        await asyncio.gather(*tasks)
        stats = metrics.get_statistics()
        print(f"Concurrent Cache Operations Performance:")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 500   # At least 500 ops/sec
        assert stats['avg_operation_time'] < 0.01     # Less than 10ms per operation
    @pytest.mark.performance
    async def test_concurrent_repository_operations(self) -> None:
        """Test concurrent repository operations."""
        repo = PostgresOrderRepository("postgresql://test:test@localhost/test")
        repo._pool = None  # Disable actual DB connection
        metrics = PerformanceMetrics()
        async def repo_operation(operation_id: int) -> None:
            """Single repository operation."""
            for i in range(10):
                order = Order(
                    id=OrderId(uuid4()),
                    trading_pair=create_trading_pair("BTC/USD"),
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=VolumeValue(Decimal("1.0")),
                    price=Price(Decimal("50000"), Currency.USD),
                    status=OrderStatus.PENDING,
                    created_at=Timestamp.now(),
                    updated_at=Timestamp.now()
                )
                start_time = time.time()
                await repo.save_order(order)
                duration = time.time() - start_time
                metrics.record_operation(duration)
        # Run concurrent operations
        tasks = [repo_operation(i) for i in range(5)]
        await asyncio.gather(*tasks)
        stats = metrics.get_statistics()
        print(f"Concurrent Repository Operations Performance:")
        print(f"  Total operations: {stats['total_operations']}")
        print(f"  Operations per second: {stats['operations_per_second']:.2f}")
        print(f"  Average operation time: {stats['avg_operation_time']*1000:.2f}ms")
        assert stats['operations_per_second'] > 10    # At least 10 ops/sec
        assert stats['avg_operation_time'] < 1.0      # Less than 1 second per operation
class TestMemoryPerformance:
    """Memory performance tests."""
    @pytest.mark.performance
    async def test_memory_usage_under_load(self) -> None:
        """Test memory usage under high load."""
        cache = DiskCache("/tmp/memory_test_cache")
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        # Generate large amount of data
        large_data = {}
        for i in range(10000):
            large_data[f"key_{i}"] = {
                "data": [i] * 1000,  # 1000 integers per entry
                "timestamp": datetime.now().isoformat(),
                "metadata": {"id": i, "type": "test"}
            }
        # Insert data
        for key, value in large_data.items():
            await cache.set(key, value)
        # Force garbage collection
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        print(f"Memory Usage Under Load:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        # Assert reasonable memory usage
        assert memory_increase < 1000  # Less than 1GB memory increase
        assert final_memory < 2000     # Less than 2GB total memory usage
    @pytest.mark.performance
    async def test_memory_leak_detection(self) -> None:
        """Test for memory leaks."""
        cache = DiskCache("/tmp/leak_test_cache")
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        # Perform multiple operations
        for cycle in range(10):
            for i in range(1000):
                key = f"cycle_{cycle}_key_{i}"
                value = {"data": [i] * 100}
                await cache.set(key, value)
                await cache.get(key)
            # Clear cache
            await cache.clear()
            # Force garbage collection
            gc.collect()
            # Check memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            print(f"Cycle {cycle}: Memory usage: {current_memory:.2f}MB")
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        print(f"Memory Leak Test:")
        print(f"  Initial memory: {initial_memory:.2f}MB")
        print(f"  Final memory: {final_memory:.2f}MB")
        print(f"  Memory increase: {memory_increase:.2f}MB")
        # Assert no significant memory leak
        assert memory_increase < 100  # Less than 100MB memory increase
class TestScalabilityPerformance:
    """Scalability performance tests."""
    @pytest.mark.performance
    async def test_cache_scalability(self) -> None:
        """Test cache scalability with increasing data size."""
        cache = DiskCache("/tmp/scalability_cache")
        metrics = PerformanceMetrics()
        data_sizes = [100, 1000, 10000, 100000]
        for size in data_sizes:
            # Generate data
            data = {f"key_{i}": f"value_{i}" for i in range(size)}
            # Benchmark insertion
            start_time = time.time()
            for key, value in data.items():
                await cache.set(key, value)
            insertion_time = time.time() - start_time
            # Benchmark retrieval
            start_time = time.time()
            for key in data.keys():
                await cache.get(key)
            retrieval_time = time.time() - start_time
            print(f"Cache Scalability (Size: {size}):")
            print(f"  Insertion time: {insertion_time:.2f}s")
            print(f"  Retrieval time: {retrieval_time:.2f}s")
            print(f"  Insertion rate: {size/insertion_time:.2f} ops/sec")
            print(f"  Retrieval rate: {size/retrieval_time:.2f} ops/sec")
            # Assert scalability
            assert insertion_time < size * 0.001  # Less than 1ms per item
            assert retrieval_time < size * 0.001  # Less than 1ms per item
    @pytest.mark.performance
    async def test_repository_scalability(self) -> None:
        """Test repository scalability with increasing data size."""
        repo = PostgresOrderRepository("postgresql://test:test@localhost/test")
        repo._pool = None  # Disable actual DB connection
        metrics = PerformanceMetrics()
        batch_sizes = [10, 50, 100, 500, 1000]
        for batch_size in batch_sizes:
            # Generate orders
            orders = []
            for i in range(batch_size):
                order = Order(
                    id=OrderId(uuid4()),
                    trading_pair="BTC/USD",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=VolumeValue(Decimal("1.0")),
                    price=Price(Decimal("50000"), Currency.USD),
                    status=OrderStatus.PENDING,
                    created_at=Timestamp.now(),
                    updated_at=Timestamp.now()
                )
                orders.append(order)
            # Benchmark bulk save
            start_time = time.time()
            await repo.bulk_save(orders)
            duration = time.time() - start_time
            print(f"Repository Scalability (Batch: {batch_size}):")
            print(f"  Bulk save time: {duration:.2f}s")
            print(f"  Rate: {batch_size/duration:.2f} ops/sec")
            # Assert scalability
            assert duration < batch_size * 0.01  # Less than 10ms per item
# Performance test configuration
def pytest_configure(config) -> Any:
    """Configure performance tests."""
    config.addinivalue_line(
        "markers", "performance: mark test as performance benchmark"
    )
def pytest_collection_modifyitems(config, items) -> Any:
    """Modify test collection for performance tests."""
    for item in items:
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.slow) 
