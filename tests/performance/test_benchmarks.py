"""
Тесты производительности (benchmarks) для критических компонентов.
"""
import pytest
import time
import asyncio
from typing import List, Dict, Any
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.trading import Trade
from domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from domain.entities.ml import Model, ModelStatus, ModelType
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.trading_pair import TradingPair as VOTradingPair
from domain.value_objects.currency import Currency
from decimal import Decimal
from infrastructure.repositories.trading_repository import InMemoryTradingRepository
from infrastructure.repositories.strategy_repository import InMemoryStrategyRepository
from infrastructure.repositories.ml_repository import InMemoryMLRepository
from infrastructure.core.optimized_database import OptimizedDatabase
from infrastructure.messaging_copy.optimized_event_bus import OptimizedEventBus
from domain.types import OrderId, TradeId, VolumeValue, TimestampValue, TradingPair
from domain.entities.strategy_parameters import StrategyParameters
from uuid import uuid4
from datetime import datetime
from domain.value_objects.symbol import Symbol
import domain.entities.trading

class TestDatabasePerformance:
    """Тесты производительности базы данных."""
    @pytest.fixture
    def database(self) -> Any:
        """Создание тестовой базы данных."""
        return OptimizedDatabase("sqlite:///test_performance.db")
    @pytest.fixture
    def sample_orders(self) -> List[Order]:
        """Создание тестовых ордеров."""
        orders = []
        for i in range(1000):
            order = Order(
                id=OrderId(uuid4()),
                trading_pair=TradingPair("BTCUSDT"),  # Исправление: используем TradingPair из domain.types
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=VolumeValue(Decimal("0.1")),  # Исправление: используем VolumeValue вместо Volume
                price=Price(Decimal("50000") + Decimal(str(i)), Currency.USDT),
                status=OrderStatus.PENDING,
                created_at=Timestamp.now()
            )
            orders.append(order)
        return orders
    @pytest.mark.asyncio
    async def test_bulk_insert_performance(self, database, sample_orders) -> None:
        """Тест производительности массовой вставки."""
        start_time = time.time()
        # Вставляем 1000 ордеров
        for i, order in enumerate(sample_orders):
            await database.save_trade(Trade(
                id=TradeId(uuid4()),
                symbol=Symbol("BTCUSDT"),
                side="buy" if i % 2 == 0 else "sell",
                price=order.price,
                volume=order.amount,
                executed_at=TimestampValue(Timestamp.now().value)
            ))
        end_time = time.time()
        execution_time = end_time - start_time
        # Проверяем, что вставка 1000 записей занимает менее 5 секунд
        assert execution_time < 5.0, f"Bulk insert took {execution_time:.2f} seconds"
        # Проверяем количество записей
        trades = await database.get_trades("BTCUSDT")
        assert len(trades) == 1000
    @pytest.mark.asyncio
    async def test_query_performance(self, database, sample_orders) -> None:
        """Тест производительности запросов."""
        # Сначала вставляем данные
        for i, order in enumerate(sample_orders):
            await database.save_trade(Trade(
                id=TradeId(uuid4()),
                symbol=Symbol("BTCUSDT"),
                side="buy" if i % 2 == 0 else "sell",
                price=order.price,
                volume=order.amount,
                executed_at=TimestampValue(Timestamp.now().value)
            ))
        # Тестируем запросы
        start_time = time.time()
        # Запрос с фильтром
        trades = await database.get_trades("BTCUSDT")
        end_time = time.time()
        query_time = end_time - start_time
        # Проверяем, что запрос занимает менее 100ms
        assert query_time < 0.1, f"Query took {query_time:.3f} seconds"
        assert len(trades) == 1000
    @pytest.mark.asyncio
    async def test_cache_performance(self, database) -> None:
        """Тест производительности кеширования."""
        # Первый запрос (кеш miss)
        start_time = time.time()
        trades1 = await database.get_trades("BTCUSDT", use_cache=True)
        first_query_time = time.time() - start_time
        # Второй запрос (кеш hit)
        start_time = time.time()
        trades2 = await database.get_trades("BTCUSDT", use_cache=True)
        second_query_time = time.time() - start_time
        # Проверяем, что кешированный запрос быстрее
        assert second_query_time < first_query_time * 0.5, "Cache should be significantly faster"
    def test_connection_pool_performance(self, database) -> None:
        """Тест производительности пула соединений."""
        metrics = database.get_performance_metrics()
        # Проверяем, что пул соединений работает
        assert 'pool_size' in metrics
        assert 'active_connections' in metrics
        assert metrics['pool_size'] > 0
class TestEventBusPerformance:
    """Тесты производительности обработчика событий."""
    @pytest.fixture
    def event_bus(self) -> Any:
        """Создание тестового обработчика событий."""
        return OptimizedEventBus(max_workers=10, queue_size=10000)
    @pytest.fixture
    def sample_events(self) -> List[Dict[str, Any]]:
        """Создание тестовых событий."""
        events = []
        for i in range(1000):
            event = {
                'name': f'test_event_{i % 10}',
                'data': {'value': i, 'timestamp': time.time()},
                'priority': 'normal',
                'source': 'test'
            }
            events.append(event)
        return events
    @pytest.mark.asyncio
    async def test_event_publishing_performance(self, event_bus, sample_events) -> None:
        """Тест производительности публикации событий."""
        await event_bus.start()
        start_time = time.time()
        # Публикуем 1000 событий
        for event in sample_events:
            event_bus.publish(**event)
        end_time = time.time()
        publishing_time = end_time - start_time
        # Проверяем, что публикация 1000 событий занимает менее 1 секунды
        assert publishing_time < 1.0, f"Event publishing took {publishing_time:.3f} seconds"
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_event_processing_performance(self, event_bus) -> None:
        """Тест производительности обработки событий."""
        processed_events = []
        def test_handler(event) -> None:
            processed_events.append(event)
        # Подписываемся на событие
        event_bus.subscribe('test_event', test_handler)
        await event_bus.start()
        # Публикуем события
        for i in range(100):
            event_bus.publish('test_event', {'value': i})
        # Ждем обработки
        await asyncio.sleep(1)
        # Проверяем, что все события обработаны
        assert len(processed_events) == 100
        await event_bus.stop()
    @pytest.mark.asyncio
    async def test_concurrent_event_processing(self, event_bus) -> None:
        """Тест параллельной обработки событий."""
        processed_events = []
        lock = asyncio.Lock()
        async def async_handler(event) -> Any:
            async with lock:
                processed_events.append(event)
            await asyncio.sleep(0.001)  # Имитируем работу
        # Подписываемся на событие
        event_bus.subscribe('test_event', async_handler, async_handler=True)
        await event_bus.start()
        # Публикуем события параллельно
        tasks = []
        for i in range(50):
            task = asyncio.create_task(
                asyncio.gather(*[
                    asyncio.create_task(event_bus.publish('test_event', {'value': j}))
                    for j in range(10)
                ])
            )
            tasks.append(task)
        await asyncio.gather(*tasks)
        # Ждем обработки
        await asyncio.sleep(2)
        # Проверяем, что все события обработаны
        assert len(processed_events) == 500
        await event_bus.stop()
    def test_event_bus_metrics(self, event_bus) -> None:
        """Тест метрик обработчика событий."""
        metrics = event_bus.get_performance_metrics()
        # Проверяем наличие метрик
        assert 'total_events_processed' in metrics
        assert 'events_per_second' in metrics
        assert 'queue_size' in metrics
        assert 'active_workers' in metrics
class TestRepositoryPerformance:
    """Тесты производительности репозиториев."""
    @pytest.fixture
    def trading_repository(self) -> Any:
        return InMemoryTradingRepository()
    @pytest.fixture
    def strategy_repository(self) -> Any:
        return InMemoryStrategyRepository()
    @pytest.fixture
    def ml_repository(self) -> Any:
        return InMemoryMLRepository()
    @pytest.fixture
    def sample_trades(self) -> List[Trade]:
        """Создание тестовых сделок."""
        trades = []
        for i in range(1000):
            trade = Trade(
                id=TradeId(uuid4()),
                order_id=OrderId(uuid4()),
                trading_pair=TradingPair("BTCUSDT"),  # Исправление: используем правильный тип
                side=domain.entities.trading.OrderSide.BUY if i % 2 == 0 else domain.entities.trading.OrderSide.SELL,  # Исправление: используем правильный тип
                quantity=Volume(Decimal("0.1"), Currency.USDT),  # Исправление: используем Volume вместо VolumeValue
                price=Price(Decimal("50000") + Decimal(str(i)), Currency.USDT),
            )
            trades.append(trade)
        return trades
    @pytest.fixture
    def sample_strategies(self) -> List[Strategy]:
        """Создание тестовых стратегий."""
        strategies = []
        for i in range(100):
            strategy = Strategy(
                id=uuid4(),
                name=f"Strategy {i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                status=StrategyStatus.ACTIVE,
                parameters=StrategyParameters(parameters={"param1": "value1"}),
                created_at=datetime.now()
            )
            strategies.append(strategy)
        return strategies
    @pytest.mark.asyncio
    async def test_trading_repository_performance(self, trading_repository, sample_trades) -> None:
        """Тест производительности торгового репозитория."""
        # Тест массовой вставки
        start_time = time.time()
        for trade in sample_trades:
            await trading_repository.save_trade(trade)
        insert_time = time.time() - start_time
        assert insert_time < 2.0, f"Bulk insert took {insert_time:.3f} seconds"
        # Тест запросов
        start_time = time.time()
        for i in range(100):
            trade = await trading_repository.get_trade(f"trade_{i}")
            assert trade is not None
        query_time = time.time() - start_time
        assert query_time < 1.0, f"Queries took {query_time:.3f} seconds"
    @pytest.mark.asyncio
    async def test_strategy_repository_performance(self, strategy_repository, sample_strategies) -> None:
        """Тест производительности репозитория стратегий."""
        # Тест массовой вставки
        start_time = time.time()
        for strategy in sample_strategies:
            await strategy_repository.save_strategy(strategy)
        insert_time = time.time() - start_time
        assert insert_time < 1.0, f"Bulk insert took {insert_time:.3f} seconds"
        # Тест запросов по типу
        start_time = time.time()
        strategies = await strategy_repository.get_strategies_by_type("trend_following")
        assert len(strategies) == 100
        query_time = time.time() - start_time
        assert query_time < 0.1, f"Query took {query_time:.3f} seconds"
    @pytest.mark.asyncio
    async def test_ml_repository_performance(self, ml_repository) -> None:
        """Тест производительности репозитория ML."""
        # Создаем тестовые модели
        models = []
        for i in range(50):
            model = Model(
                id=f"model_{i}",
                name=f"Test Model {i}",
                model_type=ModelType.PRICE_PREDICTION,
                status=ModelStatus.ACTIVE,
                accuracy=0.8 + (i * 0.001),
                created_at=Timestamp.now()
            )
            models.append(model)
        # Тест массовой вставки
        start_time = time.time()
        for model in models:
            await ml_repository.save_model(model)
        insert_time = time.time() - start_time
        assert insert_time < 1.0, f"Bulk insert took {insert_time:.3f} seconds"
        # Тест поиска лучшей модели
        start_time = time.time()
        best_model = await ml_repository.get_best_model("price_prediction")
        assert best_model is not None
        query_time = time.time() - start_time
        assert query_time < 0.1, f"Query took {query_time:.3f} seconds"
class TestMemoryUsage:
    """Тесты использования памяти."""
    def test_large_dataset_memory_usage(self) -> None:
        """Тест использования памяти при работе с большими наборами данных."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        # Создаем большой набор данных
        large_dataset = []
        for i in range(100000):
            large_dataset.append({
                'id': i,
                'data': f'data_{i}',
                'timestamp': time.time()
            })
        memory_after_creation = process.memory_info().rss
        memory_increase = memory_after_creation - initial_memory
        # Проверяем, что увеличение памяти разумное (< 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB"
        # Очищаем данные
        del large_dataset
        memory_after_cleanup = process.memory_info().rss
        memory_decrease = memory_after_creation - memory_after_cleanup
        # Проверяем, что память освободилась
        assert memory_decrease > 0, "Memory should be freed after cleanup"
class TestConcurrencyPerformance:
    """Тесты производительности при параллельной работе."""
    @pytest.mark.asyncio
    async def test_concurrent_database_operations(self) -> None:
        """Тест параллельных операций с базой данных."""
        database = OptimizedDatabase("sqlite:///test_concurrency.db")
        async def insert_operation(i) -> Any:
            trade = Trade(
                id=TradeId(f"trade_{i}"),
                order_id=OrderId(f"order_{i}"),
                trading_pair=TradingPair("BTCUSDT"),
                side=OrderSide.BUY,
                volume=Volume(Decimal("0.1"), Currency.USDT),  # Исправление: используем Volume вместо VolumeValue
                price=Price(Decimal("50000"), Currency.USDT),
                executed_at=TimestampValue(Timestamp.now().value)
            )
            await database.save_trade(trade)
        # Запускаем 100 параллельных операций вставки
        start_time = time.time()
        tasks = [insert_operation(i) for i in range(100)]
        await asyncio.gather(*tasks)
        end_time = time.time()
        execution_time = end_time - start_time
        # Проверяем, что параллельные операции выполняются эффективно
        assert execution_time < 5.0, f"Concurrent operations took {execution_time:.3f} seconds"
        # Проверяем, что все данные сохранены
        trades = await database.get_trades("BTCUSDT")
        assert len(trades) == 100
    @pytest.mark.asyncio
    async def test_concurrent_repository_operations(self) -> None:
        """Тест параллельных операций с репозиториями."""
        repository = InMemoryTradingRepository()
        async def repository_operation(i) -> Any:
            trade = Trade(
                id=TradeId(f"trade_{i}"),
                order_id=OrderId(f"order_{i}"),
                trading_pair=TradingPair("BTCUSDT"),
                side=OrderSide.BUY,
                volume=Volume(Decimal("0.1"), Currency.USDT),  # Исправление: используем Volume вместо VolumeValue
                price=Price(Decimal("50000"), Currency.USDT),
                executed_at=TimestampValue(Timestamp.now().value)
            )
            await repository.save_trade(trade)
            return await repository.get_trade(trade.id)
        # Запускаем 50 параллельных операций
        start_time = time.time()
        tasks = [repository_operation(i) for i in range(50)]
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        execution_time = end_time - start_time
        # Проверяем, что все операции выполнены успешно
        assert len(results) == 50
        assert all(result is not None for result in results)
        # Проверяем производительность
        assert execution_time < 2.0, f"Concurrent repository operations took {execution_time:.3f} seconds" 

def test_order_creation_performance() -> None:
    """Тест производительности создания ордеров."""
    start_time = time.time()
    
    orders = []
    for i in range(1000):
        order = Order(
            id=OrderId(uuid4()),
            trading_pair=TradingPair("BTCUSDT"),
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("1.0"), Currency.USDT),
            status=OrderStatus.PENDING,
        )
        orders.append(order)
    
    end_time = time.time()
    creation_time = end_time - start_time
    
    assert len(orders) == 1000
    assert creation_time < 1.0  # Создание должно быть быстрым

def test_trade_creation_performance() -> None:
    """Тест производительности создания сделок."""
    start_time = time.time()
    
    trades = []
    for i in range(1000):
        trade = Trade(
            id=TradeId(uuid4()),
            order_id=OrderId(uuid4()),
            trading_pair=TradingPair("BTCUSDT"),
            side=OrderSide.BUY,
            volume=Volume(Decimal("1.0"), Currency.USDT),
            price=Price(Decimal("50000.0"), Currency.USDT),
            executed_at=TimestampValue(Timestamp.now().value)
        )
        trades.append(trade)
    
    end_time = time.time()
    creation_time = end_time - start_time
    
    assert len(trades) == 1000
    assert creation_time < 1.0  # Создание должно быть быстрым 
