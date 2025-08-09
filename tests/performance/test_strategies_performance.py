"""
Тесты производительности для стратегий.
"""

import time
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
import statistics
from domain.entities.market import MarketData, MarketState
from domain.type_definitions import StrategyId, TradingPair, ConfidenceLevel, RiskLevel
from domain.strategies.strategy_factory import StrategyFactory, get_strategy_factory
from domain.strategies.strategy_registry import StrategyRegistry, get_strategy_registry
from domain.strategies.base_strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    ScalpingStrategy,
    ArbitrageStrategy,
)
from domain.type_definitions.strategy_types import (
    StrategyCategory,
    RiskProfile,
    Timeframe,
    StrategyConfig,
    TrendFollowingParams,
    MeanReversionParams,
    BreakoutParams,
    ScalpingParams,
    ArbitrageParams,
)
from domain.strategies.utils import StrategyUtils
from domain.strategies.validators import StrategyValidator


class TestStrategyPerformance:
    """Тесты производительности стратегий."""

    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать фабрику стратегий."""
        return StrategyFactory()

    @pytest.fixture
    def registry(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать реестр стратегий."""
        return StrategyRegistry()

    @pytest.fixture
    def large_market_dataset(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать большой набор рыночных данных."""
        dataset = []
        base_price = Decimal("50000")
        # Создаем 10000 точек данных
        for i in range(10000):
            timestamp = datetime.now() + timedelta(minutes=i)
            # Симулируем реалистичное движение цены
            trend = Decimal(str(0.001 * (i % 100)))
            noise = Decimal(str(0.0001 * (i % 10 - 5)))
            current_price = base_price + trend + noise
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                open=Price(current_price - Decimal("5"), Currency.USDT),
                high=Price(current_price + Decimal("15"), Currency.USDT),
                low=Price(current_price - Decimal("15"), Currency.USDT),
                close=Price(current_price, Currency.USDT),
                volume=Volume(Decimal("1000"), Currency.USDT),
                bid=Price(current_price - Decimal("2"), Currency.USDT),
                ask=Price(current_price + Decimal("2"), Currency.USDT),
                bid_volume=Volume(Decimal("500"), Currency.USDT),
                ask_volume=Volume(Decimal("500"), Currency.USDT),
            )
            dataset.append(data)
        return dataset

    def test_strategy_creation_performance(self, factory) -> None:
        """Тест производительности создания стратегий."""
        # Регистрируем стратегии
        strategy_types = [
            (TrendFollowingStrategy, StrategyType.TREND_FOLLOWING, "trend"),
            (MeanReversionStrategy, StrategyType.MEAN_REVERSION, "mean_reversion"),
            (BreakoutStrategy, StrategyType.BREAKOUT, "breakout"),
            (ScalpingStrategy, StrategyType.SCALPING, "scalping"),
            (ArbitrageStrategy, StrategyType.ARBITRAGE, "arbitrage"),
        ]
        for strategy_class, strategy_type, name in strategy_types:
            factory.register_strategy(
                name=f"{name}_perf",
                creator_func=strategy_class,
                strategy_type=strategy_type,
                description=f"Performance test {name}",
                version="1.0.0",
                author="Performance Test",
            )
        # Тестируем создание 100 стратегий
        creation_times = []
        for i in range(100):
            start_time = time.time()
            strategy = factory.create_strategy(
                name="trend_perf",
                trading_pairs=["BTC/USDT"],
                parameters={"short_period": 10, "long_period": 20},
                risk_level="medium",
                confidence_threshold=Decimal("0.6"),
            )
            end_time = time.time()
            creation_times.append(end_time - start_time)
        # Анализируем производительность
        avg_creation_time = statistics.mean(creation_times)
        max_creation_time = max(creation_times)
        min_creation_time = min(creation_times)
        print(f"Strategy Creation Performance:")
        print(f"  Average time: {avg_creation_time:.4f} seconds")
        print(f"  Max time: {max_creation_time:.4f} seconds")
        print(f"  Min time: {min_creation_time:.4f} seconds")
        # Проверяем, что создание достаточно быстрое
        assert avg_creation_time < 0.1, f"Average creation time too slow: {avg_creation_time}"
        assert max_creation_time < 0.5, f"Max creation time too slow: {max_creation_time}"

    def test_strategy_signal_generation_performance(self, factory, large_market_dataset) -> None:
        """Тест производительности генерации сигналов."""
        # Создаем стратегию
        factory.register_strategy(
            name="perf_trend",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Performance test trend strategy",
            version="1.0.0",
            author="Performance Test",
        )
        strategy = factory.create_strategy(
            name="perf_trend",
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level="medium",
            confidence_threshold=Decimal("0.6"),
        )
        strategy.activate()
        # Тестируем генерацию сигналов на большом наборе данных
        signal_generation_times = []
        signals_generated = 0
        for market_data in large_market_dataset:
            start_time = time.time()
            signal = strategy.generate_signal(market_data)
            end_time = time.time()
            signal_generation_times.append(end_time - start_time)
            if signal:
                signals_generated += 1
        # Анализируем производительность
        avg_generation_time = statistics.mean(signal_generation_times)
        max_generation_time = max(signal_generation_times)
        min_generation_time = min(signal_generation_times)
        print(f"Signal Generation Performance:")
        print(f"  Total signals generated: {signals_generated}")
        print(f"  Average generation time: {avg_generation_time:.6f} seconds")
        print(f"  Max generation time: {max_generation_time:.6f} seconds")
        print(f"  Min generation time: {min_generation_time:.6f} seconds")
        print(f"  Signals per second: {1/avg_generation_time:.2f}")
        # Проверяем производительность
        assert avg_generation_time < 0.001, f"Average generation time too slow: {avg_generation_time}"
        assert max_generation_time < 0.01, f"Max generation time too slow: {max_generation_time}"
        assert signals_generated > 0, "No signals generated"

    def test_strategy_registry_performance(self, registry) -> None:
        """Тест производительности реестра стратегий."""
        # Создаем много стратегий
        strategies = []
        for i in range(1000):
            strategy = TrendFollowingStrategy(
                strategy_id=StrategyId(uuid4()),
                name=f"Strategy {i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                trading_pairs=[f"PAIR{i}/USDT"],
                parameters={"param": i},
                risk_level=RiskLevel(Decimal("0.5")),
                confidence_threshold=ConfidenceLevel(Decimal("0.6")),
            )
            strategies.append(strategy)
        # Тестируем регистрацию
        registration_times = []
        for strategy in strategies:
            start_time = time.time()
            registry.register_strategy(strategy, name=strategy._name)
            end_time = time.time()
            registration_times.append(end_time - start_time)
        avg_registration_time = statistics.mean(registration_times)
        print(f"Registry Registration Performance:")
        print(f"  Average registration time: {avg_registration_time:.6f} seconds")
        # Тестируем поиск
        search_times = []
        for i in range(100):
            start_time = time.time()
            results = registry.search_strategies(name_pattern=f"Strategy {i}")
            end_time = time.time()
            search_times.append(end_time - start_time)
        avg_search_time = statistics.mean(search_times)
        print(f"Registry Search Performance:")
        print(f"  Average search time: {avg_search_time:.6f} seconds")
        # Проверяем производительность
        assert avg_registration_time < 0.001, f"Registration too slow: {avg_registration_time}"
        assert avg_search_time < 0.01, f"Search too slow: {avg_search_time}"

    def test_strategy_memory_usage(self, factory) -> None:
        """Тест использования памяти стратегиями."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        # Создаем много стратегий
        strategies = []
        for i in range(1000):
            strategy = factory.create_strategy(
                name="trend_perf",
                trading_pairs=["BTC/USDT"],
                parameters={"short_period": 10, "long_period": 20},
                risk_level="medium",
                confidence_threshold=Decimal("0.6"),
            )
            strategies.append(strategy)
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        print(f"Memory Usage Performance:")
        print(f"  Initial memory: {initial_memory:.2f} MB")
        print(f"  Final memory: {final_memory:.2f} MB")
        print(f"  Memory increase: {memory_increase:.2f} MB")
        print(f"  Memory per strategy: {memory_increase/1000:.4f} MB")
        # Проверяем, что использование памяти разумное
        assert memory_increase < 100, f"Memory usage too high: {memory_increase} MB"
        assert memory_increase / 1000 < 0.1, f"Memory per strategy too high: {memory_increase/1000} MB"

    def test_strategy_concurrent_execution(self, factory) -> None:
        """Тест конкурентного выполнения стратегий."""
        import threading
        import queue

        # Создаем стратегию
        factory.register_strategy(
            name="concurrent_trend",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Concurrent test strategy",
            version="1.0.0",
            author="Performance Test",
        )
        # Создаем тестовые данные
        test_data = MarketData(
            symbol="BTC/USDT",
            timestamp=datetime.now(),
            open=Price(Decimal("50000"), Currency.USDT),
            high=Price(Decimal("51000"), Currency.USDT),
            low=Price(Decimal("49000"), Currency.USDT),
            close=Price(Decimal("50500"), Currency.USDT),
            volume=Volume(Decimal("1000"), Currency.USDT),
            bid=Price(Decimal("50490"), Currency.USDT),
            ask=Price(Decimal("50510"), Currency.USDT),
            bid_volume=Volume(Decimal("500"), Currency.USDT),
            ask_volume=Volume(Decimal("500"), Currency.USDT),
        )

        # Функция для выполнения в потоке
        def execute_strategy(thread_id, results_queue) -> Any:
            strategy = factory.create_strategy(
                name="concurrent_trend",
                trading_pairs=["BTC/USDT"],
                parameters={"short_period": 10, "long_period": 20},
                risk_level="medium",
                confidence_threshold=Decimal("0.6"),
            )
            strategy.activate()
            start_time = time.time()
            signals_generated = 0
            for _ in range(100):
                signal = strategy.generate_signal(test_data)
                if signal:
                    signals_generated += 1
            end_time = time.time()
            execution_time = end_time - start_time
            results_queue.put(
                {"thread_id": thread_id, "execution_time": execution_time, "signals_generated": signals_generated}
            )

        # Запускаем конкурентное выполнение
        num_threads = 10
        threads = []
        results_queue = queue.Queue()
        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=execute_strategy, args=(i, results_queue))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        end_time = time.time()
        total_time = end_time - start_time
        # Собираем результаты
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        # Анализируем результаты
        execution_times = [r["execution_time"] for r in results]
        avg_execution_time = statistics.mean(execution_times)
        total_signals = sum(r["signals_generated"] for r in results)
        print(f"Concurrent Execution Performance:")
        print(f"  Number of threads: {num_threads}")
        print(f"  Total execution time: {total_time:.4f} seconds")
        print(f"  Average thread execution time: {avg_execution_time:.4f} seconds")
        print(f"  Total signals generated: {total_signals}")
        print(f"  Signals per second: {total_signals/total_time:.2f}")
        # Проверяем производительность
        assert total_time < 10, f"Total execution time too slow: {total_time}"
        assert avg_execution_time < 1, f"Average thread execution time too slow: {avg_execution_time}"
        assert total_signals > 0, "No signals generated in concurrent execution"

    def test_strategy_utils_performance(self: "TestStrategyPerformance") -> None:
        """Тест производительности утилит стратегий."""
        utils = StrategyUtils()
        # Создаем большой набор данных для тестирования
        prices = [100 + i * 0.1 + (i % 10) * 0.01 for i in range(10000)]
        # Тестируем различные утилиты
        utilities_to_test = [
            ("SMA", lambda: utils.calculate_sma(prices, 20)),
            ("EMA", lambda: utils.calculate_ema(prices, 20)),
            ("RSI", lambda: utils.calculate_rsi(prices, 14)),
            ("MACD", lambda: utils.calculate_macd(prices)),
            ("Bollinger Bands", lambda: utils.calculate_bollinger_bands(prices, 20)),
            ("ATR", lambda: utils.calculate_atr(prices, 14)),
            ("Volatility", lambda: utils.calculate_volatility(prices, 20)),
        ]
        performance_results = {}
        for utility_name, utility_func in utilities_to_test:
            start_time = time.time()
            result = utility_func()
            end_time = time.time()
            execution_time = end_time - start_time
            performance_results[utility_name] = execution_time
            print(f"{utility_name} Performance: {execution_time:.6f} seconds")
        # Проверяем, что все утилиты работают достаточно быстро
        for utility_name, execution_time in performance_results.items():
            assert execution_time < 1.0, f"{utility_name} too slow: {execution_time} seconds"

    def test_strategy_validator_performance(self: "TestStrategyPerformance") -> None:
        """Тест производительности валидатора стратегий."""
        validator = StrategyValidator()
        # Создаем много конфигураций для валидации
        configs = []
        for i in range(1000):
            config = {
                "name": f"Strategy {i}",
                "strategy_type": "trend_following",
                "trading_pairs": [f"PAIR{i}/USDT"],
                "parameters": {"short_period": 10 + (i % 10), "long_period": 20 + (i % 10), "rsi_period": 14},
                "risk_level": "medium",
                "confidence_threshold": 0.6,
            }
            configs.append(config)
        # Тестируем валидацию
        validation_times = []
        for config in configs:
            start_time = time.time()
            errors = validator.validate_strategy_config(config)
            end_time = time.time()
            validation_times.append(end_time - start_time)
        avg_validation_time = statistics.mean(validation_times)
        max_validation_time = max(validation_times)
        print(f"Validator Performance:")
        print(f"  Average validation time: {avg_validation_time:.6f} seconds")
        print(f"  Max validation time: {max_validation_time:.6f} seconds")
        print(f"  Validations per second: {1/avg_validation_time:.2f}")
        # Проверяем производительность
        assert avg_validation_time < 0.001, f"Validation too slow: {avg_validation_time}"
        assert max_validation_time < 0.01, f"Max validation time too slow: {max_validation_time}"


class TestStrategyScalability:
    """Тесты масштабируемости стратегий."""

    def test_strategy_factory_scalability(self: "TestStrategyScalability") -> None:
        """Тест масштабируемости фабрики стратегий."""
        factory = StrategyFactory()
        # Регистрируем много типов стратегий
        for i in range(100):
            factory.register_strategy(
                name=f"strategy_type_{i}",
                creator_func=TrendFollowingStrategy,
                strategy_type=StrategyType.TREND_FOLLOWING,
                description=f"Strategy type {i}",
                version="1.0.0",
                author="Scalability Test",
            )
        # Создаем много стратегий
        creation_times = []
        for i in range(1000):
            start_time = time.time()
            strategy = factory.create_strategy(
                name=f"strategy_type_{i % 100}",
                trading_pairs=["BTC/USDT"],
                parameters={"param": i},
                risk_level="medium",
            )
            end_time = time.time()
            creation_times.append(end_time - start_time)
        avg_creation_time = statistics.mean(creation_times)
        print(f"Factory Scalability:")
        print(f"  Strategies created: 1000")
        print(f"  Strategy types: 100")
        print(f"  Average creation time: {avg_creation_time:.6f} seconds")
        assert avg_creation_time < 0.01, f"Creation time too slow: {avg_creation_time}"

    def test_strategy_registry_scalability(self: "TestStrategyScalability") -> None:
        """Тест масштабируемости реестра стратегий."""
        registry = StrategyRegistry()
        # Регистрируем много стратегий
        registration_times = []
        for i in range(10000):
            strategy = TrendFollowingStrategy(
                strategy_id=StrategyId(uuid4()),
                name=f"Strategy {i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                trading_pairs=[f"PAIR{i % 100}/USDT"],
                parameters={"param": i},
                risk_level=RiskLevel(Decimal("0.5")),
                confidence_threshold=ConfidenceLevel(Decimal("0.6")),
            )
            start_time = time.time()
            registry.register_strategy(strategy, name=strategy._name)
            end_time = time.time()
            registration_times.append(end_time - start_time)
        avg_registration_time = statistics.mean(registration_times)
        # Тестируем поиск в большом реестре
        search_times = []
        for i in range(100):
            start_time = time.time()
            results = registry.search_strategies(name_pattern=f"Strategy {i}")
            end_time = time.time()
            search_times.append(end_time - start_time)
        avg_search_time = statistics.mean(search_times)
        print(f"Registry Scalability:")
        print(f"  Strategies registered: 10000")
        print(f"  Average registration time: {avg_registration_time:.6f} seconds")
        print(f"  Average search time: {avg_search_time:.6f} seconds")
        assert avg_registration_time < 0.001, f"Registration too slow: {avg_registration_time}"
        assert avg_search_time < 0.01, f"Search too slow: {avg_search_time}"


if __name__ == "__main__":
    pytest.main([__file__])
