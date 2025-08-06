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
    ArbitrageStrategy
)
from domain.strategies.strategy_types import (
    StrategyCategory, RiskProfile, Timeframe, StrategyConfig,
    TrendFollowingParams, MeanReversionParams, BreakoutParams,
    ScalpingParams, ArbitrageParams
)
from domain.strategies.utils import StrategyUtils
from domain.strategies.validators import StrategyValidator
class TestStrategyPerformance:
    """Тесты производительности стратегий."""
    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать фабрику стратегий."""
        return StrategyFactory()
        # Создаем 10000 точек данных
            # Симулируем реалистичное движение цены
        # Регистрируем стратегии
        # Тестируем создание 100 стратегий
        # Анализируем производительность
        # Проверяем, что создание достаточно быстрое
        # Создаем стратегию
        # Тестируем генерацию сигналов на большом наборе данных
        # Анализируем производительность
        # Проверяем производительность
        # Создаем много стратегий
        # Тестируем регистрацию
        # Тестируем поиск
        # Проверяем производительность
        # Создаем много стратегий
        # Проверяем, что использование памяти разумное
        # Создаем стратегию
        # Создаем тестовые данные
        # Функция для выполнения в потоке
        # Запускаем конкурентное выполнение
        # Ждем завершения всех потоков
        # Собираем результаты
        # Анализируем результаты
        # Проверяем производительность
        # Создаем большой набор данных для тестирования
        # Тестируем различные утилиты
        # Проверяем, что все утилиты работают достаточно быстро
        # Создаем много конфигураций для валидации
        # Тестируем валидацию
        # Проверяем производительность
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
                author="Scalability Test"
            )
        # Создаем много стратегий
        creation_times = []
        for i in range(1000):
            start_time = time.time()
            strategy = factory.create_strategy(
                name=f"strategy_type_{i % 100}",
                trading_pairs=["BTC/USDT"],
                parameters={"param": i},
                risk_level="medium"
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
                confidence_threshold=ConfidenceLevel(Decimal("0.6"))
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
