"""
Интеграционные тесты для стратегий.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
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
from domain.strategies.exceptions import (
    StrategyFactoryError,
    StrategyCreationError,
    StrategyValidationError,
    StrategyRegistryError,
    StrategyNotFoundError,
    StrategyDuplicateError,
)
from domain.strategies.utils import StrategyUtils
from domain.strategies.validators import StrategyValidator


class TestStrategyFactoryIntegration:
    """Интеграционные тесты для фабрики стратегий."""

    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать фабрику стратегий."""
        return StrategyFactory()

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать тестовые рыночные данные."""
        return MarketData(
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

    def test_strategy_registration_and_creation(self, factory) -> None:
        """Тест регистрации и создания стратегий."""
        # Регистрируем стратегии
        factory.register_strategy(
            name="trend_following_test",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Test trend following strategy",
            version="1.0.0",
            author="Test Author",
            required_parameters=["short_period", "long_period"],
            optional_parameters=["rsi_period"],
            supported_pairs=["BTC/USDT", "ETH/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )
        # Создаем стратегию
        strategy = factory.create_strategy(
            name="trend_following_test",
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20, "rsi_period": 14},
            risk_level="medium",
            confidence_threshold=Decimal("0.6"),
        )
        assert isinstance(strategy, TrendFollowingStrategy)
        assert strategy.get_strategy_type() == StrategyType.TREND_FOLLOWING
        assert len(strategy.get_trading_pairs()) == 1
        assert strategy.get_trading_pairs()[0] == TradingPair("BTC/USDT")

    def test_strategy_creation_with_invalid_params(self, factory) -> None:
        """Тест создания стратегии с некорректными параметрами."""
        factory.register_strategy(
            name="trend_following_test",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Test strategy",
            version="1.0.0",
            author="Test Author",
        )
        # Попытка создания с некорректными параметрами
        with pytest.raises(StrategyCreationError):
            factory.create_strategy(name="trend_following_test", trading_pairs=[], parameters={}, risk_level="invalid")

    def test_strategy_factory_statistics(self, factory) -> None:
        """Тест статистики фабрики."""
        # Регистрируем несколько стратегий
        for i in range(3):
            factory.register_strategy(
                name=f"strategy_{i}",
                creator_func=TrendFollowingStrategy,
                strategy_type=StrategyType.TREND_FOLLOWING,
                description=f"Strategy {i}",
                version="1.0.0",
                author="Test Author",
            )
        stats = factory.get_factory_stats()
        assert stats["total_strategies"] == 3
        assert "trend_following" in stats["type_distribution"]
        assert stats["type_distribution"]["trend_following"] == 3


class TestStrategyRegistryIntegration:
    """Интеграционные тесты для реестра стратегий."""

    @pytest.fixture
    def registry(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать реестр стратегий."""
        return StrategyRegistry()

    @pytest.fixture
    def sample_strategy(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать тестовую стратегию."""
        return TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT", "ETH/USDT"],
            parameters={"short_period": 10, "long_period": 20, "rsi_period": 14},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6")),
        )

    def test_strategy_registration_and_retrieval(self, registry, sample_strategy) -> None:
        """Тест регистрации и получения стратегии."""
        # Регистрируем стратегию
        strategy_id = registry.register_strategy(
            strategy=sample_strategy, name="Test Strategy", tags=["trend", "test"], priority=1
        )
        # Получаем стратегию
        retrieved_strategy = registry.get_strategy(strategy_id)
        assert retrieved_strategy is not None
        assert retrieved_strategy.get_strategy_id() == strategy_id
        # Получаем метаданные
        metadata = registry.get_strategy_metadata(strategy_id)
        assert metadata is not None
        assert metadata.name == "Test Strategy"
        assert metadata.strategy_type == StrategyType.TREND_FOLLOWING
        assert "trend" in metadata.tags
        assert metadata.priority == 1

    def test_strategy_search_and_filtering(self, registry) -> None:
        """Тест поиска и фильтрации стратегий."""
        # Создаем и регистрируем несколько стратегий
        strategies = []
        for i in range(5):
            strategy = TrendFollowingStrategy(
                strategy_id=StrategyId(uuid4()),
                name=f"Strategy {i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                trading_pairs=[f"PAIR{i}/USDT"],
                parameters={"param": i},
                risk_level=RiskLevel(Decimal("0.5")),
                confidence_threshold=ConfidenceLevel(Decimal("0.6")),
            )
            strategy_id = registry.register_strategy(
                strategy=strategy, name=f"Strategy {i}", tags=[f"tag{i}"], priority=i
            )
            strategies.append((strategy_id, strategy))
        # Активируем некоторые стратегии
        registry.update_strategy_status(strategies[0][0], StrategyStatus.ACTIVE)
        registry.update_strategy_status(strategies[1][0], StrategyStatus.ACTIVE)
        # Тестируем поиск по типу
        trend_strategies = registry.get_strategies_by_type(StrategyType.TREND_FOLLOWING)
        assert len(trend_strategies) == 5
        # Тестируем поиск по статусу
        active_strategies = registry.get_strategies_by_status(StrategyStatus.ACTIVE)
        assert len(active_strategies) == 2
        # Тестируем поиск по торговой паре
        pair_strategies = registry.get_strategies_by_pair("PAIR0/USDT")
        assert len(pair_strategies) == 1
        # Тестируем поиск по тегу
        tagged_strategies = registry.get_strategies_by_tag("tag0")
        assert len(tagged_strategies) == 1
        # Тестируем комплексный поиск
        search_results = registry.search_strategies(
            name_pattern="Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            status=StrategyStatus.ACTIVE,
            active_only=True,
        )
        assert len(search_results) == 2

    def test_strategy_performance_tracking(self, registry, sample_strategy) -> None:
        """Тест отслеживания производительности стратегии."""
        strategy_id = registry.register_strategy(strategy=sample_strategy, name="Performance Test Strategy")
        # Обновляем производительность
        registry.update_strategy_performance(
            strategy_id=strategy_id, execution_time=0.5, success=True, pnl=Decimal("100.0")
        )
        registry.update_strategy_performance(
            strategy_id=strategy_id, execution_time=0.3, success=False, pnl=Decimal("-50.0"), error_message="Test error"
        )
        # Проверяем статистику
        metadata = registry.get_strategy_metadata(strategy_id)
        assert metadata.execution_count == 2
        assert metadata.success_count == 1
        assert metadata.total_pnl == Decimal("50.0")
        assert metadata.error_count == 1
        assert metadata.last_error == "Test error"

    def test_registry_statistics(self, registry) -> None:
        """Тест статистики реестра."""
        # Создаем и регистрируем стратегии с разными статусами
        for i in range(10):
            strategy = TrendFollowingStrategy(
                strategy_id=StrategyId(uuid4()),
                name=f"Strategy {i}",
                strategy_type=StrategyType.TREND_FOLLOWING,
                trading_pairs=["BTC/USDT"],
                parameters={},
                risk_level=RiskLevel(Decimal("0.5")),
                confidence_threshold=ConfidenceLevel(Decimal("0.6")),
            )
            strategy_id = registry.register_strategy(strategy=strategy)
            # Устанавливаем разные статусы
            if i < 3:
                registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
            elif i < 5:
                registry.update_strategy_status(strategy_id, StrategyStatus.PAUSED)
            elif i < 7:
                registry.update_strategy_status(strategy_id, StrategyStatus.STOPPED)
            else:
                registry.update_strategy_status(strategy_id, StrategyStatus.ERROR)
        stats = registry.get_registry_stats()
        assert stats["total_strategies"] == 10
        assert stats["active_strategies"] == 3
        assert stats["paused_strategies"] == 2
        assert stats["stopped_strategies"] == 2
        assert stats["error_strategies"] == 3


class TestStrategyWorkflowIntegration:
    """Интеграционные тесты для полного workflow стратегий."""

    @pytest.fixture
    def factory(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать фабрику стратегий."""
        return get_strategy_factory()

    @pytest.fixture
    def registry(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать реестр стратегий."""
        return get_strategy_registry()

    @pytest.fixture
    def market_data_series(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать серию рыночных данных."""
        data_series = []
        base_price = Decimal("50000")
        for i in range(100):
            timestamp = datetime.now() + timedelta(minutes=i)
            price_change = Decimal(str(i * 0.001))  # Небольшой тренд
            current_price = base_price + price_change
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                open=Price(current_price - Decimal("10"), Currency.USDT),
                high=Price(current_price + Decimal("20"), Currency.USDT),
                low=Price(current_price - Decimal("20"), Currency.USDT),
                close=Price(current_price, Currency.USDT),
                volume=Volume(Decimal("1000"), Currency.USDT),
                bid=Price(current_price - Decimal("5"), Currency.USDT),
                ask=Price(current_price + Decimal("5"), Currency.USDT),
                bid_volume=Volume(Decimal("500"), Currency.USDT),
                ask_volume=Volume(Decimal("500"), Currency.USDT),
            )
            data_series.append(data)
        return data_series

    def test_complete_strategy_lifecycle(self, factory, registry, market_data_series) -> None:
        """Тест полного жизненного цикла стратегии."""
        # 1. Регистрируем стратегию в фабрике
        factory.register_strategy(
            name="lifecycle_test_strategy",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Test lifecycle strategy",
            version="1.0.0",
            author="Test Author",
            required_parameters=["short_period", "long_period"],
            supported_pairs=["BTC/USDT"],
        )
        # 2. Создаем стратегию через фабрику
        strategy = factory.create_strategy(
            name="lifecycle_test_strategy",
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 5, "long_period": 10, "rsi_period": 14},
            risk_level="medium",
            confidence_threshold=Decimal("0.6"),
        )
        # 3. Регистрируем стратегию в реестре
        strategy_id = registry.register_strategy(
            strategy=strategy, name="Lifecycle Test Strategy", tags=["test", "lifecycle"], priority=1
        )
        # 4. Активируем стратегию
        strategy.activate()
        registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
        # 5. Обрабатываем рыночные данные и генерируем сигналы
        signals = []
        for i, market_data in enumerate(market_data_series[20:]):  # Пропускаем первые 20 для инициализации
            try:
                signal = strategy.generate_signal(market_data)
                if signal:
                    signals.append(signal)
                    # Обновляем производительность
                    registry.update_strategy_performance(
                        strategy_id=strategy_id,
                        execution_time=0.1,
                        success=True,
                        pnl=Decimal("10.0") if signal.signal_type == SignalType.BUY else Decimal("-5.0"),
                    )
            except Exception as e:
                registry.update_strategy_performance(
                    strategy_id=strategy_id, execution_time=0.1, success=False, pnl=Decimal("0.0"), error_message=str(e)
                )
        # 6. Проверяем результаты
        assert len(signals) > 0, "Должны быть сгенерированы сигналы"
        metadata = registry.get_strategy_metadata(strategy_id)
        assert metadata.execution_count > 0
        assert metadata.is_active
        # 7. Получаем статистику
        registry_stats = registry.get_registry_stats()
        assert registry_stats["total_strategies"] > 0
        assert registry_stats["active_strategies"] > 0
        factory_stats = factory.get_factory_stats()
        assert factory_stats["total_strategies"] > 0
        assert factory_stats["successful_creations"] > 0

    def test_multiple_strategy_types_integration(self, factory, registry, market_data_series) -> None:
        """Тест интеграции нескольких типов стратегий."""
        # Регистрируем разные типы стратегий
        strategy_types = [
            (TrendFollowingStrategy, StrategyType.TREND_FOLLOWING, "trend"),
            (MeanReversionStrategy, StrategyType.MEAN_REVERSION, "mean_reversion"),
            (BreakoutStrategy, StrategyType.BREAKOUT, "breakout"),
            (ScalpingStrategy, StrategyType.SCALPING, "scalping"),
            (ArbitrageStrategy, StrategyType.ARBITRAGE, "arbitrage"),
        ]
        created_strategies = []
        for strategy_class, strategy_type, name in strategy_types:
            # Регистрируем в фабрике
            factory.register_strategy(
                name=f"{name}_test",
                creator_func=strategy_class,
                strategy_type=strategy_type,
                description=f"Test {name} strategy",
                version="1.0.0",
                author="Test Author",
                supported_pairs=["BTC/USDT"],
            )
            # Создаем стратегию
            strategy = factory.create_strategy(
                name=f"{name}_test",
                trading_pairs=["BTC/USDT"],
                parameters={},
                risk_level="medium",
                confidence_threshold=Decimal("0.6"),
            )
            # Регистрируем в реестре
            strategy_id = registry.register_strategy(
                strategy=strategy, name=f"{name.title()} Test Strategy", tags=[name, "test"]
            )
            created_strategies.append((strategy_id, strategy, strategy_type))
        # Активируем все стратегии
        for strategy_id, strategy, _ in created_strategies:
            strategy.activate()
            registry.update_strategy_status(strategy_id, StrategyStatus.ACTIVE)
        # Тестируем генерацию сигналов для всех стратегий
        for strategy_id, strategy, strategy_type in created_strategies:
            # Берем последние данные для тестирования
            test_data = market_data_series[-1]
            try:
                signal = strategy.generate_signal(test_data)
                # Обновляем производительность
                registry.update_strategy_performance(
                    strategy_id=strategy_id, execution_time=0.1, success=True, pnl=Decimal("5.0")
                )
                if signal:
                    assert signal.strategy_id == strategy.get_strategy_id()
                    assert signal.trading_pair == "BTC/USDT"
            except Exception as e:
                registry.update_strategy_performance(
                    strategy_id=strategy_id, execution_time=0.1, success=False, pnl=Decimal("0.0"), error_message=str(e)
                )
        # Проверяем статистику
        registry_stats = registry.get_registry_stats()
        assert registry_stats["total_strategies"] == len(strategy_types)
        assert registry_stats["active_strategies"] == len(strategy_types)
        # Проверяем распределение по типам
        type_distribution = registry_stats["strategy_type_distribution"]
        assert len(type_distribution) == len(strategy_types)

    def test_strategy_error_handling_integration(self, factory, registry) -> None:
        """Тест обработки ошибок в интеграции."""
        # Создаем стратегию с некорректными параметрами
        factory.register_strategy(
            name="error_test_strategy",
            creator_func=TrendFollowingStrategy,
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Error test strategy",
            version="1.0.0",
            author="Test Author",
        )
        # Попытка создания с ошибкой
        with pytest.raises(StrategyCreationError):
            factory.create_strategy(
                name="error_test_strategy", trading_pairs=[], parameters={}, risk_level="invalid"  # Пустой список
            )
        # Создаем корректную стратегию
        strategy = factory.create_strategy(
            name="error_test_strategy",
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level="medium",
        )
        strategy_id = registry.register_strategy(strategy=strategy)
        # Симулируем ошибку выполнения
        registry.update_strategy_performance(
            strategy_id=strategy_id,
            execution_time=0.1,
            success=False,
            pnl=Decimal("0.0"),
            error_message="Simulated error",
        )
        # Проверяем, что ошибка зафиксирована
        metadata = registry.get_strategy_metadata(strategy_id)
        assert metadata.error_count == 1
        assert metadata.last_error == "Simulated error"
        # Проверяем статистику ошибок
        registry_stats = registry.get_registry_stats()
        assert registry_stats["failed_executions"] > 0


class TestStrategyUtilsIntegration:
    """Интеграционные тесты для утилит стратегий."""

    @pytest.fixture
    def utils(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать утилиты стратегий."""
        return StrategyUtils()

    @pytest.fixture
    def validator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать валидатор стратегий."""
        return StrategyValidator()

    def test_strategy_validation_integration(self, validator) -> None:
        """Тест интеграции валидации стратегий."""
        # Тестируем валидацию конфигурации
        config = {
            "name": "Test Strategy",
            "strategy_type": "trend_following",
            "trading_pairs": ["BTC/USDT"],
            "parameters": {"short_period": 10, "long_period": 20, "rsi_period": 14},
            "risk_level": "medium",
            "confidence_threshold": 0.6,
        }
        errors = validator.validate_strategy_config(config)
        assert len(errors) == 0, f"Validation errors: {errors}"
        # Тестируем валидацию с ошибками
        invalid_config = {
            "name": "",
            "strategy_type": "invalid_type",
            "trading_pairs": [],
            "parameters": {"short_period": -1, "long_period": 0},
        }
        errors = validator.validate_strategy_config(invalid_config)
        assert len(errors) > 0, "Should have validation errors"

    def test_strategy_utils_integration(self, utils) -> None:
        """Тест интеграции утилит стратегий."""
        # Создаем тестовые данные
        prices = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
        # Тестируем расчет индикаторов
        sma = utils.calculate_sma(prices, 5)
        assert len(sma) == len(prices)
        assert sma[-1] > sma[0]  # Должен быть восходящий тренд
        rsi = utils.calculate_rsi(prices, 14)
        assert len(rsi) == len(prices)
        assert 0 <= rsi[-1] <= 100
        # Тестируем анализ тренда
        trend_analysis = utils.analyze_trend(prices)
        assert "direction" in trend_analysis
        assert "strength" in trend_analysis
        assert "duration" in trend_analysis
        # Тестируем анализ волатильности
        volatility_analysis = utils.analyze_volatility(prices)
        assert "current_volatility" in volatility_analysis
        assert "volatility_trend" in volatility_analysis
        assert "regime" in volatility_analysis


if __name__ == "__main__":
    pytest.main([__file__])
