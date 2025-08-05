"""
Доменные тесты для стратегий.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from domain.entities.market import MarketData, MarketState
from domain.entities.strategy import StrategyType, StrategyStatus
from domain.type_definitions import StrategyId, TradingPair, ConfidenceLevel, RiskLevel
from domain.strategies.base_strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    ScalpingStrategy,
    ArbitrageStrategy,
)
from domain.strategies.strategy_types import (
    StrategyCategory, RiskProfile, Timeframe, StrategyConfig,
    TrendFollowingParams, MeanReversionParams, BreakoutParams,
    ScalpingParams, ArbitrageParams
)
from domain.strategies.exceptions import (
    StrategyValidationError, StrategyExecutionError
)
from domain.strategies.validators import StrategyValidator


class TestStrategyDomainRules:
    """Тесты доменных правил стратегий."""
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
            ask_volume=Volume(Decimal("500"), Currency.USDT)
        )
    def test_strategy_activation_rules(self: "TestStrategyDomainRules") -> None:
        """Тест правил активации стратегии."""
        strategy = TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
        # Стратегия должна быть неактивна по умолчанию
        assert not strategy.is_active()
        # Активация стратегии
        strategy.activate()
        assert strategy.is_active()
        # Деактивация стратегии
        strategy.deactivate()
        assert not strategy.is_active()
        # Приостановка стратегии
        strategy.pause()
        assert not strategy.is_active()
    def test_strategy_parameter_validation_rules(self: "TestStrategyDomainRules") -> None:
        """Тест правил валидации параметров стратегии."""
        # Корректные параметры
        valid_params = {
            "short_period": 10,
            "long_period": 20,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        }
        strategy = TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT"],
            parameters=valid_params,
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
        # Валидация должна пройти успешно
        assert strategy.validate_parameters(valid_params)
        # Некорректные параметры
        invalid_params = {
            "short_period": -1,  # Отрицательный период
            "long_period": 0,    # Нулевой период
            "rsi_oversold": 100, # Некорректное значение RSI
            "rsi_overbought": 0  # Некорректное значение RSI
        }
        with pytest.raises(StrategyValidationError):
            strategy.validate_parameters(invalid_params)
    def test_strategy_trading_pair_rules(self: "TestStrategyDomainRules") -> None:
        """Тест правил работы с торговыми парами."""
        strategy = TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT", "ETH/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
        initial_pairs = strategy.get_trading_pairs()
        assert len(initial_pairs) == 2
        assert TradingPair("BTC/USDT") in initial_pairs
        assert TradingPair("ETH/USDT") in initial_pairs
        # Добавление новой пары
        strategy.add_trading_pair("ADA/USDT")
        updated_pairs = strategy.get_trading_pairs()
        assert len(updated_pairs) == 3
        assert TradingPair("ADA/USDT") in updated_pairs
        # Удаление пары
        strategy.remove_trading_pair("ETH/USDT")
        final_pairs = strategy.get_trading_pairs()
        assert len(final_pairs) == 2
        assert TradingPair("ETH/USDT") not in final_pairs
        # Проверка поддержки пары
        assert strategy._is_trading_pair_supported("BTC/USDT")
        assert not strategy._is_trading_pair_supported("UNKNOWN/USDT")
    def test_strategy_signal_generation_rules(self, sample_market_data) -> None:
        """Тест правил генерации сигналов."""
        strategy = TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Test Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
        # Неактивная стратегия не должна генерировать сигналы
        signal = strategy.generate_signal(sample_market_data)
        assert signal is None
        # Активируем стратегию
        strategy.activate()
        # Теперь должна генерировать сигналы
        signal = strategy.generate_signal(sample_market_data)
        if signal:
            assert signal.strategy_id == strategy.get_strategy_id()
            assert signal.trading_pair == "BTC/USDT"
            assert signal.confidence >= strategy._confidence_threshold
            assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
    def test_strategy_confidence_threshold_rules(self, sample_market_data) -> None:
        """Тест правил порога уверенности."""
        # Стратегия с высоким порогом уверенности
        high_confidence_strategy = TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="High Confidence Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.9"))  # Высокий порог
        )
        high_confidence_strategy.activate()
        # Стратегия с низким порогом уверенности
        low_confidence_strategy = TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Low Confidence Strategy",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT"],
            parameters={"short_period": 10, "long_period": 20},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.3"))  # Низкий порог
        )
        low_confidence_strategy.activate()
        # Стратегия с низким порогом должна генерировать больше сигналов
        high_signals = []
        low_signals = []
        for _ in range(10):
            high_signal = high_confidence_strategy.generate_signal(sample_market_data)
            low_signal = low_confidence_strategy.generate_signal(sample_market_data)
            if high_signal:
                high_signals.append(high_signal)
            if low_signal:
                low_signals.append(low_signal)
        # Стратегия с низким порогом должна генерировать больше сигналов
        assert len(low_signals) >= len(high_signals)
class TestStrategyBusinessLogic:
    """Тесты бизнес-логики стратегий."""
    @pytest.fixture
    def trend_following_strategy(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать трендовую стратегию."""
        return TrendFollowingStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Trend Following Test",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=["BTC/USDT"],
            parameters={
                "short_period": 10,
                "long_period": 20,
                "rsi_period": 14,
                "rsi_oversold": 30,
                "rsi_overbought": 70
            },
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
    @pytest.fixture
    def mean_reversion_strategy(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать стратегию возврата к среднему."""
        return MeanReversionStrategy(
            strategy_id=StrategyId(uuid4()),
            name="Mean Reversion Test",
            strategy_type=StrategyType.MEAN_REVERSION,
            trading_pairs=["BTC/USDT"],
            parameters={
                "lookback_period": 50,
                "deviation_threshold": Decimal("2.0"),
                "rsi_period": 14
            },
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
    def test_trend_following_business_logic(self, trend_following_strategy) -> None:
        """Тест бизнес-логики трендовой стратегии."""
        # Создаем данные с восходящим трендом
        uptrend_data = []
        base_price = Decimal("50000")
        for i in range(30):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = base_price + Decimal(str(i * 10))  # Восходящий тренд
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                open=Price(price - Decimal("5"), Currency.USDT),
                high=Price(price + Decimal("10"), Currency.USDT),
                low=Price(price - Decimal("10"), Currency.USDT),
                close=Price(price, Currency.USDT),
                volume=Volume(Decimal("1000"), Currency.USDT),
                bid=Price(price - Decimal("2"), Currency.USDT),
                ask=Price(price + Decimal("2"), Currency.USDT),
                bid_volume=Volume(Decimal("500"), Currency.USDT),
                ask_volume=Volume(Decimal("500"), Currency.USDT)
            )
            uptrend_data.append(data)
        trend_following_strategy.activate()
        # Анализируем последние данные
        last_data = uptrend_data[-1]
        analysis = trend_following_strategy.analyze_market(last_data)
        # Проверяем анализ
        assert "trend_analysis" in analysis
        assert "confidence_score" in analysis
        assert "market_regime" in analysis
        # Генерируем сигнал
        signal = trend_following_strategy.generate_signal(last_data)
        # В восходящем тренде должна генерировать сигнал на покупку
        if signal:
            assert signal.signal_type in [SignalType.BUY, SignalType.HOLD]
            assert signal.confidence >= trend_following_strategy._confidence_threshold
    def test_mean_reversion_business_logic(self, mean_reversion_strategy) -> None:
        """Тест бизнес-логики стратегии возврата к среднему."""
        # Создаем данные с отклонением от среднего
        deviation_data = []
        base_price = Decimal("50000")
        # Сначала создаем стабильные данные
        for i in range(40):
            timestamp = datetime.now() + timedelta(minutes=i)
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                open=Price(base_price, Currency.USDT),
                high=Price(base_price + Decimal("10"), Currency.USDT),
                low=Price(base_price - Decimal("10"), Currency.USDT),
                close=Price(base_price, Currency.USDT),
                volume=Volume(Decimal("1000"), Currency.USDT),
                bid=Price(base_price - Decimal("2"), Currency.USDT),
                ask=Price(base_price + Decimal("2"), Currency.USDT),
                bid_volume=Volume(Decimal("500"), Currency.USDT),
                ask_volume=Volume(Decimal("500"), Currency.USDT)
            )
            deviation_data.append(data)
        # Затем добавляем резкое отклонение вверх
        for i in range(10):
            timestamp = datetime.now() + timedelta(minutes=40 + i)
            price = base_price + Decimal("1000")  # Резкий рост
            data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                open=Price(price - Decimal("5"), Currency.USDT),
                high=Price(price + Decimal("10"), Currency.USDT),
                low=Price(price - Decimal("10"), Currency.USDT),
                close=Price(price, Currency.USDT),
                volume=Volume(Decimal("1000"), Currency.USDT),
                bid=Price(price - Decimal("2"), Currency.USDT),
                ask=Price(price + Decimal("2"), Currency.USDT),
                bid_volume=Volume(Decimal("500"), Currency.USDT),
                ask_volume=Volume(Decimal("500"), Currency.USDT)
            )
            deviation_data.append(data)
        mean_reversion_strategy.activate()
        # Анализируем данные с отклонением
        last_data = deviation_data[-1]
        analysis = mean_reversion_strategy.analyze_market(last_data)
        # Проверяем анализ
        assert "confidence_score" in analysis
        assert "market_regime" in analysis
        # Генерируем сигнал
        signal = mean_reversion_strategy.generate_signal(last_data)
        # При отклонении вверх должна генерировать сигнал на продажу
        if signal:
            assert signal.signal_type in [SignalType.SELL, SignalType.HOLD]
            assert signal.confidence >= mean_reversion_strategy._confidence_threshold
    def test_strategy_performance_tracking(self, trend_following_strategy) -> None:
        """Тест отслеживания производительности стратегии."""
        # Получаем начальную производительность
        initial_performance = trend_following_strategy.get_performance()
        assert initial_performance.total_trades == 0
        assert initial_performance.total_pnl.value == Decimal("0")
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
            ask_volume=Volume(Decimal("500"), Currency.USDT)
        )
        trend_following_strategy.activate()
        # Генерируем несколько сигналов
        signals = []
        for _ in range(5):
            signal = trend_following_strategy.generate_signal(test_data)
            if signal:
                signals.append(signal)
        # Проверяем, что сигналы генерируются
        assert len(signals) > 0
        # Проверяем, что производительность обновляется
        updated_performance = trend_following_strategy.get_performance()
        assert updated_performance is not None
    def test_strategy_parameter_updates(self, trend_following_strategy) -> None:
        """Тест обновления параметров стратегии."""
        initial_params = trend_following_strategy.get_parameters()
        assert "short_period" in initial_params
        assert initial_params["short_period"] == 10
        # Обновляем параметры
        new_params = {"short_period": 15, "long_period": 25}
        trend_following_strategy.update_parameters(new_params)
        updated_params = trend_following_strategy.get_parameters()
        assert updated_params["short_period"] == 15
        assert updated_params["long_period"] == 25
        # Проверяем, что другие параметры не изменились
        assert updated_params["rsi_period"] == initial_params["rsi_period"]
class TestStrategyDomainValidation:
    """Тесты доменной валидации стратегий."""
    @pytest.fixture
    def validator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создать валидатор стратегий."""
        return StrategyValidator()
    def test_strategy_config_validation(self, validator) -> None:
        """Тест валидации конфигурации стратегии."""
        # Корректная конфигурация
        valid_config = {
            "name": "Test Strategy",
            "strategy_type": "trend_following",
            "trading_pairs": ["BTC/USDT", "ETH/USDT"],
            "parameters": {
                "short_period": 10,
                "long_period": 20,
                "rsi_period": 14
            },
            "risk_level": "medium",
            "confidence_threshold": 0.6,
            "max_position_size": 0.1,
            "stop_loss": 0.02,
            "take_profit": 0.04
        }
        errors = validator.validate_strategy_config(valid_config)
        assert len(errors) == 0, f"Validation errors: {errors}"
        # Некорректная конфигурация
        invalid_config = {
            "name": "",  # Пустое имя
            "strategy_type": "invalid_type",  # Неверный тип
            "trading_pairs": [],  # Пустой список пар
            "parameters": {
                "short_period": -1,  # Отрицательный период
                "stop_loss": 1.5  # Слишком большой stop loss
            }
        }
        errors = validator.validate_strategy_config(invalid_config)
        assert len(errors) > 0, "Should have validation errors"
        # Проверяем конкретные ошибки
        error_messages = [error.lower() for error in errors]
        assert any("name" in error for error in error_messages)
        assert any("trading pairs" in error for error in error_messages)
        assert any("stop loss" in error for error in error_messages)
    def test_parameter_validation(self, validator) -> None:
        """Тест валидации параметров."""
        # Корректные параметры трендовой стратегии
        valid_trend_params = {
            "short_period": 10,
            "long_period": 20,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70
        }
        errors = validator.validate_parameters(valid_trend_params, "trend_following")
        assert len(errors) == 0, f"Validation errors: {errors}"
        # Некорректные параметры
        invalid_params = {
            "short_period": 0,  # Нулевой период
            "long_period": 5,   # Меньше короткого периода
            "rsi_oversold": 100,  # Некорректное значение
            "rsi_overbought": 0   # Некорректное значение
        }
        errors = validator.validate_parameters(invalid_params, "trend_following")
        assert len(errors) > 0, "Should have validation errors"
    def test_trading_pair_validation(self, validator) -> None:
        """Тест валидации торговых пар."""
        # Корректные пары
        valid_pairs = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        errors = validator.validate_trading_pairs(valid_pairs)
        assert len(errors) == 0, f"Validation errors: {errors}"
        # Некорректные пары
        invalid_pairs = ["", "BTC", "BTC/USDT/", "/USDT", "BTC-USDT"]
        errors = validator.validate_trading_pairs(invalid_pairs)
        assert len(errors) > 0, "Should have validation errors"
if __name__ == "__main__":
    pytest.main([__file__]) 
