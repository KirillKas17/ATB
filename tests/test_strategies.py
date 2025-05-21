from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from strategies.manipulation_strategies import (
    ManipulationStrategy,
    manipulation_strategy_fake_breakout,
    manipulation_strategy_stop_hunt,
)
from strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from strategies.trend_strategy import TrendStrategy
from strategies.volatility_strategy import VolatilityStrategy
from utils.data_loader import load_market_data


# Фикстуры
@pytest.fixture
def mock_market_data():
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
    data = pd.DataFrame(
        {
            "open": np.random.normal(100, 1, 100),
            "high": np.random.normal(101, 1, 100),
            "low": np.random.normal(99, 1, 100),
            "close": np.random.normal(100, 1, 100),
            "volume": np.random.normal(1000, 100, 100),
            "bid": np.random.normal(99.9, 0.1, 100),
            "ask": np.random.normal(100.1, 0.1, 100),
            "bid_volume": np.random.normal(500, 50, 100),
            "ask_volume": np.random.normal(500, 50, 100),
        },
        index=dates,
    )

    # Добавляем необходимые индикаторы
    data["rsi"] = np.random.uniform(30, 70, 100)
    data["macd"] = np.random.normal(0, 1, 100)
    data["macd_signal"] = np.random.normal(0, 1, 100)
    data["ema_fast"] = data["close"].rolling(10).mean()
    data["ema_slow"] = data["close"].rolling(30).mean()
    data["adx"] = np.random.uniform(20, 40, 100)
    data["atr"] = np.random.uniform(0.5, 2, 100)
    data["bb_upper"] = data["close"] + 2 * data["close"].rolling(20).std()
    data["bb_lower"] = data["close"] - 2 * data["close"].rolling(20).std()
    data["keltner_upper"] = data["close"] + 2 * data["atr"]
    data["keltner_lower"] = data["close"] - 2 * data["atr"]
    data["donchian_upper"] = data["high"].rolling(20).max()
    data["donchian_lower"] = data["low"].rolling(20).min()
    data["volatility"] = data["close"].pct_change().rolling(20).std()
    data["volume_ma"] = data["volume"].rolling(20).mean()
    data["correlation"] = data["close"].rolling(20).corr(data["volume"])

    return data


@pytest.fixture
def mock_market_data_with_trend():
    """Фикстура с тестовыми данными с трендом"""
    data = mock_market_data()

    # Создаем восходящий тренд
    trend = np.linspace(0, 10, len(data))
    data["close"] = data["close"] + trend
    data["high"] = data["high"] + trend
    data["low"] = data["low"] + trend
    data["open"] = data["open"] + trend

    # Обновляем индикаторы
    data["ema_fast"] = data["close"].rolling(10).mean()
    data["ema_slow"] = data["close"].rolling(30).mean()
    data["bb_upper"] = data["close"] + 2 * data["close"].rolling(20).std()
    data["bb_lower"] = data["close"] - 2 * data["close"].rolling(20).std()
    data["keltner_upper"] = data["close"] + 2 * data["atr"]
    data["keltner_lower"] = data["close"] - 2 * data["atr"]
    data["donchian_upper"] = data["high"].rolling(20).max()
    data["donchian_lower"] = data["low"].rolling(20).min()
    data["volatility"] = data["close"].pct_change().rolling(20).std()
    data["volume_ma"] = data["volume"].rolling(20).mean()
    data["correlation"] = data["close"].rolling(20).corr(data["volume"])

    return data


@pytest.fixture
def mock_market_data_with_volatility():
    """Фикстура с тестовыми данными с высокой волатильностью"""
    data = mock_market_data()

    # Увеличиваем волатильность
    volatility = np.random.normal(0, 2, len(data))
    data["close"] = data["close"] * (1 + volatility)
    data["high"] = data["high"] * (1 + volatility)
    data["low"] = data["low"] * (1 + volatility)
    data["open"] = data["open"] * (1 + volatility)

    # Обновляем индикаторы
    data["ema_fast"] = data["close"].rolling(10).mean()
    data["ema_slow"] = data["close"].rolling(30).mean()
    data["bb_upper"] = data["close"] + 2 * data["close"].rolling(20).std()
    data["bb_lower"] = data["close"] - 2 * data["close"].rolling(20).std()
    data["keltner_upper"] = data["close"] + 2 * data["atr"]
    data["keltner_lower"] = data["close"] - 2 * data["atr"]
    data["donchian_upper"] = data["high"].rolling(20).max()
    data["donchian_lower"] = data["low"].rolling(20).min()
    data["volatility"] = data["close"].pct_change().rolling(20).std()
    data["volume_ma"] = data["volume"].rolling(20).mean()
    data["correlation"] = data["close"].rolling(20).corr(data["volume"])

    return data


@pytest.fixture
def mock_market_data_with_manipulation():
    """Фикстура с тестовыми данными с манипуляциями"""
    data = mock_market_data()

    # Создаем манипуляции
    manipulation = np.zeros(len(data))
    manipulation[30:40] = 5  # Резкий рост
    manipulation[60:70] = -5  # Резкое падение
    data["close"] = data["close"] + manipulation
    data["high"] = data["high"] + manipulation
    data["low"] = data["low"] + manipulation
    data["open"] = data["open"] + manipulation

    # Обновляем индикаторы
    data["ema_fast"] = data["close"].rolling(10).mean()
    data["ema_slow"] = data["close"].rolling(30).mean()
    data["bb_upper"] = data["close"] + 2 * data["close"].rolling(20).std()
    data["bb_lower"] = data["close"] - 2 * data["close"].rolling(20).std()
    data["keltner_upper"] = data["close"] + 2 * data["atr"]
    data["keltner_lower"] = data["close"] - 2 * data["atr"]
    data["donchian_upper"] = data["high"].rolling(20).max()
    data["donchian_lower"] = data["low"].rolling(20).min()
    data["volatility"] = data["close"].pct_change().rolling(20).std()
    data["volume_ma"] = data["volume"].rolling(20).mean()
    data["correlation"] = data["close"].rolling(20).corr(data["volume"])

    return data


@pytest.fixture
def trend_strategy():
    """Фикстура для тестирования TrendStrategy"""
    config = {
        "ma_fast": 10,
        "ma_slow": 20,
        "rsi_period": 14,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "use_volume": True,
        "use_volatility": True,
        "use_correlation": True,
        "log_dir": "logs/trend",
    }
    return TrendStrategy(config)


@pytest.fixture
def volatility_strategy():
    """Фикстура для тестирования VolatilityStrategy"""
    config = {
        "atr_period": 14,
        "atr_multiplier": 2,
        "min_volatility": 0.01,
        "use_volume": True,
        "use_correlation": True,
        "use_regime": True,
        "log_dir": "logs/volatility",
    }
    return VolatilityStrategy(config)


@pytest.fixture
def manipulation_strategy():
    """Фикстура для тестирования ManipulationStrategy"""
    config = {"volume_threshold": 1.5, "imbalance_threshold": 0.7, "log_dir": "logs/manipulation"}
    return ManipulationStrategy(config)


@pytest.fixture
def regime_adaptive_strategy():
    """Фикстура для тестирования RegimeAdaptiveStrategy"""
    config = {
        "min_confidence": 0.7,
        "max_drawdown": 0.1,
        "adaptation_rate": 0.1,
        "use_volume": True,
        "use_volatility": True,
        "use_correlation": True,
        "use_regime": True,
        "log_dir": "logs/regime",
    }
    return RegimeAdaptiveStrategy(config)


# Параметризованные тесты
@pytest.mark.parametrize(
    "strategy_name,strategy_fixture,data_fixture",
    [
        ("trend", "trend_strategy", "mock_market_data_with_trend"),
        ("volatility", "volatility_strategy", "mock_market_data_with_volatility"),
        ("manipulation", "manipulation_strategy", "mock_market_data_with_manipulation"),
        ("regime_adaptive", "regime_adaptive_strategy", "mock_market_data"),
    ],
)
def test_strategy_generate_signal(strategy_name, strategy_fixture, data_fixture, request):
    """Параметризованный тест генерации сигнала"""
    strategy = request.getfixturevalue(strategy_fixture)
    data = request.getfixturevalue(data_fixture)

    # Проверяем наличие необходимых колонок
    required_columns = ["open", "high", "low", "close", "volume"]
    assert all(col in data.columns for col in required_columns), "Missing required columns"

    # Генерируем сигнал
    signal = strategy.generate_signal(data)

    # Проверяем сигнал
    assert signal is not None, "Signal should not be None"
    assert isinstance(signal, Signal), "Signal should be an instance of Signal"
    assert signal.direction in ["long", "short"], "Invalid signal direction"
    assert 0 <= signal.confidence <= 1, "Invalid confidence value"
    assert signal.stop_loss is not None and signal.stop_loss > 0, "Invalid stop loss"
    assert signal.take_profit is not None and signal.take_profit > 0, "Invalid take profit"
    assert signal.volume is not None and signal.volume > 0, "Invalid volume"
    assert isinstance(signal.timestamp, datetime), "Invalid timestamp"
    assert isinstance(signal.metadata, dict), "Invalid metadata"


# Тесты для TrendStrategy
class TestTrendStrategy:
    @pytest.mark.parametrize(
        "data_fixture",
        ["mock_market_data", "mock_market_data_with_trend", "mock_market_data_with_volatility"],
    )
    def test_calculate_indicators(self, trend_strategy, data_fixture, request):
        """Тест расчета индикаторов с разными наборами данных"""
        data = request.getfixturevalue(data_fixture)

        # Проверяем наличие необходимых колонок
        required_columns = ["open", "high", "low", "close", "volume"]
        assert all(col in data.columns for col in required_columns), "Missing required columns"

        indicators = trend_strategy.calculate_indicators(data)

        assert isinstance(indicators, dict), "Indicators should be a dictionary"
        assert "ma_fast" in indicators, "Missing ma_fast indicator"
        assert "ma_slow" in indicators, "Missing ma_slow indicator"
        assert "rsi" in indicators, "Missing rsi indicator"
        assert "volume_ma" in indicators, "Missing volume_ma indicator"
        assert "volatility" in indicators, "Missing volatility indicator"
        assert "correlation" in indicators, "Missing correlation indicator"

        # Проверка значений
        assert all(not np.isnan(x) for x in indicators["ma_fast"]), "Invalid ma_fast values"
        assert all(not np.isnan(x) for x in indicators["ma_slow"]), "Invalid ma_slow values"
        assert all(not np.isnan(x) for x in indicators["rsi"]), "Invalid rsi values"
        assert all(0 <= x <= 100 for x in indicators["rsi"]), "Invalid rsi range"

    def test_validate_signal(self, trend_strategy, mock_market_data):
        """Тест валидации сигнала"""
        signal = {"action": "buy", "confidence": 0.8, "stop_loss": 95, "take_profit": 105}

        is_valid = trend_strategy.validate_signal(signal, mock_market_data)
        assert isinstance(is_valid, bool)

    def test_calculate_position_size(self, trend_strategy, mock_market_data):
        """Тест расчета размера позиции"""
        signal = {"action": "buy", "confidence": 0.8, "stop_loss": 95, "take_profit": 105}

        position_size = trend_strategy.calculate_position_size(signal, mock_market_data)
        assert isinstance(position_size, float)
        assert position_size > 0
        assert position_size <= trend_strategy.max_position_size


# Тесты для VolatilityStrategy
class TestVolatilityStrategy:
    @pytest.mark.parametrize(
        "data_fixture", ["mock_market_data", "mock_market_data_with_volatility"]
    )
    def test_calculate_volatility(self, volatility_strategy, data_fixture, request):
        """Тест расчета волатильности с разными наборами данных"""
        data = request.getfixturevalue(data_fixture)
        volatility = volatility_strategy.calculate_volatility(data)

        assert isinstance(volatility, float)
        assert volatility >= 0

    def test_detect_volatility_regime(self, volatility_strategy, mock_market_data_with_volatility):
        """Тест определения режима волатильности"""
        regime = volatility_strategy.detect_volatility_regime(mock_market_data_with_volatility)

        assert isinstance(regime, str)
        assert regime in ["high", "low", "normal"]

    def test_calculate_risk_metrics(self, volatility_strategy, mock_market_data):
        """Тест расчета метрик риска"""
        metrics = volatility_strategy.calculate_risk_metrics(mock_market_data)

        assert isinstance(metrics, dict)
        assert "var" in metrics
        assert "cvar" in metrics
        assert "volatility" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


# Тесты для ManipulationStrategy
class TestManipulationStrategy:
    @pytest.mark.parametrize(
        "data_fixture", ["mock_market_data", "mock_market_data_with_manipulation"]
    )
    def test_detect_manipulation(self, manipulation_strategy, data_fixture, request):
        """Тест обнаружения манипуляции с разными наборами данных"""
        data = request.getfixturevalue(data_fixture)
        manipulation = manipulation_strategy.detect_manipulation(data)

        assert isinstance(manipulation, bool)

    def test_analyze_volume_profile(
        self, manipulation_strategy, mock_market_data_with_manipulation
    ):
        """Тест анализа профиля объема"""
        profile = manipulation_strategy.analyze_volume_profile(mock_market_data_with_manipulation)

        assert isinstance(profile, dict)
        assert "volume_imbalance" in profile
        assert "volume_trend" in profile
        assert "volume_anomalies" in profile
        assert "volume_delta" in profile

    def test_detect_pump_and_dump(self, manipulation_strategy, mock_market_data_with_manipulation):
        """Тест обнаружения памп-энд-дампа"""
        is_pump_and_dump = manipulation_strategy.detect_pump_and_dump(
            mock_market_data_with_manipulation
        )

        assert isinstance(is_pump_and_dump, bool)


# Тесты для RegimeAdaptiveStrategy
class TestRegimeAdaptiveStrategy:
    @pytest.mark.parametrize(
        "data_fixture",
        [
            "mock_market_data",
            "mock_market_data_with_trend",
            "mock_market_data_with_volatility",
            "mock_market_data_with_manipulation",
        ],
    )
    def test_detect_regime(self, regime_adaptive_strategy, data_fixture, request):
        """Тест определения режима с разными наборами данных"""
        data = request.getfixturevalue(data_fixture)
        regime = regime_adaptive_strategy.detect_regime(data)

        assert isinstance(regime, str)
        assert regime in ["trend", "volatility", "manipulation", "sideways"]

    @pytest.mark.parametrize(
        "current_regime,new_regime",
        [
            ("trend", "volatility"),
            ("volatility", "manipulation"),
            ("manipulation", "sideways"),
            ("sideways", "trend"),
        ],
    )
    def test_adapt_to_regime(
        self, regime_adaptive_strategy, mock_market_data, current_regime, new_regime
    ):
        """Тест адаптации к разным режимам"""
        adaptation = regime_adaptive_strategy.adapt_to_regime(
            current_regime=current_regime, new_regime=new_regime, market_data=mock_market_data
        )

        assert isinstance(adaptation, dict)
        assert "parameters" in adaptation
        assert "confidence" in adaptation
        assert 0 <= adaptation["confidence"] <= 1

    def test_validate_adaptation(self, regime_adaptive_strategy, mock_market_data):
        """Тест валидации адаптации"""
        adaptation = {
            "parameters": {"ma_fast": 5, "ma_slow": 10, "rsi_period": 14, "atr_period": 14},
            "confidence": 0.8,
        }

        is_valid = regime_adaptive_strategy.validate_adaptation(
            adaptation=adaptation, market_data=mock_market_data
        )

        assert isinstance(is_valid, bool)

    def test_calculate_regime_metrics(self, regime_adaptive_strategy, mock_market_data):
        """Тест расчета метрик режима"""
        metrics = regime_adaptive_strategy.calculate_regime_metrics(mock_market_data)

        assert isinstance(metrics, dict)
        assert "trend_strength" in metrics
        assert "volatility_level" in metrics
        assert "manipulation_probability" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
