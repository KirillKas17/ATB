from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from core.signal_processor import MarketContext, ProcessedSignal, Signal, SignalProcessor


@pytest.fixture
def market_context():
    """Фикстура для MarketContext"""
    return MarketContext(
        volatility=0.02,
        trend="up",
        volume=1000.0,
        indicators={"rsi": 65.0, "macd": 0.5},
        market_regime="bull",
        liquidity=1000000.0,
        momentum=0.05,
        sentiment=0.7,
        support_levels=[95.0, 90.0],
        resistance_levels=[105.0, 110.0],
        market_depth={"bid": 500.0, "ask": 500.0},
        correlation_matrix={"BTC/USD": {"ETH/USD": 0.8}},
        market_impact=0.001,
        volume_profile={100.0: 1000.0, 101.0: 800.0},
    )


@pytest.fixture
def signal_processor():
    """Фикстура для SignalProcessor"""
    config = {
        "min_confidence": 0.6,
        "max_position_size": 1.0,
        "risk_per_trade": 0.02,
        "correlation_threshold": 0.7,
        "impact_threshold": 0.1,
        "priority_weights": {
            "strength": 0.3,
            "confidence": 0.3,
            "risk_reward": 0.2,
            "market_regime": 0.1,
            "liquidity": 0.1,
        },
    }
    return SignalProcessor(config)


@pytest.fixture
def buy_signal():
    """Фикстура для сигнала на покупку"""
    return Signal(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        direction="buy",
        strength=0.8,
        source="test",
        confidence=0.7,
        timeframe="1h",
        entry_price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        risk_reward_ratio=3.0,
        expected_duration=10,
        priority=1,
        metadata={"strategy": "test"},
    )


@pytest.fixture
def sell_signal():
    """Фикстура для сигнала на продажу"""
    return Signal(
        timestamp=datetime.now(),
        symbol="BTC/USD",
        direction="sell",
        strength=-0.8,
        source="test",
        confidence=0.7,
        timeframe="1h",
        entry_price=100.0,
        stop_loss=105.0,
        take_profit=90.0,
        risk_reward_ratio=3.0,
        expected_duration=10,
        priority=1,
        metadata={"strategy": "test"},
    )


class TestSignalProcessor:
    """Тесты для SignalProcessor"""

    def test_initialization(self, signal_processor):
        """Тест инициализации"""
        assert signal_processor.min_confidence == 0.6
        assert signal_processor.max_position_size == 1.0
        assert signal_processor.risk_per_trade == 0.02
        assert signal_processor.correlation_threshold == 0.7
        assert signal_processor.impact_threshold == 0.1
        assert signal_processor.priority_weights["strength"] == 0.3

    def test_process_signal(self, signal_processor, buy_signal, market_context):
        """Тест обработки сигнала"""
        result = signal_processor.process_signal(buy_signal, market_context)

        assert result is not None
        assert isinstance(result, ProcessedSignal)
        assert result.signal == buy_signal
        assert result.context == market_context
        assert result.confidence >= signal_processor.min_confidence
        assert result.position_size <= signal_processor.max_position_size
        assert "var_95" in result.risk_metrics
        assert "expected_drawdown" in result.risk_metrics
        assert "sharpe_ratio" in result.risk_metrics
        assert result.execution_priority > 0
        assert result.expected_impact >= 0
        assert result.market_impact >= 0

    def test_calculate_confidence(self, signal_processor, buy_signal, market_context):
        """Тест расчета уверенности"""
        confidence = signal_processor._calculate_confidence(buy_signal, market_context)

        assert 0 <= confidence <= 1
        assert confidence >= signal_processor.min_confidence

    def test_calculate_position_size(self, signal_processor, buy_signal, market_context):
        """Тест расчета размера позиции"""
        confidence = 0.8
        position_size = signal_processor._calculate_position_size(
            buy_signal, market_context, confidence
        )

        assert 0 <= position_size <= signal_processor.max_position_size

    def test_calculate_risk_metrics(self, signal_processor, buy_signal, market_context):
        """Тест расчета метрик риска"""
        position_size = 0.5
        risk_metrics = signal_processor._calculate_risk_metrics(
            buy_signal, market_context, position_size
        )

        assert "var_95" in risk_metrics
        assert "expected_drawdown" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "sortino_ratio" in risk_metrics
        assert "max_position_risk" in risk_metrics
        assert "correlation_risk" in risk_metrics

    def test_calculate_execution_priority(self, signal_processor, buy_signal, market_context):
        """Тест расчета приоритета исполнения"""
        confidence = 0.8
        priority = signal_processor._calculate_execution_priority(
            buy_signal, market_context, confidence
        )

        assert priority > 0

    def test_calculate_expected_impact(self, signal_processor, buy_signal, market_context):
        """Тест расчета ожидаемого влияния"""
        position_size = 0.5
        impact = signal_processor._calculate_expected_impact(
            buy_signal, market_context, position_size
        )

        assert impact >= 0

    def test_calculate_correlation_impact(self, signal_processor, buy_signal, market_context):
        """Тест расчета влияния на коррелированные инструменты"""
        impact = signal_processor._calculate_correlation_impact(buy_signal, market_context)

        assert isinstance(impact, dict)
        if impact:
            assert all(0 <= v <= 1 for v in impact.values())

    def test_calculate_market_impact(self, signal_processor, buy_signal, market_context):
        """Тест расчета влияния на рынок"""
        position_size = 0.5
        impact = signal_processor._calculate_market_impact(
            buy_signal, market_context, position_size
        )

        assert impact >= 0

    def test_regime_factor(self, signal_processor):
        """Тест расчета фактора режима"""
        # Тест для бычьего рынка
        assert signal_processor._get_regime_factor("buy", "bull") == 1.0
        assert signal_processor._get_regime_factor("sell", "bull") == 0.1

        # Тест для медвежьего рынка
        assert signal_processor._get_regime_factor("sell", "bear") == 1.0
        assert signal_processor._get_regime_factor("buy", "bear") == 0.1

        # Тест для бокового рынка
        assert signal_processor._get_regime_factor("buy", "sideways") == 0.5
        assert signal_processor._get_regime_factor("sell", "sideways") == 0.5

        # Тест для волатильного рынка
        assert signal_processor._get_regime_factor("buy", "volatile") == 0.3
        assert signal_processor._get_regime_factor("sell", "volatile") == 0.3

    def test_serialization(self, signal_processor, buy_signal, market_context):
        """Тест сериализации/десериализации"""
        result = signal_processor.process_signal(buy_signal, market_context)

        # Сериализация
        result_dict = result.to_dict()

        # Проверка структуры
        assert "signal" in result_dict
        assert "context" in result_dict
        assert "confidence" in result_dict
        assert "position_size" in result_dict
        assert "risk_metrics" in result_dict
        assert "execution_priority" in result_dict
        assert "expected_impact" in result_dict
        assert "correlation_impact" in result_dict
        assert "market_impact" in result_dict

        # Десериализация
        new_result = ProcessedSignal.from_dict(result_dict)

        # Проверка равенства
        assert new_result.signal.symbol == result.signal.symbol
        assert new_result.signal.direction == result.signal.direction
        assert new_result.confidence == result.confidence
        assert new_result.position_size == result.position_size
        assert new_result.execution_priority == result.execution_priority
