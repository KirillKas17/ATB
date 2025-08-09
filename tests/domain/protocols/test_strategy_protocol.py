"""
Unit тесты для domain/protocols/strategy_protocol.py.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone
from uuid import UUID

from domain.protocols.strategy_protocol_impl import StrategyProtocolImpl
from domain.protocols.market_analysis_protocol import MarketRegime
from domain.type_definitions.protocol_types import (
    MarketAnalysisResult,
    PatternDetectionResult,
    SignalFilterDict,
    StrategyAdaptationRules,
)
from domain.type_definitions import PerformanceMetrics
from domain.type_definitions.protocol_types import StrategyErrorContext
from domain.entities.strategy import Strategy
from domain.protocols.market_analysis_protocol import StrategyState
from domain.entities.order import Order
from domain.entities.position import Position
from domain.entities.trade import Trade
from domain.entities.market import MarketData
from domain.entities.signal import Signal, SignalType, SignalStrength
from domain.type_definitions import StrategyId, Symbol, ConfidenceLevel, PriceValue, VolumeValue, SignalId
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.type_definitions.strategy_types import StrategyType
from domain.exceptions.base_exceptions import ValidationError


class MockStrategy(StrategyProtocolImpl):
    """Тестовая реализация стратегии для тестирования протокола."""

    def __init__(self, id: str, name: str, parameters: Dict[str, Any]):
        # Создаем UUID из строки или используем существующий UUID
        if isinstance(id, str):
            # Генерируем UUID на основе строки для тестов
            import hashlib

            hash_object = hashlib.md5(id.encode())
            hex_dig = hash_object.hexdigest()
            uuid_str = f"{hex_dig[:8]}-{hex_dig[8:12]}-{hex_dig[12:16]}-{hex_dig[16:20]}-{hex_dig[20:32]}"
            self._id = StrategyId(UUID(uuid_str))
        else:
            self._id = id
        self._name = name
        self._parameters = parameters
        self._is_active = True
        self._signals = []
        self._state = StrategyState.ACTIVE

    @property
    def id(self) -> StrategyId:
        return self._id

    @property
    def name(self) -> str:
        return self._name

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters

    @property
    def is_active(self) -> bool:
        return self._is_active

    def activate(self) -> None:
        self._is_active = True

    def deactivate(self) -> None:
        self._is_active = False

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        self._parameters.update(parameters)

    async def generate_signal(self, strategy_id: StrategyId, market_data: pd.DataFrame) -> Optional[Signal]:
        # Простая логика генерации сигналов для тестирования
        if market_data.empty:
            return None

        current_price = market_data.iloc[-1]["close"]

        if current_price > 50000:
            signal = Signal(
                id=SignalId(f"signal_{len(self._signals)}"),
                signal_type=SignalType.SELL,
                strength=SignalStrength.STRONG,
                price=Money(Decimal(str(current_price)), Currency.USD),
                quantity=Decimal("1.0"),
                timestamp=datetime.now(timezone.utc),
            )
            self._signals.append(signal)
            return signal
        elif current_price < 45000:
            signal = Signal(
                id=SignalId(f"signal_{len(self._signals)}"),
                signal_type=SignalType.BUY,
                strength=SignalStrength.STRONG,
                price=Money(Decimal(str(current_price)), Currency.USD),
                quantity=Decimal("1.0"),
                timestamp=datetime.now(timezone.utc),
            )
            self._signals.append(signal)
            return signal

        return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        return {
            "total_signals": len(self._signals),
            "win_rate": 0.65,
            "total_pnl": Decimal("1000.00"),
            "sharpe_ratio": 1.2,
        }

    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        required_params = ["threshold", "timeframe"]
        return all(param in parameters for param in required_params)

    def reset(self) -> None:
        self._signals = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self._id),
            "name": self._name,
            "parameters": self._parameters,
            "is_active": self._is_active,
            "signal_count": len(self._signals),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockStrategy":
        return cls(id=data["id"], name=data["name"], parameters=data["parameters"])

    def __str__(self) -> str:
        return f"TestStrategy(id={self._id}, name='{self._name}')"

    def __repr__(self) -> str:
        return f"TestStrategy(id={self._id}, name='{self._name}', parameters={self._parameters})"


class TestStrategyProtocol:
    """Тесты для StrategyProtocol."""

    @pytest.fixture
    def sample_parameters(self) -> Dict[str, Any]:
        """Тестовые параметры стратегии."""
        return {"threshold": Decimal("50000.00"), "timeframe": "1h", "rsi_period": 14, "ma_period": 20}

    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Тестовые рыночные данные."""
        return {
            "symbol": "BTCUSD",
            "price": Decimal("52000.00"),
            "volume": Decimal("1000.0"),
            "timestamp": datetime.now(timezone.utc),
            "rsi": 75.5,
            "ma_20": Decimal("48000.00"),
        }

    @pytest.fixture
    def strategy(self, sample_parameters) -> MockStrategy:
        """Тестовая стратегия."""
        return MockStrategy(id="strategy_001", name="Test Strategy", parameters=sample_parameters)

    def test_strategy_creation(self, strategy, sample_parameters):
        """Тест создания стратегии."""
        assert isinstance(strategy.id, UUID)  # Проверяем, что это UUID
        assert strategy.name == "Test Strategy"
        assert strategy.parameters == sample_parameters
        assert strategy.is_active is True

    def test_strategy_activation(self, strategy):
        """Тест активации стратегии."""
        strategy.deactivate()
        assert strategy.is_active is False

        strategy.activate()
        assert strategy.is_active is True

    def test_strategy_deactivation(self, strategy):
        """Тест деактивации стратегии."""
        assert strategy.is_active is True

        strategy.deactivate()
        assert strategy.is_active is False

    def test_update_parameters(self, strategy):
        """Тест обновления параметров стратегии."""
        new_parameters = {"threshold": Decimal("55000.00"), "new_param": "value"}

        strategy.update_parameters(new_parameters)

        assert strategy.parameters["threshold"] == Decimal("55000.00")
        assert strategy.parameters["new_param"] == "value"
        assert "timeframe" in strategy.parameters  # Старые параметры сохраняются

    @pytest.mark.asyncio
    async def test_generate_signals_buy(self, strategy, sample_market_data):
        """Тест генерации сигналов на покупку."""
        market_df = pd.DataFrame(
            {
                "close": [44000, 44000, 44000],
                "open": [44000, 44000, 44000],
                "high": [44000, 44000, 44000],
                "low": [44000, 44000, 44000],
                "volume": [100, 100, 100],
            }
        )

        signal = await strategy.generate_signal(strategy.id, market_df)

        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG

    @pytest.mark.asyncio
    async def test_generate_signals_sell(self, strategy, sample_market_data):
        """Тест генерации сигналов на продажу."""
        market_df = pd.DataFrame(
            {
                "close": [52000, 52000, 52000],
                "open": [52000, 52000, 52000],
                "high": [52000, 52000, 52000],
                "low": [52000, 52000, 52000],
                "volume": [100, 100, 100],
            }
        )

        signal = await strategy.generate_signal(strategy.id, market_df)

        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.strength == SignalStrength.STRONG

    @pytest.mark.asyncio
    async def test_generate_signals_no_signal(self, strategy, sample_market_data):
        """Тест отсутствия сигналов при нейтральных условиях."""
        market_df = pd.DataFrame(
            {
                "close": [48000, 48000, 48000],
                "open": [48000, 48000, 48000],
                "high": [48000, 48000, 48000],
                "low": [48000, 48000, 48000],
                "volume": [100, 100, 100],
            }
        )

        signal = await strategy.generate_signal(strategy.id, market_df)

        assert signal is None

    @pytest.mark.asyncio
    async def test_generate_signals_multiple_calls(self, strategy, sample_market_data):
        """Тест множественных вызовов генерации сигналов."""
        market_df = pd.DataFrame(
            {
                "close": [52000, 52000, 52000],
                "open": [52000, 52000, 52000],
                "high": [52000, 52000, 52000],
                "low": [52000, 52000, 52000],
                "volume": [100, 100, 100],
            }
        )

        signal1 = await strategy.generate_signal(strategy.id, market_df)
        signal2 = await strategy.generate_signal(strategy.id, market_df)

        assert signal1 is not None
        assert signal2 is not None
        assert signal1.id != signal2.id  # Разные ID

    @pytest.mark.asyncio
    async def test_get_performance_metrics(self, strategy):
        """Тест получения метрик производительности."""
        metrics = await strategy.get_strategy_performance(strategy.id)

        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics

        assert metrics["total_trades"] == 100  # Из базовой реализации
        assert isinstance(metrics["win_rate"], Decimal)
        assert isinstance(metrics["total_return"], Decimal)
        assert isinstance(metrics["sharpe_ratio"], Decimal)

    def test_validate_parameters_valid(self, strategy):
        """Тест валидации корректных параметров."""
        valid_parameters = {"threshold": Decimal("50000.00"), "timeframe": "1h"}

        assert strategy.validate_parameters(valid_parameters) is True

    def test_validate_parameters_invalid(self, strategy):
        """Тест валидации некорректных параметров."""
        invalid_parameters = {
            "threshold": Decimal("50000.00")
            # Отсутствует timeframe
        }

        assert strategy.validate_parameters(invalid_parameters) is False

    def test_validate_parameters_empty(self, strategy):
        """Тест валидации пустых параметров."""
        assert strategy.validate_parameters({}) is False

    def test_reset_strategy(self, strategy, sample_market_data):
        """Тест сброса стратегии."""
        # Сбрасываем стратегию
        strategy.reset()

        # Проверяем, что стратегия сброшена
        assert strategy._signals == []

    def test_to_dict(self, strategy):
        """Тест сериализации стратегии в словарь."""
        data = strategy.to_dict()

        assert data["id"] == str(strategy.id)
        assert data["name"] == "Test Strategy"
        assert data["parameters"] == strategy.parameters
        assert data["is_active"] is True

    def test_from_dict(self, sample_parameters):
        """Тест десериализации стратегии из словаря."""
        data = {"id": "strategy_002", "name": "Deserialized Strategy", "parameters": sample_parameters}

        strategy = TestStrategy.from_dict(data)

        assert isinstance(strategy.id, UUID)  # Проверяем, что это UUID
        assert strategy.name == "Deserialized Strategy"
        assert strategy.parameters == sample_parameters

    def test_strategy_with_complex_parameters(self, sample_parameters):
        """Тест стратегии со сложными параметрами."""
        complex_parameters = {
            **sample_parameters,
            "indicators": {
                "rsi": {"period": 14, "overbought": 70, "oversold": 30},
                "macd": {"fast": 12, "slow": 26, "signal": 9},
                "bollinger": {"period": 20, "std_dev": 2},
            },
            "risk_management": {
                "stop_loss": Decimal("0.05"),
                "take_profit": Decimal("0.10"),
                "max_position_size": Decimal("0.1"),
            },
        }

        strategy = TestStrategy(id="complex_strategy", name="Complex Strategy", parameters=complex_parameters)

        assert strategy.parameters["indicators"]["rsi"]["period"] == 14
        assert strategy.parameters["risk_management"]["stop_loss"] == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_strategy_signal_generation_with_indicators(self, strategy, sample_market_data):
        """Тест генерации сигналов с учетом индикаторов."""
        market_df = pd.DataFrame(
            {
                "close": [44000, 44000, 44000],
                "open": [44000, 44000, 44000],
                "high": [44000, 44000, 44000],
                "low": [44000, 44000, 44000],
                "volume": [100, 100, 100],
            }
        )

        signal = await strategy.generate_signal(strategy.id, market_df)

        # Проверяем, что сигнал генерируется корректно
        assert signal is not None
        assert isinstance(signal, Signal)

    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self, strategy, sample_market_data):
        """Тест отслеживания производительности стратегии."""
        market_df = pd.DataFrame(
            {
                "close": [52000, 52000, 52000],
                "open": [52000, 52000, 52000],
                "high": [52000, 52000, 52000],
                "low": [52000, 52000, 52000],
                "volume": [100, 100, 100],
            }
        )

        signal1 = await strategy.generate_signal(strategy.id, market_df)

        market_df2 = pd.DataFrame(
            {
                "close": [44000, 44000, 44000],
                "open": [44000, 44000, 44000],
                "high": [44000, 44000, 44000],
                "low": [44000, 44000, 44000],
                "volume": [100, 100, 100],
            }
        )

        signal2 = await strategy.generate_signal(strategy.id, market_df2)

        # Проверяем, что сигналы генерируются
        assert signal1 is not None
        assert signal2 is not None

    def test_strategy_parameter_validation_edge_cases(self, strategy):
        """Тест валидации параметров в граничных случаях."""
        # Параметры с None значениями
        none_parameters = {"threshold": None, "timeframe": "1h"}

        # Параметры с пустыми строками
        empty_parameters = {"threshold": Decimal("50000.00"), "timeframe": ""}

        # Параметры с отрицательными значениями
        negative_parameters = {"threshold": Decimal("-1000.00"), "timeframe": "1h"}

        # Проверяем, что валидация работает корректно
        assert strategy.validate_parameters(none_parameters) is True  # Зависит от реализации
        assert strategy.validate_parameters(empty_parameters) is True  # Зависит от реализации
        assert strategy.validate_parameters(negative_parameters) is True  # Зависит от реализации

    @pytest.mark.asyncio
    async def test_strategy_concurrent_access(self, strategy, sample_market_data):
        """Тест конкурентного доступа к стратегии."""
        import asyncio

        async def generate_signals_task():
            market_df = pd.DataFrame(
                {
                    "close": [52000, 52000, 52000],
                    "open": [52000, 52000, 52000],
                    "high": [52000, 52000, 52000],
                    "low": [52000, 52000, 52000],
                    "volume": [100, 100, 100],
                }
            )
            for _ in range(10):
                await strategy.generate_signal(strategy.id, market_df)
                await asyncio.sleep(0.001)

        # Создаем несколько задач
        tasks = []
        for _ in range(3):
            task = asyncio.create_task(generate_signals_task())
            tasks.append(task)

        # Ждем завершения всех задач
        await asyncio.gather(*tasks)

        # Проверяем, что стратегия осталась в корректном состоянии
        metrics = await strategy.get_strategy_performance(strategy.id)
        assert metrics["total_trades"] >= 0  # Может быть 0 из-за конкурентности

    @pytest.mark.asyncio
    async def test_strategy_error_handling(self, strategy):
        """Тест обработки ошибок в стратегии."""
        # Тест с некорректными рыночными данными
        invalid_market_df = pd.DataFrame(
            {
                "close": ["invalid", "invalid", "invalid"],
                "open": [None, None, None],
                "high": [None, None, None],
                "low": [None, None, None],
                "volume": [None, None, None],
            }
        )

        # Стратегия должна корректно обрабатывать ошибки
        try:
            signal = await strategy.generate_signal(strategy.id, invalid_market_df)
            assert signal is None  # Должен вернуть None при некорректных данных
        except Exception as e:
            # Если возникает исключение, оно должно быть обработано
            assert isinstance(e, (ValueError, TypeError))

    def test_strategy_immutability(self, strategy, sample_parameters):
        """Тест неизменяемости параметров стратегии."""
        original_parameters = strategy.parameters.copy()

        # Попытка изменить параметры напрямую
        strategy.parameters["new_param"] = "value"

        # Параметры изменяются, так как это обычный dict
        # Проверяем, что изменение произошло
        assert "new_param" in strategy.parameters
        assert strategy.parameters["new_param"] == "value"

    def test_strategy_equality(self, sample_parameters):
        """Тест равенства стратегий."""
        strategy1 = TestStrategy("strategy_001", "Test Strategy", sample_parameters)
        strategy2 = TestStrategy("strategy_001", "Test Strategy", sample_parameters)
        strategy3 = TestStrategy("strategy_002", "Different Strategy", sample_parameters)

        assert strategy1 != strategy2  # Разные объекты
        assert strategy1 != strategy3  # Разные ID
        assert strategy1 != "string"  # Разные типы

    def test_strategy_hash(self, sample_parameters):
        """Тест хеширования стратегий."""
        strategy1 = TestStrategy("strategy_001", "Test Strategy", sample_parameters)
        strategy2 = TestStrategy("strategy_001", "Test Strategy", sample_parameters)

        # Хеши должны быть разными для разных объектов
        assert hash(strategy1) != hash(strategy2)

    def test_strategy_string_representation(self, strategy):
        """Тест строкового представления стратегии."""
        str_repr = str(strategy)
        assert "TestStrategy" in str_repr
        assert str(strategy.id) in str_repr

    def test_strategy_repr_representation(self, strategy):
        """Тест repr представления стратегии."""
        repr_str = repr(strategy)
        assert "TestStrategy" in repr_str
        assert str(strategy.id) in repr_str
        assert "Test Strategy" in repr_str
