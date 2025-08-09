"""
Unit тесты для domain/value_objects/signal.py.
"""

import pytest
from typing import Dict, Any
from decimal import Decimal
from datetime import datetime, timezone

from domain.value_objects.signal import Signal, SignalType, SignalStrength, SignalDirection
from domain.exceptions.base_exceptions import ValidationError


class TestSignal:
    """Тесты для Signal."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": "signal_001",
            "type": SignalType.TECHNICAL,
            "direction": SignalDirection.BUY,
            "strength": SignalStrength.STRONG,
            "price": Decimal("50000.00"),
            "volume": Decimal("100.0"),
            "timestamp": datetime.now(timezone.utc),
            "confidence": Decimal("0.85"),
            "source": "RSI_INDICATOR",
            "metadata": {"rsi_value": 30, "timeframe": "1h"},
        }

    def test_creation(self, sample_data):
        """Тест создания Signal."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
            confidence=sample_data["confidence"],
            source=sample_data["source"],
            metadata=sample_data["metadata"],
        )

        assert signal.id == sample_data["id"]
        assert signal.type == sample_data["type"]
        assert signal.direction == sample_data["direction"]
        assert signal.strength == sample_data["strength"]
        assert signal.price == sample_data["price"]
        assert signal.volume == sample_data["volume"]
        assert signal.timestamp == sample_data["timestamp"]
        assert signal.confidence == sample_data["confidence"]
        assert signal.source == sample_data["source"]
        assert signal.metadata == sample_data["metadata"]

    def test_validation_empty_id(self, sample_data):
        """Тест валидации пустого ID."""
        with pytest.raises(ValidationError, match="Signal ID cannot be empty"):
            Signal(
                id="",
                type=sample_data["type"],
                direction=sample_data["direction"],
                strength=sample_data["strength"],
                price=sample_data["price"],
                volume=sample_data["volume"],
                timestamp=sample_data["timestamp"],
            )

    def test_validation_negative_price(self, sample_data):
        """Тест валидации отрицательной цены."""
        with pytest.raises(ValidationError, match="Price cannot be negative"):
            Signal(
                id=sample_data["id"],
                type=sample_data["type"],
                direction=sample_data["direction"],
                strength=sample_data["strength"],
                price=Decimal("-100.00"),
                volume=sample_data["volume"],
                timestamp=sample_data["timestamp"],
            )

    def test_is_buy_signal(self, sample_data):
        """Тест проверки buy сигнала."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=SignalDirection.BUY,
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
        )

        assert signal.is_buy_signal() is True
        assert signal.is_sell_signal() is False

    def test_is_sell_signal(self, sample_data):
        """Тест проверки sell сигнала."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=SignalDirection.SELL,
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
        )

        assert signal.is_sell_signal() is True
        assert signal.is_buy_signal() is False

    def test_is_strong_signal(self, sample_data):
        """Тест проверки сильного сигнала."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=SignalStrength.STRONG,
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
        )

        assert signal.is_strong_signal() is True
        assert signal.is_weak_signal() is False

    def test_get_weighted_score(self, sample_data):
        """Тест расчета взвешенного скора."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=SignalStrength.STRONG,
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
            confidence=Decimal("0.85"),
        )

        score = signal.get_weighted_score()
        expected_score = Decimal("0.85")
        assert score == expected_score

    def test_to_dict(self, sample_data):
        """Тест сериализации в словарь."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
            confidence=sample_data["confidence"],
            source=sample_data["source"],
            metadata=sample_data["metadata"],
        )

        result = signal.to_dict()

        assert result["id"] == sample_data["id"]
        assert result["type"] == sample_data["type"].value
        assert result["direction"] == sample_data["direction"].value
        assert result["strength"] == sample_data["strength"].value
        assert result["price"] == str(sample_data["price"])
        assert result["volume"] == str(sample_data["volume"])
        assert result["timestamp"] == sample_data["timestamp"].isoformat()
        assert result["confidence"] == str(sample_data["confidence"])
        assert result["source"] == sample_data["source"]
        assert result["metadata"] == sample_data["metadata"]

    def test_from_dict(self, sample_data):
        """Тест десериализации из словаря."""
        data = {
            "id": sample_data["id"],
            "type": sample_data["type"].value,
            "direction": sample_data["direction"].value,
            "strength": sample_data["strength"].value,
            "price": str(sample_data["price"]),
            "volume": str(sample_data["volume"]),
            "timestamp": sample_data["timestamp"].isoformat(),
            "confidence": str(sample_data["confidence"]),
            "source": sample_data["source"],
            "metadata": sample_data["metadata"],
        }

        signal = Signal.from_dict(data)

        assert signal.id == sample_data["id"]
        assert signal.type == sample_data["type"]
        assert signal.direction == sample_data["direction"]
        assert signal.strength == sample_data["strength"]
        assert signal.price == sample_data["price"]
        assert signal.volume == sample_data["volume"]
        assert signal.confidence == sample_data["confidence"]
        assert signal.source == sample_data["source"]
        assert signal.metadata == sample_data["metadata"]

    def test_equality(self, sample_data):
        """Тест равенства объектов."""
        signal1 = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
        )

        signal2 = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
        )

        assert signal1 == signal2
        assert hash(signal1) == hash(signal2)

    def test_str_representation(self, sample_data):
        """Тест строкового представления."""
        signal = Signal(
            id=sample_data["id"],
            type=sample_data["type"],
            direction=sample_data["direction"],
            strength=sample_data["strength"],
            price=sample_data["price"],
            volume=sample_data["volume"],
            timestamp=sample_data["timestamp"],
        )

        expected = f"Signal({sample_data['id']}: {sample_data['direction'].value} {sample_data['strength'].value} {sample_data['type'].value})"
        assert str(signal) == expected


class TestSignalType:
    """Тесты для SignalType."""

    def test_technical_value(self):
        """Тест значения TECHNICAL."""
        assert SignalType.TECHNICAL.value == "technical"

    def test_fundamental_value(self):
        """Тест значения FUNDAMENTAL."""
        assert SignalType.FUNDAMENTAL.value == "fundamental"


class TestSignalDirection:
    """Тесты для SignalDirection."""

    def test_buy_value(self):
        """Тест значения BUY."""
        assert SignalDirection.BUY.value == "buy"

    def test_sell_value(self):
        """Тест значения SELL."""
        assert SignalDirection.SELL.value == "sell"


class TestSignalStrength:
    """Тесты для SignalStrength."""

    def test_weak_value(self):
        """Тест значения WEAK."""
        assert SignalStrength.WEAK.value == "weak"

    def test_strong_value(self):
        """Тест значения STRONG."""
        assert SignalStrength.STRONG.value == "strong"

    def test_weak_weight(self):
        """Тест веса WEAK."""
        assert SignalStrength.WEAK.weight == Decimal("0.5")

    def test_strong_weight(self):
        """Тест веса STRONG."""
        assert SignalStrength.STRONG.weight == Decimal("1.0")
