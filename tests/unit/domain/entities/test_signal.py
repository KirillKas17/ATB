"""
Unit тесты для Signal.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, Any, Union, List
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.signal import (
    Signal, SignalType, SignalStrength,
    ExtendedMetadataValue, ExtendedMetadataDict
)
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class TestSignalType:
    """Тесты для SignalType."""
    
    def test_enum_values(self):
        """Тест значений enum SignalType."""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.CLOSE.value == "close"
        assert SignalType.CANCEL.value == "cancel"


class TestSignalStrength:
    """Тесты для SignalStrength."""
    
    def test_enum_values(self):
        """Тест значений enum SignalStrength."""
        assert SignalStrength.VERY_WEAK.value == "very_weak"
        assert SignalStrength.WEAK.value == "weak"
        assert SignalStrength.MEDIUM.value == "medium"
        assert SignalStrength.STRONG.value == "strong"
        assert SignalStrength.VERY_STRONG.value == "very_strong"


class TestSignal:
    """Тесты для Signal."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "strategy_id": uuid4(),
            "trading_pair": "BTC/USD",
            "signal_type": SignalType.BUY,
            "strength": SignalStrength.STRONG,
            "confidence": Decimal("0.8"),
            "price": Money(Decimal("50000"), Currency.USD),
            "quantity": Decimal("1.5"),
            "stop_loss": Money(Decimal("48000"), Currency.USD),
            "take_profit": Money(Decimal("55000"), Currency.USD),
            "timestamp": datetime.now(),
            "metadata": {
                "source": "trend_analysis",
                "indicators": ["RSI", "MACD"],
                "confidence_level": 0.9
            },
            "is_actionable": True,
            "expires_at": datetime.now() + timedelta(hours=1)
        }
    
    @pytest.fixture
    def signal(self, sample_data) -> Signal:
        """Создает тестовый сигнал."""
        return Signal(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания сигнала."""
        signal = Signal(**sample_data)
        
        assert signal.id == sample_data["id"]
        assert signal.strategy_id == sample_data["strategy_id"]
        assert signal.trading_pair == sample_data["trading_pair"]
        assert signal.signal_type == sample_data["signal_type"]
        assert signal.strength == sample_data["strength"]
        assert signal.confidence == sample_data["confidence"]
        assert signal.price == sample_data["price"]
        assert signal.quantity == sample_data["quantity"]
        assert signal.stop_loss == sample_data["stop_loss"]
        assert signal.take_profit == sample_data["take_profit"]
        assert signal.timestamp == sample_data["timestamp"]
        assert signal.metadata == sample_data["metadata"]
        assert signal.is_actionable == sample_data["is_actionable"]
        assert signal.expires_at == sample_data["expires_at"]
    
    def test_default_creation(self):
        """Тест создания сигнала с дефолтными значениями."""
        signal = Signal()
        
        assert isinstance(signal.id, uuid4().__class__)
        assert isinstance(signal.strategy_id, uuid4().__class__)
        assert signal.trading_pair == ""
        assert signal.signal_type == SignalType.HOLD
        assert signal.strength == SignalStrength.MEDIUM
        assert signal.confidence == Decimal("0.5")
        assert signal.price is None
        assert signal.quantity is None
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert isinstance(signal.timestamp, datetime)
        assert signal.metadata == {}
        assert signal.is_actionable is True
        assert signal.expires_at is None
    
    def test_validation_confidence_below_zero(self):
        """Тест валидации confidence ниже 0."""
        data = {
            "confidence": Decimal("-0.1")
        }
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Signal(**data)
    
    def test_validation_confidence_above_one(self):
        """Тест валидации confidence выше 1."""
        data = {
            "confidence": Decimal("1.1")
        }
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Signal(**data)
    
    def test_validation_negative_quantity(self):
        """Тест валидации отрицательного количества."""
        data = {
            "quantity": Decimal("-1.0")
        }
        
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Signal(**data)
    
    def test_validation_zero_quantity(self):
        """Тест валидации нулевого количества."""
        data = {
            "quantity": Decimal("0")
        }
        
        with pytest.raises(ValueError, match="Quantity must be positive"):
            Signal(**data)
    
    def test_validation_negative_price(self):
        """Тест валидации отрицательной цены."""
        data = {
            "price": Money(Decimal("-100"), Currency.USD)
        }
        
        with pytest.raises(ValueError, match="Price must be positive"):
            Signal(**data)
    
    def test_validation_zero_price(self):
        """Тест валидации нулевой цены."""
        data = {
            "price": Money(Decimal("0"), Currency.USD)
        }
        
        with pytest.raises(ValueError, match="Price must be positive"):
            Signal(**data)
    
    def test_validation_negative_stop_loss(self):
        """Тест валидации отрицательного stop loss."""
        data = {
            "stop_loss": Money(Decimal("-100"), Currency.USD)
        }
        
        with pytest.raises(ValueError, match="Stop loss must be positive"):
            Signal(**data)
    
    def test_validation_zero_stop_loss(self):
        """Тест валидации нулевого stop loss."""
        data = {
            "stop_loss": Money(Decimal("0"), Currency.USD)
        }
        
        with pytest.raises(ValueError, match="Stop loss must be positive"):
            Signal(**data)
    
    def test_validation_negative_take_profit(self):
        """Тест валидации отрицательного take profit."""
        data = {
            "take_profit": Money(Decimal("-100"), Currency.USD)
        }
        
        with pytest.raises(ValueError, match="Take profit must be positive"):
            Signal(**data)
    
    def test_validation_zero_take_profit(self):
        """Тест валидации нулевого take profit."""
        data = {
            "take_profit": Money(Decimal("0"), Currency.USD)
        }
        
        with pytest.raises(ValueError, match="Take profit must be positive"):
            Signal(**data)
    
    def test_valid_boundary_values(self):
        """Тест валидных граничных значений."""
        data = {
            "confidence": Decimal("0.0"),  # Граничное значение
            "quantity": Decimal("0.001"),  # Минимальное положительное
            "price": Money(Decimal("0.001"), Currency.USD),  # Минимальное положительное
            "stop_loss": Money(Decimal("0.001"), Currency.USD),  # Минимальное положительное
            "take_profit": Money(Decimal("0.001"), Currency.USD)  # Минимальное положительное
        }
        
        signal = Signal(**data)
        assert signal.confidence == Decimal("0.0")
        assert signal.quantity == Decimal("0.001")
        assert signal.price.value == Decimal("0.001")
        assert signal.stop_loss.value == Decimal("0.001")
        assert signal.take_profit.value == Decimal("0.001")
    
    def test_is_expired_no_expiration(self):
        """Тест проверки истечения срока - без срока истечения."""
        signal = Signal()
        assert signal.is_expired is False
    
    def test_is_expired_future_expiration(self):
        """Тест проверки истечения срока - будущая дата."""
        future_time = datetime.now() + timedelta(hours=1)
        signal = Signal(expires_at=future_time)
        assert signal.is_expired is False
    
    def test_is_expired_past_expiration(self):
        """Тест проверки истечения срока - прошедшая дата."""
        past_time = datetime.now() - timedelta(hours=1)
        signal = Signal(expires_at=past_time)
        assert signal.is_expired is True
    
    def test_risk_reward_ratio_with_all_values(self):
        """Тест расчета соотношения риск/прибыль с полными данными."""
        signal = Signal(
            price=Money(Decimal("50000"), Currency.USD),
            stop_loss=Money(Decimal("48000"), Currency.USD),
            take_profit=Money(Decimal("55000"), Currency.USD)
        )
        
        ratio = signal.risk_reward_ratio
        expected_risk = Decimal("2000")  # 50000 - 48000
        expected_reward = Decimal("5000")  # 55000 - 50000
        expected_ratio = expected_reward / expected_risk
        
        assert ratio == expected_ratio
    
    def test_risk_reward_ratio_missing_price(self):
        """Тест расчета соотношения риск/прибыль без цены."""
        signal = Signal(
            stop_loss=Money(Decimal("48000"), Currency.USD),
            take_profit=Money(Decimal("55000"), Currency.USD)
        )
        
        assert signal.risk_reward_ratio is None
    
    def test_risk_reward_ratio_missing_stop_loss(self):
        """Тест расчета соотношения риск/прибыль без stop loss."""
        signal = Signal(
            price=Money(Decimal("50000"), Currency.USD),
            take_profit=Money(Decimal("55000"), Currency.USD)
        )
        
        assert signal.risk_reward_ratio is None
    
    def test_risk_reward_ratio_missing_take_profit(self):
        """Тест расчета соотношения риск/прибыль без take profit."""
        signal = Signal(
            price=Money(Decimal("50000"), Currency.USD),
            stop_loss=Money(Decimal("48000"), Currency.USD)
        )
        
        assert signal.risk_reward_ratio is None
    
    def test_risk_reward_ratio_zero_risk(self):
        """Тест расчета соотношения риск/прибыль с нулевым риском."""
        signal = Signal(
            price=Money(Decimal("50000"), Currency.USD),
            stop_loss=Money(Decimal("50000"), Currency.USD),  # Нулевой риск
            take_profit=Money(Decimal("55000"), Currency.USD)
        )
        
        assert signal.risk_reward_ratio is None
    
    def test_risk_reward_ratio_sell_signal(self):
        """Тест расчета соотношения риск/прибыль для сигнала продажи."""
        signal = Signal(
            price=Money(Decimal("50000"), Currency.USD),
            stop_loss=Money(Decimal("52000"), Currency.USD),  # Stop loss выше цены
            take_profit=Money(Decimal("45000"), Currency.USD)  # Take profit ниже цены
        )
        
        ratio = signal.risk_reward_ratio
        expected_risk = Decimal("2000")  # 52000 - 50000
        expected_reward = Decimal("5000")  # 50000 - 45000
        expected_ratio = expected_reward / expected_risk
        
        assert ratio == expected_ratio
    
    def test_to_dict(self, signal):
        """Тест сериализации в словарь."""
        data = signal.to_dict()
        
        assert data["id"] == str(signal.id)
        assert data["strategy_id"] == str(signal.strategy_id)
        assert data["trading_pair"] == signal.trading_pair
        assert data["signal_type"] == signal.signal_type.value
        assert data["strength"] == signal.strength.value
        assert data["confidence"] == str(signal.confidence)
        assert data["price"] == str(signal.price.value)
        assert data["stop_loss"] == str(signal.stop_loss.value)
        assert data["take_profit"] == str(signal.take_profit.value)
        assert data["quantity"] == str(signal.quantity)
        assert data["timestamp"] == signal.timestamp.isoformat()
        assert data["metadata"] == signal.metadata
        assert data["is_actionable"] == signal.is_actionable
        assert data["expires_at"] == signal.expires_at.isoformat()
    
    def test_to_dict_none_values(self):
        """Тест сериализации в словарь с None значениями."""
        signal = Signal(
            price=None,
            stop_loss=None,
            take_profit=None,
            quantity=None,
            expires_at=None
        )
        data = signal.to_dict()
        
        assert data["price"] is None
        assert data["stop_loss"] is None
        assert data["take_profit"] is None
        assert data["quantity"] is None
        assert data["expires_at"] is None
    
    def test_from_dict(self, signal):
        """Тест десериализации из словаря."""
        data = signal.to_dict()
        new_signal = Signal.from_dict(data)
        
        assert new_signal.id == signal.id
        assert new_signal.strategy_id == signal.strategy_id
        assert new_signal.trading_pair == signal.trading_pair
        assert new_signal.signal_type == signal.signal_type
        assert new_signal.strength == signal.strength
        assert new_signal.confidence == signal.confidence
        assert new_signal.price.value == signal.price.value
        assert new_signal.stop_loss.value == signal.stop_loss.value
        assert new_signal.take_profit.value == signal.take_profit.value
        assert new_signal.quantity == signal.quantity
        assert new_signal.timestamp == signal.timestamp
        assert new_signal.metadata == signal.metadata
        assert new_signal.is_actionable == signal.is_actionable
        assert new_signal.expires_at == signal.expires_at
    
    def test_from_dict_invalid_uuid(self):
        """Тест десериализации из словаря с невалидным UUID."""
        data = {
            "id": "invalid_uuid",
            "strategy_id": "invalid_uuid",
            "trading_pair": "BTC/USD",
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "price": "50000",
            "stop_loss": "48000",
            "take_profit": "55000",
            "quantity": "1.5",
            "timestamp": datetime.now().isoformat(),
            "metadata": {},
            "is_actionable": True,
            "expires_at": (datetime.now() + timedelta(hours=1)).isoformat()
        }
        
        signal = Signal.from_dict(data)
        assert isinstance(signal.id, uuid4().__class__)
        assert isinstance(signal.strategy_id, uuid4().__class__)
    
    def test_from_dict_invalid_signal_type(self):
        """Тест десериализации из словаря с невалидным типом сигнала."""
        data = {
            "signal_type": "invalid_type",
            "strength": "strong",
            "confidence": "0.8"
        }
        
        signal = Signal.from_dict(data)
        assert signal.signal_type == SignalType.HOLD  # Дефолтное значение
    
    def test_from_dict_invalid_strength(self):
        """Тест десериализации из словаря с невалидной силой сигнала."""
        data = {
            "signal_type": "buy",
            "strength": "invalid_strength",
            "confidence": "0.8"
        }
        
        signal = Signal.from_dict(data)
        assert signal.strength == SignalStrength.MEDIUM  # Дефолтное значение
    
    def test_from_dict_invalid_confidence(self):
        """Тест десериализации из словаря с невалидной уверенностью."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "invalid_confidence"
        }
        
        signal = Signal.from_dict(data)
        assert signal.confidence == Decimal("0.5")  # Дефолтное значение
    
    def test_from_dict_invalid_price(self):
        """Тест десериализации из словаря с невалидной ценой."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "price": "invalid_price"
        }
        
        signal = Signal.from_dict(data)
        assert signal.price is None
    
    def test_from_dict_invalid_stop_loss(self):
        """Тест десериализации из словаря с невалидным stop loss."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "stop_loss": "invalid_stop_loss"
        }
        
        signal = Signal.from_dict(data)
        assert signal.stop_loss is None
    
    def test_from_dict_invalid_take_profit(self):
        """Тест десериализации из словаря с невалидным take profit."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "take_profit": "invalid_take_profit"
        }
        
        signal = Signal.from_dict(data)
        assert signal.take_profit is None
    
    def test_from_dict_invalid_quantity(self):
        """Тест десериализации из словаря с невалидным количеством."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "quantity": "invalid_quantity"
        }
        
        signal = Signal.from_dict(data)
        assert signal.quantity is None
    
    def test_from_dict_invalid_timestamp(self):
        """Тест десериализации из словаря с невалидной временной меткой."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "timestamp": "invalid_timestamp"
        }
        
        signal = Signal.from_dict(data)
        assert isinstance(signal.timestamp, datetime)
    
    def test_from_dict_invalid_expires_at(self):
        """Тест десериализации из словаря с невалидной датой истечения."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "expires_at": "invalid_expires_at"
        }
        
        signal = Signal.from_dict(data)
        assert signal.expires_at is None
    
    def test_from_dict_invalid_metadata(self):
        """Тест десериализации из словаря с невалидными метаданными."""
        data = {
            "signal_type": "buy",
            "strength": "strong",
            "confidence": "0.8",
            "metadata": "invalid_metadata"  # Не словарь
        }
        
        signal = Signal.from_dict(data)
        assert signal.metadata == {}
    
    def test_from_dict_empty_data(self):
        """Тест десериализации из пустого словаря."""
        data = {}
        signal = Signal.from_dict(data)
        
        assert isinstance(signal.id, uuid4().__class__)
        assert isinstance(signal.strategy_id, uuid4().__class__)
        assert signal.trading_pair == ""
        assert signal.signal_type == SignalType.HOLD
        assert signal.strength == SignalStrength.MEDIUM
        assert signal.confidence == Decimal("0.5")
        assert signal.price is None
        assert signal.quantity is None
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert isinstance(signal.timestamp, datetime)
        assert signal.metadata == {}
        assert signal.is_actionable is True
        assert signal.expires_at is None
    
    def test_extended_metadata_types(self):
        """Тест различных типов метаданных."""
        metadata = {
            "string": "test",
            "integer": 42,
            "float": 3.14,
            "decimal": Decimal("2.718"),
            "boolean": True,
            "list": ["item1", "item2"],
            "nested_dict": {"key": "value", "number": 123}
        }
        
        signal = Signal(metadata=metadata)
        assert signal.metadata == metadata
        assert isinstance(signal.metadata["string"], str)
        assert isinstance(signal.metadata["integer"], int)
        assert isinstance(signal.metadata["float"], float)
        assert isinstance(signal.metadata["decimal"], Decimal)
        assert isinstance(signal.metadata["boolean"], bool)
        assert isinstance(signal.metadata["list"], list)
        assert isinstance(signal.metadata["nested_dict"], dict) 