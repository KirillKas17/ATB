"""
Доменная сущность торгового сигнала.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, cast
from uuid import UUID, uuid4

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money

# Расширенный тип для metadata, поддерживающий вложенные структуры
ExtendedMetadataValue = Union[
    str,
    int,
    float,
    Decimal,
    bool,
    List[str],
    Dict[str, Union[str, int, float, Decimal, bool]],
]
ExtendedMetadataDict = Dict[str, ExtendedMetadataValue]


class SignalType(Enum):
    """Типы сигналов."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"
    CANCEL = "cancel"
    REVERSAL = "reversal"


class SignalStrength(Enum):
    """Сила сигнала."""

    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


@dataclass
class Signal:
    """Торговый сигнал"""

    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    trading_pair: str = ""
    signal_type: SignalType = SignalType.HOLD
    strength: SignalStrength = SignalStrength.MEDIUM
    confidence: Decimal = Decimal("0.5")
    price: Optional[Money] = None
    quantity: Optional[Decimal] = None
    stop_loss: Optional[Money] = None
    take_profit: Optional[Money] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: ExtendedMetadataDict = field(default_factory=dict)
    is_actionable: bool = True
    expires_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """
        Пост-инициализация с валидацией"""
        if self.confidence < Decimal("0") or self.confidence > Decimal("1"):
            raise ValueError("Confidence must be between 0 and 1")
        if self.quantity is not None and self.quantity <= Decimal("0"):
            raise ValueError("Quantity must be positive")
        if self.price is not None and self.price.value <= Decimal("0"):
            raise ValueError("Price must be positive")
        if self.stop_loss is not None and self.stop_loss.value <= Decimal("0"):
            raise ValueError("Stop loss must be positive")
        if self.take_profit is not None and self.take_profit.value <= Decimal("0"):
            raise ValueError("Take profit must be positive")

    @property
    def is_expired(self) -> bool:
        """Проверка истечения срока действия сигнала"""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    @property
    def signal_id(self) -> UUID:
        """Alias for id field for backward compatibility"""
        return self.id

    @property
    def risk_reward_ratio(self) -> Optional[Decimal]:
        """Расчет соотношения риск/прибыль"""
        if self.price is None or self.stop_loss is None or self.take_profit is None:
            return None
        risk = abs(self.price.value - self.stop_loss.value)
        reward = abs(self.take_profit.value - self.price.value)
        if risk == Decimal("0"):
            return None
        return reward / risk

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            str,
            int,
            float,
            Decimal,
            bool,
            List[str],
            Dict[str, Union[str, int, float, Decimal, bool]],
            None,
        ],
    ]:
        """Преобразовать в словарь"""
        return {
            "strategy_id": str(self.strategy_id),
            "id": str(self.id),
            "trading_pair": self.trading_pair,
            "signal_type": self.signal_type.value,
            "strength": self.strength.value,
            "confidence": str(self.confidence),
            "price": str(self.price.value) if self.price else None,
            "stop_loss": str(self.stop_loss.value) if self.stop_loss else None,
            "take_profit": str(self.take_profit.value) if self.take_profit else None,
            "quantity": str(self.quantity) if self.quantity else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": cast(
                Dict[str, Union[str, int, float, Decimal, bool]], self.metadata
            ),
            "is_actionable": self.is_actionable,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[
            str,
            Union[
                str,
                int,
                float,
                Decimal,
                bool,
                List[str],
                Dict[str, Union[str, int, float, Decimal, bool]],
                None,
            ],
        ],
    ) -> "Signal":
        """Создание из словаря"""
        # Безопасное извлечение и преобразование данных
        strategy_id_value = data.get("strategy_id", "")
        signal_id_value = data.get("id", "")
        trading_pair = str(data.get("trading_pair", ""))
        signal_type_value = data.get("signal_type", "hold")
        strength_value = data.get("strength", "medium")
        confidence_value = data.get("confidence", "0.5")
        price_value = data.get("price")
        stop_loss_value = data.get("stop_loss")
        take_profit_value = data.get("take_profit")
        quantity_value = data.get("quantity")
        timestamp_value = data.get("timestamp", "")
        metadata_value = data.get("metadata", {})
        is_actionable = bool(data.get("is_actionable", True))
        expires_at_value = data.get("expires_at")
        # Преобразование UUID
        try:
            strategy_id = UUID(str(strategy_id_value)) if strategy_id_value else uuid4()
        except ValueError:
            strategy_id = uuid4()
        try:
            signal_id = UUID(str(signal_id_value)) if signal_id_value else uuid4()
        except ValueError:
            signal_id = uuid4()
        # Преобразование типа сигнала
        try:
            signal_type = SignalType(str(signal_type_value))
        except ValueError:
            signal_type = SignalType.HOLD
        # Преобразование силы сигнала
        try:
            strength = SignalStrength(str(strength_value))
        except ValueError:
            strength = SignalStrength.MEDIUM
        # Преобразование уверенности
        try:
            confidence = Decimal(str(confidence_value))
        except (ValueError, TypeError):
            confidence = Decimal("0.5")
        # Преобразование цен
        price = None
        if price_value is not None:
            try:
                price_amount = Decimal(str(price_value))
                price = Money(
                    price_amount, Currency.USD
                )  # Используем USD как базовую валюту
            except (ValueError, TypeError):
                pass
        stop_loss = None
        if stop_loss_value is not None:
            try:
                stop_loss_amount = Decimal(str(stop_loss_value))
                stop_loss = Money(stop_loss_amount, Currency.USD)
            except (ValueError, TypeError):
                pass
        take_profit = None
        if take_profit_value is not None:
            try:
                take_profit_amount = Decimal(str(take_profit_value))
                take_profit = Money(take_profit_amount, Currency.USD)
            except (ValueError, TypeError):
                pass
        # Преобразование количества
        quantity = None
        if quantity_value is not None:
            try:
                quantity = Decimal(str(quantity_value))
            except (ValueError, TypeError):
                pass
        # Преобразование времени
        try:
            timestamp = (
                datetime.fromisoformat(str(timestamp_value))
                if timestamp_value
                else datetime.now()
            )
        except ValueError:
            timestamp = datetime.now()
        # Преобразование metadata
        if not isinstance(metadata_value, dict):
            metadata_value = {}
        # Преобразование времени истечения
        expires_at = None
        if expires_at_value is not None:
            try:
                expires_at = datetime.fromisoformat(str(expires_at_value))
            except ValueError:
                pass
        return cls(
            id=signal_id,
            strategy_id=strategy_id,
            trading_pair=trading_pair,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            timestamp=timestamp,
            metadata=cast(ExtendedMetadataDict, metadata_value),
            is_actionable=is_actionable,
            expires_at=expires_at,
        )
