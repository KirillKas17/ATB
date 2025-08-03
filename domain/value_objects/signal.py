"""
Промышленная реализация Value Object для торговых сигналов с расширенной функциональностью для алготрейдинга.
"""

import decimal
import hashlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, ClassVar, Dict, List, Optional, Union

from domain.types.value_object_types import (
    SignalId,
    SignalScore,
    ValidationResult,
    ValueObject,
    ValueObjectDict,
)

from .signal_config import SignalConfig, SignalDirection, SignalType, SignalStrength
from .trading_pair import TradingPair


class Signal(ValueObject):
    """
    Промышленная реализация Value Object для торговых сигналов.
    Поддерживает:
    - Различные типы сигналов с приоритизацией
    - Силу сигнала и уверенность
    - Метаданные и источники
    - Временные метки и жизненный цикл
    - Финансовые параметры и риск-менеджмент
    """

    # Константы для валидации
    MIN_CONFIDENCE: Decimal = Decimal("0.1")
    MAX_CONFIDENCE: Decimal = Decimal("1.0")
    MIN_STRENGTH: Decimal = Decimal("0.1")
    MAX_STRENGTH: Decimal = Decimal("1.0")
    # Константы для жизненного цикла
    DEFAULT_EXPIRY_SECONDS: ClassVar[int] = 3600  # 1 час
    MAX_EXPIRY_SECONDS: ClassVar[int] = 86400  # 24 часа
    # Константы для скоринга
    STRENGTH_WEIGHT: ClassVar[float] = 0.4
    CONFIDENCE_WEIGHT: ClassVar[float] = 0.3
    RECENCY_WEIGHT: ClassVar[float] = 0.3

    def __init__(
        self,
        direction: SignalDirection,
        signal_type: SignalType,
        strength: Decimal,
        confidence: Decimal,
        trading_pair: TradingPair,
        price: Optional[Decimal] = None,
        volume: Optional[Decimal] = None,
        expiry_hours: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[SignalConfig] = None,
    ) -> None:
        self._config = config or SignalConfig()
        self._direction = direction  # SignalDirection
        self._signal_type = signal_type
        self._strength = self._normalize_strength(strength)
        self._confidence = self._normalize_confidence(confidence)
        self._trading_pair = trading_pair
        self._price = price
        self._volume = volume
        self._expiry_hours = expiry_hours or self._config.default_expiry_hours
        self._metadata = metadata or {}
        self._created_at = datetime.now(timezone.utc)
        self._validate_signal()

    def _normalize_strength(self, strength: Union[Decimal, int, float]) -> Decimal:
        if isinstance(strength, (int, float)):
            return Decimal(str(strength))
        if isinstance(strength, Decimal):
            return strength
        raise ValueError(f"Invalid strength type: {type(strength)}")

    def _normalize_confidence(self, confidence: Union[Decimal, int, float]) -> Decimal:
        if isinstance(confidence, (int, float)):
            return Decimal(str(confidence))
        if isinstance(confidence, Decimal):
            return confidence
        raise ValueError(f"Invalid confidence type: {type(confidence)}")

    def _validate_signal(self) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        # Проверки типов избыточны из-за строгой типизации в конструкторе
        if self._strength < self._config.min_strength:
            errors.append(f"Strength cannot be less than {self._config.min_strength}")
        if self._strength > self._config.max_strength:
            errors.append(f"Strength cannot exceed {self._config.max_strength}")
        if self._confidence < self._config.min_confidence:
            errors.append(
                f"Confidence cannot be less than {self._config.min_confidence}"
            )
        if self._confidence > self._config.max_confidence:
            errors.append(f"Confidence cannot exceed {self._config.max_confidence}")
        if self._expiry_hours < self._config.min_expiry_hours:
            errors.append(
                f"Expiry hours cannot be less than {self._config.min_expiry_hours}"
            )
        if self._expiry_hours > self._config.max_expiry_hours:
            errors.append(f"Expiry hours cannot exceed {self._config.max_expiry_hours}")
        if self._price is not None and self._price <= 0:
            errors.append("Price must be positive")
        if self._volume is not None and self._volume <= 0:
            errors.append("Volume must be positive")
        if self._strength < Decimal("0.3"):
            warnings.append("Signal strength is very low")
        if self._confidence < Decimal("0.5"):
            warnings.append("Signal confidence is low")
        is_valid = len(errors) == 0
        result = ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings
        )
        return result

    def validate(self) -> bool:
        """Валидирует сигнал и возвращает True если валиден."""
        result = self._validate_signal()
        return result.is_valid

    @property
    def value(self) -> str:
        value_str = f"{self._direction.value}_{self._signal_type.value}_{self._strength}_{self._confidence}"
        return value_str

    @property
    def hash(self) -> str:
        return self._calculate_hash()

    def _calculate_hash(self) -> str:
        data = f"{self._direction.value}:{self._signal_type.value}:{self._strength}:{self._confidence}:{self._trading_pair.symbol}:{self._created_at.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Signal):
            return False
        return (
            self._direction == other._direction
            and self._signal_type == other._signal_type
            and self._strength == other._strength
            and self._confidence == other._confidence
            and self._trading_pair == other._trading_pair
            and self._created_at == other._created_at
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._direction,
                self._signal_type,
                self._strength,
                self._confidence,
                self._trading_pair,
                self._created_at,
            )
        )

    def __str__(self) -> str:
        return f"{self._direction.value} {self._signal_type.value} Signal (Strength: {self._strength:.2f}, Confidence: {self._confidence:.2f})"

    def __repr__(self) -> str:
        return f"Signal({self._direction.value}, {self._signal_type.value}, {self._strength}, {self._confidence}, {self._trading_pair.symbol})"

    @property
    def direction(self) -> SignalDirection:
        return self._direction

    @property
    def signal_type(self) -> SignalType:
        return self._signal_type

    @property
    def strength(self) -> Decimal:
        return self._strength

    @property
    def confidence(self) -> Decimal:
        return self._confidence

    @property
    def trading_pair(self) -> TradingPair:
        return self._trading_pair

    @property
    def price(self) -> Optional[Decimal]:
        return self._price

    @property
    def volume(self) -> Optional[Decimal]:
        return self._volume

    @property
    def expiry_hours(self) -> int:
        return self._expiry_hours

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def timestamp(self) -> datetime:
        """Алиас для created_at для совместимости с тестами."""
        return self._created_at

    @property
    def expires_at(self) -> datetime:
        return self._created_at + timedelta(hours=self._expiry_hours)

    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def time_until_expiry(self) -> timedelta:
        return self.expires_at - datetime.now(timezone.utc)

    @property
    def is_buy_signal(self) -> bool:
        return self._direction == SignalDirection.BUY

    @property
    def is_sell_signal(self) -> bool:
        return self._direction == SignalDirection.SELL

    @property
    def is_hold_signal(self) -> bool:
        return self._direction == SignalDirection.HOLD

    @property
    def is_close_signal(self) -> bool:
        return self._direction == SignalDirection.CLOSE

    @property
    def is_trading_signal(self) -> bool:
        """Проверяет, является ли сигнал торговым (BUY/SELL)."""
        return self._direction in (SignalDirection.BUY, SignalDirection.SELL)

    def get_strength_level(self) -> str:
        if self._config.strength_thresholds is None:
            return "EXTREME"
        for level, threshold in self._config.strength_thresholds.items():
            if self._strength <= threshold:
                return level
        return "EXTREME"

    def get_confidence_level(self) -> str:
        if self._config.confidence_thresholds is None:
            return "VERY_HIGH"
        for level, threshold in self._config.confidence_thresholds.items():
            if self._confidence <= threshold:
                return level
        return "VERY_HIGH"

    def get_signal_score(self) -> Decimal:
        return (self._strength + self._confidence) / 2

    @property
    def is_strong_signal(self) -> bool:
        return self._strength >= Decimal("0.7") and self._confidence >= Decimal("0.7")

    @property
    def is_weak_signal(self) -> bool:
        return self._strength < Decimal("0.3") and self._confidence < Decimal("0.3")

    @property
    def is_reliable_signal(self) -> bool:
        return self._strength >= Decimal("0.5") and self._confidence >= Decimal("0.6")

    def get_trading_recommendation(self) -> str:
        if not self.is_reliable_signal:
            return "IGNORE"
        if self.is_expired:
            return "EXPIRED"
        if self.is_buy_signal and self.is_strong_signal:
            return "STRONG_BUY"
        elif self.is_buy_signal:
            return "BUY"
        elif self.is_sell_signal and self.is_strong_signal:
            return "STRONG_SELL"
        elif self.is_sell_signal:
            return "SELL"
        elif self.is_hold_signal:
            return "HOLD"
        elif self.is_close_signal:
            return "CLOSE"
        else:
            return "UNKNOWN"

    def get_risk_level(self) -> str:
        if self._strength >= Decimal("0.8") and self._confidence >= Decimal("0.8"):
            return "LOW"
        elif self._strength >= Decimal("0.6") and self._confidence >= Decimal("0.6"):
            return "MEDIUM"
        else:
            return "HIGH"

    def combine_with(self, other: "Signal") -> "Signal":
        """Комбинирует два сигнала."""
        if not isinstance(other, Signal):
            raise ValueError("Can only combine with another Signal")
        if self._trading_pair != other._trading_pair:
            raise ValueError("Can only combine signals for the same trading pair")
        if self._direction != other._direction:
            raise ValueError("Can only combine signals with the same direction")
        if self._signal_type != other._signal_type:
            raise ValueError("Can only combine signals with the same type")
        combined_strength = (self._strength + other._strength) / 2
        combined_confidence = (self._confidence + other._confidence) / 2
        combined_metadata = {**self._metadata, **other._metadata}
        return Signal(
            direction=self._direction,
            signal_type=self._signal_type,
            strength=combined_strength,
            confidence=combined_confidence,
            trading_pair=self._trading_pair,
            price=self._price or other._price,
            volume=self._volume or other._volume,
            expiry_hours=max(self._expiry_hours, other._expiry_hours),
            metadata=combined_metadata,
            config=self._config,
        )

    def to_dict(self) -> ValueObjectDict:
        """Сериализует сигнал в словарь."""
        return ValueObjectDict(
            direction=self._direction.value,
            signal_type=self._signal_type.value,
            strength=str(self._strength),
            confidence=str(self._confidence),
            trading_pair=self._trading_pair.to_dict(),
            price=str(self._price) if self._price else None,
            volume=str(self._volume) if self._volume else None,
            expiry_hours=self._expiry_hours,
            metadata=self._metadata,
            created_at=self._created_at.isoformat(),
        )

    @classmethod
    def from_dict(cls, data: ValueObjectDict) -> "Signal":
        """Создает сигнал из словаря."""
        direction = SignalDirection(data["direction"])
        signal_type = SignalType(data["signal_type"])
        strength = Decimal(data["strength"])
        confidence = Decimal(data["confidence"])
        trading_pair = TradingPair.from_dict(data["trading_pair"])
        price = Decimal(data["price"]) if data.get("price") else None
        volume = Decimal(data["volume"]) if data.get("volume") else None
        expiry_hours = data.get("expiry_hours", 1)
        metadata = data.get("metadata", {})
        return cls(
            direction=direction,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            trading_pair=trading_pair,
            price=price,
            volume=volume,
            expiry_hours=expiry_hours,
            metadata=metadata,
        )

    @classmethod
    def create_buy_signal(
        cls,
        trading_pair: TradingPair,
        strength: str = "STRONG",
        confidence: str = "0.7",
    ) -> "Signal":
        """Создает сигнал на покупку."""
        strength_map = {"WEAK": "0.3", "MODERATE": "0.5", "STRONG": "0.7", "EXTREME": "0.9"}
        strength_value = strength_map.get(strength, "0.7")
        return cls(
            direction=SignalDirection.BUY,
            signal_type=SignalType.TECHNICAL,
            strength=Decimal(strength_value),
            confidence=Decimal(confidence),
            trading_pair=trading_pair,
        )

    @classmethod
    def create_sell_signal(
        cls,
        trading_pair: TradingPair,
        strength: str = "STRONG",
        confidence: str = "0.7",
    ) -> "Signal":
        """Создает сигнал на продажу."""
        strength_map = {"WEAK": "0.3", "MODERATE": "0.5", "STRONG": "0.7", "EXTREME": "0.9"}
        strength_value = strength_map.get(strength, "0.7")
        return cls(
            direction=SignalDirection.SELL,
            signal_type=SignalType.TECHNICAL,
            strength=Decimal(strength_value),
            confidence=Decimal(confidence),
            trading_pair=trading_pair,
        )

    @classmethod
    def create_hold_signal(
        cls,
        trading_pair: TradingPair,
        strength: str = "MODERATE",
        confidence: str = "0.5",
    ) -> "Signal":
        """Создает сигнал на удержание."""
        strength_map = {"WEAK": "0.3", "MODERATE": "0.5", "STRONG": "0.7", "EXTREME": "0.9"}
        strength_value = strength_map.get(strength, "0.5")
        return cls(
            direction=SignalDirection.HOLD,
            signal_type=SignalType.TECHNICAL,
            strength=Decimal(strength_value),
            confidence=Decimal(confidence),
            trading_pair=trading_pair,
        )

    @classmethod
    def create_close_signal(
        cls,
        trading_pair: TradingPair,
        strength: str = "MODERATE",
        confidence: str = "0.5",
    ) -> "Signal":
        """Создает сигнал на закрытие позиции."""
        strength_map = {"WEAK": "0.3", "MODERATE": "0.5", "STRONG": "0.7", "EXTREME": "0.9"}
        strength_value = strength_map.get(strength, "0.5")
        return cls(
            direction=SignalDirection.CLOSE,
            signal_type=SignalType.TECHNICAL,
            strength=Decimal(strength_value),
            confidence=Decimal(confidence),
            trading_pair=trading_pair,
        )

    def copy(self) -> "Signal":
        """Создает копию сигнала."""
        return Signal(
            direction=self._direction,
            signal_type=self._signal_type,
            strength=self._strength,
            confidence=self._confidence,
            trading_pair=self._trading_pair,
            price=self._price,
            volume=self._volume,
            expiry_hours=self._expiry_hours,
            metadata=self._metadata.copy(),
            config=self._config,
        )

    def get_combined_score(self) -> Decimal:
        """Возвращает комбинированный скор сигнала."""
        return (self._strength + self._confidence) / 2


