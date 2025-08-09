"""
Базовые типы для value objects и доменных сущностей.
"""
from decimal import Decimal
from datetime import datetime
from typing import NewType, Final, Union
from uuid import UUID

# Базовые типы для value objects
AmountType = NewType("AmountType", Decimal)
CurrencyCode = NewType("CurrencyCode", str)
TimestampValue = NewType("TimestampValue", datetime)
PercentageValue = NewType("PercentageValue", Decimal)
# Числовые типы с валидацией
NumericType = Union[int, float, Decimal]
PositiveNumeric = NewType("PositiveNumeric", Decimal)
NonNegativeNumeric = NewType("NonNegativeNumeric", Decimal)
StrictPositiveNumeric = NewType("StrictPositiveNumeric", Decimal)
# Валютные типы
CurrencyPair = NewType("CurrencyPair", str)
ExchangeRate = NewType("ExchangeRate", Decimal)
PriceLevel = NewType("PriceLevel", Decimal)
VolumeAmount = NewType("VolumeAmount", Decimal)
MoneyAmount = NewType("MoneyAmount", Decimal)
# Торговые типы
SignalId = NewType("SignalId", str)
SignalScore = NewType("SignalScore", Decimal)
OrderId = NewType("OrderId", UUID)
PositionId = NewType("PositionId", UUID)
StrategyId = NewType("StrategyId", str)
GenerationId = NewType("GenerationId", str)
FitnessScore = NewType("FitnessScore", Decimal)
Symbol = NewType("Symbol", str)
# Константы для валидации
MONEY_PRECISION: Final[int] = 8
PRICE_PRECISION: Final[int] = 8
VOLUME_PRECISION: Final[int] = 8
PERCENTAGE_PRECISION: Final[int] = 6
# Лимиты для валидации
MAX_MONEY_AMOUNT: Final[Decimal] = Decimal("999999999999.99999999")
MIN_MONEY_AMOUNT: Final[Decimal] = Decimal("-999999999999.99999999")
MAX_PRICE: Final[Decimal] = Decimal("999999999.99999999")
MIN_PRICE: Final[Decimal] = Decimal("0.00000001")
MAX_VOLUME: Final[Decimal] = Decimal("999999999999.99999999")
MIN_VOLUME: Final[Decimal] = Decimal("0.00000001")
MAX_PERCENTAGE: Final[Decimal] = Decimal("10000")
MIN_PERCENTAGE: Final[Decimal] = Decimal("-10000")

# Дополнительные типы для торговли
TradingPair = NewType("TradingPair", str)

# Перечисления
from enum import Enum

class RiskLevel(Enum):
    """Уровни риска."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class SignalDirection(Enum):
    """Направления сигналов."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold" 