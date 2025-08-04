"""
Доменная сущность TradingPair
Представляет торговую пару с базовой и котируемой валютой
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, Optional, Protocol, runtime_checkable, Any

from domain.type_definitions import TradingPair as TradingPairType
from domain.type_definitions import VolumePrecision, Symbol, PricePrecision

from ..value_objects.currency import Currency
from ..value_objects.price import Price
from ..value_objects.volume import Volume


class PairStatus(Enum):
    """Статус торговой пары."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    DELISTED = "delisted"


@runtime_checkable
class TradingPairProtocol(Protocol):
    """Протокол для торговых пар."""

    symbol: Symbol
    base_currency: Currency
    quote_currency: Currency
    is_active: bool
    min_order_size: Optional[Volume]
    max_order_size: Optional[Volume]
    price_precision: PricePrecision
    volume_precision: VolumePrecision
    created_at: datetime
    updated_at: datetime

    @property
    def status(self) -> PairStatus: ...
    @status.setter
    def status(self, value: PairStatus) -> None: ...
    @property
    def display_name(self) -> str: ...
    def validate_price(self, price: Price) -> bool: ...
    def validate_volume(self, volume: Volume) -> bool: ...
    def calculate_notional_value(self, price: Price, volume: Volume) -> Price: ...
    def deactivate(self) -> None: ...
    def activate(self) -> None: ...
    def update_precision(self, price_precision: int, volume_precision: int) -> None: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingPairProtocol": ...
@dataclass
class TradingPair:
    """
    Торговая пара - основная сущность для торговли
    Attributes:
        symbol: Символ торговой пары (например, "BTC/USDT")
        base_currency: Базовая валюта
        quote_currency: Котируемая валюта
        is_active: Активна ли пара для торговли
        min_order_size: Минимальный размер ордера
        max_order_size: Максимальный размер ордера
        price_precision: Точность цены
        volume_precision: Точность объема
        created_at: Дата создания
        updated_at: Дата обновления
    """

    symbol: Symbol
    base_currency: Currency
    quote_currency: Currency
    is_active: bool = True
    min_order_size: Optional[Volume] = None
    max_order_size: Optional[Volume] = None
    price_precision: PricePrecision = field(default=PricePrecision(8))
    volume_precision: VolumePrecision = field(default=VolumePrecision(8))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def status(self) -> PairStatus:
        """Статус торговой пары."""
        return PairStatus.ACTIVE if self.is_active else PairStatus.INACTIVE

    @status.setter
    def status(self, value: PairStatus) -> None:
        """Установка статуса торговой пары."""
        self.is_active = value == PairStatus.ACTIVE

    def __post_init__(self) -> None:
        """
        Валидация после инициализации"""
        if not self.symbol:
            raise ValueError("Symbol cannot be empty")
        if self.base_currency == self.quote_currency:
            raise ValueError("Base and quote currencies cannot be the same")
        if self.price_precision < 0 or self.volume_precision < 0:
            raise ValueError("Precision cannot be negative")

    @property
    def display_name(self) -> str:
        """Отображаемое имя пары"""
        return f"{self.base_currency.code}/{self.quote_currency.code}"

    def validate_price(self, price: Price) -> bool:
        """Валидация цены для данной пары"""
        if price.currency != self.quote_currency:
            return False
        # Проверка минимальной цены
        if price.value <= 0:
            return False
        return True

    def validate_volume(self, volume: Volume) -> bool:
        """Валидация объема для данной пары"""
        if volume.currency != self.base_currency:
            return False
        # Проверка минимального объема
        if self.min_order_size and volume.value < self.min_order_size.value:
            return False
        # Проверка максимального объема
        if self.max_order_size and volume.value > self.max_order_size.value:
            return False
        return True

    def calculate_notional_value(self, price: Price, volume: Volume) -> Price:
        """Расчет номинальной стоимости сделки"""
        if not self.validate_price(price) or not self.validate_volume(volume):
            raise ValueError("Invalid price or volume for this trading pair")
        notional_value = price.value * volume.value
        return Price(notional_value, self.quote_currency)

    def deactivate(self) -> None:
        """Деактивация торговой пары"""
        self.is_active = False
        self.updated_at = datetime.now()

    def activate(self) -> None:
        """Активация торговой пары"""
        self.is_active = True
        self.updated_at = datetime.now()

    def update_precision(self, price_precision: int, volume_precision: int) -> None:
        """Обновление точности"""
        if price_precision < 0 or volume_precision < 0:
            raise ValueError("Precision cannot be negative")
        self.price_precision = PricePrecision(price_precision)
        self.volume_precision = VolumePrecision(volume_precision)
        self.updated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "symbol": str(self.symbol),
            "base_currency": self.base_currency.code,
            "quote_currency": self.quote_currency.code,
            "is_active": self.is_active,
            "min_order_size": (
                str(self.min_order_size.value) if self.min_order_size else None
            ),
            "max_order_size": (
                str(self.max_order_size.value) if self.max_order_size else None
            ),
            "price_precision": int(self.price_precision),
            "volume_precision": int(self.volume_precision),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TradingPair":
        """Создание из словаря"""
        base_currency = Currency.from_string(data["base_currency"]) if data.get("base_currency") else None
        quote_currency = Currency.from_string(data["quote_currency"]) if data.get("quote_currency") else None
        
        if base_currency is None or quote_currency is None:
            raise ValueError("Base and quote currencies cannot be None")
            
        return cls(
            symbol=Symbol(data["symbol"]),
            base_currency=base_currency,
            quote_currency=quote_currency,
            is_active=data["is_active"],
            min_order_size=(
                Volume(Decimal(str(data["min_order_size"])), Currency.USD)
                if data.get("min_order_size")
                else None
            ),
            max_order_size=(
                Volume(Decimal(str(data["max_order_size"])), Currency.USD)
                if data.get("max_order_size")
                else None
            ),
            price_precision=PricePrecision(data["price_precision"]),
            volume_precision=VolumePrecision(data["volume_precision"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    def __eq__(self, other: object) -> bool:
        """Сравнение торговых пар"""
        if not isinstance(other, TradingPair):
            return False
        return self.symbol == other.symbol

    def __hash__(self) -> int:
        """Хеш торговой пары"""
        return hash(self.symbol)

    def __str__(self) -> str:
        """Строковое представление"""
        return self.display_name
