"""
Промышленные типы для Value Objects домена с расширенной функциональностью для алготрейдинга.
"""

import hashlib
import json
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    ClassVar,
    Dict,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
    NewType,
)

# Импорты базовых типов
from domain.types.base_types import (
    AmountType,
    CurrencyCode,
    TimestampValue,
    PercentageValue,
    NumericType,
    PositiveNumeric,
    NonNegativeNumeric,
    StrictPositiveNumeric,
    CurrencyPair,
    ExchangeRate,
    PriceLevel,
    VolumeAmount,
    MoneyAmount,
    SignalId,
    SignalScore,
    OrderId,
    PositionId,
    MONEY_PRECISION,
    PRICE_PRECISION,
    VOLUME_PRECISION,
    PERCENTAGE_PRECISION,
    MAX_MONEY_AMOUNT,
    MIN_MONEY_AMOUNT,
    MAX_PRICE,
    MIN_PRICE,
    MAX_VOLUME,
    MIN_VOLUME,
    MAX_PERCENTAGE,
    MIN_PERCENTAGE,
)

# Импорты value objects - удаляем циклические импорты
# from domain.value_objects.money import Money
# from domain.value_objects.price import Price
# from domain.value_objects.volume import Volume
# from domain.value_objects.percentage import Percentage
# from domain.value_objects.timestamp import Timestamp
# from domain.value_objects.signal import Signal
# from domain.value_objects.trading_pair import TradingPair

# Типы для кэширования
CacheKey = NewType("CacheKey", str)
CacheEntry = Dict[str, Any]
CacheStats = Dict[str, Union[int, float, str]]
# Generic тип для value objects
T = TypeVar("T", bound="ValueObject")


@runtime_checkable
class ValueObject(Protocol):
    """Базовый протокол для всех value objects."""

    @property
    @abstractmethod
    def value(self) -> Any:
        """Возвращает значение value object."""
        ...

    @property
    @abstractmethod
    def hash(self) -> str:
        """Возвращает хеш value object для кэширования."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Сериализует value object в словарь."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Валидирует value object."""
        ...

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        """Сравнение value objects."""
        ...

    @abstractmethod
    def __hash__(self) -> int:
        """Хеширование value object."""
        ...

    @abstractmethod
    def __str__(self) -> str:
        """Строковое представление."""
        ...

    @abstractmethod
    def __repr__(self) -> str:
        """Представление для отладки."""
        ...


@runtime_checkable
class CurrencyValueObject(ValueObject, Protocol):
    """Протокол для value objects с валютой."""

    @property
    @abstractmethod
    def currency(self) -> CurrencyCode:
        """Возвращает валюту."""
        ...

    @abstractmethod
    def convert_to(
        self, target_currency: CurrencyCode, rate: ExchangeRate
    ) -> "CurrencyValueObject":
        """Конвертирует в другую валюту."""
        ...


@runtime_checkable
class NumericValueObject(ValueObject, Protocol):
    """Протокол для числовых value objects."""

    @property
    @abstractmethod
    def amount(self) -> Decimal:
        """Возвращает числовое значение."""
        ...

    @abstractmethod
    def __add__(self, other: "NumericValueObject") -> "NumericValueObject":
        """Сложение."""
        ...

    @abstractmethod
    def __sub__(self, other: "NumericValueObject") -> "NumericValueObject":
        """Вычитание."""
        ...

    @abstractmethod
    def __mul__(self, other: Union[Decimal, float, int]) -> "NumericValueObject":
        """Умножение."""
        ...

    @abstractmethod
    def __truediv__(self, other: Union[Decimal, float, int]) -> "NumericValueObject":
        """Деление."""
        ...

    @abstractmethod
    def __lt__(self, other: "NumericValueObject") -> bool:
        """Меньше."""
        ...

    @abstractmethod
    def __le__(self, other: "NumericValueObject") -> bool:
        """Меньше или равно."""
        ...

    @abstractmethod
    def __gt__(self, other: "NumericValueObject") -> bool:
        """Больше."""
        ...

    @abstractmethod
    def __ge__(self, other: "NumericValueObject") -> bool:
        """Больше или равно."""
        ...


@runtime_checkable
class TradingValueObject(ValueObject, Protocol):
    """Протокол для торговых value objects."""

    @abstractmethod
    def calculate_slippage(self, target_price: "CurrencyValueObject") -> Decimal:
        """Рассчитывает проскальзывание."""
        ...

    @abstractmethod
    def calculate_fee(self, fee_rate: Decimal) -> "CurrencyValueObject":
        """Рассчитывает комиссию."""
        ...

    @abstractmethod
    def is_valid_for_trading(self) -> bool:
        """Проверяет пригодность для торговли."""
        ...


class ValueObjectDict(Dict[str, Any]):
    """Словарь для сериализации value objects."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._validate()

    def _validate(self) -> None:
        """Валидирует содержимое словаря."""
        required_keys = {"type", "value"}
        if not required_keys.issubset(self.keys()):
            raise ValueError(
                f"Missing required keys: {required_keys - set(self.keys())}"
            )


@dataclass(frozen=True)
class ValidationResult:
    """Результат валидации value object."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.is_valid and not self.errors


@dataclass(frozen=True)
class ConversionResult:
    """Результат конвертации валют."""

    original: CurrencyValueObject
    converted: CurrencyValueObject
    rate: ExchangeRate
    timestamp: TimestampValue
    fee: Optional["CurrencyValueObject"] = None

    @property
    def is_successful(self) -> bool:
        return self.converted is not None


class PrecisionMode(Enum):
    """Режимы точности для округления."""

    TRADING = auto()
    DISPLAY = auto()
    STORAGE = auto()
    CALCULATION = auto()


@dataclass
class PrecisionConfig:
    """Конфигурация точности для value objects."""

    mode: PrecisionMode
    decimal_places: int
    rounding_mode: str = "ROUND_HALF_UP"

    def get_rounding_mode(self) -> Any:
        """Возвращает режим округления."""
        return getattr(Decimal, self.rounding_mode)


class ValueObjectFactory(Protocol):
    """Протокол для фабрики value objects."""

    @abstractmethod
    def create_money(self, amount: NumericType, currency: CurrencyCode) -> "CurrencyValueObject":
        """Создает Money value object."""
        ...

    @abstractmethod
    def create_price(
        self,
        amount: NumericType,
        base_currency: CurrencyCode,
        quote_currency: CurrencyCode,
    ) -> "CurrencyValueObject":
        """Создает Price value object."""
        ...

    @abstractmethod
    def create_volume(self, amount: NumericType, currency: CurrencyCode) -> "CurrencyValueObject":
        """Создает Volume value object."""
        ...

    @abstractmethod
    def create_percentage(self, value: NumericType) -> "NumericValueObject":
        """Создает Percentage value object."""
        ...

    @abstractmethod
    def create_timestamp(self, value: Union[datetime, str, int]) -> "TimeValueObject":
        """Создает Timestamp value object."""
        ...

    @abstractmethod
    def create_signal(
        self, signal_type: str, strength: Decimal, metadata: Dict[str, Any]
    ) -> "ValueObject":
        """Создает Signal value object."""
        ...

    @abstractmethod
    def create_trading_pair(
        self, base: CurrencyCode, quote: CurrencyCode
    ) -> "ValueObject":
        """Создает TradingPair value object."""
        ...


# Типы для событий
class ValueObjectEvent(Enum):
    """События value objects."""

    CREATED = auto()
    UPDATED = auto()
    VALIDATED = auto()
    CONVERTED = auto()
    CACHED = auto()
    ERROR = auto()


@dataclass
class ValueObjectEventData:
    """Данные события value object."""

    event_type: ValueObjectEvent
    value_object: ValueObject
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.name,
            "value_object_type": type(self.value_object).__name__,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# Типы для мониторинга
@dataclass
class ValueObjectMetrics:
    """Метрики для мониторинга value objects."""

    total_created: int = 0
    total_validated: int = 0
    total_converted: int = 0
    total_errors: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_creation_time: float = 0.0
    average_validation_time: float = 0.0

    def update_creation_time(self, time_ms: float) -> None:
        """Обновляет среднее время создания."""
        self.total_created += 1
        self.average_creation_time = (
            self.average_creation_time * (self.total_created - 1) + time_ms
        ) / self.total_created

    def update_validation_time(self, time_ms: float) -> None:
        """Обновляет среднее время валидации."""
        self.total_validated += 1
        self.average_validation_time = (
            self.average_validation_time * (self.total_validated - 1) + time_ms
        ) / self.total_validated

    def to_dict(self) -> Dict[str, Union[int, float]]:
        return {
            "total_created": self.total_created,
            "total_validated": self.total_validated,
            "total_converted": self.total_converted,
            "total_errors": self.total_errors,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "average_creation_time_ms": self.average_creation_time,
            "average_validation_time_ms": self.average_validation_time,
            "cache_hit_rate": self.cache_hits
            / max(1, self.cache_hits + self.cache_misses),
        }


# Утилиты для работы с типами
def validate_numeric(
    value: NumericType,
    min_value: Optional[Decimal] = None,
    max_value: Optional[Decimal] = None,
) -> ValidationResult:
    """Валидирует числовое значение."""
    errors = []
    warnings = []
    try:
        decimal_value = Decimal(str(value))
        if min_value is not None and decimal_value < min_value:
            errors.append(f"Value {decimal_value} is less than minimum {min_value}")
        if max_value is not None and decimal_value > max_value:
            errors.append(f"Value {decimal_value} is greater than maximum {max_value}")
        if decimal_value == 0:
            warnings.append("Value is zero")
        return ValidationResult(
            is_valid=len(errors) == 0, errors=errors, warnings=warnings
        )
    except (ValueError, TypeError) as e:
        return ValidationResult(is_valid=False, errors=[f"Invalid numeric value: {e}"])


def round_decimal(
    value: Decimal, precision: int, mode: str = "ROUND_HALF_UP"
) -> Decimal:
    """Округляет Decimal с заданной точностью."""
    import decimal

    rounding_mode = getattr(decimal, mode)
    return value.quantize(Decimal(10) ** -precision, rounding=rounding_mode)


def generate_cache_key(value_object_type: str, **kwargs: Any) -> CacheKey:
    """Генерирует ключ кэша для value object."""
    key_data = {"type": value_object_type, **kwargs}
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return CacheKey(hashlib.md5(key_string.encode()).hexdigest())


@runtime_checkable
class TimeValueObject(ValueObject, Protocol):
    """Протокол для временных value objects."""

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Возвращает временную метку."""
        ...

    @abstractmethod
    def is_future(self) -> bool:
        """Проверяет, что время в будущем."""
        ...

    @abstractmethod
    def is_past(self) -> bool:
        """Проверяет, что время в прошлом."""
        ...

    @abstractmethod
    def time_difference(self, other: "TimeValueObject") -> float:
        """Вычисляет разность времени в секундах."""
        return 0.0


@runtime_checkable
class TimeValueObjectProtocol(TimeValueObject, Protocol):
    """Алиас для TimeValueObject для обратной совместимости."""

    pass


@dataclass
class ValidationContext:
    """Контекст валидации для value objects."""

    strict_mode: bool = True
    allow_negative: bool = False
    precision_mode: PrecisionMode = PrecisionMode.TRADING
    validation_rules: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strict_mode": self.strict_mode,
            "allow_negative": self.allow_negative,
            "precision_mode": self.precision_mode.name,
            "validation_rules": self.validation_rules,
        }


@dataclass
class ValueObjectConfig:
    """Конфигурация для value objects."""

    precision: Dict[str, int] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    cache_settings: Dict[str, Any] = field(default_factory=dict)
    monitoring_enabled: bool = True
    strict_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "validation_rules": self.validation_rules,
            "cache_settings": self.cache_settings,
            "monitoring_enabled": self.monitoring_enabled,
            "strict_mode": self.strict_mode,
        }


class ErrorCode(Enum):
    """Коды ошибок для value objects."""

    INVALID_VALUE = "INVALID_VALUE"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    CONVERSION_ERROR = "CONVERSION_ERROR"
    CACHE_ERROR = "CACHE_ERROR"
    PRECISION_ERROR = "PRECISION_ERROR"
    CURRENCY_ERROR = "CURRENCY_ERROR"
    TIMESTAMP_ERROR = "TIMESTAMP_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class ErrorMessage:
    """Сообщение об ошибке."""

    code: ErrorCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class ValueObjectError(Exception):
    """Исключение для value objects."""

    def __init__(self, error_message: ErrorMessage):
        self.error_message = error_message
        super().__init__(error_message.message)

    def to_dict(self) -> Dict[str, Any]:
        return self.error_message.to_dict()


# Экспорт всех типов
__all__ = [
    # Базовые типы
    "AmountType",
    "CurrencyCode",
    "TimestampValue",
    "PercentageValue",
    "NumericType",
    "PositiveNumeric",
    "NonNegativeNumeric",
    "StrictPositiveNumeric",
    "CurrencyPair",
    "ExchangeRate",
    "PriceLevel",
    "VolumeAmount",
    "MoneyAmount",
    "SignalId",
    "SignalScore",
    "OrderId",
    "PositionId",
    # Константы
    "MONEY_PRECISION",
    "PRICE_PRECISION",
    "VOLUME_PRECISION",
    "PERCENTAGE_PRECISION",
    "MAX_MONEY_AMOUNT",
    "MIN_MONEY_AMOUNT",
    "MAX_PRICE",
    "MIN_PRICE",
    "MAX_VOLUME",
    "MIN_VOLUME",
    "MAX_PERCENTAGE",
    "MIN_PERCENTAGE",
    # Кэширование
    "CacheKey",
    "CacheEntry",
    "CacheStats",
    # Протоколы
    "ValueObject",
    "CurrencyValueObject",
    "NumericValueObject",
    "TradingValueObject",
    "ValueObjectFactory",
    # Структуры данных
    "ValueObjectDict",
    "ValidationResult",
    "ConversionResult",
    "PrecisionMode",
    "PrecisionConfig",
    "ValueObjectEvent",
    "ValueObjectEventData",
    "ValueObjectMetrics",
    # Утилиты
    "validate_numeric",
    "round_decimal",
    "generate_cache_key",
    # Новые типы
    "ValidationContext",
    "ValueObjectConfig",
    "ErrorCode",
    "ErrorMessage",
    "ValueObjectError",
]
