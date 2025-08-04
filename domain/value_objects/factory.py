"""
Промышленная фабрика для создания Value Objects с валидацией, кэшированием и расширенной функциональностью.
"""

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, ClassVar, Dict, Optional, Type, TypeVar, Union, List

from domain.type_definitions.value_object_types import (
    CacheKey,
    CurrencyPair,
    ErrorCode,
    ErrorMessage,
    ValidationContext,
    ValueObject,
    ValueObjectConfig,
    ValueObjectDict,
    ValueObjectError,
)

from .currency import Currency
from .money import Money
from .percentage import Percentage
from .price import Price
from .signal import Signal, SignalDirection
from .signal_config import SignalStrength, SignalType
from .timestamp import Timestamp
from .trading_pair import TradingPair
from .volume import Volume

T = TypeVar("T", bound=ValueObject)

logger = logging.getLogger(__name__)


class ValueObjectFactory:
    """
    Промышленная фабрика для создания Value Objects.

    Поддерживает:
    - Создание из различных типов данных с валидацией
    - Кэширование часто используемых объектов
    - Сериализацию/десериализацию
    - Мониторинг производительности
    - Обработку ошибок и логирование
    """

    # Константы для кэширования
    DEFAULT_CACHE_SIZE: ClassVar[int] = 1000
    DEFAULT_CACHE_TTL: ClassVar[int] = 3600  # 1 час

    # Константы для валидации
    STRICT_MODE: ClassVar[bool] = True
    ALLOW_NEGATIVE: ClassVar[bool] = False

    def __init__(self, config: Optional[ValueObjectConfig] = None) -> None:
        """
        Инициализация фабрики.

        Args:
            config: Конфигурация фабрики
        """
        self._registry: Dict[str, Type[Any]] = {
            "Money": Money,
            "Price": Price,
            "Volume": Volume,
            "Timestamp": Timestamp,
            "Currency": Currency,
            "Percentage": Percentage,
            "Signal": Signal,
            "TradingPair": TradingPair,
        }

        self._cache: Dict[CacheKey, CacheEntry] = {}
        self._config = config or self._get_default_config()
        self._error_count = 0
        self._success_count = 0

        logger.info(
            "ValueObjectFactory initialized with %d registered types",
            len(self._registry),
        )

    def _get_default_config(self) -> ValueObjectConfig:
        """Получение конфигурации по умолчанию."""
        return ValueObjectConfig(
            precision={"price": 8, "volume": 8, "money": 8, "percentage": 4},
            validation_rules={
                "strict_mode": self.STRICT_MODE,
                "allow_negative": self.ALLOW_NEGATIVE,
            },
            cache_settings={
                "max_size": self.DEFAULT_CACHE_SIZE,
                "ttl_seconds": self.DEFAULT_CACHE_TTL,
            },
            monitoring_enabled=True,
            strict_mode=True,
        )

    def create_money(
        self,
        amount: Union[int, float, Decimal, str],
        currency: Union[Currency, str],
        validation_context: Optional[ValidationContext] = None,
    ) -> Money:
        """Создание Money value object."""
        try:
            if isinstance(currency, str):
                currency_obj = Currency.from_string(currency)
                if not currency_obj:
                    raise ValueError(f"Invalid currency: {currency}")
            else:
                currency_obj = currency

            if isinstance(amount, str):
                amount_decimal = Decimal(amount.replace(",", ""))
            else:
                amount_decimal = Decimal(str(amount))

            # Валидация в контексте
            if validation_context and validation_context.strict_mode:
                if not validation_context.allow_negative and amount_decimal < 0:
                    raise ValueError("Negative amounts not allowed in strict mode")

            money = Money(amount_decimal, currency_obj)
            self._success_count += 1
            return money

        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create Money: %s", str(e))
            raise

    def create_price(
        self,
        amount: Union[int, float, Decimal, str],
        currency: Union[Currency, str],
        validation_context: Optional[ValidationContext] = None,
    ) -> Price:
        """Создание Price value object."""
        try:
            if isinstance(currency, str):
                currency_obj = Currency.from_string(currency)
                if not currency_obj:
                    raise ValueError(f"Invalid currency: {currency}")
            else:
                currency_obj = currency

            if isinstance(amount, str):
                amount_decimal = Decimal(amount.replace(",", ""))
            else:
                amount_decimal = Decimal(str(amount))

            # Валидация в контексте
            if validation_context and validation_context.strict_mode:
                if amount_decimal < 0:
                    raise ValueError("Negative prices not allowed in strict mode")

            price = Price(amount_decimal, currency_obj)
            self._success_count += 1
            return price

        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create Price: %s", str(e))
            raise

    def create_volume(
        self,
        value: Union[int, float, Decimal, str],
        currency: Optional[Union[Currency, str]] = None,
        validation_context: Optional[ValidationContext] = None,
    ) -> Volume:
        """Создание Volume value object."""
        try:
            if isinstance(currency, str):
                currency_obj = Currency.from_string(currency)
            else:
                currency_obj = currency

            # Если валюта не указана, используем USD по умолчанию
            if currency_obj is None:
                currency_obj = Currency.USD

            if isinstance(value, str):
                value_decimal = Decimal(value.replace(",", ""))
            else:
                value_decimal = Decimal(str(value))

            # Валидация в контексте
            if validation_context and validation_context.strict_mode:
                if value_decimal < 0:
                    raise ValueError("Negative volumes not allowed in strict mode")

            volume = Volume(value_decimal, currency_obj)
            self._success_count += 1
            return volume

        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create Volume: %s", str(e))
            raise

    def create_percentage(
        self,
        value: Union[int, float, Decimal, str],
        validation_context: Optional[ValidationContext] = None,
    ) -> Percentage:
        """Создание Percentage value object."""
        try:
            if isinstance(value, str):
                # Убираем символ % если есть
                clean_value = value.replace("%", "").strip()
                value_decimal = Decimal(clean_value)
            else:
                value_decimal = Decimal(str(value))
            # Валидация в контексте
            max_precision = None
            if (
                validation_context
                and hasattr(validation_context, 'validation_rules')
                and "max_precision" in validation_context.validation_rules
            ):
                max_precision = validation_context.validation_rules["max_precision"]
            if max_precision:
                value_decimal = value_decimal.quantize(
                    Decimal(f"0.{'0' * max_precision}")
                )
            percentage = Percentage(value_decimal)
            self._success_count += 1
            return percentage
        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create Percentage: %s", str(e))
            raise

    def create_timestamp(
        self,
        value: Union[str, int, float, datetime],
        validation_context: Optional[ValidationContext] = None,
    ) -> Timestamp:
        """Создание Timestamp value object."""
        try:
            if isinstance(value, datetime):
                timestamp = Timestamp.from_datetime(value)
            elif isinstance(value, str):
                timestamp = Timestamp.from_iso(value)
            elif isinstance(value, int):
                if value > 1e10:  # Предполагаем, что это миллисекунды
                    timestamp = Timestamp.from_unix_ms(value)
                else:
                    timestamp = Timestamp.from_unix(value)
            else:
                timestamp = Timestamp.from_unix(int(value))

            self._success_count += 1
            return timestamp

        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create Timestamp: %s", str(e))
            raise

    def create_trading_pair(
        self,
        base_currency: Union[Currency, str],
        quote_currency: Union[Currency, str],
        symbol: Optional[str] = None,
        exchange: Optional[str] = None,
        validation_context: Optional[ValidationContext] = None,
    ) -> TradingPair:
        """Создание TradingPair value object."""
        try:
            if isinstance(base_currency, str):
                base_currency_obj = Currency.from_string(base_currency)
                if not base_currency_obj:
                    raise ValueError(f"Invalid base currency: {base_currency}")
            else:
                base_currency_obj = base_currency

            if isinstance(quote_currency, str):
                quote_currency_obj = Currency.from_string(quote_currency)
                if not quote_currency_obj:
                    raise ValueError(f"Invalid quote currency: {quote_currency}")
            else:
                quote_currency_obj = quote_currency

            trading_pair = TradingPair(
                base_currency_obj, quote_currency_obj, symbol, exchange
            )
            self._success_count += 1
            return trading_pair

        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create TradingPair: %s", str(e))
            raise

    def create_signal(
        self,
        signal_type: Union[str, SignalType],
        timestamp: Union[Timestamp, str, int, datetime],
        strength: Union[str, SignalStrength] = "MODERATE",
        validation_context: Optional[ValidationContext] = None,
        **kwargs: Any,
    ) -> Signal:
        """Создание Signal value object."""
        try:
            if isinstance(signal_type, str):
                # Проверяем, есть ли метод from_string у SignalType
                if hasattr(SignalType, 'from_string'):
                    signal_direction = SignalType.from_string(signal_type)
                else:
                    # Если метода нет, используем значение по умолчанию
                    signal_direction = SignalType.TECHNICAL
            else:
                signal_direction = signal_type

            if isinstance(timestamp, (str, int, datetime)):
                timestamp_obj = self.create_timestamp(timestamp, validation_context)
            else:
                timestamp_obj = timestamp

            if isinstance(strength, str):
                if strength in SignalStrength.__members__:
                    strength_decimal = Decimal(
                        "0.7"
                    )  # Значение по умолчанию для STRONG
                else:
                    strength_decimal = Decimal(strength)
            else:
                strength_decimal = Decimal("0.7")  # Значение по умолчанию

            # Создаем временную торговую пару для совместимости
            from .currency import Currency
            from .trading_pair import TradingPair

            trading_pair = TradingPair(Currency.BTC, Currency.USDT)

            signal = Signal(
                direction=signal_direction,
                signal_type=SignalType.TECHNICAL,  # Используем правильный тип
                strength=strength_decimal,
                confidence=Decimal("0.7"),  # По умолчанию
                trading_pair=trading_pair,
                **kwargs,
            )
            self._success_count += 1
            return signal

        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create Signal: %s", str(e))
            raise

    def from_dict(self, data: Dict[str, Any]) -> ValueObject:
        """Создание value object из словаря."""
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary")
            obj_type = data.get("type")
            if not obj_type:
                raise ValueError("Dictionary must contain 'type' field")
            if obj_type not in self._registry:
                raise ValueError(f"Unknown value object type: {obj_type}")
            value_object_class = self._registry[obj_type]
            if hasattr(value_object_class, "from_dict"):
                value_object = value_object_class.from_dict(data)
                if not isinstance(value_object, ValueObject):
                    raise ValueError(f"from_dict method must return ValueObject, got {type(value_object)}")
            else:
                raise ValueError(f"Class {obj_type} does not support from_dict")
            self._success_count += 1
            return value_object
        except Exception as e:
            self._error_count += 1
            logger.error("Failed to create value object from dict: %s", str(e))
            raise

    def to_dict(self, value_object: ValueObject) -> Dict[str, Any]:
        """Преобразование value object в словарь."""
        try:
            return dict(value_object.to_dict())
        except Exception as e:
            logger.error("Failed to convert value object to dict: %s", str(e))
            raise

    def register(self, name: str, value_object_class: Type[ValueObject]) -> None:
        """Регистрация нового типа value object."""
        if not hasattr(value_object_class, "to_dict"):
            raise ValueError(
                f"Class {value_object_class} must implement to_dict method"
            )
        self._registry[name] = value_object_class
        logger.info("Registered new value object type: %s", name)

    def get_registered_types(self) -> List[str]:
        """Получение списка зарегистрированных типов."""
        return list(self._registry.keys())

    def validate(self, value_object: ValueObject) -> bool:
        """Валидация value object."""
        try:
            # Проверяем, что объект можно сериализовать и десериализовать
            data = self.to_dict(value_object)
            reconstructed = self.from_dict(data)
            return value_object == reconstructed
        except Exception as e:
            logger.warning("Validation failed for value object: %s", str(e))
            return False

    def create_from_string(self, value: str, value_type: str) -> ValueObject:
        """Создание value object из строки по типу."""
        try:
            if value_type == "Currency":
                currency = Currency.from_string(value)
                if not currency:
                    raise ValueError(f"Invalid currency string: {value}")
                return currency
            elif value_type == "Percentage":
                return self.create_percentage(value)
            elif value_type == "Timestamp":
                return self.create_timestamp(value)
            elif value_type == "TradingPair":
                return TradingPair.from_symbol(value)
            else:
                raise ValueError(
                    f"Unsupported value type for string creation: {value_type}"
                )
        except Exception as e:
            logger.error("Failed to create value object from string: %s", str(e))
            raise

    def get_cached_or_create(
        self, cache_key: str, factory_func: Callable[..., ValueObject], *args: Any, **kwargs: Any
    ) -> ValueObject:
        """Получение из кэша или создание нового объекта."""
        cache_key_obj = CacheKey(cache_key)

        # Проверяем кэш
        if cache_key_obj in self._cache:
            entry = self._cache[cache_key_obj]
            if not entry.is_expired():
                # Обновляем счетчик обращений
                self._cache[cache_key_obj] = entry.increment_access()
                return entry.value

        # Создаем новый объект
        value_object = factory_func(*args, **kwargs)

        # Добавляем в кэш
        expires_at = None
        if self._config.cache_settings.get("ttl_seconds"):
            expires_at = (
                datetime.now(timezone.utc).replace(microsecond=0).timestamp()
                + self._config.cache_settings["ttl_seconds"]
            )
            expires_at = datetime.fromtimestamp(expires_at, timezone.utc)

        cache_entry = CacheEntry(
            key=cache_key_obj,
            value=value_object,
            created_at=datetime.now(timezone.utc),
            expires_at=expires_at,
        )

        # Проверяем размер кэша
        max_size = self._config.cache_settings.get("max_size", self.DEFAULT_CACHE_SIZE)
        if len(self._cache) >= max_size:
            # Удаляем самый старый элемент
            oldest_key = min(
                self._cache.keys(), key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]

        self._cache[cache_key_obj] = cache_entry
        return value_object

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()
        logger.info("Cache cleared")

    def get_cache_size(self) -> int:
        """Получение размера кэша."""
        return len(self._cache)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        if not self._cache:
            return {"size": 0, "hit_rate": 0.0, "avg_access_count": 0.0}

        total_access = sum(entry.access_count for entry in self._cache.values())
        avg_access = total_access / len(self._cache) if self._cache else 0.0

        return {
            "size": len(self._cache),
            "avg_access_count": avg_access,
            "oldest_entry": min(entry.created_at for entry in self._cache.values()) if self._cache else None,
            "newest_entry": max(entry.created_at for entry in self._cache.values()) if self._cache else None,
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Получение статистики производительности."""
        total_operations = self._success_count + self._error_count
        success_rate = (
            (self._success_count / total_operations * 100)
            if total_operations > 0
            else 0
        )

        return {
            "total_operations": total_operations,
            "success_count": self._success_count,
            "error_count": self._error_count,
            "success_rate": success_rate,
            "cache_size": self.get_cache_size(),
        }

    def reset_stats(self) -> None:
        """Сброс статистики."""
        self._success_count = 0
        self._error_count = 0
        logger.info("Performance stats reset")


# Глобальный экземпляр фабрики
factory = ValueObjectFactory()


class CacheEntry:
    def __init__(
        self,
        key: CacheKey,
        value: ValueObject,
        created_at: datetime,
        expires_at: Optional[datetime] = None,
    ):
        self.key = key
        self.value = value
        self.created_at = created_at
        self.expires_at = expires_at
        self.access_count = 1

    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def increment_access(self) -> "CacheEntry":
        self.access_count += 1
        return self
