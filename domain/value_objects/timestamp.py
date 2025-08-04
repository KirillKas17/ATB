"""
Промышленная реализация Value Object для временных меток с расширенной функциональностью для алготрейдинга.
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Optional, Union

from domain.type_definitions.base_types import TimestampValue

from domain.type_definitions.value_object_types import (
    NumericType,
    TimeValueObject,
    TimeValueObjectProtocol,
    ValidationResult,
    ValueObject,
    ValueObjectDict,
)


class Timestamp(ValueObject):
    """
    Промышленная реализация Value Object для временных меток.
    Поддерживает:
    - Точные временные операции с timezone awareness
    - Анализ временных интервалов и паттернов
    - Торговые сессии и временные окна
    - Сериализацию/десериализацию
    """

    # Константы для временных операций
    DEFAULT_TOLERANCE_SECONDS: ClassVar[int] = 1
    MAX_TOLERANCE_SECONDS: ClassVar[int] = 3600  # 1 час
    # Константы для торговых сессий
    TRADING_SESSION_START: ClassVar[int] = 9  # 9:00
    TRADING_SESSION_END: ClassVar[int] = 17  # 17:00
    # Константы для временных интервалов
    SECONDS_PER_MINUTE: ClassVar[int] = 60
    SECONDS_PER_HOUR: ClassVar[int] = 3600
    SECONDS_PER_DAY: ClassVar[int] = 86400
    SECONDS_PER_WEEK: ClassVar[int] = 604800

    def __init__(self, value: Union[datetime, str, int, float]) -> None:
        """
        Инициализация временной метки.
        Args:
            value: Временная метка (datetime, ISO string, Unix timestamp)
        """
        self._value = self._parse_timestamp(value)
        self._validate_timestamp()

    @property
    def value(self) -> TimestampValue:
        """Получение временной метки с типизацией."""
        return TimestampValue(self._value)

    def _parse_timestamp(self, value: Union[datetime, str, int, float]) -> datetime:
        """Парсинг временной метки из различных форматов."""
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            dt = datetime.fromisoformat(value)
        elif isinstance(value, int):
            if value > 1e10:  # Предполагаем, что это миллисекунды
                dt = datetime.fromtimestamp(value / 1000, timezone.utc)
            else:
                dt = datetime.fromtimestamp(value, timezone.utc)
        elif isinstance(value, float):
            dt = datetime.fromtimestamp(int(value), timezone.utc)
        else:
            raise ValueError(f"Unsupported timestamp format: {type(value)}")
        # Убеждаемся, что время имеет timezone info
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _validate_timestamp(self) -> None:
        """Валидация временной метки."""
        if not isinstance(self._value, datetime):
            raise ValueError("Timestamp value must be a datetime object")
        if self._value.tzinfo is None:
            raise ValueError("Timestamp must have timezone information")

    def _get_equality_components(self) -> tuple:
        """Компоненты для сравнения на равенство."""
        return (self._value,)

    def __lt__(self, other: "Timestamp") -> bool:
        """Сравнение меньше."""
        if not isinstance(other, Timestamp):
            raise TypeError("Can only compare Timestamp with Timestamp")
        return self._value < other._value

    def __le__(self, other: "Timestamp") -> bool:
        """Сравнение меньше или равно."""
        return self < other or self == other

    def __gt__(self, other: "Timestamp") -> bool:
        """Сравнение больше."""
        return not self <= other

    def __ge__(self, other: "Timestamp") -> bool:
        """Сравнение больше или равно."""
        return not self < other

    def to_iso(self) -> str:
        """Преобразование в ISO формат."""
        return self._value.isoformat()

    def to_unix(self) -> int:
        """Преобразование в Unix timestamp."""
        return int(self._value.timestamp())

    def to_unix_ms(self) -> int:
        """Преобразование в Unix timestamp в миллисекундах."""
        return int(self._value.timestamp() * 1000)

    def to_datetime(self) -> datetime:
        """Получение datetime объекта."""
        return self._value

    def is_future(self) -> bool:
        """Проверка, что время в будущем."""
        return self._value > datetime.now(timezone.utc)

    def is_past(self) -> bool:
        """Проверка, что время в прошлом."""
        return self._value < datetime.now(timezone.utc)

    def is_now(self, tolerance_seconds: Optional[int] = None) -> bool:
        """Проверка, что время сейчас (с допуском)."""
        tolerance = tolerance_seconds or self.DEFAULT_TOLERANCE_SECONDS
        if tolerance > self.MAX_TOLERANCE_SECONDS:
            raise ValueError(
                f"Tolerance cannot exceed {self.MAX_TOLERANCE_SECONDS} seconds"
            )
        now = datetime.now(timezone.utc)
        diff = abs((self._value - now).total_seconds())
        return diff <= tolerance

    def add_seconds(self, seconds: NumericType) -> "Timestamp":
        """Добавление секунд."""
        return Timestamp(self._value + timedelta(seconds=float(seconds)))

    def add_minutes(self, minutes: NumericType) -> "Timestamp":
        """Добавление минут."""
        return Timestamp(self._value + timedelta(minutes=float(minutes)))

    def add_hours(self, hours: NumericType) -> "Timestamp":
        """Добавление часов."""
        return Timestamp(self._value + timedelta(hours=float(hours)))

    def add_days(self, days: NumericType) -> "Timestamp":
        """Добавление дней."""
        return Timestamp(self._value + timedelta(days=float(days)))

    def subtract_seconds(self, seconds: NumericType) -> "Timestamp":
        """Вычитание секунд."""
        return Timestamp(self._value - timedelta(seconds=float(seconds)))

    def subtract_minutes(self, minutes: NumericType) -> "Timestamp":
        """Вычитание минут."""
        return Timestamp(self._value - timedelta(minutes=float(minutes)))

    def subtract_hours(self, hours: NumericType) -> "Timestamp":
        """Вычитание часов."""
        return Timestamp(self._value - timedelta(hours=float(hours)))

    def subtract_days(self, days: NumericType) -> "Timestamp":
        """Вычитание дней."""
        return Timestamp(self._value - timedelta(days=float(days)))

    def time_difference(self, other: "Timestamp") -> float:
        """Разность времени в секундах."""
        if not isinstance(other, Timestamp):
            raise TypeError("Can only calculate time difference with Timestamp")
        return (self._value - other._value).total_seconds()

    def time_difference_minutes(self, other: "Timestamp") -> float:
        """Разность времени в минутах."""
        return self.time_difference(other) / self.SECONDS_PER_MINUTE

    def time_difference_hours(self, other: "Timestamp") -> float:
        """Разность времени в часах."""
        return self.time_difference(other) / self.SECONDS_PER_HOUR

    def time_difference_days(self, other: "Timestamp") -> float:
        """Разность времени в днях."""
        return self.time_difference(other) / self.SECONDS_PER_DAY

    def is_same_day(self, other: "Timestamp") -> bool:
        """Проверка, что временные метки в один день."""
        return self._value.date() == other._value.date()

    def is_same_hour(self, other: "Timestamp") -> bool:
        """Проверка, что временные метки в один час."""
        return (
            self._value.date() == other._value.date()
            and self._value.hour == other._value.hour
        )

    def is_same_minute(self, other: "Timestamp") -> bool:
        """Проверка, что временные метки в одну минуту."""
        return (
            self._value.date() == other._value.date()
            and self._value.hour == other._value.hour
            and self._value.minute == other._value.minute
        )

    def is_weekend(self) -> bool:
        """Проверка, что время приходится на выходные."""
        return self._value.weekday() >= 5

    def is_weekday(self) -> bool:
        """Проверка, что время приходится на рабочие дни."""
        return self._value.weekday() < 5

    def is_trading_hours(
        self, start_hour: Optional[int] = None, end_hour: Optional[int] = None
    ) -> bool:
        """Проверка, что время в торговые часы."""
        start = start_hour or self.TRADING_SESSION_START
        end = end_hour or self.TRADING_SESSION_END
        return self.is_weekday() and start <= self._value.hour < end

    def round_to_minute(self) -> "Timestamp":
        """Округление до минуты."""
        rounded = self._value.replace(second=0, microsecond=0)
        return Timestamp(rounded)

    def round_to_hour(self) -> "Timestamp":
        """Округление до часа."""
        rounded = self._value.replace(minute=0, second=0, microsecond=0)
        return Timestamp(rounded)

    def round_to_day(self) -> "Timestamp":
        """Округление до дня."""
        rounded = self._value.replace(hour=0, minute=0, second=0, microsecond=0)
        return Timestamp(rounded)

    def min(self, other: "Timestamp") -> "Timestamp":
        """Минимальное значение из двух временных меток."""
        return self if self <= other else other

    def max(self, other: "Timestamp") -> "Timestamp":
        """Максимальное значение из двух временных меток."""
        return self if self >= other else other

    def is_between(self, start: "Timestamp", end: "Timestamp") -> bool:
        """Проверка, что время находится между двумя метками."""
        return start <= self <= end

    # Методы для торгового анализа
    def get_trading_session(self) -> str:
        """
        Получение торговой сессии.
        Returns:
            Торговая сессия: "PRE_MARKET", "REGULAR", "AFTER_HOURS", "CLOSED"
        """
        if not self.is_weekday():
            return "CLOSED"
        hour = self._value.hour
        if hour < self.TRADING_SESSION_START:
            return "PRE_MARKET"
        elif hour < self.TRADING_SESSION_END:
            return "REGULAR"
        else:
            return "AFTER_HOURS"

    def is_market_open(self) -> bool:
        """
        Проверка, открыт ли рынок.
        Returns:
            True, если рынок открыт.
        """
        return self.get_trading_session() == "REGULAR"

    def get_time_until_market_open(self) -> Optional[float]:
        """
        Получение времени до открытия рынка в секундах.
        Returns:
            Время до открытия или None, если рынок уже открыт.
        """
        if self.is_market_open():
            return None
        # Находим следующее открытие рынка
        next_open = self._value.replace(
            hour=self.TRADING_SESSION_START, minute=0, second=0, microsecond=0
        )
        if self._value.hour >= self.TRADING_SESSION_START:
            # Рынок закрыт сегодня, ищем следующий рабочий день
            next_open = next_open + timedelta(days=1)
            while next_open.weekday() >= 5:  # Пропускаем выходные
                next_open = next_open + timedelta(days=1)
        return (next_open - self._value).total_seconds()

    def get_time_until_market_close(self) -> Optional[float]:
        """
        Получение времени до закрытия рынка в секундах.
        Returns:
            Время до закрытия или None, если рынок уже закрыт.
        """
        if not self.is_market_open():
            return None
        next_close = self._value.replace(
            hour=self.TRADING_SESSION_END, minute=0, second=0, microsecond=0
        )
        return (next_close - self._value).total_seconds()

    def get_age_in_seconds(self) -> float:
        """
        Получение возраста временной метки в секундах.
        Returns:
            Возраст в секундах.
        """
        return (datetime.now(timezone.utc) - self._value).total_seconds()

    def is_recent(self, max_age_seconds: int = 300) -> bool:
        """
        Проверка, что временная метка недавняя.
        Args:
            max_age_seconds: Максимальный возраст в секундах
        Returns:
            True, если метка недавняя.
        """
        return self.get_age_in_seconds() <= max_age_seconds

    def is_expired(self, max_age_seconds: int = 3600) -> bool:
        """
        Проверка, что временная метка устарела.
        Args:
            max_age_seconds: Максимальный возраст в секундах
        Returns:
            True, если метка устарела.
        """
        return not self.is_recent(max_age_seconds)

    def to_dict(self) -> ValueObjectDict:
        """Преобразование в словарь."""
        return ValueObjectDict(value=self._value.isoformat(), type="Timestamp")

    @classmethod
    def from_dict(cls, data: ValueObjectDict) -> "Timestamp":
        """Создание из словаря."""
        return cls(datetime.fromisoformat(data["value"]))

    @classmethod
    def now(cls) -> "Timestamp":
        """Создание текущего времени."""
        return cls(datetime.now(timezone.utc))

    @classmethod
    def from_iso(cls, iso_string: str) -> "Timestamp":
        """Создание из ISO строки."""
        # Обрабатываем формат с 'Z' на конце (UTC timezone)
        if iso_string.endswith("Z"):
            iso_string = iso_string[:-1] + "+00:00"
        return cls(datetime.fromisoformat(iso_string))

    @classmethod
    def from_unix(cls, unix_timestamp: int) -> "Timestamp":
        """Создание из Unix timestamp."""
        return cls(datetime.fromtimestamp(unix_timestamp, timezone.utc))

    @classmethod
    def from_unix_ms(cls, unix_timestamp_ms: int) -> "Timestamp":
        """Создание из Unix timestamp в миллисекундах."""
        return cls(datetime.fromtimestamp(unix_timestamp_ms / 1000, timezone.utc))

    @classmethod
    def from_datetime(cls, dt: datetime) -> "Timestamp":
        """Создание из datetime объекта."""
        return cls(dt)

    def copy(self) -> "Timestamp":
        """Создание копии."""
        return Timestamp(self._value)

    def __str__(self) -> str:
        """Строковое представление."""
        return self._value.isoformat()

    def __repr__(self) -> str:
        """Представление для отладки."""
        return f"Timestamp({self._value})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Timestamp):
            return False
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    @property
    def hash(self) -> str:
        """Строковый хеш для кэширования (md5 от isoformat)."""
        return hashlib.md5(self._value.isoformat().encode()).hexdigest()

    def validate(self) -> bool:
        """Валидирует value object."""
        return isinstance(self._value, datetime) and self._value.tzinfo is not None
