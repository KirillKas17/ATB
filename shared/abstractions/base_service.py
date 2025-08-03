"""
Базовый класс для сервисов.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, Generic, List, Optional, TypeVar

from domain.exceptions import DomainException

T = TypeVar("T")


class BaseService(ABC, Generic[T]):
    """Базовый класс для всех сервисов."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        pass

    @abstractmethod
    def process(self, data: T) -> Any:
        """Обработка данных."""
        pass

    def get_cached_value(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        return self._cache.get(key)

    def set_cached_value(self, key: str, value: Any) -> None:
        """Установка значения в кэш."""
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()

    def get_config(self, key: str, default: Any = None) -> Any:
        """Получение конфигурации."""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Установка конфигурации."""
        self._config[key] = value

    def validate_decimal_range(
        self, value: Decimal, min_value: Decimal, max_value: Decimal, field_name: str
    ) -> bool:
        """Валидация диапазона Decimal значений."""
        if value < min_value or value > max_value:
            raise DomainException(
                f"{field_name} must be between {min_value} and {max_value}, got {value}"
            )
        return True

    def validate_string_length(
        self, value: str, min_length: int, max_length: int, field_name: str
    ) -> bool:
        """Валидация длины строки."""
        if len(value) < min_length or len(value) > max_length:
            raise DomainException(
                f"{field_name} length must be between {min_length} and {max_length}, got {len(value)}"
            )
        return True

    def validate_list_not_empty(self, value: List[Any], field_name: str) -> bool:
        """Валидация непустого списка."""
        if not value:
            raise DomainException(f"{field_name} cannot be empty")
        return True

    def safe_divide(
        self, numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
    ) -> Decimal:
        """Безопасное деление с дефолтным значением."""
        if denominator == 0:
            return default
        return numerator / denominator

    def calculate_percentage(self, part: Decimal, total: Decimal) -> Decimal:
        """Расчет процента."""
        if total == 0:
            return Decimal("0")
        return (part / total) * Decimal("100")

    def round_decimal(self, value: Decimal, places: int = 8) -> Decimal:
        """Округление Decimal с заданной точностью."""
        return round(value, places)

    def format_money(self, amount: Decimal, currency: str = "USDT") -> str:
        """Форматирование денежной суммы."""
        return f"{amount:.8f} {currency}"

    def log_operation(self, operation: str, details: Dict[str, Any]) -> None:
        """Логирование операции."""
        # Здесь можно добавить интеграцию с системой логирования
        print(f"Operation: {operation}, Details: {details}")

    def handle_error(self, error: Exception, context: str) -> None:
        """Обработка ошибок."""
        # Здесь можно добавить интеграцию с системой мониторинга ошибок
        print(f"Error in {context}: {str(error)}")

    def validate_required_fields(
        self, data: Dict[str, Any], required_fields: List[str]
    ) -> bool:
        """Валидация обязательных полей."""
        for field in required_fields:
            if field not in data or data[field] is None:
                raise DomainException(f"Required field '{field}' is missing or null")
        return True

    def sanitize_input(self, value: str) -> str:
        """Очистка входных данных."""
        if not isinstance(value, str):
            return str(value)
        return value.strip()

    def normalize_decimal(self, value: Any, precision: int = 8) -> Decimal:
        """Нормализация Decimal значения."""
        if isinstance(value, Decimal):
            return self.round_decimal(value, precision)
        elif isinstance(value, (int, float)):
            return self.round_decimal(Decimal(str(value)), precision)
        elif isinstance(value, str):
            return self.round_decimal(Decimal(value), precision)
        else:
            raise DomainException(f"Cannot convert {type(value)} to Decimal")

    def is_valid_uuid(self, value: str) -> bool:
        """Проверка валидности UUID."""
        import re

        uuid_pattern = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            re.IGNORECASE,
        )
        return bool(uuid_pattern.match(value))

    def generate_id(self) -> str:
        """Генерация уникального ID."""
        from uuid import uuid4

        return str(uuid4())

    def merge_configs(
        self, base_config: Dict[str, Any], override_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Объединение конфигураций."""
        result = base_config.copy()
        result.update(override_config)
        return result

    def get_nested_value(
        self, data: Dict[str, Any], path: str, default: Any = None
    ) -> Any:
        """Получение вложенного значения по пути."""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def set_nested_value(self, data: Dict[str, Any], path: str, value: Any) -> None:
        """Установка вложенного значения по пути."""
        keys = path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
