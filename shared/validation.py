"""
Универсальная система валидации входных данных для ATB Trading System.
Обеспечивает безопасность и целостность данных на всех уровнях архитектуры.
"""

import re
import math
import logging
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from dataclasses import dataclass

# Безопасная настройка логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class ValidationRule:
    """Правило валидации."""
    name: str
    validator: Callable[[Any], bool]
    error_message: str
    severity: str = "error"  # "error", "warning", "info"


@dataclass
class ValidationResult:
    """Результат валидации."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    field_name: Optional[str] = None


class InputValidator:
    """Универсальный валидатор входных данных."""
    
    def __init__(self) -> None:
        self.rules: Dict[str, List[ValidationRule]] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Настройка правил валидации по умолчанию."""
        
        # Правила для торговых символов
        self.rules["symbol"] = [
            ValidationRule(
                "format",
                lambda x: isinstance(x, str) and len(x) >= 6 and "/" in x,
                "Symbol must be a string in format 'BASE/QUOTE' (e.g., 'BTC/USDT')"
            ),
            ValidationRule(
                "uppercase",
                lambda x: x.isupper() if isinstance(x, str) else False,
                "Symbol must be uppercase"
            ),
            ValidationRule(
                "no_spaces",
                lambda x: " " not in x if isinstance(x, str) else False,
                "Symbol must not contain spaces"
            )
        ]
        
        # Правила для цен
        self.rules["price"] = [
            ValidationRule(
                "positive",
                lambda x: self._is_positive_number(x),
                "Price must be a positive number"
            ),
            ValidationRule(
                "reasonable_range",
                lambda x: self._is_in_range(x, 0.000001, 1000000000),
                "Price must be between 0.000001 and 1,000,000,000"
            )
        ]
        
        # Правила для объемов
        self.rules["volume"] = [
            ValidationRule(
                "positive",
                lambda x: self._is_positive_number(x),
                "Volume must be a positive number"
            ),
            ValidationRule(
                "not_zero",
                lambda x: self._is_not_zero(x),
                "Volume cannot be zero"
            )
        ]
        
        # Правила для временных интервалов
        self.rules["timeframe"] = [
            ValidationRule(
                "valid_format",
                lambda x: self._is_valid_timeframe(x),
                "Timeframe must be valid (e.g., '1m', '5m', '1h', '1d')"
            )
        ]
        
        # Правила для дат
        self.rules["datetime"] = [
            ValidationRule(
                "valid_datetime",
                lambda x: self._is_valid_datetime(x),
                "Must be a valid datetime object or ISO string"
            ),
            ValidationRule(
                "not_future",
                lambda x: self._is_not_future(x),
                "Datetime cannot be in the future",
                severity="warning"
            )
        ]
        
        # Правила для API ключей
        self.rules["api_key"] = [
            ValidationRule(
                "not_empty",
                lambda x: isinstance(x, str) and len(x.strip()) > 0,
                "API key cannot be empty"
            ),
            ValidationRule(
                "min_length",
                lambda x: len(x) >= 16 if isinstance(x, str) else False,
                "API key must be at least 16 characters long"
            ),
            ValidationRule(
                "no_spaces",
                lambda x: " " not in x if isinstance(x, str) else False,
                "API key must not contain spaces"
            )
        ]
    
    def _is_positive_number(self, value: Any) -> bool:
        """Проверка положительного числа."""
        try:
            num = float(value)
            return num > 0 and not math.isnan(num) and not math.isinf(num)
        except (ValueError, TypeError):
            return False
    
    def _is_not_zero(self, value: Any) -> bool:
        """Проверка, что число не равно нулю."""
        try:
            return float(value) != 0.0
        except (ValueError, TypeError):
            return False
    
    def _is_in_range(self, value: Any, min_val: float, max_val: float) -> bool:
        """Проверка попадания в диапазон."""
        try:
            num = float(value)
            return min_val <= num <= max_val
        except (ValueError, TypeError):
            return False
    
    def _is_valid_timeframe(self, value: Any) -> bool:
        """Проверка валидности временного интервала."""
        if not isinstance(value, str):
            return False
        
        # Поддерживаемые timeframe форматы
        pattern = r'^(\d+)([smhDW])$'
        match = re.match(pattern, value)
        
        if not match:
            return False
        
        number, unit = match.groups()
        number = int(number)
        
        # Проверяем разумные ограничения
        if unit == 's' and number > 3600:  # Максимум 1 час в секундах
            return False
        elif unit == 'm' and number > 1440:  # Максимум 1 день в минутах
            return False
        elif unit == 'h' and number > 168:  # Максимум 1 неделя в часах
            return False
        elif unit in ['D', 'W'] and number > 365:  # Максимум 1 год
            return False
        
        return True
    
    def _is_valid_datetime(self, value: Any) -> bool:
        """Проверка валидности даты/времени."""
        if isinstance(value, datetime):
            return True
        
        if isinstance(value, str):
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return True
            except ValueError:
                return False
        
        return False
    
    def _is_not_future(self, value: Any) -> bool:
        """Проверка, что дата не в будущем."""
        try:
            if isinstance(value, datetime):
                dt = value
            elif isinstance(value, str):
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                return False
            
            # Допускаем небольшое отклонение для учета задержек
            return dt <= datetime.now() + timedelta(minutes=1)
        except (ValueError, TypeError):
            return False
    
    def validate_field(self, field_name: str, value: Any, 
                      custom_rules: Optional[List[ValidationRule]] = None) -> ValidationResult:
        """Валидация одного поля."""
        errors = []
        warnings = []
        
        # Получаем правила для поля
        rules = custom_rules or self.rules.get(field_name, [])
        
        for rule in rules:
            try:
                if not rule.validator(value):
                    if rule.severity == "error":
                        errors.append(f"{field_name}: {rule.error_message}")
                    elif rule.severity == "warning":
                        warnings.append(f"{field_name}: {rule.error_message}")
            except Exception as e:
                errors.append(f"{field_name}: Validation error in rule '{rule.name}': {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            field_name=field_name
        )
    
    def validate_dict(self, data: Dict[str, Any], 
                     field_rules: Optional[Dict[str, List[ValidationRule]]] = None) -> ValidationResult:
        """Валидация словаря данных."""
        all_errors = []
        all_warnings = []
        
        for field_name, value in data.items():
            # Определяем правила для поля
            if field_rules and field_name in field_rules:
                rules = field_rules[field_name]
            else:
                rules = self.rules.get(field_name, [])
            
            if rules:  # Валидируем только если есть правила
                result = self.validate_field(field_name, value, rules)
                all_errors.extend(result.errors)
                all_warnings.extend(result.warnings)
        
        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )
    
    def add_rule(self, field_name: str, rule: ValidationRule):
        """Добавление пользовательского правила."""
        if field_name not in self.rules:
            self.rules[field_name] = []
        self.rules[field_name].append(rule)


# Глобальный экземпляр валидатора
_global_validator = InputValidator()


def validate_input(**field_rules):
    """
    Декоратор для валидации входных параметров функции.
    
    Args:
        **field_rules: Маппинг параметр -> тип валидации
        
    Example:
        @validate_input(symbol="symbol", price="price", volume="volume")
        async def place_order(symbol: str, price: float, volume: float):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Создаем словарь для валидации из kwargs
            validation_data = {}
            
            for param_name, validation_type in field_rules.items():
                if param_name in kwargs:
                    validation_data[validation_type] = kwargs[param_name]
            
            # Выполняем валидацию
            if validation_data:
                result = _global_validator.validate_dict(validation_data)
                
                if not result.is_valid:
                    error_msg = "; ".join(result.errors)
                    logger.error(f"Validation failed in {func.__name__}: {error_msg}")
                    raise ValueError(f"Input validation failed: {error_msg}")
                
                if result.warnings:
                    warning_msg = "; ".join(result.warnings)
                    logger.warning(f"Validation warnings in {func.__name__}: {warning_msg}")
            
            return await func(*args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Создаем словарь для валидации из kwargs
            validation_data = {}
            
            for param_name, validation_type in field_rules.items():
                if param_name in kwargs:
                    validation_data[validation_type] = kwargs[param_name]
            
            # Выполняем валидацию
            if validation_data:
                result = _global_validator.validate_dict(validation_data)
                
                if not result.is_valid:
                    error_msg = "; ".join(result.errors)
                    logger.error(f"Validation failed in {func.__name__}: {error_msg}")
                    raise ValueError(f"Input validation failed: {error_msg}")
                
                if result.warnings:
                    warning_msg = "; ".join(result.warnings)
                    logger.warning(f"Validation warnings in {func.__name__}: {warning_msg}")
            
            return func(*args, **kwargs)
        
        # Возвращаем подходящий wrapper в зависимости от типа функции
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def validate_trading_params(symbol: str, price: Optional[float] = None, 
                          volume: Optional[float] = None) -> ValidationResult:
    """Специализированная валидация торговых параметров."""
    data = {"symbol": symbol}
    
    if price is not None:
        data["price"] = price
    
    if volume is not None:
        data["volume"] = volume
    
    return _global_validator.validate_dict(data)


def validate_market_data(data: Dict[str, Any]) -> ValidationResult:
    """Валидация рыночных данных."""
    required_fields = ["timestamp", "open", "high", "low", "close", "volume"]
    errors = []
    
    # Проверяем наличие обязательных полей
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if errors:
        return ValidationResult(is_valid=False, errors=errors, warnings=[])
    
    # Валидируем типы и значения
    validation_data = {
        "datetime": data["timestamp"],
        "price": data["open"],  # Используем открытие как пример цены
        "volume": data["volume"]
    }
    
    result = _global_validator.validate_dict(validation_data)
    
    # Дополнительные проверки для OHLC данных
    try:
        o, h, l, c = float(data["open"]), float(data["high"]), float(data["low"]), float(data["close"])
        
        if not (l <= o <= h and l <= c <= h):
            result.errors.append("OHLC data is inconsistent: low <= open,close <= high")
        
        if h < l:
            result.errors.append("High price cannot be lower than low price")
            
    except (ValueError, TypeError) as e:
        result.errors.append(f"Invalid OHLC data types: {e}")
    
    result.is_valid = len(result.errors) == 0
    return result


# Экспорт основных функций для удобства использования
__all__ = [
    "InputValidator",
    "ValidationRule", 
    "ValidationResult",
    "validate_input",
    "validate_trading_params",
    "validate_market_data"
]