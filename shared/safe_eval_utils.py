"""
Безопасные утилиты для работы с eval операциями и парсингом данных.
"""

import ast
import json
import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union, cast

logger = logging.getLogger(__name__)

# Максимальный размер строки для парсинга (защита от DoS)
MAX_PARSE_SIZE = 10000


def safe_literal_eval(data: Any, default: Any = None, max_size: int = MAX_PARSE_SIZE) -> Any:
    """
    Безопасная альтернатива ast.literal_eval с валидацией.
    
    Args:
        data: Данные для парсинга (ожидается строка)
        default: Значение по умолчанию при ошибке
        max_size: Максимальный размер входной строки
        
    Returns:
        Распарсенное значение или default при ошибке
    """
    if not isinstance(data, str):
        logger.warning(f"Expected string input, got {type(data)}")
        return default
    
    if len(data) > max_size:
        logger.error(f"Input string too large: {len(data)} > {max_size}")
        return default
    
    # Проверка на подозрительные конструкции
    suspicious_patterns = [
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr"
    ]
    
    data_lower = data.lower()
    for pattern in suspicious_patterns:
        if pattern in data_lower:
            logger.error(f"Suspicious pattern '{pattern}' found in input data")
            return default
    
    try:
        # Пытаемся сначала JSON (быстрее и безопаснее)
        if data.strip().startswith(('{', '[', '"', "'")):
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                pass  # Попробуем ast.literal_eval
        
        # Используем ast.literal_eval как fallback
        result = ast.literal_eval(data)
        
        # Дополнительная валидация результата
        if isinstance(result, dict) and len(str(result)) > max_size:
            logger.error("Parsed dict too large")
            return default
            
        return result
        
    except (ValueError, SyntaxError, TypeError) as e:
        logger.warning(f"Failed to parse data safely: {e}")
        return default
    except Exception as e:
        logger.error(f"Unexpected error during parsing: {e}")
        return default


def safe_metadata_parse(metadata_str: Optional[str], default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Безопасный парсинг метаданных.
    
    Args:
        metadata_str: Строка с метаданными
        default: Значение по умолчанию
        
    Returns:
        Словарь с метаданными
    """
    if default is None:
        default = {}
    
    if not metadata_str or not isinstance(metadata_str, str):
        return default
    
    result = safe_literal_eval(metadata_str, default)
    
    # Убеждаемся, что результат - это словарь
    if not isinstance(result, dict):
        logger.warning(f"Expected dict from metadata, got {type(result)}")
        return default
    
    return result


def safe_numeric_convert(value: Any, target_type: type = Decimal) -> Union[Decimal, int, float, None]:
    """
    Безопасное преобразование в числовой тип.
    
    Args:
        value: Значение для преобразования
        target_type: Целевой тип (Decimal, int, float)
        
    Returns:
        Преобразованное значение или None при ошибке
    """
    if value is None:
        return None
    
    # Если уже нужный тип
    if isinstance(value, target_type):
<<<<<<< HEAD
        if target_type == Decimal:
            return cast(Decimal, value)
        elif target_type == int:
            return cast(int, value)
        elif target_type == float:
            return cast(float, value)
        else:
            return None
=======
        return value  # type: ignore[return-value]
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3
    
    try:
        if target_type == Decimal:
            if isinstance(value, (int, float)):
                return Decimal(str(value))
            elif isinstance(value, str):
                # Валидация строки перед преобразованием
                if len(value) > 50:  # Разумный лимит для числа
                    logger.error(f"Numeric string too long: {len(value)}")
                    return None
                return Decimal(value)
            else:
                # Для других типов пытаемся преобразовать через строку
                return Decimal(str(value))
        elif target_type == int:
            return int(float(value))
        elif target_type == float:
            return float(value)
        else:
<<<<<<< HEAD
            # Для других типов возвращаем None
            logger.warning(f"Unsupported target type: {target_type}")
            return None
=======
            return target_type(value)  # type: ignore[no-any-return]
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3
            
    except (ValueError, TypeError, OverflowError) as e:
        logger.warning(f"Failed to convert {value} to {target_type}: {e}")
        return None
    
    # Резервный возврат если ни одна ветка не сработала
    return None


<<<<<<< HEAD
def validate_dict_structure(data: Dict[str, Any], required_keys: Optional[List[str]] = None, allowed_keys: Optional[List[str]] = None) -> bool:
=======
def validate_dict_structure(data: Any, required_keys: list = None, allowed_keys: list = None) -> bool:
>>>>>>> e90116ec91962c532322996f100c96a0340b15b3
    """
    Валидация структуры словаря.
    
    Args:
        data: Словарь для валидации
        required_keys: Обязательные ключи
        allowed_keys: Разрешенные ключи
        
    Returns:
        True если структура валидна
    """
    if not isinstance(data, dict):
        return False
    
    if required_keys:
        missing_keys = set(required_keys) - set(data.keys())
        if missing_keys:
            logger.warning(f"Missing required keys: {missing_keys}")
            return False
    
    if allowed_keys:
        invalid_keys = set(data.keys()) - set(allowed_keys)
        if invalid_keys:
            logger.warning(f"Invalid keys found: {invalid_keys}")
            return False
    
    return True


def sanitize_for_logging(data: Any, max_length: int = 200) -> str:
    """
    Безопасная подготовка данных для логирования.
    
    Args:
        data: Данные для логирования
        max_length: Максимальная длина строки
        
    Returns:
        Безопасная строка для логирования
    """
    try:
        str_data = str(data)
        if len(str_data) > max_length:
            return str_data[:max_length] + "... (truncated)"
        return str_data
    except Exception:
        return "<unable to convert to string>"