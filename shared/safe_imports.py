"""
Система безопасных импортов для ATB Trading System.
Обеспечивает устойчивость системы при отсутствии внешних зависимостей.
"""

import sys
import logging
import warnings
from typing import Any, Dict, List, Optional, Union, Type, Callable, TypeVar
from dataclasses import dataclass
from functools import wraps

# Настройка безопасного логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class ImportStatus:
    """Статус импорта модуля."""
    module_name: str
    is_available: bool
    fallback_used: bool
    error_message: Optional[str] = None

T = TypeVar('T')

class SafeImportManager:
    """Менеджер безопасных импортов с fallback стратегиями."""
    
    def __init__(self) -> None:
        self.import_status: Dict[str, ImportStatus] = {}
        self.fallbacks: Dict[str, Any] = {}
    
    def safe_import(self, module_name: str, fallback: Any = None, required: bool = False) -> Any:
        """Безопасный импорт модуля с fallback."""
        try:
            module = __import__(module_name)
            self.import_status[module_name] = ImportStatus(
                module_name=module_name,
                is_available=True,
                fallback_used=False
            )
            return module
        except ImportError as e:
            error_msg = f"Failed to import {module_name}: {e}"
            
            if required and fallback is None:
                logger.critical(error_msg)
                raise RuntimeError(f"Critical dependency {module_name} is missing and no fallback provided")
            
            if fallback is not None:
                logger.warning(f"{error_msg}. Using fallback.")
                self.fallbacks[module_name] = fallback
                self.import_status[module_name] = ImportStatus(
                    module_name=module_name,
                    is_available=False,
                    fallback_used=True,
                    error_message=str(e)
                )
                return fallback
            
            logger.error(error_msg)
            self.import_status[module_name] = ImportStatus(
                module_name=module_name,
                is_available=False,
                fallback_used=False,
                error_message=str(e)
            )
            return None
    
    def get_status_report(self) -> Dict[str, Any]:
        """Получить отчет о статусе всех импортов."""
        return {
            "total_imports": len(self.import_status),
            "successful": sum(1 for s in self.import_status.values() if s.is_available),
            "with_fallbacks": sum(1 for s in self.import_status.values() if s.fallback_used),
            "failed": sum(1 for s in self.import_status.values() if not s.is_available and not s.fallback_used),
            "details": {name: status for name, status in self.import_status.items()}
        }

# Глобальный менеджер импортов
import_manager = SafeImportManager()

# ===== БЕЗОПАСНЫЕ ИМПОРТЫ КРИТИЧЕСКИХ ЗАВИСИМОСТЕЙ =====

# Pandas с fallback
try:
    import pandas as pd
    DataFrame = pd.DataFrame
    Series = pd.Series
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("pandas not available. Using fallback implementations.")
    PANDAS_AVAILABLE = False
    
    class DataFrameFallback:
        """Минимальная реализация DataFrame для fallback."""
        def __init__(self, data=None, **kwargs):
            self.data = data if data is not None else {}
            self._shape = (0, 0)
        
        @property
        def shape(self):
            return self._shape
        
        def dropna(self):
            return self
        
        def set_index(self, keys):
            return self
        
        def __getitem__(self, key):
            return self.data.get(key, [])
        
        def __setitem__(self, key, value):
            self.data[key] = value
        
        def get(self, key, default=None):
            return self.data.get(key, default)
    
    DataFrame = DataFrameFallback
    Series = list
    pd = type('PandasFallback', (), {
        'DataFrame': DataFrame,
        'Series': Series,
        'read_csv': lambda *args, **kwargs: DataFrame(),
        'concat': lambda *args, **kwargs: DataFrame()
    })()

# NumPy с fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    logger.warning("numpy not available. Using fallback implementations.")
    NUMPY_AVAILABLE = False
    
    class NumpyFallback:
        """Минимальная реализация numpy для fallback."""
        
        @staticmethod
        def array(data):
            return list(data) if hasattr(data, '__iter__') else [data]
        
        @staticmethod
        def mean(data):
            if not data:
                return 0.0
            return sum(data) / len(data)
        
        @staticmethod
        def std(data):
            if not data:
                return 0.0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def corrcoef(x, y=None):
            if y is None:
                return [[1.0]]
            return [[1.0, 0.0], [0.0, 1.0]]
        
        @staticmethod
        def linspace(start, stop, num):
            if num <= 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + i * step for i in range(num)]
        
        pi = 3.14159265359
        e = 2.71828182846
    
    np = NumpyFallback()

# Логирование с fallback
try:
    from loguru import logger as loguru_logger
    LOGURU_AVAILABLE = True
    primary_logger = loguru_logger
except ImportError:
    LOGURU_AVAILABLE = False
    primary_logger = logger

# Запросы с fallback
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    logger.warning("requests not available. HTTP operations will be limited.")
    REQUESTS_AVAILABLE = False
    
    class RequestsFallback:
        """Fallback для requests."""
        
        @staticmethod
        def get(*args, **kwargs):
            raise RuntimeError("HTTP requests not available - install requests library")
        
        @staticmethod
        def post(*args, **kwargs):
            raise RuntimeError("HTTP requests not available - install requests library")
    
    requests = RequestsFallback()

# Веб-сокеты с fallback
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger.warning("websockets not available. Real-time features will be limited.")
    WEBSOCKETS_AVAILABLE = False
    websockets = None

# Криптография с fallback
try:
    from cryptography.fernet import Fernet
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    logger.warning("cryptography not available. Encryption features will be limited.")
    CRYPTOGRAPHY_AVAILABLE = False
    
    class FernetFallback:
        """Fallback для Fernet."""
        def __init__(self, key):
            self.key = key
        
        def encrypt(self, data):
            logger.warning("Encryption not available - returning plain data")
            return data
        
        def decrypt(self, data):
            logger.warning("Decryption not available - returning plain data")
            return data
    
    Fernet = FernetFallback

# Обновляем статусы
import_manager.import_status.update({
    'pandas': ImportStatus('pandas', PANDAS_AVAILABLE, not PANDAS_AVAILABLE),
    'numpy': ImportStatus('numpy', NUMPY_AVAILABLE, not NUMPY_AVAILABLE),
    'loguru': ImportStatus('loguru', LOGURU_AVAILABLE, not LOGURU_AVAILABLE),
    'requests': ImportStatus('requests', REQUESTS_AVAILABLE, not REQUESTS_AVAILABLE),
    'websockets': ImportStatus('websockets', WEBSOCKETS_AVAILABLE, not WEBSOCKETS_AVAILABLE),
    'cryptography': ImportStatus('cryptography', CRYPTOGRAPHY_AVAILABLE, not CRYPTOGRAPHY_AVAILABLE),
})

def check_dependencies() -> Dict[str, Any]:
    """Проверка всех зависимостей и возврат отчета."""
    return import_manager.get_status_report()

def require_dependency(dependency_name: str) -> bool:
    """Проверка доступности критической зависимости."""
    status = import_manager.import_status.get(dependency_name)
    if status and status.is_available:
        return True
    
    logger.error(f"Critical dependency {dependency_name} is not available")
    return False

def safe_function_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> Optional[T]:
    """Безопасный вызов функции с обработкой ошибок."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Error calling {func.__name__}: {e}")
        return None

# Декоратор для безопасного выполнения
def safe_execution(fallback_value=None):
    """Декоратор для безопасного выполнения функций."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator

# Экспорт основных компонентов
__all__ = [
    'import_manager', 'SafeImportManager', 'ImportStatus',
    'pd', 'np', 'DataFrame', 'Series', 'primary_logger',
    'requests', 'websockets', 'Fernet',
    'check_dependencies', 'require_dependency', 'safe_function_call', 'safe_execution',
    'PANDAS_AVAILABLE', 'NUMPY_AVAILABLE', 'LOGURU_AVAILABLE', 
    'REQUESTS_AVAILABLE', 'WEBSOCKETS_AVAILABLE', 'CRYPTOGRAPHY_AVAILABLE'
]