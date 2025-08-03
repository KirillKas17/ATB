"""
Централизованный обработчик ошибок для Syntra.
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from functools import wraps
from datetime import datetime, timedelta
from enum import Enum

# Настройка безопасного логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class ErrorSeverity(Enum):
    """Уровни критичности ошибок."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Категории ошибок."""
    NETWORK = "network"
    DATABASE = "database"
    VALIDATION = "validation"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_API = "external_api"
    SYSTEM = "system"


class ErrorHandler:
    """Продвинутый обработчик ошибок с метриками и алертингом."""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        
    def log_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                  category: ErrorCategory = ErrorCategory.SYSTEM,
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Логирование ошибки с контекстом."""
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "severity": severity.value,
            "category": category.value,
            "context": context or {}
        }
        
        # Обновление счетчиков
        error_key = f"{category.value}_{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Добавление в историю
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history_size:
            self.error_history.pop(0)
        
        # Логирование
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]
        
        logger.log(log_level, f"[{category.value.upper()}] {type(error).__name__}: {error}")
        
        # Алерт для критических ошибок
        if severity == ErrorSeverity.CRITICAL:
            self._send_critical_alert(error_info)
    
    def _send_critical_alert(self, error_info: Dict[str, Any]) -> None:
        """Отправка критического алерта."""
        try:
            # Здесь можно интегрироваться с системой алертинга
            logger.critical(f"CRITICAL ALERT: {error_info}")
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Получение статистики ошибок."""
        total_errors = sum(self.error_counts.values())
        recent_errors = len([e for e in self.error_history 
                           if datetime.fromisoformat(e["timestamp"]) > datetime.now() - timedelta(hours=1)])
        
        return {
            "total_errors": total_errors,
            "recent_errors_1h": recent_errors,
            "error_counts": self.error_counts.copy(),
            "most_frequent": max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else None
        }


# Глобальный обработчик ошибок
global_error_handler = ErrorHandler()


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0,
                    error_types: Tuple[Type[Exception], ...] = (Exception,),
                    async_mode: bool = False):
    """
    Декоратор для повтора операций при ошибках.
    ИСПРАВЛЕНО: Правильная обработка асинхронных и синхронных функций.
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_error = None
                current_delay = delay
                
                for attempt in range(max_retries):
                    try:
                        return await func(*args, **kwargs)
                    except error_types as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Async attempt {attempt + 1} failed, retrying in {current_delay}s: {e}"
                            )
                            # ИСПРАВЛЕНО: Используем asyncio.sleep вместо time.sleep
                            await asyncio.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(f"All {max_retries} async attempts failed")
                
                global_error_handler.log_error(last_error, ErrorSeverity.HIGH, ErrorCategory.SYSTEM)
                raise last_error
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_error = None
                current_delay = delay
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except error_types as e:
                        last_error = e
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"Sync attempt {attempt + 1} failed, retrying in {current_delay}s: {e}"
                            )
                            # Для синхронных функций time.sleep допустим
                            time.sleep(current_delay)
                            current_delay *= backoff_factor
                        else:
                            logger.error(f"All {max_retries} sync attempts failed")
                
                global_error_handler.log_error(last_error, ErrorSeverity.HIGH, ErrorCategory.SYSTEM)
                raise last_error
            
            return sync_wrapper
    
    return decorator


def handle_exceptions(error_types: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                     category: ErrorCategory = ErrorCategory.SYSTEM,
                     return_value: Any = None,
                     reraise: bool = True):
    """Декоратор для обработки исключений с логированием."""
    if not isinstance(error_types, tuple):
        error_types = (error_types,)
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except error_types as e:
                    global_error_handler.log_error(e, severity, category, {
                        "function": func.__name__,
                        "args": str(args)[:100],  # Ограничиваем размер
                        "kwargs": str(kwargs)[:100]
                    })
                    
                    if reraise:
                        raise
                    return return_value
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except error_types as e:
                    global_error_handler.log_error(e, severity, category, {
                        "function": func.__name__,
                        "args": str(args)[:100],
                        "kwargs": str(kwargs)[:100]
                    })
                    
                    if reraise:
                        raise
                    return return_value
            
            return sync_wrapper
    
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, 
                log_errors: bool = True, **kwargs) -> Any:
    """Безопасное выполнение функции с обработкой ошибок."""
    try:
        if asyncio.iscoroutinefunction(func):
            # Для асинхронных функций возвращаем корутину
            return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            global_error_handler.log_error(e, ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM, {
                "function": getattr(func, '__name__', 'unknown'),
                "safe_execute": True
            })
        return default_return


class CircuitBreaker:
    """Упрощенный Circuit Breaker для предотвращения каскадных сбоев."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Проверка возможности выполнения операции."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - (self.last_failure_time or 0) > self.recovery_timeout:
                self.state = "half-open"
                return True
            return False
        else:  # half-open
            return True
    
    def record_success(self) -> None:
        """Запись успешной операции."""
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Запись неудачной операции."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
        elif self.state == "half-open":
            self.state = "open"


def circuit_breaker_decorator(circuit_breaker: CircuitBreaker):
    """Декоратор Circuit Breaker."""
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                if not circuit_breaker.can_execute():
                    raise RuntimeError("Circuit breaker is open")
                
                try:
                    result = await func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    circuit_breaker.record_failure()
                    raise
            
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                if not circuit_breaker.can_execute():
                    raise RuntimeError("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    circuit_breaker.record_success()
                    return result
                except Exception as e:
                    circuit_breaker.record_failure()
                    raise
            
            return sync_wrapper
    
    return decorator


# Экспорт основных компонентов
__all__ = [
    'ErrorHandler', 'ErrorSeverity', 'ErrorCategory',
    'global_error_handler', 'retry_on_failure', 'handle_exceptions',
    'safe_execute', 'CircuitBreaker', 'circuit_breaker_decorator'
]
