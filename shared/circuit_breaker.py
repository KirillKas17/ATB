"""
Advanced Circuit Breaker Pattern для ATB Trading System.
Обеспечивает защиту от каскадных отказов и автоматическое восстановление.
"""

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps

# Безопасная настройка логирования
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class CircuitState(Enum):
    """Состояния circuit breaker."""
    CLOSED = "closed"       # Нормальная работа
    OPEN = "open"          # Блокировка вызовов
    HALF_OPEN = "half_open" # Тестирование восстановления


@dataclass
class CircuitBreakerConfig:
    """Конфигурация circuit breaker."""
    failure_threshold: int = 5           # Количество ошибок для открытия
    recovery_timeout: float = 60.0       # Время до попытки восстановления (сек)
    expected_recovery_time: float = 30.0 # Ожидаемое время восстановления
    success_threshold: int = 3           # Успешных вызовов для закрытия
    timeout_duration: float = 10.0       # Таймаут выполнения операции
    monitor_window: int = 100            # Размер окна мониторинга
    slow_call_threshold: float = 5.0     # Порог медленного вызова (сек)
    slow_call_rate_threshold: float = 0.5 # Процент медленных вызовов для открытия


@dataclass
class CallResult:
    """Результат вызова функции."""
    success: bool
    duration: float
    timestamp: float
    error: Optional[Exception] = None


@dataclass
class CircuitBreakerMetrics:
    """Метрики circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    slow_calls: int = 0
    state_changes: int = 0
    last_state_change: Optional[datetime] = None
    call_history: List[CallResult] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        """Процент неудачных вызовов."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
    
    @property
    def slow_call_rate(self) -> float:
        """Процент медленных вызовов."""
        if self.total_calls == 0:
            return 0.0
        return self.slow_calls / self.total_calls
    
    @property
    def success_rate(self) -> float:
        """Процент успешных вызовов."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreakerError(Exception):
    """Исключение circuit breaker."""
    
    def __init__(self, message: str, state: CircuitState, metrics: CircuitBreakerMetrics) -> None:
        super().__init__(message)
        self.state = state
        self.metrics = metrics


class CircuitBreaker:
    """
    Продвинутый Circuit Breaker с адаптивным поведением.
    
    Особенности:
    - Мониторинг медленных вызовов
    - Адаптивные пороги
    - Детальные метрики
    - Автоматическое восстановление
    - Поддержка async/sync функций
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_failure_time = 0.0
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self._lock = asyncio.Lock()
        
        # Адаптивные пороги
        self._adaptive_failure_threshold = self.config.failure_threshold
        self._adaptive_recovery_timeout = self.config.recovery_timeout
        
        logger.info(f"Circuit breaker '{name}' initialized with config: {self.config}")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Выполнение функции через circuit breaker.
        
        Args:
            func: Функция для выполнения
            *args, **kwargs: Аргументы функции
            
        Returns:
            Результат выполнения функции
            
        Raises:
            CircuitBreakerError: При блокировке circuit breaker
        """
        async with self._lock:
            await self._check_state()
            
            if self.state == CircuitState.OPEN:
                self._log_blocked_call()
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is OPEN",
                    self.state,
                    self.metrics
                )
        
        # Выполняем вызов с мониторингом
        start_time = time.time()
        call_result = None
        
        try:
            # Добавляем таймаут
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.timeout_duration
                )
            else:
                # Для синхронных функций выполняем в executor
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: func(*args, **kwargs)
                )
            
            duration = time.time() - start_time
            call_result = CallResult(success=True, duration=duration, timestamp=start_time)
            
            await self._record_success(call_result)
            return result
            
        except asyncio.TimeoutError as e:
            duration = time.time() - start_time
            call_result = CallResult(success=False, duration=duration, timestamp=start_time, error=e)
            self.metrics.timeout_calls += 1
            await self._record_failure(call_result)
            raise
            
        except Exception as e:
            duration = time.time() - start_time
            call_result = CallResult(success=False, duration=duration, timestamp=start_time, error=e)
            await self._record_failure(call_result)
            raise
    
    async def _check_state(self) -> None:
        """Проверка и обновление состояния circuit breaker."""
        current_time = time.time()
        
        if self.state == CircuitState.OPEN:
            # Проверяем, можно ли перейти в HALF_OPEN
            if current_time - self.last_failure_time >= self._adaptive_recovery_timeout:
                await self._transition_to_half_open()
        
        elif self.state == CircuitState.HALF_OPEN:
            # В состоянии HALF_OPEN разрешаем ограниченное количество вызовов
            pass
    
    async def _record_success(self, call_result: CallResult) -> None:
        """Запись успешного вызова."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # Проверяем на медленный вызов
            if call_result.duration >= self.config.slow_call_threshold:
                self.metrics.slow_calls += 1
            
            self._add_to_history(call_result)
            
            # Переход из HALF_OPEN в CLOSED
            if (self.state == CircuitState.HALF_OPEN and 
                self.consecutive_successes >= self.config.success_threshold):
                await self._transition_to_closed()
            
            # Адаптивная корректировка порогов при стабильной работе
            if self.consecutive_successes > 20:
                self._adapt_thresholds_success()
    
    async def _record_failure(self, call_result: CallResult) -> None:
        """Запись неудачного вызова."""
        async with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            self.last_failure_time = time.time()
            
            self._add_to_history(call_result)
            
            # Проверяем необходимость открытия circuit breaker
            if self._should_open_circuit():
                await self._transition_to_open()
    
    def _should_open_circuit(self) -> bool:
        """Проверка условий для открытия circuit breaker."""
        # Проверка по количеству последовательных ошибок
        if self.consecutive_failures >= self._adaptive_failure_threshold:
            return True
        
        # Проверка по проценту ошибок в окне мониторинга
        recent_calls = self._get_recent_calls()
        if len(recent_calls) >= self.config.monitor_window:
            failed_in_window = sum(1 for call in recent_calls if not call.success)
            failure_rate = failed_in_window / len(recent_calls)
            
            if failure_rate >= 0.5:  # 50% ошибок
                return True
        
        # Проверка по проценту медленных вызовов
        if (len(recent_calls) >= self.config.monitor_window and 
            self.metrics.slow_call_rate >= self.config.slow_call_rate_threshold):
            return True
        
        return False
    
    def _get_recent_calls(self) -> List[CallResult]:
        """Получение недавних вызовов в окне мониторинга."""
        return self.metrics.call_history[-self.config.monitor_window:]
    
    def _add_to_history(self, call_result: CallResult) -> None:
        """Добавление результата в историю."""
        self.metrics.call_history.append(call_result)
        
        # Ограничиваем размер истории
        max_history = self.config.monitor_window * 2
        if len(self.metrics.call_history) > max_history:
            self.metrics.call_history = self.metrics.call_history[-max_history:]
    
    async def _transition_to_open(self) -> None:
        """Переход в состояние OPEN."""
        old_state = self.state
        self.state = CircuitState.OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        
        logger.warning(
            f"Circuit breaker '{self.name}' transitioned from {old_state.value} to OPEN. "
            f"Failure rate: {self.metrics.failure_rate:.2%}, "
            f"Consecutive failures: {self.consecutive_failures}"
        )
        
        # Адаптивная корректировка при открытии
        self._adapt_thresholds_failure()
    
    async def _transition_to_half_open(self) -> None:
        """Переход в состояние HALF_OPEN."""
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        self.consecutive_successes = 0
        
        logger.info(f"Circuit breaker '{self.name}' transitioned from {old_state.value} to HALF_OPEN")
    
    async def _transition_to_closed(self) -> None:
        """Переход в состояние CLOSED."""
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.metrics.state_changes += 1
        self.metrics.last_state_change = datetime.now()
        
        logger.info(
            f"Circuit breaker '{self.name}' transitioned from {old_state.value} to CLOSED. "
            f"Success rate: {self.metrics.success_rate:.2%}"
        )
    
    def _adapt_thresholds_success(self) -> None:
        """Адаптация порогов при успешной работе."""
        # Постепенно снижаем пороги при стабильной работе
        if self._adaptive_failure_threshold > self.config.failure_threshold:
            self._adaptive_failure_threshold = max(
                self.config.failure_threshold,
                self._adaptive_failure_threshold - 1
            )
        
        if self._adaptive_recovery_timeout > self.config.recovery_timeout:
            self._adaptive_recovery_timeout = max(
                self.config.recovery_timeout,
                self._adaptive_recovery_timeout * 0.9
            )
    
    def _adapt_thresholds_failure(self) -> None:
        """Адаптация порогов при частых сбоях."""
        # Увеличиваем пороги при частых сбоях
        self._adaptive_failure_threshold = min(
            self.config.failure_threshold * 2,
            self._adaptive_failure_threshold + 1
        )
        
        self._adaptive_recovery_timeout = min(
            self.config.recovery_timeout * 3,
            self._adaptive_recovery_timeout * 1.5
        )
    
    def _log_blocked_call(self) -> None:
        """Логирование заблокированного вызова."""
        logger.debug(
            f"Call blocked by circuit breaker '{self.name}'. "
            f"State: {self.state.value}, "
            f"Time since last failure: {time.time() - self.last_failure_time:.1f}s"
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение детальных метрик."""
        return {
            "name": self.name,
            "state": self.state.value,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "timeout_calls": self.metrics.timeout_calls,
                "slow_calls": self.metrics.slow_calls,
                "failure_rate": self.metrics.failure_rate,
                "slow_call_rate": self.metrics.slow_call_rate,
                "success_rate": self.metrics.success_rate,
                "state_changes": self.metrics.state_changes,
                "last_state_change": self.metrics.last_state_change.isoformat() if self.metrics.last_state_change else None
            },
            "config": {
                "failure_threshold": self._adaptive_failure_threshold,
                "recovery_timeout": self._adaptive_recovery_timeout,
                "timeout_duration": self.config.timeout_duration
            },
            "runtime": {
                "consecutive_successes": self.consecutive_successes,
                "consecutive_failures": self.consecutive_failures,
                "time_since_last_failure": time.time() - self.last_failure_time if self.last_failure_time > 0 else None
            }
        }
    
    async def reset(self) -> None:
        """Сброс circuit breaker в исходное состояние."""
        async with self._lock:
            self.state = CircuitState.CLOSED
            self.metrics = CircuitBreakerMetrics()
            self.consecutive_successes = 0
            self.consecutive_failures = 0
            self.last_failure_time = 0.0
            self._adaptive_failure_threshold = self.config.failure_threshold
            self._adaptive_recovery_timeout = self.config.recovery_timeout
            
        logger.info(f"Circuit breaker '{self.name}' reset")


# Глобальный реестр circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
    """Получение или создание circuit breaker."""
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreaker(name, config)
    return _circuit_breakers[name]


def circuit_breaker(name: str, config: Optional[CircuitBreakerConfig] = None) -> None:
    """
    Декоратор circuit breaker для функций.
    
    Args:
        name: Имя circuit breaker
        config: Конфигурация (опционально)
        
    Example:
        @circuit_breaker("external_api", CircuitBreakerConfig(failure_threshold=3))
        async def call_external_api() -> None:
            ...
    """
    def decorator(func) -> None:
        cb = get_circuit_breaker(name, config)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> None:
            return await cb.call(func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> None:
            # Для синхронных функций создаем async wrapper
            return asyncio.run(cb.call(func, *args, **kwargs))
        
        # Возвращаем подходящий wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def get_all_circuit_breakers() -> Dict[str, CircuitBreaker]:
    """Получение всех circuit breakers."""
    return _circuit_breakers.copy()


async def reset_all_circuit_breakers() -> None:
    """Сброс всех circuit breakers."""
    for cb in _circuit_breakers.values():
        await cb.reset()


# Экспорт основных компонентов
__all__ = [
    "CircuitBreaker",
    "CircuitBreakerConfig", 
    "CircuitBreakerError",
    "CircuitState",
    "circuit_breaker",
    "get_circuit_breaker",
    "get_all_circuit_breakers",
    "reset_all_circuit_breakers"
]