"""
Базовая реализация сервиса для устранения дублирования кода.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from loguru import logger

from domain.exceptions.base_exceptions import (
    DomainException,
    EvaluationError,
)
from domain.exceptions.protocol_exceptions import (
    ConfigurationError,
    ServiceError,
    ValidationError,
)
from domain.types.common_types import (
    CacheValue,
    ConfigData,
    EntityId,
    JsonData,
    OperationResult,
    ValidationResult,
)

T = TypeVar("T")


@dataclass
class ServiceMetrics:
    """Метрики сервиса."""

    operation_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_operation: Optional[datetime] = None
    cache_hits: int = 0
    cache_misses: int = 0


class BaseServiceImpl(Generic[T], ABC):
    """
    Базовая реализация сервиса с устранением дублирования.
    Предоставляет общую логику для всех сервисов:
    - Валидация входных данных
    - Обработка ошибок
    - Метрики и мониторинг
    - Кэширование результатов
    - Конфигурация
    """

    def __init__(self, config: Optional[ConfigData] = None):
        """Инициализация базового сервиса."""
        self.config = config or {}
        self.logger = logger.bind(module=self.__class__.__name__)
        # Метрики
        self._metrics = ServiceMetrics()
        # Кэш результатов
        self._result_cache: Dict[str, CacheValue] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self._cache_max_size = self.config.get("cache_max_size", 100)
        self._cache_ttl_seconds = self.config.get("cache_ttl", 300)
        # Состояние
        self._initialized = False
        self._startup_time = datetime.now()
        # Инициализация
        self._initialize_service()

    def _initialize_service(self) -> None:
        """Инициализация сервиса."""
        try:
            self._validate_config()
            self._setup_dependencies()
            self._initialized = True
            self.logger.info(
                f"Service {self.__class__.__name__} initialized successfully"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
            raise ServiceError(f"Service initialization failed: {e}")

    @property
    def is_initialized(self) -> bool:
        """Проверка инициализации сервиса."""
        return self._initialized

    # ============================================================================
    # КОНФИГУРАЦИЯ
    # ============================================================================
    def _validate_config(self) -> None:
        """Валидация конфигурации."""
        required_configs = self._get_required_configs()
        missing_configs = []
        for config_key in required_configs:
            if config_key not in self.config:
                missing_configs.append(config_key)
        if missing_configs:
            raise ConfigurationError(
                f"Missing required configs: {missing_configs}",
                config_key="required_configs",
                config_value=missing_configs,
            )

    def _get_required_configs(self) -> List[str]:
        """Получение обязательных конфигураций (переопределяется в наследниках)."""
        return []

    def _setup_dependencies(self) -> None:
        """Настройка зависимостей (переопределяется в наследниках)."""
        pass

    def get_config(self, key: str, default: Any = None) -> Any:
        """Получение значения конфигурации."""
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """Установка значения конфигурации."""
        self.config[key] = value

    # ============================================================================
    # ВАЛИДАЦИЯ
    # ============================================================================
    async def _validate_input(self, data: Any, context: str = "") -> ValidationResult:
        """Валидация входных данных."""
        errors: List[str] = []
        warnings: list[str] = []
        try:
            # Базовая валидация
            if data is None:
                errors.append(f"Input data cannot be None for context: {context}")
                return ValidationResult(
                    is_valid=False, errors=errors, warnings=warnings
                )
            # Специфичная валидация
            specific_errors, specific_warnings = await self._validate_input_specific(
                data, context
            )
            errors.extend(specific_errors)
            warnings.extend(specific_warnings)
            return ValidationResult(
                is_valid=len(errors) == 0, errors=errors, warnings=warnings
            )
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            errors.append(f"Validation failed: {str(e)}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

    async def _validate_input_specific(
        self, data: Any, context: str
    ) -> tuple[List[str], List[str]]:
        """Специфичная валидация входных данных (переопределяется в наследниках)."""
        return [], []

    # ============================================================================
    # КЭШИРОВАНИЕ
    # ============================================================================
    def _generate_cache_key(self, operation: str, data_hash: str) -> str:
        """Генерация ключа кэша."""
        return f"{self.__class__.__name__}:{operation}:{data_hash}"

    def _get_from_cache(self, key: str) -> Optional[CacheValue]:
        """Получение из кэша."""
        if key not in self._result_cache:
            self._metrics.cache_misses += 1
            return None
        # Проверка TTL
        if key in self._cache_ttl:
            if datetime.now() > self._cache_ttl[key]:
                del self._result_cache[key]
                del self._cache_ttl[key]
                self._metrics.cache_misses += 1
                return None
        self._metrics.cache_hits += 1
        return self._result_cache[key]

    def _set_cache(
        self, key: str, value: CacheValue, ttl: Optional[int] = None
    ) -> None:
        """Установка в кэш."""
        # Очистка при превышении размера
        cache_max_size = (
            int(self._cache_max_size) 
            if isinstance(self._cache_max_size, (int, float, str)) 
            else 100
        )
        if len(self._result_cache) >= cache_max_size:
            self._evict_cache()
        self._result_cache[key] = value
        ttl_seconds = ttl or self._cache_ttl_seconds
        # Проверка и преобразование типа ttl_seconds
        if isinstance(ttl_seconds, (int, float)):
            ttl_float = float(ttl_seconds)
        else:
            ttl_float = 300.0  # Значение по умолчанию
        self._cache_ttl[key] = datetime.now() + timedelta(seconds=ttl_float)

    def _evict_cache(self) -> None:
        """Вытеснение из кэша (LRU)."""
        if not self._result_cache:
            return
        # Удаляем самый старый элемент
        oldest_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
        del self._result_cache[oldest_key]
        del self._cache_ttl[oldest_key]

    def _clear_cache(self) -> None:
        """Очистка кэша."""
        self._result_cache.clear()
        self._cache_ttl.clear()

    # ============================================================================
    # ОБРАБОТКА ОШИБОК
    # ============================================================================
    def _handle_operation_error(self, operation: str, error: Exception) -> None:
        """Обработка ошибки операции."""
        self._metrics.error_count += 1
        self._metrics.last_operation = datetime.now()
        error_msg = f"Operation '{operation}' failed: {str(error)}"
        self.logger.error(error_msg)
        # Определение типа ошибки
        if isinstance(error, ValidationError):
            raise error
        elif isinstance(error, ConfigurationError):
            raise error
        else:
            raise ServiceError(error_msg)

    def _record_operation(self, operation: str, success: bool, duration: float) -> None:
        """Запись метрик операции."""
        self._metrics.operation_count += 1
        self._metrics.last_operation = datetime.now()
        # Обновление среднего времени ответа
        if self._metrics.operation_count == 1:
            self._metrics.avg_response_time = duration
        else:
            self._metrics.avg_response_time = (
                self._metrics.avg_response_time * (self._metrics.operation_count - 1)
                + duration
            ) / self._metrics.operation_count

    # ============================================================================
    # БАЗОВЫЕ ОПЕРАЦИИ
    # ============================================================================
    async def process(self, data: T) -> OperationResult:
        """Обработка данных."""
        start_time = datetime.now()
        try:
            # Проверка инициализации
            if not self._initialized:
                raise ServiceError("Service is not initialized")
            # Валидация входных данных
            validation_result = await self._validate_input(data, "process")
            if not validation_result.get("is_valid", False):
                return OperationResult(
                    success=False,
                    message=f"Validation failed: {validation_result.get('errors', [])}",
                    data=None,
                    error="VALIDATION_ERROR",
                )
            # Проверка кэша
            data_hash = str(hash(str(data)))
            cache_key = self._generate_cache_key("process", data_hash)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                duration = (datetime.now() - start_time).total_seconds() * 1000
                self._record_operation("process", True, duration)
                return OperationResult(
                    success=True,
                    message="Result retrieved from cache",
                    data=cached_result,
                    error=None,
                )
            # Обработка
            result = await self._process_impl(data)
            # Кэширование результата
            self._set_cache(cache_key, result)
            # Метрики
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._record_operation("process", True, duration)
            return OperationResult(
                success=True,
                message="Operation completed successfully",
                data=result,
                error=None,
            )
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds() * 1000
            self._record_operation("process", False, duration)
            self._handle_operation_error("process", e)
            raise

    async def validate(self, data: T) -> ValidationResult:
        """Валидация данных."""
        try:
            return await self._validate_input(data, "validate")
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False, errors=[f"Validation failed: {str(e)}"], warnings=[]
            )

    # ============================================================================
    # АБСТРАКТНЫЕ МЕТОДЫ (ДОЛЖНЫ БЫТЬ РЕАЛИЗОВАНЫ В НАСЛЕДНИКАХ)
    # ============================================================================
    @abstractmethod
    async def _process_impl(self, data: T) -> Any:
        """Реализация обработки данных."""
        pass

    # ============================================================================
    # УТИЛИТЫ
    # ============================================================================
    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик сервиса."""
        return {
            "operation_count": self._metrics.operation_count,
            "error_count": self._metrics.error_count,
            "cache_hits": self._metrics.cache_hits,
            "cache_misses": self._metrics.cache_misses,
            "cache_hit_ratio": (
                self._metrics.cache_hits
                / (self._metrics.cache_hits + self._metrics.cache_misses)
                if (self._metrics.cache_hits + self._metrics.cache_misses) > 0
                else 0
            ),
            "avg_response_time": self._metrics.avg_response_time,
            "last_operation": (
                self._metrics.last_operation.isoformat()
                if self._metrics.last_operation
                else None
            ),
            "cache_size": len(self._result_cache),
            "initialized": self._initialized,
            "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
        }

    def reset_metrics(self) -> None:
        """Сброс метрик."""
        self._metrics = ServiceMetrics()

    def get_health_status(self) -> Dict[str, Any]:
        """Получение статуса здоровья сервиса."""
        return {
            "service_name": self.__class__.__name__,
            "initialized": self._initialized,
            "uptime_seconds": (datetime.now() - self._startup_time).total_seconds(),
            "error_rate": (
                self._metrics.error_count / self._metrics.operation_count
                if self._metrics.operation_count > 0
                else 0
            ),
            "cache_efficiency": (
                self._metrics.cache_hits
                / (self._metrics.cache_hits + self._metrics.cache_misses)
                if (self._metrics.cache_hits + self._metrics.cache_misses) > 0
                else 0
            ),
        }
