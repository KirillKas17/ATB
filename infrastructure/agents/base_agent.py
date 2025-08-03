# -*- coding: utf-8 -*-
"""Базовый класс для всех агентов торговой системы."""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, runtime_checkable

from domain.types.agent_types import (
    AgentConfig,
    AgentId,
    AgentMetrics,
    AgentState,
    AgentStatus,
    AgentType,
    ProcessingResult,
    validate_confidence,
    validate_risk_score,
)
from shared.cache import CacheConfig, CacheManager
from shared.exceptions import BaseException
from domain.exceptions import ValidationError
from shared.logging import LoggerMixin


class AgentError(BaseException):
    """Базовое исключение для агентов."""


class AgentInitializationError(AgentError):
    """Ошибка инициализации агента."""


class AgentProcessingError(AgentError):
    """Ошибка обработки данных агентом."""


class AgentHealthError(AgentError):
    """Ошибка здоровья агента."""


@dataclass(frozen=True)
class AgentHealthMetrics:
    """Метрики здоровья агента."""

    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    response_time_ms: float = 0.0
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Валидация метрик здоровья."""
        if self.memory_usage_mb < 0:
            raise ValidationError("Memory usage cannot be negative", "memory_usage_mb", self.memory_usage_mb, "non_negative")
        if not 0 <= self.cpu_usage_percent <= 100:
            raise ValidationError("CPU usage must be between 0 and 100", "cpu_usage_percent", self.cpu_usage_percent, "range_0_100")
        if self.response_time_ms < 0:
            raise ValidationError("Response time cannot be negative", "response_time_ms", self.response_time_ms, "non_negative")
        if not 0 <= self.error_rate <= 1:
            raise ValidationError("Error rate must be between 0 and 1", "error_rate", self.error_rate, "range_0_1")


@runtime_checkable
class AgentLifecycleProtocol(Protocol):
    """Протокол жизненного цикла агента."""

    async def initialize(self) -> bool:
        """Инициализация агента."""
        ...

    async def start(self) -> bool:
        """Запуск агента."""
        ...

    async def stop(self) -> None:
        """Остановка агента."""
        ...

    async def cleanup(self) -> None:
        """Очистка ресурсов агента."""
        ...


@runtime_checkable
class AgentProcessingProtocol(Protocol):
    """Протокол обработки данных агента."""

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных агентом."""
        ...

    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        ...


class BaseAgent(LoggerMixin, ABC):
    """Базовый класс для всех агентов торговой системы."""

    def __init__(
        self,
        name: str,
        agent_type: AgentType,
        config: Optional[Dict[str, Any]] = None,
        cache_config: Optional[CacheConfig] = None,
    ):
        """
        Инициализация базового агента.

        Args:
            name: Имя агента
            agent_type: Тип агента
            config: Конфигурация агента
            cache_config: Конфигурация кэша
        """
        super().__init__()

        self._agent_id = AgentId(str(uuid.uuid4()))
        self._name = name
        self._agent_type = agent_type
        self._config = config or {}
        self._initialized = False
        self._running_task: Optional[asyncio.Task] = None
        self._start_time = datetime.now()
        self._last_activity = datetime.now()

        # Инициализация кэша
        self._cache_manager = CacheManager(cache_config or CacheConfig())

        # Состояние и метрики
        self._state = AgentState(
            agent_id=self._agent_id,
            status=AgentStatus.INITIALIZING,
            is_running=False,
            is_healthy=False,
            performance_score=0.0,
            confidence=0.0,
            risk_score=0.0,
        )

        # Инициализация метрик
        self._metrics = AgentMetrics(
            name=self._name,
            agent_type=self._agent_type,
            status=AgentStatus.INITIALIZING,
            total_processed=0,
            success_count=0,
            error_count=0,
            avg_processing_time_ms=0.0,
            last_error=None,
            last_update=datetime.now(),
            metadata={"agent_type": str(self._agent_type)},
        )

        # Метрики здоровья
        self._health_metrics = AgentHealthMetrics()

        # Статистика производительности
        self._processing_times: List[float] = []
        self._max_processing_history = 1000

        # Семафор для ограничения конкурентности
        self._processing_semaphore = asyncio.Semaphore(10)

        # Флаги состояния
        self._shutdown_requested = False
        self._health_check_interval = 30  # секунды

        self.logger.info(f"Initialized agent: {name} (ID: {self._agent_id})")

    @property
    def agent_id(self) -> AgentId:
        """Уникальный идентификатор агента."""
        return self._agent_id

    @property
    def name(self) -> str:
        """Имя агента."""
        return self._name

    @property
    def agent_type(self) -> AgentType:
        """Тип агента."""
        return self._agent_type

    @property
    def state(self) -> AgentState:
        """Текущее состояние агента."""
        return self._state

    @property
    def metrics(self) -> AgentMetrics:
        """Метрики агента."""
        return self._metrics

    @property
    def config(self) -> Dict[str, Any]:
        """Конфигурация агента."""
        return self._config.copy()

    @property
    def health_metrics(self) -> AgentHealthMetrics:
        """Метрики здоровья агента."""
        return self._health_metrics

    @abstractmethod
    async def initialize(self) -> bool:
        """Инициализация агента."""
        pass

    @abstractmethod
    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных агентом."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Очистка ресурсов агента."""
        pass

    async def start(self) -> bool:
        """Запуск агента."""
        try:
            if self._state.is_running:
                self.logger.warning(f"Agent {self._name} is already running")
                return True

            # Инициализация если еще не инициализирован
            if not self._initialized:
                if not await self.initialize():
                    self.logger.error(f"Failed to initialize agent: {self._name}")
                    return False

            # Запуск мониторинга здоровья
            self._running_task = asyncio.create_task(self._monitor_health())

            self._update_state(AgentStatus.RUNNING, is_running=True, is_healthy=True)
            self.logger.info(f"Started agent: {self._name}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting agent {self._name}: {e}")
            self._update_state(AgentStatus.ERROR, is_running=False, is_healthy=False)
            return False

    async def stop(self) -> None:
        """Остановка агента."""
        try:
            if not self._state.is_running:
                self.logger.warning(f"Agent {self._name} is not running")
                return

            self._shutdown_requested = True

            # Остановка задачи мониторинга
            if self._running_task and not self._running_task.done():
                self._running_task.cancel()
                try:
                    await self._running_task
                except asyncio.CancelledError:
                    pass

            await self.cleanup()
            self._update_state(AgentStatus.STOPPED, is_running=False)
            self.logger.info(f"Stopped agent: {self._name}")

        except Exception as e:
            self.logger.error(f"Error stopping agent {self._name}: {e}")

    def is_healthy(self) -> bool:
        """Проверка здоровья агента."""
        try:
            # Проверка базовых метрик
            if self._metrics.error_count > 0:
                error_rate = self._metrics.error_count / max(self._metrics.total_processed, 1)
                if error_rate > 0.1:  # Более 10% ошибок
                    return False

            # Проверка времени обработки
            if self._metrics.avg_processing_time_ms > 5000:  # Более 5 секунд
                return False

            # Проверка использования памяти
            if self._health_metrics.memory_usage_mb > 1000:  # Более 1GB
                return False

            # Проверка использования CPU
            if self._health_metrics.cpu_usage_percent > 90:  # Более 90%
                return False

            # Проверка времени последней активности
            time_since_last_activity = (datetime.now() - self._last_activity).total_seconds()
            if time_since_last_activity > 300:  # Более 5 минут
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking health: {e}")
            return False

    def get_performance(self) -> float:
        """Получение производительности агента."""
        try:
            if self._metrics.total_processed == 0:
                return 0.0

            # Базовый показатель - отношение успешных операций
            success_rate = self._metrics.success_count / self._metrics.total_processed

            # Учитываем время обработки
            time_factor = max(0, 1 - (self._metrics.avg_processing_time_ms / 1000))
            time_factor = min(1, time_factor)

            # Учитываем уверенность агента
            confidence_factor = self._state.confidence

            # Учитываем риск
            risk_factor = 1 - self._state.risk_score

            # Комбинированный показатель
            performance = (
                success_rate * 0.4 +
                time_factor * 0.2 +
                confidence_factor * 0.2 +
                risk_factor * 0.2
            )

            return max(0.0, min(1.0, performance))

        except Exception as e:
            self.logger.error(f"Error calculating performance: {e}")
            return 0.0

    def get_confidence(self) -> float:
        """Получение уверенности агента."""
        return self._state.confidence

    def get_risk_score(self) -> float:
        """Получение оценки риска агента."""
        return self._state.risk_score

    def update_confidence(self, confidence: float) -> None:
        """Обновление уверенности агента."""
        try:
            validated_confidence = validate_confidence(confidence)
            self._update_state(status=self._state.status, confidence=validated_confidence)
            self.logger.debug(f"Updated confidence for {self._name}: {validated_confidence}")
        except Exception as e:
            self.logger.error(f"Error updating confidence: {e}")

    def update_risk_score(self, risk_score: float) -> None:
        """Обновление оценки риска агента."""
        try:
            validated_risk_score = validate_risk_score(risk_score)
            self._update_state(status=self._state.status, risk_score=validated_risk_score)
            self.logger.debug(f"Updated risk score for {self._name}: {validated_risk_score}")
        except Exception as e:
            self.logger.error(f"Error updating risk score: {e}")

    def record_success(self, processing_time_ms: float = 0.0) -> None:
        """Запись успешной операции."""
        try:
            self._metrics = replace(
                self._metrics,
                total_processed=self._metrics.total_processed + 1,
                success_count=self._metrics.success_count + 1,
                last_update=datetime.now(),
            )

            # Обновление времени обработки
            avg_time = self._calculate_avg_processing_time(processing_time_ms)
            self._metrics = replace(self._metrics, avg_processing_time_ms=avg_time)

            # Обновление метрик здоровья
            self._update_health_metrics(processing_time_ms, success=True)

            # Обновление времени последней активности
            self._last_activity = datetime.now()

        except Exception as e:
            self.logger.error(f"Error recording success: {e}")

    def record_error(self, error: str, processing_time_ms: float = 0.0) -> None:
        """Запись ошибки."""
        try:
            self._metrics = replace(
                self._metrics,
                total_processed=self._metrics.total_processed + 1,
                error_count=self._metrics.error_count + 1,
                last_error=error,
                last_update=datetime.now(),
            )

            # Обновление времени обработки
            avg_time = self._calculate_avg_processing_time(processing_time_ms)
            self._metrics = replace(self._metrics, avg_processing_time_ms=avg_time)

            # Обновление метрик здоровья
            self._update_health_metrics(processing_time_ms, success=False)

            # Обновление времени последней активности
            self._last_activity = datetime.now()

            self.logger.error(f"Agent {self._name} error: {error}")

        except Exception as e:
            self.logger.error(f"Error recording error: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации агента.
        Returns:
            True если конфигурация валидна, False иначе
        """
        try:
            required_keys = self._get_required_config_keys()
            for key in required_keys:
                if key not in self._config:
                    self.logger.error(f"Missing required config key: {key}")
                    return False

                value = self._config[key]
                if not self._validate_config_value(key, value):
                    self.logger.error(f"Invalid config value for key: {key}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating config: {e}")
            return False

    def get_config(self) -> Dict[str, Any]:
        """
        Получение конфигурации агента.

        Returns:
            Копия конфигурации агента
        """
        return self._config.copy()

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        Обновление конфигурации агента.

        Args:
            new_config: Новая конфигурация

        Returns:
            True если обновление прошло успешно, False иначе
        """
        try:
            # Валидация новой конфигурации
            old_config = self._config.copy()
            self._config.update(new_config)

            if not self.validate_config():
                self._config = old_config
                return False

            self.logger.info(f"Updated config for agent: {self._name}")
            return True

        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
            return False

    def get_status_report(self) -> Dict[str, Any]:
        """
        Получение отчета о состоянии агента.

        Returns:
            Словарь с информацией о состоянии агента
        """
        try:
            return {
                "agent_id": str(self._agent_id),
                "name": self._name,
                "type": str(self._agent_type),
                "status": self._state.status.value,
                "is_running": self._state.is_running,
                "is_healthy": self._state.is_healthy,
                "performance_score": self._state.performance_score,
                "confidence": self._state.confidence,
                "risk_score": self._state.risk_score,
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
                "total_processed": self._metrics.total_processed,
                "success_count": self._metrics.success_count,
                "error_count": self._metrics.error_count,
                "avg_processing_time_ms": self._metrics.avg_processing_time_ms,
                "memory_usage_mb": self._health_metrics.memory_usage_mb,
                "cpu_usage_percent": self._health_metrics.cpu_usage_percent,
                "last_error": self._metrics.last_error,
                "last_update": self._metrics.last_update.isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error generating status report: {e}")
            return {}

    @asynccontextmanager
    async def processing_context(self) -> AsyncGenerator[None, None]:
        """Контекстный менеджер для обработки данных с ограничением конкурентности."""
        async with self._processing_semaphore:
            start_time = time.time()
            try:
                yield
            finally:
                processing_time = (time.time() - start_time) * 1000
                self._processing_times.append(processing_time)
                if len(self._processing_times) > self._max_processing_history:
                    self._processing_times.pop(0)

    def _update_state(self, status: AgentStatus, **kwargs) -> None:
        """
        Обновление состояния агента.

        Args:
            status: Новый статус
            **kwargs: Дополнительные параметры состояния
        """
        try:
            self._state = replace(self._state, status=status, **kwargs)
            self._metrics = replace(self._metrics, status=status)
        except Exception as e:
            self.logger.error(f"Error updating state: {e}")

    def _update_metrics(self) -> None:
        """Обновление метрик агента."""
        try:
            # Обновление времени обработки
            if self._processing_times:
                avg_time = sum(self._processing_times) / len(self._processing_times)
                self._metrics = replace(self._metrics, avg_processing_time_ms=avg_time)

            # Обновление использования ресурсов
            memory_usage = self._estimate_memory_usage()
            cpu_usage = self._estimate_cpu_usage()

            self._metrics = replace(
                self._metrics, memory_usage_mb=memory_usage, cpu_usage_percent=cpu_usage
            )

        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def _calculate_avg_processing_time(self, new_time_ms: float) -> float:
        """
        Расчет среднего времени обработки.

        Args:
            new_time_ms: Новое время обработки

        Returns:
            Среднее время обработки
        """
        self._processing_times.append(new_time_ms)
        if len(self._processing_times) > self._max_processing_history:
            self._processing_times.pop(0)

        return sum(self._processing_times) / len(self._processing_times)

    def _update_health_metrics(self, processing_time_ms: float, success: bool) -> None:
        """
        Обновление метрик здоровья.

        Args:
            processing_time_ms: Время обработки в миллисекундах
            success: Успешность операции
        """
        try:
            # Обновление времени отклика
            response_time = processing_time_ms

            # Обновление частоты ошибок
            total_ops = self._metrics.total_processed
            error_count = self._metrics.error_count
            error_rate = error_count / total_ops if total_ops > 0 else 0.0

            # Обновление времени работы
            uptime = (datetime.now() - self._start_time).total_seconds()

            self._health_metrics = AgentHealthMetrics(
                memory_usage_mb=self._estimate_memory_usage(),
                cpu_usage_percent=self._estimate_cpu_usage(),
                response_time_ms=response_time,
                error_rate=error_rate,
                uptime_seconds=uptime,
                last_heartbeat=datetime.now(),
            )

        except Exception as e:
            self.logger.error(f"Error updating health metrics: {e}")

    async def _monitor_health(self) -> None:
        """Мониторинг здоровья агента."""
        try:
            while not self._shutdown_requested:
                await asyncio.sleep(self._health_check_interval)

                # Проверка здоровья
                is_healthy = self.is_healthy()
                if not is_healthy and self._state.is_healthy:
                    self.logger.warning(f"Agent {self._name} became unhealthy")
                    self._update_state(AgentStatus.UNHEALTHY, is_healthy=False)
                elif is_healthy and not self._state.is_healthy:
                    self.logger.info(f"Agent {self._name} became healthy")
                    self._update_state(AgentStatus.RUNNING, is_healthy=True)

                # Обновление метрик
                self._update_metrics()

        except asyncio.CancelledError:
            self.logger.info(f"Health monitoring cancelled for agent: {self._name}")
        except Exception as e:
            self.logger.error(f"Error in health monitoring: {e}")

    def _estimate_memory_usage(self) -> float:
        """
        Оценка использования памяти агентом.

        Returns:
            Использование памяти в МБ
        """
        try:
            # Базовое использование памяти
            base_memory = 50.0  # МБ

            # Память для кэша (используем конфигурацию как приближение)
            cache_size = self._cache_manager.config.max_size * 0.1  # Примерно 0.1 МБ на запись

            # Память для истории обработки
            processing_history_size = (
                len(self._processing_times) * 0.001
            )  # 0.001 МБ на запись

            # Память для метрик
            metrics_memory = 10.0  # МБ

            total_memory = base_memory + cache_size + processing_history_size + metrics_memory
            return min(total_memory, 1000.0)  # Максимум 1 ГБ

        except Exception as e:
            self.logger.error(f"Error estimating memory usage: {e}")
            return 50.0  # Базовое значение

    def _estimate_cpu_usage(self) -> float:
        """
        Оценка использования CPU.

        Returns:
            Использование CPU в процентах
        """
        try:
            # Оценка на основе частоты обработки
            if len(self._processing_times) < 2:
                return 5.0

            recent_times = self._processing_times[-10:]  # Последние 10 операций
            avg_time = sum(recent_times) / len(recent_times)

            # Нормализация: больше времени = больше CPU
            cpu_usage = min(90.0, (avg_time / 100) * 50)  # Максимум 90%
            return max(1.0, cpu_usage)

        except Exception:
            return 5.0

    def _get_required_config_keys(self) -> List[str]:
        """
        Получение списка обязательных ключей конфигурации.

        Returns:
            Список обязательных ключей
        """
        return []

    def _validate_config_value(self, key: str, value: Any) -> bool:
        """
        Валидация значения конфигурации.

        Args:
            key: Ключ конфигурации
            value: Значение для валидации

        Returns:
            True если значение валидно, False иначе
        """
        return True
