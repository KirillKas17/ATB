"""
Базовый класс для всех сервисов application слоя.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional
from shared.base_service import SharedBaseService

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


@dataclass
class ServiceMetrics:
    """Метрики сервиса."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_request_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


class BaseApplicationService(SharedBaseService, ABC):
    """Базовый класс для всех сервисов application слоя."""

    def __init__(self, service_name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.service_name = service_name
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
        self.metrics = ServiceMetrics()
        self.is_running = False
        self.start_time = datetime.now()
        # Конфигурация по умолчанию
        self.config = config or {}
        self.timeout_seconds = self.config.get("timeout_seconds", 30)
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.max_workers = self.config.get("max_workers", 8)
        # Семафор для ограничения одновременных операций
        self._semaphore = asyncio.Semaphore(self.max_workers)

    async def start(self) -> None:
        """Запуск сервиса."""
        if self.is_running:
            self.logger.warning(f"Service {self.service_name} is already running")
            return
        self.is_running = True
        self.start_time = datetime.now()
        self.logger.info(f"Service {self.service_name} started")
        # Запускаем фоновые задачи
        asyncio.create_task(self._metrics_collector())
        asyncio.create_task(self._health_checker())

    async def stop(self) -> None:
        """Остановка сервиса."""
        if not self.is_running:
            self.logger.warning(f"Service {self.service_name} is not running")
            return
        self.is_running = False
        self.logger.info(f"Service {self.service_name} stopped")

    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья сервиса."""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "service_name": self.service_name,
            "status": "healthy" if self.is_running else "stopped",
            "uptime_seconds": uptime,
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": (
                    self.metrics.successful_requests / self.metrics.total_requests
                    if self.metrics.total_requests > 0
                    else 0.0
                ),
                "average_response_time": self.metrics.average_response_time,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
            },
            "config": {
                "timeout_seconds": self.timeout_seconds,
                "retry_attempts": self.retry_attempts,
                "max_workers": self.max_workers,
            },
        }

    async def get_metrics(self) -> ServiceMetrics:
        """Получение метрик сервиса."""
        return self.metrics

    async def reset_metrics(self) -> None:
        """Сброс метрик сервиса."""
        self.metrics = ServiceMetrics()
        self.logger.info(f"Metrics reset for service {self.service_name}")

    async def _execute_with_metrics(
        self, operation_name: str, operation: Callable, *args, **kwargs
    ) -> Any:
        """Выполнение операции с метриками."""
        start_time = datetime.now()
        self.metrics.total_requests += 1
        self.metrics.last_request_time = start_time
        try:
            async with self._semaphore:
                result = await asyncio.wait_for(
                    operation(*args, **kwargs), timeout=self.timeout_seconds
                )
                self.metrics.successful_requests += 1
                return result
        except asyncio.TimeoutError:
            self.metrics.failed_requests += 1
            self.logger.error(
                f"Operation {operation_name} timed out after {self.timeout_seconds}s"
            )
            raise
        except Exception as e:
            self.metrics.failed_requests += 1
            self.logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            # Обновляем среднее время ответа
            if self.metrics.total_requests > 0:
                self.metrics.average_response_time = (
                    self.metrics.average_response_time
                    * (self.metrics.total_requests - 1)
                    + response_time
                ) / self.metrics.total_requests

    async def _metrics_collector(self) -> None:
        """Сборщик метрик."""
        while self.is_running:
            try:
                # Собираем системные метрики
                if psutil:
                    process = psutil.Process()
                    self.metrics.memory_usage_mb = (
                        process.memory_info().rss / 1024 / 1024
                    )
                    self.metrics.cpu_usage_percent = process.cpu_percent()
                    self.metrics.uptime_seconds = (
                        datetime.now() - self.start_time
                    ).total_seconds()
                    await asyncio.sleep(60)  # Обновляем каждую минуту
                else:
                    self.logger.warning(
                        "psutil is not installed, unable to collect system metrics"
                    )
                    await asyncio.sleep(60)
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(60)

    async def _health_checker(self) -> None:
        """Проверка здоровья сервиса."""
        while self.is_running:
            try:
                health = await self.health_check()
                if health["status"] != "healthy":
                    self.logger.warning(f"Service health check failed: {health}")
                await asyncio.sleep(30)  # Проверяем каждые 30 секунд
            except Exception as e:
                self.logger.error(f"Error in health checker: {e}")
                await asyncio.sleep(30)

    @abstractmethod
    async def validate_config(self) -> bool:
        """Валидация конфигурации сервиса."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Инициализация сервиса."""
        ...
