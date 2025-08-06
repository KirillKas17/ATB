"""
Система мониторинга производительности для ATB.
Обеспечивает отслеживание производительности компонентов,
метрик и автоматическое оповещение о проблемах.
"""

import asyncio
import json
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, cast

import psutil
from loguru import logger


class MetricType(Enum):
    """Типы метрик производительности."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertLevel(Enum):
    """Уровни алертов."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Метрика производительности."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime
    component: str
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Алерт производительности."""

    level: AlertLevel
    message: str
    component: str
    metric_name: str
    current_value: float
    threshold: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class PerformanceMonitor:
    """
    Монитор производительности компонентов.
    Отслеживает метрики производительности, генерирует алерты
    и предоставляет аналитику по производительности системы.
    """

    def __init__(self) -> None:
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: List[Alert] = []
        self.thresholds: Dict[str, Dict[str, float]] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.system_metrics_enabled = True
        self._monitoring_task: Optional[asyncio.Task] = None
        # Настройка базовых порогов
        self._setup_default_thresholds()

    def _setup_default_thresholds(self) -> None:
        """Настройка порогов по умолчанию."""
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "error": 85.0, "critical": 95.0},
            "memory_usage": {"warning": 80.0, "error": 90.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "error": 90.0, "critical": 95.0},
            "response_time": {"warning": 1000.0, "error": 5000.0, "critical": 10000.0},
            "error_rate": {"warning": 5.0, "error": 10.0, "critical": 20.0},
            "throughput": {"warning": 100.0, "error": 50.0, "critical": 10.0},
        }

    async def start_monitoring(self) -> None:
        """Запуск мониторинга."""
        if self.monitoring_active:
            logger.warning("Monitoring is already active")
            return None
        self.monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")
        return None

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        if not self.monitoring_active:
            logger.warning("Monitoring is not active")
            return None
        self.monitoring_active = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
        return None

    async def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга с адаптивными интервалами."""
        base_interval = 30  # 30 секунд базовый интервал
        current_interval = base_interval
        consecutive_errors = 0
        high_load_threshold = 0.8  # Порог высокой нагрузки
        
        try:
            while self.monitoring_active:
                loop_start_time = time.time()
                
                try:
                    # Сбор метрик с адаптивным расписанием
                    if self.system_metrics_enabled:
                        await self._collect_system_metrics()
                    
                    # Проверка пороговых значений
                    threshold_violations = await self._check_thresholds()
                    
                    # Очистка старых данных (реже при низкой нагрузке)
                    if len(self.metrics) > 100:  # Очищаем только при накоплении данных
                        await self._cleanup_old_data()
                    
                    # Адаптивный интервал на основе нагрузки системы
                    if hasattr(self, 'current_cpu_usage'):
                        cpu_usage = getattr(self, 'current_cpu_usage', 0.5)
                        if cpu_usage > high_load_threshold or threshold_violations:
                            # Увеличиваем частоту мониторинга при высокой нагрузке
                            current_interval = max(base_interval // 2, 10)
                        elif cpu_usage < 0.3:
                            # Уменьшаем частоту при низкой нагрузке
                            current_interval = min(base_interval * 2, 120)
                        else:
                            current_interval = base_interval
                    
                    consecutive_errors = 0
                    
                    # Вычисляем оставшееся время до следующего цикла
                    loop_duration = time.time() - loop_start_time
                    sleep_time = max(current_interval - loop_duration, 1)
                    
                    await asyncio.sleep(sleep_time)
                    
                except Exception as e:
                    consecutive_errors += 1
                    # Экспоненциальная задержка при ошибках
                    error_delay = min(10 * (2 ** consecutive_errors), 300)
                    logger.error(f"Error in monitoring loop (attempt {consecutive_errors}): {e}")
                    await asyncio.sleep(error_delay)
                    
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        return None

    async def _collect_system_metrics(self) -> None:
        """Сбор системных метрик."""
        try:
            # Запускаем сбор метрик в отдельном потоке для избежания блокировки
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._collect_system_metrics_sync)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
        return None

    def _collect_system_metrics_sync(self) -> None:
        """Синхронный сбор системных метрик."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_usage", cpu_percent, MetricType.GAUGE, "system")
            # Memory
            memory = psutil.virtual_memory()
            self.record_metric(
                "memory_usage", memory.percent, MetricType.GAUGE, "system"
            )
            self.record_metric(
                "memory_available",
                memory.available / (1024**3),
                MetricType.GAUGE,
                "system",
            )
            # Disk
            disk = psutil.disk_usage("/")
            self.record_metric(
                "disk_usage", (disk.used / disk.total) * 100, MetricType.GAUGE, "system"
            )
            # Network
            network = psutil.net_io_counters()
            self.record_metric(
                "network_bytes_sent", network.bytes_sent, MetricType.COUNTER, "system"
            )
            self.record_metric(
                "network_bytes_recv", network.bytes_recv, MetricType.COUNTER, "system"
            )
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType,
        component: str,
        tags: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Записать метрику производительности.
        Args:
            name: Название метрики
            value: Значение метрики
            metric_type: Тип метрики
            component: Компонент системы
            tags: Дополнительные теги
            metadata: Дополнительные метаданные
        """
        metric = PerformanceMetric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            component=component,
            tags=tags or {},
            metadata=metadata or {},
        )
        self.metrics[name].append(metric)

    def record_timing(
        self,
        name: str,
        duration: float,
        component: str,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Записать метрику времени выполнения.
        Args:
            name: Название операции
            duration: Время выполнения в секундах
            component: Компонент системы
            tags: Дополнительные теги
        """
        self.record_metric(name, duration, MetricType.TIMER, component, tags)

    def record_counter(
        self,
        name: str,
        increment: float = 1.0,
        component: str = "unknown",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Записать счетчик.
        Args:
            name: Название счетчика
            increment: Инкремент
            component: Компонент системы
            tags: Дополнительные теги
        """
        self.record_metric(name, increment, MetricType.COUNTER, component, tags)

    def set_threshold(self, metric_name: str, level: str, value: float) -> None:
        """
        Установить порог для метрики.
        Args:
            metric_name: Название метрики
            level: Уровень (warning, error, critical)
            value: Значение порога
        """
        if metric_name not in self.thresholds:
            self.thresholds[metric_name] = {}
        self.thresholds[metric_name][level] = value

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Добавить обработчик алертов.
        Args:
            handler: Функция обработки алертов
        """
        self.alert_handlers.append(handler)

    async def _check_thresholds(self) -> None:
        """Проверка порогов и генерация алертов."""
        for metric_name, thresholds in self.thresholds.items():
            if metric_name not in self.metrics or not self.metrics[metric_name]:
                continue
            current_value = self.metrics[metric_name][-1].value
            for level, threshold in thresholds.items():
                if self._should_alert(metric_name, level, current_value, threshold):
                    self._create_alert(metric_name, level, current_value, threshold)

    def _should_alert(
        self, metric_name: str, level: str, current_value: float, threshold: float
    ) -> bool:
        """Определить, нужно ли создавать алерт."""
        # Проверяем, есть ли уже активный алерт для этой метрики и уровня
        for alert in self.alerts:
            if (
                alert.metric_name == metric_name
                and alert.level.value == level
                and not alert.resolved
            ):
                return False
        # Определяем условие для алерта
        if level == "warning":
            return current_value >= threshold
        elif level == "error":
            return current_value >= threshold
        elif level == "critical":
            return current_value >= threshold
        return False

    def _create_alert(
        self, metric_name: str, level: str, current_value: float, threshold: float
    ) -> None:
        """Создать алерт."""
        alert = Alert(
            level=AlertLevel(level),
            message=f"{metric_name} exceeded {level} threshold: {current_value:.2f} >= {threshold:.2f}",
            component="system",
            metric_name=metric_name,
            current_value=current_value,
            threshold=threshold,
            timestamp=datetime.now(),
        )
        self.alerts.append(alert)
        # Вызываем обработчики алертов
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        logger.warning(f"Performance alert: {alert.message}")

    def resolve_alert(self, metric_name: str, level: str) -> None:
        """
        Разрешить алерт.
        Args:
            metric_name: Название метрики
            level: Уровень алерта
        """
        for alert in self.alerts:
            if (
                alert.metric_name == metric_name
                and alert.level.value == level
                and not alert.resolved
            ):
                alert.resolved = True
                alert.resolved_at = datetime.now()
                logger.info(f"Alert resolved: {alert.message}")

    def get_metrics_summary(
        self, component: Optional[str] = None, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Получить сводку метрик.
        Args:
            component: Фильтр по компоненту
            time_window: Временное окно
        Returns:
            Словарь со сводкой метрик
        """
        summary: Dict[str, Dict[str, Any]] = {}
        cutoff_time = datetime.now() - time_window if time_window else None
        for metric_name, metric_list in self.metrics.items():
            if not metric_list:
                continue
            # Фильтруем по времени и компоненту
            filtered_metrics = [
                m
                for m in metric_list
                if (cutoff_time is None or m.timestamp >= cutoff_time)
                and (component is None or m.component == component)
            ]
            if not filtered_metrics:
                continue
            values = [m.value for m in filtered_metrics]
            summary[metric_name] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "median": statistics.median(values),
                "latest": values[-1],
                "component": filtered_metrics[0].component,
            }
        return summary

    def get_alerts_summary(
        self, resolved: Optional[bool] = None, time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Получить сводку алертов.
        Args:
            resolved: Фильтр по статусу разрешения
            time_window: Временное окно
        Returns:
            Словарь со сводкой алертов
        """
        cutoff_time = datetime.now() - time_window if time_window else None
        filtered_alerts = [
            alert
            for alert in self.alerts
            if (resolved is None or alert.resolved == resolved)
            and (cutoff_time is None or (alert.timestamp is not None and alert.timestamp >= cutoff_time))
        ]
        summary: Dict[str, Any] = {
            "total": len(filtered_alerts),
            "resolved": len([a for a in filtered_alerts if a.resolved]),
            "active": len([a for a in filtered_alerts if not a.resolved]),
            "by_level": defaultdict(int),
            "by_component": defaultdict(int),
            "by_metric": defaultdict(int),
        }
        for alert in filtered_alerts:
            summary["by_level"][alert.level.value] += 1
            summary["by_component"][alert.component] += 1
            summary["by_metric"][alert.metric_name] += 1
        return summary

    async def _cleanup_old_data(self) -> None:
        """Очистка старых данных."""
        try:
            # Удаляем метрики старше 24 часов
            cutoff_time = datetime.now() - timedelta(hours=24)
            for metric_name in list(self.metrics.keys()):
                self.metrics[metric_name] = deque(
                    [
                        m
                        for m in self.metrics[metric_name]
                        if m.timestamp >= cutoff_time
                    ],
                    maxlen=1000,
                )
            # Удаляем разрешенные алерты старше 7 дней
            cutoff_time = datetime.now() - timedelta(days=7)
            self.alerts = [
                a for a in self.alerts if not a.resolved or (a.resolved_at is not None and a.resolved_at >= cutoff_time)
            ]
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")

    def export_metrics(self, format: str = "json") -> str:
        """
        Экспорт метрик.
        Args:
            format: Формат экспорта (json, csv)
        Returns:
            Строка с экспортированными данными
        """
        if format == "json":
            data = {
                "metrics": {
                    name: [
                        {
                            "name": m.name,
                            "value": m.value,
                            "type": m.metric_type.value,
                            "timestamp": m.timestamp.isoformat(),
                            "component": m.component,
                            "tags": m.tags,
                            "metadata": m.metadata,
                        }
                        for m in metric_list
                    ]
                    for name, metric_list in self.metrics.items()
                },
                "alerts": [
                    {
                        "level": a.level.value,
                        "message": a.message,
                        "component": a.component,
                        "metric_name": a.metric_name,
                        "current_value": a.current_value,
                        "threshold": a.threshold,
                        "timestamp": a.timestamp.isoformat(),
                        "resolved": a.resolved,
                        "resolved_at": (
                            a.resolved_at.isoformat() if a.resolved_at else None
                        ),
                    }
                    for a in self.alerts
                ],
            }
            return json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Глобальный экземпляр монитора производительности
performance_monitor = PerformanceMonitor()


def monitor_performance(component: str, operation: str):
    """
    Декоратор для мониторинга производительности функций.
    Args:
        component: Компонент системы
        operation: Операция
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"{component}.{operation}", duration, component
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"{component}.{operation}_error", duration, component
                )
                performance_monitor.record_counter(
                    f"{component}.{operation}_errors", 1, component
                )
                raise

        return wrapper

    return decorator


async def monitor_async_performance(component: str, operation: str):
    """
    Декоратор для мониторинга производительности асинхронных функций.
    Args:
        component: Компонент системы
        operation: Операция
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"{component}.{operation}", duration, component
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"{component}.{operation}_error", duration, component
                )
                performance_monitor.record_counter(
                    f"{component}.{operation}_errors", 1, component
                )
                raise

        return wrapper

    return decorator
