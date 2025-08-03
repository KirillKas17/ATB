"""
Система мониторинга производительности.
Включает:
- Сбор метрик производительности
- Систему алертов
- Трейсинг запросов
- Дашборд мониторинга
"""

import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil
from loguru import logger

from domain.types.monitoring_types import (
    Alert,
    AlertHandlerProtocol,
    AlertSeverity,
    Metric,
    MetricCollectorProtocol,
    MetricType,
    TraceProtocol,
    TraceSpan,
)
from infrastructure.monitoring.monitoring_alerts import get_alert_manager


class PerformanceMonitor(MetricCollectorProtocol):
    """
    Система мониторинга производительности.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация системы мониторинга.
        Args:
            config: Конфигурация мониторинга
        """
        self.config = config or {}
        # Метрики
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.metric_configs: Dict[str, Dict[str, Any]] = {}
        # Системные метрики
        self.system_metrics: Dict[str, Any] = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_io": {"bytes_sent": 0, "bytes_recv": 0},
        }
        # Производительность приложения
        self.app_metrics = {
            "request_count": 0,
            "error_count": 0,
            "avg_response_time": 0.0,
            "active_connections": 0,
            "queue_size": 0,
        }
        # Флаги состояния
        self.is_running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        # Блокировки для потокобезопасности
        self._lock = threading.RLock()
        self._metrics_lock = threading.RLock()

    def start_monitoring(self) -> None:
        """Запуск мониторинга."""
        if self.is_running:
            return
        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Performance monitoring started")

    def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Performance monitoring stopped")

    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> None:
        """
        Запись метрики.
        Args:
            name: Имя метрики
            value: Значение
            metric_type: Тип метрики
            labels: Метки
            description: Описание
        """
        metric = Metric(
            name=name,
            value=value,
            type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {},
            description=description,
        )
        with self._metrics_lock:
            if name not in self.metrics:
                self.metrics[name] = []
            if not isinstance(self.metrics[name], list):
                self.metrics[name] = []
            self.metrics[name].append(metric)
            # Ограничиваем количество метрик в памяти
            if len(self.metrics[name]) > 1000:
                self.metrics[name] = self.metrics[name][-500:]
        # Проверяем алерты
        get_alert_manager().check_metric_alert(name, value)

    def record_counter(
        self, name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Запись счетчика."""
        self.record_metric(name, increment, MetricType.COUNTER, labels)

    def record_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Запись датчика."""
        self.record_metric(name, value, MetricType.GAUGE, labels)

    def record_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Запись гистограммы."""
        self.record_metric(name, value, MetricType.HISTOGRAM, labels)

    def record_timer(
        self, name: str, duration: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Запись таймера."""
        self.record_metric(name, duration, MetricType.TIMER, labels)

    def collect_metrics(self) -> Dict[str, float]:
        """Сбор всех метрик."""
        with self._metrics_lock:
            result: Dict[str, float] = {}
            for name, metrics in self.metrics.items():
                if metrics:
                    result[name] = float(metrics[-1].value)
            return result

    def get_metric(self, name: str) -> Optional[float]:
        """Получение конкретной метрики."""
        with self._metrics_lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
            return None

    def reset_metrics(self) -> None:
        """Сброс всех метрик."""
        with self._metrics_lock:
            self.metrics.clear()

    def get_metrics(
        self, name: Optional[str] = None, limit: int = 100
    ) -> Dict[str, List[Metric]]:
        """
        Получение метрик.
        Args:
            name: Имя метрики (если None, возвращаются все)
            limit: Лимит записей
        Returns:
            Словарь метрик
        """
        with self._metrics_lock:
            if name:
                return {name: self.metrics.get(name, [])[-limit:]}
            else:
                return {
                    metric_name: metrics[-limit:]
                    for metric_name, metrics in self.metrics.items()
                }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Получение системных метрик."""
        return self.system_metrics.copy()

    def get_app_metrics(self) -> Dict[str, Any]:
        """Получение метрик приложения."""
        return self.app_metrics.copy()

    def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга."""
        while self.is_running:
            try:
                self._collect_system_metrics()
                self._collect_app_metrics()
                self._check_system_alerts()
                time.sleep(5)  # Обновляем каждые 5 секунд
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Увеличиваем интервал при ошибке

    def _collect_system_metrics(self) -> None:
        """Сбор системных метрик."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.system_metrics["cpu_usage"] = cpu_percent
            self.record_gauge("system.cpu_usage", cpu_percent)
            # Memory
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.system_metrics["memory_usage"] = memory_percent
            self.record_gauge("system.memory_usage", memory_percent)
            # Disk
            disk = psutil.disk_usage("/")
            disk_percent = (disk.used / disk.total) * 100
            self.system_metrics["disk_usage"] = disk_percent
            self.record_gauge("system.disk_usage", disk_percent)
            # Network
            network = psutil.net_io_counters()
            self.system_metrics["network_io"]["bytes_sent"] = network.bytes_sent
            self.system_metrics["network_io"]["bytes_recv"] = network.bytes_recv
            self.record_gauge("system.network.bytes_sent", network.bytes_sent)
            self.record_gauge("system.network.bytes_recv", network.bytes_recv)
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_app_metrics(self) -> None:
        """Сбор метрик приложения."""
        try:
            # Обновляем метрики приложения
            self.record_gauge("app.request_count", self.app_metrics["request_count"])
            self.record_gauge("app.error_count", self.app_metrics["error_count"])
            self.record_gauge(
                "app.avg_response_time", self.app_metrics["avg_response_time"]
            )
            self.record_gauge(
                "app.active_connections", self.app_metrics["active_connections"]
            )
            self.record_gauge("app.queue_size", self.app_metrics["queue_size"])
        except Exception as e:
            logger.error(f"Error collecting app metrics: {e}")

    def _check_system_alerts(self) -> None:
        """Проверка системных алертов."""
        try:
            # CPU алерты
            cpu_usage = float(self.system_metrics["cpu_usage"])
            if cpu_usage > 80.0:
                get_alert_manager().create_alert(
                    title="High CPU Usage",
                    message=f"CPU usage is {cpu_usage:.1f}%",
                    severity=AlertSeverity.WARNING,
                    metric_name="cpu_usage",
                    threshold=80.0,
                    current_value=cpu_usage,
                )
            # Memory алерты
            memory_usage = float(self.system_metrics["memory_usage"])
            if memory_usage > 85.0:
                get_alert_manager().create_alert(
                    title="High Memory Usage",
                    message=f"Memory usage is {memory_usage:.1f}%",
                    severity=AlertSeverity.WARNING,
                    metric_name="memory_usage",
                    threshold=85.0,
                    current_value=memory_usage,
                )
            # Disk алерты
            disk_usage = float(self.system_metrics["disk_usage"])
            if disk_usage > 90.0:
                get_alert_manager().create_alert(
                    title="High Disk Usage",
                    message=f"Disk usage is {disk_usage:.1f}%",
                    severity=AlertSeverity.ERROR,
                    metric_name="disk_usage",
                    threshold=90.0,
                    current_value=disk_usage,
                )
        except Exception as e:
            logger.error(f"Error checking system alerts: {e}")


# Глобальный экземпляр монитора производительности
_global_monitor: Optional[PerformanceMonitor] = None


def get_monitor() -> PerformanceMonitor:
    """Получение глобального монитора производительности."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def start_monitoring() -> None:
    """Запуск мониторинга."""
    get_monitor().start_monitoring()


def stop_monitoring() -> None:
    """Остановка мониторинга."""
    get_monitor().stop_monitoring()


def record_metric(
    name: str,
    value: float,
    metric_type: MetricType = MetricType.GAUGE,
    labels: Optional[Dict[str, str]] = None,
    description: Optional[str] = None,
) -> None:
    """Запись метрики."""
    get_monitor().record_metric(name, value, metric_type, labels, description)


def record_counter(
    name: str, increment: float = 1.0, labels: Optional[Dict[str, str]] = None
) -> None:
    """Запись счетчика."""
    get_monitor().record_counter(name, increment, labels)


def record_gauge(
    name: str, value: float, labels: Optional[Dict[str, str]] = None
) -> None:
    """Запись датчика."""
    get_monitor().record_gauge(name, value, labels)


def record_histogram(
    name: str, value: float, labels: Optional[Dict[str, str]] = None
) -> None:
    """Запись гистограммы."""
    get_monitor().record_histogram(name, value, labels)


def record_timer(
    name: str, duration: float, labels: Optional[Dict[str, str]] = None
) -> None:
    """Запись таймера."""
    get_monitor().record_timer(name, duration, labels)
