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

from domain.type_definitions.monitoring_types import (
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

    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Получение значения метрики."""
        try:
            if metric_name in self.system_metrics:
                value = self.system_metrics[metric_name]
                return float(value) if value is not None else None
            return None
        except (ValueError, TypeError) as e:
            logger.error(f"Error getting metric {metric_name}: {e}")
            return None
    
    async def _check_system_alerts(self) -> None:
        """Проверка системных алертов."""
        try:
            # Проверка критических метрик
            critical_metrics = [
                ('cpu_usage', 90.0),
                ('memory_usage', 85.0),
                ('error_rate', 5.0)
            ]
            
            for metric_name, threshold in critical_metrics:
                current_value = self.get_metric_value(metric_name)
                if current_value is not None and isinstance(current_value, (int, float)):
                    if current_value > threshold:
                        await self.alert_manager.create_alert(
                            title=f"High {metric_name}",
                            message=f"{metric_name} is {current_value}%, threshold: {threshold}%",
                            severity="high",
                            current_value=float(current_value)  # Явное приведение к float
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


def get_tracer() -> "TraceProtocol":
    """Получить трейсер."""
    from infrastructure.monitoring.trace_monitor import get_trace_monitor
    return get_trace_monitor()


def get_dashboard() -> Dict[str, Any]:
    """Получение данных для дашборда мониторинга."""
    monitor = get_monitor()
    
    # Системные метрики
    system_metrics = monitor.get_system_metrics()
    
    # Метрики приложения
    app_metrics = monitor.get_app_metrics()
    
    # Собранные метрики
    collected_metrics = monitor.collect_metrics()
    
    # Статистика алертов
    alert_manager = get_alert_manager()
    alert_stats = {
        "total_alerts": len(alert_manager.get_alerts()),
        "critical_alerts": len([a for a in alert_manager.get_alerts() if a.severity == AlertSeverity.CRITICAL]),
        "warning_alerts": len([a for a in alert_manager.get_alerts() if a.severity == AlertSeverity.WARNING]),
        "info_alerts": len([a for a in alert_manager.get_alerts() if a.severity == AlertSeverity.INFO])
    }
    
    return {
        "system_metrics": system_metrics,
        "app_metrics": app_metrics,
        "collected_metrics": collected_metrics,
        "alert_stats": alert_stats,
        "monitoring_status": {
            "is_running": monitor.is_running,
            "timestamp": datetime.now().isoformat()
        }
    }


def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    source: str = "performance_monitor",
    metadata: Optional[Dict[str, Any]] = None,
) -> Alert:
    """Создать алерт."""
    alert_manager = get_alert_manager()
    return alert_manager.create_alert(title, message, severity, source, metadata or {})
