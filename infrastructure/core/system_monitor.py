# DEPRECATED: Используйте shared.monitoring

"""
Система мониторинга и алертов для Syntra.

Особенности:
- Мониторинг производительности в реальном времени
- Система алертов с различными уровнями критичности
- Метрики системы, памяти, CPU, сети
- Интеграция с внешними системами мониторинга
- Автоматическое восстановление после сбоев
- Дашборд для визуализации метрик
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil
from loguru import logger


class AlertLevel(Enum):
    """Уровни критичности алертов."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Типы метрик."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Alert:
    """Структура алерта."""

    id: str
    level: AlertLevel
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


@dataclass
class Metric:
    """Структура метрики."""

    name: str
    value: float
    type: MetricType
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


class PerformanceMonitor:
    """Монитор производительности системы."""

    def __init__(self, sample_interval: float = 1.0, max_samples: int = 3600) -> None:
        self.sample_interval = sample_interval
        self.max_samples = max_samples

        # Хранилище метрик
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))

        # Состояние системы
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Базовые метрики
        self._register_base_metrics()

    def _register_base_metrics(self) -> None:
        """Регистрация базовых метрик системы."""
        self.register_metric(
            "system.cpu_percent", MetricType.GAUGE, "CPU usage percentage"
        )
        self.register_metric(
            "system.memory_percent", MetricType.GAUGE, "Memory usage percentage"
        )
        self.register_metric(
            "system.disk_usage", MetricType.GAUGE, "Disk usage percentage"
        )
        self.register_metric(
            "system.network_io", MetricType.COUNTER, "Network I/O bytes"
        )
        self.register_metric(
            "system.process_count", MetricType.GAUGE, "Number of processes"
        )
        self.register_metric(
            "system.thread_count", MetricType.GAUGE, "Number of threads"
        )

    def register_metric(
        self, name: str, metric_type: MetricType, description: Optional[str] = None
    ) -> None:
        """Регистрация новой метрики."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.max_samples)

    def record_metric(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Запись метрики."""
        metric = Metric(
            name=name,
            value=value,
            type=MetricType.GAUGE,
            labels=labels or {},
            timestamp=datetime.now(),
        )

        self.metrics[name].append(metric)

    async def start_monitoring(self) -> None:
        """Запуск мониторинга системы."""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга."""
        if not self.is_running:
            return

        self.is_running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.sample_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.sample_interval)

    async def _collect_system_metrics(self) -> None:
        """Сбор системных метрик."""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.record_metric("system.cpu_percent", cpu_percent)

            # Memory
            memory = psutil.virtual_memory()
            self.record_metric("system.memory_percent", memory.percent)
            self.record_metric(
                "system.memory_available", memory.available / (1024**3)
            )  # GB

            # Disk
            disk = psutil.disk_usage("/")
            self.record_metric("system.disk_usage", disk.percent)
            self.record_metric("system.disk_free", disk.free / (1024**3))  # GB

            # Network
            network = psutil.net_io_counters()
            self.record_metric("system.network_bytes_sent", network.bytes_sent)
            self.record_metric("system.network_bytes_recv", network.bytes_recv)

            # Process info
            process = psutil.Process()
            self.record_metric("system.process_cpu_percent", process.cpu_percent())
            self.record_metric(
                "system.process_memory_mb", process.memory_info().rss / (1024**2)
            )
            self.record_metric("system.process_threads", process.num_threads())

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def get_metrics(
        self, name: Optional[str] = None, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Получение метрик."""
        if name:
            if name not in self.metrics:
                return {}

            metrics = list(self.metrics[name])
            if limit:
                metrics = metrics[-limit:]

            return {"name": name, "metrics": [self._metric_to_dict(m) for m in metrics]}

        # Все метрики
        result = {}
        for metric_name, metric_deque in self.metrics.items():
            metrics = list(metric_deque)
            if limit:
                metrics = metrics[-limit:]

            result[metric_name] = [self._metric_to_dict(m) for m in metrics]

        return result

    def _metric_to_dict(self, metric: Metric) -> Dict[str, Any]:
        """Преобразование метрики в словарь."""
        return {
            "name": metric.name,
            "value": metric.value,
            "type": metric.type.value,
            "timestamp": metric.timestamp.isoformat(),
            "labels": metric.labels,
            "description": metric.description,
        }


class AlertManager:
    """Менеджер алертов."""

    def __init__(self, max_alerts: int = 1000) -> None:
        self.max_alerts = max_alerts
        self.alerts: List[Alert] = []
        self.alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self.alert_rules: List[AlertRule] = []

        # Состояние
        self.is_running = False
        self.evaluation_task: Optional[asyncio.Task] = None

    def add_alert_handler(self, level: AlertLevel, handler: Callable) -> None:
        """Добавление обработчика алертов."""
        self.alert_handlers[level].append(handler)

    def add_alert_rule(self, rule: "AlertRule") -> None:
        """Добавление правила алерта."""
        self.alert_rules.append(rule)

    def create_alert(
        self,
        level: AlertLevel,
        message: str,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Alert:
        """Создание нового алерта."""
        alert = Alert(
            id=f"{source}_{int(time.time())}",
            level=level,
            message=message,
            source=source,
            metadata=metadata or {},
        )

        self.alerts.append(alert)

        # Ограничение количества алертов
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts :]

        # Вызов обработчиков
        asyncio.create_task(self._notify_handlers(alert))

        logger.log(
            (
                "WARNING"
                if level == AlertLevel.WARNING
                else (
                    "ERROR"
                    if level == AlertLevel.ERROR
                    else "CRITICAL" if level == AlertLevel.CRITICAL else "INFO"
                )
            ),
            f"Alert [{level.value.upper()}] {source}: {message}",
        )

        return alert

    async def start_evaluation(self) -> None:
        """Запуск автоматической оценки алертов."""
        if self.is_running:
            return

        self.is_running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert evaluation started")

    async def stop_evaluation(self) -> None:
        """Остановка оценки алертов."""
        if not self.is_running:
            return

        self.is_running = False

        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass

        logger.info("Alert evaluation stopped")

    async def _evaluation_loop(self) -> None:
        """Цикл оценки правил алертов."""
        while self.is_running:
            try:
                await self._evaluate_rules()
                await asyncio.sleep(30)  # Проверка каждые 30 секунд

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                await asyncio.sleep(30)

    async def _evaluate_rules(self) -> None:
        """Оценка правил алертов."""
        for rule in self.alert_rules:
            try:
                if await rule.evaluate():
                    self.create_alert(
                        level=rule.level,
                        message=rule.message,
                        source=rule.source,
                        metadata=rule.get_metadata(),
                    )
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")

    async def _notify_handlers(self, alert: Alert) -> None:
        """Уведомление обработчиков о новом алерте."""
        handlers = self.alert_handlers[alert.level]

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        acknowledged: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List[Alert]:
        """Получение алертов с фильтрацией."""
        alerts = self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]

        if limit:
            alerts = alerts[-limit:]

        return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Подтверждение алерта."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Разрешение алерта."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False


class AlertRule:
    """Правило для автоматического создания алертов."""

    def __init__(
        self,
        name: str,
        condition: Callable,
        level: AlertLevel,
        message: str,
        source: str,
    ):
        self.name = name
        self.condition = condition
        self.level = level
        self.message = message
        self.source = source
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0

    async def evaluate(self) -> bool:
        """Оценка условия правила."""
        try:
            if asyncio.iscoroutinefunction(self.condition):
                result = await self.condition()
            else:
                result = self.condition()

            if result:
                self.last_triggered = datetime.now()
                self.trigger_count += 1
                return True

            return False

        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.name}: {e}")
            return False

    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных правила."""
        return {
            "rule_name": self.name,
            "trigger_count": self.trigger_count,
            "last_triggered": (
                self.last_triggered.isoformat() if self.last_triggered else None
            ),
        }


class SystemMonitor:
    """Основной монитор системы."""

    def __init__(self) -> None:
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()

        # Интеграции
        self.integrations: Dict[str, "MonitoringIntegration"] = {}

        # Состояние
        self.is_running = False

    async def start(self) -> None:
        """Запуск системы мониторинга."""
        if self.is_running:
            return

        # Запуск компонентов
        await self.performance_monitor.start_monitoring()
        await self.alert_manager.start_evaluation()

        # Настройка базовых алертов
        self._setup_default_alerts()

        # Запуск интеграций
        for integration in self.integrations.values():
            await integration.start()

        self.is_running = True
        logger.info("System monitoring started")

    async def stop(self) -> None:
        """Остановка системы мониторинга."""
        if not self.is_running:
            return

        # Остановка компонентов
        await self.performance_monitor.stop_monitoring()
        await self.alert_manager.stop_evaluation()

        # Остановка интеграций
        for integration in self.integrations.values():
            await integration.stop()

        self.is_running = False
        logger.info("System monitoring stopped")

    def _setup_default_alerts(self) -> None:
        """Настройка базовых алертов."""
        # CPU alert
        cpu_rule = AlertRule(
            name="high_cpu_usage",
            condition=lambda: psutil.cpu_percent() > 80,
            level=AlertLevel.WARNING,
            message="High CPU usage detected",
            source="system",
        )
        self.alert_manager.add_alert_rule(cpu_rule)

        # Memory alert
        memory_rule = AlertRule(
            name="high_memory_usage",
            condition=lambda: psutil.virtual_memory().percent > 85,
            level=AlertLevel.WARNING,
            message="High memory usage detected",
            source="system",
        )
        self.alert_manager.add_alert_rule(memory_rule)

        # Disk alert
        disk_rule = AlertRule(
            name="low_disk_space",
            condition=lambda: psutil.disk_usage("/").percent > 90,
            level=AlertLevel.ERROR,
            message="Low disk space detected",
            source="system",
        )
        self.alert_manager.add_alert_rule(disk_rule)

    def add_integration(self, name: str, integration: "MonitoringIntegration") -> None:
        """Добавление интеграции мониторинга."""
        self.integrations[name] = integration

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса системы мониторинга."""
        return {
            "is_running": self.is_running,
            "performance_metrics": self.performance_monitor.get_metrics(limit=100),
            "alerts": {
                "total": len(self.alert_manager.alerts),
                "unacknowledged": len(
                    [a for a in self.alert_manager.alerts if not a.acknowledged]
                ),
                "critical": len(
                    [
                        a
                        for a in self.alert_manager.alerts
                        if a.level == AlertLevel.CRITICAL
                    ]
                ),
            },
            "integrations": list(self.integrations.keys()),
        }


class MonitoringIntegration:
    """Базовый класс для интеграций мониторинга."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_connected = False
        self.connection_retries = 0
        self.max_retries = 3

    async def start(self) -> None:
        """Запуск интеграции."""
        try:
            await self._connect()
            self.is_connected = True
            logger.info(f"Monitoring integration {self.name} started")
        except Exception as e:
            logger.error(f"Failed to start monitoring integration {self.name}: {e}")
            self.is_connected = False

    async def stop(self) -> None:
        """Остановка интеграции."""
        try:
            await self._disconnect()
            self.is_connected = False
            logger.info(f"Monitoring integration {self.name} stopped")
        except Exception as e:
            logger.error(f"Error stopping monitoring integration {self.name}: {e}")

    async def send_metric(self, metric: Metric) -> None:
        """Отправка метрики."""
        if not self.is_connected:
            logger.warning(f"Cannot send metric: {self.name} integration not connected")
            return

        try:
            await self._send_metric_impl(metric)
        except Exception as e:
            logger.error(f"Error sending metric via {self.name}: {e}")
            await self._handle_connection_error()

    async def send_alert(self, alert: Alert) -> None:
        """Отправка алерта."""
        if not self.is_connected:
            logger.warning(f"Cannot send alert: {self.name} integration not connected")
            return

        try:
            await self._send_alert_impl(alert)
        except Exception as e:
            logger.error(f"Error sending alert via {self.name}: {e}")
            await self._handle_connection_error()

    async def _connect(self) -> None:
        """Подключение к внешней системе мониторинга."""
        # Базовая реализация - можно переопределить в дочерних классах
        await asyncio.sleep(0.1)  # Имитация подключения

    async def _disconnect(self) -> None:
        """Отключение от внешней системы мониторинга."""
        # Базовая реализация - можно переопределить в дочерних классах
        await asyncio.sleep(0.1)  # Имитация отключения

    async def _send_metric_impl(self, metric: Metric) -> None:
        """Реализация отправки метрики."""
        # Базовая реализация - логирование
        logger.debug(f"[{self.name}] Metric: {metric.name}={metric.value}")

    async def _send_alert_impl(self, alert: Alert) -> None:
        """Реализация отправки алерта."""
        # Базовая реализация - логирование
        logger.info(
            f"[{self.name}] Alert: {alert.level.value.upper()} - {alert.message}"
        )

    async def _handle_connection_error(self) -> None:
        """Обработка ошибки подключения."""
        self.connection_retries += 1
        if self.connection_retries >= self.max_retries:
            logger.error(f"Max retries reached for {self.name} integration")
            self.is_connected = False
        else:
            logger.warning(
                f"Connection error for {self.name}, retry {self.connection_retries}/{self.max_retries}"
            )
            await asyncio.sleep(5)  # Пауза перед повторной попыткой


# Глобальный экземпляр монитора
system_monitor = SystemMonitor()
