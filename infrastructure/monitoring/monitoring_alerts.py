"""
Модуль алертов производительности.
Включает:
- Создание алертов
- Обработка алертов
- Управление алертами
"""

import asyncio
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from domain.type_definitions.monitoring_types import (
    Alert,
    AlertHandlerProtocol,
    AlertSeverity,
)


class AlertManager(AlertHandlerProtocol):
    """
    Менеджер алертов.
    """

    def __init__(self) -> None:
        """Инициализация менеджера алертов."""
        self.alerts: List[Alert] = []
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_thresholds: Dict[str, float] = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 1000.0,  # ms
        }
        self._lock = threading.RLock()

    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity,
        metric_name: Optional[str] = None,
        threshold: Optional[float] = None,
        current_value: Optional[float] = None,
    ) -> Alert:
        """
        Создание алерта.
        Args:
            title: Заголовок алерта
            message: Сообщение алерта
            severity: Уровень серьезности
            metric_name: Имя метрики
            threshold: Пороговое значение
            current_value: Текущее значение
        Returns:
            Созданный алерт
        """
        alert = Alert(
            id=f"alert_{int(time.time() * 1000000)}",
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
        )
        with self._lock:
            self.alerts.append(alert)
            # Ограничиваем количество алертов в памяти
            if len(self.alerts) > 1000:
                self.alerts = self.alerts[-500:]
        # Вызываем обработчики алертов
        self.handle_alert(alert)
        logger.warning(f"Alert created: {title} ({severity.value})")
        return alert

    def handle_alert(self, alert: Alert) -> None:
        """
        Обработка алерта.
        Args:
            alert: Алерт для обработки
        """
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        Добавление обработчика алертов.
        Args:
            handler: Функция-обработчик
        """
        self.alert_handlers.append(handler)

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Разрешение алерта.
        Args:
            alert_id: ID алерта
        Returns:
            True если алерт найден и разрешен
        """
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    logger.info(f"Alert resolved: {alert.title}")
                    return True
        return False

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Получение алертов.
        Args:
            severity: Фильтр по уровню серьезности
            resolved: Фильтр по статусу разрешения
            limit: Лимит записей
        Returns:
            Список алертов
        """
        with self._lock:
            alerts = self.alerts
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            if resolved is not None:
                alerts = [a for a in alerts if a.resolved == resolved]
            return alerts[-limit:]

    def get_unresolved_alerts(self) -> List[Alert]:
        """
        Получение неразрешенных алертов.
        Returns:
            Список неразрешенных алертов
        """
        return self.get_alerts(resolved=False)

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """
        Получение алертов по уровню серьезности.
        Args:
            severity: Уровень серьезности
        Returns:
            Список алертов
        """
        return self.get_alerts(severity=severity)

    def check_metric_alert(self, metric_name: str, value: float) -> Optional[Alert]:
        """
        Проверка алерта для метрики.
        Args:
            metric_name: Имя метрики
            value: Значение метрики
        Returns:
            Созданный алерт или None
        """
        if metric_name in self.alert_thresholds:
            threshold = self.alert_thresholds[metric_name]
            if value > threshold:
                severity = (
                    AlertSeverity.ERROR
                    if value > threshold * 1.5
                    else AlertSeverity.WARNING
                )
                return self.create_alert(
                    title=f"High {metric_name}",
                    message=f"{metric_name} is {value:.2f} (threshold: {threshold:.2f})",
                    severity=severity,
                    metric_name=metric_name,
                    threshold=threshold,
                    current_value=value,
                )
        return None

    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """
        Установка порога для метрики.
        Args:
            metric_name: Имя метрики
            threshold: Пороговое значение
        """
        self.alert_thresholds[metric_name] = threshold
        logger.info(f"Set threshold for {metric_name}: {threshold}")

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики алертов.
        Returns:
            Статистика алертов
        """
        with self._lock:
            total_alerts = len(self.alerts)
            unresolved_alerts = len([a for a in self.alerts if not a.resolved])
            severity_counts: Dict[str, int] = defaultdict(int)
            for alert in self.alerts:
                severity_counts[alert.severity.value] += 1
            return {
                "total_alerts": total_alerts,
                "unresolved_alerts": unresolved_alerts,
                "resolved_alerts": total_alerts - unresolved_alerts,
                "severity_counts": dict(severity_counts),
            }

    def clear_old_alerts(self, max_age_hours: int = 24) -> int:
        """
        Очистка старых алертов.
        Args:
            max_age_hours: Максимальный возраст в часах
        Returns:
            Количество удаленных алертов
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        removed_count = 0
        with self._lock:
            alerts_to_remove = []
            for alert in self.alerts:
                if alert.timestamp.timestamp() < cutoff_time and alert.resolved:
                    alerts_to_remove.append(alert)
            for alert in alerts_to_remove:
                self.alerts.remove(alert)
                removed_count += 1
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} old alerts")
        return removed_count


# Глобальный экземпляр менеджера алертов
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Получение глобального менеджера алертов."""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def create_alert(
    title: str,
    message: str,
    severity: AlertSeverity,
    metric_name: Optional[str] = None,
    threshold: Optional[float] = None,
    current_value: Optional[float] = None,
) -> Alert:
    """Создание алерта."""
    return get_alert_manager().create_alert(
        title, message, severity, metric_name, threshold, current_value
    )


def add_alert_handler(handler: Callable[[Alert], None]) -> None:
    """Добавление обработчика алертов."""
    get_alert_manager().add_alert_handler(handler)


def resolve_alert(alert_id: str) -> bool:
    """Разрешение алерта."""
    return get_alert_manager().resolve_alert(alert_id)


def get_alerts(
    severity: Optional[AlertSeverity] = None,
    resolved: Optional[bool] = None,
    limit: int = 100,
) -> List[Alert]:
    """Получение алертов."""
    return get_alert_manager().get_alerts(severity, resolved, limit)


class AlertRule:
    """Правило для создания алертов."""
    
    def __init__(self, name: str, condition: Callable[[], bool], **kwargs: Any) -> None:
        """Инициализация правила."""
        self.name: str = name
        self.condition: Callable[[], bool] = condition
        self.severity: AlertSeverity = kwargs.get("severity", AlertSeverity.WARNING)
        self.message: str = kwargs.get("message", f"Rule {name} triggered")
        self.source: str = kwargs.get("source", "system")
        self.metadata: Dict[str, Any] = kwargs.get("metadata", {})
        self.enabled: bool = kwargs.get("enabled", True)
    
    def evaluate(self) -> bool:
        """Оценка правила."""
        if not self.enabled:
            return False
        try:
            return self.condition()
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            return False
    
    def create_alert(self, manager: "AlertManager") -> Optional[Alert]:
        """Создание алерта на основе правила."""
        if self.evaluate():
            return manager.create_alert(
                title=self.name,
                message=self.message,
                severity=self.severity,
                metric_name=self.metadata.get("metric_name"),
                threshold=self.metadata.get("threshold"),
                current_value=self.metadata.get("current_value")
            )
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных правила."""
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "source": self.source,
            "enabled": self.enabled,
            "metadata": self.metadata
        }


class AlertHandler:
    """Обработчик алертов."""
    
    def __init__(self, name: str, handler_func: Callable[[Alert], Any], **kwargs: Any) -> None:
        """Инициализация обработчика."""
        self.name: str = name
        self.handler_func: Callable[[Alert], Any] = handler_func
        self.severity_filter: Optional[AlertSeverity] = kwargs.get("severity_filter")
        self.enabled: bool = kwargs.get("enabled", True)
        self.metadata: Dict[str, Any] = kwargs.get("metadata", {})
    
    async def handle(self, alert: Alert) -> None:
        """Обработка алерта."""
        if not self.enabled:
            return
        
        if self.severity_filter and alert.severity != self.severity_filter:
            return
        
        try:
            if asyncio.iscoroutinefunction(self.handler_func):
                await self.handler_func(alert)
            else:
                self.handler_func(alert)
        except Exception as e:
            logger.error(f"Error in alert handler {self.name}: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных обработчика."""
        return {
            "name": self.name,
            "severity_filter": self.severity_filter.value if self.severity_filter else None,
            "enabled": self.enabled,
            "metadata": self.metadata
        }
