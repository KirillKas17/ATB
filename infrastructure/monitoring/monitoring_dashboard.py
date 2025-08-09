"""
Модуль дашборда мониторинга.

Включает:
- Данные для дашборда
- Экспорт метрик
- Статистика производительности
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from domain.type_definitions.monitoring_types import Alert, Metric, TraceSpan


class MonitoringDashboard:
    """
    Дашборд мониторинга.
    """

    def __init__(self) -> None:
        """Инициализация дашборда."""
        pass

    def get_dashboard_data(
        self,
        system_metrics: Dict[str, float],
        app_metrics: Dict[str, Any],
        recent_alerts: List[Alert],
        metrics_data: Dict[str, List[Metric]],
    ) -> Dict[str, Any]:
        """
        Получение данных для дашборда.

        Args:
            system_metrics: Системные метрики
            app_metrics: Метрики приложения
            recent_alerts: Последние алерты
            metrics_data: Данные метрик

        Returns:
            Данные для дашборда
        """
        try:
            # Статистика метрик
            metrics_stats = {}
            for name, metrics in metrics_data.items():
                if metrics:
                    values = [m.value for m in metrics]
                    metrics_stats[name] = {
                        "count": len(values),
                        "min": min(values),
                        "max": max(values),
                        "avg": sum(values) / len(values),
                        "latest": values[-1] if values else 0,
                    }

            return {
                "system_metrics": system_metrics,
                "app_metrics": app_metrics,
                "recent_alerts": [
                    {
                        "id": alert.id,
                        "title": alert.title,
                        "message": alert.message,
                        "severity": alert.severity.value,
                        "timestamp": alert.timestamp.isoformat(),
                    }
                    for alert in recent_alerts
                ],
                "metrics_stats": metrics_stats,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}

    def export_metrics(
        self,
        metrics_data: Dict[str, List[Metric]],
        alerts_data: List[Alert],
        traces_data: List[TraceSpan],
        format: str = "json",
    ) -> str:
        """
        Экспорт метрик.

        Args:
            metrics_data: Данные метрик
            alerts_data: Данные алертов
            traces_data: Данные трейсов
            format: Формат экспорта

        Returns:
            Экспортированные данные
        """
        try:
            data = {
                "metrics": metrics_data,
                "alerts": [
                    {
                        "id": alert.id,
                        "title": alert.title,
                        "message": alert.message,
                        "severity": alert.severity.value,
                        "timestamp": alert.timestamp.isoformat(),
                        "resolved": alert.resolved,
                    }
                    for alert in alerts_data
                ],
                "traces": [
                    {
                        "id": trace.id,
                        "name": trace.name,
                        "duration": trace.duration,
                        "start_time": trace.start_time.isoformat(),
                        "end_time": (
                            trace.end_time.isoformat() if trace.end_time else None
                        ),
                    }
                    for trace in traces_data
                ],
                "export_timestamp": datetime.now().isoformat(),
            }

            if format == "json":
                return json.dumps(data, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return "{}"

    def get_performance_summary(
        self, metrics_data: Dict[str, List[Metric]], alerts_data: List[Alert]
    ) -> Dict[str, Any]:
        """
        Получение сводки производительности.

        Args:
            metrics_data: Данные метрик
            alerts_data: Данные алертов

        Returns:
            Сводка производительности
        """
        try:
            # Анализ метрик
            total_metrics = sum(len(metrics) for metrics in metrics_data.values())
            active_alerts = len([a for a in alerts_data if not a.resolved])

            # Статистика по типам метрик
            metric_types: Dict[str, int] = defaultdict(int)
            for metrics in metrics_data.values():
                for metric in metrics:
                    metric_types[metric.type.value] += 1

            # Топ метрик по частоте
            metric_frequency: Dict[str, int] = defaultdict(int)
            for metrics in metrics_data.values():
                for metric in metrics:
                    metric_frequency[metric.name] += 1

            top_metrics: Dict[str, int] = dict(
                sorted(metric_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            )

            return {
                "total_metrics": total_metrics,
                "active_alerts": active_alerts,
                "metric_types": dict(metric_types),
                "top_metrics": top_metrics,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {}

    def get_alert_summary(self, alerts_data: List[Alert]) -> Dict[str, Any]:
        """
        Получение сводки алертов.

        Args:
            alerts_data: Данные алертов

        Returns:
            Сводка алертов
        """
        try:
            severity_counts: Dict[str, int] = defaultdict(int)
            metric_alerts: Dict[str, int] = defaultdict(int)

            for alert in alerts_data:
                severity_counts[alert.severity.value] += 1
                if alert.metric_name:
                    metric_alerts[alert.metric_name] += 1

            return {
                "total_alerts": len(alerts_data),
                "unresolved_alerts": len([a for a in alerts_data if not a.resolved]),
                "severity_distribution": dict(severity_counts),
                "metric_alerts": dict(metric_alerts),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error getting alert summary: {e}")
            return {}


# Глобальный экземпляр дашборда
_global_dashboard: Optional[MonitoringDashboard] = None


def get_dashboard() -> MonitoringDashboard:
    """Получение глобального дашборда."""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = MonitoringDashboard()
    return _global_dashboard


def get_dashboard_data(
    system_metrics: Dict[str, float],
    app_metrics: Dict[str, Any],
    recent_alerts: List[Alert],
    metrics_data: Dict[str, List[Metric]],
) -> Dict[str, Any]:
    """Получение данных для дашборда."""
    return get_dashboard().get_dashboard_data(
        system_metrics, app_metrics, recent_alerts, metrics_data
    )


def export_metrics(
    metrics_data: Dict[str, List[Metric]],
    alerts_data: List[Alert],
    traces_data: List[TraceSpan],
    format: str = "json",
) -> str:
    """Экспорт метрик."""
    return get_dashboard().export_metrics(
        metrics_data, alerts_data, traces_data, format
    )


def get_performance_summary(
    metrics_data: Dict[str, List[Metric]], alerts_data: List[Alert]
) -> Dict[str, Any]:
    """Получение сводки производительности."""
    return get_dashboard().get_performance_summary(metrics_data, alerts_data)


def get_alert_summary(alerts_data: List[Alert]) -> Dict[str, Any]:
    """Получение сводки алертов."""
    return get_dashboard().get_alert_summary(alerts_data)


# Глобальные экземпляры компонентов мониторинга
_global_monitor: Optional[Any] = None
_global_alert_manager: Optional[Any] = None
_global_tracer: Optional[Any] = None
_monitoring_active: bool = False


def get_monitor() -> Any:
    """Получение глобального монитора."""
    global _global_monitor
    if _global_monitor is None:
        # Создаем простой монитор для тестов
        class SimpleMonitor:
            def __init__(self) -> None:
                self.metrics: List[Dict[str, Any]] = []
            
            def get_metrics(self) -> List[Dict[str, Any]]:
                return self.metrics
            
            def record_metric(self, name: str, value: float, tags: Dict[str, Any]) -> None:
                self.metrics.append({
                    "name": name,
                    "value": value,
                    "tags": tags,
                    "timestamp": datetime.now().isoformat()
                })
        
        _global_monitor = SimpleMonitor()
    return _global_monitor


def get_alert_manager() -> Any:
    """Получение глобального менеджера алертов."""
    global _global_alert_manager
    if _global_alert_manager is None:
        # Создаем простой менеджер алертов для тестов
        class SimpleAlertManager:
            def __init__(self) -> None:
                self.alerts: List[Dict[str, Any]] = []
            
            def get_alerts(self) -> List[Dict[str, Any]]:
                return self.alerts
            
            def create_alert(self, message: str, severity: str, source: str) -> None:
                self.alerts.append({
                    "message": message,
                    "severity": severity,
                    "source": source,
                    "timestamp": datetime.now().isoformat()
                })
        
        _global_alert_manager = SimpleAlertManager()
    return _global_alert_manager


def get_tracer() -> Any:
    """Получение глобального трейсера."""
    global _global_tracer
    if _global_tracer is None:
        # Создаем простой трейсер для тестов
        class SimpleTracer:
            def __init__(self) -> None:
                self.active_traces: List[Dict[str, Any]] = []
                self.completed_traces: List[Dict[str, Any]] = []
            
            def get_active_traces(self) -> List[Dict[str, Any]]:
                return self.active_traces
            
            def start_trace(self, trace_id: str, operation: str) -> Any:
                span = {
                    "trace_id": trace_id,
                    "operation": operation,
                    "start_time": datetime.now(),
                    "metrics": {}
                }
                self.active_traces.append(span)
                return span
            
            def end_trace(self, trace_id: str, status: str) -> None:
                for i, trace in enumerate(self.active_traces):
                    if trace["trace_id"] == trace_id:
                        trace["end_time"] = datetime.now()
                        trace["status"] = status
                        self.completed_traces.append(trace)
                        del self.active_traces[i]
                        break
        
        _global_tracer = SimpleTracer()
    return _global_tracer


def record_metric(name: str, value: float, tags: Dict[str, Any]) -> None:
    """Запись метрики."""
    monitor = get_monitor()
    if hasattr(monitor, 'record_metric'):
        monitor.record_metric(name, value, tags)


def create_alert(message: str, severity: str, source: str) -> None:
    """Создание алерта."""
    alert_manager = get_alert_manager()
    if hasattr(alert_manager, 'create_alert'):
        alert_manager.create_alert(message, severity, source)


def start_monitoring() -> None:
    """Запуск мониторинга."""
    global _monitoring_active
    _monitoring_active = True
    logger.info("Monitoring started")


def stop_monitoring() -> None:
    """Остановка мониторинга."""
    global _monitoring_active
    _monitoring_active = False
    logger.info("Monitoring stopped")
