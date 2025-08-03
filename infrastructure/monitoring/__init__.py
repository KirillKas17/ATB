"""
Модуль мониторинга производительности.

Включает:
- Логирование с трейсингом
- Мониторинг производительности
- Систему алертов
- Дашборд мониторинга
"""

# Импорты типов
from domain.types.monitoring_types import (
    Alert,
    AlertHandlerProtocol,
    AlertSeverity,
    LogContext,
    LogEntry,
    LoggerProtocol,
    LogLevel,
    Metric,
    MetricCollectorProtocol,
    MetricType,
    TraceProtocol,
    TraceSpan,
)

# Импорты из logging_system
from .logging_system import StructuredLogger, get_logger, setup_logging

# Импорты из logging_tracing
from .logging_tracing import (
    RequestTracer,
    get_tracer,
    log_critical,
    log_debug,
    log_error,
    log_info,
    log_trace,
    log_warning,
)

# Импорты из monitoring_alerts
from .monitoring_alerts import (
    AlertManager,
    add_alert_handler,
    create_alert,
    get_alert_manager,
    get_alerts,
    resolve_alert,
)

# Импорты из monitoring_dashboard
from .monitoring_dashboard import (
    MonitoringDashboard,
    export_metrics,
    get_alert_summary,
    get_dashboard,
    get_dashboard_data,
    get_performance_summary,
)

# Импорты из monitoring_tracing
from .monitoring_tracing import (
    PerformanceTracer,
    add_child_span,
    end_trace,
    start_trace,
)

# Импорты из performance_monitor
from .performance_monitor import (
    PerformanceMonitor,
    get_monitor,
    record_counter,
    record_gauge,
    record_histogram,
    record_metric,
    record_timer,
    start_monitoring,
    stop_monitoring,
)

__all__ = [
    # Логирование
    "StructuredLogger",
    "get_logger",
    "setup_logging",
    # Трейсинг логирования
    "RequestTracer",
    "get_tracer",
    "log_trace",
    "log_debug",
    "log_info",
    "log_warning",
    "log_error",
    "log_critical",
    # Мониторинг производительности
    "PerformanceMonitor",
    "get_monitor",
    "start_monitoring",
    "stop_monitoring",
    "record_metric",
    "record_counter",
    "record_gauge",
    "record_histogram",
    "record_timer",
    # Трейсинг производительности
    "PerformanceTracer",
    "start_trace",
    "end_trace",
    "add_child_span",
    # Алерты
    "AlertManager",
    "get_alert_manager",
    "create_alert",
    "add_alert_handler",
    "resolve_alert",
    "get_alerts",
    # Дашборд
    "MonitoringDashboard",
    "get_dashboard",
    "get_dashboard_data",
    "export_metrics",
    "get_performance_summary",
    "get_alert_summary",
    # Типы
    "Metric",
    "MetricType",
    "Alert",
    "AlertSeverity",
    "TraceSpan",
    "LogLevel",
    "LogContext",
    "LogEntry",
    "MetricCollectorProtocol",
    "AlertHandlerProtocol",
    "TraceProtocol",
    "LoggerProtocol",
]
