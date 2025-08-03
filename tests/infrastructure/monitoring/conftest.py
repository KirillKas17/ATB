"""
Конфигурация pytest для тестов мониторинга.
Включает:
- Фикстуры для всех компонентов мониторинга
- Моки и заглушки
- Тестовые данные
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, AsyncMock
from infrastructure.monitoring import (
    PerformanceMonitor,
    AlertManager,
    PerformanceTracer,
    MonitoringDashboard,
    get_monitor,
    get_alert_manager,
    get_tracer,
    get_dashboard
)
from domain.types.monitoring_types import (
    Metric,
    MetricType,
    Alert,
    AlertSeverity,
    LogLevel,
    TraceSpan
)
@pytest.fixture
def performance_monitor() -> Any:
    """Фикстура для PerformanceMonitor."""
    monitor = PerformanceMonitor(name="test_monitor")
    yield monitor
    # Очистка после теста
    monitor.stop_monitoring()
@pytest.fixture
def alert_manager() -> Any:
    """Фикстура для AlertManager."""
    manager = AlertManager(name="test_alerts")
    yield manager
    # Очистка после теста
    if manager.is_running:
        asyncio.run(manager.stop_evaluation())
@pytest.fixture
def performance_tracer() -> Any:
    """Фикстура для PerformanceTracer."""
    tracer = PerformanceTracer(name="test_tracer")
    yield tracer
    # Очистка после теста
    tracer.cleanup_old_traces()
@pytest.fixture
def monitoring_dashboard() -> Any:
    """Фикстура для MonitoringDashboard."""
    dashboard = MonitoringDashboard(name="test_dashboard")
    yield dashboard
    # Очистка после теста
    dashboard.cleanup_old_data()
@pytest.fixture
def sample_metrics() -> Any:
    """Фикстура с тестовыми метриками."""
    return [
        Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now(),
            type=MetricType.GAUGE,
            labels={"host": "server1"}
        ),
        Metric(
            name="memory_usage",
            value=1024,
            timestamp=datetime.now(),
            type=MetricType.GAUGE,
            labels={"host": "server1"}
        ),
        Metric(
            name="response_time",
            value=150.0,
            timestamp=datetime.now(),
            type=MetricType.HISTOGRAM,
            labels={"endpoint": "/api/users"}
        ),
        Metric(
            name="request_count",
            value=1000,
            timestamp=datetime.now(),
            type=MetricType.COUNTER,
            labels={"endpoint": "/api/users"}
        )
    ]
@pytest.fixture
def sample_alerts() -> Any:
    """Фикстура с тестовыми алертами."""
    return [
        Alert(
            alert_id="alert-1",
            message="High CPU usage",
            severity=AlertSeverity.WARNING,
            source="cpu_monitor",
            timestamp=datetime.now()
        ),
        Alert(
            alert_id="alert-2",
            message="Database connection failed",
            severity=AlertSeverity.ERROR,
            source="database_monitor",
            timestamp=datetime.now(),
            exception=ValueError("Connection timeout")
        ),
        Alert(
            alert_id="alert-3",
            message="System overload",
            severity=AlertSeverity.CRITICAL,
            source="system_monitor",
            timestamp=datetime.now()
        )
    ]
@pytest.fixture
def sample_traces() -> Any:
    """Фикстура с тестовыми трейсами."""
    return [
        TraceSpan(
            trace_id="trace-1",
            span_id="span-1",
            operation="api_request",
            parent_span_id=None
        ),
        TraceSpan(
            trace_id="trace-1",
            span_id="span-2",
            operation="database_query",
            parent_span_id="span-1"
        ),
        TraceSpan(
            trace_id="trace-2",
            span_id="span-3",
            operation="file_operation",
            parent_span_id=None
        )
    ]
@pytest.fixture
def sample_log_entries() -> Any:
    """Фикстура с тестовыми записями логов."""
    from infrastructure.monitoring.logging_system import LogEntry, LogContext
    context = LogContext(
        request_id="req-123",
        user_id="user-456",
        session_id="session-789"
    )
    return [
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Application started",
            context=context
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="High memory usage detected",
            context=context
        ),
        LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Database connection failed",
            context=context,
            exception=ValueError("Connection timeout")
        )
    ]
@pytest.fixture
def mock_system_metrics() -> Any:
    """Фикстура с моковыми системными метриками."""
    return {
        "cpu_percent": 25.5,
        "memory_percent": 60.2,
        "disk_usage": 45.8,
        "network_io": {
            "bytes_sent": 1024000,
            "bytes_recv": 2048000
        },
        "process_count": 150,
        "load_average": [1.2, 1.1, 0.9]
    }
@pytest.fixture
def mock_app_metrics() -> Any:
    """Фикстура с моковыми метриками приложения."""
    return {
        "request_count": 1000,
        "error_rate": 0.02,
        "response_time_avg": 150.0,
        "active_connections": 25,
        "queue_size": 5,
        "cache_hit_rate": 0.85
    }
@pytest.fixture
def temp_log_file() -> Any:
    """Фикстура с временным файлом для логов."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        temp_file = f.name
    yield temp_file
    # Очистка после теста
    if os.path.exists(temp_file):
        os.unlink(temp_file)
@pytest.fixture
def temp_metrics_file() -> Any:
    """Фикстура с временным файлом для метрик."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_file = f.name
    yield temp_file
    # Очистка после теста
    if os.path.exists(temp_file):
        os.unlink(temp_file)
@pytest.fixture
def mock_alert_handler() -> Any:
    """Фикстура с моковым обработчиком алертов."""
    handler = Mock()
    handler.name = "test_handler"
    handler.handle = AsyncMock()
    return handler
@pytest.fixture
def mock_metric_collector() -> Any:
    """Фикстура с моковым сборщиком метрик."""
    collector = Mock()
    collector.collect_metrics = Mock(return_value=[])
    collector.collect_system_metrics = Mock(return_value={})
    collector.collect_app_metrics = Mock(return_value={})
    return collector
@pytest.fixture
def mock_trace_collector() -> Any:
    """Фикстура с моковым сборщиком трейсов."""
    collector = Mock()
    collector.collect_traces = Mock(return_value=[])
    collector.collect_performance_metrics = Mock(return_value={})
    return collector
@pytest.fixture
def sample_dashboard_config() -> Any:
    """Фикстура с конфигурацией дашборда."""
    from infrastructure.monitoring.monitoring_dashboard import DashboardConfig
    return DashboardConfig(
        title="Test Dashboard",
        refresh_interval=30,
        max_data_points=1000,
        theme="light"
    )
@pytest.fixture
def sample_chart_config() -> Any:
    """Фикстура с конфигурацией графика."""
    from infrastructure.monitoring.monitoring_dashboard import ChartConfig
    return ChartConfig(
        name="test_chart",
        title="Test Chart",
        chart_type="line",
        metrics=["cpu_usage", "memory_usage"],
        time_range=timedelta(hours=1)
    )
@pytest.fixture
def sample_alert_rule() -> Any:
    """Фикстура с правилом алерта."""
    from infrastructure.monitoring.monitoring_alerts import AlertRule
    def condition() -> Any:
        return True  # Всегда возвращает True для тестов
    return AlertRule(
        name="test_rule",
        condition=condition,
        severity=AlertSeverity.WARNING,
        message="Test rule triggered",
        source="test_source"
    )
@pytest.fixture
def sample_alert_handler() -> Any:
    """Фикстура с обработчиком алертов."""
    from infrastructure.monitoring.monitoring_alerts import AlertHandler
    def handle_func(alert) -> Any:
        pass  # Пустая функция для тестов
    return AlertHandler(
        name="test_handler",
        handle_func=handle_func
    )
@pytest.fixture
def mock_logger() -> Any:
    """Фикстура с моковым логгером."""
    logger = Mock()
    logger.debug = Mock()
    logger.info = Mock()
    logger.warning = Mock()
    logger.error = Mock()
    logger.critical = Mock()
    logger.log = Mock()
    return logger
@pytest.fixture
def mock_tracer() -> Any:
    """Фикстура с моковым трейсером."""
    tracer = Mock()
    tracer.start_trace = Mock()
    tracer.start_span = Mock()
    tracer.end_span = Mock()
    tracer.add_tag = Mock()
    tracer.add_log = Mock()
    tracer.get_trace = Mock()
    tracer.get_active_traces = Mock(return_value={})
    tracer.get_trace_statistics = Mock(return_value={})
    return tracer
@pytest.fixture
def mock_alert_manager() -> Any:
    """Фикстура с моковым менеджером алертов."""
    manager = Mock()
    manager.create_alert = Mock()
    manager.add_alert_rule = Mock()
    manager.add_alert_handler = Mock()
    manager.get_alerts = Mock(return_value=[])
    manager.acknowledge_alert = Mock(return_value=True)
    manager.resolve_alert = Mock(return_value=True)
    manager.start_evaluation = AsyncMock()
    manager.stop_evaluation = AsyncMock()
    return manager
@pytest.fixture
def mock_dashboard() -> Any:
    """Фикстура с моковым дашбордом."""
    dashboard = Mock()
    dashboard.add_metric_data = Mock()
    dashboard.add_alert_data = Mock()
    dashboard.create_chart = Mock()
    dashboard.get_chart_data = Mock(return_value={})
    dashboard.get_dashboard_data = Mock(return_value={})
    dashboard.get_performance_summary = Mock(return_value={})
    dashboard.get_alert_summary = Mock(return_value={})
    dashboard.export_metrics = Mock(return_value="")
    return dashboard
@pytest.fixture
def async_loop() -> Any:
    """Фикстура с асинхронным циклом событий."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
@pytest.fixture
def sample_performance_data() -> Any:
    """Фикстура с тестовыми данными производительности."""
    return {
        "cpu_usage": [25.5, 30.2, 28.7, 35.1, 22.3],
        "memory_usage": [1024, 1152, 1088, 1280, 960],
        "response_time": [150.0, 180.0, 120.0, 200.0, 100.0],
        "throughput": [1000, 950, 1100, 800, 1200],
        "error_rate": [0.01, 0.02, 0.005, 0.03, 0.008]
    }
@pytest.fixture
def sample_time_series_data() -> Any:
    """Фикстура с тестовыми временными рядами."""
    now = datetime.now()
    return {
        "timestamps": [now - timedelta(minutes=i) for i in range(60, 0, -1)],
        "values": [20.0 + (i % 10) for i in range(60)],
        "labels": {"metric": "test_metric"}
    }
@pytest.fixture
def sample_alert_data() -> Any:
    """Фикстура с тестовыми данными алертов."""
    return {
        "alerts_by_severity": {
            "INFO": 5,
            "WARNING": 3,
            "ERROR": 2,
            "CRITICAL": 1
        },
        "alerts_by_source": {
            "cpu_monitor": 3,
            "memory_monitor": 2,
            "database_monitor": 2,
            "network_monitor": 1
        },
        "total_alerts": 11,
        "acknowledged_alerts": 8,
        "resolved_alerts": 6
    }
@pytest.fixture
def sample_trace_data() -> Any:
    """Фикстура с тестовыми данными трейсов."""
    return {
        "total_traces": 50,
        "total_spans": 150,
        "average_duration": 125.5,
        "duration_distribution": {
            "0-50ms": 20,
            "50-100ms": 15,
            "100-200ms": 10,
            "200-500ms": 3,
            "500ms+": 2
        },
        "traces_by_operation": {
            "api_request": 25,
            "database_query": 15,
            "file_operation": 10
        }
    }
@pytest.fixture
def sample_dashboard_data() -> Any:
    """Фикстура с тестовыми данными дашборда."""
    return {
        "metrics": {
            "cpu_usage": {"current": 25.5, "trend": "stable"},
            "memory_usage": {"current": 1024, "trend": "increasing"},
            "response_time": {"current": 150.0, "trend": "decreasing"}
        },
        "alerts": {
            "total": 5,
            "critical": 1,
            "error": 2,
            "warning": 2
        },
        "charts": {
            "cpu_chart": {"type": "line", "data_points": 100},
            "memory_chart": {"type": "line", "data_points": 100},
            "alerts_chart": {"type": "bar", "data_points": 24}
        },
        "summary": {
            "system_health": "good",
            "performance_score": 85,
            "uptime": "99.9%"
        }
    }
# Фикстуры для очистки глобального состояния
@pytest.fixture(autouse=True)
def cleanup_global_state() -> Any:
    """Автоматическая очистка глобального состояния после каждого теста."""
    yield
    # Останавливаем все мониторинги
    try:
        stop_monitoring()
    except:
        pass
    # Очищаем синглтоны
    try:
        # Очищаем кэши синглтонов (если есть)
        if hasattr(get_monitor, '_instances'):
            get_monitor._instances.clear()
        if hasattr(get_alert_manager, '_instances'):
            get_alert_manager._instances.clear()
        if hasattr(get_tracer, '_instances'):
            get_tracer._instances.clear()
        if hasattr(get_dashboard, '_instances'):
            get_dashboard._instances.clear()
    except:
        pass
# Фикстуры для настройки тестового окружения
@pytest.fixture(scope="session")
def test_environment() -> None:
    """Настройка тестового окружения."""
    # Устанавливаем переменные окружения для тестов
    os.environ["TESTING"] = "true"
    os.environ["MONITORING_LOG_LEVEL"] = "DEBUG"
    yield
    # Очистка переменных окружения
    os.environ.pop("TESTING", None)
    os.environ.pop("MONITORING_LOG_LEVEL", None)
@pytest.fixture(scope="session")
def test_data_directory() -> None:
    """Создание временной директории для тестовых данных."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir
# Фикстуры для производительности
@pytest.fixture
def performance_test_data() -> Any:
    """Фикстура с данными для тестов производительности."""
    return {
        "small_dataset": 100,
        "medium_dataset": 1000,
        "large_dataset": 10000,
        "timeout_threshold": 1.0,  # секунды
        "memory_threshold": 100 * 1024 * 1024  # 100MB
    }
@pytest.fixture
def stress_test_config() -> Any:
    """Фикстура с конфигурацией для стресс-тестов."""
    return {
        "concurrent_threads": 10,
        "iterations_per_thread": 100,
        "timeout": 30.0,
        "memory_limit": 200 * 1024 * 1024  # 200MB
    } 
