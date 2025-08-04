"""
Unit тесты для domain/protocols/monitoring.py.

Покрывает:
- MetricsCollector
- AlertManager
- HealthChecker
- PerformanceMonitor
- ProtocolMonitor
- SystemMonitor
- Декораторы мониторинга
- Обработку ошибок
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from uuid import uuid4

from domain.protocols.monitoring import (
    MetricType,
    AlertLevel,
    HealthState,
    MonitoringProtocol,
    Metric,
    Alert,
    HealthCheck,
    MetricsCollector,
    AlertManager,
    HealthChecker,
    PerformanceMonitor,
    ProtocolMonitor,
    SystemMonitor,
    monitor_protocol,
    alert_on_error,
    MetricData,
    AlertData,
    MonitoringError,
    MetricsError,
    AlertError,
    HealthCheckError,
)
from domain.types.infrastructure_types import SystemMetrics
from domain.types.common_types import HealthStatus
from shared.models.example_types import PerformanceMetrics


class TestMetricType:
    """Тесты для MetricType."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestAlertLevel:
    """Тесты для AlertLevel."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


class TestHealthState:
    """Тесты для HealthState."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert HealthState.HEALTHY.value == "healthy"
        assert HealthState.DEGRADED.value == "degraded"
        assert HealthState.UNHEALTHY.value == "unhealthy"
        assert HealthState.UNKNOWN.value == "unknown"


class TestMetric:
    """Тесты для Metric."""

    def test_creation(self):
        """Тест создания метрики."""
        timestamp = datetime.now()
        labels = {"service": "test", "version": "1.0"}
        
        metric = Metric(
            name="test_metric",
            value=42.5,
            type=MetricType.GAUGE,
            timestamp=timestamp,
            labels=labels,
            description="Test metric"
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.5
        assert metric.type == MetricType.GAUGE
        assert metric.timestamp == timestamp
        assert metric.labels == labels
        assert metric.description == "Test metric"


class TestAlert:
    """Тесты для Alert."""

    def test_creation(self):
        """Тест создания алерта."""
        alert_id = uuid4()
        timestamp = datetime.now()
        metadata = {"source": "test", "details": "test alert"}
        
        alert = Alert(
            id=alert_id,
            level=AlertLevel.WARNING,
            message="Test alert message",
            timestamp=timestamp,
            source="test_service",
            metadata=metadata,
            resolved=False
        )

        assert alert.id == alert_id
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert message"
        assert alert.timestamp == timestamp
        assert alert.source == "test_service"
        assert alert.metadata == metadata
        assert alert.resolved is False


class TestHealthCheck:
    """Тесты для HealthCheck."""

    def test_creation(self):
        """Тест создания проверки здоровья."""
        timestamp = datetime.now()
        metadata = {"component": "database", "timeout": 5.0}
        
        health_check = HealthCheck(
            name="database_check",
            status=HealthState.HEALTHY,
            timestamp=timestamp,
            response_time=0.1,
            error_message=None,
            metadata=metadata
        )

        assert health_check.name == "database_check"
        assert health_check.status == HealthState.HEALTHY
        assert health_check.timestamp == timestamp
        assert health_check.response_time == 0.1
        assert health_check.error_message is None
        assert health_check.metadata == metadata


class TestMetricsCollector:
    """Тесты для MetricsCollector."""

    @pytest.fixture
    def metrics_collector(self):
        """Фикстура сборщика метрик."""
        return MetricsCollector()

    def test_initialization(self, metrics_collector):
        """Тест инициализации."""
        assert metrics_collector.metrics == []
        assert metrics_collector.metric_history == {}

    async def test_record_counter(self, metrics_collector):
        """Тест записи счетчика."""
        await metrics_collector.record_counter("test_counter", value=5, labels={"service": "test"})
        
        assert len(metrics_collector.metrics) == 1
        metric = metrics_collector.metrics[0]
        assert metric.name == "test_counter"
        assert metric.value == 5
        assert metric.type == MetricType.COUNTER
        assert metric.labels == {"service": "test"}

    async def test_record_gauge(self, metrics_collector):
        """Тест записи датчика."""
        await metrics_collector.record_gauge("test_gauge", value=42.5, labels={"service": "test"})
        
        assert len(metrics_collector.metrics) == 1
        metric = metrics_collector.metrics[0]
        assert metric.name == "test_gauge"
        assert metric.value == 42.5
        assert metric.type == MetricType.GAUGE
        assert metric.labels == {"service": "test"}

    async def test_record_histogram(self, metrics_collector):
        """Тест записи гистограммы."""
        await metrics_collector.record_histogram("test_histogram", value=10.0, labels={"service": "test"})
        
        assert len(metrics_collector.metrics) == 1
        metric = metrics_collector.metrics[0]
        assert metric.name == "test_histogram"
        assert metric.value == 10.0
        assert metric.type == MetricType.HISTOGRAM
        assert metric.labels == {"service": "test"}

    async def test_get_metric(self, metrics_collector):
        """Тест получения метрики."""
        await metrics_collector.record_gauge("test_gauge", value=42.5)
        
        metric = await metrics_collector.get_metric("test_gauge")
        assert metric is not None
        assert metric.name == "test_gauge"
        assert metric.value == 42.5

    async def test_get_metric_history(self, metrics_collector):
        """Тест получения истории метрик."""
        await metrics_collector.record_gauge("test_gauge", value=42.5)
        await metrics_collector.record_gauge("test_gauge", value=43.0)
        
        history = await metrics_collector.get_metric_history("test_gauge", hours=24)
        assert len(history) == 2
        assert all(metric.name == "test_gauge" for metric in history)

    async def test_get_metric_summary(self, metrics_collector):
        """Тест получения сводки метрик."""
        await metrics_collector.record_gauge("test_gauge", value=10.0)
        await metrics_collector.record_gauge("test_gauge", value=20.0)
        await metrics_collector.record_gauge("test_gauge", value=30.0)
        
        summary = await metrics_collector.get_metric_summary("test_gauge", hours=24)
        assert "count" in summary
        assert "min" in summary
        assert "max" in summary
        assert "avg" in summary
        assert summary["count"] == 3

    async def test_reset_metrics(self, metrics_collector):
        """Тест сброса метрик."""
        await metrics_collector.record_gauge("test_gauge", value=42.5)
        assert len(metrics_collector.metrics) == 1
        
        await metrics_collector.reset_metrics()
        assert len(metrics_collector.metrics) == 0

    def test_record_metric_sync(self, metrics_collector):
        """Тест синхронной записи метрики."""
        metric_data = MetricData(
            name="test_metric",
            value=42.5,
            timestamp=datetime.now(),
            tags={"service": "test"}
        )
        
        metrics_collector.record_metric(metric_data)
        assert len(metrics_collector.metrics) == 1

    def test_get_metrics(self, metrics_collector):
        """Тест получения метрик."""
        metric_data = MetricData(
            name="test_metric",
            value=42.5,
            timestamp=datetime.now(),
            tags={"service": "test"}
        )
        metrics_collector.record_metric(metric_data)
        
        metrics = metrics_collector.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "test_metric"


class TestAlertManager:
    """Тесты для AlertManager."""

    @pytest.fixture
    def alert_manager(self):
        """Фикстура менеджера алертов."""
        return AlertManager()

    def test_initialization(self, alert_manager):
        """Тест инициализации."""
        assert alert_manager.alerts == []
        assert alert_manager.handlers == []

    async def test_create_alert(self, alert_manager):
        """Тест создания алерта."""
        metadata = {"source": "test", "details": "test alert"}
        
        alert = await alert_manager.create_alert(
            AlertLevel.WARNING,
            "Test alert message",
            "test_service",
            metadata
        )
        
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert message"
        assert alert.source == "test_service"
        assert alert.metadata == metadata
        assert alert.resolved is False
        assert len(alert_manager.alerts) == 1

    async def test_resolve_alert(self, alert_manager):
        """Тест разрешения алерта."""
        alert = await alert_manager.create_alert(
            AlertLevel.WARNING,
            "Test alert message",
            "test_service"
        )
        
        result = await alert_manager.resolve_alert(alert.id)
        assert result is True
        assert alert.resolved is True
        assert alert.resolved_at is not None

    async def test_get_active_alerts(self, alert_manager):
        """Тест получения активных алертов."""
        alert1 = await alert_manager.create_alert(
            AlertLevel.WARNING,
            "Active alert",
            "test_service"
        )
        alert2 = await alert_manager.create_alert(
            AlertLevel.ERROR,
            "Another active alert",
            "test_service"
        )
        
        await alert_manager.resolve_alert(alert1.id)
        
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].id == alert2.id

    async def test_get_alerts_by_level(self, alert_manager):
        """Тест получения алертов по уровню."""
        await alert_manager.create_alert(
            AlertLevel.WARNING,
            "Warning alert",
            "test_service"
        )
        await alert_manager.create_alert(
            AlertLevel.ERROR,
            "Error alert",
            "test_service"
        )
        
        warning_alerts = await alert_manager.get_alerts_by_level(AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].level == AlertLevel.WARNING

    async def test_add_handler(self, alert_manager):
        """Тест добавления обработчика."""
        handler_called = False
        
        async def test_handler(alert: Alert):
            nonlocal handler_called
            handler_called = True
        
        await alert_manager.add_handler(test_handler)
        assert len(alert_manager.handlers) == 1

    def test_send_alert_sync(self, alert_manager):
        """Тест синхронной отправки алерта."""
        alert_data = AlertData(
            level=AlertLevel.WARNING,
            message="Test alert",
            timestamp=datetime.now(),
            source="test_service",
            details={"key": "value"}
        )
        
        alert_manager.send_alert(alert_data)
        assert len(alert_manager.alerts) == 1

    def test_get_alerts(self, alert_manager):
        """Тест получения алертов."""
        alert_data = AlertData(
            level=AlertLevel.WARNING,
            message="Test alert",
            timestamp=datetime.now(),
            source="test_service"
        )
        alert_manager.send_alert(alert_data)
        
        alerts = alert_manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING


class TestHealthChecker:
    """Тесты для HealthChecker."""

    @pytest.fixture
    def health_checker(self):
        """Фикстура проверки здоровья."""
        return HealthChecker()

    def test_initialization(self, health_checker):
        """Тест инициализации."""
        assert health_checker.checks == {}
        assert health_checker.check_results == {}

    async def test_add_check(self, health_checker):
        """Тест добавления проверки."""
        async def test_check():
            return HealthCheck(
                name="test_check",
                status=HealthState.HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1
            )
        
        await health_checker.add_check("test_check", test_check)
        assert "test_check" in health_checker.checks

    async def test_run_check(self, health_checker):
        """Тест запуска проверки."""
        async def test_check():
            return HealthCheck(
                name="test_check",
                status=HealthState.HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1
            )
        
        await health_checker.add_check("test_check", test_check)
        result = await health_checker.run_check("test_check")
        
        assert result is not None
        assert result.name == "test_check"
        assert result.status == HealthState.HEALTHY

    async def test_run_all_checks(self, health_checker):
        """Тест запуска всех проверок."""
        async def check1():
            return HealthCheck(
                name="check1",
                status=HealthState.HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1
            )
        
        async def check2():
            return HealthCheck(
                name="check2",
                status=HealthState.DEGRADED,
                timestamp=datetime.now(),
                response_time=0.5
            )
        
        await health_checker.add_check("check1", check1)
        await health_checker.add_check("check2", check2)
        
        results = await health_checker.run_all_checks()
        assert len(results) == 2
        assert "check1" in results
        assert "check2" in results

    async def test_get_overall_health(self, health_checker):
        """Тест получения общего состояния здоровья."""
        async def healthy_check():
            return HealthCheck(
                name="healthy_check",
                status=HealthState.HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1
            )
        
        await health_checker.add_check("healthy_check", healthy_check)
        await health_checker.run_check("healthy_check")
        
        overall_health = await health_checker.get_overall_health()
        assert overall_health == HealthState.HEALTHY

    async def test_get_check_result(self, health_checker):
        """Тест получения результата проверки."""
        async def test_check():
            return HealthCheck(
                name="test_check",
                status=HealthState.HEALTHY,
                timestamp=datetime.now(),
                response_time=0.1
            )
        
        await health_checker.add_check("test_check", test_check)
        await health_checker.run_check("test_check")
        
        result = await health_checker.get_check_result("test_check")
        assert result is not None
        assert result.name == "test_check"

    def test_perform_health_check(self, health_checker):
        """Тест выполнения проверки здоровья."""
        health_status = health_checker.perform_health_check()
        assert isinstance(health_status, HealthStatus)


class TestPerformanceMonitor:
    """Тесты для PerformanceMonitor."""

    @pytest.fixture
    def performance_monitor(self):
        """Фикстура монитора производительности."""
        return PerformanceMonitor()

    def test_initialization(self, performance_monitor):
        """Тест инициализации."""
        assert performance_monitor.timers == {}
        assert performance_monitor.metrics == []

    async def test_start_stop_timer(self, performance_monitor):
        """Тест запуска и остановки таймера."""
        await performance_monitor.start_timer("test_timer")
        assert "test_timer" in performance_monitor.timers
        
        await asyncio.sleep(0.01)  # Небольшая задержка
        duration = await performance_monitor.stop_timer("test_timer", {"service": "test"})
        assert duration > 0

    async def test_record_operation(self, performance_monitor):
        """Тест записи операции."""
        await performance_monitor.record_operation(
            "test_operation",
            success=True,
            labels={"service": "test"}
        )
        
        assert len(performance_monitor.metrics) == 1
        metric = performance_monitor.metrics[0]
        assert metric.name == "test_operation"
        assert metric.labels == {"service": "test"}

    async def test_record_memory_usage(self, performance_monitor):
        """Тест записи использования памяти."""
        await performance_monitor.record_memory_usage("test_service", 1024 * 1024)
        
        assert len(performance_monitor.metrics) == 1
        metric = performance_monitor.metrics[0]
        assert metric.name == "test_service_memory"
        assert metric.value == 1024 * 1024

    async def test_record_error(self, performance_monitor):
        """Тест записи ошибки."""
        error = ValueError("Test error")
        await performance_monitor.record_error("test_service", error)
        
        assert len(performance_monitor.metrics) == 1
        metric = performance_monitor.metrics[0]
        assert metric.name == "test_service_error"
        assert "Test error" in str(metric.value)

    def test_record_metric(self, performance_monitor):
        """Тест записи метрики производительности."""
        perf_metrics = PerformanceMetrics(
            operation_name="test_op",
            duration=0.1,
            memory_usage=1024,
            cpu_usage=25.0,
            timestamp=datetime.now()
        )
        
        performance_monitor.record_metric(perf_metrics)
        assert len(performance_monitor.metrics) == 1

    def test_get_metrics(self, performance_monitor):
        """Тест получения метрик."""
        perf_metrics = PerformanceMetrics(
            operation_name="test_op",
            duration=0.1,
            memory_usage=1024,
            cpu_usage=25.0,
            timestamp=datetime.now()
        )
        performance_monitor.record_metric(perf_metrics)
        
        metrics = performance_monitor.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation_name == "test_op"

    def test_check_thresholds(self, performance_monitor):
        """Тест проверки порогов."""
        perf_metrics = PerformanceMetrics(
            operation_name="test_op",
            duration=0.1,
            memory_usage=1024,
            cpu_usage=25.0,
            timestamp=datetime.now()
        )
        performance_monitor.record_metric(perf_metrics)
        
        thresholds = {"duration": 0.05, "memory_usage": 2048}
        violations = performance_monitor.check_thresholds(thresholds)
        assert len(violations) > 0


class TestProtocolMonitor:
    """Тесты для ProtocolMonitor."""

    @pytest.fixture
    def protocol_monitor(self):
        """Фикстура монитора протоколов."""
        return ProtocolMonitor()

    def test_initialization(self, protocol_monitor):
        """Тест инициализации."""
        assert protocol_monitor.metrics_collector is not None
        assert protocol_monitor.alert_manager is not None
        assert protocol_monitor.health_checker is not None
        assert protocol_monitor.performance_monitor is not None
        assert protocol_monitor.initialized is False

    async def test_ensure_initialized(self, protocol_monitor):
        """Тест обеспечения инициализации."""
        await protocol_monitor._ensure_initialized()
        assert protocol_monitor.initialized is True

    async def test_monitor_protocol_health(self, protocol_monitor):
        """Тест мониторинга здоровья протокола."""
        health_check = await protocol_monitor.monitor_protocol_health("test_protocol")
        
        assert health_check is not None
        assert health_check.name == "test_protocol"
        assert health_check.status in [HealthState.HEALTHY, HealthState.DEGRADED, HealthState.UNHEALTHY]

    async def test_record_protocol_metric(self, protocol_monitor):
        """Тест записи метрики протокола."""
        await protocol_monitor.record_protocol_metric(
            "test_protocol",
            "test_metric",
            42.5,
            MetricType.GAUGE,
            {"service": "test"}
        )
        
        metrics = protocol_monitor.metrics_collector.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].name == "test_protocol_test_metric"

    async def test_create_protocol_alert(self, protocol_monitor):
        """Тест создания алерта протокола."""
        alert = await protocol_monitor.create_protocol_alert(
            "test_protocol",
            AlertLevel.WARNING,
            "Test protocol alert",
            {"details": "test"}
        )
        
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test protocol alert"
        assert alert.source == "test_protocol"

    async def test_get_protocol_metrics(self, protocol_monitor):
        """Тест получения метрик протокола."""
        await protocol_monitor.record_protocol_metric(
            "test_protocol",
            "test_metric",
            42.5
        )
        
        metrics = await protocol_monitor.get_protocol_metrics("test_protocol", hours=24)
        assert isinstance(metrics, dict)
        assert "metrics" in metrics

    async def test_get_protocol_health(self, protocol_monitor):
        """Тест получения здоровья протокола."""
        health_check = await protocol_monitor.get_protocol_health("test_protocol")
        assert health_check is not None

    async def test_get_overall_health(self, protocol_monitor):
        """Тест получения общего здоровья."""
        health_state = await protocol_monitor.get_overall_health()
        assert health_state in [HealthState.HEALTHY, HealthState.DEGRADED, HealthState.UNHEALTHY, HealthState.UNKNOWN]

    async def test_get_active_alerts(self, protocol_monitor):
        """Тест получения активных алертов."""
        await protocol_monitor.create_protocol_alert(
            "test_protocol",
            AlertLevel.WARNING,
            "Test alert"
        )
        
        alerts = await protocol_monitor.get_active_alerts()
        assert len(alerts) == 1

    async def test_get_health_report(self, protocol_monitor):
        """Тест получения отчета о здоровье."""
        report = await protocol_monitor.get_health_report()
        assert isinstance(report, dict)
        assert "overall_health" in report
        assert "active_alerts" in report
        assert "protocol_health" in report


class TestSystemMonitor:
    """Тесты для SystemMonitor."""

    @pytest.fixture
    def system_monitor(self):
        """Фикстура системного монитора."""
        return SystemMonitor()

    def test_initialization(self, system_monitor):
        """Тест инициализации."""
        assert system_monitor.metrics == []

    def test_record_metric(self, system_monitor):
        """Тест записи метрики."""
        system_metrics = SystemMetrics(
            cpu_usage=25.0,
            memory_usage=1024 * 1024,
            disk_usage=50.0,
            network_io=1000,
            timestamp=datetime.now()
        )
        
        system_monitor.record_metric(system_metrics)
        assert len(system_monitor.metrics) == 1

    def test_get_metrics(self, system_monitor):
        """Тест получения метрик."""
        system_metrics = SystemMetrics(
            cpu_usage=25.0,
            memory_usage=1024 * 1024,
            disk_usage=50.0,
            network_io=1000,
            timestamp=datetime.now()
        )
        system_monitor.record_metric(system_metrics)
        
        metrics = system_monitor.get_metrics()
        assert len(metrics) == 1
        assert metrics[0].cpu_usage == 25.0

    async def test_perform_health_check(self, system_monitor):
        """Тест выполнения проверки здоровья."""
        health_status = await system_monitor.perform_health_check()
        assert isinstance(health_status, HealthStatus)


class TestMonitoringDecorators:
    """Тесты для декораторов мониторинга."""

    async def test_monitor_protocol_decorator(self):
        """Тест декоратора мониторинга протокола."""
        @monitor_protocol("test_protocol")
        async def test_function():
            return "success"
        
        result = await test_function()
        assert result == "success"

    async def test_alert_on_error_decorator(self):
        """Тест декоратора алерта при ошибке."""
        @alert_on_error("test_protocol", AlertLevel.ERROR)
        async def test_function():
            return "success"
        
        result = await test_function()
        assert result == "success"

    async def test_alert_on_error_decorator_with_exception(self):
        """Тест декоратора алерта при ошибке с исключением."""
        @alert_on_error("test_protocol", AlertLevel.ERROR)
        async def test_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            await test_function()


class TestMonitoringIntegration:
    """Интеграционные тесты мониторинга."""

    async def test_full_monitoring_workflow(self):
        """Тест полного рабочего процесса мониторинга."""
        protocol_monitor = ProtocolMonitor()
        
        # 1. Запись метрик
        await protocol_monitor.record_protocol_metric(
            "test_protocol",
            "response_time",
            0.1,
            MetricType.HISTOGRAM
        )
        
        # 2. Создание алерта
        alert = await protocol_monitor.create_protocol_alert(
            "test_protocol",
            AlertLevel.WARNING,
            "High response time detected"
        )
        
        assert alert.level == AlertLevel.WARNING
        
        # 3. Проверка здоровья
        health_check = await protocol_monitor.monitor_protocol_health("test_protocol")
        assert health_check is not None
        
        # 4. Получение отчета
        report = await protocol_monitor.get_health_report()
        assert isinstance(report, dict)
        assert "overall_health" in report
        assert "active_alerts" in report

    async def test_monitoring_error_handling(self):
        """Тест обработки ошибок мониторинга."""
        protocol_monitor = ProtocolMonitor()
        
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await protocol_monitor.record_protocol_metric(
                "test_protocol",
                "test_metric",
                "invalid_value"  # Неправильный тип
            )


class TestMonitoringProtocol:
    """Тесты для MonitoringProtocol."""

    def test_protocol_definition(self):
        """Тест определения протокола."""
        # Проверяем, что протокол определен корректно
        assert hasattr(MonitoringProtocol, '__call__')
        
        # Проверяем методы протокола
        methods = MonitoringProtocol.__dict__.get('__annotations__', {})
        assert 'start_monitoring' in methods
        assert 'stop_monitoring' in methods
        assert 'get_health_status' in methods
        assert 'record_metric' in methods
        assert 'create_alert' in methods 