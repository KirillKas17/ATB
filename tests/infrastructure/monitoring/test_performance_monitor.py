"""
Unit тесты для модуля performance_monitor.
Тестирует:
- PerformanceMonitor
- Сбор метрик производительности
- Систему алертов
- Трейсинг запросов
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from infrastructure.monitoring.performance_monitor import (
    PerformanceMonitor,
    get_monitor,
    start_monitoring,
    stop_monitoring,
    record_metric,
    record_counter,
    record_gauge,
    record_histogram,
    record_timer,
)
from domain.type_definitions.monitoring_types import Metric, MetricType, MetricCollectorProtocol


class TestPerformanceMonitor:
    """Тесты для PerformanceMonitor."""

    def test_init_default(self: "TestPerformanceMonitor") -> None:
        """Тест инициализации с параметрами по умолчанию."""
        monitor = PerformanceMonitor()
        assert monitor.name == "performance"
        assert monitor.metrics == {}
        assert monitor.system_metrics == {}
        assert monitor.app_metrics == {}
        assert monitor.is_running is False
        assert monitor.collection_interval == 5.0

    def test_init_custom(self: "TestPerformanceMonitor") -> None:
        """Тест инициализации с пользовательскими параметрами."""
        monitor = PerformanceMonitor(name="custom_monitor", collection_interval=10.0)
        assert monitor.name == "custom_monitor"
        assert monitor.collection_interval == 10.0

    def test_register_metric(self: "TestPerformanceMonitor") -> None:
        """Тест регистрации метрики."""
        monitor = PerformanceMonitor()
        monitor.register_metric(name="test_metric", metric_type=MetricType.GAUGE, description="Test metric description")
        assert "test_metric" in monitor.metrics
        assert monitor.metrics["test_metric"]["type"] == MetricType.GAUGE
        assert monitor.metrics["test_metric"]["description"] == "Test metric description"

    def test_record_metric(self: "TestPerformanceMonitor") -> None:
        """Тест записи метрики."""
        monitor = PerformanceMonitor()
        # Регистрируем метрику
        monitor.register_metric("test_metric", MetricType.GAUGE)
        # Записываем значение
        monitor.record_metric("test_metric", 25.5, {"host": "server1"})
        # Проверяем, что метрика записана
        assert "test_metric" in monitor.metrics
        assert len(monitor.metrics["test_metric"]["values"]) == 1
        assert monitor.metrics["test_metric"]["values"][0]["value"] == 25.5

    def test_record_counter(self: "TestPerformanceMonitor") -> None:
        """Тест записи счетчика."""
        monitor = PerformanceMonitor()
        # Регистрируем счетчик
        monitor.register_metric("request_count", MetricType.COUNTER)
        # Записываем значения счетчика
        monitor.record_counter("request_count", 1, {"endpoint": "/api/users"})
        monitor.record_counter("request_count", 1, {"endpoint": "/api/users"})
        monitor.record_counter("request_count", 1, {"endpoint": "/api/posts"})
        # Проверяем, что счетчик увеличился
        assert "request_count" in monitor.metrics
        values = monitor.metrics["request_count"]["values"]
        assert len(values) == 3
        # Проверяем, что значения увеличиваются
        assert values[0]["value"] == 1
        assert values[1]["value"] == 2
        assert values[2]["value"] == 1  # Новый endpoint

    def test_record_gauge(self: "TestPerformanceMonitor") -> None:
        """Тест записи датчика."""
        monitor = PerformanceMonitor()
        # Регистрируем датчик
        monitor.register_metric("cpu_usage", MetricType.GAUGE)
        # Записываем значения датчика
        monitor.record_gauge("cpu_usage", 25.5, {"core": "0"})
        monitor.record_gauge("cpu_usage", 30.2, {"core": "0"})
        monitor.record_gauge("cpu_usage", 18.7, {"core": "1"})
        # Проверяем, что датчик записан
        assert "cpu_usage" in monitor.metrics
        values = monitor.metrics["cpu_usage"]["values"]
        assert len(values) == 3

    def test_record_histogram(self: "TestPerformanceMonitor") -> None:
        """Тест записи гистограммы."""
        monitor = PerformanceMonitor()
        # Регистрируем гистограмму
        monitor.register_metric("response_time", MetricType.HISTOGRAM)
        # Записываем значения гистограммы
        monitor.record_histogram("response_time", 100.0, {"endpoint": "/api/users"})
        monitor.record_histogram("response_time", 150.0, {"endpoint": "/api/users"})
        monitor.record_histogram("response_time", 200.0, {"endpoint": "/api/users"})
        # Проверяем, что гистограмма записана
        assert "response_time" in monitor.metrics
        values = monitor.metrics["response_time"]["values"]
        assert len(values) == 3

    def test_record_timer(self: "TestPerformanceMonitor") -> None:
        """Тест записи таймера."""
        monitor = PerformanceMonitor()
        # Регистрируем таймер
        monitor.register_metric("api_call_duration", MetricType.HISTOGRAM)
        # Используем контекстный менеджер для таймера
        with monitor.record_timer("api_call_duration", {"endpoint": "/api/users"}):
            time.sleep(0.01)  # Имитируем работу
        # Проверяем, что таймер записан
        assert "api_call_duration" in monitor.metrics
        values = monitor.metrics["api_call_duration"]["values"]
        assert len(values) == 1
        assert values[0]["value"] > 0

    def test_get_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест получения метрик."""
        monitor = PerformanceMonitor()
        # Добавляем тестовые метрики
        monitor.register_metric("cpu_usage", MetricType.GAUGE)
        monitor.record_gauge("cpu_usage", 25.5)
        monitor.record_gauge("cpu_usage", 30.2)
        monitor.register_metric("request_count", MetricType.COUNTER)
        monitor.record_counter("request_count", 1)
        monitor.record_counter("request_count", 1)
        # Получаем все метрики
        all_metrics = monitor.get_metrics()
        assert len(all_metrics) == 2
        # Получаем метрики с лимитом
        limited_metrics = monitor.get_metrics(limit=1)
        assert len(limited_metrics) == 1
        # Получаем метрики по имени
        cpu_metrics = monitor.get_metrics(name="cpu_usage")
        assert len(cpu_metrics) == 1
        assert cpu_metrics[0]["name"] == "cpu_usage"

    def test_get_system_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест получения системных метрик."""
        monitor = PerformanceMonitor()
        # Запускаем мониторинг для сбора системных метрик
        monitor.start_monitoring()
        time.sleep(0.1)  # Ждем немного для сбора метрик
        monitor.stop_monitoring()
        system_metrics = monitor.get_system_metrics()
        # Проверяем наличие основных системных метрик
        assert "cpu_percent" in system_metrics
        assert "memory_percent" in system_metrics
        assert "disk_usage" in system_metrics

    def test_get_app_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест получения метрик приложения."""
        monitor = PerformanceMonitor()
        # Добавляем метрики приложения
        monitor.record_gauge("app_metric", 42.0)
        app_metrics = monitor.get_app_metrics()
        assert "app_metric" in app_metrics
        assert app_metrics["app_metric"]["current"] == 42.0

    def test_start_monitoring(self: "TestPerformanceMonitor") -> None:
        """Тест запуска мониторинга."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        assert monitor.is_running is True

    def test_stop_monitoring(self: "TestPerformanceMonitor") -> None:
        """Тест остановки мониторинга."""
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        monitor.stop_monitoring()
        assert monitor.is_running is False

    def test_monitoring_loop(self: "TestPerformanceMonitor") -> None:
        """Тест цикла мониторинга."""
        monitor = PerformanceMonitor(collection_interval=0.1)
        # Запускаем мониторинг
        monitor.start_monitoring()
        # Ждем немного для выполнения цикла
        time.sleep(0.2)
        # Останавливаем мониторинг
        monitor.stop_monitoring()
        # Проверяем, что системные метрики собраны
        system_metrics = monitor.get_system_metrics()
        assert len(system_metrics) > 0

    def test_collect_system_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест сбора системных метрик."""
        monitor = PerformanceMonitor()
        monitor._collect_system_metrics()
        # Проверяем, что системные метрики собраны
        assert len(monitor.system_metrics) > 0
        assert "cpu_percent" in monitor.system_metrics
        assert "memory_percent" in monitor.system_metrics

    def test_collect_app_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест сбора метрик приложения."""
        monitor = PerformanceMonitor()
        # Добавляем метрики приложения
        monitor.record_gauge("app_metric", 42.0)
        monitor._collect_app_metrics()
        # Проверяем, что метрики приложения собраны
        app_metrics = monitor.get_app_metrics()
        assert "app_metric" in app_metrics

    def test_check_system_alerts(self: "TestPerformanceMonitor") -> None:
        """Тест проверки системных алертов."""
        monitor = PerformanceMonitor()
        # Mock alert manager
        mock_alert_manager = Mock()
        monitor.alert_manager = mock_alert_manager
        # Проверяем системные алерты
        monitor._check_system_alerts()
        # Проверяем, что alert manager был вызван
        assert mock_alert_manager.create_alert.called

    def test_cleanup_old_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест очистки старых метрик."""
        monitor = PerformanceMonitor()
        # Добавляем старые метрики
        old_metric = Metric(
            name="old_metric", value=25.5, timestamp=datetime.now() - timedelta(days=31), type=MetricType.GAUGE
        )
        monitor.metrics["old_metric"] = {"type": MetricType.GAUGE, "description": "Old metric", "values": [old_metric]}
        # Добавляем новые метрики
        new_metric = Metric(name="new_metric", value=30.0, timestamp=datetime.now(), type=MetricType.GAUGE)
        monitor.metrics["new_metric"] = {"type": MetricType.GAUGE, "description": "New metric", "values": [new_metric]}
        # Очищаем старые метрики
        monitor.cleanup_old_metrics(days=30)
        # Проверяем, что старые метрики удалены
        assert "old_metric" not in monitor.metrics
        assert "new_metric" in monitor.metrics

    def test_get_metric_statistics(self: "TestPerformanceMonitor") -> None:
        """Тест получения статистики метрик."""
        monitor = PerformanceMonitor()
        # Добавляем тестовые метрики
        monitor.register_metric("test_metric", MetricType.GAUGE)
        for i in range(10):
            monitor.record_gauge("test_metric", 20.0 + i)
        stats = monitor.get_metric_statistics("test_metric")
        assert "count" in stats
        assert "min" in stats
        assert "max" in stats
        assert "avg" in stats
        assert stats["count"] == 10
        assert stats["min"] == 20.0
        assert stats["max"] == 29.0

    def test_export_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест экспорта метрик."""
        monitor = PerformanceMonitor()
        # Добавляем тестовые метрики
        monitor.register_metric("test_metric", MetricType.GAUGE)
        monitor.record_gauge("test_metric", 25.5)
        # Экспортируем в JSON
        json_data = monitor.export_metrics(format="json")
        assert isinstance(json_data, str)
        data = json.loads(json_data)
        assert "metrics" in data
        assert len(data["metrics"]) == 1

    def test_performance_with_many_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест производительности с большим количеством метрик."""
        monitor = PerformanceMonitor()
        import time

        start_time = time.time()
        # Регистрируем много метрик
        for i in range(1000):
            monitor.register_metric(f"metric_{i}", MetricType.GAUGE)
            monitor.record_gauge(f"metric_{i}", 20.0 + i)
        end_time = time.time()
        duration = end_time - start_time
        # Создание 1000 метрик должно занимать менее 1 секунды
        assert duration < 1.0
        assert len(monitor.metrics) == 1000

    def test_memory_usage_with_metrics(self: "TestPerformanceMonitor") -> None:
        """Тест использования памяти с метриками."""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        monitor = PerformanceMonitor()
        # Добавляем много метрик с метаданными
        for i in range(10000):
            monitor.register_metric(f"metric_{i}", MetricType.GAUGE)
            monitor.record_gauge(f"metric_{i}", 20.0 + i, {"index": i, "data": "test" * 10})
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Увеличение памяти должно быть разумным (менее 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestGetMonitor:
    """Тесты для функции get_monitor."""

    def test_get_monitor_default(self: "TestGetMonitor") -> None:
        """Тест получения монитора по умолчанию."""
        monitor = get_monitor()
        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.name == "performance"

    def test_get_monitor_custom_name(self: "TestGetMonitor") -> None:
        """Тест получения монитора с пользовательским именем."""
        monitor = get_monitor("custom_performance")
        assert isinstance(monitor, PerformanceMonitor)
        assert monitor.name == "custom_performance"

    def test_get_monitor_singleton(self: "TestGetMonitor") -> None:
        """Тест, что get_monitor возвращает тот же экземпляр для одного имени."""
        monitor1 = get_monitor("singleton_test")
        monitor2 = get_monitor("singleton_test")
        assert monitor1 is monitor2

    def test_get_monitor_different_names(self: "TestGetMonitor") -> None:
        """Тест, что разные имена возвращают разные экземпляры."""
        monitor1 = get_monitor("monitor1")
        monitor2 = get_monitor("monitor2")
        assert monitor1 is not monitor2


class TestMonitoringFunctions:
    """Тесты для функций мониторинга."""

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_start_monitoring(self, mock_get_monitor) -> None:
        """Тест функции start_monitoring."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        start_monitoring()
        mock_monitor.start_monitoring.assert_called_once()

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_stop_monitoring(self, mock_get_monitor) -> None:
        """Тест функции stop_monitoring."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        stop_monitoring()
        mock_monitor.stop_monitoring.assert_called_once()

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_record_metric(self, mock_get_monitor) -> None:
        """Тест функции record_metric."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        record_metric("test_metric", 25.5, {"host": "server1"})
        mock_monitor.record_metric.assert_called_once_with("test_metric", 25.5, {"host": "server1"})

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_record_counter(self, mock_get_monitor) -> None:
        """Тест функции record_counter."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        record_counter("request_count", 1, {"endpoint": "/api/users"})
        mock_monitor.record_counter.assert_called_once_with("request_count", 1, {"endpoint": "/api/users"})

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_record_gauge(self, mock_get_monitor) -> None:
        """Тест функции record_gauge."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        record_gauge("cpu_usage", 25.5, {"core": "0"})
        mock_monitor.record_gauge.assert_called_once_with("cpu_usage", 25.5, {"core": "0"})

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_record_histogram(self, mock_get_monitor) -> None:
        """Тест функции record_histogram."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        record_histogram("response_time", 150.0, {"endpoint": "/api/users"})
        mock_monitor.record_histogram.assert_called_once_with("response_time", 150.0, {"endpoint": "/api/users"})

    @patch("infrastructure.monitoring.performance_monitor.get_monitor")
    def test_record_timer(self, mock_get_monitor) -> None:
        """Тест функции record_timer."""
        mock_monitor = Mock()
        mock_get_monitor.return_value = mock_monitor
        mock_context = Mock()
        mock_monitor.record_timer.return_value = mock_context
        timer = record_timer("api_call_duration", {"endpoint": "/api/users"})
        mock_monitor.record_timer.assert_called_once_with("api_call_duration", {"endpoint": "/api/users"})
        assert timer == mock_context


class TestMetricCollectorProtocol:
    """Тесты для протокола MetricCollectorProtocol."""

    def test_performance_monitor_implements_protocol(self: "TestMetricCollectorProtocol") -> None:
        """Тест, что PerformanceMonitor реализует MetricCollectorProtocol."""
        monitor = PerformanceMonitor()
        # Проверяем наличие всех методов протокола
        assert hasattr(monitor, "register_metric")
        assert hasattr(monitor, "record_metric")
        assert hasattr(monitor, "get_metrics")
        assert hasattr(monitor, "start_monitoring")
        assert hasattr(monitor, "stop_monitoring")


if __name__ == "__main__":
    pytest.main([__file__])
