"""
Unit тесты для модуля monitoring_dashboard.
Тестирует:
- MonitoringDashboard
- Функции дашборда
- Визуализацию метрик
- Экспорт данных
"""
import pytest
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from infrastructure.monitoring.monitoring_dashboard import (
    MonitoringDashboard,
    get_dashboard,
    get_dashboard_data,
    export_metrics,
    get_performance_summary,
    get_alert_summary
)
from domain.type_definitions.monitoring_types import Metric, Alert, AlertSeverity

# Локальные определения для тестов
@dataclass
class DashboardConfig:
    title: str = "Default Dashboard"
    refresh_interval: int = 30
    max_data_points: int = 1000

@dataclass
class ChartConfig:
    name: str
    title: str
    chart_type: str
    metrics: List[str]
    time_range: timedelta

@dataclass
class MetricData:
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str]

class TestMonitoringDashboard:
    """Тесты для MonitoringDashboard."""
    def test_init_default(self: "TestMonitoringDashboard") -> None:
        """Тест инициализации с параметрами по умолчанию."""
        dashboard = MonitoringDashboard()
        assert dashboard.name == "default"
        assert dashboard.config is not None
        assert dashboard.metrics_data == {}
        assert dashboard.alerts_data == []
        assert dashboard.charts == {}
        assert dashboard.is_running is False
    def test_init_custom(self: "TestMonitoringDashboard") -> None:
        """Тест инициализации с пользовательскими параметрами."""
        config = DashboardConfig(
            title="Custom Dashboard",
            refresh_interval=60,
            max_data_points=1000
        )
        dashboard = MonitoringDashboard(
            name="custom_dashboard",
            config=config
        )
        assert dashboard.name == "custom_dashboard"
        assert dashboard.config.title == "Custom Dashboard"
        assert dashboard.config.refresh_interval == 60
    def test_add_metric_data(self: "TestMonitoringDashboard") -> None:
        """Тест добавления данных метрик."""
        dashboard = MonitoringDashboard()
        # Создаем тестовые метрики
        metric1 = Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now(),
            type="gauge",
            labels={"host": "server1"}
        )
        metric2 = Metric(
            name="memory_usage",
            value=1024,
            timestamp=datetime.now(),
            type="gauge",
            labels={"host": "server1"}
        )
        # Добавляем метрики
        dashboard.add_metric_data(metric1)
        dashboard.add_metric_data(metric2)
        # Проверяем, что метрики добавлены
        assert "cpu_usage" in dashboard.metrics_data
        assert "memory_usage" in dashboard.metrics_data
        assert len(dashboard.metrics_data["cpu_usage"]) == 1
        assert len(dashboard.metrics_data["memory_usage"]) == 1
    def test_add_alert_data(self: "TestMonitoringDashboard") -> None:
        """Тест добавления данных алертов."""
        dashboard = MonitoringDashboard()
        # Создаем тестовый алерт
        alert = Alert(
            alert_id="test-alert-1",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        # Добавляем алерт
        dashboard.add_alert_data(alert)
        # Проверяем, что алерт добавлен
        assert len(dashboard.alerts_data) == 1
        assert dashboard.alerts_data[0].alert_id == "test-alert-1"
    def test_create_chart(self: "TestMonitoringDashboard") -> None:
        """Тест создания графика."""
        dashboard = MonitoringDashboard()
        chart_config = ChartConfig(
            name="cpu_chart",
            title="CPU Usage",
            chart_type="line",
            metrics=["cpu_usage"],
            time_range=timedelta(hours=1)
        )
        chart = dashboard.create_chart(chart_config)
        assert chart.name == "cpu_chart"
        assert chart.title == "CPU Usage"
        assert chart.chart_type == "line"
        assert "cpu_usage" in chart.metrics
    def test_get_chart_data(self: "TestMonitoringDashboard") -> None:
        """Тест получения данных графика."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые данные
        for i in range(10):
            metric = Metric(
                name="cpu_usage",
                value=20.0 + i,
                timestamp=datetime.now() - timedelta(minutes=i),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        # Создаем график
        chart_config = ChartConfig(
            name="cpu_chart",
            title="CPU Usage",
            chart_type="line",
            metrics=["cpu_usage"],
            time_range=timedelta(hours=1)
        )
        chart = dashboard.create_chart(chart_config)
        # Получаем данные графика
        chart_data = dashboard.get_chart_data(chart.name)
        assert chart_data is not None
        assert "labels" in chart_data
        assert "datasets" in chart_data
        assert len(chart_data["datasets"]) == 1
        assert len(chart_data["datasets"][0]["data"]) == 10
    def test_get_dashboard_data(self: "TestMonitoringDashboard") -> None:
        """Тест получения данных дашборда."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые данные
        metric = Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now(),
            type="gauge"
        )
        dashboard.add_metric_data(metric)
        alert = Alert(
            alert_id="test-alert-1",
            message="Test alert",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        dashboard.add_alert_data(alert)
        # Получаем данные дашборда
        dashboard_data = dashboard.get_dashboard_data()
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
        assert "charts" in dashboard_data
        assert "summary" in dashboard_data
        assert len(dashboard_data["metrics"]) == 1
        assert len(dashboard_data["alerts"]) == 1
    def test_get_performance_summary(self: "TestMonitoringDashboard") -> None:
        """Тест получения сводки производительности."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые метрики производительности
        metrics = [
            ("cpu_usage", 25.5),
            ("memory_usage", 1024),
            ("response_time", 150.0),
            ("throughput", 1000),
            ("error_rate", 0.01)
        ]
        for name, value in metrics:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        # Получаем сводку производительности
        summary = dashboard.get_performance_summary()
        assert "cpu_usage" in summary
        assert "memory_usage" in summary
        assert "response_time" in summary
        assert "throughput" in summary
        assert "error_rate" in summary
        assert summary["cpu_usage"]["current"] == 25.5
        assert summary["memory_usage"]["current"] == 1024
    def test_get_alert_summary(self: "TestMonitoringDashboard") -> None:
        """Тест получения сводки алертов."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые алерты
        alerts = [
            ("Alert 1", AlertSeverity.INFO),
            ("Alert 2", AlertSeverity.WARNING),
            ("Alert 3", AlertSeverity.ERROR),
            ("Alert 4", AlertSeverity.CRITICAL)
        ]
        for message, severity in alerts:
            alert = Alert(
                alert_id=f"alert-{message}",
                message=message,
                severity=severity,
                source="test_source",
                timestamp=datetime.now()
            )
            dashboard.add_alert_data(alert)
        # Получаем сводку алертов
        summary = dashboard.get_alert_summary()
        assert summary["total_alerts"] == 4
        assert summary["alerts_by_severity"]["INFO"] == 1
        assert summary["alerts_by_severity"]["WARNING"] == 1
        assert summary["alerts_by_severity"]["ERROR"] == 1
        assert summary["alerts_by_severity"]["CRITICAL"] == 1
    def test_export_metrics_json(self: "TestMonitoringDashboard") -> None:
        """Тест экспорта метрик в JSON."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые метрики
        metric = Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now(),
            type="gauge",
            labels={"host": "server1"}
        )
        dashboard.add_metric_data(metric)
        # Экспортируем в JSON
        json_data = dashboard.export_metrics(format="json")
        assert isinstance(json_data, str)
        data = json.loads(json_data)
        assert "metrics" in data
        assert len(data["metrics"]) == 1
        assert data["metrics"][0]["name"] == "cpu_usage"
    def test_export_metrics_csv(self: "TestMonitoringDashboard") -> None:
        """Тест экспорта метрик в CSV."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые метрики
        for i in range(5):
            metric = Metric(
                name="cpu_usage",
                value=20.0 + i,
                timestamp=datetime.now() - timedelta(minutes=i),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        # Экспортируем в CSV
        csv_data = dashboard.export_metrics(format="csv")
        assert isinstance(csv_data, str)
        lines = csv_data.strip().split('\n')
        assert len(lines) == 6  # Заголовок + 5 строк данных
        assert "timestamp,name,value,type" in lines[0]
    def test_export_metrics_excel(self: "TestMonitoringDashboard") -> None:
        """Тест экспорта метрик в Excel."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые метрики
        metric = Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now(),
            type="gauge"
        )
        dashboard.add_metric_data(metric)
        # Экспортируем в Excel
        excel_data = dashboard.export_metrics(format="excel")
        assert isinstance(excel_data, bytes)
        assert len(excel_data) > 0
    def test_create_time_series_chart(self: "TestMonitoringDashboard") -> None:
        """Тест создания временного ряда."""
        dashboard = MonitoringDashboard()
        # Добавляем временные данные
        for i in range(24):
            metric = Metric(
                name="cpu_usage",
                value=20.0 + (i % 10),
                timestamp=datetime.now() - timedelta(hours=i),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        # Создаем график временного ряда
        chart_config = ChartConfig(
            name="cpu_timeseries",
            title="CPU Usage Over Time",
            chart_type="line",
            metrics=["cpu_usage"],
            time_range=timedelta(hours=24)
        )
        chart = dashboard.create_chart(chart_config)
        # Получаем данные графика
        chart_data = dashboard.get_chart_data(chart.name)
        assert chart_data is not None
        assert len(chart_data["labels"]) == 24
        assert len(chart_data["datasets"][0]["data"]) == 24
    def test_create_bar_chart(self: "TestMonitoringDashboard") -> None:
        """Тест создания столбчатой диаграммы."""
        dashboard = MonitoringDashboard()
        # Добавляем данные для столбчатой диаграммы
        metrics = [
            ("server1", 25.5),
            ("server2", 30.2),
            ("server3", 18.7),
            ("server4", 35.1)
        ]
        for server, value in metrics:
            metric = Metric(
                name="cpu_usage",
                value=value,
                timestamp=datetime.now(),
                type="gauge",
                labels={"server": server}
            )
            dashboard.add_metric_data(metric)
        # Создаем столбчатую диаграмму
        chart_config = ChartConfig(
            name="server_cpu",
            title="CPU Usage by Server",
            chart_type="bar",
            metrics=["cpu_usage"],
            group_by="server"
        )
        chart = dashboard.create_chart(chart_config)
        # Получаем данные графика
        chart_data = dashboard.get_chart_data(chart.name)
        assert chart_data is not None
        assert len(chart_data["labels"]) == 4
        assert len(chart_data["datasets"][0]["data"]) == 4
    def test_create_pie_chart(self: "TestMonitoringDashboard") -> None:
        """Тест создания круговой диаграммы."""
        dashboard = MonitoringDashboard()
        # Добавляем данные для круговой диаграммы
        metrics = [
            ("error_rate", 0.05),
            ("warning_rate", 0.15),
            ("success_rate", 0.80)
        ]
        for name, value in metrics:
            metric = Metric(
                name=name,
                value=value,
                timestamp=datetime.now(),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        # Создаем круговую диаграмму
        chart_config = ChartConfig(
            name="status_distribution",
            title="Status Distribution",
            chart_type="pie",
            metrics=["error_rate", "warning_rate", "success_rate"]
        )
        chart = dashboard.create_chart(chart_config)
        # Получаем данные графика
        chart_data = dashboard.get_chart_data(chart.name)
        assert chart_data is not None
        assert len(chart_data["datasets"][0]["data"]) == 3
    def test_filter_data_by_time_range(self: "TestMonitoringDashboard") -> None:
        """Тест фильтрации данных по временному диапазону."""
        dashboard = MonitoringDashboard()
        # Добавляем данные за разные периоды
        now = datetime.now()
        for i in range(48):
            metric = Metric(
                name="cpu_usage",
                value=20.0 + (i % 10),
                timestamp=now - timedelta(hours=i),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        # Фильтруем данные за последние 24 часа
        filtered_data = dashboard.filter_data_by_time_range(
            "cpu_usage",
            time_range=timedelta(hours=24)
        )
        assert len(filtered_data) == 24
    def test_aggregate_metrics(self: "TestMonitoringDashboard") -> None:
        """Тест агрегации метрик."""
        dashboard = MonitoringDashboard()
        # Добавляем данные для агрегации
        for i in range(10):
            metric = Metric(
                name="response_time",
                value=100.0 + i * 10,
                timestamp=datetime.now() - timedelta(minutes=i),
                type="histogram"
            )
            dashboard.add_metric_data(metric)
        # Агрегируем метрики
        aggregated = dashboard.aggregate_metrics("response_time", "avg")
        assert aggregated > 0
        aggregated = dashboard.aggregate_metrics("response_time", "max")
        assert aggregated == 190.0
        aggregated = dashboard.aggregate_metrics("response_time", "min")
        assert aggregated == 100.0
    def test_create_dashboard_snapshot(self: "TestMonitoringDashboard") -> None:
        """Тест создания снимка дашборда."""
        dashboard = MonitoringDashboard()
        # Добавляем тестовые данные
        metric = Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now(),
            type="gauge"
        )
        dashboard.add_metric_data(metric)
        alert = Alert(
            alert_id="test-alert-1",
            message="Test alert",
            severity=AlertSeverity.WARNING,
            source="test_source",
            timestamp=datetime.now()
        )
        dashboard.add_alert_data(alert)
        # Создаем снимок
        snapshot = dashboard.create_snapshot()
        assert "timestamp" in snapshot
        assert "metrics" in snapshot
        assert "alerts" in snapshot
        assert "charts" in snapshot
        assert len(snapshot["metrics"]) == 1
        assert len(snapshot["alerts"]) == 1
    def test_load_dashboard_snapshot(self: "TestMonitoringDashboard") -> None:
        """Тест загрузки снимка дашборда."""
        dashboard = MonitoringDashboard()
        # Создаем тестовый снимок
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cpu_usage": [
                    {
                        "name": "cpu_usage",
                        "value": 25.5,
                        "timestamp": datetime.now().isoformat(),
                        "type": "gauge"
                    }
                ]
            },
            "alerts": [
                {
                    "alert_id": "test-alert-1",
                    "message": "Test alert",
                    "severity": "WARNING",
                    "source": "test_source",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "charts": {}
        }
        # Загружаем снимок
        dashboard.load_snapshot(snapshot)
        assert len(dashboard.metrics_data["cpu_usage"]) == 1
        assert len(dashboard.alerts_data) == 1
    def test_cleanup_old_data(self: "TestMonitoringDashboard") -> None:
        """Тест очистки старых данных."""
        dashboard = MonitoringDashboard()
        # Добавляем старые данные
        old_metric = Metric(
            name="cpu_usage",
            value=25.5,
            timestamp=datetime.now() - timedelta(days=31),
            type="gauge"
        )
        dashboard.add_metric_data(old_metric)
        # Добавляем новые данные
        new_metric = Metric(
            name="cpu_usage",
            value=30.0,
            timestamp=datetime.now(),
            type="gauge"
        )
        dashboard.add_metric_data(new_metric)
        # Очищаем старые данные
        dashboard.cleanup_old_data(days=30)
        # Проверяем, что старые данные удалены
        assert len(dashboard.metrics_data["cpu_usage"]) == 1
        assert dashboard.metrics_data["cpu_usage"][0].value == 30.0
    def test_performance_with_large_dataset(self: "TestMonitoringDashboard") -> None:
        """Тест производительности с большим набором данных."""
        dashboard = MonitoringDashboard()
        import time
        start_time = time.time()
        # Добавляем много данных
        for i in range(10000):
            metric = Metric(
                name=f"metric_{i % 10}",
                value=20.0 + (i % 100),
                timestamp=datetime.now() - timedelta(minutes=i),
                type="gauge"
            )
            dashboard.add_metric_data(metric)
        end_time = time.time()
        duration = end_time - start_time
        # Добавление 10000 метрик должно занимать менее 1 секунды
        assert duration < 1.0
        assert len(dashboard.metrics_data) == 10
    def test_memory_usage_with_dashboard(self: "TestMonitoringDashboard") -> None:
        """Тест использования памяти дашбордом."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        dashboard = MonitoringDashboard()
        # Добавляем много данных с метаданными
        for i in range(10000):
            metric = Metric(
                name=f"metric_{i}",
                value=20.0 + (i % 100),
                timestamp=datetime.now() - timedelta(minutes=i),
                type="gauge",
                labels={"index": i, "data": "test" * 10}
            )
            dashboard.add_metric_data(metric)
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Увеличение памяти должно быть разумным (менее 100MB)
        assert memory_increase < 100 * 1024 * 1024
class TestDashboardConfig:
    """Тесты для DashboardConfig."""
    def test_dashboard_config_init(self: "TestDashboardConfig") -> None:
        """Тест инициализации DashboardConfig."""
        config = DashboardConfig(
            title="Test Dashboard",
            refresh_interval=60,
            max_data_points=1000,
            theme="dark"
        )
        assert config.title == "Test Dashboard"
        assert config.refresh_interval == 60
        assert config.max_data_points == 1000
        assert config.theme == "dark"
    def test_dashboard_config_defaults(self: "TestDashboardConfig") -> None:
        """Тест значений по умолчанию DashboardConfig."""
        config = DashboardConfig()
        assert config.title == "Monitoring Dashboard"
        assert config.refresh_interval == 30
        assert config.max_data_points == 10000
        assert config.theme == "light"
    def test_dashboard_config_to_dict(self: "TestDashboardConfig") -> None:
        """Тест преобразования DashboardConfig в словарь."""
        config = DashboardConfig(
            title="Test Dashboard",
            refresh_interval=60,
            max_data_points=1000
        )
        config_dict = config.to_dict()
        assert config_dict["title"] == "Test Dashboard"
        assert config_dict["refresh_interval"] == 60
        assert config_dict["max_data_points"] == 1000
class TestChartConfig:
    """Тесты для ChartConfig."""
    def test_chart_config_init(self: "TestChartConfig") -> None:
        """Тест инициализации ChartConfig."""
        config = ChartConfig(
            name="test_chart",
            title="Test Chart",
            chart_type="line",
            metrics=["cpu_usage", "memory_usage"],
            time_range=timedelta(hours=1)
        )
        assert config.name == "test_chart"
        assert config.title == "Test Chart"
        assert config.chart_type == "line"
        assert "cpu_usage" in config.metrics
        assert "memory_usage" in config.metrics
        assert config.time_range == timedelta(hours=1)
    def test_chart_config_to_dict(self: "TestChartConfig") -> None:
        """Тест преобразования ChartConfig в словарь."""
        config = ChartConfig(
            name="test_chart",
            title="Test Chart",
            chart_type="line",
            metrics=["cpu_usage"],
            time_range=timedelta(hours=1)
        )
        config_dict = config.to_dict()
        assert config_dict["name"] == "test_chart"
        assert config_dict["title"] == "Test Chart"
        assert config_dict["chart_type"] == "line"
        assert "cpu_usage" in config_dict["metrics"]
class TestMetricData:
    """Тесты для MetricData."""
    def test_metric_data_init(self: "TestMetricData") -> None:
        """Тест инициализации MetricData."""
        data = MetricData(
            name="test_metric",
            value=1.0,
            timestamp=datetime.now(),
            labels={"host": "server1"}
        )
        assert data.name == "test_metric"
        assert data.value == 1.0
        assert data.timestamp == datetime.now()
        assert data.labels["host"] == "server1"
    def test_metric_data_to_dict(self: "TestMetricData") -> None:
        """Тест преобразования MetricData в словарь."""
        data = MetricData(
            name="test_metric",
            value=1.0,
            timestamp=datetime.now(),
            labels={"host": "server1"}
        )
        data_dict = data.to_dict()
        assert data_dict["name"] == "test_metric"
        assert data_dict["value"] == 1.0
        assert data_dict["timestamp"] == data.timestamp.isoformat()
        assert data_dict["labels"]["host"] == "server1"
class TestGetDashboard:
    """Тесты для функции get_dashboard."""
    def test_get_dashboard_default(self: "TestGetDashboard") -> None:
        """Тест получения дашборда по умолчанию."""
        dashboard = get_dashboard()
        assert isinstance(dashboard, MonitoringDashboard)
        assert dashboard.name == "default"
    def test_get_dashboard_custom_name(self: "TestGetDashboard") -> None:
        """Тест получения дашборда с пользовательским именем."""
        dashboard = get_dashboard("custom_dashboard")
        assert isinstance(dashboard, MonitoringDashboard)
        assert dashboard.name == "custom_dashboard"
    def test_get_dashboard_singleton(self: "TestGetDashboard") -> None:
        """Тест, что get_dashboard возвращает тот же экземпляр для одного имени."""
        dashboard1 = get_dashboard("singleton_test")
        dashboard2 = get_dashboard("singleton_test")
        assert dashboard1 is dashboard2
    def test_get_dashboard_different_names(self: "TestGetDashboard") -> None:
        """Тест, что разные имена возвращают разные экземпляры."""
        dashboard1 = get_dashboard("dashboard1")
        dashboard2 = get_dashboard("dashboard2")
        assert dashboard1 is not dashboard2
class TestDashboardFunctions:
    """Тесты для функций дашборда."""
    @patch('infrastructure.monitoring.monitoring_dashboard.get_dashboard')
    def test_get_dashboard_data(self, mock_get_dashboard) -> None:
        """Тест функции get_dashboard_data."""
        mock_dashboard = Mock()
        mock_get_dashboard.return_value = mock_dashboard
        mock_data = {"metrics": [], "alerts": []}
        mock_dashboard.get_dashboard_data.return_value = mock_data
        data = get_dashboard_data()
        mock_dashboard.get_dashboard_data.assert_called_once()
        assert data == mock_data
    @patch('infrastructure.monitoring.monitoring_dashboard.get_dashboard')
    def test_export_metrics(self, mock_get_dashboard) -> None:
        """Тест функции export_metrics."""
        mock_dashboard = Mock()
        mock_get_dashboard.return_value = mock_dashboard
        mock_export = "test_export_data"
        mock_dashboard.export_metrics.return_value = mock_export
        export_data = export_metrics(format="json")
        mock_dashboard.export_metrics.assert_called_once_with(format="json")
        assert export_data == mock_export
    @patch('infrastructure.monitoring.monitoring_dashboard.get_dashboard')
    def test_get_performance_summary(self, mock_get_dashboard) -> None:
        """Тест функции get_performance_summary."""
        mock_dashboard = Mock()
        mock_get_dashboard.return_value = mock_dashboard
        mock_summary = {"cpu_usage": 25.5}
        mock_dashboard.get_performance_summary.return_value = mock_summary
        summary = get_performance_summary()
        mock_dashboard.get_performance_summary.assert_called_once()
        assert summary == mock_summary
    @patch('infrastructure.monitoring.monitoring_dashboard.get_dashboard')
    def test_get_alert_summary(self, mock_get_dashboard) -> None:
        """Тест функции get_alert_summary."""
        mock_dashboard = Mock()
        mock_get_dashboard.return_value = mock_dashboard
        mock_summary = {"total_alerts": 5}
        mock_dashboard.get_alert_summary.return_value = mock_summary
        summary = get_alert_summary()
        mock_dashboard.get_alert_summary.assert_called_once()
        assert summary == mock_summary
if __name__ == "__main__":
    pytest.main([__file__]) 
