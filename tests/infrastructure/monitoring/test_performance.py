"""
Тесты производительности для модулей мониторинга.
Тестирует:
- Производительность сбора метрик
- Производительность обработки алертов
- Производительность трейсинга
- Производительность дашборда
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
import asyncio
import threading
from datetime import datetime, timedelta
from infrastructure.monitoring.monitoring_performance import (
    get_monitor,
    get_alert_manager,
    get_tracer,
    get_dashboard,
    record_metric,
    create_alert,
    start_monitoring,
    stop_monitoring,
)
from domain.type_definitions.monitoring_types import MetricType, AlertSeverity
class TestMonitoringPerformance:
    """Тесты производительности системы мониторинга."""
    def test_metric_recording_performance(self, performance_test_data) -> None:
        """Тест производительности записи метрик."""
        monitor = get_monitor("perf_test")
        # Тест записи метрик
        start_time = time.time()
        for i in range(performance_test_data["large_dataset"]):
            record_metric(f"perf_metric_{i % 10}", i, MetricType.COUNTER)
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что все метрики записаны
        metrics = monitor.get_metrics()
        assert len(metrics) >= 10  # 10 уникальных метрик
    def test_alert_creation_performance(self, performance_test_data) -> None:
        """Тест производительности создания алертов."""
        alert_manager = get_alert_manager("perf_test")
        # Тест создания алертов
        start_time = time.time()
        for i in range(performance_test_data["medium_dataset"]):
            create_alert(
                f"Performance alert {i}",
                AlertSeverity.WARNING,
                "perf_test"
            )
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что все алерты созданы
        alerts = alert_manager.get_alerts()
        assert len(alerts) >= performance_test_data["medium_dataset"]
    def test_trace_creation_performance(self, performance_test_data) -> None:
        """Тест производительности создания трейсов."""
        tracer = get_tracer("perf_test")
        # Тест создания трейсов
        start_time = time.time()
        for i in range(performance_test_data["medium_dataset"]):
            trace_id = f"perf_trace_{i}"
            span = tracer.start_trace(trace_id, f"operation_{i}")
            span.add_performance_metric("duration", 0.1)
            tracer.end_trace(trace_id, "success")
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что все трейсы созданы
        trace_stats = tracer.get_trace_statistics()
        assert trace_stats["total_traces"] >= performance_test_data["medium_dataset"]
    def test_dashboard_data_processing_performance(self, performance_test_data) -> None:
        """Тест производительности обработки данных дашборда."""
        dashboard = get_dashboard("perf_test")
        # Добавляем много данных
        for i in range(performance_test_data["medium_dataset"]):
            from infrastructure.monitoring.monitoring_dashboard import MetricData
            metric_data = MetricData(
                name=f"dashboard_metric_{i % 10}",
                values=[i],
                timestamps=[datetime.now()],
                labels={"iteration": i}
            )
            dashboard.add_metric_data(metric_data)
        # Тест получения данных дашборда
        start_time = time.time()
        dashboard_data = dashboard.get_dashboard_data()
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что данные получены
        assert "metrics" in dashboard_data
    def test_concurrent_metric_recording(self, stress_test_config) -> None:
        """Тест конкурентной записи метрик."""
        monitor = get_monitor("concurrent_test")
        def worker(worker_id: int) -> Any:
            """Рабочая функция для конкурентной записи метрик."""
            for i in range(stress_test_config["iterations_per_thread"]):
                record_metric(
                    f"concurrent_metric_{worker_id}",
                    i,
                    {"worker": worker_id, "iteration": i}
                )
        # Запускаем конкурентные потоки
        start_time = time.time()
        threads = []
        for i in range(stress_test_config["concurrent_threads"]):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < stress_test_config["timeout"]
        # Проверяем, что все метрики записаны
        metrics = monitor.get_metrics()
        expected_metrics = stress_test_config["concurrent_threads"]
        assert len(metrics) >= expected_metrics
    def test_concurrent_alert_creation(self, stress_test_config) -> None:
        """Тест конкурентного создания алертов."""
        alert_manager = get_alert_manager("concurrent_test")
        def worker(worker_id: int) -> Any:
            """Рабочая функция для конкурентного создания алертов."""
            for i in range(stress_test_config["iterations_per_thread"]):
                create_alert(
                    f"Concurrent alert {worker_id}_{i}",
                    AlertSeverity.WARNING,
                    f"worker_{worker_id}"
                )
        # Запускаем конкурентные потоки
        start_time = time.time()
        threads = []
        for i in range(stress_test_config["concurrent_threads"]):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < stress_test_config["timeout"]
        # Проверяем, что все алерты созданы
        alerts = alert_manager.get_alerts()
        expected_alerts = (
            stress_test_config["concurrent_threads"] * 
            stress_test_config["iterations_per_thread"]
        )
        assert len(alerts) >= expected_alerts
    def test_memory_usage_under_load(self, performance_test_data) -> None:
        """Тест использования памяти под нагрузкой."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        # Создаем нагрузку
        monitor = get_monitor("memory_test")
        alert_manager = get_alert_manager("memory_test")
        tracer = get_tracer("memory_test")
        # Добавляем много данных
        for i in range(performance_test_data["large_dataset"]):
            # Метрики
            record_metric(f"memory_metric_{i}", i, {"index": i, "data": "test" * 10})
            # Алерты
            if i % 100 == 0:
                create_alert(f"Memory test alert {i}", AlertSeverity.INFO, "memory_test")
            # Трейсы
            if i % 50 == 0:
                trace_id = f"memory_trace_{i}"
                span = tracer.start_trace(trace_id, f"operation_{i}")
                span.add_performance_metric("memory_usage", i)
                tracer.end_trace(trace_id, "success")
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Проверяем, что использование памяти разумное
        assert memory_increase < performance_test_data["memory_threshold"]
    def test_system_metrics_collection_performance(self: "TestMonitoringPerformance") -> None:
        """Тест производительности сбора системных метрик."""
        monitor = get_monitor("system_test")
        # Запускаем мониторинг
        start_monitoring()
        # Тестируем сбор системных метрик
        start_time = time.time()
        for _ in range(100):  # 100 циклов сбора
            monitor._collect_system_metrics()
            time.sleep(0.001)  # Небольшая пауза
        end_time = time.time()
        duration = end_time - start_time
        # Останавливаем мониторинг
        stop_monitoring()
        # Проверяем производительность
        assert duration < 5.0  # Не более 5 секунд на 100 циклов
        # Проверяем, что системные метрики собраны
        system_metrics = monitor.get_system_metrics()
        assert len(system_metrics) > 0
    def test_alert_evaluation_performance(self: "TestMonitoringPerformance") -> None:
        """Тест производительности оценки алертов."""
        alert_manager = get_alert_manager("eval_test")
        # Добавляем много правил алертов
        from infrastructure.monitoring.monitoring_alerts import AlertRule
        for i in range(100):
            def condition() -> Any:
                return i % 10 == 0  # Каждое 10-е правило срабатывает
            rule = AlertRule(
                name=f"test_rule_{i}",
                condition=condition,
                severity=AlertSeverity.WARNING,
                message=f"Test rule {i} triggered",
                source="perf_test"
            )
            alert_manager.add_alert_rule(rule)
        # Тестируем оценку правил
        start_time = time.time()
        # Запускаем оценку
        asyncio.run(alert_manager.start_evaluation())
        time.sleep(0.1)  # Ждем немного для оценки
        asyncio.run(alert_manager.stop_evaluation())
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < 2.0  # Не более 2 секунд
        # Проверяем, что алерты создались
        alerts = alert_manager.get_alerts()
        assert len(alerts) > 0
    def test_dashboard_chart_rendering_performance(self, performance_test_data) -> None:
        """Тест производительности рендеринга графиков дашборда."""
        dashboard = get_dashboard("chart_test")
        # Добавляем много данных для графиков
        for i in range(performance_test_data["medium_dataset"]):
            from infrastructure.monitoring.monitoring_dashboard import MetricData
            metric_data = MetricData(
                name="chart_metric",
                values=[i],
                timestamps=[datetime.now() - timedelta(minutes=i)],
                labels={"iteration": i}
            )
            dashboard.add_metric_data(metric_data)
        # Создаем график
        from infrastructure.monitoring.monitoring_dashboard import ChartConfig
        chart_config = ChartConfig(
            name="perf_chart",
            title="Performance Chart",
            chart_type="line",
            metrics=["chart_metric"],
            time_range=timedelta(hours=1)
        )
        chart = dashboard.create_chart(chart_config)
        # Тестируем рендеринг графика
        start_time = time.time()
        chart_data = dashboard.get_chart_data(chart.name)
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что данные графика получены
        assert chart_data is not None
        assert "datasets" in chart_data
    def test_metric_aggregation_performance(self, performance_test_data) -> None:
        """Тест производительности агрегации метрик."""
        monitor = get_monitor("agg_test")
        # Добавляем много метрик для агрегации
        for i in range(performance_test_data["medium_dataset"]):
            record_metric("agg_metric", i, {"group": i % 10})
        # Тестируем агрегацию
        start_time = time.time()
        # Различные типы агрегации
        avg_value = monitor.aggregate_metrics("agg_metric", "avg")
        max_value = monitor.aggregate_metrics("agg_metric", "max")
        min_value = monitor.aggregate_metrics("agg_metric", "min")
        count_value = monitor.aggregate_metrics("agg_metric", "count")
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем результаты агрегации
        assert avg_value > 0
        assert max_value > 0
        assert min_value >= 0
        assert count_value > 0
    def test_trace_analysis_performance(self, performance_test_data) -> None:
        """Тест производительности анализа трейсов."""
        tracer = get_tracer("analysis_test")
        # Создаем много трейсов для анализа
        for i in range(performance_test_data["medium_dataset"]):
            trace_id = f"analysis_trace_{i}"
            span = tracer.start_trace(trace_id, f"operation_{i}")
            span.add_performance_metric("duration", i % 100)
            span.add_performance_metric("memory", i % 50)
            tracer.end_trace(trace_id, "success")
        # Тестируем анализ трейсов
        start_time = time.time()
        # Получаем статистику
        trace_stats = tracer.get_trace_statistics()
        # Анализируем узкие места
        bottlenecks = tracer.analyze_performance_bottlenecks()
        # Получаем сводку производительности
        summary = tracer.get_performance_summary()
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем результаты анализа
        assert "total_traces" in trace_stats
        assert len(bottlenecks) >= 0
        assert "total_traces" in summary
    def test_export_performance(self, performance_test_data) -> None:
        """Тест производительности экспорта данных."""
        monitor = get_monitor("export_test")
        alert_manager = get_alert_manager("export_test")
        # Добавляем данные для экспорта
        for i in range(performance_test_data["medium_dataset"]):
            record_metric(f"export_metric_{i % 10}", i, {"iteration": i})
            if i % 100 == 0:
                create_alert(f"Export alert {i}", AlertSeverity.INFO, "export_test")
        # Тестируем экспорт метрик
        start_time = time.time()
        json_metrics = monitor.export_metrics(format="json")
        csv_metrics = monitor.export_metrics(format="csv")
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что экспорт выполнен
        assert len(json_metrics) > 0
        assert len(csv_metrics) > 0
    def test_cleanup_performance(self, performance_test_data) -> None:
        """Тест производительности очистки данных."""
        monitor = get_monitor("cleanup_test")
        alert_manager = get_alert_manager("cleanup_test")
        tracer = get_tracer("cleanup_test")
        # Добавляем много данных
        for i in range(performance_test_data["medium_dataset"]):
            record_metric(f"cleanup_metric_{i}", i, {"iteration": i})
            create_alert(f"Cleanup alert {i}", AlertSeverity.INFO, "cleanup_test")
            trace_id = f"cleanup_trace_{i}"
            span = tracer.start_trace(trace_id, f"operation_{i}")
            tracer.end_trace(trace_id, "success")
        # Тестируем очистку
        start_time = time.time()
        monitor.cleanup_old_metrics(days=0)  # Очищаем все
        alert_manager.cleanup_old_alerts(days=0)  # Очищаем все
        tracer.cleanup_old_traces()
        end_time = time.time()
        duration = end_time - start_time
        # Проверяем производительность
        assert duration < performance_test_data["timeout_threshold"]
        # Проверяем, что данные очищены
        metrics = monitor.get_metrics()
        alerts = alert_manager.get_alerts()
        trace_stats = tracer.get_trace_statistics()
        assert len(metrics) == 0
        assert len(alerts) == 0
        assert trace_stats["total_traces"] == 0
class TestMonitoringScalability:
    """Тесты масштабируемости системы мониторинга."""
    def test_metric_scalability(self: "TestMonitoringScalability") -> None:
        """Тест масштабируемости метрик."""
        monitor = get_monitor("scale_test")
        # Тестируем с разными объемами данных
        volumes = [100, 1000, 10000]
        for volume in volumes:
            start_time = time.time()
            for i in range(volume):
                record_metric(f"scale_metric_{i % 100}", i, {"volume": volume})
            end_time = time.time()
            duration = end_time - start_time
            # Проверяем, что время растет линейно или лучше
            assert duration < volume * 0.0001  # Не более 0.1ms на метрику
    def test_alert_scalability(self: "TestMonitoringScalability") -> None:
        """Тест масштабируемости алертов."""
        alert_manager = get_alert_manager("scale_test")
        # Тестируем с разными объемами алертов
        volumes = [10, 100, 1000]
        for volume in volumes:
            start_time = time.time()
            for i in range(volume):
                create_alert(f"Scale alert {i}", AlertSeverity.INFO, "scale_test")
            end_time = time.time()
            duration = end_time - start_time
            # Проверяем, что время растет линейно или лучше
            assert duration < volume * 0.001  # Не более 1ms на алерт
    def test_trace_scalability(self: "TestMonitoringScalability") -> None:
        """Тест масштабируемости трейсов."""
        tracer = get_tracer("scale_test")
        # Тестируем с разными объемами трейсов
        volumes = [10, 100, 1000]
        for volume in volumes:
            start_time = time.time()
            for i in range(volume):
                trace_id = f"scale_trace_{i}"
                span = tracer.start_trace(trace_id, f"operation_{i}")
                tracer.end_trace(trace_id, "success")
            end_time = time.time()
            duration = end_time - start_time
            # Проверяем, что время растет линейно или лучше
            assert duration < volume * 0.001  # Не более 1ms на трейс
    def test_memory_scalability(self: "TestMonitoringScalability") -> None:
        """Тест масштабируемости использования памяти."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        # Тестируем с разными объемами данных
        volumes = [1000, 10000, 100000]
        memory_usage = []
        for volume in volumes:
            # Очищаем предыдущие данные
            monitor = get_monitor(f"memory_scale_{volume}")
            alert_manager = get_alert_manager(f"memory_scale_{volume}")
            tracer = get_tracer(f"memory_scale_{volume}")
            initial_memory = process.memory_info().rss
            # Добавляем данные
            for i in range(volume):
                record_metric(f"memory_scale_metric_{i}", i, {"volume": volume})
                if i % 100 == 0:
                    create_alert(f"Memory scale alert {i}", AlertSeverity.INFO, "memory_scale")
                if i % 50 == 0:
                    trace_id = f"memory_scale_trace_{i}"
                    span = tracer.start_trace(trace_id, f"operation_{i}")
                    tracer.end_trace(trace_id, "success")
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            memory_usage.append(memory_increase / volume)  # Память на элемент
        # Проверяем, что использование памяти на элемент не растет экспоненциально
        for i in range(1, len(memory_usage)):
            # Каждое увеличение объема не должно увеличивать память на элемент более чем в 2 раза
            assert memory_usage[i] < memory_usage[i-1] * 2
if __name__ == "__main__":
    pytest.main([__file__]) 
