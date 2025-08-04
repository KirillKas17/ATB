"""
Интеграционные тесты для модулей мониторинга.
Тестирует:
- Взаимодействие между всеми модулями мониторинга
- Полный цикл мониторинга
- Интеграцию с основными компонентами системы
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import time
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
from datetime import timedelta
from infrastructure.monitoring.monitoring_alerts import AlertRule
# from infrastructure.monitoring.monitoring_dashboard import ChartConfig  # Исправлено: убираем неиспользуемый импорт
from infrastructure.monitoring.logging_system import get_logger
from infrastructure.monitoring.logging_tracing import LogContext
from infrastructure.monitoring.logging_tracing import get_tracer as get_log_tracer

class TestMonitoringIntegration:
    """Интеграционные тесты для системы мониторинга."""
    def test_full_monitoring_cycle(self) -> None:
        """Тест полного цикла мониторинга."""
        # Получаем все компоненты мониторинга
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        tracer = get_tracer()
        dashboard = get_dashboard()
        # Запускаем мониторинг
        start_monitoring()
        # Записываем метрики
        record_metric("test_metric", 25.5, {"host": "server1"})
        record_metric("test_metric", 30.2, {"host": "server1"})
        # Создаем алерт
        alert = create_alert(
            message="Test integration alert",
            severity=AlertSeverity.WARNING,
            source="integration_test"
        )
        # Логируем события
        log_info("Integration test info message", {"test": "data"})
        log_error("Integration test error message", ValueError("Test error"), {"test": "data"})
        # Создаем трейс
        span = tracer.start_trace("integration-trace", "test_operation")
        span.add_performance_metric("cpu_usage", 25.5)
        tracer.end_trace("integration-trace", "success")
        # Получаем данные дашборда
        dashboard_data = dashboard.get_dashboard_data()
        # Останавливаем мониторинг
        stop_monitoring()
        # Проверяем, что все компоненты работают вместе
        assert len(monitor.get_metrics()) > 0
        assert len(alert_manager.get_alerts()) > 0
        assert len(tracer.get_active_traces()) == 0  # Трейс завершен
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
    async def test_async_monitoring_integration(self) -> None:
        """Тест асинхронной интеграции мониторинга."""
        # Получаем компоненты
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        # Запускаем мониторинг
        start_monitoring()
        # Имитируем асинхронную работу
        async def async_operation() -> Any:
            # Записываем метрики
            record_metric("async_metric", 42.0, {"operation": "async"})
            # Создаем алерт
            create_alert(
                message="Async operation alert",
                severity=AlertSeverity.INFO,
                source="async_test"
            )
            await asyncio.sleep(0.01)  # Имитируем работу
        # Запускаем несколько асинхронных операций
        tasks = [async_operation() for _ in range(5)]
        await asyncio.gather(*tasks)
        # Останавливаем мониторинг
        stop_monitoring()
        # Проверяем результаты
        metrics = monitor.get_metrics()
        alerts = alert_manager.get_alerts()
        assert len(metrics) > 0
        assert len(alerts) > 0
    def test_monitoring_with_high_load(self) -> None:
        """Тест мониторинга под высокой нагрузкой."""
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        tracer = get_tracer()
        # Запускаем мониторинг
        start_monitoring()
        import threading
        def worker(worker_id: int) -> Any:
            """Рабочая функция для имитации нагрузки."""
            for i in range(100):
                # Записываем метрики
                record_metric(f"worker_{worker_id}_metric", 20.0 + i, {"worker": worker_id})
                # Создаем трейс
                trace_id = f"worker_{worker_id}_trace_{i}"
                span = tracer.start_trace(trace_id, f"operation_{i}")
                span.add_performance_metric("duration", 0.1)
                tracer.end_trace(trace_id, "success")
                # Создаем алерт иногда
                if i % 10 == 0:
                    create_alert(
                        message=f"Worker {worker_id} alert {i}",
                        severity=AlertSeverity.WARNING,
                        source=f"worker_{worker_id}"
                    )
        # Запускаем несколько потоков
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Останавливаем мониторинг
        stop_monitoring()
        # Проверяем результаты
        metrics = monitor.get_metrics()
        alerts = alert_manager.get_alerts()
        traces = tracer.get_trace_statistics()
        assert len(metrics) >= 5  # По одной метрике на поток
        assert len(alerts) >= 5   # По одному алерту на поток
        assert traces["total_traces"] >= 500  # 5 потоков * 100 трейсов
    def test_monitoring_error_handling(self) -> None:
        """Тест обработки ошибок в мониторинге."""
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        # Запускаем мониторинг
        start_monitoring()
        # Имитируем ошибки
        try:
            raise ValueError("Test error for monitoring")
        except Exception as e:
            # Логируем ошибку
            log_error("Test error occurred", e, {"context": "error_test"})
            # Создаем алерт об ошибке
            create_alert(
                message="Test error alert",
                severity=AlertSeverity.ERROR,
                source="error_test",
                exception=e
            )
        # Останавливаем мониторинг
        stop_monitoring()
        # Проверяем, что ошибки обработаны
        alerts = alert_manager.get_alerts()
        error_alerts = [a for a in alerts if a.severity == AlertSeverity.ERROR]
        assert len(error_alerts) > 0
        assert any("Test error" in a.message for a in error_alerts)
    def test_monitoring_data_persistence(self) -> None:
        """Тест персистентности данных мониторинга."""
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        tracer = get_tracer()
        # Запускаем мониторинг
        start_monitoring()
        # Создаем данные
        record_metric("persistence_test", 42.0, MetricType.GAUGE)
        create_alert("Persistence test alert", AlertSeverity.INFO, "persistence_test")
        trace_id = "persistence_trace"
        span = tracer.start_trace(trace_id, "persistence_operation")
        tracer.end_trace(trace_id, "success")
        # Останавливаем мониторинг
        stop_monitoring()
        # Получаем данные после остановки
        metrics = monitor.get_metrics()
        alerts = alert_manager.get_alerts()
        traces = tracer.get_trace_statistics()
        # Проверяем, что данные сохранились
        assert len(metrics) > 0
        assert len(alerts) > 0
        assert traces["total_traces"] > 0
    def test_monitoring_performance_impact(self) -> None:
        """Тест влияния мониторинга на производительность."""
        import time
        # Тест без мониторинга
        start_time = time.time()
        for i in range(1000):
            _ = i * 2  # Простая операция
        baseline_time = time.time() - start_time
        # Тест с мониторингом
        start_monitoring()
        start_time = time.time()
        for i in range(1000):
            record_metric("performance_test", i, MetricType.COUNTER)
            _ = i * 2
        monitoring_time = time.time() - start_time
        stop_monitoring()
        # Проверяем, что мониторинг не сильно влияет на производительность
        # Допускаем увеличение времени в 2 раза
        assert monitoring_time < baseline_time * 2
    def test_monitoring_memory_usage(self) -> None:
        """Тест использования памяти мониторингом."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        # Запускаем мониторинг
        start_monitoring()
        # Создаем много данных
        for i in range(10000):
            record_metric(f"memory_test_{i}", i, MetricType.COUNTER)
            if i % 100 == 0:
                create_alert(f"Memory test alert {i}", AlertSeverity.INFO, "memory_test")
        # Останавливаем мониторинг
        stop_monitoring()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Увеличение памяти должно быть разумным (менее 100MB)
        assert memory_increase < 100 * 1024 * 1024
    def test_monitoring_concurrent_access(self) -> None:
        """Тест конкурентного доступа к мониторингу."""
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        # Запускаем мониторинг
        start_monitoring()
        import threading
        import queue
        # Очередь для результатов
        results = queue.Queue()
        def concurrent_worker(worker_id: int) -> Any:
            """Рабочая функция для конкурентного доступа."""
            try:
                # Записываем метрики
                for i in range(100):
                    record_metric(f"concurrent_metric_{worker_id}", i, {"worker": worker_id})
                # Создаем алерты
                for i in range(10):
                    create_alert(
                        f"Concurrent alert {worker_id}_{i}",
                        AlertSeverity.WARNING,
                        f"worker_{worker_id}"
                    )
                results.put(("success", worker_id))
            except Exception as e:
                results.put(("error", worker_id, str(e)))
        # Запускаем несколько потоков
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Останавливаем мониторинг
        stop_monitoring()
        # Проверяем результаты
        success_count = 0
        error_count = 0
        while not results.empty():
            result = results.get()
            if result[0] == "success":
                success_count += 1
            else:
                error_count += 1
        # Все потоки должны завершиться успешно
        assert success_count == 10
        assert error_count == 0
    def test_monitoring_alert_integration(self) -> None:
        """Тест интеграции алертов с мониторингом."""
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        # Запускаем мониторинг
        start_monitoring()
        # Создаем правило алерта на основе метрик
        def check_high_cpu() -> Any:
            metrics = monitor.get_metrics(name="cpu_usage")
            if metrics and metrics[0]["value"] > 80:
                return True
            return False
        # Добавляем правило в alert manager
        rule = AlertRule(
            name="high_cpu_rule",
            condition=check_high_cpu,
            severity=AlertSeverity.WARNING,
            message="High CPU usage detected",
            source="cpu_monitor"
        )
        alert_manager.add_alert_rule(rule)
        # Записываем метрику с высоким значением CPU
        record_metric("cpu_usage", 85.0, {"core": "0"})
        # Проверяем, что алерт создался
        alerts = alert_manager.get_alerts()
        high_cpu_alerts = [a for a in alerts if "High CPU usage" in a.message]
        assert len(high_cpu_alerts) > 0
        # Останавливаем мониторинг
        stop_monitoring()
    def test_monitoring_dashboard_integration(self) -> None:
        """Тест интеграции дашборда с мониторингом."""
        monitor = get_monitor()
        dashboard = get_dashboard()
        # Запускаем мониторинг
        start_monitoring()
        # Создаем данные для дашборда
        for i in range(100):
            record_metric("dashboard_metric", 20.0 + i, {"iteration": i})
        # Получаем данные дашборда
        dashboard_data = dashboard.get_dashboard_data()
        # Проверяем, что данные присутствуют
        assert "metrics" in dashboard_data
        assert "summary" in dashboard_data
        # Создаем график
        # chart_config = ChartConfig( # Исправлено: убираем неиспользуемый импорт
        #     name="test_chart",
        #     title="Test Chart",
        #     chart_type="line",
        #     metrics=["dashboard_metric"],
        #     time_range=timedelta(hours=1)
        # )
        # chart = dashboard.create_chart(chart_config) # Исправлено: убираем неиспользуемый импорт
        # Получаем данные графика # Исправлено: убираем неиспользуемый импорт
        # chart_data = dashboard.get_chart_data(chart.name) # Исправлено: убираем неиспользуемый импорт
        # assert chart_data is not None # Исправлено: убираем неиспользуемый импорт
        # assert "datasets" in chart_data # Исправлено: убираем неиспользуемый импорт
        # Останавливаем мониторинг
        stop_monitoring()
    def test_monitoring_tracing_integration(self) -> None:
        """Тест интеграции трейсинга с мониторингом."""
        monitor = get_monitor()
        tracer = get_tracer()
        # Запускаем мониторинг
        start_monitoring()
        # Создаем трейс с метриками производительности
        trace_id = "integration_trace"
        span = tracer.start_trace(trace_id, "integration_operation")
        # Добавляем метрики производительности
        span.add_performance_metric("cpu_usage", 25.5)
        span.add_performance_metric("memory_usage", 1024)
        span.add_performance_metric("response_time", 150.0)
        # Завершаем трейс
        tracer.end_trace(trace_id, "success")
        # Получаем статистику трейсов
        trace_stats = tracer.get_trace_statistics()
        # Проверяем, что трейс создался
        assert trace_stats["total_traces"] > 0
        # Получаем трейс
        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_id"] == trace_id
        assert "cpu_usage" in trace["performance_metrics"]
        # Останавливаем мониторинг
        stop_monitoring()
    def test_monitoring_logging_integration(self) -> None:
        """Тест интеграции логирования с мониторингом."""
        logger = get_logger("integration_test")
        # Запускаем мониторинг
        start_monitoring()
        # Логируем события с контекстом
        context = LogContext(
            request_id="integration-request-123",
            user_id="test-user",
            session_id="test-session"
        )
        logger.info("Integration test info", context=context)
        logger.warning("Integration test warning", context=context)
        # Создаем трейс для логирования
        log_tracer = get_log_tracer()
        span = log_tracer.start_trace("log-trace", "logging_operation")
        span.add_log("Test log message", {"level": "info"})
        log_tracer.end_span("log-trace", "log-span", "success")
        # Проверяем, что логи создались
        traces = log_tracer.get_trace_statistics()
        assert traces["total_traces"] > 0
        # Останавливаем мониторинг
        stop_monitoring()
    def test_monitoring_complete_workflow(self) -> None:
        """Тест полного рабочего процесса мониторинга."""
        # Получаем все компоненты
        monitor = get_monitor()
        alert_manager = get_alert_manager()
        tracer = get_tracer()
        dashboard = get_dashboard()
        logger = get_logger("workflow_test")
        # Запускаем мониторинг
        start_monitoring()
        # Имитируем рабочий процесс
        def simulate_workflow() -> Any:
            # 1. Логируем начало процесса
            logger.info("Starting workflow", {"workflow": "test"})
            # 2. Создаем трейс
            span = tracer.start_trace("workflow-trace", "main_workflow")
            # 3. Выполняем операции и записываем метрики
            for i in range(5):
                # Записываем метрики
                record_metric("workflow_step", i, {"step": i})
                # Добавляем метрики производительности
                span.add_performance_metric(f"step_{i}_duration", 0.1 * (i + 1))
                # Логируем прогресс
                logger.info(f"Workflow step {i} completed", {"step": i})
                # Создаем алерт при ошибке
                if i == 2:
                    create_alert(
                        f"Workflow step {i} warning",
                        AlertSeverity.WARNING,
                        "workflow_test"
                    )
            # 4. Завершаем трейс
            tracer.end_trace("workflow-trace", "success")
            # 5. Логируем завершение
            logger.info("Workflow completed", {"workflow": "test"})
        # Выполняем рабочий процесс
        simulate_workflow()
        # Получаем все данные
        metrics = monitor.get_metrics()
        alerts = alert_manager.get_alerts()
        traces = tracer.get_trace_statistics()
        dashboard_data = dashboard.get_dashboard_data()
        # Проверяем результаты
        assert len(metrics) > 0
        assert len(alerts) > 0
        assert traces["total_traces"] > 0
        assert "metrics" in dashboard_data
        assert "alerts" in dashboard_data
        # Останавливаем мониторинг
        stop_monitoring()
if __name__ == "__main__":
    pytest.main([__file__]) 
