"""
Unit тесты для модуля monitoring_tracing.
Тестирует:
- PerformanceTracer
- Функции трейсинга производительности
- Измерение производительности операций
- Анализ узких мест
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from infrastructure.monitoring.monitoring_tracing import (
    PerformanceTracer,
    get_tracer,
    start_trace,
    end_trace,
    add_child_span,
    TraceSpan,
)

try:
    from infrastructure.monitoring.monitoring_tracing import PerformanceMetrics, TraceContext
except ImportError:

    class MockPerformanceMetrics:
        pass

    class MockTraceContext:
        pass


class TestPerformanceTracer:
    """Тесты для PerformanceTracer."""

    def test_init_default(self: "TestPerformanceTracer") -> None:
        """Тест инициализации с параметрами по умолчанию."""
        tracer = PerformanceTracer()
        assert tracer.name == "performance"
        assert tracer.max_traces == 1000
        assert tracer.max_duration == 300.0
        assert tracer.traces == {}
        assert tracer.active_traces == {}
        assert tracer.performance_metrics == {}

    def test_init_custom(self: "TestPerformanceTracer") -> None:
        """Тест инициализации с пользовательскими параметрами."""
        tracer = PerformanceTracer(name="custom_performance", max_traces=500, max_duration=600.0)
        assert tracer.name == "custom_performance"
        assert tracer.max_traces == 500
        assert tracer.max_duration == 600.0

    def test_start_trace(self: "TestPerformanceTracer") -> None:
        """Тест начала трейса производительности."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        operation = "test_operation"
        span = tracer.start_trace(trace_id, operation)
        assert span.trace_id == trace_id
        assert span.operation == operation
        assert span.start_time is not None
        assert span.status == "active"
        assert span.performance_metrics == {}
        # Проверяем, что span добавлен в активные трейсы
        assert trace_id in tracer.active_traces

    def test_start_trace_with_context(self: "TestPerformanceTracer") -> None:
        """Тест начала трейса с контекстом."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        operation = "test_operation"
        context = {"user_id": "user-123", "request_id": "req-456"}
        span = tracer.start_trace(trace_id, operation, context=context)
        assert span.context == context

    def test_end_trace(self: "TestPerformanceTracer") -> None:
        """Тест завершения трейса производительности."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        operation = "test_operation"
        # Создаем трейс
        span = tracer.start_trace(trace_id, operation)
        # Добавляем метрики производительности
        span.add_performance_metric("cpu_usage", 25.5)
        span.add_performance_metric("memory_usage", 1024)
        # Завершаем трейс
        tracer.end_trace(trace_id, "success")
        # Проверяем, что трейс завершен
        assert span.end_time is not None
        assert span.status == "success"
        assert span.duration > 0
        # Проверяем, что трейс удален из активных
        assert trace_id not in tracer.active_traces
        # Проверяем, что трейс сохранен
        assert trace_id in tracer.traces

    def test_end_trace_with_error(self: "TestPerformanceTracer") -> None:
        """Тест завершения трейса с ошибкой."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        operation = "test_operation"
        # Создаем трейс
        span = tracer.start_trace(trace_id, operation)
        # Завершаем трейс с ошибкой
        error = ValueError("Performance error")
        tracer.end_trace(trace_id, "error", error=error)
        # Проверяем, что трейс завершен с ошибкой
        assert span.status == "error"
        assert span.error == error

    def test_add_child_span(self: "TestPerformanceTracer") -> None:
        """Тест добавления дочернего span."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        parent_span_id = "parent-span-456"
        child_span_id = "child-span-789"
        # Создаем родительский span
        parent_span = tracer.start_trace(trace_id, "parent_operation", span_id=parent_span_id)
        # Создаем дочерний span
        child_span = tracer.add_child_span(trace_id, child_span_id, "child_operation", parent_span_id)
        assert child_span.trace_id == trace_id
        assert child_span.span_id == child_span_id
        assert child_span.operation == "child_operation"
        assert child_span.parent_span_id == parent_span_id
        assert child_span.start_time is not None
        assert child_span.status == "active"

    def test_add_performance_metric(self: "TestPerformanceTracer") -> None:
        """Тест добавления метрики производительности."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        operation = "test_operation"
        span = tracer.start_trace(trace_id, operation)
        # Добавляем метрики производительности
        span.add_performance_metric("cpu_usage", 25.5)
        span.add_performance_metric("memory_usage", 1024)
        span.add_performance_metric("response_time", 150.0)
        span.add_performance_metric("throughput", 1000)
        # Проверяем метрики
        assert span.performance_metrics["cpu_usage"] == 25.5
        assert span.performance_metrics["memory_usage"] == 1024
        assert span.performance_metrics["response_time"] == 150.0
        assert span.performance_metrics["throughput"] == 1000

    def test_get_trace(self: "TestPerformanceTracer") -> None:
        """Тест получения трейса."""
        tracer = PerformanceTracer()
        trace_id = "perf-trace-123"
        operation = "test_operation"
        # Создаем трейс
        span = tracer.start_trace(trace_id, operation)
        span.add_performance_metric("cpu_usage", 25.5)
        tracer.end_trace(trace_id, "success")
        # Получаем трейс
        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_id"] == trace_id
        assert trace["operation"] == operation
        assert trace["status"] == "success"
        assert "cpu_usage" in trace["performance_metrics"]

    def test_get_active_traces(self: "TestPerformanceTracer") -> None:
        """Тест получения активных трейсов."""
        tracer = PerformanceTracer()
        trace_id1 = "perf-trace-1"
        trace_id2 = "perf-trace-2"
        # Создаем активные трейсы
        tracer.start_trace(trace_id1, "operation1")
        tracer.start_trace(trace_id2, "operation2")
        active_traces = tracer.get_active_traces()
        assert len(active_traces) == 2
        assert trace_id1 in active_traces
        assert trace_id2 in active_traces

    def test_get_performance_metrics(self: "TestPerformanceTracer") -> None:
        """Тест получения метрик производительности."""
        tracer = PerformanceTracer()
        # Создаем несколько трейсов с метриками
        for i in range(3):
            trace_id = f"perf-trace-{i}"
            operation = f"operation{i}"
            span = tracer.start_trace(trace_id, operation)
            span.add_performance_metric("cpu_usage", 20.0 + i * 5)
            span.add_performance_metric("memory_usage", 1000 + i * 100)
            tracer.end_trace(trace_id, "success")
        metrics = tracer.get_performance_metrics()
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert len(metrics["cpu_usage"]) == 3
        assert len(metrics["memory_usage"]) == 3

    def test_analyze_performance_bottlenecks(self: "TestPerformanceTracer") -> None:
        """Тест анализа узких мест производительности."""
        tracer = PerformanceTracer()
        # Создаем трейсы с разной производительностью
        operations = [
            ("fast_operation", 50.0),  # 50ms
            ("slow_operation", 500.0),  # 500ms
            ("medium_operation", 200.0),  # 200ms
        ]
        for operation_name, duration in operations:
            trace_id = f"trace-{operation_name}"
            span = tracer.start_trace(trace_id, operation_name)
            time.sleep(duration / 1000)  # Имитируем работу
            tracer.end_trace(trace_id, "success")
        bottlenecks = tracer.analyze_performance_bottlenecks()
        assert len(bottlenecks) > 0
        # Самый медленный должен быть slow_operation
        assert "slow_operation" in [b["operation"] for b in bottlenecks]

    def test_get_performance_summary(self: "TestPerformanceTracer") -> None:
        """Тест получения сводки производительности."""
        tracer = PerformanceTracer()
        # Создаем трейсы с разной производительностью
        for i in range(5):
            trace_id = f"perf-trace-{i}"
            operation = f"operation{i}"
            span = tracer.start_trace(trace_id, operation)
            span.add_performance_metric("cpu_usage", 20.0 + i * 10)
            span.add_performance_metric("memory_usage", 1000 + i * 200)
            tracer.end_trace(trace_id, "success")
        summary = tracer.get_performance_summary()
        assert "total_traces" in summary
        assert "average_duration" in summary
        assert "slowest_operations" in summary
        assert "fastest_operations" in summary
        assert "performance_trends" in summary

    def test_cleanup_old_traces(self: "TestPerformanceTracer") -> None:
        """Тест очистки старых трейсов."""
        tracer = PerformanceTracer(max_traces=2)
        # Создаем больше трейсов, чем максимальное количество
        for i in range(5):
            trace_id = f"perf-trace-{i}"
            operation = f"operation{i}"
            span = tracer.start_trace(trace_id, operation)
            tracer.end_trace(trace_id, "success")
        # Проверяем, что старые трейсы удалены
        assert len(tracer.traces) <= 2

    def test_cleanup_expired_traces(self: "TestPerformanceTracer") -> None:
        """Тест очистки истекших трейсов."""
        tracer = PerformanceTracer(max_duration=1.0)  # 1 секунда
        trace_id = "perf-trace-123"
        operation = "test_operation"
        # Создаем трейс
        span = tracer.start_trace(trace_id, operation)
        tracer.end_trace(trace_id, "success")
        # Ждем, пока трейс истечет
        time.sleep(1.1)
        # Вызываем очистку
        tracer.cleanup_expired_traces()
        # Проверяем, что трейс удален
        assert trace_id not in tracer.traces

    def test_error_handling_invalid_trace_id(self: "TestPerformanceTracer") -> None:
        """Тест обработки ошибок с неверным trace_id."""
        tracer = PerformanceTracer()
        # Пытаемся завершить несуществующий трейс
        with pytest.raises(ValueError, match="Trace not found"):
            tracer.end_trace("invalid-trace", "success")

    def test_concurrent_tracing(self: "TestPerformanceTracer") -> None:
        """Тест конкурентного трейсинга."""
        tracer = PerformanceTracer()

        def create_trace(trace_id: str, operation: str) -> Any:
            span = tracer.start_trace(trace_id, operation)
            time.sleep(0.01)  # Имитируем работу
            tracer.end_trace(trace_id, "success")

        # Создаем несколько трейсов одновременно
        import threading

        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_trace, args=(f"perf-trace-{i}", f"operation{i}"))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все трейсы созданы
        assert len(tracer.traces) == 10

    def test_performance_tracing_accuracy(self: "TestPerformanceTracer") -> None:
        """Тест точности измерения производительности."""
        tracer = PerformanceTracer()
        trace_id = "accuracy-test"
        operation = "test_operation"
        # Создаем трейс
        span = tracer.start_trace(trace_id, operation)
        # Ждем точно 100ms
        time.sleep(0.1)
        tracer.end_trace(trace_id, "success")
        # Проверяем, что измеренная длительность близка к ожидаемой
        trace = tracer.get_trace(trace_id)
        measured_duration = trace["duration"]
        # Допускаем погрешность в 10ms
        assert abs(measured_duration - 100.0) < 10.0

    def test_memory_usage_tracking(self: "TestPerformanceTracer") -> None:
        """Тест отслеживания использования памяти."""
        tracer = PerformanceTracer()
        trace_id = "memory-test"
        operation = "memory_operation"
        span = tracer.start_trace(trace_id, operation)
        # Имитируем использование памяти
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        # Создаем объекты для увеличения памяти
        test_data = [i for i in range(10000)]
        current_memory = process.memory_info().rss
        memory_usage = current_memory - initial_memory
        span.add_performance_metric("memory_usage", memory_usage)
        tracer.end_trace(trace_id, "success")
        # Проверяем, что метрика памяти сохранена
        trace = tracer.get_trace(trace_id)
        assert "memory_usage" in trace["performance_metrics"]
        assert trace["performance_metrics"]["memory_usage"] > 0


class TestGetTracer:
    """Тесты для функции get_tracer."""

    def test_get_tracer_default(self: "TestGetTracer") -> None:
        """Тест получения трейсера по умолчанию."""
        tracer = get_tracer()
        assert isinstance(tracer, PerformanceTracer)
        assert tracer.name == "performance"

    def test_get_tracer_custom_name(self: "TestGetTracer") -> None:
        """Тест получения трейсера с пользовательским именем."""
        tracer = get_tracer("custom_performance")
        assert isinstance(tracer, PerformanceTracer)
        assert tracer.name == "custom_performance"

    def test_get_tracer_singleton(self: "TestGetTracer") -> None:
        """Тест, что get_tracer возвращает тот же экземпляр для одного имени."""
        tracer1 = get_tracer("singleton_test")
        tracer2 = get_tracer("singleton_test")
        assert tracer1 is tracer2

    def test_get_tracer_different_names(self: "TestGetTracer") -> None:
        """Тест, что разные имена возвращают разные экземпляры."""
        tracer1 = get_tracer("tracer1")
        tracer2 = get_tracer("tracer2")
        assert tracer1 is not tracer2


class TestTraceFunctions:
    """Тесты для функций трейсинга."""

    @patch("infrastructure.monitoring.monitoring_tracing.get_tracer")
    def test_start_trace(self, mock_get_tracer) -> None:
        """Тест функции start_trace."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = Mock()
        mock_tracer.start_trace.return_value = mock_span
        span = start_trace("test-trace", "test_operation")
        mock_tracer.start_trace.assert_called_once_with("test-trace", "test_operation")
        assert span == mock_span

    @patch("infrastructure.monitoring.monitoring_tracing.get_tracer")
    def test_end_trace(self, mock_get_tracer) -> None:
        """Тест функции end_trace."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        end_trace("test-trace", "success")
        mock_tracer.end_trace.assert_called_once_with("test-trace", "success")

    @patch("infrastructure.monitoring.monitoring_tracing.get_tracer")
    def test_add_child_span(self, mock_get_tracer) -> None:
        """Тест функции add_child_span."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        mock_span = Mock()
        mock_tracer.add_child_span.return_value = mock_span
        span = add_child_span("test-trace", "child-span", "child_operation", "parent-span")
        mock_tracer.add_child_span.assert_called_once_with("test-trace", "child-span", "child_operation", "parent-span")
        assert span == mock_span


class TestTraceSpan:
    """Тесты для TraceSpan."""

    def test_trace_span_init(self: "TestTraceSpan") -> None:
        """Тест инициализации TraceSpan."""
        span = TraceSpan(
            trace_id="test-trace", span_id="test-span", operation="test_operation", parent_span_id="parent-span"
        )
        assert span.trace_id == "test-trace"
        assert span.span_id == "test-span"
        assert span.operation == "test_operation"
        assert span.parent_span_id == "parent-span"
        assert span.start_time is not None
        assert span.status == "active"
        assert span.performance_metrics == {}

    def test_trace_span_add_performance_metric(self: "TestTraceSpan") -> None:
        """Тест добавления метрики производительности к TraceSpan."""
        span = TraceSpan("test-trace", "test-span", "test_operation")
        span.add_performance_metric("cpu_usage", 25.5)
        span.add_performance_metric("memory_usage", 1024)
        assert span.performance_metrics["cpu_usage"] == 25.5
        assert span.performance_metrics["memory_usage"] == 1024

    def test_trace_span_end(self: "TestTraceSpan") -> None:
        """Тест завершения TraceSpan."""
        span = TraceSpan("test-trace", "test-span", "test_operation")
        span.end("success")
        assert span.end_time is not None
        assert span.status == "success"
        assert span.duration > 0

    def test_trace_span_end_with_error(self: "TestTraceSpan") -> None:
        """Тест завершения TraceSpan с ошибкой."""
        span = TraceSpan("test-trace", "test-span", "test_operation")
        error = ValueError("Performance error")
        span.end("error", error=error)
        assert span.status == "error"
        assert span.error == error

    def test_trace_span_to_dict(self: "TestTraceSpan") -> None:
        """Тест преобразования TraceSpan в словарь."""
        span = TraceSpan("test-trace", "test-span", "test_operation")
        span.add_performance_metric("cpu_usage", 25.5)
        span.end("success")
        span_dict = span.to_dict()
        assert span_dict["trace_id"] == "test-trace"
        assert span_dict["span_id"] == "test-span"
        assert span_dict["operation"] == "test_operation"
        assert span_dict["status"] == "success"
        assert "cpu_usage" in span_dict["performance_metrics"]


class TestPerformanceMetrics:
    """Тесты для PerformanceMetrics."""

    def test_performance_metrics_init(self: "TestPerformanceMetrics") -> None:
        """Тест инициализации PerformanceMetrics."""
        metrics = PerformanceMetrics()
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0
        assert metrics.response_time == 0.0
        assert metrics.throughput == 0
        assert metrics.error_rate == 0.0

    def test_performance_metrics_update(self: "TestPerformanceMetrics") -> None:
        """Тест обновления PerformanceMetrics."""
        metrics = PerformanceMetrics()
        metrics.update(cpu_usage=25.5, memory_usage=1024, response_time=150.0, throughput=1000, error_rate=0.01)
        assert metrics.cpu_usage == 25.5
        assert metrics.memory_usage == 1024
        assert metrics.response_time == 150.0
        assert metrics.throughput == 1000
        assert metrics.error_rate == 0.01

    def test_performance_metrics_to_dict(self: "TestPerformanceMetrics") -> None:
        """Тест преобразования PerformanceMetrics в словарь."""
        metrics = PerformanceMetrics()
        metrics.update(cpu_usage=25.5, memory_usage=1024, response_time=150.0, throughput=1000, error_rate=0.01)
        metrics_dict = metrics.to_dict()
        assert metrics_dict["cpu_usage"] == 25.5
        assert metrics_dict["memory_usage"] == 1024
        assert metrics_dict["response_time"] == 150.0
        assert metrics_dict["throughput"] == 1000
        assert metrics_dict["error_rate"] == 0.01


class TestTraceContext:
    """Тесты для TraceContext."""

    def test_trace_context_init(self: "TestTraceContext") -> None:
        """Тест инициализации TraceContext."""
        context = TraceContext(trace_id="test-trace", span_id="test-span", parent_span_id="parent-span")
        assert context.trace_id == "test-trace"
        assert context.span_id == "test-span"
        assert context.parent_span_id == "parent-span"

    def test_trace_context_to_dict(self: "TestTraceContext") -> None:
        """Тест преобразования TraceContext в словарь."""
        context = TraceContext("test-trace", "test-span", "parent-span")
        context_dict = context.to_dict()
        assert context_dict["trace_id"] == "test-trace"
        assert context_dict["span_id"] == "test-span"
        assert context_dict["parent_span_id"] == "parent-span"

    def test_trace_context_from_dict(self: "TestTraceContext") -> None:
        """Тест создания TraceContext из словаря."""
        context_dict = {"trace_id": "test-trace", "span_id": "test-span", "parent_span_id": "parent-span"}
        context = TraceContext.from_dict(context_dict)
        assert context.trace_id == "test-trace"
        assert context.span_id == "test-span"
        assert context.parent_span_id == "parent-span"


class TestTraceProtocol:
    """Тесты для протокола TraceProtocol."""

    def test_performance_tracer_implements_protocol(self: "TestTraceProtocol") -> None:
        """Тест, что PerformanceTracer реализует TraceProtocol."""
        tracer = PerformanceTracer()
        # Проверяем наличие всех методов протокола
        assert hasattr(tracer, "start_trace")
        assert hasattr(tracer, "start_span")
        assert hasattr(tracer, "end_span")
        assert hasattr(tracer, "add_tag")
        assert hasattr(tracer, "add_log")
        assert hasattr(tracer, "get_trace")
        assert hasattr(tracer, "get_active_traces")
        assert hasattr(tracer, "get_trace_statistics")


if __name__ == "__main__":
    pytest.main([__file__])
