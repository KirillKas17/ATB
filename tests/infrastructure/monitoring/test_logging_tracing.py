"""
Unit тесты для модуля logging_tracing.
Тестирует:
- RequestTracer
- Функции трейсинга
- Трейсинг запросов
- Обработку ошибок
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from infrastructure.monitoring.logging_tracing import (
    RequestTracer,
    get_tracer,
    log_trace,
    log_debug,
    log_info,
    log_warning,
    log_error,
    log_critical,
)


class TestRequestTracer:
    """Тесты для RequestTracer."""

    def test_init_default(self: "TestRequestTracer") -> None:
        """Тест инициализации с параметрами по умолчанию."""
        tracer = RequestTracer()
        assert tracer.name == "default"
        assert tracer.max_spans == 1000
        assert tracer.max_duration == 300.0
        assert tracer.spans == {}
        assert tracer.active_traces == {}

    def test_init_custom(self: "TestRequestTracer") -> None:
        """Тест инициализации с пользовательскими параметрами."""
        tracer = RequestTracer(name="custom_tracer", max_spans=500, max_duration=600.0)
        assert tracer.name == "custom_tracer"
        assert tracer.max_spans == 500
        assert tracer.max_duration == 600.0

    def test_start_trace(self: "TestRequestTracer") -> None:
        """Тест начала трейса."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        assert span.trace_id == trace_id
        assert span.span_id == span_id
        assert span.operation == "test_operation"
        assert span.start_time is not None
        assert span.status == "active"
        # Проверяем, что span добавлен в активные трейсы
        assert trace_id in tracer.active_traces
        assert span_id in tracer.active_traces[trace_id]

    def test_start_span(self: "TestRequestTracer") -> None:
        """Тест начала span."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        parent_span_id = "parent-span-456"
        span_id = "child-span-789"
        # Создаем родительский span
        parent_span = tracer.start_trace(trace_id, parent_span_id, "parent_operation")
        # Создаем дочерний span
        child_span = tracer.start_span(trace_id, span_id, "child_operation", parent_span_id)
        assert child_span.trace_id == trace_id
        assert child_span.span_id == span_id
        assert child_span.operation == "child_operation"
        assert child_span.parent_span_id == parent_span_id
        assert child_span.start_time is not None
        assert child_span.status == "active"
        # Проверяем, что дочерний span добавлен в активные трейсы
        assert span_id in tracer.active_traces[trace_id]

    def test_end_span(self: "TestRequestTracer") -> None:
        """Тест завершения span."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        # Создаем span
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        # Добавляем теги
        span.add_tag("test_tag", "test_value")
        # Завершаем span
        tracer.end_span(trace_id, span_id, "success")
        # Проверяем, что span завершен
        assert span.end_time is not None
        assert span.status == "success"
        assert span.duration > 0
        # Проверяем, что span удален из активных трейсов
        assert span_id not in tracer.active_traces[trace_id]
        # Проверяем, что span сохранен в spans
        assert span_id in tracer.spans

    def test_end_span_with_error(self: "TestRequestTracer") -> None:
        """Тест завершения span с ошибкой."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        # Создаем span
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        # Завершаем span с ошибкой
        error = ValueError("Test error")
        tracer.end_span(trace_id, span_id, "error", error=error)
        # Проверяем, что span завершен с ошибкой
        assert span.status == "error"
        assert span.error == error

    def test_add_tag(self: "TestRequestTracer") -> None:
        """Тест добавления тега к span."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        # Добавляем теги
        span.add_tag("string_tag", "test_value")
        span.add_tag("int_tag", 42)
        span.add_tag("float_tag", 3.14)
        span.add_tag("bool_tag", True)
        # Проверяем теги
        assert span.tags["string_tag"] == "test_value"
        assert span.tags["int_tag"] == 42
        assert span.tags["float_tag"] == 3.14
        assert span.tags["bool_tag"] is True

    def test_add_log(self: "TestRequestTracer") -> None:
        """Тест добавления лога к span."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        # Добавляем логи
        span.add_log("Test log message", {"level": "info"})
        span.add_log("Error log message", {"level": "error", "error": "test_error"})
        # Проверяем логи
        assert len(span.logs) == 2
        assert span.logs[0]["message"] == "Test log message"
        assert span.logs[0]["fields"]["level"] == "info"
        assert span.logs[1]["message"] == "Error log message"
        assert span.logs[1]["fields"]["level"] == "error"

    def test_get_trace(self: "TestRequestTracer") -> None:
        """Тест получения трейса."""
        tracer = RequestTracer()
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        # Создаем span
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        tracer.end_span(trace_id, span_id, "success")
        # Получаем трейс
        trace = tracer.get_trace(trace_id)
        assert trace is not None
        assert trace["trace_id"] == trace_id
        assert len(trace["spans"]) == 1
        assert trace["spans"][0]["span_id"] == span_id

    def test_get_active_traces(self: "TestRequestTracer") -> None:
        """Тест получения активных трейсов."""
        tracer = RequestTracer()
        trace_id1 = "test-trace-1"
        trace_id2 = "test-trace-2"
        # Создаем активные трейсы
        tracer.start_trace(trace_id1, "span-1", "operation1")
        tracer.start_trace(trace_id2, "span-2", "operation2")
        active_traces = tracer.get_active_traces()
        assert len(active_traces) == 2
        assert trace_id1 in active_traces
        assert trace_id2 in active_traces

    def test_cleanup_old_spans(self: "TestRequestTracer") -> None:
        """Тест очистки старых spans."""
        tracer = RequestTracer(max_spans=2)
        # Создаем больше spans, чем максимальное количество
        for i in range(5):
            trace_id = f"test-trace-{i}"
            span_id = f"test-span-{i}"
            span = tracer.start_trace(trace_id, span_id, f"operation{i}")
            tracer.end_span(trace_id, span_id, "success")
        # Проверяем, что старые spans удалены
        assert len(tracer.spans) <= 2

    def test_cleanup_expired_traces(self: "TestRequestTracer") -> None:
        """Тест очистки истекших трейсов."""
        tracer = RequestTracer(max_duration=1.0)  # 1 секунда
        trace_id = "test-trace-123"
        span_id = "test-span-456"
        # Создаем span
        span = tracer.start_trace(trace_id, span_id, "test_operation")
        tracer.end_span(trace_id, span_id, "success")
        # Ждем, пока трейс истечет
        time.sleep(1.1)
        # Вызываем очистку
        tracer.cleanup_expired_traces()
        # Проверяем, что трейс удален
        assert trace_id not in tracer.spans

    def test_get_trace_statistics(self: "TestRequestTracer") -> None:
        """Тест получения статистики трейсов."""
        tracer = RequestTracer()
        # Создаем несколько трейсов
        for i in range(3):
            trace_id = f"test-trace-{i}"
            span_id = f"test-span-{i}"
            span = tracer.start_trace(trace_id, span_id, f"operation{i}")
            span.add_tag("test_tag", f"value{i}")
            tracer.end_span(trace_id, span_id, "success")
        stats = tracer.get_trace_statistics()
        assert stats["total_traces"] == 3
        assert stats["total_spans"] == 3
        assert stats["average_duration"] > 0
        assert "duration_distribution" in stats

    def test_error_handling_invalid_trace_id(self: "TestRequestTracer") -> None:
        """Тест обработки ошибок с неверным trace_id."""
        tracer = RequestTracer()
        # Пытаемся завершить несуществующий span
        with pytest.raises(ValueError, match="Trace not found"):
            tracer.end_span("invalid-trace", "invalid-span", "success")

    def test_error_handling_invalid_span_id(self: "TestRequestTracer") -> None:
        """Тест обработки ошибок с неверным span_id."""
        tracer = RequestTracer()
        # Создаем трейс
        tracer.start_trace("test-trace", "valid-span", "test_operation")
        # Пытаемся завершить несуществующий span
        with pytest.raises(ValueError, match="Span not found"):
            tracer.end_span("test-trace", "invalid-span", "success")

    def test_concurrent_tracing(self: "TestRequestTracer") -> None:
        """Тест конкурентного трейсинга."""
        tracer = RequestTracer()

        def create_trace(trace_id: str, span_id: str) -> Any:
            span = tracer.start_trace(trace_id, span_id, "test_operation")
            time.sleep(0.01)  # Имитируем работу
            tracer.end_span(trace_id, span_id, "success")

        # Создаем несколько трейсов одновременно
        import threading

        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_trace, args=(f"trace-{i}", f"span-{i}"))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все трейсы созданы
        assert len(tracer.spans) == 10

    def test_performance_tracing(self: "TestRequestTracer") -> None:
        """Тест производительности трейсинга."""
        tracer = RequestTracer()
        import time

        start_time = time.time()
        # Создаем много трейсов
        for i in range(1000):
            trace_id = f"perf-trace-{i}"
            span_id = f"perf-span-{i}"
            span = tracer.start_trace(trace_id, span_id, f"operation{i}")
            span.add_tag("test_tag", f"value{i}")
            tracer.end_span(trace_id, span_id, "success")
        end_time = time.time()
        duration = end_time - start_time
        # Создание 1000 трейсов должно занимать менее 1 секунды
        assert duration < 1.0
        assert len(tracer.spans) == 1000


class TestGetTracer:
    """Тесты для функции get_tracer."""

    def test_get_tracer_default(self: "TestGetTracer") -> None:
        """Тест получения трейсера по умолчанию."""
        tracer = get_tracer()
        assert isinstance(tracer, RequestTracer)
        assert tracer.name == "default"

    def test_get_tracer_custom_name(self: "TestGetTracer") -> None:
        """Тест получения трейсера с пользовательским именем."""
        tracer = get_tracer("custom_tracer")
        assert isinstance(tracer, RequestTracer)
        assert tracer.name == "custom_tracer"

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


class TestLogFunctions:
    """Тесты для функций логирования."""

    @patch("infrastructure.monitoring.logging_tracing.get_tracer")
    def test_log_trace(self, mock_get_tracer) -> None:
        """Тест функции log_trace."""
        mock_tracer = Mock()
        mock_get_tracer.return_value = mock_tracer
        log_trace("test-trace", "test-span", "test_operation", "Test message")
        mock_tracer.start_trace.assert_called_once_with("test-trace", "test-span", "test_operation")
        mock_tracer.end_span.assert_called_once()

    @patch("infrastructure.monitoring.logging_tracing.get_logger")
    def test_log_debug(self, mock_get_logger) -> None:
        """Тест функции log_debug."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        log_debug("Debug message", {"test": "data"})
        mock_logger.debug.assert_called_once_with("Debug message", metadata={"test": "data"})

    @patch("infrastructure.monitoring.logging_tracing.get_logger")
    def test_log_info(self, mock_get_logger) -> None:
        """Тест функции log_info."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        log_info("Info message", {"test": "data"})
        mock_logger.info.assert_called_once_with("Info message", metadata={"test": "data"})

    @patch("infrastructure.monitoring.logging_tracing.get_logger")
    def test_log_warning(self, mock_get_logger) -> None:
        """Тест функции log_warning."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        log_warning("Warning message", {"test": "data"})
        mock_logger.warning.assert_called_once_with("Warning message", metadata={"test": "data"})

    @patch("infrastructure.monitoring.logging_tracing.get_logger")
    def test_log_error(self, mock_get_logger) -> None:
        """Тест функции log_error."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        error = ValueError("Test error")
        log_error("Error message", error, {"test": "data"})
        mock_logger.error.assert_called_once_with("Error message", exception=error, metadata={"test": "data"})

    @patch("infrastructure.monitoring.logging_tracing.get_logger")
    def test_log_critical(self, mock_get_logger) -> None:
        """Тест функции log_critical."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        error = ValueError("Critical error")
        log_critical("Critical message", error, {"test": "data"})
        mock_logger.critical.assert_called_once_with("Critical message", exception=error, metadata={"test": "data"})


if __name__ == "__main__":
    pytest.main([__file__])
