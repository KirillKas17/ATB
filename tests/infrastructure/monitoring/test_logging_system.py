"""
Unit тесты для модуля logging_system.
Тестирует:
- StructuredLogger
- Функции логирования
- Конфигурацию логирования
- Обработку ошибок
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import json
import tempfile
import os
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from infrastructure.monitoring.logging_system import (
    StructuredLogger,
    get_logger,
    setup_logging,
    LogLevel,
    LogContext,
    LogEntry
)
class TestStructuredLogger:
    """Тесты для StructuredLogger."""
    def test_init_default_config(self: "TestStructuredLogger") -> None:
        """Тест инициализации с конфигурацией по умолчанию."""
        logger = StructuredLogger()
        assert logger.name == "default"
        assert logger.level == LogLevel.INFO
        assert logger.handlers == []
        assert logger.formatters == {}
        assert logger.filters == []
    def test_init_custom_config(self: "TestStructuredLogger") -> None:
        """Тест инициализации с пользовательской конфигурацией."""
        config = {
            "name": "test_logger",
            "level": "DEBUG",
            "handlers": ["console", "file"],
            "formatters": {"json": {"format": "json"}},
            "filters": ["error_filter"]
        }
        logger = StructuredLogger(config)
        assert logger.name == "test_logger"
        assert logger.level == LogLevel.DEBUG
        assert "console" in logger.handlers
        assert "file" in logger.handlers
        assert "json" in logger.formatters
    def test_add_handler(self: "TestStructuredLogger") -> None:
        """Тест добавления обработчика."""
        logger = StructuredLogger()
        # Mock handler
        handler = Mock()
        handler.name = "test_handler"
        logger.add_handler(handler)
        assert "test_handler" in logger.handlers
        assert handler in logger.handlers
    def test_add_formatter(self: "TestStructuredLogger") -> None:
        """Тест добавления форматтера."""
        logger = StructuredLogger()
        formatter_config = {"format": "json", "date_format": "%Y-%m-%d"}
        logger.add_formatter("json", formatter_config)
        assert "json" in logger.formatters
        assert logger.formatters["json"] == formatter_config
    def test_add_filter(self: "TestStructuredLogger") -> None:
        """Тест добавления фильтра."""
        logger = StructuredLogger()
        # Mock filter
        filter_func = Mock()
        filter_func.name = "test_filter"
        logger.add_filter(filter_func)
        assert "test_filter" in logger.filters
        assert filter_func in logger.filters
    def test_log_with_context(self: "TestStructuredLogger") -> None:
        """Тест логирования с контекстом."""
        logger = StructuredLogger()
        context = LogContext(
            request_id="test-123",
            user_id="user-456",
            session_id="session-789",
            correlation_id="corr-abc"
        )
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.INFO, "Test message", context=context)
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert call_args[0] == LogLevel.INFO
            assert call_args[1] == "Test message"
            assert call_args[2].request_id == "test-123"
    def test_log_with_metadata(self: "TestStructuredLogger") -> None:
        """Тест логирования с метаданными."""
        logger = StructuredLogger()
        metadata = {
            "component": "test_component",
            "operation": "test_operation",
            "duration": 1.5
        }
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.INFO, "Test message", metadata=metadata)
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert call_args[2].metadata["component"] == "test_component"
    def test_log_with_exception(self: "TestStructuredLogger") -> None:
        """Тест логирования с исключением."""
        logger = StructuredLogger()
        exception = ValueError("Test error")
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.ERROR, "Test error", exception=exception)
            mock_log.assert_called_once()
            call_args = mock_log.call_args[0]
            assert call_args[2].exception == exception
    def test_log_entry_creation(self: "TestStructuredLogger") -> None:
        """Тест создания записи лога."""
        logger = StructuredLogger()
        context = LogContext(request_id="test-123")
        metadata = {"test": "data"}
        exception = ValueError("Test error")
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.WARNING, "Test warning", context=context, metadata=metadata, exception=exception)
            mock_log.assert_called_once()
            log_entry = mock_log.call_args[0][2]
            assert isinstance(log_entry, LogEntry)
            assert log_entry.timestamp is not None
            assert log_entry.level == LogLevel.WARNING
            assert log_entry.message == "Test warning"
            assert log_entry.context == context
            assert log_entry.metadata == metadata
            assert log_entry.exception == exception
    def test_log_level_filtering(self: "TestStructuredLogger") -> None:
        """Тест фильтрации по уровню логирования."""
        logger = StructuredLogger({"level": "WARNING"})
        with patch.object(logger, '_log') as mock_log:
            # DEBUG должен быть отфильтрован
            logger.log(LogLevel.DEBUG, "Debug message")
            mock_log.assert_not_called()
            # INFO должен быть отфильтрован
            logger.log(LogLevel.INFO, "Info message")
            mock_log.assert_not_called()
            # WARNING должен пройти
            logger.log(LogLevel.WARNING, "Warning message")
            mock_log.assert_called_once()
    def test_handler_processing(self: "TestStructuredLogger") -> None:
        """Тест обработки обработчиками."""
        logger = StructuredLogger()
        # Mock handler
        handler = Mock()
        handler.name = "test_handler"
        handler.process = Mock()
        logger.add_handler(handler)
        context = LogContext(request_id="test-123")
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.INFO, "Test message", context=context)
            # Проверяем, что обработчик был вызван
            handler.process.assert_called_once()
    def test_formatter_application(self: "TestStructuredLogger") -> None:
        """Тест применения форматтеров."""
        logger = StructuredLogger()
        formatter_config = {"format": "json"}
        logger.add_formatter("json", formatter_config)
        context = LogContext(request_id="test-123")
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.INFO, "Test message", context=context, formatter="json")
            mock_log.assert_called_once()
            log_entry = mock_log.call_args[0][2]
            assert log_entry.formatter == "json"
    def test_filter_application(self: "TestStructuredLogger") -> None:
        """Тест применения фильтров."""
        logger = StructuredLogger()
        # Mock filter that returns False (should filter out)
        filter_func = Mock()
        filter_func.name = "test_filter"
        filter_func.should_log = Mock(return_value=False)
        logger.add_filter(filter_func)
        with patch.object(logger, '_log') as mock_log:
            logger.log(LogLevel.INFO, "Test message")
            # Сообщение должно быть отфильтровано
            mock_log.assert_not_called()
            filter_func.should_log.assert_called_once()
    def test_error_handling(self: "TestStructuredLogger") -> None:
        """Тест обработки ошибок в логировании."""
        logger = StructuredLogger()
        # Mock handler that raises exception
        handler = Mock()
        handler.name = "error_handler"
        handler.process = Mock(side_effect=Exception("Handler error"))
        logger.add_handler(handler)
        # Логирование не должно падать при ошибке обработчика
        try:
            logger.log(LogLevel.INFO, "Test message")
        except Exception:
            pytest.fail("Logger should handle handler errors gracefully")
    def test_performance_logging(self: "TestStructuredLogger") -> None:
        """Тест производительности логирования."""
        logger = StructuredLogger()
        import time
        start_time = time.time()
        for i in range(1000):
            logger.log(LogLevel.INFO, f"Test message {i}")
        end_time = time.time()
        duration = end_time - start_time
        # Логирование 1000 сообщений должно занимать менее 1 секунды
        assert duration < 1.0
    def test_memory_usage(self: "TestStructuredLogger") -> None:
        """Тест использования памяти."""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        logger = StructuredLogger()
        # Создаем много записей логов
        for i in range(10000):
            logger.log(LogLevel.INFO, f"Test message {i}")
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        # Увеличение памяти должно быть разумным (менее 100MB)
        assert memory_increase < 100 * 1024 * 1024
class TestGetLogger:
    """Тесты для функции get_logger."""
    def test_get_logger_default(self: "TestGetLogger") -> None:
        """Тест получения логгера по умолчанию."""
        logger = get_logger()
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "default"
    def test_get_logger_custom_name(self: "TestGetLogger") -> None:
        """Тест получения логгера с пользовательским именем."""
        logger = get_logger("custom_logger")
        assert isinstance(logger, StructuredLogger)
        assert logger.name == "custom_logger"
    def test_get_logger_singleton(self: "TestGetLogger") -> None:
        """Тест, что get_logger возвращает тот же экземпляр для одного имени."""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        assert logger1 is logger2
    def test_get_logger_different_names(self: "TestGetLogger") -> None:
        """Тест, что разные имена возвращают разные экземпляры."""
        logger1 = get_logger("logger1")
        logger2 = get_logger("logger2")
        assert logger1 is not logger2
class TestSetupLogging:
    """Тесты для функции setup_logging."""
    def test_setup_logging_default(self: "TestSetupLogging") -> None:
        """Тест настройки логирования по умолчанию."""
        with patch('infrastructure.monitoring.logging_system.StructuredLogger') as mock_logger:
            setup_logging()
            mock_logger.assert_called()
    def test_setup_logging_custom_config(self: "TestSetupLogging") -> None:
        """Тест настройки логирования с пользовательской конфигурацией."""
        config = {
            "level": "DEBUG",
            "handlers": ["console"],
            "formatters": {"json": {"format": "json"}}
        }
        with patch('infrastructure.monitoring.logging_system.StructuredLogger') as mock_logger:
            setup_logging(config)
            mock_logger.assert_called_with(config)
    def test_setup_logging_file_handler(self: "TestSetupLogging") -> None:
        """Тест настройки логирования с файловым обработчиком."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            config = {
                "handlers": ["file"],
                "file_config": {"path": temp_file.name}
            }
            try:
                with patch('infrastructure.monitoring.logging_system.StructuredLogger') as mock_logger:
                    setup_logging(config)
                    mock_logger.assert_called()
            finally:
                os.unlink(temp_file.name)
class TestLogLevel:
    """Тесты для enum LogLevel."""
    def test_log_level_values(self: "TestLogLevel") -> None:
        """Тест значений уровней логирования."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"
    def test_log_level_comparison(self: "TestLogLevel") -> None:
        """Тест сравнения уровней логирования."""
        assert LogLevel.DEBUG < LogLevel.INFO
        assert LogLevel.INFO < LogLevel.WARNING
        assert LogLevel.WARNING < LogLevel.ERROR
        assert LogLevel.ERROR < LogLevel.CRITICAL
    def test_log_level_from_string(self: "TestLogLevel") -> None:
        """Тест создания LogLevel из строки."""
        assert LogLevel.from_string("DEBUG") == LogLevel.DEBUG
        assert LogLevel.from_string("INFO") == LogLevel.INFO
        assert LogLevel.from_string("WARNING") == LogLevel.WARNING
        assert LogLevel.from_string("ERROR") == LogLevel.ERROR
        assert LogLevel.from_string("CRITICAL") == LogLevel.CRITICAL
    def test_log_level_invalid_string(self: "TestLogLevel") -> None:
        """Тест обработки неверной строки для LogLevel."""
        with pytest.raises(ValueError):
            LogLevel.from_string("INVALID")
class TestLogContext:
    """Тесты для LogContext."""
    def test_log_context_init(self: "TestLogContext") -> None:
        """Тест инициализации LogContext."""
        context = LogContext(
            request_id="req-123",
            user_id="user-456",
            session_id="session-789",
            correlation_id="corr-abc"
        )
        assert context.request_id == "req-123"
        assert context.user_id == "user-456"
        assert context.session_id == "session-789"
        assert context.correlation_id == "corr-abc"
    def test_log_context_to_dict(self: "TestLogContext") -> None:
        """Тест преобразования LogContext в словарь."""
        context = LogContext(
            request_id="req-123",
            user_id="user-456"
        )
        context_dict = context.to_dict()
        assert context_dict["request_id"] == "req-123"
        assert context_dict["user_id"] == "user-456"
        assert "timestamp" in context_dict
    def test_log_context_from_dict(self: "TestLogContext") -> None:
        """Тест создания LogContext из словаря."""
        context_dict = {
            "request_id": "req-123",
            "user_id": "user-456",
            "session_id": "session-789"
        }
        context = LogContext.from_dict(context_dict)
        assert context.request_id == "req-123"
        assert context.user_id == "user-456"
        assert context.session_id == "session-789"
class TestLogEntry:
    """Тесты для LogEntry."""
    def test_log_entry_init(self: "TestLogEntry") -> None:
        """Тест инициализации LogEntry."""
        context = LogContext(request_id="req-123")
        metadata = {"test": "data"}
        exception = ValueError("Test error")
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            context=context,
            metadata=metadata,
            exception=exception,
            formatter="json"
        )
        assert entry.level == LogLevel.INFO
        assert entry.message == "Test message"
        assert entry.context == context
        assert entry.metadata == metadata
        assert entry.exception == exception
        assert entry.formatter == "json"
    def test_log_entry_to_dict(self: "TestLogEntry") -> None:
        """Тест преобразования LogEntry в словарь."""
        context = LogContext(request_id="req-123")
        metadata = {"test": "data"}
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            context=context,
            metadata=metadata
        )
        entry_dict = entry.to_dict()
        assert entry_dict["level"] == "INFO"
        assert entry_dict["message"] == "Test message"
        assert "timestamp" in entry_dict
        assert "context" in entry_dict
        assert "metadata" in entry_dict
    def test_log_entry_json_serialization(self: "TestLogEntry") -> None:
        """Тест JSON сериализации LogEntry."""
        context = LogContext(request_id="req-123")
        metadata = {"test": "data"}
        entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            context=context,
            metadata=metadata
        )
        entry_dict = entry.to_dict()
        # Должно сериализоваться в JSON без ошибок
        json_str = json.dumps(entry_dict)
        assert isinstance(json_str, str)
        assert "Test message" in json_str
class TestLoggerProtocol:
    """Тесты для протокола LoggerProtocol."""
    def test_structured_logger_implements_protocol(self: "TestLoggerProtocol") -> None:
        """Тест, что StructuredLogger реализует LoggerProtocol."""
        logger = StructuredLogger()
        # Проверяем наличие всех методов протокола
        assert hasattr(logger, 'log')
        assert hasattr(logger, 'debug')
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'critical')
        assert hasattr(logger, 'add_handler')
        assert hasattr(logger, 'add_formatter')
        assert hasattr(logger, 'add_filter')
    def test_logger_methods(self: "TestLoggerProtocol") -> None:
        """Тест методов логирования."""
        logger = StructuredLogger()
        with patch.object(logger, 'log') as mock_log:
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            assert mock_log.call_count == 5
            assert mock_log.call_args_list[0][0][0] == LogLevel.DEBUG
            assert mock_log.call_args_list[1][0][0] == LogLevel.INFO
            assert mock_log.call_args_list[2][0][0] == LogLevel.WARNING
            assert mock_log.call_args_list[3][0][0] == LogLevel.ERROR
            assert mock_log.call_args_list[4][0][0] == LogLevel.CRITICAL
if __name__ == "__main__":
    pytest.main([__file__]) 
