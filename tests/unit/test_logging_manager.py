"""
Unit тесты для LoggingManager.
Тестирует управление логированием, включая запись логов,
фильтрацию, ротацию и анализ логов.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import logging
import tempfile
import os
from datetime import datetime, timedelta
from infrastructure.core.logging_manager import LoggingManager
class TestLoggingManager:
    """Тесты для LoggingManager."""
    @pytest.fixture
    def logging_manager(self) -> LoggingManager:
        """Фикстура для LoggingManager."""
        # Создание временного файла для логов
        temp_log = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
        temp_log.close()
        log_manager = LoggingManager()
        log_manager.initialize_logging(temp_log.name)
        yield log_manager
        # Очистка после тестов
        log_manager.cleanup()
        if os.path.exists(temp_log.name):
            os.unlink(temp_log.name)
    @pytest.fixture
    def sample_log_entry(self) -> dict:
        """Фикстура с тестовой записью лога."""
        return {
            "level": "INFO",
            "message": "Test log message",
            "source": "test_module",
            "timestamp": datetime.now(),
            "data": {
                "user_id": "user_001",
                "action": "login",
                "ip_address": "192.168.1.1"
            },
            "metadata": {
                "session_id": "session_123",
                "request_id": "req_456"
            }
        }
    def test_initialization(self, logging_manager: LoggingManager) -> None:
        """Тест инициализации менеджера логирования."""
        assert logging_manager is not None
        assert hasattr(logging_manager, 'loggers')
        assert hasattr(logging_manager, 'log_handlers')
        assert hasattr(logging_manager, 'log_formatters')
    def test_initialize_logging(self, logging_manager: LoggingManager) -> None:
        """Тест инициализации логирования."""
        # Проверка, что логирование инициализировано
        assert logging_manager.loggers is not None
        assert len(logging_manager.loggers) > 0
        # Проверка, что можно записывать логи
        test_logger = logging_manager.get_logger("test_module")
        assert test_logger is not None
        assert isinstance(test_logger, logging.Logger)
    def test_log_message(self, logging_manager: LoggingManager, sample_log_entry: dict) -> None:
        """Тест записи сообщения в лог."""
        # Запись сообщения в лог
        log_result = logging_manager.log_message(
            sample_log_entry["level"],
            sample_log_entry["message"],
            sample_log_entry["source"],
            sample_log_entry["data"]
        )
        # Проверки
        assert log_result is not None
        assert "success" in log_result
        assert "log_id" in log_result
        assert "log_time" in log_result
        assert "log_level" in log_result
        # Проверка типов данных
        assert isinstance(log_result["success"], bool)
        assert isinstance(log_result["log_id"], str)
        assert isinstance(log_result["log_time"], datetime)
        assert isinstance(log_result["log_level"], str)
    def test_log_error(self, logging_manager: LoggingManager) -> None:
        """Тест записи ошибки в лог."""
        # Создание тестовой ошибки
        error_data = {
            "error_type": "ValueError",
            "error_message": "Test error message",
            "stack_trace": "Traceback (most recent call last):\n  File test.py, line 1",
            "context": {"function": "test_function", "line": 10}
        }
        # Запись ошибки в лог
        error_result = logging_manager.log_error(
            "test_module",
            error_data
        )
        # Проверки
        assert error_result is not None
        assert "success" in error_result
        assert "error_id" in error_result
        assert "error_time" in error_result
        assert "error_level" in error_result
        # Проверка типов данных
        assert isinstance(error_result["success"], bool)
        assert isinstance(error_result["error_id"], str)
        assert isinstance(error_result["error_time"], datetime)
        assert error_result["error_level"] == "ERROR"
    def test_log_warning(self, logging_manager: LoggingManager) -> None:
        """Тест записи предупреждения в лог."""
        # Запись предупреждения в лог
        warning_result = logging_manager.log_warning(
            "test_module",
            "Test warning message",
            {"warning_type": "deprecation", "component": "old_module"}
        )
        # Проверки
        assert warning_result is not None
        assert "success" in warning_result
        assert "warning_id" in warning_result
        assert "warning_time" in warning_result
        assert warning_result["log_level"] == "WARNING"
    def test_log_info(self, logging_manager: LoggingManager) -> None:
        """Тест записи информационного сообщения в лог."""
        # Запись информационного сообщения в лог
        info_result = logging_manager.log_info(
            "test_module",
            "Test info message",
            {"info_type": "status_update", "status": "running"}
        )
        # Проверки
        assert info_result is not None
        assert "success" in info_result
        assert info_result["log_level"] == "INFO"
    def test_log_debug(self, logging_manager: LoggingManager) -> None:
        """Тест записи отладочного сообщения в лог."""
        # Запись отладочного сообщения в лог
        debug_result = logging_manager.log_debug(
            "test_module",
            "Test debug message",
            {"debug_info": "variable_value", "step": "step_1"}
        )
        # Проверки
        assert debug_result is not None
        assert "success" in debug_result
        assert debug_result["log_level"] == "DEBUG"
    def test_get_logs(self, logging_manager: LoggingManager, sample_log_entry: dict) -> None:
        """Тест получения логов."""
        # Запись нескольких сообщений в лог
        logging_manager.log_message("INFO", "Message 1", "module1")
        logging_manager.log_message("WARNING", "Message 2", "module2")
        logging_manager.log_message("ERROR", "Message 3", "module3")
        # Получение логов
        logs_result = logging_manager.get_logs(
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        # Проверки
        assert logs_result is not None
        assert "logs" in logs_result
        assert "total_logs" in logs_result
        assert "log_summary" in logs_result
        # Проверка типов данных
        assert isinstance(logs_result["logs"], list)
        assert isinstance(logs_result["total_logs"], int)
        assert isinstance(logs_result["log_summary"], dict)
        # Проверка логики
        assert logs_result["total_logs"] >= 3
    def test_filter_logs(self, logging_manager: LoggingManager) -> None:
        """Тест фильтрации логов."""
        # Запись логов с разными уровнями
        logging_manager.log_message("INFO", "Info message", "module1")
        logging_manager.log_message("WARNING", "Warning message", "module2")
        logging_manager.log_message("ERROR", "Error message", "module3")
        # Фильтрация логов
        filtered_logs = logging_manager.filter_logs(
            filters={
                "level": "ERROR",
                "source": "module3"
            }
        )
        # Проверки
        assert filtered_logs is not None
        assert isinstance(filtered_logs, list)
        assert len(filtered_logs) >= 0
        # Проверка фильтрации
        for log in filtered_logs:
            assert log["level"] == "ERROR"
            assert log["source"] == "module3"
    def test_analyze_logs(self, logging_manager: LoggingManager) -> None:
        """Тест анализа логов."""
        # Запись различных логов
        for i in range(10):
            logging_manager.log_message("INFO", f"Info message {i}", "module1")
        for i in range(5):
            logging_manager.log_message("WARNING", f"Warning message {i}", "module2")
        for i in range(3):
            logging_manager.log_message("ERROR", f"Error message {i}", "module3")
        # Анализ логов
        analysis_result = logging_manager.analyze_logs()
        # Проверки
        assert analysis_result is not None
        assert "log_statistics" in analysis_result
        assert "error_analysis" in analysis_result
        assert "performance_metrics" in analysis_result
        assert "trend_analysis" in analysis_result
        # Проверка типов данных
        assert isinstance(analysis_result["log_statistics"], dict)
        assert isinstance(analysis_result["error_analysis"], dict)
        assert isinstance(analysis_result["performance_metrics"], dict)
        assert isinstance(analysis_result["trend_analysis"], dict)
    def test_rotate_logs(self, logging_manager: LoggingManager) -> None:
        """Тест ротации логов."""
        # Ротация логов
        rotation_result = logging_manager.rotate_logs()
        # Проверки
        assert rotation_result is not None
        assert "success" in rotation_result
        assert "rotated_files" in rotation_result
        assert "rotation_time" in rotation_result
        assert "space_freed" in rotation_result
        # Проверка типов данных
        assert isinstance(rotation_result["success"], bool)
        assert isinstance(rotation_result["rotated_files"], list)
        assert isinstance(rotation_result["rotation_time"], datetime)
        assert isinstance(rotation_result["space_freed"], int)
    def test_compress_logs(self, logging_manager: LoggingManager) -> None:
        """Тест сжатия логов."""
        # Сжатие логов
        compression_result = logging_manager.compress_logs()
        # Проверки
        assert compression_result is not None
        assert "success" in compression_result
        assert "compressed_files" in compression_result
        assert "compression_ratio" in compression_result
        assert "space_saved" in compression_result
        # Проверка типов данных
        assert isinstance(compression_result["success"], bool)
        assert isinstance(compression_result["compressed_files"], list)
        assert isinstance(compression_result["compression_ratio"], float)
        assert isinstance(compression_result["space_saved"], int)
        # Проверка диапазона
        assert 0.0 <= compression_result["compression_ratio"] <= 1.0
    def test_search_logs(self, logging_manager: LoggingManager) -> None:
        """Тест поиска в логах."""
        # Запись логов с ключевыми словами
        logging_manager.log_message("INFO", "User login successful", "auth_module")
        logging_manager.log_message("ERROR", "Database connection failed", "db_module")
        logging_manager.log_message("WARNING", "High memory usage detected", "monitor_module")
        # Поиск в логах
        search_result = logging_manager.search_logs("login")
        # Проверки
        assert search_result is not None
        assert "search_results" in search_result
        assert "total_matches" in search_result
        assert "search_time" in search_result
        # Проверка типов данных
        assert isinstance(search_result["search_results"], list)
        assert isinstance(search_result["total_matches"], int)
        assert isinstance(search_result["search_time"], float)
        # Проверка результатов поиска
        assert search_result["total_matches"] >= 0
    def test_get_log_statistics(self, logging_manager: LoggingManager) -> None:
        """Тест получения статистики логов."""
        # Получение статистики
        statistics = logging_manager.get_log_statistics()
        # Проверки
        assert statistics is not None
        assert "total_logs" in statistics
        assert "logs_by_level" in statistics
        assert "logs_by_source" in statistics
        assert "logs_by_hour" in statistics
        assert "error_rate" in statistics
        # Проверка типов данных
        assert isinstance(statistics["total_logs"], int)
        assert isinstance(statistics["logs_by_level"], dict)
        assert isinstance(statistics["logs_by_source"], dict)
        assert isinstance(statistics["logs_by_hour"], dict)
        assert isinstance(statistics["error_rate"], float)
        # Проверка диапазона
        assert 0.0 <= statistics["error_rate"] <= 1.0
    def test_export_logs(self, logging_manager: LoggingManager) -> None:
        """Тест экспорта логов."""
        # Запись тестовых логов
        logging_manager.log_message("INFO", "Export test message", "test_module")
        # Экспорт логов
        export_result = logging_manager.export_logs("test_export.json")
        # Проверки
        assert export_result is not None
        assert "success" in export_result
        assert "export_path" in export_result
        assert "export_format" in export_result
        assert "export_time" in export_result
        # Проверка типов данных
        assert isinstance(export_result["success"], bool)
        assert isinstance(export_result["export_path"], str)
        assert isinstance(export_result["export_format"], str)
        assert isinstance(export_result["export_time"], datetime)
        # Очистка
        if os.path.exists("test_export.json"):
            os.unlink("test_export.json")
    def test_set_log_level(self, logging_manager: LoggingManager) -> None:
        """Тест установки уровня логирования."""
        # Установка уровня логирования
        set_level_result = logging_manager.set_log_level("test_module", "DEBUG")
        # Проверки
        assert set_level_result is not None
        assert "success" in set_level_result
        assert "module" in set_level_result
        assert "new_level" in set_level_result
        assert "set_time" in set_level_result
        # Проверка типов данных
        assert isinstance(set_level_result["success"], bool)
        assert isinstance(set_level_result["module"], str)
        assert isinstance(set_level_result["new_level"], str)
        assert isinstance(set_level_result["set_time"], datetime)
    def test_error_handling(self, logging_manager: LoggingManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            logging_manager.log_message(None, None, None)
        with pytest.raises(ValueError):
            logging_manager.get_logs(None, None)
    def test_edge_cases(self, logging_manager: LoggingManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень длинным сообщением
        long_message = "x" * 10000
        log_result = logging_manager.log_message("INFO", long_message, "test_module")
        assert log_result["success"] is True
        # Тест с пустым сообщением
        empty_message = ""
        log_result = logging_manager.log_message("INFO", empty_message, "test_module")
        assert log_result["success"] is True
        # Тест с очень большими данными
        large_data = {"large_field": "x" * 1000}
        log_result = logging_manager.log_message("INFO", "Test", "test_module", large_data)
        assert log_result["success"] is True
    def test_cleanup(self, logging_manager: LoggingManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        logging_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert logging_manager.loggers == {}
        assert logging_manager.log_handlers == {}
        assert logging_manager.log_formatters == {} 
