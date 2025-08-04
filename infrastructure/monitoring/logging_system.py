"""
Расширенная система логирования с трейсингом и структурированными логами.
Включает:
- Структурированное логирование
- Трейсинг запросов
- Контекстное логирование
- Ротацию логов
- Агрегацию логов
"""

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional

from loguru import logger

from domain.types.monitoring_types import LogContext, LogEntry, LoggerProtocol, LogLevel


class StructuredLogger(LoggerProtocol):
    """
    Структурированный логгер с трейсингом.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация структурированного логгера.
        Args:
            config: Конфигурация логирования
        """
        self.config = config or {}
        # Контекст текущего потока
        self._thread_local = threading.local()
        # Обработчики логов
        self.handlers: List[Callable[[LogEntry], None]] = []
        # Статистика логирования
        self.stats: Dict[str, Any] = {
            "total_logs": 0,
            "logs_by_level": {level.value: 0 for level in LogLevel},
            "logs_by_component": {},
            "error_count": 0,
            "warning_count": 0,
        }
        # Настройка loguru
        self._setup_loguru()
        # Флаги состояния
        self.is_initialized = False

    def _setup_loguru(self) -> None:
        """Настройка loguru."""
        # Удаляем стандартные обработчики
        logger.remove()
        # Добавляем обработчик для консоли
        logger.add(
            lambda msg: self._handle_loguru_message(msg),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="DEBUG",
        )
        # Добавляем обработчик для файла
        log_file = self.config.get("log_file", "logs/app.log")
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            rotation="10 MB",
            retention="30 days",
            compression="gz",
        )

    def _handle_loguru_message(self, message: Any) -> None:
        """Обработка сообщений loguru."""
        # Преобразуем сообщение loguru в LogEntry
        log_entry = LogEntry(
            timestamp=datetime.now(),
            level=LogLevel(message.record["level"].name),
            message=message.record["message"],
            context=self.get_current_context(),
            exception=message.record.get("exception"),
            stack_trace=message.record.get("traceback"),
        )
        # Вызываем обработчики
        for handler in self.handlers:
            try:
                handler(log_entry)
            except Exception as e:
                print(f"Error in log handler: {e}")

    def add_handler(self, handler: Callable[[LogEntry], None]) -> None:
        """
        Добавление обработчика логов.
        Args:
            handler: Функция-обработчик
        """
        self.handlers.append(handler)

    def set_context(self, context: LogContext) -> None:
        """
        Установка контекста для текущего потока.
        Args:
            context: Контекст логирования
        """
        self._thread_local.context = context

    def get_current_context(self) -> LogContext:
        """Получение текущего контекста."""
        if not hasattr(self._thread_local, "context"):
            self._thread_local.context = LogContext()
        context = self._thread_local.context
        assert isinstance(context, LogContext)
        return context

    def update_context(self, **kwargs: Any) -> None:
        """
        Обновление контекста.
        Args:
            **kwargs: Поля для обновления
        """
        context = self.get_current_context()
        for key, value in kwargs.items():
            if hasattr(context, key):
                setattr(context, key, value)
            else:
                context.extra[key] = value

    @contextmanager
    def context(self, **kwargs: Any) -> Generator[None, None, None]:
        """
        Контекстный менеджер для временного контекста.
        Args:
            **kwargs: Поля контекста
        """
        old_context = self.get_current_context()
        new_context = LogContext(
            request_id=kwargs.get("request_id", old_context.request_id),
            user_id=kwargs.get("user_id", old_context.user_id),
            session_id=kwargs.get("session_id", old_context.session_id),
            component=kwargs.get("component", old_context.component),
            operation=kwargs.get("operation", old_context.operation),
            extra={**old_context.extra, **kwargs.get("extra", {})},
        )
        self.set_context(new_context)
        try:
            yield
        finally:
            self.set_context(old_context)

    def log(self, entry: LogEntry) -> None:
        """
        Логирование записи.
        Args:
            entry: Запись лога
        """
        self._log(entry.level, entry.message)

    def trace(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне TRACE."""
        self._log(LogLevel.TRACE, message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне DEBUG."""
        self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне INFO."""
        self._log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне WARNING."""
        self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне ERROR."""
        self._log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Логирование на уровне CRITICAL."""
        self._log(LogLevel.CRITICAL, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs: Any) -> None:
        """
        Внутренний метод логирования.
        Args:
            level: Уровень логирования
            message: Сообщение
            **kwargs: Дополнительные поля
        """
        context = self.get_current_context()
        # Обновляем контекст дополнительными полями
        if kwargs:
            context.extra.update(kwargs)
        # Создаем запись лога
        log_entry = LogEntry(
            timestamp=datetime.now(), level=level, message=message, context=context
        )
        # Обновляем статистику
        self._update_stats(log_entry)
        # Логируем через loguru
        loguru_level = level.value.lower()
        loguru_message = self._format_message(log_entry)
        if hasattr(logger, loguru_level):
            getattr(logger, loguru_level)(loguru_message)

    def _format_message(self, log_entry: LogEntry) -> str:
        """
        Форматирование сообщения лога.
        Args:
            log_entry: Запись лога
        Returns:
            Отформатированное сообщение
        """
        # Базовое сообщение
        parts = [log_entry.message]
        # Добавляем контекст
        context_dict = asdict(log_entry.context)
        if context_dict["extra"]:
            parts.append(f"extra={json.dumps(context_dict['extra'])}")
        # Добавляем идентификаторы
        if context_dict["request_id"]:
            parts.append(f"request_id={context_dict['request_id']}")
        if context_dict["user_id"]:
            parts.append(f"user_id={context_dict['user_id']}")
        if context_dict["component"]:
            parts.append(f"component={context_dict['component']}")
        if context_dict["operation"]:
            parts.append(f"operation={context_dict['operation']}")
        return " | ".join(parts)

    def _update_stats(self, log_entry: LogEntry) -> None:
        """
        Обновление статистики логирования.
        Args:
            log_entry: Запись лога
        """
        self.stats["total_logs"] += 1
        self.stats["logs_by_level"][log_entry.level.value] += 1
        # Статистика по компонентам
        component = log_entry.context.component or "unknown"
        if component not in self.stats["logs_by_component"]:
            self.stats["logs_by_component"][component] = 0
        self.stats["logs_by_component"][component] += 1
        # Счетчики ошибок и предупреждений
        if log_entry.level == LogLevel.ERROR:
            self.stats["error_count"] += 1
        elif log_entry.level == LogLevel.WARNING:
            self.stats["warning_count"] += 1

    @contextmanager
    def trace_operation(self, operation: str, **kwargs: Any) -> Generator[str, None, None]:
        """
        Трейсинг операции.
        Args:
            operation: Название операции
            **kwargs: Дополнительные параметры
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())
        with self.context(operation=operation, operation_id=operation_id, **kwargs):
            self.info(f"Starting operation: {operation}")
            try:
                yield operation_id
                duration = time.time() - start_time
                self.info(
                    f"Operation completed: {operation}",
                    duration=duration,
                    status="success",
                )
            except Exception as e:
                duration = time.time() - start_time
                self.error(
                    f"Operation failed: {operation}",
                    duration=duration,
                    status="error",
                    error=str(e),
                )
                raise

    def log_performance(self, operation: str, duration: float, **kwargs: Any) -> None:
        """
        Логирование производительности.
        Args:
            operation: Название операции
            duration: Время выполнения
            **kwargs: Дополнительные метрики
        """
        self.info(f"Performance: {operation}", duration=duration, **kwargs)

    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Логирование торговой операции.
        Args:
            trade_data: Данные сделки
        """
        self.info(
            "Trade executed",
            trade_id=trade_data.get("id"),
            symbol=trade_data.get("symbol"),
            side=trade_data.get("side"),
            volume=trade_data.get("volume"),
            price=trade_data.get("price"),
            pnl=trade_data.get("pnl"),
        )

    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        Логирование ошибки.
        Args:
            error: Исключение
            context: Контекст ошибки
        """
        self.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            context=context,
            stack_trace=self._get_stack_trace(error),
        )

    def _get_stack_trace(self, error: Exception) -> str:
        """
        Получение стека вызовов для ошибки.
        Args:
            error: Исключение
        Returns:
            Стек вызовов
        """
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики логирования."""
        return self.stats.copy()

    def export_logs(
        self, level: Optional[LogLevel] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Экспорт логов.
        Args:
            level: Фильтр по уровню
            limit: Лимит записей
        Returns:
            Список логов
        """
        # В реальной реализации здесь была бы логика чтения из файлов
        # или базы данных
        return []


# Глобальный экземпляр логгера
_global_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """Получение глобального логгера."""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger()
    return _global_logger


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Настройка глобального логирования.
    Args:
        config: Конфигурация логирования
    """
    global _global_logger
    _global_logger = StructuredLogger(config)

    # Добавляем обработчик для отправки в систему мониторинга
    def monitoring_handler(log_entry: LogEntry) -> None:
        # Здесь можно добавить отправку в систему мониторинга
        pass

    _global_logger.add_handler(monitoring_handler)
