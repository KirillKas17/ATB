"""
Улучшенная система логирования для ATB.
Объединенная версия с поддержкой всех слоев архитектуры.
"""

import json
import logging
import logging.handlers
import sys
import time
from abc import ABC
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, Union, cast

from loguru import logger as loguru_logger

# Type variables
F = TypeVar("F", bound=Callable[..., Union[Any, Awaitable[Any]]])
# Удаляем стандартный обработчик loguru
loguru_logger.remove()


class StructuredFormatter(logging.Formatter):
    """Форматтер для структурированных логов."""

    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога в JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Добавляем дополнительные поля если они есть
        if hasattr(record, "context"):
            log_entry["context"] = getattr(record, "context")
        if hasattr(record, "error_code"):
            log_entry["error_code"] = getattr(record, "error_code")
        if hasattr(record, "traceback"):
            log_entry["traceback"] = getattr(record, "traceback")
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False)


class ColoredFormatter(logging.Formatter):
    """Форматтер с цветным выводом для консоли."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        formatted = super().format(record)
        return f"{color}{formatted}{reset}"


class TradingFilter(logging.Filter):
    """Фильтр для торговых логов."""

    def __init__(self, name: str = ""):
        super().__init__(name)
        self.trading_keywords = [
            "order",
            "trade",
            "position",
            "balance",
            "portfolio",
            "strategy",
            "signal",
            "risk",
            "profit",
            "loss",
        ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Фильтрация записей по торговым ключевым словам."""
        message = record.getMessage().lower()
        return any(keyword in message for keyword in self.trading_keywords)


class PerformanceFilter(logging.Filter):
    """Фильтр для логов производительности."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Фильтрация записей по производительности."""
        message = record.getMessage().lower()
        performance_keywords = ["performance", "latency", "throughput", "metrics"]
        return any(keyword in message for keyword in performance_keywords)


class LoggerManager:
    """Менеджер логгеров."""

    def __init__(self) -> None:
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def get_logger(self, name: str, level: str = "INFO") -> logging.Logger:
        """Получение или создание логгера."""
        if name in self.loggers:
            return self.loggers[name]
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        # Очищаем существующие обработчики
        logger.handlers.clear()
        # Добавляем обработчики
        self._add_console_handler(logger)
        self._add_file_handlers(logger, name)
        self.loggers[name] = logger
        return logger

    def _add_console_handler(self, logger: logging.Logger) -> None:
        """Добавление обработчика консоли."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = ColoredFormatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    def _add_file_handlers(self, logger: logging.Logger, name: str) -> None:
        """Добавление файловых обработчиков."""
        # Основной лог
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        main_handler.setLevel(logging.DEBUG)
        main_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
        )
        main_handler.setFormatter(main_formatter)
        logger.addHandler(main_handler)
        # Структурированный лог
        structured_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{name}_structured.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        structured_handler.setLevel(logging.DEBUG)
        structured_handler.setFormatter(StructuredFormatter())
        logger.addHandler(structured_handler)
        # Торговый лог
        trading_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        trading_handler.setLevel(logging.INFO)
        trading_handler.setFormatter(main_formatter)
        trading_handler.addFilter(TradingFilter())
        logger.addHandler(trading_handler)
        # Лог ошибок
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log", maxBytes=5 * 1024 * 1024, backupCount=3  # 5MB
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(main_formatter)
        logger.addHandler(error_handler)
        # Лог производительности
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(main_formatter)
        perf_handler.addFilter(PerformanceFilter())
        logger.addHandler(perf_handler)

    def set_level(self, name: str, level: str) -> None:
        """Установка уровня логирования для логгера."""
        if name in self.loggers:
            self.loggers[name].setLevel(getattr(logging, level.upper()))

    def get_log_stats(self) -> Dict[str, Any]:
        """Получение статистики логов."""
        stats = {}
        for name, logger in self.loggers.items():
            stats[name] = {
                "level": logging.getLevelName(logger.level),
                "handlers": len(logger.handlers),
                "propagate": logger.propagate,
            }
        return stats


# Глобальный менеджер логгеров
_logger_manager = LoggerManager()


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Получение логгера.
    Args:
        name: Имя логгера
        level: Уровень логирования
    Returns:
        Настроенный логгер
    """
    return _logger_manager.get_logger(name, level)


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Настройка системы логирования.
    Args:
        config: Конфигурация логирования
    """
    if config is None:
        config = {}
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.get("level", "INFO").upper()))
    # Очищаем существующие обработчики
    root_logger.handlers.clear()
    # Добавляем обработчик консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    # Настройка loguru для совместимости
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    # Добавляем вывод в консоль через loguru
    loguru_logger.add(
        sys.stdout,
        format=loguru_format,
        level=config.get("level", "INFO"),
        colorize=True,
    )
    # Добавляем вывод в файл через loguru
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    loguru_logger.add(
        log_dir / "trading_bot.log",
        format=loguru_format,
        level=config.get("level", "INFO"),
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )


def log_execution_time(func: F) -> F:
    """Декоратор для логирования времени выполнения."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.info(f"Function {func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.3f}s: {e}"
            )
            raise

    return cast(F, wrapper)


def async_log_execution_time(func: F) -> F:
    """Декоратор для логирования времени выполнения асинхронных функций."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.info(
                f"Async function {func.__name__} executed in {execution_time:.3f}s"
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger = get_logger(func.__module__)
            logger.error(
                f"Async function {func.__name__} failed after {execution_time:.3f}s: {e}"
            )
            raise

    return cast(F, wrapper)


def log_trade(trade_data: Dict[str, Any]) -> None:
    """Логирование торговой операции."""
    logger = get_logger("trading")
    logger.info(
        f"TRADE: {trade_data.get('side', 'UNKNOWN').upper()} "
        f"{trade_data.get('quantity', 0)} {trade_data.get('symbol', 'UNKNOWN')} "
        f"@ {trade_data.get('price', 0)} | "
        f"Strategy: {trade_data.get('strategy', 'unknown')} | "
        f"Time: {trade_data.get('timestamp', datetime.now().isoformat())}"
    )


def log_order(order_data: Dict[str, Any]) -> None:
    """Логирование ордера."""
    logger = get_logger("trading")
    logger.info(
        f"ORDER: {order_data.get('side', 'UNKNOWN').upper()} "
        f"{order_data.get('quantity', 0)} {order_data.get('symbol', 'UNKNOWN')} "
        f"@ {order_data.get('price', 0)} | "
        f"Status: {order_data.get('status', 'unknown')} | "
        f"Time: {order_data.get('timestamp', datetime.now().isoformat())}"
    )


def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Логирование ошибки."""
    logger = get_logger("errors")
    error_msg = f"ERROR: {type(error).__name__}: {str(error)}"
    if context:
        error_msg = f"{error_msg} | Context: {context}"
    logger.error(error_msg)
    logger.exception(error)


def log_performance(metric: str, value: float, unit: str = "") -> None:
    """Логирование метрик производительности."""
    logger = get_logger("performance")
    logger.info(f"PERFORMANCE: {metric} = {value}{unit}")


def setup_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """
    Настройка логгера (совместимость с infrastructure/core/logging.py).
    Args:
        name: Имя логгера
        level: Уровень логирования
    Returns:
        Настроенный логгер
    """
    if name is None:
        name = "trading_bot"
    setup_logging()
    return get_logger(name, level)


# Специализированные функции для инфраструктурного слоя
def log_trade_infrastructure(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    timestamp: str,
    strategy: str = "unknown",
) -> None:
    """
    Логирование торговой операции (инфраструктурный слой).
    Args:
        symbol: Символ торговой пары
        side: Сторона сделки (buy/sell)
        quantity: Количество
        price: Цена
        timestamp: Временная метка
        strategy: Название стратегии
    """
    loguru_logger.info(
        f"TRADE: {side.upper()} {quantity} {symbol} @ {price} | "
        f"Strategy: {strategy} | Time: {timestamp}"
    )


def log_error_infrastructure(
    error: Exception,
    context: str = "",
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Логирование ошибки (инфраструктурный слой).
    Args:
        error: Исключение
        context: Контекст ошибки
        additional_info: Дополнительная информация
    """
    error_msg = f"ERROR: {type(error).__name__}: {str(error)}"
    if context:
        error_msg = f"{context} | {error_msg}"
    if additional_info:
        error_msg = f"{error_msg} | Additional info: {additional_info}"
    loguru_logger.error(error_msg)
    loguru_logger.exception(error)


def log_performance_infrastructure(
    operation: str,
    duration: float,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Логирование производительности (инфраструктурный слой).
    Args:
        operation: Название операции
        duration: Длительность в секундах
        success: Успешность операции
        details: Детали операции
    """
    status = "SUCCESS" if success else "FAILED"
    msg = f"PERFORMANCE: {operation} | Duration: {duration:.3f}s | Status: {status}"
    if details:
        msg = f"{msg} | Details: {details}"
    if success:
        loguru_logger.info(msg)
    else:
        loguru_logger.warning(msg)


def log_market_data_infrastructure(
    symbol: str, price: float, volume: float, timestamp: str, source: str = "unknown"
) -> None:
    """
    Логирование рыночных данных (инфраструктурный слой).
    Args:
        symbol: Символ торговой пары
        price: Цена
        volume: Объем
        timestamp: Временная метка
        source: Источник данных
    """
    loguru_logger.debug(
        f"MARKET_DATA: {symbol} | Price: {price} | Volume: {volume} | "
        f"Source: {source} | Time: {timestamp}"
    )


def log_strategy_signal_infrastructure(
    strategy: str, symbol: str, signal_type: str, strength: float, timestamp: str
) -> None:
    """
    Логирование сигнала стратегии (инфраструктурный слой).
    Args:
        strategy: Название стратегии
        symbol: Символ торговой пары
        signal_type: Тип сигнала
        strength: Сила сигнала
        timestamp: Временная метка
    """
    loguru_logger.info(
        f"STRATEGY_SIGNAL: {strategy} | {symbol} | {signal_type} | "
        f"Strength: {strength:.3f} | Time: {timestamp}"
    )


def log_portfolio_update_infrastructure(
    portfolio_id: str, total_value: float, pnl: float, timestamp: str
) -> None:
    """
    Логирование обновления портфеля (инфраструктурный слой).
    Args:
        portfolio_id: ID портфеля
        total_value: Общая стоимость
        pnl: Прибыль/убыток
        timestamp: Временная метка
    """
    loguru_logger.info(
        f"PORTFOLIO_UPDATE: {portfolio_id} | Total Value: {total_value:.2f} | "
        f"PnL: {pnl:.2f} | Time: {timestamp}"
    )


def log_risk_alert_infrastructure(
    alert_type: str,
    message: str,
    severity: str = "WARNING",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Логирование предупреждения о рисках (инфраструктурный слой).
    Args:
        alert_type: Тип предупреждения
        message: Сообщение
        severity: Серьезность
        details: Детали
    """
    msg = f"RISK_ALERT: {alert_type} | {message} | Severity: {severity}"
    if details:
        msg = f"{msg} | Details: {details}"
    if severity.upper() == "CRITICAL":
        loguru_logger.critical(msg)
    elif severity.upper() == "ERROR":
        loguru_logger.error(msg)
    elif severity.upper() == "WARNING":
        loguru_logger.warning(msg)
    else:
        loguru_logger.info(msg)


def log_system_health_infrastructure(
    component: str,
    status: str,
    metrics: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> None:
    """
    Логирование состояния системы (инфраструктурный слой).
    Args:
        component: Компонент системы
        status: Статус
        metrics: Метрики
        timestamp: Временная метка
    """
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    msg = f"SYSTEM_HEALTH: {component} | Status: {status} | Time: {timestamp}"
    if metrics:
        msg = f"{msg} | Metrics: {metrics}"
    if status.upper() == "ERROR":
        loguru_logger.error(msg)
    elif status.upper() == "WARNING":
        loguru_logger.warning(msg)
    else:
        loguru_logger.info(msg)


class LoggerMixin(ABC):
    """Миксин для добавления логирования к классам."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._logger: Optional[logging.Logger] = None

    @property
    def logger(self) -> logging.Logger:
        """Получение логгера для класса."""
        if self._logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        return self._logger

    def log_info(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Логирование информационного сообщения."""
        self.logger.info(message, *args, **kwargs)

    def log_warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Логирование предупреждения."""
        self.logger.warning(message, *args, **kwargs)

    def log_error(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Логирование ошибки."""
        self.logger.error(message, *args, **kwargs)

    def log_debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Логирование отладочного сообщения."""
        self.logger.debug(message, *args, **kwargs)

    def log_critical(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Логирование критической ошибки."""
        self.logger.critical(message, *args, **kwargs)
