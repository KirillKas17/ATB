import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class Logger:
    """Класс для логирования."""

    _instance: Optional["Logger"] = None

    def __new__(cls, config: Optional[dict] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(config)
        return cls._instance

    def _initialize(self, config: Optional[dict] = None) -> None:
        """Инициализация логгера."""
        self.config = (
            config
            if config is not None
            else {
                "level": "INFO",
                "file": "trading_bot.log",
                "max_size": 10485760,  # 10MB
                "backup_count": 5,
            }
        )

        # Создание директории для логов
        log_dir = os.path.dirname(self.config["file"])
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Настройка логгера
        self.logger = logging.getLogger("trading_bot")
        self.logger.setLevel(getattr(logging, self.config["level"]))

        # Форматтер
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Хендлер для файла
        file_handler = RotatingFileHandler(
            self.config["file"],
            maxBytes=self.config["max_size"],
            backupCount=self.config["backup_count"],
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Хендлер для консоли
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def debug(self, message: str) -> None:
        """Логирование отладочного сообщения."""
        self.logger.debug(message)

    def info(self, message: str) -> None:
        """Логирование информационного сообщения."""
        self.logger.info(message)

    def warning(self, message: str) -> None:
        """Логирование предупреждения."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Логирование ошибки."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Логирование критической ошибки."""
        self.logger.critical(message)

    def log_trade(self, trade_data: dict) -> None:
        """Логирование информации о сделке."""
        try:
            message = (
                f"TRADE: {trade_data['pair']} | "
                f"Type: {trade_data['type']} | "
                f"Price: {trade_data['price']} | "
                f"Size: {trade_data['size']} | "
                f"P/L: {trade_data.get('pnl', 'N/A')}"
            )
            self.info(message)

        except Exception as e:
            self.error(f"Error logging trade: {e}")

    def log_signal(self, signal_data: dict) -> None:
        """Логирование информации о сигнале."""
        try:
            message = (
                f"SIGNAL: {signal_data['pair']} | "
                f"Type: {signal_data['type']} | "
                f"Confidence: {signal_data['confidence']} | "
                f"Source: {signal_data['source']}"
            )
            self.info(message)

        except Exception as e:
            self.error(f"Error logging signal: {e}")

    def log_error(self, error_data: dict) -> None:
        """Логирование информации об ошибке."""
        try:
            message = (
                f"ERROR: {error_data['type']} | "
                f"Message: {error_data['message']} | "
                f"Context: {error_data.get('context', 'N/A')}"
            )
            self.error(message)

        except Exception as e:
            self.error(f"Error logging error: {e}")

    def log_performance(self, performance_data: dict) -> None:
        """Логирование информации о производительности."""
        try:
            message = (
                f"PERFORMANCE: {performance_data['metric']} | "
                f"Value: {performance_data['value']} | "
                f"Time: {performance_data.get('time', datetime.now())}"
            )
            self.info(message)

        except Exception as e:
            self.error(f"Error logging performance: {e}")

    def set_level(self, level: str) -> None:
        """Установка уровня логирования."""
        try:
            self.logger.setLevel(getattr(logging, level))
            self.config["level"] = level

        except Exception as e:
            self.error(f"Error setting log level: {e}")

    def get_level(self) -> str:
        """Получение текущего уровня логирования."""
        return self.config["level"]

    def clear_logs(self) -> None:
        """Очистка логов."""
        try:
            # Закрытие всех хендлеров
            for handler in self.logger.handlers[:]:
                handler.close()
                self.logger.removeHandler(handler)

            # Удаление файла лога
            if os.path.exists(self.config["file"]):
                os.remove(self.config["file"])

            # Переинициализация логгера
            self._initialize(self.config)

        except Exception as e:
            self.error(f"Error clearing logs: {e}")

    def get_log_file_path(self) -> str:
        """Получение пути к файлу лога."""
        return self.config["file"]

    def get_log_size(self) -> int:
        """Получение размера файла лога."""
        try:
            if os.path.exists(self.config["file"]):
                return os.path.getsize(self.config["file"])
            return 0

        except Exception as e:
            self.error(f"Error getting log size: {e}")
            return 0
