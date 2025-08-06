"""
Умная система логирования для замены print() в аналитических модулях.
"""

import logging
import sys
from datetime import datetime
from typing import Any, Optional


class SmartAnalyticsLogger:
    """Умный логгер для аналитических модулей."""
    
    def __init__(self, module_name: str, level: int = logging.INFO) -> None:
        """
        Инициализация логгера.
        
        Args:
            module_name: Имя модуля
            level: Уровень логирования
        """
        self.logger = logging.getLogger(f"analytics.{module_name}")
        self.logger.setLevel(level)
        
        # Создаем обработчик если его нет
        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def analysis_start(self, message: str) -> None:
        """Начало анализа."""
        self.logger.info(f"🔍 [АНАЛИЗ] {message}")
    
    def analysis_progress(self, message: str, progress: Optional[float] = None) -> None:
        """Прогресс анализа."""
        if progress is not None:
            self.logger.info(f"⏳ [ПРОГРЕСС {progress:.1%}] {message}")
        else:
            self.logger.info(f"⏳ [ПРОГРЕСС] {message}")
    
    def analysis_result(self, message: str) -> None:
        """Результат анализа."""
        self.logger.info(f"📊 [РЕЗУЛЬТАТ] {message}")
    
    def analysis_complete(self, message: str) -> None:
        """Завершение анализа."""
        self.logger.info(f"✅ [ЗАВЕРШЕНО] {message}")
    
    def analysis_error(self, message: str, error: Optional[Exception] = None) -> None:
        """Ошибка анализа."""
        if error:
            self.logger.error(f"❌ [ОШИБКА] {message}: {error}")
        else:
            self.logger.error(f"❌ [ОШИБКА] {message}")
    
    def analysis_warning(self, message: str) -> None:
        """Предупреждение анализа."""
        self.logger.warning(f"⚠️ [ПРЕДУПРЕЖДЕНИЕ] {message}")
    
    def debug(self, message: str) -> None:
        """Отладочная информация."""
        self.logger.debug(f"🐛 [DEBUG] {message}")
    
    def metric(self, name: str, value: Any, unit: str = "") -> None:
        """Логирование метрики."""
        unit_str = f" {unit}" if unit else ""
        self.logger.info(f"📈 [МЕТРИКА] {name}: {value}{unit_str}")
    
    def statistics(self, stats: dict) -> None:
        """Логирование статистики."""
        self.logger.info("📋 [СТАТИСТИКА]")
        for key, value in stats.items():
            self.logger.info(f"  - {key}: {value}")
    
    def separator(self, title: str = "") -> None:
        """Разделитель в логах."""
        if title:
            self.logger.info(f"{'='*20} {title} {'='*20}")
        else:
            self.logger.info("="*60)


# Глобальные экземпляры для разных модулей
_loggers = {}


def get_analytics_logger(module_name: str) -> SmartAnalyticsLogger:
    """Получение логгера для модуля."""
    if module_name not in _loggers:
        _loggers[module_name] = SmartAnalyticsLogger(module_name)
    return _loggers[module_name]


def smart_print(*args: Any, module: str = "general", level: str = "info", **kwargs: Any) -> None:
    """
    Умная замена print() с автоматическим логированием.
    
    Args:
        *args: Аргументы для печати
        module: Имя модуля
        level: Уровень логирования
        **kwargs: Дополнительные аргументы
    """
    logger = get_analytics_logger(module)
    
    # Объединяем все аргументы в строку
    message = " ".join(str(arg) for arg in args)
    
    # Определяем тип сообщения по содержанию
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["ошибка", "error", "failed", "❌"]):
        logger.analysis_error(message)
    elif any(word in message_lower for word in ["предупреждение", "warning", "⚠️"]):
        logger.analysis_warning(message)
    elif any(word in message_lower for word in ["завершен", "complete", "готов", "✅"]):
        logger.analysis_complete(message)
    elif any(word in message_lower for word in ["анализ", "analysis", "🔍"]):
        logger.analysis_start(message)
    elif any(word in message_lower for word in ["результат", "result", "📊"]):
        logger.analysis_result(message)
    elif any(word in message_lower for word in ["прогресс", "progress", "⏳"]):
        logger.analysis_progress(message)
    else:
        # Обычное информационное сообщение
        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.analysis_warning(message)
        elif level == "error":
            logger.analysis_error(message)
        else:
            logger.logger.info(message)


# Удобные алиасы для быстрого импорта
analysis_logger = get_analytics_logger
smart_log = smart_print