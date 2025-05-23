import hashlib
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class CacheConfig:
    """Конфигурация кэширования"""

    # Параметры кэширования
    cache_dir: str = "cache"  # Директория для кэша
    max_size: int = 1000  # Максимальный размер кэша (МБ)
    ttl: int = 3600  # Время жизни кэша (сек)
    enabled: bool = True  # Включено ли кэширование

    # Параметры логирования
    log_dir: str = "logs"  # Директория для логов


class Cache:
    """Кэш для результатов анализа"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация кэша.

        Args:
            config: Словарь с параметрами кэширования
        """
        self.config = CacheConfig(**config) if config else CacheConfig()
        self._setup_logger()
        self._setup_cache()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def _setup_cache(self):
        """Настройка кэша"""
        try:
            # Создаем директорию для кэша
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)

            # Очищаем старый кэш
            self._cleanup_cache()

        except Exception as e:
            logger.error(f"Error setting up cache: {str(e)}")

    def _cleanup_cache(self):
        """Очистка старого кэша"""
        try:
            # Получаем список файлов кэша
            cache_files = list(Path(self.config.cache_dir).glob("*.pkl"))

            # Удаляем старые файлы
            current_time = datetime.now().timestamp()
            for file in cache_files:
                if current_time - file.stat().st_mtime > self.config.ttl:
                    file.unlink()

            # Проверяем размер кэша
            total_size = sum(file.stat().st_size for file in cache_files)
            if total_size > self.config.max_size * 1024 * 1024:
                # Сортируем файлы по времени модификации
                cache_files.sort(key=lambda x: x.stat().st_mtime)

                # Удаляем старые файлы
                for file in cache_files:
                    if total_size <= self.config.max_size * 1024 * 1024:
                        break
                    total_size -= file.stat().st_size
                    file.unlink()

        except Exception as e:
            logger.error(f"Error cleaning up cache: {str(e)}")

    def _generate_key(self, func: callable, *args, **kwargs) -> str:
        """
        Генерация ключа кэша.

        Args:
            func: Функция
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы

        Returns:
            str: Ключ кэша
        """
        try:
            # Создаем строку с аргументами
            key_parts = [func.__name__, str(args), str(sorted(kwargs.items()))]

            # Создаем хеш
            key = hashlib.md5("".join(key_parts).encode()).hexdigest()

            return key

        except Exception as e:
            logger.error(f"Error generating cache key: {str(e)}")
            return ""

    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша.

        Args:
            key: Ключ кэша

        Returns:
            Any: Значение из кэша или None
        """
        try:
            if not self.config.enabled:
                return None

            # Формируем путь к файлу кэша
            cache_file = Path(self.config.cache_dir) / f"{key}.pkl"

            # Проверяем существование файла
            if not cache_file.exists():
                return None

            # Проверяем время жизни
            if (
                datetime.now().timestamp() - cache_file.stat().st_mtime
                > self.config.ttl
            ):
                cache_file.unlink()
                return None

            # Читаем значение
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None

    def set(self, key: str, value: Any):
        """
        Сохранение значения в кэш.

        Args:
            key: Ключ кэша
            value: Значение для сохранения
        """
        try:
            if not self.config.enabled:
                return

            # Формируем путь к файлу кэша
            cache_file = Path(self.config.cache_dir) / f"{key}.pkl"

            # Сохраняем значение
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)

        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")

    def delete(self, key: str):
        """
        Удаление значения из кэша.

        Args:
            key: Ключ кэша
        """
        try:
            # Формируем путь к файлу кэша
            cache_file = Path(self.config.cache_dir) / f"{key}.pkl"

            # Удаляем файл
            if cache_file.exists():
                cache_file.unlink()

        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")

    def clear(self):
        """Очистка кэша"""
        try:
            # Получаем список файлов кэша
            cache_files = list(Path(self.config.cache_dir).glob("*.pkl"))

            # Удаляем файлы
            for file in cache_files:
                file.unlink()

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

    def cached(self, func: callable) -> callable:
        """
        Декоратор для кэширования результатов функции.

        Args:
            func: Функция для кэширования

        Returns:
            callable: Обернутая функция
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if not self.config.enabled:
                    return func(*args, **kwargs)

                # Генерируем ключ
                key = self._generate_key(func, *args, **kwargs)

                # Пробуем получить значение из кэша
                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value

                # Вычисляем значение
                value = func(*args, **kwargs)

                # Сохраняем в кэш
                self.set(key, value)

                return value

            except Exception as e:
                logger.error(f"Error in cached decorator: {str(e)}")
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        """Контекстный менеджер: вход"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход"""
        self._cleanup_cache()
