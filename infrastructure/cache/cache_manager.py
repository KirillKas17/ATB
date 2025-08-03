"""
Менеджер кэша для оптимизации производительности.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from loguru import logger

from .compression import CacheCompressor, CompressionType
from .encryption import CacheEncryptor, EncryptionType


@dataclass
class CacheEntry:
    """Запись в кэше."""

    value: Any
    timestamp: float
    ttl: float
    access_count: int = 0
    last_access: float = 0.0


class CacheManager:
    """Менеджер кэша с поддержкой TTL и статистики и сжатия."""

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float = 300.0,
        compression: bool = True,
        compression_type: CompressionType = CompressionType.LZ4,
        encryption: bool = True,
        encryption_type: EncryptionType = EncryptionType.AES_256,
    ):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "sets": 0}

        self.logger = logger.bind(service="CacheManager")
        self.compressor = CacheCompressor(
            compression_type=compression_type, enable_adaptive=compression
        )
        self.encryptor = (
            CacheEncryptor(encryption_type=encryption_type) if encryption else None
        )

        # Запускаем очистку устаревших записей
        self._start_cleanup_thread()

    def _start_cleanup_thread(self) -> None:
        """Запуск потока очистки."""

        def cleanup_worker() -> None:
            while True:
                try:
                    time.sleep(60)  # Проверяем каждую минуту
                    self._cleanup_expired()
                except Exception as e:
                    self.logger.error(f"Error in cleanup thread: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_expired(self) -> None:
        """Очистка устаревших записей."""
        with self.lock:
            current_time = time.time()
            expired_keys = []

            for key, entry in self.cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                self.stats["evictions"] += 1

            if expired_keys:
                self.logger.debug(
                    f"Cleaned up {len(expired_keys)} expired cache entries"
                )

    def _evict_lru(self) -> None:
        """Вытеснение наименее используемых записей."""
        if len(self.cache) < self.max_size:
            return

        # Находим запись с наименьшим количеством обращений
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.cache[k].access_count, self.cache[k].last_access),
        )

        del self.cache[lru_key]
        self.stats["evictions"] += 1
        self.logger.debug(f"Evicted LRU cache entry: {lru_key}")

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша с распаковкой."""
        with self.lock:
            if key not in self.cache:
                self.stats["misses"] += 1
                return None

            entry = self.cache[key]
            current_time = time.time()

            # Проверяем TTL
            if current_time - entry.timestamp > entry.ttl:
                del self.cache[key]
                self.stats["misses"] += 1
                self.stats["evictions"] += 1
                return None

            # Обновляем статистику
            entry.access_count += 1
            entry.last_access = current_time
            self.stats["hits"] += 1

            # Дешифрование, десериализация и распаковка
            try:
                # Сначала дешифруем, если включено шифрование
                if self.encryptor:
                    decrypted_value = self.encryptor.decrypt_value(key, entry.value)
                else:
                    decrypted_value = entry.value

                # Затем распаковываем
                return self.compressor.decompress_value(decrypted_value)
            except Exception as e:
                self.logger.error(f"Decompression/decryption error: {e}")
                return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Установка значения в кэш с сжатием."""
        with self.lock:
            # Проверяем размер кэша
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            current_time = time.time()

            # Сначала сжимаем
            try:
                compressed_value = self.compressor.compress_value(value)
            except Exception as e:
                self.logger.error(f"Compression error: {e}")
                compressed_value = value

            # Затем шифруем, если включено шифрование
            try:
                if self.encryptor:
                    final_value = self.encryptor.encrypt_value(key, compressed_value)
                else:
                    final_value = compressed_value
            except Exception as e:
                self.logger.error(f"Encryption error: {e}")
                final_value = compressed_value

            entry = CacheEntry(
                value=final_value,
                timestamp=current_time,
                ttl=ttl or self.default_ttl,
                access_count=0,
                last_access=current_time,
            )

            self.cache[key] = entry
            self.stats["sets"] += 1

    def delete(self, key: str) -> bool:
        """Удаление записи из кэша."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False

    def clear(self) -> None:
        """Очистка всего кэша."""
        with self.lock:
            self.cache.clear()
            self.logger.info("Cache cleared")

    def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        with self.lock:
            if key not in self.cache:
                return False

            entry = self.cache[key]
            current_time = time.time()

            # Проверяем TTL
            if current_time - entry.timestamp > entry.ttl:
                del self.cache[key]
                return False

            return True

    def get_or_set(
        self, key: str, factory: Callable[[], Any], ttl: Optional[float] = None
    ) -> Any:
        """Получение значения или создание через фабрику."""
        value = self.get(key)
        if value is not None:
            return value

        # Создаем новое значение
        value = factory()
        self.set(key, value, ttl)
        return value

    def get_multi(self, keys: list[str]) -> Dict[str, Any]:
        """Получение нескольких значений."""
        result = {}
        with self.lock:
            for key in keys:
                if key in self.cache:
                    entry = self.cache[key]
                    current_time = time.time()

                    # Проверяем TTL
                    if current_time - entry.timestamp > entry.ttl:
                        del self.cache[key]
                        self.stats["misses"] += 1
                        self.stats["evictions"] += 1
                        continue

                    # Обновляем статистику
                    entry.access_count += 1
                    entry.last_access = current_time
                    self.stats["hits"] += 1

                    result[key] = entry.value
                else:
                    self.stats["misses"] += 1

        return result

    def set_multi(self, data: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """Установка нескольких значений."""
        with self.lock:
            for key, value in data.items():
                # Проверяем размер кэша
                if len(self.cache) >= self.max_size:
                    self._evict_lru()

                current_time = time.time()
                entry = CacheEntry(
                    value=value,
                    timestamp=current_time,
                    ttl=ttl or self.default_ttl,
                    access_count=0,
                    last_access=current_time,
                )

                self.cache[key] = entry
                self.stats["sets"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (
                (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            )

            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "sets": self.stats["sets"],
                "hit_rate": hit_rate,
                "utilization": (
                    (len(self.cache) / self.max_size * 100) if self.max_size > 0 else 0
                ),
            }

    def get_keys(self) -> list[str]:
        """Получение всех ключей в кэше."""
        with self.lock:
            return list(self.cache.keys())

    def get_size(self) -> int:
        """Получение текущего размера кэша."""
        with self.lock:
            return len(self.cache)

    def is_full(self) -> bool:
        """Проверка заполненности кэша."""
        with self.lock:
            return len(self.cache) >= self.max_size

    def reset_stats(self) -> None:
        """Сброс статистики."""
        with self.lock:
            self.stats = {"hits": 0, "misses": 0, "evictions": 0, "sets": 0}

    def __len__(self) -> int:
        """Размер кэша."""
        return self.get_size()

    def __contains__(self, key: str) -> bool:
        """Проверка наличия ключа."""
        return self.exists(key)
