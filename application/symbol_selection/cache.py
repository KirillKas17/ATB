"""
Кэш и управление производительностью для выбора символов.
"""

import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from domain.symbols import SymbolProfile

from .types import DOASSConfig


class SymbolCache:
    """Класс кэша для выбора символов."""

    def __init__(self, config: DOASSConfig):
        self.config = config
        self.logger = logger.bind(name=self.__class__.__name__)

        # Кэш и состояние
        self._cache: Dict[str, Tuple[SymbolProfile, float]] = {}
        self._last_update: Optional[datetime] = None
        self._performance_metrics: Dict[str, Any] = {}

    def should_update(self) -> bool:
        """Проверка необходимости обновления."""
        if not self._last_update:
            return True

        time_since_update = (datetime.now() - self._last_update).total_seconds()
        return time_since_update >= self.config.update_interval_seconds

    def update_cache(self, profiles: Dict[str, SymbolProfile]) -> None:
        """Обновление кэша."""
        try:
            # Очищаем старые записи
            current_time = time.time()
            self._cache = {
                symbol: (profile, current_time)
                for symbol, (profile, timestamp) in self._cache.items()
                if current_time - timestamp < self.config.cache_ttl_seconds
            }

            # Добавляем новые записи
            current_time = time.time()
            for symbol, profile in profiles.items():
                self._cache[symbol] = (profile, current_time)

            # Ограничиваем размер кэша
            if len(self._cache) > self.config.max_cache_size:
                # Удаляем самые старые записи
                sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
                self._cache = dict(sorted_cache[-self.config.max_cache_size :])

            self._last_update = datetime.now()

        except Exception as e:
            self.logger.error(f"Error updating cache: {e}")

    def get_cached_profile(self, symbol: str) -> Optional[SymbolProfile]:
        """Получение профиля из кэша."""
        try:
            if symbol in self._cache:
                profile, timestamp = self._cache[symbol]
                current_time = time.time()

                if current_time - timestamp < self.config.cache_ttl_seconds:
                    return profile

            return None

        except Exception as e:
            self.logger.error(f"Error getting cached profile for {symbol}: {e}")
            return None

    def calculate_cache_hit_rate(self) -> float:
        """Расчет hit rate кэша."""
        try:
            if not self._cache:
                return 0.0

            current_time = time.time()
            valid_entries = sum(
                1
                for _, timestamp in self._cache.values()
                if current_time - timestamp < self.config.cache_ttl_seconds
            )

            return valid_entries / len(self._cache) if self._cache else 0.0

        except Exception:
            return 0.0

    def update_performance_metrics(
        self, processing_time: float, symbols_count: int
    ) -> None:
        """Обновление метрик производительности."""
        try:
            current_avg = self._performance_metrics.get("avg_processing_time", 0.0)
            current_count = self._performance_metrics.get("total_processing_count", 0)
            
            self._performance_metrics.update(
                {
                    "last_processing_time": processing_time,
                    "last_symbols_count": symbols_count,
                    "avg_processing_time": (current_avg + processing_time) / 2,
                    "total_processing_count": int(current_count) + 1,
                    "last_update_time": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        return self._performance_metrics.copy()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        try:
            current_time = time.time()
            valid_entries = sum(
                1
                for _, timestamp in self._cache.values()
                if current_time - timestamp < self.config.cache_ttl_seconds
            )

            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_entries,
                "hit_rate": self.calculate_cache_hit_rate(),
                "cache_size_mb": len(str(self._cache)) / (1024 * 1024),
            }

        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    def get_cached_profiles(self) -> Dict[str, SymbolProfile]:
        """Получение всех кэшированных профилей."""
        try:
            current_time = time.time()
            valid_profiles = {}
            for symbol, (profile, timestamp) in self._cache.items():
                if current_time - timestamp < self.config.cache_ttl_seconds:
                    valid_profiles[symbol] = profile
            return valid_profiles
        except Exception as e:
            self.logger.error(f"Error getting cached profiles: {e}")
            return {}

    def get_hit_rate(self) -> float:
        """Получение hit rate кэша."""
        return self.calculate_cache_hit_rate()

    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        return self.get_cache_stats()
