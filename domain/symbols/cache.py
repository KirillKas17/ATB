# -*- coding: utf-8 -*-
"""Кэширование для модуля symbols."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, cast
import logging

logger = logging.getLogger(__name__)

from domain.types import (
    MarketPhaseResult,
    OpportunityScoreResult,
    SymbolProfileCache,
    SymbolProfileProtocol,
    ValidationError,
)

from .symbol_profile import SymbolProfile


@dataclass
class CacheEntry:
    """Запись в кэше."""

    data: Any
    timestamp: datetime
    ttl: int  # Time to live в секундах

    def is_expired(self) -> bool:
        """Проверка истечения срока действия."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)


class MemorySymbolCache:
    """Кэш профилей символов в памяти."""

    def __init__(self, default_ttl: int = 300):  # 5 минут по умолчанию
        """Инициализация кэша."""
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.logger = logger.bind(name=self.__class__.__name__)

    def get_profile(self, symbol: str) -> Optional[SymbolProfileProtocol]:
        """Получение профиля из кэша."""
        try:
            if symbol not in self.cache:
                return None
            entry = self.cache[symbol]
            if entry.is_expired():
                del self.cache[symbol]
                return None
            profile = entry.data
            if isinstance(profile, SymbolProfileProtocol):
                return profile
            return None
        except Exception as e:
            self.logger.error(f"Error getting profile from cache: {e}")
            return None

    def set_profile(
        self, symbol: str, profile: SymbolProfileProtocol, ttl: Optional[int] = None
    ) -> None:
        """Сохранение профиля в кэш."""
        try:
            if not isinstance(profile, SymbolProfile):
                raise ValidationError("Profile must be a SymbolProfile instance")
            cache_ttl = ttl or self.default_ttl
            entry = CacheEntry(data=profile, timestamp=datetime.now(), ttl=cache_ttl)
            self.cache[symbol] = entry
        except Exception as e:
            self.logger.error(f"Error setting profile in cache: {e}")

    def invalidate_profile(self, symbol: str) -> None:
        """Инвалидация профиля."""
        try:
            if symbol in self.cache:
                del self.cache[symbol]
        except Exception as e:
            self.logger.error(f"Error invalidating profile: {e}")

    def clear_cache(self) -> None:
        """Очистка кэша."""
        try:
            self.cache.clear()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша."""
        try:
            total_entries = len(self.cache)
            expired_entries = sum(
                1 for entry in self.cache.values() if entry.is_expired()
            )
            valid_entries = total_entries - expired_entries
            return {
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
                "cache_size_mb": self._estimate_cache_size(),
                "hit_rate": self._calculate_hit_rate(),
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    def cleanup_expired(self) -> int:
        """Очистка истекших записей."""
        try:
            expired_keys = [
                key for key, entry in self.cache.items() if entry.is_expired()
            ]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)
        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")
            return 0

    def _estimate_cache_size(self) -> float:
        """Оценка размера кэша в МБ."""
        try:
            # Простая оценка размера объекта
            total_size = 0
            for entry in self.cache.values():
                # Примерная оценка размера SymbolProfile
                total_size += 1024  # 1KB на профиль
            return total_size / (1024 * 1024)  # Конвертация в МБ
        except Exception:
            return 0.0

    def _calculate_hit_rate(self) -> float:
        """Расчет hit rate кэша."""
        # В реальной реализации здесь нужно отслеживать hits/misses
        return 0.0


class MarketPhaseCache:
    """Кэш результатов классификации фаз рынка."""

    def __init__(self, default_ttl: int = 60):  # 1 минута по умолчанию
        """Инициализация кэша."""
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.logger = logger.bind(name=self.__class__.__name__)

    def get_phase_result(
        self, symbol: str, data_hash: str
    ) -> Optional[MarketPhaseResult]:
        """Получение результата классификации фазы."""
        try:
            cache_key = f"{symbol}_{data_hash}"
            if cache_key not in self.cache:
                return None
            entry = self.cache[cache_key]
            if entry.is_expired():
                del self.cache[cache_key]
                return None
            result = entry.data
            if isinstance(result, dict) and "phase" in result and "confidence" in result:
                return cast(MarketPhaseResult, result)
            return None
        except Exception as e:
            self.logger.error(f"Error getting phase result from cache: {e}")
            return None

    def set_phase_result(
        self,
        symbol: str,
        data_hash: str,
        result: MarketPhaseResult,
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение результата классификации фазы."""
        try:
            cache_ttl = ttl or self.default_ttl
            entry = CacheEntry(data=result, timestamp=datetime.now(), ttl=cache_ttl)
            cache_key = f"{symbol}_{data_hash}"
            self.cache[cache_key] = entry
        except Exception as e:
            self.logger.error(f"Error setting phase result in cache: {e}")

    def invalidate_symbol(self, symbol: str) -> None:
        """Инвалидация всех результатов для символа."""
        try:
            keys_to_remove = [
                key for key in self.cache.keys() if key.startswith(f"{symbol}_")
            ]
            for key in keys_to_remove:
                del self.cache[key]
        except Exception as e:
            self.logger.error(f"Error invalidating symbol results: {e}")

    def clear_cache(self) -> None:
        """Очистка кэша."""
        try:
            self.cache.clear()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")


class OpportunityScoreCache:
    """Кэш результатов расчета opportunity score."""

    def __init__(self, default_ttl: int = 120):  # 2 минуты по умолчанию
        """Инициализация кэша."""
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.logger = logger.bind(name=self.__class__.__name__)

    def get_score_result(
        self, symbol: str, data_hash: str
    ) -> Optional[OpportunityScoreResult]:
        """Получение результата расчета score."""
        try:
            cache_key = f"{symbol}_{data_hash}"
            if cache_key not in self.cache:
                return None
            entry = self.cache[cache_key]
            if entry.is_expired():
                del self.cache[cache_key]
                return None
            result = entry.data
            if isinstance(result, dict) and "symbol" in result and "total_score" in result:
                return cast(OpportunityScoreResult, result)
            return None
        except Exception as e:
            self.logger.error(f"Error getting score result from cache: {e}")
            return None

    def set_score_result(
        self,
        symbol: str,
        data_hash: str,
        result: OpportunityScoreResult,
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение результата расчета score."""
        try:
            cache_ttl = ttl or self.default_ttl
            entry = CacheEntry(data=result, timestamp=datetime.now(), ttl=cache_ttl)
            cache_key = f"{symbol}_{data_hash}"
            self.cache[cache_key] = entry
        except Exception as e:
            self.logger.error(f"Error setting score result in cache: {e}")

    def invalidate_symbol(self, symbol: str) -> None:
        """Инвалидация всех результатов для символа."""
        try:
            keys_to_remove = [
                key for key in self.cache.keys() if key.startswith(f"{symbol}_")
            ]
            for key in keys_to_remove:
                del self.cache[key]
        except Exception as e:
            self.logger.error(f"Error invalidating symbol results: {e}")

    def clear_cache(self) -> None:
        """Очистка кэша."""
        try:
            self.cache.clear()
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")


class SymbolCacheManager:
    """Менеджер кэширования для модуля symbols."""

    def __init__(self) -> None:
        """Инициализация менеджера кэширования."""
        self.profile_cache = MemorySymbolCache()
        self.phase_cache = MarketPhaseCache()
        self.score_cache = OpportunityScoreCache()
        self.logger = logger.bind(name=self.__class__.__name__)

    def get_profile(self, symbol: str) -> Optional[SymbolProfileProtocol]:
        """Получение профиля из кэша."""
        return self.profile_cache.get_profile(symbol)

    def set_profile(
        self, symbol: str, profile: SymbolProfileProtocol, ttl: Optional[int] = None
    ) -> None:
        """Сохранение профиля в кэш."""
        self.profile_cache.set_profile(symbol, profile, ttl)

    def get_phase_result(
        self, symbol: str, data_hash: str
    ) -> Optional[MarketPhaseResult]:
        """Получение результата классификации фазы."""
        return self.phase_cache.get_phase_result(symbol, data_hash)

    def set_phase_result(
        self,
        symbol: str,
        data_hash: str,
        result: MarketPhaseResult,
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение результата классификации фазы."""
        self.phase_cache.set_phase_result(symbol, data_hash, result, ttl)

    def get_score_result(
        self, symbol: str, data_hash: str
    ) -> Optional[OpportunityScoreResult]:
        """Получение результата расчета score."""
        return self.score_cache.get_score_result(symbol, data_hash)

    def set_score_result(
        self,
        symbol: str,
        data_hash: str,
        result: OpportunityScoreResult,
        ttl: Optional[int] = None,
    ) -> None:
        """Сохранение результата расчета score."""
        self.score_cache.set_score_result(symbol, data_hash, result, ttl)

    def invalidate_symbol(self, symbol: str) -> None:
        """Инвалидация всех данных для символа."""
        try:
            self.profile_cache.invalidate_profile(symbol)
            self.phase_cache.invalidate_symbol(symbol)
            self.score_cache.invalidate_symbol(symbol)
        except Exception as e:
            self.logger.error(f"Error invalidating symbol data: {e}")

    def clear_all_caches(self) -> None:
        """Очистка всех кэшей."""
        try:
            self.profile_cache.clear_cache()
            self.phase_cache.clear_cache()
            self.score_cache.clear_cache()
        except Exception as e:
            self.logger.error(f"Error clearing all caches: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Получение статистики всех кэшей."""
        try:
            return {
                "profile_cache": self.profile_cache.get_cache_stats(),
                "phase_cache": {
                    "total_entries": len(self.phase_cache.cache),
                    "expired_entries": sum(
                        1
                        for entry in self.phase_cache.cache.values()
                        if entry.is_expired()
                    ),
                },
                "score_cache": {
                    "total_entries": len(self.score_cache.cache),
                    "expired_entries": sum(
                        1
                        for entry in self.score_cache.cache.values()
                        if entry.is_expired()
                    ),
                },
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}

    def cleanup_expired(self) -> Dict[str, int]:
        """Очистка истекших записей во всех кэшах."""
        try:
            profile_cleaned = self.profile_cache.cleanup_expired()
            # Очистка фаз
            phase_cleaned = 0
            expired_keys = [
                key
                for key, entry in self.phase_cache.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.phase_cache.cache[key]
                phase_cleaned += 1
            # Очистка scores
            score_cleaned = 0
            expired_keys = [
                key
                for key, entry in self.score_cache.cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self.score_cache.cache[key]
                score_cleaned += 1
            return {
                "profile_cache": profile_cleaned,
                "phase_cache": phase_cleaned,
                "score_cache": score_cleaned,
            }
        except Exception as e:
            self.logger.error(f"Error cleaning up expired entries: {e}")
            return {"profile_cache": 0, "phase_cache": 0, "score_cache": 0}
