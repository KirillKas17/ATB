# -*- coding: utf-8 -*-
"""Менеджер торговых сессий."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from pandas import DataFrame

from domain.type_definitions.session_types import (
    SessionAnalysisResult,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .interfaces import (
    BaseSessionAnalyzer,
    SessionConfigurationRepository,
    SessionDataRepository,
    SessionRegistry,
)
from .session_marker import SessionMarker


@dataclass
class SessionManagerConfig:
    """Конфигурация менеджера сессий."""

    # Основные параметры
    auto_analysis_enabled: bool = True
    analysis_interval_minutes: int = 5
    cache_ttl_minutes: int = 30
    # Параметры анализа
    min_data_points: int = 20
    confidence_threshold: float = 0.7
    volatility_threshold: float = 1.5
    # Параметры хранения
    storage_enabled: bool = True
    backup_enabled: bool = True
    compression_enabled: bool = False

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, Optional[str]]]:
        """Преобразование в словарь."""
        return {
            "auto_analysis_enabled": self.auto_analysis_enabled,
            "analysis_interval_minutes": self.analysis_interval_minutes,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "min_data_points": self.min_data_points,
            "confidence_threshold": self.confidence_threshold,
            "volatility_threshold": self.volatility_threshold,
            "storage_enabled": self.storage_enabled,
            "backup_enabled": self.backup_enabled,
            "compression_enabled": self.compression_enabled,
        }


@dataclass
class SessionManagerState:
    """Состояние менеджера сессий."""

    # Статистика работы
    total_analyses: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    # Временные метрики
    last_analysis_time: Optional[Timestamp] = None
    average_analysis_time_ms: float = 0.0
    # Кэш
    cache_hits: int = 0
    cache_misses: int = 0
    # Ошибки
    last_error: Optional[str] = None
    error_count: int = 0

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, Optional[str]]]:
        """Преобразование в словарь."""
        return {
            "total_analyses": self.total_analyses,
            "successful_analyses": self.successful_analyses,
            "failed_analyses": self.failed_analyses,
            "last_analysis_time": (
                self.last_analysis_time.to_iso() if self.last_analysis_time else None
            ),
            "average_analysis_time_ms": self.average_analysis_time_ms,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "last_error": self.last_error,
            "error_count": self.error_count,
        }

    def get_success_rate(self) -> float:
        """Получение процента успешных анализов."""
        if self.total_analyses == 0:
            return 0.0
        return self.successful_analyses / self.total_analyses

    def get_cache_hit_rate(self) -> float:
        """Получение процента попаданий в кэш."""
        total_cache_requests = self.cache_hits + self.cache_misses
        if total_cache_requests == 0:
            return 0.0
        return self.cache_hits / total_cache_requests


class SessionManager:
    """Менеджер торговых сессий."""

    def __init__(
        self,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        analyzer: BaseSessionAnalyzer,
        data_repository: Optional[SessionDataRepository] = None,
        config_repository: Optional[SessionConfigurationRepository] = None,
        config: Optional[SessionManagerConfig] = None,
    ) -> None:
        self.registry = registry
        self.session_marker = session_marker
        self.analyzer = analyzer
        self.data_repository = data_repository
        self.config_repository = config_repository
        self.config = config or SessionManagerConfig()
        # Состояние менеджера
        self.state = SessionManagerState()
        # Кэш результатов анализа
        self._analysis_cache: Dict[str, SessionAnalysisResult] = {}
        self._cache_timestamps: Dict[str, Timestamp] = {}
        logger.info("SessionManager initialized")

    def analyze_session(
        self,
        symbol: str,
        market_data: DataFrame,
        timestamp: Optional[Timestamp] = None,
        force_analysis: bool = False,
    ) -> Optional[SessionAnalysisResult]:
        """Анализ торговой сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Проверяем кэш
            cache_key = f"{symbol}_{timestamp.to_iso()[:10]}"
            if not force_analysis and self._is_cache_valid(cache_key, timestamp):
                self.state.cache_hits += 1
                return self._analysis_cache.get(cache_key)
            self.state.cache_misses += 1
            # Проверяем минимальные требования
            if len(market_data) < self.config.min_data_points:
                logger.warning(
                    f"Insufficient data points for {symbol}: "
                    f"{len(market_data)} < {self.config.min_data_points}"
                )
                return None
            # Выполняем анализ
            import time

            start_time = time.time()
            analysis_result = self.analyzer.analyze_session(
                symbol, market_data, timestamp
            )
            analysis_time_ms = (time.time() - start_time) * 1000
            # Обновляем статистику
            self.state.total_analyses += 1
            if analysis_result:
                self.state.successful_analyses += 1
                self._update_cache(cache_key, analysis_result, timestamp)
                # Сохраняем результат
                if self.config.storage_enabled and self.data_repository:
                    self._save_analysis_result(analysis_result)
            else:
                self.state.failed_analyses += 1
            # Обновляем временные метрики
            self.state.last_analysis_time = timestamp
            self._update_average_analysis_time(analysis_time_ms)
            logger.debug(
                f"Session analysis for {symbol} completed in {analysis_time_ms:.2f}ms"
            )
            return analysis_result
        except Exception as e:
            self.state.failed_analyses += 1
            self.state.last_error = str(e)
            self.state.error_count += 1
            logger.error(f"Error analyzing session for {symbol}: {e}")
            return None

    def get_session_context(
        self, timestamp: Optional[Timestamp] = None
    ) -> Dict[str, Union[str, float, int]]:
        """Получение контекста сессии."""
        if timestamp is None:
            timestamp = Timestamp.now()
        context = self.session_marker.get_session_context(timestamp).to_dict()
        # Приводим к ожидаемому типу с преобразованием
        filtered_context: Dict[str, Union[str, float, int]] = {}
        for k, v in context.items():
            if isinstance(v, (str, float, int)):
                filtered_context[k] = v
            elif v is not None:
                # Попытаемся преобразовать в строку если не None
                filtered_context[k] = str(v)
        return filtered_context

    def get_session_profile(
        self, session_type: SessionType
    ) -> Optional[SessionProfile]:
        """Получение профиля сессии."""
        try:
            # Сначала пытаемся получить из реестра
            profile = self.registry.get_profile(session_type)
            if profile:
                return profile
            # Если нет в реестре, пытаемся загрузить из репозитория
            if self.config_repository:
                profile = self.config_repository.get_session_profile(session_type)
                if profile:
                    # Сохраняем в реестр
                    self.registry.register_profile(profile)
                    return profile
            return None
        except Exception as e:
            logger.error(f"Error getting session profile for {session_type.value}: {e}")
            return None

    def update_session_profile(
        self,
        session_type: SessionType,
        updates: Dict[str, Union[str, float, int, bool]],
    ) -> bool:
        """Обновление профиля сессии."""
        try:
            # Обновляем в реестре
            success = self.registry.update_profile(session_type, updates)
            # Обновляем в репозитории
            if success and self.config_repository:
                # Создаем обновленный профиль
                current_profile = self.config_repository.get_session_profile(session_type)
                if current_profile:
                    # Обновляем профиль
                    for key, value in updates.items():
                        if hasattr(current_profile, key):
                            setattr(current_profile, key, value)
                    self.config_repository.save_session_profile(current_profile)
            if success:
                logger.info(f"Updated session profile for {session_type.value}")
            else:
                logger.warning(
                    f"Failed to update session profile for {session_type.value}"
                )
            return success
        except Exception as e:
            logger.error(
                f"Error updating session profile for {session_type.value}: {e}"
            )
            return False

    def get_session_statistics(
        self, session_type: SessionType, lookback_days: int = 30
    ) -> Dict[str, Union[str, float, int]]:
        """Получение статистики сессии."""
        try:
            if self.data_repository:
                stats = self.data_repository.get_session_statistics(
                    session_type, lookback_days
                )
                # Приводим к ожидаемому типу с преобразованием
                filtered_stats: Dict[str, Union[str, float, int]] = {}
                for k, v in stats.items():
                    if isinstance(v, (str, float, int)):
                        filtered_stats[k] = v
                    elif v is not None:
                        # Попытаемся преобразовать в строку если не None
                        filtered_stats[k] = str(v)
                return filtered_stats
            return {}
        except Exception as e:
            logger.error(
                f"Error getting session statistics for {session_type.value}: {e}"
            )
            return {}

    def get_manager_statistics(self) -> Dict[str, Union[str, float, int, bool, None, Dict[str, Union[str, float, int, bool, None]]]]:
        """Получение статистики менеджера."""
        try:
            stats = self.state.to_dict()
            config_dict = self.config.to_dict()
            # Приводим config к правильному типу
            typed_config: Dict[str, Union[str, float, int, bool, None]] = {}
            for k, v in config_dict.items():
                if isinstance(v, (str, float, int, bool)) or v is None:
                    typed_config[k] = v
            
            result: Dict[str, Union[str, float, int, bool, None, Dict[str, Union[str, float, int, bool, None]]]] = {}
            for k, v in stats.items():
                if isinstance(v, (str, float, int, bool)) or v is None:
                    result[k] = v
            
            # Добавляем метрики и конфигурацию отдельно
            result["success_rate"] = self.state.get_success_rate()
            result["cache_hit_rate"] = self.state.get_cache_hit_rate() 
            result["config"] = typed_config
            
            return result
        except Exception as e:
            logger.error(f"Error getting manager statistics: {e}")
            return {}

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._analysis_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Session analysis cache cleared")

    def backup_data(self) -> bool:
        """Резервное копирование данных."""
        try:
            if not self.config.backup_enabled:
                logger.info("Backup is disabled")
                return False
            if self.data_repository:
                # Здесь должна быть логика резервного копирования
                logger.info("Session data backup completed")
                return True
            return False
        except Exception as e:
            logger.error(f"Error during backup: {e}")
            return False

    def _is_cache_valid(self, cache_key: str, timestamp: Timestamp) -> bool:
        """Проверка валидности кэша."""
        if cache_key not in self._cache_timestamps:
            return False
        cache_time = self._cache_timestamps[cache_key]
        cache_age_minutes = (
            timestamp.to_datetime() - cache_time.to_datetime()
        ).total_seconds() / 60
        return cache_age_minutes < self.config.cache_ttl_minutes

    def _update_cache(
        self,
        cache_key: str,
        analysis_result: SessionAnalysisResult,
        timestamp: Timestamp,
    ) -> None:
        """Обновление кэша."""
        self._analysis_cache[cache_key] = analysis_result
        self._cache_timestamps[cache_key] = timestamp
        # Очищаем старые записи кэша
        self._cleanup_cache(timestamp)

    def _cleanup_cache(self, current_timestamp: Timestamp) -> None:
        """Очистка устаревших записей кэша."""
        expired_keys: List[str] = []
        for cache_key, cache_time in self._cache_timestamps.items():
            cache_age_minutes = (
                current_timestamp.to_datetime() - cache_time.to_datetime()
            ).total_seconds() / 60
            if cache_age_minutes >= self.config.cache_ttl_minutes:
                expired_keys.append(cache_key)
        for key in expired_keys:
            del self._analysis_cache[key]
            del self._cache_timestamps[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")

    def _save_analysis_result(self, analysis_result: SessionAnalysisResult) -> None:
        """Сохранение результата анализа."""
        try:
            if self.data_repository:
                self.data_repository.save_session_analysis(analysis_result)
        except Exception as e:
            logger.error(f"Error saving analysis result: {e}")

    def _update_average_analysis_time(self, new_time_ms: float) -> None:
        """Обновление среднего времени анализа."""
        if self.state.total_analyses == 1:
            self.state.average_analysis_time_ms = new_time_ms
        else:
            # Экспоненциальное скользящее среднее
            alpha = 0.1
            self.state.average_analysis_time_ms = (
                alpha * new_time_ms + (1 - alpha) * self.state.average_analysis_time_ms
            )
