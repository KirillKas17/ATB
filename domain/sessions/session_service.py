# -*- coding: utf-8 -*-
"""Сервис торговых сессий."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import pandas as pd
from loguru import logger

from domain.type_definitions.session_types import (
    SessionAnalysisResult,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp
from domain.sessions.session_predictor import SessionPrediction

from .interfaces import (
    BaseSessionAnalyzer,
    SessionConfigurationRepository,
    SessionDataRepository,
    SessionRegistry,
)
from .session_manager import SessionManager, SessionManagerConfig
from .session_optimizer import OptimizationResult, OptimizationTarget, SessionOptimizer
from .session_predictor import SessionPredictor
from .session_marker import SessionMarker
from .session_analyzer_factory import SessionAnalyzerFactory


@dataclass
class SessionServiceConfig:
    """Конфигурация сервиса сессий."""

    # Основные параметры
    auto_analysis_enabled: bool = True
    auto_optimization_enabled: bool = False
    auto_prediction_enabled: bool = True
    # Интервалы
    analysis_interval_minutes: int = 5
    optimization_interval_hours: int = 24
    prediction_interval_minutes: int = 15
    # Параметры кэширования
    cache_enabled: bool = True
    cache_ttl_minutes: int = 30
    # Параметры хранения
    storage_enabled: bool = True
    backup_enabled: bool = True
    # Параметры производительности
    max_concurrent_operations: int = 10
    timeout_seconds: int = 60
    # Параметры логирования
    log_level: str = "INFO"
    log_detailed_results: bool = False

    def to_dict(self) -> Dict[str, Union[str, float, int, bool]]:
        """Преобразование в словарь."""
        return {
            "auto_analysis_enabled": self.auto_analysis_enabled,
            "auto_optimization_enabled": self.auto_optimization_enabled,
            "auto_prediction_enabled": self.auto_prediction_enabled,
            "analysis_interval_minutes": self.analysis_interval_minutes,
            "optimization_interval_hours": self.optimization_interval_hours,
            "prediction_interval_minutes": self.prediction_interval_minutes,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "storage_enabled": self.storage_enabled,
            "backup_enabled": self.backup_enabled,
            "max_concurrent_operations": self.max_concurrent_operations,
            "timeout_seconds": self.timeout_seconds,
            "log_level": self.log_level,
            "log_detailed_results": self.log_detailed_results,
        }

    def validate(self) -> bool:
        """Валидация конфигурации."""
        if self.analysis_interval_minutes < 1:
            logger.error(
                f"Invalid analysis_interval_minutes: {self.analysis_interval_minutes}"
            )
            return False
        if self.optimization_interval_hours < 1:
            logger.error(
                f"Invalid optimization_interval_hours: {self.optimization_interval_hours}"
            )
            return False
        if self.prediction_interval_minutes < 1:
            logger.error(
                f"Invalid prediction_interval_minutes: {self.prediction_interval_minutes}"
            )
            return False
        if self.max_concurrent_operations < 1:
            logger.error(
                f"Invalid max_concurrent_operations: {self.max_concurrent_operations}"
            )
            return False
        if self.timeout_seconds < 1:
            logger.error(f"Invalid timeout_seconds: {self.timeout_seconds}")
            return False
        return True


@dataclass
class SessionServiceState:
    """Состояние сервиса сессий."""

    # Статистика операций
    total_analyses: int = 0
    total_optimizations: int = 0
    total_predictions: int = 0
    # Успешные операции
    successful_analyses: int = 0
    successful_optimizations: int = 0
    successful_predictions: int = 0
    # Временные метрики
    last_analysis_time: Optional[Timestamp] = None
    last_optimization_time: Optional[Timestamp] = None
    last_prediction_time: Optional[Timestamp] = None
    # Производительность
    average_analysis_time_ms: float = 0.0
    average_optimization_time_ms: float = 0.0
    average_prediction_time_ms: float = 0.0
    # Ошибки
    last_error: Optional[str] = None
    error_count: int = 0

    def to_dict(self) -> Dict[str, Union[str, float, int, bool]]:
        """Преобразование в словарь."""
        return {
            "total_analyses": self.total_analyses,
            "total_optimizations": self.total_optimizations,
            "total_predictions": self.total_predictions,
            "successful_analyses": self.successful_analyses,
            "successful_optimizations": self.successful_optimizations,
            "successful_predictions": self.successful_predictions,
            "last_analysis_time": self.last_analysis_time.to_iso() if self.last_analysis_time else "",
            "last_optimization_time": self.last_optimization_time.to_iso() if self.last_optimization_time else "",
            "last_prediction_time": self.last_prediction_time.to_iso() if self.last_prediction_time else "",
            "average_analysis_time_ms": self.average_analysis_time_ms,
            "average_optimization_time_ms": self.average_optimization_time_ms,
            "average_prediction_time_ms": self.average_prediction_time_ms,
            "last_error": self.last_error or "",
            "error_count": self.error_count,
        }

    def get_analysis_success_rate(self) -> float:
        """Получение процента успешных анализов."""
        if self.total_analyses == 0:
            return 0.0
        return self.successful_analyses / self.total_analyses

    def get_optimization_success_rate(self) -> float:
        """Получение процента успешных оптимизаций."""
        if self.total_optimizations == 0:
            return 0.0
        return self.successful_optimizations / self.total_optimizations

    def get_prediction_success_rate(self) -> float:
        """Получение процента успешных прогнозов."""
        if self.total_predictions == 0:
            return 0.0
        return self.successful_predictions / self.total_predictions


class SessionService:
    """Сервис торговых сессий."""

    def __init__(
        self,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        analyzer_factory: SessionAnalyzerFactory,
        data_repository: Optional[SessionDataRepository] = None,
        config_repository: Optional[SessionConfigurationRepository] = None,
        config: Optional[SessionServiceConfig] = None,
    ) -> None:
        self.registry = registry
        self.session_marker = session_marker
        self.analyzer_factory = analyzer_factory
        self.data_repository = data_repository
        self.config_repository = config_repository
        self.config = config or SessionServiceConfig()
        # Валидируем конфигурацию
        if not self.config.validate():
            raise ValueError("Invalid session service configuration")
        # Состояние сервиса
        self.state = SessionServiceState()
        # Компоненты сервиса
        self.session_manager: Optional[SessionManager] = None
        self.session_optimizer: Optional[SessionOptimizer] = None
        self.session_predictor: Optional[SessionPredictor] = None
        # Инициализируем компоненты
        self._initialize_components()
        logger.info("SessionService initialized")

    def analyze_session(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_type: Optional[SessionType] = None,
        timestamp: Optional[Timestamp] = None,
        analyzer_name: Optional[str] = None,
        force_analysis: bool = False,
    ) -> Optional[SessionAnalysisResult]:
        """Анализ торговой сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Определяем тип сессии, если не указан
            if session_type is None:
                session_context = self.session_marker.get_session_context(timestamp)
                if session_context.primary_session:
                    session_type = session_context.primary_session.session_type
                else:
                    logger.warning(
                        f"No active session found for {symbol} at {timestamp}"
                    )
                    return None
            # Выбираем анализатор
            if analyzer_name is None:
                analyzer_name = self._select_analyzer_for_session_type(session_type)
            # Получаем анализатор
            analyzer = self.analyzer_factory.get_analyzer(analyzer_name)
            if analyzer is None:
                analyzer = self.analyzer_factory.create_analyzer(
                    analyzer_name, self.registry, self.session_marker
                )
            if analyzer is None:
                logger.error(f"Failed to create analyzer: {analyzer_name}")
                return None
            # Выполняем анализ
            import time

            start_time = time.time()
            analysis_result = analyzer.analyze_session(symbol, market_data, timestamp)
            analysis_time_ms = (time.time() - start_time) * 1000
            # Обновляем статистику
            self.state.total_analyses += 1
            if analysis_result:
                self.state.successful_analyses += 1
                self._update_average_analysis_time(analysis_time_ms)
                # Сохраняем результат
                if self.config.storage_enabled and self.data_repository:
                    self._save_analysis_result(analysis_result)
            # Обновляем временные метрики
            self.state.last_analysis_time = timestamp
            if self.config.log_detailed_results:
                logger.info(
                    f"Session analysis for {symbol} ({session_type.value}) - "
                    f"Analyzer: {analyzer_name}, "
                    f"Time: {analysis_time_ms:.2f}ms, "
                    f"Success: {analysis_result is not None}"
                )
            return analysis_result
        except Exception as e:
            self.state.last_error = str(e)
            self.state.error_count += 1
            logger.error(f"Error analyzing session for {symbol}: {e}")
            return None

    def optimize_session_profile(
        self,
        session_type: SessionType,
        target: OptimizationTarget,
        historical_data: Optional[pd.DataFrame] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[OptimizationResult]:
        """Оптимизация профиля сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Инициализируем оптимизатор, если нужно
            if self.session_optimizer is None:
                self.session_optimizer = SessionOptimizer(
                    self.registry, self.session_marker
                )
            # Выполняем оптимизацию
            import time

            start_time = time.time()
            optimization_result = self.session_optimizer.optimize_session_profile(
                session_type, target, historical_data, timestamp
            )
            optimization_time_ms = (time.time() - start_time) * 1000
            # Обновляем статистику
            self.state.total_optimizations += 1
            if optimization_result:
                self.state.successful_optimizations += 1
                self._update_average_optimization_time(optimization_time_ms)
                # Применяем оптимизированный профиль
                if self.config_repository:
                    self.config_repository.save_session_profile(
                        optimization_result.optimized_profile
                    )
            else:
                self.state.last_error = "Optimization failed"
                self.state.error_count += 1
            # Обновляем временные метрики
            self.state.last_optimization_time = timestamp
            if self.config.log_detailed_results:
                logger.info(
                    f"Session profile optimization for {session_type.value} - "
                    f"Time: {optimization_time_ms:.2f}ms, "
                    f"Success: {optimization_result is not None}"
                )
            return optimization_result
        except Exception as error_msg:
            self.state.last_error = str(error_msg)
            self.state.error_count += 1
            logger.error(
                f"Error optimizing session profile for {session_type.value}: {error_msg}"
            )
            return None

    def predict_session(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_type: SessionType,
        timestamp: Optional[Timestamp] = None,
        prediction_horizon_minutes: int = 60,
    ) -> Optional[SessionPrediction]:
        """Прогнозирование торговой сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Инициализируем предиктор, если нужно
            if self.session_predictor is None:
                self.session_predictor = SessionPredictor(
                    self.registry, self.session_marker
                )
            # Выполняем прогнозирование
            import time

            start_time = time.time()
            prediction = self.session_predictor.predict_session(
                symbol, market_data, session_type, timestamp, prediction_horizon_minutes
            )
            prediction_time_ms = (time.time() - start_time) * 1000
            # Обновляем статистику
            self.state.total_predictions += 1
            if prediction:
                self.state.successful_predictions += 1
                self._update_average_prediction_time(prediction_time_ms)
            else:
                self.state.last_error = "Prediction failed"
                self.state.error_count += 1
            # Обновляем временные метрики
            self.state.last_prediction_time = timestamp
            if self.config.log_detailed_results:
                logger.info(
                    f"Session prediction for {symbol} ({session_type.value}) - "
                    f"Time: {prediction_time_ms:.2f}ms, "
                    f"Success: {prediction is not None}"
                )
            return prediction
        except Exception as error_msg:
            self.state.last_error = str(error_msg)
            self.state.error_count += 1
            logger.error(f"Error predicting session for {symbol}: {error_msg}")
            return None

    def get_session_context(
        self, timestamp: Optional[Timestamp] = None
    ) -> Dict[str, Union[str, float, int]]:
        """Получение контекста сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            context = self.session_marker.get_session_context(timestamp)
            primary_session = getattr(context, 'primary_session', None)
            secondary_sessions = getattr(context, 'secondary_sessions', [])
            session_phase = getattr(context, 'session_phase', None)
            time_until_next = getattr(context, 'time_until_next_session', 0.0)
            current_session_duration = getattr(context, 'current_session_duration', 0.0)
            session_progress = getattr(context, 'session_progress', 0.0)
            return {
                "primary_session": primary_session.session_type.value if primary_session and hasattr(primary_session, 'session_type') else "unknown",
                "secondary_sessions": str([s.session_type.value for s in secondary_sessions if hasattr(s, 'session_type')]),
                "session_phase": session_phase.value if session_phase and hasattr(session_phase, 'value') else "unknown",
                "time_until_next": float(time_until_next),
                "current_session_duration": float(current_session_duration),
                "session_progress": float(session_progress),
            }
        except Exception as error_msg:
            logger.error(f"Error getting session context: {error_msg}")
            return {
                "primary_session": "unknown",
                "secondary_sessions": "[]",
                "session_phase": "unknown",
                "time_until_next": 0.0,
                "current_session_duration": 0.0,
                "session_progress": 0.0,
            }

    def get_session_profile(
        self, session_type: SessionType
    ) -> Optional[SessionProfile]:
        """Получение профиля сессии."""
        try:
            if self.config_repository:
                return self.config_repository.get_session_profile(session_type)
            return None
        except Exception as error_msg:
            logger.error(f"Error getting session profile for {session_type.value}: {error_msg}")
            return None

    def update_session_profile(
        self,
        session_type: SessionType,
        updates: Dict[str, Union[str, float, int, bool]],
    ) -> bool:
        """Обновление профиля сессии."""
        try:
            if self.config_repository is None:
                logger.error("Configuration repository not available")
                return False
            # Получаем текущий профиль
            current_profile = self.config_repository.get_session_profile(session_type)
            if current_profile is None:
                logger.error(f"Session profile not found for {session_type.value}")
                return False
            # Применяем обновления
            for key, value in updates.items():
                if hasattr(current_profile, key):
                    setattr(current_profile, key, value)
                else:
                    logger.warning(f"Unknown field in session profile: {key}")
            # Сохраняем обновленный профиль
            self.config_repository.save_session_profile(current_profile)
            logger.info(f"Session profile updated for {session_type.value}")
            return True
        except Exception as error_msg:
            logger.error(f"Error updating session profile for {session_type.value}: {error_msg}")
            return False

    def get_session_statistics(
        self, session_type: SessionType, lookback_days: int = 30
    ) -> Dict[str, Union[str, float, int]]:
        """Получение статистики сессии."""
        try:
            if self.data_repository is None:
                return {}
            # Получаем статистику из репозитория
            stats = self.data_repository.get_session_statistics(
                session_type, lookback_days
            )
            # Преобразуем в правильный формат
            return {
                "total_sessions": stats.get("total_sessions", 0),
                "successful_sessions": stats.get("successful_sessions", 0),
                "avg_session_duration": stats.get("avg_session_duration", 0.0),
                "avg_volume": stats.get("avg_volume", 0.0),
                "avg_volatility": stats.get("avg_volatility", 0.0),
                "success_rate": stats.get("success_rate", 0.0),
            }
        except Exception as error_msg:
            logger.error(f"Error getting session statistics for {session_type.value}: {error_msg}")
            return {}

    def get_service_statistics(self) -> Dict[str, Union[str, float, int, bool, None]]:
        """Получение статистики сервиса."""
        try:
            return {
                "total_analyses": self.state.total_analyses,
                "successful_analyses": self.state.successful_analyses,
                "analysis_success_rate": self.state.get_analysis_success_rate(),
                "total_optimizations": self.state.total_optimizations,
                "successful_optimizations": self.state.successful_optimizations,
                "optimization_success_rate": self.state.get_optimization_success_rate(),
                "total_predictions": self.state.total_predictions,
                "successful_predictions": self.state.successful_predictions,
                "prediction_success_rate": self.state.get_prediction_success_rate(),
                "average_analysis_time_ms": self.state.average_analysis_time_ms,
                "average_optimization_time_ms": self.state.average_optimization_time_ms,
                "average_prediction_time_ms": self.state.average_prediction_time_ms,
                "last_error": self.state.last_error,
                "error_count": self.state.error_count,
                "last_analysis_time": (
                    self.state.last_analysis_time.to_iso() if self.state.last_analysis_time else None
                ),
                "last_optimization_time": (
                    self.state.last_optimization_time.to_iso()
                    if self.state.last_optimization_time
                    else None
                ),
                "last_prediction_time": (
                    self.state.last_prediction_time.to_iso()
                    if self.state.last_prediction_time
                    else None
                ),
            }
        except Exception as error_msg:
            logger.error(f"Error getting service statistics: {error_msg}")
            return {}

    def backup_data(self) -> bool:
        """Резервное копирование данных."""
        try:
            if not self.config.backup_enabled:
                logger.info("Backup is disabled in configuration")
                return True
            if self.data_repository is None:
                logger.warning("Data repository not available for backup")
                return False
            # Выполняем резервное копирование, если метод существует
            if hasattr(self.data_repository, 'backup_data'):
                backup_result = self.data_repository.backup_data()
            else:
                logger.error("Data repository does not support backup_data method")
                return False
            if backup_result:
                logger.info("Data backup completed successfully")
            else:
                logger.error("Data backup failed")
            return bool(backup_result)
        except Exception as error_msg:
            logger.error(f"Error during data backup: {error_msg}")
            return False

    def clear_cache(self) -> None:
        """Очистка кэша."""
        try:
            if self.data_repository and hasattr(self.data_repository, 'clear_cache'):
                self.data_repository.clear_cache()
            logger.info("Cache cleared successfully")
        except Exception as error_msg:
            logger.error(f"Error clearing cache: {error_msg}")

    def _initialize_components(self) -> None:
        """Инициализация компонентов сервиса."""
        try:
            # Инициализируем менеджер сессий
            manager_config = SessionManagerConfig(
                auto_analysis_enabled=self.config.auto_analysis_enabled,
                analysis_interval_minutes=self.config.analysis_interval_minutes,
                cache_ttl_minutes=self.config.cache_ttl_minutes,
                min_data_points=20,
                confidence_threshold=0.7,
                volatility_threshold=1.5,
                storage_enabled=self.config.storage_enabled,
                backup_enabled=self.config.backup_enabled,
            )
            # Создаем конкретную реализацию анализатора
            from .session_analyzer import SessionAnalyzer
            analyzer = SessionAnalyzer(self.registry, self.session_marker)
            self.session_manager = SessionManager(
                self.registry,
                self.session_marker,
                analyzer,  # Передаем конкретную реализацию
                self.data_repository,
                self.config_repository,
                manager_config,
            )
            logger.info("Session service components initialized")
        except Exception as e:
            logger.error(f"Error initializing session service components: {e}")
            raise

    def _select_analyzer_for_session_type(self, session_type: SessionType) -> str:
        """Выбор подходящего анализатора для типа сессии."""
        # Простая логика выбора на основе типа сессии
        if session_type == SessionType.CRYPTO_24H:
            return "influence"  # Для криптовалют используем анализатор влияния
        else:
            return "standard"  # Для остальных используем стандартный анализатор

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

    def _update_average_optimization_time(self, new_time_ms: float) -> None:
        """Обновление среднего времени оптимизации."""
        if self.state.total_optimizations == 1:
            self.state.average_optimization_time_ms = new_time_ms
        else:
            # Экспоненциальное скользящее среднее
            alpha = 0.1
            self.state.average_optimization_time_ms = (
                alpha * new_time_ms
                + (1 - alpha) * self.state.average_optimization_time_ms
            )

    def _update_average_prediction_time(self, new_time_ms: float) -> None:
        """Обновление среднего времени прогнозирования."""
        if self.state.total_predictions == 1:
            self.state.average_prediction_time_ms = new_time_ms
        else:
            # Экспоненциальное скользящее среднее
            alpha = 0.1
            self.state.average_prediction_time_ms = (
                alpha * new_time_ms
                + (1 - alpha) * self.state.average_prediction_time_ms
            )
