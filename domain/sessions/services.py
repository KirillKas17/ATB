"""
Сервисы для модуля торговых сессий.
"""

from typing import Any, Dict, List, Optional, TypedDict
import pandas as pd

from loguru import logger

from domain.types.session_types import (
    MarketConditions,
    SessionAnalysisResult,
    SessionMetrics,
    SessionPhase,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .implementations import BaseSessionTransitionManager
from .interfaces import (
    SessionCache,
    SessionDataRepository,
    SessionDataValidator,
    SessionRegistry,
)
from .session_influence_analyzer import SessionInfluenceAnalyzer
from .session_marker import SessionMarker


class SessionContext(TypedDict):
    """Типизированный контекст сессии."""

    active_sessions: List[str]
    primary_session: Optional[str]
    current_phase: str
    time_until_next_change: Optional[float]
    session_transitions: List[Dict[str, Any]]


class SessionPrediction(TypedDict):
    """Типизированный прогноз сессии."""

    predicted_volatility: float
    predicted_volume: float
    predicted_direction_bias: float
    predicted_momentum: float
    reversal_probability: float
    continuation_probability: float
    false_breakout_probability: float
    manipulation_risk: float
    whale_activity_probability: float
    mm_activity_probability: float


class SessionHealthStatus(TypedDict):
    """Типизированный статус здоровья сервиса."""

    status: str
    timestamp: str
    components: Dict[str, str]
    metrics: Dict[str, Any]


class SessionService:
    """Сервис для работы с торговыми сессиями."""

    def __init__(
        self,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        influence_analyzer: SessionInfluenceAnalyzer,
        transition_manager: BaseSessionTransitionManager,
        cache: SessionCache,
        validator: SessionDataValidator,
        data_repository: Optional[SessionDataRepository] = None,
        # Новые инфраструктурные компоненты:
        repository: Optional[Any] = None,
        session_cache: Optional[Any] = None,
        session_validator: Optional[Any] = None,
        metrics_calculator: Optional[Any] = None,
        pattern_recognizer: Optional[Any] = None,
        transition_manager_new: Optional[Any] = None,
        predictor: Optional[Any] = None,
        optimizer: Optional[Any] = None,
        monitor: Optional[Any] = None,
        analytics: Optional[Any] = None,
        risk_analyzer: Optional[Any] = None,
    ):
        self.registry = registry
        self.session_marker = session_marker
        self.influence_analyzer = influence_analyzer
        self.transition_manager = transition_manager
        self.cache = cache
        self.validator = validator
        self.data_repository = data_repository
        # Новые компоненты:
        self.repository = repository
        self.session_cache = session_cache
        self.session_validator = session_validator
        self.metrics_calculator = metrics_calculator
        self.pattern_recognizer = pattern_recognizer
        self.transition_manager_new = transition_manager_new
        self.predictor = predictor
        self.optimizer = optimizer
        self.monitor = monitor
        self.analytics = analytics
        self.risk_analyzer = risk_analyzer

    def get_current_session_context(
        self, timestamp: Optional[Timestamp] = None
    ) -> SessionContext:
        """Получение текущего контекста сессии."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            # Создаем ключ кэша
            cache_key = f"context_{timestamp.to_iso()}"
            # Проверяем кэш
            cached_context = self.cache.get(cache_key)
            if cached_context and isinstance(cached_context, dict):
                return cached_context
            # Получаем контекст
            context = self.session_marker.get_session_context(timestamp)
            if context and hasattr(context, 'to_dict'):
                context_dict = context.to_dict()
            else:
                context_dict = {
                    "active_sessions": [],
                    "primary_session": None,
                    "current_phase": "unknown",
                    "time_until_next_change": None,
                    "session_transitions": [],
                }
            # Кэшируем результат на 5 минут
            self.cache.set(cache_key, context_dict, ttl=300)
            return context_dict
        except Exception as e:
            logger.error(f"Failed to get session context: {e}")
            return {
                "active_sessions": [],
                "primary_session": None,
                "current_phase": "unknown",
                "time_until_next_change": None,
                "session_transitions": [],
            }

    def analyze_session_influence(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[SessionAnalysisResult]:
        """Анализ влияния сессии."""
        try:
            # Валидируем данные
            if not self.validator.validate_market_data(market_data):
                logger.warning(f"Invalid market data for {symbol}")
                return None
            # Проверяем кэш
            cache_key = (
                f"analysis_{symbol}_{timestamp.to_iso() if timestamp else 'current'}"
            )
            cached_analysis = self.cache.get(cache_key)
            if cached_analysis and isinstance(cached_analysis, SessionAnalysisResult):
                return cached_analysis
            # Выполняем анализ
            analysis = self.influence_analyzer.analyze_session(
                symbol, market_data, timestamp or Timestamp.now()
            )
            if analysis:
                # Валидируем результат
                if not self.validator.validate_session_analysis(analysis):
                    logger.warning(f"Invalid analysis result for {symbol}")
                    return None
                # Кэшируем результат на 10 минут
                self.cache.set(cache_key, analysis, ttl=600)
                # Сохраняем в репозиторий, если доступен
                if self.data_repository:
                    try:
                        self.data_repository.save_session_analysis(analysis)
                    except Exception as e:
                        logger.error(f"Failed to save session analysis: {e}")
            return analysis
        except Exception as e:
            logger.error(f"Failed to analyze session influence: {e}")
            return None

    def predict_session_behavior(
        self, session_type: SessionType, market_conditions: MarketConditions
    ) -> SessionPrediction:
        """Прогноз поведения сессии."""
        try:
            # Создаем безопасный ключ кэша
            conditions_str = f"{market_conditions['volatility']:.3f}_{market_conditions['volume']:.3f}_{market_conditions['market_regime'].value}"
            cache_key = f"prediction_{session_type.value}_{conditions_str}"
            # Проверяем кэш
            cached_prediction = self.cache.get(cache_key)
            if cached_prediction and isinstance(cached_prediction, dict):
                return cached_prediction
            # Получаем профиль сессии
            profile = self.registry.get_profile(session_type)
            if not profile:
                logger.warning(f"Session profile not found for {session_type.value}")
                return self._get_default_prediction()
            # Создаем базовые метрики
            base_metrics = SessionMetrics(
                volume_change_percent=0.0,
                volatility_change_percent=0.0,
                price_direction_bias=0.0,
                momentum_strength=0.5,
                false_breakout_probability=profile.false_breakout_probability,
                reversal_probability=profile.reversal_probability,
                trend_continuation_probability=profile.continuation_probability,
                influence_duration_minutes=getattr(profile.behavior, 'typical_volatility_spike_minutes', 30),
                peak_influence_time_minutes=getattr(profile.behavior, 'peak_influence_minutes', 15),
                spread_impact=getattr(profile.behavior, 'typical_spread_impact', 0.1),
                liquidity_impact=getattr(profile.behavior, 'typical_liquidity_impact', 0.2),
                correlation_with_other_sessions=getattr(profile.behavior, 'correlation_with_other_sessions', 0.0),
            )
            # Создаем базовые условия рынка
            base_conditions = MarketConditions(
                volatility=market_conditions["volatility"],
                volume=market_conditions["volume"],
                spread=market_conditions["spread"],
                liquidity=market_conditions["liquidity"],
                momentum=market_conditions["momentum"],
                trend_strength=market_conditions["trend_strength"],
                market_regime=market_conditions["market_regime"],
                session_intensity=market_conditions["session_intensity"],
            )
            # Создаем прогноз с безопасным доступом к атрибутам
            prediction: SessionPrediction = {
                "predicted_volatility": base_metrics.get("volatility_change_percent", 0.0),
                "predicted_volume": base_metrics.get("volume_change_percent", 0.0),
                "predicted_direction_bias": base_metrics.get("price_direction_bias", 0.0),
                "predicted_momentum": base_metrics.get("momentum_strength", 0.0),
                "reversal_probability": base_metrics.get("reversal_probability", 0.0),
                "continuation_probability": base_metrics.get("trend_continuation_probability", 0.0),
                "false_breakout_probability": base_metrics.get("false_breakout_probability", 0.0),
                "manipulation_risk": 0.1,  # Базовый риск манипуляции
                "whale_activity_probability": 0.2,  # Базовая вероятность активности китов
                "mm_activity_probability": 0.3,  # Базовая вероятность активности MM
            }
            # Кэшируем результат на 15 минут
            self.cache.set(cache_key, prediction, ttl=900)
            return prediction
        except Exception as e:
            logger.error(f"Failed to predict session behavior: {e}")
            return self._get_default_prediction()

    def get_session_recommendations(
        self, symbol: str, session_type: SessionType
    ) -> List[str]:
        """Получение рекомендаций для сессии."""
        try:
            cache_key = f"recommendations_{symbol}_{session_type.value}"
            cached_recommendations = self.cache.get(cache_key)
            if cached_recommendations and isinstance(cached_recommendations, list):
                return cached_recommendations
            # Получаем базовые рекомендации
            base_recommendations = self._get_session_recommendations(session_type)
            # Добавляем специфичные для символа рекомендации
            symbol_recommendations = self._get_symbol_specific_recommendations(
                symbol, session_type
            )
            # Объединяем рекомендации
            all_recommendations = base_recommendations + symbol_recommendations
            # Кэшируем результат на 30 минут
            self.cache.set(cache_key, all_recommendations, ttl=1800)
            return all_recommendations
        except Exception as e:
            logger.error(f"Failed to get session recommendations: {e}")
            return []

    def get_session_statistics(self, session_type: SessionType) -> Dict[str, Any]:
        """Получение статистики сессии."""
        try:
            cache_key = f"statistics_{session_type.value}"
            cached_statistics = self.cache.get(cache_key)
            if cached_statistics and isinstance(cached_statistics, dict):
                return cached_statistics
            # Получаем профиль сессии
            profile = self.registry.get_profile(session_type)
            if not profile:
                logger.warning(f"Session profile not found for {session_type.value}")
                return {}
            # Создаем статистику
            statistics = {
                "session_type": session_type.value,
                "total_observations": getattr(profile, 'total_observations', 0),
                "avg_volume_change": getattr(profile, 'avg_volume_change', 0.0),
                "avg_volatility_change": getattr(profile, 'avg_volatility_change', 0.0),
                "avg_direction_bias": getattr(profile, 'avg_direction_bias', 0.0),
                "avg_confidence": getattr(profile, 'avg_confidence', 0.0),
                "bullish_count": getattr(profile, 'bullish_count', 0),
                "bearish_count": getattr(profile, 'bearish_count', 0),
                "neutral_count": getattr(profile, 'neutral_count', 0),
                "last_updated": getattr(profile, 'last_updated', ""),
            }
            # Кэшируем результат на 1 час
            self.cache.set(cache_key, statistics, ttl=3600)
            return statistics
        except Exception as e:
            logger.error(f"Failed to get session statistics: {e}")
            return {}

    def is_transition_period(self, timestamp: Optional[Timestamp] = None) -> bool:
        """Проверка, является ли текущее время переходным периодом."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            return self.transition_manager.is_transition_period(timestamp)
        except Exception as e:
            logger.error(f"Failed to check transition period: {e}")
            return False

    def get_active_transitions(
        self, timestamp: Optional[Timestamp] = None
    ) -> List[Dict[str, Any]]:
        """Получение активных переходов."""
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            transitions = self.transition_manager.get_active_transitions(timestamp)
            # Преобразуем в список словарей
            return [transition.__dict__ if hasattr(transition, '__dict__') else {} for transition in transitions]
        except Exception as e:
            logger.error(f"Failed to get active transitions: {e}")
            return []

    def update_session_profile(
        self, session_type: SessionType, updates: Dict[str, Any]
    ) -> bool:
        """Обновление профиля сессии."""
        try:
            success = self.registry.update_profile(session_type, updates)
            if success:
                # Очищаем кэш для этой сессии
                self._clear_session_cache(session_type)
            return success
        except Exception as e:
            logger.error(f"Failed to update session profile: {e}")
            return False

    def get_session_overlap(
        self, session1: SessionType, session2: SessionType
    ) -> float:
        """Получение перекрытия между сессиями."""
        try:
            if hasattr(self.transition_manager, 'get_session_overlap'):
                result = self.transition_manager.get_session_overlap(session1, session2)
                return float(result) if result is not None else 0.0
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get session overlap: {e}")
            return 0.0

    def get_session_phase(
        self, session_type: SessionType, timestamp: Optional[Timestamp] = None
    ) -> Optional[str]:
        """Получение фазы сессии."""
        try:
            if hasattr(self.transition_manager, 'get_session_phase'):
                phase = self.transition_manager.get_session_phase(session_type, timestamp)
                return phase.value if phase else None
            return None
        except Exception as e:
            logger.error(f"Failed to get session phase: {e}")
            return None

    def get_next_session_change(
        self, timestamp: Optional[Timestamp] = None
    ) -> Optional[Dict[str, Any]]:
        """Получение информации о следующем изменении сессии."""
        try:
            if hasattr(self.transition_manager, 'get_next_session_change'):
                change_info = self.transition_manager.get_next_session_change(timestamp)
                return change_info.to_dict() if change_info and hasattr(change_info, 'to_dict') else None
            return None
        except Exception as e:
            logger.error(f"Failed to get next session change: {e}")
            return None

    def clear_cache(self) -> None:
        """Очистка кэша."""
        try:
            self.cache.clear()
            logger.info("Session service cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def get_session_health_check(self) -> SessionHealthStatus:
        """Проверка здоровья сервиса."""
        try:
            components = {
                "registry": "healthy" if self.registry else "unhealthy",
                "cache": "healthy" if self.cache else "unhealthy",
                "validator": "healthy" if self.validator else "unhealthy",
                "transition_manager": "healthy" if self.transition_manager else "unhealthy",
            }
            metrics = {
                "cache_size": getattr(self.cache, 'size', 0) if hasattr(self.cache, 'size') else 0,
                "registry_profiles": len(self.registry.get_all_profiles()) if hasattr(self.registry, 'get_all_profiles') else 0,
            }
            return {
                "status": "healthy" if all(status == "healthy" for status in components.values()) else "unhealthy",
                "timestamp": Timestamp.now().to_iso(),
                "components": components,
                "metrics": metrics,
            }
        except Exception as e:
            logger.error(f"Failed to get health check: {e}")
            return {
                "status": "unhealthy",
                "timestamp": Timestamp.now().to_iso(),
                "components": {},
                "metrics": {},
            }

    def _get_default_prediction(self) -> SessionPrediction:
        """Получение прогноза по умолчанию."""
        return {
            "predicted_volatility": 0.0,
            "predicted_volume": 0.0,
            "predicted_direction_bias": 0.0,
            "predicted_momentum": 0.0,
            "reversal_probability": 0.5,
            "continuation_probability": 0.5,
            "false_breakout_probability": 0.3,
            "manipulation_risk": 0.1,
            "whale_activity_probability": 0.2,
            "mm_activity_probability": 0.3,
        }

    def _get_symbol_specific_recommendations(
        self, symbol: str, session_type: SessionType
    ) -> List[str]:
        """Получение специфичных для символа рекомендаций."""
        try:
            # Здесь можно добавить логику для получения специфичных рекомендаций
            return []
        except Exception as e:
            logger.error(f"Failed to get symbol-specific recommendations: {e}")
            return []

    def _clear_session_cache(self, session_type: SessionType) -> None:
        """Очистка кэша для конкретной сессии."""
        try:
            # Очищаем все ключи, связанные с этой сессией
            keys_to_clear = [
                f"prediction_{session_type.value}_*",
                f"recommendations_*_{session_type.value}",
                f"statistics_{session_type.value}",
            ]
            for key_pattern in keys_to_clear:
                # Здесь должна быть логика очистки по паттерну
                pass
        except Exception as e:
            logger.error(f"Failed to clear session cache: {e}")

    def _get_session_recommendations(self, session_type: SessionType) -> List[str]:
        """Получение базовых рекомендаций для сессии."""
        try:
            # Здесь можно добавить логику для получения базовых рекомендаций
            return []
        except Exception as e:
            logger.error(f"Failed to get session recommendations: {e}")
            return []

    def _update_session_profile(
        self, session_type: SessionType, updates: Dict[str, Any]
    ) -> bool:
        """Обновление профиля сессии."""
        try:
            # Здесь должна быть логика обновления профиля
            return True
        except Exception as e:
            logger.error(f"Failed to update session profile: {e}")
            return False
