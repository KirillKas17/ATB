"""
Интерфейсы для модуля торговых сессий (строгая типизация).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union
import pandas as pd

from domain.type_definitions.session_types import (
    MarketConditions,
    SessionAnalysisResult,
    SessionMetrics,
    SessionPhase,
    SessionProfile,
    SessionTransition,
    SessionType,
)
from domain.sessions.session_marker import SessionMarker
from domain.value_objects.timestamp import Timestamp


class SessionRegistry(Protocol):
    """Протокол для реестра сессий."""

    def get_profile(self, session_type: SessionType) -> Optional[SessionProfile]:
        """Получение профиля сессии."""
        ...

    def get_all_profiles(self) -> Dict[SessionType, SessionProfile]:
        """Получение всех профилей."""
        ...

    def register_profile(self, profile: SessionProfile) -> None:
        """Регистрация профиля."""
        ...

    def get_active_sessions(self, timestamp: Timestamp) -> List[SessionProfile]:
        """Получение активных сессий."""
        ...

    def get_primary_session(self, timestamp: Timestamp) -> Optional[SessionProfile]:
        """Получение основной сессии."""
        ...

    def get_session_overlap(
        self, session1: SessionType, session2: SessionType
    ) -> float:
        """Получение перекрытия сессий."""
        ...

    def get_session_recommendations(self, session_type: SessionType) -> List[str]:
        """Получение рекомендаций для сессии."""
        ...

    def get_session_statistics(self, session_type: SessionType) -> Dict[str, float]:
        """Получение статистики сессии."""
        ...

    def update_profile(
        self,
        session_type: SessionType,
        updates: Dict[str, Union[str, float, int, bool]],
    ) -> bool:
        """Обновление профиля сессии."""
        ...


class SessionTimeCalculator(Protocol):
    """Протокол для калькулятора времени сессий."""

    def is_session_active(
        self, session_type: SessionType, timestamp: Timestamp
    ) -> bool:
        """Проверка активности сессии."""
        ...

    def get_session_phase(
        self, session_type: SessionType, timestamp: Timestamp
    ) -> SessionPhase:
        """Получение фазы сессии."""
        ...

    def get_session_overlap(
        self, session1: SessionType, session2: SessionType
    ) -> float:
        """Вычисление перекрытия сессий."""
        ...

    def get_next_session_time(
        self, session_type: SessionType, timestamp: Timestamp
    ) -> Timestamp:
        """Получение времени следующей сессии."""
        ...


class SessionMetricsAnalyzer(Protocol):
    """Протокол для анализатора метрик сессий."""

    def calculate_volume_impact(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет влияния на объем."""
        ...

    def calculate_volatility_impact(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет влияния на волатильность."""
        ...

    def calculate_direction_bias(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет смещения направления."""
        ...

    def calculate_momentum_strength(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет силы импульса."""
        ...


class SessionPatternRecognizer(Protocol):
    """Протокол для распознавателя паттернов сессий."""

    def identify_session_patterns(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> List[str]:
        """Идентификация паттернов сессии."""
        ...

    def calculate_pattern_probability(
        self, pattern: str, session_profile: SessionProfile
    ) -> float:
        """Расчет вероятности паттерна."""
        ...

    def get_historical_patterns(
        self, session_type: SessionType, lookback_days: int
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Получение исторических паттернов."""
        ...


class SessionInfluencePredictor(Protocol):
    """Протокол для предиктора влияния сессий."""

    def predict_volatility(
        self, session_profile: SessionProfile, market_conditions: MarketConditions
    ) -> float:
        """Прогноз волатильности."""
        ...

    def predict_volume(
        self, session_profile: SessionProfile, market_conditions: MarketConditions
    ) -> float:
        """Прогноз объема."""
        ...

    def predict_direction(
        self, session_profile: SessionProfile, market_conditions: MarketConditions
    ) -> str:
        """Прогноз направления."""
        ...

    def calculate_confidence(
        self, session_profile: SessionProfile, market_conditions: MarketConditions
    ) -> float:
        """Расчет уверенности в прогнозе."""
        ...


class SessionTransitionAnalyzer(Protocol):
    """Протокол для анализатора переходов между сессиями."""

    def detect_upcoming_transitions(
        self, timestamp: Timestamp
    ) -> List[SessionTransition]:
        """Обнаружение предстоящих переходов."""
        ...

    def calculate_transition_impact(
        self, transition: SessionTransition, time_to_transition: int
    ) -> Dict[str, float]:
        """Расчет влияния перехода."""
        ...

    def get_manipulation_risk(
        self, transition: SessionTransition, time_to_transition: int
    ) -> float:
        """Получение риска манипуляции."""
        ...


# Абстрактные базовые классы
class BaseSessionAnalyzer(ABC):
    """Базовый абстрактный класс для анализаторов сессий."""

    def __init__(self, registry: SessionRegistry) -> None:
        self.registry = registry
        self._cache: Dict[str, object] = {}

    @abstractmethod
    def analyze_session(
        self, symbol: str, market_data: pd.DataFrame, timestamp: Timestamp
    ) -> Optional[SessionAnalysisResult]:
        """Анализ сессии."""
        pass

    @abstractmethod
    def get_session_context(self, timestamp: Timestamp) -> Dict[str, object]:
        """Получение контекста сессии."""
        pass

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()


class BaseSessionPredictor(ABC):
    """Базовый абстрактный класс для предикторов сессий."""

    def __init__(
        self, registry: SessionRegistry, analyzer: SessionMetricsAnalyzer
    ) -> None:
        self.registry = registry
        self.analyzer = analyzer
        self._historical_data: Dict[str, pd.DataFrame] = {}

    @abstractmethod
    def predict_session_behavior(
        self,
        session_type: SessionType,
        market_conditions: MarketConditions,
        timestamp: Timestamp,
    ) -> Dict[str, float]:
        """Прогноз поведения сессии."""
        pass

    @abstractmethod
    def update_historical_data(self, symbol: str, market_data: pd.DataFrame) -> None:
        """Обновление исторических данных."""
        pass

    def get_prediction_confidence(self, predictions: Dict[str, float]) -> float:
        """Получение уверенности в прогнозе."""
        # Базовая реализация - можно переопределить в дочерних классах
        if not predictions:
            return 0.0
        # Среднее значение всех прогнозов как базовая уверенность
        values = [abs(v) for v in predictions.values() if isinstance(v, (int, float))]
        return sum(values) / len(values) if values else 0.0


class BaseSessionTransitionManager(ABC):
    """Базовый абстрактный класс для менеджера переходов сессий."""

    def __init__(self, registry: SessionRegistry) -> None:
        self.registry = registry
        self._transition_cache: Dict[str, SessionTransition] = {}

    @abstractmethod
    def get_active_transitions(self, timestamp: Timestamp) -> List[SessionTransition]:
        """Получение активных переходов."""
        pass

    @abstractmethod
    def calculate_transition_metrics(
        self, transition: SessionTransition, current_metrics: SessionMetrics
    ) -> SessionMetrics:
        """Расчет метрик перехода."""
        pass

    def is_transition_period(
        self, timestamp: Timestamp, window_minutes: int = 30
    ) -> bool:
        """Проверка, является ли время переходным периодом."""
        transitions = self.get_active_transitions(timestamp)
        return len(transitions) > 0


# Интерфейсы для репозиториев
class SessionDataRepository(Protocol):
    """Протокол для репозитория данных сессий."""

    def save_session_analysis(self, analysis: SessionAnalysisResult) -> None:
        """Сохранение анализа сессии."""
        ...

    def get_session_analysis(
        self, session_type: SessionType, start_time: Timestamp, end_time: Timestamp
    ) -> List[SessionAnalysisResult]:
        """Получение анализа сессии."""
        ...

    def get_session_statistics(
        self, session_type: SessionType, lookback_days: int
    ) -> Dict[str, float]:
        """Получение статистики сессии."""
        ...


class SessionConfigurationRepository(Protocol):
    """Протокол для репозитория конфигурации сессий."""

    def save_session_profile(self, profile: SessionProfile) -> None:
        """Сохранение профиля сессии."""
        ...

    def get_session_profile(
        self, session_type: SessionType
    ) -> Optional[SessionProfile]:
        """Получение профиля сессии."""
        ...

    def get_all_session_profiles(self) -> Dict[SessionType, SessionProfile]:
        """Получение всех профилей сессий."""
        ...

    def delete_session_profile(self, session_type: SessionType) -> None:
        """Удаление профиля сессии."""
        ...


# Интерфейсы для сервисов
class SessionService(Protocol):
    """Протокол для сервисного слоя сессий."""

    def get_current_session_context(
        self, timestamp: Optional[Timestamp] = None
    ) -> Dict[str, object]:
        """Получение текущего контекста сессии."""
        ...

    def analyze_session_influence(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[SessionAnalysisResult]:
        """Анализ влияния сессии."""
        ...

    def predict_session_behavior(
        self, session_type: SessionType, market_conditions: MarketConditions
    ) -> Dict[str, float]:
        """Прогноз поведения сессии."""
        ...

    def get_session_recommendations(
        self, symbol: str, session_type: SessionType
    ) -> List[str]:
        """Получение рекомендаций для сессии."""
        ...


# Интерфейсы для фабрик
class SessionAnalyzerFactory(Protocol):
    """Протокол для фабрики анализаторов сессий."""

    def create_analyzer(
        self, 
        name: str, 
        registry: SessionRegistry, 
        session_marker: SessionMarker, 
        config: Optional[Any] = None,
        force_recreate: bool = False
    ) -> Optional[BaseSessionAnalyzer]:
        """Создание анализатора."""
        ...

    def get_analyzer(self, name: str) -> Optional[BaseSessionAnalyzer]:
        """Получение анализатора из кэша."""
        ...

    def get_available_analyzers(self) -> List[str]:
        """Получение списка доступных анализаторов."""
        ...

    def create_predictor(self, predictor_type: str) -> BaseSessionPredictor:
        """Создание предиктора."""
        ...

    def create_transition_manager(self) -> BaseSessionTransitionManager:
        """Создание менеджера переходов."""
        ...


# Интерфейсы для валидаторов
class SessionDataValidator(Protocol):
    """Протокол для валидации данных сессий."""

    def validate_market_data(self, market_data: pd.DataFrame) -> bool:
        """Валидация рыночных данных."""
        ...

    def validate_session_profile(self, profile: SessionProfile) -> bool:
        """Валидация профиля сессии."""
        ...

    def validate_session_analysis(self, analysis: SessionAnalysisResult) -> bool:
        """Валидация анализа сессии."""
        ...


# Интерфейсы для кэширования
class SessionCache(Protocol):
    """Протокол для кэша сессий."""

    def get(self, key: str) -> Optional[object]:
        """Получение из кэша."""
        ...

    def set(self, key: str, value: object, ttl: Optional[int] = None) -> None:
        """Установка в кэш."""
        ...

    def delete(self, key: str) -> None:
        """Удаление из кэша."""
        ...

    def clear(self) -> None:
        """Очистка кэша."""
        ...

    def exists(self, key: str) -> bool:
        """Проверка существования ключа."""
        ...
