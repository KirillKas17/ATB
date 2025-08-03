# -*- coding: utf-8 -*-
"""Session marker for trading sessions."""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union

from loguru import logger

from domain.types.session_types import (
    MarketConditions,
    MarketRegime,
    SessionIntensity,
    SessionPhase,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .session_profile import SessionProfileRegistry

# Глобальный реестр сессий
session_registry = SessionProfileRegistry()


@dataclass
class SessionState:
    """Состояние торговой сессии."""

    session_type: SessionType
    phase: SessionPhase
    is_active: bool
    time_until_open: Optional[timedelta] = None
    time_until_close: Optional[timedelta] = None
    overlap_with_other_sessions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Union[str, int, float, None, Dict[str, float]]]:
        """Преобразование в словарь."""
        return {
            "session_type": self.session_type.value,
            "phase": self.phase.value,
            "is_active": self.is_active,
            "time_until_open": (
                self.time_until_open.total_seconds() if self.time_until_open else None
            ),
            "time_until_close": (
                self.time_until_close.total_seconds() if self.time_until_close else None
            ),
            "overlap_with_other_sessions": self.overlap_with_other_sessions,
        }


@dataclass
class MarketSessionContext:
    """Контекст рыночных сессий."""

    timestamp: Timestamp
    primary_session: Optional[SessionState] = None
    active_sessions: List[SessionState] = field(default_factory=list)
    session_transitions: List[Dict[str, object]] = field(
        default_factory=list
    )
    market_conditions: Optional[MarketConditions] = None

    def to_dict(self) -> Dict[str, object]:
        """Преобразование в словарь."""
        return {
            "timestamp": self.timestamp.to_iso(),
            "primary_session": (
                self.primary_session.to_dict() if self.primary_session else None
            ),
            "active_sessions": [s.to_dict() for s in self.active_sessions],
            "session_transitions": self.session_transitions,
            "market_conditions": (
                dict(self.market_conditions) if self.market_conditions else None
            ),
        }


class SessionMarker:
    """Маркер торговых сессий."""

    def __init__(self, registry: Optional[SessionProfileRegistry] = None) -> None:
        self.registry: SessionProfileRegistry = registry or session_registry
        self._last_context: Optional[MarketSessionContext] = None
        self._cache: Dict[str, MarketSessionContext] = {}

    def get_session_context(
        self, timestamp: Optional[Timestamp] = None
    ) -> MarketSessionContext:
        """Получение контекста торговых сессий."""
        if timestamp is None:
            timestamp = Timestamp.now()
        # Используем новый метод без рекурсии
        context = self._get_current_session_context(timestamp)
        # Добавляем переходы сессий
        session_transitions = self._detect_session_transitions(timestamp)
        context.session_transitions = session_transitions
        # Кэшируем результат
        self._cache[timestamp.to_iso()] = context
        return context

    def _get_next_session_time(
        self, profile: SessionProfile, timestamp: Timestamp
    ) -> Timestamp:
        """Получение времени следующего открытия сессии."""
        current_time = timestamp.to_datetime()
        current_time_only = current_time.time()
        today = current_time.date()
        # Определяем время следующего открытия
        if profile.time_window.start_time > profile.time_window.end_time:
            # Сессия переходит через полночь
            if current_time_only >= profile.time_window.start_time:
                # Уже после открытия сегодня
                next_open = datetime.combine(
                    today + timedelta(days=1),
                    profile.time_window.start_time,
                    tzinfo=current_time.tzinfo,
                )
            else:
                # Еще не открылась сегодня
                next_open = datetime.combine(
                    today, profile.time_window.start_time, tzinfo=current_time.tzinfo
                )
        else:
            # Обычная сессия
            if current_time_only >= profile.time_window.start_time:
                # Уже после открытия сегодня
                next_open = datetime.combine(
                    today + timedelta(days=1),
                    profile.time_window.start_time,
                    tzinfo=current_time.tzinfo,
                )
            else:
                # Еще не открылась сегодня
                next_open = datetime.combine(
                    today, profile.time_window.start_time, tzinfo=current_time.tzinfo
                )
        return Timestamp(next_open)

    def _detect_session_transitions(
        self, timestamp: Timestamp
    ) -> List[Dict[str, object]]:
        """Определение переходов между сессиями."""
        transitions: List[Dict[str, object]] = []
        # Получаем текущий контекст
        current_context = self._get_current_session_context(timestamp)
        # Проверяем переходы на следующие 24 часа
        for hours_ahead in range(1, 25):
            future_timestamp = timestamp.add_hours(hours_ahead)
            future_context = self._get_current_session_context(future_timestamp)
            # Сравниваем активные сессии
            current_active = {s.session_type for s in current_context.active_sessions}
            future_active = {s.session_type for s in future_context.active_sessions}
            # Находим новые и завершенные сессии
            new_sessions = future_active - current_active
            ended_sessions = current_active - future_active
            # Создаем переходы
            for session_type in new_sessions:
                profile = self.registry.get_profile(session_type)
                if profile:
                    transition: Dict[str, object] = {
                        "type": "session_start",
                        "session": session_type.value,
                        "time_ahead_hours": hours_ahead,
                        "timestamp": future_timestamp.to_iso(),
                    }
                    transitions.append(transition)
            for session_type in ended_sessions:
                profile = self.registry.get_profile(session_type)
                if profile:
                    end_transition: Dict[str, object] = {
                        "type": "session_end",
                        "session": session_type.value,
                        "time_ahead_hours": hours_ahead,
                        "timestamp": future_timestamp.to_iso(),
                    }
                    transitions.append(end_transition)
        return transitions

    def _get_current_session_context(
        self, timestamp: Timestamp
    ) -> MarketSessionContext:
        """Получение контекста сессии без рекурсии."""
        # Определяем активные сессии
        active_sessions: List[SessionState] = []
        for session_type in SessionType:
            profile = self.registry.get_profile(session_type)
            if profile and self._is_session_active_at_time(profile, timestamp):
                # Определяем фазу сессии
                phase = self._determine_session_phase(profile, timestamp)
                session_state = SessionState(
                    session_type=session_type,
                    phase=phase,
                    is_active=True,
                    time_until_open=None,  # Сессия уже активна
                    time_until_close=None,  # Будет рассчитано ниже
                    overlap_with_other_sessions={},
                )
                active_sessions.append(session_state)
        # Определяем основную сессию (с наибольшим влиянием)
        primary_session = active_sessions[0] if active_sessions else None
        market_conditions = self._analyze_market_conditions(timestamp, active_sessions)
        return MarketSessionContext(
            timestamp=timestamp,
            primary_session=primary_session,
            active_sessions=active_sessions,
            session_transitions=[],  # Переходы будут добавлены отдельно
            market_conditions=market_conditions,
        )

    def _is_session_active_at_time(
        self, profile: SessionProfile, timestamp: Timestamp
    ) -> bool:
        """Проверка активности сессии в заданное время."""
        # Простая логика проверки времени
        hour = timestamp.to_datetime().hour
        if profile.session_type == SessionType.ASIAN:
            return 0 <= hour < 8
        elif profile.session_type == SessionType.LONDON:
            return 8 <= hour < 16
        elif profile.session_type == SessionType.NEW_YORK:
            return 13 <= hour < 21
        elif profile.session_type == SessionType.CRYPTO_24H:
            return True  # Криптовалюты торгуются 24/7
        return False

    def _determine_session_phase(
        self, profile: SessionProfile, timestamp: Timestamp
    ) -> SessionPhase:
        """Определение фазы сессии."""
        # Простая логика определения фазы
        hour = timestamp.to_datetime().hour
        if profile.session_type == SessionType.ASIAN:
            if 0 <= hour < 2:
                return SessionPhase.OPENING
            elif 2 <= hour < 6:
                return SessionPhase.MID_SESSION
            else:
                return SessionPhase.CLOSING
        elif profile.session_type == SessionType.LONDON:
            if 8 <= hour < 10:
                return SessionPhase.OPENING
            elif 10 <= hour < 14:
                return SessionPhase.MID_SESSION
            else:
                return SessionPhase.CLOSING
        elif profile.session_type == SessionType.NEW_YORK:
            if 13 <= hour < 15:
                return SessionPhase.OPENING
            elif 15 <= hour < 19:
                return SessionPhase.MID_SESSION
            else:
                return SessionPhase.CLOSING
        else:
            return SessionPhase.MID_SESSION

    def _analyze_market_conditions(
        self, timestamp: Timestamp, active_sessions: List[SessionState]
    ) -> MarketConditions:
        """Анализ рыночных условий на основе активных сессий."""
        if not active_sessions:
            # Нет активных сессий - используем криптовалютную сессию
            crypto_profile = self.registry.get_profile(SessionType.CRYPTO_24H)
            if crypto_profile:
                return MarketConditions(
                    volatility=1.0,
                    volume=1.0,
                    spread=1.0,
                    liquidity=0.8,
                    momentum=0.5,
                    trend_strength=0.3,
                    market_regime=crypto_profile.market_regime_tendency,
                    session_intensity=crypto_profile.intensity_profile,
                )
        # Анализируем активные сессии
        total_volume_multiplier = 0.0
        total_volatility_multiplier = 0.0
        total_spread_multiplier = 0.0
        session_count = len(active_sessions)
        for session_state in active_sessions:
            profile = self.registry.get_profile(session_state.session_type)
            if profile:
                total_volume_multiplier += profile.typical_volume_multiplier
                total_volatility_multiplier += profile.typical_volatility_multiplier
                total_spread_multiplier += profile.typical_spread_multiplier
        # Вычисляем средние значения
        avg_volume = (
            total_volume_multiplier / session_count if session_count > 0 else 1.0
        )
        avg_volatility = (
            total_volatility_multiplier / session_count if session_count > 0 else 1.0
        )
        avg_spread = (
            total_spread_multiplier / session_count if session_count > 0 else 1.0
        )
        # Определяем основную сессию для режима рынка
        primary_session = active_sessions[0] if active_sessions else None
        primary_profile = (
            self.registry.get_profile(primary_session.session_type)
            if primary_session
            else None
        )
        # market_regime и session_intensity могут быть None, подставляем значения по умолчанию
        market_regime = MarketRegime.RANGING
        session_intensity = SessionIntensity.NORMAL
        return MarketConditions(
            volatility=avg_volatility,
            volume=avg_volume,
            spread=avg_spread,
            liquidity=1.0 / avg_spread,  # Обратная зависимость
            momentum=avg_volume * avg_volatility,  # Комбинированный показатель
            trend_strength=0.5,  # Базовое значение
            market_regime=market_regime,
            session_intensity=session_intensity,
        )

    def get_current_phase(
        self, timestamp: Optional[Timestamp] = None
    ) -> Optional[SessionPhase]:
        """Получение текущей фазы основной сессии."""
        context = self.get_session_context(timestamp)
        return context.primary_session.phase if context.primary_session else None

    def get_active_session_types(
        self, timestamp: Optional[Timestamp] = None
    ) -> List[SessionType]:
        """Получение типов активных сессий."""
        context = self.get_session_context(timestamp)
        return [s.session_type for s in context.active_sessions]

    def is_session_active(
        self, session_type: SessionType, timestamp: Optional[Timestamp] = None
    ) -> bool:
        """Проверка активности сессии."""
        context = self.get_session_context(timestamp)
        return any(s.session_type == session_type for s in context.active_sessions)

    def get_session_overlap(
        self, session_type1: SessionType, session_type2: SessionType
    ) -> float:
        """Получение перекрытия между сессиями."""
        return self.registry.get_session_overlap(session_type1, session_type2)

    def get_next_session_change(
        self, timestamp: Optional[Timestamp] = None
    ) -> Optional[Dict[str, Union[str, int, float, object]]]:
        """Получение информации о следующем изменении сессии."""
        context = self.get_session_context(timestamp)
        if not context.session_transitions:
            return None
        return min(
            context.session_transitions, 
            key=lambda t: float(t["time_ahead_hours"]) if isinstance(t["time_ahead_hours"], (int, float, str)) else 0.0
        )

    def get_session_characteristics(
        self, session_type: SessionType
    ) -> Optional[Dict[str, Union[str, float, int]]]:
        stats = self.registry.get_session_statistics(session_type)
        return stats if stats else None

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()
        logger.info("Session marker cache cleared")

    def get_session_recommendations(self, session_type: SessionType) -> List[str]:
        """Получение рекомендаций для сессии."""
        return self.registry.get_session_recommendations(session_type)


# Глобальный экземпляр маркера
session_marker = SessionMarker()
