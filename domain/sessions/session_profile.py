# -*- coding: utf-8 -*-
"""Профили торговых сессий и их поведенческие характеристики."""
from datetime import time
from typing import Dict, List, Optional, Union

from loguru import logger

from domain.type_definitions.session_types import (
    SessionBehavior,
    SessionPhase,
    SessionTimeWindow,
    SessionType,
    SessionProfile,
)
from domain.value_objects.timestamp import Timestamp


class SessionProfileRegistry:
    """Реестр профилей торговых сессий."""

    def __init__(self) -> None:
        self._profiles: Dict[SessionType, SessionProfile] = {}
        self._initialize_default_profiles()

    def _initialize_default_profiles(self) -> None:
        """Инициализация профилей по умолчанию."""
        # Азиатская сессия (Токио)
        asian_behavior = SessionBehavior(
            typical_volatility_spike_minutes=45,
            volume_peak_hours=[2, 4, 6],
            quiet_hours=[1, 5],
            avg_volume_multiplier=0.8,
            avg_volatility_multiplier=0.9,
            typical_direction_bias=0.1,
            common_patterns=["asian_range", "breakout_failure"],
            false_breakout_probability=0.4,
            reversal_probability=0.25,
            overlap_impact={"london": 1.2, "new_york": 0.9},
        )
        asian_time_window = SessionTimeWindow(
            start_time=time(0, 0),  # 00:00 UTC
            end_time=time(8, 0),  # 08:00 UTC
            timezone="UTC",
        )
        asian_profile = SessionProfile(
            session_type=SessionType.ASIAN,
            time_window=asian_time_window,
            behavior=asian_behavior,
            description="Азиатская торговая сессия (Токио)",
            typical_volume_multiplier=0.8,
            typical_volatility_multiplier=0.9,
            typical_spread_multiplier=1.1,
            whale_activity_probability=0.15,
            mm_activity_probability=0.25,
            news_sensitivity=0.4,
            technical_signal_strength=0.6,
            fundamental_impact_multiplier=0.8,
            correlation_breakdown_probability=0.3,
            gap_probability=0.15,
            reversal_probability=0.25,
            continuation_probability=0.55,
            manipulation_susceptibility=0.35,
        )
        # Лондонская сессия
        london_behavior = SessionBehavior(
            typical_volatility_spike_minutes=30,
            volume_peak_hours=[2, 4, 6],
            quiet_hours=[1, 5],
            avg_volume_multiplier=1.2,
            avg_volatility_multiplier=1.1,
            typical_direction_bias=0.05,
            common_patterns=["london_breakout", "trend_continuation"],
            false_breakout_probability=0.3,
            reversal_probability=0.2,
            overlap_impact={"asian": 1.3, "new_york": 1.4},
        )
        london_time_window = SessionTimeWindow(
            start_time=time(8, 0),  # 08:00 UTC
            end_time=time(16, 0),  # 16:00 UTC
            timezone="UTC",
        )
        london_profile = SessionProfile(
            session_type=SessionType.LONDON,
            time_window=london_time_window,
            behavior=london_behavior,
            description="Лондонская торговая сессия",
            typical_volume_multiplier=1.2,
            typical_volatility_multiplier=1.1,
            typical_spread_multiplier=0.9,
            whale_activity_probability=0.2,
            mm_activity_probability=0.35,
            news_sensitivity=0.6,
            technical_signal_strength=0.75,
            fundamental_impact_multiplier=1.2,
            correlation_breakdown_probability=0.2,
            gap_probability=0.1,
            reversal_probability=0.2,
            continuation_probability=0.65,
            manipulation_susceptibility=0.25,
        )
        # Нью-Йоркская сессия
        ny_behavior = SessionBehavior(
            typical_volatility_spike_minutes=25,
            volume_peak_hours=[1, 3, 5],
            quiet_hours=[2, 4],
            avg_volume_multiplier=1.5,
            avg_volatility_multiplier=1.3,
            typical_direction_bias=0.0,
            common_patterns=["ny_momentum", "end_of_day_reversal"],
            false_breakout_probability=0.25,
            reversal_probability=0.3,
            overlap_impact={"london": 1.5, "asian": 0.8},
        )
        ny_time_window = SessionTimeWindow(
            start_time=time(13, 0),  # 13:00 UTC
            end_time=time(21, 0),  # 21:00 UTC
            timezone="UTC",
        )
        ny_profile = SessionProfile(
            session_type=SessionType.NEW_YORK,
            time_window=ny_time_window,
            behavior=ny_behavior,
            description="Нью-Йоркская торговая сессия",
            typical_volume_multiplier=1.5,
            typical_volatility_multiplier=1.3,
            typical_spread_multiplier=0.8,
            whale_activity_probability=0.25,
            mm_activity_probability=0.4,
            news_sensitivity=0.8,
            technical_signal_strength=0.8,
            fundamental_impact_multiplier=1.5,
            correlation_breakdown_probability=0.15,
            gap_probability=0.08,
            reversal_probability=0.3,
            continuation_probability=0.6,
            manipulation_susceptibility=0.2,
        )
        # Криптовалютная сессия (24/7)
        crypto_behavior = SessionBehavior(
            typical_volatility_spike_minutes=60,
            volume_peak_hours=[0, 6, 12, 18],
            quiet_hours=[3, 9, 15, 21],
            avg_volume_multiplier=1.0,
            avg_volatility_multiplier=1.2,
            typical_direction_bias=0.0,
            common_patterns=["crypto_pump", "crypto_dump", "sideways_consolidation"],
            false_breakout_probability=0.35,
            reversal_probability=0.4,
            overlap_impact={},
        )
        crypto_time_window = SessionTimeWindow(
            start_time=time(0, 0),  # 00:00 UTC
            end_time=time(23, 59),  # 23:59 UTC
            timezone="UTC",
        )
        crypto_profile = SessionProfile(
            session_type=SessionType.CRYPTO_24H,
            time_window=crypto_time_window,
            behavior=crypto_behavior,
            description="Криптовалютная торговая сессия (24/7)",
            typical_volume_multiplier=1.0,
            typical_volatility_multiplier=1.2,
            typical_spread_multiplier=1.0,
            whale_activity_probability=0.3,
            mm_activity_probability=0.45,
            news_sensitivity=0.9,
            technical_signal_strength=0.7,
            fundamental_impact_multiplier=1.8,
            correlation_breakdown_probability=0.4,
            gap_probability=0.2,
            reversal_probability=0.4,
            continuation_probability=0.5,
            manipulation_susceptibility=0.5,
        )
        # Регистрируем профили
        self._profiles[SessionType.ASIAN] = asian_profile
        self._profiles[SessionType.LONDON] = london_profile
        self._profiles[SessionType.NEW_YORK] = ny_profile
        self._profiles[SessionType.CRYPTO_24H] = crypto_profile
        logger.info(f"Initialized {len(self._profiles)} default session profiles")

    def get_profile(self, session_type: SessionType) -> Optional[SessionProfile]:
        """Получение профиля сессии."""
        return self._profiles.get(session_type)

    def get_all_profiles(self) -> Dict[SessionType, SessionProfile]:
        """Получение всех профилей."""
        return self._profiles.copy()

    def register_profile(self, profile: SessionProfile) -> None:
        """Регистрация нового профиля."""
        self._profiles[profile.session_type] = profile
        logger.info(f"Registered session profile: {profile.session_type.value}")

    def get_active_sessions(self, timestamp: Timestamp) -> List[SessionProfile]:
        """Получение активных сессий для заданного времени."""
        active_sessions: List[SessionProfile] = []
        current_time = timestamp.to_datetime().time()
        for profile in self._profiles.values():
            if profile.time_window.is_active(current_time):
                active_sessions.append(profile)
        return active_sessions

    def get_primary_session(self, timestamp: Timestamp) -> Optional[SessionProfile]:
        """Получение основной сессии (с наибольшим объемом)."""
        active_sessions = self.get_active_sessions(timestamp)
        if not active_sessions:
            return None
        # Выбираем сессию с наибольшим множителем объема
        return max(active_sessions, key=lambda p: p.typical_volume_multiplier)

    def get_session_overlap(
        self, session1: SessionType, session2: SessionType
    ) -> float:
        """Получение перекрытия сессий."""
        profile1 = self.get_profile(session1)
        profile2 = self.get_profile(session2)
        if not profile1 or not profile2:
            return 0.0
        return self._calculate_time_window_overlap(
            profile1.time_window, profile2.time_window
        )

    def _calculate_time_window_overlap(
        self, window1: SessionTimeWindow, window2: SessionTimeWindow
    ) -> float:
        """Вычисление перекрытия временных окон."""

        # Преобразуем время в минуты для удобства вычислений
        def time_to_minutes(t: time) -> int:
            return t.hour * 60 + t.minute

        start1 = time_to_minutes(window1.start_time)
        end1 = time_to_minutes(window1.end_time)
        start2 = time_to_minutes(window2.start_time)
        end2 = time_to_minutes(window2.end_time)
        # Обрабатываем случай перехода через полночь
        if end1 < start1:
            end1 += 24 * 60
        if end2 < start2:
            end2 += 24 * 60
        # Вычисляем перекрытие
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        if overlap_end <= overlap_start:
            return 0.0
        overlap_minutes = overlap_end - overlap_start
        total_minutes = min(end1 - start1, end2 - start2)
        return overlap_minutes / total_minutes if total_minutes > 0 else 0.0

    def get_session_statistics(
        self, session_type: SessionType
    ) -> Dict[str, Union[str, float, int]]:
        """Получение статистики сессии."""
        profile = self.get_profile(session_type)
        if not profile:
            return {}
        return {
            "session_type": session_type.value,
            "description": profile.description,
            "typical_volume_multiplier": profile.typical_volume_multiplier,
            "typical_volatility_multiplier": profile.typical_volatility_multiplier,
            "typical_spread_multiplier": profile.typical_spread_multiplier,
            "whale_activity_probability": profile.whale_activity_probability,
            "mm_activity_probability": profile.mm_activity_probability,
            "news_sensitivity": profile.news_sensitivity,
            "technical_signal_strength": profile.technical_signal_strength,
            "fundamental_impact_multiplier": profile.fundamental_impact_multiplier,
            "correlation_breakdown_probability": profile.correlation_breakdown_probability,
            "gap_probability": profile.gap_probability,
            "reversal_probability": profile.reversal_probability,
            "continuation_probability": profile.continuation_probability,
            "manipulation_susceptibility": profile.manipulation_susceptibility,
        }

    def update_profile(
        self,
        session_type: SessionType,
        updates: Dict[str, Union[str, float, int, bool]],
    ) -> bool:
        """Обновление профиля сессии."""
        profile = self.get_profile(session_type)
        if not profile:
            return False
        # Создаем новый профиль с обновленными значениями
        updated_profile = profile.model_copy(update=updates)
        self._profiles[session_type] = updated_profile
        logger.info(f"Updated session profile: {session_type.value}")
        return True

    def get_session_recommendations(self, session_type: SessionType) -> List[str]:
        """Получение рекомендаций для сессии."""
        profile = self.get_profile(session_type)
        if not profile:
            return []
        recommendations: List[str] = []
        # Рекомендации на основе характеристик сессии
        if profile.typical_volume_multiplier > 1.2:
            recommendations.append("Высокий объем - ожидайте активную торговлю")
        if profile.typical_volatility_multiplier > 1.2:
            recommendations.append("Высокая волатильность - используйте стоп-лоссы")
        if profile.whale_activity_probability > 0.25:
            recommendations.append(
                "Высокая вероятность активности китов - следите за крупными сделками"
            )
        if profile.manipulation_susceptibility > 0.4:
            recommendations.append(
                "Высокая подверженность манипуляциям - будьте осторожны с пробоями"
            )
        if profile.news_sensitivity > 0.7:
            recommendations.append(
                "Высокая чувствительность к новостям - следите за экономическим календарем"
            )
        return recommendations


# Глобальный экземпляр реестра
session_registry = SessionProfileRegistry()
