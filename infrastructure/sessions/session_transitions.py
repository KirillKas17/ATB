# -*- coding: utf-8 -*-
"""
Промышленный модуль управления переходами между торговыми сессиями.
"""
from typing import List
from datetime import timedelta

from domain.sessions.interfaces import BaseSessionTransitionManager, SessionRegistry
from domain.types.session_types import (
    SessionMetrics,
    SessionPhase,
    SessionTransition,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp


class SessionTransitionManager(BaseSessionTransitionManager):
    """
    Промышленная реализация менеджера переходов между сессиями.
    - Строгая типизация
    - SRP: каждая задача реализована отдельным методом
    - Полная реализация BaseSessionTransitionManager
    """

    def __init__(self, registry: SessionRegistry):
        super().__init__(registry)

    def get_active_transitions(self, timestamp: Timestamp) -> List[SessionTransition]:
        transitions: List[SessionTransition] = []
        current_time = timestamp.to_datetime()
        all_profiles = self.registry.get_all_profiles()
        for session_type, profile in all_profiles.items():
            # Пример: если сейчас не активна, а через 30 минут будет активна — это переход
            if not profile.time_window.is_active(current_time.time()):
                future_time = (current_time + timedelta(minutes=30)).time()
                if profile.time_window.is_active(future_time):
                    transitions.append(
                        SessionTransition(
                            from_session=session_type,
                            to_session=session_type,
                            transition_duration_minutes=30,
                            volume_decay_rate=0.8,
                            volatility_spike_probability=0.5,
                            gap_probability=0.2,
                            correlation_shift_probability=0.3,
                            liquidity_drain_rate=0.7,
                            manipulation_window_minutes=15,
                        )
                    )
        return transitions

    def calculate_transition_metrics(
        self, transition: SessionTransition, current_metrics: SessionMetrics
    ) -> SessionMetrics:
        # Пример: корректировка метрик на период перехода
        return SessionMetrics(
            volume_change_percent=current_metrics["volume_change_percent"]
            * transition["volume_decay_rate"],
            volatility_change_percent=current_metrics["volatility_change_percent"]
            * (1 + transition["volatility_spike_probability"]),
            price_direction_bias=current_metrics["price_direction_bias"],
            momentum_strength=current_metrics["momentum_strength"] * 0.8,
            false_breakout_probability=current_metrics["false_breakout_probability"]
            * 1.1,
            reversal_probability=current_metrics["reversal_probability"] * 1.1,
            trend_continuation_probability=current_metrics[
                "trend_continuation_probability"
            ]
            * 0.9,
            influence_duration_minutes=transition["transition_duration_minutes"],
            peak_influence_time_minutes=transition["transition_duration_minutes"] // 2,
            spread_impact=current_metrics["spread_impact"] * 1.2,
            liquidity_impact=current_metrics["liquidity_impact"]
            * transition["liquidity_drain_rate"],
            correlation_with_other_sessions=current_metrics[
                "correlation_with_other_sessions"
            ]
            * (1 - transition["correlation_shift_probability"]),
        )
