# -*- coding: utf-8 -*-
"""
Промышленный модуль оптимизации параметров торговых сессий.
"""
from typing import Any, Dict, Optional, Protocol

from domain.type_definitions.session_types import SessionProfile, SessionType


class SessionOptimizerProtocol(Protocol):
    """Протокол для оптимизации параметров сессий."""

    def optimize_weights(
        self, profiles: Dict[SessionType, SessionProfile], target: Optional[str] = None
    ) -> Dict[SessionType, float]: ...
    def optimize_schedule(
        self, profiles: Dict[SessionType, SessionProfile]
    ) -> Dict[SessionType, Any]: ...
class SessionOptimizer(SessionOptimizerProtocol):
    """
    Промышленная реализация оптимизации параметров сессий.
    - Строгая типизация
    - Расширяемость стратегий
    """

    def optimize_weights(
        self, profiles: Dict[SessionType, SessionProfile], target: Optional[str] = None
    ) -> Dict[SessionType, float]:
        # Пример: равномерное распределение весов
        n = len(profiles)
        if n == 0:
            return {}
        weights = {stype: 1.0 / n for stype in profiles}
        return weights

    def optimize_schedule(
        self, profiles: Dict[SessionType, SessionProfile]
    ) -> Dict[SessionType, Any]:
        # Пример: возвращает текущее расписание (stub)
        return {
            stype: getattr(profile, "time_window", None)
            for stype, profile in profiles.items()
        }
