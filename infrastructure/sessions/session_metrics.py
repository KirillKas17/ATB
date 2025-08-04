# -*- coding: utf-8 -*-
"""
Промышленный модуль расчёта метрик для торговых сессий.
"""
from typing import Any, Dict, Protocol
import pandas as pd
from domain.type_definitions.session_types import SessionMetrics, SessionProfile


class SessionMetricsCalculatorProtocol(Protocol):
    """
    Протокол для расчёта метрик сессии.
    """
    def calculate_volume_impact(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float: ...
    def calculate_volatility_impact(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float: ...
    def calculate_direction_bias(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float: ...
    def calculate_momentum_strength(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float: ...

class SessionMetricsCalculator(SessionMetricsCalculatorProtocol):
    """
    Промышленная реализация расчёта метрик сессий.
    - SRP: каждая метрика считается отдельным методом
    - Строгая типизация
    - Расширяемость для новых стратегий
    """

    def calculate_volume_impact(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        if market_data.empty:
            return 0.0
        base = market_data["volume"].mean()
        mult = getattr(session_profile, "typical_volume_multiplier", 1.0)
        return float(base * mult)

    def calculate_volatility_impact(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        if market_data.empty:
            return 0.0
        returns = market_data["close"].pct_change().dropna()
        base_vol = returns.std()
        mult = getattr(session_profile, "typical_volatility_multiplier", 1.0)
        return float(base_vol * mult)

    def calculate_direction_bias(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        if market_data.empty:
            return 0.0
        returns = market_data["close"].pct_change().dropna()
        bias = returns.mean()
        profile_bias = getattr(session_profile, "typical_direction_bias", 0.0)
        return float(bias + profile_bias)

    def calculate_momentum_strength(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        if market_data.empty:
            return 0.0
        close = market_data["close"]
        window = min(10, len(close))
        if window < 2:
            return 0.0
        momentum = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]
        profile_momentum = getattr(session_profile, "technical_signal_strength", 1.0)
        return float(momentum * profile_momentum)
