# -*- coding: utf-8 -*-
"""
Промышленный модуль анализа паттернов поведения торговых сессий.
"""
from typing import Any, Dict, List, Protocol
import pandas as pd
from domain.types.session_types import SessionProfile


class SessionPatternRecognizerProtocol(Protocol):
    """
    Протокол для поиска и анализа паттернов сессий.
    """
    def identify_session_patterns(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> List[str]: ...
    def calculate_pattern_probability(
        self, pattern: str, session_profile: SessionProfile
    ) -> float: ...
    def get_historical_patterns(
        self, session_type: str, lookback_days: int
    ) -> List[Dict[str, Any]]: ...

class SessionPatternRecognizer(SessionPatternRecognizerProtocol):
    """
    Промышленная реализация поиска и анализа паттернов сессий.
    - SRP: каждая задача реализована отдельным методом
    - Строгая типизация
    - Расширяемость для новых паттернов
    """

    def identify_session_patterns(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> List[str]:
        patterns: List[str] = []
        if market_data.empty:
            return patterns
        # Пример: простая логика для reversal/breakout/consolidation
        if self._detect_reversal_pattern(market_data):
            patterns.append("reversal")
        if self._detect_breakout_pattern(market_data):
            patterns.append("breakout")
        if self._detect_consolidation_pattern(market_data):
            patterns.append("consolidation")
        return patterns

    def calculate_pattern_probability(
        self, pattern: str, session_profile: SessionProfile
    ) -> float:
        # Пример: вероятность паттерна зависит от профиля сессии
        if pattern == "reversal":
            return float(getattr(session_profile, "reversal_probability", 0.2))
        if pattern == "breakout":
            return float(getattr(session_profile, "continuation_probability", 0.3))
        if pattern == "consolidation":
            return 1.0 - float(getattr(session_profile, "reversal_probability", 0.2))
        return 0.1

    def get_historical_patterns(
        self, session_type: str, lookback_days: int
    ) -> List[Dict[str, Any]]:
        # Заглушка: в реальной реализации — запрос к БД/истории
        return []

    def _detect_reversal_pattern(self, market_data: pd.DataFrame) -> bool:
        # Пример: резкий разворот цены
        returns = market_data["close"].pct_change().dropna()
        if len(returns) < 5:
            return False
        return bool((returns.iloc[-1] * returns.iloc[-2] < 0) and (
            abs(returns.iloc[-1]) > 0.01
        ))

    def _detect_breakout_pattern(self, market_data: pd.DataFrame) -> bool:
        # Пример: пробой диапазона
        if len(market_data) < 10:
            return False
        recent_high = market_data["high"].iloc[-10:].max()
        recent_low = market_data["low"].iloc[-10:].min()
        last_close = market_data["close"].iloc[-1]
        return bool(last_close > recent_high or last_close < recent_low)

    def _detect_consolidation_pattern(self, market_data: pd.DataFrame) -> bool:
        # Пример: узкий диапазон
        if len(market_data) < 10:
            return False
        high = market_data["high"].iloc[-10:].max()
        low = market_data["low"].iloc[-10:].min()
        return bool((high - low) / high < 0.01)
