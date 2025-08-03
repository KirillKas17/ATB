# -*- coding: utf-8 -*-
"""
Промышленный валидатор для инфраструктуры торговых сессий.
"""
from typing import Protocol
import pandas as pd
from domain.types.session_types import SessionAnalysisResult, SessionProfile

# Type alias для pandas
DataFrame = pd.DataFrame


class SessionValidatorProtocol(Protocol):
    """
    Протокол для валидации сессионных данных.
    """
    def validate_market_data(self, market_data: pd.DataFrame) -> bool: ...
    def validate_session_profile(self, profile: SessionProfile) -> bool: ...
    def validate_session_analysis(self, analysis: SessionAnalysisResult) -> bool: ...

class SessionValidator(SessionValidatorProtocol):
    """
    Промышленная реализация валидации сессий.
    - Проверка структуры и типов данных
    - Проверка бизнес-правил
    - Проверка на аномалии и пропуски
    """

    def validate_market_data(self, market_data: pd.DataFrame) -> bool:
        if not isinstance(market_data, pd.DataFrame):
            return False
        # Проверяем, что данные не пустые
        if market_data.empty:
            return False
        # Дополнительная проверка на наличие данных
        if not market_data.values.any():
            return False
        required_columns = {"open", "high", "low", "close", "volume"}
        if not required_columns.issubset(set(market_data.columns)):
            return False
        # Проверяем наличие пропущенных значений
        if hasattr(market_data, 'isna') and hasattr(market_data.isna(), 'any'):
            if market_data.isna().any().any():
                return False
        
        # Проверяем отрицательные значения
        numeric_columns = ["open", "high", "low", "close", "volume"]
        for col in numeric_columns:
            if col in market_data.columns:
                if (market_data[col] < 0).any():
                    return False
        return True

    def validate_session_profile(self, profile: SessionProfile) -> bool:
        if not isinstance(profile, SessionProfile):
            return False
        if not hasattr(profile, "session_type") or not hasattr(profile, "time_window"):
            return False
        if getattr(profile, "typical_volume_multiplier", 0) < 0:
            return False
        if getattr(profile, "typical_volatility_multiplier", 0) < 0:
            return False
        return True

    def validate_session_analysis(self, analysis: SessionAnalysisResult) -> bool:
        if not isinstance(analysis, SessionAnalysisResult):
            return False
        required_attrs = [
            "session_type",
            "session_phase",
            "timestamp",
            "confidence",
            "metrics",
        ]
        if not all(hasattr(analysis, attr) for attr in required_attrs):
            return False
        if not (0 <= getattr(analysis, "confidence", 0) <= 1):
            return False
        # Проверяем метрики, если они есть
        if hasattr(analysis, "metrics") and hasattr(analysis.metrics, "price_direction_bias"):
            bias = getattr(analysis.metrics, "price_direction_bias", 0)
            if not (-1 <= bias <= 1):
                return False
        return True
