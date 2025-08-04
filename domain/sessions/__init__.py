# -*- coding: utf-8 -*-
"""Модуль анализа торговых сессий."""
from typing import Any, Type

# Импортируем типы из централизованного модуля
from domain.type_definitions.session_types import (
    ConfidenceScore,
    CorrelationScore,
    InfluenceType,
    LiquidityProfile,
    MarketConditions,
    MarketRegime,
    PriceDirection,
    SessionAnalysisResult,
    SessionBehavior,
    SessionId,
    SessionIntensity,
    SessionMetrics,
    SessionPhase,
    SessionProfile,
    SessionTimeWindow,
    SessionTransition,
    SessionType,
    Symbol,
    VolatilityMultiplier,
    VolumeMultiplier,
)

from .session_influence_analyzer import (
    SessionInfluenceAnalyzer,
    SessionInfluenceMetrics,
    SessionInfluenceResult,
)

# Импортируем pandas типы для исправления проблем типизации
try:
    from pandas import DataFrame, Series
    from pandas.core.frame import DataFrame as PandasDataFrame
    from pandas.core.series import Series as PandasSeries
except ImportError:
    pass
__all__ = [
    # Основные классы
    "SessionProfile",
    "SessionType",
    "SessionPhase",
    "SessionBehavior",
    "SessionTimeWindow",
    "SessionInfluenceAnalyzer",
    "SessionInfluenceResult",
    "SessionInfluenceMetrics",
    "SessionMarker",
    "MarketSessionContext",
    "SessionState",
    "SessionProfileRegistry",
    # Типы и перечисления
    "MarketRegime",
    "SessionIntensity",
    "LiquidityProfile",
    "InfluenceType",
    "PriceDirection",
    # Типизированные структуры
    "SessionMetrics",
    "MarketConditions",
    "SessionTransition",
    "SessionAnalysisResult",
    # Новые типы
    "ConfidenceScore",
    "CorrelationScore",
    "SessionId",
    "Symbol",
    "VolumeMultiplier",
    "VolatilityMultiplier",
    # Pandas типы
    "Series",
    "DataFrame",
    "PandasSeries",
    "PandasDataFrame",
]
