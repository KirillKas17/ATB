"""
Модуль аналитической интеграции с модульной архитектурой.
"""

from .types import (
    AnalyticalData,
    AnalyticalIntegrationConfig,
    AnalyticalResult,
    IAnalyticalIntegrator,
    IntegrationConfig,
    MarketMakerContext,
    TradingRecommendation,
)

__all__ = [
    "AnalyticalIntegrationConfig",
    "AnalyticalResult",
    "TradingRecommendation",
    "AnalyticalData",
    "MarketMakerContext",
    "IntegrationConfig",
    "IAnalyticalIntegrator",
    "AnalyticalIntegrator",
    "MarketMakerAnalyticalIntegration",
]
