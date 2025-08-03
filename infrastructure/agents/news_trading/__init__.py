"""
Модуль интеграции новостной торговли с модульной архитектурой.
"""

from .types import (
    NewsAnalysis,
    NewsCategory,
    NewsImpact,
    NewsItem,
    NewsTradingConfig,
    TradingSignal,
)

__all__ = [
    "NewsCategory",
    "NewsImpact",
    "TradingSignal",
    "NewsItem",
    "NewsAnalysis",
    "NewsTradingConfig",
    "NewsTradingIntegration",
]
