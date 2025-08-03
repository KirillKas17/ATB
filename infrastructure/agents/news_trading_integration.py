"""
Интеграция новостной торговли - основной файл.
Импортирует функциональность из модульной архитектуры.
"""

from .news_trading.types import (
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
