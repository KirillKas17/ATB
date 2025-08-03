"""
Типы данных для интеграции новостной торговли.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NewsCategory(Enum):
    """Категории новостей."""

    REGULATORY = "regulatory"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MARKET = "market"
    POLITICAL = "political"
    ECONOMIC = "economic"


class NewsImpact(Enum):
    """Уровни влияния новостей."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingSignal(Enum):
    """Торговые сигналы."""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"


@dataclass
class NewsItem:
    """Новостной элемент."""

    news_id: str
    title: str
    content: str
    source: str
    category: NewsCategory
    impact: NewsImpact
    timestamp: datetime
    symbols: List[str] = field(default_factory=list)
    sentiment_score: float = 0.0
    confidence: float = 0.0
    url: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsAnalysis:
    """Анализ новостей."""

    analysis_id: str
    news_id: str
    symbol: str
    sentiment_score: float
    impact_score: float
    trading_signal: TradingSignal
    confidence: float
    reasoning: List[str]
    timestamp: datetime
    price_impact: Optional[float] = None
    volume_impact: Optional[float] = None


@dataclass
class NewsTradingConfig:
    """Конфигурация новостной торговли."""

    enable_news_trading: bool = True
    sentiment_threshold: float = 0.3
    impact_threshold: float = 0.5
    max_news_age_hours: int = 24
    enable_auto_trading: bool = False
    trading_delay_seconds: int = 30
    log_news_analysis: bool = True
