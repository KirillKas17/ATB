"""
Сущности для работы с новостями.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple


class NewsSentiment(Enum):
    """Типы настроений новостей."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class NewsCategory(Enum):
    """Категории новостей."""

    ECONOMIC = "economic"
    POLITICAL = "political"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    MARKET = "market"
    OTHER = "other"


@dataclass
class NewsItem:
    """Сущность новостного элемента."""

    id: str
    title: str
    content: str
    source: str
    published_at: datetime
    sentiment: NewsSentiment
    category: NewsCategory
    relevance_score: float
    impact_score: float
    symbols: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.id:
            raise ValueError("News ID cannot be empty")


@dataclass
class NewsAnalysis:
    """Результат анализа новостей."""

    news_item: NewsItem
    sentiment_score: float
    market_impact_prediction: float
    confidence: float
    trading_signals: List[str]
    risk_level: str

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.sentiment_score < -1 or self.sentiment_score > 1:
            raise ValueError("Sentiment score must be between -1 and 1")


@dataclass
class NewsCollection:
    """Коллекция новостей."""

    items: List[NewsItem]
    total_count: int
    filtered_count: int
    time_range: Tuple[datetime, datetime]

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.total_count < 0:
            raise ValueError("Total count cannot be negative")
