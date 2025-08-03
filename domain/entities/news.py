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
        if not self.title:
            raise ValueError("News title cannot be empty")
        if not self.content:
            raise ValueError("News content cannot be empty")
        if self.relevance_score < 0 or self.relevance_score > 1:
            raise ValueError("Relevance score must be between 0 and 1")
        if self.impact_score < 0 or self.impact_score > 1:
            raise ValueError("Impact score must be between 0 and 1")


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
        if self.market_impact_prediction < 0 or self.market_impact_prediction > 1:
            raise ValueError("Market impact prediction must be between 0 and 1")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")


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
        if self.filtered_count < 0:
            raise ValueError("Filtered count cannot be negative")
        if self.filtered_count > self.total_count:
            raise ValueError("Filtered count cannot exceed total count")
        if len(self.time_range) != 2:
            raise ValueError("Time range must have exactly 2 elements")
        if self.time_range[0] >= self.time_range[1]:
            raise ValueError("Start time must be before end time")
