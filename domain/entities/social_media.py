"""
Сущности для работы с социальными медиа.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Tuple


class SocialSentiment(Enum):
    """Типы настроений в социальных медиа."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SocialPlatform(Enum):
    """Платформы социальных медиа."""

    TWITTER = "twitter"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    TWITCH = "twitch"
    YOUTUBE = "youtube"
    OTHER = "other"


@dataclass
class SocialPost:
    """Сущность поста в социальных медиа."""

    id: str
    platform: SocialPlatform
    author: str
    content: str
    published_at: datetime
    sentiment: SocialSentiment
    engagement_score: float
    reach_score: float
    symbols: List[str]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.id:
            raise ValueError("Post ID cannot be empty")


@dataclass
class SocialAnalysis:
    """Результат анализа социальных медиа."""

    post: SocialPost
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
class SocialCollection:
    """Коллекция постов из социальных медиа."""

    posts: List[SocialPost]
    total_count: int
    filtered_count: int
    time_range: Tuple[datetime, datetime]
    platform: SocialPlatform

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.total_count < 0:
            raise ValueError("Total count cannot be negative")
