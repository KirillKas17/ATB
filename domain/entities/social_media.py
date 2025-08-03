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
        if not self.content:
            raise ValueError("Post content cannot be empty")
        if self.engagement_score < 0 or self.engagement_score > 1:
            raise ValueError("Engagement score must be between 0 and 1")
        if self.reach_score < 0 or self.reach_score > 1:
            raise ValueError("Reach score must be between 0 and 1")


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
        if self.market_impact_prediction < 0 or self.market_impact_prediction > 1:
            raise ValueError("Market impact prediction must be between 0 and 1")
        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")


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
        if self.filtered_count < 0:
            raise ValueError("Filtered count cannot be negative")
        if self.filtered_count > self.total_count:
            raise ValueError("Filtered count cannot exceed total count")
        if len(self.time_range) != 2:
            raise ValueError("Time range must have exactly 2 elements")
        if self.time_range[0] >= self.time_range[1]:
            raise ValueError("Start time must be before end time")
