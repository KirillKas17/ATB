"""
Типы данных для агента социальных сетей.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List


class SocialPlatform(Enum):
    """Платформы социальных сетей"""

    REDDIT = "reddit"
    TELEGRAM = "telegram"
    DISCORD = "discord"
    TWITTER = "twitter"


@dataclass
class SocialPost:
    """Пост из социальной сети"""

    platform: SocialPlatform
    author: str
    content: str
    timestamp: datetime
    sentiment: float = 0.0
    impact_score: float = 0.0
    engagement: Dict[str, int] = field(default_factory=dict)
    keywords: List[str] = field(default_factory=list)
    url: str = ""
    post_id: str = ""


@dataclass
class SocialSentimentResult:
    """Результат анализа настроений в социальных сетях"""

    platform: SocialPlatform
    symbol: str
    sentiment_score: float  # -1 до 1
    confidence: float  # 0 до 1
    volume: int  # количество постов
    trending_topics: List[str]
    fear_greed_index: float  # 0 до 100
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SocialMediaConfig:
    """Конфигурация агента социальных сетей"""

    cache_ttl: int = 300  # секунды
    max_posts_per_request: int = 100
    sentiment_threshold: float = 0.3
    platforms: List[str] = field(default_factory=lambda: ["reddit"])
    reddit_config: Dict[str, Any] = field(default_factory=dict)
    telegram_config: Dict[str, Any] = field(default_factory=dict)
    discord_config: Dict[str, Any] = field(default_factory=dict)
    twitter_config: Dict[str, Any] = field(default_factory=dict)
