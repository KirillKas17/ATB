"""
Типы данных для новостного агента.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class NewsSource(Enum):
    """Источник новостей"""

    NEWS = "news"
    RSS = "rss"
    TWITTER = "twitter"
    REDDIT = "reddit"
    SOCIAL_MEDIA = "social_media"


@dataclass
class NewsItem:
    """
    Класс для хранения информации о новости
    """
    timestamp: datetime
    source: NewsSource
    title: str
    content: str
    sentiment: float  # от -1 до 1
    impact_score: float  # от 0 до 1
    keywords: List[str] = field(default_factory=list)
    url: str = ""
    hash: str = ""
    social_engagement: Dict[str, int] = field(default_factory=dict)
    fear_greed_index: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class NewsAnalysis:
    """Результат анализа новостей"""

    symbol: str
    sentiment_score: float
    impact_score: float
    trending_topics: List[str]
    fear_greed_index: float
    timestamp: datetime
    confidence: float = 0.0


@dataclass
class NewsConfig:
    """Конфигурация новостного агента"""

    api_key: str = ""
    rss_feeds: List[str] = field(default_factory=list)
    twitter_api_key: str = ""
    sentiment_threshold: float = 0.3
    impact_threshold: float = 0.5
    cache_ttl: int = 300
    enable_social_media: bool = True
    relevance_threshold: float = 0.5
    max_news_age_hours: int = 24
    enable_sentiment_analysis: bool = True


@dataclass
class NewsAnalysisResult:
    """Результат анализа новостей для интеграции с системой"""

    timestamp: datetime
    pair: str
    sentiment: str  # 'positive', 'negative', 'neutral', 'mixed'
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    source: str  # 'official', 'social_media', 'analyst', 'rumor'
    confidence: float
    keywords: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    correlation_with_price: float = 0.0
    expected_impact: str = ""
    trading_recommendations: List[str] = field(default_factory=list)


@dataclass
class NewsData:
    """Данные новости."""

    id: str
    title: str
    content: str
    source: str
    timestamp: datetime
    symbols: List[str] = field(default_factory=list)
    category: str = ""
    url: str = ""


@dataclass
class NewsMetrics:
    """Метрики новостей."""

    sentiment_score: float = 0.0
    relevance_score: float = 0.0
    impact_score: float = 0.0
    confidence: float = 0.0
    error: Optional[str] = None
