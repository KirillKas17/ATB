"""
Протоколы для агентов новостей и социальных медиа.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class NewsItem:
    """Элемент новости."""

    title: str
    content: str
    source: str
    url: str
    published_at: datetime
    sentiment_score: float
    impact_score: float
    relevance_score: float
    categories: List[str]
    entities: List[str]


@dataclass
class NewsSentimentData:
    """Данные сентимента новостей."""

    sentiment_score: float  # -1.0 to 1.0
    impact_score: float  # 0.0 to 1.0
    news_items: List[NewsItem]
    trending_topics: List[str]
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    black_swan_detected: bool = False


@dataclass
class SocialSentimentResult:
    """Результат анализа социальных медиа."""

    sentiment_score: float  # -1.0 to 1.0
    fear_greed_index: float  # 0.0 to 100.0
    posts_count: int
    trending_hashtags: List[str]
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    trending_topics: Optional[List[str]] = None  # Алиас для trending_hashtags


class NewsAgentProtocol(Protocol):
    """Протокол для агента новостей."""

    async def get_news_sentiment(self, symbol: str) -> NewsSentimentData:
        """Получить сентимент новостей для символа."""
        ...

    async def get_trending_topics(self, symbol: str) -> List[str]:
        """Получить трендовые темы для символа."""
        ...

    async def get_news_impact(self, symbol: str, timeframe_hours: int = 24) -> float:
        """Получить оценку влияния новостей на символ."""
        ...


class SocialMediaAgentProtocol(Protocol):
    """Протокол для агента социальных медиа."""

    async def get_social_sentiment(self, symbol: str) -> SocialSentimentResult:
        """Получить сентимент социальных медиа для символа."""
        ...

    async def get_fear_greed_index(self, symbol: str) -> float:
        """Получить индекс страха и жадности."""
        ...

    async def get_trending_hashtags(self, symbol: str) -> List[str]:
        """Получить трендовые хештеги для символа."""
        ...


class AgentContextProtocol(Protocol):
    """Протокол для контекста агента."""

    def get_trading_config(self) -> Dict[str, Any]:
        """Получить конфигурацию торговли."""
        ...

    def get_market_service(self) -> Any:
        """Получить сервис рыночных данных."""
        ...

    def get_risk_service(self) -> Any:
        """Получить сервис управления рисками."""
        ...

    def get_portfolio_service(self) -> Any:
        """Получить сервис портфеля."""
        ...

    def get_strategy_service(self) -> Any:
        """Получить сервис стратегий."""
        ...

    def get_ml_service(self) -> Any:
        """Получить ML сервис."""
        ...

    def get_cache_service(self) -> Any:
        """Получить сервис кэша."""
        ...

    def get_logger(self) -> Any:
        """Получить логгер."""
        ...
