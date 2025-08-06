"""
Новостной агент для анализа и обработки новостей.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .types import NewsConfig, NewsData, NewsMetrics


class NewsAgent(ABC):
    """Абстрактный новостной агент."""

    def __init__(self, config: Optional[NewsConfig] = None) -> None:
        self.config = config or NewsConfig()
        self.is_active = False

    @abstractmethod
    async def process_news(self, news_data: NewsData) -> NewsMetrics:
        """Обработка новостей."""

    @abstractmethod
    async def analyze_sentiment(self, text: str) -> float:
        """Анализ настроений."""

    @abstractmethod
    async def get_relevant_news(self, symbols: List[str]) -> List[NewsData]:
        """Получение релевантных новостей."""


class DefaultNewsAgent(NewsAgent):
    """Реализация новостного агента по умолчанию."""

    async def process_news(self, news_data: NewsData) -> NewsMetrics:
        """Обработка новостей."""
        try:
            sentiment = await self.analyze_sentiment(news_data.content)
            return NewsMetrics(
                sentiment_score=sentiment,
                relevance_score=0.5,
                impact_score=0.3,
                confidence=0.7,
            )
        except Exception as e:
            return NewsMetrics(
                sentiment_score=0.0,
                relevance_score=0.0,
                impact_score=0.0,
                confidence=0.0,
                error=str(e),
            )

    async def analyze_sentiment(self, text: str) -> float:
        """Анализ настроений."""
        # Простая реализация анализа настроений
        positive_words = ["bullish", "positive", "growth", "profit", "gain"]
        negative_words = ["bearish", "negative", "loss", "decline", "crash"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count == 0 and negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    async def get_relevant_news(self, symbols: List[str]) -> List[NewsData]:
        """Получение релевантных новостей."""
        # Заглушка для получения новостей
        return []
