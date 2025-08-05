"""
Интеграция новостей и социальных медиа в торговые сигналы.
Модуль объединяет сигналы от NewsAgent, SocialMediaAgent и эволюционного
новостного агента в единые торговые сигналы с направлением, силой,
уверенностью и контекстными корректировками.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from shared.numpy_utils import np
from pydantic import BaseModel

from domain.protocols.agent_protocols import (
    NewsAgentProtocol,
    NewsSentimentData,
    SocialMediaAgentProtocol,
    SocialSentimentResult,
)
from domain.value_objects.money import Money


@dataclass
class TradingSignal:
    """Торговый сигнал на основе новостей и социальных медиа."""

    direction: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    sentiment_score: float  # -1.0 - 1.0
    fear_greed_index: float  # 0.0 - 100.0
    news_impact: float  # 0.0 - 1.0
    social_impact: float  # 0.0 - 1.0
    trending_topics: List[str]
    risk_adjustment: float  # Множитель риска
    position_size_adjustment: float  # Множитель размера позиции
    timestamp: datetime
    metadata: Dict


class NewsTradingIntegration(BaseModel):
    """Интегратор новостей и социальных медиа в торговые сигналы."""

    news_agent: NewsAgentProtocol
    social_media_agent: SocialMediaAgentProtocol
    fear_greed_thresholds: Dict[str, float] = {
        "extreme_fear": 25.0,
        "fear": 45.0,
        "neutral": 55.0,
        "greed": 75.0,
        "extreme_greed": 85.0,
    }
    sentiment_weights: Dict[str, float] = {"news": 0.4, "social": 0.3, "technical": 0.3}
    min_confidence: float = 0.3
    max_risk_multiplier: float = 2.0
    min_risk_multiplier: float = 0.5

    class Config:
        arbitrary_types_allowed = True

    async def generate_trading_signal(
        self,
        symbol: str,
        current_price: Money,
        technical_sentiment: float = 0.0,
        market_volatility: float = 0.0,
    ) -> TradingSignal:
        """
        Генерирует торговый сигнал на основе новостей и социальных медиа.
        Args:
            symbol: Торговая пара
            current_price: Текущая цена
            technical_sentiment: Технический сентимент (-1.0 - 1.0)
            market_volatility: Волатильность рынка (0.0 - 1.0)
        Returns:
            TradingSignal с направлением и параметрами
        """
        try:
            # Получаем данные от агентов
            news_data = await self.news_agent.get_news_sentiment(symbol)
            social_data = await self.social_media_agent.get_social_sentiment(symbol)
            # Рассчитываем комбинированный сентимент
            combined_sentiment = self._calculate_combined_sentiment(
                news_data, social_data, technical_sentiment
            )
            # Определяем направление сигнала
            direction = self._determine_signal_direction(combined_sentiment)
            # Рассчитываем силу сигнала
            strength = self._calculate_signal_strength(
                news_data, social_data, technical_sentiment, market_volatility
            )
            # Рассчитываем уверенность
            confidence = self._calculate_confidence(
                news_data, social_data, technical_sentiment
            )
            # Рассчитываем корректировки риска и размера позиции
            risk_adjustment, position_adjustment = self._calculate_adjustments(
                social_data.fear_greed_index, combined_sentiment, market_volatility
            )
            # Собираем трендовые темы
            trending_topics = self._extract_trending_topics(news_data, social_data)
            return TradingSignal(
                direction=direction,
                strength=strength,
                confidence=confidence,
                sentiment_score=combined_sentiment,
                fear_greed_index=social_data.fear_greed_index,
                news_impact=news_data.impact_score,
                social_impact=social_data.sentiment_score,
                trending_topics=trending_topics,
                risk_adjustment=risk_adjustment,
                position_size_adjustment=position_adjustment,
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "news_count": len(news_data.news_items),
                    "social_posts_count": social_data.posts_count,
                    "market_volatility": market_volatility,
                    "technical_sentiment": technical_sentiment,
                },
            )
        except Exception as e:
            logging.error(f"Ошибка генерации торгового сигнала: {e}")
            return self._create_default_signal()

    def _calculate_combined_sentiment(
        self,
        news_data: NewsSentimentData,
        social_data: Optional[SocialSentimentResult],
        technical_sentiment: float,
    ) -> float:
        """Рассчитывает комбинированный сентимент."""
        news_sentiment = news_data.sentiment_score
        social_sentiment = social_data.sentiment_score if social_data else 0.0
        # Взвешенное среднее
        combined = (
            news_sentiment * self.sentiment_weights["news"]
            + social_sentiment * self.sentiment_weights["social"]
            + technical_sentiment * self.sentiment_weights["technical"]
        )
        return float(np.clip(combined, -1.0, 1.0))

    def _determine_signal_direction(self, sentiment: float) -> str:
        """Определяет направление сигнала на основе сентимента."""
        if sentiment > 0.2:
            return "buy"
        elif sentiment < -0.2:
            return "sell"
        else:
            return "hold"

    def _calculate_signal_strength(
        self,
        news_data: NewsSentimentData,
        social_data: Optional[SocialSentimentResult],
        technical_sentiment: float,
        market_volatility: float,
    ) -> float:
        """Рассчитывает силу сигнала."""
        # Базовая сила на основе абсолютного значения сентимента
        social_sentiment = social_data.sentiment_score if social_data else 0.0
        base_strength = abs(news_data.sentiment_score + social_sentiment) / 2
        # Усиление при высокой волатильности
        volatility_boost = min(market_volatility * 0.3, 0.3)
        # Усиление при согласованности сигналов
        agreement_boost = 0.0
        if (news_data.sentiment_score > 0 and social_sentiment > 0) or (
            news_data.sentiment_score < 0 and social_sentiment < 0
        ):
            agreement_boost = 0.2
        strength = base_strength + volatility_boost + agreement_boost
        return float(np.clip(strength, 0.0, 1.0))

    def _calculate_confidence(
        self,
        news_data: NewsSentimentData,
        social_data: Optional[SocialSentimentResult],
        technical_sentiment: float,
    ) -> float:
        """Рассчитывает уверенность в сигнале."""
        # Базовая уверенность на основе количества данных
        news_confidence = min(len(news_data.news_items) / 10.0, 1.0)
        social_confidence = social_data.confidence if social_data else 0.0
        # Средняя уверенность
        confidence = (news_confidence + social_confidence) / 2
        # Корректировка на основе согласованности
        if social_data:
            sentiment_agreement = (
                1.0 - abs(news_data.sentiment_score - social_data.sentiment_score) / 2
            )
            confidence = (confidence + sentiment_agreement) / 2
        return float(np.clip(confidence, 0.0, 1.0))

    def _calculate_adjustments(
        self, fear_greed_index: float, sentiment: float, volatility: float
    ) -> Tuple[float, float]:
        """Рассчитывает корректировки риска и размера позиции."""
        # Корректировка риска на основе индекса страха и жадности
        if fear_greed_index < self.fear_greed_thresholds["extreme_fear"]:
            risk_multiplier = self.max_risk_multiplier
        elif fear_greed_index > self.fear_greed_thresholds["extreme_greed"]:
            risk_multiplier = self.min_risk_multiplier
        else:
            risk_multiplier = 1.0
        # Корректировка размера позиции на основе сентимента и волатильности
        sentiment_boost = abs(sentiment) * 0.3
        volatility_penalty = volatility * 0.2
        position_multiplier = 1.0 + sentiment_boost - volatility_penalty
        return risk_multiplier, float(np.clip(position_multiplier, 0.5, 2.0))

    def _extract_trending_topics(
        self, news_data: NewsSentimentData, social_data: Optional[SocialSentimentResult]
    ) -> List[str]:
        """Извлекает трендовые темы."""
        topics = []
        # Темы из новостей
        if hasattr(news_data, "trending_topics"):
            topics.extend(news_data.trending_topics)
        # Темы из социальных медиа
        if social_data and social_data.trending_topics:
            topics.extend(social_data.trending_topics)
        # Убираем дубликаты и ограничиваем количество
        unique_topics = list(set(topics))
        return unique_topics[:10]

    def _create_default_signal(self) -> TradingSignal:
        """Создает сигнал по умолчанию при ошибках."""
        return TradingSignal(
            direction="hold",
            strength=0.0,
            confidence=0.0,
            sentiment_score=0.0,
            fear_greed_index=50.0,
            news_impact=0.0,
            social_impact=0.0,
            trending_topics=[],
            risk_adjustment=1.0,
            position_size_adjustment=1.0,
            timestamp=datetime.now(timezone.utc),
            metadata={"error": True},
        )

    async def get_market_sentiment_summary(self, symbol: str) -> Dict:
        """Получает сводку рыночного сентимента."""
        try:
            news_data = await self.news_agent.get_news_sentiment(symbol)
            social_data = await self.social_media_agent.get_social_sentiment(symbol)
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "news_sentiment": news_data.sentiment_score,
                "social_sentiment": social_data.sentiment_score,
                "fear_greed_index": social_data.fear_greed_index,
                "trending_topics": social_data.trending_topics,
                "news_count": len(news_data.news_items),
                "social_posts_count": social_data.posts_count,
                "black_swan_detected": news_data.black_swan_detected,
            }
        except Exception as e:
            logging.error(f"Ошибка получения сводки сентимента: {e}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc),
                "error": str(e),
            }
