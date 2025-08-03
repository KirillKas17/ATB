"""
Протокол для анализа настроений рынка.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from enum import Enum

class SentimentType(Enum):
    """Типы настроений."""
    VERY_BEARISH = "very_bearish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    VERY_BULLISH = "very_bullish"

class SentimentSource(Enum):
    """Источники данных для анализа настроений."""
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MARKET_DATA = "market_data"

class SentimentAnalyzerProtocol(Protocol):
    """Протокол для анализа настроений рынка."""
    
    async def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ настроения текста."""
        ...
    
    async def analyze_news_sentiment(self, symbol: str, 
                                   limit: int = 100) -> Dict[str, Any]:
        """Анализ настроения новостей по символу."""
        ...
    
    async def analyze_social_sentiment(self, symbol: str,
                                     platforms: Optional[List[str]] = None) -> Dict[str, Any]:
        """Анализ настроения в социальных сетях."""
        ...
    
    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Получение общего рыночного настроения."""
        ...
    
    async def get_sentiment_indicators(self, symbol: str) -> Dict[str, float]:
        """Получение индикаторов настроения."""
        ...
    
    async def get_fear_greed_index(self) -> float:
        """Получение индекса страха и жадности."""
        ...
    
    async def analyze_whale_movements(self, symbol: str) -> Dict[str, Any]:
        """Анализ движений крупных игроков."""
        ...
    
    async def get_sentiment_history(self, symbol: str, 
                                  days: int = 30) -> List[Dict[str, Any]]:
        """Получение истории настроений."""
        ...
    
    async def calculate_sentiment_score(self, data: Dict[str, Any]) -> float:
        """Расчет общего балла настроения."""
        ...
    
    async def detect_sentiment_shift(self, symbol: str) -> Dict[str, Any]:
        """Обнаружение изменения настроения."""
        ...

class BaseSentimentAnalyzer(ABC):
    """Базовый класс для анализатора настроений."""
    
    def __init__(self):
        self._sentiment_cache = {}
        self._last_update = {}
        self._confidence_threshold = 0.7
    
    @abstractmethod
    async def fetch_news_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Получение новостных данных."""
        pass
    
    @abstractmethod
    async def fetch_social_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Получение данных социальных сетей."""
        pass
    
    def _normalize_sentiment_score(self, score: float) -> float:
        """Нормализация балла настроения в диапазон [-1, 1]."""
        return max(-1.0, min(1.0, score))
    
    def _get_sentiment_type(self, score: float) -> SentimentType:
        """Определение типа настроения по баллу."""
        if score <= -0.6:
            return SentimentType.VERY_BEARISH
        elif score <= -0.2:
            return SentimentType.BEARISH
        elif score <= 0.2:
            return SentimentType.NEUTRAL
        elif score <= 0.6:
            return SentimentType.BULLISH
        else:
            return SentimentType.VERY_BULLISH
    
    @property
    def confidence_threshold(self) -> float:
        """Порог доверия для анализа."""
        return self._confidence_threshold
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """Установка порога доверия."""
        self._confidence_threshold = max(0.0, min(1.0, threshold))