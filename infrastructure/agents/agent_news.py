"""
Новостной агент - основной файл.
Импортирует функциональность из модульной архитектуры.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

from infrastructure.agents.news_trading.integration import NewsTradingIntegration
from infrastructure.agents.news_trading.types import NewsItem, NewsAnalysis, NewsTradingConfig


@dataclass
class NewsAgent:
    """Новостной агент для анализа и торговли на основе новостей."""
    
    config: NewsTradingConfig
    integration: NewsTradingIntegration
    
    def __init__(self, config: Optional[NewsTradingConfig] = None):
        self.config = config or NewsTradingConfig()
        self.integration = NewsTradingIntegration(self.config)
    
    async def start(self) -> None:
        """Запуск новостного агента."""
        await self.integration.start()
        logger.info("NewsAgent started")
    
    async def stop(self) -> None:
        """Остановка новостного агента."""
        await self.integration.stop()
        logger.info("NewsAgent stopped")
    
    async def add_news(self, news_item: NewsItem) -> bool:
        """Добавление новости для анализа."""
        return await self.integration.add_news_item(news_item)
    
    async def get_analysis(self, symbol: str, hours: int = 24) -> List[NewsAnalysis]:
        """Получение анализа новостей для символа."""
        return await self.integration.get_news_analysis(symbol, hours)
    
    async def get_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Получение торговых сигналов на основе новостей."""
        return await self.integration.get_trading_signals(symbol)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики агента."""
        return self.integration.get_statistics()


__all__ = [
    "NewsAgent",
    "NewsSource",
    "NewsItem",
    "NewsAnalysis",
    "NewsConfig",
    "INewsProvider",
    "NewsApiProvider",
    "TwitterProvider",
    "RSSProvider",
    "SocialMediaNewsProvider",
    "SentimentAnalyzerService",
    "NewsAgentObserver",
    "PrintNewsObserver",
]
