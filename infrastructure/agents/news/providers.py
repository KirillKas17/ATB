"""
Провайдеры новостей для новостного агента.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import aiohttp
import feedparser
import pandas as pd  # type: ignore

from infrastructure.agents.social_media.agent_social_media import SocialMediaAgent

from shared.logging import setup_logger

from .types import NewsItem, NewsSource

# Type aliases для pandas
DataFrame = pd.DataFrame  # type: ignore
Series = pd.Series  # type: ignore

logger = setup_logger(__name__)


class INewsProvider(ABC):
    """Интерфейс провайдера новостей"""

    @abstractmethod
    async def fetch_news(self, pair: str) -> List[NewsItem]:
        """Получение новостей для указанной пары"""


class NewsApiProvider(INewsProvider):
    """Провайдер новостей через NewsAPI"""

    def __init__(self, api_key: str):
        self.api_key = str(api_key)

    async def fetch_news(self, pair: str) -> List[NewsItem]:
        try:
            url = f"https://newsapi.org/v2/everything?q={pair}&apiKey={self.api_key}&language=en"
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as resp:
                    data = await resp.json()
                    news_items: List[NewsItem] = []
                    for article in data.get("articles", []):
                        news_items.append(
                            NewsItem(
                                timestamp=datetime.fromisoformat(
                                    article.get(
                                        "publishedAt", datetime.now().isoformat()
                                    )
                                ),
                                source=NewsSource.NEWS,
                                title=article.get("title", ""),
                                content=article.get("description", ""),
                                sentiment=0.0,
                                impact_score=0.0,
                                url=article.get("url", ""),
                                keywords=[],
                            )
                        )
                    return news_items
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return []


class TwitterProvider(INewsProvider):
    """Провайдер новостей через Twitter API"""

    def __init__(self, api_key: str):
        self.api_key = str(api_key)

    async def fetch_news(self, pair: str) -> List[NewsItem]:
        try:
            # Пример через Tweepy или requests (здесь — заглушка для реального API)
            # В реальном проекте — использовать официальный Twitter API v2
            url = f"https://api.twitter.com/2/tweets/search/recent?query={pair}&tweet.fields=created_at,text&max_results=10"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    data = await resp.json()
                    news_items: List[NewsItem] = []
                    for tweet in data.get("data", []):
                        news_items.append(
                            NewsItem(
                                timestamp=datetime.fromisoformat(
                                    tweet.get("created_at", datetime.now().isoformat())
                                ),
                                source=NewsSource.TWITTER,
                                title=f"Tweet about {pair}",
                                content=tweet.get("text", ""),
                                sentiment=0.0,
                                impact_score=0.0,
                                url="",
                                keywords=[],
                            )
                        )
                    return news_items
        except Exception as e:
            logger.error(f"Error fetching news from Twitter: {str(e)}")
            return []


class RSSProvider(INewsProvider):
    """Провайдер новостей через RSS"""

    def __init__(self, feeds: List[str]):
        self.feeds = [str(feed) for feed in feeds]

    async def fetch_news(self, pair: str) -> List[NewsItem]:
        try:
            news_items: List[NewsItem] = []
            for feed_url in self.feeds:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries:
                    try:
                        # Приведение типов из FeedParserDict
                        title = str(entry.get("title", ""))
                        content = str(entry.get("description", ""))
                        url = str(entry.get("link", ""))
                        published = entry.get("published_parsed")
                        timestamp = (
                            datetime(*published[:6]) if published else datetime.now()
                        )

                        news_item = NewsItem(
                            timestamp=timestamp,
                            source=NewsSource.RSS,
                            title=title,
                            content=content,
                            sentiment=0.0,  # Будет рассчитано позже
                            impact_score=0.0,  # Будет рассчитано позже
                            url=url,
                            keywords=[],  # Будет заполнено позже
                        )
                        news_items.append(news_item)
                    except Exception as e:
                        logger.error(f"Error processing RSS entry: {str(e)}")
                        continue
            return news_items
        except Exception as e:
            logger.error(f"Error fetching RSS news: {str(e)}")
            return []


class SocialMediaNewsProvider(INewsProvider):
    """Провайдер новостей из социальных сетей"""

    def __init__(self, social_media_agent: SocialMediaAgent):
        self.social_media_agent = social_media_agent

    async def fetch_news(self, pair: str) -> List[NewsItem]:
        try:
            # Получаем настроения из социальных сетей
            social_result = await self.social_media_agent.get_social_sentiment(pair)

            if not social_result:
                return []

            # Создаем новостной элемент на основе социальных настроений
            news_item = NewsItem(
                timestamp=social_result.timestamp,
                source=NewsSource.SOCIAL_MEDIA,
                title=f"Social Media Sentiment for {pair}",
                content=f"Social media analysis: {social_result.sentiment_score:.3f} sentiment, {social_result.volume} posts",
                sentiment=social_result.sentiment_score,
                impact_score=min(
                    social_result.volume / 1000.0, 1.0
                ),  # Нормализуем объем
                keywords=social_result.keywords or [],
                social_engagement={
                    "posts": social_result.volume,
                    "sentiment": int(social_result.sentiment_score * 100),
                },
            )

            return [news_item]

        except Exception as e:
            logger.error(f"Error fetching social media news: {str(e)}")
            return []
