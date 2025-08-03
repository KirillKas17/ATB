"""
Social Media Data Providers для ATB Trading System.
"""

import re
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
from loguru import logger

from shared.logging import setup_logger

from .types import SocialPlatform, SocialPost, SocialSentimentResult

logger = setup_logger(__name__)


class ISocialMediaProvider(ABC):
    """Интерфейс для провайдеров социальных сетей"""

    @abstractmethod
    async def fetch_posts(self, symbol: str, limit: int = 100) -> List[SocialPost]:
        """Получение постов для символа"""
        pass

    @abstractmethod
    async def analyze_sentiment(self, posts: List[SocialPost]) -> SocialSentimentResult:
        """Анализ настроений постов"""
        pass


class RedditProvider(ISocialMediaProvider):
    """Провайдер для Reddit"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.subreddits = config.get(
            "subreddits", ["cryptocurrency", "bitcoin", "ethereum"]
        )
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.user_agent = config.get("user_agent", "ATB-Trading-Bot/1.0")
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def fetch_posts(self, symbol: str, limit: int = 100) -> List[SocialPost]:
        """Получение постов из Reddit"""
        try:
            if not self.client_id or not self.client_secret:
                logger.warning("Reddit credentials not configured")
                return []
            session = await self._get_session()
            posts = []
            for subreddit in self.subreddits:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        "q": symbol,
                        "limit": min(limit, 25),
                        "sort": "hot",
                        "t": "day",
                    }
                    headers = {"User-Agent": self.user_agent}
                    async with session.get(
                        url, params=params, headers=headers
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            for post_data in data.get("data", {}).get("children", []):
                                post = post_data.get("data", {})
                                social_post = SocialPost(
                                    platform=SocialPlatform.REDDIT,
                                    author=post.get("author", "unknown"),
                                    content=f"{post.get('title', '')} {post.get('selftext', '')}",
                                    timestamp=datetime.fromtimestamp(
                                        post.get("created_utc", 0)
                                    ),
                                    engagement={
                                        "upvotes": post.get("ups", 0),
                                        "downvotes": post.get("downs", 0),
                                        "comments": post.get("num_comments", 0),
                                    },
                                    url=f"https://reddit.com{post.get('permalink', '')}",
                                    post_id=post.get("id", ""),
                                )
                                posts.append(social_post)
                except Exception as e:
                    logger.error(f"Error fetching from r/{subreddit}: {e}")
                    continue
            return posts
        except Exception as e:
            logger.error(f"Error in Reddit fetch_posts: {e}")
            return []

    async def analyze_sentiment(self, posts: List[SocialPost]) -> SocialSentimentResult:
        """Анализ настроений Reddit постов"""
        try:
            if not posts:
                return SocialSentimentResult(
                    platform=SocialPlatform.REDDIT,
                    symbol="",
                    sentiment_score=0.0,
                    confidence=0.0,
                    volume=0,
                    trending_topics=[],
                    fear_greed_index=50.0,
                )
            positive_words = {
                "moon",
                "bull",
                "bullish",
                "pump",
                "buy",
                "hodl",
                "diamond",
                "rocket",
            }
            negative_words = {
                "dump",
                "bear",
                "bearish",
                "sell",
                "crash",
                "fud",
                "scam",
                "dead",
            }
            total_sentiment = 0.0
            for post in posts:
                content_lower = post.content.lower()
                positive_count = sum(
                    1 for word in positive_words if word in content_lower
                )
                negative_count = sum(
                    1 for word in negative_words if word in content_lower
                )
                if positive_count + negative_count > 0:
                    sentiment = (positive_count - negative_count) / (
                        positive_count + negative_count
                    )
                else:
                    sentiment = 0.0
                engagement_score = post.engagement.get(
                    "upvotes", 0
                ) - post.engagement.get("downvotes", 0)
                total_sentiment += sentiment * (1 + abs(engagement_score) / 100)
            avg_sentiment = total_sentiment / len(posts) if posts else 0.0
            confidence = min(1.0, len(posts) / 50.0)
            fear_greed = 50.0 + (avg_sentiment * 50.0)
            return SocialSentimentResult(
                platform=SocialPlatform.REDDIT,
                symbol=posts[0].post_id.split("_")[0] if posts else "",
                sentiment_score=avg_sentiment,
                confidence=confidence,
                volume=len(posts),
                trending_topics=self._extract_trending_topics(posts),
                fear_greed_index=fear_greed,
            )
        except Exception as e:
            logger.error(f"Error in Reddit analyze_sentiment: {e}")
            return SocialSentimentResult(
                platform=SocialPlatform.REDDIT,
                symbol="",
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                trending_topics=[],
                fear_greed_index=50.0,
            )

    def _extract_trending_topics(self, posts: List[SocialPost]) -> List[str]:
        """Извлечение трендовых тем"""
        try:
            all_words = []
            for post in posts:
                words = re.findall(r"\b\w+\b", post.content.lower())
                all_words.extend(words)
            # Подсчет частоты слов
            word_count: Dict[str, int] = {}
            for word in all_words:
                if len(word) > 3:  # Исключаем короткие слова
                    word_count[word] = word_count.get(word, 0) + 1
            # Возвращаем топ-5 слов
            sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
            return [word for word, count in sorted_words[:5]]
        except Exception as e:
            logger.error(f"Error extracting trending topics: {e}")
            return []


class TelegramProvider(ISocialMediaProvider):
    """Провайдер для Telegram"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config.get("bot_token")
        self.channels = config.get("channels", [])
        self.session: Optional[aiohttp.ClientSession] = None

    async def fetch_posts(self, symbol: str, limit: int = 100) -> List[SocialPost]:
        """Получение постов из Telegram"""
        try:
            if not self.bot_token:
                logger.warning("Telegram bot token not configured")
                return []
            # Упрощенная реализация - в реальном проекте нужен Telegram API
            return []
        except Exception as e:
            logger.error(f"Error in Telegram fetch_posts: {e}")
            return []

    async def analyze_sentiment(self, posts: List[SocialPost]) -> SocialSentimentResult:
        """Анализ настроений Telegram постов"""
        try:
            return SocialSentimentResult(
                platform=SocialPlatform.TELEGRAM,
                symbol="",
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                trending_topics=[],
                fear_greed_index=50.0,
            )
        except Exception as e:
            logger.error(f"Error in Telegram analyze_sentiment: {e}")
            return SocialSentimentResult(
                platform=SocialPlatform.TELEGRAM,
                symbol="",
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                trending_topics=[],
                fear_greed_index=50.0,
            )


class DiscordProvider(ISocialMediaProvider):
    """Провайдер для Discord"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.bot_token = config.get("bot_token")
        self.channels = config.get("channels", [])

    async def fetch_posts(self, symbol: str, limit: int = 100) -> List[SocialPost]:
        """Получение постов из Discord"""
        try:
            if not self.bot_token:
                logger.warning("Discord bot token not configured")
                return []
            # Упрощенная реализация - в реальном проекте нужен Discord API
            return []
        except Exception as e:
            logger.error(f"Error in Discord fetch_posts: {e}")
            return []

    async def analyze_sentiment(self, posts: List[SocialPost]) -> SocialSentimentResult:
        """Анализ настроений Discord постов"""
        try:
            return SocialSentimentResult(
                platform=SocialPlatform.DISCORD,
                symbol="",
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                trending_topics=[],
                fear_greed_index=50.0,
            )
        except Exception as e:
            logger.error(f"Error in Discord analyze_sentiment: {e}")
            return SocialSentimentResult(
                platform=SocialPlatform.DISCORD,
                symbol="",
                sentiment_score=0.0,
                confidence=0.0,
                volume=0,
                trending_topics=[],
                fear_greed_index=50.0,
            )
