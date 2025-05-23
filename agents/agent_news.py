import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

import feedparser
import nltk
import numpy as np
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer as VaderSentimentAnalyzer
from transformers import pipeline as transformers_pipeline

from utils.logger import setup_logger

logger = setup_logger(__name__)


class NewsSource(Enum):
    """Источник новостей"""

    NEWS = "news"
    RSS = "rss"
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class NewsItem:
    """Класс для хранения информации о новости"""

    timestamp: datetime
    source: NewsSource
    title: str
    content: str
    sentiment: float  # от -1 до 1
    impact_score: float  # от 0 до 1
    keywords: List[str] = field(default_factory=list)
    url: str = ""
    hash: str = ""

    def __post_init__(self):
        """Проверка и приведение типов после инициализации"""
        self.timestamp = (
            datetime.fromisoformat(self.timestamp)
            if isinstance(self.timestamp, str)
            else self.timestamp
        )
        self.source = (
            NewsSource(self.source.value)
            if isinstance(self.source, str)
            else self.source
        )
        self.title = str(self.title)
        self.content = str(self.content)
        self.sentiment = float(self.sentiment)
        self.impact_score = float(self.impact_score)
        self.keywords = [str(k) for k in self.keywords]
        self.url = str(self.url)
        self.hash = str(self.hash)


class INewsProvider(ABC):
    @abstractmethod
    async def fetch_news(self, pair: str) -> List[NewsItem]:
        pass


class NewsApiProvider(INewsProvider):
    def __init__(self, api_key: str):
        self.api_key = str(api_key)

    async def fetch_news(self, pair: str) -> List[NewsItem]:
        try:
            # ... реализация асинхронного запроса к NewsAPI ...
            return []
        except Exception as e:
            logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return []


class TwitterProvider(INewsProvider):
    def __init__(self, api_key: str):
        self.api_key = str(api_key)

    async def fetch_news(self, pair: str) -> List[NewsItem]:
        try:
            # ... реализация асинхронного запроса к Twitter API ...
            return []
        except Exception as e:
            logger.error(f"Error fetching news from Twitter: {str(e)}")
            return []


class RSSProvider(INewsProvider):
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


class LRUCache:
    """Кэш с LRU политикой вытеснения"""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.cache: Dict[str, Any] = {}
        self.order: List[str] = []

    def __getitem__(self, key: str) -> Any:
        if key in self.cache:
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None

    def __setitem__(self, key: str, value: Any) -> None:
        key = str(key)
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.cache.get(str(key), default)


class SentimentAnalyzerService:
    def __init__(self):
        self.nltk_analyzer: Optional[VaderSentimentAnalyzer] = None
        self.transformer_analyzer: Optional[Callable[[str], List[Dict[str, Any]]]] = (
            None
        )

        try:
            nltk.download("vader_lexicon", quiet=True)
            self.nltk_analyzer = VaderSentimentAnalyzer()
        except Exception as e:
            logger.error(f"Error initializing NLTK: {str(e)}")

        try:
            self.transformer_analyzer = transformers_pipeline(
                "sentiment-analysis",
                model="finiteautomata/bertweet-base-sentiment-analysis",
            )
        except Exception as e:
            logger.error(f"Error initializing transformers: {str(e)}")

    def analyze(self, text: str) -> float:
        try:
            text = str(text)
            scores: List[float] = []

            if self.nltk_analyzer:
                nltk_score = self.nltk_analyzer.polarity_scores(text)
                scores.append(float(nltk_score["compound"]))

            if self.transformer_analyzer:
                transformer_score = self.transformer_analyzer(text)[0]
                # Преобразуем метку в числовой score
                label_to_score = {"POS": 1.0, "NEU": 0.0, "NEG": -1.0}
                scores.append(
                    float(label_to_score.get(transformer_score["label"], 0.0))
                )

            return float(np.mean(scores)) if scores else 0.0
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0


class NewsAgentObserver(ABC):
    @abstractmethod
    def notify(self, event: str, data: Any) -> None:
        pass


class NewsAgent:
    """
    Агент анализа новостей: асинхронный сбор, фильтрация, скоринг, black swan, подписка на события.
    TODO: Вынести работу с API, кэшированием, анализом в отдельные классы/модули (SRP).
    """

    config: Dict[str, Any]
    providers: List[INewsProvider]
    cache_service: LRUCache
    sentiment_service: SentimentAnalyzerService
    observers: List[NewsAgentObserver]
    news_history: Dict[str, List[NewsItem]]
    black_swan_alerts: List[Dict]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента анализа новостей.
        :param config: словарь параметров
        """
        self.config = config or {
            "news_api_key": "",
            "twitter_api_key": "",
            "rss_feeds": [
                "https://cointelegraph.com/rss",
                "https://coindesk.com/arc/outboundfeeds/rss/",
                "https://www.theblock.co/rss.xml",
            ],
            "keywords": [
                "bitcoin",
                "ethereum",
                "crypto",
                "blockchain",
                "defi",
                "nft",
                "regulation",
                "ban",
                "adoption",
            ],
            "sentiment_threshold": 0.3,
            "impact_threshold": 0.5,
            "cache_size": 1000,
            "black_swan_keywords": [
                "hack",
                "exploit",
                "ban",
                "regulation",
                "collapse",
                "bankruptcy",
                "scam",
            ],
        }
        self.providers = []
        if self.config.get("news_api_key"):
            self.providers.append(NewsApiProvider(str(self.config["news_api_key"])))
        if self.config.get("twitter_api_key"):
            self.providers.append(TwitterProvider(str(self.config["twitter_api_key"])))
        self.providers.append(
            RSSProvider([str(feed) for feed in self.config["rss_feeds"]])
        )
        self.cache_service = LRUCache(int(self.config.get("cache_size", 1000)))
        self.sentiment_service = SentimentAnalyzerService()
        self.observers = []
        self.news_history = {}
        self.black_swan_alerts = []

    async def fetch_and_process_news(self, pair: str):
        tasks = [provider.fetch_news(pair) for provider in self.providers]
        all_news = await asyncio.gather(*tasks)
        news_items = [item for sublist in all_news for item in sublist]
        for item in news_items:
            item.hash = hashlib.sha256((item.title + item.content).encode()).hexdigest()
            item.sentiment = self.sentiment_service.analyze(
                item.title + " " + item.content
            )
            self.cache_service[item.hash] = item
        self.news_history.setdefault(pair, []).extend(news_items)
        self._notify_observers("news_update", news_items)

    def _notify_observers(self, event: str, data: Any):
        for observer in self.observers:
            observer.notify(event, data)

    def analyze_sentiment(self, pair: str) -> Dict[str, float]:
        """
        Анализ настроения для торговой пары.
        :param pair: тикер пары
        :return: словарь с метриками настроения
        """
        try:
            news_items = self._get_relevant_news(pair)
            if not news_items:
                return {"sentiment": 0.0, "confidence": 0.0, "impact": 0.0}
            sentiments = []
            impacts = []
            for item in news_items:
                if abs(item.sentiment) > self.config["sentiment_threshold"]:
                    sentiments.append(item.sentiment)
                    impacts.append(item.impact_score)
            if not sentiments:
                return {"sentiment": 0.0, "confidence": 0.0, "impact": 0.0}
            avg_sentiment = np.mean(sentiments)
            avg_impact = np.mean(impacts)
            confidence = min(len(sentiments) / 10, 1.0)
            return {
                "sentiment": avg_sentiment,
                "confidence": confidence,
                "impact": avg_impact,
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {pair}: {str(e)}")
            return {"sentiment": 0.0, "confidence": 0.0, "impact": 0.0}

    def impact_score(self, news_item: NewsItem) -> float:
        """
        Расчет оценки влияния новости.
        :param news_item: объект новости
        :return: float (0..1)
        """
        try:
            source_weight = {"news": 0.4, "twitter": 0.3, "rss": 0.3}
            source_score = source_weight.get(news_item.source, 0.1)
            keyword_score = len(news_item.keywords) / len(self.config["keywords"])
            sentiment_score = abs(news_item.sentiment)
            impact = source_score * 0.4 + keyword_score * 0.3 + sentiment_score * 0.3
            return min(impact, 1.0)
        except Exception as e:
            logger.error(f"Error calculating impact score: {str(e)}")
            return 0.0

    def check_for_black_swans(self) -> List[Dict]:
        """
        Проверка на наличие событий типа "черный лебедь".
        :return: список обнаруженных событий
        """
        try:
            black_swans: List[Dict] = []
            recent_news = self._get_recent_news()
            for item in recent_news:
                if any(
                    keyword in item.title.lower() or keyword in item.content.lower()
                    for keyword in self.config["black_swan_keywords"]
                ):
                    impact = self.impact_score(item)
                    if impact > self.config["impact_threshold"]:
                        black_swan = {
                            "timestamp": item.timestamp,
                            "title": item.title,
                            "source": item.source,
                            "impact": impact,
                            "url": item.url,
                        }
                        black_swans.append(black_swan)
                        self.black_swan_alerts.append(black_swan)
            return black_swans
        except Exception as e:
            logger.error(f"Error checking for black swans: {str(e)}")
            return []

    def _get_relevant_news(self, pair: str) -> List[NewsItem]:
        """Получение релевантных новостей для пары"""
        try:
            # Получение новостей из разных источников
            news_items = []

            # NewsAPI
            if self.config["news_api_key"]:
                news_items.extend(self._get_newsapi_news(pair))

            # Twitter
            if self.config["twitter_api_key"]:
                news_items.extend(self._get_twitter_news(pair))

            # RSS
            news_items.extend(self._get_rss_news(pair))

            # Фильтрация дубликатов
            filtered_items = []
            seen_titles = set()

            for item in news_items:
                if item.title not in seen_titles:
                    seen_titles.add(item.title)
                    filtered_items.append(item)

            return filtered_items

        except Exception as e:
            logger.error(f"Error getting relevant news for {pair}: {str(e)}")
            return []

    def _get_newsapi_news(self, pair: str) -> List[NewsItem]:
        """Получение новостей из NewsAPI"""
        try:
            if not self.config["news_api_key"]:
                return []

            # Формирование запроса
            query = f"{pair} OR {pair.replace('USDT', '')}"
            url = f"https://newsapi.org/v2/everything?q={query}&apiKey={self.config['news_api_key']}"

            response = requests.get(url)
            if response.status_code != 200:
                return []

            news_data = response.json()
            news_items = []

            for article in news_data.get("articles", []):
                # Анализ настроения
                sentiment = self._analyze_sentiment(
                    article["title"] + " " + article["description"]
                )

                # Создание объекта новости
                news_item = NewsItem(
                    timestamp=datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    ),
                    source="news",
                    title=article["title"],
                    content=article["description"],
                    sentiment=sentiment,
                    impact_score=0.0,  # Будет рассчитано позже
                    keywords=self._extract_keywords(
                        article["title"] + " " + article["description"]
                    ),
                    url=article["url"],
                )

                # Расчет влияния
                news_item.impact_score = self.impact_score(news_item)

                news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Error getting NewsAPI news: {str(e)}")
            return []

    def _get_twitter_news(self, pair: str) -> List[NewsItem]:
        """Получение новостей из Twitter"""
        try:
            if not self.config["twitter_api_key"]:
                return []

            # Здесь должна быть реализация получения твитов
            # через Twitter API
            return []

        except Exception as e:
            logger.error(f"Error getting Twitter news: {str(e)}")
            return []

    def _get_rss_news(self, pair: str) -> List[NewsItem]:
        """Получение новостей из RSS-лент"""
        try:
            news_items = []

            for feed_url in self.config["rss_feeds"]:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    # Проверка релевантности
                    if not self._is_relevant(
                        entry.title + " " + entry.description, pair
                    ):
                        continue

                    # Анализ настроения
                    sentiment = self._analyze_sentiment(
                        entry.title + " " + entry.description
                    )

                    # Создание объекта новости
                    news_item = NewsItem(
                        timestamp=datetime.fromtimestamp(
                            time.mktime(entry.published_parsed)
                        ),
                        source="rss",
                        title=entry.title,
                        content=entry.description,
                        sentiment=sentiment,
                        impact_score=0.0,  # Будет рассчитано позже
                        keywords=self._extract_keywords(
                            entry.title + " " + entry.description
                        ),
                        url=entry.link,
                    )

                    # Расчет влияния
                    news_item.impact_score = self.impact_score(news_item)

                    news_items.append(news_item)

            return news_items

        except Exception as e:
            logger.error(f"Error getting RSS news: {str(e)}")
            return []

    def _analyze_sentiment(self, text: str) -> float:
        """Анализ настроения текста"""
        try:
            # Использование NLTK
            if self.sentiment_service.nltk_analyzer:
                scores = self.sentiment_service.nltk_analyzer.polarity_scores(text)
                return scores["compound"]

            # Использование трансформеров
            if self.sentiment_service.transformer_analyzer:
                result = self.sentiment_service.transformer_analyzer(text)[0]
                if result["label"] == "POS":
                    return result["score"]
                elif result["label"] == "NEG":
                    return -result["score"]
                else:
                    return 0.0

            return 0.0

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0

    def _extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        try:
            # Простая реализация через поиск ключевых слов
            text_lower = text.lower()
            return [
                keyword for keyword in self.config["keywords"] if keyword in text_lower
            ]

        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []

    def _is_relevant(self, text: str, pair: str) -> bool:
        """Проверка релевантности текста для пары"""
        try:
            # Проверка на наличие пары в тексте
            if pair in text or pair.replace("USDT", "") in text:
                return True

            # Проверка на ключевые слова
            return any(keyword in text.lower() for keyword in self.config["keywords"])

        except Exception as e:
            logger.error(f"Error checking relevance: {str(e)}")
            return False

    def _get_recent_news(self) -> List[NewsItem]:
        """Получение последних новостей"""
        try:
            # Объединение новостей из всех источников
            all_news = []

            for pair_news in self.news_history.values():
                all_news.extend(pair_news)

            # Сортировка по времени
            all_news.sort(key=lambda x: x.timestamp, reverse=True)

            # Фильтрация по времени (последние 24 часа)
            recent_time = datetime.now() - timedelta(days=1)
            return [item for item in all_news if item.timestamp > recent_time]

        except Exception as e:
            logger.error(f"Error getting recent news: {str(e)}")
            return []

    def _format_news(self, news: List[NewsItem]) -> str:
        """Форматирование новостей в строку"""
        try:
            if not news:
                return ""
            return " ".join([f"{item.title} {item.content}" for item in news])
        except Exception as e:
            logger.error(f"Error formatting news: {str(e)}")
            return ""

    def _analyze_news_impact(self, news: List[NewsItem]) -> Dict[str, float]:
        """Анализ влияния новостей"""
        try:
            if not news:
                return {"sentiment": 0.0, "impact": 0.0}

            text = self._format_news(news)
            sentiment = self.sentiment_service.analyze(text)

            return {"sentiment": sentiment, "impact": abs(sentiment)}
        except Exception as e:
            logger.error(f"Error analyzing news impact: {str(e)}")
            return {"sentiment": 0.0, "impact": 0.0}

    def _calculate_sentiment_score(self, text: str) -> float:
        """Расчет сентимента текста."""
        try:
            result = self.sentiment_service.transformer_analyzer(text)[0]
            if isinstance(result, dict) and "score" in result:
                return float(result["score"])
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating sentiment: {str(e)}")
            return 0.0

    def _get_source_weight(self, news_source: NewsSource) -> float:
        """Получение веса источника новостей."""
        return float(self.source_weights.get(news_source.value, 0.5))

    async def _process_news(self, news: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка новости."""
        try:
            text = " ".join(news.get("text", []))
            sentiment = self._calculate_sentiment_score(text)
            return {
                "text": text,
                "sentiment": float(sentiment),
                "source": NewsSource.NEWS.value,
                "timestamp": news.get("timestamp", datetime.now()),
            }
        except Exception as e:
            logger.error(f"Error processing news: {str(e)}")
            return {}

    async def _process_rss(self, feed: List[Any]) -> Dict[str, Any]:
        """Обработка RSS-ленты."""
        try:
            text = " ".join(
                [
                    entry.get("title", "") + " " + entry.get("description", "")
                    for entry in feed
                ]
            )
            sentiment = self._calculate_sentiment_score(text)
            timestamp = (
                datetime.fromtimestamp(time.mktime(feed[0].published_parsed))
                if feed
                else datetime.now()
            )
            return {
                "text": text,
                "sentiment": float(sentiment),
                "source": NewsSource.RSS.value,
                "timestamp": timestamp,
            }
        except Exception as e:
            logger.error(f"Error processing RSS: {str(e)}")
            return {}

    async def _process_twitter(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Обработка твитов."""
        try:
            text = " ".join([tweet.get("text", "") for tweet in tweets])
            sentiment = self._calculate_sentiment_score(text)
            return {
                "text": text,
                "sentiment": float(sentiment),
                "source": NewsSource.TWITTER.value,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error processing Twitter: {str(e)}")
            return {}

    async def _process_reddit(self, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Обработка постов Reddit."""
        try:
            text = " ".join(
                [
                    post.get("title", "") + " " + post.get("selftext", "")
                    for post in posts
                ]
            )
            sentiment = self._calculate_sentiment_score(text)
            return {
                "text": text,
                "sentiment": float(sentiment),
                "source": NewsSource.REDDIT.value,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"Error processing Reddit: {str(e)}")
            return {}
