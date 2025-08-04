"""
Основной новостной агент с модульной архитектурой.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from domain.type_definitions.agent_types import ProcessingResult, AgentType, AgentConfig
from infrastructure.agents.base_agent import AgentStatus, BaseAgent
from infrastructure.agents.social_media.agent_social_media import SocialMediaAgent
from shared.logging import setup_logger

from .observers import NewsObserverManager
from .providers import (
    INewsProvider,
    NewsApiProvider,
    RSSProvider,
    SocialMediaNewsProvider,
    TwitterProvider,
)
from .services import LRUCache, SentimentAnalyzerService
from .types import NewsItem

logger = setup_logger(__name__)

# Заглушка для NewsAgentObserver, если не определён
try:
    from .observers import NewsAgentObserver
except ImportError:
    from typing import Any
    # Убираем дублирующее определение класса
    pass

class NewsAgent(BaseAgent):
    """
    Агент анализа новостей: асинхронный сбор, фильтрация, скоринг, black swan, подписка на события.
    Расширен интеграцией с социальными сетями.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента новостей.
        :param config: словарь с параметрами новостного анализа
        """
        news_config = config or {
            "cache_size": 1000,
            "cache_ttl": 300,  # 5 минут
            "sentiment_threshold": 0.3,
            "impact_threshold": 0.5,
            "black_swan_threshold": 0.8,
            "max_news_age": 24,  # часы
            "update_interval": 60,  # секунды
            "providers": {
                "newsapi": {"enabled": True, "api_key": ""},
                "twitter": {"enabled": False, "api_key": ""},
                "rss": {"enabled": True, "feeds": []},
                "social_media": {"enabled": True},
            },
        }
        
        # [2] создаю AgentConfig для базового агента
        agent_config: Dict[str, Any] = {
            "name": "NewsAgent",
            "agent_type": "news_analyzer",  # [1] правильный AgentType
            "max_position_size": 1000.0,  # [2] обязательное поле
            "max_portfolio_risk": 0.02,  # [2] обязательное поле
            "max_risk_per_trade": 0.01,  # [2] обязательное поле
            "confidence_threshold": 0.7,  # [2] обязательное поле
            "risk_threshold": 0.3,  # [2] обязательное поле
            "performance_threshold": 0.6,  # [2] обязательное поле
            "rebalance_interval": 300,  # [2] обязательное поле
            "processing_timeout_ms": 30000,  # [2] обязательное поле
            "retry_attempts": 3,  # [2] обязательное поле
            "enable_evolution": False,  # [2] обязательное поле
            "enable_learning": True,  # [2] обязательное поле
            "metadata": {"news_config": news_config}  # Исправление: используем строковый ключ
        }
        
        super().__init__(
            name="NewsAgent",
            agent_type="news_analyzer",  # [1] правильный AgentType
            config=agent_config  # config должен быть dict[str, Any]
        )

        # Инициализация компонентов
        self._providers: List[INewsProvider] = []
        self._cache_service = LRUCache(news_config.get("cache_size", 1000))
        self._sentiment_service = SentimentAnalyzerService()
        self._observer_manager = NewsObserverManager()
        self._news_history: Dict[str, List[NewsItem]] = {}
        self._black_swan_alerts: List[Dict] = []
        self._social_media_agent: Optional[SocialMediaAgent] = None

        # Инициализация провайдеров
        self._initialize_providers()

    @property
    def providers(self) -> List[INewsProvider]:
        return self._providers

    @property
    def cache_service(self) -> LRUCache:
        return self._cache_service

    @property
    def sentiment_service(self) -> SentimentAnalyzerService:
        return self._sentiment_service

    @property
    def observer_manager(self) -> NewsObserverManager:
        return self._observer_manager

    @property
    def news_history(self) -> Dict[str, List[NewsItem]]:
        return self._news_history

    @property
    def black_swan_alerts(self) -> List[Dict]:
        return self._black_swan_alerts

    @property
    def social_media_agent(self) -> Optional[SocialMediaAgent]:
        return self._social_media_agent

    async def initialize(self) -> bool:
        """Инициализация агента новостей."""
        try:
            # Валидация конфигурации
            if not self.validate_config():
                return False

            # Инициализация провайдеров
            await self._initialize_providers_async()

            self._update_state(AgentStatus.HEALTHY)
            self.update_confidence(0.7)

            logger.info(f"NewsAgent initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize NewsAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных для анализа новостей."""
        start_time = datetime.now()

        try:
            if isinstance(data, dict):
                pair = data.get("pair", "")
                if not pair:
                    raise ValueError("Pair is required for news analysis")

                # Получение и анализ новостей
                news_result = await self.fetch_and_process_news(pair)

                # Анализ настроений
                sentiment_result = self.analyze_sentiment(pair)

                # Проверка на black swan события
                black_swan_events = self.check_for_black_swans()

                result_data = {
                    "news": news_result,
                    "sentiment": sentiment_result,
                    "black_swan_events": black_swan_events,
                    "fear_greed_index": self.get_fear_greed_index(pair),
                    "trending_topics": self.get_trending_topics(pair),
                }

                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)

                return ProcessingResult(
                    success=True,
                    data=result_data,
                    confidence=0.7,
                    risk_score=0.3,
                    processing_time_ms=processing_time,
                    timestamp=datetime.now(),  # [3] обязательное поле
                    metadata={"agent_type": "news_analyzer"},  # [3] обязательное поле
                    errors=[],  # [3] обязательное поле
                    warnings=[]  # [3] обязательное поле
                )
            else:
                raise ValueError("Invalid data format for NewsAgent")

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                risk_score=1.0,
                processing_time_ms=processing_time,
                timestamp=datetime.now(),  # [3] обязательное поле
                metadata={"agent_type": "news_analyzer"},  # [3] обязательное поле
                errors=[str(e)],  # [3] обязательное поле
                warnings=[]  # [3] обязательное поле
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов агента новостей."""
        try:
            # Очистка кэша
            self._cache_service = LRUCache(0)

            # Очистка истории
            self._news_history.clear()
            self._black_swan_alerts.clear()

            # Очистка наблюдателей
            self._observer_manager.clear_observers()

            logger.info("NewsAgent cleanup completed")

        except Exception as e:
            logger.error(f"Error during NewsAgent cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации агента новостей."""
        try:
            required_keys = [
                "cache_size",
                "cache_ttl",
                "sentiment_threshold",
                "impact_threshold",
                "black_swan_threshold",
                "max_news_age",
                "update_interval",
            ]
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"Missing required config key: {key}")
                    return False
                value = self.config[key]
                if not isinstance(value, (int, float)) or float(value) <= 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    def _initialize_providers(self) -> None:
        """Инициализация провайдеров новостей."""
        try:
            providers = self.config.get("providers", {})
            providers_config = dict(providers) if isinstance(providers, dict) else {}
            
            # NewsAPI провайдер
            if providers_config.get("newsapi", {}).get("enabled", False):
                newsapi_config = providers_config["newsapi"]
                self._providers.append(
                    NewsApiProvider(api_key=newsapi_config.get("api_key", ""))
                )
            
            # Twitter провайдер
            if providers_config.get("twitter", {}).get("enabled", False):
                twitter_config = providers_config["twitter"]
                self._providers.append(
                    TwitterProvider(api_key=twitter_config.get("api_key", ""))
                )
            
            # RSS провайдер
            if providers_config.get("rss", {}).get("enabled", False):
                rss_config = providers_config["rss"]
                self._providers.append(
                    RSSProvider(feeds=rss_config.get("feeds", []))
                )
            
            # Social Media провайдер
            if providers_config.get("social_media", {}).get("enabled", False):
                if self._social_media_agent:
                    self._providers.append(
                        SocialMediaNewsProvider(self._social_media_agent)
                    )
                else:
                    # Создаем заглушку для SocialMediaAgent
                    dummy_agent = SocialMediaAgent()
                    self._providers.append(
                        SocialMediaNewsProvider(dummy_agent)
                    )
            
            logger.info(f"Initialized {len(self._providers)} news providers")
            
        except Exception as e:
            logger.error(f"Error initializing providers: {e}")

    async def _initialize_providers_async(self) -> None:
        """Асинхронная инициализация провайдеров."""
        try:
            for provider in self._providers:
                if hasattr(provider, "initialize"):
                    await provider.initialize()
        except Exception as e:
            logger.error(f"Error in async provider initialization: {e}")

    def add_observer(self, observer: NewsAgentObserver) -> None:
        """Добавление наблюдателя."""
        self._observer_manager.add_observer(observer)

    def remove_observer(self, observer: NewsAgentObserver) -> None:
        """Удаление наблюдателя."""
        self._observer_manager.remove_observer(observer)

    def set_social_media_agent(self, agent: SocialMediaAgent) -> None:
        """Установка агента социальных сетей."""
        self._social_media_agent = agent

    async def fetch_and_process_news(self, pair: str) -> Dict[str, Any]:
        """Получение и обработка новостей."""
        try:
            news_items = []
            
            # Получение новостей от всех провайдеров
            for provider in self._providers:
                try:
                    provider_news = await provider.fetch_news(pair)
                    news_items.extend(provider_news)
                except Exception as e:
                    logger.error(f"Error fetching news from provider {provider}: {e}")
            
            # Фильтрация и обработка новостей
            filtered_news = []
            for item in news_items:
                if self._is_relevant(item.title, pair):
                    # Анализ настроений
                    sentiment = self._sentiment_service.analyze(item.title)
                    item.sentiment = sentiment
                    
                    # Проверка на black swan
                    if float(str(sentiment)) > float(self.config.get("black_swan_threshold", 0.8)):
                        self._black_swan_alerts.append({
                            "title": item.title,
                            "sentiment": sentiment,
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    filtered_news.append(item)
            
            # Сохранение в историю
            if pair not in self._news_history:
                self._news_history[pair] = []
            self._news_history[pair].extend(filtered_news)
            
            # Ограничение размера истории
            max_history = 100
            if len(self._news_history[pair]) > max_history:
                self._news_history[pair] = self._news_history[pair][-max_history:]
            
            return {
                "total_news": len(news_items),
                "filtered_news": len(filtered_news),
                "black_swan_count": len(self._black_swan_alerts),
                "latest_news": [item.to_dict() if hasattr(item, 'to_dict') else item.__dict__ for item in filtered_news[-5:]]
            }
            
        except Exception as e:
            logger.error(f"Error fetching and processing news: {e}")
            return {"error": str(e)}

    def analyze_sentiment(self, pair: str) -> Dict[str, float]:
        """Анализ настроений для пары."""
        try:
            if pair not in self._news_history:
                return {"overall_sentiment": 0.0, "confidence": 0.0}
            
            news_items = self._news_history[pair]
            if not news_items:
                return {"overall_sentiment": 0.0, "confidence": 0.0}
            
            # Расчет общего настроения
            overall_sentiment = self._calculate_overall_sentiment(news_items)
            
            # Расчет уверенности на основе количества новостей
            confidence = min(1.0, len(news_items) / 10.0)
            
            return {
                "overall_sentiment": overall_sentiment,
                "confidence": confidence,
                "news_count": len(news_items)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"overall_sentiment": 0.0, "confidence": 0.0}

    def check_for_black_swans(self) -> List[Dict]:
        """Проверка на black swan события."""
        try:
            # Возвращаем последние black swan события
            recent_alerts = self._black_swan_alerts[-10:]  # Последние 10
            return recent_alerts
        except Exception as e:
            logger.error(f"Error checking for black swans: {e}")
            return []

    def get_social_sentiment(self, pair: str) -> Optional[Any]:
        """Получение настроений из социальных сетей."""
        try:
            if self._social_media_agent:
                return self._social_media_agent.get_social_sentiment(pair)
            return None
        except Exception as e:
            logger.error(f"Error getting social sentiment: {e}")
            return None

    def get_fear_greed_index(self, pair: str) -> float:
        """Получение индекса страха и жадности."""
        try:
            # Упрощенная реализация
            # В реальности здесь был бы расчет на основе различных метрик
            sentiment = self.analyze_sentiment(pair)
            overall_sentiment = sentiment.get("overall_sentiment", 0.0)
            
            # Преобразование настроения в индекс страха/жадности
            # 0 = крайний страх, 100 = крайняя жадность
            fear_greed = (overall_sentiment + 1.0) * 50.0  # Нормализация от -1..1 к 0..100
            return max(0.0, min(100.0, fear_greed))
            
        except Exception as e:
            logger.error(f"Error calculating fear/greed index: {e}")
            return 50.0  # Нейтральное значение

    def get_trending_topics(self, pair: str) -> List[str]:
        """Получение трендовых тем."""
        try:
            if pair not in self._news_history:
                return []
            
            news_items = self._news_history[pair]
            if not news_items:
                return []
            
            # Упрощенная реализация - извлекаем ключевые слова из заголовков
            topics = []
            for item in news_items[-20:]:  # Последние 20 новостей
                words = item.title.lower().split()
                # Фильтруем короткие слова и стоп-слова
                keywords = [word for word in words if len(word) > 3 and word not in ["the", "and", "for", "with"]]
                topics.extend(keywords[:3])  # Берем первые 3 ключевых слова
            
            # Подсчитываем частоту и возвращаем топ-5
            from collections import Counter
            topic_counts = Counter(topics)
            return [topic for topic, count in topic_counts.most_common(5)]
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []

    def _is_relevant(self, text: str, pair: str) -> bool:
        """Проверка релевантности текста для пары."""
        try:
            # Упрощенная проверка релевантности
            text_lower = text.lower()
            pair_lower = pair.lower()
            
            # Проверяем наличие символов пары в тексте
            if pair_lower in text_lower:
                return True
            
            # Проверяем ключевые слова
            keywords = ["crypto", "bitcoin", "ethereum", "trading", "market", "price"]
            return any(keyword in text_lower for keyword in keywords)
            
        except Exception as e:
            logger.error(f"Error checking relevance: {e}")
            return False

    def _calculate_overall_sentiment(self, news_items: List[NewsItem]) -> float:
        """Расчет общего настроения."""
        try:
            if not news_items:
                return 0.0
            
            sentiments = [item.sentiment for item in news_items if hasattr(item, 'sentiment')]
            if not sentiments:
                return 0.0
            
            return sum(sentiments) / len(sentiments)
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {e}")
            return 0.0

    def _calculate_overall_impact(self, news_items: List[NewsItem]) -> float:
        """Расчет общего влияния."""
        try:
            if not news_items:
                return 0.0
            
            # Упрощенная реализация
            # В реальности здесь был бы более сложный анализ влияния
            return len(news_items) / 100.0  # Нормализованное влияние
            
        except Exception as e:
            logger.error(f"Error calculating overall impact: {e}")
            return 0.0
