"""
Агент для анализа социальных сетей.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from domain.type_definitions.agent_types import AgentConfig, AgentStatus, ProcessingResult, AgentType
from infrastructure.agents.social_media.types import SocialSentimentResult
from infrastructure.agents.base_agent import BaseAgent
from infrastructure.agents.social_media.providers import (
    DiscordProvider,
    ISocialMediaProvider,
    RedditProvider,
    TelegramProvider,
)
from infrastructure.agents.base_agent import AgentState

logger = logging.getLogger(__name__)


class SocialMediaAgent(BaseAgent):
    """Агент для анализа настроений в социальных сетях."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        """Инициализация агента социальных сетей."""
        if config is None:
            config = {
                "name": "SocialMediaAgent",
                "agent_type": "social_media",  # [1] правильный AgentType
                "max_position_size": 1000.0,
                "max_portfolio_risk": 0.02,
                "max_risk_per_trade": 0.01,
                "confidence_threshold": 0.7,  # [1] обязательное поле
                "risk_threshold": 0.5,  # [1] обязательное поле
                "performance_threshold": 0.6,  # [1] обязательное поле
                "rebalance_interval": 300,  # [1] обязательное поле
                "processing_timeout_ms": 30000,  # [1] обязательное поле
                "retry_attempts": 3,  # [1] обязательное поле
                "enable_evolution": False,  # [1] обязательное поле
                "enable_learning": True,  # [1] обязательное поле
                "metadata": {  # [1] обязательное поле
                    "cache_ttl": 300,
                    "max_posts_per_request": 100,
                    "sentiment_threshold": 0.5,
                    "platforms": ["reddit", "telegram", "discord"],
                    "reddit": {},
                    "telegram": {},
                    "discord": {},
                }
            }
        name = config.get("name", "SocialMediaAgent")
        # Инициализация базового агента
        super().__init__(
            name=name,
            agent_type="social_media",  # Исправление: используем строковое значение из Literal
            config=config.__dict__ if config else {},  # Исправление: преобразуем AgentConfig в dict
        )
        
        # Состояние агента
        self._state = AgentState(
            agent_id=self.agent_id,  # [2] добавляю agent_id
            status=AgentStatus.INITIALIZING,  # [2] правильный статус
            is_running=False,
            is_healthy=True,
            performance_score=0.0,
            confidence=0.5,
            risk_score=0.5,
        )
        self._providers: List[ISocialMediaProvider] = []
        self._sentiment_cache: Dict[str, SocialSentimentResult] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._initialize_providers()

    @property
    def providers(self) -> List[ISocialMediaProvider]:
        return self._providers

    @property
    def sentiment_cache(self) -> Dict[str, SocialSentimentResult]:
        return self._sentiment_cache

    @property
    def cache_timestamps(self) -> Dict[str, datetime]:
        return self._cache_timestamps

    async def initialize(self) -> bool:
        """Инициализация агента социальных сетей."""
        try:
            # Валидация конфигурации
            if not self.validate_config():
                return False
            # Инициализация провайдеров
            await self._initialize_providers_async()
            self._state = AgentState(
                agent_id=self._state.agent_id,
                status=AgentStatus.HEALTHY,
                is_running=self._state.is_running,
                is_healthy=self._state.is_healthy,
                performance_score=self._state.performance_score,
                confidence=self._state.confidence,
                risk_score=self._state.risk_score,
                error_count=self._state.error_count,
                last_update=self._state.last_update,
                metadata=self._state.metadata,
            )
            self.update_confidence(0.7)
            logger.info(f"SocialMediaAgent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SocialMediaAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных для анализа социальных сетей."""
        start_time = datetime.now()
        try:
            if isinstance(data, dict):
                symbol = data.get("symbol", "")
                if not symbol:
                    raise ValueError("Symbol is required for social media analysis")
                # Получение настроений из социальных сетей
                sentiment_result = await self.get_social_sentiment(symbol)
                # Получение индекса страха и жадности
                fear_greed_index = self.get_fear_greed_index(symbol)
                # Получение истории настроений
                sentiment_history = self.get_sentiment_history(symbol)
                result_data = {
                    "sentiment": sentiment_result,
                    "fear_greed_index": fear_greed_index,
                    "sentiment_history": sentiment_history,
                    "platforms": [getattr(p, "platform_name", "unknown") for p in self.providers],  # Исправлено: используем правильный атрибут
                }
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)
                return ProcessingResult(
                    success=True,
                    data=result_data,
                    confidence=0.7,  # [3] добавляю confidence
                    risk_score=0.3,  # [3] добавляю risk_score
                    processing_time_ms=int(processing_time),
                    errors=[]
                )
            else:
                raise ValueError("Invalid data format for SocialMediaAgent")
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={},
                confidence=0.0,  # [3] добавляю confidence
                risk_score=1.0,  # [3] добавляю risk_score
                processing_time_ms=int(processing_time),
                errors=[str(e)]
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов агента социальных сетей."""
        try:
            # Очистка кэша
            self._sentiment_cache.clear()
            self._cache_timestamps.clear()
            # Закрытие сессий провайдеров
            for provider in self.providers:
                if hasattr(provider, "session") and provider.session:
                    await provider.session.close()
            logger.info("SocialMediaAgent cleanup completed")
        except Exception as e:
            logger.error(f"Error during SocialMediaAgent cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации."""
        try:
            if not self.config:
                logger.error("Config is required")
                return False
            # Проверка обязательных ключей
            required_keys = [
                "max_posts_per_request",
                "sentiment_threshold",
            ]
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"Missing required config key: {key}")
                    return False
                value = self.config[key]
                if not isinstance(value, (int, float)) or value <= 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False
            # Проверка платформ
            if "platforms" not in self.config:
                logger.error("Missing platforms configuration")
                return False
            return True
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False

    def _initialize_providers(self) -> None:
        """Инициализация провайдеров социальных сетей."""
        try:
            platforms = self.config.get("platforms", [])
            if not isinstance(platforms, list):
                logger.error("Platforms config must be a list")
                return
            # Reddit провайдер
            if "reddit" in platforms:
                reddit_config = self.config.get("reddit", {})
                if isinstance(reddit_config, dict):
                    self._providers.append(RedditProvider(reddit_config))
            # Telegram провайдер
            if "telegram" in platforms:
                telegram_config = self.config.get("telegram", {})
                if isinstance(telegram_config, dict):
                    self._providers.append(TelegramProvider(telegram_config))
            # Discord провайдер
            if "discord" in platforms:
                discord_config = self.config.get("discord", {})
                if isinstance(discord_config, dict):
                    self._providers.append(DiscordProvider(discord_config))
            logger.info(f"Initialized {len(self._providers)} social media providers")
        except Exception as e:
            logger.error(f"Error initializing providers: {e}")

    async def _initialize_providers_async(self) -> None:
        """Асинхронная инициализация провайдеров."""
        try:
            # Здесь можно добавить асинхронную инициализацию провайдеров
            # например, проверку доступности API
            pass
        except Exception as e:
            logger.error(f"Error in async provider initialization: {e}")

    def _get_cached_sentiment(self, symbol: str) -> Optional[SocialSentimentResult]:
        """Получение настроений из кэша."""
        try:
            if symbol not in self._sentiment_cache:
                return None
            # Проверка актуальности кэша
            cache_time = self._cache_timestamps.get(symbol)
            if not cache_time:
                return None
            cache_age = (datetime.now() - cache_time).total_seconds()
            cache_ttl = self.config.get("cache_ttl", 300)
            if isinstance(cache_ttl, (int, float)) and cache_age > cache_ttl:
                # Удаляем устаревшие данные
                del self._sentiment_cache[symbol]
                del self._cache_timestamps[symbol]
                return None
            return self._sentiment_cache[symbol]
        except Exception as e:
            logger.error(f"Error getting cached sentiment: {e}")
            return None

    def _cache_sentiment(self, symbol: str, result: SocialSentimentResult) -> None:
        """Сохранение настроений в кэш."""
        try:
            self._sentiment_cache[symbol] = result
            self._cache_timestamps[symbol] = datetime.now()
        except Exception as e:
            logger.error(f"Error caching sentiment: {e}")

    def _cleanup_old_cache(self) -> None:
        """Очистка устаревшего кэша."""
        try:
            current_time = datetime.now()
            expired_symbols = []
            cache_ttl = self.config.get("cache_ttl", 300)
            for symbol, cache_time in self._cache_timestamps.items():
                cache_age = (current_time - cache_time).total_seconds()
                if isinstance(cache_ttl, (int, float)) and cache_age > cache_ttl:
                    expired_symbols.append(symbol)
            for symbol in expired_symbols:
                del self._sentiment_cache[symbol]
                del self._cache_timestamps[symbol]
            if expired_symbols:
                logger.info(f"Cleaned up {len(expired_symbols)} expired cache entries")
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")

    async def get_social_sentiment(
        self, symbol: str
    ) -> Optional[SocialSentimentResult]:
        """Получение настроений из социальных сетей."""
        try:
            # Проверяем кэш
            cached_result = self._get_cached_sentiment(symbol)
            if cached_result:
                return cached_result
            # Получаем данные от всех провайдеров
            all_results = []
            for provider in self.providers:
                try:
                    max_posts = self.config.get("max_posts_per_request", 100)
                    if isinstance(max_posts, int):
                        posts = await provider.fetch_posts(symbol, max_posts)
                        if posts:
                            sentiment = await provider.analyze_sentiment(posts)
                            if sentiment:
                                all_results.append(sentiment)
                except Exception as e:
                    logger.error(f"Error fetching data from provider: {e}")
            # Объединяем результаты
            if all_results:
                combined_result = self._combine_sentiment_results(all_results, symbol)
                self._cache_sentiment(symbol, combined_result)
                return combined_result
            return None
        except Exception as e:
            logger.error(f"Error getting social sentiment: {e}")
            return None

    def _combine_sentiment_results(
        self, results: List[SocialSentimentResult], symbol: str
    ) -> SocialSentimentResult:
        """Объединение результатов анализа настроений."""
        try:
            if not results:
                return SocialSentimentResult(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    overall_sentiment=0.0,
                    positive_sentiment=0.0,
                    negative_sentiment=0.0,
                    neutral_sentiment=0.0,
                    confidence=0.0,
                    volume=0,
                    platforms=[],
                )
            # Взвешенное среднее по доверию
            total_weight = sum(r.confidence for r in results)
            if total_weight == 0:
                total_weight = len(results)
            overall_sentiment = sum(r.overall_sentiment * r.confidence for r in results) / total_weight
            positive_sentiment = sum(r.positive_sentiment * r.confidence for r in results) / total_weight
            negative_sentiment = sum(r.negative_sentiment * r.confidence for r in results) / total_weight
            neutral_sentiment = sum(r.neutral_sentiment * r.confidence for r in results) / total_weight
            confidence = sum(r.confidence for r in results) / len(results)
            volume = sum(r.volume for r in results)
            platforms = list(set().union(*[r.platforms for r in results]))
            return SocialSentimentResult(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=overall_sentiment,
                positive_sentiment=positive_sentiment,
                negative_sentiment=negative_sentiment,
                neutral_sentiment=neutral_sentiment,
                confidence=confidence,
                volume=volume,
                platforms=platforms,
            )
        except Exception as e:
            logger.error(f"Error combining sentiment results: {e}")
            return SocialSentimentResult(
                symbol=symbol,
                timestamp=datetime.now(),
                overall_sentiment=0.0,
                positive_sentiment=0.0,
                negative_sentiment=0.0,
                neutral_sentiment=0.0,
                confidence=0.0,
                volume=0,
                platforms=[],
            )

    def get_fear_greed_index(self, symbol: str) -> float:
        """Получение индекса страха и жадности."""
        try:
            # Простая реализация на основе настроений
            sentiment_result = self._sentiment_cache.get(symbol)
            if sentiment_result:
                # Нормализуем настроения к индексу 0-100
                sentiment_score = (sentiment_result.overall_sentiment + 1) * 50
                return max(0, min(100, sentiment_score))
            return 50.0  # Нейтральное значение
        except Exception as e:
            logger.error(f"Error calculating fear/greed index: {e}")
            return 50.0

    def get_sentiment_history(
        self, symbol: str, hours: int = 24
    ) -> List[SocialSentimentResult]:
        """Получение истории настроений."""
        try:
            # В реальной реализации здесь должна быть база данных
            # Пока возвращаем пустой список
            return []
        except Exception as e:
            logger.error(f"Error getting sentiment history: {e}")
            return []
