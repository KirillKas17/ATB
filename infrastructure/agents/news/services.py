"""
Сервисы для работы с новостями
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import nltk
from shared.numpy_utils import np
from nltk.sentiment.vader import VaderSentimentAnalyzer
from transformers import pipeline as transformers_pipeline

from shared.logging import setup_logger

logger = setup_logger(__name__)


class LRUCache:
    """LRU кэш для новостей"""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []

    def __getitem__(self, key: str) -> Any:
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = value
        self.access_order.append(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class SentimentAnalyzerService:
    """Сервис анализа настроений"""

    def __init__(self) -> None:
        self.vader_analyzer: Optional[VaderSentimentAnalyzer] = None
        self.transformer_pipeline: Optional[Any] = None
        self._initialize_analyzers()

    def _initialize_analyzers(self) -> None:
        """Инициализация анализаторов настроений"""
        try:
            # Инициализация NLTK
            try:
                nltk.data.find("vader_lexicon")
            except LookupError:
                nltk.download("vader_lexicon")
            self.vader_analyzer = VaderSentimentAnalyzer()
            # Инициализация Transformers (опционально)
            try:
                self.transformer_pipeline = transformers_pipeline(
                    task="sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
                )
            except Exception as e:
                logger.warning(f"Could not initialize transformer pipeline: {e}")
                self.transformer_pipeline = None
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzers: {e}")

    def analyze(self, text: str) -> float:
        """Анализ настроений текста"""
        try:
            if not text or not text.strip():
                return 0.0
            # VADER анализ
            if self.vader_analyzer is None:
                return 0.0
            vader_scores = self.vader_analyzer.polarity_scores(text)
            vader_sentiment = vader_scores["compound"]
            # Transformers анализ (если доступен)
            transformer_sentiment = 0.0
            if self.transformer_pipeline:
                try:
                    result = self.transformer_pipeline(text[:512])[
                        0
                    ]  # Ограничиваем длину
                    label = result["label"]
                    score = result["score"]
                    if label == "POSITIVE":
                        transformer_sentiment = score
                    elif label == "NEGATIVE":
                        transformer_sentiment = -score
                    else:  # NEUTRAL
                        transformer_sentiment = 0.0
                except Exception as e:
                    logger.warning(f"Transformer analysis failed: {e}")
            # Комбинируем результаты
            if self.transformer_pipeline:
                combined_sentiment = (vader_sentiment + transformer_sentiment) / 2
            else:
                combined_sentiment = vader_sentiment
            return float(np.clip(combined_sentiment, -1.0, 1.0))
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return 0.0

    def extract_keywords(self, text: str) -> List[str]:
        """Извлечение ключевых слов из текста"""
        try:
            if not text:
                return []
            # Простое извлечение ключевых слов
            words = text.lower().split()
            # Фильтруем стоп-слова и короткие слова
            stop_words = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "in",
                "on",
                "at",
                "to",
                "for",
                "of",
                "with",
                "by",
            }
            keywords = [
                word for word in words if len(word) > 3 and word not in stop_words
            ]
            # Возвращаем уникальные ключевые слова
            return list(set(keywords))[:10]  # Ограничиваем количество
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def calculate_impact_score(self, news_item: Any) -> float:
        """Расчет оценки влияния новости"""
        try:
            # Базовые факторы
            source_weight = self._get_source_weight(news_item.source)
            sentiment_impact = abs(news_item.sentiment)
            engagement_impact = self._calculate_engagement_impact(
                news_item.social_engagement
            )
            # Время актуальности
            time_factor = self._calculate_time_factor(news_item.timestamp)
            # Комбинированная оценка
            impact_score = (
                source_weight * 0.3
                + sentiment_impact * 0.3
                + engagement_impact * 0.2
                + time_factor * 0.2
            )
            return float(np.clip(impact_score, 0.0, 1.0))
        except Exception as e:
            logger.error(f"Error calculating impact score: {e}")
            return 0.0

    def _get_source_weight(self, source: Any) -> float:
        """Получение веса источника"""
        weights = {
            "news": 0.8,
            "rss": 0.7,
            "twitter": 0.6,
            "reddit": 0.5,
            "social_media": 0.6,
        }
        return weights.get(source.value, 0.5)

    def _calculate_engagement_impact(self, engagement: Dict[str, int]) -> float:
        """Расчет влияния вовлеченности"""
        try:
            total_engagement = sum(engagement.values())
            return min(total_engagement / 1000.0, 1.0)  # Нормализуем
        except Exception:
            return 0.0

    def _calculate_time_factor(self, timestamp: datetime) -> float:
        """Расчет временного фактора"""
        try:
            age_hours = (datetime.now() - timestamp).total_seconds() / 3600
            if age_hours <= 1:
                return 1.0
            elif age_hours <= 24:
                return 0.8
            elif age_hours <= 72:
                return 0.5
            else:
                return 0.2
        except Exception:
            return 0.5
