"""
Unit тесты для SocialMediaAnalyzer.
Тестирует анализ социальных сетей, включая мониторинг настроений,
анализ трендов, отслеживание влиятельных аккаунтов и анализ активности.
"""
import pytest
from datetime import datetime, timedelta
from infrastructure.core.social_media_analyzer import SocialMediaAnalyzer
from typing import List, Dict, Any

class TestSocialMediaAnalyzer:
    """Тесты для SocialMediaAnalyzer."""
    @pytest.fixture
    def social_media_analyzer(self) -> SocialMediaAnalyzer:
        """Фикстура для SocialMediaAnalyzer."""
        return SocialMediaAnalyzer()
    @pytest.fixture
    def sample_social_data(self) -> list:
        """Фикстура с тестовыми данными социальных сетей."""
        return [
            {
                "id": "post_001",
                "platform": "twitter",
                "username": "crypto_expert",
                "content": "Bitcoin is showing strong momentum! 🚀 #BTC #crypto",
                "timestamp": datetime.now() - timedelta(hours=2),
                "likes": 1500,
                "retweets": 300,
                "comments": 200,
                "sentiment_score": 0.8,
                "followers_count": 50000
            },
            {
                "id": "post_002",
                "platform": "reddit",
                "username": "crypto_trader",
                "content": "Ethereum network congestion is causing high fees. Not good for adoption.",
                "timestamp": datetime.now() - timedelta(hours=1),
                "upvotes": 500,
                "downvotes": 50,
                "comments": 150,
                "sentiment_score": -0.4,
                "karma": 10000
            },
            {
                "id": "post_003",
                "platform": "telegram",
                "username": "crypto_news",
                "content": "Breaking: Major exchange announces new listing!",
                "timestamp": datetime.now() - timedelta(minutes=30),
                "views": 5000,
                "forwards": 200,
                "sentiment_score": 0.6,
                "channel_members": 100000
            }
        ]
    def test_initialization(self, social_media_analyzer: SocialMediaAnalyzer) -> None:
        """Тест инициализации анализатора социальных сетей."""
        assert social_media_analyzer is not None
        assert hasattr(social_media_analyzer, 'platform_connectors')
        assert hasattr(social_media_analyzer, 'sentiment_analyzers')
        assert hasattr(social_media_analyzer, 'trend_detectors')
    def test_collect_social_data(self, social_media_analyzer: SocialMediaAnalyzer) -> None:
        """Тест сбора данных из социальных сетей."""
        # Сбор данных
        collection_result = social_media_analyzer.collect_social_data(
            platforms=["twitter", "reddit", "telegram"],
            keywords=["bitcoin", "ethereum", "crypto"],
            time_range=timedelta(hours=24)
        )
        # Проверки
        assert collection_result is not None
        assert "collected_data" in collection_result
        assert "total_posts" in collection_result
        assert "platform_stats" in collection_result
        assert "collection_time" in collection_result
        # Проверка типов данных
        assert isinstance(collection_result["collected_data"], list)
        assert isinstance(collection_result["total_posts"], int)
        assert isinstance(collection_result["platform_stats"], dict)
        assert isinstance(collection_result["collection_time"], datetime)
        # Проверка логики
        assert collection_result["total_posts"] >= 0
    def test_analyze_sentiment(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест анализа настроений в социальных сетях."""
        # Анализ настроений
        sentiment_result = social_media_analyzer.analyze_sentiment(sample_social_data)
        # Проверки
        assert sentiment_result is not None
        assert "overall_sentiment" in sentiment_result
        assert "sentiment_by_platform" in sentiment_result
        assert "sentiment_trends" in sentiment_result
        assert "sentiment_confidence" in sentiment_result
        # Проверка типов данных
        assert isinstance(sentiment_result["overall_sentiment"], float)
        assert isinstance(sentiment_result["sentiment_by_platform"], dict)
        assert isinstance(sentiment_result["sentiment_trends"], dict)
        assert isinstance(sentiment_result["sentiment_confidence"], float)
        # Проверка диапазонов
        assert -1.0 <= sentiment_result["overall_sentiment"] <= 1.0
        assert 0.0 <= sentiment_result["sentiment_confidence"] <= 1.0
    def test_detect_trends(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест обнаружения трендов."""
        # Обнаружение трендов
        trends_result = social_media_analyzer.detect_trends(sample_social_data)
        # Проверки
        assert trends_result is not None
        assert "trending_topics" in trends_result
        assert "trend_strength" in trends_result
        assert "trend_duration" in trends_result
        assert "trend_prediction" in trends_result
        # Проверка типов данных
        assert isinstance(trends_result["trending_topics"], list)
        assert isinstance(trends_result["trend_strength"], dict)
        assert isinstance(trends_result["trend_duration"], dict)
        assert isinstance(trends_result["trend_prediction"], dict)
    def test_analyze_influencers(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест анализа влиятельных аккаунтов."""
        # Анализ влиятельных аккаунтов
        influencers_result = social_media_analyzer.analyze_influencers(sample_social_data)
        # Проверки
        assert influencers_result is not None
        assert "top_influencers" in influencers_result
        assert "influence_scores" in influencers_result
        assert "engagement_rates" in influencers_result
        assert "influence_categories" in influencers_result
        # Проверка типов данных
        assert isinstance(influencers_result["top_influencers"], list)
        assert isinstance(influencers_result["influence_scores"], dict)
        assert isinstance(influencers_result["engagement_rates"], dict)
        assert isinstance(influencers_result["influence_categories"], dict)
    def test_analyze_engagement(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест анализа вовлеченности."""
        # Анализ вовлеченности
        engagement_result = social_media_analyzer.analyze_engagement(sample_social_data)
        # Проверки
        assert engagement_result is not None
        assert "engagement_metrics" in engagement_result
        assert "engagement_by_platform" in engagement_result
        assert "engagement_trends" in engagement_result
        assert "viral_content" in engagement_result
        # Проверка типов данных
        assert isinstance(engagement_result["engagement_metrics"], dict)
        assert isinstance(engagement_result["engagement_by_platform"], dict)
        assert isinstance(engagement_result["engagement_trends"], dict)
        assert isinstance(engagement_result["viral_content"], list)
    def test_monitor_brand_mentions(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест мониторинга упоминаний брендов."""
        # Мониторинг упоминаний
        mentions_result = social_media_analyzer.monitor_brand_mentions(
            sample_social_data,
            brands=["Bitcoin", "Ethereum", "Binance"]
        )
        # Проверки
        assert mentions_result is not None
        assert "brand_mentions" in mentions_result
        assert "mention_sentiment" in mentions_result
        assert "mention_volume" in mentions_result
        assert "mention_reach" in mentions_result
        # Проверка типов данных
        assert isinstance(mentions_result["brand_mentions"], dict)
        assert isinstance(mentions_result["mention_sentiment"], dict)
        assert isinstance(mentions_result["mention_volume"], dict)
        assert isinstance(mentions_result["mention_reach"], dict)
    def test_analyze_activity_patterns(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест анализа паттернов активности."""
        # Анализ паттернов активности
        patterns_result = social_media_analyzer.analyze_activity_patterns(sample_social_data)
        # Проверки
        assert patterns_result is not None
        assert "activity_timeline" in patterns_result
        assert "peak_hours" in patterns_result
        assert "activity_by_day" in patterns_result
        assert "user_behavior" in patterns_result
        # Проверка типов данных
        assert isinstance(patterns_result["activity_timeline"], dict)
        assert isinstance(patterns_result["peak_hours"], list)
        assert isinstance(patterns_result["activity_by_day"], dict)
        assert isinstance(patterns_result["user_behavior"], dict)
    def test_detect_social_signals(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест обнаружения социальных сигналов."""
        # Обнаружение сигналов
        signals_result = social_media_analyzer.detect_social_signals(sample_social_data)
        # Проверки
        assert signals_result is not None
        assert "detected_signals" in signals_result
        assert "signal_strength" in signals_result
        assert "signal_confidence" in signals_result
        assert "signal_categories" in signals_result
        # Проверка типов данных
        assert isinstance(signals_result["detected_signals"], list)
        assert isinstance(signals_result["signal_strength"], dict)
        assert isinstance(signals_result["signal_confidence"], dict)
        assert isinstance(signals_result["signal_categories"], dict)
    def test_calculate_social_metrics(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест расчета социальных метрик."""
        # Расчет метрик
        metrics = social_media_analyzer.calculate_social_metrics(sample_social_data)
        # Проверки
        assert metrics is not None
        assert "total_posts" in metrics
        assert "total_engagement" in metrics
        assert "avg_sentiment" in metrics
        assert "reach_estimate" in metrics
        assert "viral_coefficient" in metrics
        # Проверка типов данных
        assert isinstance(metrics["total_posts"], int)
        assert isinstance(metrics["total_engagement"], int)
        assert isinstance(metrics["avg_sentiment"], float)
        assert isinstance(metrics["reach_estimate"], int)
        assert isinstance(metrics["viral_coefficient"], float)
        # Проверка логики
        assert metrics["total_posts"] == len(sample_social_data)
    def test_validate_social_data(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест валидации социальных данных."""
        # Валидация данных
        validation_result = social_media_analyzer.validate_social_data(sample_social_data[0])
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "data_quality_score" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["data_quality_score"], float)
        # Проверка диапазона
        assert 0.0 <= validation_result["data_quality_score"] <= 1.0
    def test_filter_social_data(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест фильтрации социальных данных."""
        # Фильтрация данных
        filter_criteria = {
            "min_engagement": 100,
            "min_sentiment": 0.0,
            "platforms": ["twitter", "reddit"],
            "time_range": timedelta(hours=24)
        }
        filtered_result = social_media_analyzer.filter_social_data(sample_social_data, filter_criteria)
        # Проверки
        assert filtered_result is not None
        assert "filtered_data" in filtered_result
        assert "filter_stats" in filtered_result
        assert "filter_criteria" in filtered_result
        # Проверка типов данных
        assert isinstance(filtered_result["filtered_data"], list)
        assert isinstance(filtered_result["filter_stats"], dict)
        assert isinstance(filtered_result["filter_criteria"], dict)
        # Проверка логики
        assert len(filtered_result["filtered_data"]) <= len(sample_social_data)
    def test_get_social_statistics(self, social_media_analyzer: SocialMediaAnalyzer, sample_social_data: list) -> None:
        """Тест получения статистики социальных сетей."""
        # Получение статистики
        statistics = social_media_analyzer.get_social_statistics(sample_social_data)
        # Проверки
        assert statistics is not None
        assert "platform_distribution" in statistics
        assert "sentiment_distribution" in statistics
        assert "engagement_distribution" in statistics
        assert "user_activity" in statistics
        # Проверка типов данных
        assert isinstance(statistics["platform_distribution"], dict)
        assert isinstance(statistics["sentiment_distribution"], dict)
        assert isinstance(statistics["engagement_distribution"], dict)
        assert isinstance(statistics["user_activity"], dict)
    def test_error_handling(self, social_media_analyzer: SocialMediaAnalyzer) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            social_media_analyzer.analyze_sentiment(None)
        with pytest.raises(ValueError):
            social_media_analyzer.validate_social_data(None)
    def test_edge_cases(self, social_media_analyzer: SocialMediaAnalyzer) -> None:
        """Тест граничных случаев."""
        # Тест с пустыми данными
        empty_data: List[Dict[str, Any]] = []
        result = social_media_analyzer.analyze_sentiment(empty_data)
        assert result is not None
        assert isinstance(result, dict)
        # Тест с очень большими данными
        large_data = [{"id": f"post_{i}", "content": "x" * 1000} for i in range(1000)]
        metrics = social_media_analyzer.calculate_social_metrics(large_data)
        assert metrics is not None
    def test_cleanup(self, social_media_analyzer: SocialMediaAnalyzer) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        social_media_analyzer.cleanup()
        # Проверка, что ресурсы освобождены
        assert social_media_analyzer.platform_connectors == {}
        assert social_media_analyzer.sentiment_analyzers == {}
        assert social_media_analyzer.trend_detectors == {} 
