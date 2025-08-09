"""
Unit тесты для Social Media.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
from unittest.mock import Mock, patch

from domain.entities.social_media import SocialPost, SocialAnalysis, SocialCollection, SocialSentiment, SocialPlatform


class TestSocialSentiment:
    """Тесты для SocialSentiment."""

    def test_enum_values(self):
        """Тест значений enum SocialSentiment."""
        assert SocialSentiment.POSITIVE.value == "positive"
        assert SocialSentiment.NEGATIVE.value == "negative"
        assert SocialSentiment.NEUTRAL.value == "neutral"
        assert SocialSentiment.MIXED.value == "mixed"


class TestSocialPlatform:
    """Тесты для SocialPlatform."""

    def test_enum_values(self):
        """Тест значений enum SocialPlatform."""
        assert SocialPlatform.TWITTER.value == "twitter"
        assert SocialPlatform.REDDIT.value == "reddit"
        assert SocialPlatform.TELEGRAM.value == "telegram"
        assert SocialPlatform.DISCORD.value == "discord"
        assert SocialPlatform.TWITCH.value == "twitch"
        assert SocialPlatform.YOUTUBE.value == "youtube"
        assert SocialPlatform.OTHER.value == "other"


class TestSocialPost:
    """Тесты для SocialPost."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": "post_001",
            "platform": SocialPlatform.TWITTER,
            "author": "@crypto_trader",
            "content": "Bitcoin is going to the moon! 🚀 #BTC #crypto",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.POSITIVE,
            "engagement_score": 0.8,
            "reach_score": 0.7,
            "symbols": ["BTC", "ETH"],
            "metadata": {"likes": 150, "retweets": 25, "replies": 10},
        }

    @pytest.fixture
    def social_post(self, sample_data) -> SocialPost:
        """Создает тестовый пост в социальных медиа."""
        return SocialPost(**sample_data)

    def test_creation(self, sample_data):
        """Тест создания поста в социальных медиа."""
        post = SocialPost(**sample_data)

        assert post.id == sample_data["id"]
        assert post.platform == sample_data["platform"]
        assert post.author == sample_data["author"]
        assert post.content == sample_data["content"]
        assert post.published_at == sample_data["published_at"]
        assert post.sentiment == sample_data["sentiment"]
        assert post.engagement_score == sample_data["engagement_score"]
        assert post.reach_score == sample_data["reach_score"]
        assert post.symbols == sample_data["symbols"]
        assert post.metadata == sample_data["metadata"]

    def test_validation_empty_id(self):
        """Тест валидации пустого ID."""
        data = {
            "id": "",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "Test content",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": 0.5,
            "reach_score": 0.5,
            "symbols": [],
            "metadata": {},
        }

        with pytest.raises(ValueError, match="Post ID cannot be empty"):
            SocialPost(**data)

    def test_validation_empty_content(self):
        """Тест валидации пустого содержимого."""
        data = {
            "id": "test_id",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": 0.5,
            "reach_score": 0.5,
            "symbols": [],
            "metadata": {},
        }

        with pytest.raises(ValueError, match="Post content cannot be empty"):
            SocialPost(**data)

    def test_validation_engagement_score_below_zero(self):
        """Тест валидации engagement_score ниже 0."""
        data = {
            "id": "test_id",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "Test content",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": -0.1,
            "reach_score": 0.5,
            "symbols": [],
            "metadata": {},
        }

        with pytest.raises(ValueError, match="Engagement score must be between 0 and 1"):
            SocialPost(**data)

    def test_validation_engagement_score_above_one(self):
        """Тест валидации engagement_score выше 1."""
        data = {
            "id": "test_id",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "Test content",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": 1.1,
            "reach_score": 0.5,
            "symbols": [],
            "metadata": {},
        }

        with pytest.raises(ValueError, match="Engagement score must be between 0 and 1"):
            SocialPost(**data)

    def test_validation_reach_score_below_zero(self):
        """Тест валидации reach_score ниже 0."""
        data = {
            "id": "test_id",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "Test content",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": 0.5,
            "reach_score": -0.1,
            "symbols": [],
            "metadata": {},
        }

        with pytest.raises(ValueError, match="Reach score must be between 0 and 1"):
            SocialPost(**data)

    def test_validation_reach_score_above_one(self):
        """Тест валидации reach_score выше 1."""
        data = {
            "id": "test_id",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "Test content",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": 0.5,
            "reach_score": 1.1,
            "symbols": [],
            "metadata": {},
        }

        with pytest.raises(ValueError, match="Reach score must be between 0 and 1"):
            SocialPost(**data)

    def test_valid_boundary_values(self):
        """Тест валидных граничных значений."""
        data = {
            "id": "test_id",
            "platform": SocialPlatform.TWITTER,
            "author": "test_author",
            "content": "Test content",
            "published_at": datetime.now(),
            "sentiment": SocialSentiment.NEUTRAL,
            "engagement_score": 0.0,  # Граничное значение
            "reach_score": 1.0,  # Граничное значение
            "symbols": [],
            "metadata": {},
        }

        post = SocialPost(**data)
        assert post.engagement_score == 0.0
        assert post.reach_score == 1.0


class TestSocialAnalysis:
    """Тесты для SocialAnalysis."""

    @pytest.fixture
    def social_post(self) -> SocialPost:
        """Создает тестовый пост в социальных медиа."""
        return SocialPost(
            id="test_post",
            platform=SocialPlatform.TWITTER,
            author="@test_user",
            content="Test post content",
            published_at=datetime.now(),
            sentiment=SocialSentiment.POSITIVE,
            engagement_score=0.8,
            reach_score=0.7,
            symbols=["BTC"],
            metadata={},
        )

    @pytest.fixture
    def sample_data(self, social_post) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "post": social_post,
            "sentiment_score": 0.8,
            "market_impact_prediction": 0.7,
            "confidence": 0.9,
            "trading_signals": ["BUY", "HOLD"],
            "risk_level": "medium",
        }

    @pytest.fixture
    def social_analysis(self, sample_data) -> SocialAnalysis:
        """Создает тестовый анализ социальных медиа."""
        return SocialAnalysis(**sample_data)

    def test_creation(self, sample_data):
        """Тест создания анализа социальных медиа."""
        analysis = SocialAnalysis(**sample_data)

        assert analysis.post == sample_data["post"]
        assert analysis.sentiment_score == sample_data["sentiment_score"]
        assert analysis.market_impact_prediction == sample_data["market_impact_prediction"]
        assert analysis.confidence == sample_data["confidence"]
        assert analysis.trading_signals == sample_data["trading_signals"]
        assert analysis.risk_level == sample_data["risk_level"]

    def test_validation_sentiment_score_below_minus_one(self):
        """Тест валидации sentiment_score ниже -1."""
        data = {
            "post": Mock(spec=SocialPost),
            "sentiment_score": -1.1,
            "market_impact_prediction": 0.5,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low",
        }

        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            SocialAnalysis(**data)

    def test_validation_sentiment_score_above_one(self):
        """Тест валидации sentiment_score выше 1."""
        data = {
            "post": Mock(spec=SocialPost),
            "sentiment_score": 1.1,
            "market_impact_prediction": 0.5,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low",
        }

        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            SocialAnalysis(**data)

    def test_validation_market_impact_prediction_below_zero(self):
        """Тест валидации market_impact_prediction ниже 0."""
        data = {
            "post": Mock(spec=SocialPost),
            "sentiment_score": 0.5,
            "market_impact_prediction": -0.1,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low",
        }

        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            SocialAnalysis(**data)

    def test_validation_market_impact_prediction_above_one(self):
        """Тест валидации market_impact_prediction выше 1."""
        data = {
            "post": Mock(spec=SocialPost),
            "sentiment_score": 0.5,
            "market_impact_prediction": 1.1,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low",
        }

        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            SocialAnalysis(**data)

    def test_validation_confidence_below_zero(self):
        """Тест валидации confidence ниже 0."""
        data = {
            "post": Mock(spec=SocialPost),
            "sentiment_score": 0.5,
            "market_impact_prediction": 0.5,
            "confidence": -0.1,
            "trading_signals": [],
            "risk_level": "low",
        }

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SocialAnalysis(**data)

    def test_validation_confidence_above_one(self):
        """Тест валидации confidence выше 1."""
        data = {
            "post": Mock(spec=SocialPost),
            "sentiment_score": 0.5,
            "market_impact_prediction": 0.5,
            "confidence": 1.1,
            "trading_signals": [],
            "risk_level": "low",
        }

        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SocialAnalysis(**data)

    def test_valid_boundary_values(self, social_post):
        """Тест валидных граничных значений."""
        data = {
            "post": social_post,
            "sentiment_score": -1.0,  # Граничное значение
            "market_impact_prediction": 0.0,  # Граничное значение
            "confidence": 1.0,  # Граничное значение
            "trading_signals": [],
            "risk_level": "high",
        }

        analysis = SocialAnalysis(**data)
        assert analysis.sentiment_score == -1.0
        assert analysis.market_impact_prediction == 0.0
        assert analysis.confidence == 1.0


class TestSocialCollection:
    """Тесты для SocialCollection."""

    @pytest.fixture
    def social_posts(self) -> List[SocialPost]:
        """Создает список тестовых постов в социальных медиа."""
        return [
            SocialPost(
                id=f"post_{i}",
                platform=SocialPlatform.TWITTER,
                author=f"user_{i}",
                content=f"Post content {i}",
                published_at=datetime.now(),
                sentiment=SocialSentiment.NEUTRAL,
                engagement_score=0.5,
                reach_score=0.5,
                symbols=[],
                metadata={},
            )
            for i in range(3)
        ]

    @pytest.fixture
    def time_range(self) -> Tuple[datetime, datetime]:
        """Создает временной диапазон."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        return (start_time, end_time)

    @pytest.fixture
    def sample_data(self, social_posts, time_range) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "posts": social_posts,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": time_range,
            "platform": SocialPlatform.TWITTER,
        }

    @pytest.fixture
    def social_collection(self, sample_data) -> SocialCollection:
        """Создает тестовую коллекцию постов в социальных медиа."""
        return SocialCollection(**sample_data)

    def test_creation(self, sample_data):
        """Тест создания коллекции постов в социальных медиа."""
        collection = SocialCollection(**sample_data)

        assert collection.posts == sample_data["posts"]
        assert collection.total_count == sample_data["total_count"]
        assert collection.filtered_count == sample_data["filtered_count"]
        assert collection.time_range == sample_data["time_range"]
        assert collection.platform == sample_data["platform"]

    def test_validation_negative_total_count(self, social_posts, time_range):
        """Тест валидации отрицательного total_count."""
        data = {
            "posts": social_posts,
            "total_count": -1,
            "filtered_count": 3,
            "time_range": time_range,
            "platform": SocialPlatform.TWITTER,
        }

        with pytest.raises(ValueError, match="Total count cannot be negative"):
            SocialCollection(**data)

    def test_validation_negative_filtered_count(self, social_posts, time_range):
        """Тест валидации отрицательного filtered_count."""
        data = {
            "posts": social_posts,
            "total_count": 10,
            "filtered_count": -1,
            "time_range": time_range,
            "platform": SocialPlatform.TWITTER,
        }

        with pytest.raises(ValueError, match="Filtered count cannot be negative"):
            SocialCollection(**data)

    def test_validation_filtered_count_exceeds_total(self, social_posts, time_range):
        """Тест валидации filtered_count больше total_count."""
        data = {
            "posts": social_posts,
            "total_count": 5,
            "filtered_count": 10,
            "time_range": time_range,
            "platform": SocialPlatform.TWITTER,
        }

        with pytest.raises(ValueError, match="Filtered count cannot exceed total count"):
            SocialCollection(**data)

    def test_validation_invalid_time_range_length(self, social_posts):
        """Тест валидации неправильной длины time_range."""
        data = {
            "posts": social_posts,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": (datetime.now(),),  # Только один элемент
            "platform": SocialPlatform.TWITTER,
        }

        with pytest.raises(ValueError, match="Time range must have exactly 2 elements"):
            SocialCollection(**data)

    def test_validation_invalid_time_range_order(self, social_posts):
        """Тест валидации неправильного порядка времени."""
        start_time = datetime.now()
        end_time = start_time - timedelta(hours=1)  # end_time раньше start_time

        data = {
            "posts": social_posts,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": (start_time, end_time),
            "platform": SocialPlatform.TWITTER,
        }

        with pytest.raises(ValueError, match="Start time must be before end time"):
            SocialCollection(**data)

    def test_validation_equal_time_range(self, social_posts):
        """Тест валидации равных временных меток."""
        time = datetime.now()

        data = {
            "posts": social_posts,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": (time, time),  # Одинаковые временные метки
            "platform": SocialPlatform.TWITTER,
        }

        with pytest.raises(ValueError, match="Start time must be before end time"):
            SocialCollection(**data)

    def test_valid_boundary_values(self, social_posts, time_range):
        """Тест валидных граничных значений."""
        data = {
            "posts": social_posts,
            "total_count": 0,  # Граничное значение
            "filtered_count": 0,  # Граничное значение
            "time_range": time_range,
            "platform": SocialPlatform.TWITTER,
        }

        collection = SocialCollection(**data)
        assert collection.total_count == 0
        assert collection.filtered_count == 0

    def test_valid_equal_counts(self, social_posts, time_range):
        """Тест валидного случая когда filtered_count равен total_count."""
        data = {
            "posts": social_posts,
            "total_count": 3,
            "filtered_count": 3,  # Равен total_count
            "time_range": time_range,
            "platform": SocialPlatform.TWITTER,
        }

        collection = SocialCollection(**data)
        assert collection.total_count == 3
        assert collection.filtered_count == 3

    def test_different_platforms(self, social_posts, time_range):
        """Тест различных платформ социальных медиа."""
        platforms = [
            SocialPlatform.TWITTER,
            SocialPlatform.REDDIT,
            SocialPlatform.TELEGRAM,
            SocialPlatform.DISCORD,
            SocialPlatform.TWITCH,
            SocialPlatform.YOUTUBE,
            SocialPlatform.OTHER,
        ]

        for platform in platforms:
            data = {
                "posts": social_posts,
                "total_count": 10,
                "filtered_count": 3,
                "time_range": time_range,
                "platform": platform,
            }

            collection = SocialCollection(**data)
            assert collection.platform == platform
