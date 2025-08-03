"""
Тесты для сущностей социальных медиа.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any

from domain.entities.social_media import (
    SocialPost,
    SocialAnalysis,
    SocialCollection,
    SocialSentiment,
    SocialPlatform,
)


class TestSocialSentiment:
    """Тесты для перечисления настроений в социальных медиа."""

    def test_sentiment_values(self) -> None:
        """Тест значений настроений."""
        assert SocialSentiment.POSITIVE.value == "positive"
        assert SocialSentiment.NEGATIVE.value == "negative"
        assert SocialSentiment.NEUTRAL.value == "neutral"
        assert SocialSentiment.MIXED.value == "mixed"


class TestSocialPlatform:
    """Тесты для перечисления платформ социальных медиа."""

    def test_platform_values(self) -> None:
        """Тест значений платформ."""
        assert SocialPlatform.TWITTER.value == "twitter"
        assert SocialPlatform.REDDIT.value == "reddit"
        assert SocialPlatform.TELEGRAM.value == "telegram"
        assert SocialPlatform.DISCORD.value == "discord"
        assert SocialPlatform.TWITCH.value == "twitch"
        assert SocialPlatform.YOUTUBE.value == "youtube"
        assert SocialPlatform.OTHER.value == "other"


class TestSocialPost:
    """Тесты для сущности поста в социальных медиа."""

    @pytest.fixture
    def valid_social_post(self) -> SocialPost:
        """Фикстура с валидным постом в социальных медиа."""
        return SocialPost(
            id="post_001",
            platform=SocialPlatform.TWITTER,
            author="test_user",
            content="Test post content about BTC",
            published_at=datetime.now(),
            sentiment=SocialSentiment.POSITIVE,
            engagement_score=0.8,
            reach_score=0.6,
            symbols=["BTC/USD", "ETH/USD"],
            metadata={"likes": 100, "retweets": 50},
        )

    def test_social_post_creation(self, valid_social_post: SocialPost) -> None:
        """Тест создания поста в социальных медиа."""
        assert valid_social_post.id == "post_001"
        assert valid_social_post.platform == SocialPlatform.TWITTER
        assert valid_social_post.author == "test_user"
        assert valid_social_post.content == "Test post content about BTC"
        assert isinstance(valid_social_post.published_at, datetime)
        assert valid_social_post.sentiment == SocialSentiment.POSITIVE
        assert valid_social_post.engagement_score == 0.8
        assert valid_social_post.reach_score == 0.6
        assert valid_social_post.symbols == ["BTC/USD", "ETH/USD"]
        assert valid_social_post.metadata == {"likes": 100, "retweets": 50}

    def test_social_post_validation_empty_id(self) -> None:
        """Тест валидации пустого ID."""
        with pytest.raises(ValueError, match="Post ID cannot be empty"):
            SocialPost(
                id="",
                platform=SocialPlatform.TWITTER,
                author="test_user",
                content="Test content",
                published_at=datetime.now(),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=0.8,
                reach_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_social_post_validation_empty_content(self) -> None:
        """Тест валидации пустого содержимого."""
        with pytest.raises(ValueError, match="Post content cannot be empty"):
            SocialPost(
                id="post_001",
                platform=SocialPlatform.TWITTER,
                author="test_user",
                content="",
                published_at=datetime.now(),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=0.8,
                reach_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_social_post_validation_engagement_score_too_low(self) -> None:
        """Тест валидации слишком низкого engagement_score."""
        with pytest.raises(ValueError, match="Engagement score must be between 0 and 1"):
            SocialPost(
                id="post_001",
                platform=SocialPlatform.TWITTER,
                author="test_user",
                content="Test content",
                published_at=datetime.now(),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=-0.1,
                reach_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_social_post_validation_engagement_score_too_high(self) -> None:
        """Тест валидации слишком высокого engagement_score."""
        with pytest.raises(ValueError, match="Engagement score must be between 0 and 1"):
            SocialPost(
                id="post_001",
                platform=SocialPlatform.TWITTER,
                author="test_user",
                content="Test content",
                published_at=datetime.now(),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=1.1,
                reach_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_social_post_validation_reach_score_too_low(self) -> None:
        """Тест валидации слишком низкого reach_score."""
        with pytest.raises(ValueError, match="Reach score must be between 0 and 1"):
            SocialPost(
                id="post_001",
                platform=SocialPlatform.TWITTER,
                author="test_user",
                content="Test content",
                published_at=datetime.now(),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=0.8,
                reach_score=-0.1,
                symbols=[],
                metadata={},
            )

    def test_social_post_validation_reach_score_too_high(self) -> None:
        """Тест валидации слишком высокого reach_score."""
        with pytest.raises(ValueError, match="Reach score must be between 0 and 1"):
            SocialPost(
                id="post_001",
                platform=SocialPlatform.TWITTER,
                author="test_user",
                content="Test content",
                published_at=datetime.now(),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=0.8,
                reach_score=1.1,
                symbols=[],
                metadata={},
            )

    def test_social_post_boundary_values(self) -> None:
        """Тест граничных значений для scores."""
        post = SocialPost(
            id="post_001",
            platform=SocialPlatform.REDDIT,
            author="test_user",
            content="Test content",
            published_at=datetime.now(),
            sentiment=SocialSentiment.NEUTRAL,
            engagement_score=0.0,
            reach_score=1.0,
            symbols=[],
            metadata={},
        )
        assert post.engagement_score == 0.0
        assert post.reach_score == 1.0


class TestSocialAnalysis:
    """Тесты для результата анализа социальных медиа."""

    @pytest.fixture
    def valid_social_post(self) -> SocialPost:
        """Фикстура с валидным постом в социальных медиа."""
        return SocialPost(
            id="post_001",
            platform=SocialPlatform.TELEGRAM,
            author="test_user",
            content="Test post content",
            published_at=datetime.now(),
            sentiment=SocialSentiment.POSITIVE,
            engagement_score=0.8,
            reach_score=0.6,
            symbols=["BTC/USD"],
            metadata={},
        )

    @pytest.fixture
    def valid_social_analysis(self, valid_social_post: SocialPost) -> SocialAnalysis:
        """Фикстура с валидным анализом социальных медиа."""
        return SocialAnalysis(
            post=valid_social_post,
            sentiment_score=0.7,
            market_impact_prediction=0.5,
            confidence=0.8,
            trading_signals=["BUY", "HOLD"],
            risk_level="MEDIUM",
        )

    def test_social_analysis_creation(self, valid_social_analysis: SocialAnalysis) -> None:
        """Тест создания анализа социальных медиа."""
        assert valid_social_analysis.post.id == "post_001"
        assert valid_social_analysis.sentiment_score == 0.7
        assert valid_social_analysis.market_impact_prediction == 0.5
        assert valid_social_analysis.confidence == 0.8
        assert valid_social_analysis.trading_signals == ["BUY", "HOLD"]
        assert valid_social_analysis.risk_level == "MEDIUM"

    def test_social_analysis_validation_sentiment_score_too_low(self, valid_social_post: SocialPost) -> None:
        """Тест валидации слишком низкого sentiment_score."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            SocialAnalysis(
                post=valid_social_post,
                sentiment_score=-1.1,
                market_impact_prediction=0.5,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_social_analysis_validation_sentiment_score_too_high(self, valid_social_post: SocialPost) -> None:
        """Тест валидации слишком высокого sentiment_score."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            SocialAnalysis(
                post=valid_social_post,
                sentiment_score=1.1,
                market_impact_prediction=0.5,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_social_analysis_validation_market_impact_too_low(self, valid_social_post: SocialPost) -> None:
        """Тест валидации слишком низкого market_impact_prediction."""
        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            SocialAnalysis(
                post=valid_social_post,
                sentiment_score=0.5,
                market_impact_prediction=-0.1,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_social_analysis_validation_market_impact_too_high(self, valid_social_post: SocialPost) -> None:
        """Тест валидации слишком высокого market_impact_prediction."""
        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            SocialAnalysis(
                post=valid_social_post,
                sentiment_score=0.5,
                market_impact_prediction=1.1,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_social_analysis_validation_confidence_too_low(self, valid_social_post: SocialPost) -> None:
        """Тест валидации слишком низкого confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SocialAnalysis(
                post=valid_social_post,
                sentiment_score=0.5,
                market_impact_prediction=0.5,
                confidence=-0.1,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_social_analysis_validation_confidence_too_high(self, valid_social_post: SocialPost) -> None:
        """Тест валидации слишком высокого confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            SocialAnalysis(
                post=valid_social_post,
                sentiment_score=0.5,
                market_impact_prediction=0.5,
                confidence=1.1,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_social_analysis_boundary_values(self, valid_social_post: SocialPost) -> None:
        """Тест граничных значений."""
        analysis = SocialAnalysis(
            post=valid_social_post,
            sentiment_score=-1.0,
            market_impact_prediction=0.0,
            confidence=1.0,
            trading_signals=[],
            risk_level="HIGH",
        )
        assert analysis.sentiment_score == -1.0
        assert analysis.market_impact_prediction == 0.0
        assert analysis.confidence == 1.0


class TestSocialCollection:
    """Тесты для коллекции постов из социальных медиа."""

    @pytest.fixture
    def sample_social_posts(self) -> list[SocialPost]:
        """Фикстура с примерными постами в социальных медиа."""
        return [
            SocialPost(
                id=f"post_{i}",
                platform=SocialPlatform.TWITTER,
                author=f"user_{i}",
                content=f"Post content {i}",
                published_at=datetime.now() + timedelta(hours=i),
                sentiment=SocialSentiment.POSITIVE,
                engagement_score=0.8,
                reach_score=0.6,
                symbols=["BTC/USD"],
                metadata={},
            )
            for i in range(1, 4)
        ]

    @pytest.fixture
    def valid_time_range(self) -> tuple[datetime, datetime]:
        """Фикстура с валидным временным диапазоном."""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=24)
        return (start_time, end_time)

    def test_social_collection_creation(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест создания коллекции постов из социальных медиа."""
        collection = SocialCollection(
            posts=sample_social_posts,
            total_count=10,
            filtered_count=3,
            time_range=valid_time_range,
            platform=SocialPlatform.TWITTER,
        )
        assert len(collection.posts) == 3
        assert collection.total_count == 10
        assert collection.filtered_count == 3
        assert collection.time_range == valid_time_range
        assert collection.platform == SocialPlatform.TWITTER

    def test_social_collection_validation_negative_total_count(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест валидации отрицательного total_count."""
        with pytest.raises(ValueError, match="Total count cannot be negative"):
            SocialCollection(
                posts=sample_social_posts,
                total_count=-1,
                filtered_count=3,
                time_range=valid_time_range,
                platform=SocialPlatform.REDDIT,
            )

    def test_social_collection_validation_negative_filtered_count(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест валидации отрицательного filtered_count."""
        with pytest.raises(ValueError, match="Filtered count cannot be negative"):
            SocialCollection(
                posts=sample_social_posts,
                total_count=10,
                filtered_count=-1,
                time_range=valid_time_range,
                platform=SocialPlatform.TELEGRAM,
            )

    def test_social_collection_validation_filtered_exceeds_total(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест валидации превышения filtered_count над total_count."""
        with pytest.raises(ValueError, match="Filtered count cannot exceed total count"):
            SocialCollection(
                posts=sample_social_posts,
                total_count=2,
                filtered_count=3,
                time_range=valid_time_range,
                platform=SocialPlatform.DISCORD,
            )

    def test_social_collection_validation_invalid_time_range_length(self, sample_social_posts: list[SocialPost]) -> None:
        """Тест валидации неправильной длины time_range."""
        invalid_time_range = (datetime.now(),)  # Только один элемент
        with pytest.raises(ValueError, match="Time range must have exactly 2 elements"):
            SocialCollection(
                posts=sample_social_posts,
                total_count=10,
                filtered_count=3,
                time_range=invalid_time_range,
                platform=SocialPlatform.TWITCH,
            )

    def test_social_collection_validation_invalid_time_range_order(self, sample_social_posts: list[SocialPost]) -> None:
        """Тест валидации неправильного порядка времени."""
        start_time = datetime.now()
        end_time = start_time - timedelta(hours=1)  # Конец раньше начала
        invalid_time_range = (start_time, end_time)
        with pytest.raises(ValueError, match="Start time must be before end time"):
            SocialCollection(
                posts=sample_social_posts,
                total_count=10,
                filtered_count=3,
                time_range=invalid_time_range,
                platform=SocialPlatform.YOUTUBE,
            )

    def test_social_collection_boundary_values(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест граничных значений."""
        collection = SocialCollection(
            posts=sample_social_posts,
            total_count=0,
            filtered_count=0,
            time_range=valid_time_range,
            platform=SocialPlatform.OTHER,
        )
        assert collection.total_count == 0
        assert collection.filtered_count == 0

    def test_social_collection_equal_counts(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест равных значений total_count и filtered_count."""
        collection = SocialCollection(
            posts=sample_social_posts,
            total_count=3,
            filtered_count=3,
            time_range=valid_time_range,
            platform=SocialPlatform.TWITTER,
        )
        assert collection.total_count == 3
        assert collection.filtered_count == 3

    def test_social_collection_different_platforms(self, sample_social_posts: list[SocialPost], valid_time_range: tuple[datetime, datetime]) -> None:
        """Тест различных платформ."""
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
            collection = SocialCollection(
                posts=sample_social_posts,
                total_count=10,
                filtered_count=3,
                time_range=valid_time_range,
                platform=platform,
            )
            assert collection.platform == platform 