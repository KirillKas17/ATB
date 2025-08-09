"""
Тесты для сущностей новостей.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any

from domain.entities.news import (
    NewsItem,
    NewsAnalysis,
    NewsCollection,
    NewsSentiment,
    NewsCategory,
)


class TestNewsSentiment:
    """Тесты для перечисления настроений новостей."""

    def test_sentiment_values(self: "TestNewsSentiment") -> None:
        """Тест значений настроений."""
        assert NewsSentiment.POSITIVE.value == "positive"
        assert NewsSentiment.NEGATIVE.value == "negative"
        assert NewsSentiment.NEUTRAL.value == "neutral"


class TestNewsCategory:
    """Тесты для перечисления категорий новостей."""

    def test_category_values(self: "TestNewsCategory") -> None:
        """Тест значений категорий."""
        assert NewsCategory.ECONOMIC.value == "economic"
        assert NewsCategory.POLITICAL.value == "political"
        assert NewsCategory.TECHNICAL.value == "technical"
        assert NewsCategory.REGULATORY.value == "regulatory"
        assert NewsCategory.MARKET.value == "market"
        assert NewsCategory.OTHER.value == "other"


class TestNewsItem:
    """Тесты для сущности новостного элемента."""

    @pytest.fixture
    def valid_news_item(self) -> NewsItem:
        """Фикстура с валидным новостным элементом."""
        return NewsItem(
            id="news_001",
            title="Test News Title",
            content="Test news content",
            source="Test Source",
            published_at=datetime.now(),
            sentiment=NewsSentiment.POSITIVE,
            category=NewsCategory.ECONOMIC,
            relevance_score=0.8,
            impact_score=0.6,
            symbols=["BTC/USD", "ETH/USD"],
            metadata={"key": "value"},
        )

    def test_news_item_creation(self, valid_news_item: NewsItem) -> None:
        """Тест создания новостного элемента."""
        assert valid_news_item.id == "news_001"
        assert valid_news_item.title == "Test News Title"
        assert valid_news_item.content == "Test news content"
        assert valid_news_item.source == "Test Source"
        assert isinstance(valid_news_item.published_at, datetime)
        assert valid_news_item.sentiment == NewsSentiment.POSITIVE
        assert valid_news_item.category == NewsCategory.ECONOMIC
        assert valid_news_item.relevance_score == 0.8
        assert valid_news_item.impact_score == 0.6
        assert valid_news_item.symbols == ["BTC/USD", "ETH/USD"]
        assert valid_news_item.metadata == {"key": "value"}

    def test_news_item_validation_empty_id(self: "TestNewsItem") -> None:
        """Тест валидации пустого ID."""
        with pytest.raises(ValueError, match="News ID cannot be empty"):
            NewsItem(
                id="",
                title="Test Title",
                content="Test content",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=0.8,
                impact_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_news_item_validation_empty_title(self: "TestNewsItem") -> None:
        """Тест валидации пустого заголовка."""
        with pytest.raises(ValueError, match="News title cannot be empty"):
            NewsItem(
                id="news_001",
                title="",
                content="Test content",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=0.8,
                impact_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_news_item_validation_empty_content(self: "TestNewsItem") -> None:
        """Тест валидации пустого содержимого."""
        with pytest.raises(ValueError, match="News content cannot be empty"):
            NewsItem(
                id="news_001",
                title="Test Title",
                content="",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=0.8,
                impact_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_news_item_validation_relevance_score_too_low(self: "TestNewsItem") -> None:
        """Тест валидации слишком низкого relevance_score."""
        with pytest.raises(ValueError, match="Relevance score must be between 0 and 1"):
            NewsItem(
                id="news_001",
                title="Test Title",
                content="Test content",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=-0.1,
                impact_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_news_item_validation_relevance_score_too_high(self: "TestNewsItem") -> None:
        """Тест валидации слишком высокого relevance_score."""
        with pytest.raises(ValueError, match="Relevance score must be between 0 and 1"):
            NewsItem(
                id="news_001",
                title="Test Title",
                content="Test content",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=1.1,
                impact_score=0.6,
                symbols=[],
                metadata={},
            )

    def test_news_item_validation_impact_score_too_low(self: "TestNewsItem") -> None:
        """Тест валидации слишком низкого impact_score."""
        with pytest.raises(ValueError, match="Impact score must be between 0 and 1"):
            NewsItem(
                id="news_001",
                title="Test Title",
                content="Test content",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=0.8,
                impact_score=-0.1,
                symbols=[],
                metadata={},
            )

    def test_news_item_validation_impact_score_too_high(self: "TestNewsItem") -> None:
        """Тест валидации слишком высокого impact_score."""
        with pytest.raises(ValueError, match="Impact score must be between 0 and 1"):
            NewsItem(
                id="news_001",
                title="Test Title",
                content="Test content",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=0.8,
                impact_score=1.1,
                symbols=[],
                metadata={},
            )

    def test_news_item_boundary_values(self: "TestNewsItem") -> None:
        """Тест граничных значений для scores."""
        news_item = NewsItem(
            id="news_001",
            title="Test Title",
            content="Test content",
            source="Test Source",
            published_at=datetime.now(),
            sentiment=NewsSentiment.NEUTRAL,
            category=NewsCategory.MARKET,
            relevance_score=0.0,
            impact_score=1.0,
            symbols=[],
            metadata={},
        )
        assert news_item.relevance_score == 0.0
        assert news_item.impact_score == 1.0


class TestNewsAnalysis:
    """Тесты для результата анализа новостей."""

    @pytest.fixture
    def valid_news_item(self) -> NewsItem:
        """Фикстура с валидным новостным элементом."""
        return NewsItem(
            id="news_001",
            title="Test News Title",
            content="Test news content",
            source="Test Source",
            published_at=datetime.now(),
            sentiment=NewsSentiment.POSITIVE,
            category=NewsCategory.ECONOMIC,
            relevance_score=0.8,
            impact_score=0.6,
            symbols=["BTC/USD"],
            metadata={},
        )

    @pytest.fixture
    def valid_news_analysis(self, valid_news_item: NewsItem) -> NewsAnalysis:
        """Фикстура с валидным анализом новостей."""
        return NewsAnalysis(
            news_item=valid_news_item,
            sentiment_score=0.7,
            market_impact_prediction=0.5,
            confidence=0.8,
            trading_signals=["BUY", "HOLD"],
            risk_level="MEDIUM",
        )

    def test_news_analysis_creation(self, valid_news_analysis: NewsAnalysis) -> None:
        """Тест создания анализа новостей."""
        assert valid_news_analysis.news_item.id == "news_001"
        assert valid_news_analysis.sentiment_score == 0.7
        assert valid_news_analysis.market_impact_prediction == 0.5
        assert valid_news_analysis.confidence == 0.8
        assert valid_news_analysis.trading_signals == ["BUY", "HOLD"]
        assert valid_news_analysis.risk_level == "MEDIUM"

    def test_news_analysis_validation_sentiment_score_too_low(self, valid_news_item: NewsItem) -> None:
        """Тест валидации слишком низкого sentiment_score."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            NewsAnalysis(
                news_item=valid_news_item,
                sentiment_score=-1.1,
                market_impact_prediction=0.5,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_news_analysis_validation_sentiment_score_too_high(self, valid_news_item: NewsItem) -> None:
        """Тест валидации слишком высокого sentiment_score."""
        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            NewsAnalysis(
                news_item=valid_news_item,
                sentiment_score=1.1,
                market_impact_prediction=0.5,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_news_analysis_validation_market_impact_too_low(self, valid_news_item: NewsItem) -> None:
        """Тест валидации слишком низкого market_impact_prediction."""
        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            NewsAnalysis(
                news_item=valid_news_item,
                sentiment_score=0.5,
                market_impact_prediction=-0.1,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_news_analysis_validation_market_impact_too_high(self, valid_news_item: NewsItem) -> None:
        """Тест валидации слишком высокого market_impact_prediction."""
        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            NewsAnalysis(
                news_item=valid_news_item,
                sentiment_score=0.5,
                market_impact_prediction=1.1,
                confidence=0.8,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_news_analysis_validation_confidence_too_low(self, valid_news_item: NewsItem) -> None:
        """Тест валидации слишком низкого confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            NewsAnalysis(
                news_item=valid_news_item,
                sentiment_score=0.5,
                market_impact_prediction=0.5,
                confidence=-0.1,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_news_analysis_validation_confidence_too_high(self, valid_news_item: NewsItem) -> None:
        """Тест валидации слишком высокого confidence."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            NewsAnalysis(
                news_item=valid_news_item,
                sentiment_score=0.5,
                market_impact_prediction=0.5,
                confidence=1.1,
                trading_signals=[],
                risk_level="LOW",
            )

    def test_news_analysis_boundary_values(self, valid_news_item: NewsItem) -> None:
        """Тест граничных значений."""
        analysis = NewsAnalysis(
            news_item=valid_news_item,
            sentiment_score=-1.0,
            market_impact_prediction=0.0,
            confidence=1.0,
            trading_signals=[],
            risk_level="HIGH",
        )
        assert analysis.sentiment_score == -1.0
        assert analysis.market_impact_prediction == 0.0
        assert analysis.confidence == 1.0


class TestNewsCollection:
    """Тесты для коллекции новостей."""

    @pytest.fixture
    def sample_news_items(self) -> list[NewsItem]:
        """Фикстура с примерными новостными элементами."""
        return [
            NewsItem(
                id=f"news_{i}",
                title=f"News {i}",
                content=f"Content {i}",
                source=f"Source {i}",
                published_at=datetime.now() + timedelta(hours=i),
                sentiment=NewsSentiment.POSITIVE,
                category=NewsCategory.ECONOMIC,
                relevance_score=0.8,
                impact_score=0.6,
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

    def test_news_collection_creation(
        self, sample_news_items: list[NewsItem], valid_time_range: tuple[datetime, datetime]
    ) -> None:
        """Тест создания коллекции новостей."""
        collection = NewsCollection(
            items=sample_news_items,
            total_count=10,
            filtered_count=3,
            time_range=valid_time_range,
        )
        assert len(collection.items) == 3
        assert collection.total_count == 10
        assert collection.filtered_count == 3
        assert collection.time_range == valid_time_range

    def test_news_collection_validation_negative_total_count(
        self, sample_news_items: list[NewsItem], valid_time_range: tuple[datetime, datetime]
    ) -> None:
        """Тест валидации отрицательного total_count."""
        with pytest.raises(ValueError, match="Total count cannot be negative"):
            NewsCollection(
                items=sample_news_items,
                total_count=-1,
                filtered_count=3,
                time_range=valid_time_range,
            )

    def test_news_collection_validation_negative_filtered_count(
        self, sample_news_items: list[NewsItem], valid_time_range: tuple[datetime, datetime]
    ) -> None:
        """Тест валидации отрицательного filtered_count."""
        with pytest.raises(ValueError, match="Filtered count cannot be negative"):
            NewsCollection(
                items=sample_news_items,
                total_count=10,
                filtered_count=-1,
                time_range=valid_time_range,
            )

    def test_news_collection_validation_filtered_exceeds_total(
        self, sample_news_items: list[NewsItem], valid_time_range: tuple[datetime, datetime]
    ) -> None:
        """Тест валидации превышения filtered_count над total_count."""
        with pytest.raises(ValueError, match="Filtered count cannot exceed total count"):
            NewsCollection(
                items=sample_news_items,
                total_count=2,
                filtered_count=3,
                time_range=valid_time_range,
            )

    def test_news_collection_validation_invalid_time_range_length(self, sample_news_items: list[NewsItem]) -> None:
        """Тест валидации неправильной длины time_range."""
        invalid_time_range = (datetime.now(),)  # Только один элемент
        with pytest.raises(ValueError, match="Time range must have exactly 2 elements"):
            NewsCollection(
                items=sample_news_items,
                total_count=10,
                filtered_count=3,
                time_range=invalid_time_range,
            )

    def test_news_collection_validation_invalid_time_range_order(self, sample_news_items: list[NewsItem]) -> None:
        """Тест валидации неправильного порядка времени."""
        start_time = datetime.now()
        end_time = start_time - timedelta(hours=1)  # Конец раньше начала
        invalid_time_range = (start_time, end_time)
        with pytest.raises(ValueError, match="Start time must be before end time"):
            NewsCollection(
                items=sample_news_items,
                total_count=10,
                filtered_count=3,
                time_range=invalid_time_range,
            )

    def test_news_collection_boundary_values(
        self, sample_news_items: list[NewsItem], valid_time_range: tuple[datetime, datetime]
    ) -> None:
        """Тест граничных значений."""
        collection = NewsCollection(
            items=sample_news_items,
            total_count=0,
            filtered_count=0,
            time_range=valid_time_range,
        )
        assert collection.total_count == 0
        assert collection.filtered_count == 0

    def test_news_collection_equal_counts(
        self, sample_news_items: list[NewsItem], valid_time_range: tuple[datetime, datetime]
    ) -> None:
        """Тест равных значений total_count и filtered_count."""
        collection = NewsCollection(
            items=sample_news_items,
            total_count=3,
            filtered_count=3,
            time_range=valid_time_range,
        )
        assert collection.total_count == 3
        assert collection.filtered_count == 3
