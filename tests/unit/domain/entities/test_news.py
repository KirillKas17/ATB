"""
Unit тесты для News.

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

from domain.entities.news import (
    NewsItem, NewsAnalysis, NewsCollection,
    NewsSentiment, NewsCategory
)


class TestNewsSentiment:
    """Тесты для NewsSentiment."""
    
    def test_enum_values(self):
        """Тест значений enum NewsSentiment."""
        assert NewsSentiment.POSITIVE.value == "positive"
        assert NewsSentiment.NEGATIVE.value == "negative"
        assert NewsSentiment.NEUTRAL.value == "neutral"


class TestNewsCategory:
    """Тесты для NewsCategory."""
    
    def test_enum_values(self):
        """Тест значений enum NewsCategory."""
        assert NewsCategory.ECONOMIC.value == "economic"
        assert NewsCategory.POLITICAL.value == "political"
        assert NewsCategory.TECHNICAL.value == "technical"
        assert NewsCategory.REGULATORY.value == "regulatory"
        assert NewsCategory.MARKET.value == "market"
        assert NewsCategory.OTHER.value == "other"


class TestNewsItem:
    """Тесты для NewsItem."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": "news_001",
            "title": "Bitcoin Reaches New All-Time High",
            "content": "Bitcoin has reached a new all-time high of $50,000...",
            "source": "CryptoNews",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.POSITIVE,
            "category": NewsCategory.MARKET,
            "relevance_score": 0.8,
            "impact_score": 0.7,
            "symbols": ["BTC", "ETH"],
            "metadata": {"author": "John Doe", "read_time": 3}
        }
    
    @pytest.fixture
    def news_item(self, sample_data) -> NewsItem:
        """Создает тестовый новостной элемент."""
        return NewsItem(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания новостного элемента."""
        news_item = NewsItem(**sample_data)
        
        assert news_item.id == sample_data["id"]
        assert news_item.title == sample_data["title"]
        assert news_item.content == sample_data["content"]
        assert news_item.source == sample_data["source"]
        assert news_item.published_at == sample_data["published_at"]
        assert news_item.sentiment == sample_data["sentiment"]
        assert news_item.category == sample_data["category"]
        assert news_item.relevance_score == sample_data["relevance_score"]
        assert news_item.impact_score == sample_data["impact_score"]
        assert news_item.symbols == sample_data["symbols"]
        assert news_item.metadata == sample_data["metadata"]
    
    def test_validation_empty_id(self):
        """Тест валидации пустого ID."""
        data = {
            "id": "",
            "title": "Test Title",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 0.5,
            "impact_score": 0.5,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="News ID cannot be empty"):
            NewsItem(**data)
    
    def test_validation_empty_title(self):
        """Тест валидации пустого заголовка."""
        data = {
            "id": "test_id",
            "title": "",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 0.5,
            "impact_score": 0.5,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="News title cannot be empty"):
            NewsItem(**data)
    
    def test_validation_empty_content(self):
        """Тест валидации пустого содержимого."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 0.5,
            "impact_score": 0.5,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="News content cannot be empty"):
            NewsItem(**data)
    
    def test_validation_relevance_score_below_zero(self):
        """Тест валидации relevance_score ниже 0."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": -0.1,
            "impact_score": 0.5,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="Relevance score must be between 0 and 1"):
            NewsItem(**data)
    
    def test_validation_relevance_score_above_one(self):
        """Тест валидации relevance_score выше 1."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 1.1,
            "impact_score": 0.5,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="Relevance score must be between 0 and 1"):
            NewsItem(**data)
    
    def test_validation_impact_score_below_zero(self):
        """Тест валидации impact_score ниже 0."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 0.5,
            "impact_score": -0.1,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="Impact score must be between 0 and 1"):
            NewsItem(**data)
    
    def test_validation_impact_score_above_one(self):
        """Тест валидации impact_score выше 1."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 0.5,
            "impact_score": 1.1,
            "symbols": [],
            "metadata": {}
        }
        
        with pytest.raises(ValueError, match="Impact score must be between 0 and 1"):
            NewsItem(**data)
    
    def test_valid_boundary_values(self):
        """Тест валидных граничных значений."""
        data = {
            "id": "test_id",
            "title": "Test Title",
            "content": "Test Content",
            "source": "Test Source",
            "published_at": datetime.now(),
            "sentiment": NewsSentiment.NEUTRAL,
            "category": NewsCategory.OTHER,
            "relevance_score": 0.0,  # Граничное значение
            "impact_score": 1.0,     # Граничное значение
            "symbols": [],
            "metadata": {}
        }
        
        news_item = NewsItem(**data)
        assert news_item.relevance_score == 0.0
        assert news_item.impact_score == 1.0


class TestNewsAnalysis:
    """Тесты для NewsAnalysis."""
    
    @pytest.fixture
    def news_item(self) -> NewsItem:
        """Создает тестовый новостной элемент."""
        return NewsItem(
            id="test_news",
            title="Test News",
            content="Test content",
            source="Test Source",
            published_at=datetime.now(),
            sentiment=NewsSentiment.POSITIVE,
            category=NewsCategory.MARKET,
            relevance_score=0.8,
            impact_score=0.7,
            symbols=["BTC"],
            metadata={}
        )
    
    @pytest.fixture
    def sample_data(self, news_item) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "news_item": news_item,
            "sentiment_score": 0.8,
            "market_impact_prediction": 0.7,
            "confidence": 0.9,
            "trading_signals": ["BUY", "HOLD"],
            "risk_level": "medium"
        }
    
    @pytest.fixture
    def news_analysis(self, sample_data) -> NewsAnalysis:
        """Создает тестовый анализ новостей."""
        return NewsAnalysis(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания анализа новостей."""
        analysis = NewsAnalysis(**sample_data)
        
        assert analysis.news_item == sample_data["news_item"]
        assert analysis.sentiment_score == sample_data["sentiment_score"]
        assert analysis.market_impact_prediction == sample_data["market_impact_prediction"]
        assert analysis.confidence == sample_data["confidence"]
        assert analysis.trading_signals == sample_data["trading_signals"]
        assert analysis.risk_level == sample_data["risk_level"]
    
    def test_validation_sentiment_score_below_minus_one(self):
        """Тест валидации sentiment_score ниже -1."""
        data = {
            "news_item": Mock(spec=NewsItem),
            "sentiment_score": -1.1,
            "market_impact_prediction": 0.5,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low"
        }
        
        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            NewsAnalysis(**data)
    
    def test_validation_sentiment_score_above_one(self):
        """Тест валидации sentiment_score выше 1."""
        data = {
            "news_item": Mock(spec=NewsItem),
            "sentiment_score": 1.1,
            "market_impact_prediction": 0.5,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low"
        }
        
        with pytest.raises(ValueError, match="Sentiment score must be between -1 and 1"):
            NewsAnalysis(**data)
    
    def test_validation_market_impact_prediction_below_zero(self):
        """Тест валидации market_impact_prediction ниже 0."""
        data = {
            "news_item": Mock(spec=NewsItem),
            "sentiment_score": 0.5,
            "market_impact_prediction": -0.1,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low"
        }
        
        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            NewsAnalysis(**data)
    
    def test_validation_market_impact_prediction_above_one(self):
        """Тест валидации market_impact_prediction выше 1."""
        data = {
            "news_item": Mock(spec=NewsItem),
            "sentiment_score": 0.5,
            "market_impact_prediction": 1.1,
            "confidence": 0.8,
            "trading_signals": [],
            "risk_level": "low"
        }
        
        with pytest.raises(ValueError, match="Market impact prediction must be between 0 and 1"):
            NewsAnalysis(**data)
    
    def test_validation_confidence_below_zero(self):
        """Тест валидации confidence ниже 0."""
        data = {
            "news_item": Mock(spec=NewsItem),
            "sentiment_score": 0.5,
            "market_impact_prediction": 0.5,
            "confidence": -0.1,
            "trading_signals": [],
            "risk_level": "low"
        }
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            NewsAnalysis(**data)
    
    def test_validation_confidence_above_one(self):
        """Тест валидации confidence выше 1."""
        data = {
            "news_item": Mock(spec=NewsItem),
            "sentiment_score": 0.5,
            "market_impact_prediction": 0.5,
            "confidence": 1.1,
            "trading_signals": [],
            "risk_level": "low"
        }
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            NewsAnalysis(**data)
    
    def test_valid_boundary_values(self, news_item):
        """Тест валидных граничных значений."""
        data = {
            "news_item": news_item,
            "sentiment_score": -1.0,  # Граничное значение
            "market_impact_prediction": 0.0,  # Граничное значение
            "confidence": 1.0,  # Граничное значение
            "trading_signals": [],
            "risk_level": "high"
        }
        
        analysis = NewsAnalysis(**data)
        assert analysis.sentiment_score == -1.0
        assert analysis.market_impact_prediction == 0.0
        assert analysis.confidence == 1.0


class TestNewsCollection:
    """Тесты для NewsCollection."""
    
    @pytest.fixture
    def news_items(self) -> List[NewsItem]:
        """Создает список тестовых новостных элементов."""
        return [
            NewsItem(
                id=f"news_{i}",
                title=f"News {i}",
                content=f"Content {i}",
                source="Test Source",
                published_at=datetime.now(),
                sentiment=NewsSentiment.NEUTRAL,
                category=NewsCategory.OTHER,
                relevance_score=0.5,
                impact_score=0.5,
                symbols=[],
                metadata={}
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
    def sample_data(self, news_items, time_range) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "items": news_items,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": time_range
        }
    
    @pytest.fixture
    def news_collection(self, sample_data) -> NewsCollection:
        """Создает тестовую коллекцию новостей."""
        return NewsCollection(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания коллекции новостей."""
        collection = NewsCollection(**sample_data)
        
        assert collection.items == sample_data["items"]
        assert collection.total_count == sample_data["total_count"]
        assert collection.filtered_count == sample_data["filtered_count"]
        assert collection.time_range == sample_data["time_range"]
    
    def test_validation_negative_total_count(self, news_items, time_range):
        """Тест валидации отрицательного total_count."""
        data = {
            "items": news_items,
            "total_count": -1,
            "filtered_count": 3,
            "time_range": time_range
        }
        
        with pytest.raises(ValueError, match="Total count cannot be negative"):
            NewsCollection(**data)
    
    def test_validation_negative_filtered_count(self, news_items, time_range):
        """Тест валидации отрицательного filtered_count."""
        data = {
            "items": news_items,
            "total_count": 10,
            "filtered_count": -1,
            "time_range": time_range
        }
        
        with pytest.raises(ValueError, match="Filtered count cannot be negative"):
            NewsCollection(**data)
    
    def test_validation_filtered_count_exceeds_total(self, news_items, time_range):
        """Тест валидации filtered_count больше total_count."""
        data = {
            "items": news_items,
            "total_count": 5,
            "filtered_count": 10,
            "time_range": time_range
        }
        
        with pytest.raises(ValueError, match="Filtered count cannot exceed total count"):
            NewsCollection(**data)
    
    def test_validation_invalid_time_range_length(self, news_items):
        """Тест валидации неправильной длины time_range."""
        data = {
            "items": news_items,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": (datetime.now(),)  # Только один элемент
        }
        
        with pytest.raises(ValueError, match="Time range must have exactly 2 elements"):
            NewsCollection(**data)
    
    def test_validation_invalid_time_range_order(self, news_items):
        """Тест валидации неправильного порядка времени."""
        start_time = datetime.now()
        end_time = start_time - timedelta(hours=1)  # end_time раньше start_time
        
        data = {
            "items": news_items,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": (start_time, end_time)
        }
        
        with pytest.raises(ValueError, match="Start time must be before end time"):
            NewsCollection(**data)
    
    def test_validation_equal_time_range(self, news_items):
        """Тест валидации равных временных меток."""
        time = datetime.now()
        
        data = {
            "items": news_items,
            "total_count": 10,
            "filtered_count": 3,
            "time_range": (time, time)  # Одинаковые временные метки
        }
        
        with pytest.raises(ValueError, match="Start time must be before end time"):
            NewsCollection(**data)
    
    def test_valid_boundary_values(self, news_items, time_range):
        """Тест валидных граничных значений."""
        data = {
            "items": news_items,
            "total_count": 0,  # Граничное значение
            "filtered_count": 0,  # Граничное значение
            "time_range": time_range
        }
        
        collection = NewsCollection(**data)
        assert collection.total_count == 0
        assert collection.filtered_count == 0
    
    def test_valid_equal_counts(self, news_items, time_range):
        """Тест валидного случая когда filtered_count равен total_count."""
        data = {
            "items": news_items,
            "total_count": 3,
            "filtered_count": 3,  # Равен total_count
            "time_range": time_range
        }
        
        collection = NewsCollection(**data)
        assert collection.total_count == 3
        assert collection.filtered_count == 3 