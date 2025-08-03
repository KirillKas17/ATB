"""
Тесты для доменных моделей в domain/entities/models.py
"""
import pytest
import pandas as pd
from datetime import datetime
from decimal import Decimal

from domain.entities.models import (
    MarketData,
    Model,
    Prediction,
    Order,
    Position,
    SystemState
)


class TestMarketData:
    """Тесты для класса MarketData."""

    def test_market_data_creation(self):
        """Тест создания MarketData."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        market_data = MarketData(
            symbol="BTCUSDT",
            timeframe="1h",
            data=data
        )
        
        assert market_data.symbol == "BTCUSDT"
        assert market_data.timeframe == "1h"
        assert len(market_data.data) == 3
        assert market_data.latest_price == 105.0
        assert market_data.latest_volume == 1200.0

    def test_market_data_validation_empty_data(self):
        """Тест валидации пустых данных."""
        empty_data = pd.DataFrame()
        
        with pytest.raises(ValueError, match="data cannot be empty"):
            MarketData("BTCUSDT", "1h", empty_data)

    def test_market_data_validation_wrong_type(self):
        """Тест валидации неправильного типа данных."""
        with pytest.raises(ValueError, match="data must be a pandas DataFrame"):
            MarketData("BTCUSDT", "1h", "not a dataframe")

    def test_market_data_validation_missing_columns(self):
        """Тест валидации отсутствующих колонок."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'close': [103, 104]
            # missing 'low' and 'volume'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            MarketData("BTCUSDT", "1h", incomplete_data)

    def test_latest_price(self):
        """Тест получения последней цены."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        assert market_data.latest_price == 105.0

    def test_latest_volume(self):
        """Тест получения последнего объема."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        assert market_data.latest_volume == 1200.0

    def test_price_change(self):
        """Тест расчета изменения цены."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        expected_change = (105.0 / 103.0) - 1
        assert market_data.price_change == pytest.approx(expected_change)

    def test_price_change_single_point(self):
        """Тест изменения цены для одной точки данных."""
        data = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [103],
            'volume': [1000]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        assert market_data.price_change == 0.0

    def test_volatility(self):
        """Тест расчета волатильности."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        assert market_data.volatility > 0

    def test_volatility_single_point(self):
        """Тест волатильности для одной точки данных."""
        data = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [103],
            'volume': [1000]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        assert market_data.volatility == 0.0

    def test_get_ohlcv(self):
        """Тест получения OHLCV данных."""
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104],
            'volume': [1000, 1100]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        ohlcv = market_data.get_ohlcv()
        
        assert isinstance(ohlcv, pd.DataFrame)
        assert list(ohlcv.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert len(ohlcv) == 2

    def test_get_price_series(self):
        """Тест получения временного ряда цен."""
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104],
            'volume': [1000, 1100]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        price_series = market_data.get_price_series()
        
        assert isinstance(price_series, pd.Series)
        assert list(price_series.values) == [103, 104]

    def test_get_volume_series(self):
        """Тест получения временного ряда объемов."""
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104],
            'volume': [1000, 1100]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data)
        volume_series = market_data.get_volume_series()
        
        assert isinstance(volume_series, pd.Series)
        assert list(volume_series.values) == [1000, 1100]

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        data = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104],
            'volume': [1000, 1100]
        })
        
        market_data = MarketData("BTCUSDT", "1h", data, metadata={"test": "value"})
        result = market_data.to_dict()
        
        assert result["symbol"] == "BTCUSDT"
        assert result["timeframe"] == "1h"
        assert result["data_length"] == 2
        assert result["latest_price"] == 104.0
        assert result["metadata"] == {"test": "value"}


class TestModel:
    """Тесты для класса Model."""

    def test_model_creation(self):
        """Тест создания модели."""
        model = Model(
            name="test_model",
            type="regression",
            parameters={"learning_rate": 0.01},
            metadata={"description": "test"}
        )
        
        assert model.name == "test_model"
        assert model.type == "regression"
        assert model.parameters == {"learning_rate": 0.01}
        assert model.metadata == {"description": "test"}

    def test_model_validation_invalid_type(self):
        """Тест валидации неправильного типа модели."""
        with pytest.raises(ValueError, match="Invalid model type"):
            Model("test_model", "invalid_type")

    def test_model_valid_types(self):
        """Тест валидации правильных типов моделей."""
        valid_types = ["classification", "regression", "clustering"]
        
        for model_type in valid_types:
            model = Model("test_model", model_type)
            assert model.type == model_type

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        model = Model(
            name="test_model",
            type="classification",
            parameters={"max_depth": 10},
            metadata={"description": "test"}
        )
        
        result = model.to_dict()
        
        assert result["name"] == "test_model"
        assert result["type"] == "classification"
        assert result["parameters"] == {"max_depth": 10}
        assert result["metadata"] == {"description": "test"}


class TestPrediction:
    """Тесты для класса Prediction."""

    def test_prediction_creation(self):
        """Тест создания предсказания."""
        prediction = Prediction(
            model_name="test_model",
            symbol="BTCUSDT",
            prediction=50000.0,
            confidence=0.8,
            metadata={"source": "test"}
        )
        
        assert prediction.model_name == "test_model"
        assert prediction.symbol == "BTCUSDT"
        assert prediction.prediction == 50000.0
        assert prediction.confidence == 0.8
        assert prediction.metadata == {"source": "test"}

    def test_prediction_validation_confidence_too_low(self):
        """Тест валидации слишком низкой уверенности."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction("test_model", "BTCUSDT", 50000.0, -0.1)

    def test_prediction_validation_confidence_too_high(self):
        """Тест валидации слишком высокой уверенности."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction("test_model", "BTCUSDT", 50000.0, 1.1)

    def test_prediction_validation_confidence_boundaries(self):
        """Тест валидации граничных значений уверенности."""
        # Должно работать
        Prediction("test_model", "BTCUSDT", 50000.0, 0.0)
        Prediction("test_model", "BTCUSDT", 50000.0, 1.0)
        Prediction("test_model", "BTCUSDT", 50000.0, 0.5)

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        prediction = Prediction(
            model_name="test_model",
            symbol="BTCUSDT",
            prediction=50000.0,
            confidence=0.8,
            metadata={"source": "test"}
        )
        
        result = prediction.to_dict()
        
        assert result["model_name"] == "test_model"
        assert result["symbol"] == "BTCUSDT"
        assert result["prediction"] == 50000.0
        assert result["confidence"] == 0.8
        assert result["metadata"] == {"source": "test"}


class TestOrder:
    """Тесты для класса Order."""

    def test_order_creation(self):
        """Тест создания ордера."""
        order = Order(
            id="order_123",
            pair="BTCUSDT",
            type="limit",
            side="buy",
            price=50000.0,
            size=0.1,
            status="open",
            metadata={"strategy": "test"}
        )
        
        assert order.id == "order_123"
        assert order.pair == "BTCUSDT"
        assert order.type == "limit"
        assert order.side == "buy"
        assert order.price == 50000.0
        assert order.size == 0.1
        assert order.status == "open"
        assert order.metadata == {"strategy": "test"}

    def test_order_validation_invalid_type(self):
        """Тест валидации неправильного типа ордера."""
        with pytest.raises(ValueError, match="Invalid order type"):
            Order("order_123", "BTCUSDT", "invalid", "buy", 50000.0, 0.1, "open")

    def test_order_validation_invalid_side(self):
        """Тест валидации неправильной стороны ордера."""
        with pytest.raises(ValueError, match="Invalid order side"):
            Order("order_123", "BTCUSDT", "limit", "invalid", 50000.0, 0.1, "open")

    def test_order_validation_invalid_status(self):
        """Тест валидации неправильного статуса ордера."""
        with pytest.raises(ValueError, match="Invalid order status"):
            Order("order_123", "BTCUSDT", "limit", "buy", 50000.0, 0.1, "invalid")

    def test_order_validation_negative_price(self):
        """Тест валидации отрицательной цены."""
        with pytest.raises(ValueError, match="Price must be positive"):
            Order("order_123", "BTCUSDT", "limit", "buy", -50000.0, 0.1, "open")

    def test_order_validation_negative_size(self):
        """Тест валидации отрицательного размера."""
        with pytest.raises(ValueError, match="Size must be positive"):
            Order("order_123", "BTCUSDT", "limit", "buy", 50000.0, -0.1, "open")

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        order = Order(
            id="order_123",
            pair="BTCUSDT",
            type="limit",
            side="buy",
            price=50000.0,
            size=0.1,
            status="open",
            metadata={"strategy": "test"}
        )
        
        result = order.to_dict()
        
        assert result["id"] == "order_123"
        assert result["pair"] == "BTCUSDT"
        assert result["type"] == "limit"
        assert result["side"] == "buy"
        assert result["price"] == 50000.0
        assert result["size"] == 0.1
        assert result["status"] == "open"
        assert result["metadata"] == {"strategy": "test"}


class TestPosition:
    """Тесты для класса Position."""

    def test_position_creation(self):
        """Тест создания позиции."""
        position = Position(
            pair="BTCUSDT",
            side="long",
            size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            pnl=100.0,
            leverage=10.0,
            metadata={"strategy": "test"}
        )
        
        assert position.pair == "BTCUSDT"
        assert position.side == "long"
        assert position.size == 0.1
        assert position.entry_price == 50000.0
        assert position.current_price == 51000.0
        assert position.pnl == 100.0
        assert position.leverage == 10.0
        assert position.metadata == {"strategy": "test"}

    def test_position_validation_invalid_side(self):
        """Тест валидации неправильной стороны позиции."""
        with pytest.raises(ValueError, match="Invalid position side"):
            Position("BTCUSDT", "invalid", 0.1, 50000.0, 51000.0, 100.0, 10.0)

    def test_position_validation_negative_size(self):
        """Тест валидации отрицательного размера."""
        with pytest.raises(ValueError, match="Size must be positive"):
            Position("BTCUSDT", "long", -0.1, 50000.0, 51000.0, 100.0, 10.0)

    def test_position_validation_negative_entry_price(self):
        """Тест валидации отрицательной цены входа."""
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position("BTCUSDT", "long", 0.1, -50000.0, 51000.0, 100.0, 10.0)

    def test_position_validation_negative_current_price(self):
        """Тест валидации отрицательной текущей цены."""
        with pytest.raises(ValueError, match="Current price must be positive"):
            Position("BTCUSDT", "long", 0.1, 50000.0, -51000.0, 100.0, 10.0)

    def test_position_validation_negative_leverage(self):
        """Тест валидации отрицательного плеча."""
        with pytest.raises(ValueError, match="Leverage must be positive"):
            Position("BTCUSDT", "long", 0.1, 50000.0, 51000.0, 100.0, -10.0)

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        position = Position(
            pair="BTCUSDT",
            side="long",
            size=0.1,
            entry_price=50000.0,
            current_price=51000.0,
            pnl=100.0,
            leverage=10.0,
            metadata={"strategy": "test"}
        )
        
        result = position.to_dict()
        
        assert result["pair"] == "BTCUSDT"
        assert result["side"] == "long"
        assert result["size"] == 0.1
        assert result["entry_price"] == 50000.0
        assert result["current_price"] == 51000.0
        assert result["pnl"] == 100.0
        assert result["leverage"] == 10.0
        assert result["metadata"] == {"strategy": "test"}


class TestSystemState:
    """Тесты для класса SystemState."""

    def test_system_state_creation(self):
        """Тест создания состояния системы."""
        state = SystemState(
            is_running=True,
            is_healthy=False,
            metadata={"version": "1.0"}
        )
        
        assert state.is_running is True
        assert state.is_healthy is False
        assert state.metadata == {"version": "1.0"}

    def test_system_state_defaults(self):
        """Тест значений по умолчанию."""
        state = SystemState()
        
        assert state.is_running is False
        assert state.is_healthy is True
        assert state.metadata == {}

    def test_to_dict(self):
        """Тест преобразования в словарь."""
        state = SystemState(
            is_running=True,
            is_healthy=False,
            metadata={"version": "1.0"}
        )
        
        result = state.to_dict()
        
        assert result["is_running"] is True
        assert result["is_healthy"] is False
        assert result["metadata"] == {"version": "1.0"} 