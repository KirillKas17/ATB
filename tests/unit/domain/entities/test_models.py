"""
Unit тесты для Models.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime
from typing import Dict, Any
from unittest.mock import Mock, patch

from domain.entities.models import (
    MarketData, Model, Prediction, Order, Position, SystemState
)


class TestMarketData:
    """Тесты для MarketData."""
    
    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """Создает тестовые OHLCV данные."""
        dates = pd.date_range('2023-01-01', periods=10, freq='1H')
        data = {
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }
        return pd.DataFrame(data, index=dates)
    
    @pytest.fixture
    def sample_data(self, sample_ohlcv_data) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "data": sample_ohlcv_data,
            "timestamp": datetime.now(),
            "metadata": {"source": "binance", "quality": "high"}
        }
    
    @pytest.fixture
    def market_data(self, sample_data) -> MarketData:
        """Создает тестовые рыночные данные."""
        return MarketData(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания рыночных данных."""
        market_data = MarketData(**sample_data)
        
        assert market_data.symbol == sample_data["symbol"]
        assert market_data.timeframe == sample_data["timeframe"]
        assert market_data.data.equals(sample_data["data"])
        assert market_data.timestamp == sample_data["timestamp"]
        assert market_data.metadata == sample_data["metadata"]
    
    def test_validation_invalid_data_type(self):
        """Тест валидации неправильного типа данных."""
        data = {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "data": "invalid_data",  # Не DataFrame
            "timestamp": datetime.now()
        }
        
        with pytest.raises(ValueError, match="data must be a pandas DataFrame"):
            MarketData(**data)
    
    def test_validation_empty_dataframe(self):
        """Тест валидации пустого DataFrame."""
        data = {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "data": pd.DataFrame(),  # Пустой DataFrame
            "timestamp": datetime.now()
        }
        
        with pytest.raises(ValueError, match="data cannot be empty"):
            MarketData(**data)
    
    def test_validation_missing_columns(self):
        """Тест валидации отсутствующих колонок."""
        incomplete_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100]
            # Отсутствуют 'close' и 'volume'
        })
        
        data = {
            "symbol": "BTC/USD",
            "timeframe": "1h",
            "data": incomplete_data,
            "timestamp": datetime.now()
        }
        
        with pytest.raises(ValueError, match="Missing required columns"):
            MarketData(**data)
    
    def test_latest_price(self, market_data):
        """Тест получения последней цены."""
        latest_price = market_data.latest_price
        assert latest_price == 110.0
        assert isinstance(latest_price, float)
    
    def test_latest_volume(self, market_data):
        """Тест получения последнего объема."""
        latest_volume = market_data.latest_volume
        assert latest_volume == 1900.0
        assert isinstance(latest_volume, float)
    
    def test_price_change(self, market_data):
        """Тест расчета изменения цены."""
        price_change = market_data.price_change
        expected_change = (110.0 / 101.0) - 1
        assert abs(price_change - expected_change) < 1e-10
        assert isinstance(price_change, float)
    
    def test_price_change_single_row(self):
        """Тест расчета изменения цены для одной строки."""
        single_row_data = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101],
            'volume': [1000]
        })
        
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=single_row_data
        )
        
        price_change = market_data.price_change
        assert price_change == 0.0
    
    def test_volatility(self, market_data):
        """Тест расчета волатильности."""
        volatility = market_data.volatility
        assert volatility > 0
        assert isinstance(volatility, float)
    
    def test_volatility_single_row(self):
        """Тест расчета волатильности для одной строки."""
        single_row_data = pd.DataFrame({
            'open': [100],
            'high': [102],
            'low': [99],
            'close': [101],
            'volume': [1000]
        })
        
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=single_row_data
        )
        
        volatility = market_data.volatility
        assert volatility == 0.0
    
    def test_get_ohlcv(self, market_data):
        """Тест получения OHLCV данных."""
        ohlcv = market_data.get_ohlcv()
        
        assert isinstance(ohlcv, pd.DataFrame)
        assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]
        assert len(ohlcv) == len(market_data.data)
        assert ohlcv.equals(market_data.data[["open", "high", "low", "close", "volume"]])
    
    def test_get_price_series(self, market_data):
        """Тест получения временного ряда цен."""
        price_series = market_data.get_price_series()
        
        assert isinstance(price_series, pd.Series)
        assert price_series.name == "close"
        assert len(price_series) == len(market_data.data)
        assert price_series.equals(market_data.data["close"])
    
    def test_get_volume_series(self, market_data):
        """Тест получения временного ряда объемов."""
        volume_series = market_data.get_volume_series()
        
        assert isinstance(volume_series, pd.Series)
        assert volume_series.name == "volume"
        assert len(volume_series) == len(market_data.data)
        assert volume_series.equals(market_data.data["volume"])
    
    def test_to_dict(self, market_data):
        """Тест сериализации в словарь."""
        data_dict = market_data.to_dict()
        
        assert data_dict["symbol"] == market_data.symbol
        assert data_dict["timeframe"] == market_data.timeframe
        assert data_dict["timestamp"] == market_data.timestamp.isoformat()
        assert data_dict["data_length"] == len(market_data.data)
        assert data_dict["latest_price"] == market_data.latest_price
        assert data_dict["price_change"] == market_data.price_change
        assert data_dict["volatility"] == market_data.volatility
        assert data_dict["metadata"] == market_data.metadata


class TestModel:
    """Тесты для Model."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "name": "RandomForestClassifier",
            "type": "classification",
            "parameters": {
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            },
            "metadata": {
                "version": "1.0.0",
                "training_data_size": 10000,
                "accuracy": 0.85
            },
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
    
    @pytest.fixture
    def model(self, sample_data) -> Model:
        """Создает тестовую модель."""
        return Model(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания модели."""
        model = Model(**sample_data)
        
        assert model.name == sample_data["name"]
        assert model.type == sample_data["type"]
        assert model.parameters == sample_data["parameters"]
        assert model.metadata == sample_data["metadata"]
        assert model.created_at == sample_data["created_at"]
        assert model.updated_at == sample_data["updated_at"]
    
    def test_validation_invalid_type(self):
        """Тест валидации неправильного типа модели."""
        data = {
            "name": "TestModel",
            "type": "invalid_type"  # Невалидный тип
        }
        
        with pytest.raises(ValueError, match="Invalid model type"):
            Model(**data)
    
    def test_validation_valid_types(self):
        """Тест валидации правильных типов моделей."""
        valid_types = ["classification", "regression", "clustering"]
        
        for model_type in valid_types:
            model = Model(name="TestModel", type=model_type)
            assert model.type == model_type
    
    def test_to_dict(self, model):
        """Тест сериализации в словарь."""
        data_dict = model.to_dict()
        
        assert data_dict["name"] == model.name
        assert data_dict["type"] == model.type
        assert data_dict["parameters"] == model.parameters
        assert data_dict["metadata"] == model.metadata
        assert data_dict["created_at"] == model.created_at.isoformat()
        assert data_dict["updated_at"] == model.updated_at.isoformat()


class TestPrediction:
    """Тесты для Prediction."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "model_name": "RandomForestClassifier",
            "symbol": "BTC/USD",
            "prediction": 0.75,
            "confidence": 0.85,
            "timestamp": datetime.now(),
            "metadata": {
                "features_used": ["rsi", "macd", "volume"],
                "prediction_type": "price_direction"
            }
        }
    
    @pytest.fixture
    def prediction(self, sample_data) -> Prediction:
        """Создает тестовое предсказание."""
        return Prediction(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания предсказания."""
        pred = Prediction(**sample_data)
        
        assert pred.model_name == sample_data["model_name"]
        assert pred.symbol == sample_data["symbol"]
        assert pred.prediction == sample_data["prediction"]
        assert pred.confidence == sample_data["confidence"]
        assert pred.timestamp == sample_data["timestamp"]
        assert pred.metadata == sample_data["metadata"]
    
    def test_validation_confidence_below_zero(self):
        """Тест валидации confidence ниже 0."""
        data = {
            "model_name": "TestModel",
            "symbol": "BTC/USD",
            "prediction": 0.5,
            "confidence": -0.1  # Ниже 0
        }
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction(**data)
    
    def test_validation_confidence_above_one(self):
        """Тест валидации confidence выше 1."""
        data = {
            "model_name": "TestModel",
            "symbol": "BTC/USD",
            "prediction": 0.5,
            "confidence": 1.1  # Выше 1
        }
        
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction(**data)
    
    def test_validation_boundary_confidence(self):
        """Тест валидации граничных значений confidence."""
        # Граничные значения должны быть валидными
        boundary_values = [0.0, 0.5, 1.0]
        
        for confidence in boundary_values:
            pred = Prediction(
                model_name="TestModel",
                symbol="BTC/USD",
                prediction=0.5,
                confidence=confidence
            )
            assert pred.confidence == confidence
    
    def test_to_dict(self, prediction):
        """Тест сериализации в словарь."""
        data_dict = prediction.to_dict()
        
        assert data_dict["model_name"] == prediction.model_name
        assert data_dict["symbol"] == prediction.symbol
        assert data_dict["prediction"] == prediction.prediction
        assert data_dict["confidence"] == prediction.confidence
        assert data_dict["timestamp"] == prediction.timestamp.isoformat()
        assert data_dict["metadata"] == prediction.metadata


class TestOrder:
    """Тесты для Order."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": "order_123",
            "pair": "BTC/USD",
            "type": "limit",
            "side": "buy",
            "price": 50000.0,
            "size": 1.5,
            "status": "open",
            "timestamp": datetime.now(),
            "metadata": {
                "strategy": "trend_following",
                "risk_level": "medium"
            }
        }
    
    @pytest.fixture
    def order(self, sample_data) -> Order:
        """Создает тестовый ордер."""
        return Order(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания ордера."""
        order = Order(**sample_data)
        
        assert order.id == sample_data["id"]
        assert order.pair == sample_data["pair"]
        assert order.type == sample_data["type"]
        assert order.side == sample_data["side"]
        assert order.price == sample_data["price"]
        assert order.size == sample_data["size"]
        assert order.status == sample_data["status"]
        assert order.timestamp == sample_data["timestamp"]
        assert order.metadata == sample_data["metadata"]
    
    def test_validation_invalid_type(self):
        """Тест валидации неправильного типа ордера."""
        data = {
            "id": "order_123",
            "pair": "BTC/USD",
            "type": "invalid_type",  # Невалидный тип
            "side": "buy",
            "price": 50000.0,
            "size": 1.5,
            "status": "open"
        }
        
        with pytest.raises(ValueError, match="Invalid order type"):
            Order(**data)
    
    def test_validation_invalid_side(self):
        """Тест валидации неправильной стороны ордера."""
        data = {
            "id": "order_123",
            "pair": "BTC/USD",
            "type": "limit",
            "side": "invalid_side",  # Невалидная сторона
            "price": 50000.0,
            "size": 1.5,
            "status": "open"
        }
        
        with pytest.raises(ValueError, match="Invalid order side"):
            Order(**data)
    
    def test_validation_invalid_status(self):
        """Тест валидации неправильного статуса ордера."""
        data = {
            "id": "order_123",
            "pair": "BTC/USD",
            "type": "limit",
            "side": "buy",
            "price": 50000.0,
            "size": 1.5,
            "status": "invalid_status"  # Невалидный статус
        }
        
        with pytest.raises(ValueError, match="Invalid order status"):
            Order(**data)
    
    def test_validation_non_positive_price(self):
        """Тест валидации неположительной цены."""
        data = {
            "id": "order_123",
            "pair": "BTC/USD",
            "type": "limit",
            "side": "buy",
            "price": 0.0,  # Нулевая цена
            "size": 1.5,
            "status": "open"
        }
        
        with pytest.raises(ValueError, match="Price must be positive"):
            Order(**data)
    
    def test_validation_non_positive_size(self):
        """Тест валидации неположительного размера."""
        data = {
            "id": "order_123",
            "pair": "BTC/USD",
            "type": "limit",
            "side": "buy",
            "price": 50000.0,
            "size": -1.0,  # Отрицательный размер
            "status": "open"
        }
        
        with pytest.raises(ValueError, match="Size must be positive"):
            Order(**data)
    
    def test_validation_valid_values(self):
        """Тест валидации правильных значений."""
        valid_types = ["market", "limit", "stop", "stop_limit"]
        valid_sides = ["buy", "sell"]
        valid_statuses = ["open", "closed", "canceled", "pending"]
        
        for order_type in valid_types:
            for side in valid_sides:
                for status in valid_statuses:
                    order = Order(
                        id="order_123",
                        pair="BTC/USD",
                        type=order_type,
                        side=side,
                        price=50000.0,
                        size=1.5,
                        status=status
                    )
                    assert order.type == order_type
                    assert order.side == side
                    assert order.status == status
    
    def test_to_dict(self, order):
        """Тест сериализации в словарь."""
        data_dict = order.to_dict()
        
        assert data_dict["id"] == order.id
        assert data_dict["pair"] == order.pair
        assert data_dict["type"] == order.type
        assert data_dict["side"] == order.side
        assert data_dict["price"] == order.price
        assert data_dict["size"] == order.size
        assert data_dict["status"] == order.status
        assert data_dict["timestamp"] == order.timestamp.isoformat()
        assert data_dict["metadata"] == order.metadata


class TestPosition:
    """Тесты для Position."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "pair": "BTC/USD",
            "side": "long",
            "size": 2.0,
            "entry_price": 50000.0,
            "current_price": 52000.0,
            "pnl": 4000.0,
            "leverage": 1.0,
            "entry_time": datetime.now(),
            "metadata": {
                "strategy": "trend_following",
                "risk_level": "medium"
            }
        }
    
    @pytest.fixture
    def position(self, sample_data) -> Position:
        """Создает тестовую позицию."""
        return Position(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания позиции."""
        position = Position(**sample_data)
        
        assert position.pair == sample_data["pair"]
        assert position.side == sample_data["side"]
        assert position.size == sample_data["size"]
        assert position.entry_price == sample_data["entry_price"]
        assert position.current_price == sample_data["current_price"]
        assert position.pnl == sample_data["pnl"]
        assert position.leverage == sample_data["leverage"]
        assert position.entry_time == sample_data["entry_time"]
        assert position.metadata == sample_data["metadata"]
    
    def test_validation_invalid_side(self):
        """Тест валидации неправильной стороны позиции."""
        data = {
            "pair": "BTC/USD",
            "side": "invalid_side",  # Невалидная сторона
            "size": 2.0,
            "entry_price": 50000.0,
            "current_price": 52000.0,
            "pnl": 4000.0,
            "leverage": 1.0
        }
        
        with pytest.raises(ValueError, match="Invalid position side"):
            Position(**data)
    
    def test_validation_non_positive_size(self):
        """Тест валидации неположительного размера."""
        data = {
            "pair": "BTC/USD",
            "side": "long",
            "size": 0.0,  # Нулевой размер
            "entry_price": 50000.0,
            "current_price": 52000.0,
            "pnl": 4000.0,
            "leverage": 1.0
        }
        
        with pytest.raises(ValueError, match="Size must be positive"):
            Position(**data)
    
    def test_validation_non_positive_entry_price(self):
        """Тест валидации неположительной цены входа."""
        data = {
            "pair": "BTC/USD",
            "side": "long",
            "size": 2.0,
            "entry_price": -100.0,  # Отрицательная цена
            "current_price": 52000.0,
            "pnl": 4000.0,
            "leverage": 1.0
        }
        
        with pytest.raises(ValueError, match="Entry price must be positive"):
            Position(**data)
    
    def test_validation_non_positive_current_price(self):
        """Тест валидации неположительной текущей цены."""
        data = {
            "pair": "BTC/USD",
            "side": "long",
            "size": 2.0,
            "entry_price": 50000.0,
            "current_price": 0.0,  # Нулевая цена
            "pnl": 4000.0,
            "leverage": 1.0
        }
        
        with pytest.raises(ValueError, match="Current price must be positive"):
            Position(**data)
    
    def test_validation_non_positive_leverage(self):
        """Тест валидации неположительного плеча."""
        data = {
            "pair": "BTC/USD",
            "side": "long",
            "size": 2.0,
            "entry_price": 50000.0,
            "current_price": 52000.0,
            "pnl": 4000.0,
            "leverage": -1.0  # Отрицательное плечо
        }
        
        with pytest.raises(ValueError, match="Leverage must be positive"):
            Position(**data)
    
    def test_validation_valid_sides(self):
        """Тест валидации правильных сторон позиции."""
        valid_sides = ["long", "short"]
        
        for side in valid_sides:
            position = Position(
                pair="BTC/USD",
                side=side,
                size=2.0,
                entry_price=50000.0,
                current_price=52000.0,
                pnl=4000.0,
                leverage=1.0
            )
            assert position.side == side
    
    def test_to_dict(self, position):
        """Тест сериализации в словарь."""
        data_dict = position.to_dict()
        
        assert data_dict["pair"] == position.pair
        assert data_dict["side"] == position.side
        assert data_dict["size"] == position.size
        assert data_dict["entry_price"] == position.entry_price
        assert data_dict["current_price"] == position.current_price
        assert data_dict["pnl"] == position.pnl
        assert data_dict["leverage"] == position.leverage
        assert data_dict["entry_time"] == position.entry_time.isoformat()
        assert data_dict["metadata"] == position.metadata


class TestSystemState:
    """Тесты для SystemState."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "is_running": True,
            "is_healthy": True,
            "last_update": datetime.now(),
            "metadata": {
                "version": "1.0.0",
                "uptime": "24h",
                "active_strategies": 5
            }
        }
    
    @pytest.fixture
    def system_state(self, sample_data) -> SystemState:
        """Создает тестовое состояние системы."""
        return SystemState(**sample_data)
    
    def test_creation(self, sample_data):
        """Тест создания состояния системы."""
        state = SystemState(**sample_data)
        
        assert state.is_running == sample_data["is_running"]
        assert state.is_healthy == sample_data["is_healthy"]
        assert state.last_update == sample_data["last_update"]
        assert state.metadata == sample_data["metadata"]
    
    def test_default_creation(self):
        """Тест создания состояния системы с дефолтными значениями."""
        state = SystemState()
        
        assert state.is_running is False
        assert state.is_healthy is True
        assert isinstance(state.last_update, datetime)
        assert state.metadata == {}
    
    def test_system_states(self):
        """Тест различных состояний системы."""
        # Работающая и здоровая система
        running_healthy = SystemState(is_running=True, is_healthy=True)
        assert running_healthy.is_running is True
        assert running_healthy.is_healthy is True
        
        # Работающая, но нездоровая система
        running_unhealthy = SystemState(is_running=True, is_healthy=False)
        assert running_unhealthy.is_running is True
        assert running_unhealthy.is_healthy is False
        
        # Остановленная система
        stopped = SystemState(is_running=False, is_healthy=True)
        assert stopped.is_running is False
        assert stopped.is_healthy is True
        
        # Остановленная и нездоровая система
        stopped_unhealthy = SystemState(is_running=False, is_healthy=False)
        assert stopped_unhealthy.is_running is False
        assert stopped_unhealthy.is_healthy is False
    
    def test_to_dict(self, system_state):
        """Тест сериализации в словарь."""
        data_dict = system_state.to_dict()
        
        assert data_dict["is_running"] == system_state.is_running
        assert data_dict["is_healthy"] == system_state.is_healthy
        assert data_dict["last_update"] == system_state.last_update.isoformat()
        assert data_dict["metadata"] == system_state.metadata
    
    def test_system_state_with_extensive_metadata(self):
        """Тест состояния системы с расширенными метаданными."""
        metadata = {
            "system_info": {
                "version": "2.1.0",
                "build_date": "2023-12-01",
                "python_version": "3.9.0"
            },
            "performance_metrics": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1
            },
            "trading_metrics": {
                "active_strategies": 8,
                "total_positions": 15,
                "daily_pnl": 2500.0,
                "monthly_pnl": 45000.0
            },
            "risk_metrics": {
                "current_drawdown": 0.05,
                "max_drawdown": 0.12,
                "sharpe_ratio": 1.8,
                "var_95": 0.03
            },
            "network_status": {
                "exchange_connections": 3,
                "data_feed_latency": 0.15,
                "order_execution_latency": 0.25
            }
        }
        
        state = SystemState(
            is_running=True,
            is_healthy=True,
            metadata=metadata
        )
        
        assert state.metadata == metadata
        assert state.metadata["system_info"]["version"] == "2.1.0"
        assert state.metadata["performance_metrics"]["cpu_usage"] == 45.2
        assert state.metadata["trading_metrics"]["active_strategies"] == 8
        assert state.metadata["risk_metrics"]["sharpe_ratio"] == 1.8 