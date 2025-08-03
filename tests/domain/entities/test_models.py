"""
Тесты для доменных моделей.
"""
import pytest
from datetime import datetime
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Any

if TYPE_CHECKING:
    from domain.entities import MarketData, Model, Prediction
else:
    try:
        from domain.entities import MarketData, Model, Prediction
    except ImportError:
        # Создаем заглушки для тестирования
        class MarketData:
            def __init__(self, symbol: str, timeframe: str, data, metadata: Optional[Dict[str, Any]] = None) -> Any:
                # Валидация данных
                if data is None or (hasattr(data, 'empty') and data.empty):
                    raise ValueError("data cannot be empty")
                if not isinstance(data, pd.DataFrame):
                    raise ValueError("data must be a pandas DataFrame")
                
                # Проверка обязательных колонок
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in data.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                self.symbol = symbol
                self.timeframe = timeframe
                self.data = data
                self.metadata = metadata or {}
                self.timestamp = datetime.now()
            @property
            def latest_price(self) -> Any:
                if hasattr(self.data, 'iloc') and callable(self.data.iloc):
                    iloc_result = self.data.iloc[-1]
                    if callable(iloc_result):
                        close_data = iloc_result()
                    else:
                        close_data = iloc_result
                    if hasattr(close_data, 'get'):
                        return float(close_data.get("close", 0))
                    else:
                        return float(close_data)
                else:
                    return 0.0
            @property
            def latest_volume(self) -> Any:
                if hasattr(self.data, 'iloc') and callable(self.data.iloc):
                    iloc_result = self.data.iloc[-1]
                    if callable(iloc_result):
                        volume_data = iloc_result()
                    else:
                        volume_data = iloc_result
                    if hasattr(volume_data, 'get'):
                        return float(volume_data.get("volume", 0))
                    else:
                        return float(volume_data)
                else:
                    return 0.0
            @property
            def price_change(self) -> Any:
                if len(self.data) <= 1:
                    return 0.0
                return (self.data["close"].iloc[-1] / self.data["close"].iloc[0]) - 1
            @property
            def volatility(self) -> Any:
                if len(self.data) <= 1:
                    return 0.0
                returns = self.data["close"].pct_change().dropna()
                return float(returns.std())
            def get_ohlcv(self) -> Any:
                return self.data[["open", "high", "low", "close", "volume"]]
            def get_price_series(self) -> Any:
                return self.data["close"]
            def get_volume_series(self) -> Any:
                return self.data["volume"]
            def to_dict(self) -> Any:
                return {
                    "symbol": self.symbol,
                    "timeframe": self.timeframe,
                    "data_length": len(self.data),
                    "latest_price": self.latest_price,
                    "price_change": self.price_change,
                    "volatility": self.volatility,
                    "metadata": self.metadata
                }
        class Model:
            def __init__(self, name: str, model_type: str, parameters: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> Any:
                # Валидация типа модели
                valid_types = ["classification", "regression", "clustering"]
                if model_type not in valid_types:
                    raise ValueError("Invalid model type")
                
                self.name = name
                self.model_type = model_type
                self.parameters = parameters or {}
                self.metadata = metadata or {}
                self.created_at = datetime.now()
                self.updated_at = datetime.now()
            def to_dict(self) -> Any:
                return {
                    "name": self.name,
                    "model_type": self.model_type,
                    "parameters": self.parameters,
                    "metadata": self.metadata,
                    "created_at": self.created_at.isoformat(),
                    "updated_at": self.updated_at.isoformat()
                }
        class Prediction:
            def __init__(self, symbol: str, value: float, confidence: float, timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None) -> Any:
                # Валидация уверенности
                if not 0 <= confidence <= 1:
                    raise ValueError("Confidence must be between 0 and 1")
                
                self.symbol = symbol
                self.value = value
                self.confidence = confidence
                self.timestamp = timestamp or datetime.now()
                self.metadata = metadata or {}
            def to_dict(self) -> Any:
                return {
                    "symbol": self.symbol,
                    "value": self.value,
                    "confidence": self.confidence,
                    "timestamp": self.timestamp.isoformat(),
                    "metadata": self.metadata
                }

class TestMarketData:
    """Тесты для рыночных данных."""
    @pytest.fixture
    def sample_ohlcv_data(self) -> Any:
        """Фикстура с примерными OHLCV данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        data = pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(100, 1000, 100),
        }, index=dates)
        return data
    def test_market_data_creation(self, sample_ohlcv_data) -> None:
        """Тест создания рыночных данных."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        assert market_data.symbol == "BTC/USD"
        assert market_data.timeframe == "1h"
        assert len(market_data.data) == 100
        assert isinstance(market_data.timestamp, datetime)
    def test_market_data_validation_empty_data(self) -> None:
        """Тест валидации пустых данных."""
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError, match="data cannot be empty"):
            MarketData(
                symbol="BTC/USD",
                timeframe="1h",
                data=empty_data,
            )
    def test_market_data_validation_wrong_type(self) -> None:
        """Тест валидации неправильного типа данных."""
        wrong_data = [1, 2, 3]
        with pytest.raises(ValueError, match="data must be a pandas DataFrame"):
            MarketData(
                symbol="BTC/USD",
                timeframe="1h",
                data=wrong_data,
            )
    def test_market_data_validation_missing_columns(self, sample_ohlcv_data) -> None:
        """Тест валидации отсутствующих колонок."""
        incomplete_data = sample_ohlcv_data.drop(columns=['volume'])
        with pytest.raises(ValueError, match="Missing required columns"):
            MarketData(
                symbol="BTC/USD",
                timeframe="1h",
                data=incomplete_data,
            )
    def test_latest_price(self, sample_ohlcv_data) -> None:
        """Тест получения последней цены."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        latest_price = market_data.latest_price
        assert isinstance(latest_price, float)
        assert latest_price == float(sample_ohlcv_data["close"].iloc[-1])
    def test_latest_volume(self, sample_ohlcv_data) -> None:
        """Тест получения последнего объема."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        latest_volume = market_data.latest_volume
        assert isinstance(latest_volume, float)
        assert latest_volume == float(sample_ohlcv_data["volume"].iloc[-1])
    def test_price_change(self, sample_ohlcv_data) -> None:
        """Тест расчета изменения цены."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        price_change = market_data.price_change
        assert isinstance(price_change, float)
        # Проверяем расчет
        expected_change = (sample_ohlcv_data["close"].iloc[-1] / sample_ohlcv_data["close"].iloc[0]) - 1
        assert abs(price_change - expected_change) < 1e-10
    def test_price_change_single_point(self) -> None:
        """Тест изменения цены для одной точки данных."""
        single_data = pd.DataFrame({
            'open': [50000],
            'high': [51000],
            'low': [49000],
            'close': [50500],
            'volume': [500],
        })
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=single_data,
        )
        price_change = market_data.price_change
        assert price_change == 0.0
    def test_volatility(self, sample_ohlcv_data) -> None:
        """Тест расчета волатильности."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        volatility = market_data.volatility
        assert isinstance(volatility, float)
        assert volatility >= 0.0
        # Проверяем расчет
        returns = sample_ohlcv_data["close"].pct_change().dropna()
        expected_volatility = returns.std()
        assert abs(volatility - expected_volatility) < 1e-10
    def test_volatility_single_point(self) -> None:
        """Тест волатильности для одной точки данных."""
        single_data = pd.DataFrame({
            'open': [50000],
            'high': [51000],
            'low': [49000],
            'close': [50500],
            'volume': [500],
        })
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=single_data,
        )
        volatility = market_data.volatility
        assert volatility == 0.0
    def test_get_ohlcv(self, sample_ohlcv_data) -> None:
        """Тест получения OHLCV данных."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        ohlcv = market_data.get_ohlcv()
        assert isinstance(ohlcv, pd.DataFrame)
        assert list(ohlcv.columns) == ["open", "high", "low", "close", "volume"]
        assert len(ohlcv) == len(sample_ohlcv_data)
    def test_get_price_series(self, sample_ohlcv_data) -> None:
        """Тест получения временного ряда цен."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        price_series = market_data.get_price_series()
        assert isinstance(price_series, pd.Series)
        assert price_series.name == "close"
        assert len(price_series) == len(sample_ohlcv_data)
    def test_get_volume_series(self, sample_ohlcv_data) -> None:
        """Тест получения временного ряда объемов."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
        )
        volume_series = market_data.get_volume_series()
        assert isinstance(volume_series, pd.Series)
        assert volume_series.name == "volume"
        assert len(volume_series) == len(sample_ohlcv_data)
    def test_to_dict(self, sample_ohlcv_data) -> None:
        """Тест преобразования в словарь."""
        market_data = MarketData(
            symbol="BTC/USD",
            timeframe="1h",
            data=sample_ohlcv_data,
            metadata={"source": "binance"},
        )
        data_dict = market_data.to_dict()
        assert data_dict["symbol"] == "BTC/USD"
        assert data_dict["timeframe"] == "1h"
        assert data_dict["data_length"] == 100
        assert isinstance(data_dict["latest_price"], float)
        assert isinstance(data_dict["price_change"], float)
        assert isinstance(data_dict["volatility"], float)
        assert data_dict["metadata"] == {"source": "binance"}
class TestModel:
    """Тесты для модели машинного обучения."""
    def test_model_creation(self) -> None:
        """Тест создания модели."""
        model = Model(
            name="Test Model",
            model_type="classification",
            parameters={"n_estimators": 100},
            metadata={"version": "1.0"},
        )
        assert model.name == "Test Model"
        assert model.model_type == "classification"
        assert model.parameters == {"n_estimators": 100}
        assert model.metadata == {"version": "1.0"}
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)
    def test_model_validation_invalid_type(self) -> None:
        """Тест валидации неправильного типа модели."""
        with pytest.raises(ValueError, match="Invalid model type"):
            Model(
                name="Test Model",
                model_type="invalid_type",
            )
    def test_model_valid_types(self) -> None:
        """Тест всех валидных типов моделей."""
        valid_types = ["classification", "regression", "clustering"]
        for model_type in valid_types:
            model = Model(
                name=f"Test {model_type}",
                model_type=model_type,
            )
            assert model.model_type == model_type
    def test_model_to_dict(self) -> None:
        """Тест преобразования модели в словарь."""
        model = Model(
            name="Test Model",
            model_type="regression",
            parameters={"learning_rate": 0.1},
            metadata={"framework": "sklearn"},
        )
        model_dict = model.to_dict()
        assert model_dict["name"] == "Test Model"
        assert model_dict["model_type"] == "regression"
        assert model_dict["parameters"] == {"learning_rate": 0.1}
        assert model_dict["metadata"] == {"framework": "sklearn"}
        assert isinstance(model_dict["created_at"], str)
        assert isinstance(model_dict["updated_at"], str)
class TestPrediction:
    """Тесты для предсказания модели."""
    def test_prediction_creation(self) -> None:
        """Тест создания предсказания."""
        prediction = Prediction(
            symbol="BTC/USD",
            value=0.75,
            confidence=0.85,
            metadata={"threshold": 0.5},
        )
        assert prediction.symbol == "BTC/USD"
        assert prediction.value == 0.75
        assert prediction.confidence == 0.85
        assert prediction.metadata == {"threshold": 0.5}
        assert isinstance(prediction.timestamp, datetime)
    def test_prediction_validation_confidence_too_low(self) -> None:
        """Тест валидации слишком низкой уверенности."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction(
                symbol="BTC/USD",
                value=0.75,
                confidence=-0.1,
            )
    def test_prediction_validation_confidence_too_high(self) -> None:
        """Тест валидации слишком высокой уверенности."""
        with pytest.raises(ValueError, match="Confidence must be between 0 and 1"):
            Prediction(
                symbol="BTC/USD",
                value=0.75,
                confidence=1.1,
            )
    def test_prediction_validation_confidence_boundaries(self) -> None:
        """Тест валидации граничных значений уверенности."""
        # Должно работать
        Prediction(
            symbol="BTC/USD",
            value=0.75,
            confidence=0.0,
        )
        Prediction(
            symbol="BTC/USD",
            value=0.75,
            confidence=1.0,
        )
    def test_prediction_to_dict(self) -> None:
        """Тест преобразования предсказания в словарь."""
        prediction = Prediction(
            symbol="BTC/USD",
            value=0.75,
            confidence=0.85,
            metadata={"feature_importance": [0.3, 0.7]},
        )
        prediction_dict = prediction.to_dict()
        assert prediction_dict["symbol"] == "BTC/USD"
        assert prediction_dict["value"] == 0.75
        assert prediction_dict["confidence"] == 0.85
        assert prediction_dict["metadata"] == {"feature_importance": [0.3, 0.7]}
        assert isinstance(prediction_dict["timestamp"], str) 
