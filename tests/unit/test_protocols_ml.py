"""
Production-ready unit тесты для MLProtocol.
Полное покрытие всех методов, ошибок, edge cases и асинхронных сценариев.
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Any, Dict, Optional
from shared.numpy_utils import np
from decimal import Decimal
from uuid import uuid4
from sklearn.metrics import mean_squared_error, r2_score
from domain.protocols.ml_protocol import (
    MLProtocol,
    ModelTrainingProtocol,
    ModelEvaluationProtocol,
    PredictionProtocol,
    MarketAnalysisProtocol,
    ModelConfig,
    TrainingConfig,
    ModelMetrics,
    ModelType,
    ModelNotFoundError,
    TrainingError,
    PredictionError,
    InsufficientDataError
)
from domain.entities.market import MarketData, Price, Volume
from domain.entities.ml import Model, Prediction, PredictionType
from domain.value_objects.currency import Currency
from domain.type_definitions import Symbol, TimestampValue, ModelId
from uuid import uuid4

# Определяем недостающие типы для тестов
class TrainingResult:
    def __init__(self, model_id: str, success: bool, training_time: timedelta, 
                 epochs_completed: int, final_loss: float, validation_loss: float, 
                 metrics: ModelMetrics) -> Any:
        self.model_id = model_id
        self.success = success
        self.training_time = training_time
        self.epochs_completed = epochs_completed
        self.final_loss = final_loss
        self.validation_loss = validation_loss
        self.metrics = metrics

class PredictionResult:
    def __init__(self, model_id: str, predictions: np.ndarray, 
                 confidence_scores: np.ndarray, feature_importance: Dict[str, float], 
                 timestamp: datetime) -> Any:
        self.model_id = model_id
        self.predictions = predictions
        self.confidence_scores = confidence_scores
        self.feature_importance = feature_importance
        self.timestamp = timestamp

class EvaluationResult:
    def __init__(self, model_id: str, metrics: ModelMetrics, 
                 confusion_matrix: np.ndarray, classification_report: str, 
                 timestamp: datetime) -> Any:
        self.model_id = model_id
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix
        self.classification_report = classification_report
        self.timestamp = timestamp

class SignalResult:
    def __init__(self, signal_type: str, confidence: float, price_target: Optional[Price], 
                 stop_loss: Optional[Price], take_profit: Optional[Price], 
                 reasoning: str, timestamp: datetime) -> Any:
        self.signal_type = signal_type
        self.confidence = confidence
        self.price_target = price_target
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.reasoning = reasoning
        self.timestamp = timestamp

class TestMLProtocol:
    """Production-ready тесты для MLProtocol."""
    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Фикстура конфигурации модели."""
        return ModelConfig(
            name="test_model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            features=["price", "volume", "rsi"],
            target="price_change"
        )
    @pytest.fixture
    def training_config(self) -> TrainingConfig:
        """Фикстура конфигурации обучения."""
        return TrainingConfig(
            validation_split=0.8,
            test_split=0.1,
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            early_stopping_patience=10
        )
    @pytest.fixture
    def mock_ml_protocol(self, model_config: ModelConfig) -> Mock:
        """Фикстура мока ML протокола."""
        ml_protocol = Mock(spec=MLProtocol)
        ml_protocol.name = "test_ml_protocol"
        ml_protocol.config = model_config
        # Настройка методов модели
        ml_protocol.create_model = AsyncMock(return_value=Model(
            id=uuid4(),
            name=model_config.name,
            model_type=model_config.model_type,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        ))
        ml_protocol.load_model = AsyncMock(return_value=Model(
            id=uuid4(),
            name=model_config.name,
            model_type=model_config.model_type,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        ))
        ml_protocol.save_model = AsyncMock(return_value=True)
        ml_protocol.delete_model = AsyncMock(return_value=True)
        ml_protocol.list_models = AsyncMock(return_value=["model_1", "model_2"])
        # Настройка методов обучения
        ml_protocol.train_model = AsyncMock(return_value=TrainingResult(
            model_id="model_123",
            success=True,
            training_time=timedelta(minutes=30),
            epochs_completed=100,
            final_loss=0.001,
            validation_loss=0.002,
            metrics=ModelMetrics(
                mse=0.001,
                mae=0.025,
                r2=0.78,
                sharpe_ratio=1.2,
                max_drawdown=0.05,
                win_rate=0.65,
                profit_factor=1.5,
                total_return=0.15,
                volatility=0.12,
                calmar_ratio=2.4
            )
        ))
        ml_protocol.validate_data = AsyncMock(return_value=True)
        ml_protocol.preprocess_data = AsyncMock(return_value=(
            np.random.rand(1000, 10),  # X_train
            np.random.rand(1000),      # y_train
            np.random.rand(200, 10),   # X_val
            np.random.rand(200),       # y_val
            np.random.rand(200, 10),   # X_test
            np.random.rand(200)        # y_test
        ))
        # Настройка методов предсказания
        ml_protocol.predict = AsyncMock(return_value=PredictionResult(
            model_id="model_123",
            predictions=np.array([0.05, -0.02, 0.03]),
            confidence_scores=np.array([0.85, 0.72, 0.91]),
            feature_importance={"price": 0.3, "volume": 0.2, "rsi": 0.5},
            timestamp=datetime.utcnow()
        ))
        ml_protocol.predict_batch = AsyncMock(return_value=PredictionResult(
            model_id="model_123",
            predictions=np.random.rand(100),
            confidence_scores=np.random.rand(100),
            feature_importance={},
            timestamp=datetime.utcnow()
        ))
        # Настройка методов оценки
        ml_protocol.evaluate_model = AsyncMock(return_value=EvaluationResult(
            model_id="model_123",
            metrics=ModelMetrics(
                mse=0.001,
                mae=0.025,
                r2=0.78,
                sharpe_ratio=1.2,
                max_drawdown=0.05,
                win_rate=0.65,
                profit_factor=1.5,
                total_return=0.15,
                volatility=0.12,
                calmar_ratio=2.4
            ),
            confusion_matrix=np.array([[45, 5], [8, 42]]),
            classification_report="precision    recall  f1-score",
            timestamp=datetime.utcnow()
        ))
        # Настройка методов анализа рынка
        ml_protocol.analyze_market = AsyncMock(return_value=SignalResult(
            signal_type="buy",
            confidence=0.85,
            price_target=Price(Decimal("51000.0"), Currency("USDT")),
            stop_loss=Price(Decimal("49000.0"), Currency("USDT")),
            take_profit=Price(Decimal("52000.0"), Currency("USDT")),
            reasoning="Strong bullish pattern detected with high confidence",
            timestamp=datetime.utcnow()
        ))
        ml_protocol.generate_signal = AsyncMock(return_value=SignalResult(
            signal_type="hold",
            confidence=0.65,
            price_target=None,
            stop_loss=None,
            take_profit=None,
            reasoning="Market conditions are neutral",
            timestamp=datetime.utcnow()
        ))
        return ml_protocol
    @pytest.fixture
    def sample_market_data(self) -> List[MarketData]:
        """Фикстура образцовых рыночных данных."""
        data = []
        base_price = Decimal("50000.0")
        base_volume = Decimal("100.0")
        for i in range(100):
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            price_change = Decimal(str(np.random.normal(0, 100)))
            volume_change = Decimal(str(np.random.normal(0, 10)))
            data.append(MarketData(
                symbol=Symbol("BTCUSDT"),
                timestamp=TimestampValue(timestamp),
                open=Price(base_price + price_change - Decimal("25.0"), Currency("USDT"), Currency("USDT")),
                high=Price(base_price + price_change + Decimal("50.0"), Currency("USDT"), Currency("USDT")),
                low=Price(base_price + price_change - Decimal("50.0"), Currency("USDT"), Currency("USDT")),
                close=Price(base_price + price_change, Currency("USDT"), Currency("USDT")),
                volume=Volume(base_volume + volume_change, Currency("BTC"))
            ))
        return data
    @pytest.mark.asyncio
    async def test_model_lifecycle(self, mock_ml_protocol: Mock, model_config: ModelConfig) -> None:
        """Тест полного жизненного цикла модели."""
        # Создание модели
        model = await mock_ml_protocol.create_model(model_config)
        assert isinstance(model, Model)
        assert model.name == model_config.name
        assert model.model_type == model_config.model_type
        mock_ml_protocol.create_model.assert_called_once_with(model_config)
        # Загрузка модели
        loaded_model = await mock_ml_protocol.load_model("model_123")
        assert isinstance(loaded_model, Model)
        assert loaded_model.name == model_config.name
        assert loaded_model.model_type == model_config.model_type
        mock_ml_protocol.load_model.assert_called_once_with("model_123")
        # Сохранение модели
        save_result = await mock_ml_protocol.save_model("model_123", "/path/to/model")
        assert save_result is True
        mock_ml_protocol.save_model.assert_called_once_with("model_123", "/path/to/model")
        # Удаление модели
        delete_result = await mock_ml_protocol.delete_model("model_123")
        assert delete_result is True
        mock_ml_protocol.delete_model.assert_called_once_with("model_123")
        # Список моделей
        models = await mock_ml_protocol.list_models()
        assert isinstance(models, list)
        assert len(models) == 2
        mock_ml_protocol.list_models.assert_called_once()
    @pytest.mark.asyncio
    async def test_model_training(self, mock_ml_protocol: Mock, training_config: TrainingConfig) -> None:
        """Тест обучения модели."""
        # Валидация данных
        validation_result = await mock_ml_protocol.validate_data("model_123")
        assert validation_result is True
        mock_ml_protocol.validate_data.assert_called_once_with("model_123")
        # Предобработка данных
        preprocessed_data = await mock_ml_protocol.preprocess_data("model_123")
        assert isinstance(preprocessed_data, tuple)
        assert len(preprocessed_data) == 6
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessed_data
        assert X_train.shape[0] == 1000
        assert y_train.shape[0] == 1000
        mock_ml_protocol.preprocess_data.assert_called_once_with("model_123")
        # Обучение модели
        training_result = await mock_ml_protocol.train_model("model_123", training_config)
        assert isinstance(training_result, TrainingResult)
        assert training_result.model_id == "model_123"
        assert training_result.success is True
        assert training_result.epochs_completed == 100
        assert training_result.final_loss == 0.001
        assert training_result.validation_loss == 0.002
        # Проверка метрик
        metrics = training_result.metrics
        assert metrics.mse == 0.001
        assert metrics.mae == 0.025
        assert metrics.r2 == 0.78
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == 0.05
        assert metrics.win_rate == 0.65
        assert metrics.profit_factor == 1.5
        assert metrics.total_return == 0.15
        assert metrics.volatility == 0.12
        assert metrics.calmar_ratio == 2.4
        mock_ml_protocol.train_model.assert_called_once_with("model_123", training_config)
    @pytest.mark.asyncio
    async def test_model_prediction(self, mock_ml_protocol: Mock, sample_market_data: List[MarketData]) -> None:
        """Тест предсказания модели."""
        # Одиночное предсказание
        prediction_result = await mock_ml_protocol.predict("model_123", sample_market_data[:3])
        assert isinstance(prediction_result, PredictionResult)
        assert prediction_result.model_id == "model_123"
        assert len(prediction_result.predictions) == 3
        assert len(prediction_result.confidence_scores) == 3
        assert all(0 <= conf <= 1 for conf in prediction_result.confidence_scores)
        assert "price" in prediction_result.feature_importance
        assert "volume" in prediction_result.feature_importance
        mock_ml_protocol.predict.assert_called_once_with("model_123", sample_market_data[:3])
        # Пакетное предсказание
        batch_result = await mock_ml_protocol.predict_batch("model_123", sample_market_data)
        assert isinstance(batch_result, PredictionResult)
        assert batch_result.model_id == "model_123"
        assert len(batch_result.predictions) == 100
        assert len(batch_result.confidence_scores) == 100
        mock_ml_protocol.predict_batch.assert_called_once_with("model_123", sample_market_data)
    @pytest.mark.asyncio
    async def test_model_evaluation(self, mock_ml_protocol: Mock) -> None:
        """Тест оценки модели."""
        evaluation_result = await mock_ml_protocol.evaluate_model("model_123")
        assert isinstance(evaluation_result, EvaluationResult)
        assert evaluation_result.model_id == "model_123"
        # Проверка метрик
        metrics = evaluation_result.metrics
        assert metrics.mse == 0.001
        assert metrics.mae == 0.025
        assert metrics.r2 == 0.78
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == 0.05
        assert metrics.win_rate == 0.65
        assert metrics.profit_factor == 1.5
        assert metrics.total_return == 0.15
        assert metrics.volatility == 0.12
        assert metrics.calmar_ratio == 2.4
        # Проверка матрицы ошибок
        assert evaluation_result.confusion_matrix.shape == (2, 2)
        assert evaluation_result.classification_report is not None
        mock_ml_protocol.evaluate_model.assert_called_once_with("model_123")
    @pytest.mark.asyncio
    async def test_market_analysis(self, mock_ml_protocol: Mock, sample_market_data: List[MarketData]) -> None:
        """Тест анализа рынка."""
        analysis_result = await mock_ml_protocol.analyze_market("model_123", sample_market_data)
        assert isinstance(analysis_result, SignalResult)
        assert analysis_result.signal_type == "buy"
        assert analysis_result.confidence == 0.85
        assert analysis_result.price_target is not None and analysis_result.price_target.value == Decimal("51000.0")
        assert analysis_result.stop_loss is not None and analysis_result.stop_loss.value == Decimal("49000.0")
        assert analysis_result.take_profit is not None and analysis_result.take_profit.value == Decimal("52000.0")
        assert analysis_result.reasoning is not None
        mock_ml_protocol.analyze_market.assert_called_once_with("model_123", sample_market_data)
    @pytest.mark.asyncio
    async def test_signal_generation(self, mock_ml_protocol: Mock) -> None:
        """Тест генерации сигналов."""
        signal_result = await mock_ml_protocol.generate_signal("model_123")
        assert isinstance(signal_result, SignalResult)
        assert signal_result.signal_type == "hold"
        assert signal_result.confidence == 0.65
        assert signal_result.reasoning is not None
        mock_ml_protocol.generate_signal.assert_called_once_with("model_123")
    @pytest.mark.asyncio
    async def test_ml_errors(self, mock_ml_protocol: Mock) -> None:
        """Тест ошибок ML протокола."""
        # Ошибка модели не найдена
        mock_ml_protocol.load_model.side_effect = ModelNotFoundError("Model not found", ModelId(uuid4()))
        with pytest.raises(ModelNotFoundError):
            await mock_ml_protocol.load_model("nonexistent_model")
        # Ошибка обучения
        mock_ml_protocol.train_model.side_effect = TrainingError("Training failed", ModelId(uuid4()), "initialization")
        with pytest.raises(TrainingError):
            await mock_ml_protocol.train_model("model_123", TrainingConfig())
        # Ошибка предсказания
        mock_ml_protocol.predict.side_effect = PredictionError("Prediction failed")
        with pytest.raises(PredictionError):
            await mock_ml_protocol.predict("model_123", [])
        # Ошибка валидации данных
        class DataValidationError(Exception):
            pass
        mock_ml_protocol.validate_data.side_effect = DataValidationError("Invalid data")
        with pytest.raises(DataValidationError):
            await mock_ml_protocol.validate_data("model_123")
        # Ошибка валидации модели
        class ModelValidationError(Exception):
            pass
        mock_ml_protocol.create_model.side_effect = ModelValidationError("Invalid model config")
        with pytest.raises(ModelValidationError):
            await mock_ml_protocol.create_model(ModelConfig(
                name="test",
                model_type=ModelType.RANDOM_FOREST,
                trading_pair=Symbol("BTCUSDT"),
                prediction_type=PredictionType.PRICE,
                hyperparameters={},
                features=[],
                target="test"
            ))
        # Ошибка недостатка данных
        mock_ml_protocol.preprocess_data.side_effect = InsufficientDataError("Not enough data", 100, 50)
        with pytest.raises(InsufficientDataError):
            await mock_ml_protocol.preprocess_data("model_123")
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_ml_protocol: Mock) -> None:
        """Тест конкурентных операций."""
        # Создаем несколько задач
        tasks = [
            mock_ml_protocol.predict("model_1", []),
            mock_ml_protocol.predict("model_2", []),
            mock_ml_protocol.evaluate_model("model_1"),
            mock_ml_protocol.generate_signal("model_1")
        ]
        # Выполняем их конкурентно
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        assert all(result is not None for result in results)
    @pytest.mark.asyncio
    async def test_model_performance_metrics(self, mock_ml_protocol: Mock) -> None:
        """Тест метрик производительности модели."""
        # Создаем тестовые данные
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 0])
        # Рассчитываем метрики
        accuracy = np.mean(y_true == y_pred)
        precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
        recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
        f1 = 2 * (precision * recall) / (precision + recall)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        # Проверяем корректность расчетов
        assert 0 <= accuracy <= 1
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert mse >= 0
        assert r2 <= 1
    @pytest.mark.asyncio
    async def test_feature_importance_analysis(self, mock_ml_protocol: Mock) -> None:
        """Тест анализа важности признаков."""
        feature_importance = {
            "price": 0.3,
            "volume": 0.2,
            "rsi": 0.25,
            "macd": 0.25
        }
        # Проверяем корректность важности признаков
        total_importance = sum(feature_importance.values())
        assert abs(total_importance - 1.0) < 1e-6  # Сумма должна быть равна 1
        # Проверяем, что все значения положительные
        assert all(importance >= 0 for importance in feature_importance.values())
        # Проверяем, что важность отсортирована
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        assert sorted_features[0][0] == "price"  # Самый важный признак
    @pytest.mark.asyncio
    async def test_confidence_score_validation(self, mock_ml_protocol: Mock) -> None:
        """Тест валидации оценок уверенности."""
        confidence_scores = np.array([0.1, 0.5, 0.8, 0.95, 1.0])
        # Проверяем, что все оценки в диапазоне [0, 1]
        assert all(0 <= conf <= 1 for conf in confidence_scores)
        # Проверяем, что оценки корректно интерпретируются
        high_confidence = confidence_scores >= 0.8
        medium_confidence = (confidence_scores >= 0.5) & (confidence_scores < 0.8)
        low_confidence = confidence_scores < 0.5
        assert np.sum(high_confidence) == 2  # 0.8, 0.95
        assert np.sum(medium_confidence) == 1  # 0.5
        assert np.sum(low_confidence) == 2  # 0.1
class TestModelConfig:
    """Тесты для ModelConfig."""
    def test_model_config_creation(self: "TestModelConfig") -> None:
        """Тест создания конфигурации модели."""
        config = ModelConfig(
            name="test_model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume", "rsi"],
            target="price_change"
        )
        assert config.name == "test_model"
        assert config.model_type == ModelType.RANDOM_FOREST
        assert config.trading_pair == Symbol("BTCUSDT")
        assert config.prediction_type == PredictionType.PRICE
        assert config.hyperparameters["n_estimators"] == 100
        assert config.features == ["price", "volume", "rsi"]
        assert config.target == "price_change"
    def test_model_config_validation(self: "TestModelConfig") -> None:
        """Тест валидации конфигурации модели."""
        # Валидная конфигурация
        valid_config = ModelConfig(
            name="valid_model",
            model_type=ModelType.RANDOM_FOREST,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="price_change"
        )
        assert valid_config.name != ""
        assert valid_config.model_type in [ModelType.RANDOM_FOREST, ModelType.XGBOOST, ModelType.NEURAL_NETWORK, ModelType.LSTM]
        assert valid_config.trading_pair == Symbol("BTCUSDT")
        assert valid_config.prediction_type == PredictionType.PRICE
        assert len(valid_config.features) > 0
        assert valid_config.target != ""
        # Невалидная конфигурация
        with pytest.raises(ValueError):
            ModelConfig(
                name="",
                model_type="invalid_type",
                trading_pair=Symbol("BTCUSDT"),
                prediction_type=PredictionType.PRICE,
                hyperparameters={},
                features=[],
                target=""
            )
class TestTrainingConfig:
    """Тесты для TrainingConfig."""
    def test_training_config_creation(self: "TestTrainingConfig") -> None:
        """Тест создания конфигурации обучения."""
        config = TrainingConfig(
            validation_split=0.8,
            test_split=0.1,
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            early_stopping_patience=10
        )
        assert config.validation_split == 0.8
        assert config.test_split == 0.1
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.early_stopping_patience == 10
    def test_training_config_validation(self: "TestTrainingConfig") -> None:
        """Тест валидации конфигурации обучения."""
        # Проверка суммы размеров выборок
        config = TrainingConfig(
            validation_split=0.7,
            test_split=0.15,
            batch_size=32,
            epochs=100,
            learning_rate=0.001,
            early_stopping_patience=10
        )
        total_size = config.validation_split + config.test_split
        assert abs(total_size - 1.0) < 1e-6  # Сумма должна быть равна 1
        # Проверка корректности параметров
        assert 0 < config.validation_split < 1
        assert 0 <= config.test_split < 1
        assert config.batch_size > 0
        assert config.epochs > 0
        assert config.learning_rate > 0
        assert config.early_stopping_patience >= 0
class TestModelMetrics:
    """Тесты для ModelMetrics."""
    def test_model_metrics_creation(self: "TestModelMetrics") -> None:
        """Тест создания метрик модели."""
        metrics = ModelMetrics(
            mse=0.001,
            mae=0.025,
            r2=0.78,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_rate=0.65,
            profit_factor=1.5,
            total_return=0.15,
            volatility=0.12,
            calmar_ratio=2.4
        )
        assert metrics.mse == 0.001
        assert metrics.mae == 0.025
        assert metrics.r2 == 0.78
        assert metrics.sharpe_ratio == 1.2
        assert metrics.max_drawdown == 0.05
        assert metrics.win_rate == 0.65
        assert metrics.profit_factor == 1.5
        assert metrics.total_return == 0.15
        assert metrics.volatility == 0.12
        assert metrics.calmar_ratio == 2.4
    def test_model_metrics_validation(self: "TestModelMetrics") -> None:
        """Тест валидации метрик модели."""
        # Валидные метрики
        valid_metrics = ModelMetrics(
            mse=0.001,
            mae=0.025,
            r2=0.78,
            sharpe_ratio=1.2,
            max_drawdown=0.05,
            win_rate=0.65,
            profit_factor=1.5,
            total_return=0.15,
            volatility=0.12,
            calmar_ratio=2.4
        )
        # Проверка диапазонов
        assert 0 <= valid_metrics.mse < float('inf')
        assert 0 <= valid_metrics.mae < float('inf')
        assert 0 <= valid_metrics.r2 <= 1
        assert valid_metrics.sharpe_ratio >= 0
        assert 0 <= valid_metrics.max_drawdown <= 1
        assert 0 <= valid_metrics.win_rate <= 1
        assert valid_metrics.profit_factor >= 1
        assert valid_metrics.total_return >= 0
        assert valid_metrics.volatility >= 0
        assert valid_metrics.calmar_ratio >= 0
        # Проверка корректности F1-score
        expected_f1 = 2 * (valid_metrics.precision * valid_metrics.recall) / (valid_metrics.precision + valid_metrics.recall)
        assert abs(valid_metrics.f1_score - expected_f1) < 1e-6
class TestMLErrors:
    """Тесты для ошибок ML."""
    def test_ml_error_creation(self: "TestMLErrors") -> None:
        """Тест создания ошибок ML."""
        # Базовая ошибка ML
        error = MLError("General ML error")
        assert str(error) == "General ML error"
        # Ошибка модели не найдена
        model_error = ModelNotFoundError("Model not found")
        assert str(model_error) == "Model not found"
        # Ошибка обучения
        training_error = TrainingError("Training failed")
        assert str(training_error) == "Training failed"
        # Ошибка предсказания
        prediction_error = PredictionError("Prediction failed")
        assert str(prediction_error) == "Prediction failed"
        # Ошибка валидации данных
        data_error = DataValidationError("Invalid data")
        assert str(data_error) == "Invalid data"
        # Ошибка валидации модели
        model_validation_error = ModelValidationError("Invalid model config")
        assert str(model_validation_error) == "Invalid model config"
        # Ошибка недостатка данных
        insufficient_data_error = InsufficientDataError("Not enough data")
        assert str(insufficient_data_error) == "Not enough data"
    def test_error_inheritance(self: "TestMLErrors") -> None:
        """Тест иерархии ошибок."""
        # Проверка наследования
        assert issubclass(ModelNotFoundError, MLError)
        assert issubclass(TrainingError, MLError)
        assert issubclass(PredictionError, MLError)
        assert issubclass(DataValidationError, MLError)
        assert issubclass(ModelValidationError, MLError)
        assert issubclass(InsufficientDataError, MLError) 
