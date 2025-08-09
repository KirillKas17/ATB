"""
Unit тесты для ML.

Покрывает:
- Основной функционал
- Валидацию данных
- Бизнес-логику
- Обработку ошибок
- Сериализацию/десериализацию
"""

import pytest
from shared.numpy_utils import np
import pandas as pd
from decimal import Decimal
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import Mock, patch
from uuid import uuid4

from domain.entities.ml import (
    Model,
    Prediction,
    ModelEnsemble,
    MLPipeline,
    ModelType,
    ModelStatus,
    PredictionType,
    ModelProtocol,
    MLModelInterface,
)
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class TestModel:
    """Тесты для Model."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "name": "Test Model",
            "description": "Test model description",
            "model_type": ModelType.LINEAR_REGRESSION,
            "status": ModelStatus.INACTIVE,
            "version": "1.0.0",
            "trading_pair": "BTC/USDT",
            "prediction_type": PredictionType.PRICE,
            "hyperparameters": {"learning_rate": 0.01, "epochs": 100},
            "features": ["price", "volume", "rsi"],
            "target": "next_price",
            "accuracy": Decimal("0.85"),
            "precision": Decimal("0.82"),
            "recall": Decimal("0.88"),
            "f1_score": Decimal("0.85"),
            "mse": Decimal("0.001"),
            "mae": Decimal("0.002"),
            "metadata": {"framework": "sklearn"},
        }

    @pytest.fixture
    def model(self, sample_data) -> Model:
        """Создает тестовую модель."""
        return Model(**sample_data)

    def test_creation(self, sample_data):
        """Тест создания модели."""
        model = Model(**sample_data)

        assert model.id == sample_data["id"]
        assert model.name == sample_data["name"]
        assert model.description == sample_data["description"]
        assert model.model_type == sample_data["model_type"]
        assert model.status == sample_data["status"]
        assert model.version == sample_data["version"]
        assert model.trading_pair == sample_data["trading_pair"]
        assert model.prediction_type == sample_data["prediction_type"]
        assert model.hyperparameters == sample_data["hyperparameters"]
        assert model.features == sample_data["features"]
        assert model.target == sample_data["target"]
        assert model.accuracy == sample_data["accuracy"]
        assert model.precision == sample_data["precision"]
        assert model.recall == sample_data["recall"]
        assert model.f1_score == sample_data["f1_score"]
        assert model.mse == sample_data["mse"]
        assert model.mae == sample_data["mae"]
        assert model.metadata == sample_data["metadata"]

    def test_default_creation(self):
        """Тест создания модели с дефолтными значениями."""
        model = Model()

        assert isinstance(model.id, uuid4().__class__)
        assert model.name == ""
        assert model.description == ""
        assert model.model_type == ModelType.LINEAR_REGRESSION
        assert model.status == ModelStatus.INACTIVE
        assert model.version == "1.0.0"
        assert model.trading_pair == ""
        assert model.prediction_type == PredictionType.PRICE
        assert model.hyperparameters == {}
        assert model.features == []
        assert model.target == ""
        assert model.accuracy == Decimal("0")
        assert model.precision == Decimal("0")
        assert model.recall == Decimal("0")
        assert model.f1_score == Decimal("0")
        assert model.mse == Decimal("0")
        assert model.mae == Decimal("0")
        assert model.metadata == {}

    def test_metric_normalization(self):
        """Тест нормализации метрик."""
        model = Model(accuracy=0.85, precision=0.82, recall=0.88, f1_score=0.85, mse=0.001, mae=0.002)

        assert isinstance(model.accuracy, Decimal)
        assert isinstance(model.precision, Decimal)
        assert isinstance(model.recall, Decimal)
        assert isinstance(model.f1_score, Decimal)
        assert isinstance(model.mse, Decimal)
        assert isinstance(model.mae, Decimal)

        assert model.accuracy == Decimal("0.85")
        assert model.precision == Decimal("0.82")
        assert model.recall == Decimal("0.88")
        assert model.f1_score == Decimal("0.85")
        assert model.mse == Decimal("0.001")
        assert model.mae == Decimal("0.002")

    def test_activate(self, model):
        """Тест активации модели."""
        old_updated_at = model.updated_at

        model.activate()

        assert model.status == ModelStatus.ACTIVE
        assert model.updated_at > old_updated_at

    def test_deactivate(self, model):
        """Тест деактивации модели."""
        model.status = ModelStatus.ACTIVE
        old_updated_at = model.updated_at

        model.deactivate()

        assert model.status == ModelStatus.INACTIVE
        assert model.updated_at > old_updated_at

    def test_mark_trained(self, model):
        """Тест отметки как обученной."""
        old_updated_at = model.updated_at

        model.mark_trained()

        assert model.trained_at is not None
        assert model.updated_at > old_updated_at

    def test_update_metrics(self, model):
        """Тест обновления метрик."""
        old_updated_at = model.updated_at
        new_metrics = {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92,
            "f1_score": 0.90,
            "mse": 0.0005,
            "mae": 0.001,
        }

        model.update_metrics(new_metrics)

        assert model.accuracy == Decimal("0.90")
        assert model.precision == Decimal("0.88")
        assert model.recall == Decimal("0.92")
        assert model.f1_score == Decimal("0.90")
        assert model.mse == Decimal("0.0005")
        assert model.mae == Decimal("0.001")
        assert model.updated_at > old_updated_at

    def test_is_ready_for_prediction(self, model):
        """Тест готовности к предсказаниям."""
        # Модель не готова (неактивна, не обучена, низкая точность)
        assert model.is_ready_for_prediction() is False

        # Активируем модель
        model.activate()
        assert model.is_ready_for_prediction() is False

        # Отмечаем как обученную
        model.mark_trained()
        assert model.is_ready_for_prediction() is False

        # Устанавливаем высокую точность
        model.accuracy = Decimal("0.85")
        assert model.is_ready_for_prediction() is True

        # Проверяем с низкой точностью
        model.accuracy = Decimal("0.3")
        assert model.is_ready_for_prediction() is False

    def test_to_dict(self, model):
        """Тест сериализации в словарь."""
        data = model.to_dict()

        assert data["id"] == str(model.id)
        assert data["name"] == model.name
        assert data["description"] == model.description
        assert data["model_type"] == model.model_type.value
        assert data["status"] == model.status.value
        assert data["version"] == model.version
        assert data["trading_pair"] == model.trading_pair
        assert data["prediction_type"] == model.prediction_type.value
        assert data["accuracy"] == str(model.accuracy)
        assert data["precision"] == str(model.precision)
        assert data["recall"] == str(model.recall)
        assert data["f1_score"] == str(model.f1_score)
        assert data["mse"] == str(model.mse)
        assert data["mae"] == str(model.mae)
        assert "created_at" in data
        assert "updated_at" in data
        assert "metadata" in data

    def test_model_type_enum(self):
        """Тест enum ModelType."""
        assert ModelType.LINEAR_REGRESSION.value == "linear_regression"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.GRADIENT_BOOSTING.value == "gradient_boosting"
        assert ModelType.NEURAL_NETWORK.value == "neural_network"
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.TRANSFORMER.value == "transformer"
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.CUSTOM.value == "custom"
        assert ModelType.XGBOOST.value == "xgboost"

    def test_model_status_enum(self):
        """Тест enum ModelStatus."""
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.TRAINED.value == "trained"
        assert ModelStatus.EVALUATING.value == "evaluating"
        assert ModelStatus.ACTIVE.value == "active"
        assert ModelStatus.INACTIVE.value == "inactive"
        assert ModelStatus.ERROR.value == "error"
        assert ModelStatus.DEPRECATED.value == "deprecated"

    def test_prediction_type_enum(self):
        """Тест enum PredictionType."""
        assert PredictionType.PRICE.value == "price"
        assert PredictionType.DIRECTION.value == "direction"
        assert PredictionType.VOLATILITY.value == "volatility"
        assert PredictionType.VOLUME.value == "volume"
        assert PredictionType.REGIME.value == "regime"
        assert PredictionType.SIGNAL.value == "signal"
        assert PredictionType.PROBABILITY.value == "probability"


class TestPrediction:
    """Тесты для Prediction."""

    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "id": uuid4(),
            "model_id": uuid4(),
            "trading_pair": "BTC/USDT",
            "prediction_type": PredictionType.PRICE,
            "value": Money(Decimal("50000"), Currency.USD),
            "confidence": Decimal("0.85"),
            "features": {"price": 50000, "volume": 1000, "rsi": 65},
            "metadata": {"model_version": "1.0.0"},
        }

    @pytest.fixture
    def prediction(self, sample_data) -> Prediction:
        """Создает тестовое предсказание."""
        return Prediction(**sample_data)

    def test_creation(self, sample_data):
        """Тест создания предсказания."""
        pred = Prediction(**sample_data)

        assert pred.id == sample_data["id"]
        assert pred.model_id == sample_data["model_id"]
        assert pred.trading_pair == sample_data["trading_pair"]
        assert pred.prediction_type == sample_data["prediction_type"]
        assert pred.value == sample_data["value"]
        assert pred.confidence == sample_data["confidence"]
        assert pred.features == sample_data["features"]
        assert pred.metadata == sample_data["metadata"]

    def test_default_creation(self):
        """Тест создания предсказания с дефолтными значениями."""
        pred = Prediction()

        assert isinstance(pred.id, uuid4().__class__)
        assert isinstance(pred.model_id, uuid4().__class__)
        assert pred.trading_pair == ""
        assert pred.prediction_type == PredictionType.PRICE
        assert pred.value == Decimal("0")
        assert pred.confidence == Decimal("0.5")
        assert pred.features == {}
        assert pred.metadata == {}

    def test_value_normalization_price_type(self):
        """Тест нормализации значения для типа PRICE."""
        pred = Prediction(prediction_type=PredictionType.PRICE, value=50000)

        assert isinstance(pred.value, Money)
        assert pred.value.amount == Decimal("50000")
        assert pred.value.currency == Currency.USD

    def test_value_normalization_other_types(self):
        """Тест нормализации значения для других типов."""
        pred = Prediction(prediction_type=PredictionType.DIRECTION, value=0.75)

        assert isinstance(pred.value, Decimal)
        assert pred.value == Decimal("0.75")

    def test_confidence_normalization(self):
        """Тест нормализации уверенности."""
        pred = Prediction(confidence=0.85)

        assert isinstance(pred.confidence, Decimal)
        assert pred.confidence == Decimal("0.85")

    def test_is_high_confidence(self, prediction):
        """Тест проверки высокой уверенности."""
        prediction.confidence = Decimal("0.85")
        assert prediction.is_high_confidence() is True

        prediction.confidence = Decimal("0.65")
        assert prediction.is_high_confidence() is False

    def test_is_medium_confidence(self, prediction):
        """Тест проверки средней уверенности."""
        prediction.confidence = Decimal("0.65")
        assert prediction.is_medium_confidence() is True

        prediction.confidence = Decimal("0.85")
        assert prediction.is_medium_confidence() is False

        prediction.confidence = Decimal("0.45")
        assert prediction.is_medium_confidence() is False

    def test_is_low_confidence(self, prediction):
        """Тест проверки низкой уверенности."""
        prediction.confidence = Decimal("0.45")
        assert prediction.is_low_confidence() is True

        prediction.confidence = Decimal("0.65")
        assert prediction.is_low_confidence() is False

    def test_to_dict(self, prediction):
        """Тест сериализации в словарь."""
        data = prediction.to_dict()

        assert data["id"] == str(prediction.id)
        assert data["model_id"] == str(prediction.model_id)
        assert data["trading_pair"] == prediction.trading_pair
        assert data["prediction_type"] == prediction.prediction_type.value
        assert data["value"] == str(prediction.value.amount)
        assert data["confidence"] == str(prediction.confidence)
        assert "timestamp" in data
        assert data["features"] == prediction.features
        assert data["metadata"] == prediction.metadata


class TestModelEnsemble:
    """Тесты для ModelEnsemble."""

    @pytest.fixture
    def model1(self) -> Model:
        """Создает первую тестовую модель."""
        return Model(
            id=uuid4(),
            name="Model 1",
            model_type=ModelType.LINEAR_REGRESSION,
            status=ModelStatus.ACTIVE,
            trading_pair="BTC/USDT",
            accuracy=Decimal("0.85"),
        )

    @pytest.fixture
    def model2(self) -> Model:
        """Создает вторую тестовую модель."""
        return Model(
            id=uuid4(),
            name="Model 2",
            model_type=ModelType.RANDOM_FOREST,
            status=ModelStatus.ACTIVE,
            trading_pair="BTC/USDT",
            accuracy=Decimal("0.88"),
        )

    @pytest.fixture
    def ensemble(self, model1, model2) -> ModelEnsemble:
        """Создает тестовый ансамбль."""
        return ModelEnsemble(
            id=uuid4(),
            name="Test Ensemble",
            description="Test ensemble description",
            models=[model1, model2],
            weights={model1.id: Decimal("0.6"), model2.id: Decimal("0.4")},
            voting_method="weighted_average",
        )

    def test_creation(self, model1, model2):
        """Тест создания ансамбля."""
        ensemble = ModelEnsemble(
            id=uuid4(),
            name="Test Ensemble",
            description="Test description",
            models=[model1, model2],
            weights={model1.id: Decimal("0.6"), model2.id: Decimal("0.4")},
            voting_method="weighted_average",
        )

        assert ensemble.name == "Test Ensemble"
        assert ensemble.description == "Test description"
        assert len(ensemble.models) == 2
        assert ensemble.weights[model1.id] == Decimal("0.6")
        assert ensemble.weights[model2.id] == Decimal("0.4")
        assert ensemble.voting_method == "weighted_average"
        assert ensemble.is_active is True

    def test_default_creation(self):
        """Тест создания ансамбля с дефолтными значениями."""
        ensemble = ModelEnsemble()

        assert isinstance(ensemble.id, uuid4().__class__)
        assert ensemble.name == ""
        assert ensemble.description == ""
        assert ensemble.models == []
        assert ensemble.weights == {}
        assert ensemble.voting_method == "weighted_average"
        assert ensemble.is_active is True

    def test_add_model(self, ensemble, model1):
        """Тест добавления модели."""
        new_model = Model(
            id=uuid4(),
            name="New Model",
            model_type=ModelType.GRADIENT_BOOSTING,
            status=ModelStatus.ACTIVE,
            trading_pair="BTC/USDT",
        )
        old_updated_at = ensemble.updated_at

        ensemble.add_model(new_model, Decimal("0.3"))

        assert len(ensemble.models) == 3
        assert new_model in ensemble.models
        assert ensemble.weights[new_model.id] == Decimal("0.3")
        assert ensemble.updated_at > old_updated_at

    def test_add_model_duplicate(self, ensemble, model1):
        """Тест добавления дублирующей модели."""
        initial_count = len(ensemble.models)
        old_updated_at = ensemble.updated_at

        ensemble.add_model(model1, Decimal("0.5"))

        assert len(ensemble.models) == initial_count
        assert ensemble.updated_at == old_updated_at

    def test_remove_model(self, ensemble, model1):
        """Тест удаления модели."""
        old_updated_at = ensemble.updated_at

        ensemble.remove_model(model1.id)

        assert len(ensemble.models) == 1
        assert model1 not in ensemble.models
        assert model1.id not in ensemble.weights
        assert ensemble.updated_at > old_updated_at

    def test_remove_model_nonexistent(self, ensemble):
        """Тест удаления несуществующей модели."""
        initial_count = len(ensemble.models)
        old_updated_at = ensemble.updated_at

        ensemble.remove_model(uuid4())

        assert len(ensemble.models) == initial_count
        assert ensemble.updated_at == old_updated_at

    def test_update_weight(self, ensemble, model1):
        """Тест обновления веса модели."""
        old_updated_at = ensemble.updated_at

        ensemble.update_weight(model1.id, Decimal("0.8"))

        assert ensemble.weights[model1.id] == Decimal("0.8")
        assert ensemble.updated_at > old_updated_at

    def test_update_weight_nonexistent(self, ensemble):
        """Тест обновления веса несуществующей модели."""
        old_updated_at = ensemble.updated_at

        ensemble.update_weight(uuid4(), Decimal("0.5"))

        assert ensemble.updated_at == old_updated_at

    def test_get_active_models(self, ensemble, model1, model2):
        """Тест получения активных моделей."""
        # Все модели активны
        active_models = ensemble.get_active_models()
        assert len(active_models) == 2
        assert model1 in active_models
        assert model2 in active_models

        # Деактивируем одну модель
        model1.status = ModelStatus.INACTIVE
        active_models = ensemble.get_active_models()
        assert len(active_models) == 1
        assert model2 in active_models
        assert model1 not in active_models

    def test_predict(self, ensemble, model1, model2):
        """Тест предсказания ансамблем."""
        # Устанавливаем высокую точность для готовности к предсказаниям
        model1.accuracy = Decimal("0.85")
        model2.accuracy = Decimal("0.88")

        features = {"price": 50000, "volume": 1000}

        prediction = ensemble.predict(features)

        assert prediction is not None
        assert prediction.model_id == ensemble.id
        assert prediction.trading_pair == model1.trading_pair
        assert prediction.prediction_type == model1.prediction_type
        assert prediction.features == features

    def test_predict_no_active_models(self, ensemble):
        """Тест предсказания без активных моделей."""
        # Деактивируем все модели
        for model in ensemble.models:
            model.status = ModelStatus.INACTIVE

        features = {"price": 50000, "volume": 1000}

        prediction = ensemble.predict(features)

        assert prediction is None


class TestMLPipeline:
    """Тесты для MLPipeline."""

    @pytest.fixture
    def model1(self) -> Model:
        """Создает первую тестовую модель."""
        return Model(
            id=uuid4(),
            name="Model 1",
            model_type=ModelType.LINEAR_REGRESSION,
            status=ModelStatus.ACTIVE,
            trading_pair="BTC/USDT",
        )

    @pytest.fixture
    def model2(self) -> Model:
        """Создает вторую тестовую модель."""
        return Model(
            id=uuid4(),
            name="Model 2",
            model_type=ModelType.RANDOM_FOREST,
            status=ModelStatus.INACTIVE,
            trading_pair="BTC/USDT",
        )

    @pytest.fixture
    def ensemble(self, model1) -> ModelEnsemble:
        """Создает тестовый ансамбль."""
        return ModelEnsemble(id=uuid4(), name="Test Ensemble", models=[model1], is_active=True)

    @pytest.fixture
    def pipeline(self, model1, model2, ensemble) -> MLPipeline:
        """Создает тестовый пайплайн."""
        return MLPipeline(
            id=uuid4(),
            name="Test Pipeline",
            description="Test pipeline description",
            models=[model1, model2],
            ensembles=[ensemble],
            feature_engineering={"scaling": "standard", "encoding": "onehot"},
            data_preprocessing={"normalization": True, "outlier_removal": True},
            evaluation_metrics=["accuracy", "precision", "recall"],
            is_active=True,
        )

    def test_creation(self, model1, model2, ensemble):
        """Тест создания пайплайна."""
        pipeline = MLPipeline(
            id=uuid4(),
            name="Test Pipeline",
            description="Test description",
            models=[model1, model2],
            ensembles=[ensemble],
            feature_engineering={"scaling": "standard"},
            data_preprocessing={"normalization": True},
            evaluation_metrics=["accuracy", "precision"],
            is_active=True,
        )

        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "Test description"
        assert len(pipeline.models) == 2
        assert len(pipeline.ensembles) == 1
        assert pipeline.feature_engineering == {"scaling": "standard"}
        assert pipeline.data_preprocessing == {"normalization": True}
        assert pipeline.evaluation_metrics == ["accuracy", "precision"]
        assert pipeline.is_active is True

    def test_default_creation(self):
        """Тест создания пайплайна с дефолтными значениями."""
        pipeline = MLPipeline()

        assert isinstance(pipeline.id, uuid4().__class__)
        assert pipeline.name == ""
        assert pipeline.description == ""
        assert pipeline.models == []
        assert pipeline.ensembles == []
        assert pipeline.feature_engineering == {}
        assert pipeline.data_preprocessing == {}
        assert pipeline.evaluation_metrics == []
        assert pipeline.is_active is True

    def test_add_model(self, pipeline, model1):
        """Тест добавления модели."""
        new_model = Model(
            id=uuid4(),
            name="New Model",
            model_type=ModelType.GRADIENT_BOOSTING,
            status=ModelStatus.ACTIVE,
            trading_pair="BTC/USDT",
        )
        old_updated_at = pipeline.updated_at

        pipeline.add_model(new_model)

        assert len(pipeline.models) == 3
        assert new_model in pipeline.models
        assert pipeline.updated_at > old_updated_at

    def test_add_model_duplicate(self, pipeline, model1):
        """Тест добавления дублирующей модели."""
        initial_count = len(pipeline.models)
        old_updated_at = pipeline.updated_at

        pipeline.add_model(model1)

        assert len(pipeline.models) == initial_count
        assert pipeline.updated_at == old_updated_at

    def test_add_ensemble(self, pipeline, ensemble):
        """Тест добавления ансамбля."""
        new_ensemble = ModelEnsemble(id=uuid4(), name="New Ensemble", models=[], is_active=True)
        old_updated_at = pipeline.updated_at

        pipeline.add_ensemble(new_ensemble)

        assert len(pipeline.ensembles) == 2
        assert new_ensemble in pipeline.ensembles
        assert pipeline.updated_at > old_updated_at

    def test_add_ensemble_duplicate(self, pipeline, ensemble):
        """Тест добавления дублирующего ансамбля."""
        initial_count = len(pipeline.ensembles)
        old_updated_at = pipeline.updated_at

        pipeline.add_ensemble(ensemble)

        assert len(pipeline.ensembles) == initial_count
        assert pipeline.updated_at == old_updated_at

    def test_get_best_model(self, pipeline, model1, model2):
        """Тест получения лучшей модели."""
        # Устанавливаем разные значения точности
        model1.accuracy = Decimal("0.85")
        model2.accuracy = Decimal("0.88")

        best_model = pipeline.get_best_model("accuracy")
        assert best_model == model2

        # Тест с несуществующей метрикой
        best_model = pipeline.get_best_model("nonexistent")
        assert best_model is not None  # Возвращает модель с максимальным значением 0

    def test_get_best_model_empty(self):
        """Тест получения лучшей модели при пустом списке."""
        pipeline = MLPipeline()

        best_model = pipeline.get_best_model("accuracy")
        assert best_model is None

    def test_get_active_models(self, pipeline, model1, model2):
        """Тест получения активных моделей."""
        active_models = pipeline.get_active_models()
        assert len(active_models) == 1
        assert model1 in active_models
        assert model2 not in active_models

    def test_get_active_ensembles(self, pipeline, ensemble):
        """Тест получения активных ансамблей."""
        active_ensembles = pipeline.get_active_ensembles()
        assert len(active_ensembles) == 1
        assert ensemble in active_ensembles

        # Деактивируем ансамбль
        ensemble.is_active = False
        active_ensembles = pipeline.get_active_ensembles()
        assert len(active_ensembles) == 0

    def test_to_dict(self, pipeline):
        """Тест сериализации в словарь."""
        data = pipeline.to_dict()

        assert data["id"] == str(pipeline.id)
        assert data["name"] == pipeline.name
        assert data["description"] == pipeline.description
        assert data["models_count"] == 2
        assert data["ensembles_count"] == 1
        assert data["active_models"] == 1
        assert data["active_ensembles"] == 1
        assert data["is_active"] == pipeline.is_active
        assert "created_at" in data
        assert "updated_at" in data


class TestMLModelInterface:
    """Тесты для MLModelInterface."""

    class ConcreteMLModel(MLModelInterface):
        """Конкретная реализация для тестирования."""

        def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
            return {"accuracy": 0.85, "precision": 0.82}

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0.5, 0.7, 0.3])

        def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
            return {"accuracy": 0.83, "precision": 0.80}

        def save(self, path: str) -> bool:
            return True

        def load(self, path: str) -> bool:
            return True

        def get_feature_importance(self) -> Dict[str, float]:
            return {"feature1": 0.6, "feature2": 0.4}

    def test_concrete_ml_model(self):
        """Тест конкретной реализации ML модели."""
        model = self.ConcreteMLModel()

        X = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [4, 5, 6]})
        y = pd.Series([0, 1, 0])

        # Тест обучения
        train_metrics = model.train(X, y)
        assert train_metrics["accuracy"] == 0.85
        assert train_metrics["precision"] == 0.82

        # Тест предсказания
        predictions = model.predict(X)
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 3

        # Тест оценки
        eval_metrics = model.evaluate(X, y)
        assert eval_metrics["accuracy"] == 0.83
        assert eval_metrics["precision"] == 0.80

        # Тест сохранения
        assert model.save("test_model.pkl") is True

        # Тест загрузки
        assert model.load("test_model.pkl") is True

        # Тест важности признаков
        feature_importance = model.get_feature_importance()
        assert feature_importance["feature1"] == 0.6
        assert feature_importance["feature2"] == 0.4
