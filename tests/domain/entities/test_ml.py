"""Тесты для ML entities."""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4
from unittest.mock import Mock, patch

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
    """Тесты для класса Model."""

    def test_model_creation(self: "TestModel") -> None:
        """Тест создания модели."""
        model = Model(
            name="Test Model",
            description="Test Description",
            model_type=ModelType.LINEAR_REGRESSION,
            trading_pair="BTCUSDT",
            prediction_type=PredictionType.PRICE
        )
        
        assert model.name == "Test Model"
        assert model.description == "Test Description"
        assert model.model_type == ModelType.LINEAR_REGRESSION
        assert model.status == ModelStatus.INACTIVE
        assert model.trading_pair == "BTCUSDT"
        assert model.prediction_type == PredictionType.PRICE
        assert isinstance(model.id, UUID)
        assert isinstance(model.created_at, datetime)
        assert isinstance(model.updated_at, datetime)

    def test_model_default_values(self: "TestModel") -> None:
        """Тест значений по умолчанию."""
        model = Model()
        
        assert model.name == ""
        assert model.description == ""
        assert model.model_type == ModelType.LINEAR_REGRESSION
        assert model.status == ModelStatus.INACTIVE
        assert model.version == "1.0.0"
        assert model.trading_pair == ""
        assert model.prediction_type == PredictionType.PRICE
        assert model.accuracy == Decimal("0")
        assert model.precision == Decimal("0")
        assert model.recall == Decimal("0")
        assert model.f1_score == Decimal("0")
        assert model.mse == Decimal("0")
        assert model.mae == Decimal("0")

    def test_model_post_init_conversion(self: "TestModel") -> None:
        """Тест конвертации типов в __post_init__."""
        model = Model(
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            mse=0.15,
            mae=0.12
        )
        
        assert isinstance(model.accuracy, Decimal)
        assert isinstance(model.precision, Decimal)
        assert isinstance(model.recall, Decimal)
        assert isinstance(model.f1_score, Decimal)
        assert isinstance(model.mse, Decimal)
        assert isinstance(model.mae, Decimal)

    def test_model_activate(self: "TestModel") -> None:
        """Тест активации модели."""
        model = Model()
        original_updated_at = model.updated_at
        
        model.activate()
        
        assert model.status == ModelStatus.ACTIVE
        assert model.updated_at >= original_updated_at

    def test_model_deactivate(self: "TestModel") -> None:
        """Тест деактивации модели."""
        model = Model()
        model.status = ModelStatus.ACTIVE
        original_updated_at = model.updated_at
        
        model.deactivate()
        
        assert model.status == ModelStatus.INACTIVE
        assert model.updated_at >= original_updated_at

    def test_model_mark_trained(self: "TestModel") -> None:
        """Тест отметки модели как обученной."""
        model = Model()
        original_updated_at = model.updated_at
        
        model.mark_trained()
        
        assert model.trained_at is not None
        assert model.updated_at >= original_updated_at

    def test_model_update_metrics(self: "TestModel") -> None:
        """Тест обновления метрик."""
        model = Model()
        metrics = {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "mse": 0.15,
            "mae": 0.12
        }
        
        model.update_metrics(metrics)
        
        assert model.accuracy == Decimal("0.85")
        assert model.precision == Decimal("0.82")
        assert model.recall == Decimal("0.88")
        assert model.f1_score == Decimal("0.85")
        assert model.mse == Decimal("0.15")
        assert model.mae == Decimal("0.12")

    def test_model_is_ready_for_prediction(self: "TestModel") -> None:
        """Тест готовности модели к предсказаниям."""
        model = Model()
        assert not model.is_ready_for_prediction()
        
        model.status = ModelStatus.ACTIVE
        model.mark_trained()
        model.accuracy = Decimal("0.85")
        assert model.is_ready_for_prediction()

    def test_model_serialization(self: "TestModel") -> None:
        """Тест сериализации модели."""
        model = Model(
            name="Test Model",
            description="Test Description",
            model_type=ModelType.LINEAR_REGRESSION,
            status=ModelStatus.ACTIVE,
            trading_pair="BTCUSDT",
            prediction_type=PredictionType.PRICE,
            accuracy=Decimal("0.85"),
            precision=Decimal("0.82"),
            recall=Decimal("0.88"),
            f1_score=Decimal("0.85"),
            mse=Decimal("0.15"),
            mae=Decimal("0.12"),
            metadata={"test": "data"}
        )
        model.mark_trained()
        
        data = model.to_dict()
        
        assert data["name"] == "Test Model"
        assert data["description"] == "Test Description"
        assert data["model_type"] == "linear_regression"
        assert data["status"] == "active"
        assert data["trading_pair"] == "BTCUSDT"
        assert data["prediction_type"] == "price"
        assert data["accuracy"] == "0.85"
        assert data["precision"] == "0.82"
        assert data["recall"] == "0.88"
        assert data["f1_score"] == "0.85"
        assert data["mse"] == "0.15"
        assert data["mae"] == "0.12"
        assert data["metadata"] == {"test": "data"}
        assert data["trained_at"] is not None


class TestPrediction:
    """Тесты для класса Prediction."""

    def test_prediction_creation(self: "TestPrediction") -> None:
        """Тест создания предсказания."""
        prediction = Prediction(
            trading_pair="BTCUSDT",
            prediction_type=PredictionType.PRICE,
            value=Money(Decimal("50000"), Currency.USD),
            confidence=Decimal("0.85")
        )
        
        assert prediction.trading_pair == "BTCUSDT"
        assert prediction.prediction_type == PredictionType.PRICE
        assert isinstance(prediction.value, Money)
        assert prediction.value.amount == Decimal("50000")
        assert prediction.confidence == Decimal("0.85")
        assert isinstance(prediction.id, UUID)
        assert isinstance(prediction.timestamp, datetime)

    def test_prediction_post_init_conversion(self: "TestPrediction") -> None:
        """Тест конвертации типов в __post_init__."""
        # Ценовое предсказание
        price_prediction = Prediction(
            prediction_type=PredictionType.PRICE,
            value=50000,
            confidence=0.85
        )
        assert isinstance(price_prediction.value, Money)
        assert isinstance(price_prediction.confidence, Decimal)
        
        # Неценовое предсказание
        direction_prediction = Prediction(
            prediction_type=PredictionType.DIRECTION,
            value=1,
            confidence=0.75
        )
        assert isinstance(direction_prediction.value, Decimal)
        assert isinstance(direction_prediction.confidence, Decimal)

    def test_prediction_confidence_levels(self: "TestPrediction") -> None:
        """Тест уровней уверенности."""
        high_conf = Prediction(confidence=Decimal("0.85"))
        medium_conf = Prediction(confidence=Decimal("0.65"))
        low_conf = Prediction(confidence=Decimal("0.35"))
        
        assert high_conf.is_high_confidence()
        assert medium_conf.is_medium_confidence()
        assert low_conf.is_low_confidence()

    def test_prediction_serialization(self: "TestPrediction") -> None:
        """Тест сериализации предсказания."""
        prediction = Prediction(
            trading_pair="BTCUSDT",
            prediction_type=PredictionType.PRICE,
            value=Money(Decimal("50000"), Currency.USD),
            confidence=Decimal("0.85"),
            features={"feature1": 1.0, "feature2": 2.0},
            metadata={"source": "model1"}
        )
        
        data = prediction.to_dict()
        
        assert data["trading_pair"] == "BTCUSDT"
        assert data["prediction_type"] == "price"
        assert data["value"] == "50000.00000000"
        assert data["confidence"] == "0.85"
        assert data["features"] == {"feature1": 1.0, "feature2": 2.0}
        assert data["metadata"] == {"source": "model1"}


class TestModelEnsemble:
    """Тесты для класса ModelEnsemble."""

    def test_ensemble_creation(self: "TestModelEnsemble") -> None:
        """Тест создания ансамбля."""
        ensemble = ModelEnsemble(
            name="Test Ensemble",
            description="Test Description",
            voting_method="weighted_average"
        )
        
        assert ensemble.name == "Test Ensemble"
        assert ensemble.description == "Test Description"
        assert ensemble.voting_method == "weighted_average"
        assert ensemble.is_active is True
        assert isinstance(ensemble.id, UUID)
        assert isinstance(ensemble.created_at, datetime)
        assert isinstance(ensemble.updated_at, datetime)

    def test_ensemble_add_model(self: "TestModelEnsemble") -> None:
        """Тест добавления модели в ансамбль."""
        ensemble = ModelEnsemble()
        model = Model(name="Test Model")
        original_updated_at = ensemble.updated_at
        
        ensemble.add_model(model, weight=Decimal("0.6"))
        
        assert len(ensemble.models) == 1
        assert ensemble.models[0] == model
        assert ensemble.weights[model.id] == Decimal("0.6")
        assert ensemble.updated_at >= original_updated_at

    def test_ensemble_add_duplicate_model(self: "TestModelEnsemble") -> None:
        """Тест добавления дубликата модели."""
        ensemble = ModelEnsemble()
        model = Model(name="Test Model")
        
        ensemble.add_model(model, weight=Decimal("0.6"))
        ensemble.add_model(model, weight=Decimal("0.8"))  # Дубликат
        
        assert len(ensemble.models) == 1
        assert ensemble.weights[model.id] == Decimal("0.6")  # Вес не изменился

    def test_ensemble_remove_model(self: "TestModelEnsemble") -> None:
        """Тест удаления модели из ансамбля."""
        ensemble = ModelEnsemble()
        model = Model(name="Test Model")
        
        ensemble.add_model(model, weight=Decimal("0.6"))
        original_updated_at = ensemble.updated_at
        
        ensemble.remove_model(model.id)
        
        assert len(ensemble.models) == 0
        assert model.id not in ensemble.weights
        assert ensemble.updated_at >= original_updated_at

    def test_ensemble_update_weight(self: "TestModelEnsemble") -> None:
        """Тест обновления веса модели."""
        ensemble = ModelEnsemble()
        model = Model(name="Test Model")
        
        ensemble.add_model(model, weight=Decimal("0.6"))
        original_updated_at = ensemble.updated_at
        
        ensemble.update_weight(model.id, Decimal("0.8"))
        
        assert ensemble.weights[model.id] == Decimal("0.8")
        assert ensemble.updated_at >= original_updated_at

    def test_ensemble_get_active_models(self: "TestModelEnsemble") -> None:
        """Тест получения активных моделей."""
        ensemble = ModelEnsemble()
        
        # Неактивная модель
        inactive_model = Model(name="Inactive Model")
        inactive_model.status = ModelStatus.INACTIVE
        
        # Активная модель
        active_model = Model(name="Active Model")
        active_model.status = ModelStatus.ACTIVE
        active_model.mark_trained()
        active_model.accuracy = Decimal("0.85")
        
        ensemble.add_model(inactive_model)
        ensemble.add_model(active_model)
        
        active_models = ensemble.get_active_models()
        
        assert len(active_models) == 1
        assert active_models[0] == active_model

    def test_ensemble_predict_no_active_models(self: "TestModelEnsemble") -> None:
        """Тест предсказания без активных моделей."""
        ensemble = ModelEnsemble()
        
        prediction = ensemble.predict({"feature1": 1.0})
        
        assert prediction is None

    def test_ensemble_predict_with_active_models(self: "TestModelEnsemble") -> None:
        """Тест предсказания с активными моделями."""
        ensemble = ModelEnsemble()
        
        # Создаем активную модель
        model = Model(name="Active Model")
        model.status = ModelStatus.ACTIVE
        model.mark_trained()
        model.accuracy = Decimal("0.85")
        model.trading_pair = "BTCUSDT"
        model.prediction_type = PredictionType.PRICE
        
        ensemble.add_model(model, weight=Decimal("1.0"))
        
        prediction = ensemble.predict({"feature1": 1.0})
        
        assert prediction is not None
        assert prediction.model_id == ensemble.id
        assert prediction.trading_pair == "BTCUSDT"
        assert prediction.prediction_type == PredictionType.PRICE
        assert prediction.confidence == Decimal("0.6")


class TestMLPipeline:
    """Тесты для класса MLPipeline."""

    def test_pipeline_creation(self: "TestMLPipeline") -> None:
        """Тест создания пайплайна."""
        pipeline = MLPipeline(
            name="Test Pipeline",
            description="Test Description"
        )
        
        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "Test Description"
        assert pipeline.is_active is True
        assert isinstance(pipeline.id, UUID)
        assert isinstance(pipeline.created_at, datetime)
        assert isinstance(pipeline.updated_at, datetime)

    def test_pipeline_add_model(self: "TestMLPipeline") -> None:
        """Тест добавления модели в пайплайн."""
        pipeline = MLPipeline()
        model = Model(name="Test Model")
        original_updated_at = pipeline.updated_at
        
        pipeline.add_model(model)
        
        assert len(pipeline.models) == 1
        assert pipeline.models[0] == model
        assert pipeline.updated_at >= original_updated_at

    def test_pipeline_add_duplicate_model(self: "TestMLPipeline") -> None:
        """Тест добавления дубликата модели."""
        pipeline = MLPipeline()
        model = Model(name="Test Model")
        
        pipeline.add_model(model)
        pipeline.add_model(model)  # Дубликат
        
        assert len(pipeline.models) == 1

    def test_pipeline_add_ensemble(self: "TestMLPipeline") -> None:
        """Тест добавления ансамбля в пайплайн."""
        pipeline = MLPipeline()
        ensemble = ModelEnsemble(name="Test Ensemble")
        original_updated_at = pipeline.updated_at
        
        pipeline.add_ensemble(ensemble)
        
        assert len(pipeline.ensembles) == 1
        assert pipeline.ensembles[0] == ensemble
        assert pipeline.updated_at >= original_updated_at

    def test_pipeline_get_best_model(self: "TestMLPipeline") -> None:
        """Тест получения лучшей модели."""
        pipeline = MLPipeline()
        
        model1 = Model(name="Model 1", accuracy=Decimal("0.8"))
        model2 = Model(name="Model 2", accuracy=Decimal("0.9"))
        model3 = Model(name="Model 3", accuracy=Decimal("0.7"))
        
        pipeline.add_model(model1)
        pipeline.add_model(model2)
        pipeline.add_model(model3)
        
        best_model = pipeline.get_best_model("accuracy")
        
        assert best_model == model2

    def test_pipeline_get_best_model_empty(self: "TestMLPipeline") -> None:
        """Тест получения лучшей модели из пустого пайплайна."""
        pipeline = MLPipeline()
        
        best_model = pipeline.get_best_model("accuracy")
        
        assert best_model is None

    def test_pipeline_get_active_models(self: "TestMLPipeline") -> None:
        """Тест получения активных моделей."""
        pipeline = MLPipeline()
        
        inactive_model = Model(name="Inactive Model")
        inactive_model.status = ModelStatus.INACTIVE
        
        active_model = Model(name="Active Model")
        active_model.status = ModelStatus.ACTIVE
        
        pipeline.add_model(inactive_model)
        pipeline.add_model(active_model)
        
        active_models = pipeline.get_active_models()
        
        assert len(active_models) == 1
        assert active_models[0] == active_model

    def test_pipeline_get_active_ensembles(self: "TestMLPipeline") -> None:
        """Тест получения активных ансамблей."""
        pipeline = MLPipeline()
        
        inactive_ensemble = ModelEnsemble(name="Inactive Ensemble")
        inactive_ensemble.is_active = False
        
        active_ensemble = ModelEnsemble(name="Active Ensemble")
        active_ensemble.is_active = True
        
        pipeline.add_ensemble(inactive_ensemble)
        pipeline.add_ensemble(active_ensemble)
        
        active_ensembles = pipeline.get_active_ensembles()
        
        assert len(active_ensembles) == 1
        assert active_ensembles[0] == active_ensemble

    def test_pipeline_serialization(self: "TestMLPipeline") -> None:
        """Тест сериализации пайплайна."""
        pipeline = MLPipeline(
            name="Test Pipeline",
            description="Test Description",
            is_active=True
        )
        
        model = Model(name="Test Model")
        ensemble = ModelEnsemble(name="Test Ensemble")
        
        pipeline.add_model(model)
        pipeline.add_ensemble(ensemble)
        
        data = pipeline.to_dict()
        
        assert data["name"] == "Test Pipeline"
        assert data["description"] == "Test Description"
        assert data["models_count"] == 1
        assert data["ensembles_count"] == 1
        assert data["active_models"] == 0  # Модель неактивна по умолчанию
        assert data["active_ensembles"] == 1
        assert data["is_active"] is True


class TestModelProtocol:
    """Тесты для протокола ModelProtocol."""

    def test_model_protocol_interface(self: "TestModelProtocol") -> None:
        """Тест интерфейса протокола."""
        # Создаем mock объект, соответствующий протоколу
        mock_model = Mock(spec=ModelProtocol)
        mock_model.predict.return_value = [1, 2, 3]
        mock_model.fit.return_value = None
        mock_model.score.return_value = 0.85
        
        # Проверяем, что методы существуют
        assert hasattr(mock_model, 'predict')
        assert hasattr(mock_model, 'fit')
        assert hasattr(mock_model, 'score')
        
        # Проверяем вызовы
        result = mock_model.predict([1, 2, 3])
        assert result == [1, 2, 3]
        
        mock_model.fit([1, 2, 3], [1, 0, 1])
        mock_model.fit.assert_called_once()
        
        score = mock_model.score([1, 2, 3], [1, 0, 1])
        assert score == 0.85


class TestMLModelInterface:
    """Тесты для абстрактного интерфейса MLModelInterface."""

    def test_ml_model_interface_abstract_methods(self: "TestMLModelInterface") -> None:
        """Тест абстрактных методов интерфейса."""
        # Создаем mock объект, реализующий интерфейс
        mock_model = Mock(spec=MLModelInterface)
        
        # Проверяем наличие абстрактных методов
        assert hasattr(mock_model, 'train')
        assert hasattr(mock_model, 'predict')
        assert hasattr(mock_model, 'evaluate')
        assert hasattr(mock_model, 'save')
        assert hasattr(mock_model, 'load')
        assert hasattr(mock_model, 'get_feature_importance')
        
        # Проверяем, что методы абстрактные
        with pytest.raises(TypeError):
            MLModelInterface()  # Нельзя создать экземпляр абстрактного класса


class TestModelType:
    """Тесты для перечисления ModelType."""

    def test_model_type_values(self: "TestModelType") -> None:
        """Тест значений типов моделей."""
        assert ModelType.LINEAR_REGRESSION.value == "linear_regression"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.GRADIENT_BOOSTING.value == "gradient_boosting"
        assert ModelType.NEURAL_NETWORK.value == "neural_network"
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.TRANSFORMER.value == "transformer"
        assert ModelType.ENSEMBLE.value == "ensemble"
        assert ModelType.CUSTOM.value == "custom"
        assert ModelType.XGBOOST.value == "xgboost"


class TestModelStatus:
    """Тесты для перечисления ModelStatus."""

    def test_model_status_values(self: "TestModelStatus") -> None:
        """Тест значений статусов моделей."""
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.TRAINED.value == "trained"
        assert ModelStatus.EVALUATING.value == "evaluating"
        assert ModelStatus.ACTIVE.value == "active"
        assert ModelStatus.INACTIVE.value == "inactive"
        assert ModelStatus.ERROR.value == "error"
        assert ModelStatus.DEPRECATED.value == "deprecated"


class TestPredictionType:
    """Тесты для перечисления PredictionType."""

    def test_prediction_type_values(self: "TestPredictionType") -> None:
        """Тест значений типов предсказаний."""
        assert PredictionType.PRICE.value == "price"
        assert PredictionType.DIRECTION.value == "direction"
        assert PredictionType.VOLATILITY.value == "volatility"
        assert PredictionType.VOLUME.value == "volume"
        assert PredictionType.REGIME.value == "regime"
        assert PredictionType.SIGNAL.value == "signal"
        assert PredictionType.PROBABILITY.value == "probability"
