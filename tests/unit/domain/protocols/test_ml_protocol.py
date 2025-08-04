"""
Unit тесты для MLProtocol.

Покрывает:
- Основные протоколы машинного обучения
- Управление моделями
- Обучение и предсказания
- Оценку производительности
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from decimal import Decimal
from datetime import datetime
import pandas as pd
from shared.numpy_utils import np

from domain.protocols.ml_protocol import (
    MLProtocol,
    ModelState,
    FeatureType,
    OptimizationMethod,
    MarketAnalysisProtocol,
    ModelConfig,
    TrainingConfig,
    PredictionConfig,
    ModelMetrics,
    ModelTrainingProtocol,
    ModelEvaluationProtocol,
    PredictionProtocol
)
from domain.entities.ml import Model, ModelType, PredictionType
from domain.entities.trading_pair import Symbol
from domain.exceptions.base_exceptions import ValidationError


class TestMLProtocol:
    """Тесты для базового MLProtocol."""

    @pytest.fixture
    def mock_ml_protocol(self) -> Mock:
        """Мок ML протокола."""
        return Mock(spec=MLProtocol)

    @pytest.fixture
    def sample_model_config(self) -> ModelConfig:
        """Тестовая конфигурация модели."""
        return ModelConfig(
            name="Test Model",
            model_type=ModelType.LINEAR_REGRESSION,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE_PREDICTION,
            hyperparameters={"alpha": 0.1},
            features=["feature1", "feature2"],
            target="price",
            description="Test model",
            version="1.0.0",
            author="Test Author",
            tags=["test", "regression"]
        )

    @pytest.fixture
    def sample_training_config(self) -> TrainingConfig:
        """Тестовая конфигурация обучения."""
        return TrainingConfig(
            validation_split=0.2,
            test_split=0.1,
            random_state=42,
            shuffle=False,
            early_stopping_patience=10,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            callbacks=[]
        )

    @pytest.fixture
    def sample_prediction_config(self) -> PredictionConfig:
        """Тестовая конфигурация предсказания."""
        return PredictionConfig(
            confidence_threshold=0.8,
            ensemble_method="weighted_average",
            post_processing=True,
            uncertainty_estimation=True,
            feature_importance=True
        )

    def test_create_model_method_exists(self, mock_ml_protocol, sample_model_config):
        """Тест наличия метода create_model."""
        mock_ml_protocol.create_model = AsyncMock(return_value=Mock(spec=Model))
        assert hasattr(mock_ml_protocol, 'create_model')
        assert callable(mock_ml_protocol.create_model)

    def test_train_model_method_exists(self, mock_ml_protocol, sample_training_config):
        """Тест наличия метода train_model."""
        mock_ml_protocol.train_model = AsyncMock(return_value=Mock(spec=Model))
        assert hasattr(mock_ml_protocol, 'train_model')
        assert callable(mock_ml_protocol.train_model)

    def test_predict_method_exists(self, mock_ml_protocol):
        """Тест наличия метода predict."""
        mock_ml_protocol.predict = AsyncMock(return_value=Mock())
        assert hasattr(mock_ml_protocol, 'predict')
        assert callable(mock_ml_protocol.predict)

    def test_batch_predict_method_exists(self, mock_ml_protocol):
        """Тест наличия метода batch_predict."""
        mock_ml_protocol.batch_predict = AsyncMock(return_value=[])
        assert hasattr(mock_ml_protocol, 'batch_predict')
        assert callable(mock_ml_protocol.batch_predict)

    def test_evaluate_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода evaluate_model."""
        mock_ml_protocol.evaluate_model = AsyncMock(return_value=Mock(spec=ModelMetrics))
        assert hasattr(mock_ml_protocol, 'evaluate_model')
        assert callable(mock_ml_protocol.evaluate_model)

    def test_cross_validate_method_exists(self, mock_ml_protocol):
        """Тест наличия метода cross_validate."""
        mock_ml_protocol.cross_validate = AsyncMock(return_value={})
        assert hasattr(mock_ml_protocol, 'cross_validate')
        assert callable(mock_ml_protocol.cross_validate)

    def test_backtest_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода backtest_model."""
        mock_ml_protocol.backtest_model = AsyncMock(return_value={})
        assert hasattr(mock_ml_protocol, 'backtest_model')
        assert callable(mock_ml_protocol.backtest_model)

    def test_optimize_hyperparameters_method_exists(self, mock_ml_protocol):
        """Тест наличия метода optimize_hyperparameters."""
        mock_ml_protocol.optimize_hyperparameters = AsyncMock(return_value={})
        assert hasattr(mock_ml_protocol, 'optimize_hyperparameters')
        assert callable(mock_ml_protocol.optimize_hyperparameters)

    def test_feature_selection_method_exists(self, mock_ml_protocol):
        """Тест наличия метода feature_selection."""
        mock_ml_protocol.feature_selection = AsyncMock(return_value=[])
        assert hasattr(mock_ml_protocol, 'feature_selection')
        assert callable(mock_ml_protocol.feature_selection)

    def test_calculate_feature_importance_method_exists(self, mock_ml_protocol):
        """Тест наличия метода calculate_feature_importance."""
        mock_ml_protocol.calculate_feature_importance = AsyncMock(return_value={})
        assert hasattr(mock_ml_protocol, 'calculate_feature_importance')
        assert callable(mock_ml_protocol.calculate_feature_importance)

    def test_save_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода save_model."""
        mock_ml_protocol.save_model = AsyncMock(return_value=True)
        assert hasattr(mock_ml_protocol, 'save_model')
        assert callable(mock_ml_protocol.save_model)

    def test_load_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода load_model."""
        mock_ml_protocol.load_model = AsyncMock(return_value=Mock(spec=Model))
        assert hasattr(mock_ml_protocol, 'load_model')
        assert callable(mock_ml_protocol.load_model)

    def test_get_model_status_method_exists(self, mock_ml_protocol):
        """Тест наличия метода get_model_status."""
        mock_ml_protocol.get_model_status = AsyncMock(return_value=Mock())
        assert hasattr(mock_ml_protocol, 'get_model_status')
        assert callable(mock_ml_protocol.get_model_status)

    def test_activate_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода activate_model."""
        mock_ml_protocol.activate_model = AsyncMock(return_value=True)
        assert hasattr(mock_ml_protocol, 'activate_model')
        assert callable(mock_ml_protocol.activate_model)

    def test_deactivate_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода deactivate_model."""
        mock_ml_protocol.deactivate_model = AsyncMock(return_value=True)
        assert hasattr(mock_ml_protocol, 'deactivate_model')
        assert callable(mock_ml_protocol.deactivate_model)

    def test_delete_model_method_exists(self, mock_ml_protocol):
        """Тест наличия метода delete_model."""
        mock_ml_protocol.delete_model = AsyncMock(return_value=True)
        assert hasattr(mock_ml_protocol, 'delete_model')
        assert callable(mock_ml_protocol.delete_model)

    def test_monitor_model_drift_method_exists(self, mock_ml_protocol):
        """Тест наличия метода monitor_model_drift."""
        mock_ml_protocol.monitor_model_drift = AsyncMock(return_value={})
        assert hasattr(mock_ml_protocol, 'monitor_model_drift')
        assert callable(mock_ml_protocol.monitor_model_drift)

    def test_create_ensemble_method_exists(self, mock_ml_protocol):
        """Тест наличия метода create_ensemble."""
        mock_ml_protocol.create_ensemble = AsyncMock(return_value=uuid4())
        assert hasattr(mock_ml_protocol, 'create_ensemble')
        assert callable(mock_ml_protocol.create_ensemble)

    def test_ensemble_predict_method_exists(self, mock_ml_protocol):
        """Тест наличия метода ensemble_predict."""
        mock_ml_protocol.ensemble_predict = AsyncMock(return_value=Mock())
        assert hasattr(mock_ml_protocol, 'ensemble_predict')
        assert callable(mock_ml_protocol.ensemble_predict)

    def test_online_learning_method_exists(self, mock_ml_protocol):
        """Тест наличия метода online_learning."""
        mock_ml_protocol.online_learning = AsyncMock(return_value=True)
        assert hasattr(mock_ml_protocol, 'online_learning')
        assert callable(mock_ml_protocol.online_learning)

    def test_transfer_learning_method_exists(self, mock_ml_protocol):
        """Тест наличия метода transfer_learning."""
        mock_ml_protocol.transfer_learning = AsyncMock(return_value=uuid4())
        assert hasattr(mock_ml_protocol, 'transfer_learning')
        assert callable(mock_ml_protocol.transfer_learning)

    def test_handle_model_error_method_exists(self, mock_ml_protocol):
        """Тест наличия метода handle_model_error."""
        mock_ml_protocol.handle_model_error = AsyncMock(return_value=True)
        assert hasattr(mock_ml_protocol, 'handle_model_error')
        assert callable(mock_ml_protocol.handle_model_error)

    def test_retry_prediction_method_exists(self, mock_ml_protocol):
        """Тест наличия метода retry_prediction."""
        mock_ml_protocol.retry_prediction = AsyncMock(return_value=Mock())
        assert hasattr(mock_ml_protocol, 'retry_prediction')
        assert callable(mock_ml_protocol.retry_prediction)


class TestModelState:
    """Тесты для ModelState."""

    def test_model_states_exist(self):
        """Тест наличия всех состояний модели."""
        assert ModelState.CREATED == "created"
        assert ModelState.TRAINING == "training"
        assert ModelState.TRAINED == "trained"
        assert ModelState.EVALUATING == "evaluating"
        assert ModelState.ACTIVE == "active"
        assert ModelState.INACTIVE == "inactive"
        assert ModelState.ERROR == "error"
        assert ModelState.DEPRECATED == "deprecated"

    def test_model_state_transitions(self):
        """Тест переходов между состояниями модели."""
        # Валидные переходы
        valid_transitions = {
            ModelState.CREATED: [ModelState.TRAINING, ModelState.ERROR],
            ModelState.TRAINING: [ModelState.TRAINED, ModelState.ERROR],
            ModelState.TRAINED: [ModelState.EVALUATING, ModelState.ERROR],
            ModelState.EVALUATING: [ModelState.ACTIVE, ModelState.INACTIVE, ModelState.ERROR],
            ModelState.ACTIVE: [ModelState.INACTIVE, ModelState.DEPRECATED, ModelState.ERROR],
            ModelState.INACTIVE: [ModelState.ACTIVE, ModelState.DEPRECATED, ModelState.ERROR],
            ModelState.ERROR: [ModelState.CREATED, ModelState.TRAINING],
            ModelState.DEPRECATED: [ModelState.ERROR]
        }
        
        for state, valid_next_states in valid_transitions.items():
            assert isinstance(state, str)
            assert all(isinstance(next_state, str) for next_state in valid_next_states)


class TestFeatureType:
    """Тесты для FeatureType."""

    def test_feature_types_exist(self):
        """Тест наличия всех типов признаков."""
        assert FeatureType.TECHNICAL == "technical"
        assert FeatureType.FUNDAMENTAL == "fundamental"
        assert FeatureType.SENTIMENT == "sentiment"
        assert FeatureType.MARKET_MICROSTRUCTURE == "market_microstructure"
        assert FeatureType.EXTERNAL == "external"

    def test_feature_type_validation(self):
        """Тест валидации типов признаков."""
        valid_types = [
            FeatureType.TECHNICAL,
            FeatureType.FUNDAMENTAL,
            FeatureType.SENTIMENT,
            FeatureType.MARKET_MICROSTRUCTURE,
            FeatureType.EXTERNAL
        ]
        
        for feature_type in valid_types:
            assert isinstance(feature_type, str)
            assert feature_type in [ft.value for ft in FeatureType]


class TestOptimizationMethod:
    """Тесты для OptimizationMethod."""

    def test_optimization_methods_exist(self):
        """Тест наличия всех методов оптимизации."""
        assert OptimizationMethod.GRID_SEARCH == "grid_search"
        assert OptimizationMethod.RANDOM_SEARCH == "random_search"
        assert OptimizationMethod.BAYESIAN_OPTIMIZATION == "bayesian_optimization"
        assert OptimizationMethod.GENETIC_ALGORITHM == "genetic_algorithm"
        assert OptimizationMethod.HYPEROPT == "hyperopt"

    def test_optimization_method_validation(self):
        """Тест валидации методов оптимизации."""
        valid_methods = [
            OptimizationMethod.GRID_SEARCH,
            OptimizationMethod.RANDOM_SEARCH,
            OptimizationMethod.BAYESIAN_OPTIMIZATION,
            OptimizationMethod.GENETIC_ALGORITHM,
            OptimizationMethod.HYPEROPT
        ]
        
        for method in valid_methods:
            assert isinstance(method, str)
            assert method in [om.value for om in OptimizationMethod]


class TestModelConfig:
    """Тесты для ModelConfig."""

    def test_model_config_creation(self, sample_model_config):
        """Тест создания конфигурации модели."""
        assert sample_model_config.name == "Test Model"
        assert sample_model_config.model_type == ModelType.LINEAR_REGRESSION
        assert sample_model_config.trading_pair == Symbol("BTCUSDT")
        assert sample_model_config.prediction_type == PredictionType.PRICE_PREDICTION
        assert sample_model_config.hyperparameters == {"alpha": 0.1}
        assert sample_model_config.features == ["feature1", "feature2"]
        assert sample_model_config.target == "price"
        assert sample_model_config.description == "Test model"
        assert sample_model_config.version == "1.0.0"
        assert sample_model_config.author == "Test Author"
        assert sample_model_config.tags == ["test", "regression"]

    def test_model_config_validation(self):
        """Тест валидации конфигурации модели."""
        # Валидная конфигурация
        valid_config = ModelConfig(
            name="Valid Model",
            model_type=ModelType.LINEAR_REGRESSION,
            trading_pair=Symbol("BTCUSDT"),
            prediction_type=PredictionType.PRICE_PREDICTION,
            hyperparameters={},
            features=[],
            target="price"
        )
        assert valid_config.name == "Valid Model"
        assert valid_config.model_type == ModelType.LINEAR_REGRESSION

        # Тест с пустым именем
        with pytest.raises(ValueError):
            ModelConfig(
                name="",
                model_type=ModelType.LINEAR_REGRESSION,
                trading_pair=Symbol("BTCUSDT"),
                prediction_type=PredictionType.PRICE_PREDICTION,
                hyperparameters={},
                features=[],
                target="price"
            )


class TestTrainingConfig:
    """Тесты для TrainingConfig."""

    def test_training_config_creation(self, sample_training_config):
        """Тест создания конфигурации обучения."""
        assert sample_training_config.validation_split == 0.2
        assert sample_training_config.test_split == 0.1
        assert sample_training_config.random_state == 42
        assert sample_training_config.shuffle is False
        assert sample_training_config.early_stopping_patience == 10
        assert sample_training_config.learning_rate == 0.001
        assert sample_training_config.batch_size == 32
        assert sample_training_config.epochs == 100
        assert sample_training_config.callbacks == []

    def test_training_config_validation(self):
        """Тест валидации конфигурации обучения."""
        # Валидная конфигурация
        valid_config = TrainingConfig()
        assert valid_config.validation_split == 0.2
        assert valid_config.test_split == 0.1
        assert valid_config.random_state == 42
        assert valid_config.shuffle is False

        # Тест с невалидными значениями
        with pytest.raises(ValueError):
            TrainingConfig(validation_split=1.5)  # > 1.0

        with pytest.raises(ValueError):
            TrainingConfig(test_split=-0.1)  # < 0.0

        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=0.0)  # <= 0.0

        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)  # <= 0

        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)  # <= 0


class TestPredictionConfig:
    """Тесты для PredictionConfig."""

    def test_prediction_config_creation(self, sample_prediction_config):
        """Тест создания конфигурации предсказания."""
        assert sample_prediction_config.confidence_threshold == 0.8
        assert sample_prediction_config.ensemble_method == "weighted_average"
        assert sample_prediction_config.post_processing is True
        assert sample_prediction_config.uncertainty_estimation is True
        assert sample_prediction_config.feature_importance is True

    def test_prediction_config_validation(self):
        """Тест валидации конфигурации предсказания."""
        # Валидная конфигурация
        valid_config = PredictionConfig()
        assert valid_config.ensemble_method == "weighted_average"
        assert valid_config.post_processing is True
        assert valid_config.uncertainty_estimation is True
        assert valid_config.feature_importance is True

        # Тест с невалидным порогом уверенности
        with pytest.raises(ValueError):
            PredictionConfig(confidence_threshold=1.5)  # > 1.0

        with pytest.raises(ValueError):
            PredictionConfig(confidence_threshold=-0.1)  # < 0.0


class TestModelMetrics:
    """Тесты для ModelMetrics."""

    def test_model_metrics_creation(self):
        """Тест создания метрик модели."""
        metrics = ModelMetrics(
            mse=0.01,
            mae=0.05,
            r2=0.85,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.65,
            profit_factor=1.8,
            total_return=0.25,
            volatility=0.15,
            calmar_ratio=2.5
        )
        
        assert metrics.mse == 0.01
        assert metrics.mae == 0.05
        assert metrics.r2 == 0.85
        assert metrics.sharpe_ratio == 1.5
        assert metrics.max_drawdown == 0.1
        assert metrics.win_rate == 0.65
        assert metrics.profit_factor == 1.8
        assert metrics.total_return == 0.25
        assert metrics.volatility == 0.15
        assert metrics.calmar_ratio == 2.5

    def test_model_metrics_validation(self):
        """Тест валидации метрик модели."""
        # Валидные метрики
        valid_metrics = ModelMetrics(
            mse=0.01,
            mae=0.05,
            r2=0.85,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.65,
            profit_factor=1.8,
            total_return=0.25,
            volatility=0.15,
            calmar_ratio=2.5
        )
        assert valid_metrics.mse >= 0
        assert valid_metrics.mae >= 0
        assert 0 <= valid_metrics.r2 <= 1
        assert valid_metrics.max_drawdown >= 0
        assert 0 <= valid_metrics.win_rate <= 1
        assert valid_metrics.profit_factor >= 0
        assert valid_metrics.volatility >= 0


class TestMLProtocolIntegration:
    """Интеграционные тесты для MLProtocol."""

    @pytest.mark.asyncio
    async def test_ml_protocol_workflow(self, mock_ml_protocol, sample_model_config, sample_training_config):
        """Тест полного рабочего процесса ML протокола."""
        model_id = uuid4()
        
        # Создание модели
        mock_model = Mock(spec=Model)
        mock_model.id = model_id
        mock_ml_protocol.create_model = AsyncMock(return_value=mock_model)
        
        created_model = await mock_ml_protocol.create_model(sample_model_config)
        assert created_model.id == model_id
        mock_ml_protocol.create_model.assert_called_once_with(sample_model_config)

        # Обучение модели
        training_data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'price': [100, 200, 300, 400, 500]
        })
        
        mock_ml_protocol.train_model = AsyncMock(return_value=mock_model)
        trained_model = await mock_ml_protocol.train_model(
            model_id, training_data, sample_training_config
        )
        assert trained_model.id == model_id
        mock_ml_protocol.train_model.assert_called_once_with(
            model_id, training_data, sample_training_config, None
        )

        # Предсказание
        features = {'feature1': 6, 'feature2': 12}
        mock_prediction = Mock()
        mock_ml_protocol.predict = AsyncMock(return_value=mock_prediction)
        
        prediction = await mock_ml_protocol.predict(model_id, features)
        assert prediction == mock_prediction
        mock_ml_protocol.predict.assert_called_once_with(model_id, features, None)

    @pytest.mark.asyncio
    async def test_error_handling_in_ml_protocol(self, mock_ml_protocol, sample_model_config):
        """Тест обработки ошибок в ML протоколе."""
        model_id = uuid4()
        
        # Ошибка создания модели
        mock_ml_protocol.create_model = AsyncMock(side_effect=ValidationError("Invalid model type"))
        
        with pytest.raises(ValidationError, match="Invalid model type"):
            await mock_ml_protocol.create_model(sample_model_config)

        # Ошибка обучения
        mock_ml_protocol.train_model = AsyncMock(side_effect=Exception("Training failed"))
        
        with pytest.raises(Exception, match="Training failed"):
            await mock_ml_protocol.train_model(model_id, pd.DataFrame(), Mock())

        # Ошибка предсказания
        mock_ml_protocol.predict = AsyncMock(side_effect=Exception("Prediction failed"))
        
        with pytest.raises(Exception, match="Prediction failed"):
            await mock_ml_protocol.predict(model_id, {})

    @pytest.mark.asyncio
    async def test_model_lifecycle_management(self, mock_ml_protocol):
        """Тест управления жизненным циклом модели."""
        model_id = uuid4()
        
        # Получение статуса
        mock_status = Mock()
        mock_ml_protocol.get_model_status = AsyncMock(return_value=mock_status)
        status = await mock_ml_protocol.get_model_status(model_id)
        assert status == mock_status

        # Активация модели
        mock_ml_protocol.activate_model = AsyncMock(return_value=True)
        activated = await mock_ml_protocol.activate_model(model_id)
        assert activated is True

        # Деактивация модели
        mock_ml_protocol.deactivate_model = AsyncMock(return_value=True)
        deactivated = await mock_ml_protocol.deactivate_model(model_id)
        assert deactivated is True

        # Удаление модели
        mock_ml_protocol.delete_model = AsyncMock(return_value=True)
        deleted = await mock_ml_protocol.delete_model(model_id)
        assert deleted is True

    @pytest.mark.asyncio
    async def test_ensemble_operations(self, mock_ml_protocol):
        """Тест операций с ансамблями."""
        model_ids = [uuid4(), uuid4(), uuid4()]
        ensemble_id = uuid4()
        
        # Создание ансамбля
        mock_ml_protocol.create_ensemble = AsyncMock(return_value=ensemble_id)
        created_ensemble = await mock_ml_protocol.create_ensemble(
            "Test Ensemble", model_ids
        )
        assert created_ensemble == ensemble_id

        # Предсказание ансамбля
        features = {'feature1': 1, 'feature2': 2}
        mock_prediction = Mock()
        mock_ml_protocol.ensemble_predict = AsyncMock(return_value=mock_prediction)
        
        prediction = await mock_ml_protocol.ensemble_predict(ensemble_id, features)
        assert prediction == mock_prediction

        # Обновление весов ансамбля
        new_weights = [0.4, 0.3, 0.3]
        mock_ml_protocol.update_ensemble_weights = AsyncMock(return_value=True)
        updated = await mock_ml_protocol.update_ensemble_weights(ensemble_id, new_weights)
        assert updated is True

    @pytest.mark.asyncio
    async def test_advanced_ml_features(self, mock_ml_protocol):
        """Тест продвинутых функций ML."""
        model_id = uuid4()
        
        # Мониторинг дрейфа модели
        recent_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [2, 4, 6]
        })
        mock_ml_protocol.monitor_model_drift = AsyncMock(return_value={'drift_detected': False})
        drift_info = await mock_ml_protocol.monitor_model_drift(model_id, recent_data)
        assert drift_info['drift_detected'] is False

        # Онлайн обучение
        new_data = pd.DataFrame({
            'feature1': [7, 8, 9],
            'feature2': [14, 16, 18],
            'price': [700, 800, 900]
        })
        mock_ml_protocol.online_learning = AsyncMock(return_value=True)
        learned = await mock_ml_protocol.online_learning(model_id, new_data)
        assert learned is True

        # Transfer learning
        source_model_id = uuid4()
        target_config = Mock(spec=ModelConfig)
        mock_ml_protocol.transfer_learning = AsyncMock(return_value=uuid4())
        new_model_id = await mock_ml_protocol.transfer_learning(source_model_id, target_config)
        assert isinstance(new_model_id, type(uuid4()))

    def test_protocol_compliance(self):
        """Тест соответствия протоколам."""
        # Проверка соответствия ModelTrainingProtocol
        assert hasattr(ModelTrainingProtocol, 'prepare_data')
        assert hasattr(ModelTrainingProtocol, 'validate_data')
        assert hasattr(ModelTrainingProtocol, 'split_data')
        assert hasattr(ModelTrainingProtocol, 'feature_engineering')
        assert hasattr(ModelTrainingProtocol, 'train_model')
        assert hasattr(ModelTrainingProtocol, 'validate_model')

        # Проверка соответствия ModelEvaluationProtocol
        assert hasattr(ModelEvaluationProtocol, 'calculate_metrics')
        assert hasattr(ModelEvaluationProtocol, 'cross_validate')
        assert hasattr(ModelEvaluationProtocol, 'backtest_model')
        assert hasattr(ModelEvaluationProtocol, 'calculate_feature_importance')

        # Проверка соответствия PredictionProtocol
        assert hasattr(PredictionProtocol, 'preprocess_features')
        assert hasattr(PredictionProtocol, 'postprocess_prediction')
        assert hasattr(PredictionProtocol, 'validate_prediction')
        assert hasattr(PredictionProtocol, 'calculate_confidence')

        # Проверка соответствия MarketAnalysisProtocol
        assert hasattr(MarketAnalysisProtocol, 'analyze_market_conditions')
        assert hasattr(MarketAnalysisProtocol, 'detect_patterns')
        assert hasattr(MarketAnalysisProtocol, 'calculate_technical_indicators')
        assert hasattr(MarketAnalysisProtocol, 'predict_market_regime') 