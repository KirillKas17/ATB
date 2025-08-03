"""
Unit тесты для ML сервисов.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import tempfile
import os
import pandas as pd
import numpy as np
from infrastructure.external_services.ml_services import (
    MLServiceConfig,
    FeatureEngineer,
    ModelManager,
    ProductionMLService,
    MLServiceAdapter,
    LocalMLService
)
from domain.entities.ml import Model, ModelType, ModelStatus
from domain.types.external_service_types import (
    MLModelConfig,
    MLPredictionRequest
)
class TestMLServiceConfig:
    """Тесты конфигурации ML сервиса."""
    def test_default_config(self) -> None:
        """Тест конфигурации по умолчанию."""
        config = MLServiceConfig()
        assert config.service_url == "http://localhost:8001"
        assert config.models_dir == "./models"
        assert config.cache_dir == "./cache"
        assert config.max_models == 100
        assert config.model_timeout == 300
        assert config.prediction_timeout == 30
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.enable_auto_scaling is True
        assert config.enable_feature_engineering is True
        assert config.enable_hyperparameter_optimization is True
        assert config.enable_ensemble_learning is True
        assert config.enable_online_learning is True
        assert config.batch_size == 32
        assert config.learning_rate == 0.01
        assert config.max_iterations == 1000
        assert config.validation_split == 0.2
        assert config.test_split == 0.2
        assert config.random_state == 42
    def test_custom_config(self) -> None:
        """Тест кастомной конфигурации."""
        config = MLServiceConfig(
            service_url="http://custom:8002",
            models_dir="/custom/models",
            max_models=50,
            enable_caching=False
        )
        assert config.service_url == "http://custom:8002"
        assert config.models_dir == "/custom/models"
        assert config.max_models == 50
        assert config.enable_caching is False
class TestFeatureEngineer:
    """Тесты инженера признаков."""
    @pytest.fixture
    def feature_engineer(self) -> Any:
        """Создание экземпляра инженера признаков."""
        return FeatureEngineer()
    @pytest.fixture
    def sample_data(self) -> Any:
        """Тестовые данные."""
        dates = pd.date_range('2023-01-01', periods=100, freq='H')
        return pd.DataFrame({
            'timestamp': [int(d.timestamp() * 1000) for d in dates],
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000000, 200000, 100)
        })
    def test_create_technical_indicators(self, feature_engineer, sample_data) -> None:
        """Тест создания технических индикаторов."""
        result = feature_engineer.create_technical_indicators(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Проверяем наличие основных индикаторов
        expected_indicators = [
            'sma_5', 'sma_20', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_middle', 'bb_upper', 'bb_lower', 'bb_width',
            'volume_sma', 'volume_ratio', 'price_change',
            'volatility', 'momentum', 'rate_of_change'
        ]
        for indicator in expected_indicators:
            assert indicator in result.columns
        # Проверяем, что нет NaN в результате
        assert not result.isna().all().any()
    def test_create_advanced_features(self, feature_engineer, sample_data) -> None:
        """Тест создания продвинутых признаков."""
        result = feature_engineer.create_advanced_features(sample_data)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Проверяем наличие продвинутых признаков
        expected_features = [
            'fractal_dimension', 'price_entropy', 'volume_entropy',
            'wavelet_coeff', 'bid_ask_spread', 'price_efficiency',
            'hour', 'day_of_week', 'month',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        for feature in expected_features:
            assert feature in result.columns
    def test_calculate_fractal_dimension(self, feature_engineer) -> None:
        """Тест расчета фрактальной размерности."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = feature_engineer._calculate_fractal_dimension(series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
    def test_calculate_entropy(self, feature_engineer) -> None:
        """Тест расчета энтропии."""
        series = pd.Series([1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        result = feature_engineer._calculate_entropy(series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        assert (result >= 0).all()
    def test_calculate_wavelet_coefficient(self, feature_engineer) -> None:
        """Тест расчета вейвлет коэффициента."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = feature_engineer._calculate_wavelet_coefficient(series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
    def test_calculate_price_efficiency(self, feature_engineer) -> None:
        """Тест расчета эффективности цены."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = feature_engineer._calculate_price_efficiency(series)
        assert isinstance(result, pd.Series)
        assert len(result) == len(series)
        assert (result >= 0).all()
class TestModelManager:
    """Тесты менеджера моделей."""
    @pytest.fixture
    def model_manager(self) -> Any:
        """Создание экземпляра менеджера моделей."""
        config = MLServiceConfig()
        return ModelManager(config)
    @pytest.fixture
    def sample_model_config(self) -> Any:
        """Тестовая конфигурация модели."""
        return MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            name="test_model",
            version="1.0.0",
            hyperparameters={"n_estimators": 100}
        )
    @pytest.mark.asyncio
    async def test_create_model(self, model_manager, sample_model_config) -> None:
        """Тест создания модели."""
        model_id = await model_manager.create_model(sample_model_config)
        assert isinstance(model_id, str)
        assert len(model_id) > 0
    @pytest.mark.asyncio
    async def test_get_model(self, model_manager, sample_model_config) -> None:
        """Тест получения модели."""
        model_id = await model_manager.create_model(sample_model_config)
        model = await model_manager.get_model(model_id)
        assert model is not None
        assert model.id == model_id
        assert model.name == sample_model_config.name
    @pytest.mark.asyncio
    async def test_update_model(self, model_manager, sample_model_config) -> None:
        """Тест обновления модели."""
        model_id = await model_manager.create_model(sample_model_config)
        updates = {"status": ModelStatus.TRAINED}
        success = await model_manager.update_model(model_id, updates)
        assert success is True
        model = await model_manager.get_model(model_id)
        assert model.status == ModelStatus.TRAINED
    @pytest.mark.asyncio
    async def test_delete_model(self, model_manager, sample_model_config) -> None:
        """Тест удаления модели."""
        model_id = await model_manager.create_model(sample_model_config)
        success = await model_manager.delete_model(model_id)
        assert success is True
        model = await model_manager.get_model(model_id)
        assert model is None
    @pytest.mark.asyncio
    async def test_list_models(self, model_manager, sample_model_config) -> None:
        """Тест списка моделей."""
        # Создаем несколько моделей
        model_ids = []
        for i in range(3):
            config = MLModelConfig(
                model_type=ModelType.RANDOM_FOREST,
                name=f"test_model_{i}",
                version="1.0.0",
                hyperparameters={"n_estimators": 100}
            )
            model_id = await model_manager.create_model(config)
            model_ids.append(model_id)
        models = await model_manager.list_models()
        assert isinstance(models, list)
        assert len(models) >= 3
        model_names = [model.name for model in models]
        for i in range(3):
            assert f"test_model_{i}" in model_names
class TestProductionMLService:
    """Тесты производственного ML сервиса."""
    @pytest.fixture
    def ml_service(self) -> Any:
        """Создание экземпляра ML сервиса."""
        config = MLServiceConfig()
        return ProductionMLService(config)
    @pytest.fixture
    def sample_training_data(self) -> Any:
        """Тестовые данные для обучения."""
        return {
            "features": np.random.normal(0, 1, (100, 10)),
            "target": np.random.normal(0, 1, 100),
            "feature_names": [f"feature_{i}" for i in range(10)],
            "target_name": "price"
        }
    @pytest.fixture
    def sample_model_config(self) -> Any:
        """Тестовая конфигурация модели."""
        return MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            name="test_model",
            version="1.0.0",
            hyperparameters={"n_estimators": 100}
        )
    @pytest.mark.asyncio
    async def test_train_model(self, ml_service, sample_model_config, sample_training_data) -> None:
        """Тест обучения модели."""
        model_id = await ml_service.train_model(sample_model_config, sample_training_data)
        assert isinstance(model_id, str)
        assert len(model_id) > 0
    @pytest.mark.asyncio
    async def test_predict(self, ml_service, sample_model_config, sample_training_data) -> None:
        """Тест предсказания."""
        # Сначала обучаем модель
        model_id = await ml_service.train_model(sample_model_config, sample_training_data)
        # Создаем запрос на предсказание
        request = MLPredictionRequest(
            model_id=model_id,
            features={"feature_0": 0.5, "feature_1": 0.3},
            confidence_threshold=0.8
        )
        prediction = await ml_service.predict(request)
        assert isinstance(prediction, dict)
        assert "prediction" in prediction
        assert "confidence" in prediction
        assert "model_id" in prediction
    @pytest.mark.asyncio
    async def test_evaluate_model(self, ml_service, sample_model_config, sample_training_data) -> None:
        """Тест оценки модели."""
        # Сначала обучаем модель
        model_id = await ml_service.train_model(sample_model_config, sample_training_data)
        # Создаем тестовые данные
        test_data = {
            "features": np.random.normal(0, 1, (50, 10)),
            "target": np.random.normal(0, 1, 50)
        }
        metrics = await ml_service.evaluate_model(model_id, test_data)
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
    @pytest.mark.asyncio
    async def test_save_load_model(self, ml_service, sample_model_config, sample_training_data) -> None:
        """Тест сохранения и загрузки модели."""
        # Сначала обучаем модель
        model_id = await ml_service.train_model(sample_model_config, sample_training_data)
        # Сохраняем модель
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            success = await ml_service.save_model(model_id, model_path)
            assert success is True
            # Загружаем модель
            success = await ml_service.load_model(model_id, model_path)
            assert success is True
    @pytest.mark.asyncio
    async def test_get_model_status(self, ml_service, sample_model_config, sample_training_data) -> None:
        """Тест получения статуса модели."""
        # Сначала обучаем модель
        model_id = await ml_service.train_model(sample_model_config, sample_training_data)
        status = await ml_service.get_model_status(model_id)
        assert isinstance(status, dict)
        assert "model_id" in status
        assert "status" in status
        assert "created_at" in status
    def test_create_model_object(self, ml_service) -> None:
        """Тест создания объекта модели."""
        model_types = [
            ModelType.RANDOM_FOREST,
            ModelType.GRADIENT_BOOSTING,
            ModelType.LINEAR_REGRESSION,
            ModelType.SVR,
            ModelType.NEURAL_NETWORK
        ]
        for model_type in model_types:
            model = ml_service._create_model_object(model_type, {})
            assert model is not None
    def test_calculate_metrics(self, ml_service) -> None:
        """Тест расчета метрик."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        metrics = ml_service._calculate_metrics(y_true, y_pred)
        assert isinstance(metrics, dict)
        assert "mse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert all(isinstance(v, float) for v in metrics.values())
class TestMLServiceAdapter:
    """Тесты адаптера ML сервиса."""
    @pytest.fixture
    def adapter(self) -> Any:
        """Создание экземпляра адаптера."""
        return MLServiceAdapter("http://test:8001")
    @pytest.mark.asyncio
    async def test_train_model(self, adapter) -> None:
        """Тест обучения модели через адаптер."""
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {"model_id": "test_id"}
            mock_post.return_value.status_code = 200
            model = await adapter.train_model(
                ModelType.RANDOM_FOREST,
                {"features": [[1, 2, 3]], "target": [1]},
                {"n_estimators": 100}
            )
            assert isinstance(model, Model)
            assert model.id == "test_id"
    @pytest.mark.asyncio
    async def test_predict(self, adapter) -> None:
        """Тест предсказания через адаптер."""
        with patch('requests.post') as mock_post:
            mock_post.return_value.json.return_value = {
                "prediction": 1.5,
                "confidence": 0.9
            }
            mock_post.return_value.status_code = 200
            prediction = await adapter.predict("test_id", {"feature_1": 0.5})
            assert isinstance(prediction, dict)
            assert "prediction" in prediction
            assert "confidence" in prediction
class TestLocalMLService:
    """Тесты локального ML сервиса."""
    @pytest.fixture
    def local_service(self) -> Any:
        """Создание экземпляра локального сервиса."""
        return LocalMLService()
    @pytest.mark.asyncio
    async def test_train_model(self, local_service) -> None:
        """Тест обучения модели в локальном сервисе."""
        model = await local_service.train_model(
            ModelType.RANDOM_FOREST,
            {"features": [[1, 2, 3]], "target": [1]},
            {"n_estimators": 100}
        )
        assert isinstance(model, Model)
        assert model.id is not None
        assert model.type == ModelType.RANDOM_FOREST
    @pytest.mark.asyncio
    async def test_predict(self, local_service) -> None:
        """Тест предсказания в локальном сервисе."""
        # Сначала обучаем модель
        model = await local_service.train_model(
            ModelType.RANDOM_FOREST,
            {"features": [[1, 2, 3], [4, 5, 6]], "target": [1, 2]},
            {"n_estimators": 100}
        )
        # Делаем предсказание
        prediction = await local_service.predict(model.id, {"feature_1": 0.5})
        assert isinstance(prediction, dict)
        assert "prediction" in prediction
    @pytest.mark.asyncio
    async def test_evaluate_model(self, local_service) -> None:
        """Тест оценки модели в локальном сервисе."""
        # Сначала обучаем модель
        model = await local_service.train_model(
            ModelType.RANDOM_FOREST,
            {"features": [[1, 2, 3], [4, 5, 6]], "target": [1, 2]},
            {"n_estimators": 100}
        )
        # Оцениваем модель
        metrics = await local_service.evaluate_model(
            model.id,
            {"features": [[1, 2, 3]], "target": [1]}
        )
        assert isinstance(metrics, dict)
        assert "mse" in metrics or "accuracy" in metrics
    @pytest.mark.asyncio
    async def test_save_load_model(self, local_service) -> None:
        """Тест сохранения и загрузки модели в локальном сервисе."""
        # Сначала обучаем модель
        model = await local_service.train_model(
            ModelType.RANDOM_FOREST,
            {"features": [[1, 2, 3], [4, 5, 6]], "target": [1, 2]},
            {"n_estimators": 100}
        )
        # Сохраняем модель
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pkl")
            success = await local_service.save_model(model.id, model_path)
            assert success is True
            # Загружаем модель
            loaded_model = await local_service.load_model(model.id, model_path)
            assert isinstance(loaded_model, Model)
            assert loaded_model.id == model.id
class TestErrorHandling:
    """Тесты обработки ошибок."""
    @pytest.mark.asyncio
    async def test_invalid_model_type(self) -> None:
        """Тест обработки неверного типа модели."""
        service = ProductionMLService()
        with pytest.raises(ValueError):
            service._create_model_object("invalid_type", {})
    @pytest.mark.asyncio
    async def test_empty_training_data(self) -> None:
        """Тест обработки пустых данных обучения."""
        service = ProductionMLService()
        config = MLModelConfig(
            model_type=ModelType.RANDOM_FOREST,
            name="test",
            version="1.0.0",
            hyperparameters={}
        )
        with pytest.raises(ValueError):
            await service.train_model(config, {})
    @pytest.mark.asyncio
    async def test_invalid_prediction_request(self) -> None:
        """Тест обработки неверного запроса предсказания."""
        service = ProductionMLService()
        with pytest.raises(ValueError):
            await service.predict(None)
    @pytest.mark.asyncio
    async def test_model_not_found(self) -> None:
        """Тест обработки отсутствующей модели."""
        service = ProductionMLService()
        request = MLPredictionRequest(
            model_id="non_existent",
            features={"feature_1": 0.5},
            confidence_threshold=0.8
        )
        with pytest.raises(ValueError):
            await service.predict(request) 
