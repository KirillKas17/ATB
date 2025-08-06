"""
Unit тесты для ML сервисов.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import tempfile
import os
import pandas as pd
from shared.numpy_utils import np
from infrastructure.external_services.ml_services import (
    MLServiceConfig,
    FeatureEngineer,
    ModelManager,
    ProductionMLService,
    MLServiceAdapter,
    LocalMLService
)
from domain.entities.ml import Model, ModelType, ModelStatus
from domain.type_definitions.external_service_types import (
    MLModelConfig,
    MLPredictionRequest
)
class TestMLServiceConfig:
    """Тесты конфигурации ML сервиса."""
    def test_default_config(self: "TestMLServiceConfig") -> None:
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
    def test_custom_config(self: "TestMLServiceConfig") -> None:
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
    def feature_engineer(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра инженера признаков."""
        return FeatureEngineer()
        # Проверяем наличие основных индикаторов
        # Проверяем, что нет NaN в результате
        # Проверяем наличие продвинутых признаков
class TestModelManager:
    """Тесты менеджера моделей."""
    @pytest.fixture
    def model_manager(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра менеджера моделей."""
        config = MLServiceConfig()
        return ModelManager(config)
        # Создаем несколько моделей
class TestProductionMLService:
    """Тесты производственного ML сервиса."""
    @pytest.fixture
    def ml_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра ML сервиса."""
        config = MLServiceConfig()
        return ProductionMLService(config)
        # Сначала обучаем модель
        # Создаем запрос на предсказание
        # Сначала обучаем модель
        # Создаем тестовые данные
        # Сначала обучаем модель
        # Сохраняем модель
            # Загружаем модель
        # Сначала обучаем модель
class TestMLServiceAdapter:
    """Тесты адаптера ML сервиса."""
    @pytest.fixture
    def adapter(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра адаптера."""
        return MLServiceAdapter("http://test:8001")
class TestLocalMLService:
    """Тесты локального ML сервиса."""
    @pytest.fixture
    def local_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Создание экземпляра локального сервиса."""
        return LocalMLService()
        # Сначала обучаем модель
        # Делаем предсказание
        # Сначала обучаем модель
        # Оцениваем модель
        # Сначала обучаем модель
        # Сохраняем модель
            # Загружаем модель
class TestErrorHandling:
    """Тесты обработки ошибок."""
    @pytest.mark.asyncio
    def test_invalid_model_type(self: "TestErrorHandling") -> None:
        """Тест обработки неверного типа модели."""
        service = ProductionMLService()
        with pytest.raises(ValueError):
            service._create_model_object("invalid_type", {})
    @pytest.mark.asyncio
    def test_empty_training_data(self: "TestErrorHandling") -> None:
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
    def test_invalid_prediction_request(self: "TestErrorHandling") -> None:
        """Тест обработки неверного запроса предсказания."""
        service = ProductionMLService()
        with pytest.raises(ValueError):
            await service.predict(None)
    @pytest.mark.asyncio
    def test_model_not_found(self: "TestErrorHandling") -> None:
        """Тест обработки отсутствующей модели."""
        service = ProductionMLService()
        request = MLPredictionRequest(
            model_id="non_existent",
            features={"feature_1": 0.5},
            confidence_threshold=0.8
        )
        with pytest.raises(ValueError):
            await service.predict(request) 
