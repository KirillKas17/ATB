"""
Unit тесты для MLRepository.

Покрывает:
- Основной функционал репозитория ML моделей
- CRUD операции для моделей
- Управление предсказаниями
- Фильтрацию и поиск
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from uuid import uuid4

from domain.entities.ml import Model, ModelType, Prediction
from domain.repositories.ml_repository import MLRepository, InMemoryMLRepository
from domain.type_definitions import EntityId
from domain.exceptions.base_exceptions import ValidationError


class TestMLRepository:
    """Тесты для абстрактного MLRepository."""

    @pytest.fixture
    def mock_ml_repository(self) -> Mock:
        """Мок репозитория ML моделей."""
        return Mock(spec=MLRepository)

    @pytest.fixture
    def sample_model(self) -> Model:
        """Тестовая ML модель."""
        return Model(
            id=uuid4(),
            name="Test Model",
            model_type=ModelType.REGRESSION,
            version="1.0.0",
            trading_pair="BTC/USDT",
            parameters={"learning_rate": 0.01, "epochs": 100},
            is_active=True,
        )

    @pytest.fixture
    def sample_prediction(self, sample_model) -> Prediction:
        """Тестовое предсказание."""
        return Prediction(
            id=uuid4(),
            model_id=sample_model.id,
            trading_pair="BTC/USDT",
            predicted_value=50000.0,
            confidence=0.85,
            timestamp="2023-01-01T00:00:00Z",
            features={"price": 49000.0, "volume": 1000000.0},
        )

    def test_save_model_method_exists(self, mock_ml_repository, sample_model):
        """Тест наличия метода save_model."""
        mock_ml_repository.save_model = AsyncMock(return_value=sample_model)
        assert hasattr(mock_ml_repository, "save_model")
        assert callable(mock_ml_repository.save_model)

    def test_get_model_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_model."""
        mock_ml_repository.get_model = AsyncMock(return_value=None)
        assert hasattr(mock_ml_repository, "get_model")
        assert callable(mock_ml_repository.get_model)

    def test_get_all_models_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_all_models."""
        mock_ml_repository.get_all_models = AsyncMock(return_value=[])
        assert hasattr(mock_ml_repository, "get_all_models")
        assert callable(mock_ml_repository.get_all_models)

    def test_delete_model_method_exists(self, mock_ml_repository):
        """Тест наличия метода delete_model."""
        mock_ml_repository.delete_model = AsyncMock(return_value=True)
        assert hasattr(mock_ml_repository, "delete_model")
        assert callable(mock_ml_repository.delete_model)

    def test_get_models_by_type_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_models_by_type."""
        mock_ml_repository.get_models_by_type = AsyncMock(return_value=[])
        assert hasattr(mock_ml_repository, "get_models_by_type")
        assert callable(mock_ml_repository.get_models_by_type)

    def test_get_models_by_trading_pair_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_models_by_trading_pair."""
        mock_ml_repository.get_models_by_trading_pair = AsyncMock(return_value=[])
        assert hasattr(mock_ml_repository, "get_models_by_trading_pair")
        assert callable(mock_ml_repository.get_models_by_trading_pair)

    def test_get_active_models_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_active_models."""
        mock_ml_repository.get_active_models = AsyncMock(return_value=[])
        assert hasattr(mock_ml_repository, "get_active_models")
        assert callable(mock_ml_repository.get_active_models)

    def test_save_prediction_method_exists(self, mock_ml_repository, sample_prediction):
        """Тест наличия метода save_prediction."""
        mock_ml_repository.save_prediction = AsyncMock(return_value=sample_prediction)
        assert hasattr(mock_ml_repository, "save_prediction")
        assert callable(mock_ml_repository.save_prediction)

    def test_get_predictions_by_model_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_predictions_by_model."""
        mock_ml_repository.get_predictions_by_model = AsyncMock(return_value=[])
        assert hasattr(mock_ml_repository, "get_predictions_by_model")
        assert callable(mock_ml_repository.get_predictions_by_model)

    def test_get_latest_prediction_method_exists(self, mock_ml_repository):
        """Тест наличия метода get_latest_prediction."""
        mock_ml_repository.get_latest_prediction = AsyncMock(return_value=None)
        assert hasattr(mock_ml_repository, "get_latest_prediction")
        assert callable(mock_ml_repository.get_latest_prediction)


class TestInMemoryMLRepository:
    """Тесты для InMemoryMLRepository."""

    @pytest.fixture
    def repository(self) -> InMemoryMLRepository:
        """Экземпляр репозитория."""
        return InMemoryMLRepository()

    @pytest.fixture
    def sample_model(self) -> Model:
        """Тестовая ML модель."""
        return Model(
            id=uuid4(),
            name="Test Model",
            model_type=ModelType.REGRESSION,
            version="1.0.0",
            trading_pair="BTC/USDT",
            parameters={"learning_rate": 0.01, "epochs": 100},
            is_active=True,
        )

    @pytest.fixture
    def sample_models(self) -> List[Model]:
        """Список тестовых ML моделей."""
        return [
            Model(
                id=uuid4(),
                name="Model 1",
                model_type=ModelType.REGRESSION,
                version="1.0.0",
                trading_pair="BTC/USDT",
                is_active=True,
            ),
            Model(
                id=uuid4(),
                name="Model 2",
                model_type=ModelType.CLASSIFICATION,
                version="2.0.0",
                trading_pair="ETH/USDT",
                is_active=False,
            ),
            Model(
                id=uuid4(),
                name="Model 3",
                model_type=ModelType.REGRESSION,
                version="1.5.0",
                trading_pair="BTC/USDT",
                is_active=True,
            ),
        ]

    @pytest.fixture
    def sample_prediction(self, sample_model) -> Prediction:
        """Тестовое предсказание."""
        return Prediction(
            id=uuid4(),
            model_id=sample_model.id,
            trading_pair="BTC/USDT",
            predicted_value=50000.0,
            confidence=0.85,
            timestamp="2023-01-01T00:00:00Z",
            features={"price": 49000.0, "volume": 1000000.0},
        )

    @pytest.fixture
    def sample_predictions(self, sample_model) -> List[Prediction]:
        """Список тестовых предсказаний."""
        return [
            Prediction(
                id=uuid4(),
                model_id=sample_model.id,
                trading_pair="BTC/USDT",
                predicted_value=50000.0,
                confidence=0.85,
                timestamp="2023-01-01T00:00:00Z",
                features={"price": 49000.0, "volume": 1000000.0},
            ),
            Prediction(
                id=uuid4(),
                model_id=sample_model.id,
                trading_pair="BTC/USDT",
                predicted_value=51000.0,
                confidence=0.90,
                timestamp="2023-01-01T01:00:00Z",
                features={"price": 50000.0, "volume": 1100000.0},
            ),
            Prediction(
                id=uuid4(),
                model_id=sample_model.id,
                trading_pair="BTC/USDT",
                predicted_value=52000.0,
                confidence=0.88,
                timestamp="2023-01-01T02:00:00Z",
                features={"price": 51000.0, "volume": 1200000.0},
            ),
        ]

    @pytest.mark.asyncio
    async def test_save_model(self, repository, sample_model):
        """Тест сохранения модели."""
        saved_model = await repository.save_model(sample_model)

        assert saved_model == sample_model
        assert EntityId(sample_model.id) in repository._models
        assert repository._models[EntityId(sample_model.id)] == sample_model

    @pytest.mark.asyncio
    async def test_get_model_existing(self, repository, sample_model):
        """Тест получения существующей модели по ID."""
        await repository.save_model(sample_model)

        retrieved_model = await repository.get_model(EntityId(sample_model.id))

        assert retrieved_model == sample_model

    @pytest.mark.asyncio
    async def test_get_model_not_existing(self, repository):
        """Тест получения несуществующей модели по ID."""
        model_id = EntityId(uuid4())
        retrieved_model = await repository.get_model(model_id)

        assert retrieved_model is None

    @pytest.mark.asyncio
    async def test_get_all_models_empty(self, repository):
        """Тест получения всех моделей из пустого репозитория."""
        models = await repository.get_all_models()

        assert models == []

    @pytest.mark.asyncio
    async def test_get_all_models_with_models(self, repository, sample_models):
        """Тест получения всех моделей."""
        for model in sample_models:
            await repository.save_model(model)

        models = await repository.get_all_models()

        assert len(models) == 3
        assert all(model in models for model in sample_models)

    @pytest.mark.asyncio
    async def test_delete_model_existing(self, repository, sample_model):
        """Тест удаления существующей модели."""
        await repository.save_model(sample_model)

        result = await repository.delete_model(EntityId(sample_model.id))

        assert result is True
        assert EntityId(sample_model.id) not in repository._models

    @pytest.mark.asyncio
    async def test_delete_model_not_existing(self, repository):
        """Тест удаления несуществующей модели."""
        model_id = EntityId(uuid4())
        result = await repository.delete_model(model_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_get_models_by_type(self, repository, sample_models):
        """Тест получения моделей по типу."""
        for model in sample_models:
            await repository.save_model(model)

        regression_models = await repository.get_models_by_type(ModelType.REGRESSION)
        classification_models = await repository.get_models_by_type(ModelType.CLASSIFICATION)

        assert len(regression_models) == 2
        assert len(classification_models) == 1
        assert all(m.model_type == ModelType.REGRESSION for m in regression_models)
        assert all(m.model_type == ModelType.CLASSIFICATION for m in classification_models)

    @pytest.mark.asyncio
    async def test_get_models_by_trading_pair(self, repository, sample_models):
        """Тест получения моделей по торговой паре."""
        for model in sample_models:
            await repository.save_model(model)

        btc_models = await repository.get_models_by_trading_pair("BTC/USDT")
        eth_models = await repository.get_models_by_trading_pair("ETH/USDT")

        assert len(btc_models) == 2
        assert len(eth_models) == 1
        assert all(getattr(m, "trading_pair", None) == "BTC/USDT" for m in btc_models)
        assert all(getattr(m, "trading_pair", None) == "ETH/USDT" for m in eth_models)

    @pytest.mark.asyncio
    async def test_get_active_models(self, repository, sample_models):
        """Тест получения активных моделей."""
        for model in sample_models:
            await repository.save_model(model)

        active_models = await repository.get_active_models()

        assert len(active_models) == 2
        assert all(getattr(m, "is_active", False) for m in active_models)

    @pytest.mark.asyncio
    async def test_save_prediction(self, repository, sample_model, sample_prediction):
        """Тест сохранения предсказания."""
        await repository.save_model(sample_model)

        saved_prediction = await repository.save_prediction(sample_prediction)

        assert saved_prediction == sample_prediction
        assert EntityId(sample_model.id) in repository._predictions
        assert sample_prediction in repository._predictions[EntityId(sample_model.id)]

    @pytest.mark.asyncio
    async def test_save_multiple_predictions(self, repository, sample_model, sample_predictions):
        """Тест сохранения нескольких предсказаний."""
        await repository.save_model(sample_model)

        for prediction in sample_predictions:
            await repository.save_prediction(prediction)

        model_predictions = repository._predictions[EntityId(sample_model.id)]
        assert len(model_predictions) == 3
        assert all(prediction in model_predictions for prediction in sample_predictions)

    @pytest.mark.asyncio
    async def test_get_predictions_by_model(self, repository, sample_model, sample_predictions):
        """Тест получения предсказаний модели."""
        await repository.save_model(sample_model)

        for prediction in sample_predictions:
            await repository.save_prediction(prediction)

        predictions = await repository.get_predictions_by_model(EntityId(sample_model.id))

        assert len(predictions) == 3
        assert all(prediction in predictions for prediction in sample_predictions)

    @pytest.mark.asyncio
    async def test_get_predictions_by_model_with_limit(self, repository, sample_model, sample_predictions):
        """Тест получения предсказаний модели с лимитом."""
        await repository.save_model(sample_model)

        for prediction in sample_predictions:
            await repository.save_prediction(prediction)

        predictions = await repository.get_predictions_by_model(EntityId(sample_model.id), limit=2)

        assert len(predictions) == 2
        assert predictions == sample_predictions[:2]

    @pytest.mark.asyncio
    async def test_get_predictions_by_model_no_predictions(self, repository, sample_model):
        """Тест получения предсказаний модели без предсказаний."""
        await repository.save_model(sample_model)

        predictions = await repository.get_predictions_by_model(EntityId(sample_model.id))

        assert predictions == []

    @pytest.mark.asyncio
    async def test_get_latest_prediction_with_predictions(self, repository, sample_model, sample_predictions):
        """Тест получения последнего предсказания."""
        await repository.save_model(sample_model)

        for prediction in sample_predictions:
            await repository.save_prediction(prediction)

        latest_prediction = await repository.get_latest_prediction(EntityId(sample_model.id))

        assert latest_prediction == sample_predictions[-1]

    @pytest.mark.asyncio
    async def test_get_latest_prediction_no_predictions(self, repository, sample_model):
        """Тест получения последнего предсказания без предсказаний."""
        await repository.save_model(sample_model)

        latest_prediction = await repository.get_latest_prediction(EntityId(sample_model.id))

        assert latest_prediction is None

    @pytest.mark.asyncio
    async def test_get_latest_prediction_not_existing_model(self, repository):
        """Тест получения последнего предсказания несуществующей модели."""
        model_id = EntityId(uuid4())

        latest_prediction = await repository.get_latest_prediction(model_id)

        assert latest_prediction is None

    @pytest.mark.asyncio
    async def test_save_method_compatibility(self, repository, sample_model):
        """Тест совместимости метода save."""
        saved_model = await repository.save(sample_model)

        assert saved_model == sample_model
        assert EntityId(sample_model.id) in repository._models

    @pytest.mark.asyncio
    async def test_get_by_id_method_compatibility(self, repository, sample_model):
        """Тест совместимости метода get_by_id."""
        await repository.save_model(sample_model)

        retrieved_model = await repository.get_by_id(EntityId(sample_model.id))

        assert retrieved_model == sample_model

    @pytest.mark.asyncio
    async def test_update_method_compatibility(self, repository, sample_model):
        """Тест совместимости метода update."""
        await repository.save_model(sample_model)

        # Изменяем модель
        sample_model.name = "Updated Model"
        updated_model = await repository.update(sample_model)

        assert updated_model == sample_model
        assert repository._models[EntityId(sample_model.id)].name == "Updated Model"

    @pytest.mark.asyncio
    async def test_delete_method_compatibility(self, repository, sample_model):
        """Тест совместимости метода delete."""
        await repository.save_model(sample_model)

        result = await repository.delete(EntityId(sample_model.id))

        assert result is True
        assert EntityId(sample_model.id) not in repository._models

    @pytest.mark.asyncio
    async def test_delete_method_compatibility_not_existing(self, repository):
        """Тест совместимости метода delete для несуществующей модели."""
        model_id = EntityId(uuid4())

        result = await repository.delete(model_id)

        assert result is False

    @pytest.mark.asyncio
    async def test_repository_isolation(self: "TestInMemoryMLRepository") -> None:
        """Тест изоляции между экземплярами репозитория."""
        repo1 = InMemoryMLRepository()
        repo2 = InMemoryMLRepository()

        model1 = Model(
            id=uuid4(),
            name="Model 1",
            model_type=ModelType.REGRESSION,
            version="1.0.0",
            trading_pair="BTC/USDT",
            is_active=True,
        )

        model2 = Model(
            id=uuid4(),
            name="Model 2",
            model_type=ModelType.CLASSIFICATION,
            version="2.0.0",
            trading_pair="ETH/USDT",
            is_active=False,
        )

        await repo1.save_model(model1)
        await repo2.save_model(model2)

        assert len(await repo1.get_all_models()) == 1
        assert len(await repo2.get_all_models()) == 1
        assert await repo1.get_model(EntityId(model1.id)) == model1
        assert await repo2.get_model(EntityId(model2.id)) == model2
        assert await repo1.get_model(EntityId(model2.id)) is None
        assert await repo2.get_model(EntityId(model1.id)) is None

    @pytest.mark.asyncio
    async def test_prediction_isolation_between_models(self, repository, sample_models):
        """Тест изоляции предсказаний между моделями."""
        for model in sample_models:
            await repository.save_model(model)

        # Создаем предсказания для разных моделей
        prediction1 = Prediction(
            id=uuid4(),
            model_id=sample_models[0].id,
            trading_pair="BTC/USDT",
            predicted_value=50000.0,
            confidence=0.85,
            timestamp="2023-01-01T00:00:00Z",
            features={"price": 49000.0},
        )

        prediction2 = Prediction(
            id=uuid4(),
            model_id=sample_models[1].id,
            trading_pair="ETH/USDT",
            predicted_value=3000.0,
            confidence=0.90,
            timestamp="2023-01-01T00:00:00Z",
            features={"price": 2900.0},
        )

        await repository.save_prediction(prediction1)
        await repository.save_prediction(prediction2)

        # Проверяем изоляцию
        model1_predictions = await repository.get_predictions_by_model(EntityId(sample_models[0].id))
        model2_predictions = await repository.get_predictions_by_model(EntityId(sample_models[1].id))

        assert len(model1_predictions) == 1
        assert len(model2_predictions) == 1
        assert model1_predictions[0] == prediction1
        assert model2_predictions[0] == prediction2

    @pytest.mark.asyncio
    async def test_model_parameters_persistence(self, repository):
        """Тест сохранения параметров модели."""
        model = Model(
            id=uuid4(),
            name="Test Model",
            model_type=ModelType.REGRESSION,
            version="1.0.0",
            trading_pair="BTC/USDT",
            parameters={"learning_rate": 0.01, "epochs": 100, "batch_size": 32},
            is_active=True,
        )

        await repository.save_model(model)
        retrieved_model = await repository.get_model(EntityId(model.id))

        assert retrieved_model is not None
        assert retrieved_model.parameters == {"learning_rate": 0.01, "epochs": 100, "batch_size": 32}

    @pytest.mark.asyncio
    async def test_prediction_features_persistence(self, repository, sample_model):
        """Тест сохранения признаков предсказания."""
        await repository.save_model(sample_model)

        prediction = Prediction(
            id=uuid4(),
            model_id=sample_model.id,
            trading_pair="BTC/USDT",
            predicted_value=50000.0,
            confidence=0.85,
            timestamp="2023-01-01T00:00:00Z",
            features={"price": 49000.0, "volume": 1000000.0, "rsi": 65.5, "macd": 0.002},
        )

        await repository.save_prediction(prediction)
        model_predictions = await repository.get_predictions_by_model(EntityId(sample_model.id))

        assert len(model_predictions) == 1
        saved_prediction = model_predictions[0]
        assert saved_prediction.features == {"price": 49000.0, "volume": 1000000.0, "rsi": 65.5, "macd": 0.002}

    @pytest.mark.asyncio
    async def test_model_versioning(self, repository):
        """Тест версионирования моделей."""
        model_v1 = Model(
            id=uuid4(),
            name="Test Model",
            model_type=ModelType.REGRESSION,
            version="1.0.0",
            trading_pair="BTC/USDT",
            is_active=True,
        )

        model_v2 = Model(
            id=uuid4(),
            name="Test Model",
            model_type=ModelType.REGRESSION,
            version="2.0.0",
            trading_pair="BTC/USDT",
            is_active=True,
        )

        await repository.save_model(model_v1)
        await repository.save_model(model_v2)

        all_models = await repository.get_all_models()
        assert len(all_models) == 2

        # Проверяем, что версии сохранились
        versions = [model.version for model in all_models]
        assert "1.0.0" in versions
        assert "2.0.0" in versions
