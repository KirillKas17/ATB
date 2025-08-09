"""
Unit тесты для ModelManager
"""

import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from sklearn.preprocessing import StandardScaler
from domain.entities.ml import Model, ModelStatus
from domain.type_definitions import ModelId
from domain.type_definitions.external_service_types import MLModelConfig, MLModelType, PredictionType
from infrastructure.external_services.ml.config import MLServiceConfig
from infrastructure.external_services.ml.model_manager import ModelManager


class TestModelManager:
    """Тесты для ModelManager."""

    @pytest.fixture
    def config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Конфигурация для тестов."""
        return MLServiceConfig(
            models_dir="/tmp/test_models",
            cache_dir="/tmp/test_cache",
            max_models=10,
        )

    @pytest.fixture
    def model_manager(self, config) -> Any:
        """Экземпляр ModelManager."""
        return ModelManager(config)

    @pytest.fixture
    def sample_model_config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Пример конфигурации модели."""
        return MLModelConfig(
            name="test_model",
            model_type=MLModelType.RANDOM_FOREST,
            trading_pair="BTC/USD",
            prediction_type=PredictionType.PRICE,
            hyperparameters={"n_estimators": 100},
            features=["price", "volume"],
            target="direction",
            description="Test model",
        )

    @pytest.fixture
    def sample_model(self, sample_model_config) -> Any:
        """Пример модели."""
        return Model(
            id=ModelId("1"),
            name=sample_model_config["name"],
            model_type=sample_model_config["model_type"].value,
            trading_pair=str(sample_model_config["trading_pair"]),
            prediction_type=sample_model_config["prediction_type"].value,
            hyperparameters=sample_model_config["hyperparameters"],
            features=sample_model_config["features"],
            target=sample_model_config["target"],
            description=sample_model_config["description"],
            status=ModelStatusType.CREATED,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_init_creates_directories(self, config, tmp_path) -> None:
        """Тест создания директорий при инициализации."""
        config.models_dir = str(tmp_path / "models")
        config.cache_dir = str(tmp_path / "cache")
        manager = ModelManager(config)
        assert Path(config.models_dir).exists()
        assert Path(config.cache_dir).exists()

    @pytest.mark.asyncio
    async def test_create_model_success(self, model_manager, sample_model_config) -> None:
        """Тест успешного создания модели."""
        model_id = await model_manager.create_model(sample_model_config)
        assert model_id is not None
        assert str(model_id) == "1"
        model = await model_manager.get_model(model_id)
        assert model is not None
        assert model.name == sample_model_config["name"]
        assert model.model_type == sample_model_config["model_type"].value
        assert model.trading_pair == str(sample_model_config["trading_pair"])
        assert model.status == ModelStatusType.CREATED

    @pytest.mark.asyncio
    async def test_create_model_increments_id(self, model_manager, sample_model_config) -> None:
        """Тест инкремента ID при создании моделей."""
        model_id1 = await model_manager.create_model(sample_model_config)
        sample_model_config["name"] = "test_model_2"
        model_id2 = await model_manager.create_model(sample_model_config)
        assert str(model_id1) == "1"
        assert str(model_id2) == "2"

    @pytest.mark.asyncio
    async def test_get_model_exists(self, model_manager, sample_model_config) -> None:
        """Тест получения существующей модели."""
        model_id = await model_manager.create_model(sample_model_config)
        model = await model_manager.get_model(model_id)
        assert model is not None
        assert model.id == model_id

    @pytest.mark.asyncio
    async def test_get_model_not_exists(self, model_manager) -> None:
        """Тест получения несуществующей модели."""
        model = await model_manager.get_model(ModelId("999"))
        assert model is None

    @pytest.mark.asyncio
    async def test_update_model_success(self, model_manager, sample_model_config) -> None:
        """Тест успешного обновления модели."""
        model_id = await model_manager.create_model(sample_model_config)
        updates = {"name": "updated_name", "description": "updated_description"}
        result = await model_manager.update_model(model_id, updates)
        assert result is True
        model = await model_manager.get_model(model_id)
        assert model.name == "updated_name"
        assert model.description == "updated_description"
        assert model.updated_at > model.created_at

    @pytest.mark.asyncio
    async def test_update_model_not_exists(self, model_manager) -> None:
        """Тест обновления несуществующей модели."""
        updates = {"name": "updated_name"}
        result = await model_manager.update_model(ModelId("999"), updates)
        assert result is False

    @pytest.mark.asyncio
    async def test_update_model_invalid_field(self, model_manager, sample_model_config) -> None:
        """Тест обновления с невалидным полем."""
        model_id = await model_manager.create_model(sample_model_config)
        updates = {"invalid_field": "value"}
        result = await model_manager.update_model(model_id, updates)
        assert result is True  # Невалидные поля игнорируются

    @pytest.mark.asyncio
    async def test_delete_model_success(self, model_manager, sample_model_config) -> None:
        """Тест успешного удаления модели."""
        model_id = await model_manager.create_model(sample_model_config)
        result = await model_manager.delete_model(model_id)
        assert result is True
        model = await model_manager.get_model(model_id)
        assert model is None

    @pytest.mark.asyncio
    async def test_delete_model_not_exists(self, model_manager) -> None:
        """Тест удаления несуществующей модели."""
        result = await model_manager.delete_model(ModelId("999"))
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_model_removes_files(self, model_manager, sample_model_config, tmp_path) -> None:
        """Тест удаления файлов модели."""
        model_manager.config.models_dir = str(tmp_path / "models")
        Path(model_manager.config.models_dir).mkdir(parents=True, exist_ok=True)
        model_id = await model_manager.create_model(sample_model_config)
        # Создаем файл модели
        model_path = Path(model_manager.config.models_dir) / f"{model_id}.pkl"
        model_path.write_text("test")
        result = await model_manager.delete_model(model_id)
        assert result is True
        assert not model_path.exists()

    @pytest.mark.asyncio
    async def test_list_models_empty(self, model_manager) -> None:
        """Тест списка моделей (пустой)."""
        models = await model_manager.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_list_models_with_models(self, model_manager, sample_model_config) -> None:
        """Тест списка моделей с моделями."""
        await model_manager.create_model(sample_model_config)
        sample_model_config["name"] = "test_model_2"
        await model_manager.create_model(sample_model_config)
        models = await model_manager.list_models()
        assert len(models) == 2
        assert all(isinstance(model, Model) for model in models)

    @pytest.mark.asyncio
    async def test_get_model_by_name_exists(self, model_manager, sample_model_config) -> None:
        """Тест получения модели по имени (существует)."""
        await model_manager.create_model(sample_model_config)
        model = await model_manager.get_model_by_name("test_model")
        assert model is not None
        assert model.name == "test_model"

    @pytest.mark.asyncio
    async def test_get_model_by_name_not_exists(self, model_manager) -> None:
        """Тест получения модели по имени (не существует)."""
        model = await model_manager.get_model_by_name("nonexistent")
        assert model is None

    @pytest.mark.asyncio
    async def test_get_models_by_status(self, model_manager, sample_model_config) -> None:
        """Тест получения моделей по статусу."""
        await model_manager.create_model(sample_model_config)
        sample_model_config["name"] = "test_model_2"
        await model_manager.create_model(sample_model_config)
        # Обновляем статус одной модели
        model_id = ModelId("1")
        await model_manager.update_model(model_id, {"status": ModelStatus.TRAINED})
        created_models = await model_manager.get_models_by_status(ModelStatusType.CREATED)
        trained_models = await model_manager.get_models_by_status(ModelStatus.TRAINED)
        assert len(created_models) == 1
        assert len(trained_models) == 1
        assert created_models[0].status == ModelStatusType.CREATED
        assert trained_models[0].status == ModelStatus.TRAINED

    @pytest.mark.asyncio
    async def test_get_models_by_trading_pair(self, model_manager, sample_model_config) -> None:
        """Тест получения моделей по торговой паре."""
        await model_manager.create_model(sample_model_config)
        sample_model_config["name"] = "test_model_2"
        sample_model_config["trading_pair"] = "ETH/USD"
        await model_manager.create_model(sample_model_config)
        btc_models = await model_manager.get_models_by_trading_pair("BTC/USD")
        eth_models = await model_manager.get_models_by_trading_pair("ETH/USD")
        assert len(btc_models) == 1
        assert len(eth_models) == 1
        assert btc_models[0].trading_pair == "BTC/USD"
        assert eth_models[0].trading_pair == "ETH/USD"

    @pytest.mark.asyncio
    async def test_get_model_metrics_exists(self, model_manager, sample_model_config) -> None:
        """Тест получения метрик модели (существует)."""
        model_id = await model_manager.create_model(sample_model_config)
        metrics = await model_manager.get_model_metrics(model_id)
        assert metrics["id"] == "1"
        assert metrics["name"] == "test_model"
        assert metrics["status"] == ModelStatusType.CREATED.value
        assert metrics["accuracy"] == 0.0
        assert metrics["features_count"] == 2
        assert metrics["trading_pair"] == "BTC/USD"
        assert metrics["prediction_type"] == PredictionType.PRICE.value
        assert "created_at" in metrics
        assert "updated_at" in metrics

    @pytest.mark.asyncio
    async def test_get_model_metrics_not_exists(self, model_manager) -> None:
        """Тест получения метрик модели (не существует)."""
        metrics = await model_manager.get_model_metrics(ModelId("999"))
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_backup_model_success(self, model_manager, sample_model_config, tmp_path) -> None:
        """Тест успешного резервного копирования."""
        model_id = await model_manager.create_model(sample_model_config)
        # Добавляем объект модели
        model_manager.model_objects[model_id] = MagicMock()
        model_manager.feature_engineers[model_id].scaler = StandardScaler()
        model_manager.feature_engineers[model_id].feature_names = ["feature1", "feature2"]
        backup_path = str(tmp_path / "backup")
        result = await model_manager.backup_model(model_id, backup_path)
        assert result is True
        backup_file = Path(backup_path) / f"{model_id}_backup.pkl"
        assert backup_file.exists()
        # Проверяем содержимое backup
        with open(backup_file, "rb") as f:
            backup_data = pickle.load(f)
        assert "model" in backup_data
        assert "scaler" in backup_data
        assert "feature_names" in backup_data
        assert "model_info" in backup_data
        assert "backup_timestamp" in backup_data

    @pytest.mark.asyncio
    async def test_backup_model_not_exists(self, model_manager, tmp_path) -> None:
        """Тест резервного копирования несуществующей модели."""
        backup_path = str(tmp_path / "backup")
        result = await model_manager.backup_model(ModelId("999"), backup_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_backup_model_no_model_object(self, model_manager, sample_model_config, tmp_path) -> None:
        """Тест резервного копирования без объекта модели."""
        model_id = await model_manager.create_model(sample_model_config)
        backup_path = str(tmp_path / "backup")
        result = await model_manager.backup_model(model_id, backup_path)
        assert result is False

    @pytest.mark.asyncio
    async def test_restore_model_success(self, model_manager, sample_model, tmp_path) -> None:
        """Тест успешного восстановления модели."""
        # Создаем backup файл
        backup_data = {
            "model": MagicMock(),
            "scaler": StandardScaler(),
            "feature_names": ["feature1", "feature2"],
            "model_info": sample_model,
            "backup_timestamp": datetime.now().isoformat(),
        }
        backup_file = tmp_path / "test_backup.pkl"
        with open(backup_file, "wb") as f:
            pickle.dump(backup_data, f)
        model_id = await model_manager.restore_model(str(backup_file))
        assert model_id is not None
        assert str(model_id) == "1"
        model = await model_manager.get_model(model_id)
        assert model is not None
        assert model.name == sample_model.name

    @pytest.mark.asyncio
    async def test_restore_model_file_not_exists(self, model_manager) -> None:
        """Тест восстановления из несуществующего файла."""
        model_id = await model_manager.restore_model("/nonexistent/path")
        assert model_id is None

    @pytest.mark.asyncio
    async def test_restore_model_invalid_file(self, model_manager, tmp_path) -> None:
        """Тест восстановления из невалидного файла."""
        backup_file = tmp_path / "invalid_backup.pkl"
        backup_file.write_text("invalid data")
        model_id = await model_manager.restore_model(str(backup_file))
        assert model_id is None

    @pytest.mark.asyncio
    async def test_get_model_size_exists(self, model_manager, sample_model_config, tmp_path) -> None:
        """Тест получения размера модели (файл существует)."""
        model_manager.config.models_dir = str(tmp_path / "models")
        Path(model_manager.config.models_dir).mkdir(parents=True, exist_ok=True)
        model_id = await model_manager.create_model(sample_model_config)
        # Создаем файл модели
        model_path = Path(model_manager.config.models_dir) / f"{model_id}.pkl"
        model_path.write_text("test content")
        size = await model_manager.get_model_size(model_id)
        assert size > 0

    @pytest.mark.asyncio
    async def test_get_model_size_not_exists(self, model_manager) -> None:
        """Тест получения размера модели (файл не существует)."""
        size = await model_manager.get_model_size(ModelId("999"))
        assert size == 0

    @pytest.mark.asyncio
    async def test_cleanup_old_models(self, model_manager, sample_model_config) -> None:
        """Тест очистки старых моделей."""
        # Создаем старую модель
        model_id1 = await model_manager.create_model(sample_model_config)
        model = await model_manager.get_model(model_id1)
        model.created_at = datetime.now() - timedelta(days=31)
        model.updated_at = datetime.now() - timedelta(days=31)
        # Создаем новую модель
        sample_model_config["name"] = "new_model"
        model_id2 = await model_manager.create_model(sample_model_config)
        # Создаем обученную старую модель (не должна удаляться)
        sample_model_config["name"] = "trained_old_model"
        model_id3 = await model_manager.create_model(sample_model_config)
        await model_manager.update_model(model_id3, {"status": ModelStatus.TRAINED})
        model = await model_manager.get_model(model_id3)
        model.created_at = datetime.now() - timedelta(days=31)
        model.updated_at = datetime.now() - timedelta(days=31)
        deleted_count = await model_manager.cleanup_old_models(max_age_days=30)
        assert deleted_count == 1
        # Проверяем, что старая модель удалена
        assert await model_manager.get_model(model_id1) is None
        # Проверяем, что новая модель осталась
        assert await model_manager.get_model(model_id2) is not None
        # Проверяем, что обученная старая модель осталась
        assert await model_manager.get_model(model_id3) is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, model_manager, sample_model_config) -> None:
        """Тест конкурентных операций."""
        # Создаем несколько моделей одновременно
        tasks = []
        for i in range(5):
            config = MLModelConfig(
                name=f"model_{i}",
                model_type=MLModelType.RANDOM_FOREST,
                trading_pair="BTC/USD",
                prediction_type=PredictionType.PRICE,
                hyperparameters={"n_estimators": 100},
                features=["price", "volume"],
                target="direction",
                description=f"Test model {i}",
            )
            tasks.append(model_manager.create_model(config))
        model_ids = await asyncio.gather(*tasks)
        assert len(model_ids) == 5
        assert len(set(str(mid) for mid in model_ids)) == 5  # Все ID уникальны
        # Проверяем, что все модели созданы
        models = await model_manager.list_models()
        assert len(models) == 5

    @pytest.mark.asyncio
    async def test_model_manager_thread_safety(self, model_manager, sample_model_config) -> None:
        """Тест потокобезопасности."""
        # Создаем модель
        model_id = await model_manager.create_model(sample_model_config)

        # Выполняем конкурентные операции
        async def update_model() -> Any:
            return await model_manager.update_model(model_id, {"name": "updated"})

        async def get_model() -> Any:
            return await model_manager.get_model(model_id)

        async def delete_model() -> Any:
            return await model_manager.delete_model(model_id)

        # Запускаем операции одновременно
        results = await asyncio.gather(update_model(), get_model(), delete_model(), return_exceptions=True)
        # Проверяем, что операции завершились без исключений
        assert all(not isinstance(result, Exception) for result in results)
