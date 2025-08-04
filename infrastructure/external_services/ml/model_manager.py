"""
Менеджер моделей машинного обучения.
"""

import asyncio
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger

from domain.entities.ml import Model, ModelStatus, ModelType, PredictionType
from domain.type_definitions import ModelId
from domain.type_definitions.external_service_types import (
    FeatureName,
    MLModelConfig,
    ModelName,
    ModelPath,
    ModelVersion,
    TargetName,
)
from domain.type_definitions.protocol_types import ModelStatusType
from infrastructure.external_services.ml_services import (
    FeatureEngineer,
    MLServiceConfig,
)


class ModelManager:
    """Менеджер моделей машинного обучения."""

    def __init__(self, config: MLServiceConfig):
        self.config = config
        self.models: Dict[ModelId, Model] = {}
        self.model_objects: Dict[ModelId, Any] = {}
        self.scalers: Dict[ModelId, Any] = {}
        self.feature_engineers: Dict[ModelId, FeatureEngineer] = {}
        # Исправление: используем asyncio.Lock вместо threading.Lock для асинхронных операций
        self.lock = asyncio.Lock()
        # Создаем директории
        Path(config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)

    async def create_model(self, config: MLModelConfig) -> ModelId:
        """Создание новой модели."""
        model_id = ModelId(UUID(str(len(self.models) + 1)))
        model = Model(
            id=model_id,
            name=config["name"],
            model_type=ModelType(config["model_type"].value),
            trading_pair=str(config["trading_pair"]),
            prediction_type=PredictionType(config["prediction_type"].value),
            hyperparameters=config["hyperparameters"],
            features=list(config["features"]),
            target=config["target"],
            description=config["description"],
            status=ModelStatus.INACTIVE,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        # Исправление: теперь self.lock - это asyncio.Lock, который поддерживает async with
        async with self.lock:
            self.models[model_id] = model
            self.feature_engineers[model_id] = FeatureEngineer()
        logger.info(f"Created model {model_id} with name {config['name']}")
        return model_id

    async def get_model(self, model_id: ModelId) -> Optional[Model]:
        """Получение модели."""
        return self.models.get(model_id)

    async def update_model(self, model_id: ModelId, updates: Dict[str, Any]) -> bool:
        """Обновление модели."""
        if model_id not in self.models:
            return False
        # Исправление: теперь self.lock - это asyncio.Lock, который поддерживает async with
        async with self.lock:
            model = self.models[model_id]
            for key, value in updates.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.now()
        return True

    async def delete_model(self, model_id: ModelId) -> bool:
        """Удаление модели."""
        if model_id not in self.models:
            return False
        # Исправление: теперь self.lock - это asyncio.Lock, который поддерживает async with
        async with self.lock:
            del self.models[model_id]
            if model_id in self.model_objects:
                del self.model_objects[model_id]
            if model_id in self.scalers:
                del self.scalers[model_id]
            if model_id in self.feature_engineers:
                del self.feature_engineers[model_id]
        # Удаляем файл модели
        model_path = Path(self.config.models_dir) / f"{model_id}.pkl"
        if model_path.exists():
            model_path.unlink()
        return True

    async def list_models(self) -> List[Model]:
        """Список всех моделей."""
        return list(self.models.values())

    async def get_model_by_name(self, name: str) -> Optional[Model]:
        """Получение модели по имени."""
        for model in self.models.values():
            if model.name == name:
                return model
        return None

    async def get_models_by_status(self, status: ModelStatus) -> List[Model]:
        """Получение моделей по статусу."""
        return [model for model in self.models.values() if model.status == status]

    async def get_models_by_trading_pair(self, trading_pair: str) -> List[Model]:
        """Получение моделей по торговой паре."""
        return [
            model
            for model in self.models.values()
            if model.trading_pair == trading_pair
        ]

    async def get_model_metrics(self, model_id: ModelId) -> Dict[str, Any]:
        """Получение метрик модели."""
        model = await self.get_model(model_id)
        if not model:
            return {}
        return {
            "id": str(model.id),
            "name": model.name,
            "status": model.status.value,
            "accuracy": float(model.accuracy) if model.accuracy else 0.0,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat(),
            "features_count": len(model.features),
            "trading_pair": model.trading_pair,
            "prediction_type": model.prediction_type,
        }

    async def backup_model(self, model_id: ModelId, backup_path: str) -> bool:
        """Резервное копирование модели."""
        try:
            model = await self.get_model(model_id)
            if not model:
                return False
            model_object = self.model_objects.get(model_id)
            if not model_object:
                return False
            feature_engineer = self.feature_engineers[model_id]
            backup_data = {
                "model": model_object,
                "scaler": feature_engineer.scaler,
                "feature_names": feature_engineer.feature_names,
                "model_info": model,
                "backup_timestamp": datetime.now().isoformat(),
            }
            backup_file = Path(backup_path) / f"{model_id}_backup.pkl"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            with open(backup_file, "wb") as f:
                pickle.dump(backup_data, f)
            logger.info(f"Model {model_id} backed up to {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Error backing up model {model_id}: {str(e)}")
            return False

    async def restore_model(self, backup_path: str) -> Optional[ModelId]:
        """Восстановление модели из резервной копии."""
        try:
            with open(backup_path, "rb") as f:
                backup_data = pickle.load(f)
            model_info = backup_data["model_info"]
            model_object = backup_data["model"]
            scaler = backup_data["scaler"]
            feature_names = backup_data["feature_names"]
            # Создаем новую модель
            model_id = ModelId(UUID(str(len(self.models) + 1)))
            # Обновляем ID и временные метки
            model_info.id = model_id
            model_info.created_at = datetime.now()
            model_info.updated_at = datetime.now()
            # Исправление: теперь self.lock - это asyncio.Lock, который поддерживает async with
            async with self.lock:
                self.models[model_id] = model_info
                self.model_objects[model_id] = model_object
                self.scalers[model_id] = scaler
                feature_engineer = FeatureEngineer()
                feature_engineer.scaler = scaler
                feature_engineer.feature_names = feature_names
                self.feature_engineers[model_id] = feature_engineer
            logger.info(f"Model restored from backup: {model_id}")
            return model_id
        except Exception as e:
            logger.error(f"Error restoring model from {backup_path}: {str(e)}")
            return None

    async def get_model_size(self, model_id: ModelId) -> int:
        """Получение размера модели в байтах."""
        try:
            model_path = Path(self.config.models_dir) / f"{model_id}.pkl"
            if model_path.exists():
                return model_path.stat().st_size
            return 0
        except Exception:
            return 0

    async def cleanup_old_models(self, max_age_days: int = 30) -> int:
        """Очистка старых моделей."""
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            models_to_delete = []
            for model_id, model in self.models.items():
                if (
                    model.updated_at < cutoff_date
                    and model.status != ModelStatus.TRAINED
                ):
                    models_to_delete.append(model_id)
            deleted_count = 0
            for model_id in models_to_delete:
                if await self.delete_model(model_id):
                    deleted_count += 1
            logger.info(f"Cleaned up {deleted_count} old models")
            return deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old models: {str(e)}")
            return 0
