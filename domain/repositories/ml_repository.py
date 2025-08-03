"""
Репозиторий для работы с ML моделями.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from domain.entities.ml import Model, ModelType, Prediction
from domain.protocols.repository_protocol import RepositoryProtocol
from domain.types import EntityId  # type: ignore[attr-defined]


class MLRepository(RepositoryProtocol):
    """Репозиторий для работы с ML моделями."""

    @abstractmethod
    async def save_model(self, model: Model) -> Model:
        """Сохранение модели."""

    @abstractmethod
    async def get_model(self, model_id: EntityId) -> Optional[Model]:
        """Получение модели по ID."""

    @abstractmethod
    async def get_all_models(self) -> List[Model]:
        """Получение всех моделей."""

    @abstractmethod
    async def delete_model(self, model_id: EntityId) -> bool:
        """Удаление модели."""

    @abstractmethod
    async def get_models_by_type(self, model_type: ModelType) -> List[Model]:
        """Получение моделей по типу."""

    @abstractmethod
    async def get_models_by_trading_pair(self, trading_pair: str) -> List[Model]:
        """Получение моделей по торговой паре."""

    @abstractmethod
    async def get_active_models(self) -> List[Model]:
        """Получение активных моделей."""

    @abstractmethod
    async def save_prediction(self, prediction: Prediction) -> Prediction:
        """Сохранение предсказания."""

    @abstractmethod
    async def get_predictions_by_model(
        self, model_id: EntityId, limit: int = 100
    ) -> List[Prediction]:
        """Получение предсказаний модели."""

    @abstractmethod
    async def get_latest_prediction(self, model_id: EntityId) -> Optional[Prediction]:
        """Получение последнего предсказания модели."""


class InMemoryMLRepository(MLRepository):
    def __init__(self) -> None:
        """Инициализация репозитория."""
        self._models: Dict[EntityId, Model] = {}
        self._predictions: Dict[EntityId, List[Prediction]] = {}
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._metrics: Dict[str, Any] = {}

    async def save(self, entity: Model) -> Model:
        """Сохранить модель."""
        self._models[entity.id] = entity
        return entity

    async def get_by_id(self, entity_id: EntityId) -> Optional[Model]:
        """Получить модель по ID."""
        return self._models.get(entity_id)

    async def update(self, entity: Model) -> Model:
        """Обновить модель."""
        if entity.id in self._models:
            self._models[entity.id] = entity
        return entity

    async def delete(self, entity_id: EntityId) -> bool:
        """Удалить модель."""
        if entity_id in self._models:
            del self._models[entity_id]
            return True
        return False

    async def save_model(self, model: Model) -> Model:
        self._models[EntityId(model.id)] = model
        return model

    async def get_model(self, model_id: EntityId) -> Optional[Model]:
        return self._models.get(model_id)

    async def get_all_models(self) -> List[Model]:
        return list(self._models.values())

    async def delete_model(self, model_id: EntityId) -> bool:
        return self._models.pop(model_id, None) is not None

    async def get_models_by_type(self, model_type: ModelType) -> List[Model]:
        return [m for m in self._models.values() if m.model_type == model_type]

    async def get_models_by_trading_pair(self, trading_pair: str) -> List[Model]:
        return [
            m
            for m in self._models.values()
            if getattr(m, "trading_pair", None) == trading_pair
        ]

    async def get_active_models(self) -> List[Model]:
        return [m for m in self._models.values() if getattr(m, "is_active", False)]

    async def save_prediction(self, prediction: Prediction) -> Prediction:
        model_id = EntityId(prediction.model_id)
        if model_id not in self._predictions:
            self._predictions[model_id] = []
        self._predictions[model_id].append(prediction)
        return prediction

    async def get_predictions_by_model(
        self, model_id: EntityId, limit: int = 100
    ) -> List[Prediction]:
        predictions: List[Prediction] = self._predictions.get(model_id, [])
        return predictions[:limit]

    async def get_latest_prediction(self, model_id: EntityId) -> Optional[Prediction]:
        predictions: List[Prediction] = self._predictions.get(model_id, [])
        return predictions[-1] if predictions else None
