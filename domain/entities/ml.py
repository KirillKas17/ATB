"""
Доменные сущности машинного обучения.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union
from uuid import UUID, uuid4

from shared.numpy_utils import np
import pandas as pd

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money


class ModelType(Enum):
    """Типы моделей."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    CUSTOM = "custom"
    XGBOOST = "xgboost"


class ModelStatus(Enum):
    """Статусы модели."""

    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class PredictionType(Enum):
    """Типы предсказаний."""

    PRICE = "price"
    DIRECTION = "direction"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    REGIME = "regime"
    SIGNAL = "signal"
    PROBABILITY = "probability"


@dataclass
class Model:
    """Модель машинного обучения"""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    model_type: ModelType = ModelType.LINEAR_REGRESSION
    status: ModelStatus = ModelStatus.INACTIVE
    version: str = "1.0.0"
    trading_pair: str = ""
    prediction_type: PredictionType = PredictionType.PRICE
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    target: str = ""
    accuracy: Union[Decimal, int, float, str] = Decimal("0")
    precision: Union[Decimal, int, float, str] = Decimal("0")
    recall: Union[Decimal, int, float, str] = Decimal("0")
    f1_score: Union[Decimal, int, float, str] = Decimal("0")
    mse: Union[Decimal, int, float, str] = Decimal("0")
    mae: Union[Decimal, int, float, str] = Decimal("0")
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    trained_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Валидация и нормализация метрик
        self._normalize_metric("accuracy")
        self._normalize_metric("precision")
        self._normalize_metric("recall")
        self._normalize_metric("f1_score")
        self._normalize_metric("mse")
        self._normalize_metric("mae")

    def _normalize_metric(self, metric_name: str) -> None:
        """Нормализовать метрику к типу Decimal"""
        current_value = getattr(self, metric_name)
        if not isinstance(current_value, Decimal):
            normalized_value = Decimal(str(current_value))
            object.__setattr__(self, metric_name, normalized_value)

    def activate(self) -> None:
        """Активировать модель"""
        self.status = ModelStatus.ACTIVE
        self.updated_at = datetime.now()

    def deactivate(self) -> None:
        """Деактивировать модель"""
        self.status = ModelStatus.INACTIVE
        self.updated_at = datetime.now()

    def mark_trained(self) -> None:
        """Отметить как обученную"""
        self.trained_at = datetime.now()
        self.updated_at = datetime.now()

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Обновить метрики модели"""
        for key, value in metrics.items():
            if hasattr(self, key):
                if isinstance(value, (int, float, str)):
                    setattr(self, key, Decimal(str(value)))
                else:
                    setattr(self, key, value)
        self.updated_at = datetime.now()

    def is_ready_for_prediction(self) -> bool:
        """Готова ли модель для предсказаний"""
        # Убеждаемся, что accuracy нормализована к Decimal
        accuracy_decimal = self.accuracy if isinstance(self.accuracy, Decimal) else Decimal(str(self.accuracy))
        return (
            self.status == ModelStatus.ACTIVE
            and self.trained_at is not None
            and accuracy_decimal > Decimal("0.5")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "model_type": self.model_type.value,
            "status": self.status.value,
            "version": self.version,
            "trading_pair": self.trading_pair,
            "prediction_type": self.prediction_type.value,
            "accuracy": str(self.accuracy),
            "precision": str(self.precision),
            "recall": str(self.recall),
            "f1_score": str(self.f1_score),
            "mse": str(self.mse),
            "mae": str(self.mae),
            "created_at": self.created_at.isoformat(),
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class Prediction:
    """Предсказание модели"""

    id: UUID = field(default_factory=uuid4)
    model_id: UUID = field(default_factory=uuid4)
    trading_pair: str = ""
    prediction_type: PredictionType = PredictionType.PRICE
    value: Union[Decimal, Money, int, float, str] = field(default_factory=lambda: Decimal("0"))
    confidence: Union[Decimal, int, float, str] = Decimal("0.5")
    timestamp: datetime = field(default_factory=datetime.now)
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Валидация и нормализация значений
        self._normalize_value()
        self._normalize_confidence()

    def _normalize_value(self) -> None:
        """Нормализовать значение предсказания"""
        current_value = self.value
        # Проверяем тип значения и нормализуем при необходимости
        if isinstance(current_value, (int, float, str)):
            if self.prediction_type == PredictionType.PRICE:
                object.__setattr__(self, "value", Money(Decimal(str(current_value)), Currency.USD))
            else:
                object.__setattr__(self, "value", Decimal(str(current_value)))

    def _normalize_confidence(self) -> None:
        """Нормализовать уверенность к типу Decimal"""
        current_confidence = self.confidence
        # Проверяем тип уверенности и нормализуем при необходимости
        if isinstance(current_confidence, (int, float, str)):
            normalized_confidence = Decimal(str(current_confidence))
            object.__setattr__(self, "confidence", normalized_confidence)

    def is_high_confidence(self) -> bool:
        """Высокая ли уверенность"""
        confidence_decimal = self.confidence if isinstance(self.confidence, Decimal) else Decimal(str(self.confidence))
        return confidence_decimal > Decimal("0.7")

    def is_medium_confidence(self) -> bool:
        """Средняя ли уверенность"""
        confidence_decimal = self.confidence if isinstance(self.confidence, Decimal) else Decimal(str(self.confidence))
        return Decimal("0.5") <= confidence_decimal <= Decimal("0.7")

    def is_low_confidence(self) -> bool:
        """Низкая ли уверенность"""
        confidence_decimal = self.confidence if isinstance(self.confidence, Decimal) else Decimal(str(self.confidence))
        return confidence_decimal < Decimal("0.5")

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь"""
        return {
            "id": str(self.id),
            "model_id": str(self.model_id),
            "trading_pair": self.trading_pair,
            "prediction_type": self.prediction_type.value,
            "value": (
                str(self.value.value)
                if hasattr(self.value, "value") and isinstance(self.value, Money)
                else str(self.value)
            ),
            "confidence": str(self.confidence) if isinstance(self.confidence, (Decimal, Money)) else str(Decimal(str(self.confidence))),
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "metadata": self.metadata,
        }


@dataclass
class ModelEnsemble:
    """Ансамбль моделей"""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    models: List[Model] = field(default_factory=list)
    weights: Dict[UUID, Decimal] = field(default_factory=dict)
    voting_method: str = "weighted_average"  # weighted_average, majority, stacking
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_model(self, model: Model, weight: Decimal = Decimal("1")) -> None:
        """Добавить модель в ансамбль"""
        if model.id not in [m.id for m in self.models]:
            self.models.append(model)
            self.weights[model.id] = weight
            self.updated_at = datetime.now()

    def remove_model(self, model_id: UUID) -> None:
        """Удалить модель из ансамбля"""
        self.models = [m for m in self.models if m.id != model_id]
        if model_id in self.weights:
            del self.weights[model_id]
        self.updated_at = datetime.now()

    def update_weight(self, model_id: UUID, weight: Decimal) -> None:
        """Обновить вес модели"""
        if model_id in self.weights:
            self.weights[model_id] = weight
            self.updated_at = datetime.now()

    def get_active_models(self) -> List[Model]:
        """Получить активные модели"""
        return [m for m in self.models if m.is_ready_for_prediction()]

    def predict(self, features: Dict[str, Any]) -> Optional[Prediction]:
        """Сделать предсказание ансамблем"""
        active_models = self.get_active_models()
        if not active_models:
            return None
        predictions = []
        total_weight = Decimal("0")
        for model in active_models:
            weight = self.weights.get(model.id, Decimal("1"))
            # Здесь должна быть логика получения предсказания от модели
            # Пока возвращаем заглушку
            predictions.append((model, weight))
            total_weight += weight
        if total_weight == 0:
            return None
        # Взвешенное среднее предсказаний
        weighted_value = Decimal("0")
        for model, weight in predictions:
            # Упрощенная логика - в реальности здесь будет вызов модели
            weighted_value += weight
        avg_value = weighted_value / total_weight
        return Prediction(
            model_id=self.id,
            trading_pair=active_models[0].trading_pair if active_models else "",
            prediction_type=(
                active_models[0].prediction_type
                if active_models
                else PredictionType.PRICE
            ),
            value=avg_value,
            confidence=Decimal("0.6"),  # Средняя уверенность для ансамбля
            features=features,
        )


@dataclass
class MLPipeline:
    """ML пайплайн"""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    models: List[Model] = field(default_factory=list)
    ensembles: List[ModelEnsemble] = field(default_factory=list)
    feature_engineering: Dict[str, Any] = field(default_factory=dict)
    data_preprocessing: Dict[str, Any] = field(default_factory=dict)
    evaluation_metrics: List[str] = field(default_factory=list)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def add_model(self, model: Model) -> None:
        """Добавить модель"""
        if model.id not in [m.id for m in self.models]:
            self.models.append(model)
            self.updated_at = datetime.now()

    def add_ensemble(self, ensemble: ModelEnsemble) -> None:
        """Добавить ансамбль"""
        if ensemble.id not in [e.id for e in self.ensembles]:
            self.ensembles.append(ensemble)
            self.updated_at = datetime.now()

    def get_best_model(self, metric: str = "accuracy") -> Optional[Model]:
        """Получить лучшую модель по метрике"""
        if not self.models:
            return None
        return max(self.models, key=lambda m: getattr(m, metric, Decimal("0")))

    def get_active_models(self) -> List[Model]:
        """Получить активные модели"""
        return [m for m in self.models if m.status == ModelStatus.ACTIVE]

    def get_active_ensembles(self) -> List[ModelEnsemble]:
        """Получить активные ансамбли"""
        return [e for e in self.ensembles if e.is_active]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь"""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "models_count": len(self.models),
            "ensembles_count": len(self.ensembles),
            "active_models": len(self.get_active_models()),
            "active_ensembles": len(self.get_active_ensembles()),
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ModelProtocol(Protocol):
    """Протокол модели."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучение."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Оценка производительности."""
        ...


class MLModelInterface(ABC):
    """Интерфейс модели машинного обучения."""

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Обучение модели."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Предсказание."""

    @abstractmethod
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Оценка модели."""

    @abstractmethod
    def save(self, path: str) -> bool:
        """Сохранение модели."""

    @abstractmethod
    def load(self, path: str) -> bool:
        """Загрузка модели."""

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков."""
