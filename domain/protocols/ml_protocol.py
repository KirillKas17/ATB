"""
Промышленный протокол для машинного обучения в алготрейдинге.
Обеспечивает типобезопасную работу с ML моделями и предсказаниями.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    runtime_checkable,
)
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

from domain.entities.ml import Model, ModelStatus, ModelType, Prediction, PredictionType
from domain.exceptions import (
    EvaluationError,
    InsufficientDataError,
    InvalidFeaturesError,
    InvalidModelTypeError,
    ModelLoadError,
    ModelNotFoundError,
    ModelNotReadyError,
    ModelSaveError,
    PredictionError,
    TrainingError,
)
from domain.types import (
    ConfidenceLevel,
    MetadataDict,
    ModelId,
    PerformanceScore,
    PredictionId,
    PriceValue,
    Symbol,
    TimestampValue,
    VolumeValue,
)


class ModelState(Enum):
    """Состояния ML модели."""

    CREATED = "created"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATING = "evaluating"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DEPRECATED = "deprecated"


class FeatureType(Enum):
    """Типы признаков."""

    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    EXTERNAL = "external"


class OptimizationMethod(Enum):
    """Методы оптимизации гиперпараметров."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    HYPEROPT = "hyperopt"


class MarketAnalysisProtocol(Protocol):
    """Протокол анализа рынка."""

    async def analyze_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ рыночных условий."""
        ...

    async def detect_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттернов."""
        ...

    async def calculate_technical_indicators(
        self, data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Расчет технических индикаторов."""
        ...

    async def predict_market_regime(self, data: pd.DataFrame) -> str:
        """Предсказание режима рынка."""
        ...


@dataclass(frozen=True)
class ModelConfig:
    """Конфигурация ML модели."""

    name: str
    model_type: ModelType
    trading_pair: Symbol
    prediction_type: PredictionType
    hyperparameters: Dict[str, Any]
    features: List[str]
    target: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class TrainingConfig:
    """Конфигурация обучения."""

    validation_split: float = 0.2
    test_split: float = 0.1
    random_state: int = 42
    shuffle: bool = False  # Для временных рядов
    early_stopping_patience: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    callbacks: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class PredictionConfig:
    """Конфигурация предсказания."""

    confidence_threshold: Optional[ConfidenceLevel] = None
    ensemble_method: str = "weighted_average"
    post_processing: bool = True
    uncertainty_estimation: bool = True
    feature_importance: bool = True


@dataclass(frozen=True)
class ModelMetrics:
    """Метрики модели."""

    mse: float
    mae: float
    r2: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    volatility: float
    calmar_ratio: float


@runtime_checkable
class ModelTrainingProtocol(Protocol):
    """Протокол обучения моделей."""

    async def prepare_data(
        self, raw_data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]: ...
    async def validate_data(self, data: pd.DataFrame) -> bool: ...
    async def split_data(
        self, data: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: ...
    async def feature_engineering(self, data: pd.DataFrame) -> pd.DataFrame: ...
    async def train_model(
        self, model: Model, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Model: ...
    async def validate_model(
        self, model: Model, X_val: pd.DataFrame, y_val: pd.Series
    ) -> float: ...
@runtime_checkable
class ModelEvaluationProtocol(Protocol):
    """Протокол оценки моделей."""

    async def calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]: ...
    async def cross_validate(
        self, model: Model, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]: ...
    async def backtest_model(
        self, model: Model, test_data: pd.DataFrame
    ) -> Dict[str, float]: ...
    async def calculate_feature_importance(
        self, model: Model, X: pd.DataFrame
    ) -> Dict[str, float]: ...
@runtime_checkable
class PredictionProtocol(Protocol):
    """Протокол предсказаний."""

    async def preprocess_features(self, features: Dict[str, Any]) -> np.ndarray: ...
    async def postprocess_prediction(
        self, raw_prediction: np.ndarray
    ) -> Prediction: ...
    async def validate_prediction(self, prediction: Prediction) -> bool: ...
    async def calculate_confidence(
        self, prediction: Prediction, model: Model
    ) -> ConfidenceLevel: ...


T = TypeVar("T", bound=Model)


class MLProtocol(ABC):
    """
    Промышленный протокол для машинного обучения в алготрейдинге.
    Обеспечивает полный цикл работы с ML моделями:
    - Подготовка и валидация данных
    - Обучение и оптимизация моделей
    - Предсказания и их валидация
    - Оценка производительности
    - Управление жизненным циклом моделей
    - Ансамблирование и адаптация
    """

    def __init__(self) -> None:
        """Инициализация ML протокола."""
        self.logger = logging.getLogger(__name__)
        self._models_cache: Dict[ModelId, Model] = {}
        self._predictions_cache: Dict[PredictionId, Prediction] = {}
        self._training_jobs: Dict[ModelId, asyncio.Task] = {}
        self._model_versions: Dict[str, List[ModelId]] = {}

    # ============================================================================
    # УПРАВЛЕНИЕ МОДЕЛЯМИ
    # ============================================================================
    @abstractmethod
    async def create_model(self, config: ModelConfig) -> Model:
        """
        Создание новой ML модели.
        Args:
            config: Конфигурация модели
        Returns:
            Model: Созданная модель
        Raises:
            InvalidModelTypeError: Неверный тип модели
            InvalidFeaturesError: Неверные признаки
        """
        pass

    @abstractmethod
    async def train_model(
        self,
        model_id: ModelId,
        training_data: pd.DataFrame,
        config: TrainingConfig,
        validation_data: Optional[pd.DataFrame] = None,
    ) -> Model:
        """
        Обучение ML модели.
        Args:
            model_id: ID модели
            training_data: Данные для обучения
            config: Конфигурация обучения
            validation_data: Данные для валидации
        Returns:
            Model: Обученная модель
        Raises:
            ModelNotFoundError: Модель не найдена
            InsufficientDataError: Недостаточно данных
            TrainingError: Ошибка обучения
        """
        pass

    @abstractmethod
    async def predict(
        self,
        model_id: ModelId,
        features: Dict[str, Any],
        config: Optional[PredictionConfig] = None,
    ) -> Optional[Prediction]:
        """
        Выполнение предсказания.
        Args:
            model_id: ID модели
            features: Признаки для предсказания
            config: Конфигурация предсказания
        Returns:
            Optional[Prediction]: Предсказание или None
        Raises:
            ModelNotFoundError: Модель не найдена
            ModelNotReadyError: Модель не готова
            InvalidFeaturesError: Неверные признаки
        """
        pass

    @abstractmethod
    async def batch_predict(
        self,
        model_id: ModelId,
        features_batch: List[Dict[str, Any]],
        config: Optional[PredictionConfig] = None,
    ) -> List[Prediction]:
        """
        Пакетное предсказание.
        Args:
            model_id: ID модели
            features_batch: Пакет признаков
            config: Конфигурация предсказания
        Returns:
            List[Prediction]: Список предсказаний
        """
        pass

    # ============================================================================
    # ОЦЕНКА И ВАЛИДАЦИЯ
    # ============================================================================
    @abstractmethod
    async def evaluate_model(
        self,
        model_id: ModelId,
        test_data: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> ModelMetrics:
        """
        Оценка производительности модели.
        Args:
            model_id: ID модели
            test_data: Тестовые данные
            metrics: Список метрик для расчета
        Returns:
            ModelMetrics: Метрики производительности
        Raises:
            ModelNotFoundError: Модель не найдена
            EvaluationError: Ошибка оценки
        """
        pass

    @abstractmethod
    async def cross_validate(
        self,
        model_id: ModelId,
        data: pd.DataFrame,
        cv_folds: int = 5,
        cv_method: str = "time_series",
    ) -> Dict[str, List[float]]:
        """
        Кросс-валидация модели.
        Args:
            model_id: ID модели
            data: Данные для валидации
            cv_folds: Количество фолдов
            cv_method: Метод кросс-валидации
        Returns:
            Dict[str, List[float]]: Результаты кросс-валидации
        """
        pass

    @abstractmethod
    async def backtest_model(
        self,
        model_id: ModelId,
        historical_data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
        transaction_cost: Decimal = Decimal("0.001"),
        slippage: Decimal = Decimal("0.0001"),
    ) -> Dict[str, Any]:
        """
        Бэктестинг модели.
        Args:
            model_id: ID модели
            historical_data: Исторические данные
            initial_capital: Начальный капитал
            transaction_cost: Стоимость транзакций
            slippage: Проскальзывание
        Returns:
            Dict[str, Any]: Результаты бэктестинга
        """
        pass

    @abstractmethod
    async def optimize_hyperparameters(
        self,
        model_id: ModelId,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        optimization_method: OptimizationMethod = OptimizationMethod.BAYESIAN_OPTIMIZATION,
        n_trials: int = 100,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Оптимизация гиперпараметров.
        Args:
            model_id: ID модели
            training_data: Данные для обучения
            validation_data: Данные для валидации
            param_grid: Сетка параметров
            optimization_method: Метод оптимизации
            n_trials: Количество попыток
            cv_folds: Количество фолдов для CV
        Returns:
            Dict[str, Any]: Результаты оптимизации
        """
        pass

    @abstractmethod
    async def feature_selection(
        self,
        model_id: ModelId,
        data: pd.DataFrame,
        method: str = "mutual_info",
        n_features: Optional[int] = None,
        threshold: float = 0.01,
    ) -> List[str]:
        """
        Отбор признаков.
        Args:
            model_id: ID модели
            data: Данные
            method: Метод отбора
            n_features: Количество признаков
            threshold: Порог значимости
        Returns:
            List[str]: Отобранные признаки
        """
        pass

    @abstractmethod
    async def calculate_feature_importance(
        self, model_id: ModelId, data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Расчет важности признаков.
        Args:
            model_id: ID модели
            data: Данные
        Returns:
            Dict[str, float]: Важность признаков
        """
        pass

    # ============================================================================
    # СОХРАНЕНИЕ И ЗАГРУЗКА
    # ============================================================================
    @abstractmethod
    async def save_model(self, model_id: ModelId, path: str) -> bool:
        """
        Сохранение модели.
        Args:
            model_id: ID модели
            path: Путь для сохранения
        Returns:
            bool: Успешность сохранения
        Raises:
            ModelNotFoundError: Модель не найдена
            ModelSaveError: Ошибка сохранения
        """
        pass

    @abstractmethod
    async def load_model(self, model_id: ModelId, path: str) -> Model:
        """
        Загрузка модели.
        Args:
            model_id: ID модели
            path: Путь к модели
        Returns:
            Model: Загруженная модель
        Raises:
            ModelNotFoundError: Модель не найдена
            ModelLoadError: Ошибка загрузки
        """
        pass

    @abstractmethod
    async def export_model_metadata(self, model_id: ModelId, path: str) -> bool:
        """
        Экспорт метаданных модели.
        Args:
            model_id: ID модели
            path: Путь для экспорта
        Returns:
            bool: Успешность экспорта
        """
        pass

    @abstractmethod
    async def import_model_metadata(self, path: str) -> ModelConfig:
        """
        Импорт метаданных модели.
        Args:
            path: Путь к метаданным
        Returns:
            ModelConfig: Конфигурация модели
        """
        pass

    # ============================================================================
    # УПРАВЛЕНИЕ ЖИЗНЕННЫМ ЦИКЛОМ
    # ============================================================================
    @abstractmethod
    async def get_model_status(self, model_id: ModelId) -> ModelStatus:
        """
        Получение статуса модели.
        Args:
            model_id: ID модели
        Returns:
            ModelStatus: Статус модели
        """
        pass

    @abstractmethod
    async def activate_model(self, model_id: ModelId) -> bool:
        """
        Активация модели.
        Args:
            model_id: ID модели
        Returns:
            bool: Успешность активации
        """
        pass

    @abstractmethod
    async def deactivate_model(self, model_id: ModelId) -> bool:
        """
        Деактивация модели.
        Args:
            model_id: ID модели
        Returns:
            bool: Успешность деактивации
        """
        pass

    @abstractmethod
    async def delete_model(self, model_id: ModelId) -> bool:
        """
        Удаление модели.
        Args:
            model_id: ID модели
        Returns:
            bool: Успешность удаления
        """
        pass

    @abstractmethod
    async def archive_model(self, model_id: ModelId) -> bool:
        """
        Архивирование модели.
        Args:
            model_id: ID модели
        Returns:
            bool: Успешность архивирования
        """
        pass

    # ============================================================================
    # МОНИТОРИНГ И АНАЛИТИКА
    # ============================================================================
    @abstractmethod
    async def get_model_performance_history(
        self,
        model_id: ModelId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получение истории производительности модели.
        Args:
            model_id: ID модели
            start_date: Начальная дата
            end_date: Конечная дата
        Returns:
            List[Dict[str, Any]]: История производительности
        """
        pass

    @abstractmethod
    async def get_prediction_history(
        self,
        model_id: ModelId,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Prediction]:
        """
        Получение истории предсказаний.
        Args:
            model_id: ID модели
            limit: Лимит записей
            start_date: Начальная дата
            end_date: Конечная дата
        Returns:
            List[Prediction]: История предсказаний
        """
        pass

    @abstractmethod
    async def monitor_model_drift(
        self,
        model_id: ModelId,
        recent_data: pd.DataFrame,
        drift_threshold: float = 0.1,
        drift_method: str = "ks_test",
    ) -> Dict[str, Any]:
        """
        Мониторинг дрифта модели.
        Args:
            model_id: ID модели
            recent_data: Недавние данные
            drift_threshold: Порог дрифта
            drift_method: Метод определения дрифта
        Returns:
            Dict[str, Any]: Результаты мониторинга дрифта
        """
        pass

    @abstractmethod
    async def calculate_model_confidence(
        self, model_id: ModelId, prediction: Prediction
    ) -> ConfidenceLevel:
        """
        Расчет уверенности модели.
        Args:
            model_id: ID модели
            prediction: Предсказание
        Returns:
            ConfidenceLevel: Уровень уверенности
        """
        pass

    # ============================================================================
    # АНСАМБЛИРОВАНИЕ
    # ============================================================================
    @abstractmethod
    async def create_ensemble(
        self,
        name: str,
        models: List[ModelId],
        weights: Optional[List[float]] = None,
        voting_method: str = "weighted_average",
        meta_learner: Optional[str] = None,
    ) -> ModelId:
        """
        Создание ансамбля моделей.
        Args:
            name: Название ансамбля
            models: Список моделей
            weights: Веса моделей
            voting_method: Метод голосования
            meta_learner: Мета-обучатель
        Returns:
            ModelId: ID ансамбля
        """
        pass

    @abstractmethod
    async def ensemble_predict(
        self,
        ensemble_id: ModelId,
        features: Dict[str, Any],
        config: Optional[PredictionConfig] = None,
    ) -> Prediction:
        """
        Предсказание ансамбля.
        Args:
            ensemble_id: ID ансамбля
            features: Признаки
            config: Конфигурация предсказания
        Returns:
            Prediction: Предсказание ансамбля
        """
        pass

    @abstractmethod
    async def update_ensemble_weights(
        self, ensemble_id: ModelId, new_weights: List[float]
    ) -> bool:
        """
        Обновление весов ансамбля.
        Args:
            ensemble_id: ID ансамбля
            new_weights: Новые веса
        Returns:
            bool: Успешность обновления
        """
        pass

    # ============================================================================
    # АДАПТАЦИЯ И ОБУЧЕНИЕ
    # ============================================================================
    @abstractmethod
    async def online_learning(
        self,
        model_id: ModelId,
        new_data: pd.DataFrame,
        learning_rate: float = 0.01,
        batch_size: int = 1,
    ) -> bool:
        """
        Онлайн обучение модели.
        Args:
            model_id: ID модели
            new_data: Новые данные
            learning_rate: Скорость обучения
            batch_size: Размер батча
        Returns:
            bool: Успешность обучения
        """
        pass

    @abstractmethod
    async def transfer_learning(
        self,
        source_model_id: ModelId,
        target_config: ModelConfig,
        fine_tune_layers: Optional[List[str]] = None,
    ) -> ModelId:
        """
        Перенос обучения.
        Args:
            source_model_id: ID исходной модели
            target_config: Конфигурация целевой модели
            fine_tune_layers: Слои для тонкой настройки
        Returns:
            ModelId: ID новой модели
        """
        pass

    @abstractmethod
    async def adaptive_learning(
        self, model_id: ModelId, market_regime: str, adaptation_rules: Dict[str, Any]
    ) -> bool:
        """
        Адаптивное обучение.
        Args:
            model_id: ID модели
            market_regime: Рыночный режим
            adaptation_rules: Правила адаптации
        Returns:
            bool: Успешность адаптации
        """
        pass

    # ============================================================================
    # ОБРАБОТКА ОШИБОК И ВОССТАНОВЛЕНИЕ
    # ============================================================================
    @abstractmethod
    async def handle_model_error(self, model_id: ModelId, error: Exception) -> bool:
        """
        Обработка ошибки модели.
        Args:
            model_id: ID модели
            error: Ошибка
        Returns:
            bool: Успешность обработки
        """
        pass

    @abstractmethod
    async def retry_prediction(
        self,
        model_id: ModelId,
        features: Dict[str, Any],
        max_retries: int = 3,
        fallback_model_id: Optional[ModelId] = None,
    ) -> Optional[Prediction]:
        """
        Повторная попытка предсказания.
        Args:
            model_id: ID модели
            features: Признаки
            max_retries: Максимальное количество попыток
            fallback_model_id: ID резервной модели
        Returns:
            Optional[Prediction]: Предсказание или None
        """
        pass

    @abstractmethod
    async def validate_model_integrity(self, model_id: ModelId) -> bool:
        """
        Валидация целостности модели.
        Args:
            model_id: ID модели
        Returns:
            bool: Целостность модели
        """
        pass

    @abstractmethod
    async def recover_model_state(
        self, model_id: ModelId, recovery_point: Optional[datetime] = None
    ) -> bool:
        """
        Восстановление состояния модели.
        Args:
            model_id: ID модели
            recovery_point: Точка восстановления
        Returns:
            bool: Успешность восстановления
        """
        pass

    # ============================================================================
    # УТИЛИТЫ
    # ============================================================================
    async def _validate_training_data(self, data: pd.DataFrame) -> bool:
        """Валидация данных для обучения."""
        if data.empty:
            return False
        if data.isnull().any().any():
            return False
        if len(data) < 100:  # Минимальное количество записей
            return False
        return True

    async def _validate_features(
        self, features: Dict[str, Any], expected_features: List[str]
    ) -> bool:
        """Валидация признаков."""
        for feature in expected_features:
            if feature not in features:
                return False
        return True

    async def _calculate_prediction_confidence(
        self, prediction: Prediction, model_metrics: ModelMetrics
    ) -> ConfidenceLevel:
        """Расчет уверенности в предсказании."""
        # Простая эвристика на основе метрик модели
        base_confidence = min(model_metrics.r2, 0.95)
        return ConfidenceLevel(Decimal(str(base_confidence)))

    async def _preprocess_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Предобработка признаков."""
        # Базовая реализация - нормализация
        feature_values = list(features.values())
        return np.array(feature_values).reshape(1, -1)

    async def _postprocess_prediction(
        self, raw_prediction: np.ndarray, model: Model, confidence: ConfidenceLevel
    ) -> Prediction:
        """Постобработка предсказания."""
        # Создание объекта предсказания
        return Prediction(
            id=PredictionId(uuid4()),
            model_id=model.id,
            value=raw_prediction[0],
            confidence=confidence,
            timestamp=datetime.now(),
            metadata=MetadataDict({}),
        )

    async def _cache_model(self, model: Model) -> None:
        """Кэширование модели."""
        self._models_cache[ModelId(model.id)] = model

    async def _get_cached_model(self, model_id: ModelId) -> Optional[Model]:
        """Получение модели из кэша."""
        return self._models_cache.get(model_id)

    async def _clear_model_cache(self, model_id: Optional[ModelId] = None) -> None:
        """Очистка кэша моделей."""
        if model_id:
            self._models_cache.pop(model_id, None)
        else:
            self._models_cache.clear()
