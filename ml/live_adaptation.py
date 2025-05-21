import asyncio
import json
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Deque, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import bayes_opt as bo
import joblib
import numpy as np
import pandas as pd
from deap import algorithms, base, creator, tools
from loguru import logger
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ml.model_selector import ModelSelector
from ml.pattern_discovery import PatternDiscovery
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Интерфейсы для тестирования
class MarketStateInterface(ABC):
    """Интерфейс для работы с состоянием рынка"""

    @abstractmethod
    async def get_current_regime(self, pair: str, timeframe: str) -> str:
        pass

    @abstractmethod
    async def get_previous_regime(self, pair: str, timeframe: str) -> str:
        pass


class AdaptableModel(ABC):
    """Базовый класс для адаптивных моделей"""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание на новых данных"""
        pass

    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обновление модели на новых данных"""
        pass

    @abstractmethod
    def get_confidence(self, X: np.ndarray) -> np.ndarray:
        """Оценка уверенности в предсказаниях"""
        pass


class ModelSelectorInterface(ABC):
    """Интерфейс для выбора моделей"""

    @abstractmethod
    def select_model(self, market_regime: str) -> AdaptableModel:
        pass


class PatternDiscoveryInterface(ABC):
    """Интерфейс для обнаружения паттернов"""

    @abstractmethod
    def discover_patterns(self, data: pd.DataFrame) -> List[Dict]:
        pass


@runtime_checkable
class AdaptableModel(Protocol):
    """Протокол для адаптируемых моделей"""

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None
    ) -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...


@dataclass
class AdaptationConfig:
    """Конфигурация адаптации"""

    min_samples: int = 100
    max_samples: int = 1000
    update_interval: int = 1  # часов
    retrain_threshold: float = 0.1
    confidence_threshold: float = 0.7
    max_retries: int = 3
    cache_size: int = 1000
    compression: bool = True
    metrics_window: int = 24  # часов
    drift_threshold: float = 0.05
    ensemble_size: int = 3
    feature_batch_size: int = 100
    learning_rate: float = 0.01
    bo_iterations: int = 10
    max_metrics_age_days: int = 30
    metrics_dir: str = "metrics"


@dataclass
class AdaptationMetrics:
    """Метрики адаптации"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confidence: float
    drift_score: float
    last_update: datetime
    samples_count: int
    retrain_count: int
    error_count: int


@dataclass
class AdaptationState:
    """Состояние адаптации модели"""

    timestamp: datetime
    performance: float
    parameters: Dict[str, float]
    market_conditions: Dict[str, float]


class LiveAdaptationModel:
    """Адаптация моделей в реальном времени"""

    def __init__(self, config: Optional[AdaptationConfig] = None):
        """Инициализация адаптации"""
        self.config = config or AdaptationConfig()
        self.adaptation_dir = Path("adaptation")
        self.adaptation_dir.mkdir(parents=True, exist_ok=True)

        # Данные
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []

        # Модели
        self.models = {}
        self.scalers = {}
        self.metrics = {}

        # Кэш
        self._prediction_cache = {}
        self._feature_cache = {}

        # Инициализация компонентов
        self.model: Optional[AdaptableModel] = None
        self.market_state: Optional[MarketStateInterface] = None
        self.model_selector: Optional[ModelSelectorInterface] = None
        self.pattern_discovery: Optional[PatternDiscoveryInterface] = None

        # Хранение метрик с ограничением размера
        self.metrics: Dict[str, Dict[str, AdaptationMetrics]] = {}

        # История адаптации с ограничением размера
        self.adaptation_history: Deque[AdaptationState] = deque(maxlen=self.config.max_samples)

        # Блокировки для потокобезопасности
        self._metrics_lock = Lock()
        self._model_lock = Lock()
        self._history_lock = Lock()

        # Инициализация моделей
        self._init_models()

        # Загрузка состояния
        self._load_state()

    async def initialize(self) -> None:
        """Асинхронная инициализация"""
        from core.market_state import MarketState

        self.market_state = MarketState()
        await self.market_state.initialize_ml()

        self.model_selector = ModelSelector()
        self.pattern_discovery = PatternDiscovery()

    def _init_models(self) -> None:
        """Инициализация моделей"""
        try:
            # Инициализация онлайн-классификатора
            self.online_classifier = SGDClassifier(
                learning_rate="constant", eta0=self.config.learning_rate
            )

            # Инициализация нормализации
            self.scaler = StandardScaler()

            # Инициализация GA
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

            self.toolbox = base.Toolbox()
            self.toolbox.register("attr_float", np.random.uniform, -1, 1)
            self.toolbox.register(
                "individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=10
            )
            self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

            self.toolbox.register("evaluate", self._evaluate_individual)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
            self.toolbox.register("select", tools.selTournament, tournsize=3)

            # Инициализация BO
            self.bo = bo.BayesianOptimization(
                f=self._evaluate_bo,
                pbounds={
                    "learning_rate": (0.001, 0.1),
                    "batch_size": (16, 64),
                    "momentum": (0.8, 0.99),
                },
            )

            # Инициализация ансамбля моделей
            self.models = {
                "rf": RandomForestRegressor(n_estimators=100),
                "gb": GradientBoostingRegressor(n_estimators=100),
                "mlp": MLPRegressor(hidden_layer_sizes=(100, 50)),
                "svr": SVR(kernel="rbf"),
            }

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка признаков с кэшированием и батчингом

        Args:
            data: DataFrame с данными

        Returns:
            pd.DataFrame: Подготовленные признаки
        """
        try:
            # Копирование данных
            features = data.copy()

            # Батчинг для больших наборов данных
            if len(features) > self.config.feature_batch_size:
                features = features.tail(self.config.feature_batch_size)

            # Добавление технических индикаторов
            features["sma_20"] = features["close"].rolling(window=20).mean()
            features["sma_50"] = features["close"].rolling(window=50).mean()
            features["sma_200"] = features["close"].rolling(window=200).mean()

            features["volatility"] = features["close"].rolling(window=20).std()

            delta = features["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = 100 - (100 / (1 + rs))

            exp1 = features["close"].ewm(span=12, adjust=False).mean()
            exp2 = features["close"].ewm(span=26, adjust=False).mean()
            features["macd"] = exp1 - exp2
            features["signal"] = features["macd"].ewm(span=9, adjust=False).mean()

            features["bb_middle"] = features["close"].rolling(window=20).mean()
            features["bb_std"] = features["close"].rolling(window=20).std()
            features["bb_upper"] = features["bb_middle"] + (features["bb_std"] * 2)
            features["bb_lower"] = features["bb_middle"] - (features["bb_std"] * 2)

            # Добавление паттернов
            patterns = self.pattern_discovery.discover_patterns(features)
            for pattern in patterns:
                features[f"pattern_{pattern['name']}"] = pattern["value"]

            # Удаление пропусков
            features.dropna(inplace=True)

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    async def _cleanup_old_metrics(self) -> None:
        """Очистка старых метрик"""
        try:
            async with self._metrics_lock:
                current_time = datetime.now()
                max_age = pd.Timedelta(days=self.config.max_metrics_age_days)

                for pair in list(self.metrics.keys()):
                    for timeframe in list(self.metrics[pair].keys()):
                        metric = self.metrics[pair][timeframe]
                        if current_time - metric.last_update > max_age:
                            del self.metrics[pair][timeframe]

                    if not self.metrics[pair]:
                        del self.metrics[pair]

        except Exception as e:
            logger.error(f"Error cleaning up old metrics: {str(e)}")

    async def _update_metrics(
        self,
        pair: str,
        timeframe: str,
        model: AdaptableModel,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> None:
        """
        Обновление метрик

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм
            model: Адаптированная модель
            features: Признаки
            target: Целевая переменная
        """
        try:
            async with self._metrics_lock:
                # Расчет метрик
                predictions = model.predict(features)
                accuracy = accuracy_score(target, predictions)
                precision = precision_score(target, predictions, average="weighted")
                recall = recall_score(target, predictions, average="weighted")
                f1 = f1_score(target, predictions, average="weighted")

                # Обновление метрик
                self.metrics[pair] = self.metrics.get(pair, {})
                self.metrics[pair][timeframe] = AdaptationMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    confidence=1.0,  # TODO: Добавить оценку уверенности
                    drift_score=0.0,  # TODO: Добавить оценку дрейфа
                    last_update=datetime.now(),
                    samples_count=len(self.data_buffer),
                    retrain_count=self.metrics.get(pair, {})
                    .get(timeframe, AdaptationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0))
                    .retrain_count,
                    error_count=self.metrics.get(pair, {})
                    .get(timeframe, AdaptationMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0))
                    .error_count,
                )

                # Сохранение метрик
                await self._save_metrics()

                # Очистка старых метрик
                await self._cleanup_old_metrics()

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    async def _save_metrics(self) -> None:
        """Сохранение метрик"""
        try:
            metrics_dir = Path(self.config.metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)

            metrics_file = metrics_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            metrics_dict = {
                pair: {
                    tf: {
                        "accuracy": m.accuracy,
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1": m.f1,
                        "confidence": m.confidence,
                        "drift_score": m.drift_score,
                        "last_update": m.last_update.isoformat(),
                        "samples_count": m.samples_count,
                        "retrain_count": m.retrain_count,
                        "error_count": m.error_count,
                    }
                    for tf, m in timeframes.items()
                }
                for pair, timeframes in self.metrics.items()
            }

            async with asyncio.Lock():
                with open(metrics_file, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    async def update(self, market_data: pd.DataFrame, performance: float) -> Dict[str, float]:
        """
        Обновление состояния адаптации

        Args:
            market_data: Рыночные данные
            performance: Производительность

        Returns:
            Dict[str, float]: Обновленные параметры
        """
        try:
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, market_data]).tail(
                self.config.max_samples
            )

            if len(self.data_buffer) < self.config.min_samples:
                return {}

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)

            # Обновление состояния
            state = AdaptationState(
                timestamp=datetime.now(),
                performance=performance,
                parameters=self._get_current_parameters(),
                market_conditions=features,
            )

            async with self._history_lock:
                self.adaptation_history.append(state)

            # Оптимизация параметров
            if len(self.adaptation_history) >= self.config.min_samples:
                new_params = await self._optimize_parameters()
                return new_params

            return self._get_current_parameters()

        except Exception as e:
            logger.error(f"Error updating adaptation: {str(e)}")
            return self._get_current_parameters()

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Извлечение признаков из данных

        Args:
            data: DataFrame с данными

        Returns:
            pd.DataFrame: Признаки
        """
        try:
            features = {}

            # Батчинг для больших наборов данных
            if len(data) > self.config.feature_batch_size:
                data = data.tail(self.config.feature_batch_size)

            # Волатильность
            features["volatility"] = float(data["close"].pct_change().std())

            # Тренд
            features["trend"] = float(data["close"].pct_change().mean())

            # Объем
            features["volume"] = float(data["volume"].mean())

            # RSI
            delta = data["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features["rsi"] = float(100 - (100 / (1 + rs)).iloc[-1])

            # MACD
            exp1 = data["close"].ewm(span=12, adjust=False).mean()
            exp2 = data["close"].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            features["macd"] = float(macd.iloc[-1])
            features["macd_signal"] = float(signal.iloc[-1])

            # Bollinger Bands
            bb_middle = data["close"].rolling(window=20).mean()
            bb_std = data["close"].rolling(window=20).std()
            bb_upper = bb_middle + (bb_std * 2)
            bb_lower = bb_middle - (bb_std * 2)
            features["bb_position"] = float(
                (data["close"].iloc[-1] - bb_lower.iloc[-1])
                / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
            )

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return {}

    def _get_current_parameters(self) -> Dict[str, float]:
        """
        Получение текущих параметров

        Returns:
            Dict[str, float]: Параметры
        """
        try:
            if not self.adaptation_history:
                return {}

            return self.adaptation_history[-1].parameters

        except Exception as e:
            logger.error(f"Error getting current parameters: {str(e)}")
            return {}

    async def _optimize_parameters(self) -> Dict[str, float]:
        """
        Оптимизация параметров

        Returns:
            Dict[str, float]: Оптимизированные параметры
        """
        try:
            # Подготовка данных
            async with self._history_lock:
                X = np.array([state.market_conditions for state in self.adaptation_history])
                y = np.array([state.performance for state in self.adaptation_history])

            # Обучение моделей
            predictions = {}
            for name, model in self.models.items():
                model.fit(X, y)
                predictions[name] = model.predict(X)

            # Взвешенное усреднение предсказаний
            weights = self._calculate_model_weights(predictions, y)
            ensemble_prediction = np.zeros_like(y)
            for name, pred in predictions.items():
                ensemble_prediction += weights[name] * pred

            # Оптимизация
            def objective(params):
                return -np.mean(ensemble_prediction)

            # Bayesian Optimization
            optimizer = bo.BayesianOptimization(
                f=objective,
                pbounds={
                    "volatility": (0.0, 1.0),
                    "trend": (-1.0, 1.0),
                    "volume": (0.0, 1.0),
                    "rsi": (0.0, 100.0),
                    "macd": (-1.0, 1.0),
                    "bb_position": (0.0, 1.0),
                },
            )

            optimizer.maximize(init_points=5, n_iter=self.config.bo_iterations)

            return optimizer.max["params"]

        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return self._get_current_parameters()

    def _calculate_model_weights(
        self, predictions: Dict[str, np.ndarray], y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Расчет весов моделей

        Args:
            predictions: Предсказания моделей
            y_true: Истинные значения

        Returns:
            Dict[str, float]: Веса моделей
        """
        try:
            weights = {}
            total_error = 0

            for name, pred in predictions.items():
                error = np.mean((pred - y_true) ** 2)
                weights[name] = 1.0 / (error + 1e-10)
                total_error += weights[name]

            # Нормализация весов
            for name in weights:
                weights[name] /= total_error

            return weights

        except Exception as e:
            logger.error(f"Error calculating model weights: {str(e)}")
            return {name: 1.0 / len(predictions) for name in predictions.keys()}

    def _individual_to_params(self, individual: List[float]) -> Dict:
        """Преобразование особи GA в параметры"""
        try:
            return {
                "learning_rate": individual[0],
                "batch_size": int(individual[1] * 100),
                "momentum": individual[2],
                "dropout": individual[3],
                "l1_ratio": individual[4],
                "l2_ratio": individual[5],
                "epsilon": individual[6],
                "beta1": individual[7],
                "beta2": individual[8],
                "gamma": individual[9],
            }

        except Exception as e:
            logger.error(f"Error converting individual to params: {str(e)}")
            return {}

    def _create_model_with_params(self, params: Dict) -> AdaptableModel:
        """Создание модели с параметрами"""
        try:
            return SGDClassifier(
                learning_rate="constant",
                eta0=params["learning_rate"],
                batch_size=params["batch_size"],
                momentum=params["momentum"],
                l1_ratio=params["l1_ratio"],
                l2_ratio=params["l2_ratio"],
                epsilon=params["epsilon"],
                beta1=params["beta1"],
                beta2=params["beta2"],
                gamma=params["gamma"],
            )

        except Exception as e:
            logger.error(f"Error creating model with params: {str(e)}")
            return None

    def _update_model_params(self, model: AdaptableModel, individual: List[float]):
        """Обновление параметров модели"""
        try:
            params = self._individual_to_params(individual)
            model.set_params(**params)

        except Exception as e:
            logger.error(f"Error updating model params: {str(e)}")

    def _load_state(self):
        """Загрузка состояния"""
        try:
            state_file = self.adaptation_dir / "state.json"
            if state_file.exists():
                with open(state_file, "r") as f:
                    state = json.load(f)
                    self.metrics_history = state.get("metrics_history", [])
                    self.metrics = state.get("metrics", {})
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния: {e}")

    def _save_state(self):
        """Сохранение состояния"""
        try:
            state_file = self.adaptation_dir / "state.json"
            state = {"metrics_history": self.metrics_history, "metrics": self.metrics}
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Расчет метрик"""
        try:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average="weighted")),
                "recall": float(recall_score(y_true, y_pred, average="weighted")),
                "f1": float(f1_score(y_true, y_pred, average="weighted")),
            }
        except Exception as e:
            logger.error(f"Ошибка расчета метрик: {e}")
            return {}

    def _detect_drift(self, current_metrics: Dict, historical_metrics: List[Dict]) -> float:
        """Обнаружение дрейфа"""
        try:
            if not historical_metrics:
                return 0.0

            # Расчет средних метрик
            avg_metrics = {
                "accuracy": np.mean([m["accuracy"] for m in historical_metrics]),
                "precision": np.mean([m["precision"] for m in historical_metrics]),
                "recall": np.mean([m["recall"] for m in historical_metrics]),
                "f1": np.mean([m["f1"] for m in historical_metrics]),
            }

            # Расчет отклонения
            drift = np.mean(
                [
                    abs(current_metrics["accuracy"] - avg_metrics["accuracy"]),
                    abs(current_metrics["precision"] - avg_metrics["precision"]),
                    abs(current_metrics["recall"] - avg_metrics["recall"]),
                    abs(current_metrics["f1"] - avg_metrics["f1"]),
                ]
            )

            return float(drift)

        except Exception as e:
            logger.error(f"Ошибка обнаружения дрейфа: {e}")
            return 0.0

    def update(self, df: pd.DataFrame, model_id: str, model: Any):
        """Обновление модели"""
        try:
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, df]).tail(self.config.max_samples)

            if len(self.data_buffer) < self.config.min_samples:
                return

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)

            # Подготовка данных
            X = features.values
            y = (self.data_buffer["close"].shift(-1) > self.data_buffer["close"]).values[:-1]
            X = X[:-1]  # Убираем последнюю строку, так как для нее нет целевой переменной

            # Нормализация
            if model_id not in self.scalers:
                self.scalers[model_id] = StandardScaler()
            X_scaled = self.scalers[model_id].fit_transform(X)

            # Обучение модели
            model.fit(X_scaled, y)
            self.models[model_id] = model

            # Расчет метрик
            y_pred = model.predict(X_scaled)
            metrics = self._calculate_metrics(y, y_pred)

            # Обновление истории метрик
            self.metrics_history.append(
                {"timestamp": datetime.now(), "model_id": model_id, **metrics}
            )

            # Ограничение истории
            self.metrics_history = self.metrics_history[-self.config.metrics_window :]

            # Обнаружение дрейфа
            drift_score = self._detect_drift(
                metrics, [m for m in self.metrics_history if m["model_id"] == model_id]
            )

            # Обновление метрик
            self.metrics[model_id] = AdaptationMetrics(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                confidence=1.0 - drift_score,
                drift_score=drift_score,
                last_update=datetime.now(),
                samples_count=len(self.data_buffer),
                retrain_count=self.metrics.get(model_id, {}).get("retrain_count", 0) + 1,
                error_count=self.metrics.get(model_id, {}).get("error_count", 0),
            ).__dict__

            # Сохранение состояния
            self._save_state()

            logger.info(f"Модель {model_id} обновлена. Метрики: {metrics}")

        except Exception as e:
            logger.error(f"Ошибка обновления модели: {e}")
            if model_id in self.metrics:
                self.metrics[model_id]["error_count"] += 1
            raise

    def predict(self, df: pd.DataFrame, model_id: str) -> Tuple[np.ndarray, float]:
        """Предсказание"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Модель {model_id} не найдена")

            # Извлечение признаков
            features = self._extract_features(df)

            # Нормализация
            X_scaled = self.scalers[model_id].transform(features.values)

            # Предсказание
            predictions = self.models[model_id].predict(X_scaled)
            probabilities = self.models[model_id].predict_proba(X_scaled)

            # Расчет уверенности
            confidence = float(np.mean(np.max(probabilities, axis=1)))

            return predictions, confidence

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            raise

    def get_metrics(self, model_id: Optional[str] = None) -> Dict:
        """Получение метрик"""
        if model_id:
            return self.metrics.get(model_id, {})
        return self.metrics

    def get_history(self, model_id: Optional[str] = None) -> List[Dict]:
        """Получение истории"""
        if model_id:
            return [m for m in self.metrics_history if m["model_id"] == model_id]
        return self.metrics_history

    def reset(self):
        """Сброс состояния"""
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []
        self.models = {}
        self.scalers = {}
        self.metrics = {}
        self._prediction_cache.clear()
        self._feature_cache.clear()

    def _evaluate_individual(self, individual):
        """Оценка индивидуума для генетического алгоритма"""
        try:
            # Преобразование индивидуума в параметры модели
            params = self._individual_to_params(individual)

            # Создание модели с параметрами
            model = self._create_model_with_params(params)
            if model is None:
                return (-float("inf"),)

            # Подготовка данных для оценки
            if len(self.data_buffer) < self.config.min_samples:
                return (-float("inf"),)

            features = self._extract_features(self.data_buffer)
            X = features.values
            y = (self.data_buffer["close"].shift(-1) > self.data_buffer["close"]).values[:-1]
            X = X[:-1]  # Убираем последнюю строку

            # Нормализация
            X_scaled = self.scaler.fit_transform(X)

            # Обучение и оценка модели
            model.fit(X_scaled, y)
            predictions = model.predict(X_scaled)

            # Расчет метрик
            accuracy = accuracy_score(y, predictions)
            precision = precision_score(y, predictions)
            recall = recall_score(y, predictions)
            f1 = f1_score(y, predictions)

            # Общий скор как среднее метрик
            score = (accuracy + precision + recall + f1) / 4

            return (score,)

        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return (-float("inf"),)


class LiveAdaptation:
    """Класс для адаптации моделей в реальном времени"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "adaptation_window": 1000,
            "min_samples": 100,
            "update_interval": 100,
            "confidence_threshold": 0.7,
        }
        self.adaptation_state = {}
        self._last_update = None

    def update_state(self, data: pd.DataFrame, trades: List[Dict]) -> Dict[str, Any]:
        """
        Обновление состояния адаптации

        Args:
            data: Рыночные данные
            trades: История сделок

        Returns:
            Dict[str, Any]: Обновленное состояние адаптации
        """
        try:
            if len(data) < self.config["min_samples"]:
                return self.adaptation_state

            # Расчет метрик
            volatility = data["close"].pct_change().std()
            trend = self._calculate_trend(data)
            market_regime = self._detect_regime(data)

            # Обновление состояния
            self.adaptation_state = {
                "volatility": volatility,
                "trend": trend,
                "market_regime": market_regime,
                "position_size_factor": self._calculate_position_size_factor(volatility, trend),
                "stop_loss_factor": self._calculate_stop_loss_factor(volatility),
                "take_profit_factor": self._calculate_take_profit_factor(trend),
                "commission_factor": self._calculate_commission_factor(volatility),
                "slippage_factor": self._calculate_slippage_factor(volatility),
                "last_update": datetime.now(),
            }

            self._last_update = datetime.now()
            return self.adaptation_state

        except Exception as e:
            logger.error(f"Error updating adaptation state: {str(e)}")
            return self.adaptation_state

    def _calculate_trend(self, data: pd.DataFrame) -> float:
        """Расчет тренда"""
        try:
            returns = data["close"].pct_change()
            return returns.mean() / returns.std() if returns.std() != 0 else 0
        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            return 0

    def _detect_regime(self, data: pd.DataFrame) -> str:
        """Определение режима рынка"""
        try:
            volatility = data["close"].pct_change().std()
            trend = self._calculate_trend(data)

            if abs(trend) > 0.5:
                return "trend"
            elif volatility > 0.02:
                return "volatile"
            elif abs(trend) < 0.1:
                return "sideways"
            else:
                return "normal"

        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return "normal"

    def _calculate_position_size_factor(self, volatility: float, trend: float) -> float:
        """Расчет фактора размера позиции"""
        try:
            base_factor = 1.0

            # Корректировка по волатильности
            if volatility > 0.02:
                base_factor *= 0.8
            elif volatility < 0.005:
                base_factor *= 1.2

            # Корректировка по тренду
            if abs(trend) > 0.5:
                base_factor *= 1.2
            elif abs(trend) < 0.1:
                base_factor *= 0.8

            return max(0.5, min(1.5, base_factor))

        except Exception as e:
            logger.error(f"Error calculating position size factor: {str(e)}")
            return 1.0

    def _calculate_stop_loss_factor(self, volatility: float) -> float:
        """Расчет фактора стоп-лосса"""
        try:
            if volatility > 0.02:
                return 1.5
            elif volatility < 0.005:
                return 0.8
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating stop loss factor: {str(e)}")
            return 1.0

    def _calculate_take_profit_factor(self, trend: float) -> float:
        """Расчет фактора тейк-профита"""
        try:
            if abs(trend) > 0.5:
                return 1.5
            elif abs(trend) < 0.1:
                return 0.8
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating take profit factor: {str(e)}")
            return 1.0

    def _calculate_commission_factor(self, volatility: float) -> float:
        """Расчет фактора комиссии"""
        try:
            if volatility > 0.02:
                return 1.2
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating commission factor: {str(e)}")
            return 1.0

    def _calculate_slippage_factor(self, volatility: float) -> float:
        """Расчет фактора проскальзывания"""
        try:
            if volatility > 0.02:
                return 1.5
            elif volatility < 0.005:
                return 0.8
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating slippage factor: {str(e)}")
            return 1.0
