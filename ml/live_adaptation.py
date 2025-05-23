import asyncio
import glob
import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import (Any, Deque, Dict, List, Optional, Protocol, Tuple, Union,
                    runtime_checkable)

import aiofiles
import numpy as np
import pandas as pd
import ta
from deap import algorithms, base, creator, tools
from loguru import logger
from scipy.stats import entropy
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from skopt import bayesian_optimization as bo

from ml.model_selector import ModelSelector
from ml.pattern_discovery import PatternDiscovery
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


# Интерфейсы для тестирования
class MarketStateInterface(Protocol):
    """Интерфейс состояния рынка"""
    
    def initialize_ml(self) -> None:
        """Инициализация ML компонентов"""
        ...

    def get_state(self) -> Dict[str, Any]:
        """Получение текущего состояния"""
        ...


@runtime_checkable
class AdaptableModel(Protocol):
    """Протокол для адаптируемых моделей"""

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray: ...
    def partial_fit(
        self, X: np.ndarray, y: np.ndarray, classes: Optional[np.ndarray] = None
    ) -> None: ...
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def get_params(self) -> Dict[str, Union[float, int, str]]: ...
    def set_params(self, **params: Any) -> None: ...


class ModelSelectorInterface(Protocol):
    """Интерфейс выбора моделей"""
    
    def select_model(self, state: Dict[str, Any]) -> str:
        """Выбор модели"""
        ...

    def update_weights(self, model_id: str, performance: float) -> None:
        """Обновление весов"""
        ...


class PatternDiscoveryInterface(Protocol):
    """Интерфейс обнаружения паттернов"""
    
    def discover_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение паттернов"""
        ...

    def update_patterns(self, new_patterns: List[Dict[str, Any]]) -> None:
        """Обновление паттернов"""
        ...


@dataclass
class AdaptationConfig:
    """Конфигурация адаптации"""

    learning_rate: float = 0.01
    max_iter: int = 1000
    tol: float = 1e-3
    min_samples: int = 100
    max_metrics_age_days: int = 7
    history_size: int = 1000
    drift_threshold: float = 0.1
    confidence_threshold: float = 0.8
    retrain_interval: int = 1000
    save_interval: int = 100
    load_interval: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "min_samples": self.min_samples,
            "max_metrics_age_days": self.max_metrics_age_days,
            "history_size": self.history_size,
            "drift_threshold": self.drift_threshold,
            "confidence_threshold": self.confidence_threshold,
            "retrain_interval": self.retrain_interval,
            "save_interval": self.save_interval,
            "load_interval": self.load_interval,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptationConfig":
        """Создание из словаря"""
        return cls(
            learning_rate=float(data["learning_rate"]),
            max_iter=int(data["max_iter"]),
            tol=float(data["tol"]),
            min_samples=int(data["min_samples"]),
            max_metrics_age_days=int(data["max_metrics_age_days"]),
            history_size=int(data["history_size"]),
            drift_threshold=float(data["drift_threshold"]),
            confidence_threshold=float(data["confidence_threshold"]),
            retrain_interval=int(data["retrain_interval"]),
            save_interval=int(data["save_interval"]),
            load_interval=int(data["load_interval"]),
        )


@dataclass
class AdaptationMetrics:
    """Метрики адаптации"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    drift_score: float
    confidence: float
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "drift_score": self.drift_score,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptationMetrics":
        """Создание из словаря"""
        return cls(
            accuracy=float(data["accuracy"]),
            precision=float(data["precision"]),
            recall=float(data["recall"]),
            f1=float(data["f1"]),
            drift_score=float(data["drift_score"]),
            confidence=float(data["confidence"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class AdaptationHistory:
    """История адаптации"""

    metrics: AdaptationMetrics
    model_id: str
    parameters: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь"""
        return {
            "metrics": self.metrics.to_dict(),
            "model_id": self.model_id,
            "parameters": self.parameters,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptationHistory":
        """Создание из словаря"""
        return cls(
            metrics=AdaptationMetrics.from_dict(data["metrics"]),
            model_id=str(data["model_id"]),
            parameters=dict(data["parameters"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class MarketState(MarketStateInterface):
    """Состояние рынка"""

    def __init__(self):
        """Инициализация"""
        self.timestamp = datetime.now()
        self.price = 0.0
        self.volume = 0.0
        self.volatility = 0.0
        self.trend = ""
        self.indicators = {}
        self.market_regime = ""
        self.liquidity = 0.0
        self.momentum = 0.0
        self.sentiment = 0.0
        self.support_levels = []
        self.resistance_levels = []
        self.market_depth = {}
        self.correlation_matrix = np.array([])
        self.market_impact = 0.0
        self.volume_profile = {}

    def initialize_ml(self) -> None:
        """Инициализация ML компонентов"""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Получение текущего состояния"""
        return {
            "timestamp": self.timestamp,
            "price": self.price,
            "volume": self.volume,
            "volatility": self.volatility,
            "trend": self.trend,
            "indicators": self.indicators,
            "market_regime": self.market_regime,
            "liquidity": self.liquidity,
            "momentum": self.momentum,
            "sentiment": self.sentiment,
            "support_levels": self.support_levels,
            "resistance_levels": self.resistance_levels,
            "market_depth": self.market_depth,
            "correlation_matrix": self.correlation_matrix.tolist(),
            "market_impact": self.market_impact,
            "volume_profile": self.volume_profile,
        }


class LiveAdaptationModel:
    """Модель адаптации в реальном времени"""

    def __init__(
        self,
        market_state: MarketStateInterface,
        model_selector: ModelSelectorInterface,
        pattern_discovery: PatternDiscoveryInterface,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Инициализация"""
        self.market_state = market_state
        self.model_selector = model_selector
        self.pattern_discovery = pattern_discovery
        self.config = config or {}

        # Инициализация ML компонентов
        self.market_state.initialize_ml()

        # Инициализация моделей
        self.models: Dict[str, AdaptableModel] = {}
        self.model_weights: Dict[str, float] = {}
        self.metrics: Dict[str, AdaptationMetrics] = {}
        self.adaptation_history: List[AdaptationMetrics] = []

        # Инициализация нормализации
        self.scaler = StandardScaler()

        # Инициализация онлайн-классификатора
        self.online_classifier = SGDClassifier(
            learning_rate="constant",
            eta0=self.config.get("learning_rate", 0.01),
            max_iter=self.config.get("max_iter", 1000),
            tol=self.config.get("tol", 1e-3),
        )

        # Инициализация GA
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_float", np.random.uniform, -1, 1)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_float,
            n=10,
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual
        )

        self.toolbox.register("evaluate", self._evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register(
            "mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2
        )
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Инициализация BO
        self.bo = bo.BayesianOptimization(
            f=self._evaluate_bo,
            pbounds={
                "learning_rate": (0.001, 0.1),
                "max_iter": (100, 1000),
                "tol": (1e-4, 1e-2),
            },
        )

        # Инициализация блокировок
        self.metrics_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        self.history_lock = asyncio.Lock()

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
            self.online_classifier: AdaptableModel = SGDClassifier(
                learning_rate="constant",
                eta0=self.config.learning_rate,
                batch_size=self.config.batch_size,
                momentum=self.config.momentum,
            )

            # Инициализация нормализации
            self.scaler = StandardScaler()

            # Инициализация GA
            if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

            self.toolbox = base.Toolbox()
            self.toolbox.register("attr_float", np.random.uniform, -1, 1)
            self.toolbox.register(
                "individual",
                tools.initRepeat,
                creator.Individual,
                self.toolbox.attr_float,
                n=10,
            )
            self.toolbox.register(
                "population", tools.initRepeat, list, self.toolbox.individual
            )

            self.toolbox.register("evaluate", self._evaluate_individual)
            self.toolbox.register("mate", tools.cxTwoPoint)
            self.toolbox.register(
                "mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2
            )
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
            # Удаление старых файлов метрик
            metrics_files = glob.glob(
                os.path.join(self.config.metrics_dir, "metrics_*.json")
            )
            for file in metrics_files:
                file_time = datetime.fromtimestamp(os.path.getctime(file))
                if (datetime.now() - file_time).days > self.config.max_metrics_age_days:
                    os.remove(file)

            # Удаление старых метрик из памяти
            current_time = datetime.now()
                for pair in list(self.metrics.keys()):
                    for timeframe in list(self.metrics[pair].keys()):
                    metrics = self.metrics[pair][timeframe]
                    if (current_time - metrics.timestamp).days > self.config.max_metrics_age_days:
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
        """Обновление метрик"""
        try:
            async with self.metrics_lock:
                # Расчет метрик
                predictions = model.predict(features)
                accuracy = float(accuracy_score(target, predictions))
                precision = float(precision_score(target, predictions, average="weighted"))
                recall = float(recall_score(target, predictions, average="weighted"))
                f1 = float(f1_score(target, predictions, average="weighted"))

                # Обновление метрик
                self.metrics[pair] = self.metrics.get(pair, {})
                self.metrics[pair][timeframe] = AdaptationMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    drift_score=0.0,  # TODO: Добавить оценку дрейфа
                    confidence=1.0,  # TODO: Добавить оценку уверенности
                    timestamp=datetime.now(),
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
            metrics_path = os.path.join(
                self.config.metrics_dir,
                f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            # Создание директории если не существует
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

            # Преобразование метрик в словарь
            metrics_dict = {}
            for pair, timeframes in self.metrics.items():
                metrics_dict[pair] = {}
                for timeframe, metrics in timeframes.items():
                    metrics_dict[pair][timeframe] = metrics.to_dict()

            # Сохранение метрик
            with open(metrics_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")

    async def _load_metrics(self) -> None:
        """Загрузка метрик"""
        try:
            # Поиск последнего файла метрик
            metrics_files = glob.glob(
                os.path.join(self.config.metrics_dir, "metrics_*.json")
            )
            if not metrics_files:
                return

            latest_metrics = max(metrics_files, key=os.path.getctime)

            # Загрузка метрик
            with open(latest_metrics, "r") as f:
                metrics_dict = json.load(f)

            # Преобразование метрик в объекты
            for pair, timeframes in metrics_dict.items():
                self.metrics[pair] = {}
                for timeframe, metrics in timeframes.items():
                    self.metrics[pair][timeframe] = AdaptationMetrics.from_dict(metrics)

        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")

    async def update(
        self,
        market_data: pd.DataFrame,
        performance: float,
        pair: str,
        timeframe: str,
    ) -> None:
        """Обновление адаптации"""
        try:
            # Проверка входных данных
            if market_data.empty:
                logger.warning("Empty market data received")
                return

            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, market_data])
            if len(self.data_buffer) > self.config.max_buffer_size:
                self.data_buffer = self.data_buffer.tail(self.config.max_buffer_size)

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)
            if features.empty:
                logger.warning("Failed to extract features")
                return

            # Подготовка целевой переменной
            target = (self.data_buffer["close"].shift(-1) > self.data_buffer["close"]).astype(int)
            target = target[:-1]  # Удаляем последнее значение, так как у нас нет следующей цены
            features = features[:-1]  # Удаляем последнюю строку признаков

            # Загрузка или создание модели
            model = await self._load_model(pair, timeframe)
            if model is None:
                model = SGDClassifier(
                    learning_rate="constant",
                    eta0=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    momentum=self.config.momentum,
                )

            # Адаптация модели
            await self._adapt_model(pair, timeframe, model, features, target)

            # Обновление метрик
            await self._update_metrics(pair, timeframe, model, features, target)

            # Сохранение метрик
            await self._save_metrics()

        except Exception as e:
            logger.error(f"Error updating adaptation: {str(e)}")

    async def predict(
        self,
        market_data: pd.DataFrame,
        pair: str,
        timeframe: str,
    ) -> Tuple[float, float]:
        """Получение предсказания"""
        try:
            # Проверка входных данных
            if market_data.empty:
                logger.warning("Empty market data received")
                return 0.0, 0.0

            # Извлечение признаков
            features = self._extract_features(market_data)
            if features.empty:
                logger.warning("Failed to extract features")
                return 0.0, 0.0

            # Загрузка модели
            model = await self._load_model(pair, timeframe)
            if model is None:
                logger.warning("No model found")
                return 0.0, 0.0

            # Получение предсказания
            prediction = model.predict_proba(features.values)[-1]
            confidence = float(max(prediction))

            # Определение направления
            direction = 1.0 if prediction[1] > prediction[0] else -1.0

            return direction, confidence

        except Exception as e:
            logger.error(f"Error getting prediction: {str(e)}")
            return 0.0, 0.0

    async def get_metrics(
        self,
        pair: str,
        timeframe: str,
    ) -> Optional[AdaptationMetrics]:
        """Получение метрик"""
        try:
            return self.metrics.get(pair, {}).get(timeframe)

        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return None

    def _extract_features(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        """Извлечение признаков из данных"""
        try:
            features = {}
            
            # Технические индикаторы
            features["rsi"] = ta.momentum.RSIIndicator(data["close"]).rsi().iloc[-1]
            features["macd"] = ta.trend.MACD(data["close"]).macd().iloc[-1]
            features["bb_upper"] = ta.volatility.BollingerBands(data["close"]).bollinger_hband().iloc[-1]
            features["bb_lower"] = ta.volatility.BollingerBands(data["close"]).bollinger_lband().iloc[-1]
            
            # Волатильность
            features["volatility"] = data["close"].pct_change().std()
            
            # Объемы
            features["volume_ma"] = data["volume"].rolling(window=20).mean().iloc[-1]
            features["volume_std"] = data["volume"].rolling(window=20).std().iloc[-1]
            
            # Ценовые метрики
            features["price_ma"] = data["close"].rolling(window=20).mean().iloc[-1]
            features["price_std"] = data["close"].rolling(window=20).std().iloc[-1]

            return features

        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return None

    def _get_current_parameters(self) -> Dict[str, float]:
        """Получение текущих параметров"""
        try:
            if not self.adaptation_history:
                return {}

            return self.adaptation_history[-1].parameters

        except Exception as e:
            logger.error(f"Error getting current parameters: {str(e)}")
            return {}

    async def _optimize_parameters(self, model: AdaptableModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Оптимизация параметров"""
        try:
            # Обучение моделей
                model.fit(X, y)

            # Получение параметров модели
            params = model.get_params()

            return params

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

    def _calculate_metrics(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
        """Расчет метрик качества"""
        try:
            # Получение предсказаний
            y_pred = self._ensemble_predict(X)

            # Расчет метрик
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average="weighted")
            recall = recall_score(y, y_pred, average="weighted")
            f1 = f1_score(y, y_pred, average="weighted")

            return accuracy, precision, recall, f1

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return 0.0, 0.0, 0.0, 0.0

    def _calculate_drift_score(self, X: np.ndarray) -> float:
        """Расчет показателя дрейфа"""
        try:
            # Расчет средних значений
            mean_current = np.mean(X, axis=0)
            mean_historical = np.mean(self.historical_data, axis=0)

            # Расчет ковариационных матриц
            cov_current = np.cov(X.T)
            cov_historical = np.cov(self.historical_data.T)

            # Расчет расстояния между распределениями
            mean_diff = np.linalg.norm(mean_current - mean_historical)
            cov_diff = np.linalg.norm(cov_current - cov_historical)

            # Нормализация метрик
            mean_score = 1.0 / (1.0 + mean_diff)
            cov_score = 1.0 / (1.0 + cov_diff)

            # Комбинирование метрик
            drift_score = 0.5 * mean_score + 0.5 * cov_score

            return drift_score

        except Exception as e:
            logger.error(f"Error calculating drift score: {str(e)}")
                return 0.0

    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Расчет уверенности в предсказаниях"""
        try:
            # Получение предсказаний от всех моделей
            predictions = []
            for model in self.models.values():
                pred = model.predict(X)
                predictions.append(pred)

            # Расчет точности предсказаний
            accuracy = np.mean([np.std(pred) for pred in predictions])

            # Расчет энтропии предсказаний
            entropy_value = np.mean([entropy(pred) for pred in predictions])

            # Нормализация энтропии
            normalized_entropy = 1.0 - (entropy_value / np.log(len(predictions)))

            # Комбинирование метрик
            confidence = 0.7 * accuracy + 0.3 * normalized_entropy

            return confidence

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def _calculate_performance(self) -> float:
        """Расчет общей производительности"""
        try:
            # Расчет всех метрик
            accuracy, precision, recall, f1 = self._calculate_metrics(
                self.current_data, self.target_data
            )
            drift_score = self._calculate_drift_score(self.current_data)
            confidence = self._calculate_confidence(self.current_data)

            # Взвешенная сумма метрик
            performance = (
                0.3 * accuracy
                + 0.2 * precision
                + 0.2 * recall
                + 0.1 * f1
                + 0.1 * drift_score
                + 0.1 * confidence
            )

            return performance

        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return 0.0

    def update(self, df: pd.DataFrame, model_id: str, model: Any):
        """Обновление модели"""
        try:
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, df]).tail(
                self.config.max_samples
            )

            if len(self.data_buffer) < self.config.min_samples:
                return

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)

            # Подготовка данных
            X = features.values
            y = (
                self.data_buffer["close"].shift(-1) > self.data_buffer["close"]
            ).values[:-1]
            X = X[
                :-1
            ]  # Убираем последнюю строку, так как для нее нет целевой переменной

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
            drift_score = self._calculate_drift_score(X_scaled)

            # Обновление метрик
            self.metrics[model_id] = AdaptationMetrics(
                accuracy=metrics[0],
                precision=metrics[1],
                recall=metrics[2],
                f1=metrics[3],
                drift_score=drift_score,
                confidence=self._calculate_confidence(X_scaled),
                timestamp=datetime.now(),
            )

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

    def _evaluate_individual(self, individual: List[float]) -> Tuple[float,]:
        """Оценка особи в GA"""
        try:
            # Преобразование параметров
            params = {
                "learning_rate": individual[0],
                "max_iter": int(individual[1] * 1000),
                "tol": individual[2],
            }
            
            # Обучение модели
            model = SGDClassifier(**params)
            model.fit(self.current_data, self.target_data)
            
            # Оценка качества
            score = model.score(self.current_data, self.target_data)
            
            return (score,)
            
        except Exception as e:
            logger.error(f"Error evaluating individual: {str(e)}")
            return (0.0,)

    def _evaluate_bo(self, **params: float) -> float:
        """Оценка параметров в BO"""
        try:
            # Преобразование параметров
            model_params = {
                "learning_rate": params["learning_rate"],
                "max_iter": int(params["max_iter"]),
                "tol": params["tol"],
            }
            
            # Обучение модели
            model = SGDClassifier(**model_params)
            model.fit(self.current_data, self.target_data)
            
            # Оценка качества
            score = model.score(self.current_data, self.target_data)
            
            return score

        except Exception as e:
            logger.error(f"Error evaluating BO parameters: {str(e)}")
            return 0.0

    def _optimize_parameters(self) -> Dict[str, float]:
        """Оптимизация параметров"""
        try:
            # Оптимизация с помощью GA
            pop = self.toolbox.population(n=50)
            result, _ = algorithms.eaSimple(
                pop,
                self.toolbox,
                cxpb=0.7,
                mutpb=0.3,
                ngen=10,
                verbose=False,
            )
            
            best_individual = tools.selBest(result, k=1)[0]
            ga_params = {
                "learning_rate": best_individual[0],
                "max_iter": int(best_individual[1] * 1000),
                "tol": best_individual[2],
            }
            
            # Оптимизация с помощью BO
            self.bo.maximize(init_points=5, n_iter=10)
            bo_params = self.bo.max["params"]
            
            # Комбинирование результатов
            final_params = {
                "learning_rate": (ga_params["learning_rate"] + bo_params["learning_rate"]) / 2,
                "max_iter": int((ga_params["max_iter"] + bo_params["max_iter"]) / 2),
                "tol": (ga_params["tol"] + bo_params["tol"]) / 2,
            }
            
            return final_params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}")
            return {
                "learning_rate": 0.01,
                "max_iter": 1000,
                "tol": 1e-3,
            }

    async def _adapt_model(
        self, model_id: str, features: pd.DataFrame, target: pd.Series
    ) -> None:
        """Адаптация модели"""
        try:
            # Получение модели
            model = self.models.get(model_id)
            if model is None:
                logger.error(f"Model {model_id} not found")
                return

            # Подготовка данных
            X = features.values
            y = target.values

            # Оптимизация параметров
            params = self._optimize_parameters()

            # Обновление параметров модели
            model.set_params(**params)

            # Обучение модели
            model.fit(X, y)

            # Обновление метрик
            metrics = self._calculate_metrics(X, y)
            drift_score = self._calculate_drift_score(X)

            # Обновление метрик
            self.metrics[model_id] = AdaptationMetrics(
                accuracy=metrics[0],
                precision=metrics[1],
                recall=metrics[2],
                f1=metrics[3],
                drift_score=drift_score,
                confidence=self._calculate_confidence(X),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Error adapting model {model_id}: {str(e)}")
            raise

    async def _update_model_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обновление весов моделей"""
        try:
            total_error = 0.0
            errors = {}

            for name, model in self.models.items():
                pred = model.predict(X)
                error = np.mean(np.abs(pred - y))
                errors[name] = error
                total_error += error

            if total_error > 0:
                for name in self.models:
                    self.model_weights[name] = 1 - (errors[name] / total_error)
            else:
                for name in self.models:
                    self.model_weights[name] = 1.0 / len(self.models)

        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")
            raise

    async def _update_metrics(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обновление метрик"""
        try:
            async with self.metrics_lock:
                # Расчет метрик
                accuracy, precision, recall, f1 = self._calculate_metrics(X, y)
                drift_score = self._calculate_drift_score(X)
                confidence = self._calculate_confidence(X)

                # Обновление метрик
                self.metrics = AdaptationMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    drift_score=drift_score,
                    confidence=confidence,
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise

    async def save_state(self, path: str) -> None:
        """Сохранение состояния"""
        try:
            state = {
                "models": {
                    name: {
                        "model": model,
                        "weights": self.model_weights[name],
                    }
                    for name, model in self.models.items()
                },
                "metrics": self.metrics,
                "history": self.adaptation_history,
                "scaler": self.scaler,
                "config": self.config,
            }

            async with aiofiles.open(path, "wb") as f:
                await f.write(pickle.dumps(state))

        except Exception as e:
            logger.error(f"Error saving state: {str(e)}")
            raise

    async def load_state(self, path: str) -> None:
        """Загрузка состояния"""
        try:
            async with aiofiles.open(path, "rb") as f:
                state = pickle.loads(await f.read())

            self.models = state["models"]
            self.metrics = state["metrics"]
            self.adaptation_history = state["history"]
            self.scaler = state["scaler"]
            self.config = state["config"]

        except Exception as e:
            logger.error(f"Error loading state: {str(e)}")
            raise


class LiveAdaptation:
    """Адаптация в реальном времени"""

    def __init__(self, config: Optional[AdaptationConfig] = None):
        """Инициализация"""
        self.config = config or AdaptationConfig()
        self.adaptation_dir = Path("adaptation")
        self.adaptation_dir.mkdir(parents=True, exist_ok=True)

        # Данные
        self.data_buffer = pd.DataFrame()
        self.metrics_history = []

        # Модели
        self.models: Dict[str, AdaptableModel] = {}
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
        self.adaptation_history: List[AdaptationHistory] = []

        # Блокировки для потокобезопасности
        self._metrics_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()
        self._history_lock = asyncio.Lock()

    async def update(self, data: pd.DataFrame) -> None:
        """Обновление адаптации"""
        try:
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, data])
            if len(self.data_buffer) > self.config.min_samples:
                self.data_buffer = self.data_buffer.iloc[-self.config.min_samples:]

            # Извлечение признаков
            features = self._extract_features(self.data_buffer)
            if features is None:
                return

            # Подготовка данных
            X, y = self._prepare_data(features)
            if X is None or y is None:
                return

            # Обучение моделей
            await self._train_ensemble(X, y)

            # Обновление весов
            await self._update_model_weights(X, y)

            # Обновление метрик
            await self._update_metrics(X, y)

            # Обновление истории
            await self._update_history()

        except Exception as e:
            logger.error(f"Error updating adaptation: {str(e)}")
            raise

    async def _train_ensemble(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обучение ансамбля моделей"""
        try:
            for name, model in self.models.items():
                model.fit(X, y)
                logger.info(f"Trained {name} model")

        except Exception as e:
            logger.error(f"Error training ensemble: {str(e)}")
            raise

    async def _update_model_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обновление весов моделей"""
        try:
            total_error = 0.0
            errors = {}

            for name, model in self.models.items():
                pred = model.predict(X)
                error = np.mean(np.abs(pred - y))
                errors[name] = error
                total_error += error

            if total_error > 0:
                for name in self.models:
                    self.model_weights[name] = 1 - (errors[name] / total_error)
            else:
                for name in self.models:
                    self.model_weights[name] = 1.0 / len(self.models)

        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")
            raise

    async def _update_metrics(self, X: np.ndarray, y: np.ndarray) -> None:
        """Обновление метрик"""
        try:
            async with self._metrics_lock:
                # Расчет метрик
                accuracy, precision, recall, f1 = self._calculate_metrics(X, y)
                drift_score = self._calculate_drift_score(X)
                confidence = self._calculate_confidence(X)

                # Обновление метрик
                self.metrics = AdaptationMetrics(
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    drift_score=drift_score,
                    confidence=confidence,
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise

    async def _update_history(self) -> None:
        """Обновление истории адаптации"""
        try:
            async with self._history_lock:
                self.adaptation_history.append(self.metrics)
                if len(self.adaptation_history) > self.config.history_size:
                    self.adaptation_history.pop(0)

        except Exception as e:
            logger.error(f"Error updating history: {str(e)}")
            raise
