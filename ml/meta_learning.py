import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import talib
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

from ml.dataset_manager import DatasetManager


@dataclass
class MetaConfig:
    """Конфигурация мета-обучения"""

    min_samples: int = 100
    max_samples: int = 1000
    update_interval: int = 1  # часов
    cache_size: int = 1000
    compression: bool = True
    metrics_window: int = 24  # часов
    ensemble_size: int = 3
    feature_importance_threshold: float = 0.1
    n_trials: int = 100
    cv_splits: int = 5
    early_stopping_rounds: int = 10
    learning_rate: float = 0.01
    max_depth: int = 5
    min_samples_split: int = 2
    min_samples_leaf: int = 1


@dataclass
class MetaMetrics:
    """Метрики мета-обучения"""

    accuracy: float
    precision: float
    recall: float
    f1: float
    confidence: float
    last_update: datetime
    samples_count: int
    model_count: int
    error_count: int
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]


@dataclass
class ModelContext:
    """Контекст модели для метаобучения"""

    model_id: str
    performance: float
    parameters: Dict[str, Any]
    features: pd.DataFrame
    predictions: np.ndarray
    confidence: float
    market_regime: str
    timestamp: pd.Timestamp


class MetaLearner:
    """Мета-обучение для адаптации моделей"""

    def __init__(self, config: Optional[MetaConfig] = None):
        """Инициализация мета-обучения"""
        self.config = config or MetaConfig()
        self.meta_dir = Path("meta")
        self.meta_dir.mkdir(parents=True, exist_ok=True)

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

        # Блокировки
        self._model_lock = Lock()
        self._metrics_lock = Lock()

        # Загрузка состояния
        self._load_state()

    def _load_state(self):
        """Загрузка состояния"""
        try:
            state_file = self.meta_dir / "state.json"
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
            state_file = self.meta_dir / "state.json"
            state = {"metrics_history": self.metrics_history, "metrics": self.metrics}
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Извлечение признаков"""
        try:
            features = pd.DataFrame()

            # Ценовые признаки
            features["returns"] = df["close"].pct_change()
            features["log_returns"] = np.log1p(features["returns"])
            features["volatility"] = features["returns"].rolling(20).std()

            # Технические индикаторы
            features["rsi"] = talib.RSI(df["close"].values)
            features["macd"], features["macd_signal"], _ = talib.MACD(
                df["close"].values
            )
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = (
                talib.BBANDS(df["close"].values)
            )
            features["atr"] = talib.ATR(
                df["high"].values, df["low"].values, df["close"].values
            )
            features["adx"] = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values
            )

            # Объемные признаки
            features["volume_ma"] = df["volume"].rolling(20).mean()
            features["volume_std"] = df["volume"].rolling(20).std()
            features["volume_ratio"] = df["volume"] / features["volume_ma"]

            # Моментум
            features["momentum"] = talib.MOM(df["close"].values, timeperiod=10)
            features["roc"] = talib.ROC(df["close"].values, timeperiod=10)

            # Волатильность
            features["high_low_ratio"] = df["high"] / df["low"]
            features["close_open_ratio"] = df["close"] / df["open"]

            # Тренд
            features["trend"] = talib.ADX(
                df["high"].values, df["low"].values, df["close"].values
            )
            features["trend_strength"] = abs(features["trend"])

            # Нормализация
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features.fillna(0)),
                columns=features.columns,
                index=features.index,
            )

            return features_scaled

        except Exception as e:
            logger.error(f"Ошибка извлечения признаков: {e}")
            return pd.DataFrame()

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

    def _get_feature_importance(self, model_id: str) -> Dict[str, float]:
        """Получение важности признаков"""
        try:
            if model_id not in self.models:
                return {}

            model = self.models[model_id]
            if not hasattr(model, "feature_importances_"):
                return {}

            return {
                feature: float(importance)
                for feature, importance in zip(
                    model.feature_names_in_, model.feature_importances_
                )
            }
        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {e}")
            return {}

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Оптимизация гиперпараметров"""
        try:

            def objective(trial):
                # Параметры для оптимизации
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                }

                # Создание модели
                model = GradientBoostingClassifier(**params)

                # Кросс-валидация
                scores = []
                for i in range(self.config.cv_splits):
                    # Разделение данных
                    split_size = len(X) // self.config.cv_splits
                    start_idx = i * split_size
                    end_idx = (i + 1) * split_size

                    X_train = np.concatenate([X[:start_idx], X[end_idx:]])
                    y_train = np.concatenate([y[:start_idx], y[end_idx:]])
                    X_val = X[start_idx:end_idx]
                    y_val = y[start_idx:end_idx]

                    # Обучение и оценка
                    model.fit(X_train, y_train)
                    score = model.score(X_val, y_val)
                    scores.append(score)

                return np.mean(scores)

            # Оптимизация
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.config.n_trials)

            return study.best_params

        except Exception as e:
            logger.error(f"Ошибка оптимизации гиперпараметров: {e}")
            return {}

    def update(self, df: pd.DataFrame, model_id: str, model: Any):
        """Обновление модели с проверкой входных данных"""
        try:
            # Проверка на пустой датасет
            if df is None or df.empty:
                logger.error(f"Пустой датасет для update, model_id={model_id}")
                return
            # Проверка на NaN/None
            if df.isnull().values.any() or (
                df.applymap(lambda x: x is None).values.any()
            ):
                logger.error(
                    f"Датасет содержит NaN или None для update, model_id={model_id}"
                )
                return
            # Добавление данных в буфер
            self.data_buffer = pd.concat([self.data_buffer, df]).tail(
                self.config.max_samples
            )
            if len(self.data_buffer) < self.config.min_samples:
                return
            # Извлечение признаков
            features = self._extract_features(self.data_buffer)
            # Проверка признаков
            if (
                features.empty
                or features.isnull().values.any()
                or (features.applymap(lambda x: x is None).values.any())
            ):
                logger.error(
                    f"Признаки пусты или содержат NaN/None для update, model_id={model_id}"
                )
                return
            # Подготовка данных
            X = features.values
            y = (
                self.data_buffer["close"].shift(-1) > self.data_buffer["close"]
            ).values[:-1]
            X = X[
                :-1
            ]  # Убираем последнюю строку, так как для нее нет целевой переменной
            # Проверка целевой переменной
            if len(y) == 0 or np.any(pd.isna(y)) or np.any([v is None for v in y]):
                logger.error(
                    f"Целевая переменная пуста или содержит NaN/None для update, model_id={model_id}"
                )
                return
            # Нормализация
            if model_id not in self.scalers:
                self.scalers[model_id] = StandardScaler()
            X_scaled = self.scalers[model_id].fit_transform(X)
            # Оптимизация гиперпараметров
            best_params = self._optimize_hyperparameters(X_scaled, y)
            # Обучение модели
            model.set_params(**best_params)
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
            # Получение важности признаков
            feature_importance = self._get_feature_importance(model_id)
            # Обновление метрик
            self.metrics[model_id] = MetaMetrics(
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1=metrics["f1"],
                confidence=1.0,  # TODO: Добавить оценку уверенности
                last_update=datetime.now(),
                samples_count=len(self.data_buffer),
                model_count=self.metrics.get(model_id, {}).get("model_count", 0) + 1,
                error_count=self.metrics.get(model_id, {}).get("error_count", 0),
                feature_importance=feature_importance,
                hyperparameters=best_params,
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
        """Предсказание с проверкой входных данных"""
        try:
            if model_id not in self.models:
                raise ValueError(f"Модель {model_id} не найдена")
            # Проверка на пустой датасет
            if df is None or df.empty:
                logger.error(f"Пустой датасет для predict, model_id={model_id}")
                return np.array([]), 0.0
            # Проверка на NaN/None
            if df.isnull().values.any() or (
                df.applymap(lambda x: x is None).values.any()
            ):
                logger.error(
                    f"Датасет содержит NaN или None для predict, model_id={model_id}"
                )
                return np.array([]), 0.0
            # Извлечение признаков
            features = self._extract_features(df)
            # Проверка признаков
            if (
                features.empty
                or features.isnull().values.any()
                or (features.applymap(lambda x: x is None).values.any())
            ):
                logger.error(
                    f"Признаки пусты или содержат NaN/None для predict, model_id={model_id}"
                )
                return np.array([]), 0.0
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
            return np.array([]), 0.0

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


class MetaLearning:
    def __init__(self, config: Dict = None, update_interval=60, min_samples=100):
        """
        Инициализация мета-обучения.

        Args:
            config (Dict): Конфигурация параметров
            update_interval (int): Интервал обновления статуса пары
            min_samples (int): Минимальное количество образцов для обучения
        """
        self.config = config or {
            "min_meta_confidence": 0.7,  # минимальная уверенность
            "min_meta_win_rate": 0.6,  # минимальный винрейт
            "max_meta_loss": 0.3,  # максимальная функция потерь
            "meta_window": 100,  # окно мета-обучения
            "learning_rate": 0.001,  # скорость обучения
            "batch_size": 32,  # размер батча
            "n_estimators": 10,  # количество моделей
            "metrics_dir": "meta_metrics",  # директория для метрик
        }

        self.update_interval = update_interval
        self.min_samples = min_samples
        self.meta_features = {}
        self.meta_strategies = {}
        self.pair_statuses = {}

        # Хранение метрик и контекста
        self.metrics: Dict[str, Dict[str, MetaMetrics]] = {}
        self.context: Dict[str, Dict[str, ModelContext]] = {}

        # Инициализация моделей
        self._init_models()

    def _init_models(self):
        """Инициализация моделей"""
        try:
            # Инициализация стекинга
            self.stacking = StackingClassifier(
                estimators=[
                    ("lr", LogisticRegression()),
                    ("svm", SVC(probability=True)),
                    ("rf", RandomForestClassifier()),
                ],
                final_estimator=LogisticRegression(),
                cv=5,
            )

            # Инициализация RL
            self.env = DummyVecEnv([lambda: gym.make("TradingEnv-v0")])
            self.rl_model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.config["learning_rate"],
                batch_size=self.config["batch_size"],
                verbose=0,
            )

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")

    def select_model(
        self, pair: str, timeframe: str, data: pd.DataFrame, models: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Выбор оптимальной модели.

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм
            data: Данные
            models: Словарь моделей

        Returns:
            Tuple[str, float]: Выбранная модель и уверенность
        """
        try:
            # Получение контекста
            context = self._get_context(pair, timeframe, data)

            # Проверка необходимости мета-обучения
            if self._need_meta_learning(pair, timeframe, context):
                self._meta_learn(pair, timeframe, data, models)

            # Выбор модели
            model_name, confidence = self._select_best_model(
                pair, timeframe, context, models
            )

            # Обновление метрик
            self._update_metrics(pair, timeframe, model_name, confidence)

            return model_name, confidence

        except Exception as e:
            logger.error(f"Error selecting model: {str(e)}")
            return list(models.keys())[0], 0.0

    def _get_context(
        self, pair: str, timeframe: str, data: pd.DataFrame
    ) -> ModelContext:
        """Получение расширенного контекста для модели.

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм
            data: DataFrame с данными

        Returns:
            ModelContext: Контекст модели
        """
        try:
            if data is None or data.empty:
                return None
            volatility = (
                data["close"].pct_change().std() if not data["close"].empty else 0.0
            )
            volume = data["volume"].mean() if "volume" in data else 0.0
            trend = self._get_trend(data)
            atr = calculate_atr(data, 14)
            volatility_profile = {
                "current": volatility,
                "historical": (
                    data["close"].pct_change().rolling(50, min_periods=5).std().mean()
                    if not data["close"].empty
                    else 0.0
                ),
                "atr": atr.iloc[-1] if not atr.empty else 0.0,
                "volatility_ratio": (
                    atr.iloc[-1] / data["close"].iloc[-1]
                    if not atr.empty and data["close"].iloc[-1] != 0
                    else 0.0
                ),
            }
            volume_profile = {
                "current": volume,
                "trend": (
                    data["volume"].pct_change(20).mean() if "volume" in data else 0.0
                ),
                "relative": (
                    volume / data["volume"].rolling(50, min_periods=5).mean().iloc[-1]
                    if "volume" in data
                    and data["volume"].rolling(50, min_periods=5).mean().iloc[-1] != 0
                    else 0.0
                ),
                "distribution": (
                    calculate_volume_profile(data) if not data.empty else {}
                ),
            }
            trend_profile = {
                "direction": trend,
                "strength": (
                    abs(data["close"].iloc[-1] - data["close"].iloc[-20])
                    / data["close"].iloc[-20]
                    if len(data) > 20 and data["close"].iloc[-20] != 0
                    else 0.0
                ),
                "acceleration": (
                    data["close"].pct_change(5).mean() if len(data) > 5 else 0.0
                ),
                "consistency": (
                    self._calculate_trend_consistency(data) if not data.empty else 0.0
                ),
            }
            structure_profile = {
                "support_resistance": (
                    calculate_market_structure(data) if not data.empty else []
                ),
                "liquidity_zones": (
                    calculate_liquidity_zones(data) if not data.empty else []
                ),
                "fractals": calculate_fractals(data) if not data.empty else [],
                "imbalance": calculate_imbalance(data) if not data.empty else 0.0,
            }
            manipulation_profile = {
                "fakeouts": self._identify_fakeouts(data) if not data.empty else [],
                "stop_hunts": self._identify_stop_hunts(data) if not data.empty else [],
                "liquidity_hunts": (
                    self._identify_liquidity_hunts(data) if not data.empty else []
                ),
            }
            correlation_profile = (
                self._calculate_correlations(data) if not data.empty else {}
            )
            context = ModelContext(
                market_regime=self._get_market_regime(data),
                volatility=volatility_profile,
                volume=volume_profile,
                trend=trend_profile,
                timeframes=[timeframe],
                features={
                    "structure": structure_profile,
                    "manipulation": manipulation_profile,
                    "correlation": correlation_profile,
                },
            )
            if pair not in self.context:
                self.context[pair] = {}
            self.context[pair][timeframe] = context
            return context
        except Exception as e:
            logger.error(f"Error getting context: {str(e)}")
            return None

    def _get_market_regime(self, data: pd.DataFrame) -> str:
        """Расширенное определение рыночного режима.

        Args:
            data: DataFrame с рыночными данными

        Returns:
            str: Тип рыночного режима
        """
        try:
            if data is None or data.empty:
                return "unknown"
            returns = data["close"].pct_change()
            volatility = returns.std() if not returns.isna().all() else 0.0
            data["volume"].mean() if "volume" in data else 0.0
            ema_20 = calculate_ema(data["close"], 20)
            calculate_ema(data["close"], 50)
            ema_200 = calculate_ema(data["close"], 200)
            trend_strength = (
                abs(ema_20.iloc[-1] - ema_200.iloc[-1]) / ema_200.iloc[-1]
                if ema_200.iloc[-1] != 0
                else 0.0
            )
            trend_direction = "up" if ema_20.iloc[-1] > ema_200.iloc[-1] else "down"
            atr = calculate_atr(data, 14)
            volatility_ratio = (
                atr.iloc[-1] / data["close"].iloc[-1]
                if data["close"].iloc[-1] != 0
                else 0.0
            )
            volume_ma = data["volume"].rolling(20, min_periods=5).mean()
            volume_trend = (
                data["volume"].iloc[-1] / volume_ma.iloc[-1]
                if volume_ma.iloc[-1] != 0
                else 0.0
            )
            calculate_rsi(data["close"], 14)
            data["close"].pct_change(10)
            support_resistance = calculate_market_structure(data)
            if not support_resistance:
                support_resistance = [data["close"].iloc[-1]]
            price_position = (
                (data["close"].iloc[-1] - min(support_resistance))
                / (max(support_resistance) - min(support_resistance))
                if max(support_resistance) != min(support_resistance)
                else 0.5
            )
            imbalance = calculate_imbalance(data)
            fakeouts = self._identify_fakeouts(data)
            if volatility > 0.02:
                if len(fakeouts) > 0:
                    return "manipulation"
                else:
                    return "volatile"
            elif trend_strength > 0.05:
                if trend_direction == "up":
                    return "trend_up"
                else:
                    return "trend_down"
            elif abs(imbalance) > 0.7:
                return "imbalanced"
            elif volume_trend > 1.5:
                return "volume_spike"
            else:
                return "range"
        except Exception as e:
            logger.error(f"Error getting market regime: {str(e)}")
            return "unknown"

    def _get_trend(self, data: pd.DataFrame) -> str:
        """Определение тренда"""
        try:
            # Расчет SMA
            sma20 = data["close"].rolling(window=20).mean()
            sma50 = data["close"].rolling(window=50).mean()

            # Определение тренда
            if sma20.iloc[-1] > sma50.iloc[-1]:
                return "up"
            elif sma20.iloc[-1] < sma50.iloc[-1]:
                return "down"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Error getting trend: {str(e)}")
            return "unknown"

    def _get_features(self, data: pd.DataFrame) -> List[str]:
        """Получение признаков"""
        try:
            # Расчет технических индикаторов
            features = []

            # RSI
            rsi = talib.RSI(data["close"])
            if not rsi.isna().all():
                features.append("rsi")

            # MACD
            macd, _, _ = talib.MACD(data["close"])
            if not macd.isna().all():
                features.append("macd")

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(data["close"])
            if not bb_upper.isna().all():
                features.append("bb")

            # Volume
            if not data["volume"].isna().all():
                features.append("volume")

            return features

        except Exception as e:
            logger.error(f"Error getting features: {str(e)}")
            return []

    def _need_meta_learning(
        self, pair: str, timeframe: str, context: ModelContext
    ) -> bool:
        """Проверка необходимости мета-обучения"""
        try:
            # Проверка метрик
            metrics = self.metrics.get(pair, {}).get(timeframe)
            if metrics is None:
                return True

            # Проверка уверенности
            if metrics.meta_confidence < self.config["min_meta_confidence"]:
                return True

            # Проверка винрейта
            if metrics.meta_win_rate < self.config["min_meta_win_rate"]:
                return True

            # Проверка функции потерь
            if metrics.meta_loss > self.config["max_meta_loss"]:
                return True

            return False

        except Exception as e:
            logger.error(f"Error checking meta learning need: {str(e)}")
            return True

    def _meta_learn(
        self, pair: str, timeframe: str, data: pd.DataFrame, models: Dict[str, Any]
    ):
        """Мета-обучение"""
        try:
            # Подготовка данных
            features = self._prepare_features(data)
            target = self._prepare_target(data)

            # Обучение стекинга
            self.stacking.fit(features, target)

            # Обучение RL
            self.rl_model.learn(total_timesteps=1000, progress_bar=False)

        except Exception as e:
            logger.error(f"Error in meta learning: {str(e)}")

    def _select_best_model(
        self, pair: str, timeframe: str, context: ModelContext, models: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Выбор лучшей модели"""
        try:
            best_model = None
            best_confidence = 0.0

            for model_name, model in models.items():
                # Оценка модели
                confidence = self._evaluate_model(model, context, pair, timeframe)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_model = model_name

            return best_model, best_confidence

        except Exception as e:
            logger.error(f"Error selecting best model: {str(e)}")
            return list(models.keys())[0], 0.0

    def _evaluate_model(
        self, model: Any, context: ModelContext, pair: str, timeframe: str
    ) -> float:
        """Оценка модели"""
        try:
            predictions = (
                model.predict(context.features)
                if context and hasattr(model, "predict")
                else []
            )
            if (
                not hasattr(context, "target")
                or context.target is None
                or len(context.target) == 0
                or len(predictions) == 0
            ):
                return 0.0
            accuracy = accuracy_score(context.target, predictions)
            precision = precision_score(context.target, predictions)
            recall = recall_score(context.target, predictions)
            f1 = f1_score(context.target, predictions)
            confidence = accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1 * 0.2
            return confidence
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return 0.0

    def _update_metrics(
        self, pair: str, timeframe: str, model_name: str, confidence: float
    ):
        """Обновление метрик"""
        try:
            # Создание метрик
            metrics = MetaMetrics(
                accuracy=0.0,  # TODO: Рассчитать
                precision=0.0,  # TODO: Рассчитать
                recall=0.0,  # TODO: Рассчитать
                f1=0.0,  # TODO: Рассчитать
                confidence=confidence,
                last_update=datetime.now(),
                samples_count=0,
                model_count=0,
                error_count=0,
                feature_importance={},
                hyperparameters={},
            )

            # Сохранение метрик
            if pair not in self.metrics:
                self.metrics[pair] = {}
            self.metrics[pair][timeframe] = metrics

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков"""
        try:
            features = pd.DataFrame()

            # Технические индикаторы
            features["rsi"] = talib.RSI(data["close"])
            features["macd"], features["macd_signal"], _ = talib.MACD(data["close"])
            features["bb_upper"], features["bb_middle"], features["bb_lower"] = (
                talib.BBANDS(data["close"])
            )
            features["atr"] = talib.ATR(data["high"], data["low"], data["close"])

            # Свечные характеристики
            features["body_size"] = abs(data["close"] - data["open"])
            features["upper_shadow"] = data["high"] - data[["open", "close"]].max(
                axis=1
            )
            features["lower_shadow"] = data[["open", "close"]].min(axis=1) - data["low"]
            features["is_bullish"] = (data["close"] > data["open"]).astype(int)

            # Объемные характеристики
            features["volume_ma"] = talib.SMA(data["volume"], timeperiod=20)
            features["volume_ratio"] = data["volume"] / features["volume_ma"]

            # Волатильность
            features["volatility"] = data["close"].pct_change().rolling(window=20).std()

            # Тренд
            features["trend"] = talib.ADX(data["high"], data["low"], data["close"])

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return pd.DataFrame()

    def _prepare_target(self, data: pd.DataFrame) -> pd.Series:
        """Подготовка целевой переменной"""
        try:
            # Расчет будущего движения цены
            future_returns = data["close"].pct_change().shift(-1)

            # Бинаризация
            target = (future_returns > 0).astype(int)

            return target

        except Exception as e:
            logger.error(f"Error preparing target: {str(e)}")
            return pd.Series()

    def get_meta_metrics(self, pair: str, timeframe: str) -> Optional[MetaMetrics]:
        """Получение метрик мета-обучения"""
        try:
            return self.metrics.get(pair, {}).get(timeframe)

        except Exception as e:
            logger.error(f"Error getting meta metrics: {str(e)}")
            return None

    def get_model_context(self, pair: str, timeframe: str) -> Optional[ModelContext]:
        """Получение контекста модели"""
        try:
            return self.context.get(pair, {}).get(timeframe)

        except Exception as e:
            logger.error(f"Error getting model context: {str(e)}")
            return None

    def init_pair_structure(self, pair: str) -> None:
        """Initialize directory structure and status tracking for a new trading pair.

        Args:
            pair: Trading pair symbol (e.g., 'BTC/USDT')
        """
        # Create pair-specific directories
        pair_dir = Path(f"data/{pair.replace('/', '_')}")
        pair_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (pair_dir / "models").mkdir(exist_ok=True)
        (pair_dir / "backtests").mkdir(exist_ok=True)
        (pair_dir / "logs").mkdir(exist_ok=True)

        # Initialize pair status
        self.pair_statuses[pair] = {
            "is_trade_ready": False,
            "meta_status": "initializing",
            "progress": {
                "data_collection": 0.0,
                "model_training": 0.0,
                "backtest_completion": 0.0,
                "correlation_analysis": 0.0,
            },
            "last_update": datetime.now().isoformat(),
            "correlations": {},
            "meta_features": {},
        }

        # Initialize meta-features for the pair
        self.meta_features[pair] = {
            "volatility": 0.0,
            "trend": 0.0,
            "volume": 0.0,
            "correlation_strength": 0.0,
            "market_regime": "unknown",
        }

        logger.info(f"Initialized structure for pair {pair}")

    def update_pair_status(self, pair: str, status_updates: dict) -> None:
        """Update the status of a trading pair.

        Args:
            pair: Trading pair symbol
            status_updates: Dictionary with status updates
        """
        if pair not in self.pair_statuses:
            self.init_pair_structure(pair)

        self.pair_statuses[pair].update(status_updates)
        self.pair_statuses[pair]["last_update"] = datetime.now().isoformat()

    def get_pair_status(self, pair: str) -> dict:
        """Get current status of a trading pair.

        Args:
            pair: Trading pair symbol

        Returns:
            Dictionary with pair status
        """
        if pair not in self.pair_statuses:
            self.init_pair_structure(pair)
        return self.pair_statuses[pair]

    def calculate_pair_readiness(
        self, pair: str, wr_threshold: float = 0.55, min_samples: int = 100
    ) -> bool:
        """Оценивает готовность пары к лайв-торговле по данным из DatasetManager."""
        stats = DatasetManager.get_statistics(pair)
        if not stats or stats.get("total", 0) < min_samples:
            self.pair_statuses[pair] = {"ready": False, "reason": "Недостаточно данных"}
            logger.info(
                f"Пара {pair} НЕ готова к лайв: мало данных ({stats.get('total', 0)})"
            )
            return False
        wr = stats.get("win", 0) / stats.get("total", 1)
        regimes = self._get_regimes_variety(pair)
        if wr < wr_threshold:
            self.pair_statuses[pair] = {
                "ready": False,
                "reason": f"WR ниже порога: {wr:.2f}",
            }
            logger.info(f"Пара {pair} НЕ готова к лайв: winrate={wr:.2f}")
            return False
        if regimes < 3:
            self.pair_statuses[pair] = {
                "ready": False,
                "reason": f"Мало режимов: {regimes}",
            }
            logger.info(f"Пара {pair} НЕ готова к лайв: мало режимов ({regimes})")
            return False
        self.pair_statuses[pair] = {"ready": True, "reason": "OK"}
        logger.info(f"Пара {pair} готова к лайв")
        return True

    def _get_regimes_variety(self, pair: str) -> int:
        """Возвращает количество уникальных рыночных режимов в датасете пары."""
        data = DatasetManager.load_dataset(pair)
        regimes = set(x.get("market_regime") for x in data if x.get("market_regime"))
        return len(regimes)

    def retrain_from_backtest(self, pair: str):
        """Переобучает модель по историческим данным из DatasetManager."""
        data = DatasetManager.load_dataset(pair)
        if not data:
            logger.warning(f"Нет данных для переобучения по паре {pair}")
            return
        X = [x["features"] for x in data]
        y = [1 if x.get("result", {}).get("win") else 0 for x in data]
        # Здесь можно использовать self.stacking или RL-модель
        try:
            self.stacking.fit(X, y)
            logger.info(f"Мета-модель переобучена по {len(X)} примерам для {pair}")
        except Exception as e:
            logger.error(f"Ошибка при переобучении stacking: {e}")
        # RL-обучение можно реализовать отдельно (совместимо)

    def score_patterns(self, pair: str, top_n: int = 5) -> List[dict]:
        """Выбирает топ-N паттернов, которые чаще всего приводят к профиту."""
        data = DatasetManager.load_dataset(pair)
        if not data:
            return []
        pattern_counter = {}
        for x in data:
            pattern = (
                x["features"].get("pattern") if x["features"].get("pattern") else None
            )
            if not pattern:
                continue
            if pattern not in pattern_counter:
                pattern_counter[pattern] = {"count": 0, "profit": 0, "wins": 0}
            pattern_counter[pattern]["count"] += 1
            pattern_counter[pattern]["profit"] += x.get("result", {}).get("PnL", 0)
            if x.get("result", {}).get("win"):
                pattern_counter[pattern]["wins"] += 1
        # Сортировка по winrate и прибыли
        scored = [
            {
                "pattern": k,
                "count": v["count"],
                "avg_profit": v["profit"] / v["count"],
                "winrate": v["wins"] / v["count"],
            }
            for k, v in pattern_counter.items()
            if v["count"] >= 3
        ]
        scored.sort(key=lambda x: (x["winrate"], x["avg_profit"]), reverse=True)
        return scored[:top_n]

    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Расчет согласованности тренда."""
        try:
            # Расчет направления свечей
            candle_direction = (data["close"] > data["open"]).astype(int)

            # Расчет согласованности
            consistency = candle_direction.rolling(20).mean().iloc[-1]

            return consistency

        except Exception as e:
            logger.error(f"Error calculating trend consistency: {str(e)}")
            return 0.5

    def _identify_stop_hunts(self, data: pd.DataFrame) -> List[Dict]:
        """Определение выносов стопов."""
        try:
            stop_hunts = []

            # Анализ движения цены
            for i in range(1, len(data)):
                # Проверка резкого движения
                price_spike = (
                    abs(data["close"].iloc[i] - data["open"].iloc[i])
                    > data["atr"].iloc[i] * 2
                )

                # Проверка объема
                volume_spike = (
                    data["volume"].iloc[i]
                    > data["volume"].rolling(20).mean().iloc[i] * 1.5
                )

                # Проверка возврата
                price_return = (
                    abs(data["close"].iloc[i + 1] - data["close"].iloc[i])
                    < data["atr"].iloc[i] * 0.5
                )

                if price_spike and volume_spike and price_return:
                    stop_hunts.append(
                        {
                            "timestamp": data.index[i],
                            "price": data["close"].iloc[i],
                            "volume": data["volume"].iloc[i],
                            "atr": data["atr"].iloc[i],
                        }
                    )

            return stop_hunts

        except Exception as e:
            logger.error(f"Error identifying stop hunts: {str(e)}")
            return []

    def _identify_liquidity_hunts(self, data: pd.DataFrame) -> List[Dict]:
        """Определение охоты за ликвидностью."""
        try:
            liquidity_hunts = []

            # Анализ объема и цены
            for i in range(1, len(data)):
                # Проверка объема
                volume_spike = (
                    data["volume"].iloc[i]
                    > data["volume"].rolling(20).mean().iloc[i] * 2
                )

                # Проверка движения цены
                price_move = (
                    abs(data["close"].iloc[i] - data["open"].iloc[i])
                    < data["atr"].iloc[i] * 0.5
                )

                # Проверка спреда
                spread_increase = (
                    data["high"].iloc[i] - data["low"].iloc[i] > data["atr"].iloc[i]
                )

                if volume_spike and price_move and spread_increase:
                    liquidity_hunts.append(
                        {
                            "timestamp": data.index[i],
                            "volume": data["volume"].iloc[i],
                            "spread": data["high"].iloc[i] - data["low"].iloc[i],
                            "atr": data["atr"].iloc[i],
                        }
                    )

            return liquidity_hunts

        except Exception as e:
            logger.error(f"Error identifying liquidity hunts: {str(e)}")
            return []

    def _calculate_correlations(self, data: pd.DataFrame) -> Dict:
        """Расчет корреляций между метриками."""
        try:
            # Подготовка данных
            metrics = pd.DataFrame(
                {
                    "price": data["close"],
                    "volume": data["volume"],
                    "volatility": data["close"].pct_change().rolling(20).std(),
                    "momentum": data["close"].pct_change(10),
                    "volume_trend": data["volume"].pct_change(20),
                }
            )

            # Расчет корреляций
            correlations = metrics.corr()

            return correlations.to_dict()

        except Exception as e:
            logger.error(f"Error calculating correlations: {str(e)}")
            return {}
