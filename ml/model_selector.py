import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import catboost as cb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from prophet import Prophet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Conv1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.dataset_manager import DatasetManager


@dataclass
class ModelMetrics:
    """Метрики модели"""

    win_rate: float  # процент успешных сделок
    accuracy: float  # точность
    precision: float  # прецизионность
    recall: float  # полнота
    f1: float  # F1-score
    last_update: datetime  # время последнего обновления
    training_time: float  # время обучения
    inference_time: float  # время предсказания


@dataclass
class ModelConfig:
    """Конфигурация модели"""

    model_type: str  # тип модели
    params: Dict  # параметры модели
    features: List[str]  # используемые фичи
    target: str  # целевая переменная
    timeframes: List[str]  # таймфреймы
    market_regimes: List[str]  # режимы рынка


class ModelSelector:
    def __init__(self, min_accuracy=0.7, max_models=3, **kwargs):
        """
        Инициализация селектора моделей.

        Args:
            config (Dict): Конфигурация параметров
        """
        self.config = kwargs or {
            "min_win_rate": 0.6,  # минимальный винрейт
            "min_accuracy": min_accuracy,  # минимальная точность
            "retrain_threshold": 0.1,  # порог для переобучения
            "max_models_per_pair": max_models,  # максимальное количество моделей на пару
            "validation_window": 100,  # окно валидации
            "model_dir": "models",  # директория для сохранения моделей
            "feature_importance_threshold": 0.05,  # порог важности фич
            "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],  # таймфреймы
            "market_regimes": ["trend", "range", "reversal", "manipulation", "volatile", "panic"],
        }

        # Создание директории для моделей
        os.makedirs(self.config["model_dir"], exist_ok=True)

        # Хранение моделей и метрик
        self.models: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, ModelMetrics]]] = {}
        self.feature_importance: Dict[str, Dict[str, pd.DataFrame]] = {}

        # Инициализация моделей
        self._init_models()

    def _init_models(self):
        """Инициализация моделей"""
        try:
            # CatBoost
            self.catboost_params = {
                "iterations": 1000,
                "learning_rate": 0.1,
                "depth": 6,
                "loss_function": "Logloss",
                "eval_metric": "AUC",
                "random_seed": 42,
            }

            # XGBoost
            self.xgboost_params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 1000,
                "objective": "binary:logistic",
                "eval_metric": "auc",
                "random_state": 42,
            }

            # LSTM
            self.lstm_params = {
                "units": 64,
                "dropout": 0.2,
                "recurrent_dropout": 0.2,
                "batch_size": 32,
                "epochs": 100,
                "patience": 10,
            }

            # CNN
            self.cnn_params = {
                "filters": 64,
                "kernel_size": 3,
                "pool_size": 2,
                "batch_size": 32,
                "epochs": 100,
                "patience": 10,
            }

            # Transformer
            self.transformer_params = {
                "model_name": "bert-base-uncased",
                "num_labels": 2,
                "batch_size": 16,
                "epochs": 5,
                "learning_rate": 2e-5,
            }

            # ARIMA
            self.arima_params = {"order": (5, 1, 0), "seasonal_order": (1, 1, 1, 12)}

            # Prophet
            self.prophet_params = {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
            }

        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")

    def select_model(
        self,
        pair: str,
        timeframe: str,
        market_regime: str,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Optional[Any]:
        """
        Выбор оптимальной модели.

        Args:
            pair: Торговая пара
            timeframe: Таймфрейм
            market_regime: Режим рынка
            features: Признаки
            target: Целевая переменная

        Returns:
            Optional[Any]: Выбранная модель
        """
        try:
            # Проверка существующих моделей
            if pair in self.models and timeframe in self.models[pair]:
                # Получение метрик
                metrics = self.metrics[pair][timeframe]

                # Проверка необходимости переобучения
                if self._need_retraining(metrics, market_regime):
                    return self._retrain_model(pair, timeframe, market_regime, features, target)

                # Выбор лучшей модели
                best_model = self._select_best_model(metrics)
                if best_model:
                    return best_model

            # Обучение новой модели
            return self._train_new_model(pair, timeframe, market_regime, features, target)

        except Exception as e:
            logger.error(f"Error selecting model: {str(e)}")
            return None

    def _need_retraining(self, metrics: Dict[str, ModelMetrics], market_regime: str) -> bool:
        """Проверка необходимости переобучения"""
        try:
            for model_metrics in metrics.values():
                # Проверка винрейта
                if model_metrics.win_rate < self.config["min_win_rate"]:
                    return True

                # Проверка точности
                if model_metrics.accuracy < self.config["min_accuracy"]:
                    return True

                # Проверка актуальности
                time_diff = datetime.now() - model_metrics.last_update
                if time_diff.days > 7:  # неделя
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking retraining need: {str(e)}")
            return True

    def _select_best_model(self, metrics: Dict[str, ModelMetrics]) -> Optional[Any]:
        """Выбор лучшей модели"""
        try:
            if not metrics:
                logger.warning("Нет доступных метрик для выбора модели")
                return None

            best_model = None
            best_score = -float("inf")

            for model_type, model_metrics in metrics.items():
                if not isinstance(model_metrics, ModelMetrics):
                    logger.warning(f"Некорректный формат метрик для модели {model_type}")
                    continue

                # Расчет общего скора
                score = (
                    model_metrics.win_rate * 0.4
                    + model_metrics.accuracy * 0.3
                    + model_metrics.f1 * 0.3
                )

                if score > best_score:
                    best_score = score
                    best_model = model_type

            return best_model

        except Exception as e:
            logger.error(f"Ошибка выбора лучшей модели: {e}")
            return None

    def _train_new_model(
        self,
        pair: str,
        timeframe: str,
        market_regime: str,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Optional[Any]:
        """Обучение новой модели"""
        try:
            # Разделение данных
            tscv = TimeSeriesSplit(n_splits=5)

            # Выбор типа модели
            model_type = self._select_model_type(market_regime)

            # Обучение модели
            if model_type == "catboost":
                model = self._train_catboost(features, target)
            elif model_type == "xgboost":
                model = self._train_xgboost(features, target)
            elif model_type == "lstm":
                model = self._train_lstm(features, target)
            elif model_type == "cnn":
                model = self._train_cnn(features, target)
            elif model_type == "transformer":
                model = self._train_transformer(features, target)
            elif model_type == "arima":
                model = self._train_arima(features, target)
            elif model_type == "prophet":
                model = self._train_prophet(features, target)
            else:
                return None

            # Сохранение модели
            self._save_model(pair, timeframe, model_type, model)

            # Обновление метрик
            self._update_metrics(pair, timeframe, model_type, model, features, target)

            return model

        except Exception as e:
            logger.error(f"Error training new model: {str(e)}")
            return None

    def _select_model_type(self, market_regime: str) -> str:
        """Выбор типа модели"""
        try:
            if market_regime == "trend":
                return "catboost"  # для трендовых данных
            elif market_regime == "range":
                return "lstm"  # для боковика
            elif market_regime == "reversal":
                return "transformer"  # для разворотов
            elif market_regime == "manipulation":
                return "cnn"  # для манипуляций
            elif market_regime == "volatile":
                return "xgboost"  # для волатильности
            elif market_regime == "panic":
                return "arima"  # для паники
            else:
                return "catboost"  # по умолчанию

        except Exception as e:
            logger.error(f"Error selecting model type: {str(e)}")
            return "catboost"

    def _train_catboost(self, features: pd.DataFrame, target: pd.Series) -> cb.CatBoostClassifier:
        """Обучение CatBoost"""
        try:
            model = cb.CatBoostClassifier(**self.catboost_params)
            model.fit(features, target)
            return model

        except Exception as e:
            logger.error(f"Error training CatBoost: {str(e)}")
            return None

    def _train_xgboost(self, features: pd.DataFrame, target: pd.Series) -> xgb.XGBClassifier:
        """Обучение XGBoost"""
        try:
            model = xgb.XGBClassifier(**self.xgboost_params)
            model.fit(features, target)
            return model

        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            return None

    def _train_lstm(self, features: pd.DataFrame, target: pd.Series) -> Sequential:
        """Обучение LSTM"""
        try:
            # Подготовка данных
            X = features.values.reshape((features.shape[0], features.shape[1], 1))
            y = target.values

            # Создание модели
            model = Sequential(
                [
                    LSTM(
                        self.lstm_params["units"],
                        dropout=self.lstm_params["dropout"],
                        recurrent_dropout=self.lstm_params["recurrent_dropout"],
                        return_sequences=True,
                    ),
                    LSTM(
                        self.lstm_params["units"] // 2,
                        dropout=self.lstm_params["dropout"],
                        recurrent_dropout=self.lstm_params["recurrent_dropout"],
                    ),
                    Dense(1, activation="sigmoid"),
                ]
            )

            # Компиляция
            model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

            # Обучение
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=self.lstm_params["patience"]
            )

            model.fit(
                X,
                y,
                batch_size=self.lstm_params["batch_size"],
                epochs=self.lstm_params["epochs"],
                callbacks=[early_stopping],
                validation_split=0.2,
            )

            return model

        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            return None

    def _train_cnn(self, features: pd.DataFrame, target: pd.Series) -> Sequential:
        """Обучение CNN"""
        try:
            # Подготовка данных
            X = features.values.reshape((features.shape[0], features.shape[1], 1))
            y = target.values

            # Создание модели
            model = Sequential(
                [
                    Conv1D(
                        self.cnn_params["filters"],
                        self.cnn_params["kernel_size"],
                        activation="relu",
                        input_shape=(features.shape[1], 1),
                    ),
                    MaxPooling1D(self.cnn_params["pool_size"]),
                    Conv1D(
                        self.cnn_params["filters"] // 2,
                        self.cnn_params["kernel_size"],
                        activation="relu",
                    ),
                    MaxPooling1D(self.cnn_params["pool_size"]),
                    Dense(1, activation="sigmoid"),
                ]
            )

            # Компиляция
            model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

            # Обучение
            early_stopping = EarlyStopping(monitor="val_loss", patience=self.cnn_params["patience"])

            model.fit(
                X,
                y,
                batch_size=self.cnn_params["batch_size"],
                epochs=self.cnn_params["epochs"],
                callbacks=[early_stopping],
                validation_split=0.2,
            )

            return model

        except Exception as e:
            logger.error(f"Error training CNN: {str(e)}")
            return None

    def _train_transformer(self, features: pd.DataFrame, target: pd.Series) -> Any:
        """Обучение Transformer"""
        try:
            # Подготовка данных
            tokenizer = AutoTokenizer.from_pretrained(self.transformer_params["model_name"])

            # Создание модели
            model = AutoModelForSequenceClassification.from_pretrained(
                self.transformer_params["model_name"],
                num_labels=self.transformer_params["num_labels"],
            )

            # Обучение
            # TODO: Реализовать обучение трансформера

            return model

        except Exception as e:
            logger.error(f"Error training Transformer: {str(e)}")
            return None

    def _train_arima(self, features: pd.DataFrame, target: pd.Series) -> ARIMA:
        """Обучение ARIMA"""
        try:
            model = ARIMA(target, order=self.arima_params["order"])
            model = model.fit()
            return model

        except Exception as e:
            logger.error(f"Error training ARIMA: {str(e)}")
            return None

    def _train_prophet(self, features: pd.DataFrame, target: pd.Series) -> Prophet:
        """Обучение Prophet"""
        try:
            # Подготовка данных
            df = pd.DataFrame({"ds": features.index, "y": target})

            # Создание и обучение модели
            model = Prophet(**self.prophet_params)
            model.fit(df)

            return model

        except Exception as e:
            logger.error(f"Error training Prophet: {str(e)}")
            return None

    def _save_model(self, pair: str, timeframe: str, model_type: str, model: Any):
        """Сохранение модели"""
        try:
            # Создание директории
            model_path = os.path.join(self.config["model_dir"], pair, timeframe)
            os.makedirs(model_path, exist_ok=True)

            # Сохранение модели
            model_file = os.path.join(model_path, f"{model_type}.joblib")
            joblib.dump(model, model_file)

            # Сохранение метаданных
            metadata = {
                "model_type": model_type,
                "last_update": datetime.now().isoformat(),
                "metrics": self.metrics.get(pair, {})
                .get(timeframe, {})
                .get(model_type, {})
                .__dict__,
                "feature_importance": (
                    self.feature_importance.get(pair, {}).get(timeframe, {}).to_dict()
                    if pair in self.feature_importance
                    and timeframe in self.feature_importance[pair]
                    else {}
                ),
            }

            metadata_file = os.path.join(model_path, f"{model_type}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def _load_model(self, pair: str, timeframe: str, model_type: str) -> Optional[Dict]:
        """Загрузка модели и метаданных"""
        try:
            model_path = os.path.join(self.config["model_dir"], pair, timeframe)

            # Загрузка модели
            model_file = os.path.join(model_path, f"{model_type}.joblib")
            if not os.path.exists(model_file):
                logger.warning(f"Файл модели не найден: {model_file}")
                return None

            model = joblib.load(model_file)

            # Загрузка метаданных
            metadata_file = os.path.join(model_path, f"{model_type}_metadata.json")
            if not os.path.exists(metadata_file):
                logger.warning(f"Файл метаданных не найден: {metadata_file}")
                return None

            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            # Проверка формата метаданных
            if not isinstance(metadata, dict):
                raise ValueError(f"Ожидался словарь, получен {type(metadata)}")

            # Проверка обязательных полей
            required_fields = ["model_type", "last_update", "metrics", "feature_importance"]
            for field in required_fields:
                if field not in metadata:
                    raise ValueError(f"Отсутствует обязательное поле: {field}")

            return {"model": model, "metadata": metadata}

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return None

    def _update_metrics(
        self,
        pair: str,
        timeframe: str,
        model_type: str,
        model: Any,
        features: pd.DataFrame,
        target: pd.Series,
    ):
        """Обновление метрик"""
        try:
            # Получение предсказаний
            if model_type in ["catboost", "xgboost"]:
                predictions = model.predict(features)
            elif model_type in ["lstm", "cnn"]:
                predictions = (model.predict(features) > 0.5).astype(int)
            elif model_type == "transformer":
                # TODO: Реализовать предсказания для трансформера
                predictions = np.zeros(len(target))
            elif model_type in ["arima", "prophet"]:
                predictions = model.predict(features)

            # Расчет метрик
            metrics = ModelMetrics(
                win_rate=sum(predictions == target) / len(target),
                accuracy=accuracy_score(target, predictions),
                precision=precision_score(target, predictions),
                recall=recall_score(target, predictions),
                f1=f1_score(target, predictions),
                last_update=datetime.now(),
                training_time=0.0,  # TODO: Добавить измерение времени
                inference_time=0.0,  # TODO: Добавить измерение времени
            )

            # Сохранение метрик
            if pair not in self.metrics:
                self.metrics[pair] = {}
            if timeframe not in self.metrics[pair]:
                self.metrics[pair][timeframe] = {}

            self.metrics[pair][timeframe][model_type] = metrics

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")

    def get_model_metrics(self, pair: str, timeframe: str) -> Dict[str, ModelMetrics]:
        """Получение метрик моделей"""
        try:
            return self.metrics.get(pair, {}).get(timeframe, {})

        except Exception as e:
            logger.error(f"Error getting model metrics: {str(e)}")
            return {}

    def get_feature_importance(self, pair: str, timeframe: str) -> pd.DataFrame:
        """Получение важности признаков"""
        try:
            return self.feature_importance.get(pair, {}).get(timeframe, pd.DataFrame())

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return pd.DataFrame()

    def retrain_if_dataset_updated(self, pair: str, timeframe: str):
        """Переобучает модель, если датасет для пары обновился."""
        dataset = DatasetManager.load_dataset(pair)
        if not dataset:
            logger.warning(f"Нет данных для переобучения по паре {pair}")
            return
        X = pd.DataFrame([x["features"] for x in dataset])
        y = pd.Series([1 if x.get("result", {}).get("win") else 0 for x in dataset])
        t0 = time.time()
        model = self._train_new_model(pair, timeframe, "all", X, y)
        self._save_model(pair, timeframe, "dynamic", model)
        self._update_metadata(pair, timeframe, len(dataset), time.time() - t0)
        logger.info(f"Модель для {pair} переобучена по {len(dataset)} примерам")

    def _update_metadata(self, pair: str, timeframe: str, dataset_size: int, train_time: float):
        """Обновляет метаданные обучения."""
        stats = DatasetManager.get_statistics(pair)
        wr = stats.get("win", 0) / stats.get("total", 1) if stats else 0
        meta = {
            "last_train_time": datetime.now().isoformat(),
            "dataset_size": dataset_size,
            "win_rate": wr,
        }
        if not hasattr(self, "train_metadata"):
            self.train_metadata = {}
        if pair not in self.train_metadata:
            self.train_metadata[pair] = {}
        self.train_metadata[pair][timeframe] = meta

    def get_training_quality(self, pair: str, timeframe: str) -> dict:
        """Возвращает качество обучения: confidence, pattern complexity, diversity, estimated profitability."""
        dataset = DatasetManager.load_dataset(pair)
        if not dataset:
            return {"confidence": 0, "pattern_complexity": 0, "diversity": 0, "profitability": 0}
        # Confidence: средний accuracy по последней модели
        metrics = self.metrics.get(pair, {}).get(timeframe, {})
        confidence = getattr(metrics, "accuracy", 0) if metrics else 0
        # Pattern complexity: среднее число уникальных паттернов на 100 примеров
        patterns = [x["features"].get("pattern") for x in dataset if x["features"].get("pattern")]
        pattern_complexity = len(set(patterns)) / (len(dataset) / 100) if dataset else 0
        # Diversity: число уникальных режимов
        diversity = len(set(x.get("market_regime") for x in dataset if x.get("market_regime")))
        # Profitability: средний PnL
        profitability = sum(x.get("result", {}).get("PnL", 0) for x in dataset) / len(dataset)
        return {
            "confidence": confidence,
            "pattern_complexity": pattern_complexity,
            "diversity": diversity,
            "profitability": profitability,
        }

    def dashboard_retrain(self, pair: str, timeframe: str):
        """Переобучение по команде с дашборда."""
        self.retrain_if_dataset_updated(pair, timeframe)
        logger.info(f"Переобучение по запросу dashboard для {pair} {timeframe}")
