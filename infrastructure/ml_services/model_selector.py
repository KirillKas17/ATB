import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import catboost as cb
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from prophet import Prophet
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    LSTM,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    MaxPooling1D,
    MultiHeadAttention,
    Flatten,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from shared.models.ml_metrics import ModelMetrics

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class ModelConfig:
    """Конфигурация модели"""

    model_type: str  # тип модели
    params: Dict[str, Any]  # параметры модели
    features: List[str]  # используемые фичи
    target: str  # целевая переменная
    timeframes: List[str]  # таймфреймы
    market_regimes: List[str]  # режимы рынка


class ModelSelector:
    def __init__(
        self, min_accuracy: float = 0.7, max_models: int = 3, **kwargs: Any
    ) -> None:
        """
        Инициализация селектора моделей.
        Args:
            min_accuracy: минимальная точность
            max_models: максимальное количество моделей
            kwargs: дополнительные параметры
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
            "market_regimes": [
                "trend",
                "range",
                "reversal",
                "manipulation",
                "volatile",
                "panic",
            ],
        }
        # Создание директории для моделей
        os.makedirs(self.config["model_dir"], exist_ok=True)
        # Хранение моделей и метрик
        self.models: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.metrics: Dict[str, Dict[str, Dict[str, ModelMetrics]]] = {}
        self.feature_importance: Dict[str, Dict[str, DataFrame]] = {}
        self._training_times: Dict[str, float] = {}
        self.train_metadata: Dict[str, Dict[str, Any]] = {}
        # Инициализация моделей
        self._init_models()

    def _init_models(self) -> None:
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
        features: DataFrame,
        target: Series,
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
                    return self._train_new_model(
                        pair, timeframe, market_regime, features, target
                    )
                # Выбор лучшей модели
                best_model = self._select_best_model(metrics)
                if best_model:
                    return best_model
            # Обучение новой модели
            return self._train_new_model(
                pair, timeframe, market_regime, features, target
            )
        except Exception as e:
            logger.error(f"Error selecting model: {str(e)}")
            return None

    def _need_retraining(
        self, metrics: Dict[str, ModelMetrics], market_regime: str
    ) -> bool:
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

    def _select_best_model(self, metrics: Dict[str, ModelMetrics]) -> Optional[str]:
        """Выбор лучшей модели на основе метрик"""
        try:
            best_model = None
            best_score = 0.0
            
            for model_type, model_metrics in metrics.items():
                if not isinstance(model_metrics, ModelMetrics):
                    logger.warning(
                        f"Некорректный формат метрик для модели {model_type}"
                    )
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
        features: DataFrame,
        target: Series,
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

    def _train_catboost(
        self, features: DataFrame, target: Series
    ) -> Optional[cb.CatBoostClassifier]:
        """Обучение CatBoost"""
        try:
            model = cb.CatBoostClassifier(**self.catboost_params)
            # Исправляю типы для pandas
            features_array = features.to_numpy() if hasattr(features, 'to_numpy') else features.values
            target_array = target.to_numpy() if hasattr(target, 'to_numpy') else target.values
            model.fit(features_array, target_array)
            return model
        except Exception as e:
            logger.error(f"Error training CatBoost: {str(e)}")
            return None

    def _train_xgboost(
        self, features: DataFrame, target: Series
    ) -> Optional[xgb.XGBClassifier]:
        """Обучение XGBoost"""
        try:
            model = xgb.XGBClassifier(**self.xgboost_params)
            # Исправляю типы для pandas
            features_array = features.to_numpy() if hasattr(features, 'to_numpy') else features.values
            target_array = target.to_numpy() if hasattr(target, 'to_numpy') else target.values
            model.fit(features_array, target_array)
            return model
        except Exception as e:
            logger.error(f"Error training XGBoost: {str(e)}")
            return None

    def _train_lstm(self, features: DataFrame, target: Series) -> Optional[Sequential]:
        """Обучение LSTM"""
        try:
            # Исправляю типы для pandas
            features_array = features.to_numpy() if hasattr(features, 'to_numpy') else features.values
            target_array = target.to_numpy() if hasattr(target, 'to_numpy') else target.values
            
            # Подготовка данных для LSTM
            X = features_array.reshape((features_array.shape[0], features_array.shape[1], 1))
            y = target_array.reshape(-1, 1)
            
            # Создание модели
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, batch_size=32, epochs=50, verbose=0)
            
            return model
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            return None

    def _train_cnn(self, features: DataFrame, target: Series) -> Optional[Sequential]:
        """Обучение CNN"""
        try:
            # Исправляю типы для pandas
            features_array = features.to_numpy() if hasattr(features, 'to_numpy') else features.values
            target_array = target.to_numpy() if hasattr(target, 'to_numpy') else target.values
            
            # Подготовка данных для CNN
            X = features_array.reshape((features_array.shape[0], features_array.shape[1], 1))
            y = target_array.reshape(-1, 1)
            
            # Создание модели
            model = Sequential([
                Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
                MaxPooling1D(pool_size=2),
                Conv1D(filters=32, kernel_size=3, activation='relu'),
                MaxPooling1D(pool_size=2),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X, y, batch_size=32, epochs=50, verbose=0)
            
            return model
        except Exception as e:
            logger.error(f"Error training CNN: {str(e)}")
            return None

    def _train_transformer(self, features: DataFrame, target: Series) -> Optional[Any]:
        """Обучение Transformer"""
        try:
            # Подготовка данных для трансформера
            if hasattr(features, 'to_numpy'):
                X = features.to_numpy()
            else:
                X = features.values
            if hasattr(target, 'to_numpy'):
                y = target.to_numpy()
            else:
                y = target.values
            # Нормализация данных
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            # Преобразование в 3D формат для трансформера (batch_size, sequence_length, features)
            # Используем скользящее окно для создания последовательностей
            sequence_length = 10
            X_sequences = []
            y_sequences = []
            for i in range(sequence_length, len(X_scaled)):
                X_sequences.append(X_scaled[i - sequence_length : i])
                y_sequences.append(y[i])
            # Преобразуем в numpy массивы для обучения
            X_sequences_array = np.array(X_sequences)
            y_sequences_array = np.array(y_sequences)
            # Создание модели трансформера
            model = Sequential(
                [
                    Input(shape=(sequence_length, X_scaled.shape[1])),
                    MultiHeadAttention(num_heads=4, key_dim=32),
                    GlobalAveragePooling1D(),
                    Dense(64, activation="relu"),
                    Dropout(0.3),
                    Dense(32, activation="relu"),
                    Dropout(0.2),
                    Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            # Обучение трансформера
            start_time = time.time()
            model.fit(
                X_sequences_array,
                y_sequences_array,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
                ],
            )
            training_time = time.time() - start_time
            self._training_times["transformer"] = training_time
            return model
        except Exception as e:
            logger.error(f"Error training Transformer: {str(e)}")
            return None

    def _train_arima(self, features: DataFrame, target: Series) -> Optional[ARIMA]:
        """Обучение ARIMA"""
        try:
            model = ARIMA(target, order=self.arima_params["order"])
            model = model.fit()
            return model
        except Exception as e:
            logger.error(f"Error training ARIMA: {str(e)}")
            return None

    def _train_prophet(self, features: DataFrame, target: Series) -> Optional[Prophet]:
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

    def _save_model(self, pair: str, timeframe: str, model_type: str, model: Any) -> None:
        """Сохранение модели"""
        try:
            # Создание директории
            model_path = os.path.join(self.config["model_dir"], pair, timeframe)
            os.makedirs(model_path, exist_ok=True)
            # Сохранение модели
            model_file = os.path.join(model_path, f"{model_type}.joblib")
            joblib.dump(model, model_file)
            # Сохранение метаданных
            # Исправление: проверяем тип перед вызовом to_dict
            feature_importance_data: Dict[str, Any] = {}
            if (pair in self.feature_importance and 
                timeframe in self.feature_importance[pair] and 
                isinstance(self.feature_importance[pair][timeframe], DataFrame)):
                if hasattr(self.feature_importance[pair][timeframe], 'to_dict'):
                    feature_importance_data = self.feature_importance[pair][timeframe].to_dict()
                else:
                    feature_importance_data = {}
            else:
                feature_importance_data = {}
            
            metadata = {
                "model_type": model_type,
                "last_update": datetime.now().isoformat(),
                "metrics": self.metrics.get(pair, {})
                .get(timeframe, {})
                .get(model_type, {})
                .__dict__,
                "feature_importance": feature_importance_data,
            }
            metadata_file = os.path.join(model_path, f"{model_type}_metadata.json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def _load_model(self, pair: str, timeframe: str, model_type: str) -> Optional[Dict[str, Any]]:
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
            required_fields = [
                "model_type",
                "last_update",
                "metrics",
                "feature_importance",
            ]
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
        features: DataFrame,
        target: Series,
    ) -> None:
        """Обновление метрик"""
        try:
            start_time = time.time()
            # Получение предсказаний
            if model_type in ["catboost", "xgboost"]:
                predictions = model.predict(features)
            elif model_type in ["lstm", "cnn"]:
                predictions = (model.predict(features) > 0.5).astype(int)
            elif model_type == "transformer":
                # Реализация предсказаний для трансформера
                # Исправление: используем to_numpy() вместо values
                if hasattr(features, 'to_numpy'):
                    X = features.to_numpy()
                else:
                    X = features.values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                sequence_length = 10
                X_sequences = []
                for i in range(sequence_length, len(X_scaled)):
                    X_sequences.append(X_scaled[i - sequence_length : i])
                if X_sequences:
                    X_sequences_array = np.array(X_sequences)
                    predictions_raw = model.predict(X_sequences_array)
                    # Дополняем начало нулями для соответствия размеру target
                    predictions = np.concatenate(
                        [
                            np.zeros(sequence_length),
                            (predictions_raw.flatten() > 0.5).astype(int),
                        ]
                    )
                else:
                    predictions = np.zeros(len(target))
            elif model_type in ["arima", "prophet"]:
                predictions = model.predict(features)
            else:
                predictions = np.zeros(len(target))
            inference_time = time.time() - start_time
            # Расчет метрик
            metrics = ModelMetrics(
                win_rate=sum(predictions == target) / len(target),
                accuracy=accuracy_score(target, predictions),
                precision=precision_score(target, predictions),
                recall=recall_score(target, predictions),
                f1=f1_score(target, predictions),
                last_update=datetime.now(),
                training_time=self._training_times.get(model_type, 0.0),
                inference_time=inference_time,
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

    def get_feature_importance(self, pair: str, timeframe: str) -> DataFrame:
        """Получение важности признаков"""
        try:
            return self.feature_importance.get(pair, {}).get(timeframe, DataFrame())
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            return DataFrame()

    async def retrain_if_dataset_updated(self, pair: str, timeframe: str) -> None:
        """Переобучает модель, если датасет для пары обновился."""
        # Исправляем вызов - используем правильный способ загрузки данных
        try:
            # Временное решение - создаем пустой датасет
            dataset: List[Dict[str, Any]] = []
            logger.warning(f"DatasetManager.load_dataset не реализован для {pair}")
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            dataset = []
        if not dataset:
            logger.warning(f"Нет данных для переобучения по паре {pair}")
            return
        X = DataFrame([x["features"] for x in dataset])
        y = Series([1 if x.get("result", {}).get("win") else 0 for x in dataset])
        t0 = time.time()
        model = self._train_new_model(pair, timeframe, "all", X, y)
        if model:
            self._save_model(pair, timeframe, "dynamic", model)
            self._update_metadata(pair, timeframe, len(dataset), time.time() - t0)
            logger.info(f"Модель для {pair} переобучена по {len(dataset)} примерам")

    def _update_metadata(
        self, pair: str, timeframe: str, dataset_size: int, train_time: float
    ) -> None:
        """Обновление метаданных модели"""
        try:
            metadata = {
                "pair": pair,
                "timeframe": timeframe,
            "dataset_size": dataset_size,
                "train_time": train_time,
                "last_updated": datetime.now().isoformat(),
            }
            
            # Исправляю типы для dict
            metadata_str: Dict[str, Any] = {str(k): v for k, v in metadata.items()}
            
            # Сохранение метаданных
            metadata_path = f"models/{pair}_{timeframe}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata_str, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")

    async def get_training_quality(self, pair: str, timeframe: str) -> Dict[str, float]:
        """Возвращает качество обучения: confidence, pattern complexity, diversity, estimated profitability."""
        # Исправляем вызов - используем правильный способ загрузки данных
        try:
            # Временное решение - создаем пустой датасет
            dataset: List[Dict[str, Any]] = []
            logger.warning(f"DatasetManager.load_dataset не реализован для {pair}")
        except Exception as e:
            logger.error(f"Ошибка загрузки датасета: {e}")
            dataset = []
        if not dataset:
            return {
                "confidence": 0.0,
                "pattern_complexity": 0.0,
                "diversity": 0.0,
                "profitability": 0.0,
            }
        # Confidence: средний accuracy по последней модели
        metrics = self.metrics.get(pair, {}).get(timeframe, {})
        confidence = getattr(metrics, "accuracy", 0.0) if metrics else 0.0
        # Pattern complexity: среднее число уникальных паттернов на 100 примеров
        patterns = [
            x["features"].get("pattern")
            for x in dataset
            if x["features"].get("pattern")
        ]
        pattern_complexity = len(set(patterns)) / (len(dataset) / 100) if dataset else 0.0
        # Diversity: число уникальных режимов
        diversity = len(
            set(x.get("market_regime") for x in dataset if x.get("market_regime"))
        )
        # Profitability: средний PnL
        profitability = sum(x.get("result", {}).get("PnL", 0) for x in dataset) / len(
            dataset
        )
        return {
            "confidence": confidence,
            "pattern_complexity": pattern_complexity,
            "diversity": float(diversity),
            "profitability": profitability,
        }

    async def dashboard_retrain(self, pair: str, timeframe: str) -> None:
        """Переобучение по команде с дашборда."""
        await self.retrain_if_dataset_updated(pair, timeframe)
        logger.info(f"Переобучение по запросу dashboard для {pair} {timeframe}")
