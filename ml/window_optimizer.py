import warnings
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore")


@dataclass
class WindowConfig:
    """Конфигурация оптимизатора окна"""

    min_window: int = 150
    max_window: int = 2000
    default_window: int = 300
    n_trials: int = 100
    cv_splits: int = 5
    test_size: float = 0.2
    random_state: int = 42
    model_type: str = "rf"  # 'rf' или 'gb'
    feature_importance_threshold: float = 0.01
    ensemble_size: int = 3
    update_interval: int = 24  # часов


@dataclass
class WindowMetrics:
    """Метрики оптимизатора окна"""

    mse: float
    r2: float
    feature_importance: Dict[str, float]
    prediction_time: float
    last_update: datetime
    window_size: int
    confidence: float


class WindowSizeOptimizer:
    """Оптимизатор размера окна"""

    def __init__(self, config: Optional[WindowConfig] = None):
        """Инициализация оптимизатора"""
        self.config = config or WindowConfig()

        # Модели
        self.models = []
        self.scalers = []
        self.feature_names = None

        # Метрики
        self.metrics = None

        # Режимы рынка
        self.regime_mapping = {
            "трендовый": 0,
            "боковой": 1,
            "разворотный": 2,
            "манипуляционный": 3,
            "волатильный": 4,
            "неизвестный": 5,
        }

        # Кэш
        self._prediction_cache = {}
        self._feature_cache = {}

        # Загрузка моделей
        self._load_models()

    def _load_models(self):
        """Загрузка моделей"""
        try:
            # Создание директории для моделей
            models_dir = Path("models")
            models_dir.mkdir(parents=True, exist_ok=True)

            for i in range(self.config.ensemble_size):
                model_path = models_dir / f"window_size_model_{i}.pkl"
                scaler_path = models_dir / f"window_size_scaler_{i}.pkl"

                try:
                    if not model_path.exists() or not scaler_path.exists():
                        logger.warning(f"Файлы модели {i} не найдены: {model_path}, {scaler_path}")
                        continue

                    model = joblib.load(model_path)
                    scaler = joblib.load(scaler_path)

                    if model is None or scaler is None:
                        logger.warning(f"Модель {i} или скейлер загружены как None")
                        continue

                    if not hasattr(model, "predict"):
                        logger.warning(f"Модель {i} не имеет метода predict")
                        continue

                    self.models.append(model)
                    self.scalers.append(scaler)
                    logger.info(f"Модель {i} успешно загружена")
                except Exception as e:
                    logger.warning(f"Не удалось загрузить модель {i}: {e}")
                    continue

            if not self.models:
                logger.warning(
                    "Не удалось загрузить ни одной модели, используем размер окна по умолчанию"
                )
                return

        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            return

    def _initialize_models(self):
        """Инициализация новых моделей"""
        try:
            for _ in range(self.config.ensemble_size):
                if self.config.model_type == "rf":
                    model = RandomForestRegressor(
                        n_estimators=100, max_depth=10, random_state=self.config.random_state
                    )
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=100, max_depth=5, random_state=self.config.random_state
                    )

                scaler = RobustScaler()
                self.models.append(model)
                self.scalers.append(scaler)

        except Exception as e:
            logger.error(f"Ошибка инициализации моделей: {e}")
            raise

    @lru_cache(maxsize=1000)
    def extract_features(self, df: pd.DataFrame, meta: Dict) -> Dict:
        """Извлечение признаков с кэшированием"""
        try:
            features = {
                "volatility": meta.get("volatility", 0.5),
                "trend_strength": meta.get("trend_strength", 0.5),
                "regime_encoded": self.encode_regime(meta.get("regime", "неизвестный")),
                "atr": df["atr"].iloc[-1] if "atr" in df.columns else 0,
                "adx": df["adx"].iloc[-1] if "adx" in df.columns else 0,
                "rsi": df["rsi"].iloc[-1] if "rsi" in df.columns else 50,
                "bollinger_width": (
                    (df["bb_upper"].iloc[-1] - df["bb_lower"].iloc[-1])
                    if all(x in df.columns for x in ["bb_upper", "bb_lower"])
                    else 0
                ),
                "volume_ma_ratio": (
                    (df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1])
                    if "volume" in df.columns
                    else 1.0
                ),
                "price_range": (
                    (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-1]
                    if all(x in df.columns for x in ["high", "low", "close"])
                    else 0
                ),
                "momentum": (
                    (df["close"].iloc[-1] / df["close"].iloc[-5] - 1)
                    if "close" in df.columns
                    else 0
                ),
            }

            # Нормализация признаков
            for key in features:
                if isinstance(features[key], (int, float)):
                    features[key] = float(features[key])

            return features

        except Exception as e:
            logger.error(f"Ошибка при извлечении признаков: {e}")
            return self._get_default_features()

    def _get_default_features(self) -> Dict:
        """Получение признаков по умолчанию"""
        return {
            "volatility": 0.5,
            "trend_strength": 0.5,
            "regime_encoded": self.encode_regime("неизвестный"),
            "atr": 0,
            "adx": 0,
            "rsi": 50,
            "bollinger_width": 0,
            "volume_ma_ratio": 1.0,
            "price_range": 0,
            "momentum": 0,
        }

    def encode_regime(self, regime: str) -> int:
        """Кодирование режима рынка"""
        return self.regime_mapping.get(regime.lower(), 5)

    def _optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Оптимизация гиперпараметров"""

        def objective(trial):
            if self.config.model_type == "rf":
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 30),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                }
                model = RandomForestRegressor(**params, random_state=self.config.random_state)
            else:
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                }
                model = GradientBoostingRegressor(**params, random_state=self.config.random_state)

            cv = TimeSeriesSplit(n_splits=self.config.cv_splits)
            scores = []

            for train_idx, val_idx in cv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)

            return np.mean(scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials)

        return study.best_params

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Обучение моделей"""
        try:
            # Оптимизация гиперпараметров
            best_params = self._optimize_hyperparameters(X, y)
            logger.info(f"Лучшие параметры: {best_params}")

            # Разделение данных
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )

            # Обучение ансамбля
            for i in range(self.config.ensemble_size):
                if self.config.model_type == "rf":
                    model = RandomForestRegressor(
                        **best_params, random_state=self.config.random_state + i
                    )
                else:
                    model = GradientBoostingRegressor(
                        **best_params, random_state=self.config.random_state + i
                    )

                scaler = RobustScaler()

                # Масштабирование признаков
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Обучение модели
                model.fit(X_train_scaled, y_train)

                # Оценка
                val_pred = model.predict(X_val_scaled)
                mse = mean_squared_error(y_val, val_pred)
                r2 = r2_score(y_val, val_pred)

                logger.info(f"Модель {i} - MSE: {mse:.3f}, R2: {r2:.3f}")

                # Сохранение
                self.models[i] = model
                self.scalers[i] = scaler

                joblib.dump(model, f"models/window_size_model_{i}.pkl")
                joblib.dump(scaler, f"models/window_size_scaler_{i}.pkl")

            # Обновление метрик
            self._update_metrics(X_val, y_val)

        except Exception as e:
            logger.error(f"Ошибка при обучении: {e}")
            raise

    def _update_metrics(self, X: pd.DataFrame, y: pd.Series):
        """Обновление метрик"""
        try:
            predictions = []
            for model, scaler in zip(self.models, self.scalers):
                X_scaled = scaler.transform(X)
                pred = model.predict(X_scaled)
                predictions.append(pred)

            ensemble_pred = np.mean(predictions, axis=0)

            self.metrics = WindowMetrics(
                mse=mean_squared_error(y, ensemble_pred),
                r2=r2_score(y, ensemble_pred),
                feature_importance=self._get_feature_importance(),
                prediction_time=0.0,
                last_update=datetime.now(),
                window_size=int(np.mean(ensemble_pred)),
                confidence=1.0 - np.std(predictions, axis=0).mean() / np.mean(ensemble_pred),
            )

        except Exception as e:
            logger.error(f"Ошибка обновления метрик: {e}")

    def _get_feature_importance(self) -> Dict[str, float]:
        """Получение важности признаков"""
        try:
            importance = {}
            for model in self.models:
                if hasattr(model, "feature_importances_"):
                    for name, imp in zip(self.feature_names, model.feature_importances_):
                        importance[name] = importance.get(name, 0) + imp

            # Нормализация
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            # Фильтрация по порогу
            importance = {
                k: v for k, v in importance.items() if v >= self.config.feature_importance_threshold
            }

            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        except Exception as e:
            logger.error(f"Ошибка получения важности признаков: {e}")
            return {}

    def predict(self, features: Dict) -> Tuple[int, float]:
        """Предсказание размера окна"""
        try:
            if not self.models:
                logger.warning("Модели не загружены, используем размер окна по умолчанию")
                return self.config.default_window, 0.0

            # Преобразование признаков в DataFrame
            X = pd.DataFrame([features])

            # Предсказания моделей
            predictions = []
            confidences = []

            for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
                if model is None or scaler is None:
                    logger.warning(f"Пропускаем None модель или скейлер {i}")
                    continue

                if not hasattr(model, "predict"):
                    logger.warning(f"Модель {i} не имеет метода predict")
                    continue

                try:
                    # Масштабирование признаков
                    X_scaled = scaler.transform(X)

                    # Предсказание
                    pred = model.predict(X_scaled)[0]
                    predictions.append(pred)

                    # Оценка уверенности
                    if hasattr(model, "predict_proba"):
                        conf = model.predict_proba(X_scaled)[0].max()
                    else:
                        conf = 1.0
                    confidences.append(conf)
                except Exception as e:
                    logger.warning(f"Ошибка предсказания модели {i}: {e}")
                    continue

            if not predictions:
                logger.warning("Нет валидных предсказаний, используем размер окна по умолчанию")
                return self.config.default_window, 0.0

            # Усреднение предсказаний
            window_size = int(np.mean(predictions))
            confidence = np.mean(confidences) if confidences else 0.0

            # Ограничение размера окна
            window_size = max(self.config.min_window, min(window_size, self.config.max_window))

            return window_size, confidence

        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return self.config.default_window, 0.0

    def get_metrics(self) -> Optional[WindowMetrics]:
        """Получение метрик"""
        return self.metrics

    def reset_metrics(self):
        """Сброс метрик"""
        self.metrics = None
        self._prediction_cache.clear()
        self._feature_cache.clear()
