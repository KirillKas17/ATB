"""
Модуль адаптивного обучения в реальном времени.
Обеспечивает непрерывную адаптацию моделей к изменениям рынка
в реальном времени с минимальной задержкой.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler

# Type aliases
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class AdaptationConfig:
    """Конфигурация адаптивного обучения."""

    # Временные параметры
    adaptation_window: int = 300  # 5 минут
    retrain_interval: int = 3600  # 1 час
    prediction_horizon: int = 60  # 1 минута
    # Пороги адаптации
    performance_threshold: float = 0.7
    drift_threshold: float = 0.1
    adaptation_threshold: float = 0.05
    # Параметры моделей
    max_models: int = 5
    ensemble_size: int = 3
    feature_window: int = 100
    # Параметры обучения
    learning_rate: float = 0.01
    batch_size: int = 32
    max_iterations: int = 1000
    # Пути сохранения
    models_path: str = "models/live_adaptation"
    cache_path: str = "cache/adaptation"


@dataclass
class ModelPerformance:
    """Производительность модели."""

    model_id: str
    mse: float
    mae: float
    r2_score: float
    prediction_accuracy: float
    adaptation_speed: float
    last_update: datetime
    is_active: bool = True


@dataclass
class MarketDrift:
    """Информация о дрифте рынка."""

    timestamp: datetime
    drift_score: float
    feature_drift: Dict[str, float]
    target_drift: float
    confidence: float
    detected_regime: str


class LiveAdaptation:
    """
    Система адаптивного обучения в реальном времени.
    Обеспечивает:
    - Непрерывный мониторинг производительности моделей
    - Обнаружение дрифта рынка
    - Адаптивное переобучение моделей
    - Ансамблевое предсказание
    """

    def __init__(self, config: Optional[AdaptationConfig] = None) -> None:
        """Инициализация системы адаптивного обучения."""
        self.config = config or AdaptationConfig()
        # Создание директорий
        Path(self.config.models_path).mkdir(parents=True, exist_ok=True)
        Path(self.config.cache_path).mkdir(parents=True, exist_ok=True)
        # Модели
        self.models: Dict[str, Any] = {}
        self.ensemble_models: Dict[str, Any] = {}
        self.model_performances: Dict[str, ModelPerformance] = {}
        # Данные
        self.feature_buffer: List[List[float]] = []
        self.target_buffer: List[float] = []
        self.prediction_buffer: List[Dict[str, Any]] = []
        # Мониторинг дрифта
        self.drift_history: List[MarketDrift] = []
        self.last_drift_check = datetime.now()
        # Состояние
        self.is_adapting: bool = False
        self.adaptation_task: Optional[asyncio.Task] = None
        # Предобработка
        self.scaler = StandardScaler()
        self.is_scaler_fitted: bool = False
        logger.info("LiveAdaptation initialized")

    async def start_adaptation(self) -> None:
        """Запуск адаптивного обучения."""
        if self.is_adapting:
            logger.warning("Live adaptation is already running")
            return
        self.is_adapting = True
        self.adaptation_task = asyncio.create_task(self._adaptation_loop())
        logger.info("Live adaptation started")

    async def stop_adaptation(self) -> None:
        """Остановка адаптивного обучения."""
        if not self.is_adapting:
            return
        self.is_adapting = False
        if self.adaptation_task:
            self.adaptation_task.cancel()
            try:
                await self.adaptation_task
            except asyncio.CancelledError:
                pass
        logger.info("Live adaptation stopped")

    async def _adaptation_loop(self) -> None:
        """Основной цикл адаптивного обучения."""
        logger.info("Starting live adaptation loop")
        while self.is_adapting:
            try:
                # Проверка необходимости адаптации
                if await self._should_adapt():
                    await self._perform_adaptation()
                # Проверка дрифта рынка
                await self._check_market_drift()
                # Обновление производительности моделей
                await self._update_model_performance()
                # Очистка старых моделей
                await self._cleanup_old_models()
                # Ожидание следующей итерации
                await asyncio.sleep(self.config.adaptation_window)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptation loop: {e}")
                await asyncio.sleep(60)

    async def _should_adapt(self) -> bool:
        """Проверка необходимости адаптации."""
        try:
            # Проверка производительности
            if not self.model_performances:
                return True
            avg_performance = np.mean(
                [
                    perf.prediction_accuracy
                    for perf in self.model_performances.values()
                    if perf.is_active
                ]
            )
            if avg_performance < self.config.performance_threshold:
                logger.info(f"Low performance detected: {avg_performance:.3f}")
                return True
            # Проверка дрифта
            if self.drift_history:
                latest_drift = self.drift_history[-1]
                if latest_drift.drift_score > self.config.drift_threshold:
                    logger.info(
                        f"Market drift detected: {latest_drift.drift_score:.3f}"
                    )
                    return True
            # Проверка времени последней адаптации
            if not self.model_performances:
                return True
            last_adaptation = max(
                perf.last_update for perf in self.model_performances.values()
            )
            if datetime.now() - last_adaptation > timedelta(
                seconds=self.config.retrain_interval
            ):
                logger.info("Scheduled adaptation triggered")
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking adaptation need: {e}")
            return True

    async def _perform_adaptation(self) -> None:
        """Выполнение адаптации моделей."""
        try:
            logger.info("Starting model adaptation")
            # Подготовка данных
            X, y = await self._prepare_training_data()
            if X is None or y is None or len(X) < 50:
                logger.warning("Insufficient data for adaptation")
                return
            # Создание новых моделей
            new_models = await self._create_models(X, y)
            # Оценка новых моделей
            model_scores = await self._evaluate_models(new_models, X, y)
            # Выбор лучших моделей
            best_models = await self._select_best_models(model_scores)
            # Обновление ансамбля
            await self._update_ensemble(best_models)
            # Сохранение моделей
            await self._save_models()
            logger.info(
                f"Adaptation completed. Active models: {len(self.ensemble_models)}"
            )
        except Exception as e:
            logger.error(f"Error performing adaptation: {e}")

    async def _prepare_training_data(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Подготовка данных для обучения."""
        try:
            if len(self.feature_buffer) < 50:
                return None, None
            # Преобразование в numpy массивы
            X = np.array(list(self.feature_buffer))
            y = np.array(list(self.target_buffer))
            # Нормализация признаков
            if not self.is_scaler_fitted:
                X = self.scaler.fit_transform(X)
                self.is_scaler_fitted = True
            else:
                X = self.scaler.transform(X)
            return X, y
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None

    async def _create_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Создание новых моделей."""
        models = {}
        try:
            # Random Forest
            rf_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            rf_model.fit(X, y)
            models["rf"] = rf_model
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=self.config.learning_rate,
                max_depth=6,
                random_state=42,
            )
            gb_model.fit(X, y)
            models["gb"] = gb_model
            # Ridge Regression
            ridge_model = Ridge(alpha=1.0, random_state=42)
            ridge_model.fit(X, y)
            models["ridge"] = ridge_model
            # Linear Regression
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            models["lr"] = lr_model
            logger.info(f"Created {len(models)} new models")
        except Exception as e:
            logger.error(f"Error creating models: {e}")
        return models

    async def _evaluate_models(
        self, models: Dict[str, Any], X: np.ndarray, y: np.ndarray
    ) -> Dict[str, ModelPerformance]:
        """Оценка производительности моделей."""
        performances = {}
        try:
            for model_id, model in models.items():
                # Предсказания
                y_pred = model.predict(X)
                # Метрики
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2_score = model.score(X, y)
                # Точность предсказания (процент правильных направлений)
                direction_accuracy = self._calculate_direction_accuracy(y, y_pred)
                # Скорость адаптации (время обучения)
                adaptation_speed = 1.0  # Заглушка
                performances[model_id] = ModelPerformance(
                    model_id=model_id,
                    mse=mse,
                    mae=mae,
                    r2_score=r2_score,
                    prediction_accuracy=direction_accuracy,
                    adaptation_speed=adaptation_speed,
                    last_update=datetime.now(),
                )
            logger.info(f"Evaluated {len(performances)} models")
        except Exception as e:
            logger.error(f"Error evaluating models: {e}")
        return performances

    async def _select_best_models(
        self, performances: Dict[str, ModelPerformance]
    ) -> List[str]:
        """Выбор лучших моделей для ансамбля."""
        try:
            # Сортировка по точности предсказания
            sorted_models = sorted(
                performances.items(),
                key=lambda x: x[1].prediction_accuracy,
                reverse=True,
            )
            # Выбор топ моделей
            best_models = [
                model_id for model_id, _ in sorted_models[: self.config.ensemble_size]
            ]
            logger.info(f"Selected best models: {best_models}")
            return best_models
        except Exception as e:
            logger.error(f"Error selecting best models: {e}")
            return []

    async def _update_ensemble(self, best_model_ids: List[str]) -> None:
        """Обновление ансамбля моделей."""
        try:
            # Обновление списка ансамблевых моделей
            self.ensemble_models = {
                model_id: self.models[model_id] for model_id in best_model_ids
            }
            # Обновление производительности
            for model_id in best_model_ids:
                if model_id in self.model_performances:
                    self.model_performances[model_id].is_active = True
                    self.model_performances[model_id].last_update = datetime.now()
            # Деактивация старых моделей
            for model_id in self.model_performances:
                if model_id not in best_model_ids:
                    self.model_performances[model_id].is_active = False
            logger.info(f"Ensemble updated: {self.ensemble_models}")
        except Exception as e:
            logger.error(f"Error updating ensemble: {e}")

    async def _check_market_drift(self) -> None:
        """Проверка дрифта рынка."""
        try:
            if len(self.feature_buffer) < 100:
                return
            # Расчет дрифта признаков
            feature_drift = self._calculate_feature_drift()
            # Расчет дрифта целевой переменной
            target_drift = self._calculate_target_drift()
            # Общий дрифт
            drift_score = float(
                (np.mean(list(feature_drift.values())) + target_drift) / 2
            )
            # Определение режима рынка
            detected_regime = self._detect_market_regime(drift_score)
            # Создание записи о дрифте
            drift_record = MarketDrift(
                timestamp=datetime.now(),
                drift_score=drift_score,
                feature_drift=feature_drift,
                target_drift=target_drift,
                confidence=0.8,  # Заглушка
                detected_regime=detected_regime,
            )
            self.drift_history.append(drift_record)
            # Ограничение истории
            if len(self.drift_history) > 1000:
                self.drift_history = self.drift_history[-1000:]
            logger.debug(f"Market drift: {drift_score:.3f}, regime: {detected_regime}")
        except Exception as e:
            logger.error(f"Error checking market drift: {e}")

    def _calculate_feature_drift(self) -> Dict[str, float]:
        """Расчет дрифта признаков."""
        try:
            if len(self.feature_buffer) < 50:
                return {}
            # Разделение на старые и новые данные
            split_point = len(self.feature_buffer) // 2
            old_features = list(self.feature_buffer)[:split_point]
            new_features = list(self.feature_buffer)[split_point:]
            old_features_array = np.array(old_features)
            new_features_array = np.array(new_features)
            # Расчет статистик
            drift_scores: Dict[str, float] = {}
            for i in range(old_features_array.shape[1]):
                old_mean = float(np.mean(old_features_array[:, i]))
                new_mean = float(np.mean(new_features_array[:, i]))
                old_std = float(np.std(old_features_array[:, i]))
                new_std = float(np.std(new_features_array[:, i]))
                # Расстояние между распределениями
                drift = abs(new_mean - old_mean) / (old_std + 1e-8)
                drift_scores[f"feature_{i}"] = min(drift, 1.0)
            return drift_scores
        except Exception as e:
            logger.error(f"Error calculating feature drift: {e}")
            return {}

    def _calculate_target_drift(self) -> float:
        """Расчет дрифта целевой переменной."""
        try:
            if len(self.target_buffer) < 50:
                return 0.0
            # Разделение на старые и новые данные
            split_point = len(self.target_buffer) // 2
            old_targets = list(self.target_buffer)[:split_point]
            new_targets = list(self.target_buffer)[split_point:]
            old_mean = float(np.mean(old_targets))
            new_mean = float(np.mean(new_targets))
            old_std = float(np.std(old_targets))
            # Расстояние между распределениями
            drift = abs(new_mean - old_mean) / (old_std + 1e-8)
            return min(drift, 1.0)
        except Exception as e:
            logger.error(f"Error calculating target drift: {e}")
            return 0.0

    def _detect_market_regime(self, drift_score: float) -> str:
        """Определение режима рынка на основе дрифта."""
        if drift_score < 0.1:
            return "stable"
        elif drift_score < 0.3:
            return "trending"
        elif drift_score < 0.5:
            return "volatile"
        else:
            return "chaotic"

    async def _update_model_performance(self) -> None:
        """Обновление производительности моделей."""
        try:
            if not self.prediction_buffer:
                return
            # Получение последних предсказаний
            recent_predictions = list(self.prediction_buffer)[-100:]
            for prediction in recent_predictions:
                model_id = prediction.get("model_id")
                if model_id in self.model_performances:
                    # Обновление метрик
                    actual = prediction.get("actual")
                    predicted = prediction.get("predicted")
                    if actual is not None and predicted is not None:
                        error = abs(actual - predicted)
                        self.model_performances[model_id].mae = (
                            self.model_performances[model_id].mae * 0.9 + error * 0.1
                        )
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")

    async def _cleanup_old_models(self) -> None:
        """Очистка старых моделей."""
        try:
            # Удаление неактивных моделей старше 1 часа
            cutoff_time = datetime.now() - timedelta(hours=1)
            models_to_remove = []
            for model_id, performance in self.model_performances.items():
                if not performance.is_active and performance.last_update < cutoff_time:
                    models_to_remove.append(model_id)
            for model_id in models_to_remove:
                del self.model_performances[model_id]
                if model_id in self.models:
                    del self.models[model_id]
            if models_to_remove:
                logger.info(f"Cleaned up {len(models_to_remove)} old models")
        except Exception as e:
            logger.error(f"Error cleaning up old models: {e}")

    async def _save_models(self) -> None:
        """Сохранение моделей."""
        try:
            import os

            os.makedirs(self.config.models_path, exist_ok=True)
            for model_id in self.ensemble_models:
                if model_id in self.models:
                    model_path = f"{self.config.models_path}/{model_id}.joblib"
                    joblib.dump(self.models[model_id], model_path)
            # Сохранение скалера
            scaler_path = f"{self.config.models_path}/scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            logger.debug("Models saved successfully")
        except Exception as e:
            logger.error(f"Error saving models: {e}")

    def add_data_point(self, features: List[float], target: float) -> None:
        """Добавление новой точки данных."""
        try:
            self.feature_buffer.append(features)
            self.target_buffer.append(target)
        except Exception as e:
            logger.error(f"Error adding data point: {e}")

    async def predict(self, features: List[float]) -> Dict[str, Any]:
        """Предсказание с использованием ансамбля моделей."""
        try:
            if not self.ensemble_models:
                return {"prediction": 0.0, "confidence": 0.0, "model_count": 0}
            # Нормализация признаков
            X = np.array([features])
            if self.is_scaler_fitted:
                X = self.scaler.transform(X)
            # Предсказания от всех моделей
            predictions = []
            for model_id in self.ensemble_models:
                if model_id in self.models:
                    pred = self.models[model_id].predict(X)[0]
                    predictions.append(pred)
            if not predictions:
                return {"prediction": 0.0, "confidence": 0.0, "model_count": 0}
            # Ансамблевое предсказание (среднее)
            ensemble_prediction = float(np.mean(predictions))
            # Уверенность (обратная дисперсии)
            confidence = float(1.0 / (1.0 + np.var(predictions)))
            # Сохранение предсказания
            prediction_record = {
                "timestamp": datetime.now(),
                "features": features,
                "prediction": ensemble_prediction,
                "confidence": confidence,
                "model_predictions": predictions,
                "model_count": len(predictions),
            }
            self.prediction_buffer.append(prediction_record)
            return {
                "prediction": ensemble_prediction,
                "confidence": confidence,
                "model_count": len(predictions),
                "model_predictions": predictions,
            }
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"prediction": 0.0, "confidence": 0.0, "model_count": 0}

    def _calculate_direction_accuracy(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        """Расчет точности направления предсказания."""
        try:
            if len(y_true) < 2:
                return 0.0
            # Изменения в реальных значениях
            true_changes = np.diff(y_true)
            true_directions = np.sign(true_changes)
            # Изменения в предсказаниях
            pred_changes = np.diff(y_pred)
            pred_directions = np.sign(pred_changes)
            # Точность направления
            correct_directions = np.sum(true_directions == pred_directions)
            total_directions = len(true_directions)
            return (
                correct_directions / total_directions if total_directions > 0 else 0.0
            )
        except Exception as e:
            logger.error(f"Error calculating direction accuracy: {e}")
            return 0.0

    def get_adaptation_status(self) -> Dict[str, Any]:
        """Получение статуса адаптации."""
        return {
            "is_adapting": self.is_adapting,
            "active_models": len(self.ensemble_models),
            "total_models": len(self.model_performances),
            "data_points": len(self.feature_buffer),
            "last_drift_score": (
                self.drift_history[-1].drift_score if self.drift_history else 0.0
            ),
            "current_regime": (
                self.drift_history[-1].detected_regime
                if self.drift_history
                else "unknown"
            ),
        }

    def get_model_performance(self) -> Dict[str, Any]:
        """Получение производительности моделей."""
        return {
            model_id: {
                "mse": perf.mse,
                "mae": perf.mae,
                "r2_score": perf.r2_score,
                "prediction_accuracy": perf.prediction_accuracy,
                "is_active": perf.is_active,
                "last_update": perf.last_update.isoformat(),
            }
            for model_id, perf in self.model_performances.items()
        }
