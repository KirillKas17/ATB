"""
Модуль для машинного обучения и моделей.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
from shared.numpy_utils import np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Настройка логирования
logger = logging.getLogger(__name__)


class MLModelManager:
    """Менеджер машинного обучения"""

    def __init__(self, models_dir: str = "models") -> None:
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        self.model_metrics: Dict[str, Dict[str, float]] = {}
        # Создание директории для моделей
        os.makedirs(models_dir, exist_ok=True)

    def train_models(self, X: DataFrame, y: Series) -> Dict[str, float]:
        """Обучение моделей"""
        try:
            # Масштабирование данных
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers["main"] = scaler
            # Сохранение имен признаков
            self.feature_names = list(X.columns)
            # Обучение моделей
            models = {
                "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "logistic_regression": LogisticRegression(random_state=42, max_iter=1000),
                "svm": SVC(probability=True, random_state=42),
                "neural_network": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
            }
            for name, model in models.items():
                # Обучение
                model.fit(X_scaled, y)
                # Предсказания
                y_pred = model.predict(X_scaled)
                # Метрики
                metrics = {
                    "accuracy": float(accuracy_score(y, y_pred)),
                    "precision": float(
                        precision_score(y, y_pred, average="weighted", zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y, y_pred, average="weighted", zero_division=0)
                    ),
                    "f1": float(
                        f1_score(y, y_pred, average="weighted", zero_division=0)
                    ),
                }
                # Сохранение модели и метрик
                self.models[name] = model
                self.model_metrics[name] = metrics
                logger.info(
                    f"Модель {name} обучена. Accuracy: {metrics['accuracy']:.3f}"
                )
            # Сохранение моделей
            self._save_models()
            return self._get_best_model_metrics()
        except Exception as e:
            logger.error(f"Ошибка при обучении моделей: {e}")
            return {}

    def predict(
        self, X: DataFrame, model_name: str = "ensemble"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Предсказание с использованием модели"""
        try:
            if model_name == "ensemble":
                return self._ensemble_predict(X)
            if model_name not in self.models:
                raise ValueError(f"Модель {model_name} не найдена")
            # Масштабирование
            scaler = self.scalers.get("main")
            if scaler is None:
                raise ValueError("Scaler не найден")
            X_scaled = scaler.transform(X)
            # Предсказание
            model = self.models[model_name]
            predictions = model.predict(X_scaled)
            probabilities = model.predict_proba(X_scaled)
            return predictions, probabilities
        except Exception as e:
            logger.error(f"Ошибка при предсказании: {e}")
            return np.array([]), np.array([])

    def _ensemble_predict(self, X: DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Ансамблевое предсказание"""
        try:
            if not self.models:
                raise ValueError("Нет обученных моделей")
            # Масштабирование
            scaler = self.scalers.get("main")
            if scaler is None:
                raise ValueError("Scaler не найден")
            X_scaled = scaler.transform(X)
            # Предсказания всех моделей
            all_predictions = []
            all_probabilities = []
            for name, model in self.models.items():
                pred = model.predict(X_scaled)
                proba = model.predict_proba(X_scaled)
                all_predictions.append(pred)
                all_probabilities.append(proba)
            # Голосование
            ensemble_predictions = np.mean(all_predictions, axis=0)
            ensemble_probabilities = np.mean(all_probabilities, axis=0)
            # Округление предсказаний
            ensemble_predictions = np.round(ensemble_predictions).astype(int)
            return ensemble_predictions, ensemble_probabilities
        except Exception as e:
            logger.error(f"Ошибка при ансамблевом предсказании: {e}")
            return np.array([]), np.array([])

    def get_feature_importance(
        self, model_name: str = "random_forest"
    ) -> Dict[str, float]:
        """Получение важности признаков"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Модель {model_name} не найдена")
            model = self.models[model_name]
            if hasattr(model, "feature_importances_"):
                importance = model.feature_importances_
            elif hasattr(model, "coef_"):
                importance = np.abs(model.coef_[0])
            else:
                return {}
            # Создание словаря важности признаков
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = float(importance[i])
            return feature_importance
        except Exception as e:
            logger.error(f"Ошибка при получении важности признаков: {e}")
            return {}

    def get_model_metrics(self) -> Dict[str, Dict[str, float]]:
        """Получение метрик всех моделей"""
        return self.model_metrics.copy()

    def _get_best_model_metrics(self) -> Dict[str, float]:
        """Получение метрик лучшей модели"""
        if not self.model_metrics:
            return {}
        # Поиск лучшей модели по F1-score
        best_model = max(self.model_metrics.items(), key=lambda x: x[1]["f1"])
        return best_model[1]  # Возвращаем только метрики, без имени модели

    def _save_models(self) -> None:
        """Сохранение моделей"""
        try:
            for name, model in self.models.items():
                model_path = os.path.join(self.models_dir, f"{name}.joblib")
                joblib.dump(model, model_path)
                logger.info(f"Модель {name} сохранена в {model_path}")
            # Сохранение scaler
            scaler_path = os.path.join(self.models_dir, "scaler.joblib")
            joblib.dump(self.scalers["main"], scaler_path)
            # Сохранение метрик
            metrics_path = os.path.join(self.models_dir, "metrics.json")
            import json

            with open(metrics_path, "w") as f:
                json.dump(self.model_metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Ошибка при сохранении моделей: {e}")

    def load_models(self) -> bool:
        """Загрузка сохраненных моделей"""
        try:
            # Загрузка scaler
            scaler_path = os.path.join(self.models_dir, "scaler.joblib")
            if os.path.exists(scaler_path):
                self.scalers["main"] = joblib.load(scaler_path)
            # Загрузка моделей
            for name in [
                "random_forest",
                "gradient_boosting",
                "logistic_regression",
                "svm",
                "neural_network",
            ]:
                model_path = os.path.join(self.models_dir, f"{name}.joblib")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    logger.info(f"Модель {name} загружена")
            # Загрузка метрик
            metrics_path = os.path.join(self.models_dir, "metrics.json")
            if os.path.exists(metrics_path):
                import json

                with open(metrics_path, "r") as f:
                    self.model_metrics = json.load(f)
            return len(self.models) > 0
        except Exception as e:
            logger.error(f"Ошибка при загрузке моделей: {e}")
            return False

    def update_model(
        self, X_new: DataFrame, y_new: Series, model_name: str = "random_forest"
    ) -> bool:
        """Обновление модели новыми данными"""
        try:
            if model_name not in self.models:
                raise ValueError(f"Модель {model_name} не найдена")
            # Масштабирование новых данных
            scaler = self.scalers.get("main")
            if scaler is None:
                raise ValueError("Scaler не найден")
            X_new_scaled = scaler.transform(X_new)
            # Обновление модели
            model = self.models[model_name]
            model.fit(X_new_scaled, y_new)
            # Обновление метрик
            y_pred = model.predict(X_new_scaled)
            metrics = {
                "accuracy": float(accuracy_score(y_new, y_pred)),
                "precision": float(
                    precision_score(y_new, y_pred, average="weighted", zero_division=0)
                ),
                "recall": float(
                    recall_score(y_new, y_pred, average="weighted", zero_division=0)
                ),
                "f1": float(
                    f1_score(y_new, y_pred, average="weighted", zero_division=0)
                ),
            }
            self.model_metrics[model_name] = metrics
            # Сохранение обновленной модели
            self._save_models()
            logger.info(f"Модель {model_name} обновлена")
            return True
        except Exception as e:
            logger.error(f"Ошибка при обновлении модели: {e}")
            return False

    def evaluate_model_performance(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Оценка производительности модели на новых данных"""
        try:
            if not self.models:
                raise ValueError("Нет обученных моделей")
            # Масштабирование
            scaler = self.scalers.get("main")
            if scaler is None:
                raise ValueError("Scaler не найден")
            X_scaled = scaler.transform(X)
            # Оценка всех моделей
            best_metrics = {}
            best_f1: float = 0.0  # Явно указываем тип float
            for name, model in self.models.items():
                y_pred = model.predict(X_scaled)
                metrics = {
                    "accuracy": float(accuracy_score(y, y_pred)),
                    "precision": float(
                        precision_score(y, y_pred, average="weighted", zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y, y_pred, average="weighted", zero_division=0)
                    ),
                    "f1": float(
                        f1_score(y, y_pred, average="weighted", zero_division=0)
                    ),
                }
                if metrics["f1"] > best_f1:
                    best_f1 = metrics["f1"]
                    best_metrics = metrics
            return best_metrics
        except Exception as e:
            logger.error(f"Ошибка при оценке производительности: {e}")
            return {}
