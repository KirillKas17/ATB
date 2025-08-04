"""
Онлайн-обучающий анализатор решений.
"""

import numpy as np
from typing import Dict, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class OnlineLearningReasoner:
    """Онлайн-обучающий анализатор решений"""

    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.online_model = LogisticRegression(random_state=42)
        self.metrics: Dict[str, float] = {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        self.feature_names: List[str] = []
        self.drift_detector = None
        self.predictions_history: List[int] = []
        self.targets_history: List[int] = []
        self.is_trained = False

    def update(self, features: Dict[str, float], target: int, prediction: int) -> None:
        """Обновление модели на основе новых данных"""
        if not self.is_trained:
            # Первоначальное обучение
            X = np.array([list(features.values())])
            y = np.array([target])
            self.online_model.fit(X, y)
            self.feature_names = list(features.keys())
            self.is_trained = True
        else:
            # Онлайн обновление
            X = np.array([list(features.values())])
            y = np.array([target])
            # Частичное обновление весов
            self.online_model.partial_fit(X, y, classes=np.array([0, 1]))
        # Обновление истории
        self.predictions_history.append(prediction)
        self.targets_history.append(target)
        # Обновление метрик
        self._update_metrics()

    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Предсказание с уверенностью"""
        if not self.is_trained:
            return 0, 0.0
        X = np.array([list(features.values())])
        prediction = self.online_model.predict(X)[0]
        confidence = np.max(self.online_model.predict_proba(X)[0])
        return int(prediction), float(confidence)

    def get_metrics(self) -> Dict[str, float]:
        """Получение метрик производительности"""
        return self.metrics.copy()

    def detect_drift(self) -> bool:
        """Обнаружение дрейфа данных"""
        if len(self.predictions_history) < 100:
            return False
        # Простая эвристика для обнаружения дрейфа
        recent_accuracy = accuracy_score(
            self.targets_history[-50:], self.predictions_history[-50:]
        )
        overall_accuracy = self.metrics["accuracy"]
        drift_detected: bool = abs(recent_accuracy - overall_accuracy) > 0.1
        return drift_detected

    def _update_metrics(self) -> None:
        """Обновление метрик производительности"""
        if len(self.targets_history) < 10:
            return
        y_true = np.array(self.targets_history)
        y_pred = np.array(self.predictions_history)
        self.metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        self.metrics["precision"] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        self.metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        self.metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
