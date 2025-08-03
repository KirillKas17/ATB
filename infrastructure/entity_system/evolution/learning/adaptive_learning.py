"""Адаптивное обучение для системы."""

from datetime import datetime
from typing import Any, Dict, List

import numpy as np

from domain.types.entity_system_types import FitnessScore, GeneDict


class AdaptiveLearning:
    """Адаптивное обучение для системы."""

    def __init__(self) -> None:
        self.learning_rate: float = 0.01
        self.momentum: float = 0.9
        self.adaptation_history: List[Dict[str, Any]] = []
        self.current_parameters: GeneDict = {}
        self.performance_history: List[Dict[str, Any]] = []

    async def adapt_parameters(
        self, current_performance: FitnessScore, target_performance: FitnessScore
    ) -> GeneDict:
        """Адаптация параметров на основе производительности."""
        # Расчет ошибки
        error = target_performance - current_performance
        # Адаптация параметров
        adapted_parameters: GeneDict = {}
        for key, value in self.current_parameters.items():
            if isinstance(value, (int, float)):
                # Градиентный спуск с моментом
                gradient = error * self.learning_rate
                adapted_parameters[key] = value + gradient
        # Сохранение истории
        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "current_performance": current_performance,
            "target_performance": target_performance,
            "error": error,
            "old_parameters": self.current_parameters.copy(),
            "new_parameters": adapted_parameters,
        }
        self.adaptation_history.append(adaptation_record)
        self.current_parameters = adapted_parameters
        return adapted_parameters

    def set_parameters(self, parameters: GeneDict) -> None:
        """Установка текущих параметров."""
        self.current_parameters = parameters.copy()

    def update_performance(self, performance: FitnessScore) -> None:
        """Обновление производительности."""
        self.performance_history.append(
            {"timestamp": datetime.now().isoformat(), "performance": performance}
        )

    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Получение статистики адаптации."""
        if not self.adaptation_history:
            return {}
        errors = [record["error"] for record in self.adaptation_history]
        performances = [
            record["current_performance"] for record in self.adaptation_history
        ]
        return {
            "total_adaptations": len(self.adaptation_history),
            "average_error": np.mean(errors),
            "error_std": np.std(errors),
            "performance_trend": np.polyfit(range(len(performances)), performances, 1)[
                0
            ],
            "last_adaptation": self.adaptation_history[-1]["timestamp"],
        }
