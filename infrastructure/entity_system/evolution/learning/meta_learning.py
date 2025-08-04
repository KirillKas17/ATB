"""Мета-обучение для оптимизации процесса обучения."""

import random
from typing import Any, Dict, List, Optional

from domain.type_definitions.entity_system_types import FitnessScore


class MetaLearning:
    """Мета-обучение для оптимизации процесса обучения."""

    def __init__(self) -> None:
        self.learning_strategies: Dict[str, List[str]] = {}
        self.strategy_performance: Dict[str, Dict[str, FitnessScore]] = {}
        self.meta_parameters: Dict[str, float] = {
            "learning_rate": 0.01,
            "exploration_rate": 0.1,
            "adaptation_threshold": 0.05,
        }

    async def learn_optimal_strategy(
        self, problem_type: str, available_strategies: List[str]
    ) -> str:
        """Обучение оптимальной стратегии для типа проблемы."""
        if problem_type not in self.learning_strategies:
            self.learning_strategies[problem_type] = available_strategies
            self.strategy_performance[problem_type] = {
                s: 0.5 for s in available_strategies
            }
        # Выбор стратегии (epsilon-greedy)
        if random.random() < self.meta_parameters["exploration_rate"]:
            # Исследование - случайная стратегия
            return random.choice(available_strategies)
        else:
            # Эксплуатация - лучшая известная стратегия
            performances = self.strategy_performance[problem_type]
            return max(performances, key=lambda k: performances[k])

    def update_strategy_performance(
        self, problem_type: str, strategy: str, performance: FitnessScore
    ) -> None:
        """Обновление производительности стратегии."""
        if problem_type in self.strategy_performance:
            current_perf = self.strategy_performance[problem_type].get(strategy, 0.5)
            # Экспоненциальное скользящее среднее
            alpha = self.meta_parameters["learning_rate"]
            new_perf = alpha * performance + (1 - alpha) * current_perf
            self.strategy_performance[problem_type][strategy] = new_perf

    def get_best_strategy(self, problem_type: str) -> Optional[str]:
        """Получение лучшей стратегии для типа проблемы."""
        if problem_type in self.strategy_performance:
            performances = self.strategy_performance[problem_type]
            return max(performances, key=lambda k: performances[k])
        return None

    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Получение статистики мета-обучения."""
        stats = {
            "total_problem_types": len(self.learning_strategies),
            "strategy_performance": self.strategy_performance,
            "meta_parameters": self.meta_parameters,
        }
        # Анализ эффективности стратегий
        strategy_effectiveness: Dict[str, Dict[str, Any]] = {}
        for problem_type, performances in self.strategy_performance.items():
            best_strategy = max(performances, key=lambda k: performances[k])
            best_performance = performances[best_strategy]
            strategy_effectiveness[problem_type] = {
                "best_strategy": best_strategy,
                "best_performance": best_performance,
                "strategy_count": len(performances),
            }
        stats["strategy_effectiveness"] = strategy_effectiveness
        return stats
