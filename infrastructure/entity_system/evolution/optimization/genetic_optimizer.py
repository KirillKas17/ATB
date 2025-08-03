"""Генетический оптимизатор для параметров системы."""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from domain.types.entity_system_types import FitnessScore, GeneDict

from ..genetic.engine import EvolutionEngine


class GeneticOptimizer:
    """Генетический оптимизатор для параметров системы."""

    def __init__(self) -> None:
        self.evolution_engine = EvolutionEngine()
        self.optimization_history: List[Dict[str, Any]] = []
        self.best_solutions: Dict[str, GeneDict] = {}

        # Шаблоны параметров для оптимизации
        self.parameter_templates: Dict[str, GeneDict] = {
            "code_analysis": {
                "complexity_threshold": 10,
                "quality_threshold": 70,
                "performance_weight": 0.3,
                "maintainability_weight": 0.4,
                "readability_weight": 0.3,
            },
            "strategy_optimization": {
                "risk_tolerance": 0.5,
                "profit_target": 0.02,
                "stop_loss": 0.01,
                "position_size": 0.1,
                "max_positions": 5,
            },
            "system_configuration": {
                "analysis_interval": 3600,
                "experiment_duration": 1800,
                "confidence_threshold": 0.8,
                "improvement_threshold": 0.05,
                "max_concurrent_experiments": 3,
            },
        }

    async def optimize_parameters(
        self, parameter_type: str, fitness_function: Callable[[GeneDict], FitnessScore]
    ) -> GeneDict:
        """Оптимизация параметров определенного типа."""
        if parameter_type not in self.parameter_templates:
            raise ValueError(f"Неизвестный тип параметров: {parameter_type}")

        logger.info(f"Начало оптимизации параметров типа: {parameter_type}")

        # Настройка эволюционного движка
        self.evolution_engine.set_gene_template(
            self.parameter_templates[parameter_type]
        )
        self.evolution_engine.set_fitness_function(fitness_function)

        # Инициализация и запуск эволюции
        self.evolution_engine.initialize_population()
        best_individual = await self.evolution_engine.evolve()

        # Сохранение результата
        optimization_result = {
            "parameter_type": parameter_type,
            "timestamp": datetime.now().isoformat(),
            "best_parameters": best_individual.genes,
            "best_fitness": best_individual.fitness,
            "statistics": self.evolution_engine.get_evolution_statistics(),
        }

        self.optimization_history.append(optimization_result)
        self.best_solutions[parameter_type] = best_individual.genes

        logger.info(
            f"Оптимизация завершена. Лучшая приспособленность: {best_individual.fitness:.4f}"
        )

        return best_individual.genes

    async def optimize_code_analysis_parameters(
        self, sample_data: List[Dict[str, Any]]
    ) -> GeneDict:
        """Оптимизация параметров анализа кода."""

        def fitness_function(parameters: GeneDict) -> FitnessScore:
            total_score = 0.0

            for data_point in sample_data:
                complexity = data_point.get("complexity", 0)
                quality = data_point.get("quality", 100)

                if complexity <= parameters["complexity_threshold"]:
                    total_score += parameters["performance_weight"]

                if quality >= parameters["quality_threshold"]:
                    total_score += parameters["maintainability_weight"]

                total_score += parameters["readability_weight"] * (quality / 100)

            return total_score / len(sample_data) if sample_data else 0.0

        return await self.optimize_parameters("code_analysis", fitness_function)

    async def optimize_strategy_parameters(
        self, historical_performance: List[Dict[str, Any]]
    ) -> GeneDict:
        """Оптимизация параметров стратегии."""

        def fitness_function(parameters: GeneDict) -> FitnessScore:
            total_return = 0.0
            max_drawdown = 0.0
            win_rate = 0.0

            for trade in historical_performance:
                profit_target = float(parameters["profit_target"])
                stop_loss = float(parameters["stop_loss"])
                position_size = float(parameters["position_size"])
                if trade.get("profit", 0) > profit_target:
                    total_return += position_size * profit_target
                    win_rate += 1
                elif trade.get("loss", 0) > stop_loss:
                    total_return -= position_size * stop_loss

            win_rate = (
                win_rate / len(historical_performance)
                if historical_performance
                else 0.0
            )

            # Преобразуем все значения в float для безопасных операций
            total_return_float = float(total_return)
            win_rate_float = float(win_rate)
            max_drawdown_float = float(max_drawdown)
            
            # Явно приводим все значения к float для безопасных операций
            fitness = total_return_float * 0.4 + win_rate_float * 0.3 + (1.0 - max_drawdown_float) * 0.3
            return max(0.0, fitness)

        return await self.optimize_parameters("strategy_optimization", fitness_function)

    def get_optimization_history(
        self, parameter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Получение истории оптимизации."""
        if parameter_type:
            return [
                result
                for result in self.optimization_history
                if result["parameter_type"] == parameter_type
            ]
        return self.optimization_history

    def get_best_solution(self, parameter_type: str) -> Optional[GeneDict]:
        """Получение лучшего решения для типа параметров."""
        return self.best_solutions.get(parameter_type)
