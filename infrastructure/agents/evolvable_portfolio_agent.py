"""
Эволюционный агент управления портфелем.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class EvolvablePortfolioConfig:
    """Конфигурация эволюционного портфельного агента."""

    evolution_rate: float = 0.1
    mutation_rate: float = 0.05
    population_size: int = 100
    generations: int = 50


class EvolvablePortfolioAgent(ABC):
    """Абстрактный эволюционный портфельный агент."""

    def __init__(self, config: Optional[EvolvablePortfolioConfig] = None) -> None:
        self.config = config or EvolvablePortfolioConfig()
        self.is_active: bool = False
        self.generation: int = 0
        self.fitness_history: List[float] = []

    @abstractmethod
    async def evolve_portfolio_strategy(
        self, portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Эволюция стратегии портфеля."""

    @abstractmethod
    async def calculate_fitness(self, strategy: Dict[str, Any]) -> float:
        """Расчет приспособленности стратегии."""

    @abstractmethod
    async def mutate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Мутация стратегии."""

    @abstractmethod
    async def crossover_strategies(
        self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Скрещивание стратегий."""


class DefaultEvolvablePortfolioAgent(EvolvablePortfolioAgent):
    """Реализация эволюционного портфельного агента по умолчанию."""

    async def evolve_portfolio_strategy(
        self, portfolio_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Эволюция стратегии портфеля."""
        try:
            # Простая реализация эволюционного алгоритма
            population = self._initialize_population()

            for generation in range(self.config.generations):
                # Оценка приспособленности
                fitness_scores = []
                for strategy in population:
                    fitness = await self.calculate_fitness(strategy)
                    fitness_scores.append(fitness)

                # Отбор лучших
                best_strategies = self._select_best(population, fitness_scores)

                # Создание нового поколения
                new_population = []
                for _ in range(self.config.population_size):
                    if len(best_strategies) >= 2:
                        parent1, parent2 = self._select_parents(best_strategies)
                        child = await self.crossover_strategies(parent1, parent2)
                        child = await self.mutate_strategy(child)
                    else:
                        child = best_strategies[0] if best_strategies else {}
                    new_population.append(child)

                population = new_population
                self.generation = generation

            # Возврат лучшей стратегии
            best_fitness = -float("inf")
            best_strategy = {}
            for strategy in population:
                fitness = await self.calculate_fitness(strategy)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = strategy
            return best_strategy

        except Exception as e:
            return {"error": str(e)}

    async def calculate_fitness(self, strategy: Dict[str, Any]) -> float:
        """Расчет приспособленности стратегии."""
        # Простая функция приспособленности
        sharpe_ratio = float(strategy.get("sharpe_ratio", 0.0))
        max_drawdown = float(strategy.get("max_drawdown", 1.0))
        return sharpe_ratio * (1 - max_drawdown)

    async def mutate_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Мутация стратегии."""
        import random

        mutated = strategy.copy()
        for key in mutated:
            if (
                isinstance(mutated[key], (int, float))
                and random.random() < self.config.mutation_rate
            ):
                # Простая мутация - добавление случайного значения
                mutated[key] += random.uniform(-0.1, 0.1)

        return mutated

    async def crossover_strategies(
        self, strategy1: Dict[str, Any], strategy2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Скрещивание стратегий."""
        import random

        child = {}
        for key in strategy1:
            if random.random() < 0.5:
                child[key] = strategy1[key]
            else:
                child[key] = strategy2.get(key, strategy1[key])

        return child

    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Инициализация популяции."""
        population = []
        for _ in range(self.config.population_size):
            strategy = {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "allocation": {},
                "risk_tolerance": 0.5,
            }
            population.append(strategy)
        return population

    def _select_best(
        self, population: List[Dict[str, Any]], fitness_scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Отбор лучших стратегий."""
        # Отбираем топ 20% стратегий
        sorted_indices = sorted(
            range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True
        )
        top_count = max(1, len(population) // 5)
        return [population[i] for i in sorted_indices[:top_count]]

    def _select_parents(
        self, strategies: List[Dict[str, Any]]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Выбор родителей для скрещивания."""
        import random

        selected = random.sample(strategies, 2)
        return selected[0], selected[1]
