"""Движок эволюционной оптимизации."""

import random
from shared.numpy_utils import np
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from domain.types.entity_system_types import EvolutionStats, FitnessScore, GeneDict

from .individual import Individual
from .population import Population


class EvolutionEngine:
    """Движок эволюционной оптимизации."""

    def __init__(self, population_size: int = 50, max_generations: int = 100) -> None:
        self.population_size: int = population_size
        self.max_generations: int = max_generations
        self.population: Optional[Population] = None
        self.fitness_function: Optional[Callable[[GeneDict], FitnessScore]] = None
        self.gene_template: GeneDict = {}
        # Параметры эволюции
        self.mutation_rate: float = 0.1
        self.crossover_rate: float = 0.8
        self.elite_size: int = 2
        self.tournament_size: int = 3
        # История эволюции
        self.evolution_history: List[Dict[str, Any]] = []
        self.best_fitness_history: List[FitnessScore] = []
        self.average_fitness_history: List[FitnessScore] = []
        # Статистика
        self.generations_completed: int = 0
        self.total_evaluations: int = 0
        self.convergence_generation: Optional[int] = None

    def set_fitness_function(
        self, fitness_func: Callable[[GeneDict], FitnessScore]
    ) -> None:
        """Установка функции приспособленности."""
        self.fitness_function = fitness_func

    def set_gene_template(self, template: GeneDict) -> None:
        """Установка шаблона генов."""
        self.gene_template = template

    def initialize_population(self) -> None:
        """Инициализация начальной популяции."""
        if not self.gene_template:
            raise ValueError("Не установлен шаблон генов")
        individuals: List[Individual] = []
        for _ in range(self.population_size):
            genes = self._generate_random_genes()
            individual = Individual(genes)
            individuals.append(individual)
        self.population = Population(individuals)
        logger.info(f"Инициализирована популяция из {self.population_size} индивидов")

    def _generate_random_genes(self) -> GeneDict:
        """Генерация случайных генов."""
        genes: GeneDict = {}
        for key, value in self.gene_template.items():
            if isinstance(value, bool):
                genes[key] = random.choice([True, False])
            elif isinstance(value, int):
                genes[key] = random.randint(0, value * 2)
            elif isinstance(value, float):
                genes[key] = random.uniform(0, value * 2)
            elif isinstance(value, str):
                length = random.randint(3, 10)
                genes[key] = "".join(
                    random.choices("abcdefghijklmnopqrstuvwxyz", k=length)
                )
            elif isinstance(value, list):
                genes[key] = random.choice(value)
            else:
                genes[key] = value
        return genes

    async def evolve(self) -> Individual:
        """Запуск эволюционного процесса."""
        if not self.population or not self.fitness_function:
            raise ValueError(
                "Популяция или функция приспособленности не инициализированы"
            )
        logger.info("Начало эволюционного процесса")
        # Оценка начальной популяции
        await self._evaluate_population()
        
        # Проверяем, что у нас есть лучший индивид
        if not self.population or not self.population.best_individual:
            raise ValueError("Не удалось оценить начальную популяцию")
            
        best_fitness = self.population.best_individual.fitness
        stagnation_count = 0
        for generation in range(self.max_generations):
            # Эволюция популяции
            self.population = self.population.evolve(
                self.mutation_rate,
                self.crossover_rate,
                self.elite_size,
                self.tournament_size,
            )
            # Оценка новой популяции
            await self._evaluate_population()
            
            # Проверяем, что у нас есть лучший индивид после оценки
            if not self.population or not self.population.best_individual:
                logger.error("Популяция пуста после оценки")
                break
                
            # Обновление статистики
            self.generations_completed = generation + 1
            self.best_fitness_history.append(self.population.best_individual.fitness)
            self.average_fitness_history.append(self.population.average_fitness)
            # Проверка сходимости
            if self.population.best_individual.fitness > best_fitness:
                best_fitness = self.population.best_individual.fitness
                stagnation_count = 0
            else:
                stagnation_count += 1
            # Остановка при сходимости
            if stagnation_count >= 20:
                self.convergence_generation = generation
                logger.info(f"Эволюция сошлась на поколении {generation}")
                break
            # Логирование прогресса
            if generation % 10 == 0:
                logger.info(
                    f"Поколение {generation}: лучшая приспособленность = {self.population.best_individual.fitness:.4f}"
                )
        
        # Проверяем финальный результат
        if not self.population or not self.population.best_individual:
            raise ValueError("Эволюция завершилась без результата")
            
        logger.info(
            f"Эволюция завершена. Лучшая приспособленность: {self.population.best_individual.fitness:.4f}"
        )
        return self.population.best_individual

    async def _evaluate_population(self) -> None:
        """Оценка приспособленности популяции."""
        if not self.fitness_function or not self.population:
            return
        for individual in self.population.individuals:
            try:
                if individual is not None:
                    individual.fitness = self.fitness_function(individual.genes)
                    self.total_evaluations += 1
            except Exception as e:
                logger.error(f"Ошибка оценки индивида {individual.id if individual else 'Unknown'}: {e}")
                if individual is not None:
                    individual.fitness = 0.0
        self.population._update_statistics()

    def get_evolution_statistics(self) -> EvolutionStats:
        """Получение статистики эволюции."""
        # Получаем лучшую приспособленность
        best_fitness = 0.0
        if self.population and self.population.best_individual:
            best_fitness = self.population.best_individual.fitness
            
        # Получаем среднюю приспособленность
        average_fitness = 0.0
        if self.population:
            average_fitness = self.population.average_fitness
            
        # Вычисляем диверсиность (стандартное отклонение приспособленности)
        diversity = 0.0
        if self.population:
            diversity = self.population.fitness_std
            
        stats: EvolutionStats = {
            "generation": self.generations_completed,
            "best_fitness": best_fitness,
            "average_fitness": average_fitness,
            "diversity": diversity,
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
            "population_size": self.population_size,
            "timestamp": datetime.now(),
        }
        return stats
