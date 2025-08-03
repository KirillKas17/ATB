"""Популяция индивидов в генетическом алгоритме."""

import random
from typing import List, Optional

import numpy as np

from domain.types.entity_system_types import FitnessScore

from .individual import Individual


class Population:
    """Популяция индивидов."""

    def __init__(self, individuals: Optional[List[Individual]] = None) -> None:
        self.individuals: List[Individual] = individuals or []
        self.generation: int = 0
        self.best_individual: Optional[Individual] = None
        self.worst_individual: Optional[Individual] = None
        self.average_fitness: FitnessScore = 0.0
        self.fitness_std: float = 0.0

    def add_individual(self, individual: Individual) -> None:
        """Добавление индивида в популяцию."""
        self.individuals.append(individual)
        self._update_statistics()

    def remove_individual(self, individual: Individual) -> None:
        """Удаление индивида из популяции."""
        if individual in self.individuals:
            self.individuals.remove(individual)
            self._update_statistics()

    def get_best_individuals(self, count: int = 1) -> List[Individual]:
        """Получение лучших индивидов."""
        sorted_individuals = sorted(
            self.individuals, key=lambda x: x.fitness, reverse=True
        )
        return sorted_individuals[:count]

    def get_worst_individuals(self, count: int = 1) -> List[Individual]:
        """Получение худших индивидов."""
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness)
        return sorted_individuals[:count]

    def select_tournament(self, tournament_size: int = 3) -> Individual:
        """Турнирная селекция."""
        tournament = random.sample(
            self.individuals, min(tournament_size, len(self.individuals))
        )
        return max(tournament, key=lambda x: x.fitness)

    def select_roulette(self) -> Individual:
        """Рулеточная селекция."""
        total_fitness = sum(ind.fitness for ind in self.individuals)
        if total_fitness <= 0:
            return random.choice(self.individuals)
        r = random.uniform(0, total_fitness)
        current_sum = 0.0
        for individual in self.individuals:
            current_sum += individual.fitness
            if current_sum >= r:
                return individual
        return self.individuals[-1]

    def evolve(
        self,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elite_size: int = 2,
        tournament_size: int = 3,
    ) -> "Population":
        """Эволюция популяции."""
        new_population = Population()
        new_population.generation = self.generation + 1
        # Элитизм - сохранение лучших индивидов
        elite = self.get_best_individuals(elite_size)
        for individual in elite:
            new_population.add_individual(individual)
        # Генерация новых индивидов
        while len(new_population.individuals) < len(self.individuals):
            # Селекция родителей
            parent1 = self.select_tournament(tournament_size)
            parent2 = self.select_tournament(tournament_size)
            # Скрещивание
            child1, child2 = parent1.crossover(parent2, crossover_rate)
            # Мутация
            if random.random() < mutation_rate:
                child1 = child1.mutate(mutation_rate)
            if random.random() < mutation_rate:
                child2 = child2.mutate(mutation_rate)
            new_population.add_individual(child1)
            if len(new_population.individuals) < len(self.individuals):
                new_population.add_individual(child2)
        return new_population

    def _update_statistics(self) -> None:
        """Обновление статистики популяции."""
        if not self.individuals:
            return
        fitnesses = [ind.fitness for ind in self.individuals]
        self.average_fitness = float(np.mean(fitnesses))
        self.fitness_std = float(np.std(fitnesses))
        self.best_individual = max(self.individuals, key=lambda x: x.fitness)
        self.worst_individual = min(self.individuals, key=lambda x: x.fitness)

    def __len__(self) -> int:
        return len(self.individuals)
