"""
Индивид в генетическом алгоритме.
Представляет решение в виде набора генов и методов для мутации и скрещивания.
"""

import copy
import hashlib
import json
import random
from typing import Tuple

from shared.numpy_utils import np

from domain.types.entity_system_types import FitnessScore, GeneDict


class Individual:
    """Индивид в генетическом алгоритме."""

    def __init__(self, genes: GeneDict, fitness: FitnessScore = 0.0) -> None:
        """Инициализация индивида.
        Args:
            genes: Словарь генов (параметров)
            fitness: Значение приспособленности
        """
        self.genes: GeneDict = genes
        self.fitness: FitnessScore = fitness
        self.age: int = 0
        self.generation: int = 0
        self.mutation_count: int = 0
        self.crossover_count: int = 0
        self.id: str = self._generate_id()

    def _generate_id(self) -> str:
        """Генерация уникального ID на основе генов."""
        genes_str = json.dumps(self.genes, sort_keys=True)
        return hashlib.md5(genes_str.encode()).hexdigest()[:8]

    def mutate(self, mutation_rate: float = 0.1) -> "Individual":
        """Мутация индивида.
        Args:
            mutation_rate: Вероятность мутации каждого гена
        Returns:
            Новый мутированный индивид
        """
        mutated_genes = copy.deepcopy(self.genes)
        for key, value in mutated_genes.items():
            if random.random() < mutation_rate:
                if isinstance(value, (int, float)):
                    # Гауссова мутация для числовых значений
                    noise = np.random.normal(0, abs(value) * 0.1)
                    mutated_genes[key] = value + noise
                elif isinstance(value, str):
                    # Случайная замена символа для строк
                    if len(value) > 0:
                        chars = list(value)
                        idx = random.randint(0, len(chars) - 1)
                        chars[idx] = random.choice(
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                        )
                        mutated_genes[key] = "".join(chars)
                # Для других типов оставляем без изменений
        mutated_individual = Individual(mutated_genes, self.fitness)
        mutated_individual.age = self.age
        mutated_individual.generation = self.generation + 1
        mutated_individual.mutation_count = self.mutation_count + 1
        return mutated_individual

    def crossover(
        self, other: "Individual", crossover_rate: float = 0.8
    ) -> Tuple["Individual", "Individual"]:
        """Скрещивание с другим индивидом.
        Args:
            other: Другой индивид для скрещивания
            crossover_rate: Вероятность скрещивания
        Returns:
            Кортеж из двух потомков
        """
        if random.random() > crossover_rate:
            return copy.deepcopy(self), copy.deepcopy(other)
        child1_genes: GeneDict = {}
        child2_genes: GeneDict = {}
        for key in self.genes:
            if random.random() < 0.5:
                child1_genes[key] = copy.deepcopy(self.genes[key])
                child2_genes[key] = copy.deepcopy(other.genes[key])
            else:
                child1_genes[key] = copy.deepcopy(other.genes[key])
                child2_genes[key] = copy.deepcopy(self.genes[key])
        child1 = Individual(child1_genes)
        child2 = Individual(child2_genes)
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        child1.crossover_count = 1
        child2.crossover_count = 1
        return child1, child2

    def __str__(self) -> str:
        return f"Individual(id={self.id}, fitness={self.fitness:.4f}, generation={self.generation})"

    def __repr__(self) -> str:
        return self.__str__()
