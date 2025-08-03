"""
Сверхпродвинутый модуль нейроэволюции для торговых стратегий
Включает NEAT, Evolution Strategies, Genetic Programming, Quantum Evolution
"""

import asyncio
import math
import pickle
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

# Попытка импорта дополнительных библиотек
try:
    import deap
    from deap import base, creator, tools, algorithms, gp
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP not available, using simplified evolution")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


@dataclass
class NeuroEvolutionConfig:
    """Конфигурация нейроэволюции"""
    
    # Популяция
    population_size: int = 150
    elite_size: int = 15
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    
    # Нейросети
    max_nodes: int = 100
    max_connections: int = 200
    activation_functions: List[str] = field(default_factory=lambda: [
        'relu', 'tanh', 'sigmoid', 'leaky_relu', 'swish', 'gelu'
    ])
    
    # Эволюция
    generations: int = 100
    fitness_threshold: float = 0.95
    speciation_threshold: float = 3.0
    compatibility_disjoint_coeff: float = 1.0
    compatibility_weight_coeff: float = 0.5
    
    # Quantum Evolution
    quantum_enabled: bool = True
    quantum_noise: float = 0.1
    quantum_entanglement: float = 0.05
    
    # Парето-оптимизация
    pareto_objectives: List[str] = field(default_factory=lambda: [
        'profit', 'sharpe_ratio', 'max_drawdown', 'win_rate'
    ])
    
    # Adaptive parameters
    adaptive_mutation: bool = True
    adaptive_crossover: bool = True
    diversity_pressure: float = 0.1


class ActivationFunction:
    """Коллекция функций активации"""
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)
    
    @staticmethod
    def tanh(x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)
    
    @staticmethod
    def sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)
    
    @staticmethod
    def leaky_relu(x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, 0.1)
    
    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    @staticmethod
    def gelu(x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)
    
    @staticmethod
    def get_function(name: str) -> Callable:
        return getattr(ActivationFunction, name, ActivationFunction.relu)


class NeuralNode:
    """Узел нейронной сети"""
    
    def __init__(self, node_id: int, node_type: str = "hidden", 
                 activation: str = "relu", bias: float = 0.0):
        self.id = node_id
        self.type = node_type  # input, hidden, output
        self.activation = activation
        self.bias = bias
        self.value = 0.0
        self.layer = 0  # For topological sorting
        
    def activate(self, x: torch.Tensor) -> torch.Tensor:
        """Применить функцию активации"""
        activation_fn = ActivationFunction.get_function(self.activation)
        return activation_fn(x + self.bias)


class Connection:
    """Соединение между узлами"""
    
    def __init__(self, from_node: int, to_node: int, weight: float = 0.0, enabled: bool = True):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = 0


class NEATGenome:
    """Геном NEAT (NeuroEvolution of Augmenting Topologies)"""
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes: Dict[int, NeuralNode] = {}
        self.connections: Dict[Tuple[int, int], Connection] = {}
        self.fitness = 0.0
        self.adjusted_fitness = 0.0
        self.species_id = 0
        
        # Создаем базовые узлы
        self._create_basic_structure()
        
    def _create_basic_structure(self):
        """Создает базовую структуру сети"""
        node_id = 0
        
        # Input nodes
        for i in range(self.input_size):
            self.nodes[node_id] = NeuralNode(node_id, "input")
            node_id += 1
            
        # Output nodes
        for i in range(self.output_size):
            self.nodes[node_id] = NeuralNode(node_id, "output", "tanh")
            node_id += 1
            
        # Базовые соединения от всех входов ко всем выходам
        for input_id in range(self.input_size):
            for output_id in range(self.input_size, self.input_size + self.output_size):
                weight = random.uniform(-1, 1)
                self.connections[(input_id, output_id)] = Connection(input_id, output_id, weight)
    
    def add_node(self, innovation_tracker: Dict[Tuple[int, int], int]):
        """Добавляет новый узел (мутация)"""
        if not self.connections:
            return
            
        # Выбираем случайное соединение
        connection_key = random.choice(list(self.connections.keys()))
        connection = self.connections[connection_key]
        
        if not connection.enabled:
            return
            
        # Отключаем старое соединение
        connection.enabled = False
        
        # Создаем новый узел
        new_node_id = max(self.nodes.keys()) + 1
        activation = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
        self.nodes[new_node_id] = NeuralNode(new_node_id, "hidden", activation)
        
        # Создаем два новых соединения
        self.connections[(connection.from_node, new_node_id)] = Connection(
            connection.from_node, new_node_id, 1.0
        )
        self.connections[(new_node_id, connection.to_node)] = Connection(
            new_node_id, connection.to_node, connection.weight
        )
    
    def add_connection(self):
        """Добавляет новое соединение (мутация)"""
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts:
            from_node = random.choice(list(self.nodes.keys()))
            to_node = random.choice(list(self.nodes.keys()))
            
            # Проверяем валидность соединения
            if (from_node, to_node) not in self.connections and from_node != to_node:
                # Проверяем, что не создаем обратное соединение
                if not self._creates_cycle(from_node, to_node):
                    weight = random.uniform(-1, 1)
                    self.connections[(from_node, to_node)] = Connection(from_node, to_node, weight)
                    break
                    
            attempts += 1
    
    def _creates_cycle(self, from_node: int, to_node: int) -> bool:
        """Проверяет, создает ли новое соединение цикл"""
        if not NETWORKX_AVAILABLE:
            return False  # Упрощенная проверка
            
        graph = nx.DiGraph()
        
        # Добавляем существующие соединения
        for (f, t), conn in self.connections.items():
            if conn.enabled:
                graph.add_edge(f, t)
                
        # Добавляем новое соединение
        graph.add_edge(from_node, to_node)
        
        return not nx.is_directed_acyclic_graph(graph)
    
    def mutate_weights(self, mutation_rate: float):
        """Мутирует веса соединений"""
        for connection in self.connections.values():
            if random.random() < mutation_rate:
                if random.random() < 0.1:  # Полная замена веса
                    connection.weight = random.uniform(-1, 1)
                else:  # Небольшое изменение
                    connection.weight += random.gauss(0, 0.1)
                    connection.weight = max(-5, min(5, connection.weight))  # Clamp
    
    def mutate_bias(self, mutation_rate: float):
        """Мутирует смещения узлов"""
        for node in self.nodes.values():
            if node.type != "input" and random.random() < mutation_rate:
                if random.random() < 0.1:
                    node.bias = random.uniform(-1, 1)
                else:
                    node.bias += random.gauss(0, 0.1)
                    node.bias = max(-2, min(2, node.bias))
    
    def mutate_activation(self, mutation_rate: float, available_activations: List[str]):
        """Мутирует функции активации"""
        for node in self.nodes.values():
            if node.type == "hidden" and random.random() < mutation_rate:
                node.activation = random.choice(available_activations)
    
    def crossover(self, other: 'NEATGenome') -> 'NEATGenome':
        """Скрещивание двух геномов"""
        child = NEATGenome(self.input_size, self.output_size)
        child.nodes.clear()
        child.connections.clear()
        
        # Берем узлы от более приспособленного родителя
        if self.fitness >= other.fitness:
            primary_parent = self
            secondary_parent = other
        else:
            primary_parent = other
            secondary_parent = self
            
        # Копируем все узлы от первичного родителя
        for node_id, node in primary_parent.nodes.items():
            child.nodes[node_id] = NeuralNode(node_id, node.type, node.activation, node.bias)
            
        # Скрещиваем соединения
        all_connection_keys = set(self.connections.keys()) | set(other.connections.keys())
        
        for key in all_connection_keys:
            if key in self.connections and key in other.connections:
                # Общее соединение - выбираем случайно
                parent_conn = random.choice([self.connections[key], other.connections[key]])
                child.connections[key] = Connection(
                    parent_conn.from_node, parent_conn.to_node, 
                    parent_conn.weight, parent_conn.enabled
                )
            elif key in primary_parent.connections:
                # Соединение только у более приспособленного родителя
                parent_conn = primary_parent.connections[key]
                child.connections[key] = Connection(
                    parent_conn.from_node, parent_conn.to_node,
                    parent_conn.weight, parent_conn.enabled
                )
                
        return child
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Прямое распространение через сеть"""
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)
            
        batch_size = inputs.size(0)
        
        # Топологическая сортировка узлов
        sorted_nodes = self._topological_sort()
        
        # Инициализация значений узлов
        node_values = {}
        
        # Устанавливаем входные значения
        for i, node_id in enumerate(sorted_nodes):
            if self.nodes[node_id].type == "input":
                if i < inputs.size(1):
                    node_values[node_id] = inputs[:, i]
                else:
                    node_values[node_id] = torch.zeros(batch_size)
            else:
                node_values[node_id] = torch.zeros(batch_size)
        
        # Распространяем сигнал через сеть
        for node_id in sorted_nodes:
            if self.nodes[node_id].type == "input":
                continue
                
            # Суммируем входные сигналы
            input_sum = torch.zeros(batch_size)
            for (from_id, to_id), connection in self.connections.items():
                if to_id == node_id and connection.enabled and from_id in node_values:
                    input_sum += node_values[from_id] * connection.weight
            
            # Применяем активацию
            node_values[node_id] = self.nodes[node_id].activate(input_sum)
        
        # Собираем выходные значения
        outputs = []
        for node_id in sorted_nodes:
            if self.nodes[node_id].type == "output":
                outputs.append(node_values[node_id])
                
        if outputs:
            return torch.stack(outputs, dim=1)
        else:
            return torch.zeros(batch_size, self.output_size)
    
    def _topological_sort(self) -> List[int]:
        """Топологическая сортировка узлов"""
        if not NETWORKX_AVAILABLE:
            # Упрощенная сортировка
            return sorted(self.nodes.keys())
            
        graph = nx.DiGraph()
        
        # Добавляем узлы
        for node_id in self.nodes.keys():
            graph.add_node(node_id)
            
        # Добавляем соединения
        for (from_id, to_id), connection in self.connections.items():
            if connection.enabled:
                graph.add_edge(from_id, to_id)
        
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXError:
            # Если есть циклы, возвращаем простую сортировку
            return sorted(self.nodes.keys())
    
    def compatibility_distance(self, other: 'NEATGenome', 
                             c1: float = 1.0, c2: float = 1.0, c3: float = 0.5) -> float:
        """Вычисляет расстояние совместимости между геномами"""
        
        # Получаем все innovation numbers для соединений
        innovations1 = set(self.connections.keys())
        innovations2 = set(other.connections.keys())
        
        # Disjoint и excess соединения
        matching = innovations1 & innovations2
        disjoint = (innovations1 | innovations2) - matching
        
        # Разница в весах совпадающих соединений
        weight_diff = 0.0
        if matching:
            for key in matching:
                weight_diff += abs(self.connections[key].weight - other.connections[key].weight)
            weight_diff /= len(matching)
        
        # Нормализация по размеру большего генома
        N = max(len(self.connections), len(other.connections), 1)
        
        distance = (c1 * len(disjoint)) / N + c3 * weight_diff
        
        return distance


class QuantumInspiredEvolution:
    """Квантово-вдохновленная эволюция"""
    
    def __init__(self, config: NeuroEvolutionConfig):
        self.config = config
        self.quantum_bits = {}  # Квантовые биты для каждого параметра
        
    def initialize_quantum_population(self, population: List[NEATGenome]):
        """Инициализация квантовой популяции"""
        for i, genome in enumerate(population):
            self.quantum_bits[i] = self._create_quantum_state(genome)
    
    def _create_quantum_state(self, genome: NEATGenome) -> Dict[str, complex]:
        """Создает квантовое состояние для генома"""
        quantum_state = {}
        
        # Квантовые биты для весов соединений
        for key, connection in genome.connections.items():
            phase = random.uniform(0, 2 * math.pi)
            amplitude = abs(connection.weight)
            quantum_state[f"weight_{key}"] = amplitude * (math.cos(phase) + 1j * math.sin(phase))
            
        return quantum_state
    
    def quantum_mutation(self, genome: NEATGenome, genome_id: int) -> NEATGenome:
        """Квантовая мутация"""
        if genome_id not in self.quantum_bits:
            return genome
            
        mutated_genome = NEATGenome(genome.input_size, genome.output_size)
        mutated_genome.nodes = genome.nodes.copy()
        mutated_genome.connections = genome.connections.copy()
        
        quantum_state = self.quantum_bits[genome_id]
        
        # Применяем квантовые флуктуации к весам
        for key, connection in mutated_genome.connections.items():
            quantum_key = f"weight_{key}"
            if quantum_key in quantum_state:
                # Квантовое вмешательство
                quantum_bit = quantum_state[quantum_key]
                phase_shift = random.uniform(-self.config.quantum_noise, self.config.quantum_noise)
                
                # Новая фаза
                new_phase = math.atan2(quantum_bit.imag, quantum_bit.real) + phase_shift
                amplitude = abs(quantum_bit)
                
                # Обновляем вес на основе квантового состояния
                quantum_weight = amplitude * math.cos(new_phase)
                connection.weight += quantum_weight * self.config.quantum_noise
                
                # Обновляем квантовое состояние
                quantum_state[quantum_key] = amplitude * (math.cos(new_phase) + 1j * math.sin(new_phase))
        
        return mutated_genome
    
    def quantum_entanglement(self, genome1: NEATGenome, genome2: NEATGenome, 
                           id1: int, id2: int) -> Tuple[NEATGenome, NEATGenome]:
        """Квантовая запутанность между геномами"""
        if id1 not in self.quantum_bits or id2 not in self.quantum_bits:
            return genome1, genome2
            
        # Создаем запутанные состояния
        entangled1 = NEATGenome(genome1.input_size, genome1.output_size)
        entangled2 = NEATGenome(genome2.input_size, genome2.output_size)
        
        entangled1.nodes = genome1.nodes.copy()
        entangled1.connections = genome1.connections.copy()
        entangled2.nodes = genome2.nodes.copy()
        entangled2.connections = genome2.connections.copy()
        
        # Применяем запутанность к общим соединениям
        common_connections = set(genome1.connections.keys()) & set(genome2.connections.keys())
        
        for key in common_connections:
            if random.random() < self.config.quantum_entanglement:
                # Создаем запутанную пару
                weight1 = entangled1.connections[key].weight
                weight2 = entangled2.connections[key].weight
                
                # Bell state - максимально запутанное состояние
                entangled_weight = (weight1 + weight2) / 2
                entanglement_factor = random.uniform(-0.1, 0.1)
                
                entangled1.connections[key].weight = entangled_weight + entanglement_factor
                entangled2.connections[key].weight = entangled_weight - entanglement_factor
        
        return entangled1, entangled2


class NSGA2Selection:
    """Non-dominated Sorting Genetic Algorithm II для многокритериальной оптимизации"""
    
    def __init__(self, objectives: List[str]):
        self.objectives = objectives
        
    def fast_non_dominated_sort(self, population: List[NEATGenome]) -> List[List[int]]:
        """Быстрая сортировка по недоминированности"""
        fronts = [[]]
        
        dominated = [[] for _ in range(len(population))]
        domination_count = [0] * len(population)
        
        # Для каждой особи
        for i in range(len(population)):
            for j in range(len(population)):
                if i != j:
                    if self._dominates(population[i], population[j]):
                        dominated[i].append(j)
                    elif self._dominates(population[j], population[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        # Строим остальные фронты
        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            if next_front:
                fronts.append(next_front)
            current_front += 1
        
        return fronts[:-1] if fronts[-1] == [] else fronts
    
    def _dominates(self, genome1: NEATGenome, genome2: NEATGenome) -> bool:
        """Проверяет, доминирует ли genome1 над genome2"""
        objectives1 = self._get_objectives(genome1)
        objectives2 = self._get_objectives(genome2)
        
        better_in_at_least_one = False
        for obj1, obj2 in zip(objectives1, objectives2):
            if obj1 < obj2:  # Предполагаем минимизацию
                return False
            elif obj1 > obj2:
                better_in_at_least_one = True
                
        return better_in_at_least_one
    
    def _get_objectives(self, genome: NEATGenome) -> List[float]:
        """Получает значения целевых функций"""
        # Здесь должны быть реальные метрики торговой стратегии
        return [
            getattr(genome, 'profit', 0.0),
            getattr(genome, 'sharpe_ratio', 0.0),
            -getattr(genome, 'max_drawdown', 0.0),  # Минимизируем просадку
            getattr(genome, 'win_rate', 0.0)
        ]
    
    def crowding_distance(self, front: List[int], population: List[NEATGenome]) -> List[float]:
        """Вычисляет crowding distance для фронта"""
        distances = [0.0] * len(front)
        
        for obj_idx in range(len(self.objectives)):
            # Сортируем по текущей цели
            front_sorted = sorted(front, key=lambda i: self._get_objectives(population[i])[obj_idx])
            
            # Крайние точки получают бесконечное расстояние
            distances[front.index(front_sorted[0])] = float('inf')
            distances[front.index(front_sorted[-1])] = float('inf')
            
            # Вычисляем расстояния для остальных
            obj_range = (self._get_objectives(population[front_sorted[-1]])[obj_idx] - 
                        self._get_objectives(population[front_sorted[0]])[obj_idx])
            
            if obj_range > 0:
                for i in range(1, len(front_sorted) - 1):
                    distance = (self._get_objectives(population[front_sorted[i+1]])[obj_idx] - 
                               self._get_objectives(population[front_sorted[i-1]])[obj_idx]) / obj_range
                    distances[front.index(front_sorted[i])] += distance
        
        return distances


class AdaptiveParameterController:
    """Адаптивное управление параметрами эволюции"""
    
    def __init__(self, config: NeuroEvolutionConfig):
        self.config = config
        self.generation = 0
        self.fitness_history = []
        self.diversity_history = []
        
    def update_parameters(self, population: List[NEATGenome]) -> NeuroEvolutionConfig:
        """Обновляет параметры на основе состояния популяции"""
        self.generation += 1
        
        # Анализируем текущую популяцию
        avg_fitness = np.mean([genome.fitness for genome in population])
        diversity = self._calculate_diversity(population)
        
        self.fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        # Адаптивная мутация
        if self.config.adaptive_mutation:
            self.config.mutation_rate = self._adapt_mutation_rate(diversity)
            
        # Адаптивное скрещивание
        if self.config.adaptive_crossover:
            self.config.crossover_rate = self._adapt_crossover_rate(avg_fitness)
            
        return self.config
    
    def _calculate_diversity(self, population: List[NEATGenome]) -> float:
        """Вычисляет разнообразие популяции"""
        if len(population) < 2:
            return 0.0
            
        total_distance = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = population[i].compatibility_distance(population[j])
                total_distance += distance
                comparisons += 1
                
        return total_distance / comparisons if comparisons > 0 else 0.0
    
    def _adapt_mutation_rate(self, diversity: float) -> float:
        """Адаптирует коэффициент мутации"""
        # Если разнообразие низкое, увеличиваем мутацию
        if diversity < 1.0:
            return min(0.3, self.config.mutation_rate * 1.1)
        elif diversity > 3.0:
            return max(0.05, self.config.mutation_rate * 0.9)
        else:
            return self.config.mutation_rate
    
    def _adapt_crossover_rate(self, avg_fitness: float) -> float:
        """Адаптирует коэффициент скрещивания"""
        # Если фитнес не растет, увеличиваем скрещивание
        if len(self.fitness_history) > 5:
            recent_improvement = (self.fitness_history[-1] - self.fitness_history[-6]) / 5
            if recent_improvement < 0.01:
                return min(0.9, self.config.crossover_rate * 1.05)
            else:
                return max(0.5, self.config.crossover_rate * 0.95)
        
        return self.config.crossover_rate


class Species:
    """Вид в NEAT алгоритме"""
    
    def __init__(self, representative: NEATGenome, species_id: int):
        self.id = species_id
        self.representative = representative
        self.members: List[NEATGenome] = [representative]
        self.avg_fitness = 0.0
        self.max_fitness = 0.0
        self.generations_without_improvement = 0
        
    def add_member(self, genome: NEATGenome):
        """Добавляет члена в вид"""
        self.members.append(genome)
        genome.species_id = self.id
        
    def update_fitness(self):
        """Обновляет фитнес вида"""
        if not self.members:
            return
            
        fitnesses = [member.fitness for member in self.members]
        old_max = self.max_fitness
        
        self.avg_fitness = np.mean(fitnesses)
        self.max_fitness = max(fitnesses)
        
        if self.max_fitness <= old_max:
            self.generations_without_improvement += 1
        else:
            self.generations_without_improvement = 0
            
    def adjust_fitness(self):
        """Корректирует фитнес членов вида (fitness sharing)"""
        for member in self.members:
            member.adjusted_fitness = member.fitness / len(self.members)


class NEATEvolution:
    """Основной класс NEAT эволюции"""
    
    def __init__(self, config: NeuroEvolutionConfig, input_size: int, output_size: int):
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        
        # Компоненты эволюции
        self.quantum_evolution = QuantumInspiredEvolution(config) if config.quantum_enabled else None
        self.nsga2 = NSGA2Selection(config.pareto_objectives)
        self.adaptive_controller = AdaptiveParameterController(config)
        
        # Состояние эволюции
        self.population: List[NEATGenome] = []
        self.species: List[Species] = []
        self.generation = 0
        self.innovation_counter = 0
        self.global_innovations: Dict[Tuple[int, int], int] = {}
        
        # История эволюции
        self.fitness_history = []
        self.complexity_history = []
        
    def initialize_population(self) -> List[NEATGenome]:
        """Инициализирует начальную популяцию"""
        self.population = []
        
        for _ in range(self.config.population_size):
            genome = NEATGenome(self.input_size, self.output_size)
            
            # Добавляем случайные скрытые узлы
            num_hidden = random.randint(0, 5)
            for _ in range(num_hidden):
                genome.add_node(self.global_innovations)
                
            # Случайные дополнительные соединения
            num_connections = random.randint(0, 3)
            for _ in range(num_connections):
                genome.add_connection()
                
            # Мутируем веса
            genome.mutate_weights(0.8)
            
            self.population.append(genome)
            
        # Инициализируем квантовую популяцию
        if self.quantum_evolution:
            self.quantum_evolution.initialize_quantum_population(self.population)
            
        return self.population
    
    async def evolve_generation(self, fitness_function: Callable[[NEATGenome], float]) -> List[NEATGenome]:
        """Эволюционирует одно поколение"""
        
        # Оценка фитнеса
        await self._evaluate_population(fitness_function)
        
        # Видообразование
        self._speciate_population()
        
        # Корректировка фитнеса
        self._adjust_fitness()
        
        # Адаптивные параметры
        self.config = self.adaptive_controller.update_parameters(self.population)
        
        # Создание нового поколения
        new_population = await self._create_new_generation()
        
        # Обновление статистики
        self._update_statistics()
        
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    async def _evaluate_population(self, fitness_function: Callable[[NEATGenome], float]):
        """Оценивает фитнес популяции"""
        tasks = []
        
        for genome in self.population:
            task = asyncio.create_task(self._evaluate_genome(genome, fitness_function))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
    
    async def _evaluate_genome(self, genome: NEATGenome, fitness_function: Callable[[NEATGenome], float]):
        """Оценивает отдельный геном"""
        try:
            fitness = fitness_function(genome)
            genome.fitness = fitness
        except Exception as e:
            logger.error(f"Error evaluating genome: {e}")
            genome.fitness = 0.0
    
    def _speciate_population(self):
        """Разделение популяции на виды"""
        # Очищаем старые виды
        for species in self.species:
            species.members.clear()
            
        unassigned = self.population.copy()
        
        # Назначаем геномы существующим видам
        for genome in self.population:
            assigned = False
            
            for species in self.species:
                distance = genome.compatibility_distance(
                    species.representative, 
                    self.config.compatibility_disjoint_coeff,
                    self.config.compatibility_disjoint_coeff,
                    self.config.compatibility_weight_coeff
                )
                
                if distance < self.config.speciation_threshold:
                    species.add_member(genome)
                    unassigned.remove(genome)
                    assigned = True
                    break
                    
            if not assigned and genome in unassigned:
                # Создаем новый вид
                new_species = Species(genome, len(self.species))
                self.species.append(new_species)
                unassigned.remove(genome)
                
        # Удаляем пустые виды
        self.species = [s for s in self.species if s.members]
        
        # Обновляем фитнес видов
        for species in self.species:
            species.update_fitness()
    
    def _adjust_fitness(self):
        """Корректирует фитнес с учетом видов"""
        for species in self.species:
            species.adjust_fitness()
    
    async def _create_new_generation(self) -> List[NEATGenome]:
        """Создает новое поколение"""
        new_population = []
        
        # Элитизм - сохраняем лучших
        elite = self._select_elite()
        new_population.extend(elite)
        
        # Определяем количество потомков для каждого вида
        offspring_counts = self._calculate_offspring_counts()
        
        # Создаем потомков для каждого вида
        for species, count in zip(self.species, offspring_counts):
            if count > 0:
                offspring = await self._create_species_offspring(species, count)
                new_population.extend(offspring)
                
        # Дополняем до нужного размера популяции
        while len(new_population) < self.config.population_size:
            parent = random.choice(self.population)
            child = self._mutate_genome(parent)
            new_population.append(child)
            
        return new_population[:self.config.population_size]
    
    def _select_elite(self) -> List[NEATGenome]:
        """Выбирает элиту популяции"""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_population[:self.config.elite_size]
    
    def _calculate_offspring_counts(self) -> List[int]:
        """Вычисляет количество потомков для каждого вида"""
        total_adjusted_fitness = sum(
            sum(member.adjusted_fitness for member in species.members)
            for species in self.species
        )
        
        if total_adjusted_fitness == 0:
            # Равномерное распределение
            base_count = self.config.population_size // len(self.species)
            return [base_count] * len(self.species)
            
        offspring_counts = []
        remaining_population = self.config.population_size - self.config.elite_size
        
        for species in self.species:
            species_fitness = sum(member.adjusted_fitness for member in species.members)
            proportion = species_fitness / total_adjusted_fitness
            count = int(remaining_population * proportion)
            offspring_counts.append(count)
            
        return offspring_counts
    
    async def _create_species_offspring(self, species: Species, count: int) -> List[NEATGenome]:
        """Создает потомков для вида"""
        offspring = []
        
        for _ in range(count):
            if random.random() < self.config.crossover_rate and len(species.members) > 1:
                # Скрещивание
                parent1 = self._tournament_selection(species.members)
                parent2 = self._tournament_selection(species.members)
                child = parent1.crossover(parent2)
            else:
                # Мутация
                parent = self._tournament_selection(species.members)
                child = NEATGenome(parent.input_size, parent.output_size)
                child.nodes = parent.nodes.copy()
                child.connections = parent.connections.copy()
                
            # Применяем мутации
            child = self._mutate_genome(child)
            
            # Квантовая мутация
            if self.quantum_evolution:
                child = self.quantum_evolution.quantum_mutation(child, len(offspring))
                
            offspring.append(child)
            
        return offspring
    
    def _tournament_selection(self, candidates: List[NEATGenome], tournament_size: int = 3) -> NEATGenome:
        """Турнирный отбор"""
        tournament = random.sample(candidates, min(tournament_size, len(candidates)))
        return max(tournament, key=lambda x: x.adjusted_fitness)
    
    def _mutate_genome(self, genome: NEATGenome) -> NEATGenome:
        """Применяет мутации к геному"""
        # Мутация весов
        genome.mutate_weights(self.config.mutation_rate)
        
        # Мутация смещений
        genome.mutate_bias(self.config.mutation_rate * 0.5)
        
        # Мутация функций активации
        genome.mutate_activation(self.config.mutation_rate * 0.1, self.config.activation_functions)
        
        # Структурные мутации
        if random.random() < 0.03:  # 3% шанс добавить узел
            genome.add_node(self.global_innovations)
            
        if random.random() < 0.05:  # 5% шанс добавить соединение
            genome.add_connection()
            
        return genome
    
    def _update_statistics(self):
        """Обновляет статистику эволюции"""
        if self.population:
            avg_fitness = np.mean([genome.fitness for genome in self.population])
            max_fitness = max(genome.fitness for genome in self.population)
            avg_complexity = np.mean([len(genome.connections) for genome in self.population])
            
            self.fitness_history.append({
                'generation': self.generation,
                'avg_fitness': avg_fitness,
                'max_fitness': max_fitness,
                'num_species': len(self.species)
            })
            
            self.complexity_history.append({
                'generation': self.generation,
                'avg_connections': avg_complexity,
                'avg_nodes': np.mean([len(genome.nodes) for genome in self.population])
            })
            
            logger.info(f"Generation {self.generation}: "
                       f"Avg Fitness: {avg_fitness:.4f}, "
                       f"Max Fitness: {max_fitness:.4f}, "
                       f"Species: {len(self.species)}, "
                       f"Avg Complexity: {avg_complexity:.1f}")
    
    def get_best_genome(self) -> Optional[NEATGenome]:
        """Возвращает лучший геном"""
        if not self.population:
            return None
        return max(self.population, key=lambda x: x.fitness)
    
    def save_evolution_state(self, filepath: str):
        """Сохраняет состояние эволюции"""
        state = {
            'population': self.population,
            'species': self.species,
            'generation': self.generation,
            'config': self.config,
            'fitness_history': self.fitness_history,
            'complexity_history': self.complexity_history
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_evolution_state(self, filepath: str):
        """Загружает состояние эволюции"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        self.population = state['population']
        self.species = state['species']
        self.generation = state['generation']
        self.config = state['config']
        self.fitness_history = state.get('fitness_history', [])
        self.complexity_history = state.get('complexity_history', [])


# Фабрика для создания различных типов эволюции
class EvolutionFactory:
    """Фабрика эволюционных алгоритмов"""
    
    @staticmethod
    def create_neat_evolution(input_size: int, output_size: int, 
                             config: Optional[NeuroEvolutionConfig] = None) -> NEATEvolution:
        """Создает NEAT эволюцию"""
        if config is None:
            config = NeuroEvolutionConfig()
        return NEATEvolution(config, input_size, output_size)
    
    @staticmethod
    def create_quantum_neat(input_size: int, output_size: int) -> NEATEvolution:
        """Создает квантовую NEAT эволюцию"""
        config = NeuroEvolutionConfig()
        config.quantum_enabled = True
        config.quantum_noise = 0.15
        config.quantum_entanglement = 0.1
        return NEATEvolution(config, input_size, output_size)
    
    @staticmethod
    def create_multiobjective_neat(input_size: int, output_size: int, 
                                  objectives: List[str]) -> NEATEvolution:
        """Создает многокритериальную NEAT эволюцию"""
        config = NeuroEvolutionConfig()
        config.pareto_objectives = objectives
        config.population_size = 200  # Больше для многокритериальной оптимизации
        return NEATEvolution(config, input_size, output_size)


# Экспорт основных классов
__all__ = [
    'NeuroEvolutionConfig',
    'NEATGenome', 
    'NEATEvolution',
    'QuantumInspiredEvolution',
    'NSGA2Selection',
    'AdaptiveParameterController',
    'EvolutionFactory'
]