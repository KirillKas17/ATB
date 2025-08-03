"""
Оптимизатор стратегий для улучшения существующих стратегий.
"""
import random
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, cast, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from domain.types.evolution_types import (
    StrategyCandidate,
    EvolutionContext,
    IndicatorParameters,
    FilterParameters,
    EntryCondition,
    ExitCondition,
)

# Mock для optuna
class optuna_mock:
    class Trial:
        def suggest_float(self, name: str, low: float, high: float) -> float:
            return random.uniform(low, high)

        def suggest_int(self, name: str, low: int, high: int) -> int:
            return random.randint(low, high)

        def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
            return random.choice(choices)

    class Study:
        def __init__(self, direction: str = "minimize"):
            self.direction = direction
            self.trials: List[Any] = []

        def optimize(self, objective: Callable[[optuna_mock.Trial], float], n_trials: int) -> None:
            for _ in range(n_trials):
                trial = optuna_mock.Trial()
                objective(trial)

def create_study(direction: str = "minimize") -> optuna_mock.Study:
    return optuna_mock.Study(direction)

# Импортируем StrategyFitnessEvaluator из правильного места
from domain.evolution.strategy_fitness import StrategyFitnessEvaluator

class StrategyOptimizer:
    """Оптимизатор стратегий."""

    def __init__(
        self, context: EvolutionContext, fitness_evaluator: StrategyFitnessEvaluator
    ):
        self.context = context
        self.fitness_evaluator = fitness_evaluator
        self.optimization_history: List[Dict[str, Any]] = []

    def optimize_strategy(
        self,
        candidate: StrategyCandidate,
        historical_data: pd.DataFrame,
        optimization_type: str = "genetic",
        max_iterations: int = 100,
    ) -> StrategyCandidate:
        """Оптимизировать стратегию."""
        if optimization_type == "genetic":
            return self._genetic_optimization(candidate, historical_data, max_iterations)
        elif optimization_type == "bayesian":
            return self._bayesian_optimization(candidate, historical_data, max_iterations)
        elif optimization_type == "gradient":
            return self._gradient_optimization(candidate, historical_data, max_iterations)
        else:
            raise ValueError(f"Unknown optimization type: {optimization_type}")

    def _genetic_optimization(
        self,
        candidate: StrategyCandidate,
        historical_data: pd.DataFrame,
        max_iterations: int,
    ) -> StrategyCandidate:
        """Генетическая оптимизация."""
        population = [candidate]
        best_candidate = candidate
        best_fitness = self._evaluate_fitness(candidate, historical_data)

        for generation in range(max_iterations):
            # Создание новой популяции
            new_population = []
            for _ in range(len(population)):
                # Селекция
                parent1 = self._tournament_selection(population, historical_data)
                parent2 = self._tournament_selection(population, historical_data)
                # Скрещивание
                child = self._crossover(parent1, parent2)
                # Мутация
                if random.random() < 0.3:
                    child = self._mutate(child)
                new_population.append(child)

            # Оценка новой популяции
            for candidate in new_population:
                fitness = self._evaluate_fitness(candidate, historical_data)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_candidate = candidate

            population = new_population

            # Сохранение истории
            self.optimization_history.append({
                "generation": generation,
                "best_fitness": float(best_fitness),
                "optimization_type": "genetic",
            })

        return best_candidate

    def _bayesian_optimization(
        self,
        candidate: StrategyCandidate,
        historical_data: pd.DataFrame,
        max_iterations: int,
    ) -> StrategyCandidate:
        """Байесовская оптимизация."""
        study = create_study("maximize")

        def objective(trial: optuna_mock.Trial) -> float:
            # Создать параметры для trial
            params = self._create_trial_parameters(trial, candidate)
            # Применить параметры
            optimized_candidate = self._apply_parameters(candidate, params)
            # Оценить fitness
            fitness = self._evaluate_fitness(optimized_candidate, historical_data)
            return float(fitness)

        study.optimize(objective, max_iterations)
        return candidate  # Упрощенно возвращаем исходную стратегию

    def _gradient_optimization(
        self,
        candidate: StrategyCandidate,
        historical_data: pd.DataFrame,
        max_iterations: int,
    ) -> StrategyCandidate:
        """Градиентная оптимизация."""
        # Преобразуем стратегию в вектор
        initial_vector = self._strategy_to_vector(candidate)

        def objective(params: np.ndarray) -> float:
            # Преобразуем вектор обратно в стратегию
            strategy = self._vector_to_strategy(params, candidate)
            # Оцениваем fitness
            fitness = self._evaluate_fitness(strategy, historical_data)
            return -float(fitness)  # Минимизируем отрицательный fitness

        # Оптимизация
        result = minimize(objective, initial_vector, method="L-BFGS-B", options={"maxiter": max_iterations})
        if result.success:
            return self._vector_to_strategy(result.x, candidate)
        return candidate

    def _evaluate_fitness(
        self, candidate: StrategyCandidate, historical_data: pd.DataFrame
    ) -> Decimal:
        """Оценить fitness стратегии."""
        try:
            evaluation = self.fitness_evaluator.evaluate_strategy(
                candidate, historical_data
            )
            # Предполагаем, что результат имеет поле fitness_score
            return getattr(evaluation, 'fitness_score', Decimal("0"))
        except Exception:
            return Decimal("0")

    def _tournament_selection(
        self,
        population: List[StrategyCandidate],
        historical_data: pd.DataFrame,
        tournament_size: int = 3,
    ) -> StrategyCandidate:
        """Турнирная селекция."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        best = tournament[0]
        best_fitness = self._evaluate_fitness(best, historical_data)

        for candidate in tournament[1:]:
            fitness = self._evaluate_fitness(candidate, historical_data)
            if fitness > best_fitness:
                best = candidate
                best_fitness = fitness

        return best

    def _crossover(
        self, parent1: StrategyCandidate, parent2: StrategyCandidate
    ) -> StrategyCandidate:
        """Скрещивание стратегий."""
        child = parent1.clone()
        child.name = f"Child_{parent1.name}_{parent2.name}"
        child.generation = max(parent1.generation, parent2.generation) + 1
        child.parent_ids = [parent1.id, parent2.id]

        # Скрещивание индикаторов
        child.indicators = []
        for i in range(max(len(parent1.indicators), len(parent2.indicators))):
            if i < len(parent1.indicators) and i < len(parent2.indicators):
                if random.random() < 0.5:
                    child.indicators.append(parent1.indicators[i].clone())
                else:
                    child.indicators.append(parent2.indicators[i].clone())
            elif i < len(parent1.indicators):
                child.indicators.append(parent1.indicators[i].clone())
            else:
                child.indicators.append(parent2.indicators[i].clone())

        # Скрещивание фильтров
        child.filters = []
        for i in range(max(len(parent1.filters), len(parent2.filters))):
            if i < len(parent1.filters) and i < len(parent2.filters):
                if random.random() < 0.5:
                    child.filters.append(parent1.filters[i].clone())
                else:
                    child.filters.append(parent2.filters[i].clone())
            elif i < len(parent1.filters):
                child.filters.append(parent1.filters[i].clone())
            else:
                child.filters.append(parent2.filters[i].clone())

        # Скрещивание правил входа
        child.entry_rules = []
        for i in range(max(len(parent1.entry_rules), len(parent2.entry_rules))):
            if i < len(parent1.entry_rules) and i < len(parent2.entry_rules):
                if random.random() < 0.5:
                    child.entry_rules.append(parent1.entry_rules[i].clone())
                else:
                    child.entry_rules.append(parent2.entry_rules[i].clone())
            elif i < len(parent1.entry_rules):
                child.entry_rules.append(parent1.entry_rules[i].clone())
            else:
                child.entry_rules.append(parent2.entry_rules[i].clone())
        # Скрещивание правил выхода
        child.exit_rules = []
        for i in range(max(len(parent1.exit_rules), len(parent2.exit_rules))):
            if i < len(parent1.exit_rules) and i < len(parent2.exit_rules):
                if random.random() < 0.5:
                    child.exit_rules.append(parent1.exit_rules[i].clone())
                else:
                    child.exit_rules.append(parent2.exit_rules[i].clone())
            elif i < len(parent1.exit_rules):
                child.exit_rules.append(parent1.exit_rules[i].clone())
            else:
                child.exit_rules.append(parent2.exit_rules[i].clone())
        # Скрещивание параметров исполнения
        child.position_size_pct = (
            parent1.position_size_pct + parent2.position_size_pct
        ) / 2
        child.max_positions = (parent1.max_positions + parent2.max_positions) // 2
        child.min_holding_time = (parent1.min_holding_time + parent2.min_holding_time) // 2
        child.max_holding_time = (parent1.max_holding_time + parent2.max_holding_time) // 2
        return child

    def _mutate(self, candidate: StrategyCandidate) -> StrategyCandidate:
        """Мутация стратегии."""
        mutated = candidate.clone()
        mutated.name = f"Mutated_{candidate.name}"
        mutated.generation = candidate.generation + 1
        mutated.parent_ids = [candidate.id]
        mutated.increment_mutation_count()
        # Мутация индикаторов
        for indicator in mutated.indicators:
            if random.random() < 0.3:  # 30% вероятность мутации
                indicator_params = dict(indicator.parameters)
                mutated_params = self._mutate_parameters(indicator_params)
                # Создаем параметры индикаторов напрямую
                indicator_kwargs: Dict[str, Any] = {}
                if 'period' in mutated_params and isinstance(mutated_params['period'], int):
                    indicator_kwargs['period'] = int(mutated_params['period'])
                if 'fast' in mutated_params and isinstance(mutated_params['fast'], int):
                    indicator_kwargs['fast'] = int(mutated_params['fast'])
                if 'slow' in mutated_params and isinstance(mutated_params['slow'], int):
                    indicator_kwargs['slow'] = int(mutated_params['slow'])
                if 'signal' in mutated_params and isinstance(mutated_params['signal'], int):
                    indicator_kwargs['signal'] = int(mutated_params['signal'])
                if 'alpha' in mutated_params and isinstance(mutated_params['alpha'], (int, float)):
                    indicator_kwargs['alpha'] = float(mutated_params['alpha'])
                if 'beta' in mutated_params and isinstance(mutated_params['beta'], (int, float)):
                    indicator_kwargs['beta'] = float(mutated_params['beta'])
                if 'gamma' in mutated_params and isinstance(mutated_params['gamma'], (int, float)):
                    indicator_kwargs['gamma'] = float(mutated_params['gamma'])
                indicator.parameters = dict(self._create_indicator_parameters(indicator_kwargs))
        # Мутация фильтров
        for filter_config in mutated.filters:
            if random.random() < 0.3:
                filter_params = dict(filter_config.parameters)
                mutated_params = self._mutate_parameters(filter_params)
                # Создаем параметры фильтров напрямую
                filter_config.parameters = dict(self._create_filter_parameters(mutated_params))
        # Мутация параметров исполнения
        if random.random() < 0.5:
            mutated.position_size_pct = Decimal(
                str(max(0.01, min(0.5, float(mutated.position_size_pct) * random.uniform(0.8, 1.2))))
            )
        if random.random() < 0.5:
            mutated.max_positions = int(max(1, min(10, mutated.max_positions + random.randint(-1, 1))))
        if random.random() < 0.5:
            mutated.min_holding_time = int(max(30, mutated.min_holding_time + random.randint(-300, 300)))
        if random.random() < 0.5:
            mutated.max_holding_time = int(max(mutated.min_holding_time + 300, mutated.max_holding_time + random.randint(-3600, 3600)))
        return mutated

    def _mutate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Мутация параметров только по допустимым ключам TypedDict."""
        # Список допустимых ключей для индикаторов и фильтров
        allowed_keys = {
            'period', 'fast', 'slow', 'signal', 'acceleration', 'maximum', 'k_period', 'd_period', 'std_dev', 'multiplier',
            'period1', 'period2', 'period3', 'threshold', 'smoothing', 'alpha', 'beta', 'gamma',
            'min_atr', 'max_atr', 'min_width', 'max_width', 'min_volume', 'spike_threshold', 'min_adx', 'trend_period',
            'start_hour', 'end_hour', 'excluded_days', 'regime', 'max_correlation', 'volatility_threshold', 'volume_threshold', 'trend_strength'
        }
        mutated = parameters.copy()
        for key in list(mutated.keys()):
            if key not in allowed_keys:
                del mutated[key]
        for key, value in mutated.items():
            if isinstance(value, (int, float)):
                if random.random() < 0.3:
                    if isinstance(value, int):
                        mutated[key] = int(max(1, value + random.randint(-2, 2)))
                    else:
                        mutated[key] = float(value * random.uniform(0.8, 1.2))
            elif isinstance(value, list) and key == 'excluded_days':
                mutated[key] = [int(float(x)) for x in value if isinstance(x, (int, float))]
            elif isinstance(value, str) and key == 'regime':
                mutated[key] = str(value)
        return dict(mutated)  # всегда возвращаем dict

    def _create_trial_parameters(
        self, trial: optuna_mock.Trial, candidate: StrategyCandidate
    ) -> Dict[str, Any]:
        """Создать параметры для trial с разделением по индикаторам и фильтрам."""
        params: Dict[str, Any] = {}
        
        # Параметры индикаторов - создаем отдельные словари для каждого индикатора
        for i, indicator in enumerate(candidate.indicators):
            indicator_params: Dict[str, Any] = {}
            for param_name, param_value in indicator.parameters.items():
                if isinstance(param_value, int):
                    indicator_params[param_name] = trial.suggest_int(
                        f"indicator_{i}_{param_name}", 
                        max(1, param_value - 5), 
                        param_value + 5
                    )
                elif isinstance(param_value, float):
                    indicator_params[param_name] = trial.suggest_float(
                        f"indicator_{i}_{param_name}", 
                        param_value * 0.5, 
                        param_value * 1.5
                    )
            params[f"indicator_{i}"] = indicator_params
        
        # Параметры фильтров - создаем отдельные словари для каждого фильтра
        for i, filter_config in enumerate(candidate.filters):
            filter_params: Dict[str, Any] = {}
            for param_name, param_value in filter_config.parameters.items():
                if isinstance(param_value, int):
                    filter_params[param_name] = trial.suggest_int(
                        f"filter_{i}_{param_name}", 
                        max(1, param_value - 5), 
                        param_value + 5
                    )
                elif isinstance(param_value, float):
                    filter_params[param_name] = trial.suggest_float(
                        f"filter_{i}_{param_name}", 
                        param_value * 0.5, 
                        param_value * 1.5
                    )
            params[f"filter_{i}"] = filter_params
        
        # Параметры исполнения
        params["position_size_pct"] = trial.suggest_float("position_size_pct", 0.01, 0.5)
        params["max_positions"] = trial.suggest_int("max_positions", 1, 10)
        params["min_holding_time"] = trial.suggest_int("min_holding_time", 30, 3600)
        params["max_holding_time"] = trial.suggest_int("max_holding_time", 3600, 86400)
        
        return dict(params)  # всегда возвращаем dict

    def _apply_parameters(
        self, candidate: StrategyCandidate, params: Dict[str, Any]
    ) -> StrategyCandidate:
        """Применить параметры к стратегии."""
        optimized = candidate.clone()
        optimized.name = f"Optimized_{candidate.name}"
        
        # Список допустимых ключей для индикаторов и фильтров
        indicator_allowed_keys = {
            'period', 'fast', 'slow', 'signal', 'acceleration', 'maximum', 'k_period', 'd_period', 'std_dev', 'multiplier',
            'period1', 'period2', 'period3', 'threshold', 'smoothing', 'alpha', 'beta', 'gamma'
        }
        filter_allowed_keys = {
            'min_atr', 'max_atr', 'min_width', 'max_width', 'min_volume', 'spike_threshold', 'min_adx', 'trend_period',
            'start_hour', 'end_hour', 'excluded_days', 'regime', 'max_correlation', 'volatility_threshold', 'volume_threshold', 'trend_strength'
        }
        # Применить параметры индикаторов
        for i, indicator in enumerate(optimized.indicators):
            indicator_key = f"indicator_{i}"
            if indicator_key in params:
                indicator_params = params[indicator_key]
                # Создаем параметры индикаторов напрямую
                indicator_kwargs: Dict[str, Any] = {}
                if 'period' in indicator_params and isinstance(indicator_params['period'], int):
                    indicator_kwargs['period'] = int(indicator_params['period'])
                if 'fast' in indicator_params and isinstance(indicator_params['fast'], int):
                    indicator_kwargs['fast'] = int(indicator_params['fast'])
                if 'slow' in indicator_params and isinstance(indicator_params['slow'], int):
                    indicator_kwargs['slow'] = int(indicator_params['slow'])
                if 'signal' in indicator_params and isinstance(indicator_params['signal'], int):
                    indicator_kwargs['signal'] = int(indicator_params['signal'])
                if 'alpha' in indicator_params and isinstance(indicator_params['alpha'], (int, float)):
                    indicator_kwargs['alpha'] = float(indicator_params['alpha'])
                if 'beta' in indicator_params and isinstance(indicator_params['beta'], (int, float)):
                    indicator_kwargs['beta'] = float(indicator_params['beta'])
                if 'gamma' in indicator_params and isinstance(indicator_params['gamma'], (int, float)):
                    indicator_kwargs['gamma'] = float(indicator_params['gamma'])
                indicator.parameters = dict(self._create_indicator_parameters(indicator_kwargs))
        
        # Применить параметры фильтров
        for i, filter_config in enumerate(optimized.filters):
            filter_key = f"filter_{i}"
            if filter_key in params:
                filter_params = params[filter_key]
                # Создаем параметры фильтров напрямую
                filter_config.parameters = dict(self._create_filter_parameters(filter_params))
        # Применить параметры исполнения
        if "position_size_pct" in params:
            optimized.position_size_pct = Decimal(str(params["position_size_pct"]))
        if "max_positions" in params:
            optimized.max_positions = int(params["max_positions"])
        if "min_holding_time" in params:
            optimized.min_holding_time = int(params["min_holding_time"])
        if "max_holding_time" in params:
            optimized.max_holding_time = int(params["max_holding_time"])
        return optimized

    def _strategy_to_vector(self, candidate: StrategyCandidate) -> np.ndarray:
        """Преобразовать стратегию в вектор параметров."""
        vector: List[float] = []
        # Параметры индикаторов
        for indicator in candidate.indicators:
            for param_value in indicator.parameters.values():
                if isinstance(param_value, (int, float)):
                    vector.append(float(param_value))
        # Параметры фильтров
        for filter_config in candidate.filters:
            for param_value in filter_config.parameters.values():
                if isinstance(param_value, (int, float)):
                    vector.append(float(param_value))
        # Параметры исполнения
        vector.extend([
            float(candidate.position_size_pct),
            float(candidate.max_positions),
            float(candidate.min_holding_time),
            float(candidate.max_holding_time)
        ])
        return np.array(vector)

    def _vector_to_strategy(
        self, vector: np.ndarray, template: StrategyCandidate
    ) -> StrategyCandidate:
        """Преобразовать вектор параметров в стратегию."""
        strategy = template.clone()
        strategy.name = f"Vector_{template.name}"
        idx = 0
        indicator_allowed_keys = [
            'period', 'fast', 'slow', 'signal', 'acceleration', 'maximum', 'k_period', 'd_period', 'std_dev', 'multiplier',
            'period1', 'period2', 'period3', 'threshold', 'smoothing', 'alpha', 'beta', 'gamma'
        ]
        filter_allowed_keys = [
            'min_atr', 'max_atr', 'min_width', 'max_width', 'min_volume', 'spike_threshold', 'min_adx', 'trend_period',
            'start_hour', 'end_hour', 'excluded_days', 'regime', 'max_correlation', 'volatility_threshold', 'volume_threshold', 'trend_strength'
        ]
        # Параметры индикаторов
        for indicator in strategy.indicators:
            indicator_kwargs: Dict[str, Any] = {}
            # Проверяем каждый параметр явно
            if 'period' in indicator.parameters and isinstance(indicator.parameters['period'], (int, float)):
                indicator_kwargs['period'] = int(float(vector[idx]))
                idx += 1
            if 'fast' in indicator.parameters and isinstance(indicator.parameters['fast'], (int, float)):
                indicator_kwargs['fast'] = int(float(vector[idx]))
                idx += 1
            if 'slow' in indicator.parameters and isinstance(indicator.parameters['slow'], (int, float)):
                indicator_kwargs['slow'] = int(float(vector[idx]))
                idx += 1
            if 'signal' in indicator.parameters and isinstance(indicator.parameters['signal'], (int, float)):
                indicator_kwargs['signal'] = int(float(vector[idx]))
                idx += 1
            if 'acceleration' in indicator.parameters and isinstance(indicator.parameters['acceleration'], (int, float)):
                indicator_kwargs['acceleration'] = float(vector[idx])
                idx += 1
            if 'maximum' in indicator.parameters and isinstance(indicator.parameters['maximum'], (int, float)):
                indicator_kwargs['maximum'] = float(vector[idx])
                idx += 1
            if 'k_period' in indicator.parameters and isinstance(indicator.parameters['k_period'], (int, float)):
                indicator_kwargs['k_period'] = int(float(vector[idx]))
                idx += 1
            if 'd_period' in indicator.parameters and isinstance(indicator.parameters['d_period'], (int, float)):
                indicator_kwargs['d_period'] = int(float(vector[idx]))
                idx += 1
            if 'std_dev' in indicator.parameters and isinstance(indicator.parameters['std_dev'], (int, float)):
                indicator_kwargs['std_dev'] = float(vector[idx])
                idx += 1
            if 'multiplier' in indicator.parameters and isinstance(indicator.parameters['multiplier'], (int, float)):
                indicator_kwargs['multiplier'] = float(vector[idx])
                idx += 1
            if 'period1' in indicator.parameters and isinstance(indicator.parameters['period1'], (int, float)):
                indicator_kwargs['period1'] = int(float(vector[idx]))
                idx += 1
            if 'period2' in indicator.parameters and isinstance(indicator.parameters['period2'], (int, float)):
                indicator_kwargs['period2'] = int(float(vector[idx]))
                idx += 1
            if 'period3' in indicator.parameters and isinstance(indicator.parameters['period3'], (int, float)):
                indicator_kwargs['period3'] = int(float(vector[idx]))
                idx += 1
            if 'threshold' in indicator.parameters and isinstance(indicator.parameters['threshold'], (int, float)):
                indicator_kwargs['threshold'] = float(vector[idx])
                idx += 1
            if 'smoothing' in indicator.parameters and isinstance(indicator.parameters['smoothing'], (int, float)):
                indicator_kwargs['smoothing'] = int(float(vector[idx]))
                idx += 1
            if 'alpha' in indicator.parameters and isinstance(indicator.parameters['alpha'], (int, float)):
                indicator_kwargs['alpha'] = float(vector[idx])
                idx += 1
            if 'beta' in indicator.parameters and isinstance(indicator.parameters['beta'], (int, float)):
                indicator_kwargs['beta'] = float(vector[idx])
                idx += 1
            if 'gamma' in indicator.parameters and isinstance(indicator.parameters['gamma'], (int, float)):
                indicator_kwargs['gamma'] = float(vector[idx])
                idx += 1
            indicator.parameters = dict(self._create_indicator_parameters(indicator_kwargs))
        # Параметры фильтров
        for filter_config in strategy.filters:
            filter_kwargs: Dict[str, Any] = {}
            # Проверяем каждый параметр явно
            if 'min_atr' in filter_config.parameters and isinstance(filter_config.parameters['min_atr'], (int, float)):
                filter_kwargs['min_atr'] = float(vector[idx])
                idx += 1
            if 'max_atr' in filter_config.parameters and isinstance(filter_config.parameters['max_atr'], (int, float)):
                filter_kwargs['max_atr'] = float(vector[idx])
                idx += 1
            if 'min_width' in filter_config.parameters and isinstance(filter_config.parameters['min_width'], (int, float)):
                filter_kwargs['min_width'] = float(vector[idx])
                idx += 1
            if 'max_width' in filter_config.parameters and isinstance(filter_config.parameters['max_width'], (int, float)):
                filter_kwargs['max_width'] = float(vector[idx])
                idx += 1
            if 'min_volume' in filter_config.parameters and isinstance(filter_config.parameters['min_volume'], (int, float)):
                filter_kwargs['min_volume'] = int(float(vector[idx]))
                idx += 1
            if 'spike_threshold' in filter_config.parameters and isinstance(filter_config.parameters['spike_threshold'], (int, float)):
                filter_kwargs['spike_threshold'] = float(vector[idx])
                idx += 1
            if 'min_adx' in filter_config.parameters and isinstance(filter_config.parameters['min_adx'], (int, float)):
                filter_kwargs['min_adx'] = int(float(vector[idx]))
                idx += 1
            if 'trend_period' in filter_config.parameters and isinstance(filter_config.parameters['trend_period'], (int, float)):
                filter_kwargs['trend_period'] = int(float(vector[idx]))
                idx += 1
            if 'start_hour' in filter_config.parameters and isinstance(filter_config.parameters['start_hour'], (int, float)):
                filter_kwargs['start_hour'] = int(float(vector[idx]))
                idx += 1
            if 'end_hour' in filter_config.parameters and isinstance(filter_config.parameters['end_hour'], (int, float)):
                filter_kwargs['end_hour'] = int(float(vector[idx]))
                idx += 1
            if 'max_correlation' in filter_config.parameters and isinstance(filter_config.parameters['max_correlation'], (int, float)):
                filter_kwargs['max_correlation'] = float(vector[idx])
                idx += 1
            if 'volatility_threshold' in filter_config.parameters and isinstance(filter_config.parameters['volatility_threshold'], (int, float)):
                filter_kwargs['volatility_threshold'] = float(vector[idx])
                idx += 1
            if 'volume_threshold' in filter_config.parameters and isinstance(filter_config.parameters['volume_threshold'], (int, float)):
                filter_kwargs['volume_threshold'] = float(vector[idx])
                idx += 1
            if 'trend_strength' in filter_config.parameters and isinstance(filter_config.parameters['trend_strength'], (int, float)):
                filter_kwargs['trend_strength'] = float(vector[idx])
                idx += 1
            filter_config.parameters = dict(self._create_filter_parameters(filter_kwargs))
        # Параметры исполнения
        strategy.position_size_pct = Decimal(str(vector[idx]))
        strategy.max_positions = int(float(vector[idx + 1]))
        strategy.min_holding_time = int(float(vector[idx + 2]))
        strategy.max_holding_time = int(float(vector[idx + 3]))
        return strategy

    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Получить историю оптимизации."""
        return self.optimization_history

    def optimize_population(
        self,
        population: List[StrategyCandidate],
        historical_data: pd.DataFrame,
        optimization_type: str = "genetic",
        max_iterations: int = 50,
    ) -> List[StrategyCandidate]:
        """Оптимизировать популяцию стратегий."""
        optimized_population = []
        for candidate in population:
            try:
                optimized = self.optimize_strategy(
                    candidate, historical_data, optimization_type, max_iterations
                )
                optimized_population.append(optimized)
            except Exception as e:
                print(f"Ошибка оптимизации стратегии {candidate.name}: {e}")
                optimized_population.append(candidate)
        return optimized_population

    def _create_indicator_parameters(self, params: dict[str, Any]) -> IndicatorParameters:
        """Создать IndicatorParameters безопасно, исключая None и неподходящие типы."""
        # Создаем TypedDict напрямую
        result: IndicatorParameters = {}
        for key, value in params.items():
            if value is not None:
                if key == 'period' and isinstance(value, int):
                    result['period'] = value
                elif key == 'fast' and isinstance(value, int):
                    result['fast'] = value
                elif key == 'slow' and isinstance(value, int):
                    result['slow'] = value
                elif key == 'signal' and isinstance(value, int):
                    result['signal'] = value
                elif key == 'k_period' and isinstance(value, int):
                    result['k_period'] = value
                elif key == 'd_period' and isinstance(value, int):
                    result['d_period'] = value
                elif key == 'period1' and isinstance(value, int):
                    result['period1'] = value
                elif key == 'period2' and isinstance(value, int):
                    result['period2'] = value
                elif key == 'period3' and isinstance(value, int):
                    result['period3'] = value
                elif key == 'smoothing' and isinstance(value, int):
                    result['smoothing'] = value
                elif key == 'acceleration' and isinstance(value, float):
                    result['acceleration'] = value
                elif key == 'maximum' and isinstance(value, float):
                    result['maximum'] = value
                elif key == 'std_dev' and isinstance(value, float):
                    result['std_dev'] = value
                elif key == 'multiplier' and isinstance(value, float):
                    result['multiplier'] = value
                elif key == 'threshold' and isinstance(value, float):
                    result['threshold'] = value
                elif key == 'alpha' and isinstance(value, float):
                    result['alpha'] = value
                elif key == 'beta' and isinstance(value, float):
                    result['beta'] = value
                elif key == 'gamma' and isinstance(value, float):
                    result['gamma'] = value
        return result

    def _create_filter_parameters(self, params: dict[str, Any]) -> FilterParameters:
        """Создать FilterParameters безопасно, исключая None и неподходящие типы."""
        # Создаем TypedDict напрямую
        result: FilterParameters = {}
        for key, value in params.items():
            if value is not None:
                if key == 'min_volume' and isinstance(value, int):
                    result['min_volume'] = value
                elif key == 'min_adx' and isinstance(value, int):
                    result['min_adx'] = value
                elif key == 'trend_period' and isinstance(value, int):
                    result['trend_period'] = value
                elif key == 'start_hour' and isinstance(value, int):
                    result['start_hour'] = value
                elif key == 'end_hour' and isinstance(value, int):
                    result['end_hour'] = value
                elif key == 'min_atr' and isinstance(value, float):
                    result['min_atr'] = value
                elif key == 'max_atr' and isinstance(value, float):
                    result['max_atr'] = value
                elif key == 'min_width' and isinstance(value, float):
                    result['min_width'] = value
                elif key == 'max_width' and isinstance(value, float):
                    result['max_width'] = value
                elif key == 'spike_threshold' and isinstance(value, float):
                    result['spike_threshold'] = value
                elif key == 'max_correlation' and isinstance(value, float):
                    result['max_correlation'] = value
                elif key == 'volatility_threshold' and isinstance(value, float):
                    result['volatility_threshold'] = value
                elif key == 'volume_threshold' and isinstance(value, float):
                    result['volume_threshold'] = value
                elif key == 'trend_strength' and isinstance(value, float):
                    result['trend_strength'] = value
                elif key == 'excluded_days' and isinstance(value, list) and all(isinstance(x, int) for x in value):
                    result['excluded_days'] = value
                elif key == 'regime' and isinstance(value, str):
                    result['regime'] = value
        return result