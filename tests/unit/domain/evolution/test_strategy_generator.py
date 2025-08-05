"""
Unit тесты для strategy_generator.py.

Покрывает:
- StrategyGenerator - генератор стратегий для эволюционного алгоритма
- Методы генерации случайных стратегий
- Методы генерации популяции
- Методы генерации стратегий от родителей
- Создание условий входа и выхода
- Валидация и оптимизация стратегий
"""

import pytest
import random
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

from domain.evolution.strategy_generator import StrategyGenerator
from domain.evolution.strategy_model import (
    EvolutionContext,
    EvolutionStatus,
    StrategyCandidate,
    EntryCondition,
    ExitCondition
)
from domain.type_definitions.evolution_types import (
    FitnessScore,
    FitnessWeights,
    StrategyPerformance
)
from domain.type_definitions.technical_types import SignalType
from domain.type_definitions.strategy_types import StrategyType
from domain.exceptions.base_exceptions import ValidationError


class TestStrategyGenerator:
    """Тесты для StrategyGenerator."""

    @pytest.fixture
    def sample_context(self) -> EvolutionContext:
        """Тестовый контекст эволюции."""
        return EvolutionContext(
            min_accuracy=Decimal("0.6"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.1"),
            min_sharpe=Decimal("1.0"),
            population_size=50,
            generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=5
        )

    @pytest.fixture
    def generator(self, sample_context: EvolutionContext) -> StrategyGenerator:
        """Тестовый генератор."""
        return StrategyGenerator(context=sample_context)

    def test_initialization(self, sample_context: EvolutionContext) -> None:
        """Тест инициализации."""
        generator = StrategyGenerator(context=sample_context)
        
        assert generator.context == sample_context
        assert generator.context.population_size == 50
        assert generator.context.generations == 100
        assert generator.context.mutation_rate == 0.1
        assert generator.context.crossover_rate == 0.8
        assert generator.context.elite_size == 5

    def test_generate_random_strategy(self, generator: StrategyGenerator) -> None:
        """Тест генерации случайной стратегии."""
        strategy = generator.generate_random_strategy()
        
        assert isinstance(strategy, StrategyCandidate)
        assert strategy.id is not None
        assert isinstance(strategy.name, str)
        assert isinstance(strategy.description, str)
        assert isinstance(strategy.entry_conditions, list)
        assert isinstance(strategy.exit_conditions, list)
        assert isinstance(strategy.parameters, dict)
        assert strategy.status == EvolutionStatus.CREATED

    def test_generate_random_strategy_with_name(self, generator: StrategyGenerator) -> None:
        """Тест генерации случайной стратегии с именем."""
        strategy = generator.generate_random_strategy(name="Test Strategy")
        
        assert strategy.name == "Test Strategy"
        assert isinstance(strategy, StrategyCandidate)

    def test_generate_population(self, generator: StrategyGenerator) -> None:
        """Тест генерации популяции."""
        population = generator.generate_population()
        
        assert isinstance(population, list)
        assert len(population) == generator.context.population_size
        assert all(isinstance(s, StrategyCandidate) for s in population)

    def test_generate_population_custom_size(self, generator: StrategyGenerator) -> None:
        """Тест генерации популяции с кастомным размером."""
        custom_size = 20
        population = generator.generate_population(size=custom_size)
        
        assert isinstance(population, list)
        assert len(population) == custom_size
        assert all(isinstance(s, StrategyCandidate) for s in population)

    def test_generate_from_parents(self, generator: StrategyGenerator) -> None:
        """Тест генерации стратегии от родителей."""
        # Создаем родительские стратегии
        parent1 = generator.generate_random_strategy(name="Parent 1")
        parent2 = generator.generate_random_strategy(name="Parent 2")
        
        child = generator.generate_from_parents(parent1, parent2)
        
        assert isinstance(child, StrategyCandidate)
        assert child.id != parent1.id
        assert child.id != parent2.id
        assert isinstance(child.name, str)
        assert isinstance(child.description, str)
        assert isinstance(child.entry_conditions, list)
        assert isinstance(child.exit_conditions, list)
        assert isinstance(child.parameters, dict)

    def test_generate_from_parents_with_mutation(self, generator: StrategyGenerator) -> None:
        """Тест генерации стратегии от родителей с мутацией."""
        parent1 = generator.generate_random_strategy(name="Parent 1")
        parent2 = generator.generate_random_strategy(name="Parent 2")
        
        child = generator.generate_from_parents(parent1, parent2, apply_mutation=True)
        
        assert isinstance(child, StrategyCandidate)
        assert child.id != parent1.id
        assert child.id != parent2.id

    def test_create_random_entry_condition(self, generator: StrategyGenerator) -> None:
        """Тест создания случайного условия входа."""
        condition = generator._create_random_entry_condition()
        
        assert isinstance(condition, EntryCondition)
        assert isinstance(condition.indicator, str)
        assert isinstance(condition.operator, str)
        assert isinstance(condition.value, (int, float))
        assert isinstance(condition.timeframe, str)

    def test_create_random_exit_condition(self, generator: StrategyGenerator) -> None:
        """Тест создания случайного условия выхода."""
        condition = generator._create_random_exit_condition()
        
        assert isinstance(condition, ExitCondition)
        assert isinstance(condition.indicator, str)
        assert isinstance(condition.operator, str)
        assert isinstance(condition.value, (int, float))
        assert isinstance(condition.timeframe, str)

    def test_create_random_parameters(self, generator: StrategyGenerator) -> None:
        """Тест создания случайных параметров."""
        parameters = generator._create_random_parameters()
        
        assert isinstance(parameters, dict)
        assert "position_size" in parameters
        assert "stop_loss" in parameters
        assert "take_profit" in parameters
        assert isinstance(parameters["position_size"], (int, float))
        assert isinstance(parameters["stop_loss"], (int, float))
        assert isinstance(parameters["take_profit"], (int, float))

    def test_crossover_conditions(self, generator: StrategyGenerator) -> None:
        """Тест скрещивания условий."""
        parent1_conditions = [
            EntryCondition(indicator="sma", operator="crossover", value=50.0, timeframe="1h"),
            EntryCondition(indicator="rsi", operator="less_than", value=30.0, timeframe="1h")
        ]
        parent2_conditions = [
            EntryCondition(indicator="macd", operator="greater_than", value=0.0, timeframe="1h"),
            EntryCondition(indicator="volume", operator="greater_than", value=1000.0, timeframe="1h")
        ]
        
        child_conditions = generator._crossover_conditions(parent1_conditions, parent2_conditions)
        
        assert isinstance(child_conditions, list)
        assert len(child_conditions) > 0
        assert all(isinstance(c, EntryCondition) for c in child_conditions)

    def test_crossover_parameters(self, generator: StrategyGenerator) -> None:
        """Тест скрещивания параметров."""
        parent1_params = {
            "position_size": 0.1,
            "stop_loss": 0.01,
            "take_profit": 0.02
        }
        parent2_params = {
            "position_size": 0.2,
            "stop_loss": 0.02,
            "take_profit": 0.04
        }
        
        child_params = generator._crossover_parameters(parent1_params, parent2_params)
        
        assert isinstance(child_params, dict)
        assert "position_size" in child_params
        assert "stop_loss" in child_params
        assert "take_profit" in child_params
        assert isinstance(child_params["position_size"], (int, float))
        assert isinstance(child_params["stop_loss"], (int, float))
        assert isinstance(child_params["take_profit"], (int, float))

    def test_mutate_condition(self, generator: StrategyGenerator) -> None:
        """Тест мутации условия."""
        condition = EntryCondition(
            indicator="sma",
            operator="crossover",
            value=50.0,
            timeframe="1h"
        )
        
        mutated_condition = generator._mutate_condition(condition)
        
        assert isinstance(mutated_condition, EntryCondition)
        assert isinstance(mutated_condition.indicator, str)
        assert isinstance(mutated_condition.operator, str)
        assert isinstance(mutated_condition.value, (int, float))
        assert isinstance(mutated_condition.timeframe, str)

    def test_mutate_parameters(self, generator: StrategyGenerator) -> None:
        """Тест мутации параметров."""
        parameters = {
            "position_size": 0.1,
            "stop_loss": 0.01,
            "take_profit": 0.02
        }
        
        mutated_params = generator._mutate_parameters(parameters)
        
        assert isinstance(mutated_params, dict)
        assert "position_size" in mutated_params
        assert "stop_loss" in mutated_params
        assert "take_profit" in mutated_params
        assert isinstance(mutated_params["position_size"], (int, float))
        assert isinstance(mutated_params["stop_loss"], (int, float))
        assert isinstance(mutated_params["take_profit"], (int, float))

    def test_validate_strategy(self, generator: StrategyGenerator) -> None:
        """Тест валидации стратегии."""
        strategy = generator.generate_random_strategy()
        
        is_valid = generator._validate_strategy(strategy)
        
        assert isinstance(is_valid, bool)

    def test_optimize_strategy(self, generator: StrategyGenerator) -> None:
        """Тест оптимизации стратегии."""
        strategy = generator.generate_random_strategy()
        
        optimized_strategy = generator._optimize_strategy(strategy)
        
        assert isinstance(optimized_strategy, StrategyCandidate)
        assert optimized_strategy.id == strategy.id

    def test_generate_strategy_name(self, generator: StrategyGenerator) -> None:
        """Тест генерации имени стратегии."""
        name = generator._generate_strategy_name()
        
        assert isinstance(name, str)
        assert len(name) > 0

    def test_generate_strategy_description(self, generator: StrategyGenerator) -> None:
        """Тест генерации описания стратегии."""
        description = generator._generate_strategy_description()
        
        assert isinstance(description, str)
        assert len(description) > 0

    def test_get_random_indicator(self, generator: StrategyGenerator) -> None:
        """Тест получения случайного индикатора."""
        indicator = generator._get_random_indicator()
        
        assert isinstance(indicator, str)
        assert len(indicator) > 0

    def test_get_random_operator(self, generator: StrategyGenerator) -> None:
        """Тест получения случайного оператора."""
        operator = generator._get_random_operator()
        
        assert isinstance(operator, str)
        assert len(operator) > 0

    def test_get_random_timeframe(self, generator: StrategyGenerator) -> None:
        """Тест получения случайного таймфрейма."""
        timeframe = generator._get_random_timeframe()
        
        assert isinstance(timeframe, str)
        assert len(timeframe) > 0

    def test_get_random_value(self, generator: StrategyGenerator) -> None:
        """Тест получения случайного значения."""
        value = generator._get_random_value("sma")
        
        assert isinstance(value, (int, float))

    def test_error_handling_invalid_context(self: "TestStrategyGenerator") -> None:
        """Тест обработки ошибок с невалидным контекстом."""
        invalid_context = Mock(spec=EvolutionContext)
        invalid_context.population_size = -1
        
        with pytest.raises(ValueError):
            StrategyGenerator(context=invalid_context)

    def test_error_handling_invalid_parents(self, generator: StrategyGenerator) -> None:
        """Тест обработки ошибок с невалидными родителями."""
        invalid_parent = Mock(spec=StrategyCandidate)
        valid_parent = generator.generate_random_strategy()
        
        with pytest.raises(ValueError):
            generator.generate_from_parents(invalid_parent, valid_parent)

    def test_performance_with_large_population(self, generator: StrategyGenerator) -> None:
        """Тест производительности с большой популяцией."""
        large_size = 100
        
        start_time = datetime.now()
        population = generator.generate_population(size=large_size)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Обработка должна быть быстрой (менее 2 секунд)
        assert processing_time < 2.0
        assert len(population) == large_size
        assert all(isinstance(s, StrategyCandidate) for s in population)

    def test_diversity_in_population(self, generator: StrategyGenerator) -> None:
        """Тест разнообразия в популяции."""
        population = generator.generate_population(size=20)
        
        # Проверяем, что стратегии имеют разные ID
        strategy_ids = [s.id for s in population]
        unique_ids = set(strategy_ids)
        
        assert len(unique_ids) == len(strategy_ids)  # Все ID уникальны

    def test_inheritance_from_parents(self, generator: StrategyGenerator) -> None:
        """Тест наследования от родителей."""
        parent1 = generator.generate_random_strategy(name="Parent 1")
        parent2 = generator.generate_random_strategy(name="Parent 2")
        
        child = generator.generate_from_parents(parent1, parent2)
        
        # Ребенок должен иметь характеристики от обоих родителей
        assert child.entry_conditions is not None
        assert child.exit_conditions is not None
        assert child.parameters is not None
        assert len(child.entry_conditions) > 0 or len(parent1.entry_conditions) > 0 or len(parent2.entry_conditions) > 0

    def test_mutation_effectiveness(self, generator: StrategyGenerator) -> None:
        """Тест эффективности мутации."""
        original_strategy = generator.generate_random_strategy()
        
        # Создаем копию для мутации
        mutated_strategy = StrategyCandidate(
            id=original_strategy.id,
            name=original_strategy.name,
            description=original_strategy.description,
            entry_conditions=original_strategy.entry_conditions.copy(),
            exit_conditions=original_strategy.exit_conditions.copy(),
            parameters=original_strategy.parameters.copy(),
            status=original_strategy.status
        )
        
        # Применяем мутацию
        generator._mutate_condition(mutated_strategy.entry_conditions[0])
        generator._mutate_parameters(mutated_strategy.parameters)
        
        # Проверяем, что что-то изменилось
        assert (mutated_strategy.entry_conditions != original_strategy.entry_conditions or
                mutated_strategy.parameters != original_strategy.parameters)

    def test_strategy_complexity(self, generator: StrategyGenerator) -> None:
        """Тест сложности стратегий."""
        strategies = [generator.generate_random_strategy() for _ in range(10)]
        
        for strategy in strategies:
            # Проверяем, что стратегия имеет разумную сложность
            assert len(strategy.entry_conditions) <= 5  # Не слишком много условий входа
            assert len(strategy.exit_conditions) <= 5   # Не слишком много условий выхода
            assert len(strategy.parameters) >= 3        # Минимум параметров

    def test_parameter_bounds(self, generator: StrategyGenerator) -> None:
        """Тест границ параметров."""
        strategy = generator.generate_random_strategy()
        
        params = strategy.parameters
        
        # Проверяем разумные границы параметров
        assert 0 < params["position_size"] <= 1.0  # Размер позиции от 0 до 100%
        assert 0 < params["stop_loss"] <= 0.5      # Стоп-лосс от 0 до 50%
        assert 0 < params["take_profit"] <= 1.0    # Тейк-профит от 0 до 100% 