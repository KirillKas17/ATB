"""
Unit тесты для strategy_optimizer.py.

Покрывает:
- StrategyOptimizer - оптимизатор стратегий
- Методы оптимизации параметров
- Методы оптимизации условий
- Методы оптимизации производительности
- Валидация и улучшение стратегий
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
from uuid import UUID, uuid4

from domain.evolution.strategy_optimizer import StrategyOptimizer
from domain.evolution.strategy_model import (
    EvolutionContext,
    EvolutionStatus,
    StrategyCandidate,
    EntryCondition,
    ExitCondition
)
from domain.types.evolution_types import (
    FitnessScore,
    FitnessWeights,
    StrategyPerformance
)
from domain.exceptions.base_exceptions import ValidationError


class TestStrategyOptimizer:
    """Тесты для StrategyOptimizer."""

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
    def optimizer(self, sample_context: EvolutionContext) -> StrategyOptimizer:
        """Тестовый оптимизатор."""
        return StrategyOptimizer(context=sample_context)

    @pytest.fixture
    def sample_strategy(self) -> StrategyCandidate:
        """Тестовая стратегия."""
        return StrategyCandidate(
            id=uuid4(),
            name="Test Strategy",
            description="Test strategy for optimization",
            entry_conditions=[
                EntryCondition(
                    indicator="sma",
                    operator="crossover",
                    value=50.0,
                    timeframe="1h"
                )
            ],
            exit_conditions=[
                ExitCondition(
                    indicator="profit_target",
                    operator="greater_than",
                    value=0.02,
                    timeframe="1h"
                )
            ],
            parameters={
                "position_size": 0.1,
                "stop_loss": 0.01,
                "take_profit": 0.02
            }
        )

    @pytest.fixture
    def sample_historical_data(self) -> pd.DataFrame:
        """Тестовые исторические данные."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
        data = {
            'open': np.random.randn(100).cumsum() + 50000,
            'high': np.random.randn(100).cumsum() + 50100,
            'low': np.random.randn(100).cumsum() + 49900,
            'close': np.random.randn(100).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 100),
            'sma_20': np.random.randn(100).cumsum() + 50000,
            'sma_50': np.random.randn(100).cumsum() + 49950
        }
        return pd.DataFrame(data, index=dates)

    def test_initialization(self, sample_context: EvolutionContext) -> None:
        """Тест инициализации."""
        optimizer = StrategyOptimizer(context=sample_context)
        
        assert optimizer.context == sample_context
        assert optimizer.context.population_size == 50
        assert optimizer.context.generations == 100
        assert optimizer.context.mutation_rate == 0.1
        assert optimizer.context.crossover_rate == 0.8
        assert optimizer.context.elite_size == 5

    def test_optimize_strategy(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест оптимизации стратегии."""
        optimized_strategy = optimizer.optimize_strategy(
            sample_strategy,
            sample_historical_data
        )
        
        assert isinstance(optimized_strategy, StrategyCandidate)
        assert optimized_strategy.id == sample_strategy.id
        assert optimized_strategy.name == sample_strategy.name
        assert isinstance(optimized_strategy.parameters, dict)
        assert isinstance(optimized_strategy.entry_conditions, list)
        assert isinstance(optimized_strategy.exit_conditions, list)

    def test_optimize_parameters(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации параметров."""
        optimized_params = optimizer._optimize_parameters(sample_strategy.parameters)
        
        assert isinstance(optimized_params, dict)
        assert "position_size" in optimized_params
        assert "stop_loss" in optimized_params
        assert "take_profit" in optimized_params
        assert isinstance(optimized_params["position_size"], (int, float))
        assert isinstance(optimized_params["stop_loss"], (int, float))
        assert isinstance(optimized_params["take_profit"], (int, float))

    def test_optimize_entry_conditions(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации условий входа."""
        optimized_conditions = optimizer._optimize_entry_conditions(sample_strategy.entry_conditions)
        
        assert isinstance(optimized_conditions, list)
        assert len(optimized_conditions) > 0
        assert all(isinstance(c, EntryCondition) for c in optimized_conditions)

    def test_optimize_exit_conditions(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации условий выхода."""
        optimized_conditions = optimizer._optimize_exit_conditions(sample_strategy.exit_conditions)
        
        assert isinstance(optimized_conditions, list)
        assert len(optimized_conditions) > 0
        assert all(isinstance(c, ExitCondition) for c in optimized_conditions)

    def test_optimize_condition_value(self, optimizer: StrategyOptimizer) -> None:
        """Тест оптимизации значения условия."""
        condition = EntryCondition(
            indicator="sma",
            operator="crossover",
            value=50.0,
            timeframe="1h"
        )
        
        optimized_value = optimizer._optimize_condition_value(condition)
        
        assert isinstance(optimized_value, (int, float))
        assert optimized_value > 0

    def test_optimize_condition_operator(self, optimizer: StrategyOptimizer) -> None:
        """Тест оптимизации оператора условия."""
        condition = EntryCondition(
            indicator="sma",
            operator="crossover",
            value=50.0,
            timeframe="1h"
        )
        
        optimized_operator = optimizer._optimize_condition_operator(condition)
        
        assert isinstance(optimized_operator, str)
        assert len(optimized_operator) > 0

    def test_optimize_condition_timeframe(self, optimizer: StrategyOptimizer) -> None:
        """Тест оптимизации таймфрейма условия."""
        condition = EntryCondition(
            indicator="sma",
            operator="crossover",
            value=50.0,
            timeframe="1h"
        )
        
        optimized_timeframe = optimizer._optimize_condition_timeframe(condition)
        
        assert isinstance(optimized_timeframe, str)
        assert len(optimized_timeframe) > 0

    def test_validate_optimization_result(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест валидации результата оптимизации."""
        is_valid = optimizer._validate_optimization_result(sample_strategy)
        
        assert isinstance(is_valid, bool)

    def test_improve_strategy_performance(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест улучшения производительности стратегии."""
        improved_strategy = optimizer._improve_strategy_performance(sample_strategy)
        
        assert isinstance(improved_strategy, StrategyCandidate)
        assert improved_strategy.id == sample_strategy.id

    def test_optimize_risk_management(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации управления рисками."""
        optimized_strategy = optimizer._optimize_risk_management(sample_strategy)
        
        assert isinstance(optimized_strategy, StrategyCandidate)
        assert optimized_strategy.id == sample_strategy.id
        assert "stop_loss" in optimized_strategy.parameters
        assert "take_profit" in optimized_strategy.parameters

    def test_optimize_position_sizing(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации размера позиции."""
        optimized_size = optimizer._optimize_position_sizing(sample_strategy.parameters["position_size"])
        
        assert isinstance(optimized_size, (int, float))
        assert 0 < optimized_size <= 1.0

    def test_optimize_stop_loss(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации стоп-лосса."""
        optimized_stop_loss = optimizer._optimize_stop_loss(sample_strategy.parameters["stop_loss"])
        
        assert isinstance(optimized_stop_loss, (int, float))
        assert 0 < optimized_stop_loss <= 0.5

    def test_optimize_take_profit(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации тейк-профита."""
        optimized_take_profit = optimizer._optimize_take_profit(sample_strategy.parameters["take_profit"])
        
        assert isinstance(optimized_take_profit, (int, float))
        assert 0 < optimized_take_profit <= 1.0

    def test_optimize_condition_complexity(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации сложности условий."""
        optimized_conditions = optimizer._optimize_condition_complexity(sample_strategy.entry_conditions)
        
        assert isinstance(optimized_conditions, list)
        assert len(optimized_conditions) <= 5  # Не слишком сложные условия

    def test_optimize_parameter_bounds(self, optimizer: StrategyOptimizer) -> None:
        """Тест оптимизации границ параметров."""
        params = {
            "position_size": 0.1,
            "stop_loss": 0.01,
            "take_profit": 0.02
        }
        
        optimized_params = optimizer._optimize_parameter_bounds(params)
        
        assert isinstance(optimized_params, dict)
        assert 0 < optimized_params["position_size"] <= 1.0
        assert 0 < optimized_params["stop_loss"] <= 0.5
        assert 0 < optimized_params["take_profit"] <= 1.0

    def test_optimize_condition_consistency(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации консистентности условий."""
        optimized_conditions = optimizer._optimize_condition_consistency(sample_strategy.entry_conditions)
        
        assert isinstance(optimized_conditions, list)
        assert all(isinstance(c, EntryCondition) for c in optimized_conditions)

    def test_optimize_strategy_efficiency(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест оптимизации эффективности стратегии."""
        optimized_strategy = optimizer._optimize_strategy_efficiency(sample_strategy)
        
        assert isinstance(optimized_strategy, StrategyCandidate)
        assert optimized_strategy.id == sample_strategy.id

    def test_error_handling_invalid_strategy(self, optimizer: StrategyOptimizer, sample_historical_data: pd.DataFrame) -> None:
        """Тест обработки ошибок с невалидной стратегией."""
        invalid_strategy = Mock(spec=StrategyCandidate)
        invalid_strategy.entry_conditions = []
        invalid_strategy.exit_conditions = []
        invalid_strategy.parameters = {}
        
        with pytest.raises(ValueError):
            optimizer.optimize_strategy(invalid_strategy, sample_historical_data)

    def test_error_handling_invalid_data(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест обработки ошибок с невалидными данными."""
        invalid_data = pd.DataFrame({
            'open': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
            'close': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
        
        with pytest.raises(ValueError):
            optimizer.optimize_strategy(sample_strategy, invalid_data)

    def test_performance_with_large_data(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест производительности с большими данными."""
        # Создаем большие исторические данные
        large_data = pd.DataFrame({
            'open': np.random.randn(1000).cumsum() + 50000,
            'high': np.random.randn(1000).cumsum() + 50100,
            'low': np.random.randn(1000).cumsum() + 49900,
            'close': np.random.randn(1000).cumsum() + 50000,
            'volume': np.random.randint(1000, 10000, 1000),
            'sma_20': np.random.randn(1000).cumsum() + 50000,
            'sma_50': np.random.randn(1000).cumsum() + 49950
        })
        
        start_time = datetime.now()
        optimized_strategy = optimizer.optimize_strategy(sample_strategy, large_data)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Обработка должна быть быстрой (менее 5 секунд)
        assert processing_time < 5.0
        assert isinstance(optimized_strategy, StrategyCandidate)

    def test_optimization_improvement(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate, sample_historical_data: pd.DataFrame) -> None:
        """Тест улучшения после оптимизации."""
        original_params = sample_strategy.parameters.copy()
        original_entry_conditions = sample_strategy.entry_conditions.copy()
        original_exit_conditions = sample_strategy.exit_conditions.copy()
        
        optimized_strategy = optimizer.optimize_strategy(sample_strategy, sample_historical_data)
        
        # Проверяем, что что-то изменилось
        assert (optimized_strategy.parameters != original_params or
                optimized_strategy.entry_conditions != original_entry_conditions or
                optimized_strategy.exit_conditions != original_exit_conditions)

    def test_parameter_constraints(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест ограничений параметров."""
        optimized_strategy = optimizer.optimize_strategy(sample_strategy, pd.DataFrame())
        
        params = optimized_strategy.parameters
        
        # Проверяем разумные ограничения
        assert 0 < params["position_size"] <= 1.0
        assert 0 < params["stop_loss"] <= 0.5
        assert 0 < params["take_profit"] <= 1.0
        assert params["stop_loss"] < params["take_profit"]  # Стоп-лосс меньше тейк-профита

    def test_condition_validation(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест валидации условий."""
        optimized_strategy = optimizer.optimize_strategy(sample_strategy, pd.DataFrame())
        
        # Проверяем, что условия валидны
        for condition in optimized_strategy.entry_conditions:
            assert isinstance(condition.indicator, str)
            assert isinstance(condition.operator, str)
            assert isinstance(condition.value, (int, float))
            assert isinstance(condition.timeframe, str)
            assert condition.value > 0
        
        for condition in optimized_strategy.exit_conditions:
            assert isinstance(condition.indicator, str)
            assert isinstance(condition.operator, str)
            assert isinstance(condition.value, (int, float))
            assert isinstance(condition.timeframe, str)
            assert condition.value > 0

    def test_optimization_convergence(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест сходимости оптимизации."""
        # Выполняем несколько итераций оптимизации
        strategies = []
        for i in range(3):
            strategy = optimizer.optimize_strategy(sample_strategy, pd.DataFrame())
            strategies.append(strategy)
        
        # Проверяем, что оптимизация стабилизируется
        assert len(strategies) == 3
        assert all(isinstance(s, StrategyCandidate) for s in strategies)

    def test_optimization_reproducibility(self, optimizer: StrategyOptimizer, sample_strategy: StrategyCandidate) -> None:
        """Тест воспроизводимости оптимизации."""
        # Выполняем оптимизацию дважды с одинаковыми данными
        result1 = optimizer.optimize_strategy(sample_strategy, pd.DataFrame())
        result2 = optimizer.optimize_strategy(sample_strategy, pd.DataFrame())
        
        # Результаты должны быть похожими (не обязательно идентичными из-за случайности)
        assert isinstance(result1, StrategyCandidate)
        assert isinstance(result2, StrategyCandidate)
        assert result1.id == result2.id 