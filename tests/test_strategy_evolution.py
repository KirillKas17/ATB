"""
Тесты для модуля эволюции стратегий.
"""
from decimal import Decimal
from uuid import uuid4
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.evolution import EvolutionOrchestrator
from domain.evolution import (EntryRule, EvolutionContext, EvolutionStatus,
                              ExitRule, FilterConfig, FilterType,
                              IndicatorConfig, IndicatorType, SignalType,
                              StrategyCandidate, StrategyFitnessEvaluator,
                              StrategyGenerator, StrategySelector,
                              StrategyType)
from infrastructure.evolution import StrategyStorage
class TestStrategyModel:
    """Тесты для моделей стратегий."""
    def test_strategy_candidate_creation(self: "TestStrategyModel") -> None:
        """Тест создания кандидата стратегии."""
        candidate = StrategyCandidate(
            name="Test Strategy",
            description="Test description",
            strategy_type=StrategyType.TREND_FOLLOWING,
        )
        assert candidate.name == "Test Strategy"
        assert candidate.description == "Test description"
        assert candidate.strategy_type == StrategyType.TREND_FOLLOWING
        assert candidate.status == EvolutionStatus.GENERATED
        assert candidate.generation == 0
        assert len(candidate.indicators) == 0
        assert len(candidate.filters) == 0
    def test_strategy_candidate_with_indicators(self: "TestStrategyModel") -> None:
        """Тест кандидата стратегии с индикаторами."""
        candidate = StrategyCandidate()
        indicator = IndicatorConfig(
            name="SMA",
            indicator_type=IndicatorType.TREND,
            parameters={"period": 20},
            weight=Decimal("1.0"),
        )
        candidate.add_indicator(indicator)
        assert len(candidate.indicators) == 1
        assert candidate.indicators[0].name == "SMA"
        assert candidate.indicators[0].parameters["period"] == 20
    def test_strategy_candidate_with_filters(self: "TestStrategyModel") -> None:
        """Тест кандидата стратегии с фильтрами."""
        candidate = StrategyCandidate()
        filter_config = FilterConfig(
            name="VolatilityFilter",
            filter_type=FilterType.VOLATILITY,
            parameters={"min_atr": 0.01},
            threshold=Decimal("0.5"),
        )
        candidate.add_filter(filter_config)
        assert len(candidate.filters) == 1
        assert candidate.filters[0].name == "VolatilityFilter"
        assert candidate.filters[0].parameters["min_atr"] == 0.01
    def test_strategy_candidate_with_rules(self: "TestStrategyModel") -> None:
        """Тест кандидата стратегии с правилами."""
        candidate = StrategyCandidate()
        entry_rule = EntryRule(
            conditions=[{"indicator": "SMA", "condition": "price_above"}],
            signal_type=SignalType.BUY,
            confidence_threshold=Decimal("0.7"),
        )
        exit_rule = ExitRule(
            stop_loss_pct=Decimal("0.02"), take_profit_pct=Decimal("0.04")
        )
        candidate.add_entry_rule(entry_rule)
        candidate.add_exit_rule(exit_rule)
        assert len(candidate.entry_rules) == 1
        assert len(candidate.exit_rules) == 1
        assert candidate.entry_rules[0].signal_type == SignalType.BUY
        assert candidate.exit_rules[0].stop_loss_pct == Decimal("0.02")
    def test_strategy_candidate_serialization(self: "TestStrategyModel") -> None:
        """Тест сериализации кандидата стратегии."""
        candidate = StrategyCandidate(
            name="Test Strategy", strategy_type=StrategyType.MEAN_REVERSION
        )
        # Добавить индикатор
        indicator = IndicatorConfig(
            name="RSI", indicator_type=IndicatorType.MOMENTUM, parameters={"period": 14}
        )
        candidate.add_indicator(indicator)
        # Сериализация
        data = candidate.to_dict()
        # Десериализация
        restored = StrategyCandidate.from_dict(data)
        assert restored.name == candidate.name
        assert restored.strategy_type == candidate.strategy_type
        assert len(restored.indicators) == len(candidate.indicators)
        assert restored.indicators[0].name == candidate.indicators[0].name
class TestEvolutionContext:
    """Тесты для контекста эволюции."""
    def test_evolution_context_creation(self: "TestEvolutionContext") -> None:
        """Тест создания контекста эволюции."""
        context = EvolutionContext(
            name="Test Evolution",
            description="Test description",
            population_size=50,
            generations=100,
            mutation_rate=Decimal("0.1"),
            min_accuracy=Decimal("0.82"),
        )
        assert context.name == "Test Evolution"
        assert context.population_size == 50
        assert context.generations == 100
        assert context.mutation_rate == Decimal("0.1")
        assert context.min_accuracy == Decimal("0.82")
    def test_evolution_context_serialization(self: "TestEvolutionContext") -> None:
        """Тест сериализации контекста эволюции."""
        context = EvolutionContext(
            name="Test Context", population_size=30, generations=50
        )
        data = context.to_dict()
        restored = EvolutionContext.from_dict(data)
        assert restored.name == context.name
        assert restored.population_size == context.population_size
        assert restored.generations == context.generations
class TestStrategyGenerator:
    """Тесты для генератора стратегий."""
    def test_generator_creation(self: "TestStrategyGenerator") -> None:
        """Тест создания генератора."""
        context = EvolutionContext(population_size=50, generations=100)
        generator = StrategyGenerator(context)
        assert generator.context == context
        assert generator.generation_count == 0
    def test_random_strategy_generation(self: "TestStrategyGenerator") -> None:
        """Тест генерации случайной стратегии."""
        context = EvolutionContext()
        generator = StrategyGenerator(context)
        strategy = generator.generate_random_strategy("Test")
        assert strategy.name.startswith("Test")
        assert strategy.strategy_type in StrategyType
        assert len(strategy.indicators) > 0
        assert len(strategy.filters) > 0
        assert len(strategy.entry_rules) > 0
        assert len(strategy.exit_rules) > 0
    def test_population_generation(self: "TestStrategyGenerator") -> None:
        """Тест генерации популяции."""
        context = EvolutionContext(population_size=10)
        generator = StrategyGenerator(context)
        population = generator.generate_population()
        assert len(population) == 10
        assert all(isinstance(s, StrategyCandidate) for s in population)
    def test_crossover_strategies(self: "TestStrategyGenerator") -> None:
        """Тест скрещивания стратегий."""
        context = EvolutionContext()
        generator = StrategyGenerator(context)
        # Создать родителей
        parent1 = generator.generate_random_strategy("Parent1")
        parent2 = generator.generate_random_strategy("Parent2")
        # Скрещивание
        child = generator._crossover_strategies(parent1, parent2)
        assert child.strategy_type in [parent1.strategy_type, parent2.strategy_type]
        assert len(child.indicators) > 0
        assert len(child.parent_ids) == 2
    def test_mutation_strategy(self: "TestStrategyGenerator") -> None:
        """Тест мутации стратегии."""
        context = EvolutionContext()
        generator = StrategyGenerator(context)
        original = generator.generate_random_strategy("Original")
        mutated = generator._mutate_strategy(original)
        assert mutated.id != original.id
        assert mutated.mutation_count == 0  # Счетчик увеличивается в генераторе
class TestStrategyFitnessEvaluator:
    """Тесты для оценщика эффективности."""
    def test_evaluator_creation(self: "TestStrategyFitnessEvaluator") -> None:
        """Тест создания оценщика."""
        evaluator = StrategyFitnessEvaluator()
        assert evaluator.evaluation_results == {}
    def test_evaluate_strategy(self: "TestStrategyFitnessEvaluator") -> None:
        """Тест оценки стратегии."""
        evaluator = StrategyFitnessEvaluator()
        # Создать тестовую стратегию
        candidate = StrategyCandidate(
            name="Test Strategy", strategy_type=StrategyType.TREND_FOLLOWING
        )
        # Добавить простые правила
        entry_rule = EntryRule(
            conditions=[{"simple": True}], signal_type=SignalType.BUY
        )
        candidate.add_entry_rule(entry_rule)
        # Создать тестовые данные
        dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="1H")
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": [100 + i for i in range(len(dates))],
                "high": [101 + i for i in range(len(dates))],
                "low": [99 + i for i in range(len(dates))],
                "close": [100.5 + i for i in range(len(dates))],
                "volume": [1000000 for _ in range(len(dates))],
            }
        )
        # Оценить стратегию
        result = evaluator.evaluate_strategy(candidate, data)
        assert isinstance(result, StrategyEvaluationResult)
        assert result.strategy_id == candidate.id
        assert result.total_trades >= 0
        assert result.accuracy >= 0
        assert result.profitability >= 0
    def test_fitness_score_calculation(self: "TestStrategyFitnessEvaluator") -> None:
        """Тест расчета fitness score."""
        StrategyFitnessEvaluator()
        # Создать результат оценки
        result = StrategyEvaluationResult(
            strategy_id=uuid4(),
            accuracy=Decimal("0.8"),
            profitability=Decimal("0.1"),
            max_drawdown_pct=Decimal("0.05"),
            sharpe_ratio=Decimal("2.0"),
            total_trades=50,
        )
        fitness = result.get_fitness_score()
        assert isinstance(fitness, Decimal)
        assert fitness > 0
    def test_approval_criteria_check(self: "TestStrategyFitnessEvaluator") -> None:
        """Тест проверки критериев одобрения."""
        context = EvolutionContext(
            min_accuracy=Decimal("0.8"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.15"),
            min_sharpe=Decimal("1.0"),
        )
        # Создать результат оценки, который должен быть одобрен
        result = StrategyEvaluationResult(
            strategy_id=uuid4(),
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.1"),
            max_drawdown_pct=Decimal("0.1"),
            sharpe_ratio=Decimal("1.5"),
            total_trades=20,
        )
        is_approved = result.check_approval_criteria(context)
        assert is_approved == True
        assert result.is_approved == True
class TestStrategySelector:
    """Тесты для селектора стратегий."""
    def test_selector_creation(self: "TestStrategySelector") -> None:
        """Тест создания селектора."""
        context = EvolutionContext()
        selector = StrategySelector(context)
        assert selector.context == context
    def test_select_by_fitness(self: "TestStrategySelector") -> None:
        """Тест отбора по fitness."""
        context = EvolutionContext()
        selector = StrategySelector(context)
        # Создать тестовые кандидаты и оценки
        candidates = []
        evaluations = []
        for i in range(5):
            candidate = StrategyCandidate(name=f"Strategy_{i}")
            candidates.append(candidate)
            evaluation = StrategyEvaluationResult(
                strategy_id=candidate.id,
                accuracy=Decimal("0.8"),
                profitability=Decimal("0.1"),
                max_drawdown_pct=Decimal("0.1"),
                sharpe_ratio=Decimal("1.5"),
                total_trades=20,
            )
            evaluations.append(evaluation)
        # Отбор топ-3
        selected = selector.select_top_strategies(
            candidates, evaluations, n=3, selection_method="fitness"
        )
        assert len(selected) == 3
        assert all(isinstance(s, StrategyCandidate) for s in selected)
    def test_multi_criteria_selection(self: "TestStrategySelector") -> None:
        """Тест многокритериального отбора."""
        context = EvolutionContext()
        selector = StrategySelector(context)
        # Создать тестовые данные
        candidates = [StrategyCandidate(name=f"Strategy_{i}") for i in range(3)]
        evaluations = []
        for candidate in candidates:
            evaluation = StrategyEvaluationResult(
                strategy_id=candidate.id,
                accuracy=Decimal("0.85"),
                profitability=Decimal("0.1"),
                max_drawdown_pct=Decimal("0.1"),
                sharpe_ratio=Decimal("1.5"),
                total_trades=20,
            )
            evaluations.append(evaluation)
        selected = selector.select_top_strategies(
            candidates, evaluations, n=2, selection_method="multi_criteria"
        )
        assert len(selected) == 2
    def test_selection_statistics(self: "TestStrategySelector") -> None:
        """Тест статистики отбора."""
        context = EvolutionContext()
        selector = StrategySelector(context)
        # Создать тестовые данные
        candidates = [StrategyCandidate(name=f"Strategy_{i}") for i in range(5)]
        evaluations = []
        for candidate in candidates:
            evaluation = StrategyEvaluationResult(
                strategy_id=candidate.id,
                accuracy=Decimal("0.8"),
                profitability=Decimal("0.1"),
                max_drawdown_pct=Decimal("0.1"),
                sharpe_ratio=Decimal("1.5"),
                total_trades=20,
            )
            evaluations.append(evaluation)
        stats = selector.get_selection_statistics(candidates, evaluations)
        assert "total_candidates" in stats
        assert "approval_rate" in stats
        assert "fitness_stats" in stats
class TestStrategyStorage:
    """Тесты для хранилища стратегий."""
    def test_storage_creation(self: "TestStrategyStorage") -> None:
        """Тест создания хранилища."""
        storage = StrategyStorage(":memory:")  # In-memory database
        assert storage.engine is not None
    def test_save_and_retrieve_candidate(self: "TestStrategyStorage") -> None:
        """Тест сохранения и извлечения кандидата."""
        storage = StrategyStorage(":memory:")
        # Создать кандидата
        candidate = StrategyCandidate(
            name="Test Candidate", strategy_type=StrategyType.TREND_FOLLOWING
        )
        # Сохранить
        storage.save_strategy_candidate(candidate)
        # Извлечь
        retrieved = storage.get_strategy_candidate(candidate.id)
        assert retrieved is not None
        assert retrieved.name == candidate.name
        assert retrieved.strategy_type == candidate.strategy_type
    def test_save_and_retrieve_evaluation(self: "TestStrategyStorage") -> None:
        """Тест сохранения и извлечения оценки."""
        storage = StrategyStorage(":memory:")
        # Создать оценку
        evaluation = StrategyEvaluationResult(
            strategy_id=uuid4(),
            accuracy=Decimal("0.8"),
            profitability=Decimal("0.1"),
            total_trades=20,
        )
        # Сохранить
        storage.save_evaluation_result(evaluation)
        # Извлечь
        retrieved = storage.get_evaluation_result(evaluation.id)
        assert retrieved is not None
        assert retrieved.strategy_id == evaluation.strategy_id
        assert retrieved.accuracy == evaluation.accuracy
    def test_storage_statistics(self: "TestStrategyStorage") -> None:
        """Тест статистики хранилища."""
        storage = StrategyStorage(":memory:")
        # Добавить тестовые данные
        candidate = StrategyCandidate()
        storage.save_strategy_candidate(candidate)
        evaluation = StrategyEvaluationResult(
            strategy_id=candidate.id, is_approved=True
        )
        storage.save_evaluation_result(evaluation)
        stats = storage.get_statistics()
        assert "candidates_by_status" in stats
        assert "total_evaluations" in stats
        assert "approved_evaluations" in stats
        assert stats["total_evaluations"] == 1
        assert stats["approved_evaluations"] == 1
class TestEvolutionOrchestrator:
    """Тесты для оркестратора эволюции."""
    def test_orchestrator_creation(self: "TestEvolutionOrchestrator") -> None:
        """Тест создания оркестратора."""
        context = EvolutionContext()
        # Мок репозитория
        class MockRepository:
            async def save(self, strategy) -> Any:
                pass
        # Мок провайдера данных
        def mock_data_provider(pairs, start, end) -> Any:
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, end, freq="1H"),
                    "open": [100] * 24,
                    "high": [101] * 24,
                    "low": [99] * 24,
                    "close": [100.5] * 24,
                    "volume": [1000000] * 24,
                }
            )
        orchestrator = EvolutionOrchestrator(
            context=context,
            strategy_repository=MockRepository(),
            market_data_provider=mock_data_provider,
        )
        assert orchestrator.context == context
        assert orchestrator.current_generation == 0
    def test_single_generation(self: "TestEvolutionOrchestrator") -> None:
        """Тест одного поколения эволюции."""
        context = EvolutionContext(population_size=5, generations=1)
        class MockRepository:
            async def save(self, strategy) -> Any:
                pass
        def mock_data_provider(pairs, start, end) -> Any:
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range(start, end, freq="1H"),
                    "open": [100] * 24,
                    "high": [101] * 24,
                    "low": [99] * 24,
                    "close": [100.5] * 24,
                    "volume": [1000000] * 24,
                }
            )
        orchestrator = EvolutionOrchestrator(
            context=context,
            strategy_repository=MockRepository(),
            market_data_provider=mock_data_provider,
        )
        # Запустить одно поколение
        import asyncio
        result = asyncio.run(orchestrator.run_single_generation(["BTC/USD"]))
        assert "generation" in result
        assert "population_size" in result
        assert "approved_in_generation" in result
if __name__ == "__main__":
    pytest.main([__file__])
