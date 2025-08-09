"""
Unit тесты для domain/evolution/strategy_selection.py
"""

import pytest
from shared.numpy_utils import np
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from domain.evolution.strategy_selection import StrategySelector
from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import EvolutionContext, StrategyCandidate
from domain.type_definitions.strategy_types import StrategyType


class TestStrategySelector:
    """Тесты для StrategySelector."""

    @pytest.fixture
    def evolution_context(self):
        """Фикстура для контекста эволюции."""
        return EvolutionContext(
            name="Test Evolution",
            min_accuracy=Decimal("0.8"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.15"),
            min_sharpe=Decimal("1.0"),
        )

    @pytest.fixture
    def strategy_candidates(self):
        """Фикстура для кандидатов стратегий."""
        candidates = []
        for i in range(5):
            candidate = StrategyCandidate(
                name=f"Strategy {i}",
                strategy_type=StrategyType.TREND if i % 2 == 0 else StrategyType.MEAN_REVERSION,
            )
            candidates.append(candidate)
        return candidates

    @pytest.fixture
    def strategy_evaluations(self):
        """Фикстура для оценок стратегий."""
        evaluations = []
        for i in range(5):
            evaluation = StrategyEvaluationResult(
                accuracy=Decimal("0.85") + Decimal(str(i * 0.02)),
                profitability=Decimal("0.06") + Decimal(str(i * 0.01)),
                sharpe_ratio=Decimal("1.2") + Decimal(str(i * 0.1)),
                max_drawdown_pct=Decimal("0.12") - Decimal(str(i * 0.01)),
                total_trades=15 + i * 5,
            )
            evaluations.append(evaluation)
        return evaluations

    def test_initialization(self, evolution_context):
        """Тест инициализации селектора."""
        selector = StrategySelector(evolution_context)
        assert selector.context == evolution_context
        assert selector.selection_history == []

    def test_select_top_strategies_fitness(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест отбора по fitness score."""
        selector = StrategySelector(evolution_context)
        selected = selector.select_top_strategies(
            strategy_candidates, strategy_evaluations, n=3, selection_method="fitness"
        )
        assert len(selected) == 3
        # Проверить, что выбраны стратегии с лучшими fitness scores
        assert selected[0].name == "Strategy 4"  # Лучший fitness
        assert selected[1].name == "Strategy 3"
        assert selected[2].name == "Strategy 2"

    def test_select_top_strategies_multi_criteria(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест отбора по множественным критериям."""
        selector = StrategySelector(evolution_context)
        selected = selector.select_top_strategies(
            strategy_candidates, strategy_evaluations, n=3, selection_method="multi_criteria"
        )
        assert len(selected) <= 3
        # Проверить историю отбора
        assert len(selector.selection_history) == 1
        history_entry = selector.selection_history[0]
        assert history_entry["method"] == "multi_criteria"
        assert history_entry["total_candidates"] == 5

    def test_select_top_strategies_pareto(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест отбора по парето-фронту."""
        selector = StrategySelector(evolution_context)
        selected = selector.select_top_strategies(
            strategy_candidates, strategy_evaluations, n=3, selection_method="pareto"
        )
        assert len(selected) <= 3
        # Проверить историю отбора
        assert len(selector.selection_history) == 1
        history_entry = selector.selection_history[0]
        assert history_entry["method"] == "pareto"

    def test_select_top_strategies_tournament(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест турнирного отбора."""
        selector = StrategySelector(evolution_context)
        selected = selector.select_top_strategies(
            strategy_candidates, strategy_evaluations, n=3, selection_method="tournament"
        )
        assert len(selected) <= 3
        # Проверить историю отбора
        assert len(selector.selection_history) == 1
        history_entry = selector.selection_history[0]
        assert history_entry["method"] == "tournament"

    def test_select_top_strategies_invalid_method(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест с невалидным методом отбора."""
        selector = StrategySelector(evolution_context)
        with pytest.raises(ValueError, match="Неизвестный метод отбора"):
            selector.select_top_strategies(strategy_candidates, strategy_evaluations, n=3, selection_method="invalid")

    def test_select_top_strategies_mismatched_lengths(self, evolution_context, strategy_candidates):
        """Тест с несовпадающими длинами списков."""
        selector = StrategySelector(evolution_context)
        evaluations = [StrategyEvaluationResult() for _ in range(3)]  # Меньше чем кандидатов
        with pytest.raises(ValueError, match="Количество кандидатов и оценок должно совпадать"):
            selector.select_top_strategies(strategy_candidates, evaluations, n=3)

    def test_meets_basic_criteria_valid(self, evolution_context):
        """Тест соответствия базовым критериям - валидный случай."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        assert selector._meets_basic_criteria(evaluation) is True

    def test_meets_basic_criteria_invalid_accuracy(self, evolution_context):
        """Тест соответствия базовым критериям - невалидная точность."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.75"),  # Ниже минимума
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        assert selector._meets_basic_criteria(evaluation) is False

    def test_meets_basic_criteria_invalid_profitability(self, evolution_context):
        """Тест соответствия базовым критериям - невалидная прибыльность."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.03"),  # Ниже минимума
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        assert selector._meets_basic_criteria(evaluation) is False

    def test_meets_basic_criteria_invalid_drawdown(self, evolution_context):
        """Тест соответствия базовым критериям - невалидная просадка."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.20"),  # Выше максимума
            total_trades=15,
        )
        assert selector._meets_basic_criteria(evaluation) is False

    def test_meets_basic_criteria_invalid_sharpe(self, evolution_context):
        """Тест соответствия базовым критериям - невалидный коэффициент Шарпа."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("0.8"),  # Ниже минимума
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        assert selector._meets_basic_criteria(evaluation) is False

    def test_meets_basic_criteria_insufficient_trades(self, evolution_context):
        """Тест соответствия базовым критериям - недостаточно сделок."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=5,  # Меньше 10
        )
        assert selector._meets_basic_criteria(evaluation) is False

    def test_calculate_composite_score(self, evolution_context):
        """Тест расчета комплексного score."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        score = selector._calculate_composite_score(evaluation)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_calculate_composite_score_extreme_values(self, evolution_context):
        """Тест расчета комплексного score с экстремальными значениями."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("1.0"),  # Максимальная точность
            profitability=Decimal("0.5"),  # Высокая прибыльность
            sharpe_ratio=Decimal("3.0"),  # Максимальный коэффициент Шарпа
            max_drawdown_pct=Decimal("0.0"),  # Минимальная просадка
            total_trades=15,
        )
        score = selector._calculate_composite_score(evaluation)
        assert score > 0.8  # Должен быть высокий score

    def test_find_pareto_frontier(self, evolution_context):
        """Тест поиска парето-фронта."""
        selector = StrategySelector(evolution_context)
        candidates = [
            StrategyCandidate(name="Strategy 1"),
            StrategyCandidate(name="Strategy 2"),
        ]
        evaluations = [
            StrategyEvaluationResult(
                accuracy=Decimal("0.85"),
                profitability=Decimal("0.06"),
                sharpe_ratio=Decimal("1.2"),
                max_drawdown_pct=Decimal("0.12"),
                total_trades=15,
            ),
            StrategyEvaluationResult(
                accuracy=Decimal("0.90"),
                profitability=Decimal("0.08"),
                sharpe_ratio=Decimal("1.5"),
                max_drawdown_pct=Decimal("0.10"),
                total_trades=20,
            ),
        ]
        pairs = list(zip(candidates, evaluations))
        pareto_frontier = selector._find_pareto_frontier(pairs)
        assert len(pareto_frontier) >= 1
        # Второй кандидат должен доминировать над первым
        assert len(pareto_frontier) == 1
        assert pareto_frontier[0][0].name == "Strategy 2"

    def test_dominates_true(self, evolution_context):
        """Тест доминирования - истинный случай."""
        selector = StrategySelector(evolution_context)
        evaluation1 = StrategyEvaluationResult(
            accuracy=Decimal("0.90"),
            profitability=Decimal("0.08"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown_pct=Decimal("0.10"),
            total_trades=20,
        )
        evaluation2 = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        assert selector._dominates(evaluation1, evaluation2) is True

    def test_dominates_false(self, evolution_context):
        """Тест доминирования - ложный случай."""
        selector = StrategySelector(evolution_context)
        evaluation1 = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        evaluation2 = StrategyEvaluationResult(
            accuracy=Decimal("0.90"),
            profitability=Decimal("0.08"),
            sharpe_ratio=Decimal("1.5"),
            max_drawdown_pct=Decimal("0.10"),
            total_trades=20,
        )
        assert selector._dominates(evaluation1, evaluation2) is False

    def test_dominates_equal(self, evolution_context):
        """Тест доминирования - равные оценки."""
        selector = StrategySelector(evolution_context)
        evaluation1 = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        evaluation2 = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        assert selector._dominates(evaluation1, evaluation2) is False

    def test_filter_by_criteria(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест фильтрации по критериям."""
        selector = StrategySelector(evolution_context)
        criteria = {
            "accuracy": {"min": 0.86},
            "profitability": {"min": 0.07},
        }
        filtered = selector.filter_by_criteria(strategy_candidates, strategy_evaluations, criteria)
        assert len(filtered) < len(strategy_candidates)
        # Проверить, что отфильтрованные стратегии соответствуют критериям
        for candidate in filtered:
            idx = strategy_candidates.index(candidate)
            evaluation = strategy_evaluations[idx]
            assert float(evaluation.accuracy) >= 0.86
            assert float(evaluation.profitability) >= 0.07

    def test_filter_by_criteria_mismatched_lengths(self, evolution_context, strategy_candidates):
        """Тест фильтрации с несовпадающими длинами."""
        selector = StrategySelector(evolution_context)
        evaluations = [StrategyEvaluationResult() for _ in range(3)]
        criteria = {"accuracy": {"min": 0.8}}
        with pytest.raises(ValueError, match="Количество кандидатов и оценок должно совпадать"):
            selector.filter_by_criteria(strategy_candidates, evaluations, criteria)

    def test_meets_custom_criteria_simple(self, evolution_context):
        """Тест соответствия пользовательским критериям - простые критерии."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        criteria = {"accuracy": 0.8}
        assert selector._meets_custom_criteria(evaluation, criteria) is True

    def test_meets_custom_criteria_complex(self, evolution_context):
        """Тест соответствия пользовательским критериям - сложные критерии."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        criteria = {
            "accuracy": {"min": 0.8, "max": 0.9},
            "profitability": {"min": 0.05},
        }
        assert selector._meets_custom_criteria(evaluation, criteria) is True

    def test_meets_custom_criteria_invalid(self, evolution_context):
        """Тест соответствия пользовательским критериям - невалидные критерии."""
        selector = StrategySelector(evolution_context)
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.85"),
            profitability=Decimal("0.06"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown_pct=Decimal("0.12"),
            total_trades=15,
        )
        criteria = {"nonexistent_field": 0.8}
        assert selector._meets_custom_criteria(evaluation, criteria) is False

    def test_select_diverse_strategies_strategy_type(
        self, evolution_context, strategy_candidates, strategy_evaluations
    ):
        """Тест выбора разнообразных стратегий по типу стратегии."""
        selector = StrategySelector(evolution_context)
        selected = selector.select_diverse_strategies(
            strategy_candidates, strategy_evaluations, n=4, diversity_metric="strategy_type"
        )
        assert len(selected) <= 4
        # Проверить разнообразие типов стратегий
        strategy_types = [c.strategy_type for c in selected]
        assert len(set(strategy_types)) > 1

    @patch("sklearn.cluster.KMeans")
    def test_select_diverse_strategies_performance_cluster(
        self, mock_kmeans, evolution_context, strategy_candidates, strategy_evaluations
    ):
        """Тест выбора разнообразных стратегий по кластеризации производительности."""
        # Мокаем KMeans
        mock_kmeans_instance = Mock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1, 0, 1, 0])
        mock_kmeans.return_value = mock_kmeans_instance

        selector = StrategySelector(evolution_context)
        selected = selector.select_diverse_strategies(
            strategy_candidates, strategy_evaluations, n=3, diversity_metric="performance_cluster"
        )
        assert len(selected) <= 3

    def test_select_diverse_strategies_invalid_metric(
        self, evolution_context, strategy_candidates, strategy_evaluations
    ):
        """Тест выбора разнообразных стратегий с невалидной метрикой."""
        selector = StrategySelector(evolution_context)
        with pytest.raises(ValueError, match="Неизвестная метрика разнообразия"):
            selector.select_diverse_strategies(
                strategy_candidates, strategy_evaluations, n=3, diversity_metric="invalid"
            )

    def test_select_diverse_strategies_mismatched_lengths(self, evolution_context, strategy_candidates):
        """Тест выбора разнообразных стратегий с несовпадающими длинами."""
        selector = StrategySelector(evolution_context)
        evaluations = [StrategyEvaluationResult() for _ in range(3)]
        with pytest.raises(ValueError, match="Количество кандидатов и оценок должно совпадать"):
            selector.select_diverse_strategies(strategy_candidates, evaluations, n=3)

    def test_get_selection_statistics(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест получения статистики отбора."""
        selector = StrategySelector(evolution_context)
        stats = selector.get_selection_statistics(strategy_candidates, strategy_evaluations)

        assert "total_candidates" in stats
        assert "approved_count" in stats
        assert "approval_rate" in stats
        assert "fitness_stats" in stats
        assert "strategy_type_distribution" in stats
        assert "criteria_statistics" in stats
        assert "selection_history" in stats

        assert stats["total_candidates"] == 5
        assert isinstance(stats["approval_rate"], float)
        assert "mean" in stats["fitness_stats"]
        assert "max" in stats["fitness_stats"]
        assert "min" in stats["fitness_stats"]

    def test_get_selection_statistics_empty(self, evolution_context):
        """Тест получения статистики отбора для пустых списков."""
        selector = StrategySelector(evolution_context)
        stats = selector.get_selection_statistics([], [])
        assert stats == {}

    def test_selection_history_recording(self, evolution_context, strategy_candidates, strategy_evaluations):
        """Тест записи в историю отбора."""
        selector = StrategySelector(evolution_context)

        # Выполнить несколько отборов
        selector.select_top_strategies(strategy_candidates, strategy_evaluations, n=3, selection_method="fitness")
        selector.select_top_strategies(strategy_candidates, strategy_evaluations, n=2, selection_method="tournament")

        assert len(selector.selection_history) == 2
        assert selector.selection_history[0]["method"] == "fitness"
        assert selector.selection_history[1]["method"] == "tournament"
        assert "timestamp" in selector.selection_history[0]
        assert "total_candidates" in selector.selection_history[0]

    def test_tournament_selection_with_small_pool(self, evolution_context):
        """Тест турнирного отбора с небольшим пулом кандидатов."""
        selector = StrategySelector(evolution_context)
        candidates = [StrategyCandidate(name=f"Strategy {i}") for i in range(2)]
        evaluations = [
            StrategyEvaluationResult(
                accuracy=Decimal("0.85"),
                profitability=Decimal("0.06"),
                sharpe_ratio=Decimal("1.2"),
                max_drawdown_pct=Decimal("0.12"),
                total_trades=15,
            ),
            StrategyEvaluationResult(
                accuracy=Decimal("0.90"),
                profitability=Decimal("0.08"),
                sharpe_ratio=Decimal("1.5"),
                max_drawdown_pct=Decimal("0.10"),
                total_trades=20,
            ),
        ]
        selected = selector._select_by_tournament(candidates, evaluations, n=3, tournament_size=3)
        assert len(selected) <= 2  # Не может выбрать больше, чем есть кандидатов

    def test_pareto_frontier_with_single_candidate(self, evolution_context):
        """Тест парето-фронта с одним кандидатом."""
        selector = StrategySelector(evolution_context)
        candidates = [StrategyCandidate(name="Strategy 1")]
        evaluations = [
            StrategyEvaluationResult(
                accuracy=Decimal("0.85"),
                profitability=Decimal("0.06"),
                sharpe_ratio=Decimal("1.2"),
                max_drawdown_pct=Decimal("0.12"),
                total_trades=15,
            )
        ]
        pairs = list(zip(candidates, evaluations))
        pareto_frontier = selector._find_pareto_frontier(pairs)
        assert len(pareto_frontier) == 1
        assert pareto_frontier[0][0].name == "Strategy 1"

    def test_composite_score_edge_cases(self, evolution_context):
        """Тест комплексного score с граничными случаями."""
        selector = StrategySelector(evolution_context)

        # Тест с нулевыми значениями
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("0.0"),
            profitability=Decimal("0.0"),
            sharpe_ratio=Decimal("0.0"),
            max_drawdown_pct=Decimal("1.0"),
            total_trades=15,
        )
        score = selector._calculate_composite_score(evaluation)
        assert score >= 0.0

        # Тест с максимальными значениями
        evaluation = StrategyEvaluationResult(
            accuracy=Decimal("1.0"),
            profitability=Decimal("1.0"),
            sharpe_ratio=Decimal("3.0"),
            max_drawdown_pct=Decimal("0.0"),
            total_trades=15,
        )
        score = selector._calculate_composite_score(evaluation)
        assert score <= 1.0
