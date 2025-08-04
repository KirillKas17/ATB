"""
Селектор стратегий для фильтрации по критериям качества.
"""

from datetime import datetime
from typing import Any, Dict, List

from shared.numpy_utils import np

from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import EvolutionContext, StrategyCandidate


class StrategySelector:
    """Селектор стратегий."""

    def __init__(self, context: EvolutionContext):
        self.context = context
        self.selection_history: List[Dict[str, Any]] = []

    def select_top_strategies(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int = 10,
        selection_method: str = "fitness",
    ) -> List[StrategyCandidate]:
        """Выбрать топ стратегий."""
        if len(candidates) != len(evaluations):
            raise ValueError("Количество кандидатов и оценок должно совпадать")
        if selection_method == "fitness":
            return self._select_by_fitness(candidates, evaluations, n)
        elif selection_method == "multi_criteria":
            return self._select_by_multi_criteria(candidates, evaluations, n)
        elif selection_method == "pareto":
            return self._select_by_pareto_frontier(candidates, evaluations, n)
        elif selection_method == "tournament":
            return self._select_by_tournament(candidates, evaluations, n)
        else:
            raise ValueError(f"Неизвестный метод отбора: {selection_method}")

    def _select_by_fitness(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int,
    ) -> List[StrategyCandidate]:
        """Отбор по fitness score."""
        # Создать пары (кандидат, оценка)
        pairs = list(zip(candidates, evaluations))
        # Отсортировать по fitness score
        sorted_pairs = sorted(
            pairs, key=lambda x: x[1].get_fitness_score(), reverse=True
        )
        # Выбрать топ n
        selected_candidates: List[StrategyCandidate] = [
            pair[0] for pair in sorted_pairs[:n]
        ]
        # Записать в историю
        self.selection_history.append(
            {
                "timestamp": datetime.now(),
                "method": "fitness",
                "total_candidates": len(candidates),
                "selected_count": len(selected_candidates),
                "best_fitness": (
                    float(sorted_pairs[0][1].get_fitness_score())
                    if sorted_pairs
                    else 0.0
                ),
            }
        )
        return selected_candidates

    def _select_by_multi_criteria(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int,
    ) -> List[StrategyCandidate]:
        """Отбор по множественным критериям."""
        # Создать пары (кандидат, оценка)
        pairs = list(zip(candidates, evaluations))
        # Фильтровать по базовым критериям
        filtered_pairs = []
        for candidate, evaluation in pairs:
            if self._meets_basic_criteria(evaluation):
                filtered_pairs.append((candidate, evaluation))
        if not filtered_pairs:
            return []
        # Рассчитать комплексный score
        scored_pairs = []
        for candidate, evaluation in filtered_pairs:
            score = self._calculate_composite_score(evaluation)
            scored_pairs.append((candidate, evaluation, score))
        # Отсортировать по комплексному score
        sorted_pairs = sorted(scored_pairs, key=lambda x: x[2], reverse=True)
        # Выбрать топ n
        selected_candidates = [pair[0] for pair in sorted_pairs[:n]]
        # Записать в историю
        self.selection_history.append(
            {
                "timestamp": datetime.now(),
                "method": "multi_criteria",
                "total_candidates": len(candidates),
                "filtered_count": len(filtered_pairs),
                "selected_count": len(selected_candidates),
                "best_score": float(sorted_pairs[0][2]) if sorted_pairs else 0.0,
            }
        )
        return selected_candidates

    def _select_by_pareto_frontier(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int,
    ) -> List[StrategyCandidate]:
        """Отбор по парето-фронту."""
        # Создать пары (кандидат, оценка)
        pairs = list(zip(candidates, evaluations))
        # Фильтровать по базовым критериям
        filtered_pairs = []
        for candidate, evaluation in pairs:
            if self._meets_basic_criteria(evaluation):
                filtered_pairs.append((candidate, evaluation))
        if not filtered_pairs:
            return []
        # Найти парето-оптимальные решения
        pareto_frontier = self._find_pareto_frontier(filtered_pairs)
        # Если парето-фронт меньше n, добавить остальные по fitness
        if len(pareto_frontier) < n:
            remaining_pairs = [
                (c, e) for c, e in filtered_pairs if (c, e) not in pareto_frontier
            ]
            remaining_sorted = sorted(
                remaining_pairs, key=lambda x: x[1].get_fitness_score(), reverse=True
            )
            pareto_frontier.extend(remaining_sorted[: n - len(pareto_frontier)])
        # Выбрать топ n из парето-фронта
        selected_candidates = [pair[0] for pair in pareto_frontier[:n]]
        # Записать в историю
        self.selection_history.append(
            {
                "timestamp": datetime.now(),
                "method": "pareto",
                "total_candidates": len(candidates),
                "filtered_count": len(filtered_pairs),
                "pareto_size": len(self._find_pareto_frontier(filtered_pairs)),
                "selected_count": len(selected_candidates),
            }
        )
        return selected_candidates

    def _select_by_tournament(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int,
        tournament_size: int = 3,
    ) -> List[StrategyCandidate]:
        """Турнирный отбор."""
        selected_candidates: List[StrategyCandidate] = []
        pairs = list(zip(candidates, evaluations))
        while len(selected_candidates) < n and pairs:
            # Провести турнир
            tournament_pairs = np.random.choice(
                len(pairs), min(tournament_size, len(pairs)), replace=False
            )
            tournament_pairs = np.array([pairs[i] for i in tournament_pairs])
            # Выбрать победителя турнира
            winner = max(tournament_pairs, key=lambda x: x[1].get_fitness_score())
            selected_candidates.append(winner[0])
            # Удалить победителя из пула
            pairs.remove(winner)
        # Записать в историю
        self.selection_history.append(
            {
                "timestamp": datetime.now(),
                "method": "tournament",
                "total_candidates": len(candidates),
                "selected_count": len(selected_candidates),
                "tournament_size": tournament_size,
            }
        )
        return selected_candidates

    def _meets_basic_criteria(self, evaluation: StrategyEvaluationResult) -> bool:
        """Проверить соответствие базовым критериям."""
        return (
            evaluation.accuracy >= self.context.min_accuracy
            and evaluation.profitability >= self.context.min_profitability
            and evaluation.max_drawdown_pct <= self.context.max_drawdown
            and evaluation.sharpe_ratio >= self.context.min_sharpe
            and evaluation.total_trades
            >= 10  # Минимум сделок для статистической значимости
        )

    def _calculate_composite_score(self, evaluation: StrategyEvaluationResult) -> float:
        """Рассчитать комплексный score."""
        # Нормализованные компоненты (0-1)
        accuracy_score = min(float(evaluation.accuracy), 1.0)
        profitability_score = min(max(float(evaluation.profitability), 0.0), 1.0)
        sharpe_score = min(max(float(evaluation.sharpe_ratio), 0.0), 3.0) / 3.0
        drawdown_score = max(0.0, 1.0 - float(evaluation.max_drawdown_pct))
        # Взвешенная сумма
        composite_score = (
            accuracy_score * 0.3
            + profitability_score * 0.3
            + sharpe_score * 0.2
            + drawdown_score * 0.2
        )
        return composite_score

    def _find_pareto_frontier(
        self, pairs: List[tuple[StrategyCandidate, StrategyEvaluationResult]]
    ) -> List[tuple[StrategyCandidate, StrategyEvaluationResult]]:
        """Найти парето-оптимальные решения."""
        pareto_frontier = []
        for i, (candidate1, evaluation1) in enumerate(pairs):
            is_dominated = False
            for j, (candidate2, evaluation2) in enumerate(pairs):
                if i != j and self._dominates(evaluation2, evaluation1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_frontier.append((candidate1, evaluation1))
        return pareto_frontier

    def _dominates(
        self,
        evaluation1: StrategyEvaluationResult,
        evaluation2: StrategyEvaluationResult,
    ) -> bool:
        """Проверить, доминирует ли evaluation1 над evaluation2."""
        # evaluation1 доминирует, если она лучше или равна по всем критериям
        # и строго лучше хотя бы по одному
        better_or_equal = (
            evaluation1.accuracy >= evaluation2.accuracy
            and evaluation1.profitability >= evaluation2.profitability
            and evaluation1.sharpe_ratio >= evaluation2.sharpe_ratio
            and evaluation1.max_drawdown_pct <= evaluation2.max_drawdown_pct
        )
        strictly_better = (
            evaluation1.accuracy > evaluation2.accuracy
            or evaluation1.profitability > evaluation2.profitability
            or evaluation1.sharpe_ratio > evaluation2.sharpe_ratio
            or evaluation1.max_drawdown_pct < evaluation2.max_drawdown_pct
        )
        return better_or_equal and strictly_better

    def filter_by_criteria(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        criteria: Dict[str, Any],
    ) -> List[StrategyCandidate]:
        """Фильтровать стратегии по заданным критериям."""
        if len(candidates) != len(evaluations):
            raise ValueError("Количество кандидатов и оценок должно совпадать")
        filtered_candidates = []
        filtered_evaluations = []
        for candidate, evaluation in zip(candidates, evaluations):
            if self._meets_custom_criteria(evaluation, criteria):
                filtered_candidates.append(candidate)
                filtered_evaluations.append(evaluation)
        return filtered_candidates

    def _meets_custom_criteria(
        self, evaluation: StrategyEvaluationResult, criteria: Dict[str, Any]
    ) -> bool:
        """Проверить соответствие пользовательским критериям."""
        for criterion, value in criteria.items():
            if hasattr(evaluation, criterion):
                actual_value = getattr(evaluation, criterion)
                if isinstance(value, dict):
                    # Сложные критерии с операторами
                    if "min" in value and actual_value < value["min"]:
                        return False
                    if "max" in value and actual_value > value["max"]:
                        return False
                    if "equals" in value and actual_value != value["equals"]:
                        return False
                else:
                    # Простые критерии (минимум)
                    if actual_value < value:
                        return False
            else:
                # Неизвестный критерий
                return False
        return True

    def select_diverse_strategies(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int = 10,
        diversity_metric: str = "strategy_type",
    ) -> List[StrategyCandidate]:
        """Выбрать разнообразные стратегии."""
        if len(candidates) != len(evaluations):
            raise ValueError("Количество кандидатов и оценок должно совпадать")
        if diversity_metric == "strategy_type":
            return self._select_by_strategy_type_diversity(candidates, evaluations, n)
        elif diversity_metric == "performance_cluster":
            return self._select_by_performance_clustering(candidates, evaluations, n)
        else:
            raise ValueError(f"Неизвестная метрика разнообразия: {diversity_metric}")

    def _select_by_strategy_type_diversity(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int,
    ) -> List[StrategyCandidate]:
        """Выбрать стратегии с разнообразием типов."""
        # Группировать по типу стратегии
        strategy_types: Dict[str, List[tuple[StrategyCandidate, StrategyEvaluationResult]]] = {}
        for candidate, evaluation in zip(candidates, evaluations):
            strategy_type = candidate.strategy_type.value
            if strategy_type not in strategy_types:
                strategy_types[strategy_type] = []
            strategy_types[strategy_type].append((candidate, evaluation))
        # Выбрать лучших из каждой группы
        selected_candidates = []
        candidates_per_group = max(1, n // len(strategy_types))
        for strategy_type, group in strategy_types.items():
            # Отсортировать группу по fitness
            sorted_group = sorted(
                group, key=lambda x: x[1].get_fitness_score(), reverse=True
            )
            # Выбрать лучших из группы
            selected_from_group = sorted_group[:candidates_per_group]
            selected_candidates.extend([pair[0] for pair in selected_from_group])
        # Если выбрано меньше n, добавить остальные по fitness
        if len(selected_candidates) < n:
            remaining_pairs = [
                (c, e)
                for c, e in zip(candidates, evaluations)
                if c not in selected_candidates
            ]
            remaining_sorted = sorted(
                remaining_pairs, key=lambda x: x[1].get_fitness_score(), reverse=True
            )
            additional_candidates = [
                pair[0] for pair in remaining_sorted[: n - len(selected_candidates)]
            ]
            selected_candidates.extend(additional_candidates)
        return selected_candidates[:n]

    def _select_by_performance_clustering(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
        n: int,
    ) -> List[StrategyCandidate]:
        """Выбрать стратегии с кластеризацией по производительности."""
        # Создать матрицу признаков для кластеризации
        features = []
        for evaluation in evaluations:
            feature_vector = [
                float(evaluation.accuracy),
                float(evaluation.profitability),
                float(evaluation.sharpe_ratio),
                float(evaluation.max_drawdown_pct),
                float(evaluation.total_trades),
            ]
            features.append(feature_vector)
        features_array = np.array(features)
        # Нормализовать признаки
        features_normalized = (
            features_array - features_array.mean(axis=0)
        ) / features_array.std(axis=0)
        # Простая кластеризация по k-means
        from sklearn.cluster import KMeans

        n_clusters = min(n, len(candidates))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_normalized)
        # Выбрать лучших из каждого кластера
        selected_candidates = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Найти лучшего в кластере
                best_in_cluster = max(
                    cluster_indices, key=lambda i: evaluations[i].get_fitness_score()
                )
                selected_candidates.append(candidates[best_in_cluster])
        return selected_candidates[:n]

    def get_selection_statistics(
        self,
        candidates: List[StrategyCandidate],
        evaluations: List[StrategyEvaluationResult],
    ) -> Dict[str, Any]:
        """Получить статистику отбора."""
        if not candidates or not evaluations:
            return {}
        # Базовые метрики
        total_candidates = len(candidates)
        approved_count = sum(1 for e in evaluations if e.is_approved)
        avg_fitness = np.mean([float(e.get_fitness_score()) for e in evaluations])
        max_fitness = max([float(e.get_fitness_score()) for e in evaluations])
        min_fitness = min([float(e.get_fitness_score()) for e in evaluations])
        # Распределение по типам стратегий
        strategy_types: Dict[str, int] = {}
        for candidate in candidates:
            strategy_type = candidate.strategy_type.value
            strategy_types[strategy_type] = strategy_types.get(strategy_type, 0) + 1
        # Статистика по критериям
        criteria_stats = {
            "accuracy": {
                "mean": np.mean([float(e.accuracy) for e in evaluations]),
                "std": np.std([float(e.accuracy) for e in evaluations]),
                "min": min([float(e.accuracy) for e in evaluations]),
                "max": max([float(e.accuracy) for e in evaluations]),
            },
            "profitability": {
                "mean": np.mean([float(e.profitability) for e in evaluations]),
                "std": np.std([float(e.profitability) for e in evaluations]),
                "min": min([float(e.profitability) for e in evaluations]),
                "max": max([float(e.profitability) for e in evaluations]),
            },
            "sharpe_ratio": {
                "mean": np.mean([float(e.sharpe_ratio) for e in evaluations]),
                "std": np.std([float(e.sharpe_ratio) for e in evaluations]),
                "min": min([float(e.sharpe_ratio) for e in evaluations]),
                "max": max([float(e.sharpe_ratio) for e in evaluations]),
            },
            "max_drawdown": {
                "mean": np.mean([float(e.max_drawdown_pct) for e in evaluations]),
                "std": np.std([float(e.max_drawdown_pct) for e in evaluations]),
                "min": min([float(e.max_drawdown_pct) for e in evaluations]),
                "max": max([float(e.max_drawdown_pct) for e in evaluations]),
            },
        }
        return {
            "total_candidates": total_candidates,
            "approved_count": approved_count,
            "approval_rate": (
                approved_count / total_candidates if total_candidates > 0 else 0.0
            ),
            "fitness_stats": {
                "mean": avg_fitness,
                "max": max_fitness,
                "min": min_fitness,
                "std": np.std([float(e.get_fitness_score()) for e in evaluations]),
            },
            "strategy_type_distribution": strategy_types,
            "criteria_statistics": criteria_stats,
            "selection_history": self.selection_history,
        }
