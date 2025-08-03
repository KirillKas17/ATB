"""
Главный управляющий модуль эволюции стратегий.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

import pandas as pd

from domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from domain.evolution.strategy_fitness import (
    StrategyEvaluationResult,
    StrategyFitnessEvaluator,
)
from domain.evolution.strategy_generator import StrategyGenerator
from domain.evolution.strategy_model import (
    EvolutionContext,
    EvolutionStatus,
    StrategyCandidate,
)
from domain.evolution.strategy_optimizer import StrategyOptimizer
from domain.evolution.strategy_selection import StrategySelector
from domain.protocols.repository_protocol import StrategyRepositoryProtocol
from domain.value_objects.timestamp import Timestamp

# Импорты infrastructure/evolution
from infrastructure.evolution import (
    EvolutionBackup,
    EvolutionCache,
    EvolutionMigration,
    StrategyStorage,
)


class EvolutionOrchestrator:
    """Оркестратор эволюции стратегий."""

    def __init__(
        self,
        context: EvolutionContext,
        strategy_repository: StrategyRepositoryProtocol,
        market_data_provider: Callable[[str, datetime, datetime], pd.DataFrame],
        on_strategy_approved: Optional[
            Callable[[StrategyCandidate, StrategyEvaluationResult], None]
        ] = None,
        strategy_storage: Optional[StrategyStorage] = None,
        evolution_cache: Optional[EvolutionCache] = None,
        evolution_backup: Optional[EvolutionBackup] = None,
        evolution_migration: Optional[EvolutionMigration] = None,
    ):
        self.context = context
        self.strategy_repository = strategy_repository
        self.market_data_provider = market_data_provider
        self.on_strategy_approved = on_strategy_approved
        # Инициализация компонентов infrastructure/evolution
        self.strategy_storage = strategy_storage or StrategyStorage()
        self.evolution_cache = evolution_cache or EvolutionCache()
        self.evolution_backup = evolution_backup or EvolutionBackup()
        self.evolution_migration = evolution_migration or EvolutionMigration(storage=self.strategy_storage)
        # Инициализация компонентов
        self.fitness_evaluator = StrategyFitnessEvaluator()
        self.strategy_generator = StrategyGenerator(context)
        self.strategy_optimizer = StrategyOptimizer(context, self.fitness_evaluator)
        self.strategy_selector = StrategySelector(context)
        # Состояние эволюции
        self.current_generation = 0
        self.population: List[StrategyCandidate] = []
        self.evaluations: Dict[UUID, StrategyEvaluationResult] = {}
        self.approved_strategies: List[StrategyCandidate] = []
        self.evolution_history: List[Dict[str, Any]] = []
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        # Статистика
        self.stats = {
            "total_generations": 0,
            "total_candidates_generated": 0,
            "total_candidates_evaluated": 0,
            "total_strategies_approved": 0,
            "best_fitness_achieved": 0.0,
            "start_time": datetime.now(),
        }

    async def start_evolution(self, trading_pairs: Optional[List[str]] = None) -> None:
        """Запустить процесс эволюции."""
        self.logger.info("Запуск эволюции стратегий")
        if trading_pairs is None:
            trading_pairs = ["BTC/USD", "ETH/USD", "ADA/USD"]
        try:
            # Инициализация первой популяции
            await self._initialize_population()
            # Основной цикл эволюции
            for generation in range(self.context.generations):
                self.current_generation = generation
                self.logger.info(
                    f"Поколение {generation + 1}/{self.context.generations}"
                )
                # Получить исторические данные
                historical_data = await self._get_historical_data(trading_pairs)
                # Оценить текущую популяцию
                await self._evaluate_population(historical_data)
                # Выбрать лучших
                selected_candidates = self._select_best_candidates()
                # Создать новое поколение
                await self._create_next_generation(selected_candidates)
                # Обновить статистику
                self._update_statistics()
                # Проверить условия остановки
                if self._should_stop_evolution():
                    self.logger.info("Условия остановки эволюции выполнены")
                    break
                # Пауза между поколениями
                await asyncio.sleep(1)
            # Финальная обработка
            await self._finalize_evolution()
        except Exception as e:
            self.logger.error(f"Ошибка в процессе эволюции: {e}")
            raise

    async def _initialize_population(self) -> None:
        """Инициализировать начальную популяцию."""
        self.logger.info("Инициализация начальной популяции")
        # Генерировать случайные стратегии
        self.population = self.strategy_generator.generate_population(
            self.context.population_size
        )
        self.stats["total_candidates_generated"] = int(str(self.stats.get("total_candidates_generated", 0))) + len(self.population)
        self.logger.info(f"Создано {len(self.population)} стратегий")

    async def _get_historical_data(self, trading_pairs: Optional[List[str]] = None) -> pd.DataFrame:
        """Получить исторические данные."""
        if trading_pairs is None:
            trading_pairs = ["BTC/USD", "ETH/USD", "ADA/USD"]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)  # Год данных
        all_data = []
        for pair in trading_pairs:
            try:
                data = self.market_data_provider(pair, start_date, end_date)
                if isinstance(data, pd.DataFrame) and not data.empty:
                    data["trading_pair"] = pair
                    all_data.append(data)
            except Exception as e:
                self.logger.warning(f"Не удалось получить данные для {pair}: {e}")
        if not all_data:
            raise ValueError("Не удалось получить исторические данные")
        # Объединить данные
        combined_data = pd.concat(all_data, ignore_index=True)
        if not combined_data.empty:
            combined_data = combined_data.sort_values("timestamp").reset_index(drop=True)
        return combined_data

    async def _evaluate_population(self, historical_data: pd.DataFrame) -> None:
        """Оценить популяцию стратегий."""
        self.logger.info(f"Оценка {len(self.population)} стратегий")
        evaluations = {}
        for i, candidate in enumerate(self.population):
            try:
                self.logger.debug(
                    f"Оценка стратегии {i+1}/{len(self.population)}: {candidate.name}"
                )
                # Оценить стратегию
                evaluation = self.fitness_evaluator.evaluate_strategy(
                    candidate, historical_data
                )
                evaluations[candidate.id] = evaluation
                # Проверить критерии одобрения
                if evaluation.check_approval_criteria(self.context):
                    candidate.update_status(EvolutionStatus.APPROVED)
                    self.approved_strategies.append(candidate)
                    # Уведомить о одобрении
                    if self.on_strategy_approved:
                        await self._notify_strategy_approved(candidate, evaluation)
                    self.logger.info(
                        f"Стратегия {candidate.name} одобрена (fitness: {evaluation.get_fitness_score()})"
                    )
                else:
                    candidate.update_status(EvolutionStatus.REJECTED)
                    self.logger.debug(
                        f"Стратегия {candidate.name} отклонена: {evaluation.approval_reason}"
                    )
            except Exception as e:
                self.logger.error(f"Ошибка оценки стратегии {candidate.name}: {e}")
                # Исправление: используем существующий статус вместо ERROR
                candidate.update_status(EvolutionStatus.REJECTED)
        self.evaluations.update(evaluations)
        self.stats["total_candidates_evaluated"] = int(str(self.stats.get("total_candidates_evaluated", 0))) + len(self.population)

    def _select_best_candidates(self) -> List[StrategyCandidate]:
        """Выбрать лучших кандидатов."""
        evaluations_list = [
            self.evaluations[c.id] for c in self.population if c.id in self.evaluations
        ]
        if not evaluations_list:
            return []
        # Выбрать топ стратегий
        selected_count = max(1, self.context.population_size // 2)
        selected_candidates = self.strategy_selector.select_top_strategies(
            self.population, evaluations_list, selected_count, "multi_criteria"
        )
        self.logger.info(f"Выбрано {len(selected_candidates)} лучших стратегий")
        return selected_candidates

    async def _create_next_generation(
        self, selected_candidates: List[StrategyCandidate]
    ) -> None:
        """Создать следующее поколение."""
        if not selected_candidates:
            self.logger.warning("Нет кандидатов для создания нового поколения")
            return
        self.logger.info("Создание нового поколения")
        # Элитизм - сохранить лучших
        elite_size = min(self.context.elite_size, len(selected_candidates))
        elite_candidates = selected_candidates[:elite_size]
        # Создать потомков
        children_count = self.context.population_size - elite_size
        children = self.strategy_generator.generate_from_parents(
            selected_candidates, children_count
        )
        # Оптимизировать потомков
        if children:
            historical_data = await self._get_historical_data(
                ["BTC/USD"]
            )  # Упрощенно для оптимизации
            optimized_children = self.strategy_optimizer.optimize_population(
                children, historical_data, "genetic", max_iterations=20
            )
        else:
            optimized_children = []
        # Сформировать новую популяцию
        new_population = elite_candidates + optimized_children
        # Добавить случайные стратегии для разнообразия
        if len(new_population) < self.context.population_size:
            random_count = self.context.population_size - len(new_population)
            random_strategies = self.strategy_generator.generate_population(
                random_count
            )
            new_population.extend(random_strategies)
        # Обновить популяцию
        self.population = new_population[: self.context.population_size]
        self.strategy_generator.generation_count += 1
        # Исправление: добавляем проверки типов для операций с object
        current_generated = self.stats.get("total_candidates_generated", 0)
        if isinstance(current_generated, (int, float)):
            self.stats["total_candidates_generated"] = int(current_generated) + len(children) + len(random_strategies)
        else:
            self.stats["total_candidates_generated"] = len(children) + len(random_strategies)
        self.logger.info(f"Создано новое поколение: {len(self.population)} стратегий")

    def _update_statistics(self) -> None:
        """Обновить статистику эволюции."""
        self.stats["total_generations"] = int(str(self.stats.get("total_generations", 0))) + 1
        
        # Найти лучший fitness в текущем поколении
        if self.evaluations:
            best_fitness = max(
                evaluation.get_fitness_score()
                for evaluation in self.evaluations.values()
                if evaluation is not None
            )
            current_best = self.stats.get("best_fitness_achieved", 0.0)
            if isinstance(current_best, (int, float)) and isinstance(best_fitness, (int, float)):
                self.stats["best_fitness_achieved"] = max(float(current_best), float(best_fitness))
        
        # Обновить количество одобренных стратегий
        self.stats["total_strategies_approved"] = self.stats.get("total_strategies_approved", 0) + 1  # type: ignore[operator]
        
        # Добавить в историю
        generation_stats = {
            "generation": self.current_generation,
            "population_size": len(self.population),
            "approved_count": len(self.approved_strategies),
            "best_fitness": self.stats.get("best_fitness_achieved", 0.0),
            "timestamp": datetime.now(),
        }
        self.evolution_history.append(generation_stats)

    def _should_stop_evolution(self) -> bool:
        """Проверить условия остановки эволюции."""
        # Остановка по количеству одобренных стратегий
        if len(self.approved_strategies) >= 10:
            self.logger.info("Достигнуто достаточное количество одобренных стратегий")
            return True
        # Остановка по отсутствию улучшений
        if len(self.evolution_history) >= 10:
            recent_generations = self.evolution_history[-10:]
            recent_fitnesses = [g["best_fitness"] for g in recent_generations]
            if max(recent_fitnesses) - min(recent_fitnesses) < 0.01:
                self.logger.info("Нет улучшений в последних 10 поколениях")
                return True
        return False

    async def _finalize_evolution(self) -> None:
        """Завершить эволюцию."""
        self.logger.info("Завершение эволюции")
        # Сохранить одобренные стратегии
        await self._save_approved_strategies()
        # Создать финальный отчет
        final_report = self._create_final_report()
        self.logger.info(
            f"Эволюция завершена. Одобрено стратегий: {len(self.approved_strategies)}"
        )
        self.logger.info(f"Лучший fitness: {self.stats['best_fitness_achieved']}")

    async def _save_approved_strategies(self) -> None:
        """Сохранить одобренные стратегии."""
        for candidate in self.approved_strategies:
            try:
                # Преобразовать в доменную стратегию
                strategy = self._convert_to_domain_strategy(candidate)
                # Сохранить в репозитории
                await self.strategy_repository.save(strategy)
                current_count = self.stats.get("total_strategies_approved", 0)
                self.stats["total_strategies_approved"] = current_count + 1
            except Exception as e:
                self.logger.error(f"Ошибка сохранения стратегии {candidate.name}: {e}")

    def _convert_to_domain_strategy(self, candidate: StrategyCandidate) -> Strategy:
        """Преобразовать кандидата в доменную стратегию."""
        # Приведение типа strategy_type к domain.entities.strategy.StrategyType
        strategy = Strategy(
            id=candidate.id,
            name=candidate.name,
            description=candidate.description,
            strategy_type=StrategyType(candidate.strategy_type.value) if hasattr(candidate.strategy_type, 'value') else StrategyType(candidate.strategy_type),
            status=StrategyStatus.ACTIVE,
            trading_pairs=getattr(candidate, 'trading_pairs', []),
            is_active=True,
            created_at=candidate.created_at,
            updated_at=datetime.now(),
        )
        # Установить параметры
        strategy.parameters.parameters = {
            "position_size_pct": str(candidate.position_size_pct),
            "max_positions": candidate.max_positions,
            "min_holding_time": candidate.min_holding_time,
            "max_holding_time": candidate.max_holding_time,
            "indicators": [str(ind.to_dict()) for ind in candidate.indicators],
            "filters": [str(filt.to_dict()) for filt in candidate.filters],
            "entry_rules": [str(rule.to_dict()) for rule in candidate.entry_rules],
            "exit_rules": [str(rule.to_dict()) for rule in candidate.exit_rules],
        }
        # Установить метаданные
        strategy.metadata = {
            "evolution_generation": candidate.generation,
            "parent_ids": [str(pid) for pid in getattr(candidate, 'parent_ids', [])],
            "mutation_count": candidate.mutation_count,
            "evolution_context": self.context.to_dict(),
        }
        return strategy

    async def _notify_strategy_approved(
        self, candidate: StrategyCandidate, evaluation: StrategyEvaluationResult
    ) -> None:
        """Уведомить об одобрении стратегии."""
        if self.on_strategy_approved and callable(self.on_strategy_approved):
            try:
                if asyncio.iscoroutinefunction(self.on_strategy_approved):
                    await self.on_strategy_approved(candidate, evaluation)
                else:
                    self.on_strategy_approved(candidate, evaluation)
            except Exception as e:
                self.logger.error(f"Ошибка уведомления об одобрении: {e}")

    def _create_final_report(self) -> Dict[str, Any]:
        """Создать финальный отчет."""
        return {
            "evolution_summary": {
                "total_generations": self.stats["total_generations"],
                "total_candidates_generated": self.stats["total_candidates_generated"],
                "total_candidates_evaluated": self.stats["total_candidates_evaluated"],
                "total_strategies_approved": self.stats["total_strategies_approved"],
                "best_fitness_achieved": self.stats["best_fitness_achieved"],
                "start_time": self.stats["start_time"].isoformat() if isinstance(self.stats["start_time"], datetime) else str(self.stats["start_time"]),
                "end_time": datetime.now().isoformat(),
                "duration_hours": (
                    (datetime.now() - self.stats["start_time"]).total_seconds() / 3600
                    if isinstance(self.stats["start_time"], datetime) else 0.0
                ),
            },
            "approved_strategies": [
                {
                    "id": str(s.id),
                    "name": s.name,
                    "strategy_type": s.strategy_type.value if hasattr(s.strategy_type, 'value') else str(s.strategy_type),
                    "generation": s.generation,
                    "fitness_score": (
                        float(self.evaluations[s.id].get_fitness_score())
                        if s.id in self.evaluations
                        else 0.0
                    ),
                }
                for s in self.approved_strategies
            ],
            "evolution_history": self.evolution_history,
            "context": self.context.to_dict(),
        }

    def get_current_status(self) -> Dict[str, Any]:
        """Получить текущий статус эволюции."""
        return {
            "current_generation": self.current_generation,
            "population_size": len(self.population),
            "approved_strategies_count": len(self.approved_strategies),
            "best_fitness": self.stats["best_fitness_achieved"],
            "evolution_stats": self.stats,
            "context": self.context.to_dict(),
        }

    def get_approved_strategies(self) -> List[StrategyCandidate]:
        """Получить одобренные стратегии."""
        return self.approved_strategies.copy()

    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Получить историю эволюции."""
        return self.evolution_history.copy()

    async def pause_evolution(self) -> None:
        """Приостановить эволюцию."""
        self.logger.info("Эволюция приостановлена")
        # Здесь можно добавить логику приостановки

    async def resume_evolution(self) -> None:
        """Возобновить эволюцию."""
        self.logger.info("Эволюция возобновлена")
        # Здесь можно добавить логику возобновления

    async def stop_evolution(self) -> None:
        """Остановить эволюцию."""
        self.logger.info("Эволюция остановлена")
        # Здесь можно добавить логику остановки

    def get_selection_statistics(self) -> Dict[str, Any]:
        """Получить статистику отбора."""
        evaluations_list = [
            self.evaluations[c.id] for c in self.population if c.id in self.evaluations
        ]
        return self.strategy_selector.get_selection_statistics(
            self.population, evaluations_list
        )

    async def run_single_generation(
        self, trading_pairs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Запустить одно поколение эволюции."""
        if trading_pairs is None:
            trading_pairs = ["BTC/USD", "ETH/USD", "ADA/USD"]
        try:
            # Получить исторические данные
            historical_data = await self._get_historical_data(trading_pairs)
            # Оценить текущую популяцию
            await self._evaluate_population(historical_data)
            # Выбрать лучших
            selected_candidates = self._select_best_candidates()
            # Создать новое поколение
            await self._create_next_generation(selected_candidates)
            # Обновить статистику
            self._update_statistics()
            return {
                "generation": self.current_generation,
                "population_size": len(self.population),
                "approved_count": len(self.approved_strategies),
                "best_fitness": self.stats["best_fitness_achieved"],
            }
        except Exception as e:
            self.logger.error(f"Ошибка в одном поколении: {e}")
            raise

    # Методы интеграции с infrastructure/evolution
    async def save_strategy_to_storage(self, candidate: StrategyCandidate) -> bool:
        """Сохранить стратегию в хранилище."""
        try:
            result = await self.strategy_storage.save_strategy_candidate(candidate)  # type: ignore[func-returns-value]
            self.logger.info(f"Стратегия {candidate.name} сохранена в хранилище")
            return result is not None
        except Exception as e:
            self.logger.error(f"Ошибка сохранения стратегии {candidate.name}: {e}")
            return False

    async def load_strategies_from_storage(self) -> List[StrategyCandidate]:
        """Загрузить стратегии из хранилища."""
        try:
            candidates = await self.strategy_storage.get_strategy_candidates()
            self.logger.info(f"Загружено {len(candidates)} стратегий из хранилища")
            return candidates if candidates is not None else []
        except Exception as e:
            self.logger.error(f"Ошибка загрузки стратегий: {e}")
            return []

    async def cache_strategy_evaluation(
        self, candidate_id: UUID, evaluation: StrategyEvaluationResult
    ) -> bool:
        """Кэшировать оценку стратегии."""
        try:
            result = await self.evolution_cache.set_evaluation(candidate_id, evaluation)
            return result is not None
        except Exception as e:
            self.logger.error(f"Ошибка кэширования оценки: {e}")
            return False

    async def get_cached_evaluation(
        self, candidate_id: UUID
    ) -> Optional[StrategyEvaluationResult]:
        """Получить кэшированную оценку стратегии."""
        try:
            result = await self.evolution_cache.get_evaluation(candidate_id)
            return result if result is not None else None
        except Exception as e:
            self.logger.error(f"Ошибка получения кэшированной оценки: {e}")
            return None

    async def create_evolution_backup(self) -> bool:
        """Создать резервную копию эволюции."""
        try:
            backup_data = {
                "population": [c.to_dict() for c in self.population],
                "evaluations": {
                    str(k): v.to_dict() for k, v in self.evaluations.items()
                },
                "approved_strategies": [c.to_dict() for c in self.approved_strategies],
                "evolution_history": self.evolution_history,
                "stats": self.stats,
                "current_generation": self.current_generation,
            }
            backup_metadata = self.evolution_backup.create_backup("evolution_backup")
            self.logger.info(f"Создан бэкап эволюции: {backup_metadata}")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка создания резервной копии: {e}")
            return False

    async def restore_evolution_from_backup(self, backup_id: str) -> bool:
        """Восстановить эволюцию из резервной копии."""
        try:
            backup_data = self.evolution_backup._load_backup_data(backup_id)
            self.evolution_backup._restore_backup_data(backup_data)
            self.logger.info(f"Восстановлен бэкап эволюции: {backup_id}")
            # Восстановление состояния
            self.population = [
                StrategyCandidate.from_dict(c) for c in backup_data["population"]
            ]
            self.evaluations = {
                UUID(k): StrategyEvaluationResult.from_dict(v)
                for k, v in backup_data["evaluations"].items()
            }
            self.approved_strategies = [
                StrategyCandidate.from_dict(c)
                for c in backup_data["approved_strategies"]
            ]
            self.evolution_history = backup_data["evolution_history"]
            self.stats = backup_data["stats"]
            self.current_generation = backup_data["current_generation"]
            self.logger.info("Эволюция восстановлена из резервной копии")
            return True
        except Exception as e:
            self.logger.error(f"Ошибка восстановления из резервной копии: {e}")
            return False

    async def run_evolution_migration(self) -> bool:
        """Запустить миграцию эволюции."""
        try:
            result = await self.evolution_migration.run_migration()
            self.logger.info("Миграция эволюции выполнена")
            return result is not None
        except Exception as e:
            self.logger.error(f"Ошибка миграции эволюции: {e}")
            return False

    async def get_evolution_metrics(self) -> Dict[str, Any]:
        """Получить метрики эволюции."""
        try:
            cache_stats = await self.evolution_cache.get_statistics()  # type: ignore[misc]
            storage_stats = self.strategy_storage.get_statistics()
            backup_stats = self.evolution_backup.list_backups()
            return {
                "cache": cache_stats,
                "storage": storage_stats,
                "backup": backup_stats,
                "evolution": self.stats,
                "current_generation": self.current_generation,
                "population_size": len(self.population),
                "approved_strategies_count": len(self.approved_strategies),
            }
        except Exception as e:
            self.logger.error(f"Ошибка получения метрик эволюции: {e}")
            return {}
