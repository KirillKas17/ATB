"""
Функции сериализации/десериализации для infrastructure/evolution слоя.
"""

import json
import logging
from decimal import Decimal
from uuid import UUID

from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import (
    EntryRule,
    EvolutionContext,
    EvolutionStatus,
    ExitRule,
    FilterConfig,
    FilterType,
    IndicatorConfig,
    IndicatorType,
    SignalType,
    StrategyCandidate,
    StrategyType,
)
from infrastructure.evolution.exceptions import SerializationError
from infrastructure.evolution.models import (
    EvolutionContextModel,
    StrategyCandidateModel,
    StrategyEvaluationModel,
)

logger = logging.getLogger(__name__)


def candidate_to_model(candidate: StrategyCandidate) -> StrategyCandidateModel:
    """
    Преобразовать доменный объект StrategyCandidate в ORM-модель.
    Args:
        candidate: Доменный объект кандидата стратегии
    Returns:
        ORM-модель для сохранения в БД
    Raises:
        SerializationError: При ошибке сериализации
    """
    try:
        return StrategyCandidateModel(
            id=str(candidate.id),
            name=candidate.name,
            description=candidate.description,
            strategy_type=candidate.strategy_type.value,
            status=candidate.status.value,
            generation=candidate.generation,
            parent_ids=json.dumps([str(pid) for pid in candidate.parent_ids]),
            mutation_count=candidate.mutation_count,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
            indicators_config=json.dumps(
                [ind.to_dict() for ind in candidate.indicators]
            ),
            filters_config=json.dumps([filt.to_dict() for filt in candidate.filters]),
            entry_rules_config=json.dumps(
                [rule.to_dict() for rule in candidate.entry_rules]
            ),
            exit_rules_config=json.dumps(
                [rule.to_dict() for rule in candidate.exit_rules]
            ),
            position_size_pct=str(float(candidate.position_size_pct)),
            max_positions=candidate.max_positions,
            min_holding_time=candidate.min_holding_time,
            max_holding_time=candidate.max_holding_time,
            meta_data=json.dumps(candidate.metadata),
        )
    except Exception as e:
        logger.error(f"Ошибка сериализации StrategyCandidate: {e}")
        raise SerializationError(
            f"Не удалось сериализовать StrategyCandidate: {e}",
            "serialization_error",
            {"candidate_id": str(candidate.id)},
        )


def model_to_candidate(model: StrategyCandidateModel) -> StrategyCandidate:
    """
    Преобразовать ORM-модель в доменный объект StrategyCandidate.
    Args:
        model: ORM-модель из БД
    Returns:
        Доменный объект кандидата стратегии
    Raises:
        SerializationError: При ошибке десериализации
    """
    try:
        # Восстановить индикаторы
        indicators_data = json.loads(model.indicators_config)
        indicators = []
        for ind_data in indicators_data:
            indicator = IndicatorConfig.from_dict(ind_data)
            indicators.append(indicator)
        # Восстановить фильтры
        filters_data = json.loads(model.filters_config)
        filters = []
        for filt_data in filters_data:
            filter_config = FilterConfig.from_dict(filt_data)
            filters.append(filter_config)
        # Восстановить правила входа
        entry_rules_data = json.loads(model.entry_rules_config)
        entry_rules = []
        for rule_data in entry_rules_data:
            entry_rule = EntryRule.from_dict(rule_data)
            entry_rules.append(entry_rule)
        # Восстановить правила выхода
        exit_rules_data = json.loads(model.exit_rules_config)
        exit_rules = []
        for rule_data in exit_rules_data:
            exit_rule = ExitRule.from_dict(rule_data)
            exit_rules.append(exit_rule)
        return StrategyCandidate(
            id=UUID(model.id),
            name=model.name,
            description=model.description,
            strategy_type=StrategyType(model.strategy_type),
            status=EvolutionStatus(model.status),
            generation=model.generation,
            parent_ids=[UUID(pid) for pid in json.loads(model.parent_ids)],
            mutation_count=model.mutation_count,
            created_at=model.created_at,
            updated_at=model.updated_at,
            indicators=indicators,
            filters=filters,
            entry_rules=entry_rules,
            exit_rules=exit_rules,
            position_size_pct=Decimal(str(model.position_size_pct)),
            max_positions=model.max_positions,
            min_holding_time=model.min_holding_time,
            max_holding_time=model.max_holding_time,
            metadata=json.loads(model.meta_data),
        )
    except Exception as e:
        logger.error(f"Ошибка десериализации StrategyCandidate: {e}")
        raise SerializationError(
            f"Не удалось десериализовать StrategyCandidate: {e}",
            "deserialization_error",
            {"model_id": model.id},
        )


def evaluation_to_model(
    evaluation: StrategyEvaluationResult,
) -> StrategyEvaluationModel:
    """
    Преобразовать доменный объект StrategyEvaluationResult в ORM-модель.
    Args:
        evaluation: Доменный объект результата оценки
    Returns:
        ORM-модель для сохранения в БД
    Raises:
        SerializationError: При ошибке сериализации
    """
    try:
        return StrategyEvaluationModel(
            id=str(evaluation.id),
            strategy_id=str(evaluation.strategy_id),
            total_trades=evaluation.total_trades,
            winning_trades=evaluation.winning_trades,
            losing_trades=evaluation.losing_trades,
            win_rate=str(float(evaluation.win_rate)),
            accuracy=str(float(evaluation.accuracy)),
            total_pnl=str(float(evaluation.total_pnl)),
            net_pnl=str(float(evaluation.net_pnl)),
            profitability=str(float(evaluation.profitability)),
            profit_factor=str(float(evaluation.profit_factor)),
            max_drawdown=str(float(evaluation.max_drawdown)),
            max_drawdown_pct=str(float(evaluation.max_drawdown_pct)),
            sharpe_ratio=str(float(evaluation.sharpe_ratio)),
            sortino_ratio=str(float(evaluation.sortino_ratio)),
            calmar_ratio=str(float(evaluation.calmar_ratio)),
            average_trade=str(float(evaluation.average_trade)),
            best_trade=str(float(evaluation.best_trade)),
            worst_trade=str(float(evaluation.worst_trade)),
            average_win=str(float(evaluation.average_win)),
            average_loss=str(float(evaluation.average_loss)),
            largest_win=str(float(evaluation.largest_win)),
            largest_loss=str(float(evaluation.largest_loss)),
            average_holding_time=evaluation.average_holding_time,
            total_trading_time=evaluation.total_trading_time,
            start_date=evaluation.start_date,
            end_date=evaluation.end_date,
            is_approved=evaluation.is_approved,
            approval_reason=evaluation.approval_reason,
            fitness_score=str(float(evaluation.get_fitness_score())),
            evaluation_time=evaluation.evaluation_time,
            meta_data=json.dumps(evaluation.metadata),
        )
    except Exception as e:
        logger.error(f"Ошибка сериализации StrategyEvaluationResult: {e}")
        raise SerializationError(
            f"Не удалось сериализовать StrategyEvaluationResult: {e}",
            "serialization_error",
            {"evaluation_id": str(evaluation.id)},
        )


def model_to_evaluation(model: StrategyEvaluationModel) -> StrategyEvaluationResult:
    """
    Преобразовать ORM-модель в доменный объект StrategyEvaluationResult.
    Args:
        model: ORM-модель из БД
    Returns:
        Доменный объект результата оценки
    Raises:
        SerializationError: При ошибке десериализации
    """
    try:
        return StrategyEvaluationResult(
            id=UUID(model.id),
            strategy_id=UUID(model.strategy_id),
            total_trades=model.total_trades,
            winning_trades=model.winning_trades,
            losing_trades=model.losing_trades,
            win_rate=Decimal(model.win_rate),
            accuracy=Decimal(model.accuracy),
            total_pnl=Decimal(model.total_pnl),
            net_pnl=Decimal(model.net_pnl),
            profitability=Decimal(model.profitability),
            profit_factor=Decimal(model.profit_factor),
            max_drawdown=Decimal(model.max_drawdown),
            max_drawdown_pct=Decimal(model.max_drawdown_pct),
            sharpe_ratio=Decimal(model.sharpe_ratio),
            sortino_ratio=Decimal(model.sortino_ratio),
            calmar_ratio=Decimal(model.calmar_ratio),
            average_trade=Decimal(model.average_trade),
            best_trade=Decimal(model.best_trade),
            worst_trade=Decimal(model.worst_trade),
            average_win=Decimal(model.average_win),
            average_loss=Decimal(model.average_loss),
            largest_win=Decimal(model.largest_win),
            largest_loss=Decimal(model.largest_loss),
            average_holding_time=model.average_holding_time,
            total_trading_time=model.total_trading_time,
            start_date=model.start_date,
            end_date=model.end_date,
            is_approved=model.is_approved,
            approval_reason=model.approval_reason,
            evaluation_time=model.evaluation_time,
            metadata=json.loads(model.meta_data),
        )
    except Exception as e:
        logger.error(f"Ошибка десериализации StrategyEvaluationResult: {e}")
        raise SerializationError(
            f"Не удалось десериализовать StrategyEvaluationResult: {e}",
            "deserialization_error",
            {"model_id": model.id},
        )


def context_to_model(context: EvolutionContext) -> EvolutionContextModel:
    """
    Преобразовать доменный объект EvolutionContext в ORM-модель.
    Args:
        context: Доменный объект контекста эволюции
    Returns:
        ORM-модель для сохранения в БД
    Raises:
        SerializationError: При ошибке сериализации
    """
    try:
        return EvolutionContextModel(
            id=str(context.id),
            name=context.name,
            description=context.description,
            created_at=context.created_at,
            updated_at=context.updated_at,
            population_size=context.population_size,
            generations=context.generations,
            mutation_rate=str(float(context.mutation_rate)),
            crossover_rate=str(float(context.crossover_rate)),
            elite_size=context.elite_size,
            min_accuracy=str(float(context.min_accuracy)),
            min_profitability=str(float(context.min_profitability)),
            max_drawdown=str(float(context.max_drawdown)),
            min_sharpe=str(float(context.min_sharpe)),
            max_indicators=context.max_indicators,
            max_filters=context.max_filters,
            max_entry_rules=context.max_entry_rules,
            max_exit_rules=context.max_exit_rules,
            meta_data=json.dumps(context.metadata),
        )
    except Exception as e:
        logger.error(f"Ошибка сериализации EvolutionContext: {e}")
        raise SerializationError(
            f"Не удалось сериализовать EvolutionContext: {e}",
            "serialization_error",
            {"context_id": str(context.id)},
        )


def model_to_context(model: EvolutionContextModel) -> EvolutionContext:
    """
    Преобразовать ORM-модель в доменный объект EvolutionContext.
    Args:
        model: ORM-модель из БД
    Returns:
        Доменный объект контекста эволюции
    Raises:
        SerializationError: При ошибке десериализации
    """
    try:
        return EvolutionContext(
            id=UUID(model.id),
            name=model.name,
            description=model.description,
            population_size=model.population_size,
            generations=model.generations,
            mutation_rate=Decimal(model.mutation_rate),
            crossover_rate=Decimal(model.crossover_rate),
            elite_size=model.elite_size,
            min_accuracy=Decimal(model.min_accuracy),
            min_profitability=Decimal(model.min_profitability),
            max_drawdown=Decimal(model.max_drawdown),
            min_sharpe=Decimal(model.min_sharpe),
            max_indicators=model.max_indicators,
            max_filters=model.max_filters,
            max_entry_rules=model.max_entry_rules,
            max_exit_rules=model.max_exit_rules,
            created_at=model.created_at,
            updated_at=model.updated_at,
            metadata=json.loads(model.meta_data),
        )
    except Exception as e:
        logger.error(f"Ошибка десериализации EvolutionContext: {e}")
        raise SerializationError(
            f"Не удалось десериализовать EvolutionContext: {e}",
            "deserialization_error",
            {"model_id": model.id},
        )
