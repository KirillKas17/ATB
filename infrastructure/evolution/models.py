"""
SQLModel-модели для infrastructure/evolution слоя.
"""

from datetime import datetime

from sqlmodel import Field, SQLModel


class StrategyCandidateModel(SQLModel, table=True):  # type: ignore[call-arg]
    """Модель для хранения кандидатов стратегий."""

    __tablename__ = "strategy_candidates"
    id: str = Field(primary_key=True)
    name: str
    description: str
    strategy_type: str
    status: str
    generation: int
    parent_ids: str  # JSON string
    mutation_count: int
    created_at: datetime
    updated_at: datetime
    # Конфигурация стратегии (JSON)
    indicators_config: str
    filters_config: str
    entry_rules_config: str
    exit_rules_config: str
    # Параметры исполнения
    position_size_pct: str
    max_positions: int
    min_holding_time: int
    max_holding_time: int
    # Метаданные
    meta_data: str  # JSON string


class StrategyEvaluationModel(SQLModel, table=True):  # type: ignore[call-arg]
    """Модель для хранения результатов оценки стратегий."""

    __tablename__ = "strategy_evaluations"
    id: str = Field(primary_key=True)
    strategy_id: str
    evaluation_time: datetime
    # Основные метрики
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: str
    accuracy: str
    # Финансовые метрики
    total_pnl: str
    net_pnl: str
    profitability: str
    profit_factor: str
    # Риск-метрики
    max_drawdown: str
    max_drawdown_pct: str
    sharpe_ratio: str
    sortino_ratio: str
    calmar_ratio: str
    # Дополнительные метрики
    average_trade: str
    best_trade: str
    worst_trade: str
    average_win: str
    average_loss: str
    largest_win: str
    largest_loss: str
    # Временные метрики
    average_holding_time: int
    total_trading_time: int
    start_date: datetime
    end_date: datetime
    # Статус оценки
    is_approved: bool
    approval_reason: str
    fitness_score: str
    # Метаданные
    meta_data: str  # JSON string


class EvolutionContextModel(SQLModel, table=True):  # type: ignore[call-arg]
    """Модель для хранения контекстов эволюции."""

    __tablename__ = "evolution_contexts"
    id: str = Field(primary_key=True)
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    # Параметры эволюции
    population_size: int
    generations: int
    mutation_rate: str
    crossover_rate: str
    elite_size: int
    # Критерии отбора
    min_accuracy: str
    min_profitability: str
    max_drawdown: str
    min_sharpe: str
    # Ограничения
    max_indicators: int
    max_filters: int
    max_entry_rules: int
    max_exit_rules: int
    # Метаданные
    meta_data: str  # JSON string
