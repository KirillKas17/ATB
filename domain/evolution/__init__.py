"""
Модуль эволюции стратегий.
Этот модуль содержит компоненты для эволюционного развития торговых стратегий:
- Модели стратегий и кандидатов
- Оценка эффективности (fitness)
- Генерация новых стратегий
- Оптимизация существующих стратегий
- Отбор лучших стратегий
"""

# Экспорт типов из evolution_types
from domain.entities.signal import SignalType
from domain.type_definitions.strategy_types import StrategyType
from domain.type_definitions.evolution_types import (
    EvolutionConfig,  # NewType; Literal; Final constants; TypedDict; Protocol; Enum; Dataclass
)
from domain.type_definitions.evolution_types import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_ELITE_SIZE,
    DEFAULT_GENERATIONS,
    DEFAULT_MUTATION_RATE,
    DEFAULT_POPULATION_SIZE,
    MAX_DRAWDOWN_THRESHOLD,
    MIN_ACCURACY_THRESHOLD,
    MIN_PROFITABILITY_THRESHOLD,
    MIN_SHARPE_THRESHOLD,
    AccuracyScore,
    ComplexityScore,
    ConsistencyScore,
    CrossoverStrategy,
    CrossoverType,
    DiversityScore,
    EntryCondition,
    EvaluationStatus,
    EvolutionMetrics,
    EvolutionOrchestratorProtocol,
    EvolutionPhase,
    ExitCondition,
    FilterParameters,
    FitnessComponent,
    FitnessEvaluatorProtocol,
    FitnessScore,
    FitnessWeights,
    IndicatorParameters,
    MutationStrategy,
    MutationType,
    OptimizationMethod,
    OptimizationResult,
    ProfitabilityScore,
    RiskScore,
    SelectionMethod,
    SelectionStatistics,
    SelectionStrategy,
    StrategyGeneratorProtocol,
    StrategyOptimizerProtocol,
    StrategyPerformance,
    StrategySelectorProtocol,
    TradePosition,
)

from .strategy_fitness import (
    StrategyEvaluationResult,
    StrategyFitnessEvaluator,
    TradeResult,
)
from .strategy_generator import StrategyGenerator
from .strategy_model import (
    EntryRule,
    EvolutionContext,
    EvolutionStatus,
    ExitRule,
    FilterConfig,
    FilterType,
    IndicatorConfig,
    IndicatorType,
    StrategyCandidate,
)
from .strategy_selection import StrategySelector

__all__ = [
    # Модели
    "StrategyCandidate",
    "EvolutionContext",
    "EvolutionStatus",
    "IndicatorConfig",
    "FilterConfig",
    "EntryRule",
    "ExitRule",
    "IndicatorType",
    "FilterType",
    # Оценка эффективности
    "StrategyFitnessEvaluator",
    "StrategyEvaluationResult",
    "TradeResult",
    # Компоненты эволюции
    "StrategyGenerator",
    "StrategyOptimizer",
    "StrategySelector",
    # Типы
    "SignalType",
    "StrategyType",
    "FitnessScore",
    "AccuracyScore",
    "ProfitabilityScore",
    "RiskScore",
    "ConsistencyScore",
    "DiversityScore",
    "ComplexityScore",
    "OptimizationMethod",
    "SelectionMethod",
    "MutationType",
    "CrossoverType",
    "EvaluationStatus",
    "DEFAULT_POPULATION_SIZE",
    "DEFAULT_GENERATIONS",
    "DEFAULT_MUTATION_RATE",
    "DEFAULT_CROSSOVER_RATE",
    "DEFAULT_ELITE_SIZE",
    "MIN_ACCURACY_THRESHOLD",
    "MIN_PROFITABILITY_THRESHOLD",
    "MAX_DRAWDOWN_THRESHOLD",
    "MIN_SHARPE_THRESHOLD",
    "IndicatorParameters",
    "FilterParameters",
    "EntryCondition",
    "ExitCondition",
    "TradePosition",
    "OptimizationResult",
    "SelectionStatistics",
    "EvolutionMetrics",
    "StrategyPerformance",
    "EvolutionConfigDict",
    "FitnessEvaluatorProtocol",
    "StrategyGeneratorProtocol",
    "StrategyOptimizerProtocol",
    "StrategySelectorProtocol",
    "EvolutionOrchestratorProtocol",
    "EvolutionPhase",
    "FitnessComponent",
    "MutationStrategy",
    "CrossoverStrategy",
    "SelectionStrategy",
    "EvolutionConfig",
    "FitnessWeights",
]
