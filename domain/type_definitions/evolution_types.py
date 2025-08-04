"""
Типы для модуля эволюции стратегий.
Содержит все типы, используемые в модуле evolution:
- TypedDict для структурированных данных
- Protocol для интерфейсов
- NewType для типобезопасности
- Literal для констант
- Final для неизменяемых значений
"""
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Union,
)
from uuid import UUID

import pandas as pd

# Импорт для типизации
if TYPE_CHECKING:
    from domain.evolution.strategy_fitness import StrategyEvaluationResult
    from domain.evolution.strategy_model import StrategyCandidate, EvolutionContext

# Literal типы для констант (доступны во время выполнения)
OptimizationMethod = Literal["genetic", "bayesian", "gradient", "particle_swarm", "simulated_annealing"]
SelectionMethod = Literal["fitness", "multi_criteria", "pareto", "tournament", "diversity"]
MutationType = Literal["parameter", "structure", "hybrid", "adaptive"]
CrossoverType = Literal["uniform", "single_point", "two_point", "arithmetic", "blend"]
EvaluationStatus = Literal["pending", "running", "completed", "failed", "cancelled"]

# Final константы (доступны во время выполнения)
DEFAULT_POPULATION_SIZE: Final = 50
DEFAULT_GENERATIONS: Final = 100
DEFAULT_MUTATION_RATE: Final = Decimal("0.1")
DEFAULT_CROSSOVER_RATE: Final = Decimal("0.8")
DEFAULT_ELITE_SIZE: Final = 5
MIN_ACCURACY_THRESHOLD: Final = Decimal("0.82")
MIN_PROFITABILITY_THRESHOLD: Final = Decimal("0.05")
MAX_DRAWDOWN_THRESHOLD: Final = Decimal("0.15")
MIN_SHARPE_THRESHOLD: Final = Decimal("1.0")

# NewType для типобезопасности (доступны во время выполнения)
FitnessScore = NewType('FitnessScore', Decimal)
AccuracyScore = NewType('AccuracyScore', Decimal)
ProfitabilityScore = NewType('ProfitabilityScore', Decimal)
RiskScore = NewType('RiskScore', Decimal)
ConsistencyScore = NewType('ConsistencyScore', Decimal)
DiversityScore = NewType('DiversityScore', Decimal)
ComplexityScore = NewType('ComplexityScore', Decimal)

# TypedDict для структурированных данных
class IndicatorParameters(TypedDict, total=False):
    """Параметры индикатора."""
    period: int
    fast: int
    slow: int
    signal: int
    acceleration: float
    maximum: float
    k_period: int
    d_period: int
    std_dev: float
    multiplier: float
    period1: int
    period2: int
    period3: int
    threshold: float
    smoothing: int
    alpha: float
    beta: float
    gamma: float
class FilterParameters(TypedDict, total=False):
    """Параметры фильтра."""
    min_atr: float
    max_atr: float
    min_width: float
    max_width: float
    min_volume: int
    spike_threshold: float
    min_adx: int
    trend_period: int
    start_hour: int
    end_hour: int
    excluded_days: List[int]
    regime: str
    max_correlation: float
    volatility_threshold: float
    volume_threshold: float
    trend_strength: float
class EntryCondition(TypedDict):
    """Условие входа в позицию."""
    indicator: str
    condition: str
    period: Optional[int]
    direction: Optional[str]
    threshold: Optional[float]
    operator: Optional[str]
    value: Optional[Union[float, str, bool]]
class ExitCondition(TypedDict):
    """Условие выхода из позиции."""
    indicator: str
    condition: str
    period: Optional[int]
    threshold: Optional[float]
    operator: Optional[str]
    value: Optional[Union[float, str, bool]]
class TradePosition(TypedDict):
    """Торговая позиция."""
    id: str
    entry_time: datetime
    entry_price: Decimal
    quantity: Decimal
    signal_type: str
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    trailing_stop: bool
    trailing_distance: Optional[Decimal]
    current_price: Optional[Decimal]
    unrealized_pnl: Optional[Decimal]
    holding_time: Optional[int]
class OptimizationResult(TypedDict):
    """Результат оптимизации."""
    generation: int
    best_fitness: float
    avg_fitness: float
    optimization_type: str
    timestamp: datetime
    parameters: Dict[str, Any]
    convergence_rate: float
    diversity: float
    improvement_rate: float
class SelectionStatistics(TypedDict):
    """Статистика отбора."""
    total_candidates: int
    filtered_count: int
    selected_count: int
    best_score: float
    method: str
    timestamp: datetime
    diversity_score: float
    convergence_score: float
class EvolutionMetrics(TypedDict):
    """Метрики эволюции."""
    generation: int
    population_size: int
    best_fitness: float
    avg_fitness: float
    diversity: float
    convergence_rate: float
    improvement_rate: float
    stagnation_count: int
    mutation_rate: float
    crossover_rate: float
class StrategyPerformance(TypedDict):
    """Производительность стратегии."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    total_pnl: float
    net_pnl: float
    average_trade: float
    best_trade: float
    worst_trade: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
# Protocol интерфейсы
class FitnessEvaluatorProtocol(Protocol):
    """Протокол для оценки fitness стратегий."""
    def evaluate_strategy(
        self,
        candidate: 'StrategyCandidate',
        historical_data: pd.DataFrame,
        initial_capital: Decimal = Decimal("10000"),
    ) -> 'StrategyEvaluationResult':
        """Оценить стратегию."""
        ...
    def get_fitness_score(self, evaluation: 'StrategyEvaluationResult') -> FitnessScore:
        """Получить fitness score."""
        ...
    def get_evaluation_result(self, strategy_id: UUID) -> Optional['StrategyEvaluationResult']:
        """Получить результат оценки."""
        ...
    def get_all_results(self) -> List['StrategyEvaluationResult']:
        """Получить все результаты."""
        ...
    def get_approved_strategies(self) -> List['StrategyEvaluationResult']:
        """Получить одобренные стратегии."""
        ...
    def get_top_strategies(self, n: int = 10) -> List['StrategyEvaluationResult']:
        """Получить топ стратегии."""
        ...
    def clear_results(self) -> None:
        """Очистить результаты."""
        ...
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Получить статистику оценки."""
        ...
class StrategyGeneratorProtocol(Protocol):
    """Протокол для генерации стратегий."""
    def generate_random_strategy(self, name_prefix: str = "Evolved") -> 'StrategyCandidate':
        """Сгенерировать случайную стратегию."""
        ...
    def generate_population(self, size: Optional[int] = None) -> List['StrategyCandidate']:
        """Сгенерировать популяцию стратегий."""
        ...
    def generate_from_parents(
        self, 
        parents: List['StrategyCandidate'], 
        num_children: int
    ) -> List['StrategyCandidate']:
        """Сгенерировать потомков от родителей."""
        ...
    def mutate_strategy(self, candidate: 'StrategyCandidate') -> 'StrategyCandidate':
        """Мутировать стратегию."""
        ...
    def crossover_strategies(
        self, 
        parent1: 'StrategyCandidate', 
        parent2: 'StrategyCandidate'
    ) -> 'StrategyCandidate':
        """Скрестить стратегии."""
        ...
class StrategyOptimizerProtocol(Protocol):
    """Протокол для оптимизации стратегий."""
    def optimize_strategy(
        self,
        candidate: 'StrategyCandidate',
        historical_data: pd.DataFrame,
        optimization_type: OptimizationMethod = "genetic",
        max_iterations: int = 100,
    ) -> 'StrategyCandidate':
        """Оптимизировать стратегию."""
        ...
    def optimize_population(
        self,
        population: List['StrategyCandidate'],
        historical_data: pd.DataFrame,
        optimization_type: OptimizationMethod = "genetic",
        max_iterations: int = 50,
    ) -> List['StrategyCandidate']:
        """Оптимизировать популяцию."""
        ...
    def get_optimization_history(self) -> List[OptimizationResult]:
        """Получить историю оптимизации."""
        ...
class StrategySelectorProtocol(Protocol):
    """Протокол для отбора стратегий."""
    def select_top_strategies(
        self,
        candidates: List['StrategyCandidate'],
        evaluations: List['StrategyEvaluationResult'],
        n: int = 10,
        selection_method: SelectionMethod = "fitness",
    ) -> List['StrategyCandidate']:
        """Выбрать топ стратегии."""
        ...
    def filter_by_criteria(
        self,
        candidates: List['StrategyCandidate'],
        evaluations: List['StrategyEvaluationResult'],
        criteria: Dict[str, Any],
    ) -> List['StrategyCandidate']:
        """Фильтровать по критериям."""
        ...
    def select_diverse_strategies(
        self,
        candidates: List['StrategyCandidate'],
        evaluations: List['StrategyEvaluationResult'],
        n: int = 10,
        diversity_metric: str = "strategy_type",
    ) -> List['StrategyCandidate']:
        """Выбрать разнообразные стратегии."""
        ...
    def get_selection_statistics(
        self,
        candidates: List['StrategyCandidate'],
        evaluations: List['StrategyEvaluationResult'],
    ) -> SelectionStatistics:
        """Получить статистику отбора."""
        ...
class EvolutionOrchestratorProtocol(Protocol):
    """Протокол для оркестрации эволюции."""
    def run_evolution(
        self,
        context: 'EvolutionContext',
        historical_data: pd.DataFrame,
        max_generations: Optional[int] = None,
    ) -> List['StrategyCandidate']:
        """Запустить эволюцию."""
        ...
    def get_evolution_metrics(self) -> EvolutionMetrics:
        """Получить метрики эволюции."""
        ...
    def get_best_strategies(self, n: int = 10) -> List['StrategyCandidate']:
        """Получить лучшие стратегии."""
        ...
    def save_evolution_state(self, filepath: str) -> None:
        """Сохранить состояние эволюции."""
        ...
    def load_evolution_state(self, filepath: str) -> None:
        """Загрузить состояние эволюции."""
        ...
# Enums
class EvolutionPhase(Enum):
    """Фазы эволюции."""
    INITIALIZATION = "initialization"
    GENERATION = "generation"
    EVALUATION = "evaluation"
    SELECTION = "selection"
    CROSSOVER = "crossover"
    MUTATION = "mutation"
    OPTIMIZATION = "optimization"
    CONVERGENCE = "convergence"
    TERMINATION = "termination"
class FitnessComponent(Enum):
    """Компоненты fitness функции."""
    ACCURACY = "accuracy"
    PROFITABILITY = "profitability"
    RISK = "risk"
    CONSISTENCY = "consistency"
    DIVERSITY = "diversity"
    COMPLEXITY = "complexity"
class MutationStrategy(Enum):
    """Стратегии мутации."""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    DIFFERENTIAL = "differential"
    POLYNOMIAL = "polynomial"
    SWAP = "swap"
    INVERSION = "inversion"
    INSERTION = "insertion"
    DELETION = "deletion"
class CrossoverStrategy(Enum):
    """Стратегии скрещивания."""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    ARITHMETIC = "arithmetic"
    BLEND = "blend"
    SIMULATED_BINARY = "simulated_binary"
    ORDERED = "ordered"
    PARTIALLY_MAPPED = "partially_mapped"
class SelectionStrategy(Enum):
    """Стратегии отбора."""
    ROULETTE_WHEEL = "roulette_wheel"
    TOURNAMENT = "tournament"
    RANK_BASED = "rank_based"
    ELITISM = "elitism"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    TRUNCATION = "truncation"
# Dataclasses для конфигурации
@dataclass
class EvolutionConfig:
    """Конфигурация эволюции."""
    # Основные параметры
    population_size: int = DEFAULT_POPULATION_SIZE
    generations: int = DEFAULT_GENERATIONS
    mutation_rate: Decimal = DEFAULT_MUTATION_RATE
    crossover_rate: Decimal = DEFAULT_CROSSOVER_RATE
    elite_size: int = DEFAULT_ELITE_SIZE
    # Критерии качества
    min_accuracy: Decimal = MIN_ACCURACY_THRESHOLD
    min_profitability: Decimal = MIN_PROFITABILITY_THRESHOLD
    max_drawdown: Decimal = MAX_DRAWDOWN_THRESHOLD
    min_sharpe: Decimal = MIN_SHARPE_THRESHOLD
    # Ограничения
    max_indicators: int = 10
    max_filters: int = 5
    max_entry_rules: int = 3
    max_exit_rules: int = 3
    # Методы
    optimization_method: OptimizationMethod = "genetic"
    selection_method: SelectionMethod = "fitness"
    mutation_type: MutationStrategy = MutationStrategy.GAUSSIAN
    crossover_type: CrossoverStrategy = CrossoverStrategy.UNIFORM
    # Дополнительные параметры
    tournament_size: int = 3
    diversity_weight: Decimal = Decimal("0.1")
    complexity_penalty: Decimal = Decimal("0.05")
    convergence_threshold: Decimal = Decimal("0.001")
    stagnation_limit: int = 20
    def __post_init__(self) -> None:
        """Валидация конфигурации."""
        if self.population_size <= 0:
            raise ValueError("Population size must be positive")
        if self.generations <= 0:
            raise ValueError("Generations must be positive")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        if self.elite_size < 0 or self.elite_size > self.population_size:
            raise ValueError("Elite size must be between 0 and population size")
@dataclass
class FitnessWeights:
    """Веса компонентов fitness функции."""
    accuracy: Decimal = Decimal("0.3")
    profitability: Decimal = Decimal("0.3")
    risk: Decimal = Decimal("0.2")
    consistency: Decimal = Decimal("0.2")
    diversity: Decimal = Decimal("0.0")
    complexity: Decimal = Decimal("0.0")
    def __post_init__(self) -> None:
        """Валидация весов."""
        total_weight = sum([
            self.accuracy, self.profitability, self.risk,
            self.consistency, self.diversity, self.complexity
        ])
        if total_weight != Decimal("1.0"):
            raise ValueError(f"Sum of weights must equal 1.0, got {total_weight}")
        for weight in [self.accuracy, self.profitability, self.risk, 
                      self.consistency, self.diversity, self.complexity]:
            if weight < 0:
                raise ValueError(f"Weight cannot be negative: {weight}")