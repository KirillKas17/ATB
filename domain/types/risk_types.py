"""
Промышленные типы для риск-анализа.
Содержит строго типизированные структуры данных для анализа рисков,
оптимизации портфеля и управления рисками.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    TypedDict,
    Union,
    runtime_checkable,
)
import pandas as pd
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.percentage import Percentage


class RiskMetricType(Enum):
    """Типы метрик риска."""

    VAR = "value_at_risk"
    CVAR = "conditional_value_at_risk"
    SHARPE = "sharpe_ratio"
    SORTINO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    BETA = "beta"
    CORRELATION = "correlation"
    INFORMATION_RATIO = "information_ratio"
    CALMAR_RATIO = "calmar_ratio"
    TREYNOR_RATIO = "treynor_ratio"
    JENSEN_ALPHA = "jensen_alpha"
    TRACKING_ERROR = "tracking_error"
    DOWNSIDE_DEVIATION = "downside_deviation"
    UPSIDE_POTENTIAL_RATIO = "upside_potential_ratio"
    GAIN_LOSS_RATIO = "gain_loss_ratio"
    PROFIT_FACTOR = "profit_factor"
    RECOVERY_FACTOR = "recovery_factor"


class RiskLevel(Enum):
    """Уровни риска."""

    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PortfolioOptimizationMethod(Enum):
    """Методы оптимизации портфеля."""

    SHARPE_MAXIMIZATION = "sharpe_maximization"
    RISK_MINIMIZATION = "risk_minimization"
    RETURN_MAXIMIZATION = "return_maximization"
    BLACK_LITTERMAN = "black_litterman"
    RISK_PARITY = "risk_parity"
    MEAN_VARIANCE = "mean_variance"


@dataclass(frozen=True)
class RiskMetrics:
    """Метрики риска портфеля."""

    # Основные метрики
    volatility: Decimal = field(default=Decimal("0"))
    sharpe_ratio: Decimal = field(default=Decimal("0"))
    sortino_ratio: Decimal = field(default=Decimal("0"))
    max_drawdown: Decimal = field(default=Decimal("0"))
    var_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    cvar_95: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    var_99: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    cvar_99: Money = field(default_factory=lambda: Money(Decimal("0"), Currency.USD))
    # Дополнительные метрики
    beta: Decimal = field(default=Decimal("0"))
    alpha: Decimal = field(default=Decimal("0"))
    information_ratio: Decimal = field(default=Decimal("0"))
    calmar_ratio: Decimal = field(default=Decimal("0"))
    treynor_ratio: Decimal = field(default=Decimal("0"))
    jensen_alpha: Decimal = field(default=Decimal("0"))
    tracking_error: Decimal = field(default=Decimal("0"))
    downside_deviation: Decimal = field(default=Decimal("0"))
    upside_potential_ratio: Decimal = field(default=Decimal("0"))
    gain_loss_ratio: Decimal = field(default=Decimal("0"))
    profit_factor: Decimal = field(default=Decimal("0"))
    recovery_factor: Decimal = field(default=Decimal("0"))
    risk_adjusted_return: Decimal = field(default=Decimal("0"))
    # Метаданные
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    confidence_level: Decimal = field(default=Decimal("0.95"))
    risk_free_rate: Decimal = field(default=Decimal("0.02"))
    data_points: int = field(default=0)

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.volatility < 0:
            raise ValueError("Volatility cannot be negative")
        if self.max_drawdown > 0:
            raise ValueError("Max drawdown should be negative")


@dataclass(frozen=True)
class PositionRisk:
    """Риск отдельной позиции."""

    symbol: str
    position_size: Money
    market_value: Money
    unrealized_pnl: Money
    var_95: Money
    var_99: Money
    beta: Decimal
    volatility: Decimal
    correlation_with_portfolio: Decimal
    contribution_to_portfolio_risk: Decimal
    risk_level: RiskLevel
    # Метаданные
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    position_age_days: int = field(default=0)
    last_rebalance_date: Optional[datetime] = field(default=None)


@dataclass(frozen=True)
class PortfolioRisk:
    """Риск портфеля."""

    total_value: Money
    total_risk: Decimal
    risk_metrics: RiskMetrics
    position_risks: List[PositionRisk]
    correlation_matrix: pd.DataFrame
    risk_decomposition: Dict[str, Decimal]
    # Дополнительные метрики
    diversification_ratio: Decimal = field(default=Decimal("0"))
    concentration_risk: Decimal = field(default=Decimal("0"))
    liquidity_risk: Decimal = field(default=Decimal("0"))
    currency_risk: Decimal = field(default=Decimal("0"))
    sector_risk: Decimal = field(default=Decimal("0"))
    # Метаданные
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    portfolio_id: Optional[str] = field(default=None)
    risk_model_version: str = field(default="1.0")

    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.total_value.value <= 0:
            raise ValueError("Total portfolio value must be positive")
        if self.total_risk < 0:
            raise ValueError("Total risk cannot be negative")


@dataclass(frozen=True)
class RiskOptimizationResult:
    """Результат оптимизации портфеля."""

    optimal_weights: Dict[str, Decimal]
    expected_return: Decimal
    expected_risk: Decimal
    sharpe_ratio: Decimal
    optimization_method: PortfolioOptimizationMethod
    # Дополнительные результаты
    efficient_frontier: pd.DataFrame
    risk_contribution: Dict[str, Decimal]
    return_contribution: Dict[str, Decimal]
    rebalancing_recommendations: List[str]
    # Метаданные
    optimization_timestamp: datetime = field(default_factory=datetime.now)
    constraints_applied: List[str] = field(default_factory=list)
    optimization_time_seconds: float = field(default=0.0)
    convergence_status: str = field(default="converged")


@dataclass(frozen=True)
class StressTestResult:
    """Результат стресс-тестирования."""

    scenario_name: str
    portfolio_value_change: Money
    var_change: Money
    max_drawdown_change: Decimal
    affected_positions: List[str]
    # Дополнительные метрики
    correlation_breakdown: bool = field(default=False)
    liquidity_impact: Decimal = field(default=Decimal("0"))
    recovery_time_days: Optional[int] = field(default=None)
    # Метаданные
    test_timestamp: datetime = field(default_factory=datetime.now)
    scenario_probability: Decimal = field(default=Decimal("0.01"))


@dataclass(frozen=True)
class RiskReport:
    """Полный отчет по рискам."""

    portfolio_risk: PortfolioRisk
    risk_metrics: RiskMetrics
    optimization_result: Optional[RiskOptimizationResult] = field(default=None)
    stress_test_results: List[StressTestResult] = field(default_factory=list)
    # Рекомендации
    risk_recommendations: List[str] = field(default_factory=list)
    rebalancing_suggestions: List[str] = field(default_factory=list)
    risk_alerts: List[str] = field(default_factory=list)
    # Метаданные
    report_timestamp: datetime = field(default_factory=datetime.now)
    report_period: str = field(default="daily")
    risk_model_parameters: Dict[str, Any] = field(default_factory=dict)


# Protocol для сервиса анализа рисков
@runtime_checkable
class RiskAnalysisServiceProtocol(Protocol):
    """Протокол для сервиса анализа рисков."""

    def calculate_risk_metrics(
        self, returns: pd.Series, risk_free_rate: Decimal = Decimal("0.02")
    ) -> RiskMetrics: ...
    def calculate_var(
        self, returns: pd.Series, confidence_level: Decimal = Decimal("0.95")
    ) -> Decimal: ...
    def calculate_cvar(
        self, returns: pd.Series, confidence_level: Decimal = Decimal("0.95")
    ) -> Decimal: ...
    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal: ...
    def calculate_sortino_ratio(
        self, returns: pd.Series, risk_free_rate: Decimal = Decimal("0.02")
    ) -> Decimal: ...
    def calculate_max_drawdown(self, prices: pd.Series) -> Decimal: ...
    def calculate_beta(
        self, asset_returns: pd.Series, market_returns: pd.Series
    ) -> Decimal: ...
    def calculate_correlation_matrix(
        self, returns_df: pd.DataFrame
    ) -> pd.DataFrame: ...
    def calculate_portfolio_risk(
        self, positions: List[PositionRisk], market_data: pd.DataFrame
    ) -> PortfolioRisk: ...
    def optimize_portfolio(
        self,
        returns_df: pd.DataFrame,
        target_return: Optional[Decimal] = None,
        risk_free_rate: Decimal = Decimal("0.02"),
        method: PortfolioOptimizationMethod = PortfolioOptimizationMethod.SHARPE_MAXIMIZATION,
    ) -> RiskOptimizationResult: ...
    def perform_stress_test(
        self, portfolio_risk: PortfolioRisk, scenarios: List[Dict[str, Any]]
    ) -> List[StressTestResult]: ...
    def generate_risk_report(
        self,
        portfolio_risk: PortfolioRisk,
        include_optimization: bool = True,
        include_stress_tests: bool = True,
    ) -> RiskReport: ...


# Устаревшие типы для обратной совместимости
class RiskMetricsResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    volatility: float
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation: float
    exposure: float
    details: Dict[str, Any]


class PortfolioRiskResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    total_value: float
    total_risk: float
    var_95: float
    var_99: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    beta: float
    correlation_matrix: Any
    position_risks: List[Any]


class LegacyRiskOptimizationResult(TypedDict, total=False):
    """Устаревший тип для обратной совместимости."""

    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    details: Dict[str, Any]
