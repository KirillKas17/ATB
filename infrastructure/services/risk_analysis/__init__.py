"""
Модуль риск-анализа.

Содержит декомпозированные компоненты для анализа рисков,
оптимизации портфеля и стресс-тестирования.
"""

# Оптимизация портфеля
from .portfolio_optimization import (
    calc_concentration_risk,
    calc_diversification_ratio,
    calc_portfolio_return,
    calc_portfolio_variance,
    calc_portfolio_volatility,
    calc_risk_contribution,
    calc_sharpe_ratio_weights,
    generate_rebalancing_recommendations,
    optimize_portfolio_weights,
)

# Рекомендации
from .recommendations import (
    assess_risk_level,
    generate_portfolio_insights,
    generate_rebalancing_suggestions,
    generate_risk_alerts,
    generate_risk_recommendations,
)

# Метрики риска
from .risk_metrics import (
    calc_alpha,
    calc_beta,
    calc_calmar,
    calc_downside_deviation,
    calc_gain_loss_ratio,
    calc_information_ratio,
    calc_max_drawdown,
    calc_parametric_cvar,
    calc_parametric_var,
    calc_profit_factor,
    calc_recovery_factor,
    calc_risk_adjusted_return,
    calc_sharpe,
    calc_sortino,
    calc_treynor,
    calc_upside_potential_ratio,
    calc_volatility,
)

# Стресс-тестирование
from .stress_testing import (
    apply_stress_scenario,
    calc_scenario_impact,
    generate_default_scenarios,
    perform_stress_test,
    validate_scenario,
)

# Утилиты
from .utils import (
    RiskAnalysisCache,
    clean_cache,
    convert_to_decimal,
    convert_to_money,
    create_empty_optimization_result,
    create_empty_portfolio_risk,
    create_empty_risk_metrics,
    extract_returns_from_market_data,
    validate_market_data,
    validate_portfolio_data,
    validate_returns_data,
)

__all__ = [
    # Risk metrics
    "calc_volatility",
    "calc_sharpe",
    "calc_sortino",
    "calc_max_drawdown",
    "calc_alpha",
    "calc_beta",
    "calc_information_ratio",
    "calc_calmar",
    "calc_treynor",
    "calc_downside_deviation",
    "calc_upside_potential_ratio",
    "calc_gain_loss_ratio",
    "calc_profit_factor",
    "calc_recovery_factor",
    "calc_risk_adjusted_return",
    "calc_parametric_var",
    "calc_parametric_cvar",
    # Portfolio optimization
    "optimize_portfolio_weights",
    "calc_portfolio_return",
    "calc_portfolio_variance",
    "calc_sharpe_ratio_weights",
    "calc_risk_contribution",
    "calc_diversification_ratio",
    "calc_concentration_risk",
    "calc_portfolio_volatility",
    "generate_rebalancing_recommendations",
    # Stress testing
    "apply_stress_scenario",
    "perform_stress_test",
    "generate_default_scenarios",
    "calc_scenario_impact",
    "validate_scenario",
    # Recommendations
    "generate_risk_recommendations",
    "generate_rebalancing_suggestions",
    "generate_risk_alerts",
    "assess_risk_level",
    "generate_portfolio_insights",
    # Utils
    "validate_returns_data",
    "validate_market_data",
    "extract_returns_from_market_data",
    "create_empty_risk_metrics",
    "create_empty_portfolio_risk",
    "create_empty_optimization_result",
    "convert_to_decimal",
    "convert_to_money",
    "clean_cache",
    "validate_portfolio_data",
    "RiskAnalysisCache",
]
