import numpy as np
import pandas as pd

"""
Модуль оптимизации портфеля.
Содержит промышленные функции для оптимизации портфеля,
расчёта весов, декомпозиции риска и метрик диверсификации.
"""

from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from scipy.optimize import Bounds, LinearConstraint, minimize

from domain.types.risk_types import PortfolioOptimizationMethod

__all__ = [
    "optimize_portfolio_weights",
    "calc_portfolio_return",
    "calc_portfolio_variance",
    "calc_sharpe_ratio_weights",
    "calc_risk_contribution",
    "calc_diversification_ratio",
    "calc_concentration_risk",
    "calc_portfolio_volatility",
    "generate_rebalancing_recommendations",
]


def optimize_portfolio_weights(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    method: PortfolioOptimizationMethod = PortfolioOptimizationMethod.SHARPE_MAXIMIZATION,
    target_return: Optional[float] = None,
    max_weight: float = 0.3,
    min_weight: float = 0.01,
) -> Tuple[np.ndarray, bool, float]:
    """
    Оптимизация весов портфеля.
    Returns:
        Tuple[weights, success, optimization_time]
    """
    n_assets = len(expected_returns)
    # Целевая функция
    if method == PortfolioOptimizationMethod.SHARPE_MAXIMIZATION:
        objective = lambda w: -calc_sharpe_ratio_weights(
            w, expected_returns, covariance_matrix, risk_free_rate
        )
    elif method == PortfolioOptimizationMethod.RISK_MINIMIZATION:
        objective = lambda w: calc_portfolio_variance(w, covariance_matrix)
    elif method == PortfolioOptimizationMethod.RETURN_MAXIMIZATION:
        objective = lambda w: -calc_portfolio_return(w, expected_returns)
    else:
        objective = lambda w: -calc_sharpe_ratio_weights(
            w, expected_returns, covariance_matrix, risk_free_rate
        )
    # Ограничения
    bounds = Bounds([min_weight] * n_assets, [max_weight] * n_assets)
    constraints = [LinearConstraint(np.ones(n_assets), 1.0, 1.0)]  # Сумма весов = 1
    if target_return is not None:
        # Исправление: безопасное получение numpy array из Series
        if hasattr(expected_returns, 'values'):
            returns_array: np.ndarray = expected_returns.to_numpy()
        else:
            returns_array_alt: np.ndarray = np.array(expected_returns)
        constraints.append(
            LinearConstraint(returns_array, target_return, target_return)
        )
    # Начальные веса (равные)
    initial_weights = np.ones(n_assets) / n_assets
    # Оптимизация
    result = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )
    return result.x, result.success, 0.0  # Упрощённо без времени


def calc_portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """Расчёт ожидаемой доходности портфеля."""
    # Исправлено: безопасное извлечение значений из Series
    if hasattr(expected_returns, 'to_numpy'):
        returns_array_np = expected_returns.to_numpy()
    elif hasattr(expected_returns, 'values'):
        returns_array_np = expected_returns.values
    else:
        returns_array_np = np.array(expected_returns)
    return float(np.dot(weights, returns_array_np))


def calc_portfolio_variance(
    weights: np.ndarray, covariance_matrix: pd.DataFrame
) -> float:
    """Расчёт дисперсии портфеля."""
    # Исправлено: безопасное извлечение значений из DataFrame
    if hasattr(covariance_matrix, 'to_numpy'):
        cov_array = covariance_matrix.to_numpy()
    elif hasattr(covariance_matrix, 'values'):
        cov_array = covariance_matrix.values
    else:
        cov_array = np.array(covariance_matrix)
    return float(np.dot(weights.T, np.dot(cov_array, weights)))


def calc_portfolio_volatility(
    weights: np.ndarray, covariance_matrix: pd.DataFrame
) -> float:
    """Расчёт волатильности портфеля."""
    return float(np.sqrt(calc_portfolio_variance(weights, covariance_matrix)))


def calc_sharpe_ratio_weights(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Расчёт коэффициента Шарпа для весов."""
    portfolio_return = calc_portfolio_return(weights, expected_returns)
    portfolio_volatility = calc_portfolio_volatility(weights, covariance_matrix)
    if portfolio_volatility > 0:
        return float((portfolio_return - risk_free_rate) / portfolio_volatility)
    return 0.0


def calc_risk_contribution(
    weights: np.ndarray, covariance_matrix: pd.DataFrame
) -> Dict[str, Decimal]:
    """Расчёт вклада в риск."""
    portfolio_variance = calc_portfolio_variance(weights, covariance_matrix)
    if portfolio_variance > 0:
        # Исправлено: безопасное извлечение значений из DataFrame
        if hasattr(covariance_matrix, 'to_numpy'):
            cov_array = covariance_matrix.to_numpy()
        elif hasattr(covariance_matrix, 'values'):
            cov_array = covariance_matrix.values
        else:
            cov_array = np.array(covariance_matrix)
        marginal_contributions = np.dot(cov_array, weights)
        risk_contributions = (
            weights * marginal_contributions / np.sqrt(portfolio_variance)
        )
        return {
            f"asset_{i}": Decimal(str(contrib))
            for i, contrib in enumerate(risk_contributions)
        }
    return {}


def calc_diversification_ratio(
    weights: np.ndarray, volatilities: np.ndarray, correlation_matrix: pd.DataFrame
) -> float:
    """Расчёт коэффициента диверсификации."""
    weighted_vol = np.sum(weights * volatilities)
    # Исправлено: безопасное извлечение значений из DataFrame
    if hasattr(correlation_matrix, 'to_numpy'):
        corr_array = correlation_matrix.to_numpy()
    elif hasattr(correlation_matrix, 'values'):
        corr_array = correlation_matrix.values
    else:
        corr_array = np.array(correlation_matrix)
    portfolio_vol = np.sqrt(
        np.dot(
            weights.T,
            np.dot(
                corr_array * np.outer(volatilities, volatilities),
                weights,
            ),
        )
    )
    if portfolio_vol > 0:
        return float(weighted_vol / portfolio_vol)
    return 1.0


def calc_concentration_risk(weights: np.ndarray) -> float:
    """Расчёт риска концентрации (Herfindahl index)."""
    return float(np.sum(weights**2))


def generate_rebalancing_recommendations(
    weights: Dict[str, Decimal], max_weight: float = 0.3, min_weight: float = 0.01
) -> List[str]:
    """Генерация рекомендаций по ребалансировке."""
    recommendations = []
    for asset, weight in weights.items():
        if weight > max_weight:
            recommendations.append(
                f"Reduce position in {asset}: {weight:.1%} > {max_weight:.1%}"
            )
        elif weight < min_weight:
            recommendations.append(
                f"Increase position in {asset}: {weight:.1%} < {min_weight:.1%}"
            )
    return recommendations
