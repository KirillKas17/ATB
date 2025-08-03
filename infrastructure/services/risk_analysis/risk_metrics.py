import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Union

# Метрики риска
__all__ = [
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
]
TRADING_DAYS = 252


def calc_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    return float(np.std(returns_array) * np.sqrt(TRADING_DAYS))


def calc_sharpe(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    excess_return = float(np.mean(returns_array)) - risk_free_rate / TRADING_DAYS
    volatility = float(np.std(returns_array)) * np.sqrt(TRADING_DAYS)
    if volatility > 0:
        return excess_return / volatility
    return 0.0


def calc_sortino(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    excess_return = float(np.mean(returns_array)) - risk_free_rate / TRADING_DAYS
    # Исправляем сравнение с pandas.Series
    downside_returns = returns_array[returns_array < 0.0]
    if len(downside_returns) == 0:
        return 0.0
    downside_deviation = float(np.std(downside_returns)) * np.sqrt(TRADING_DAYS)
    if downside_deviation > 0:
        return excess_return / downside_deviation
    return 0.0


def calc_max_drawdown(returns: pd.Series) -> float:
    if len(returns) < 2:
        return 0.0
    # Исправляем использование cumprod для pandas.Series
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    cumulative_returns = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    return float(np.min(drawdown))


def calc_alpha(returns: pd.Series, risk_free_rate: float, beta: float) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    market_return = float(np.mean(returns_array)) * TRADING_DAYS
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    actual_return = float(np.mean(returns_array)) * TRADING_DAYS
    return float(actual_return - expected_return)


def calc_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    if asset_returns.empty or market_returns.empty:
        return 0.0
    common_index = asset_returns.index.intersection(market_returns.index)
    if len(common_index) < 2:
        return 0.0
    asset_aligned = asset_returns.loc[common_index]
    market_aligned = market_returns.loc[common_index]
    # Исправляем использование numpy функций
    asset_array: np.ndarray = asset_aligned.to_numpy() if hasattr(asset_aligned, 'to_numpy') else np.asarray(asset_aligned)
    market_array: np.ndarray = market_aligned.to_numpy() if hasattr(market_aligned, 'to_numpy') else np.asarray(market_aligned)
    covariance = np.cov(asset_array, market_array)[0, 1]
    market_variance = np.var(market_array)
    if market_variance > 0:
        return float(covariance / market_variance)
    return 0.0


def calc_information_ratio(returns: pd.Series, tracking_error: float) -> float:
    if tracking_error > 0:
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        return float(np.mean(returns_array)) / tracking_error
    return 0.0


def calc_calmar(returns: pd.Series, max_drawdown: float) -> float:
    if max_drawdown != 0:
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        return float(np.mean(returns_array)) * TRADING_DAYS / abs(max_drawdown)
    return 0.0


def calc_treynor(returns: pd.Series, risk_free_rate: float, beta: float) -> float:
    if beta > 0:
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        excess_return = float(np.mean(returns_array)) - risk_free_rate / TRADING_DAYS
        return excess_return / beta
    return 0.0


def calc_downside_deviation(returns: pd.Series) -> float:
    # Исправляем сравнение с pandas.Series
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    downside_returns = returns_array[returns_array < 0.0]
    if len(downside_returns) > 0:
        return float(np.std(downside_returns)) * np.sqrt(TRADING_DAYS)
    return 0.0


def calc_upside_potential_ratio(returns: pd.Series, downside_deviation: float) -> float:
    if downside_deviation > 0:
        # Исправляем сравнение с pandas.Series
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        upside_returns = returns_array[returns_array > 0.0]
        if len(upside_returns) > 0:
            return float(np.mean(upside_returns)) / downside_deviation
    return 0.0


def calc_gain_loss_ratio(returns: pd.Series) -> float:
    # Исправляем сравнения с pandas.Series
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    gains = returns_array[returns_array > 0.0]
    losses = returns_array[returns_array < 0.0]
    if len(losses) > 0 and len(gains) > 0:
        return abs(float(np.mean(gains)) / float(np.mean(losses)))
    return 0.0


def calc_profit_factor(returns: pd.Series) -> float:
    # Исправляем сравнения с pandas.Series
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    gains = returns_array[returns_array > 0.0]
    losses = returns_array[returns_array < 0.0]
    if float(np.sum(losses)) != 0:
        return abs(float(np.sum(gains)) / float(np.sum(losses)))
    return 0.0


def calc_recovery_factor(returns: pd.Series, max_drawdown: float) -> float:
    if max_drawdown != 0:
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        total_return = float(np.sum(returns_array))
        return abs(total_return / max_drawdown)
    return 0.0


def calc_risk_adjusted_return(returns: pd.Series, volatility: float) -> float:
    if volatility > 0:
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
        return float(np.mean(returns_array)) / volatility
    return 0.0


def calc_parametric_var(returns: pd.Series, confidence_level: float) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    return float(np.percentile(returns_array, (1 - confidence_level) * 100))


def calc_parametric_cvar(returns: pd.Series, confidence_level: float) -> float:
    if len(returns) < 2:
        return 0.0
    returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    var = np.percentile(returns_array, (1 - confidence_level) * 100)
    returns_filtered = returns_array[returns_array <= var]
    if len(returns_filtered) > 0:
        return float(np.mean(returns_filtered))
    return 0.0
