import pandas as pd
from shared.numpy_utils import np
# Миграция из utils/math_utils.py
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union


def calculate_fibonacci_levels(high: Decimal, low: Decimal) -> List[Decimal]:
    """Расчет уровней Фибоначчи с использованием точной арифметики."""
    diff = high - low
    return [
        high,
        high - Decimal("0.236") * diff,
        high - Decimal("0.382") * diff,
        high - Decimal("0.5") * diff,
        high - Decimal("0.618") * diff,
        high - Decimal("0.786") * diff,
        low,
    ]


def calculate_win_rate(trades: list) -> float:
    if not trades:
        return 0.0
    winning_trades = sum(1 for trade in trades if trade.get("pnl", 0) > 0)
    return winning_trades / len(trades)


# Дополнительные функции из utils/math_utils.py
ArrayLike = Union[pd.Series, np.ndarray]


def normalize(data: ArrayLike, method: str = "minmax") -> np.ndarray:
    """
    Normalize data using specified method
    Args:
        data: Input data
        method: 'minmax' or 'zscore'
    Returns:
        Normalized data as numpy array
    """
    data = np.asarray(data)
    if method == "minmax":
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == "zscore":
        return (data - np.mean(data)) / np.std(data)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def zscore(data: ArrayLike, window: Optional[int] = None) -> np.ndarray:
    """
    Calculate Z-score with optional rolling window
    Args:
        data: Input data
        window: Rolling window size (None for global)
    Returns:
        Z-scores as numpy array
    """
    data = np.asarray(data)
    if window is None:
        return (data - np.mean(data)) / np.std(data)
    else:
        result = np.zeros_like(data)
        for i in range(len(data)):
            if i < window:
                result[i] = np.nan
            else:
                window_data = data[i - window : i]
                result[i] = (data[i] - np.mean(window_data)) / np.std(window_data)
        return result


def pct_change(data: ArrayLike, periods: int = 1) -> np.ndarray:
    """
    Calculate percentage change
    Args:
        data: Input data
        periods: Number of periods to shift
    Returns:
        Percentage changes as numpy array
    """
    data = np.asarray(data)
    return (data - np.roll(data, periods)) / np.roll(data, periods)


def correlation(x: ArrayLike, y: ArrayLike, window: Optional[int] = None) -> np.ndarray:
    """
    Calculate correlation between two series
    Args:
        x: First series
        y: Second series
        window: Rolling window size (None for global)
    Returns:
        Correlation values as numpy array
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if window is None:
        return np.corrcoef(x, y)[0, 1]
    else:
        result = np.zeros(len(x))
        for i in range(len(x)):
            if i < window:
                result[i] = np.nan
            else:
                window_x = x[i - window : i]
                window_y = y[i - window : i]
                result[i] = np.corrcoef(window_x, window_y)[0, 1]
        return result


def covariance(x: ArrayLike, y: ArrayLike, window: Optional[int] = None) -> np.ndarray:
    """
    Calculate covariance between two series
    Args:
        x: First series
        y: Second series
        window: Rolling window size (None for global)
    Returns:
        Covariance values as numpy array
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if window is None:
        return np.cov(x, y)[0, 1]
    else:
        result = np.zeros(len(x))
        for i in range(len(x)):
            if i < window:
                result[i] = np.nan
            else:
                window_x = x[i - window : i]
                window_y = y[i - window : i]
                result[i] = np.cov(window_x, window_y)[0, 1]
        return result


@dataclass
class PerformanceMetrics:
    """Performance metrics for trading strategy"""

    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    drawdown_duration: int


def calculate_performance(
    returns: ArrayLike, risk_free_rate: float = 0.0
) -> PerformanceMetrics:
    """
    Calculate performance metrics
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
    Returns:
        PerformanceMetrics object
    """
    returns = np.asarray(returns)
    # CAGR
    total_return = np.prod(1 + returns) - 1
    years = len(returns) / 252  # Assuming daily data
    cagr = (1 + total_return) ** (1 / years) - 1
    # Sharpe Ratio
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    # Sortino Ratio
    downside_returns = returns[returns < 0]
    sortino = np.sqrt(252) * np.mean(excess_returns) / np.std(downside_returns)
    # Drawdown
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    # Drawdown duration
    drawdown_indices = np.where(drawdown == max_drawdown)[0]
    if len(drawdown_indices) > 0:
        drawdown_duration = int(drawdown_indices[-1] - drawdown_indices[0])
    else:
        drawdown_duration = 0
    return PerformanceMetrics(
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        drawdown_duration=drawdown_duration,
    )


def calculate_drawdown(returns: ArrayLike) -> Tuple[np.ndarray, float, int]:
    """
    Calculate drawdown series and statistics
    Args:
        returns: Array of returns
    Returns:
        Tuple of (drawdown series, max drawdown, max drawdown duration)
    """
    returns = np.asarray(returns)
    cumulative_returns = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    max_drawdown_idx = np.where(drawdown == max_drawdown)[0]
    max_drawdown_duration = max_drawdown_idx[-1] - max_drawdown_idx[0]
    return drawdown, max_drawdown, max_drawdown_duration


def calculate_sharpe_ratio(
    returns: Union[List[float], np.ndarray, pd.Series], risk_free_rate: float = 0.0
) -> float:
    """
    Расчет коэффициента Шарпа.
    Args:
        returns: Доходности
        risk_free_rate: Безрисковая ставка
    Returns:
        float: Коэффициент Шарпа
    """
    if isinstance(returns, list):
        returns_array = np.array(returns)
    elif isinstance(returns, pd.Series):
        returns_array = returns.to_numpy() if hasattr(returns, 'to_numpy') else np.asarray(returns)
    else:
        returns_array = np.asarray(returns)
    excess_returns = returns_array - risk_free_rate
    if len(excess_returns) == 0:
        return 0.0
    result = (
        np.mean(excess_returns) / np.std(excess_returns)
        if np.std(excess_returns) != 0
        else 0.0
    )
    return float(result)


def calculate_drawdown_metrics(
    equity_curve: Union[List[float], np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Расчет просадки.
    Args:
        equity_curve: Кривая капитала
    Returns:
        Dict: Максимальная просадка и её длительность
    """
    if isinstance(equity_curve, list):
        equity_array = np.array(equity_curve)
    elif isinstance(equity_curve, pd.Series):
        equity_array = equity_curve.to_numpy() if hasattr(equity_curve, 'to_numpy') else np.asarray(equity_curve)
    else:
        equity_array = np.asarray(equity_curve)
    # Расчет максимумов
    running_max = np.maximum.accumulate(equity_array)
    # Расчет просадок
    drawdowns = (equity_array - running_max) / running_max
    # Максимальная просадка
    max_drawdown = np.min(drawdowns)
    # Длительность максимальной просадки
    drawdown_periods = np.where(drawdowns == max_drawdown)[0]
    if len(drawdown_periods) > 0:
        max_drawdown_duration = len(drawdown_periods)
    else:
        max_drawdown_duration = 0
    return {
        "max_drawdown": float(max_drawdown),
        "max_drawdown_duration": float(max_drawdown_duration),
    }


def calculate_support_resistance(
    prices: List[Decimal], window: int = 20
) -> Tuple[List[Decimal], List[Decimal]]:
    """Расчет уровней поддержки и сопротивления с точной арифметикой"""
    if len(prices) < window:
        return [], []
    supports = []
    resistances = []
    for i in range(window, len(prices)):
        window_prices = prices[i - window : i]
        local_min = min(window_prices)
        local_max = max(window_prices)
        if prices[i] == local_min:
            supports.append(prices[i])
        if prices[i] == local_max:
            resistances.append(prices[i])
    return supports, resistances


def calculate_liquidity_impact(
    order_size: Decimal, 
    orderbook_data: List[Tuple[Decimal, Decimal]], 
    side: str = "buy"
) -> Dict[str, Decimal]:
    """
    Расчет влияния ордера на ликвидность и ожидаемого проскальзывания.
    
    Args:
        order_size: Размер ордера
        orderbook_data: Список (цена, объем) для соответствующей стороны
        side: "buy" или "sell"
    
    Returns:
        Словарь с метриками влияния на ликвидность
    """
    if not orderbook_data or order_size <= 0:
        return {
            "slippage_pct": Decimal("0"),
            "avg_price": Decimal("0"),
            "liquidity_consumed_pct": Decimal("0"),
            "market_impact": Decimal("0")
        }
    
    # Сортируем ордербук: для покупки по возрастанию цены, для продажи по убыванию
    sorted_orders = sorted(orderbook_data, key=lambda x: x[0], reverse=(side == "sell"))
    
    remaining_size = order_size
    total_cost = Decimal("0")
    consumed_levels = 0
    total_available_volume = sum(vol for _, vol in sorted_orders)
    
    best_price = sorted_orders[0][0] if sorted_orders else Decimal("0")
    
    # Проходим по уровням ордербука
    for price, volume in sorted_orders:
        if remaining_size <= 0:
            break
            
        fill_volume = min(remaining_size, volume)
        total_cost += fill_volume * price
        remaining_size -= fill_volume
        consumed_levels += 1
        
        if remaining_size <= 0:
            break
    
    if order_size == remaining_size:  # Не удалось исполнить ордер
        return {
            "slippage_pct": Decimal("100"),  # Максимальное проскальзывание
            "avg_price": Decimal("0"),
            "liquidity_consumed_pct": Decimal("100"),
            "market_impact": Decimal("100")
        }
    
    filled_size = order_size - remaining_size
    avg_price = total_cost / filled_size if filled_size > 0 else Decimal("0")
    
    # Расчет проскальзывания
    slippage_pct = abs((avg_price - best_price) / best_price * 100) if best_price > 0 else Decimal("0")
    
    # Расчет потребленной ликвидности
    liquidity_consumed_pct = (filled_size / total_available_volume * 100) if total_available_volume > 0 else Decimal("0")
    
    # Оценка влияния на рынок (упрощенная)
    market_impact = min(slippage_pct * 2, Decimal("100"))  # Влияние больше проскальзывания
    
    return {
        "slippage_pct": slippage_pct,
        "avg_price": avg_price,
        "liquidity_consumed_pct": liquidity_consumed_pct,
        "market_impact": market_impact,
        "filled_size": filled_size,
        "remaining_size": remaining_size,
        "levels_consumed": consumed_levels
    }


def should_split_order(
    order_size: Decimal, 
    orderbook_data: List[Tuple[Decimal, Decimal]], 
    max_slippage_pct: Decimal = Decimal("0.5"),
    max_liquidity_impact_pct: Decimal = Decimal("10")
) -> Dict[str, Union[bool, List[Decimal]]]:
    """
    Определяет, нужно ли разбить крупный ордер на части.
    
    Args:
        order_size: Размер ордера
        orderbook_data: Данные ордербука
        max_slippage_pct: Максимально допустимое проскальзывание (%)
        max_liquidity_impact_pct: Максимальное влияние на ликвидность (%)
    
    Returns:
        Словарь с рекомендациями по разбиению ордера
    """
    impact = calculate_liquidity_impact(order_size, orderbook_data)
    
    should_split = (
        impact["slippage_pct"] > max_slippage_pct or 
        impact["liquidity_consumed_pct"] > max_liquidity_impact_pct
    )
    
    suggested_chunks = []
    if should_split:
        # Простая стратегия разбиения на равные части
        num_chunks = min(5, max(2, int(impact["liquidity_consumed_pct"] / max_liquidity_impact_pct)))
        chunk_size = order_size / num_chunks
        suggested_chunks = [chunk_size] * num_chunks
    
    return {
        "should_split": should_split,
        "suggested_chunks": suggested_chunks,
        "reason": f"Slippage: {impact['slippage_pct']:.2f}%, Liquidity impact: {impact['liquidity_consumed_pct']:.2f}%",
        "total_impact": impact
    }
