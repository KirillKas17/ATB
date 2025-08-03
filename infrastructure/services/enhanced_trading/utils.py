"""
Утилиты для enhanced trading.
Содержит вспомогательные функции для валидации, преобразований,
кэширования и общих утилит enhanced trading.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from domain.entities.order import OrderSide, OrderStatus, OrderType
from domain.value_objects import Currency, Money

__all__ = [
    "validate_order_data",
    "validate_market_data",
    "validate_strategy_data",
    "create_empty_order",
    "create_empty_execution_plan",
    "create_empty_sentiment_analysis",
    "convert_to_decimal",
    "convert_to_money",
    "clean_cache",
    "validate_trading_parameters",
    "calculate_performance_metrics",
    "normalize_data",
    "EnhancedTradingCache",
]


def validate_order_data(order: Dict[str, Any]) -> bool:
    """Валидация данных ордера."""
    if not order:
        return False
    required_fields = ["symbol", "side", "order_type", "quantity"]
    for field in required_fields:
        if field not in order:
            return False
    # Проверяем типы
    if not isinstance(order["symbol"], str) or not order["symbol"]:
        return False
    if order["side"] not in [OrderSide.BUY, OrderSide.SELL]:
        return False
    if order["order_type"] not in [
        OrderType.MARKET,
        OrderType.LIMIT,
        OrderType.STOP,
        OrderType.STOP_LIMIT,
    ]:
        return False
    if (
        not isinstance(order["quantity"], (int, float, Decimal))
        or order["quantity"] <= 0
    ):
        return False
    return True


def validate_market_data(market_data: pd.DataFrame) -> bool:
    """Валидация рыночных данных."""
    if market_data is None or market_data.empty:
        return False
    required_columns = ["open", "high", "low", "close"]
    for col in required_columns:
        if col not in market_data.columns:
            return False
    # Проверяем логику данных
    if (market_data["high"] < market_data["low"]).any():
        return False
    if (market_data["high"] < market_data["open"]).any() or (
        market_data["high"] < market_data["close"]
    ).any():
        return False
    if (market_data["low"] > market_data["open"]).any() or (
        market_data["low"] > market_data["close"]
    ).any():
        return False
    # Проверяем на отрицательные цены
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if (market_data[col] <= 0).any():
            return False
    return True


def validate_strategy_data(strategy_data: Dict[str, Any]) -> bool:
    """Валидация данных стратегии."""
    if not strategy_data:
        return False
    required_fields = ["strategy_type", "parameters"]
    for field in required_fields:
        if field not in strategy_data:
            return False
    if not strategy_data["parameters"]:
        return False
    return True


def validate_trading_parameters(parameters: Dict[str, Any]) -> bool:
    """Валидация торговых параметров."""
    if not parameters:
        return False
    # Проверяем обязательные параметры
    required_params = ["symbol", "side", "quantity"]
    for param in required_params:
        if param not in parameters:
            return False
    # Проверяем числовые параметры
    numeric_params = ["quantity", "price", "stop_loss", "take_profit"]
    for param in numeric_params:
        if param in parameters:
            value = parameters[param]
            if not isinstance(value, (int, float, Decimal)) or value <= 0:
                return False
    return True


def create_empty_order() -> Dict[str, Any]:
    """Создание пустого ордера."""
    return {
        "symbol": "",
        "side": OrderSide.BUY,
        "order_type": OrderType.MARKET,
        "quantity": Decimal("0"),
        "price": None,
        "status": OrderStatus.PENDING,
        "created_at": datetime.now(),
        "execution_time": None,
        "filled_quantity": Decimal("0"),
        "average_price": None,
        "commission": Money(Decimal("0"), Currency.USD),
        "metadata": {},
    }


def create_empty_execution_plan() -> Dict[str, Any]:
    """Создание пустого плана исполнения."""
    return {
        "strategy_type": None,
        "parameters": {},
        "execution_steps": [],
        "risk_limits": {},
        "performance_targets": {},
        "created_at": datetime.now(),
        "status": "pending",
        "progress": Decimal("0"),
        "estimated_completion": None,
    }


def create_empty_sentiment_analysis() -> Dict[str, Any]:
    """Создание пустого анализа настроений."""
    return {
        "overall_sentiment": "neutral",
        "sentiment_score": Decimal("0"),
        "confidence": "low",
        "sources": [],
        "trend": "stable",
        "timestamp": datetime.now(),
        "alerts": [],
    }


def convert_to_decimal(value: Union[float, int, str, Decimal]) -> Decimal:
    """Конвертация значения в Decimal."""
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float)):
        return Decimal(str(value))
    elif isinstance(value, str):
        return Decimal(value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to Decimal")


def convert_to_money(
    value: Union[float, int, str, Decimal], currency: Currency = Currency.USD
) -> Money:
    """Конвертация значения в Money."""
    decimal_value = convert_to_decimal(value)
    return Money(decimal_value, currency)


class EnhancedTradingCache:
    """Кэш для enhanced trading."""

    def __init__(self, ttl_hours: int = 1):
        self.cache: Dict[str, Any] = {}
        self.timestamps: Dict[str, datetime] = {}
        self.ttl = timedelta(hours=ttl_hours)

    def get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        if key not in self.cache:
            return None
        if datetime.now() - self.timestamps[key] > self.ttl:
            del self.cache[key]
            del self.timestamps[key]
            return None
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """Установка значения в кэш."""
        self.cache[key] = value
        self.timestamps[key] = datetime.now()

    def clear(self) -> None:
        """Очистка кэша."""
        self.cache.clear()
        self.timestamps.clear()

    def clean_expired(self) -> None:
        """Очистка истёкших записей."""
        current_time = datetime.now()
        expired_keys = [
            key
            for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]


def clean_cache(cache: EnhancedTradingCache) -> None:
    """Очистка кэша."""
    cache.clean_expired()


def calculate_performance_metrics(
    orders: List[Dict[str, Any]], market_data: pd.DataFrame
) -> Dict[str, Any]:
    """Расчёт метрик производительности."""
    if not orders:
        return {
            "total_orders": 0,
            "filled_orders": 0,
            "fill_rate": Decimal("0"),
            "average_execution_time": timedelta(0),
            "total_commission": Money(Decimal("0"), Currency.USD),
            "total_pnl": Money(Decimal("0"), Currency.USD),
        }
    total_orders = len(orders)
    filled_orders = sum(
        1 for order in orders if order.get("status") == OrderStatus.FILLED
    )
    fill_rate = (
        Decimal(str(filled_orders / total_orders)) if total_orders > 0 else Decimal("0")
    )
    # Среднее время исполнения
    execution_times = []
    for order in orders:
        if order.get("execution_time") and order.get("created_at"):
            execution_time = order["execution_time"] - order["created_at"]
            execution_times.append(execution_time)
    avg_execution_time = (
        sum(execution_times, timedelta(0)) / len(execution_times)
        if execution_times
        else timedelta(0)
    )
    # Общая комиссия
    total_commission = sum(
        order.get("commission", Money(Decimal("0"), Currency.USD)) for order in orders
    )
    # Общий PnL (упрощённо)
    total_pnl = Money(Decimal("0"), Currency.USD)
    for order in orders:
        if order.get("filled_quantity") and order.get("average_price"):
            # Упрощённый расчёт PnL
            pnl = (
                order["filled_quantity"] * order["average_price"] * Decimal("0.001")
            )  # 0.1% прибыль
            total_pnl += Money(pnl, Currency.USD)
    return {
        "total_orders": total_orders,
        "filled_orders": filled_orders,
        "fill_rate": fill_rate,
        "average_execution_time": avg_execution_time,
        "total_commission": total_commission,
        "total_pnl": total_pnl,
    }


def normalize_data(data: pd.Series, method: str = "minmax") -> pd.Series:
    """Нормализация данных."""
    if not isinstance(data, pd.Series) or data.empty:
        return pd.Series()
    if method == "minmax":
        min_val = data.min()
        max_val = data.max()
        if max_val > min_val:
            return (data - min_val) / (max_val - min_val)
        else:
            return data
    elif method == "zscore":
        mean_val = data.mean()
        std_val = data.std()
        if std_val > 0:
            return (data - mean_val) / std_val
        else:
            return data
    else:
        return data


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Расчёт доходностей."""
    if not isinstance(prices, pd.Series):
        return pd.Series()
    return prices.pct_change().dropna()


def calculate_volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Расчёт волатильности."""
    if not isinstance(prices, pd.Series):
        return pd.Series()
    returns = calculate_returns(prices)
    return returns.rolling(window=window).std()


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Расчёт коэффициента Шарпа."""
    if returns.empty:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # Дневная безрисковая ставка
    if returns.std() > 0:
        return float(excess_returns.mean() / returns.std() * np.sqrt(252))
    else:
        return 0.0


def calculate_max_drawdown(returns: pd.Series) -> float:
    """Расчёт максимальной просадки."""
    if returns.empty:
        return 0.0
    cumulative_returns = (1 + returns).cumprod()
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    return float(drawdown.min())


def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт корреляционной матрицы."""
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        return pd.DataFrame()
    # Удаляем строки с NaN
    clean_df = returns_df.dropna()
    if len(clean_df) < 20:
        return pd.DataFrame()
    # Рассчитываем корреляцию
    correlation_matrix = clean_df.corr()
    # Заполняем NaN значения нулями
    correlation_matrix = correlation_matrix.fillna(0)
    return correlation_matrix


def detect_outliers(
    data: pd.Series, method: str = "iqr", threshold: float = 1.5
) -> pd.Series:
    """Обнаружение выбросов в данных."""
    if not isinstance(data, pd.Series) or data.empty:
        return pd.Series()
    
    if method == "iqr":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        # Определение выбросов
        outliers_result = data[(data < lower_bound) | (data > upper_bound)]
    elif method == "zscore":
        z_scores = np.abs((data - data.mean()) / data.std())
        outliers_result = z_scores > float(threshold)
    else:
        outliers_result = pd.Series(False, index=data.index)
    return outliers_result


def smooth_data(data: pd.Series, method: str = "sma", window: int = 5) -> pd.Series:
    """Сглаживание данных."""
    if not isinstance(data, pd.Series) or data.empty:
        return pd.Series()
    if method == "sma":
        return data.rolling(window=window).mean()
    elif method == "ema":
        return data.ewm(span=window).mean()
    elif method == "median":
        return data.rolling(window=window).median()
    else:
        return data


def calculate_moving_average(
    prices: pd.Series, window: int, method: str = "sma"
) -> pd.Series:
    """Расчёт скользящих средних."""
    if not isinstance(prices, pd.Series) or prices.empty:
        return pd.Series()
    if method == "sma":
        return prices.rolling(window=window).mean()
    elif method == "ema":
        return prices.ewm(span=window).mean()
    elif method == "wma":
        weights = np.arange(1, window + 1)
        return prices.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    else:
        return prices.rolling(window=window).mean()


def calculate_bollinger_bands(
    prices: pd.Series, window: int = 20, std_dev: float = 2.0
) -> Dict[str, pd.Series]:
    """Расчёт полос Боллинджера."""
    if not isinstance(prices, pd.Series) or prices.empty:
        return {"upper": pd.Series(), "middle": pd.Series(), "lower": pd.Series()}
    middle = calculate_moving_average(prices, window)
    std = prices.rolling(window=window).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return {"upper": upper, "middle": middle, "lower": lower}


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Расчёт RSI."""
    if not isinstance(prices, pd.Series) or prices.empty:
        return pd.Series()
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = delta.where(delta < 0, 0).abs()
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Dict[str, pd.Series]:
    """Расчёт MACD."""
    if not isinstance(prices, pd.Series) or prices.empty:
        return {"macd": pd.Series(), "signal": pd.Series(), "histogram": pd.Series()}
    ema_fast = calculate_moving_average(prices, fast_period, "ema")
    ema_slow = calculate_moving_average(prices, slow_period, "ema")
    macd_line = ema_fast - ema_slow
    signal_line = calculate_moving_average(macd_line, signal_period, "ema")
    histogram = macd_line - signal_line
    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Расчёт Average True Range."""
    if not all(isinstance(s, pd.Series) for s in [high, low, close]) or high.empty or low.empty or close.empty:
        return pd.Series()
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = calculate_moving_average(tr, period, "ema")
    return atr


def calculate_volume_profile(
    market_data: pd.DataFrame, price_levels: int = 100
) -> Dict[str, np.ndarray]:
    """Расчёт профиля объёма."""
    if not isinstance(market_data, pd.DataFrame) or market_data.empty or "volume" not in market_data.columns:
        return {}
    high = market_data["high"]
    low = market_data["low"]
    close = market_data["close"]
    volume = market_data["volume"]
    price_range = high.max() - low.min()
    price_step = price_range / price_levels
    volume_profile = np.zeros(price_levels)
    price_levels_array = np.linspace(low.min(), high.max(), price_levels)
    for i in range(len(close.index)):
        price_level = int((close.iloc[i] - low.min()) / price_step)
        if 0 <= price_level < price_levels:
            volume_profile[price_level] += volume.iloc[i]
    return {"price_levels": price_levels_array, "volume_profile": volume_profile}
