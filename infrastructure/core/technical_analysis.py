"""
Технический анализ для торговых стратегий.
"""

import pandas as pd
from shared.numpy_utils import np
try:
    import talib as ta
    TALIB_AVAILABLE = True
except ImportError:
    ta = None
    TALIB_AVAILABLE = False
from typing import Dict, List, Optional, Any, Tuple, Union, cast
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from pandas import Series, DataFrame
from numpy.typing import ArrayLike

def ema(data: ArrayLike, period: int) -> Series:
    """Exponential Moving Average"""
    data_array = np.asarray(data, dtype=np.float64)
    return pd.Series(data_array, dtype=float).ewm(span=period).mean()

def sma(data: ArrayLike, period: int) -> Series:
    """Simple Moving Average"""
    data_array = np.asarray(data, dtype=np.float64)
    return pd.Series(data_array, dtype=float).rolling(window=period).mean()

def wma(data: ArrayLike, period: int) -> Series:
    """Weighted Moving Average"""
    data_array = np.asarray(data, dtype=np.float64)
    weights = np.arange(1, period + 1)
    return pd.Series(data_array, dtype=float).rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )

def rsi(data: ArrayLike, period: int = 14) -> Series:
    """Relative Strength Index"""
    data_array = np.asarray(data, dtype=np.float64)
    return ta.RSI(pd.Series(data_array, dtype=float), timeperiod=period)

def stoch_rsi(
    data: ArrayLike, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[Series, Series]:
    """Stochastic RSI"""
    rsi_data = rsi(data, period)
    rsi_array = np.asarray(rsi_data, dtype=np.float64)
    k = pd.Series(rsi_array, dtype=float).rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    return k, d

def bollinger_bands(
    data: ArrayLike, period: int = 20, std_dev: float = 2.0
) -> Tuple[Series, Series, Series]:
    """Bollinger Bands"""
    data_array = np.asarray(data, dtype=np.float64)
    series = pd.Series(data_array, dtype=float)
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def keltner_channels(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Tuple[Series, Series, Series]:
    """Keltner Channels"""
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    high_series = pd.Series(high_array, dtype=float)
    low_series = pd.Series(low_array, dtype=float)
    close_series = pd.Series(close_array, dtype=float)
    
    typical_price = (high_series + low_series + close_series) / 3
    atr_val = atr(high, low, close, atr_period)
    
    middle = typical_price.rolling(window=period).mean()
    upper = middle + (atr_val * multiplier)
    lower = middle - (atr_val * multiplier)
    
    return upper, middle, lower

@dataclass
class MACD:
    """Moving Average Convergence Divergence"""

    macd: Series
    signal: Series
    histogram: Series

def macd(
    data: ArrayLike,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACD:
    """Moving Average Convergence Divergence"""
    data_array = np.asarray(data, dtype=np.float64)
    series = pd.Series(data_array, dtype=float)
    ema_fast = series.ewm(span=fast_period).mean()
    ema_slow = series.ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    return MACD(macd_line, signal_line, histogram)

def adx(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> Series:
    """Average Directional Index"""
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    high_series = pd.Series(high_array, dtype=float)
    low_series = pd.Series(low_array, dtype=float)
    close_series = pd.Series(close_array, dtype=float)
    tr1 = high_series - low_series
    tr2: pd.Series = abs(high_series - close_series.shift(1))
    tr3: pd.Series = abs(low_series - close_series.shift(1))
    # Используем DataFrame для concat
    tr_df = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3})
    tr = tr_df.max(axis=1)
    atr_val = tr.rolling(period).mean()
    up_move = high_series - high_series.shift(1)
    down_move = low_series.shift(1) - low_series
    # Calculate directional movement
    plus_dm = up_move.where((up_move.gt(down_move)) & (up_move.gt(0)), 0.0)
    minus_dm = down_move.where((down_move.gt(up_move)) & (down_move.gt(0)), 0.0)
    plus_di = 100 * pd.Series(plus_dm, index=up_move.index, dtype=float).rolling(window=period).mean() / atr_val
    minus_di = 100 * pd.Series(minus_dm, index=down_move.index, dtype=float).rolling(window=period).mean() / atr_val
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_result = dx.rolling(period).mean()
    return adx_result

def cci(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 20
) -> Series:
    """Commodity Channel Index"""
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    tp = (pd.Series(high_array, dtype=float) + pd.Series(low_array, dtype=float) + pd.Series(close_array, dtype=float)) / 3
    tp_ma = tp.rolling(window=period).mean()
    tp_md = pd.Series(tp, dtype=float).rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - tp_ma) / (0.015 * tp_md)

def atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> Series:
    """Average True Range"""
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    return ta.ATR(pd.Series(high_array, dtype=float), pd.Series(low_array, dtype=float), pd.Series(close_array, dtype=float), timeperiod=period)

def vwap(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike
) -> Series:
    """Volume Weighted Average Price"""
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    volume_array = np.asarray(volume, dtype=np.float64)
    typical_price = (pd.Series(high_array, dtype=float) + pd.Series(low_array, dtype=float) + pd.Series(close_array, dtype=float)) / 3
    volume_series = pd.Series(volume_array, dtype=float)
    return (typical_price * volume_series).cumsum() / volume_series.cumsum()

def fractals(
    high: ArrayLike, low: ArrayLike, n: int = 2
) -> Tuple[Series, Series]:
    """Fractal Patterns"""
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    high_series = pd.Series(high_array, dtype=float)
    low_series = pd.Series(low_array, dtype=float)
    bullish_fractal = pd.Series(False, index=high_series.index)
    bearish_fractal = pd.Series(False, index=low_series.index)
    for i in range(n, len(high_series) - n):
        # Bullish fractal
        if hasattr(low_series, 'iloc') and hasattr(high_series, 'iloc'):
            try:
                # Проверка условий для bullish fractal
                left_lower = all(low_series.iloc[i] < low_series.iloc[i - j] for j in range(1, n + 1))
                right_lower = all(low_series.iloc[i] < low_series.iloc[i + j] for j in range(1, n + 1))
                
                if left_lower and right_lower:
                    if hasattr(bullish_fractal, 'iloc'):
                        bullish_fractal.iloc[i] = True
                
                # Проверка условий для bearish fractal
                left_higher = all(high_series.iloc[i] > high_series.iloc[i - j] for j in range(1, n + 1))
                right_higher = all(high_series.iloc[i] > high_series.iloc[i + j] for j in range(1, n + 1))
                
                if left_higher and right_higher:
                    if hasattr(bearish_fractal, 'iloc'):
                        bearish_fractal.iloc[i] = True
            except (IndexError, TypeError):
                # Если iloc недоступен, используем обычную индексацию
                left_lower = all(low_series[i] < low_series[i - j] for j in range(1, n + 1))
                right_lower = all(low_series[i] < low_series[i + j] for j in range(1, n + 1))
                
                if left_lower and right_lower:
                    bullish_fractal[i] = True
                
                # Bearish fractal
                left_higher = all(high_series[i] > high_series[i - j] for j in range(1, n + 1))
                right_higher = all(high_series[i] > high_series[i + j] for j in range(1, n + 1))
                
                if left_higher and right_higher:
                    bearish_fractal[i] = True
    return bullish_fractal, bearish_fractal

def volume_delta(buy_volume: ArrayLike, sell_volume: ArrayLike) -> Series:
    """Volume Delta (Buy Volume - Sell Volume)"""
    buy_array = np.asarray(buy_volume, dtype=np.float64)
    sell_array = np.asarray(sell_volume, dtype=np.float64)
    return pd.Series(buy_array, dtype=float) - pd.Series(sell_array, dtype=float)

def order_imbalance(bid_volume: ArrayLike, ask_volume: ArrayLike) -> Series:
    """Order Book Imbalance"""
    bid_array = np.asarray(bid_volume, dtype=np.float64)
    ask_array = np.asarray(ask_volume, dtype=np.float64)
    total_volume = pd.Series(bid_array, dtype=float) + pd.Series(ask_array, dtype=float)
    return (pd.Series(bid_array, dtype=float) - pd.Series(ask_array, dtype=float)) / total_volume

def calculate_fuzzy_support_resistance(
    prices: np.ndarray, window: int = 50, std_factor: float = 1.5
) -> Dict[str, List[Dict]]:
    """
    Вычисление нечетких зон поддержки и сопротивления.
    Args:
        prices: Массив цен
        window: Размер окна для расчета
        std_factor: Множитель стандартного отклонения
    Returns:
        Dict с зонами поддержки и сопротивления
    """
    if len(prices) < window:
        return {"support": [], "resistance": []}
    
    support_zones = []
    resistance_zones = []
    
    for i in range(window, len(prices)):
        window_prices = prices[i - window : i]
        current_price = prices[i]
        
        # Расчет статистик окна
        mean_price = np.mean(window_prices)
        std_price = np.std(window_prices)
        
        # Определение зон
        support_threshold = mean_price - std_factor * std_price
        resistance_threshold = mean_price + std_factor * std_price
        
        # Проверка на поддержку
        if current_price <= support_threshold:
            support_zones.append(
                {
                    "price": float(current_price),
                    "index": i,
                    "strength": float((support_threshold - current_price) / std_price),
                    "volume": 1.0,  # Placeholder
                }
            )
        
        # Проверка на сопротивление
        if current_price >= resistance_threshold:
            resistance_zones.append(
                {
                    "price": float(current_price),
                    "index": i,
                    "strength": float(
                        (current_price - resistance_threshold) / std_price
                    ),
                    "volume": 1.0,  # Placeholder
                }
            )
    
    return {"support": support_zones, "resistance": resistance_zones}

def cluster_price_levels(
    prices: np.ndarray, eps: float = 0.5, min_samples: int = 3
) -> List[Dict]:
    """
    Кластеризация ценовых уровней с помощью DBSCAN.
    Args:
        prices: Массив цен
        eps: Параметр eps для DBSCAN
        min_samples: Минимальное количество образцов для кластера
    Returns:
        Список кластеров с их характеристиками
    """
    if len(prices) < min_samples:
        return []
    
    # Подготовка данных для кластеризации
    X = prices.reshape(-1, 1)
    
    # Кластеризация
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    
    # Анализ кластеров
    clusters: List[Dict] = []
    unique_labels = set(clustering.labels_)
    
    for label in unique_labels:
        if label == -1:  # Шум
            continue
        
        cluster_mask = clustering.labels_ == label
        cluster_prices = prices[cluster_mask]
        
        clusters.append(
            {
                "label": int(label),
                "prices": cluster_prices.tolist(),
                "mean_price": float(np.mean(cluster_prices)),
                "std_price": float(np.std(cluster_prices)),
                "count": int(len(cluster_prices)),
                "min_price": float(np.min(cluster_prices)),
                "max_price": float(np.max(cluster_prices)),
            }
        )
    
    return clusters

def merge_overlapping_zones(
    zones: List[Dict], overlap_threshold: float = 0.5
) -> List[Dict]:
    """
    Объединение перекрывающихся зон.
    Args:
        zones: Список зон
        overlap_threshold: Порог перекрытия для объединения
    Returns:
        Список объединенных зон
    """
    if not zones:
        return zones
    
    merged_zones = []
    used = set()
    
    for i, zone1 in enumerate(zones):
        if i in used:
            continue
        
        current_zone = zone1.copy()
        used.add(i)
        
        for j, zone2 in enumerate(zones[i + 1:], i + 1):
            if j in used:
                continue
            
            # Проверка перекрытия
            if (
                abs(zone1["price"] - zone2["price"]) / max(zone1["price"], zone2["price"])
                <= overlap_threshold
            ):
                # Объединение зон
                current_zone["price"] = (zone1["price"] + zone2["price"]) / 2
                current_zone["strength"] = max(zone1["strength"], zone2["strength"])
                current_zone["volume"] = zone1["volume"] + zone2["volume"]
                used.add(j)
        
        merged_zones.append(current_zone)
    
    return merged_zones

def get_significant_levels(
    prices: np.ndarray,
    window: int = 50,
    std_factor: float = 1.5,
    eps: float = 0.5,
    min_samples: int = 3,
    overlap_threshold: float = 0.5,
) -> Dict[str, List[Dict]]:
    """
    Получение значимых уровней поддержки и сопротивления.
    Args:
        prices: Массив цен
        window: Размер окна для анализа
        std_factor: Множитель стандартного отклонения
        eps: Параметр eps для кластеризации
        min_samples: Минимальное количество образцов
        overlap_threshold: Порог перекрытия
    Returns:
        Dict с уровнями поддержки и сопротивления
    """
    # Получение нечетких зон
    fuzzy_zones = calculate_fuzzy_support_resistance(prices, window, std_factor)
    
    # Кластеризация уровней
    support_clusters = cluster_price_levels(
        np.array([zone["price"] for zone in fuzzy_zones["support"]]),
        eps,
        min_samples,
    )
    resistance_clusters = cluster_price_levels(
        np.array([zone["price"] for zone in fuzzy_zones["resistance"]]),
        eps,
        min_samples,
    )
    
    # Объединение перекрывающихся зон
    merged_support = merge_overlapping_zones(
        [{"price": cluster["mean_price"], "strength": 1.0, "volume": cluster["count"]} for cluster in support_clusters],
        overlap_threshold,
    )
    merged_resistance = merge_overlapping_zones(
        [{"price": cluster["mean_price"], "strength": 1.0, "volume": cluster["count"]} for cluster in resistance_clusters],
        overlap_threshold,
    )
    
    return {
        "support": merged_support,
        "resistance": merged_resistance,
    }

def calculate_fibonacci_levels(high: float, low: float) -> Dict[str, float]:
    """
    Расчет уровней Фибоначчи.
    Args:
        high: Максимальная цена
        low: Минимальная цена
    Returns:
        Dict с уровнями Фибоначчи
    """
    diff = high - low
    return {
        "0.0": low,
        "0.236": low + 0.236 * diff,
        "0.382": low + 0.382 * diff,
        "0.5": low + 0.5 * diff,
        "0.618": low + 0.618 * diff,
        "0.786": low + 0.786 * diff,
        "1.0": high,
    }

# Дополнительные функции для работы с Series
def calculate_ema(data: Series, period: int = 20) -> Series:
    """Расчет экспоненциальной скользящей средней"""
    return data.ewm(span=period).mean()

def calculate_rsi(data: Series, period: int = 14) -> Series:
    """Расчет RSI"""
    result = ta.RSI(data, timeperiod=period)
    return pd.Series(result, index=data.index, dtype=float)

def calculate_macd(
    data: Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[Series, Series, Series]:
    """Расчет MACD"""
    ema_fast = data.ewm(span=fast_period).mean()
    ema_slow = data.ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(
    data: Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[Series, Series, Series]:
    """Расчет полос Боллинджера"""
    middle = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def calculate_atr(
    high: Series, low: Series, close: Series, period: int = 14
) -> Series:
    """Расчет ATR"""
    result = ta.ATR(high, low, close, timeperiod=period)
    return pd.Series(result, index=high.index, dtype=float)

def calculate_fractals(
    high: Series, low: Series, period: int = 2
) -> Tuple[Series, Series]:
    """Расчет фракталов."""
    if len(high) < 2 * period + 1 or len(low) < 2 * period + 1:
        return pd.Series(), pd.Series()
    
    bullish_fractal = pd.Series(False, index=high.index)
    bearish_fractal = pd.Series(False, index=high.index)
    
    for i in range(period, len(high) - period):
        if hasattr(low, 'iloc') and hasattr(high, 'iloc') and hasattr(bullish_fractal, 'iloc') and hasattr(bearish_fractal, 'iloc'):
            # Бычий фрактал
            is_bullish = True
            for j in range(1, period + 1):
                if float(low.iloc[i]) >= float(low.iloc[i - j]) or float(low.iloc[i]) >= float(low.iloc[i + j]):
                    is_bullish = False
                    break
            if is_bullish:
                bullish_fractal.iloc[i] = True
            
            # Медвежий фрактал
            is_bearish = True
            for j in range(1, period + 1):
                if float(high.iloc[i]) <= float(high.iloc[i - j]) or float(high.iloc[i]) <= float(high.iloc[i + j]):
                    is_bearish = False
                    break
            if is_bearish:
                bearish_fractal.iloc[i] = True
    
    return bullish_fractal, bearish_fractal

def calculate_support_resistance(
    high: Series, low: Series, close: Series, period: int = 20
) -> Tuple[float, float]:
    """Расчет уровней поддержки и сопротивления"""
    resistance = cast(pd.Series, high.rolling(window=period).max()).iloc[-1]
    support = cast(pd.Series, low.rolling(window=period).min()).iloc[-1]
    return float(support), float(resistance)

def calculate_volume_profile(
    volume: Series, price: Series, bins: int = 10
) -> Tuple[np.ndarray, Any]:  # pd.IntervalIndex simplified
    """Расчет профиля объема"""
    price = cast(pd.Series, price)
    volume = cast(pd.Series, volume)
    
    if hasattr(price, 'groupby') and len(price) > 0:
        try:
            price_bins = pd.cut(price, bins=bins)
            volume_profile = volume.groupby(price_bins).sum()
            return volume_profile.values, volume_profile.index
        except Exception:
            return np.array([]), pd.IntervalIndex([])
    else:
        return np.array([]), pd.IntervalIndex([])

def calculate_momentum(data: Series, period: int = 14) -> Series:
    """Расчет моментума"""
    return data.pct_change(periods=period)

def calculate_volatility(data: Series, period: int = 20) -> Series:
    """Расчет волатильности"""
    return data.rolling(window=period).std()

def calculate_liquidity_zones(
    price: Series, volume: Series, window: int = 20, threshold: float = 0.1
) -> Dict[str, List[float]]:
    """
    Расчет зон ликвидности.
    Args:
        price: Series с ценами
        volume: Series с объемами
        window: Размер окна
        threshold: Порог для определения зоны
    Returns:
        Dict с зонами ликвидности
    """
    # Расчет среднего объема
    avg_volume = volume.rolling(window=window).mean()
    
    # Определение зон высокой ликвидности
    high_liquidity_mask = volume > (avg_volume * (1 + threshold))
    low_liquidity_mask = volume < (avg_volume * (1 - threshold))
    
    high_liquidity_prices = price[high_liquidity_mask].tolist()
    low_liquidity_prices = price[low_liquidity_mask].tolist()
    
    return {
        "high_liquidity": high_liquidity_prices,
        "low_liquidity": low_liquidity_prices,
    }

def calculate_market_structure(
    high: Series, low: Series, close: Series, window: int = 20
) -> Dict[str, Union[float, List[float], str]]:
    """
    Расчет структуры рынка.
    Args:
        high: Series с максимальными ценами
        low: Series с минимальными ценами
        close: Series с ценами закрытия
        window: Размер окна
    Returns:
        Dict с характеристиками структуры рынка
    """
    # Тренд
    trend = "uptrend" if close.iloc[-1] > close.iloc[-window] else "downtrend"
    
    # Волатильность
    volatility = close.pct_change().rolling(window=window).std().iloc[-1]
    
    # Диапазон
    price_range = (high.rolling(window=window).max() - low.rolling(window=window).min()).iloc[-1]
    
    # Моментум
    momentum = (close.iloc[-1] - close.iloc[-window]) / close.iloc[-window]
    
    return {
        "trend": trend,
        "volatility": float(volatility),
        "price_range": float(price_range),
        "momentum": float(momentum),
        "highs": high.rolling(window=window).max().dropna().tolist(),
        "lows": low.rolling(window=window).min().dropna().tolist(),
    }

def calculate_wave_clusters(
    prices: np.ndarray, window: int = 20, min_points: int = 3, eps: float = 0.5
) -> List[Dict[str, Union[float, List[Dict[str, Any]], int]]]:
    """
    Кластеризация волн цены.
    Args:
        prices: Массив цен
        window: Размер окна
        min_points: Минимальное количество точек
        eps: Параметр eps для DBSCAN
    Returns:
        Список кластеров волн
    """
    if len(prices) < window:
        return []
    
    waves = []
    
    for i in range(window, len(prices)):
        window_prices = prices[i - window : i]
        
        # Поиск локальных экстремумов
        local_max = np.max(window_prices)
        local_min = np.min(window_prices)
        
        # Определение волн
        if prices[i] > local_max:
            waves.append({
                "type": "impulse",
                "start": i - window,
                "end": i,
                "amplitude": float(prices[i] - local_min),
                "price": float(prices[i]),
            })
        elif prices[i] < local_min:
            waves.append({
                "type": "correction",
                "start": i - window,
                "end": i,
                "amplitude": float(local_max - prices[i]),
                "price": float(prices[i]),
            })
    
    # Кластеризация волн по амплитуде
    if len(waves) >= min_points:
        amplitudes = np.array([wave["amplitude"] for wave in waves]).reshape(-1, 1)
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(amplitudes)
        
        # Группировка волн по кластерам
        clusters: Dict[int, List[Dict[str, Any]]] = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(waves[i])
        
        # Формирование результата
        result = []
        for label, cluster_waves in clusters.items():
            if label == -1:  # Шум
                continue
            
            result.append({
                "cluster_id": int(label),
                "waves": cluster_waves,
                "avg_amplitude": float(np.mean([w["amplitude"] for w in cluster_waves])),
                "count": len(cluster_waves),
            })
        
        return result
    
    return []

def calculate_ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
) -> Dict[str, Union[Series, float, int]]:
    """Расчет индикатора Ишимоку."""
    if len(high) < max(tenkan_period, kijun_period, senkou_span_b_period):
        return {}
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = pd.Series(high).rolling(window=tenkan_period).max()
    tenkan_low = pd.Series(low).rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = pd.Series(high).rolling(window=kijun_period).max()
    kijun_low = pd.Series(low).rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    
    # Senkou Span B (Leading Span B)
    senkou_span_b_high = pd.Series(high).rolling(window=senkou_span_b_period).max()
    senkou_span_b_low = pd.Series(low).rolling(window=senkou_span_b_period).min()
    senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(displacement)
    
    # Chikou Span (Lagging Span)
    chikou_span = pd.Series(index=range(len(close)), dtype=float)
    if len(close) > displacement:
        # Безопасное копирование значений
        close_values = close[displacement:]
        if len(close_values) > 0:
            chikou_span.iloc[:-displacement] = close_values
    
    return {
        "tenkan_sen": tenkan_sen.values,
        "kijun_sen": kijun_sen.values,
        "senkou_span_a": senkou_span_a.values,
        "senkou_span_b": senkou_span_b.values,
        "chikou_span": chikou_span.values,
    }

def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[Series, Series]:
    """Расчет стохастического осциллятора."""
    if len(high) < k_period or len(low) < k_period or len(close) < k_period:
        return pd.Series(), pd.Series()
    
    high_array = np.asarray(high, dtype=np.float64)
    low_array = np.asarray(low, dtype=np.float64)
    close_array = np.asarray(close, dtype=np.float64)
    high_series = pd.Series(high_array)
    low_series = pd.Series(low_array)
    close_series = pd.Series(close_array)
    
    # %K
    lowest_low = low_series.rolling(window=k_period).min()
    highest_high = high_series.rolling(window=k_period).max()
    
    # Безопасное вычисление %K
    k_percent = pd.Series(0.0, index=close_series.index)
    for i in range(len(close_series)):
        if i >= k_period - 1:
            close_val = float(close_series.iloc[i])
            low_val = float(lowest_low.iloc[i])
            high_val = float(highest_high.iloc[i])
            
            if high_val - low_val > 0:
                k_percent.iloc[i] = 100 * (close_val - low_val) / (high_val - low_val)
    
    # %D (сглаженная %K)
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent

def calculate_obv(close: Series, volume: Series) -> Series:
    """Расчет On-Balance Volume."""
    if len(close) != len(volume) or len(close) == 0:
        return pd.Series(dtype=float)
    
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = float(volume.iloc[0])
    
    for i in range(1, len(close)):
        current_close = float(close.iloc[i])
        prev_close = float(close.iloc[i - 1])
        current_volume = float(volume.iloc[i])
        prev_obv = float(obv.iloc[i - 1])
        
        if current_close > prev_close:
            obv.iloc[i] = prev_obv + current_volume
        elif current_close < prev_close:
            obv.iloc[i] = prev_obv - current_volume
        else:
            obv.iloc[i] = prev_obv
    
    return obv

def calculate_vwap(
    high: Series, low: Series, close: Series, volume: Series
) -> Series:
    """Расчет VWAP (Volume Weighted Average Price)."""
    if len(high) != len(low) or len(high) != len(close) or len(high) != len(volume) or len(high) == 0:
        return pd.Series(dtype=float)
    
    typical_price = (high + low + close) / 3
    vwap = pd.Series(index=high.index, dtype=float)
    
    # Безопасное вычисление VWAP
    cumulative_tp_volume = 0.0
    cumulative_volume = 0.0
    
    for i in range(len(high)):
        tp_val = float(typical_price.iloc[i])
        vol_val = float(volume.iloc[i])
        
        cumulative_tp_volume += tp_val * vol_val
        cumulative_volume += vol_val
        
        if cumulative_volume > 0:
            vwap.iloc[i] = cumulative_tp_volume / cumulative_volume
        else:
            vwap.iloc[i] = 0.0
    
    return vwap

def calculate_adx(
    high: Series, low: Series, close: Series, period: int = 14
) -> Series:
    """
    Расчет Average Directional Index (ADX).
    Args:
        high: Series с максимальными ценами
        low: Series с минимальными ценами
        close: Series с ценами закрытия
        period: Период для расчета
    Returns:
        Series с ADX
    """
    result = ta.ADX(high, low, close, timeperiod=period)
    return pd.Series(result, index=high.index, dtype=float)

def calculate_volume_delta(data: Series, period: int = 20) -> Series:
    """
    Расчет дельта-объема.
    Args:
        data: Series с данными
        period: Период для расчета
    Returns:
        Series с дельта-объемом
    """
    result = data.rolling(window=period).apply(
        lambda x: np.sum(x[x > 0]) - np.sum(x[x < 0])
    )
    return pd.Series(result, index=data.index, dtype=float)
