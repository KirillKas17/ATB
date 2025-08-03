"""
Технический анализ для торговых стратегий.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

from numpy.typing import ArrayLike
from sklearn.cluster import DBSCAN


def ema(data: Union[np.ndarray, List[float], pd.Series], period: int) -> pd.Series:
    """Экспоненциальная скользящая средняя"""
    return pd.Series(data).ewm(span=period, adjust=False).mean()


def sma(data: Union[np.ndarray, List[float], pd.Series], period: int) -> pd.Series:
    """Простая скользящая средняя"""
    return pd.Series(data).rolling(window=period).mean()


def wma(data: Union[np.ndarray, List[float], pd.Series], period: int) -> pd.Series:
    """Взвешенная скользящая средняя"""
    weights = np.arange(1, period + 1)
    return pd.Series(data).rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def rsi(data: Union[np.ndarray, List[float], pd.Series], period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = pd.Series(data).diff()
    if hasattr(delta, 'gt') and hasattr(delta, 'lt'):
        gain = delta.where(delta.gt(0), 0).rolling(window=period).mean()
        loss = delta.where(delta.lt(0), 0).abs().rolling(window=period).mean()
    else:
        gain = pd.Series()
        loss = pd.Series()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def stoch_rsi(
    data: Union[np.ndarray, List[float], pd.Series], period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI"""
    rsi_values = rsi(data, period)
    stoch_k = (rsi_values - rsi_values.rolling(window=period).min()) / (
        rsi_values.rolling(window=period).max() - rsi_values.rolling(window=period).min()
    ) * 100
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    return stoch_k, stoch_d


def bollinger_bands(
    data: Union[np.ndarray, List[float], pd.Series], period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Полосы Боллинджера"""
    sma_values = sma(data, period)
    std = pd.Series(data).rolling(window=period).std()
    upper_band = sma_values + (std * std_dev)
    lower_band = sma_values - (std * std_dev)
    return upper_band, sma_values, lower_band


def keltner_channels(
    high: Union[np.ndarray, List[float], pd.Series],
    low: Union[np.ndarray, List[float], pd.Series],
    close: Union[np.ndarray, List[float], pd.Series],
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Каналы Келтнера"""
    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    atr_values = atr(high, low, close, atr_period)
    middle_line = typical_price.rolling(window=period).mean()
    upper_channel = middle_line + (atr_values * multiplier)
    lower_channel = middle_line - (atr_values * multiplier)
    return upper_channel, middle_line, lower_channel


@dataclass
class MACD:
    """Moving Average Convergence Divergence"""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


def macd(
    data: Union[np.ndarray, List[float], pd.Series],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACD:
    """MACD индикатор"""
    ema_fast = ema(data, fast_period)
    ema_slow = ema(data, slow_period)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return MACD(macd=macd_line, signal=signal_line, histogram=histogram)


def adx(
    high: Union[np.ndarray, List[float], pd.Series], 
    low: Union[np.ndarray, List[float], pd.Series], 
    close: Union[np.ndarray, List[float], pd.Series], 
    period: int = 14
) -> pd.Series:
    """Average Directional Index"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # True Range
    tr1 = high_series - low_series
    tr2: pd.Series = abs(high_series - close_series.shift(1))
    tr3: pd.Series = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high_series - high_series.shift(1)
    dm_minus = low_series.shift(1) - low_series
    
    # Используем pandas методы вместо numpy
    if isinstance(dm_plus, pd.Series) and isinstance(dm_minus, pd.Series):
        dm_plus_condition = (dm_plus.gt(dm_minus)) & (dm_plus.gt(0))
        dm_minus_condition = (dm_minus.gt(dm_plus)) & (dm_minus.gt(0))
        
        dm_plus = dm_plus.where(dm_plus_condition, 0)
        dm_minus = dm_minus.where(dm_minus_condition, 0)
    # Fallback для не-pandas типов
    dm_plus = pd.Series(0, index=dm_plus.index if hasattr(dm_plus, 'index') else range(len(dm_plus)))
    dm_minus = pd.Series(0, index=dm_minus.index if hasattr(dm_minus, 'index') else range(len(dm_minus)))
    
    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # DI values
    di_plus = (dm_plus_smooth / tr_smooth) * 100
    di_minus = (dm_minus_smooth / tr_smooth) * 100
    
    # DX and ADX
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus) * 100
    adx_values = dx.rolling(window=period).mean()
    
    return adx_values


def cci(
    high: Union[np.ndarray, List[float], pd.Series], 
    low: Union[np.ndarray, List[float], pd.Series], 
    close: Union[np.ndarray, List[float], pd.Series], 
    period: int = 20
) -> pd.Series:
    """Commodity Channel Index"""
    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    sma_tp = sma(typical_price, period)
    mean_deviation = typical_price.rolling(window=period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical_price - sma_tp) / (0.015 * mean_deviation)


def atr(
    high: Union[np.ndarray, List[float], pd.Series], 
    low: Union[np.ndarray, List[float], pd.Series], 
    close: Union[np.ndarray, List[float], pd.Series], 
    period: int = 14
) -> pd.Series:
    """Average True Range"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    tr1 = high_series - low_series
    tr2: pd.Series = (high_series - close_series.shift(1)).abs()
    tr3: pd.Series = (low_series - close_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(window=period).mean()


def vwap(
    high: Union[np.ndarray, List[float], pd.Series], 
    low: Union[np.ndarray, List[float], pd.Series], 
    close: Union[np.ndarray, List[float], pd.Series], 
    volume: Union[np.ndarray, List[float], pd.Series]
) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    volume_series = pd.Series(volume)
    return (typical_price * volume_series).cumsum() / volume_series.cumsum()


def fractals(
    high: Union[np.ndarray, List[float], pd.Series], 
    low: Union[np.ndarray, List[float], pd.Series], 
    n: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """Фракталы Билла Вильямса"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Bullish fractals (верхние фракталы)
    bullish_fractals: pd.Series = pd.Series(False, index=high_series.index)
    for i in range(n, len(high_series) - n):
        if len(high_series) > i and len(bullish_fractals) > i:
            # Проверка условий для bullish fractal
            left_higher = all(high_series.iloc[i] > high_series.iloc[i-j] for j in range(1, n+1))
            right_higher = all(high_series.iloc[i] > high_series.iloc[i+j] for j in range(1, n+1))
            
            if left_higher and right_higher:
                bullish_fractals.iloc[i] = True
    
    # Bearish fractals (нижние фракталы)
    bearish_fractals: pd.Series = pd.Series(False, index=low_series.index)
    for i in range(n, len(low_series) - n):
        if len(low_series) > i and len(bearish_fractals) > i:
            # Проверка условий для bearish fractal
            left_lower = all(low_series.iloc[i] < low_series.iloc[i-j] for j in range(1, n+1))
            right_lower = all(low_series.iloc[i] < low_series.iloc[i+j] for j in range(1, n+1))
            
            if left_lower and right_lower:
                bearish_fractals.iloc[i] = True
    
    return bullish_fractals, bearish_fractals


def volume_delta(buy_volume: Union[np.ndarray, List[float], pd.Series], sell_volume: Union[np.ndarray, List[float], pd.Series]) -> pd.Series:
    """Дельта объема"""
    return pd.Series(buy_volume) - pd.Series(sell_volume)


def order_imbalance(bid_volume: Union[np.ndarray, List[float], pd.Series], ask_volume: Union[np.ndarray, List[float], pd.Series]) -> pd.Series:
    """Дисбаланс ордеров"""
    return pd.Series(bid_volume) - pd.Series(ask_volume)


def calculate_fuzzy_support_resistance(
    prices: np.ndarray, window: int = 50, std_factor: float = 1.5
) -> Dict[str, List[Dict]]:
    """
    Расчет нечетких зон поддержки и сопротивления.
    Args:
        prices: Массив цен
        window: Размер окна для анализа
        std_factor: Множитель стандартного отклонения
    Returns:
        Словарь с зонами поддержки и сопротивления
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
    clusters = []
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
        overlap_threshold: Порог перекрытия
    Returns:
        Список объединенных зон
    """
    if not zones:
        return []
    
    # Сортируем зоны по цене
    sorted_zones = sorted(zones, key=lambda x: x["price"])
    merged_zones = []
    
    current_zone = sorted_zones[0].copy()
    
    for zone in sorted_zones[1:]:
        # Проверяем перекрытие
        price_diff = abs(zone["price"] - current_zone["price"])
        avg_price = (zone["price"] + current_zone["price"]) / 2
        overlap_ratio = price_diff / avg_price if avg_price > 0 else 0
        
        if overlap_ratio <= overlap_threshold:
            # Объединяем зоны
            current_zone["price"] = (current_zone["price"] + zone["price"]) / 2
            current_zone["strength"] = max(current_zone["strength"], zone["strength"])
            current_zone["volume"] = current_zone["volume"] + zone["volume"]
        else:
            # Добавляем текущую зону и начинаем новую
            merged_zones.append(current_zone)
            current_zone = zone.copy()
    
    # Добавляем последнюю зону
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
        eps: Параметр eps для DBSCAN
        min_samples: Минимальное количество образцов для кластера
        overlap_threshold: Порог перекрытия
    Returns:
        Словарь с зонами поддержки и сопротивления
    """
    # Получаем нечеткие зоны
    fuzzy_zones = calculate_fuzzy_support_resistance(prices, window, std_factor)
    
    # Кластеризуем уровни
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
    
    # Объединяем перекрывающиеся зоны
    merged_support = merge_overlapping_zones(
        [{"price": cluster["mean_price"], "strength": 1.0, "volume": 1.0} for cluster in support_clusters],
        overlap_threshold,
    )
    merged_resistance = merge_overlapping_zones(
        [{"price": cluster["mean_price"], "strength": 1.0, "volume": 1.0} for cluster in resistance_clusters],
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
        Словарь с уровнями Фибоначчи
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
        "1.618": high + 0.618 * diff,
        "2.618": high + 1.618 * diff,
    }


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Расчет EMA"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Расчет MACD"""
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(
    data: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Расчет полос Боллинджера"""
    sma = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Расчет ATR"""
    tr1 = high - low
    tr2: pd.Series = abs(high - close.shift(1))
    tr3: pd.Series = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_fractals(
    high: pd.Series, low: pd.Series, period: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """Расчет фракталов"""
    # Bullish fractals
    bullish_fractals: pd.Series = pd.Series(False, index=high.index)
    for i in range(period, len(high) - period):
        if len(high) > i and len(bullish_fractals) > i:
            # Проверка условий для bullish fractal
            left_higher = all(high.iloc[i] > high.iloc[i-j] for j in range(1, period+1))
            right_higher = all(high.iloc[i] > high.iloc[i+j] for j in range(1, period+1))
            
            if left_higher and right_higher:
                bullish_fractals.iloc[i] = True
    
    # Bearish fractals
    bearish_fractals: pd.Series = pd.Series(False, index=low.index)
    for i in range(period, len(low) - period):
        if len(low) > i and len(bearish_fractals) > i:
            # Проверка условий для bearish fractal
            left_lower = all(low.iloc[i] < low.iloc[i-j] for j in range(1, period+1))
            right_lower = all(low.iloc[i] < low.iloc[i+j] for j in range(1, period+1))
            
            if left_lower and right_lower:
                bearish_fractals.iloc[i] = True
    
    return bullish_fractals, bearish_fractals


def calculate_support_resistance(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> Tuple[float, float]:
    """Расчет уровней поддержки и сопротивления"""
    resistance = high.rolling(window=period).max().iloc[-1]
    support = low.rolling(window=period).min().iloc[-1]
    return float(support), float(resistance)


def calculate_volume_profile(
    volume: pd.Series, price: pd.Series, bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Расчет профиля объема"""
    # Создаем гистограмму
    price_array = np.asarray(price.values, dtype=np.float64)
    volume_array = np.asarray(volume.values, dtype=np.float64)
    hist, bin_edges = np.histogram(price_array, bins=bins, weights=volume_array)
    return hist, bin_edges


def calculate_momentum(data: pd.Series, period: int = 14) -> pd.Series:
    """Расчет моментума"""
    return data.diff(period)


def calculate_volatility(data: pd.Series, period: int = 20) -> pd.Series:
    """Расчет волатильности"""
    return data.rolling(window=period).std()


def calculate_liquidity_zones(
    price: pd.Series, volume: pd.Series, window: int = 20, threshold: float = 0.1
) -> Dict[str, List[float]]:
    """Расчет зон ликвидности"""
    # Находим уровни с высоким объемом
    volume_ma = volume.rolling(window=window).mean()
    high_volume_mask = volume > (volume_ma * (1 + threshold))
    
    # Группируем цены по уровням
    price_levels = price[high_volume_mask].unique()
    
    # Сортируем по объему
    level_volumes = []
    for level in price_levels:
        level_volume = volume[price == level].sum()
        level_volumes.append((level, level_volume))
    
    level_volumes.sort(key=lambda x: x[1], reverse=True)
    
    # Возвращаем топ уровни
    top_levels = [float(level) for level, _ in level_volumes[:10]]
    
    return {
        "liquidity_levels": top_levels,
        "volume_profile": [float(vol) for _, vol in level_volumes[:10]]
    }


def calculate_market_structure(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> Dict[str, Any]:
    """Расчет структуры рынка"""
    # Используем .values для numpy операций
    high_values = high.values
    low_values = low.values
    close_values = close.values
    
    # Расчет структуры
    structure = {
        'trend': 'neutral',
        'strength': 0.0,
        'support': float(np.min(np.asarray(low_values))),
        'resistance': float(np.max(np.asarray(high_values))),
        'volatility': float(np.std(np.asarray(close_values))),
        'momentum': float((close_values[-1] - close_values[0]) / close_values[0] * 100)
    }
    
    return structure


def calculate_wave_clusters(
    prices: np.ndarray, window: int = 20, min_points: int = 3, eps: float = 0.5
) -> List[Dict[str, Any]]:
    """Расчет кластеров волн"""
    if len(prices) < window:
        return []
    
    # Находим локальные экстремумы
    peaks = []
    troughs = []
    
    for i in range(1, len(prices) - 1):
        if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
            peaks.append((i, prices[i]))
        elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
            troughs.append((i, prices[i]))
    
    # Кластеризуем экстремумы
    all_extrema = peaks + troughs
    if len(all_extrema) < min_points:
        return []
    
    extrema_prices = np.array([price for _, price in all_extrema])
    X = extrema_prices.reshape(-1, 1)
    
    clustering = DBSCAN(eps=eps, min_samples=min_points).fit(X)
    
    # Анализируем кластеры
    clusters = []
    unique_labels = set(clustering.labels_)
    
    for label in unique_labels:
        if label == -1:  # Шум
            continue
        
        cluster_mask = clustering.labels_ == label
        cluster_extrema = [all_extrema[i] for i in range(len(all_extrema)) if cluster_mask[i]]
        
        cluster_prices = [price for _, price in cluster_extrema]
        cluster_indices = [idx for idx, _ in cluster_extrema]
        
        clusters.append({
            "label": int(label),
            "mean_price": float(np.mean(cluster_prices)),
            "std_price": float(np.std(cluster_prices)) if len(cluster_prices) > 1 else 0.0,
            "count": len(cluster_prices),
            "indices": cluster_indices,
            "prices": cluster_prices,
            "type": "peak" if len([p for p in peaks if p in cluster_extrema]) > len([t for t in troughs if t in cluster_extrema]) else "trough"
        })
    
    return clusters


def calculate_ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
) -> Dict[str, np.ndarray]:
    """Расчет Ишимоку"""
    # Используем numpy массивы напрямую
    tenkan_sen = (np.max(high[-tenkan_period:]) + np.min(low[-tenkan_period:])) / 2
    kijun_sen = (np.max(high[-kijun_period:]) + np.min(low[-kijun_period:])) / 2
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_b = (np.max(high[-senkou_span_b_period:]) + np.min(low[-senkou_span_b_period:])) / 2
    
    return {
        'tenkan_sen': np.array([tenkan_sen]),
        'kijun_sen': np.array([kijun_sen]),
        'senkou_span_a': np.array([senkou_span_a]),
        'senkou_span_b': np.array([senkou_span_b]),
        'chikou_span': np.array([close[-displacement]])
    }


def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Расчет стохастического осциллятора"""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # %K
    lowest_low = low_series.rolling(window=k_period).min()
    highest_high = high_series.rolling(window=k_period).max()
    k_percent = ((close_series - lowest_low) / (highest_high - lowest_low)) * 100
    
    # %D
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return np.array(k_percent.values), np.array(d_percent.values)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Расчет On-Balance Volume"""
    obv: pd.Series = pd.Series(index=close.index, dtype=float)
    
    # Инициализация первого значения
    if len(obv) > 0 and len(volume) > 0:
        obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if len(close) > i and len(close) > i-1 and len(obv) > i and len(obv) > i-1 and len(volume) > i:
            current_close = close.iloc[i]
            prev_close = close.iloc[i-1]
            current_volume = volume.iloc[i]
            prev_obv = obv.iloc[i-1]
            
            if current_close > prev_close:
                obv.iloc[i] = prev_obv + current_volume
            elif current_close < prev_close:
                obv.iloc[i] = prev_obv - current_volume
            else:
                obv.iloc[i] = prev_obv
    
    return obv


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Расчет VWAP"""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Расчет ADX"""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high - high.shift(1)
    dm_minus = low.shift(1) - low
    
    dm_plus = dm_plus.where((dm_plus.gt(dm_minus)) & (dm_plus.gt(0)), 0)
    dm_minus = dm_minus.where((dm_minus.gt(dm_plus)) & (dm_minus.gt(0)), 0)
    
    # Smoothed values
    tr_smooth = tr.rolling(window=period).mean()
    dm_plus_smooth = dm_plus.rolling(window=period).mean()
    dm_minus_smooth = dm_minus.rolling(window=period).mean()
    
    # DI values
    di_plus = (dm_plus_smooth / tr_smooth) * 100
    di_minus = (dm_minus_smooth / tr_smooth) * 100
    
    # DX and ADX
    dx = (di_plus - di_minus).abs() / (di_plus + di_minus) * 100
    adx_values = dx.rolling(window=period).mean()
    
    return adx_values
