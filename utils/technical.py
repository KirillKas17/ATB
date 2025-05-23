from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger
from sklearn.cluster import DBSCAN

# Type aliases
ArrayLike = Union[pd.Series, np.ndarray]


def ema(data: ArrayLike, period: int) -> pd.Series:
    """Exponential Moving Average"""
    return pd.Series(data).ewm(span=period, adjust=False).mean()


def sma(data: ArrayLike, period: int) -> pd.Series:
    """Simple Moving Average"""
    return pd.Series(data).rolling(window=period).mean()


def wma(data: ArrayLike, period: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return (
        pd.Series(data)
        .rolling(window=period)
        .apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    )


def rsi(data: ArrayLike, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    return ta.rsi(data, length=period)


def stoch_rsi(
    data: ArrayLike, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic RSI"""
    rsi_val = rsi(data, period)
    stoch = (rsi_val - rsi_val.rolling(period).min()) / (
        rsi_val.rolling(period).max() - rsi_val.rolling(period).min()
    )
    k = stoch.rolling(smooth_k).mean() * 100
    d = k.rolling(smooth_d).mean()
    return k, d


def bollinger_bands(
    data: ArrayLike, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    bb = ta.bbands(data, length=period, std=std_dev)
    return bb["BBL_20_2.0"], bb["BBM_20_2.0"], bb["BBU_20_2.0"]


def keltner_channels(
    high: ArrayLike,
    low: ArrayLike,
    close: ArrayLike,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels"""
    middle = ema(close, period)
    atr_val = atr(high, low, close, atr_period)
    upper = middle + (atr_val * multiplier)
    lower = middle - (atr_val * multiplier)
    return upper, middle, lower


@dataclass
class MACD:
    """Moving Average Convergence Divergence"""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


def macd(
    data: ArrayLike,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACD:
    """Calculate MACD"""
    macd = ta.macd(data, fast=fast_period, slow=slow_period, signal=signal_period)
    return MACD(macd["MACD_12_26_9"], macd["MACDs_12_26_9"], macd["MACDh_12_26_9"])


def adx(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> pd.Series:
    """Average Directional Index"""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx


def cci(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 20
) -> pd.Series:
    """Commodity Channel Index"""
    tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    tp_ma = sma(tp, period)
    tp_md = pd.Series(tp).rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - tp_ma) / (0.015 * tp_md)


def atr(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int = 14
) -> pd.Series:
    """Average True Range"""
    return ta.atr(high, low, close, length=period)


def vwap(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, volume: ArrayLike
) -> pd.Series:
    """Volume Weighted Average Price"""
    typical_price = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    volume = pd.Series(volume)
    return (typical_price * volume).cumsum() / volume.cumsum()


def fractals(
    high: ArrayLike, low: ArrayLike, n: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """Fractal Patterns"""
    high = pd.Series(high)
    low = pd.Series(low)

    bullish_fractal = pd.Series(False, index=high.index)
    bearish_fractal = pd.Series(False, index=low.index)

    for i in range(n, len(high) - n):
        # Bullish fractal
        if all(low[i] < low[i - j] for j in range(1, n + 1)) and all(
            low[i] < low[i + j] for j in range(1, n + 1)
        ):
            bullish_fractal.iloc[i] = True

        # Bearish fractal
        if all(high[i] > high[i - j] for j in range(1, n + 1)) and all(
            high[i] > high[i + j] for j in range(1, n + 1)
        ):
            bearish_fractal.iloc[i] = True

    return bullish_fractal, bearish_fractal


def volume_delta(buy_volume: ArrayLike, sell_volume: ArrayLike) -> pd.Series:
    """Volume Delta (Buy Volume - Sell Volume)"""
    return pd.Series(buy_volume) - pd.Series(sell_volume)


def order_imbalance(bid_volume: ArrayLike, ask_volume: ArrayLike) -> pd.Series:
    """Order Book Imbalance"""
    total_volume = pd.Series(bid_volume) + pd.Series(ask_volume)
    return (pd.Series(bid_volume) - pd.Series(ask_volume)) / total_volume


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
    try:
        # Инициализация списков для зон
        support_zones = []
        resistance_zones = []

        # Расчет скользящих статистик
        rolling_mean = pd.Series(prices).rolling(window=window).mean()
        rolling_std = pd.Series(prices).rolling(window=window).std()

        # Поиск локальных минимумов и максимумов
        for i in range(window, len(prices) - window):
            # Проверка на локальный минимум
            if all(prices[i] <= prices[i - window : i]) and all(
                prices[i] <= prices[i + 1 : i + window + 1]
            ):
                zone = {
                    "price": float(prices[i]),
                    "mean": float(rolling_mean[i]),
                    "std": float(rolling_std[i]),
                    "upper": float(rolling_mean[i] + std_factor * rolling_std[i]),
                    "lower": float(rolling_mean[i] - std_factor * rolling_std[i]),
                    "strength": float(
                        1 / (rolling_std[i] + 1e-6)
                    ),  # Сила обратно пропорциональна std
                }
                support_zones.append(zone)

            # Проверка на локальный максимум
            if all(prices[i] >= prices[i - window : i]) and all(
                prices[i] >= prices[i + 1 : i + window + 1]
            ):
                zone = {
                    "price": float(prices[i]),
                    "mean": float(rolling_mean[i]),
                    "std": float(rolling_std[i]),
                    "upper": float(rolling_mean[i] + std_factor * rolling_std[i]),
                    "lower": float(rolling_mean[i] - std_factor * rolling_std[i]),
                    "strength": float(1 / (rolling_std[i] + 1e-6)),
                }
                resistance_zones.append(zone)

        return {"support": support_zones, "resistance": resistance_zones}

    except Exception as e:
        logger.error(f"Error calculating fuzzy support/resistance: {str(e)}")
        return {"support": [], "resistance": []}


def cluster_price_levels(
    prices: np.ndarray, eps: float = 0.5, min_samples: int = 3
) -> List[Dict]:
    """
    Кластеризация ценовых уровней.

    Args:
        prices: Массив цен
        eps: Максимальное расстояние между точками в кластере
        min_samples: Минимальное количество точек в кластере

    Returns:
        List[Dict]: Список кластеров с их характеристиками
    """
    try:
        # Подготовка данных
        X = prices.reshape(-1, 1)

        # Кластеризация
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        # Анализ кластеров
        clusters = []
        for label in set(clustering.labels_):
            if label == -1:  # Пропускаем шум
                continue

            # Получение точек кластера
            cluster_points = prices[clustering.labels_ == label]

            # Расчет характеристик
            cluster = {
                "center": float(np.mean(cluster_points)),
                "std": float(np.std(cluster_points)),
                "size": len(cluster_points),
                "min": float(np.min(cluster_points)),
                "max": float(np.max(cluster_points)),
            }

            clusters.append(cluster)

        return clusters

    except Exception as e:
        logger.error(f"Error clustering price levels: {str(e)}")
        return []


def merge_overlapping_zones(
    zones: List[Dict], overlap_threshold: float = 0.5
) -> List[Dict]:
    """
    Объединение перекрывающихся зон.

    Args:
        zones: Список зон
        overlap_threshold: Порог перекрытия

    Returns:
        List[Dict]: Объединенные зоны
    """
    try:
        if not zones:
            return []

        # Сортировка зон по цене
        zones = sorted(zones, key=lambda x: x["price"])

        merged = []
        current = zones[0]

        for next_zone in zones[1:]:
            # Проверка перекрытия
            overlap = min(current["upper"], next_zone["upper"]) - max(
                current["lower"], next_zone["lower"]
            )
            total_range = max(current["upper"], next_zone["upper"]) - min(
                current["lower"], next_zone["lower"]
            )

            if overlap / total_range > overlap_threshold:
                # Объединение зон
                current = {
                    "price": (current["price"] + next_zone["price"]) / 2,
                    "mean": (current["mean"] + next_zone["mean"]) / 2,
                    "std": max(current["std"], next_zone["std"]),
                    "upper": max(current["upper"], next_zone["upper"]),
                    "lower": min(current["lower"], next_zone["lower"]),
                    "strength": max(current["strength"], next_zone["strength"]),
                }
            else:
                merged.append(current)
                current = next_zone

        merged.append(current)
        return merged

    except Exception as e:
        logger.error(f"Error merging overlapping zones: {str(e)}")
        return zones


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
        window: Размер окна для расчета
        std_factor: Множитель стандартного отклонения
        eps: Максимальное расстояние между точками в кластере
        min_samples: Минимальное количество точек в кластере
        overlap_threshold: Порог перекрытия

    Returns:
        Dict[str, List[Dict]]: Словарь с уровнями поддержки и сопротивления
    """
    try:
        # Получение нечетких зон
        fuzzy_zones = calculate_fuzzy_support_resistance(
            prices, window=window, std_factor=std_factor
        )

        # Кластеризация цен
        clusters = cluster_price_levels(prices, eps=eps, min_samples=min_samples)

        # Объединение зон
        support = merge_overlapping_zones(
            fuzzy_zones["support"]
            + [
                {
                    "price": c["center"],
                    "mean": c["center"],
                    "std": c["std"],
                    "upper": c["max"],
                    "lower": c["min"],
                    "strength": c["size"],
                }
                for c in clusters
            ],
            overlap_threshold=overlap_threshold,
        )

        resistance = merge_overlapping_zones(
            fuzzy_zones["resistance"]
            + [
                {
                    "price": c["center"],
                    "mean": c["center"],
                    "std": c["std"],
                    "upper": c["max"],
                    "lower": c["min"],
                    "strength": c["size"],
                }
                for c in clusters
            ],
            overlap_threshold=overlap_threshold,
        )

        return {"support": support, "resistance": resistance}

    except Exception as e:
        logger.error(f"Error getting significant levels: {str(e)}")
        return {"support": [], "resistance": []}


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
        "0.236": low + diff * 0.236,
        "0.382": low + diff * 0.382,
        "0.5": low + diff * 0.5,
        "0.618": low + diff * 0.618,
        "0.786": low + diff * 0.786,
        "1.0": high,
    }


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Расчет EMA"""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Расчет RSI"""
    return ta.rsi(data, length=period)


def calculate_macd(
    data: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Расчет MACD"""
    macd = ta.macd(data, fast=fast_period, slow=slow_period, signal=signal_period)
    return macd["MACD_12_26_9"], macd["MACDs_12_26_9"], macd["MACDh_12_26_9"]


def calculate_bollinger_bands(
    data: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Расчет линий Боллинджера"""
    bb = ta.bbands(data, length=period, std=std_dev)
    return bb["BBL_20_2.0"], bb["BBM_20_2.0"], bb["BBU_20_2.0"]


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Расчет ATR"""
    return ta.atr(high, low, close, length=period)


def calculate_fractals(
    high: pd.Series, low: pd.Series, period: int = 2
) -> Tuple[pd.Series, pd.Series]:
    """Расчет фракталов"""
    # Верхние фракталы
    upper_fractals = pd.Series(False, index=high.index)
    for i in range(period, len(high) - period):
        if all(high[i] > high[i - j] for j in range(1, period + 1)) and all(
            high[i] > high[i + j] for j in range(1, period + 1)
        ):
            upper_fractals[i] = True

    # Нижние фракталы
    lower_fractals = pd.Series(False, index=low.index)
    for i in range(period, len(low) - period):
        if all(low[i] < low[i - j] for j in range(1, period + 1)) and all(
            low[i] < low[i + j] for j in range(1, period + 1)
        ):
            lower_fractals[i] = True

    return upper_fractals, lower_fractals


def calculate_support_resistance(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> Tuple[float, float]:
    """Расчет уровней поддержки и сопротивления"""
    recent_high = high.rolling(window=period).max()
    recent_low = low.rolling(window=period).min()

    resistance = recent_high.mean()
    support = recent_low.mean()

    return support, resistance


def calculate_volume_profile(
    volume: pd.Series, price: pd.Series, bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Расчет профиля объема"""
    hist, bins = np.histogram(price, bins=bins, weights=volume)
    return hist, bins


def calculate_momentum(data: pd.Series, period: int = 14) -> pd.Series:
    """Расчет моментума"""
    return data.pct_change(periods=period)


def calculate_volatility(data: pd.Series, period: int = 20) -> pd.Series:
    """Расчет волатильности"""
    return data.pct_change().rolling(window=period).std()


def calculate_liquidity_zones(
    price: pd.Series, volume: pd.Series, window: int = 20, threshold: float = 0.1
) -> Dict[str, List[float]]:
    """Расчет зон ликвидности"""
    # Нормализация объема
    volume_norm = volume / volume.rolling(window=window).mean()

    # Поиск зон с высокой ликвидностью
    high_liquidity = price[volume_norm > (1 + threshold)]
    low_liquidity = price[volume_norm < (1 - threshold)]

    return {
        "high_liquidity": high_liquidity.tolist(),
        "low_liquidity": low_liquidity.tolist(),
    }


def calculate_market_structure(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> Dict[str, Any]:
    """Расчет структуры рынка"""
    # Расчет локальных максимумов и минимумов
    local_high = high.rolling(window=window, center=True).max()
    local_low = low.rolling(window=window, center=True).min()

    # Определение тренда
    trend = pd.Series(index=close.index)
    for i in range(window, len(close)):
        if close[i] > local_high[i - 1]:
            trend[i] = 1  # Восходящий тренд
        elif close[i] < local_low[i - 1]:
            trend[i] = -1  # Нисходящий тренд
        else:
            trend[i] = 0  # Боковой тренд

    return {
        "trend": trend,
        "local_high": local_high,
        "local_low": local_low,
        "structure": {
            "is_uptrend": trend.iloc[-1] == 1,
            "is_downtrend": trend.iloc[-1] == -1,
            "is_sideways": trend.iloc[-1] == 0,
        },
    }


def calculate_wave_clusters(
    prices: np.ndarray, window: int = 20, min_points: int = 3, eps: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Вычисление волновых кластеров на основе ценовых движений.

    Args:
        prices: Массив цен
        window: Размер окна для анализа
        min_points: Минимальное количество точек в кластере
        eps: Максимальное расстояние между точками в кластере

    Returns:
        List[Dict]: Список волновых кластеров с их характеристиками
    """
    try:
        # Подготовка данных
        X = prices.reshape(-1, 1)

        # Кластеризация
        clustering = DBSCAN(eps=eps, min_samples=min_points).fit(X)

        # Анализ кластеров
        wave_clusters = []
        for label in set(clustering.labels_):
            if label == -1:  # Пропускаем шум
                continue

            # Получение точек кластера
            cluster_points = prices[clustering.labels_ == label]

            # Расчет характеристик волны
            wave = {
                "start_price": float(cluster_points[0]),
                "end_price": float(cluster_points[-1]),
                "amplitude": float(abs(cluster_points[-1] - cluster_points[0])),
                "direction": "up" if cluster_points[-1] > cluster_points[0] else "down",
                "duration": len(cluster_points),
                "mean_price": float(np.mean(cluster_points)),
                "std_price": float(np.std(cluster_points)),
            }

            wave_clusters.append(wave)

        return wave_clusters

    except Exception as e:
        logger.error(f"Error calculating wave clusters: {str(e)}")
        return []


def calculate_ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    displacement: int = 26,
) -> Dict[str, np.ndarray]:
    """
    Расчет индикатора Ишимоку Кинко Хайо

    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        close: Массив цен закрытия
        tenkan_period: Период для Tenkan-sen (Conversion Line)
        kijun_period: Период для Kijun-sen (Base Line)
        senkou_span_b_period: Период для Senkou Span B
        displacement: Смещение для Senkou Span A и B

    Returns:
        Словарь с компонентами Ишимоку:
        - tenkan_sen: Conversion Line
        - kijun_sen: Base Line
        - senkou_span_a: Leading Span A
        - senkou_span_b: Leading Span B
        - chikou_span: Lagging Span
    """
    try:
        # Расчет Tenkan-sen (Conversion Line)
        tenkan_high = pd.Series(high).rolling(window=tenkan_period).max()
        tenkan_low = pd.Series(low).rolling(window=tenkan_period).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2

        # Расчет Kijun-sen (Base Line)
        kijun_high = pd.Series(high).rolling(window=kijun_period).max()
        kijun_low = pd.Series(low).rolling(window=kijun_period).min()
        kijun_sen = (kijun_high + kijun_low) / 2

        # Расчет Senkou Span A (Leading Span A)
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

        # Расчет Senkou Span B (Leading Span B)
        senkou_span_b_high = pd.Series(high).rolling(window=senkou_span_b_period).max()
        senkou_span_b_low = pd.Series(low).rolling(window=senkou_span_b_period).min()
        senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(
            displacement
        )

        # Расчет Chikou Span (Lagging Span)
        chikou_span = pd.Series(close).shift(-displacement)

        return {
            "tenkan_sen": tenkan_sen.values,
            "kijun_sen": kijun_sen.values,
            "senkou_span_a": senkou_span_a.values,
            "senkou_span_b": senkou_span_b.values,
            "chikou_span": chikou_span.values,
        }

    except Exception as e:
        logger.error(f"Ошибка при расчете Ишимоку: {e}")
        return {
            "tenkan_sen": np.zeros_like(close),
            "kijun_sen": np.zeros_like(close),
            "senkou_span_a": np.zeros_like(close),
            "senkou_span_b": np.zeros_like(close),
            "chikou_span": np.zeros_like(close),
        }


def calculate_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Расчет стохастического осциллятора

    Args:
        high: Массив максимальных цен
        low: Массив минимальных цен
        close: Массив цен закрытия
        k_period: Период для %K
        d_period: Период для %D

    Returns:
        Tuple[np.ndarray, np.ndarray]: %K и %D
    """
    try:
        # Расчет %K
        lowest_low = pd.Series(low).rolling(window=k_period).min()
        highest_high = pd.Series(high).rolling(window=k_period).max()

        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))

        # Расчет %D (скользящее среднее %K)
        d = k.rolling(window=d_period).mean()

        return k.values, d.values

    except Exception as e:
        logger.error(f"Ошибка при расчете стохастика: {e}")
        return np.zeros_like(close), np.zeros_like(close)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Расчет On Balance Volume (OBV)"""
    try:
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]

        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        return obv

    except Exception as e:
        logger.error(f"Ошибка при расчете OBV: {e}")
        return pd.Series(index=close.index)


def calculate_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Расчет Volume Weighted Average Price (VWAP)"""
    try:
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    except Exception as e:
        logger.error(f"Ошибка при расчете VWAP: {e}")
        return pd.Series(index=high.index)


def calculate_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Расчет Average Directional Index (ADX)"""
    try:
        # Расчет True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Расчет Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Расчет +DI и -DI
        tr_ema = tr.ewm(span=period, adjust=False).mean()
        plus_di = (
            100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / tr_ema
        )
        minus_di = (
            100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / tr_ema
        )

        # Расчет DX и ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    except Exception as e:
        logger.error(f"Ошибка при расчете ADX: {e}")
        return pd.Series(index=high.index)
