from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import talib
from loguru import logger
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler

from utils.logger import setup_logger

from .technical import (
    MACD,
    adx,
    atr,
    bollinger_bands,
    ema,
    fractals,
    get_significant_levels,
    keltner_channels,
    macd,
    order_imbalance,
    rsi,
    sma,
    stoch_rsi,
    volume_delta,
    vwap,
)

logger = setup_logger(__name__)

# Type aliases
ArrayLike = Union[pd.Series, np.ndarray]


@dataclass
class IndicatorConfig:
    """Конфигурация индикаторов"""

    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    fractal_period: int = 2
    cluster_window: int = 20
    volatility_window: int = 20
    volume_profile_bins: int = 24
    use_cache: bool = True
    parallel_processing: bool = True
    max_workers: int = 4


class IndicatorCalculator:
    """Калькулятор индикаторов с оптимизацией и кэшированием"""

    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or IndicatorConfig()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self._scaler = StandardScaler()

    def __del__(self):
        self._executor.shutdown(wait=True)

    @lru_cache(maxsize=128)
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех технических индикаторов с оптимизацией"""
        try:
            df = data.copy()

            if self.config.parallel_processing:
                futures = []

                # Трендовые индикаторы
                futures.append(
                    self._executor.submit(self._calculate_trend_indicators, df)
                )

                # Волатильность и моментум
                futures.append(
                    self._executor.submit(self._calculate_volatility_momentum, df)
                )

                # Объемные индикаторы
                futures.append(
                    self._executor.submit(self._calculate_volume_indicators, df)
                )

                # Уровни и структура
                futures.append(
                    self._executor.submit(self._calculate_structure_indicators, df)
                )

                # Собираем результаты
                for future in futures:
                    result = future.result()
                    df.update(result)

            else:
                df.update(self._calculate_trend_indicators(df))
                df.update(self._calculate_volatility_momentum(df))
                df.update(self._calculate_volume_indicators(df))
                df.update(self._calculate_structure_indicators(df))

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return data

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет трендовых индикаторов"""
        result = pd.DataFrame(index=df.index)

        # Скользящие средние
        result["sma_20"] = sma(df["close"], 20)
        result["sma_50"] = sma(df["close"], 50)
        result["sma_200"] = sma(df["close"], 200)

        result["ema_20"] = ema(df["close"], 20)
        result["ema_50"] = ema(df["close"], 50)
        result["ema_200"] = ema(df["close"], 200)

        # MACD
        macd_result = macd(
            df["close"],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        result["macd"] = macd_result.macd
        result["macd_signal"] = macd_result.signal
        result["macd_hist"] = macd_result.histogram

        # ADX
        result["adx"], result["plus_di"], result["minus_di"] = adx(
            df["high"], df["low"], df["close"]
        )

        return result

    def _calculate_volatility_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов волатильности и моментума"""
        result = pd.DataFrame(index=df.index)

        # Волатильность
        result["atr"] = atr(df["high"], df["low"], df["close"], self.config.atr_period)

        # Моментум
        result["rsi"] = rsi(df["close"], self.config.rsi_period)
        stoch_k, stoch_d = stoch_rsi(df["close"])
        result["stoch_k"] = stoch_k
        result["stoch_d"] = stoch_d

        # Bollinger Bands
        result["bb_upper"], result["bb_middle"], result["bb_lower"] = bollinger_bands(
            df["close"], self.config.bb_period, self.config.bb_std
        )

        # Keltner Channels
        result["kc_upper"], result["kc_middle"], result["kc_lower"] = keltner_channels(
            df["high"], df["low"], df["close"]
        )

        return result

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет объемных индикаторов"""
        result = pd.DataFrame(index=df.index)

        # VWAP
        result["vwap"] = vwap(df["high"], df["low"], df["close"], df["volume"])

        # Volume Delta
        if "buy_volume" in df.columns and "sell_volume" in df.columns:
            result["volume_delta"] = volume_delta(df["buy_volume"], df["sell_volume"])

        # Order Imbalance
        if "bid_volume" in df.columns and "ask_volume" in df.columns:
            result["order_imbalance"] = order_imbalance(
                df["bid_volume"], df["ask_volume"]
            )

        # Volume Profile
        volume_profile = self.calculate_volume_profile(df)
        result["volume_poc"] = volume_profile["poc"]
        result["volume_value_area"] = volume_profile["value_area"]

        return result

    def _calculate_structure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет структурных индикаторов"""
        result = pd.DataFrame(index=df.index)

        # Фракталы
        result["bullish_fractal"], result["bearish_fractal"] = fractals(
            df["high"], df["low"], self.config.fractal_period
        )

        # Уровни
        levels = get_significant_levels(df["close"].values)
        result["support_levels"] = [levels["support"]] * len(df)
        result["resistance_levels"] = [levels["resistance"]] * len(df)

        # Структура рынка
        market_structure = self.calculate_market_structure(df)
        result["market_structure"] = [market_structure["structure"]] * len(df)
        result["trend_strength"] = [market_structure["trend_strength"]] * len(df)

        return result

    def calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет профиля объема с оптимизацией"""
        try:
            # Нормализация цен
            price_range = data["high"].max() - data["low"].min()
            normalized_prices = (data["close"] - data["low"].min()) / price_range

            # Создание гистограммы
            hist, bins = np.histogram(
                normalized_prices,
                bins=self.config.volume_profile_bins,
                weights=data["volume"],
            )

            # Поиск POC (Point of Control)
            poc_idx = np.argmax(hist)
            poc_price = bins[poc_idx]

            # Расчет Value Area (70% объема)
            total_volume = np.sum(hist)
            target_volume = total_volume * 0.7

            sorted_volumes = np.sort(hist)[::-1]
            cumulative_volume = np.cumsum(sorted_volumes)
            value_area_idx = np.searchsorted(cumulative_volume, target_volume)

            value_area_prices = bins[: value_area_idx + 1]

            return {
                "poc": poc_price,
                "value_area": value_area_prices.tolist(),
                "histogram": hist.tolist(),
                "bins": bins.tolist(),
            }

        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {"poc": None, "value_area": [], "histogram": [], "bins": []}

    def calculate_market_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Расчет структуры рынка с продвинутым анализом"""
        try:
            # Определение тренда
            sma_20 = sma(df["close"], 20)
            sma_50 = sma(df["close"], 50)

            # Определение структуры
            if sma_20.iloc[-1] > sma_50.iloc[-1]:
                structure = "uptrend"
            elif sma_20.iloc[-1] < sma_50.iloc[-1]:
                structure = "downtrend"
            else:
                structure = "sideways"

            # Расчет силы тренда
            adx_value = adx(df["high"], df["low"], df["close"])[0].iloc[-1]
            rsi_value = rsi(df["close"]).iloc[-1]

            trend_strength = (adx_value / 100) * (abs(rsi_value - 50) / 50)

            # Определение волатильности
            atr_value = atr(df["high"], df["low"], df["close"]).iloc[-1]
            avg_price = df["close"].mean()
            volatility = atr_value / avg_price

            return {
                "structure": structure,
                "trend_strength": float(trend_strength),
                "volatility": float(volatility),
                "adx": float(adx_value),
                "rsi": float(rsi_value),
            }

        except Exception as e:
            logger.error(f"Error calculating market structure: {str(e)}")
            return {
                "structure": "unknown",
                "trend_strength": 0.0,
                "volatility": 0.0,
                "adx": 0.0,
                "rsi": 50.0,
            }

    def calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет продвинутых индикаторов"""
        try:
            # Волновой анализ
            wave_clusters = self.calculate_wave_clusters(data)

            # Анализ ликвидности
            liquidity_zones = self.calculate_liquidity_zones(data)

            # Анализ дисбаланса
            imbalance = self.calculate_imbalance(data)

            # Анализ волатильности
            volatility = self.calculate_volatility(data["close"])

            return {
                "wave_clusters": wave_clusters,
                "liquidity_zones": liquidity_zones,
                "imbalance": imbalance,
                "volatility": volatility,
            }

        except Exception as e:
            logger.error(f"Error calculating advanced indicators: {str(e)}")
            return {}

    def calculate_wave_clusters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет волновых кластеров с оптимизацией"""
        try:
            # Нормализация данных
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=self.config.cluster_window).std()

            # Определение волн с использованием scipy
            peaks, _ = find_peaks(returns.values, height=volatility.values)
            troughs, _ = find_peaks(-returns.values, height=volatility.values)

            # Классификация волн
            waves = pd.Series(0, index=data.index)
            waves.iloc[peaks] = 1
            waves.iloc[troughs] = -1

            # Определение кластеров
            clusters = []
            current_cluster = {
                "start": None,
                "end": None,
                "direction": 0,
                "strength": 0,
            }

            for i in range(len(waves)):
                if waves[i] != 0:
                    if current_cluster["start"] is None:
                        current_cluster["start"] = i
                        current_cluster["direction"] = waves[i]
                        current_cluster["strength"] = abs(returns[i])
                    elif waves[i] != current_cluster["direction"]:
                        current_cluster["end"] = i - 1
                        clusters.append(current_cluster)
                        current_cluster = {
                            "start": i,
                            "end": None,
                            "direction": waves[i],
                            "strength": abs(returns[i]),
                        }

            if current_cluster["start"] is not None:
                current_cluster["end"] = len(waves) - 1
                clusters.append(current_cluster)

            return {"waves": waves, "clusters": clusters, "volatility": volatility}

        except Exception as e:
            logger.error(f"Error calculating wave clusters: {str(e)}")
            return {"waves": pd.Series(), "clusters": [], "volatility": pd.Series()}

    def calculate_liquidity_zones(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Расчет зон ликвидности с оптимизацией"""
        try:
            # Нормализация цен
            price_range = df["high"].max() - df["low"].min()
            normalized_prices = (df["close"] - df["low"].min()) / price_range

            # Расчет объемов на уровнях
            volume_levels = pd.cut(normalized_prices, bins=20)
            volume_profile = df.groupby(volume_levels)["volume"].sum()

            # Определение значимых уровней
            mean_volume = volume_profile.mean()
            std_volume = volume_profile.std()
            significant_levels = volume_profile[
                volume_profile > mean_volume + std_volume
            ]

            # Конвертация обратно в цены
            support_levels = []
            resistance_levels = []

            for level in significant_levels.index:
                price = level.left * price_range + df["low"].min()
                if price < df["close"].iloc[-1]:
                    support_levels.append(price)
                else:
                    resistance_levels.append(price)

            return {
                "support": sorted(support_levels),
                "resistance": sorted(resistance_levels),
            }

        except Exception as e:
            logger.error(f"Error calculating liquidity zones: {str(e)}")
            return {"support": [], "resistance": []}

    def calculate_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """Расчет дисбаланса с оптимизацией"""
        try:
            # Нормализация данных
            price_range = data["high"].max() - data["low"].min()
            normalized_prices = (data["close"] - data["low"].min()) / price_range

            # Расчет дисбаланса
            imbalance = pd.Series(0.0, index=data.index)

            for i in range(1, len(data)):
                price_change = normalized_prices[i] - normalized_prices[i - 1]
                volume_change = data["volume"][i] / data["volume"][i - 1]

                imbalance[i] = price_change * volume_change

            # Сглаживание
            imbalance = imbalance.rolling(window=20).mean()

            return imbalance

        except Exception as e:
            logger.error(f"Error calculating imbalance: {str(e)}")
            return pd.Series(0.0, index=data.index)


# Создаем глобальный экземпляр калькулятора
calculator = IndicatorCalculator()


def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Расчет всех технических индикаторов"""
    return calculator.calculate_indicators(data)


def calculate_advanced_indicators(data: pd.DataFrame) -> Dict[str, Any]:
    """Расчет продвинутых индикаторов"""
    return calculator.calculate_advanced_indicators(data)


def calculate_rsi(data: ArrayLike, period: int = 14) -> pd.Series:
    """Relative Strength Index"""
    return rsi(data, period)


def calculate_macd(
    data: ArrayLike,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACD:
    """Moving Average Convergence Divergence"""
    return macd(data, fast_period, slow_period, signal_period)


def calculate_bollinger_bands(
    data: ArrayLike, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    return bollinger_bands(data, period, std_dev)


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """Расчет EMA"""
    return talib.EMA(data, timeperiod=period)


def calculate_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Расчет ATR"""
    return talib.ATR(high, low, close, timeperiod=period)


def calculate_fractals(
    high: pd.Series, low: pd.Series, period: int = 2
) -> Dict[str, Any]:
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

    # Определение паттерна разворота
    reversal_pattern = False
    if len(upper_fractals) > 0 and len(lower_fractals) > 0:
        last_upper = (
            upper_fractals[upper_fractals].index[-1] if any(upper_fractals) else None
        )
        last_lower = (
            lower_fractals[lower_fractals].index[-1] if any(lower_fractals) else None
        )

        if last_upper and last_lower:
            reversal_pattern = last_lower > last_upper

    return {
        "upper_fractals": upper_fractals,
        "lower_fractals": lower_fractals,
        "reversal_pattern": reversal_pattern,
    }


def calculate_wave_clusters(data: pd.DataFrame) -> Dict[str, Any]:
    """Расчет волновых кластеров"""
    # Простая реализация для примера
    returns = data["close"].pct_change()
    volatility = returns.rolling(window=20).std()

    # Определение волн
    waves = pd.Series(0, index=data.index)
    for i in range(1, len(returns)):
        if returns[i] > volatility[i]:
            waves[i] = 1  # Восходящая волна
        elif returns[i] < -volatility[i]:
            waves[i] = -1  # Нисходящая волна

    # Определение кластеров
    clusters = []
    current_cluster = {"start": None, "end": None, "direction": 0}

    for i in range(len(waves)):
        if waves[i] != 0:
            if current_cluster["start"] is None:
                current_cluster["start"] = i
                current_cluster["direction"] = waves[i]
            elif waves[i] != current_cluster["direction"]:
                current_cluster["end"] = i - 1
                clusters.append(current_cluster)
                current_cluster = {"start": i, "end": None, "direction": waves[i]}

    if current_cluster["start"] is not None:
        current_cluster["end"] = len(waves) - 1
        clusters.append(current_cluster)

    return {"waves": waves, "clusters": clusters}


def calculate_volume_delta(data: pd.DataFrame) -> pd.Series:
    """Расчет Volume Delta"""
    return data["volume"] * np.sign(data["close"] - data["open"])


def calculate_order_book_imbalance(order_book: pd.DataFrame) -> pd.Series:
    """Расчет дисбаланса стакана"""
    if "bid_volume" in order_book.columns and "ask_volume" in order_book.columns:
        total_volume = order_book["bid_volume"] + order_book["ask_volume"]
        return (order_book["bid_volume"] - order_book["ask_volume"]) / total_volume
    return pd.Series(0, index=order_book.index)


def calculate_vwap(data: pd.DataFrame) -> pd.Series:
    """Расчет VWAP"""
    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    return (typical_price * data["volume"]).cumsum() / data["volume"].cumsum()


def detect_volume_spike(volume: pd.Series, threshold: float = 1.5) -> bool:
    """Обнаружение всплеска объема"""
    mean_volume = volume.rolling(window=20).mean()
    std_volume = volume.rolling(window=20).std()
    return volume.iloc[-1] > (mean_volume.iloc[-1] + threshold * std_volume.iloc[-1])


def get_historical_extremes(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """Получение исторических экстремумов"""
    return {
        "high": data["high"].rolling(window=window).max().iloc[-1],
        "low": data["low"].rolling(window=window).min().iloc[-1],
        "close_high": data["close"].rolling(window=window).max().iloc[-1],
        "close_low": data["close"].rolling(window=window).min().iloc[-1],
    }


def calculate_volume_acceleration(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Расчет ускорения объема"""
    volume_ma = data["volume"].rolling(window=window).mean()
    volume_std = data["volume"].rolling(window=window).std()
    return (data["volume"] - volume_ma) / volume_std


def calculate_volatility(
    prices: Union[pd.Series, np.ndarray], window: int = 20, annualize: bool = True
) -> float:
    """
    Расчет волатильности

    Args:
        prices: Цены
        window: Размер окна
        annualize: Годовой расчет

    Returns:
        Значение волатильности
    """
    try:
        if isinstance(prices, pd.Series):
            returns = prices.pct_change().dropna()
        else:
            returns = np.diff(prices) / prices[:-1]

        volatility = np.std(returns, ddof=1)

        if annualize:
            volatility *= np.sqrt(252)  # Годовая волатильность

        return float(volatility)

    except Exception as e:
        logger.error(f"Ошибка расчета волатильности: {str(e)}")
        raise


def calculate_volume_profile(data: pd.DataFrame, bins: int = 24) -> Dict[str, Any]:
    """
    Расчет Volume Profile

    Args:
        data: DataFrame с данными
        bins: Количество бинов

    Returns:
        Словарь с данными Volume Profile
    """
    try:
        # Создаем бины по цене
        price_bins = np.linspace(data["low"].min(), data["high"].max(), bins)

        # Считаем объем в каждом бине
        volume_profile = np.zeros(bins - 1)
        for i in range(len(price_bins) - 1):
            mask = (data["close"] >= price_bins[i]) & (
                data["close"] < price_bins[i + 1]
            )
            volume_profile[i] = data.loc[mask, "volume"].sum()

        # Находим POC (Point of Control)
        poc_index = np.argmax(volume_profile)
        poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2

        return {
            "price_levels": price_bins,
            "volume_profile": volume_profile,
            "poc_price": poc_price,
            "value_area": {
                "high": price_bins[poc_index + 1],
                "low": price_bins[poc_index],
            },
        }

    except Exception as e:
        logger.error(f"Ошибка расчета Volume Profile: {str(e)}")
        return {}


def calculate_imbalance(data: pd.DataFrame) -> pd.Series:
    """
    Расчет дисбаланса объема

    Args:
        data: DataFrame с данными

    Returns:
        Series с дисбалансом
    """
    try:
        if "buy_volume" in data.columns and "sell_volume" in data.columns:
            return (data["buy_volume"] - data["sell_volume"]) / (
                data["buy_volume"] + data["sell_volume"]
            )
        return pd.Series(0, index=data.index)

    except Exception as e:
        logger.error(f"Ошибка расчета дисбаланса: {str(e)}")
        return pd.Series(0, index=data.index)


def calculate_liquidity_zones(
    df: pd.DataFrame, window: int = 20
) -> Dict[str, List[float]]:
    """Расчет зон ликвидности"""
    try:
        # Расчет объемного профиля
        price_bins = np.linspace(df["low"].min(), df["high"].max(), 50)
        volume_profile = np.zeros_like(price_bins)

        for i in range(len(df)):
            price = df["close"].iloc[i]
            volume = df["volume"].iloc[i]
            bin_idx = np.digitize(price, price_bins) - 1
            if 0 <= bin_idx < len(volume_profile):
                volume_profile[bin_idx] += volume

        # Поиск локальных максимумов объема
        peaks = []
        for i in range(1, len(volume_profile) - 1):
            if (
                volume_profile[i] > volume_profile[i - 1]
                and volume_profile[i] > volume_profile[i + 1]
            ):
                peaks.append((price_bins[i], volume_profile[i]))

        # Сортировка по объему
        peaks.sort(key=lambda x: x[1], reverse=True)

        # Выбор топ-5 зон
        top_zones = peaks[:5]

        return {
            "support": [p[0] for p in top_zones if p[0] < df["close"].iloc[-1]],
            "resistance": [p[0] for p in top_zones if p[0] > df["close"].iloc[-1]],
        }

    except Exception as e:
        print(f"Error calculating liquidity zones: {str(e)}")
        return {"support": [], "resistance": []}


def calculate_market_structure(df: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
    """Расчет структуры рынка"""
    # Расчет локальных максимумов и минимумов
    local_high = df["high"].rolling(window=window, center=True).max()
    local_low = df["low"].rolling(window=window, center=True).min()

    # Определение тренда
    trend = pd.Series(index=df.index)
    for i in range(window, len(df)):
        if df["close"].iloc[i] > local_high.iloc[i - 1]:
            trend.iloc[i] = 1  # Восходящий тренд
        elif df["close"].iloc[i] < local_low.iloc[i - 1]:
            trend.iloc[i] = -1  # Нисходящий тренд
        else:
            trend.iloc[i] = 0  # Боковой тренд

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
