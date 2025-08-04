"""
Модуль технических индикаторов.
Содержит промышленные функции для расчёта всех технических индикаторов:
трендовые, осцилляторы, объёмы, волатильность.
"""

# ВАЖНО: Для корректной работы mypy с pandas используйте pandas-stubs: pip install pandas-stubs

import pandas as pd
from shared.numpy_utils import np
from typing import Dict, List, Optional, Tuple, Union, Any, cast

# Типы для pandas
from pandas import Series, DataFrame

# Удаляю алиасы Series = pd.Series, DataFrame = pd.DataFrame

__all__ = [
    # Трендовые индикаторы
    "calc_sma",
    "calc_ema",
    "calc_wma",
    "calc_dema",
    "calc_tema",
    "calc_bollinger_bands",
    "calc_parabolic_sar",
    "calc_ichimoku",
    "calc_macd",
    "calc_adx",
    "calc_cci",
    "calc_aroon",
    # Осцилляторы
    "calc_rsi",
    "calc_stochastic",
    "calc_williams_r",
    "calc_momentum",
    "calc_roc",
    "calc_mfi",
    "calc_obv",
    "calc_cci_oscillator",
    "calc_ultimate_oscillator",
    "calc_awesome_oscillator",
    # Индикаторы объёма
    "calc_volume_sma",
    "calc_volume_ema",
    "calc_volume_ratio",
    "calc_volume_profile",
    "calc_vwap",
    "calc_accumulation_distribution",
    "calc_chaikin_money_flow",
    "calc_volume_price_trend",
    # Индикаторы волатильности
    "calc_atr",
    "calc_natr",
    "calc_keltner_channels",
    "calc_donchian_channels",
    "calc_standard_deviation",
    "calc_historical_volatility",
    # Утилиты
    "validate_ohlcv_data",
    "calculate_returns",
    "detect_divergence",
]


def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    """Валидация OHLCV данных."""
    if data is None or data.empty:
        return False
    required_columns = ["open", "high", "low", "close"]
    if not all(col in data.columns for col in required_columns):
        return False
    # Проверяем логику данных
    if (data["high"] < data["low"]).any():
        return False
    if (data["high"] < data["open"]).any() or (data["high"] < data["close"]).any():
        return False
    if (data["low"] > data["open"]).any() or (data["low"] > data["close"]).any():
        return False
    return True


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Расчёт доходностей."""
    return prices.pct_change().dropna()


# Трендовые индикаторы
def calc_sma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Простая скользящая средняя."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    # Проверяем наличие метода rolling у Series
    if hasattr(prices, 'rolling'):
        result = prices.rolling(window=period).mean()
        return result
    else:
        # Альтернативная реализация без rolling
        return pd.Series()


def calc_ema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Экспоненциальная скользящая средняя."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    # Проверяем наличие метода ewm у Series
    if hasattr(prices, 'ewm'):
        result = prices.ewm(span=period).mean()
        return result
    else:
        # Альтернативная реализация без ewm
        return pd.Series()


def calc_wma(prices: pd.Series, period: int = 20) -> pd.Series:
    """Взвешенная скользящая средняя."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    weights = np.arange(1, period + 1)
    # Проверяем наличие метода rolling у Series
    if hasattr(prices, 'rolling'):
        result = prices.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
        return result
    else:
        return pd.Series()


def calc_dema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Двойная экспоненциальная скользящая средняя."""
    ema1 = calc_ema(prices, period)
    ema2 = calc_ema(ema1, period)
    # Проверяем, что ema1 и ema2 являются Series
    if isinstance(ema1, pd.Series) and isinstance(ema2, pd.Series):
        result = 2 * ema1 - ema2
        return result
    else:
        return pd.Series()


def calc_tema(prices: pd.Series, period: int = 20) -> pd.Series:
    """Тройная экспоненциальная скользящая средняя."""
    ema1 = calc_ema(prices, period)
    ema2 = calc_ema(ema1, period)
    ema3 = calc_ema(ema2, period)
    # Проверяем, что все ema являются Series
    if all(isinstance(x, pd.Series) for x in [ema1, ema2, ema3]):
        result = 3 * ema1 - 3 * ema2 + ema3
        return result
    else:
        return pd.Series()


def calc_bollinger_bands(
    prices: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Полосы Боллинджера."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series(), pd.Series(), pd.Series()
    sma = calc_sma(prices, period)
    # Проверяем наличие метода rolling у Series
    if hasattr(prices, 'rolling'):
        std = prices.rolling(window=period).std()
        # Проверяем, что sma и std являются Series
        if isinstance(sma, pd.Series) and isinstance(std, pd.Series):
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band, sma, lower_band
    return pd.Series(), pd.Series(), pd.Series()


def calc_parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    acceleration: float = 0.02,
    maximum: float = 0.2,
) -> pd.Series:
    """Parabolic SAR (Stop and Reverse)."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series()
    
    # Создаем Series с правильными типами
    sar = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=bool)
    
    # Инициализация
    if len(low) > 0:
        sar.iloc[0] = float(low.iloc[0])
    if len(trend) > 0:
        trend.iloc[0] = True  # True = восходящий тренд
    
    for i in range(1, len(close)):
        if i < len(trend) and bool(trend.iloc[i - 1]):
            # Восходящий тренд
            if i < len(sar) and i < len(high) and i < len(low):
                sar.iloc[i] = float(sar.iloc[i - 1]) + acceleration * (
                    float(high.iloc[i - 1]) - float(sar.iloc[i - 1])
                )
                sar.iloc[i] = min(
                    float(sar.iloc[i]),
                    float(low.iloc[i - 1]),
                    float(low.iloc[i - 2]) if i > 1 else float(low.iloc[i - 1]),
                )
                if i < len(close) and float(close.iloc[i]) < float(sar.iloc[i]):
                    trend.iloc[i] = False
                    sar.iloc[i] = float(high.iloc[i - 1])
                else:
                    trend.iloc[i] = True
        else:
            # Нисходящий тренд
            if i < len(sar) and i < len(high) and i < len(low):
                sar.iloc[i] = float(sar.iloc[i - 1]) - acceleration * (
                    float(sar.iloc[i - 1]) - float(low.iloc[i - 1])
                )
                sar.iloc[i] = max(
                    float(sar.iloc[i]),
                    float(high.iloc[i - 1]),
                    float(high.iloc[i - 2]) if i > 1 else float(high.iloc[i - 1]),
                )
                if i < len(close) and float(close.iloc[i]) > float(sar.iloc[i]):
                    trend.iloc[i] = True
                    sar.iloc[i] = float(low.iloc[i - 1])
                else:
                    trend.iloc[i] = False
    
    return sar


def calc_ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
) -> Dict[str, pd.Series]:
    """Ишимоку Кинко Хё."""
    if not validate_ohlcv_data(pd.DataFrame({"high": high, "low": low})):
        return {}
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_span_b_high = high.rolling(window=senkou_span_b_period).max()
    senkou_span_b_low = low.rolling(window=senkou_span_b_period).min()
    senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    chikou_span = pd.Series(index=high.index, dtype=float)
    if len(high) > kijun_period:
        # Безопасное копирование значений
        high_values = high.iloc[kijun_period:].values
        if len(high_values) > 0:
            chikou_span.iloc[:-kijun_period] = high_values
    
    return {
        "tenkan_sen": tenkan_sen,
        "kijun_sen": kijun_sen,
        "senkou_span_a": senkou_span_a,
        "senkou_span_b": senkou_span_b,
        "chikou_span": chikou_span,
    }


def calc_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (Moving Average Convergence Divergence)."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series(), pd.Series(), pd.Series()
    ema_fast = calc_ema(prices, fast_period)
    ema_slow = calc_ema(prices, slow_period)
    # Проверяем, что ema_fast и ema_slow являются Series
    if isinstance(ema_fast, pd.Series) and isinstance(ema_slow, pd.Series):
        macd_line = ema_fast - ema_slow
        signal_line = calc_ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    return pd.Series(), pd.Series(), pd.Series()


def calc_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """ADX (Average Directional Index)."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series(), pd.Series(), pd.Series()
    
    # True Range
    tr1_adx = high - low
    
    # Проверяем наличие метода shift у Series
    if hasattr(close, 'shift'):
        tr2_adx_shift = (high - close.shift(1)).abs()
        tr3_adx_shift = (low - close.shift(1)).abs()
        tr2_adx = tr2_adx_shift
        tr3_adx = tr3_adx_shift
    else:
        # Безопасная альтернатива без shift
        tr2_adx_no_shift = (high - close).abs()
        tr3_adx_no_shift = (low - close).abs()
        tr2_adx = tr2_adx_no_shift
        tr3_adx = tr3_adx_no_shift
    
    tr_adx = pd.DataFrame({"tr1": tr1_adx, "tr2": tr2_adx, "tr3": tr3_adx}).max(axis=1)
    
    # Directional Movement
    if hasattr(high, 'shift'):
        dm_plus = high - high.shift(1)
        dm_minus = low.shift(1) - low
    else:
        dm_plus = high - high
        dm_minus = low - low
    
    # Безопасные сравнения
    dm_plus_safe = pd.Series(0.0, index=high.index)
    dm_minus_safe = pd.Series(0.0, index=high.index)
    
    for i in range(len(dm_plus)):
        if i > 0:
            plus_val = float(dm_plus.iloc[i])
            minus_val = float(dm_minus.iloc[i])
            if plus_val > minus_val and plus_val > 0.0:
                dm_plus_safe.iloc[i] = plus_val
            if minus_val > plus_val and minus_val > 0.0:
                dm_minus_safe.iloc[i] = minus_val
    
    # Smoothing
    tr_smooth = tr_adx.rolling(window=period).mean()
    dm_plus_smooth = dm_plus_safe.rolling(window=period).mean()
    dm_minus_smooth = dm_minus_safe.rolling(window=period).mean()
    
    # DI+ and DI-
    dx = pd.Series(0.0, index=high.index)
    for i in range(len(dm_plus_smooth)):
        plus_val = float(dm_plus_smooth.iloc[i])
        minus_val = float(dm_minus_smooth.iloc[i])
        tr_val = float(tr_smooth.iloc[i])
        
        if plus_val + minus_val > 0 and tr_val > 0:
            dx.iloc[i] = 100 * abs(plus_val - minus_val) / (plus_val + minus_val)
    
    adx = calc_ema(dx, period)
    return adx, dm_plus_smooth, dm_minus_smooth


def calc_cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """CCI (Commodity Channel Index)."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series()
    
    typical_price = (high + low + close) / 3
    sma_tp = calc_sma(typical_price, period)
    
    # Проверяем наличие метода rolling у Series
    if hasattr(typical_price, 'rolling'):
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
    else:
        # Безопасная альтернатива без rolling
        mean_deviation = pd.Series(0.0, index=typical_price.index)
        for i in range(period - 1, len(typical_price)):
            window = typical_price.iloc[i - period + 1:i + 1]
            window_mean = window.mean()
            deviations = [abs(float(val) - float(window_mean)) for val in window]
            mean_deviation.iloc[i] = sum(deviations) / len(deviations)
    
    # Безопасное вычисление CCI
    cci = pd.Series(0.0, index=typical_price.index)
    for i in range(len(typical_price)):
        tp_val = float(typical_price.iloc[i])
        sma_val = float(sma_tp.iloc[i])
        md_val = float(mean_deviation.iloc[i])
        
        if md_val > 0:
            cci.iloc[i] = (tp_val - sma_val) / (0.015 * md_val)
    
    return cci


def calc_aroon(
    high: pd.Series, low: pd.Series, period: int = 25
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aroon Indicator."""
    if not validate_ohlcv_data(pd.DataFrame({"high": high, "low": low})):
        return pd.Series(), pd.Series(), pd.Series()

    def aroon_up(series: pd.Series) -> pd.Series:
        # Проверяем наличие метода rolling у Series
        if hasattr(series, 'rolling'):
            return series.rolling(window=period).apply(
                lambda x: (x.argmax() / (period - 1)) * 100
            )
        else:
            # Безопасная альтернатива без rolling
            result = pd.Series(0.0, index=series.index)
            for i in range(period - 1, len(series)):
                window = series.iloc[i - period + 1:i + 1]
                max_idx = window.idxmax()
                window_idx = window.index.get_loc(max_idx)
                result.iloc[i] = (window_idx / (period - 1)) * 100
            return result

    def aroon_down(series: pd.Series) -> pd.Series:
        # Проверяем наличие метода rolling у Series
        if hasattr(series, 'rolling'):
            return series.rolling(window=period).apply(
                lambda x: (x.argmin() / (period - 1)) * 100
            )
        else:
            # Безопасная альтернатива без rolling
            result = pd.Series(0.0, index=series.index)
            for i in range(period - 1, len(series)):
                window = series.iloc[i - period + 1:i + 1]
                min_idx = window.idxmin()
                window_idx = window.index.get_loc(min_idx)
                result.iloc[i] = (window_idx / (period - 1)) * 100
            return result

    aroon_up_line = aroon_up(high)
    aroon_down_line = aroon_down(low)
    
    # Безопасное вычисление Aroon Oscillator
    aroon_oscillator = pd.Series(0.0, index=high.index)
    for i in range(len(high)):
        up_val = float(aroon_up_line.iloc[i])
        down_val = float(aroon_down_line.iloc[i])
        aroon_oscillator.iloc[i] = up_val - down_val
    
    return aroon_up_line, aroon_down_line, aroon_oscillator


# Осцилляторы
def calc_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """RSI (Relative Strength Index)."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    # Проверяем наличие метода diff у Series
    if hasattr(prices, 'diff'):
        delta = prices.diff()
        # Используем pandas методы для безопасного сравнения
        gain_mask: pd.Series = delta.gt(0.0)
        loss_mask: pd.Series = delta.lt(0.0)
        gain = delta.where(gain_mask, 0.0)
        loss = -delta.where(loss_mask, 0.0)
        # Проверяем наличие метода rolling у Series
        if hasattr(gain, 'rolling'):
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # Защита от деления на ноль
            avg_loss = avg_loss.replace(0, np.nan)
            rs = avg_gain / avg_loss
            rs = rs.fillna(0)  # Если avg_loss=0, то rs=0 (нет потерь -> RSI=100)
            
            # Защита от деления на ноль в финальной формуле
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.fillna(100)  # Если rs=0, то RSI=100
            return rsi
    return pd.Series()


def calc_stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Stochastic Oscillator."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series(), pd.Series()
    # Проверяем наличие метода rolling у Series
    if hasattr(low, 'rolling') and hasattr(high, 'rolling'):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = calc_sma(k_percent, d_period)
        return k_percent, d_percent
    else:
        return pd.Series(), pd.Series()


def calc_williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series()
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def calc_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
    """Momentum Oscillator."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    result = prices - prices.shift(period)
    return result


def calc_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    result = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return result


def calc_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    ):
        return pd.Series()
    # Вычисляем типичную цену
    typical_price = (high + low + close) / 3
    # Вычисляем денежный поток
    money_flow = typical_price * volume
    # Определяем положительный и отрицательный денежный поток
    positive_flow = pd.Series(0.0, index=money_flow.index)
    negative_flow = pd.Series(0.0, index=money_flow.index)
    # Сравниваем текущую типичную цену с предыдущей
    for i in range(1, len(typical_price)):
        if typical_price.iloc[i] > typical_price.iloc[i - 1]:
            positive_flow.iloc[i] = money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
            negative_flow.iloc[i] = money_flow.iloc[i]
    # Вычисляем скользящие средние
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    # Защита от деления на ноль в MFI
    negative_mf = negative_mf.replace(0, np.nan)
    money_ratio = positive_mf / negative_mf
    money_ratio = money_ratio.fillna(0)  # Если negative_mf=0, то money_ratio=0
    
    # Вычисляем MFI с защитой от деления на ноль
    mfi = 100 - (100 / (1 + money_ratio))
    mfi = mfi.fillna(100)  # Если money_ratio=0, то MFI=100
    return mfi


def calc_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    if not validate_ohlcv_data(pd.DataFrame({"close": close, "volume": volume})):
        return pd.Series()
    obv = pd.Series(0.0, index=close.index)
    obv.iloc[0] = volume.iloc[0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i - 1]:
            obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i - 1]
    return obv


def calc_cci_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """CCI как осциллятор."""
    return calc_cci(high, low, close, period)


def calc_ultimate_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period1: int = 7,
    period2: int = 14,
    period3: int = 28,
    weight1: float = 4.0,
    weight2: float = 2.0,
    weight3: float = 1.0,
) -> pd.Series:
    """Ultimate Oscillator."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series()
    # Вычисляем True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Вычисляем Buying Pressure
    bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
    # Вычисляем средние для разных периодов
    avg7 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg14 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg28 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    # Вычисляем Ultimate Oscillator
    uo = 100 * ((weight1 * avg7) + (weight2 * avg14) + (weight3 * avg28)) / (weight1 + weight2 + weight3)
    return uo


def calc_awesome_oscillator(
    high: pd.Series, low: pd.Series, period1: int = 5, period2: int = 34
) -> pd.Series:
    """Awesome Oscillator."""
    if not validate_ohlcv_data(pd.DataFrame({"high": high, "low": low})):
        return pd.Series()
    # Вычисляем средние цены
    median_price = (high + low) / 2
    # Вычисляем скользящие средние
    fast_ma = median_price.rolling(window=period1).mean()
    slow_ma = median_price.rolling(window=period2).mean()
    # Вычисляем Awesome Oscillator
    ao = fast_ma - slow_ma
    return ao


# Индикаторы объёма
def calc_volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Скользящая средняя объёма."""
    return calc_sma(volume, period)


def calc_volume_ema(volume: pd.Series, period: int = 20) -> pd.Series:
    """Экспоненциальная скользящая средняя объёма."""
    return calc_ema(volume, period)


def calc_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Отношение текущего объёма к среднему."""
    volume_sma = calc_volume_sma(volume, period)
    return volume / volume_sma


def calc_volume_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    price_levels: int = 100,
) -> Dict[str, np.ndarray]:
    """Volume Profile."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    ):
        return {"price_levels": np.array([]), "volume_profile": np.array([])}
    
    price_step = (float(high.max()) - float(low.min())) / price_levels
    if price_step <= 0:
        return {"price_levels": np.array([]), "volume_profile": np.array([])}
    
    volume_profile: np.ndarray = np.zeros(price_levels)
    price_levels_array: np.ndarray = np.linspace(float(low.min()), float(high.max()), price_levels)
    
    for i in range(len(close)):
        if hasattr(close, 'iloc') and hasattr(volume, 'iloc'):
            close_val = float(close.iloc[i])
            low_min = float(low.min())
            price_level = int((close_val - low_min) / price_step)
            if 0 <= price_level < price_levels:
                volume_profile[price_level] += float(volume.iloc[i])
    
    return {"price_levels": price_levels_array, "volume_profile": volume_profile}


def calc_vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Volume Weighted Average Price."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    ):
        return pd.Series()
    # Вычисляем типичную цену
    typical_price = (high + low + close) / 3
    # Вычисляем VWAP
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap


def calc_accumulation_distribution(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """Accumulation/Distribution Line."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    ):
        return pd.Series()
    # Вычисляем Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    # Вычисляем Money Flow Volume
    mfv = mfm * volume
    # Вычисляем Accumulation/Distribution Line
    ad_line = mfv.cumsum()
    return ad_line


def calc_chaikin_money_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    ):
        return pd.Series()
    # Вычисляем Money Flow Multiplier
    mfm = ((close - low) - (high - close)) / (high - low)
    # Вычисляем Money Flow Volume
    mfv = mfm * volume
    # Вычисляем Chaikin Money Flow
    cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
    return cmf


def calc_volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Price Trend."""
    if not validate_ohlcv_data(pd.DataFrame({"close": close, "volume": volume})):
        return pd.Series()
    # Вычисляем процентное изменение цены
    price_change = close.pct_change()
    # Вычисляем Volume Price Trend
    vpt = (price_change * volume).cumsum()
    return vpt


# Индикаторы волатильности
def calc_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Average True Range."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series()
    # Вычисляем True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Вычисляем ATR
    atr = tr.rolling(window=period).mean()
    return atr


def calc_natr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Normalized Average True Range."""
    atr = calc_atr(high, low, close, period)
    natr = (atr / close) * 100
    return natr


def calc_keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    multiplier: float = 2.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner Channels."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series(), pd.Series(), pd.Series()
    # Вычисляем среднюю цену
    typical_price = (high + low + close) / 3
    # Вычисляем ATR
    atr = calc_atr(high, low, close, period)
    # Вычисляем скользящую среднюю
    middle_line = typical_price.rolling(window=period).mean()
    # Вычисляем верхнюю и нижнюю линии
    upper_line = middle_line + (multiplier * atr)
    lower_line = middle_line - (multiplier * atr)
    return upper_line, middle_line, lower_line


def calc_donchian_channels(
    high: pd.Series, low: pd.Series, period: int = 20
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channels."""
    if not validate_ohlcv_data(pd.DataFrame({"high": high, "low": low})):
        return pd.Series(), pd.Series(), pd.Series()
    # Вычисляем верхнюю и нижнюю линии
    upper_line = high.rolling(window=period).max()
    lower_line = low.rolling(window=period).min()
    # Вычисляем среднюю линию
    middle_line = (upper_line + lower_line) / 2
    return upper_line, middle_line, lower_line


def calc_standard_deviation(prices: pd.Series, period: int = 20) -> pd.Series:
    """Standard Deviation."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    std = prices.rolling(window=period).std()
    return std


def calc_historical_volatility(
    prices: pd.Series, period: int = 20, annualization_factor: float = 252.0
) -> pd.Series:
    """Historical Volatility."""
    if not validate_ohlcv_data(pd.DataFrame({"close": prices})):
        return pd.Series()
    # Вычисляем логарифмические доходности
    log_returns = np.log(prices / prices.shift(1))
    # Вычисляем историческую волатильность
    volatility = log_returns.rolling(window=period).std() * np.sqrt(annualization_factor)
    return volatility


# Утилиты
def detect_divergence(
    price: pd.Series, indicator: pd.Series, lookback_period: int = 10
) -> Dict[str, bool]:
    """Detect divergence between price and indicator."""
    if not validate_ohlcv_data(pd.DataFrame({"close": price})):
        return {"bullish": False, "bearish": False}
    
    # Находим локальные максимумы и минимумы
    price_highs = price.rolling(window=lookback_period, center=True).max()
    price_lows = price.rolling(window=lookback_period, center=True).min()
    indicator_highs = indicator.rolling(window=lookback_period, center=True).max()
    indicator_lows = indicator.rolling(window=lookback_period, center=True).min()
    
    # Проверяем дивергенцию
    bullish_divergence = False
    bearish_divergence = False
    
    # Простая логика проверки дивергенции
    if len(price) > lookback_period:
        recent_price_low = price_lows.iloc[-lookback_period:].min()
        recent_indicator_low = indicator_lows.iloc[-lookback_period:].min()
        earlier_price_low = price_lows.iloc[:-lookback_period].min()
        earlier_indicator_low = indicator_lows.iloc[:-lookback_period].min()
        
        if recent_price_low < earlier_price_low and recent_indicator_low > earlier_indicator_low:
            bullish_divergence = True
        
        recent_price_high = price_highs.iloc[-lookback_period:].max()
        recent_indicator_high = indicator_highs.iloc[-lookback_period:].max()
        earlier_price_high = price_highs.iloc[:-lookback_period].max()
        earlier_indicator_high = indicator_highs.iloc[:-lookback_period].max()
        
        if recent_price_high > earlier_price_high and recent_indicator_high < earlier_indicator_high:
            bearish_divergence = True
    
    return {"bullish": bullish_divergence, "bearish": bearish_divergence}
