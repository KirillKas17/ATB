"""
Модуль анализа структуры рынка.
Содержит промышленные функции для анализа структуры рынка:
уровни поддержки/сопротивления, паттерны, тренды, фибоначчи.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from domain.types.technical_types import (
    PatternType,
    SupportResistanceLevel,
    TrendDirection,
)
from infrastructure.services.technical_analysis.utils import validate_ohlcv_data

# Импорт функций из модуля indicators
from .indicators import (
    calc_adx,
    calc_atr,
    calc_sma,
    calc_volume_profile,
    detect_divergence,
)

__all__ = [
    "identify_support_resistance_levels",
    "detect_trend_direction",
    "find_swing_points",
    "calculate_fibonacci_levels",
    "identify_chart_patterns",
    "analyze_market_structure",
    "detect_breakouts",
    "calculate_pivot_points",
    "identify_consolidation_zones",
    "analyze_volume_profile",
    "detect_divergence_patterns",
]


def _safe_extract_value(obj: Any) -> Any:
    """Безопасное извлечение значения из объекта, который может быть callable."""
    if callable(obj):
        return obj()
    return obj


def _safe_extract_series(obj: Any) -> pd.Series:
    """Безопасное извлечение pandas Series из объекта."""
    if callable(obj):
        obj = obj()
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    return obj


def _safe_index(obj, *args, **kwargs) -> Any:
    """Безопасная индексация объекта."""
    if callable(obj):
        obj = obj()
    return obj.__getitem__(*args, **kwargs) if hasattr(obj, '__getitem__') else obj


def _safe_assign(obj, key, value):
    """Безопасное присваивание значения объекту."""
    if callable(obj):
        obj = obj()
    if hasattr(obj, '__setitem__'):
        obj[key] = value
    return obj


def find_swing_points(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """Поиск точек разворота (swing points)."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    swing_highs = pd.Series(index=high.index, dtype=float)
    swing_lows = pd.Series(index=low.index, dtype=float)
    
    for i in range(window, len(high) - window):
        # Swing high
        if (high.iloc[i - window : i] <= high.iloc[i]).all() and (
            high.iloc[i + 1 : i + window + 1] <= high.iloc[i]
        ).all():
            _safe_assign(swing_highs, i, _safe_index(high, i))
        
        # Swing low
        if (low.iloc[i - window : i] >= low.iloc[i]).all() and (
            low.iloc[i + 1 : i + window + 1] >= low.iloc[i]
        ).all():
            _safe_assign(swing_lows, i, _safe_index(low, i))
    
    return swing_highs, swing_lows


def identify_support_resistance_levels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: Optional[pd.Series] = None,
    tolerance: float = 0.02,
) -> Dict[str, List[SupportResistanceLevel]]:
    """Идентификация уровней поддержки и сопротивления."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return {"support": [], "resistance": []}
    
    # Группируем близкие ценовые уровни
    high_grouped = _group_price_levels(high, tolerance)
    low_grouped = _group_price_levels(low, tolerance)
    
    support_levels = []
    resistance_levels = []
    
    # Уровни сопротивления из максимумов
    for price, group in high_grouped.items():
        if len(group) >= 2:  # Минимум 2 касания
            resistance_levels.append(
                SupportResistanceLevel(
                    price=price,
                    strength=float(len(group)),
                    touches=len(group),
                    level_type="resistance",
                    timestamp=datetime.now(),
                )
            )
    # Уровни поддержки из минимумов
    for price, group in low_grouped.items():
        if len(group) >= 2:  # Минимум 2 касания
            support_levels.append(
                SupportResistanceLevel(
                    price=price,
                    strength=float(len(group)),
                    touches=len(group),
                    level_type="support",
                    timestamp=datetime.now(),
                )
            )
    return {
        "support": sorted(support_levels, key=lambda x: x.price),
        "resistance": sorted(resistance_levels, key=lambda x: x.price, reverse=True),
    }


def _group_price_levels(
    price_series: pd.Series, tolerance: float
) -> Dict[float, pd.Series]:
    """Группировка ценовых уровней по толерантности."""
    grouped = {}
    if isinstance(price_series, pd.Series):
        sorted_prices = price_series.sort_values()  # type: ignore
    else:
        # Альтернативный способ сортировки для не-pandas объектов
        sorted_prices = pd.Series(sorted(price_series))
    current_group = []
    current_level = None
    for price in sorted_prices:
        if current_level is None:
            current_level = price
            current_group = [price]
        elif abs(price - current_level) / current_level <= tolerance:
            current_group.append(price)
        else:
            # Сохраняем группу
            if current_group:
                avg_price = np.mean(current_group)
                grouped[avg_price] = pd.Series(current_group, dtype=float)
            # Начинаем новую группу
            current_level = price
            current_group = [price]
    # Сохраняем последнюю группу
    if current_group:
        avg_price = np.mean(current_group)
        grouped[avg_price] = pd.Series(current_group, dtype=float)
    return grouped


def detect_trend_direction(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    short_period: int = 10,
    long_period: int = 50,
) -> TrendDirection:
    """Определение направления тренда."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return TrendDirection.SIDEWAYS
    methods = []
    # 1. Скользящие средние
    sma_short = _safe_extract_series(calc_sma(close, short_period))
    sma_long = _safe_extract_series(calc_sma(close, long_period))
    
    if float(sma_short.iloc[-1]) > float(sma_long.iloc[-1]):  # type: ignore
        methods.append(TrendDirection.UP)
    else:
        methods.append(TrendDirection.DOWN)
    
    # 2. Swing points
    swing_highs, swing_lows = find_swing_points(high, low, close)
    recent_highs = swing_highs.dropna().tail(3)
    recent_lows = swing_lows.dropna().tail(3)
    
    if len(recent_highs) >= 2 and len(recent_lows) >= 2:
        if (
            float(recent_highs.iloc[-1]) > float(recent_highs.iloc[-2])  # type: ignore
            and float(recent_lows.iloc[-1]) > float(recent_lows.iloc[-2])  # type: ignore
        ):
            methods.append(TrendDirection.UP)
        elif (
            float(recent_highs.iloc[-1]) < float(recent_highs.iloc[-2])  # type: ignore
            and float(recent_lows.iloc[-1]) < float(recent_lows.iloc[-2])  # type: ignore
        ):
            methods.append(TrendDirection.DOWN)
        else:
            methods.append(TrendDirection.SIDEWAYS)
    
    # 3. ADX для силы тренда
    adx_result = calc_adx(high, low, close)
    if isinstance(adx_result, tuple) and len(adx_result) == 3:
        adx: pd.Series = _safe_extract_series(adx_result[0])
        di_plus: pd.Series = _safe_extract_series(adx_result[1])
        di_minus: pd.Series = _safe_extract_series(adx_result[2])
        
        if hasattr(adx, 'iloc') and float(adx.iloc[-1]) > 25:  # Сильный тренд  # type: ignore[index]
            if hasattr(di_plus, 'iloc') and hasattr(di_minus, 'iloc') and float(di_plus.iloc[-1]) > float(di_minus.iloc[-1]):  # type: ignore[index]
                methods.append(TrendDirection.UP)
            else:
                methods.append(TrendDirection.DOWN)
        else:
            methods.append(TrendDirection.SIDEWAYS)
    
    up_count = methods.count(TrendDirection.UP)
    down_count = methods.count(TrendDirection.DOWN)
    sideways_count = methods.count(TrendDirection.SIDEWAYS)
    
    if up_count > down_count and up_count > sideways_count:
        return TrendDirection.UP
    elif down_count > up_count and down_count > sideways_count:
        return TrendDirection.DOWN
    else:
        return TrendDirection.SIDEWAYS


def calculate_fibonacci_levels(
    high: pd.Series, low: pd.Series, close: pd.Series, trend_direction: TrendDirection
) -> Dict[str, float]:
    """Расчёт уровней Фибоначчи."""
    if not validate_ohlcv_data(pd.DataFrame({"high": high, "low": low, "close": close})):
        return {}
    
    swing_highs, swing_lows = find_swing_points(high, low, close)
    
    if trend_direction == TrendDirection.UP:
        # Восходящий тренд: от минимума к максимуму
        start_price = float(swing_lows.dropna().iloc[-1])  # type: ignore[index]
        end_price = float(swing_highs.dropna().iloc[-1])  # type: ignore[index]
    else:
        # Нисходящий тренд: от максимума к минимуму
        start_price = float(swing_highs.dropna().iloc[-1])  # type: ignore[index]
        end_price = float(swing_lows.dropna().iloc[-1])  # type: ignore[index]
    
    price_range = end_price - start_price
    fibonacci_levels = {
        "0.0": start_price,
        "0.236": start_price + 0.236 * price_range,
        "0.382": start_price + 0.382 * price_range,
        "0.5": start_price + 0.5 * price_range,
        "0.618": start_price + 0.618 * price_range,
        "0.786": start_price + 0.786 * price_range,
        "1.0": end_price,
        "1.272": start_price + 1.272 * price_range,
        "1.618": start_price + 1.618 * price_range,
    }
    return fibonacci_levels


def identify_chart_patterns(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """Идентификация графических паттернов."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return []
    patterns = []
    # Двойная вершина
    double_top = _detect_double_top(high, low, close)
    if double_top:
        patterns.append(double_top)
    # Двойное дно
    double_bottom = _detect_double_bottom(high, low, close)
    if double_bottom:
        patterns.append(double_bottom)
    # Голова и плечи
    head_shoulders = _detect_head_shoulders(high, low, close)
    if head_shoulders:
        patterns.append(head_shoulders)
    # Перевёрнутая голова и плечи
    inverse_head_shoulders = _detect_inverse_head_shoulders(high, low, close)
    if inverse_head_shoulders:
        patterns.append(inverse_head_shoulders)
    # Треугольники
    triangles = _detect_triangles(high, low, close)
    patterns.extend(triangles)
    # Флаги и вымпелы
    flags_pennants = _detect_flags_pennants(high, low, close)
    patterns.extend(flags_pennants)
    return patterns


def _detect_double_top(
    high: pd.Series, low: pd.Series, close: pd.Series, tolerance: float = 0.02
) -> Optional[Dict[str, Any]]:
    """Обнаружение двойной вершины."""
    swing_highs, _ = find_swing_points(high, low, close)
    recent_highs = swing_highs.dropna().tail(5)
    if len(recent_highs) < 2:
        return None
    # Проверяем последние две вершины
    peak1 = recent_highs.iloc[-2]  # type: ignore
    peak2 = recent_highs.iloc[-1]  # type: ignore
    # Проверяем, что вершины близки по цене
    if abs(peak1 - peak2) / peak1 <= tolerance:
        # Проверяем, что между вершинами есть впадина
        valley_idx = recent_highs.index[-2]  # type: ignore
        valley_price = low.loc[valley_idx : recent_highs.index[-1]].min()  # type: ignore
        if valley_price < min(peak1, peak2):
            return {
                "pattern": "DOUBLE_TOP",
                "peaks": [peak1, peak2],
                "valley": valley_price,
                "breakout_level": valley_price,
                "target": valley_price - (max(peak1, peak2) - valley_price),
                "confidence": 0.8,
            }
    return None


def _detect_double_bottom(
    high: pd.Series, low: pd.Series, close: pd.Series, tolerance: float = 0.02
) -> Optional[Dict[str, Any]]:
    """Обнаружение двойного дна."""
    _, swing_lows = find_swing_points(high, low, close)
    recent_lows = swing_lows.dropna().tail(5)
    if len(recent_lows) < 2:
        return None
    # Проверяем последние два дна
    bottom1 = recent_lows.iloc[-2]  # type: ignore
    bottom2 = recent_lows.iloc[-1]  # type: ignore
    # Проверяем, что дна близки по цене
    if abs(bottom1 - bottom2) / bottom1 <= tolerance:
        # Проверяем, что между днами есть пик
        peak_idx = recent_lows.index[-2]  # type: ignore
        peak_price = high.loc[peak_idx : recent_lows.index[-1]].max()  # type: ignore
        if peak_price > max(bottom1, bottom2):
            return {
                "pattern": "DOUBLE_BOTTOM",
                "bottoms": [bottom1, bottom2],
                "peak": peak_price,
                "breakout_level": peak_price,
                "target": peak_price + (peak_price - min(bottom1, bottom2)),
                "confidence": 0.8,
            }
    return None


def _detect_head_shoulders(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Optional[Dict[str, Any]]:
    """Обнаружение паттерна голова и плечи."""
    swing_highs, _ = find_swing_points(high, low, close)
    recent_highs = swing_highs.dropna().tail(7)
    if len(recent_highs) < 5:
        return None
    # Ищем три вершины: левое плечо, голова, правое плечо
    peaks = recent_highs.values
    peak_indices = recent_highs.index
    # Проверяем паттерн: левое плечо < голова > правое плечо
    if hasattr(peaks, '__len__') and len(peaks) >= 5:
        left_shoulder = peaks[-5] if hasattr(peaks, "__getitem__") else 0 if hasattr(peaks, '__getitem__') else 0
        head = peaks[-3] if hasattr(peaks, "__getitem__") else 0 if hasattr(peaks, '__getitem__') else 0
        right_shoulder = peaks[-1] if hasattr(peaks, "__getitem__") else 0 if hasattr(peaks, '__getitem__') else 0
        if (
            left_shoulder < head
            and head > right_shoulder
            and abs(left_shoulder - right_shoulder) / left_shoulder < 0.1
        ):
            # Находим линию шеи
            neckline_start = low.loc[peak_indices[-5]]  # type: ignore
            neckline_end = low.loc[peak_indices[-1]]  # type: ignore
            return {
                "pattern": "HEAD_AND_SHOULDERS",
                "left_shoulder": left_shoulder,
                "head": head,
                "right_shoulder": right_shoulder,
                "neckline": [neckline_start, neckline_end],
                "breakout_level": neckline_end,
                "target": neckline_end - (head - neckline_end),
                "confidence": 0.85,
            }
    return None


def _detect_inverse_head_shoulders(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Optional[Dict[str, Any]]:
    """Обнаружение перевёрнутого паттерна голова и плечи."""
    _, swing_lows = find_swing_points(high, low, close)
    recent_lows = swing_lows.dropna().tail(7)
    if len(recent_lows) < 5:
        return None
    # Ищем три дна: левое плечо, голова, правое плечо
    bottoms = recent_lows.values
    bottom_indices = recent_lows.index
    # Проверяем паттерн: левое плечо > голова < правое плечо
    if hasattr(bottoms, "__len__") and len(bottoms) >= 5:
        left_shoulder = bottoms[-5] if hasattr(bottoms, "__getitem__") else 0
        head = bottoms[-3] if hasattr(bottoms, "__getitem__") else 0
        right_shoulder = bottoms[-1] if hasattr(bottoms, "__getitem__") else 0
        if (
            left_shoulder > head
            and head < right_shoulder
            and abs(left_shoulder - right_shoulder) / left_shoulder < 0.1
        ):
            # Находим линию шеи
            neckline_start = high.loc[bottom_indices[-5]]  # type: ignore
            neckline_end = high.loc[bottom_indices[-1]]  # type: ignore
            return {
                "pattern": "INVERSE_HEAD_AND_SHOULDERS",
                "left_shoulder": left_shoulder,
                "head": head,
                "right_shoulder": right_shoulder,
                "neckline": [neckline_start, neckline_end],
                "breakout_level": neckline_end,
                "target": neckline_end + (neckline_end - head),
                "confidence": 0.85,
            }
    return None


def _detect_triangles(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> List[Dict[str, Any]]:
    """Обнаружение треугольных паттернов."""
    triangles = []
    # Анализируем последние 20 баров
    recent_highs = high.tail(20)
    recent_lows = low.tail(20)
    
    # Находим линии тренда
    high_slope, high_intercept = _fit_trend_line(
        recent_highs.index, np.array(recent_highs.values)
    )
    low_slope, low_intercept = _fit_trend_line(
        recent_lows.index, np.array(recent_lows.values)
    )
    
    # Безопасная проверка для len
    if hasattr(recent_highs.values, '__len__') and hasattr(recent_lows.values, '__len__'):
        high_values = recent_highs.values if hasattr(recent_highs, "values") else np.array([])
        low_values = recent_lows.values if hasattr(recent_lows, "values") else np.array([])
    else:
        high_values = np.array(recent_highs.values)
        low_values = np.array(recent_lows.values)
    
    # Определяем тип треугольника
    if high_slope < 0 and low_slope > 0:
        # Симметричный треугольник
        triangles.append(
            {
                "pattern": "SYMMETRICAL_TRIANGLE",
                "upper_line": {"slope": high_slope, "intercept": high_intercept},
                "lower_line": {"slope": low_slope, "intercept": low_intercept},
                "breakout_direction": "unknown",
                "confidence": 0.7,
            }
        )
    elif high_slope < 0 and abs(low_slope) < 0.001:
        # Нисходящий треугольник
        triangles.append(
            {
                "pattern": "DESCENDING_TRIANGLE",
                "upper_line": {"slope": high_slope, "intercept": high_intercept},
                "lower_line": {"slope": low_slope, "intercept": low_intercept},
                "breakout_direction": "down",
                "confidence": 0.75,
            }
        )
    elif abs(high_slope) < 0.001 and low_slope > 0:
        # Восходящий треугольник
        triangles.append(
            {
                "pattern": "ASCENDING_TRIANGLE",
                "upper_line": {"slope": high_slope, "intercept": high_intercept},
                "lower_line": {"slope": low_slope, "intercept": low_intercept},
                "breakout_direction": "up",
                "confidence": 0.75,
            }
        )
    return triangles


def _fit_trend_line(x: pd.Index, y: np.ndarray) -> Tuple[float, float]:
    """Подгонка линии тренда."""
    # Проверяем, что x не является callable
    if callable(x):
        x = x()
    if callable(y):
        y = y()
    
    # Проверяем, что y является numpy array
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # Исправление: приводим x к list для корректной работы len()
    if hasattr(x, '__len__'):
        x_length = len(x)
    else:
        x_length = len(x.tolist())  # type: ignore
    
    if x_length < 2:
        return 0.0, 0.0
    x_numeric = np.arange(x_length)
    slope, intercept, _, _, _ = np.polyfit(x_numeric, y, 1)
    return slope, intercept


def _detect_flags_pennants(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> List[Dict[str, Any]]:
    """Обнаружение флагов и вымпелов."""
    flags_pennants = []
    # Анализируем последние 15 баров
    recent_highs = high.tail(15)
    recent_lows = low.tail(15)
    
    # Находим линии тренда
    high_slope, high_intercept = _fit_trend_line(
        recent_highs.index, np.array(recent_highs.values)
    )
    low_slope, low_intercept = _fit_trend_line(
        recent_lows.index, np.array(recent_lows.values)
    )
    # Проверяем параллельность линий (флаг)
    if abs(high_slope - low_slope) < 0.01 and abs(high_slope) > 0.001:
        flags_pennants.append(
            {
                "pattern": "FLAG",
                "upper_line": {"slope": high_slope, "intercept": high_intercept},
                "lower_line": {"slope": low_slope, "intercept": low_intercept},
                "breakout_direction": "up" if high_slope > 0 else "down",
                "confidence": 0.7,
            }
        )
    # Проверяем схождение линий (вымпел)
    elif abs(high_slope - low_slope) > 0.01:
        flags_pennants.append(
            {
                "pattern": "PENNANT",
                "upper_line": {"slope": high_slope, "intercept": high_intercept},
                "lower_line": {"slope": low_slope, "intercept": low_intercept},
                "breakout_direction": "unknown",
                "confidence": 0.65,
            }
        )
    return flags_pennants


def detect_breakouts(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    support_resistance_levels: Dict[str, List[SupportResistanceLevel]],
    volume: Optional[pd.Series] = None,
) -> List[Dict[str, Any]]:
    """Обнаружение пробоев уровней."""
    breakouts = []
    current_price = close.iloc[-1]  # type: ignore
    current_volume = volume.iloc[-1] if volume is not None else None  # type: ignore
    # Проверяем пробои уровней сопротивления
    for level in support_resistance_levels["resistance"]:
        if current_price > level.price * (1 + 0.01):  # 1% выше уровня
            volume_confirmation = False
            if current_volume:
                volume_confirmation = current_volume > 1000  # Простая проверка объема
            breakouts.append(
                {
                    "type": "resistance_breakout",
                    "level": float(level.price),
                    "price": float(current_price),
                    "strength": level.strength,
                    "volume_confirmation": volume_confirmation,
                    "timestamp": close.index[-1],
                }
            )
    # Проверяем пробои уровней поддержки
    for level in support_resistance_levels["support"]:
        if current_price < level.price * (1 - 0.01):  # 1% ниже уровня
            volume_confirmation = False
            if current_volume:
                volume_confirmation = current_volume > 1000  # Простая проверка объема
            breakouts.append(
                {
                    "type": "support_breakdown",
                    "level": float(level.price),
                    "price": float(current_price),
                    "strength": level.strength,
                    "volume_confirmation": volume_confirmation,
                    "timestamp": close.index[-1],
                }
            )
    return breakouts


def calculate_pivot_points(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> Dict[str, float]:
    """Расчёт точек разворота."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return {}
    # Используем последние данные
    prev_high = high.iloc[-2]  # type: ignore
    prev_low = low.iloc[-2]  # type: ignore
    prev_close = close.iloc[-2]  # type: ignore
    # Классические точки разворота
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    return {"pivot": pivot, "r1": r1, "r2": r2, "r3": r3, "s1": s1, "s2": s2, "s3": s3}


def identify_consolidation_zones(
    high: pd.Series, low: pd.Series, close: pd.Series, min_periods: int = 10
) -> List[Dict[str, Any]]:
    """Идентификация зон консолидации."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return []
    consolidation_zones = []
    # Анализируем волатильность
    atr_result = calc_atr(high, low, close, 14)
    atr = _safe_extract_series(atr_result)
    if isinstance(atr, pd.Series):
        atr_percent = (atr / close) * 100
        # Ищем периоды низкой волатильности
        low_volatility = atr_percent < atr_percent.rolling(window=20).mean() * 0.7
        # Группируем последовательные периоды низкой волатильности
        consolidation_start = None
        consolidation_end = None
        for i in range(len(low_volatility)):
            if low_volatility.iloc[i] and consolidation_start is None:  # type: ignore
                consolidation_start = i
            elif not low_volatility.iloc[i] and consolidation_start is not None:  # type: ignore
                consolidation_end = i
                if consolidation_end - consolidation_start >= min_periods:
                    zone_high = high.iloc[consolidation_start:consolidation_end].max()  # type: ignore
                    zone_low = low.iloc[consolidation_start:consolidation_end].min()  # type: ignore
                    consolidation_zones.append(
                        {
                            "start": consolidation_start,
                            "end": consolidation_end,
                            "high": zone_high,
                            "low": zone_low,
                            "range": zone_high - zone_low,
                            "duration": consolidation_end - consolidation_start,
                        }
                    )
                consolidation_start = None
                consolidation_end = None
    return consolidation_zones


def analyze_market_structure(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Комплексный анализ структуры рынка."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close})
    ):
        return {}
    # Определяем тренд
    trend = detect_trend_direction(high, low, close)
    # Находим уровни поддержки и сопротивления
    levels = identify_support_resistance_levels(high, low, close, volume)
    # Ищем паттерны
    patterns = identify_chart_patterns(high, low, close, volume)
    # Рассчитываем точки разворота
    pivot_points = calculate_pivot_points(high, low, close)
    # Ищем зоны консолидации
    consolidation_zones = identify_consolidation_zones(high, low, close)
    # Обнаруживаем пробои
    breakouts = detect_breakouts(high, low, close, levels, volume)
    # Рассчитываем уровни Фибоначчи
    fibonacci_levels = calculate_fibonacci_levels(high, low, close, trend)
    return {
        "trend_direction": trend,
        "support_resistance_levels": levels,
        "patterns": patterns,
        "pivot_points": pivot_points,
        "consolidation_zones": consolidation_zones,
        "breakouts": breakouts,
        "fibonacci_levels": fibonacci_levels,
        "analysis_timestamp": close.index[-1],  # type: ignore
    }


def analyze_volume_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    price_levels: int = 100,
) -> Dict[str, Any]:
    """Анализ профиля объёма."""
    if not validate_ohlcv_data(
        pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})
    ):
        return {}
    
    profile_result = _safe_extract_value(calc_volume_profile(high, low, close, volume, price_levels))
    if not profile_result:
        return {}
    
    # Находим POC (Point of Control)
    poc_idx = np.argmax(profile_result["volume_profile"])
    poc_price = profile_result["price_levels"][poc_idx]
    
    # Находим Value Area (70% объёма)
    total_volume = np.sum(profile_result["volume_profile"])
    target_volume = total_volume * 0.7
    sorted_indices = np.argsort(profile_result["volume_profile"])[::-1]
    cumulative_volume = 0
    value_area_indices = []
    
    for idx in sorted_indices:
        cumulative_volume += profile_result["volume_profile"][idx]
        value_area_indices.append(idx)
        if cumulative_volume >= target_volume:
            break
    
    value_area_prices = [profile_result["price_levels"][i] for i in value_area_indices]
    
    return {
        "poc": poc_price,
        "value_area": {
            "high": max(value_area_prices),
            "low": min(value_area_prices),
        },
        "price_levels": profile_result["price_levels"],
        "volume_profile": profile_result["volume_profile"],
    }


def detect_divergence_patterns(
    price: pd.Series, indicator: pd.Series, lookback_period: int = 10
) -> List[Dict[str, Any]]:
    """Обнаружение паттернов дивергенции."""
    divergences = []
    # Бычья дивергенция
    bullish_div = detect_divergence(price, indicator, lookback_period)
    if bullish_div and bullish_div.get("bullish_divergence"):
        divergences.append(
            {
                "type": "bullish_divergence",
                "strength": bullish_div.get("strength", 0.7),
                "timestamp": price.index[-1],  # type: ignore
                "description": "Price making lower lows while indicator making higher lows",
            }
        )
    # Медвежья дивергенция
    bearish_div = detect_divergence(price, indicator, lookback_period)
    if bearish_div and bearish_div.get("bearish_divergence"):
        divergences.append(
            {
                "type": "bearish_divergence",
                "strength": bearish_div.get("strength", 0.7),
                "timestamp": price.index[-1],  # type: ignore
                "description": "Price making higher highs while indicator making lower highs",
            }
        )
    return divergences
