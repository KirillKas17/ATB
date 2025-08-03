# -*- coding: utf-8 -*-
"""
Модуль генерации торговых сигналов.
Содержит промышленные функции для генерации торговых сигналов
на основе технического анализа.
"""

import pandas as pd
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from domain.types.technical_types import (
    PatternType,
    SignalStrength,
    SignalType,
    TradingSignal,
)

# Импорт функций из модуля indicators
from .indicators import (
    calc_adx,
    calc_bollinger_bands,
    calc_keltner_channels,
    calc_macd,
    calc_rsi,
    calc_sma,
    calc_stochastic,
    calc_williams_r,
    detect_divergence,
)

__all__ = [
    "generate_trend_signals",
    "generate_momentum_signals",
    "generate_breakout_signals",
    "generate_divergence_signals",
    "generate_pattern_signals",
    "combine_signals",
    "calculate_signal_strength",
    "validate_signal",
    "generate_composite_signal",
]


def _safe_float_to_decimal(value: float) -> Decimal:
    """Безопасное преобразование float в Decimal."""
    return Decimal(str(value))


def _safe_extract_series(obj: Any) -> pd.Series:
    """Безопасное извлечение pandas Series из объекта."""
    if callable(obj):
        obj = obj()
    if not isinstance(obj, pd.Series):
        obj = pd.Series(obj)
    return pd.Series(obj)


def _safe_extract_tuple(obj: Any) -> Tuple[Any, ...]:
    """Безопасное извлечение tuple из объекта."""
    if callable(obj):
        obj = obj()
    if not isinstance(obj, tuple):
        obj = (obj,)
    return tuple(obj)


def generate_trend_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series] = None,
) -> List[TradingSignal]:
    """Генерация сигналов на основе трендовых индикаторов."""
    signals = []
    # MACD сигналы
    macd_result = _safe_extract_tuple(calc_macd(close))
    if isinstance(macd_result, (list, tuple)) and len(macd_result) == 3:
        macd_line = _safe_extract_series(macd_result[0])
        signal_line = _safe_extract_series(macd_result[1])
        histogram = _safe_extract_series(macd_result[2])
        if len(macd_line) > 0 and len(signal_line) > 0:
            # Бычий сигнал
            if (float(histogram.iloc[-1] if hasattr(histogram, "iloc") else histogram[-1]) > 0
                and float(macd_line.iloc[-1] if hasattr(macd_line, "iloc") else macd_line[-1]) > float(signal_line.iloc[-1] if hasattr(signal_line, "iloc") else signal_line[-1])):
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        indicator="MACD",
                        price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                        timestamp=close.index[-1],
                        description="MACD bullish crossover",
                        confidence=0.7,
                    )
                )
            # Медвежий сигнал
            elif (float(histogram.iloc[-1] if hasattr(histogram, "iloc") else histogram[-1]) < 0
                and float(macd_line.iloc[-1] if hasattr(macd_line, "iloc") else macd_line[-1]) < float(signal_line.iloc[-1] if hasattr(signal_line, "iloc") else signal_line[-1])):
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        indicator="MACD",
                        price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                        timestamp=close.index[-1],
                        description="MACD bearish crossover",
                        confidence=0.7,
                    )
                )

    # ADX сигналы
    adx_result = _safe_extract_tuple(calc_adx(high, low, close))
    if isinstance(adx_result, (list, tuple)) and len(adx_result) == 3:
        adx = _safe_extract_series(adx_result[0])
        di_plus = _safe_extract_series(adx_result[1])
        di_minus = _safe_extract_series(adx_result[2])
        if len(adx) > 0 and len(di_plus) > 0 and len(di_minus) > 0:
            if float(adx.iloc[-1] if hasattr(adx, "iloc") else adx[-1]) > 25:
                if float(di_plus.iloc[-1] if hasattr(di_plus, "iloc") else di_plus[-1]) > float(di_minus.iloc[-1] if hasattr(di_minus, "iloc") else di_minus[-1]):
                    signals.append(
                        TradingSignal(
                            signal_type=SignalType.BUY,
                            strength=SignalStrength.MEDIUM,
                            indicator="ADX",
                            price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                            timestamp=close.index[-1],
                            description="ADX bullish trend",
                            confidence=0.65,
                        )
                    )
                else:
                    signals.append(
                        TradingSignal(
                            signal_type=SignalType.SELL,
                            strength=SignalStrength.MEDIUM,
                            indicator="ADX",
                            price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                            timestamp=close.index[-1],
                            description="ADX bearish trend",
                            confidence=0.65,
                        )
                    )

    # Moving Average сигналы
    sma_20 = _safe_extract_series(calc_sma(close, 20))
    sma_50 = _safe_extract_series(calc_sma(close, 50))
    if len(sma_20) > 1 and len(sma_50) > 1:
        # Золотой крест
        sma_20_prev = float(sma_20.iloc[-2] if hasattr(sma_20, "iloc") else sma_20[-2])
        sma_50_prev = float(sma_50.iloc[-2] if hasattr(sma_50, "iloc") else sma_50[-2])
        sma_20_curr = float(sma_20.iloc[-1] if hasattr(sma_20, "iloc") else sma_20[-1])
        sma_50_curr = float(sma_50.iloc[-1] if hasattr(sma_50, "iloc") else sma_50[-1])
        
        if sma_20_prev < sma_50_prev and sma_20_curr > sma_50_curr:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.STRONG,
                    indicator="Moving Average",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description="Golden Cross (SMA 20 > SMA 50)",
                    confidence=0.8,
                )
            )
        # Мертвый крест
        elif sma_20_prev > sma_50_prev and sma_20_curr < sma_50_curr:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.STRONG,
                    indicator="Moving Average",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description="Death Cross (SMA 20 < SMA 50)",
                    confidence=0.8,
                )
            )
    return signals


def generate_momentum_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series] = None,
) -> List[TradingSignal]:
    """Генерация сигналов на основе осцилляторов."""
    signals = []
    # RSI сигналы
    rsi = _safe_extract_series(calc_rsi(close))
    if len(rsi) > 0:
        # Перепроданность
        if float(rsi.iloc[-1] if hasattr(rsi, "iloc") else rsi[-1]) < 30:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MEDIUM,
                    indicator="RSI",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description="RSI oversold condition",
                    confidence=0.6,
                )
            )
        # Перекупленность
        elif float(rsi.iloc[-1] if hasattr(rsi, "iloc") else rsi[-1]) > 70:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MEDIUM,
                    indicator="RSI",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description="RSI overbought condition",
                    confidence=0.6,
                )
            )
    # Stochastic сигналы
    stochastic_result = _safe_extract_tuple(calc_stochastic(high, low, close))
    if isinstance(stochastic_result, (list, tuple)) and len(stochastic_result) == 2:
        k_percent = _safe_extract_series(stochastic_result[0])
        d_percent = _safe_extract_series(stochastic_result[1])
        if len(k_percent) > 0 and len(d_percent) > 0:
            # Перепроданность
            if k_percent.iloc[-1] if hasattr(k_percent, "iloc") else k_percent[-1] < 20 and d_percent.iloc[-1] if hasattr(d_percent, "iloc") else d_percent[-1] < 20:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        indicator="Stochastic",
                        price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                        timestamp=close.index[-1],
                        description="Stochastic oversold condition",
                        confidence=0.6,
                    )
                )
            # Перекупленность
            elif k_percent.iloc[-1] if hasattr(k_percent, "iloc") else k_percent[-1] > 80 and d_percent.iloc[-1] if hasattr(d_percent, "iloc") else d_percent[-1] > 80:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        indicator="Stochastic",
                        price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                        timestamp=close.index[-1],
                        description="Stochastic overbought condition",
                        confidence=0.6,
                    )
                )
    # Williams %R сигналы
    williams_r = _safe_extract_series(calc_williams_r(high, low, close))
    if len(williams_r) > 0:
        # Перепроданность
        if float(williams_r.iloc[-1] if hasattr(williams_r, "iloc") else williams_r[-1]) < -80:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MEDIUM,
                    indicator="Williams %R",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description="Williams %R oversold condition",
                    confidence=0.6,
                )
            )
        # Перекупленность
        elif float(williams_r.iloc[-1]) > -20:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MEDIUM,
                    indicator="Williams %R",
                    price=_safe_float_to_decimal(float(close.iloc[-1])),
                    timestamp=close.index[-1],
                    description="Williams %R overbought condition",
                    confidence=0.6,
                )
            )
    return signals


def generate_breakout_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series] = None,
) -> List[TradingSignal]:
    """Генерация сигналов на основе пробоев уровней."""
    signals = []

    # Bollinger Bands пробои
    bb_result = _safe_extract_tuple(calc_bollinger_bands(close))
    if isinstance(bb_result, (list, tuple)) and len(bb_result) == 3:
        upper_band: pd.Series = _safe_extract_series(bb_result[0])
        middle_band: pd.Series = _safe_extract_series(bb_result[1])
        lower_band: pd.Series = _safe_extract_series(bb_result[2])
        if len(upper_band) > 0 and len(lower_band) > 0:
            # Пробой верхней полосы
            if close.iloc[-1] > upper_band.iloc[-1]:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        indicator="Bollinger Bands",
                        price=_safe_float_to_decimal(float(close.iloc[-1])),
                        timestamp=close.index[-1],
                        description="Price broke above upper Bollinger Band",
                        confidence=0.65,
                    )
                )
            # Пробой нижней полосы
            elif close.iloc[-1] < lower_band.iloc[-1]:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        indicator="Bollinger Bands",
                        price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                        timestamp=close.index[-1],
                        description="Price broke below lower Bollinger Band",
                        confidence=0.65,
                    )
                )

    # Keltner Channels пробои
    kc_result = _safe_extract_tuple(calc_keltner_channels(high, low, close))
    if isinstance(kc_result, (list, tuple)) and len(kc_result) == 3:
        upper_kc: pd.Series = _safe_extract_series(kc_result[0])
        middle_kc: pd.Series = _safe_extract_series(kc_result[1])
        lower_kc: pd.Series = _safe_extract_series(kc_result[2])
        if len(upper_kc) > 0 and len(lower_kc) > 0:
            # Пробой верхнего канала
            if close.iloc[-1] > upper_kc.iloc[-1]:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.SELL,
                        strength=SignalStrength.MEDIUM,
                        indicator="Keltner Channels",
                        price=_safe_float_to_decimal(float(close.iloc[-1])),
                        timestamp=close.index[-1],
                        description="Price broke above upper Keltner Channel",
                        confidence=0.6,
                    )
                )
            # Пробой нижнего канала
            elif close.iloc[-1] < lower_kc.iloc[-1]:
                signals.append(
                    TradingSignal(
                        signal_type=SignalType.BUY,
                        strength=SignalStrength.MEDIUM,
                        indicator="Keltner Channels",
                        price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                        timestamp=close.index[-1],
                        description="Price broke below lower Keltner Channel",
                        confidence=0.6,
                    )
                )
    return signals


def generate_divergence_signals(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series] = None,
) -> List[TradingSignal]:
    """Генерация сигналов на основе дивергенций."""
    signals = []
    # RSI дивергенции
    rsi_result = _safe_extract_series(calc_rsi(close))
    if len(rsi_result) > 0:
        divergence_result = detect_divergence(close, rsi_result)
        if divergence_result:
            for div in divergence_result:
                if isinstance(div, dict):
                    div_type = div.get("type", "")
                    div_description = div.get("description", "")
                else:
                    div_type = str(div)
                    div_description = str(div)
                if div_type == "bullish":
                    signals.append(
                        TradingSignal(
                            signal_type=SignalType.BUY,
                            strength=SignalStrength.STRONG,
                            indicator="RSI Divergence",
                            price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                            timestamp=close.index[-1],
                            description=f"Bullish RSI divergence: {div_description}",
                            confidence=0.8,
                        )
                    )
                elif div_type == "bearish":
                    signals.append(
                        TradingSignal(
                            signal_type=SignalType.SELL,
                            strength=SignalStrength.STRONG,
                            indicator="RSI Divergence",
                            price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                            timestamp=close.index[-1],
                            description=f"Bearish RSI divergence: {div_description}",
                            confidence=0.8,
                        )
                    )
    return signals


def generate_pattern_signals(
    patterns: List[Dict[str, Any]], close: pd.Series
) -> List[TradingSignal]:
    """Генерация сигналов на основе паттернов."""
    signals = []
    for pattern in patterns:
        pattern_type = pattern.get("type")
        confidence = pattern.get("confidence", 0.5)
        pattern_name = pattern.get("name", "Unknown")
        if pattern_type == PatternType.BULLISH:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.BUY,
                    strength=SignalStrength.MEDIUM,
                    indicator="Pattern Recognition",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description=f"Bullish pattern detected: {pattern_name}",
                    confidence=confidence,
                )
            )
        elif pattern_type == PatternType.BEARISH:
            signals.append(
                TradingSignal(
                    signal_type=SignalType.SELL,
                    strength=SignalStrength.MEDIUM,
                    indicator="Pattern Recognition",
                    price=_safe_float_to_decimal(float(close.iloc[-1] if hasattr(close, "iloc") else close[-1])),
                    timestamp=close.index[-1],
                    description=f"Bearish pattern detected: {pattern_name}",
                    confidence=confidence,
                )
            )
    return signals


def calculate_signal_strength(signal: TradingSignal) -> float:
    """Расчет силы сигнала."""
    base_strength = {
        SignalStrength.WEAK: 0.3,
        SignalStrength.MEDIUM: 0.6,
        SignalStrength.STRONG: 0.9,
    }.get(signal.strength, 0.5)

    # Корректировка на основе уверенности
    adjusted_strength = base_strength * signal.confidence
    return min(1.0, max(0.0, adjusted_strength))


def validate_signal(signal: TradingSignal) -> bool:
    """Валидация торгового сигнала."""
    if not signal.price or signal.price <= 0:
        return False
    if not signal.timestamp:
        return False
    if signal.confidence < 0 or signal.confidence > 1:
        return False
    if not signal.description:
        return False
    return True


def combine_signals(signals: List[TradingSignal]) -> Optional[TradingSignal]:
    """Объединение нескольких сигналов в один."""
    if not signals:
        return None

    # Подсчет весов по типам сигналов
    buy_weight = 0.0
    sell_weight = 0.0

    for signal in signals:
        weight = calculate_signal_strength(signal)
        if signal.signal_type == SignalType.BUY:
            buy_weight += weight
        elif signal.signal_type == SignalType.SELL:
            sell_weight += weight

    # Определение итогового сигнала
    if buy_weight > sell_weight and buy_weight > 0.5:
        return TradingSignal(
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG if buy_weight > 0.8 else SignalStrength.MEDIUM,
            indicator="Combined Signals",
            price=signals[0].price,  # Используем цену первого сигнала
            timestamp=signals[0].timestamp,
            description=f"Combined {len(signals)} signals (Buy weight: {buy_weight:.2f})",
            confidence=buy_weight,
        )
    elif sell_weight > buy_weight and sell_weight > 0.5:
        return TradingSignal(
            signal_type=SignalType.SELL,
            strength=SignalStrength.STRONG if sell_weight > 0.8 else SignalStrength.MEDIUM,
            indicator="Combined Signals",
            price=signals[0].price,  # Используем цену первого сигнала
            timestamp=signals[0].timestamp,
            description=f"Combined {len(signals)} signals (Sell weight: {sell_weight:.2f})",
            confidence=sell_weight,
        )

    return None


def generate_composite_signal(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: Optional[pd.Series] = None,
    patterns: Optional[List[Dict[str, Any]]] = None,
) -> Optional[TradingSignal]:
    """Генерация композитного сигнала на основе всех индикаторов."""
    all_signals = []

    # Собираем сигналы от всех источников
    all_signals.extend(generate_trend_signals(close, high, low, volume))
    all_signals.extend(generate_momentum_signals(close, high, low, volume))
    all_signals.extend(generate_breakout_signals(close, high, low, volume))
    all_signals.extend(generate_divergence_signals(close, high, low, volume))

    if patterns:
        all_signals.extend(generate_pattern_signals(patterns, close))

    # Фильтруем валидные сигналы
    valid_signals = [s for s in all_signals if validate_signal(s)]

    # Объединяем сигналы
    return combine_signals(valid_signals)
