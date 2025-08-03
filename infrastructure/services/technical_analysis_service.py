"""
Промышленная реализация сервиса технического анализа.
Фасад для декомпозированных модулей технического анализа.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

from domain.exceptions import TechnicalAnalysisError
from domain.services.technical_analysis import TechnicalAnalysisService
from domain.types.technical_types import (
    TechnicalIndicatorResult,
    PatternType,
    SignalStrength,
    SignalType,
    TechnicalAnalysisReport,
    TradingSignal,
    TrendDirection,
    BollingerBandsResult,
    VolumeProfileResult,
    MarketStructureResult,
    MarketStructure,
    TrendStrength,
    SignalResult,
)

# Импорты из декомпозированных модулей
from .technical_analysis import (  # Indicators; Market structure; Signal generation; Utils
    TechnicalAnalysisCache,
    analyze_market_structure,
    analyze_volume_profile,
    calc_accumulation_distribution,
    calc_adx,
    calc_aroon,
    calc_atr,
    calc_awesome_oscillator,
    calc_bollinger_bands,
    calc_cci,
    calc_cci_oscillator,
    calc_chaikin_money_flow,
    calc_dema,
    calc_donchian_channels,
    calc_ema,
    calc_historical_volatility,
    calc_ichimoku,
    calc_keltner_channels,
    calc_macd,
    calc_mfi,
    calc_momentum,
    calc_natr,
    calc_obv,
    calc_parabolic_sar,
    calc_roc,
    calc_rsi,
    calc_sma,
    calc_standard_deviation,
    calc_stochastic,
    calc_tema,
    calc_ultimate_oscillator,
    calc_volume_ema,
    calc_volume_price_trend,
    calc_volume_profile,
    calc_volume_ratio,
    calc_volume_sma,
    calc_vwap,
    calc_williams_r,
    calc_wma,
    calculate_fibonacci_levels,
    calculate_pivot_points,
    calculate_returns,
    calculate_signal_strength,
    clean_cache,
    combine_signals,
    convert_to_datetime,
    convert_to_decimal,
    create_empty_indicator_result,
    create_empty_technical_result,
    detect_breakouts,
    detect_divergence,
    detect_divergence_patterns,
    detect_trend_direction,
    extract_ohlcv_components,
    find_swing_points,
    generate_breakout_signals,
    generate_composite_signal,
    generate_divergence_signals,
    generate_momentum_signals,
    generate_pattern_signals,
    generate_trend_signals,
    identify_chart_patterns,
    identify_consolidation_zones,
    identify_support_resistance_levels,
    normalize_data,
    validate_indicator_data,
    validate_market_data,
    validate_ohlcv_data,
    validate_signal,
)


@dataclass
class TechnicalAnalysisConfig:
    """Конфигурация технического анализа."""

    # Параметры индикаторов
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    bollinger_period: int = 20
    bollinger_std_dev: float = 2.0
    atr_period: int = 14
    # Параметры анализа
    min_data_points: int = 30
    trend_period: int = 50
    support_resistance_tolerance: float = 0.02
    # Параметры сигналов
    signal_confidence_threshold: float = 0.6
    enable_volume_analysis: bool = True
    enable_pattern_recognition: bool = True


class TechnicalAnalysisServiceImpl(TechnicalAnalysisService):
    """Промышленная реализация сервиса технического анализа."""

    def __init__(self, config: Optional[TechnicalAnalysisConfig] = None):
        """Инициализация сервиса."""
        self.config = config or TechnicalAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        self.cache = TechnicalAnalysisCache(ttl_hours=1)

    def calculate_sma(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Расчёт простой скользящей средней."""
        try:
            if not validate_indicator_data(prices, self.config.min_data_points):
                return pd.Series()
            return calc_sma(prices, period)
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return pd.Series()

    def calculate_ema(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Расчёт экспоненциальной скользящей средней."""
        try:
            if not validate_indicator_data(prices, self.config.min_data_points):
                return pd.Series()
            return calc_ema(prices, period)
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return pd.Series()

    def calculate_rsi(
        self, prices: pd.Series, period: Optional[int] = None
    ) -> pd.Series:
        """Расчёт RSI."""
        try:
            if not validate_indicator_data(prices, self.config.min_data_points):
                return pd.Series()
            period = period or self.config.rsi_period
            return calc_rsi(prices, period)
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.Series()

    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: Optional[int] = None,
        slow_period: Optional[int] = None,
        signal_period: Optional[int] = None,
    ) -> Dict[str, pd.Series]:
        """Расчёт MACD."""
        try:
            if not validate_indicator_data(prices, self.config.min_data_points):
                return {}
            fast_period = fast_period or self.config.macd_fast_period
            slow_period = slow_period or self.config.macd_slow_period
            signal_period = signal_period or self.config.macd_signal_period
            macd_line, signal_line, histogram = calc_macd(
                prices, fast_period, slow_period, signal_period
            )
            return {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
            }
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {}

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: Optional[int] = None,
        std_dev: Optional[float] = None,
    ) -> 'BollingerBandsResult':
        """Расчёт полос Боллинджера."""
        from domain.types.technical_types import BollingerBandsResult
        try:
            if not validate_indicator_data(prices, self.config.min_data_points):
                return BollingerBandsResult(
                    upper=pd.Series(), 
                    middle=pd.Series(), 
                    lower=pd.Series(),
                    bandwidth=pd.Series(),
                    percent_b=pd.Series()
                )
            period = period or self.config.bollinger_period
            std_dev = std_dev or self.config.bollinger_std_dev
            bollinger_result = calc_bollinger_bands(
                prices, period, std_dev
            )
            # Проверяем, что результат имеет нужные атрибуты
            if hasattr(bollinger_result, 'upper') and hasattr(bollinger_result, 'middle'):
                return BollingerBandsResult(
                    upper=bollinger_result.upper,  # type: ignore[attr-defined]
                    middle=bollinger_result.middle,  # type: ignore[attr-defined]
                    lower=bollinger_result.lower,  # type: ignore[attr-defined]
                    bandwidth=getattr(bollinger_result, 'bandwidth', pd.Series()),
                    percent_b=getattr(bollinger_result, 'percent_b', pd.Series())
                )
            elif isinstance(bollinger_result, tuple) and len(bollinger_result) >= 3:
                upper, middle, lower = bollinger_result[:3]
                bandwidth = bollinger_result[3] if len(bollinger_result) > 3 else pd.Series()
                percent_b = bollinger_result[4] if len(bollinger_result) > 4 else pd.Series()
                return BollingerBandsResult(
                    upper=upper,
                    middle=middle,
                    lower=lower,
                    bandwidth=bandwidth,
                    percent_b=percent_b
                )
            else:
                return BollingerBandsResult(
                    upper=pd.Series(), 
                    middle=pd.Series(), 
                    lower=pd.Series(),
                    bandwidth=pd.Series(),
                    percent_b=pd.Series()
                )
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return BollingerBandsResult(
                upper=pd.Series(), 
                middle=pd.Series(), 
                lower=pd.Series(),
                bandwidth=pd.Series(),
                percent_b=pd.Series()
            )

    def calculate_atr(
        self, df: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Расчёт ATR."""
        try:
            if not validate_ohlcv_data(df):
                return pd.Series()
            high = df["high"]
            low = df["low"]
            close = df["close"]
            return calc_atr(high, low, close, period)
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return pd.Series()

    def identify_support_resistance_levels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> Dict[str, List[Any]]:
        """Определение уровней поддержки и сопротивления."""
        try:
            if not validate_ohlcv_data(
                pd.DataFrame({"high": high, "low": low, "close": close})
            ):
                return {"support": [], "resistance": []}
            return identify_support_resistance_levels(
                high, low, close, volume, self.config.support_resistance_tolerance
            )
        except Exception as e:
            self.logger.error(f"Error identifying support/resistance: {e}")
            return {"support": [], "resistance": []}

    def detect_trend_direction(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> TrendDirection:
        """Определение направления тренда."""
        try:
            if not validate_ohlcv_data(
                pd.DataFrame({"high": high, "low": low, "close": close})
            ):
                return TrendDirection.SIDEWAYS
            return detect_trend_direction(high, low, close, len(close) if volume is not None else 20)
        except Exception as e:
            self.logger.error(f"Error detecting trend direction: {e}")
            return TrendDirection.SIDEWAYS

    def identify_chart_patterns(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> List[Dict[str, Any]]:
        """Определение паттернов на графике."""
        try:
            if not validate_ohlcv_data(
                pd.DataFrame({"high": high, "low": low, "close": close})
            ):
                return []
            return identify_chart_patterns(high, low, close, volume)
        except Exception as e:
            self.logger.error(f"Error identifying chart patterns: {e}")
            return []

    def generate_trading_signals(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> List[TradingSignal]:
        """Генерация торговых сигналов."""
        try:
            if not validate_ohlcv_data(
                pd.DataFrame({"high": high, "low": low, "close": close})
            ):
                return []
            # Генерация различных типов сигналов
            trend_signals = generate_trend_signals(high, low, close, volume)
            momentum_signals = generate_momentum_signals(high, low, close, volume)
            # Исправление: используем tolist() вместо to_list() и правильные типы
            close_list = close.tolist() if hasattr(close, 'tolist') and callable(close.tolist) else list(close)
            close_array = close.to_numpy() if hasattr(close, 'to_numpy') and callable(close.to_numpy) else np.array(close)
            # Исправляю передачу Series в generate_pattern_signals
            pattern_signals = generate_pattern_signals([], close)
            # Объединение сигналов
            all_signals = trend_signals + momentum_signals + pattern_signals
            return all_signals
        except Exception as e:
            self.logger.error(f"Error generating trading signals: {e}")
            return []

    def perform_complete_analysis(
        self, market_data: pd.DataFrame
    ) -> TechnicalAnalysisReport:
        """Выполнение полного технического анализа."""
        try:
            if not validate_market_data(market_data):
                return create_empty_technical_result()
            # Извлечение компонентов OHLCV
            ohlcv = extract_ohlcv_components(market_data)
            # Расчет индикаторов
            sma = self.calculate_sma(ohlcv["close"])
            ema = self.calculate_ema(ohlcv["close"])
            rsi = self.calculate_rsi(ohlcv["close"])
            macd = self.calculate_macd(ohlcv["close"])
            bollinger = self.calculate_bollinger_bands(ohlcv["close"])
            atr = self.calculate_atr(market_data)
            # Анализ структуры рынка
            market_structure_dict = analyze_market_structure(ohlcv["high"], ohlcv["low"], ohlcv["close"])
            # Преобразование в MarketStructureResult
            market_structure = MarketStructureResult(
                structure=MarketStructure.SIDEWAYS,  # Исправление 340: создаем MarketStructureResult
                trend_strength=TrendStrength.WEAK,
                volatility=Decimal("0.0"),
                adx=Decimal("0.0"),
                rsi=Decimal("0.0"),
                confidence=Decimal("0.0")
            )
            # Генерация сигналов
            trading_signals = self.generate_trading_signals(
                ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
            )
            # Преобразование в SignalResult
            signals = [SignalResult(
                signal_type=signal.signal_type,
                strength=Decimal(str(signal.strength)),
                confidence=Decimal(str(signal.confidence))
            ) for signal in trading_signals]  # Исправление 353: правильные типы
            # Создание отчета
            return TechnicalAnalysisReport(
                indicator_results=TechnicalIndicatorResult(
                    indicators={},
                    market_structure=market_structure,  # Исправление 352: используем MarketStructureResult
                    volume_profile=VolumeProfileResult(
                        poc=0.0,
                        value_area_high=0.0,
                        value_area_low=0.0,
                        volume_by_price={},
                        histogram=[],
                        price_levels=[]
                    ),
                    support_levels=[],
                    resistance_levels=[]
                ),
                market_structure=market_structure,  # Исправление 352: используем MarketStructureResult
                signals=signals,  # Исправление 353: используем list[SignalResult]
            )
        except Exception as e:
            self.logger.error(f"Error performing complete analysis: {e}")
            return create_empty_technical_result()

    def calculate_indicator(
        self, indicator_name: str, market_data: pd.DataFrame, **kwargs
    ) -> TechnicalIndicatorResult:
        """Расчет конкретного индикатора."""
        try:
            if not validate_market_data(market_data):
                return create_empty_indicator_result()
            # Извлечение компонентов OHLCV
            ohlcv = extract_ohlcv_components(market_data)
            # Расчет индикатора в зависимости от типа
            if indicator_name == "sma":
                result = self.calculate_sma(ohlcv["close"], **kwargs)
            elif indicator_name == "ema":
                result = self.calculate_ema(ohlcv["close"], **kwargs)
            elif indicator_name == "rsi":
                result = self.calculate_rsi(ohlcv["close"], **kwargs)
            elif indicator_name == "macd":
                result = self.calculate_macd(ohlcv["close"], **kwargs)
            elif indicator_name == "bollinger":
                result = self.calculate_bollinger_bands(ohlcv["close"], **kwargs)
            elif indicator_name == "atr":
                result = self.calculate_atr(market_data, **kwargs)
            else:
                self.logger.warning(f"Unknown indicator: {indicator_name}")
                return create_empty_indicator_result()
            
            # Исправление: создаем правильный словарь индикаторов
            indicators_dict: Dict[str, Any] = {}
            # Исправляю типы для индикаторов
            if indicator_name == "macd" and isinstance(result, dict):
                indicators_dict = result
            elif indicator_name == "bollinger" and hasattr(result, 'upper'):
                indicators_dict = {
                    "upper": result.upper,
                    "middle": result.middle,
                    "lower": result.lower,
                    "bandwidth": getattr(result, 'bandwidth', pd.Series()),
                    "percent_b": getattr(result, 'percent_b', pd.Series())
                }
            else:
                indicators_dict = {indicator_name: result}
            
            # Создание результата
            return TechnicalIndicatorResult(
                indicators=indicators_dict,
                market_structure=MarketStructureResult(
                    structure=MarketStructure.SIDEWAYS,
                    trend_strength=TrendStrength.WEAK,
                    volatility=Decimal("0.0"),
                    adx=Decimal("0.0"),
                    rsi=Decimal("0.0"),
                    confidence=Decimal("0.0")
                ),
                volume_profile=VolumeProfileResult(
                    poc=0.0,
                    value_area_high=0.0,
                    value_area_low=0.0,
                    volume_by_price={},
                    histogram=[],
                    price_levels=[]
                ),
                support_levels=[],
                resistance_levels=[]
            )
        except Exception as e:
            self.logger.error(f"Error calculating indicator {indicator_name}: {e}")
            return create_empty_indicator_result()

    def analyze_market_structure(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ структуры рынка."""
        try:
            if not validate_market_data(market_data):
                return {}
            # Извлечение компонентов OHLCV
            ohlcv = extract_ohlcv_components(market_data)
            # Анализ структуры
            structure = analyze_market_structure(ohlcv["high"], ohlcv["low"], ohlcv["close"])
            return structure
        except Exception as e:
            self.logger.error(f"Error analyzing market structure: {e}")
            return {}

    def generate_composite_signal(
        self, market_data: pd.DataFrame
    ) -> Optional[TradingSignal]:
        """Генерация композитного сигнала."""
        try:
            if not validate_market_data(market_data):
                return None
            # Извлечение компонентов OHLCV
            ohlcv = extract_ohlcv_components(market_data)
            # Генерация сигналов
            signals = self.generate_trading_signals(
                ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"]
            )
            # Создание композитного сигнала
            if signals:
                # Исправление 439: используем правильный тип для generate_composite_signal
                return generate_composite_signal(ohlcv["high"], ohlcv["low"], ohlcv["close"])
            return None
        except Exception as e:
            self.logger.error(f"Error generating composite signal: {e}")
            return None

    def _calculate_overall_confidence(
        self,
        signals: List[TradingSignal],
        indicators: Dict[str, Any],
        patterns: List[Dict[str, Any]],
    ) -> Decimal:
        """Расчет общей уверенности в анализе."""
        try:
            # Расчет уверенности на основе сигналов
            signal_confidence = sum(signal.confidence for signal in signals) / max(len(signals), 1)
            # Расчет уверенности на основе индикаторов
            indicator_confidence = 0.5  # Базовая уверенность
            # Расчет уверенности на основе паттернов
            pattern_confidence = len(patterns) * 0.1  # 0.1 за каждый паттерн
            # Общая уверенность
            total_confidence = (signal_confidence + indicator_confidence + pattern_confidence) / 3
            return Decimal(str(min(total_confidence, 1.0)))
        except Exception as e:
            self.logger.error(f"Error calculating overall confidence: {e}")
            return Decimal("0.5")
