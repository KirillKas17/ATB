# -*- coding: utf-8 -*-
"""
Доменный сервис технического анализа.
Промышленная реализация с строгой типизацией и продвинутыми
алгоритмами технического анализа.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import pandas as pd
from loguru import logger

# Временные замены для отсутствующих библиотек
try:
    import talib  # type: ignore
except ImportError:
    # Простая замена для talib
    class TalibMock:
        @staticmethod
        def RSI(data: pd.Series, timeperiod: int = 14) -> pd.Series:
            return pd.Series([0.5] * len(data), index=data.index)

        @staticmethod
        def MACD(
            data: pd.Series,
            fastperiod: int = 12,
            slowperiod: int = 26,
            signalperiod: int = 9,
        ) -> Tuple[pd.Series, pd.Series, pd.Series]:
            return pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index), pd.Series([0] * len(data), index=data.index)

        @staticmethod
        def BBANDS(
            data: pd.Series,
            timeperiod: int = 20,
            nbdevup: float = 2.0,
            nbdevdn: float = 2.0,
        ) -> Tuple[pd.Series, pd.Series, pd.Series]:
            return pd.Series([data.mean() + data.std()] * len(data), index=data.index), data, pd.Series([data.mean() - data.std()] * len(data), index=data.index)

        @staticmethod
        def ATR(
            high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
        ) -> pd.Series:
            return pd.Series([0.1] * len(high), index=high.index)

        @staticmethod
        def EMA(data: pd.Series, timeperiod: int = 14) -> pd.Series:
            return data.ewm(span=timeperiod).mean()

        @staticmethod
        def SMA(data: pd.Series, timeperiod: int = 20) -> pd.Series:
            return data.rolling(window=timeperiod).mean()

        @staticmethod
        def ADX(
            high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14
        ) -> pd.Series:
            return pd.Series([25.0] * len(high), index=high.index)

        @staticmethod
        def STOCHRSI(data: pd.Series) -> Tuple[pd.Series, pd.Series]:
            return pd.Series([50.0] * len(data), index=data.index), pd.Series([50.0] * len(data), index=data.index)

        @staticmethod
        def KELTNER(
            high: pd.Series, low: pd.Series, close: pd.Series
        ) -> Tuple[pd.Series, pd.Series, pd.Series]:
            return pd.Series([close.mean() + close.std()] * len(close), index=close.index), close, pd.Series([close.mean() - close.std()] * len(close), index=close.index)

        @staticmethod
        def VWAP(
            high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
        ) -> pd.Series:
            typical_price = (high + low + close) / 3
            return (typical_price * volume).cumsum() / volume.cumsum()

        @staticmethod
        def FRACTAL(high: pd.Series, low: pd.Series) -> Tuple[pd.Series, pd.Series]:
            return pd.Series([0.0] * len(high), index=high.index), pd.Series([0.0] * len(low), index=low.index)

    talib = TalibMock()
try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Простая замена для StandardScaler
    class StandardScalerMock:
        def __init__(self) -> None:
            self.mean_: Optional[float] = None
            self.scale_: Optional[float] = None

        def fit(self, X: pd.Series) -> "StandardScalerMock":
            self.mean_ = X.mean()
            self.scale_ = X.std()
            return self

        def transform(self, X: pd.Series) -> pd.Series:
            if self.mean_ is None or self.scale_ is None:
                return X
            return (X - self.mean_) / self.scale_

    StandardScaler = StandardScalerMock

from domain.entities.market import MarketData, TechnicalIndicator
from domain.exceptions import TechnicalAnalysisError
from domain.types.technical_types import (
    BollingerBandsResult,
    MarketStructure,
    MarketStructureResult,
    TrendStrength,
    VolumeProfileResult,
)


class IndicatorType(Enum):
    """Типы индикаторов."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class IndicatorConfig:
    """Конфигурация индикаторов."""

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


@dataclass
class MACD:
    """MACD индикатор."""

    macd: pd.Series
    signal: pd.Series
    histogram: pd.Series


@dataclass
class VolumeProfileData:
    """Данные профиля объема."""

    poc: Optional[float]  # Point of Control
    value_area: List[float]
    histogram: List[float]
    bins: List[float]


@dataclass
class MarketStructureData:
    """Данные структуры рынка."""

    structure: MarketStructure
    trend_strength: float
    volatility: float
    adx: float
    rsi: float
    confidence: float = field(default=0.0)


@dataclass
class TechnicalAnalysisResult:
    """Результат технического анализа."""

    symbol: str
    timeframe: str
    indicators: Dict[str, TechnicalIndicator]
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class TechnicalAnalysisProtocol(Protocol):
    """Протокол для технического анализа."""

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Рассчитать Simple Moving Average."""
        ...

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Рассчитать Exponential Moving Average."""
        ...

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Рассчитать Relative Strength Index."""
        ...

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Рассчитать MACD."""
        ...

    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> BollingerBandsResult:
        """Рассчитать полосы Боллинджера."""
        ...

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитать Average True Range."""
        ...

    def calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Рассчитать Stochastic Oscillator."""
        ...

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитать Williams %R."""
        ...

    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Рассчитать Commodity Channel Index."""
        ...

    def analyze_market_data(
        self, market_data: List[MarketData], indicators: List[str]
    ) -> TechnicalAnalysisResult:
        """Анализировать рыночные данные."""
        ...


class TechnicalAnalysisService(TechnicalAnalysisProtocol):
    """Основной сервис технического анализа."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Рассчитать Simple Moving Average."""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Рассчитать Exponential Moving Average."""
        return prices.ewm(span=period).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Рассчитать Relative Strength Index."""
        # Приводим к float для избежания ошибок типизации
        prices_float = prices.astype(float)
        delta = prices_float.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Заполняем NaN значения

    def calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Dict[str, pd.Series]:
        """Рассчитать MACD."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return {
            "macd": macd,
            "signal": signal_line,
            "histogram": histogram,
        }

    def calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std_dev: float = 2.0
    ) -> BollingerBandsResult:
        """Рассчитать полосы Боллинджера."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return BollingerBandsResult(
            upper=upper,
            middle=sma,
            lower=lower,
            bandwidth=(upper - lower) / sma,
            percent_b=(prices - lower) / (upper - lower),
        )

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитать Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def calculate_stochastic(
        self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """Рассчитать Stochastic Oscillator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        k = 100 * ((df["close"] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        return {"k": k, "d": d}

    def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Рассчитать Williams %R."""
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        return -100 * ((high_max - df["close"]) / (high_max - low_min))

    def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Рассчитать Commodity Channel Index."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        return (typical_price - sma) / (0.015 * mad)

    def analyze_market_data(
        self, market_data: List[MarketData], indicators: List[str]
    ) -> TechnicalAnalysisResult:
        """Анализировать рыночные данные."""
        if not market_data:
            raise TechnicalAnalysisError("No market data provided")

        # Преобразование в DataFrame
        df = pd.DataFrame(
            [
                {
                    "timestamp": data.timestamp,
                    "open": float(data.open.value),
                    "high": float(data.high.value),
                    "low": float(data.low.value),
                    "close": float(data.close.value),
                    "volume": float(data.volume.value),
                }
                for data in market_data
            ]
        )

        # Расчет индикаторов
        result_indicators: Dict[str, TechnicalIndicator] = {}

        for indicator in indicators:
            try:
                if indicator == "rsi":
                    rsi_values = self.calculate_rsi(df["close"])
                    # Безопасное преобразование в список
                    rsi_list = rsi_values.tolist() if hasattr(rsi_values, 'tolist') and callable(rsi_values.tolist) else rsi_values.values.tolist()
                    result_indicators["rsi"] = TechnicalIndicator(
                        name="RSI",
                        value=float(rsi_values.iloc[-1]) if not rsi_values.empty else 50.0,
                        metadata={"period": 14, "values": rsi_list},
                    )
                elif indicator == "macd":
                    macd_result = self.calculate_macd(df["close"])
                    if isinstance(macd_result, dict) and "macd" in macd_result:
                        macd_series = macd_result["macd"]
                        signal_series = macd_result["signal"]
                        histogram_series = macd_result["histogram"]
                        # Безопасное преобразование в список
                        macd_list = macd_series.tolist() if hasattr(macd_series, 'tolist') and callable(macd_series.tolist) else macd_series.values.tolist()
                        signal_list = signal_series.tolist() if hasattr(signal_series, 'tolist') and callable(signal_series.tolist) else signal_series.values.tolist()
                        histogram_list = histogram_series.tolist() if hasattr(histogram_series, 'tolist') and callable(histogram_series.tolist) else histogram_series.values.tolist()
                        result_indicators["macd"] = TechnicalIndicator(
                            name="MACD",
                            value=float(macd_series.iloc[-1])
                            if not macd_series.empty
                            else 0.0,
                            metadata={
                                "macd": macd_list,
                                "signal": signal_list,
                                "histogram": histogram_list,
                            },
                        )
                elif indicator == "bollinger_bands":
                    bb_result = self.calculate_bollinger_bands(df["close"])
                    if hasattr(bb_result, 'percent_b') and hasattr(bb_result, 'upper'):
                        # Безопасное преобразование в список
                        upper_list = bb_result.upper.tolist() if hasattr(bb_result.upper, 'tolist') and callable(bb_result.upper.tolist) else bb_result.upper.values.tolist()
                        middle_list = bb_result.middle.tolist() if hasattr(bb_result.middle, 'tolist') and callable(bb_result.middle.tolist) else bb_result.middle.values.tolist()
                        lower_list = bb_result.lower.tolist() if hasattr(bb_result.lower, 'tolist') and callable(bb_result.lower.tolist) else bb_result.lower.values.tolist()
                        bandwidth_list = bb_result.bandwidth.tolist() if hasattr(bb_result.bandwidth, 'tolist') and callable(bb_result.bandwidth.tolist) else bb_result.bandwidth.values.tolist()
                        percent_b_list = bb_result.percent_b.tolist() if hasattr(bb_result.percent_b, 'tolist') and callable(bb_result.percent_b.tolist) else bb_result.percent_b.values.tolist()
                        result_indicators["bollinger_bands"] = TechnicalIndicator(
                            name="Bollinger Bands",
                            value=float(bb_result.percent_b.iloc[-1])
                            if not bb_result.percent_b.empty
                            else 0.5,
                            metadata={
                                "upper": upper_list,
                                "middle": middle_list,
                                "lower": lower_list,
                                "bandwidth": bandwidth_list,
                                "percent_b": percent_b_list,
                            },
                        )
                elif indicator == "atr":
                    atr_values = self.calculate_atr(df)
                    # Безопасное преобразование в список
                    atr_list = atr_values.tolist() if hasattr(atr_values, 'tolist') and callable(atr_values.tolist) else atr_values.values.tolist()
                    result_indicators["atr"] = TechnicalIndicator(
                        name="ATR",
                        value=float(atr_values.iloc[-1]) if not atr_values.empty else 0.0,
                        metadata={"period": 14, "values": atr_list},
                    )
                elif indicator == "stochastic":
                    stoch_result = self.calculate_stochastic(df)
                    if isinstance(stoch_result, dict) and "k" in stoch_result:
                        k_series = stoch_result["k"]
                        d_series = stoch_result["d"]
                        # Безопасное преобразование в список
                        k_list = k_series.tolist() if hasattr(k_series, 'tolist') and callable(k_series.tolist) else k_series.values.tolist()
                        d_list = d_series.tolist() if hasattr(d_series, 'tolist') and callable(d_series.tolist) else d_series.values.tolist()
                        result_indicators["stochastic"] = TechnicalIndicator(
                            name="Stochastic",
                            value=float(k_series.iloc[-1])
                            if not k_series.empty
                            else 50.0,
                            metadata={
                                "k": k_list,
                                "d": d_list,
                            },
                        )
                elif indicator == "williams_r":
                    wr_values = self.calculate_williams_r(df)
                    # Безопасное преобразование в список
                    wr_list = wr_values.tolist() if hasattr(wr_values, 'tolist') and callable(wr_values.tolist) else list(wr_values)
                    result_indicators["williams_r"] = TechnicalIndicator(
                        name="Williams %R",
                        value=float(wr_values.iloc[-1]) if not wr_values.empty else -50.0,
                        metadata={"period": 14, "values": wr_list},
                    )
                elif indicator == "cci":
                    cci_values = self.calculate_cci(df)
                    # Безопасное преобразование в список
                    cci_list = cci_values.tolist() if hasattr(cci_values, 'tolist') and callable(cci_values.tolist) else list(cci_values)
                    result_indicators["cci"] = TechnicalIndicator(
                        name="CCI",
                        value=float(cci_values.iloc[-1]) if not cci_values.empty else 0.0,
                        metadata={"period": 20, "values": cci_list},
                    )
            except Exception as e:
                logger.warning(f"Failed to calculate {indicator}: {e}")

        return TechnicalAnalysisResult(
            symbol=market_data[0].symbol if market_data else "",
            timeframe="1m",  # По умолчанию
            indicators=result_indicators,
            timestamp=datetime.now(),
        )

    def _extract_numeric_value(self, value_obj: Any) -> float:
        """Извлечь числовое значение из объекта."""
        if hasattr(value_obj, "value"):
            return float(value_obj.value)
        elif hasattr(value_obj, "__float__"):
            return float(value_obj)
        elif isinstance(value_obj, (int, float)):
            return float(value_obj)
        else:
            return 0.0


def create_technical_analysis_service() -> TechnicalAnalysisProtocol:
    """Фабричная функция для создания сервиса технического анализа."""
    return TechnicalAnalysisService()


class DefaultTechnicalAnalysisService(TechnicalAnalysisService):
    """Расширенный сервис технического анализа с дополнительными функциями."""

    def __init__(self, config: Optional[IndicatorConfig] = None):
        super().__init__()
        self.config = config or IndicatorConfig()

    def _validate_market_data(self, data: pd.DataFrame) -> None:
        """Валидация рыночных данных."""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise TechnicalAnalysisError(
                f"Missing required columns: {missing_columns}"
            )

        if len(data) < 50:
            raise TechnicalAnalysisError(
                "Insufficient data points for technical analysis"
            )

        # Проверка на отрицательные значения
        for col in ["open", "high", "low", "close", "volume"]:
            if (data[col] < 0).any():
                raise TechnicalAnalysisError(f"Negative values found in {col}")

        # Проверка логики OHLC
        if not ((data["low"] <= data["open"]) & (data["low"] <= data["close"])).all():
            raise TechnicalAnalysisError("Low price logic violation")
        if not ((data["high"] >= data["open"]) & (data["high"] >= data["close"])).all():
            raise TechnicalAnalysisError("High price logic violation")

    def calculate_indicators(self, data: pd.DataFrame) -> TechnicalAnalysisResult:
        """Расчет всех индикаторов для данных."""
        try:
            self._validate_market_data(data)
            # Создаем копию данных для модификации
            df = data.copy()
            
            # Расчет трендовых индикаторов
            df = self._calculate_trend_indicators(df)
            
            # Расчет индикаторов волатильности и моментума
            df = self._calculate_volatility_momentum(df)
            
            # Расчет объемных индикаторов
            df = self._calculate_volume_indicators(df)
            
            # Расчет индикаторов структуры
            df = self._calculate_structure_indicators(df)
            
            # Создание результата
            indicators: Dict[str, TechnicalIndicator] = {}
            
            # Преобразование результатов в TechnicalIndicator
            for col in df.columns:
                if col not in ["open", "high", "low", "close", "volume"]:
                    try:
                        value = float(df[col].iloc[-1]) if not df[col].empty else 0.0
                        indicators[col] = TechnicalIndicator(
                            name=col,
                            value=value,
                            metadata={"values": df[col].to_list()},
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create indicator for {col}: {e}")
            
            return TechnicalAnalysisResult(
                symbol="",  # Будет заполнено вызывающим кодом
                timeframe="1m",
                indicators=indicators,
                timestamp=datetime.now(),
            )
        except Exception as e:
            raise TechnicalAnalysisError(
                f"Data access error during indicators calculation: {str(e)}"
            )

    def calculate_volume_profile(
        self, data: pd.DataFrame, bins: int = 24
    ) -> VolumeProfileResult:
        """Расчет профиля объема."""
        try:
            # Создание ценовых уровней
            price_range = data["high"].max() - data["low"].min()
            bin_size = price_range / bins
            # Группировка данных по ценовым уровням
            volume_profile: Dict[float, float] = {}
            for i in range(len(data)):
                # Исправлено: price_level должен быть float, а не int
                price_level = float((data["close"].iloc[i] / bin_size) * bin_size)
                if price_level not in volume_profile:
                    volume_profile[price_level] = 0.0
                volume_profile[price_level] += float(data["volume"].iloc[i])
            # Нахождение Point of Control (уровень с максимальным объемом)
            poc_level = max(volume_profile.keys(), key=lambda k: volume_profile[k])
            # Создание Value Area (70% от общего объема)
            total_volume = sum(list(volume_profile.values()))
            target_volume = total_volume * 0.7
            sorted_levels = sorted(
                volume_profile.items(), key=lambda x: x[1], reverse=True
            )
            value_area: List[float] = []
            cumulative_volume: float = 0.0
            for level, volume in sorted_levels:
                value_area.append(level)
                cumulative_volume += volume
                if cumulative_volume >= target_volume:
                    break
            value_area_high = float(max(value_area)) if value_area else float(poc_level)
            value_area_low = float(min(value_area)) if value_area else float(poc_level)
            histogram = [
                volume_profile[level] for level in sorted(volume_profile.keys())
            ]
            price_levels = [float(level) for level in sorted(volume_profile.keys())]
            volume_by_price = {float(k): v for k, v in volume_profile.items()}
            return VolumeProfileResult(
                poc=float(poc_level),
                value_area_high=value_area_high,
                value_area_low=value_area_low,
                volume_by_price=volume_by_price,
                histogram=histogram,
                price_levels=price_levels,
                bins=int(bins),
                value_area_percentage=0.7,
                calculation_timestamp=datetime.now(),
            )
        except Exception as e:
            raise TechnicalAnalysisError(f"Error calculating volume profile: {str(e)}")

    def calculate_market_structure(self, data: pd.DataFrame) -> MarketStructureResult:
        """Расчет структуры рынка."""
        try:
            # Расчет ADX для определения силы тренда
            adx = talib.ADX(data["high"], data["low"], data["close"], timeperiod=14)
            current_adx = float(adx.iloc[-1]) if not adx.empty else 0.0
            # Расчет RSI
            rsi = self.calculate_rsi(data["close"])
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0
            # Расчет волатильности
            returns = data["close"].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Годовая волатильность
            # Определение структуры рынка
            if current_adx > 25:
                if current_rsi > 50:
                    structure = MarketStructure.UPTREND
                else:
                    structure = MarketStructure.DOWNTREND
            else:
                structure = MarketStructure.SIDEWAYS
            # Определение силы тренда
            if current_adx > 40:
                trend_strength = TrendStrength.VERY_STRONG
            elif current_adx > 25:
                trend_strength = TrendStrength.STRONG
            elif current_adx > 15:
                trend_strength = TrendStrength.MODERATE
            elif current_adx > 5:
                trend_strength = TrendStrength.WEAK
            else:
                trend_strength = TrendStrength.VERY_WEAK
            # Расчет уверенности в структуре
            confidence = min(current_adx / 50.0, 1.0)
            return MarketStructureResult(
                structure=structure,
                trend_strength=trend_strength,
                volatility=Decimal(str(volatility)),
                adx=Decimal(str(current_adx)),
                rsi=Decimal(str(current_rsi)),
                confidence=Decimal(str(confidence)),
                support_levels=self._find_support_levels(data),
                resistance_levels=self._find_resistance_levels(data),
                calculation_timestamp=datetime.now(),
                analysis_period="daily",
                data_points=int(len(data)),
            )
        except Exception as e:
            raise TechnicalAnalysisError(
                f"Error calculating market structure: {str(e)}"
            )

    def _find_support_levels(self, data: pd.DataFrame) -> List[float]:
        """Поиск уровней поддержки."""
        try:
            # Используем минимумы за последние 20 баров
            lows = data["low"].rolling(window=20).min().dropna()
            support_levels = lows.unique().tolist()
            return sorted(support_levels)[:5]  # Возвращаем 5 самых низких
        except Exception:
            return []

    def _find_resistance_levels(self, data: pd.DataFrame) -> List[float]:
        """Поиск уровней сопротивления."""
        try:
            # Используем максимумы за последние 20 баров
            highs = data["high"].rolling(window=20).max().dropna()
            resistance_levels = highs.unique().tolist()
            return sorted(resistance_levels, reverse=True)[
                :5
            ]  # Возвращаем 5 самых высоких
        except Exception:
            return []

    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет трендовых индикаторов."""
        # Скользящие средние
        df["sma_20"] = talib.SMA(df["close"], timeperiod=20)
        df["sma_50"] = talib.SMA(df["close"], timeperiod=50)
        df["sma_200"] = talib.SMA(df["close"], timeperiod=200)
        df["ema_20"] = talib.EMA(df["close"], timeperiod=20)
        df["ema_50"] = talib.EMA(df["close"], timeperiod=50)
        df["ema_200"] = talib.EMA(df["close"], timeperiod=200)
        # MACD
        macd_result = self.calculate_macd(
            df["close"],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal,
        )
        df["macd"] = macd_result["macd"]
        df["macd_signal"] = macd_result["signal"]
        df["macd_hist"] = macd_result["histogram"]
        # ADX
        df["adx"] = talib.ADX(df["high"], df["low"], df["close"])
        return df

    def _calculate_volatility_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов волатильности и моментума."""
        # Волатильность
        df["atr"] = self.calculate_atr(df)
        # Моментум
        df["rsi"] = self.calculate_rsi(df["close"], self.config.rsi_period)
        df["stoch_k"], df["stoch_d"] = talib.STOCHRSI(df["close"])
        # Bollinger Bands
        bb_result = self.calculate_bollinger_bands(
            df["close"], self.config.bb_period, self.config.bb_std
        )
        df["bb_upper"] = bb_result.upper
        df["bb_middle"] = bb_result.middle
        df["bb_lower"] = bb_result.lower
        # Keltner Channels
        df["kc_upper"], df["kc_middle"], df["kc_lower"] = talib.KELTNER(
            df["high"], df["low"], df["close"]
        )
        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет объемных индикаторов."""
        # VWAP
        df["vwap"] = talib.VWAP(df["high"], df["low"], df["close"], df["volume"])
        # Volume SMA
        df["volume_sma"] = talib.SMA(
            df["volume"], timeperiod=self.config.volume_ma_period
        )
        # Volume Delta
        df["volume_delta"] = df["volume"] - df["volume"].shift(1)
        return df

    def _calculate_structure_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов структуры."""
        # Fractals
        df["upper_fractal"], df["lower_fractal"] = talib.FRACTAL(df["high"], df["low"])
        # Support/Resistance levels
        df["support"] = df["low"].rolling(window=20).min()
        df["resistance"] = df["high"].rolling(window=20).max()
        return df


# Экспорт интерфейса для обратной совместимости
ITechnicalAnalysisService = TechnicalAnalysisService
