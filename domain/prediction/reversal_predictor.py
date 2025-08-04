# -*- coding: utf-8 -*-
"""Reversal Predictor Domain Model for Price Reversal Prediction."""
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from scipy.signal import find_peaks

from domain.prediction.reversal_signal import (
    CandlestickPattern,
    DivergenceSignal,
    MeanReversionBand,
    MomentumAnalysis,
    ReversalSignal,
)
from domain.type_definitions.prediction_types import (
    CandlestickPatternType,
    ConfidenceScore,
    DivergenceType,
    OHLCVData,
    OrderBookData,
    PredictionConfig,
    ReversalDirection,
    SignalStrength,
    SignalStrengthScore,
)
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume_profile import VolumeProfile
from domain.interfaces.price_pattern_extractor import PricePatternExtractor


class ReversalPredictor:
    """Продвинутый прогнозатор разворотов цены."""

    def __init__(self, config: Optional[PredictionConfig] = None) -> None:
        """
        Инициализация прогнозатора.
        Args:
            config: Конфигурация прогнозирования
        """
        self.config: PredictionConfig = config or {}
        self.pattern_extractor: PricePatternExtractor = PricePatternExtractor()
        logger.info(f"ReversalPredictor initialized with config: {self.config}")

    def predict_reversal(
        self,
        symbol: str,
        market_data: OHLCVData,
        order_book: Optional[OrderBookData] = None,
    ) -> Optional[ReversalSignal]:
        """
        Прогнозирование разворота цены.
        Args:
            symbol: Символ торговой пары
            market_data: OHLCV данные
            order_book: Данные ордербука (опционально)
        Returns:
            Optional[ReversalSignal]: Сигнал разворота или None
        """
        try:
            df: pd.DataFrame = (
                market_data
                if isinstance(market_data, pd.DataFrame)
                else pd.DataFrame(market_data)
            )
            lookback = self.config.get("lookback_period", 100)
            if len(df) < lookback:
                logger.warning(
                    f"Недостаточно данных для прогнозирования: {len(df)} < {lookback}"
                )
                return None
            
            min_confidence = self.config.get("min_confidence", 0.3)
            min_signal_strength = self.config.get("min_signal_strength", 0.4)
            prediction_horizon = self.config.get(
                "prediction_horizon", timedelta(hours=4)
            )
            high_pivots, low_pivots = self.pattern_extractor.extract_pivot_points(df)
            volume_profile_raw = self.pattern_extractor.extract_volume_profile(df)
            volume_profile = (
                volume_profile_raw
                if isinstance(volume_profile_raw, VolumeProfile)
                else None
            )
            liquidity_clusters = self.pattern_extractor.extract_liquidity_clusters(
                order_book
            )
            divergence_signals = self._analyze_divergences(df)
            candlestick_patterns = self._analyze_candlestick_patterns(df)
            momentum_analysis = self._analyze_momentum(df)
            mean_reversion_band = self._analyze_mean_reversion(df)
            direction = self._determine_reversal_direction(
                df, high_pivots, low_pivots, divergence_signals, candlestick_patterns
            )
            if direction == ReversalDirection.NEUTRAL:
                logger.debug(f"Нейтральное направление для {symbol}")
                return None
            
            pivot_price = self._calculate_reversal_level(
                direction, df, high_pivots, low_pivots, volume_profile
            )
            confidence = self._calculate_confidence(
                direction, divergence_signals, candlestick_patterns, momentum_analysis
            )
            if confidence < min_confidence:
                logger.debug(
                    f"Недостаточная уверенность для {symbol}: {confidence:.3f}"
                )
                return None
            
            signal_strength = self._calculate_signal_strength(
                confidence, divergence_signals, candlestick_patterns, momentum_analysis
            )
            if signal_strength < min_signal_strength:
                logger.debug(
                    f"Недостаточная сила сигнала для {symbol}: {signal_strength:.3f}"
                )
                return None
            
            signal = ReversalSignal(
                symbol=symbol,
                direction=direction,
                pivot_price=pivot_price,
                confidence=confidence,
                horizon=prediction_horizon,
                signal_strength=signal_strength,
                timestamp=Timestamp(datetime.now()),
                pivot_points=high_pivots + low_pivots,
                volume_profile=volume_profile,
                liquidity_clusters=liquidity_clusters,
                divergence_signals=divergence_signals,
                candlestick_patterns=candlestick_patterns,
                momentum_analysis=momentum_analysis,
                mean_reversion_band=mean_reversion_band,
            )
            logger.info(
                f"Создан сигнал разворота для {symbol}: {direction.value}, "
                f"цена={pivot_price.value:.2f}, уверенность={confidence:.3f}, "
                f"сила={signal_strength:.3f}"
            )
            return signal
        except Exception as e:
            logger.error(f"Ошибка прогнозирования разворота для {symbol}: {e}")
            return None

    def _analyze_divergences(self, data: pd.DataFrame) -> List[DivergenceSignal]:
        """Анализ дивергенций RSI и MACD."""
        try:
            # Исправляем передачу DataFrame вместо OHLCVData в методы анализа
            df: pd.DataFrame = data  # Убираем ненужное преобразование
            divergences: List[DivergenceSignal] = []
            rsi_divergences = self._detect_rsi_divergences(df)
            divergences.extend(rsi_divergences)
            macd_divergences = self._detect_macd_divergences(df)
            divergences.extend(macd_divergences)
            return divergences
        except Exception as e:
            logger.error(f"Ошибка анализа дивергенций: {e}")
            return []

    def _detect_rsi_divergences(self, data: pd.DataFrame) -> List[DivergenceSignal]:
        """Обнаружение дивергенций RSI."""
        try:
            df: pd.DataFrame = data
            rsi = self._calculate_rsi(df["close"], period=14)
            price_highs = self._find_peaks(np.asarray(df["high"]), "high")
            price_lows = self._find_peaks(np.asarray(df["low"]), "low")
            rsi_highs = self._find_peaks(np.asarray(rsi), "high")
            rsi_lows = self._find_peaks(np.asarray(rsi), "low")
            divergences: List[DivergenceSignal] = []
            # Исправление: price_highs, rsi_highs — это списки индексов, а не функции
            if len(price_highs) >= 2 and len(rsi_highs) >= 2:
                if (
                    float(df["high"].iloc[price_highs[-1]]) > float(df["high"].iloc[price_highs[-2]])
                    and float(rsi.iloc[rsi_highs[-1]]) < float(rsi.iloc[rsi_highs[-2]])
                ):
                    divergences.append(
                        DivergenceSignal(
                            type=DivergenceType.BEARISH_REGULAR,
                            indicator="RSI",
                            price_highs=[
                                float(df["high"].iloc[i]) for i in price_highs[-2:]
                            ],
                            price_lows=[],
                            indicator_highs=[
                                float(rsi.iloc[i]) for i in rsi_highs[-2:]
                            ],
                            indicator_lows=[],
                            strength=float(0.7),
                            confidence=ConfidenceScore(0.8),
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
            if len(price_lows) >= 2 and len(rsi_lows) >= 2:
                if (
                    float(df["low"].iloc[price_lows[-1]]) < float(df["low"].iloc[price_lows[-2]])
                    and float(rsi.iloc[rsi_lows[-1]]) > float(rsi.iloc[rsi_lows[-2]])
                ):
                    divergences.append(
                        DivergenceSignal(
                            type=DivergenceType.BULLISH_REGULAR,
                            indicator="RSI",
                            price_highs=[],
                            price_lows=[
                                float(df["low"].iloc[i]) for i in price_lows[-2:]
                            ],
                            indicator_highs=[],
                            indicator_lows=[float(rsi.iloc[i]) for i in rsi_lows[-2:]],
                            strength=float(0.7),
                            confidence=ConfidenceScore(0.8),
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
            return divergences
        except Exception as e:
            logger.error(f"Ошибка анализа дивергенций RSI: {e}")
            return []

    def _detect_macd_divergences(self, data: pd.DataFrame) -> List[DivergenceSignal]:
        """Обнаружение дивергенций MACD."""
        try:
            df: pd.DataFrame = data
            # Вычисляем MACD
            macd, signal = self._calculate_macd(df["close"])
            # Находим экстремумы цены и MACD
            price_highs = self._find_peaks(np.asarray(df["high"]), "high")
            price_lows = self._find_peaks(np.asarray(df["low"]), "low")
            macd_highs = self._find_peaks(np.asarray(macd), "high")
            macd_lows = self._find_peaks(np.asarray(macd), "low")
            divergences: List[DivergenceSignal] = []
            # Проверяем медвежьи дивергенции
            if len(price_highs) >= 2 and len(macd_highs) >= 2:
                if (
                    float(df["high"].iloc[price_highs[-1]]) > float(df["high"].iloc[price_highs[-2]])
                    and float(macd.iloc[macd_highs[-1]]) < float(macd.iloc[macd_highs[-2]])
                ):
                    divergences.append(
                        DivergenceSignal(
                            type=DivergenceType.BEARISH_REGULAR,
                            indicator="MACD",
                            price_highs=[
                                float(df["high"].iloc[i]) for i in price_highs[-2:]
                            ],
                            price_lows=[],
                            indicator_highs=[
                                float(macd.iloc[i]) for i in macd_highs[-2:]
                            ],
                            indicator_lows=[],
                            strength=float(0.6),
                            confidence=ConfidenceScore(0.7),
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
            # Проверяем бычьи дивергенции
            if len(price_lows) >= 2 and len(macd_lows) >= 2:
                if (
                    float(df["low"].iloc[price_lows[-1]]) < float(df["low"].iloc[price_lows[-2]])
                    and float(macd.iloc[macd_lows[-1]]) > float(macd.iloc[macd_lows[-2]])
                ):
                    divergences.append(
                        DivergenceSignal(
                            type=DivergenceType.BULLISH_REGULAR,
                            indicator="MACD",
                            price_highs=[],
                            price_lows=[
                                float(df["low"].iloc[i]) for i in price_lows[-2:]
                            ],
                            indicator_highs=[],
                            indicator_lows=[
                                float(macd.iloc[i]) for i in macd_lows[-2:]
                            ],
                            strength=float(0.6),
                            confidence=ConfidenceScore(0.7),
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
            return divergences
        except Exception as e:
            logger.error(f"Ошибка анализа дивергенций MACD: {e}")
            return []

    def _analyze_candlestick_patterns(
        self, data: pd.DataFrame
    ) -> List[CandlestickPattern]:
        """Анализ свечных паттернов."""
        try:
            patterns: List[CandlestickPattern] = []
            # Исправляем использование callable как индекса
            for i in range(len(data)):
                # Исправление: безопасное обращение к данным
                if hasattr(data, 'iloc'):
                    candle = data.iloc[i]
                else:
                    candle = data[i]
                
                if self._is_doji(candle):
                    # Исправляем передачу параметров в CandlestickPattern
                    patterns.append(
                        CandlestickPattern(
                            name="doji",
                            direction=ReversalDirection.NEUTRAL,
                            strength=0.6,
                            confirmation_level=0.5,
                            volume_confirmation=False,
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
                elif self._is_hammer(candle):
                    # Исправляем передачу параметров в CandlestickPattern
                    patterns.append(
                        CandlestickPattern(
                            name="hammer",
                            direction=ReversalDirection.BULLISH,
                            strength=0.7,
                            confirmation_level=0.6,
                            volume_confirmation=True,
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
                elif self._is_shooting_star(candle):
                    # Исправляем передачу параметров в CandlestickPattern
                    patterns.append(
                        CandlestickPattern(
                            name="shooting_star",
                            direction=ReversalDirection.BEARISH,
                            strength=0.7,
                            confirmation_level=0.6,
                            volume_confirmation=True,
                            timestamp=Timestamp(datetime.now()),
                        )
                    )
            return patterns
        except Exception as e:
            logger.error(f"Ошибка анализа свечных паттернов: {e}")
            return []

    def _analyze_momentum(self, data: pd.DataFrame) -> Optional[MomentumAnalysis]:
        """Анализ импульса."""
        try:
            close_prices = data["close"]
            if len(close_prices) < 20:
                return None
            
            # Проверяем, что momentum не None
            momentum_series = close_prices.diff(5)
            momentum_value = float(momentum_series.mean()) if not momentum_series.empty else 0.0
            acceleration_series: pd.Series = momentum_series.diff(1)
            acceleration_value = float(acceleration_series.mean()) if not acceleration_series.empty else 0.0
            volume_momentum_series = data["volume"].diff(5) if "volume" in data.columns else pd.Series()
            volume_momentum_value = float(volume_momentum_series.mean()) if not volume_momentum_series.empty else 0.0
            
            return MomentumAnalysis(
                timestamp=Timestamp(datetime.now()),
                momentum_loss=float(abs(momentum_value)),
                velocity_change=momentum_value,
                acceleration=acceleration_value,
                volume_momentum=volume_momentum_value,
                price_momentum=momentum_value,
                momentum_divergence=None
            )
        except Exception as e:
            logger.error(f"Ошибка анализа импульса: {e}")
            return None

    def _analyze_mean_reversion(
        self, data: pd.DataFrame
    ) -> Optional[MeanReversionBand]:
        """Анализ среднего возврата."""
        try:
            close_prices = data["close"]
            if len(close_prices) < 20:
                return None
            
            # Исправляем использование Series методов
            sma_20 = close_prices.rolling(window=20).mean()
            sma_50 = close_prices.rolling(window=50).mean()
            
            current_price = close_prices.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            # Исправляем использование Series методов
            volatility = close_prices.pct_change().std()
            
            # Исправляем передачу параметров в MeanReversionBand
            middle_line = Price(Decimal(str(current_sma_20)), Currency("USD"))
            upper_band = Price(Decimal(str(current_sma_20 + 2 * volatility * current_sma_20)), Currency("USD"))
            lower_band = Price(Decimal(str(current_sma_20 - 2 * volatility * current_sma_20)), Currency("USD"))
            
            return MeanReversionBand(
                upper_band=upper_band,
                lower_band=lower_band,
                middle_line=middle_line,
                deviation=float(abs(current_price - current_sma_20) / current_sma_20),
                band_width=float(4 * volatility * current_sma_20),
                current_position=float((current_price - lower_band.value) / (upper_band.value - lower_band.value)),
                timestamp=Timestamp(datetime.now()),
            )
        except Exception as e:
            logger.error(f"Ошибка анализа среднего возврата: {e}")
            return None

    def _determine_reversal_direction(
        self,
        data: pd.DataFrame,
        high_pivots: List[Any],
        low_pivots: List[Any],
        divergence_signals: List[DivergenceSignal],
        candlestick_patterns: List[CandlestickPattern],
    ) -> ReversalDirection:
        """Определение направления разворота."""
        try:
            bullish_signals = 0
            bearish_signals = 0
            
            # Анализ дивергенций
            for signal in divergence_signals:
                if signal.type in [DivergenceType.BULLISH_REGULAR, DivergenceType.BULLISH_HIDDEN]:
                    bullish_signals += 1
                elif signal.type in [DivergenceType.BEARISH_REGULAR, DivergenceType.BEARISH_HIDDEN]:
                    bearish_signals += 1
            
            # Анализ свечных паттернов
            for pattern in candlestick_patterns:
                if pattern.name in ["hammer", "doji"]:
                    bullish_signals += 1
                elif pattern.name == "shooting_star":
                    bearish_signals += 1
            
            # Анализ пивотов
            if len(high_pivots) > len(low_pivots):
                bearish_signals += 1
            elif len(low_pivots) > len(high_pivots):
                bullish_signals += 1
            
            if bullish_signals > bearish_signals:
                return ReversalDirection.BULLISH
            elif bearish_signals > bullish_signals:
                return ReversalDirection.BEARISH
            else:
                return ReversalDirection.NEUTRAL
        except Exception as e:
            logger.error(f"Ошибка определения направления разворота: {e}")
            return ReversalDirection.NEUTRAL

    def _calculate_reversal_level(
        self,
        direction: ReversalDirection,
        data: pd.DataFrame,
        high_pivots: List[Any],
        low_pivots: List[Any],
        volume_profile: Optional[VolumeProfile],
    ) -> Price:
        """Расчет уровня разворота."""
        try:
            current_price = float(data["close"].iloc[-1])
            
            if direction == ReversalDirection.BULLISH:
                if low_pivots:
                    # Исправляем передачу Decimal в Price
                    return Price(Decimal(str(min(low_pivots))), Currency("USD"))
                else:
                    return Price(Decimal(str(current_price * 0.95)), Currency("USD"))
            elif direction == ReversalDirection.BEARISH:
                if high_pivots:
                    return Price(Decimal(str(max(high_pivots))), Currency("USD"))
                else:
                    return Price(Decimal(str(current_price * 1.05)), Currency("USD"))
            else:
                return Price(Decimal(str(current_price)), Currency("USD"))
        except Exception as e:
            logger.error(f"Ошибка расчета уровня разворота: {e}")
            return Price(Decimal("0"), Currency("USD"))

    def _calculate_confidence(
        self,
        direction: ReversalDirection,
        divergence_signals: List[DivergenceSignal],
        candlestick_patterns: List[CandlestickPattern],
        momentum_analysis: Optional[MomentumAnalysis],
    ) -> ConfidenceScore:
        """Расчет уверенности сигнала."""
        try:
            confidence = 0.5  # Базовая уверенность
            
            # Исправляем возвращаемые типы для ConfidenceScore
            for signal in divergence_signals:
                confidence += float(signal.confidence) * 0.2
            
            for pattern in candlestick_patterns:
                confidence += float(pattern.confirmation_level) * 0.1
            
            if momentum_analysis:
                # Исправляем использование атрибутов MomentumAnalysis
                if momentum_analysis.momentum_loss > 0.1 and direction == ReversalDirection.BEARISH:
                    confidence += 0.2
                elif momentum_analysis.momentum_loss < -0.1 and direction == ReversalDirection.BULLISH:
                    confidence += 0.2
            
            return ConfidenceScore(min(confidence, 1.0))
        except Exception as e:
            logger.error(f"Ошибка расчета уверенности: {e}")
            return ConfidenceScore(0.5)

    def _calculate_signal_strength(
        self,
        confidence: ConfidenceScore,
        divergence_signals: List[DivergenceSignal],
        candlestick_patterns: List[CandlestickPattern],
        momentum_analysis: Optional[MomentumAnalysis],
    ) -> SignalStrengthScore:
        """Расчет силы сигнала."""
        try:
            strength = float(confidence) * 0.6  # Базовая сила
            
            # Добавляем силу от дивергенций
            for signal in divergence_signals:
                strength += float(signal.strength) * 0.2
            
            # Добавляем силу от свечных паттернов
            for pattern in candlestick_patterns:
                strength += float(pattern.strength) * 0.1
            
            # Добавляем силу от импульса
            if momentum_analysis:
                momentum_value = float(momentum_analysis.momentum_loss)
                strength += abs(momentum_value) * 0.1
            
            return SignalStrengthScore(min(strength, 1.0))
        except Exception as e:
            logger.error(f"Ошибка расчета силы сигнала: {e}")
            return SignalStrengthScore(0.5)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI."""
        try:
            if len(prices) < period:
                return pd.Series([50.0] * len(prices))
            
            # Явное приведение типов для корректной работы с mypy
            prices_float = prices.astype(float)
            delta = prices_float.diff()
            gain = delta.where(delta > 0.0, 0.0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0.0, 0.0)).rolling(window=period).mean()
            
            # Избегаем деления на ноль
            rs = gain / loss.replace(0.0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            logger.error(f"Ошибка расчета RSI: {e}")
            return pd.Series([50.0] * len(prices))

    def _calculate_macd(
        self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Расчет MACD."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd, signal_line
        except Exception as e:
            logger.error(f"Ошибка расчета MACD: {e}")
            return pd.Series([0.0] * len(prices)), pd.Series([0.0] * len(prices))

    def _find_peaks(self, arr: np.ndarray, peak_type: str) -> List[int]:
        """Поиск пиков в массиве."""
        try:
            if len(arr) == 0:
                return []
            arr_mean = float(np.mean(arr))
            if peak_type == "high":
                peaks, _ = find_peaks(arr, height=arr_mean)
            else:  # low
                peaks, _ = find_peaks(-arr, height=-arr_mean)
            return [int(x) for x in peaks.tolist()]
        except Exception as e:
            logger.error(f"Ошибка поиска пиков: {e}")
            return []

    def _is_doji(self, candle: pd.Series) -> bool:
        """Проверка на паттерн Doji."""
        try:
            body_size = abs(float(candle["close"]) - float(candle["open"]))
            total_range = float(candle["high"]) - float(candle["low"])
            return body_size <= total_range * 0.1
        except Exception as e:
            logger.error(f"Ошибка проверки Doji: {e}")
            return False

    def _is_hammer(self, candle: pd.Series) -> bool:
        """Проверка на паттерн Hammer."""
        try:
            body_size = abs(float(candle["close"]) - float(candle["open"]))
            lower_shadow = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
            upper_shadow = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
            return bool(lower_shadow > 2 * body_size and upper_shadow < body_size)
        except Exception as e:
            logger.error(f"Ошибка проверки Hammer: {e}")
            return False

    def _is_shooting_star(self, candle: pd.Series) -> bool:
        """Проверка на паттерн Shooting Star."""
        try:
            body_size = abs(float(candle["close"]) - float(candle["open"]))
            lower_shadow = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
            upper_shadow = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
            return bool(upper_shadow > 2 * body_size and lower_shadow < body_size)
        except Exception as e:
            logger.error(f"Ошибка проверки Shooting Star: {e}")
            return False
