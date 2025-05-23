from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class RegimeConfig:
    """Конфигурация адаптивной стратегии"""

    min_confidence: float = 0.7
    max_drawdown: float = 0.1
    adaptation_rate: float = 0.1
    atr_period: int = 14
    atr_multiplier: float = 2.0
    volume_ma_period: int = 20
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    trailing_stop: bool = True
    trailing_step: float = 0.002
    partial_close: bool = True
    partial_close_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    partial_close_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    max_trades: Optional[int] = None
    min_volume: float = 0.0
    min_volatility: float = 0.0
    max_spread: float = 0.01
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"
    use_volume: bool = True
    use_volatility: bool = True
    use_correlation: bool = True
    use_regime: bool = True

    # Параметры для разных режимов
    trend_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "bb_period": 20,
            "bb_std": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 25.0,
        }
    )

    volatility_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "rsi_period": 10,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "bb_period": 15,
            "bb_std": 2.5,
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 5,
            "adx_period": 10,
            "adx_threshold": 30.0,
        }
    )

    sideways_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "rsi_period": 20,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "bb_period": 25,
            "bb_std": 1.8,
            "macd_fast": 16,
            "macd_slow": 32,
            "macd_signal": 9,
            "adx_period": 20,
            "adx_threshold": 20.0,
        }
    )


class RegimeAdaptiveStrategy(BaseStrategy):
    """Адаптивная стратегия, меняющая параметры в зависимости от режима рынка (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        if config is None:
            config = {}
        self.config = (
            RegimeConfig(**config) if not isinstance(config, RegimeConfig) else config
        )
        super().__init__(self.config.__dict__)
        self.position = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.partial_closes = []
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/regime_adaptive_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных данных.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с результатами анализа
        """
        try:
            is_valid, error = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")

            regime = self.detect_regime(data)
            params = self._get_regime_params(regime)
            indicators = self._calculate_indicators(data, params)
            market_state = self._analyze_market_state(data, indicators, regime)
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "regime": regime,
                "indicators": indicators,
                "market_state": market_state,
                "risk_metrics": risk_metrics,
                "timestamp": data.index[-1],
            }
        except Exception as e:
            logger.error(f"Error analyzing data: {str(e)}")
            return {}

    def generate_signal(self, data: pd.DataFrame) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            analysis = self.analyze(data)
            if not analysis:
                return None

            regime = analysis["regime"]
            indicators = analysis["indicators"]
            market_state = analysis["market_state"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, indicators):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(
                data, indicators, market_state, regime
            )
            if signal:
                self._update_position_state(signal, data)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(
        self, data: pd.DataFrame, indicators: Dict[str, float]
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов

        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка объема
            if data["volume"].iloc[-1] < self.config.min_volume:
                return False

            # Проверка волатильности
            if indicators["volatility"] < self.config.min_volatility:
                return False

            # Проверка спреда
            spread = (data["high"].iloc[-1] - data["low"].iloc[-1]) / data[
                "close"
            ].iloc[-1]
            if spread > self.config.max_spread:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            regime: Режим рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            if self.position is None:
                return self._generate_entry_signal(
                    data, indicators, market_state, regime
                )
            else:
                return self._generate_exit_signal(
                    data, indicators, market_state, regime
                )

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            regime: Режим рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]
            params = self._get_regime_params(regime)

            # Проверяем условия для длинной позиции
            if (
                indicators["rsi"] < params["rsi_oversold"]
                and current_price < indicators["bb_lower"]
                and indicators["macd"] > indicators["macd_signal"]
                and indicators["adx"] > params["adx_threshold"]
            ):

                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, indicators["atr"])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = (
                    current_price - indicators["atr"] * self.config.atr_multiplier
                )
                take_profit = (
                    current_price + indicators["atr"] * self.config.atr_multiplier * 2
                )

                return Signal(
                    direction="long",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=self._calculate_confidence(indicators, regime),
                    timestamp=data.index[-1],
                    metadata={
                        "regime": regime,
                        "indicators": indicators,
                        "market_state": market_state,
                    },
                )

            # Проверяем условия для короткой позиции
            elif (
                indicators["rsi"] > params["rsi_overbought"]
                and current_price > indicators["bb_upper"]
                and indicators["macd"] < indicators["macd_signal"]
                and indicators["adx"] > params["adx_threshold"]
            ):

                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, indicators["atr"])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = (
                    current_price + indicators["atr"] * self.config.atr_multiplier
                )
                take_profit = (
                    current_price - indicators["atr"] * self.config.atr_multiplier * 2
                )

                return Signal(
                    direction="short",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=self._calculate_confidence(indicators, regime),
                    timestamp=data.index[-1],
                    metadata={
                        "regime": regime,
                        "indicators": indicators,
                        "market_state": market_state,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            regime: Режим рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if current_price <= self.stop_loss or current_price >= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss_or_take_profit",
                            "regime": regime,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            elif self.position == "short":
                if current_price >= self.stop_loss or current_price <= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss_or_take_profit",
                            "regime": regime,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop and self.trailing_stop:
                if self.position == "long" and current_price > self.trailing_stop:
                    self.trailing_stop = (
                        current_price - indicators["atr"] * self.config.trailing_step
                    )
                elif self.position == "short" and current_price < self.trailing_stop:
                    self.trailing_stop = (
                        current_price + indicators["atr"] * self.config.trailing_step
                    )

                if (
                    self.position == "long" and current_price <= self.trailing_stop
                ) or (self.position == "short" and current_price >= self.trailing_stop):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "trailing_stop",
                            "regime": regime,
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            # Проверяем частичное закрытие
            if self.config.partial_close and self.partial_closes:
                for level, size in zip(
                    self.config.partial_close_levels, self.config.partial_close_sizes
                ):
                    if (
                        self.position == "long"
                        and current_price >= self.take_profit * level
                    ):
                        return Signal(
                            direction="partial_close",
                            entry_price=current_price,
                            volume=size,
                            timestamp=data.index[-1],
                            confidence=1.0,
                            metadata={
                                "reason": "partial_close",
                                "level": level,
                                "size": size,
                                "regime": regime,
                                "indicators": indicators,
                                "market_state": market_state,
                            },
                        )
                    elif (
                        self.position == "short"
                        and current_price <= self.take_profit * level
                    ):
                        return Signal(
                            direction="partial_close",
                            entry_price=current_price,
                            volume=size,
                            timestamp=data.index[-1],
                            confidence=1.0,
                            metadata={
                                "reason": "partial_close",
                                "level": level,
                                "size": size,
                                "regime": regime,
                                "indicators": indicators,
                                "market_state": market_state,
                            },
                        )

            return None

        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _update_position_state(self, signal: Signal, data: pd.DataFrame):
        """
        Обновление состояния позиции.

        Args:
            signal: Торговый сигнал
            data: DataFrame с OHLCV данными
        """
        try:
            if signal.direction in ["long", "short"]:
                self.position = signal.direction
                self.stop_loss = signal.stop_loss
                self.take_profit = signal.take_profit
                self.trailing_stop = signal.entry_price
                self.partial_closes = []
            elif signal.direction == "close":
                self.position = None
                self.stop_loss = None
                self.take_profit = None
                self.trailing_stop = None
                self.partial_closes = []
            elif signal.direction == "partial_close":
                self.partial_closes.append(signal.volume)

        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        Определение режима рынка.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            str: Режим рынка ('trend', 'volatility', 'sideways')
        """
        try:
            # Расчет волатильности
            returns = data["close"].pct_change()
            volatility = returns.std()

            # Расчет тренда
            sma_20 = data["close"].rolling(window=20).mean()
            sma_50 = data["close"].rolling(window=50).mean()
            trend_strength = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]

            # Расчет ADX
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            plus_dm = data["high"].diff()
            minus_dm = data["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (
                plus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean()
            )
            minus_di = 100 * (
                minus_dm.rolling(window=14).mean() / tr.rolling(window=14).mean()
            )
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=14).mean()

            # Определение режима
            if trend_strength > 0.02 and adx.iloc[-1] > 25:
                return "trend"
            elif volatility > 0.015:
                return "volatility"
            else:
                return "sideways"

        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return "sideways"

    def _get_regime_params(self, regime: str) -> Dict[str, Any]:
        """
        Получение параметров для режима.

        Args:
            regime: Режим рынка

        Returns:
            Dict с параметрами
        """
        try:
            if regime == "trend":
                return self.config.trend_params
            elif regime == "volatility":
                return self.config.volatility_params
            else:
                return self.config.sideways_params

        except Exception as e:
            logger.error(f"Error getting regime parameters: {str(e)}")
            return self.config.sideways_params

    def _calculate_indicators(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Расчет индикаторов.

        Args:
            data: DataFrame с OHLCV данными
            params: Параметры индикаторов

        Returns:
            Dict с значениями индикаторов
        """
        try:
            # RSI
            delta = data["close"].diff()
            gain = (
                (delta.where(delta > 0, 0)).rolling(window=params["rsi_period"]).mean()
            )
            loss = (
                (-delta.where(delta < 0, 0)).rolling(window=params["rsi_period"]).mean()
            )
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Bollinger Bands
            bb_middle = data["close"].rolling(window=params["bb_period"]).mean()
            bb_std = data["close"].rolling(window=params["bb_period"]).std()
            bb_upper = bb_middle + (bb_std * params["bb_std"])
            bb_lower = bb_middle - (bb_std * params["bb_std"])

            # MACD
            ema_fast = data["close"].ewm(span=params["macd_fast"]).mean()
            ema_slow = data["close"].ewm(span=params["macd_slow"]).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=params["macd_signal"]).mean()

            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=self.config.atr_period).mean()

            # ADX
            plus_dm = data["high"].diff()
            minus_dm = data["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (
                plus_dm.rolling(window=params["adx_period"]).mean()
                / tr.rolling(window=params["adx_period"]).mean()
            )
            minus_di = 100 * (
                minus_dm.rolling(window=params["adx_period"]).mean()
                / tr.rolling(window=params["adx_period"]).mean()
            )
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=params["adx_period"]).mean()

            # Волатильность
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=20).std()

            # Volume MA
            volume_ma = (
                data["volume"].rolling(window=self.config.volume_ma_period).mean()
            )

            return {
                "rsi": rsi.iloc[-1],
                "bb_upper": bb_upper.iloc[-1],
                "bb_middle": bb_middle.iloc[-1],
                "bb_lower": bb_lower.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": macd_signal.iloc[-1],
                "atr": atr.iloc[-1],
                "adx": adx.iloc[-1],
                "plus_di": plus_di.iloc[-1],
                "minus_di": minus_di.iloc[-1],
                "volatility": volatility.iloc[-1],
                "volume_ma": volume_ma.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def _analyze_market_state(
        self, data: pd.DataFrame, indicators: Dict[str, float], regime: str
    ) -> Dict[str, Any]:
        """
        Анализ состояния рынка.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            regime: Режим рынка

        Returns:
            Dict с состоянием рынка
        """
        try:
            # Тренд
            trend = "up" if indicators["macd"] > indicators["macd_signal"] else "down"

            # Сила тренда
            trend_strength = "strong" if indicators["adx"] > 25 else "weak"

            # Волатильность
            volatility = "high" if indicators["volatility"] > 0.015 else "low"

            # Объем
            volume = (
                "high" if data["volume"].iloc[-1] > indicators["volume_ma"] else "low"
            )

            # Перекупленность/перепроданность
            overbought = indicators["rsi"] > 70
            oversold = indicators["rsi"] < 30

            return {
                "regime": regime,
                "trend": trend,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "volume": volume,
                "overbought": overbought,
                "oversold": oversold,
            }

        except Exception as e:
            logger.error(f"Error analyzing market state: {str(e)}")
            return {}

    def _calculate_position_size(self, price: float, atr: float) -> float:
        """
        Расчет размера позиции.

        Args:
            price: Текущая цена
            atr: ATR

        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            base_size = self.config.risk_per_trade

            # Корректировка на волатильность
            volatility_factor = 1 / (1 + atr / price)

            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(position_size, self.config.max_position_size)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _calculate_confidence(self, indicators: Dict[str, float], regime: str) -> float:
        """
        Расчет уверенности в сигнале.

        Args:
            indicators: Значения индикаторов
            regime: Режим рынка

        Returns:
            float: Уверенность (0-1)
        """
        try:
            # Нормализация индикаторов
            rsi_conf = 1 - abs(indicators["rsi"] - 50) / 50
            macd_conf = abs(indicators["macd"] - indicators["macd_signal"]) / abs(
                indicators["macd_signal"]
            )
            adx_conf = indicators["adx"] / 100
            bb_conf = (
                1
                - abs(indicators["bb_middle"] - indicators["bb_lower"])
                / indicators["bb_middle"]
            )

            # Взвешенная сумма в зависимости от режима
            if regime == "trend":
                confidence = (
                    0.3 * macd_conf + 0.3 * adx_conf + 0.2 * rsi_conf + 0.2 * bb_conf
                )
            elif regime == "volatility":
                confidence = (
                    0.3 * bb_conf + 0.3 * rsi_conf + 0.2 * macd_conf + 0.2 * adx_conf
                )
            else:  # sideways
                confidence = (
                    0.3 * rsi_conf + 0.3 * bb_conf + 0.2 * macd_conf + 0.2 * adx_conf
                )

            return min(max(confidence, 0), 1)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
