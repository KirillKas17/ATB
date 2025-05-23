from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class VolatilityConfig:
    """Конфигурация стратегии волатильности"""

    volatility_window: int = 20
    bb_period: int = 20
    bb_std: float = 2.0
    min_volatility: float = 0.01
    max_volatility: float = 0.05
    atr_period: int = 14
    atr_multiplier: float = 2.0
    keltner_period: int = 20
    keltner_multiplier: float = 2.0
    donchian_period: int = 20
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
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


class VolatilityStrategy(BaseStrategy):
    """Стратегия торговли по волатильности (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            VolatilityConfig(**config)
            if config and not isinstance(config, VolatilityConfig)
            else (config or VolatilityConfig())
        )
        self.position = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.partial_closes = []
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        log_dir = getattr(self.config, "log_dir", "logs")
        logger.add(
            f"{log_dir}/volatility_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            indicators = self._calculate_indicators(data)
            market_state = self._analyze_market_state(data, indicators)
            risk_metrics = self.calculate_risk_metrics(data)

            return {
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

            indicators = analysis["indicators"]
            market_state = analysis["market_state"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, indicators):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(data, indicators, market_state)
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
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            if self.position is None:
                return self._generate_entry_signal(data, indicators, market_state)
            else:
                return self._generate_exit_signal(data, indicators, market_state)

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем условия для входа в длинную позицию
            if (
                indicators["volatility"] > self.config.min_volatility
                and current_price < indicators["bb_lower"]
                and indicators["rsi"] < self.config.rsi_oversold
            ):

                # Рассчитываем уровни
                atr = indicators["atr"]
                stop_loss = current_price - atr * self.config.atr_multiplier
                take_profit = current_price + atr * self.config.atr_multiplier * 2

                # Рассчитываем объем
                volume = self._calculate_position_size(current_price, atr)

                # Рассчитываем уверенность
                confidence = self._calculate_confidence(indicators)

                if confidence >= self.config.min_confidence:
                    return Signal(
                        direction="long",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=confidence,
                        timestamp=data.index[-1],
                        metadata={
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )

            # Проверяем условия для входа в короткую позицию
            elif (
                indicators["volatility"] > self.config.min_volatility
                and current_price > indicators["bb_upper"]
                and indicators["rsi"] > self.config.rsi_overbought
            ):

                # Рассчитываем уровни
                atr = indicators["atr"]
                stop_loss = current_price + atr * self.config.atr_multiplier
                take_profit = current_price - atr * self.config.atr_multiplier * 2

                # Рассчитываем объем
                volume = self._calculate_position_size(current_price, atr)

                # Рассчитываем уверенность
                confidence = self._calculate_confidence(indicators)

                if confidence >= self.config.min_confidence:
                    return Signal(
                        direction="short",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=confidence,
                        timestamp=data.index[-1],
                        metadata={
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
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка

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

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет индикаторов.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с значениями индикаторов
        """
        try:
            # Волатильность
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=self.config.volatility_window).std()

            # Bollinger Bands
            bb_middle = data["close"].rolling(window=self.config.bb_period).mean()
            bb_std = data["close"].rolling(window=self.config.bb_period).std()
            bb_upper = bb_middle + (bb_std * self.config.bb_std)
            bb_lower = bb_middle - (bb_std * self.config.bb_std)

            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=self.config.atr_period).mean()

            # Keltner Channels
            keltner_middle = (
                data["close"].rolling(window=self.config.keltner_period).mean()
            )
            keltner_atr = atr.rolling(window=self.config.keltner_period).mean()
            keltner_upper = keltner_middle + (
                keltner_atr * self.config.keltner_multiplier
            )
            keltner_lower = keltner_middle - (
                keltner_atr * self.config.keltner_multiplier
            )

            # Donchian Channels
            donchian_upper = (
                data["high"].rolling(window=self.config.donchian_period).max()
            )
            donchian_lower = (
                data["low"].rolling(window=self.config.donchian_period).min()
            )
            donchian_middle = (donchian_upper + donchian_lower) / 2

            # RSI
            delta = data["close"].diff()
            gain = (
                (delta.where(delta > 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            loss = (
                (-delta.where(delta < 0, 0))
                .rolling(window=self.config.rsi_period)
                .mean()
            )
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # Volume MA
            volume_ma = (
                data["volume"].rolling(window=self.config.volume_ma_period).mean()
            )

            return {
                "volatility": volatility.iloc[-1],
                "bb_upper": bb_upper.iloc[-1],
                "bb_middle": bb_middle.iloc[-1],
                "bb_lower": bb_lower.iloc[-1],
                "atr": atr.iloc[-1],
                "keltner_upper": keltner_upper.iloc[-1],
                "keltner_middle": keltner_middle.iloc[-1],
                "keltner_lower": keltner_lower.iloc[-1],
                "donchian_upper": donchian_upper.iloc[-1],
                "donchian_middle": donchian_middle.iloc[-1],
                "donchian_lower": donchian_lower.iloc[-1],
                "rsi": rsi.iloc[-1],
                "volume_ma": volume_ma.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def _analyze_market_state(
        self, data: pd.DataFrame, indicators: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Анализ состояния рынка.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов

        Returns:
            Dict с состоянием рынка
        """
        try:
            # Волатильность
            volatility_state = (
                "high"
                if indicators["volatility"] > self.config.max_volatility
                else "low"
            )

            # Bollinger Bands
            bb_width = (indicators["bb_upper"] - indicators["bb_lower"]) / indicators[
                "bb_middle"
            ]
            bb_state = "expanding" if bb_width > bb_width * 1.1 else "contracting"

            # Keltner Channels
            keltner_width = (
                indicators["keltner_upper"] - indicators["keltner_lower"]
            ) / indicators["keltner_middle"]
            keltner_state = (
                "expanding" if keltner_width > keltner_width * 1.1 else "contracting"
            )

            # Donchian Channels
            donchian_width = (
                indicators["donchian_upper"] - indicators["donchian_lower"]
            ) / indicators["donchian_middle"]
            donchian_state = (
                "expanding" if donchian_width > donchian_width * 1.1 else "contracting"
            )

            # RSI
            rsi_state = (
                "overbought"
                if indicators["rsi"] > self.config.rsi_overbought
                else (
                    "oversold"
                    if indicators["rsi"] < self.config.rsi_oversold
                    else "neutral"
                )
            )

            # Объем
            volume_state = (
                "high" if data["volume"].iloc[-1] > indicators["volume_ma"] else "low"
            )

            return {
                "volatility_state": volatility_state,
                "bb_state": bb_state,
                "keltner_state": keltner_state,
                "donchian_state": donchian_state,
                "rsi_state": rsi_state,
                "volume_state": volume_state,
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

    def _calculate_confidence(self, indicators: Dict[str, float]) -> float:
        """
        Расчет уверенности в сигнале.

        Args:
            indicators: Значения индикаторов

        Returns:
            float: Уверенность (0-1)
        """
        try:
            # Нормализация индикаторов
            bb_conf = (
                1
                - abs(indicators["bb_middle"] - indicators["bb_lower"])
                / indicators["bb_middle"]
            )
            keltner_conf = (
                1
                - abs(indicators["keltner_middle"] - indicators["keltner_lower"])
                / indicators["keltner_middle"]
            )
            donchian_conf = (
                1
                - abs(indicators["donchian_middle"] - indicators["donchian_lower"])
                / indicators["donchian_middle"]
            )
            rsi_conf = 1 - abs(indicators["rsi"] - 50) / 50
            volatility_conf = (
                1
                - abs(indicators["volatility"] - self.config.min_volatility)
                / self.config.max_volatility
            )

            # Взвешенная сумма
            confidence = (
                0.3 * bb_conf
                + 0.3 * keltner_conf
                + 0.2 * donchian_conf
                + 0.1 * rsi_conf
                + 0.1 * volatility_conf
            )

            return min(max(confidence, 0), 1)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0

    def get_parameters(self) -> Dict[str, Any]:
        """Получение параметров стратегии"""
        return {
            "volatility_window": self.config.volatility_window,
            "bb_period": self.config.bb_period,
            "bb_std": self.config.bb_std,
            "min_volatility": self.config.min_volatility,
            "max_volatility": self.config.max_volatility,
            "atr_period": self.config.atr_period,
            "atr_multiplier": self.config.atr_multiplier,
            "keltner_period": self.config.keltner_period,
            "keltner_multiplier": self.config.keltner_multiplier,
            "donchian_period": self.config.donchian_period,
            "rsi_period": self.config.rsi_period,
            "rsi_overbought": self.config.rsi_overbought,
            "rsi_oversold": self.config.rsi_oversold,
            "volume_ma_period": self.config.volume_ma_period,
            "risk_per_trade": self.config.risk_per_trade,
            "max_position_size": self.config.max_position_size,
            "trailing_stop": self.config.trailing_stop,
            "trailing_step": self.config.trailing_step,
            "partial_close": self.config.partial_close,
            "partial_close_levels": self.config.partial_close_levels,
            "partial_close_sizes": self.config.partial_close_sizes,
            "max_trades": self.config.max_trades,
            "min_volume": self.config.min_volume,
            "max_spread": self.config.max_spread,
            "symbols": self.config.symbols,
            "timeframes": self.config.timeframes,
            "log_dir": self.config.log_dir,
        }
