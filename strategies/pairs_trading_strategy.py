from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class PairsTradingConfig:
    """Конфигурация парной торговой стратегии"""

    # Параметры парной торговли
    lookback_period: int = 100  # Период для расчета статистик
    z_score_threshold: float = 2.0  # Порог для входа в позицию
    min_correlation: float = 0.7  # Минимальная корреляция между инструментами
    min_cointegration: float = 0.7  # Минимальный уровень коинтеграции
    min_half_life: int = 5  # Минимальное время полураспада
    max_half_life: int = 50  # Максимальное время полураспада
    hedge_ratio: float = 1.0  # Соотношение хеджирования

    # Параметры индикаторов
    rsi_period: int = 14
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    atr_period: int = 14
    atr_multiplier: float = 1.0

    # Параметры управления рисками
    risk_per_trade: float = 0.02
    max_position_size: float = 0.2
    trailing_stop: bool = True
    trailing_step: float = 0.002
    partial_close: bool = True
    partial_close_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    partial_close_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class PairsTradingStrategy(BaseStrategy):
    """Парная торговая стратегия (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            PairsTradingConfig(**config)
            if config and not isinstance(config, PairsTradingConfig)
            else (config or PairsTradingConfig())
        )
        self.position = None
        self.hedge_position = None
        self.stop_loss = None
        self.take_profit = None
        self.trailing_stop = None
        self.partial_closes = []
        self.spread_mean = None
        self.spread_std = None
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/pairs_trading_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            # Расчет статистик спреда
            spread_stats = self._calculate_spread_statistics(data)

            return {
                "indicators": indicators,
                "market_state": market_state,
                "risk_metrics": risk_metrics,
                "spread_stats": spread_stats,
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
            spread_stats = analysis["spread_stats"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, indicators, spread_stats):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(
                data, indicators, market_state, spread_stats
            )
            if signal:
                self._update_position_state(signal, data)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        spread_stats: Dict[str, float],
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            spread_stats: Статистики спреда

        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка коинтеграции
            if spread_stats["cointegration"] < self.config.min_cointegration:
                return False

            # Проверка времени полураспада
            if not (
                self.config.min_half_life
                <= spread_stats["half_life"]
                <= self.config.max_half_life
            ):
                return False

            # Проверка корреляции
            if spread_stats["correlation"] < self.config.min_correlation:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _calculate_spread_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет статистик спреда.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с статистиками спреда
        """
        try:
            if len(self.config.symbols) < 2:
                return {}

            # Расчет спреда
            spread = (
                data[self.config.symbols[0]]["close"]
                - data[self.config.symbols[1]]["close"]
            )

            # Расчет статистик
            mean = spread.rolling(window=self.config.lookback_period).mean()
            std = spread.rolling(window=self.config.lookback_period).std()
            z_score = (spread - mean) / std

            # Расчет коинтеграции
            cointegration = self._calculate_cointegration(data)

            # Расчет времени полураспада
            half_life = self._calculate_half_life(spread)

            # Расчет корреляции
            correlation = data[self.config.symbols[0]]["close"].corr(
                data[self.config.symbols[1]]["close"]
            )

            return {
                "mean": mean.iloc[-1],
                "std": std.iloc[-1],
                "z_score": z_score.iloc[-1],
                "cointegration": cointegration,
                "half_life": half_life,
                "correlation": correlation,
            }

        except Exception as e:
            logger.error(f"Error calculating spread statistics: {str(e)}")
            return {}

    def _calculate_cointegration(self, data: pd.DataFrame) -> float:
        """
        Расчет коинтеграции между инструментами.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Уровень коинтеграции
        """
        try:
            if len(self.config.symbols) < 2:
                return 0.0

            # Расчет коинтеграции
            x = data[self.config.symbols[0]]["close"]
            y = data[self.config.symbols[1]]["close"]

            # Регрессия
            beta = np.cov(x, y)[0, 1] / np.var(x)
            spread = y - beta * x

            # Тест на стационарность
            adf_stat = self._calculate_adf_statistic(spread)

            return abs(adf_stat)

        except Exception as e:
            logger.error(f"Error calculating cointegration: {str(e)}")
            return 0.0

    def _calculate_adf_statistic(self, series: pd.Series) -> float:
        """
        Расчет статистики теста Дики-Фуллера.

        Args:
            series: Временной ряд

        Returns:
            float: Статистика теста
        """
        try:
            # Расчет разницы
            diff = series.diff().dropna()

            # Регрессия
            y = diff
            x = series.shift(1).dropna()
            x = x[:-1]
            y = y[1:]

            # Расчет статистики
            beta = np.cov(x, y)[0, 1] / np.var(x)
            residuals = y - beta * x
            std_error = np.std(residuals)

            return beta / std_error

        except Exception as e:
            logger.error(f"Error calculating ADF statistic: {str(e)}")
            return 0.0

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Расчет времени полураспада.

        Args:
            spread: Спред между инструментами

        Returns:
            float: Время полураспада
        """
        try:
            # Расчет разницы
            diff = spread.diff().dropna()

            # Регрессия
            y = diff
            x = spread.shift(1).dropna()
            x = x[:-1]
            y = y[1:]

            # Расчет времени полураспада
            beta = np.cov(x, y)[0, 1] / np.var(x)
            half_life = -np.log(2) / beta

            return half_life

        except Exception as e:
            logger.error(f"Error calculating half life: {str(e)}")
            return 0.0

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        spread_stats: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            spread_stats: Статистики спреда

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            data["close"].iloc[-1]

            if self.position is None:
                return self._generate_entry_signal(
                    data, indicators, market_state, spread_stats
                )
            else:
                return self._generate_exit_signal(
                    data, indicators, market_state, spread_stats
                )

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        spread_stats: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            spread_stats: Статистики спреда

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем условия для длинной позиции
            if spread_stats["z_score"] < -self.config.z_score_threshold:
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, indicators["atr"])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = (
                    current_price - indicators["atr"] * self.config.atr_multiplier
                )
                take_profit = current_price + (current_price - stop_loss) * 2

                return Signal(
                    direction="long",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=self._calculate_confidence(indicators, spread_stats),
                    timestamp=data.index[-1],
                    metadata={
                        "indicators": indicators,
                        "market_state": market_state,
                        "spread_stats": spread_stats,
                    },
                )

            # Проверяем условия для короткой позиции
            elif spread_stats["z_score"] > self.config.z_score_threshold:
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, indicators["atr"])

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = (
                    current_price + indicators["atr"] * self.config.atr_multiplier
                )
                take_profit = current_price - (stop_loss - current_price) * 2

                return Signal(
                    direction="short",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=self._calculate_confidence(indicators, spread_stats),
                    timestamp=data.index[-1],
                    metadata={
                        "indicators": indicators,
                        "market_state": market_state,
                        "spread_stats": spread_stats,
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
        spread_stats: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            spread_stats: Статистики спреда

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if current_price <= self.stop_loss:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss",
                            "indicators": indicators,
                            "market_state": market_state,
                            "spread_stats": spread_stats,
                        },
                    )
                elif current_price >= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "take_profit",
                            "indicators": indicators,
                            "market_state": market_state,
                            "spread_stats": spread_stats,
                        },
                    )

            elif self.position == "short":
                if current_price >= self.stop_loss:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss",
                            "indicators": indicators,
                            "market_state": market_state,
                            "spread_stats": spread_stats,
                        },
                    )
                elif current_price <= self.take_profit:
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "take_profit",
                            "indicators": indicators,
                            "market_state": market_state,
                            "spread_stats": spread_stats,
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
                            "spread_stats": spread_stats,
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
                                "spread_stats": spread_stats,
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
                                "spread_stats": spread_stats,
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

                # Создаем хеджирующую позицию
                self._create_hedge_position(signal, data)
            elif signal.direction == "close":
                self.position = None
                self.hedge_position = None
                self.stop_loss = None
                self.take_profit = None
                self.trailing_stop = None
                self.partial_closes = []
            elif signal.direction == "partial_close":
                self.partial_closes.append(signal.volume)

        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def _create_hedge_position(self, signal: Signal, data: pd.DataFrame):
        """
        Создание хеджирующей позиции.

        Args:
            signal: Торговый сигнал
            data: DataFrame с OHLCV данными
        """
        try:
            if len(self.config.symbols) < 2:
                return

            # Выбираем инструмент для хеджирования
            hedge_symbol = self._select_hedge_symbol(data)
            if not hedge_symbol:
                return

            # Создаем хеджирующую позицию
            hedge_volume = signal.volume * self.config.hedge_ratio
            hedge_direction = "short" if signal.direction == "long" else "long"

            self.hedge_position = {
                "symbol": hedge_symbol,
                "direction": hedge_direction,
                "volume": hedge_volume,
                "entry_price": data[hedge_symbol]["close"].iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error creating hedge position: {str(e)}")

    def _select_hedge_symbol(self, data: pd.DataFrame) -> Optional[str]:
        """
        Выбор инструмента для хеджирования.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Optional[str]: Символ инструмента для хеджирования
        """
        try:
            if len(self.config.symbols) < 2:
                return None

            # Находим инструмент с наибольшей корреляцией
            correlations = []
            for symbol in self.config.symbols:
                if symbol != self.position:
                    corr = data[self.position]["close"].corr(data[symbol]["close"])
                    correlations.append((symbol, abs(corr)))

            if not correlations:
                return None

            return max(correlations, key=lambda x: x[1])[0]

        except Exception as e:
            logger.error(f"Error selecting hedge symbol: {str(e)}")
            return None

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет индикаторов.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с значениями индикаторов
        """
        try:
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

            # MACD
            exp1 = data["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
            exp2 = data["close"].ewm(span=self.config.macd_slow, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
            macd_hist = macd - signal

            # Bollinger Bands
            sma = data["close"].rolling(window=self.config.bollinger_period).mean()
            std = data["close"].rolling(window=self.config.bollinger_period).std()
            upper_band = sma + (std * self.config.bollinger_std)
            lower_band = sma - (std * self.config.bollinger_std)

            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            atr = true_range.rolling(window=self.config.atr_period).mean()

            return {
                "rsi": rsi.iloc[-1],
                "macd": macd.iloc[-1],
                "macd_signal": signal.iloc[-1],
                "macd_hist": macd_hist.iloc[-1],
                "bollinger_upper": upper_band.iloc[-1],
                "bollinger_lower": lower_band.iloc[-1],
                "atr": atr.iloc[-1],
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
            # Направление тренда
            trend_direction = "up" if indicators["macd_hist"] > 0 else "down"

            # Волатильность
            volatility = (
                "high" if indicators["atr"] > data["close"].iloc[-1] * 0.01 else "low"
            )

            # Объем
            volume_state = (
                "high"
                if data["volume"].iloc[-1]
                > data["volume"].rolling(window=20).mean().iloc[-1]
                else "low"
            )

            return {
                "trend_direction": trend_direction,
                "volatility": volatility,
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

    def _calculate_confidence(
        self, indicators: Dict[str, float], spread_stats: Dict[str, float]
    ) -> float:
        """
        Расчет уверенности в сигнале.

        Args:
            indicators: Значения индикаторов
            spread_stats: Статистики спреда

        Returns:
            float: Уверенность (0-1)
        """
        try:
            # Нормализация индикаторов
            rsi_conf = 1 - abs(indicators["rsi"] - 50) / 50
            macd_conf = min(abs(indicators["macd_hist"]) / abs(indicators["macd"]), 1)
            bollinger_conf = (
                1
                - abs(indicators["bollinger_upper"] - indicators["bollinger_lower"])
                / indicators["bollinger_upper"]
            )

            # Нормализация статистик спреда
            z_score_conf = (
                1 - abs(spread_stats["z_score"]) / self.config.z_score_threshold
            )
            cointegration_conf = spread_stats["cointegration"]
            half_life_conf = 1 - abs(
                spread_stats["half_life"]
                - (self.config.min_half_life + self.config.max_half_life) / 2
            ) / (self.config.max_half_life - self.config.min_half_life)

            # Взвешенная сумма
            confidence = (
                0.2 * rsi_conf
                + 0.2 * macd_conf
                + 0.2 * bollinger_conf
                + 0.2 * z_score_conf
                + 0.2 * cointegration_conf
                + 0.2 * half_life_conf
            )

            return min(max(confidence, 0), 1)

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
