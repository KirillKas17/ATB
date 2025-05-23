from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class BreakoutConfig:
    """Конфигурация стратегии пробоя"""

    # Параметры пробоя
    breakout_period: int = 20  # Период для определения пробоя
    breakout_threshold: float = 0.02  # Порог пробоя
    min_volume_multiplier: float = 1.5  # Множитель объема для подтверждения
    confirmation_periods: int = 3  # Количество периодов для подтверждения
    false_breakout_threshold: float = 0.01  # Порог ложного пробоя

    # Параметры входа
    entry_threshold: float = 0.01  # Порог для входа
    min_volume: float = 1000.0  # Минимальный объем
    min_volatility: float = 0.01  # Минимальная волатильность
    max_spread: float = 0.001  # Максимальный спред

    # Параметры выхода
    take_profit: float = 0.03  # Тейк-профит
    stop_loss: float = 0.015  # Стоп-лосс
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_step: float = 0.005  # Шаг трейлинг-стопа

    # Параметры управления рисками
    max_position_size: float = 1.0  # Максимальный размер позиции
    max_daily_trades: int = 10  # Максимальное количество сделок в день
    max_daily_loss: float = 0.02  # Максимальный дневной убыток
    risk_per_trade: float = 0.02  # Риск на сделку

    # Параметры мониторинга
    price_deviation_threshold: float = 0.002  # Порог отклонения цены
    volume_deviation_threshold: float = 0.5  # Порог отклонения объема
    liquidity_threshold: float = 10000.0  # Порог ликвидности
    trend_strength_threshold: float = 0.7  # Порог силы тренда

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class BreakoutStrategy(BaseStrategy):
    """Стратегия пробоя (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            BreakoutConfig(**config)
            if config and not isinstance(config, BreakoutConfig)
            else (config or BreakoutConfig())
        )
        self.position = None
        self.entry_price = None
        self.breakout_level = None
        self.total_position = 0.0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/breakout_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            # Расчет волатильности
            volatility = self._calculate_volatility(data)

            # Расчет спреда
            spread = self._calculate_spread(data)

            # Анализ ликвидности
            liquidity = self._analyze_liquidity(data)

            # Анализ тренда
            trend = self._analyze_trend(data)

            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
                "trend": trend,
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

            volatility = analysis["volatility"]
            spread = analysis["spread"]
            liquidity = analysis["liquidity"]
            trend = analysis["trend"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(
                data, volatility, spread, liquidity, trend
            ):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(
                data, volatility, spread, liquidity, trend
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
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        trend: Dict[str, Any],
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            trend: Показатели тренда

        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка волатильности
            if volatility < self.config.min_volatility:
                return False

            # Проверка спреда
            if spread > self.config.max_spread:
                return False

            # Проверка ликвидности
            if liquidity["volume"] < self.config.min_volume:
                return False

            if liquidity["depth"] < self.config.liquidity_threshold:
                return False

            # Проверка силы тренда
            if trend["strength"] < self.config.trend_strength_threshold:
                return False

            # Проверка размера позиции
            if self.total_position >= self.config.max_position_size:
                return False

            # Проверка количества сделок
            if self.daily_trades >= self.config.max_daily_trades:
                return False

            # Проверка дневного убытка
            if self.daily_pnl <= -self.config.max_daily_loss:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Расчет волатильности.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Текущая волатильность
        """
        try:
            # Расчет волатильности как стандартного отклонения доходности
            returns = data["close"].pct_change()
            volatility = returns.std()
            return volatility

        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _calculate_spread(self, data: pd.DataFrame) -> float:
        """
        Расчет спреда.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Текущий спред
        """
        try:
            # Расчет спреда как разницы между ценами
            spread = (data["ask"] - data["bid"]).iloc[-1]
            return spread

        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return 0.0

    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Анализ ликвидности.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с показателями ликвидности
        """
        try:
            # Расчет объема
            volume = data["volume"].iloc[-1]

            # Расчет глубины рынка
            depth = (data["ask_volume"] + data["bid_volume"]).iloc[-1]

            # Расчет спреда объема
            volume_spread = abs(data["ask_volume"] - data["bid_volume"]).iloc[-1]

            return {"volume": volume, "depth": depth, "volume_spread": volume_spread}

        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {}

    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ тренда.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с показателями тренда
        """
        try:
            # Расчет скользящих средних
            sma_short = data["close"].rolling(window=10).mean()
            sma_long = data["close"].rolling(window=30).mean()

            # Расчет направления тренда
            trend_direction = "up" if sma_short.iloc[-1] > sma_long.iloc[-1] else "down"

            # Расчет силы тренда
            trend_strength = (
                abs(sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            )

            # Расчет уровней поддержки и сопротивления
            support = (
                data["low"].rolling(window=self.config.breakout_period).min().iloc[-1]
            )
            resistance = (
                data["high"].rolling(window=self.config.breakout_period).max().iloc[-1]
            )

            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "support": support,
                "resistance": resistance,
            }

        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {}

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        trend: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            trend: Показатели тренда

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            data["close"].iloc[-1]

            # Проверяем вход в позицию
            if not self.position:
                return self._generate_entry_signal(
                    data, volatility, spread, liquidity, trend
                )

            # Проверяем выход из позиции
            return self._generate_exit_signal(
                data, volatility, spread, liquidity, trend
            )

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        trend: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            trend: Показатели тренда

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем пробой уровня
            if current_price > trend["resistance"]:
                # Проверяем объем
                if (
                    data["volume"].iloc[-1]
                    > data["volume"].rolling(window=20).mean().iloc[-1]
                    * self.config.min_volume_multiplier
                ):
                    # Проверяем подтверждение
                    if self._check_breakout_confirmation(data, "up"):
                        # Рассчитываем размер позиции
                        volume = self._calculate_position_size(
                            current_price, volatility
                        )

                        # Устанавливаем стоп-лосс и тейк-профит
                        stop_loss = current_price * (1 - self.config.stop_loss)
                        take_profit = current_price * (1 + self.config.take_profit)

                        self.breakout_level = trend["resistance"]

                        return Signal(
                            direction="long",
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            volume=volume,
                            confidence=min(1.0, trend["strength"]),
                            timestamp=datetime.now(),
                            metadata={
                                "volatility": volatility,
                                "spread": spread,
                                "liquidity": liquidity,
                                "trend": trend,
                                "breakout_level": trend["resistance"],
                            },
                        )

            elif current_price < trend["support"]:
                # Проверяем объем
                if (
                    data["volume"].iloc[-1]
                    > data["volume"].rolling(window=20).mean().iloc[-1]
                    * self.config.min_volume_multiplier
                ):
                    # Проверяем подтверждение
                    if self._check_breakout_confirmation(data, "down"):
                        # Рассчитываем размер позиции
                        volume = self._calculate_position_size(
                            current_price, volatility
                        )

                        # Устанавливаем стоп-лосс и тейк-профит
                        stop_loss = current_price * (1 + self.config.stop_loss)
                        take_profit = current_price * (1 - self.config.take_profit)

                        self.breakout_level = trend["support"]

                        return Signal(
                            direction="short",
                            entry_price=current_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            volume=volume,
                            confidence=min(1.0, trend["strength"]),
                            timestamp=datetime.now(),
                            metadata={
                                "volatility": volatility,
                                "spread": spread,
                                "liquidity": liquidity,
                                "trend": trend,
                                "breakout_level": trend["support"],
                            },
                        )

            return None

        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _check_breakout_confirmation(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Проверка подтверждения пробоя.

        Args:
            data: DataFrame с OHLCV данными
            direction: Направление пробоя ('up' или 'down')

        Returns:
            bool: Результат проверки
        """
        try:
            # Проверяем последние периоды
            for i in range(self.config.confirmation_periods):
                if direction == "up":
                    if data["close"].iloc[-(i + 1)] <= self.breakout_level:
                        return False
                else:
                    if data["close"].iloc[-(i + 1)] >= self.breakout_level:
                        return False

            return True

        except Exception as e:
            logger.error(f"Error checking breakout confirmation: {str(e)}")
            return False

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        trend: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            trend: Показатели тренда

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
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "trend": trend,
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
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "trend": trend,
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
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "trend": trend,
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
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "trend": trend,
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self.config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self.config.trailing_step)

            # Проверяем ложный пробой
            if self._check_false_breakout(data):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=1.0,
                    metadata={
                        "reason": "false_breakout",
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
                        "trend": trend,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _check_false_breakout(self, data: pd.DataFrame) -> bool:
        """
        Проверка ложного пробоя.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            bool: Результат проверки
        """
        try:
            if not self.breakout_level:
                return False

            current_price = data["close"].iloc[-1]

            if self.position == "long":
                # Проверяем возврат цены ниже уровня пробоя
                if current_price < self.breakout_level * (
                    1 - self.config.false_breakout_threshold
                ):
                    return True
            else:
                # Проверяем возврат цены выше уровня пробоя
                if current_price > self.breakout_level * (
                    1 + self.config.false_breakout_threshold
                ):
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking false breakout: {str(e)}")
            return False

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
                self.entry_price = signal.entry_price
                self.stop_loss = signal.stop_loss
                self.take_profit = signal.take_profit
                self.total_position += signal.volume
                self.daily_trades += 1
                self.last_trade_time = data.index[-1]
            elif signal.direction == "close":
                # Обновляем дневной P&L
                if self.position == "long":
                    self.daily_pnl += (
                        data["close"].iloc[-1] - self.entry_price
                    ) * self.total_position
                else:
                    self.daily_pnl += (
                        self.entry_price - data["close"].iloc[-1]
                    ) * self.total_position

                self.position = None
                self.entry_price = None
                self.stop_loss = None
                self.take_profit = None
                self.breakout_level = None
                self.total_position = 0.0

        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def _calculate_position_size(self, price: float, volatility: float) -> float:
        """
        Расчет размера позиции.

        Args:
            price: Текущая цена
            volatility: Текущая волатильность

        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            base_size = self.config.risk_per_trade

            # Корректировка на волатильность
            volatility_factor = 1 / (1 + volatility)

            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(
                position_size, self.config.max_position_size - self.total_position
            )

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
