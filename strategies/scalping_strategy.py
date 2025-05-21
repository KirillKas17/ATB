from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal, StrategyMetrics


@dataclass
class ScalpingConfig:
    """Конфигурация скальпинг-стратегии"""

    # Параметры входа
    entry_threshold: float = 0.001  # Порог для входа
    min_volume: float = 1000.0  # Минимальный объем
    min_volatility: float = 0.0005  # Минимальная волатильность
    max_spread: float = 0.0003  # Максимальный спред
    min_tick_size: float = 0.0001  # Минимальный размер тика

    # Параметры выхода
    take_profit: float = 0.002  # Тейк-профит
    stop_loss: float = 0.001  # Стоп-лосс
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_step: float = 0.0005  # Шаг трейлинг-стопа
    max_holding_time: int = 300  # Максимальное время удержания (сек)

    # Параметры управления рисками
    max_position_size: float = 1.0  # Максимальный размер позиции
    max_daily_trades: int = 100  # Максимальное количество сделок в день
    max_daily_loss: float = 0.02  # Максимальный дневной убыток
    risk_per_trade: float = 0.01  # Риск на сделку

    # Параметры мониторинга
    price_deviation_threshold: float = 0.0005  # Порог отклонения цены
    volume_deviation_threshold: float = 0.5  # Порог отклонения объема
    liquidity_threshold: float = 10000.0  # Порог ликвидности
    order_book_depth: int = 10  # Глубина стакана

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1m"])
    log_dir: str = "logs"


class ScalpingStrategy(BaseStrategy):
    """Скальпинг-стратегия (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            ScalpingConfig(**config)
            if config and not isinstance(config, ScalpingConfig)
            else (config or ScalpingConfig())
        )
        self.position = None
        self.entry_time = None
        self.total_position = 0.0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/scalping_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
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

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, volatility, spread, liquidity):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(data, volatility, spread, liquidity)
            if signal:
                self._update_position_state(signal, data)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(
        self, data: pd.DataFrame, volatility: float, spread: float, liquidity: Dict[str, float]
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности

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

    def _generate_trading_signal(
        self, data: pd.DataFrame, volatility: float, spread: float, liquidity: Dict[str, float]
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем вход в позицию
            if not self.position:
                return self._generate_entry_signal(data, volatility, spread, liquidity)

            # Проверяем выход из позиции
            return self._generate_exit_signal(data, volatility, spread, liquidity)

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self, data: pd.DataFrame, volatility: float, spread: float, liquidity: Dict[str, float]
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем отклонение цены
            price_deviation = (
                abs(current_price - data["close"].rolling(window=20).mean().iloc[-1])
                / current_price
            )
            if price_deviation > self.config.price_deviation_threshold:
                return None

            # Проверяем отклонение объема
            volume_deviation = (
                abs(data["volume"].iloc[-1] - data["volume"].rolling(window=20).mean().iloc[-1])
                / data["volume"].iloc[-1]
            )
            if volume_deviation > self.config.volume_deviation_threshold:
                return None

            # Рассчитываем размер позиции
            volume = self._calculate_position_size(current_price, volatility)

            # Устанавливаем стоп-лосс и тейк-профит
            stop_loss = current_price * (1 - self.config.stop_loss)
            take_profit = current_price * (1 + self.config.take_profit)

            # Определяем направление
            direction = "long" if data["close"].iloc[-1] > data["close"].iloc[-2] else "short"

            return Signal(
                direction=direction,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                confidence=min(1.0, volatility / self.config.min_volatility),
                timestamp=datetime.now(),
                metadata={
                    "volatility": volatility,
                    "spread": spread,
                    "liquidity": liquidity,
                    "price_deviation": price_deviation,
                    "volume_deviation": volume_deviation,
                },
            )

        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self, data: pd.DataFrame, volatility: float, spread: float, liquidity: Dict[str, float]
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности

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
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self.config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self.config.trailing_step)

            # Проверяем время удержания
            if (
                self.entry_time
                and (datetime.now() - self.entry_time).total_seconds()
                > self.config.max_holding_time
            ):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=1.0,
                    metadata={
                        "reason": "max_holding_time",
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
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
                self.total_position += signal.volume
                self.entry_time = datetime.now()
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
                self.stop_loss = None
                self.take_profit = None
                self.total_position = 0.0
                self.entry_time = None

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
            position_size = min(position_size, self.config.max_position_size - self.total_position)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
