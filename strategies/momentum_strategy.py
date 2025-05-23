from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class MomentumConfig:
    """Конфигурация стратегии импульса"""

    # Параметры импульса
    momentum_period: int = 14  # Период для расчета импульса
    momentum_threshold: float = 0.02  # Порог импульса
    volume_threshold: float = 1.5  # Порог объема
    trend_period: int = 20  # Период для определения тренда
    trend_threshold: float = 0.01  # Порог тренда

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
    momentum_strength_threshold: float = 0.7  # Порог силы импульса

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class MomentumStrategy(BaseStrategy):
    """Стратегия импульса (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            MomentumConfig(**config)
            if config and not isinstance(config, MomentumConfig)
            else (config or MomentumConfig())
        )
        self.position = None
        self.entry_price = None
        self.total_position = 0.0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.last_trade_time = None
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/momentum_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            # Анализ импульса
            momentum = self._analyze_momentum(data)

            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
                "momentum": momentum,
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
            momentum = analysis["momentum"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(
                data, volatility, spread, liquidity, momentum
            ):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(
                data, volatility, spread, liquidity, momentum
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
        momentum: Dict[str, Any],
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            momentum: Показатели импульса

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

            # Проверка силы импульса
            if momentum["strength"] < self.config.momentum_strength_threshold:
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

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ импульса.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с показателями импульса
        """
        try:
            # Расчет импульса
            momentum = data["close"].pct_change(periods=self.config.momentum_period)

            # Расчет тренда
            trend = data["close"].pct_change(periods=self.config.trend_period)

            # Расчет силы импульса
            momentum_strength = abs(momentum.iloc[-1]) / momentum.std()

            # Расчет направления импульса
            momentum_direction = "up" if momentum.iloc[-1] > 0 else "down"

            # Расчет ускорения
            acceleration = momentum.diff().iloc[-1]

            return {
                "value": momentum.iloc[-1],
                "trend": trend.iloc[-1],
                "strength": momentum_strength,
                "direction": momentum_direction,
                "acceleration": acceleration,
            }

        except Exception as e:
            logger.error(f"Error analyzing momentum: {str(e)}")
            return {}

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        momentum: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            momentum: Показатели импульса

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            data["close"].iloc[-1]

            # Проверяем вход в позицию
            if not self.position:
                return self._generate_entry_signal(
                    data, volatility, spread, liquidity, momentum
                )

            # Проверяем выход из позиции
            return self._generate_exit_signal(
                data, volatility, spread, liquidity, momentum
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
        momentum: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            momentum: Показатели импульса

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем импульс вверх
            if (
                momentum["direction"] == "up"
                and momentum["value"] > self.config.momentum_threshold
                and momentum["trend"] > self.config.trend_threshold
                and momentum["acceleration"] > 0
            ):

                # Проверяем объем
                if (
                    data["volume"].iloc[-1]
                    > data["volume"].rolling(window=20).mean().iloc[-1]
                    * self.config.volume_threshold
                ):
                    # Рассчитываем размер позиции
                    volume = self._calculate_position_size(current_price, volatility)

                    # Устанавливаем стоп-лосс и тейк-профит
                    stop_loss = current_price * (1 - self.config.stop_loss)
                    take_profit = current_price * (1 + self.config.take_profit)

                    return Signal(
                        direction="long",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=min(1.0, momentum["strength"]),
                        timestamp=datetime.now(),
                        metadata={
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "momentum": momentum,
                        },
                    )

            # Проверяем импульс вниз
            elif (
                momentum["direction"] == "down"
                and momentum["value"] < -self.config.momentum_threshold
                and momentum["trend"] < -self.config.trend_threshold
                and momentum["acceleration"] < 0
            ):

                # Проверяем объем
                if (
                    data["volume"].iloc[-1]
                    > data["volume"].rolling(window=20).mean().iloc[-1]
                    * self.config.volume_threshold
                ):
                    # Рассчитываем размер позиции
                    volume = self._calculate_position_size(current_price, volatility)

                    # Устанавливаем стоп-лосс и тейк-профит
                    stop_loss = current_price * (1 + self.config.stop_loss)
                    take_profit = current_price * (1 - self.config.take_profit)

                    return Signal(
                        direction="short",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=min(1.0, momentum["strength"]),
                        timestamp=datetime.now(),
                        metadata={
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "momentum": momentum,
                        },
                    )

            return None

        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        momentum: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            momentum: Показатели импульса

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
                            "momentum": momentum,
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
                            "momentum": momentum,
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
                            "momentum": momentum,
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
                            "momentum": momentum,
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self.config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self.config.trailing_step)

            # Проверяем ослабление импульса
            if self._check_momentum_weakening(momentum):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=1.0,
                    metadata={
                        "reason": "momentum_weakening",
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
                        "momentum": momentum,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _check_momentum_weakening(self, momentum: Dict[str, Any]) -> bool:
        """
        Проверка ослабления импульса.

        Args:
            momentum: Показатели импульса

        Returns:
            bool: Результат проверки
        """
        try:
            if self.position == "long":
                # Проверяем ослабление восходящего импульса
                if momentum["direction"] == "down" or momentum["acceleration"] < 0:
                    return True
            else:
                # Проверяем ослабление нисходящего импульса
                if momentum["direction"] == "up" or momentum["acceleration"] > 0:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking momentum weakening: {str(e)}")
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
