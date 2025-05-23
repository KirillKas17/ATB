from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class MartingaleConfig:
    """Конфигурация стратегии Мартингейла"""

    # Параметры Мартингейла
    initial_bet: float = 0.01  # Начальная ставка
    multiplier: float = 2.0  # Множитель ставки
    max_bet: float = 1.0  # Максимальная ставка
    max_steps: int = 5  # Максимальное количество шагов
    reset_on_win: bool = True  # Сброс на начальную ставку после выигрыша

    # Параметры управления рисками
    max_daily_loss: float = 0.1  # Максимальный дневной убыток
    max_consecutive_losses: int = 3  # Максимальное количество последовательных убытков
    min_balance: float = 100.0  # Минимальный баланс
    risk_per_trade: float = 0.02  # Риск на сделку
    max_position_size: float = 0.2  # Максимальный размер позиции

    # Параметры входа
    entry_threshold: float = 0.02  # Порог для входа
    min_volume: float = 1000.0  # Минимальный объем
    min_volatility: float = 0.01  # Минимальная волатильность
    max_spread: float = 0.001  # Максимальный спред

    # Параметры выхода
    take_profit: float = 0.02  # Тейк-профит
    stop_loss: float = 0.01  # Стоп-лосс
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_step: float = 0.005  # Шаг трейлинг-стопа

    # Параметры мониторинга
    price_deviation_threshold: float = 0.002  # Порог отклонения цены
    volume_deviation_threshold: float = 0.5  # Порог отклонения объема
    liquidity_threshold: float = 10000.0  # Порог ликвидности

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1m"])
    log_dir: str = "logs"


class MartingaleStrategy(BaseStrategy):
    """Стратегия Мартингейла (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            MartingaleConfig(**config)
            if config and not isinstance(config, MartingaleConfig)
            else (config or MartingaleConfig())
        )
        self.position = None
        self.current_bet = self.config.initial_bet
        self.step = 0
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.last_trade_time = None
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/martingale_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
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
            # Проверка лимитов
            if self.daily_loss >= self.config.max_daily_loss:
                return False

            if self.consecutive_losses >= self.config.max_consecutive_losses:
                return False

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
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
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
            data["close"].iloc[-1]

            if self.position is None:
                return self._generate_entry_signal(data, volatility, spread, liquidity)
            else:
                return self._generate_exit_signal(data, volatility, spread, liquidity)

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
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

            # Проверяем условия для входа
            if (
                abs(current_price - data["close"].iloc[-2]) / data["close"].iloc[-2]
                >= self.config.entry_threshold
            ):
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, volatility)

                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price * (1 - self.config.stop_loss)
                take_profit = current_price * (1 + self.config.take_profit)

                return Signal(
                    direction=(
                        "long" if current_price > data["close"].iloc[-2] else "short"
                    ),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=min(1.0, volatility / self.config.min_volatility),
                    timestamp=data.index[-1],
                    metadata={
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
                        "step": self.step,
                        "current_bet": self.current_bet,
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
                            "step": self.step,
                            "current_bet": self.current_bet,
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
                            "step": self.step,
                            "current_bet": self.current_bet,
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
                            "step": self.step,
                            "current_bet": self.current_bet,
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
                            "step": self.step,
                            "current_bet": self.current_bet,
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self.config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self.config.trailing_step)

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
                self.last_trade_time = data.index[-1]
            elif signal.direction == "close":
                # Обновляем статистику
                if signal.metadata.get("reason") == "stop_loss":
                    self.consecutive_losses += 1
                    self.daily_loss += self.current_bet
                    self.step += 1
                    if self.step < self.config.max_steps:
                        self.current_bet = min(
                            self.current_bet * self.config.multiplier,
                            self.config.max_bet,
                        )
                else:
                    self.consecutive_losses = 0
                    if self.config.reset_on_win:
                        self.current_bet = self.config.initial_bet
                        self.step = 0

                self.position = None
                self.stop_loss = None
                self.take_profit = None

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
            base_size = self.current_bet

            # Корректировка на волатильность
            volatility_factor = 1 / (1 + volatility)

            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(position_size, self.config.max_position_size)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
