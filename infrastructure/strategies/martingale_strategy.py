from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
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

    def __init__(
        self, config: Optional[Union[Dict[str, Any], MartingaleConfig]] = None
    ):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        if isinstance(config, MartingaleConfig):
            super().__init__(asdict(config))
            self._config = config
        elif isinstance(config, dict):
            super().__init__(config)
            self._config = MartingaleConfig(**config)
        else:
            super().__init__(None)
            self._config = MartingaleConfig()
        self.position: Optional[str] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.current_bet: float = self._config.initial_bet
        self.step: int = 0
        self.consecutive_losses: int = 0
        self.daily_loss: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self._setup_logger()

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
            if self.daily_loss >= self._config.max_daily_loss:
                return False
            if self.consecutive_losses >= self._config.max_consecutive_losses:
                return False
            # Проверка волатильности
            volatility = float(volatility) if pd.notna(volatility) else 0.0
            if volatility < self._config.min_volatility:
                return False
            # Проверка спреда
            spread = float(spread) if pd.notna(spread) else 0.0
            if spread > self._config.max_spread:
                return False
            # Проверка ликвидности
            volume = liquidity.get("volume", 0.0)
            volume = float(volume) if pd.notna(volume) else 0.0
            if volume < self._config.min_volume:
                return False
            depth = liquidity.get("depth", 0.0)
            depth = float(depth) if pd.notna(depth) else 0.0
            if depth < self._config.liquidity_threshold:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        try:
            returns = data["close"].pct_change()
            volatility = returns.std()
            return float(volatility) if pd.notna(volatility) else 0.0
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.0

    def _calculate_spread(self, data: pd.DataFrame) -> float:
        try:
            spread = (data["ask"] - data["bid"]).iloc[-1]
            return float(spread) if pd.notna(spread) else 0.0
        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return 0.0

    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, float]:
        try:
            volume = data["volume"].iloc[-1]
            volume = float(volume) if pd.notna(volume) else 0.0
            depth = (data["ask_volume"] + data["bid_volume"]).iloc[-1]
            depth = float(depth) if pd.notna(depth) else 0.0
            volume_spread = abs(data["ask_volume"] - data["bid_volume"]).iloc[-1]
            volume_spread = float(volume_spread) if pd.notna(volume_spread) else 0.0
            return {"volume": volume, "depth": depth, "volume_spread": volume_spread}
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"volume": 0.0, "depth": 0.0, "volume_spread": 0.0}

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
            # Проверяем достаточное количество данных
            if len(data) < 2:
                return None
            current_price = data["close"].iloc[-1]
            if pd.isna(current_price):
                return None
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
            # Проверяем достаточное количество данных
            if len(data) < 2:
                return None
            current_price = data["close"].iloc[-1]
            previous_price = data["close"].iloc[-2]
            current_price = float(current_price) if pd.notna(current_price) else 0.0
            previous_price = float(previous_price) if pd.notna(previous_price) else 0.0
            if current_price <= 0 or previous_price <= 0:
                return None
            # Проверяем условия для входа
            price_change = abs(current_price - previous_price) / previous_price
            if price_change >= self._config.entry_threshold:
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, volatility)
                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price * (1 - self._config.stop_loss)
                take_profit = current_price * (1 + self._config.take_profit)
                return Signal(
                    direction=(
                        "long" if current_price > previous_price else "short"
                    ),
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=min(1.0, volatility / self._config.min_volatility),
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
            current_price = float(current_price) if pd.notna(current_price) else 0.0
            if current_price <= 0:
                return None
            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if self.stop_loss and current_price <= self.stop_loss:
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
                elif self.take_profit and current_price >= self.take_profit:
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
                if self.stop_loss and current_price >= self.stop_loss:
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
                elif self.take_profit and current_price <= self.take_profit:
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
            if self._config.trailing_stop and self.take_profit:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self._config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self._config.trailing_step)
            return None
        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _update_position_state(self, signal: Signal, data: pd.DataFrame) -> None:
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
                    if self.step < self._config.max_steps:
                        self.current_bet = min(
                            self.current_bet * self._config.multiplier,
                            self._config.max_bet,
                        )
                else:
                    self.consecutive_losses = 0
                    if self._config.reset_on_win:
                        self.current_bet = self._config.initial_bet
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
            position_size = min(position_size, self._config.max_position_size)
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
