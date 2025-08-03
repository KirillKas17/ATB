from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class GridConfig:
    """Конфигурация сеточной стратегии"""

    # Параметры сетки
    grid_levels: int = 10  # Количество уровней сетки
    grid_spacing: float = 0.01  # Расстояние между уровнями
    grid_volume: float = 0.1  # Объем на уровень
    grid_direction: str = "both"  # Направление сетки (up/down/both)
    grid_shift: float = 0.0  # Смещение сетки относительно текущей цены
    # Параметры управления рисками
    max_position_size: float = 1.0  # Максимальный размер позиции
    max_grid_levels: int = 20  # Максимальное количество уровней
    min_profit_threshold: float = 0.001  # Минимальный порог прибыли
    max_loss_threshold: float = 0.002  # Максимальный порог убытка
    risk_per_trade: float = 0.02  # Риск на сделку
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


class GridStrategy(BaseStrategy):
    """Сеточная стратегия (расширенная)"""

    def __init__(self, config: Optional[Union[Dict[str, Any], GridConfig]] = None):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        if isinstance(config, GridConfig):
            super().__init__(asdict(config))
            self._config = config
        elif isinstance(config, dict):
            super().__init__(config)
            self._config = GridConfig(**config)
        else:
            super().__init__(None)
            self._config = GridConfig()
        self.position: Optional[str] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.grid_levels: Dict[float, Dict[str, Any]] = {}
        self.active_orders: Dict[str, Any] = {}
        self.total_position: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self._config.log_dir}/grid_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
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
            # Проверка волатильности
            if volatility < self._config.min_volatility:
                return False
            # Проверка спреда
            if spread > self._config.max_spread:
                return False
            # Проверка ликвидности
            if liquidity["volume"] < self._config.min_volume:
                return False
            if liquidity["depth"] < self._config.liquidity_threshold:
                return False
            # Проверка размера позиции
            if self.total_position >= self._config.max_position_size:
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
            return float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0  # type: ignore
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
            return float(spread) if spread is not None and not pd.isna(spread) else 0.0  # type: ignore
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
            return {
                "volume": float(volume) if volume is not None and not pd.isna(volume) else 0.0,  # type: ignore
                "depth": float(depth) if depth is not None and not pd.isna(depth) else 0.0,  # type: ignore
                "volume_spread": float(volume_spread) if volume_spread is not None and not pd.isna(volume_spread) else 0.0  # type: ignore
            }
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
            current_price = data["close"].iloc[-1]
            # Обновляем уровни сетки
            self._update_grid_levels(current_price)
            # Проверяем срабатывание уровней
            for level, order in self.grid_levels.items():
                if not order["active"]:
                    if (
                        self._config.grid_direction in ["up", "both"]
                        and current_price <= level
                    ) or (
                        self._config.grid_direction in ["down", "both"]
                        and current_price >= level
                    ):
                        return self._generate_entry_signal(
                            current_price, level, volatility, spread, liquidity
                        )
            # Проверяем выход из позиции
            if self.position:
                return self._generate_exit_signal(data, volatility, spread, liquidity)
            return None
        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _update_grid_levels(self, current_price: float) -> None:
        """
        Обновление уровней сетки.
        Args:
            current_price: Текущая цена
        """
        try:
            # Очищаем неактивные уровни
            self.grid_levels = {
                level: order
                for level, order in self.grid_levels.items()
                if order["active"]
            }
            # Добавляем новые уровни
            if len(self.grid_levels) < self._config.max_grid_levels:
                base_price = current_price * (1 + self._config.grid_shift)
                for i in range(self._config.grid_levels):
                    level = base_price * (1 + i * self._config.grid_spacing)
                    if level not in self.grid_levels:
                        self.grid_levels[level] = {
                            "active": True,
                            "volume": self._config.grid_volume,
                            "direction": (
                                "buy"
                                if self._config.grid_direction in ["up", "both"]
                                else "sell"
                            ),
                        }
        except Exception as e:
            logger.error(f"Error updating grid levels: {str(e)}")

    def _generate_entry_signal(
        self,
        current_price: float,
        level: float,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.
        Args:
            current_price: Текущая цена
            level: Уровень сетки
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            # Рассчитываем размер позиции
            volume = self._calculate_position_size(current_price, volatility)
            # Устанавливаем стоп-лосс и тейк-профит
            stop_loss = current_price * (1 - self._config.stop_loss)
            take_profit = current_price * (1 + self._config.take_profit)
            # Деактивируем уровень
            self.grid_levels[level]["active"] = False
            return Signal(
                direction=(
                    "long" if self.grid_levels[level]["direction"] == "buy" else "short"
                ),
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                confidence=min(1.0, volatility / self._config.min_volatility),
                timestamp=datetime.now(),
                metadata={
                    "level": level,
                    "volatility": volatility,
                    "spread": spread,
                    "liquidity": liquidity,
                },
            )
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
                if self.stop_loss is not None and current_price <= self.stop_loss:
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
                elif self.take_profit is not None and current_price >= self.take_profit:
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
                if self.stop_loss is not None and current_price >= self.stop_loss:
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
                elif self.take_profit is not None and current_price <= self.take_profit:
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
            if self._config.trailing_stop and self.take_profit is not None:
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
                self.total_position += signal.volume
                self.last_trade_time = data.index[-1]
            elif signal.direction == "close":
                self.position = None
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
            base_size = self._config.grid_volume
            # Корректировка на волатильность
            volatility_factor = 1 / (1 + volatility)
            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            max_available = self._config.max_position_size - self.total_position
            position_size = min(position_size, max_available) if max_available is not None else position_size
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
