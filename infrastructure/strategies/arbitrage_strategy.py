from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class ArbitrageConfig:
    """Конфигурация арбитражной стратегии"""

    # Параметры арбитража
    min_spread: float = 0.001  # Минимальный спред для входа
    max_spread: float = 0.05  # Максимальный спред для входа
    spread_window: int = 20  # Окно для расчета среднего спреда
    spread_std_multiplier: float = 2.0  # Множитель стандартного отклонения
    min_volume: float = 1000.0  # Минимальный объем
    max_slippage: float = 0.001  # Максимальное проскальзывание
    # Параметры управления рисками
    risk_per_trade: float = 0.02  # Риск на сделку
    max_position_size: float = 0.2  # Максимальный размер позиции
    max_daily_trades: int = 10  # Максимальное количество сделок в день
    max_open_positions: int = 3  # Максимальное количество открытых позиций
    min_profit_threshold: float = 0.001  # Минимальный порог прибыли
    max_loss_threshold: float = 0.002  # Максимальный порог убытка
    # Параметры исполнения
    execution_timeout: int = 5  # Таймаут исполнения в секундах
    retry_attempts: int = 3  # Количество попыток исполнения
    partial_fill: bool = True  # Разрешить частичное исполнение
    min_fill_ratio: float = 0.8  # Минимальное соотношение исполнения
    # Параметры мониторинга
    price_deviation_threshold: float = 0.002  # Порог отклонения цены
    volume_deviation_threshold: float = 0.5  # Порог отклонения объема
    liquidity_threshold: float = 10000.0  # Порог ликвидности
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1m"])
    log_dir: str = "logs"


class ArbitrageStrategy(BaseStrategy):
    """Арбитражная стратегия (расширенная)"""

    def __init__(self, config: Optional[Union[Dict[str, Any], ArbitrageConfig]] = None):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        # Преобразуем конфигурацию в словарь для базового класса
        config_dict = None
        if config is not None:
            if isinstance(config, ArbitrageConfig):
                config_dict = config.__dict__.copy()
            else:
                config_dict = config
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию для этого класса
        if isinstance(config, ArbitrageConfig):
            self._config = config
        elif config is not None:
            self._config = ArbitrageConfig(**config)
        else:
            self._config = ArbitrageConfig()
        self.position: Optional[str] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.daily_trades = 0
        self.open_positions = 0
        self.last_trade_time: Optional[datetime] = None
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self._config.log_dir}/arbitrage_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
            # Расчет спреда
            spread = self._calculate_spread(data)
            # Расчет статистик спреда
            spread_stats = self._calculate_spread_stats(spread)
            # Анализ ликвидности
            liquidity = self._analyze_liquidity(data)
            # Анализ исполнения
            execution = self._analyze_execution(data)
            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)
            return {
                "spread": spread,
                "spread_stats": spread_stats,
                "liquidity": liquidity,
                "execution": execution,
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
            spread = analysis["spread"]
            spread_stats = analysis["spread_stats"]
            liquidity = analysis["liquidity"]
            execution = analysis["execution"]
            # Проверяем базовые условия
            if not self._check_basic_conditions(
                data, spread, spread_stats, liquidity, execution
            ):
                return None
            # Генерируем сигнал
            signal = self._generate_trading_signal(
                data, spread, spread_stats, liquidity, execution
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
        spread: float,
        spread_stats: Dict[str, float],
        liquidity: Dict[str, float],
        execution: Dict[str, float],
    ) -> bool:
        """
        Проверка базовых условий для торговли.
        Args:
            data: DataFrame с OHLCV данными
            spread: Текущий спред
            spread_stats: Статистики спреда
            liquidity: Показатели ликвидности
            execution: Показатели исполнения
        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка лимитов
            if self.daily_trades >= self._config.max_daily_trades:
                return False
            if self.open_positions >= self._config.max_open_positions:
                return False
            # Проверка спреда
            if not (self._config.min_spread <= spread <= self._config.max_spread):
                return False
            # Проверка отклонения от среднего
            if (
                abs(spread - spread_stats["mean"])
                > spread_stats["std"] * self._config.spread_std_multiplier
            ):
                return False
            # Проверка ликвидности
            if liquidity["volume"] < self._config.min_volume:
                return False
            if liquidity["depth"] < self._config.liquidity_threshold:
                return False
            # Проверка исполнения
            if execution["slippage"] > self._config.max_slippage:
                return False
            if execution["fill_ratio"] < self._config.min_fill_ratio:
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _calculate_spread(self, data: pd.DataFrame) -> float:
        """
        Расчет спреда между bid и ask.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            float: Текущий спред
        """
        try:
            # Расчет спреда как разницы между ценами
            spread = (data["ask"] - data["bid"]).iloc[-1]
            return float(spread)
        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return 0.0

    def _calculate_spread_stats(self, spread: float) -> Dict[str, float]:
        """
        Расчет статистик спреда.
        Args:
            spread: Текущий спред
        Returns:
            Dict с статистиками
        """
        try:
            # Расчет скользящего среднего и стандартного отклонения
            spread_ma = (
                pd.Series(spread).rolling(window=self._config.spread_window).mean()
            )
            spread_std = (
                pd.Series(spread).rolling(window=self._config.spread_window).std()
            )
            return {
                "mean": spread_ma.iloc[-1],
                "std": spread_std.iloc[-1],
                "min": spread_ma.iloc[-1] - spread_std.iloc[-1] * self._config.spread_std_multiplier,
                "max": spread_ma.iloc[-1] + spread_std.iloc[-1] * self._config.spread_std_multiplier,
            }
        except Exception as e:
            logger.error(f"Error calculating spread stats: {str(e)}")
            return {}

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

    def _analyze_execution(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Анализ исполнения.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с показателями исполнения
        """
        try:
            # Расчет проскальзывания
            slippage = (
                abs(data["close"] - data["vwap"]).iloc[-1] / data["vwap"].iloc[-1]
            )
            # Расчет соотношения исполнения
            fill_ratio = data["filled_volume"].iloc[-1] / data["volume"].iloc[-1]
            # Расчет времени исполнения
            execution_time = (
                (data["execution_time"] - data["order_time"]).iloc[-1].total_seconds()
            )
            return {
                "slippage": slippage,
                "fill_ratio": fill_ratio,
                "execution_time": execution_time,
            }
        except Exception as e:
            logger.error(f"Error analyzing execution: {str(e)}")
            return {}

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        spread: float,
        spread_stats: Dict[str, float],
        liquidity: Dict[str, float],
        execution: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        Args:
            data: DataFrame с OHLCV данными
            spread: Текущий спред
            spread_stats: Статистики спреда
            liquidity: Показатели ликвидности
            execution: Показатели исполнения
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            data["close"].iloc[-1]
            if self.position is None:
                return self._generate_entry_signal(
                    data, spread, spread_stats, liquidity, execution
                )
            else:
                return self._generate_exit_signal(
                    data, spread, spread_stats, liquidity, execution
                )
        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        spread: float,
        spread_stats: Dict[str, float],
        liquidity: Dict[str, float],
        execution: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.
        Args:
            data: DataFrame с OHLCV данными
            spread: Текущий спред
            spread_stats: Статистики спреда
            liquidity: Показатели ликвидности
            execution: Показатели исполнения
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]
            # Проверяем условия для длинной позиции
            if spread > spread_stats["mean"] + spread_stats["std"]:
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, spread)
                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price * (1 - self._config.max_loss_threshold)
                take_profit = current_price * (1 + self._config.min_profit_threshold)
                return Signal(
                    direction="long",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=min(1.0, spread / spread_stats["max"]),
                    timestamp=data.index[-1],
                    metadata={
                        "spread": spread,
                        "spread_stats": spread_stats,
                        "liquidity": liquidity,
                        "execution": execution,
                    },
                )
            # Проверяем условия для короткой позиции
            elif spread < spread_stats["mean"] - spread_stats["std"]:
                # Рассчитываем размер позиции
                volume = self._calculate_position_size(current_price, spread)
                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price * (1 + self._config.max_loss_threshold)
                take_profit = current_price * (1 - self._config.min_profit_threshold)
                return Signal(
                    direction="short",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=min(
                        1.0, abs(spread - spread_stats["min"]) / spread_stats["std"]
                    ),
                    timestamp=data.index[-1],
                    metadata={
                        "spread": spread,
                        "spread_stats": spread_stats,
                        "liquidity": liquidity,
                        "execution": execution,
                    },
                )
            return None
        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        spread: float,
        spread_stats: Dict[str, float],
        liquidity: Dict[str, float],
        execution: Dict[str, float],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.
        Args:
            data: DataFrame с OHLCV данными
            spread: Текущий спред
            spread_stats: Статистики спреда
            liquidity: Показатели ликвидности
            execution: Показатели исполнения
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
                            "spread": spread,
                            "spread_stats": spread_stats,
                            "liquidity": liquidity,
                            "execution": execution,
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
                            "spread": spread,
                            "spread_stats": spread_stats,
                            "liquidity": liquidity,
                            "execution": execution,
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
                            "spread": spread,
                            "spread_stats": spread_stats,
                            "liquidity": liquidity,
                            "execution": execution,
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
                            "spread": spread,
                            "spread_stats": spread_stats,
                            "liquidity": liquidity,
                            "execution": execution,
                        },
                    )
            # Проверяем возврат спреда к среднему
            if (self.position == "long" and spread <= spread_stats["mean"]) or (
                self.position == "short" and spread >= spread_stats["mean"]
            ):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=0.8,
                    metadata={
                        "reason": "spread_mean_reversion",
                        "spread": spread,
                        "spread_stats": spread_stats,
                        "liquidity": liquidity,
                        "execution": execution,
                    },
                )
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
                self.daily_trades += 1
                self.open_positions += 1
                self.last_trade_time = data.index[-1]
            elif signal.direction == "close":
                self.position = None
                self.stop_loss = None
                self.take_profit = None
                self.open_positions -= 1
        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def _calculate_position_size(self, price: float, spread: float) -> float:
        """
        Расчет размера позиции.
        Args:
            price: Текущая цена
            spread: Текущий спред
        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            base_size = self._config.risk_per_trade
            # Корректировка на спред
            spread_factor = 1 / (1 + spread / price)
            # Корректировка на максимальный размер
            position_size = base_size * spread_factor
            position_size = min(position_size, self._config.max_position_size)
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
