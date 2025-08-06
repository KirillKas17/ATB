from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from loguru import logger
from shared.signal_validator import validate_trading_signal
from shared.decimal_utils import TradingDecimal, to_trading_decimal

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

    def __init__(self, config: Optional[Union[Dict[str, Any], BreakoutConfig]] = None):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        # Преобразуем конфигурацию в словарь для базового класса
        if isinstance(config, BreakoutConfig):
            config_dict = {
                "breakout_period": config.breakout_period,
                "breakout_threshold": config.breakout_threshold,
                "min_volume_multiplier": config.min_volume_multiplier,
                "confirmation_periods": config.confirmation_periods,
                "false_breakout_threshold": config.false_breakout_threshold,
                "entry_threshold": config.entry_threshold,
                "min_volume": config.min_volume,
                "min_volatility": config.min_volatility,
                "max_spread": config.max_spread,
                "take_profit": config.take_profit,
                "stop_loss": config.stop_loss,
                "trailing_stop": config.trailing_stop,
                "trailing_step": config.trailing_step,
                "max_position_size": config.max_position_size,
                "max_daily_trades": config.max_daily_trades,
                "max_daily_loss": config.max_daily_loss,
                "risk_per_trade": config.risk_per_trade,
                "price_deviation_threshold": config.price_deviation_threshold,
                "volume_deviation_threshold": config.volume_deviation_threshold,
                "liquidity_threshold": config.liquidity_threshold,
                "trend_strength_threshold": config.trend_strength_threshold,
                "symbols": config.symbols,
                "timeframes": config.timeframes,
                "log_dir": config.log_dir,
            }
        else:
            config_dict = config or {}
        
        super().__init__(config_dict)
        
        # Инициализируем конфигурацию
        if isinstance(config, BreakoutConfig):
            self._config = config
        elif config:
            self._config = BreakoutConfig(**config)
        else:
            self._config = BreakoutConfig()
        
        # Состояние позиции
        self.position: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.breakout_level: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.total_position: float = 0.0
        self.daily_trades: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_time: Optional[datetime] = None
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self._config.log_dir}/breakout_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
            # Проверка силы тренда
            if trend["strength"] < self._config.trend_strength_threshold:
                return False
            # Проверка размера позиции
            if self.total_position >= self._config.max_position_size:
                return False
            # Проверка количества сделок
            if self.daily_trades >= self._config.max_daily_trades:
                return False
            # Проверка дневного убытка
            if self.daily_pnl <= -self._config.max_daily_loss:
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
            returns = data["close"].pct_change().dropna()
            return returns.std()
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
            spread = (data["high"] - data["low"]) / data["close"]
            return spread.mean()
        except Exception as e:
            logger.error(f"Error calculating spread: {str(e)}")
            return 0.0

    def _analyze_liquidity(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Анализ ликвидности.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict[str, float]: Показатели ликвидности
        """
        try:
            volume = data["volume"].mean()
            depth = volume * data["close"].mean()
            return {"volume": volume, "depth": depth}
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"volume": 0.0, "depth": 0.0}

    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ тренда.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict[str, Any]: Показатели тренда
        """
        try:
            # Простая скользящая средняя
            sma = data["close"].rolling(window=20).mean()
            # Экспоненциальная скользящая средняя
            ema = data["close"].ewm(span=20).mean()
            # Сила тренда с защитой от деления на ноль
            sma_last = sma.iloc[-1]
            if sma_last != 0:
                trend_strength = abs(data["close"].iloc[-1] - sma_last) / sma_last
            else:
                trend_strength = 0.0
            # Направление тренда
            if data["close"].iloc[-1] > sma.iloc[-1]:
                direction = "up"
            elif data["close"].iloc[-1] < sma.iloc[-1]:
                direction = "down"
            else:
                direction = "sideways"
            return {
                "sma": sma.iloc[-1],
                "ema": ema.iloc[-1],
                "strength": trend_strength,
                "direction": direction,
            }
        except Exception as e:
            logger.error(f"Error analyzing trend: {str(e)}")
            return {"sma": 0.0, "ema": 0.0, "strength": 0.0, "direction": "sideways"}

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
            # Если есть открытая позиция, проверяем условия выхода
            if self.position:
                return self._generate_exit_signal(
                    data, volatility, spread, liquidity, trend
                )
            # Иначе генерируем сигнал на вход
            return self._generate_entry_signal(
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
            high = data["high"].rolling(window=self._config.breakout_period).max()
            low = data["low"].rolling(window=self._config.breakout_period).min()
            # Проверяем пробой вверх
            if current_price > high.iloc[-2] * (1 + self._config.breakout_threshold):
                if self._check_breakout_confirmation(data, "up"):
                    position_size = self._calculate_position_size(current_price, volatility)
                    # Используем Decimal для точных расчетов
                    current_price_decimal = to_trading_decimal(current_price)
                    stop_loss_decimal = TradingDecimal.calculate_stop_loss(
                        current_price_decimal, "long", to_trading_decimal(self._config.stop_loss * 100)
                    )
                    take_profit_decimal = TradingDecimal.calculate_take_profit(
                        current_price_decimal, "long", to_trading_decimal(self._config.take_profit * 100)
                    )
                    stop_loss = float(stop_loss_decimal)
                    take_profit = float(take_profit_decimal)
                    signal = Signal(
                        direction="long",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=position_size,
                        confidence=0.8,
                        timestamp=data.index[-1],
                        metadata={
                            "reason": "breakout_up",
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "trend": trend,
                        },
                    )
                    # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ сигнала
                    if not validate_trading_signal(signal):
                        return None
                    return signal
            # Проверяем пробой вниз
            elif current_price < low.iloc[-2] * (1 - self._config.breakout_threshold):
                if self._check_breakout_confirmation(data, "down"):
                    position_size = self._calculate_position_size(current_price, volatility)
                    # Используем Decimal для точных расчетов
                    stop_loss_decimal = TradingDecimal.calculate_stop_loss(
                        current_price_decimal, "short", to_trading_decimal(self._config.stop_loss * 100)
                    )
                    take_profit_decimal = TradingDecimal.calculate_take_profit(
                        current_price_decimal, "short", to_trading_decimal(self._config.take_profit * 100)
                    )
                    stop_loss = float(stop_loss_decimal)
                    take_profit = float(take_profit_decimal)
                    signal = Signal(
                        direction="short",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=position_size,
                        confidence=0.8,
                        timestamp=data.index[-1],
                        metadata={
                            "reason": "breakout_down",
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "trend": trend,
                        },
                    )
                    # КРИТИЧЕСКАЯ ВАЛИДАЦИЯ сигнала
                    if not validate_trading_signal(signal):
                        return None
                    return signal
            return None
        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _check_breakout_confirmation(self, data: pd.DataFrame, direction: str) -> bool:
        """
        Проверка подтверждения пробоя.
        Args:
            data: DataFrame с OHLCV данными
            direction: Направление пробоя
        Returns:
            bool: Результат проверки
        """
        try:
            # Проверяем объем
            current_volume = data["volume"].iloc[-1]
            avg_volume = data["volume"].rolling(window=20).mean().iloc[-1]
            if current_volume < avg_volume * self._config.min_volume_multiplier:
                return False
            # Проверяем подтверждение в течение нескольких периодов
            for i in range(1, self._config.confirmation_periods + 1):
                if i >= len(data):
                    return False
                if direction == "up":
                    if data["close"].iloc[-i] <= data["close"].iloc[-i-1]:
                        return False
                else:
                    if data["close"].iloc[-i] >= data["close"].iloc[-i-1]:
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
            if self.position == "long" and self.stop_loss is not None and self.take_profit is not None:
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
            elif self.position == "short" and self.stop_loss is not None and self.take_profit is not None:
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
            if self._config.trailing_stop and self.take_profit is not None:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self._config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self._config.trailing_step)
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
                    1 - self._config.false_breakout_threshold
                ):
                    return True
            else:
                # Проверяем возврат цены выше уровня пробоя
                if current_price > self.breakout_level * (
                    1 + self._config.false_breakout_threshold
                ):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking false breakout: {str(e)}")
            return False

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
                self.entry_price = signal.entry_price
                self.stop_loss = signal.stop_loss
                self.take_profit = signal.take_profit
                self.total_position += signal.volume or 0.0
                self.daily_trades += 1
                self.last_trade_time = data.index[-1]
            elif signal.direction == "close":
                # Обновляем дневной P&L
                if self.position == "long" and self.entry_price is not None:
                    self.daily_pnl += (
                        data["close"].iloc[-1] - self.entry_price
                    ) * self.total_position
                elif self.position == "short" and self.entry_price is not None:
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
            base_size = self._config.risk_per_trade
            # Корректировка на волатильность
            volatility_factor = 1 / (1 + volatility)
            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(
                position_size, self._config.max_position_size - self.total_position
            )
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0
