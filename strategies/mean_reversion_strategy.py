from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, Signal, StrategyMetrics


@dataclass
class MeanReversionConfig:
    """Конфигурация стратегии возврата к среднему"""

    # Параметры возврата к среднему
    mean_period: int = 20  # Период для расчета среднего
    std_period: int = 20  # Период для расчета стандартного отклонения
    z_score_threshold: float = 2.0  # Порог Z-оценки
    min_reversion_periods: int = 3  # Минимальное количество периодов для подтверждения
    max_reversion_periods: int = 10  # Максимальное количество периодов для подтверждения

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
    reversion_strength_threshold: float = 0.7  # Порог силы возврата

    # Адаптивные параметры
    adaptive_mean: bool = True  # Адаптивное среднее
    adaptive_std: bool = True  # Адаптивное стандартное отклонение
    adaptive_z_score: bool = True  # Адаптивный Z-score
    adaptive_volatility: bool = True  # Адаптивная волатильность
    adaptive_position_sizing: bool = True  # Адаптивный размер позиции

    # Параметры для адаптации
    adaptation_window: int = 100  # Окно для адаптации
    adaptation_threshold: float = 0.1  # Порог для адаптации
    adaptation_speed: float = 0.1  # Скорость адаптации
    adaptation_method: str = "ewm"  # Метод адаптации (ewm, kalman, particle)

    # Параметры для фильтрации сигналов
    use_trend_filter: bool = True  # Использовать фильтр тренда
    use_volume_filter: bool = True  # Использовать фильтр объема
    use_volatility_filter: bool = True  # Использовать фильтр волатильности
    use_correlation_filter: bool = True  # Использовать фильтр корреляции

    # Параметры для фильтров
    trend_period: int = 50  # Период для определения тренда
    volume_ma_period: int = 20  # Период для скользящего среднего объема
    volatility_ma_period: int = 20  # Период для скользящего среднего волатильности
    correlation_period: int = 20  # Период для расчета корреляции

    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"


class MeanReversionStrategy(BaseStrategy):
    """Стратегия возврата к среднему (расширенная)"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.config = (
            MeanReversionConfig(**config)
            if config and not isinstance(config, MeanReversionConfig)
            else (config or MeanReversionConfig())
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
            f"{self.config.log_dir}/mean_reversion_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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

            # Анализ возврата к среднему
            reversion = self._analyze_reversion(data)

            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "volatility": volatility,
                "spread": spread,
                "liquidity": liquidity,
                "reversion": reversion,
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
            reversion = analysis["reversion"]

            # Проверяем базовые условия
            if not self._check_basic_conditions(data, volatility, spread, liquidity, reversion):
                return None

            # Генерируем сигнал
            signal = self._generate_trading_signal(data, volatility, spread, liquidity, reversion)
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
        reversion: Dict[str, Any],
    ) -> bool:
        """
        Проверка базовых условий для торговли.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему

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

            # Проверка силы возврата
            if reversion["strength"] < self.config.reversion_strength_threshold:
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

    def _analyze_reversion(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ возврата к среднему.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с показателями возврата к среднему
        """
        try:
            # Расчет скользящего среднего
            mean = data["close"].rolling(window=self.config.mean_period).mean()

            # Расчет стандартного отклонения
            std = data["close"].rolling(window=self.config.std_period).std()

            # Расчет Z-оценки
            z_score = (data["close"] - mean) / std

            # Расчет силы возврата
            reversion_strength = abs(z_score.iloc[-1]) / z_score.std()

            # Расчет направления возврата
            reversion_direction = "up" if z_score.iloc[-1] < 0 else "down"

            # Расчет скорости возврата
            reversion_speed = abs(z_score.diff().iloc[-1])

            # Расчет количества периодов отклонения
            deviation_periods = self._calculate_deviation_periods(z_score)

            return {
                "z_score": z_score.iloc[-1],
                "mean": mean.iloc[-1],
                "std": std.iloc[-1],
                "strength": reversion_strength,
                "direction": reversion_direction,
                "speed": reversion_speed,
                "deviation_periods": deviation_periods,
            }

        except Exception as e:
            logger.error(f"Error analyzing reversion: {str(e)}")
            return {}

    def _calculate_deviation_periods(self, z_score: pd.Series) -> int:
        """
        Расчет количества периодов отклонения.

        Args:
            z_score: Серия Z-оценок

        Returns:
            int: Количество периодов отклонения
        """
        try:
            current_z = z_score.iloc[-1]
            periods = 0

            for z in reversed(z_score[:-1]):
                if (current_z > 0 and z > 0) or (current_z < 0 and z < 0):
                    periods += 1
                else:
                    break

            return periods

        except Exception as e:
            logger.error(f"Error calculating deviation periods: {str(e)}")
            return 0

    def _calculate_adaptive_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет адаптивных параметров.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с адаптивными параметрами
        """
        try:
            window = self.config.adaptation_window
            returns = data["close"].pct_change().dropna()

            # Адаптивное среднее
            if self.config.adaptive_mean:
                mean = returns.ewm(span=window, adjust=False).mean()
            else:
                mean = returns.rolling(window=self.config.mean_period).mean()

            # Адаптивное стандартное отклонение
            if self.config.adaptive_std:
                std = returns.ewm(span=window, adjust=False).std()
            else:
                std = returns.rolling(window=self.config.std_period).std()

            # Адаптивный Z-score
            if self.config.adaptive_z_score:
                z_score = (returns - mean) / std
                z_score = z_score.ewm(span=window, adjust=False).mean()
            else:
                z_score = (returns - mean) / std

            # Адаптивная волатильность
            if self.config.adaptive_volatility:
                volatility = returns.ewm(span=window, adjust=False).std()
            else:
                volatility = returns.rolling(window=self.config.std_period).std()

            return {"mean": mean, "std": std, "z_score": z_score, "volatility": volatility}

        except Exception as e:
            logger.error(f"Error calculating adaptive parameters: {str(e)}")
            return {}

    def _apply_filters(self, data: pd.DataFrame, analysis: Dict[str, Any]) -> bool:
        """
        Применение фильтров к сигналу.

        Args:
            data: DataFrame с OHLCV данными
            analysis: Результаты анализа

        Returns:
            bool: Результат фильтрации
        """
        try:
            # Фильтр тренда
            if self.config.use_trend_filter:
                trend = self._calculate_trend(data)
                if trend["strength"] > self.config.trend_threshold:
                    return False

            # Фильтр объема
            if self.config.use_volume_filter:
                volume_ma = data["volume"].rolling(window=self.config.volume_ma_period).mean()
                if data["volume"].iloc[-1] < volume_ma.iloc[-1] * 0.8:
                    return False

            # Фильтр волатильности
            if self.config.use_volatility_filter:
                volatility_ma = (
                    analysis["volatility"].rolling(window=self.config.volatility_ma_period).mean()
                )
                if analysis["volatility"] < volatility_ma.iloc[-1] * 0.5:
                    return False

            # Фильтр корреляции
            if self.config.use_correlation_filter:
                correlation = self._calculate_correlation(data)
                if abs(correlation) > 0.7:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error applying filters: {str(e)}")
            return False

    def _calculate_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет тренда.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с параметрами тренда
        """
        try:
            # Расчет скользящего среднего
            ma = data["close"].rolling(window=self.config.trend_period).mean()

            # Расчет наклона тренда
            slope = (ma - ma.shift(1)) / ma.shift(1)

            # Расчет силы тренда
            trend_strength = abs(slope).mean()

            # Определение направления тренда
            trend_direction = "up" if slope.iloc[-1] > 0 else "down"

            return {
                "direction": trend_direction,
                "strength": trend_strength,
                "slope": slope.iloc[-1],
            }

        except Exception as e:
            logger.error(f"Error calculating trend: {str(e)}")
            return {"direction": "unknown", "strength": 0.0, "slope": 0.0}

    def _calculate_correlation(self, data: pd.DataFrame) -> float:
        """
        Расчет корреляции.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Коэффициент корреляции
        """
        try:
            # Расчет корреляции между ценой и объемом
            returns = data["close"].pct_change()
            volume_change = data["volume"].pct_change()

            correlation = returns.corr(volume_change)

            return correlation

        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return 0.0

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        reversion: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала с учетом фильтров.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            # Получаем адаптивные параметры
            adaptive_params = self._calculate_adaptive_parameters(data)

            # Применяем фильтры
            if not self._apply_filters(data, {"volatility": volatility}):
                return None

            # Генерируем сигнал входа
            if not self.position:
                signal = self._generate_entry_signal(data, volatility, spread, liquidity, reversion)
                if signal:
                    # Корректируем сигнал с учетом адаптивных параметров
                    if self.config.adaptive_position_sizing:
                        signal.volume = self._calculate_adaptive_position_size(
                            signal, adaptive_params
                        )
                    return signal

            # Генерируем сигнал выхода
            else:
                signal = self._generate_exit_signal(data, volatility, spread, liquidity, reversion)
                if signal:
                    return signal

            return None

        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        volatility: float,
        spread: float,
        liquidity: Dict[str, float],
        reversion: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему

        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            current_price = data["close"].iloc[-1]

            # Проверяем условия для длинной позиции
            if (
                reversion["z_score"] < -self.config.z_score_threshold
                and reversion["direction"] == "up"
                and reversion["deviation_periods"] >= self.config.min_reversion_periods
                and reversion["deviation_periods"] <= self.config.max_reversion_periods
            ):

                # Проверяем объем
                if data["volume"].iloc[-1] > data["volume"].rolling(window=20).mean().iloc[-1]:
                    # Рассчитываем размер позиции
                    volume = self._calculate_position_size(current_price, volatility)

                    # Устанавливаем стоп-лосс и тейк-профит
                    stop_loss = current_price * (1 - self.config.stop_loss)
                    take_profit = reversion["mean"]  # Цель - возврат к среднему

                    return Signal(
                        direction="long",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=min(1.0, reversion["strength"]),
                        timestamp=datetime.now(),
                        metadata={
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "reversion": reversion,
                        },
                    )

            # Проверяем условия для короткой позиции
            elif (
                reversion["z_score"] > self.config.z_score_threshold
                and reversion["direction"] == "down"
                and reversion["deviation_periods"] >= self.config.min_reversion_periods
                and reversion["deviation_periods"] <= self.config.max_reversion_periods
            ):

                # Проверяем объем
                if data["volume"].iloc[-1] > data["volume"].rolling(window=20).mean().iloc[-1]:
                    # Рассчитываем размер позиции
                    volume = self._calculate_position_size(current_price, volatility)

                    # Устанавливаем стоп-лосс и тейк-профит
                    stop_loss = current_price * (1 + self.config.stop_loss)
                    take_profit = reversion["mean"]  # Цель - возврат к среднему

                    return Signal(
                        direction="short",
                        entry_price=current_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        volume=volume,
                        confidence=min(1.0, reversion["strength"]),
                        timestamp=datetime.now(),
                        metadata={
                            "volatility": volatility,
                            "spread": spread,
                            "liquidity": liquidity,
                            "reversion": reversion,
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
        reversion: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.

        Args:
            data: DataFrame с OHLCV данными
            volatility: Текущая волатильность
            spread: Текущий спред
            liquidity: Показатели ликвидности
            reversion: Показатели возврата к среднему

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
                            "reversion": reversion,
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
                            "reversion": reversion,
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
                            "reversion": reversion,
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
                            "reversion": reversion,
                        },
                    )

            # Проверяем трейлинг-стоп
            if self.config.trailing_stop:
                if self.position == "long" and current_price > self.take_profit:
                    self.take_profit = current_price * (1 - self.config.trailing_step)
                elif self.position == "short" and current_price < self.take_profit:
                    self.take_profit = current_price * (1 + self.config.trailing_step)

            # Проверяем ослабление возврата
            if self._check_reversion_weakening(reversion):
                return Signal(
                    direction="close",
                    entry_price=current_price,
                    timestamp=data.index[-1],
                    confidence=1.0,
                    metadata={
                        "reason": "reversion_weakening",
                        "volatility": volatility,
                        "spread": spread,
                        "liquidity": liquidity,
                        "reversion": reversion,
                    },
                )

            return None

        except Exception as e:
            logger.error(f"Error generating exit signal: {str(e)}")
            return None

    def _check_reversion_weakening(self, reversion: Dict[str, Any]) -> bool:
        """
        Проверка ослабления возврата.

        Args:
            reversion: Показатели возврата к среднему

        Returns:
            bool: Результат проверки
        """
        try:
            if self.position == "long":
                # Проверяем ослабление восходящего возврата
                if reversion["direction"] == "down" or reversion["speed"] < 0:
                    return True
            else:
                # Проверяем ослабление нисходящего возврата
                if reversion["direction"] == "up" or reversion["speed"] < 0:
                    return True

            return False

        except Exception as e:
            logger.error(f"Error checking reversion weakening: {str(e)}")
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
            position_size = min(position_size, self.config.max_position_size - self.total_position)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _calculate_adaptive_position_size(
        self, signal: Signal, adaptive_params: Dict[str, Any]
    ) -> float:
        """
        Расчет адаптивного размера позиции.

        Args:
            signal: Торговый сигнал
            adaptive_params: Адаптивные параметры

        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            position_size = self._calculate_position_size(
                signal.entry_price, adaptive_params["volatility"].iloc[-1]
            )

            # Корректировка на Z-score
            z_score = adaptive_params["z_score"].iloc[-1]
            position_size *= 1 - abs(z_score) / 3.0

            # Корректировка на волатильность
            volatility = adaptive_params["volatility"].iloc[-1]
            position_size *= 1 - volatility

            # Ограничение размера позиции
            position_size = min(position_size, self.config.max_position_size)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating adaptive position size: {str(e)}")
            return 0.0
