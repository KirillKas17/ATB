import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import talib
from loguru import logger

from .base_strategy import BaseStrategy, Signal

warnings.filterwarnings("ignore")


@dataclass
class HedgingConfig:
    """Конфигурация стратегии хеджирования"""

    # Основные параметры
    hedge_ratio: float = 1.0  # Соотношение хеджирования
    correlation_threshold: float = 0.7  # Минимальная корреляция для хеджирования
    volatility_threshold: float = 0.02  # Порог волатильности
    min_spread: float = 0.001  # Минимальный спред
    max_spread: float = 0.05  # Максимальный спред

    # Параметры управления рисками
    max_position_size: float = 0.1  # Максимальный размер позиции
    stop_loss: float = 0.02  # Стоп-лосс
    take_profit: float = 0.04  # Тейк-профит
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_stop_activation: float = 0.01  # Активация трейлинг-стопа
    trailing_stop_distance: float = 0.005  # Дистанция трейлинг-стопа

    # Параметры анализа
    lookback_period: int = 20  # Период для анализа
    min_trades: int = 10  # Минимальное количество сделок
    confidence_threshold: float = 0.7  # Порог уверенности

    # Параметры оптимизации
    optimization_period: int = 100  # Период для оптимизации
    optimization_metric: str = "sharpe_ratio"  # Метрика оптимизации
    reoptimization_interval: int = 24  # Интервал переоптимизации (часы)

    # Дополнительные параметры
    use_volume: bool = True  # Учитывать объем
    use_volatility: bool = True  # Учитывать волатильность
    use_correlation: bool = True  # Учитывать корреляцию
    use_regime: bool = True  # Учитывать рыночный режим
    use_sentiment: bool = False  # Учитывать настроения рынка

    # Параметры для режимов рынка
    regime_threshold: float = 0.6  # Порог для определения режима
    trend_threshold: float = 0.7  # Порог для определения тренда
    volatility_regime_threshold: float = 0.02  # Порог волатильности для режима

    # Параметры для настроений рынка
    sentiment_threshold: float = 0.6  # Порог настроений
    sentiment_lookback: int = 5  # Период для анализа настроений

    # Параметры для управления капиталом
    risk_per_trade: float = 0.02  # Риск на сделку
    max_drawdown: float = 0.1  # Максимальная просадка
    position_sizing: str = "kelly"  # Метод расчета размера позиции

    # Параметры для мониторинга
    monitor_interval: int = 1  # Интервал мониторинга (минуты)
    alert_threshold: float = 0.05  # Порог для алертов

    # Параметры для логирования
    log_level: str = "INFO"  # Уровень логирования
    log_interval: int = 1  # Интервал логирования (часы)

    # Параметры для тестирования
    test_mode: bool = False  # Режим тестирования
    test_balance: float = 10000.0  # Начальный баланс для тестирования

    # Дополнительные настройки
    additional: Dict[str, Any] = field(default_factory=dict)


class HedgingStrategy(BaseStrategy):
    """Стратегия хеджирования"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация стратегии хеджирования.

        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        self.hedge_config = HedgingConfig(**self.config)
        self._initialize_indicators()
        self._setup_optimization()
        self._setup_monitoring()

    def _initialize_indicators(self):
        """Инициализация индикаторов"""
        self.indicators = {
            "sma": lambda x: talib.SMA(x, timeperiod=self.hedge_config.lookback_period),
            "ema": lambda x: talib.EMA(x, timeperiod=self.hedge_config.lookback_period),
            "rsi": lambda x: talib.RSI(x, timeperiod=self.hedge_config.lookback_period),
            "macd": lambda x: talib.MACD(
                x, fastperiod=12, slowperiod=26, signalperiod=9
            ),
            "bollinger": lambda x: talib.BBANDS(
                x, timeperiod=self.hedge_config.lookback_period, nbdevup=2, nbdevdn=2
            ),
            "atr": lambda x: talib.ATR(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.hedge_config.lookback_period,
            ),
            "adx": lambda x: talib.ADX(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.hedge_config.lookback_period,
            ),
        }

    def _setup_optimization(self):
        """Настройка оптимизации"""
        self.last_optimization = datetime.now()
        self.optimization_results = {}

    def _setup_monitoring(self):
        """Настройка мониторинга"""
        self.monitoring_data = {
            "positions": [],
            "trades": [],
            "metrics": {},
            "alerts": [],
        }

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ рыночных данных.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с результатами анализа
        """
        try:
            # Валидация данных
            is_valid, error = self.validate_data(data)
            if not is_valid:
                logger.error(f"Invalid data: {error}")
                return {}

            # Расчет индикаторов
            indicators = {}
            for name, func in self.indicators.items():
                try:
                    if name in ["macd", "bollinger"]:
                        indicators[name] = func(data["close"])
                    elif name in ["atr", "adx"]:
                        indicators[name] = func(data)
                    else:
                        indicators[name] = func(data["close"])
                except Exception as e:
                    logger.error(f"Error calculating {name}: {str(e)}")

            # Анализ корреляции
            correlation = self._analyze_correlation(data)

            # Анализ волатильности
            volatility = self._analyze_volatility(data)

            # Анализ спреда
            spread = self._analyze_spread(data)

            # Определение режима рынка
            regime = self._determine_market_regime(data, indicators)

            # Анализ настроений
            sentiment = (
                self._analyze_sentiment(data)
                if self.hedge_config.use_sentiment
                else None
            )

            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)

            return {
                "indicators": indicators,
                "correlation": correlation,
                "volatility": volatility,
                "spread": spread,
                "regime": regime,
                "sentiment": sentiment,
                "risk_metrics": risk_metrics,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
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
            # Анализ данных
            analysis = self.analyze(data)
            if not analysis:
                return None

            # Проверка условий для хеджирования
            if not self._check_hedging_conditions(analysis):
                return None

            # Расчет направления сигнала
            direction = self._calculate_signal_direction(data, analysis)
            if not direction:
                return None

            # Расчет уровней входа
            entry_price = data["close"].iloc[-1]
            stop_loss = self._calculate_stop_loss(entry_price, direction, analysis)
            take_profit = self._calculate_take_profit(entry_price, direction, analysis)

            # Расчет объема
            volume = self._calculate_position_size(entry_price, stop_loss, analysis)

            # Расчет уверенности
            confidence = self._calculate_confidence(analysis)

            # Создание сигнала
            signal = Signal(
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=volume,
                confidence=confidence,
                metadata={"analysis": analysis, "timestamp": datetime.now()},
            )

            # Проверка сигнала
            if not self._validate_signal(signal):
                return None

            return signal

        except Exception as e:
            logger.error(f"Error in generate_signal: {str(e)}")
            return None

    def _check_hedging_conditions(self, analysis: Dict[str, Any]) -> bool:
        """
        Проверка условий для хеджирования.

        Args:
            analysis: Результаты анализа

        Returns:
            bool: True если условия выполнены
        """
        try:
            # Проверка корреляции
            if self.hedge_config.use_correlation:
                if analysis["correlation"] < self.hedge_config.correlation_threshold:
                    return False

            # Проверка волатильности
            if self.hedge_config.use_volatility:
                if analysis["volatility"] < self.hedge_config.volatility_threshold:
                    return False

            # Проверка спреда
            spread = analysis["spread"]
            if (
                spread < self.hedge_config.min_spread
                or spread > self.hedge_config.max_spread
            ):
                return False

            # Проверка режима рынка
            if self.hedge_config.use_regime:
                if analysis["regime"] == "unknown":
                    return False

            # Проверка настроений
            if self.hedge_config.use_sentiment and analysis["sentiment"]:
                if analysis["sentiment"] < self.hedge_config.sentiment_threshold:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error in _check_hedging_conditions: {str(e)}")
            return False

    def _calculate_signal_direction(
        self, data: pd.DataFrame, analysis: Dict[str, Any]
    ) -> Optional[str]:
        """
        Расчет направления сигнала.

        Args:
            data: DataFrame с OHLCV данными
            analysis: Результаты анализа

        Returns:
            Optional[str]: 'long' или 'short' или None
        """
        try:
            # Получаем индикаторы
            indicators = analysis["indicators"]

            # Анализ тренда
            sma = indicators["sma"]
            ema = indicators["ema"]
            macd, signal, hist = indicators["macd"]

            # Определение тренда
            current_price = data["close"].iloc[-1]
            current_sma = sma.iloc[-1]
            current_ema = ema.iloc[-1]
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]

            # Проверка условий для long
            long_conditions = [
                current_price > current_sma,
                current_price > current_ema,
                current_macd > current_signal,
                hist.iloc[-1] > 0,
            ]

            # Проверка условий для short
            short_conditions = [
                current_price < current_sma,
                current_price < current_ema,
                current_macd < current_signal,
                hist.iloc[-1] < 0,
            ]

            # Определение направления
            if sum(long_conditions) >= 3:
                return "long"
            elif sum(short_conditions) >= 3:
                return "short"

            return None

        except Exception as e:
            logger.error(f"Error in _calculate_signal_direction: {str(e)}")
            return None

    def _calculate_stop_loss(
        self, entry_price: float, direction: str, analysis: Dict[str, Any]
    ) -> float:
        """
        Расчет уровня стоп-лосса.

        Args:
            entry_price: Цена входа
            direction: Направление позиции
            analysis: Результаты анализа

        Returns:
            float: Уровень стоп-лосса
        """
        try:
            # Получаем ATR
            analysis["indicators"]["atr"].iloc[-1]

            # Базовый стоп-лосс
            if direction == "long":
                stop_loss = entry_price * (1 - self.hedge_config.stop_loss)
            else:
                stop_loss = entry_price * (1 + self.hedge_config.stop_loss)

            # Корректировка на волатильность
            if self.hedge_config.use_volatility:
                volatility = analysis["volatility"]
                stop_loss = stop_loss * (1 - volatility)

            return stop_loss

        except Exception as e:
            logger.error(f"Error in _calculate_stop_loss: {str(e)}")
            return entry_price * (1 - self.hedge_config.stop_loss)

    def _calculate_take_profit(
        self, entry_price: float, direction: str, analysis: Dict[str, Any]
    ) -> float:
        """
        Расчет уровня тейк-профита.

        Args:
            entry_price: Цена входа
            direction: Направление позиции
            analysis: Результаты анализа

        Returns:
            float: Уровень тейк-профита
        """
        try:
            # Базовый тейк-профит
            if direction == "long":
                take_profit = entry_price * (1 + self.hedge_config.take_profit)
            else:
                take_profit = entry_price * (1 - self.hedge_config.take_profit)

            # Корректировка на волатильность
            if self.hedge_config.use_volatility:
                volatility = analysis["volatility"]
                take_profit = take_profit * (1 + volatility)

            return take_profit

        except Exception as e:
            logger.error(f"Error in _calculate_take_profit: {str(e)}")
            return entry_price * (1 + self.hedge_config.take_profit)

    def _calculate_position_size(
        self, entry_price: float, stop_loss: float, analysis: Dict[str, Any]
    ) -> float:
        """
        Расчет размера позиции.

        Args:
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            analysis: Результаты анализа

        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            risk_amount = (
                self.hedge_config.test_balance * self.hedge_config.risk_per_trade
            )
            risk_per_unit = abs(entry_price - stop_loss)
            position_size = risk_amount / risk_per_unit

            # Корректировка на уверенность
            confidence = self._calculate_confidence(analysis)
            position_size *= confidence

            # Ограничение размера позиции
            position_size = min(
                position_size,
                self.hedge_config.test_balance * self.hedge_config.max_position_size,
            )

            return position_size

        except Exception as e:
            logger.error(f"Error in _calculate_position_size: {str(e)}")
            return 0.0

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """
        Расчет уверенности в сигнале.

        Args:
            analysis: Результаты анализа

        Returns:
            float: Уверенность (0-1)
        """
        try:
            # Базовые факторы
            factors = []

            # Корреляция
            if self.hedge_config.use_correlation:
                correlation = analysis["correlation"]
                factors.append(correlation)

            # Волатильность
            if self.hedge_config.use_volatility:
                volatility = analysis["volatility"]
                factors.append(1 - volatility)

            # Режим рынка
            if self.hedge_config.use_regime:
                regime = analysis["regime"]
                if regime == "trend":
                    factors.append(0.8)
                elif regime == "sideways":
                    factors.append(0.5)
                else:
                    factors.append(0.3)

            # Настроения
            if self.hedge_config.use_sentiment and analysis["sentiment"]:
                sentiment = analysis["sentiment"]
                factors.append(sentiment)

            # Расчет итоговой уверенности
            if factors:
                confidence = sum(factors) / len(factors)
                return min(max(confidence, 0), 1)

            return 0.5

        except Exception as e:
            logger.error(f"Error in _calculate_confidence: {str(e)}")
            return 0.5

    def _validate_signal(self, signal: Signal) -> bool:
        """
        Валидация сигнала.

        Args:
            signal: Торговый сигнал

        Returns:
            bool: True если сигнал валиден
        """
        try:
            # Проверка уверенности
            if signal.confidence < self.hedge_config.confidence_threshold:
                return False

            # Проверка размера позиции
            if signal.volume <= 0:
                return False

            # Проверка стоп-лосса
            if signal.stop_loss:
                if (
                    signal.direction == "long"
                    and signal.stop_loss >= signal.entry_price
                ):
                    return False
                if (
                    signal.direction == "short"
                    and signal.stop_loss <= signal.entry_price
                ):
                    return False

            # Проверка тейк-профита
            if signal.take_profit:
                if (
                    signal.direction == "long"
                    and signal.take_profit <= signal.entry_price
                ):
                    return False
                if (
                    signal.direction == "short"
                    and signal.take_profit >= signal.entry_price
                ):
                    return False

            return True

        except Exception as e:
            logger.error(f"Error in _validate_signal: {str(e)}")
            return False

    def _analyze_correlation(self, data: pd.DataFrame) -> float:
        """
        Анализ корреляции.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Коэффициент корреляции
        """
        try:
            # Расчет корреляции между ценами закрытия
            returns = data["close"].pct_change().dropna()
            if len(returns) < 2:
                return 0.0
            return returns.autocorr()
        except Exception as e:
            logger.error(f"Error in _analyze_correlation: {str(e)}")
            return 0.0

    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """
        Анализ волатильности.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Волатильность
        """
        try:
            # Расчет волатильности
            returns = data["close"].pct_change().dropna()
            if len(returns) < 2:
                return 0.0
            return returns.std() * (252**0.5)
        except Exception as e:
            logger.error(f"Error in _analyze_volatility: {str(e)}")
            return 0.0

    def _analyze_spread(self, data: pd.DataFrame) -> float:
        """
        Анализ спреда.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            float: Спред
        """
        try:
            # Расчет спреда
            high_low_spread = (data["high"] - data["low"]) / data["low"]
            return high_low_spread.mean()
        except Exception as e:
            logger.error(f"Error in _analyze_spread: {str(e)}")
            return 0.0

    def _determine_market_regime(
        self, data: pd.DataFrame, indicators: Dict[str, Any]
    ) -> str:
        """
        Определение режима рынка.

        Args:
            data: DataFrame с OHLCV данными
            indicators: Словарь с индикаторами

        Returns:
            str: Режим рынка ('trend', 'sideways', 'unknown')
        """
        try:
            # Получаем индикаторы
            adx = indicators["adx"].iloc[-1]
            rsi = indicators["rsi"].iloc[-1]

            # Определение тренда
            if adx > self.hedge_config.trend_threshold:
                return "trend"

            # Определение боковика
            if 30 <= rsi <= 70:
                return "sideways"

            return "unknown"

        except Exception as e:
            logger.error(f"Error in _determine_market_regime: {str(e)}")
            return "unknown"

    def _analyze_sentiment(self, data: pd.DataFrame) -> Optional[float]:
        """
        Анализ настроений рынка.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Optional[float]: Значение настроений (0-1) или None
        """
        try:
            if not self.hedge_config.use_sentiment:
                return None

            # Расчет настроений на основе объема и цены
            volume_change = data["volume"].pct_change()
            price_change = data["close"].pct_change()

            # Корреляция между объемом и ценой
            correlation = volume_change.corr(price_change)

            # Нормализация
            sentiment = (correlation + 1) / 2

            return sentiment

        except Exception as e:
            logger.error(f"Error in _analyze_sentiment: {str(e)}")
            return None

    def optimize(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Оптимизация параметров стратегии.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с оптимизированными параметрами
        """
        try:
            # Проверка необходимости оптимизации
            if (
                datetime.now() - self.last_optimization
            ).total_seconds() < self.hedge_config.reoptimization_interval * 3600:
                return self.optimization_results

            # Параметры для оптимизации
            params = {
                "hedge_ratio": np.arange(0.5, 2.0, 0.1),
                "correlation_threshold": np.arange(0.5, 0.9, 0.1),
                "volatility_threshold": np.arange(0.01, 0.05, 0.005),
                "stop_loss": np.arange(0.01, 0.05, 0.005),
                "take_profit": np.arange(0.02, 0.1, 0.01),
            }

            # Оптимизация
            best_params = {}
            best_metric = float("-inf")

            for hedge_ratio in params["hedge_ratio"]:
                for corr_thresh in params["correlation_threshold"]:
                    for vol_thresh in params["volatility_threshold"]:
                        for sl in params["stop_loss"]:
                            for tp in params["take_profit"]:
                                # Обновляем параметры
                                self.hedge_config.hedge_ratio = hedge_ratio
                                self.hedge_config.correlation_threshold = corr_thresh
                                self.hedge_config.volatility_threshold = vol_thresh
                                self.hedge_config.stop_loss = sl
                                self.hedge_config.take_profit = tp

                                # Тестируем параметры
                                results = self._test_parameters(data)
                                metric = results.get(
                                    self.hedge_config.optimization_metric, 0
                                )

                                # Обновляем лучшие параметры
                                if metric > best_metric:
                                    best_metric = metric
                                    best_params = {
                                        "hedge_ratio": hedge_ratio,
                                        "correlation_threshold": corr_thresh,
                                        "volatility_threshold": vol_thresh,
                                        "stop_loss": sl,
                                        "take_profit": tp,
                                    }

            # Сохраняем результаты
            self.optimization_results = best_params
            self.last_optimization = datetime.now()

            # Обновляем параметры
            for param, value in best_params.items():
                setattr(self.hedge_config, param, value)

            return best_params

        except Exception as e:
            logger.error(f"Error in optimize: {str(e)}")
            return {}

    def _test_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Тестирование параметров стратегии.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с метриками
        """
        try:
            # Генерируем сигналы
            signals = []
            for i in range(len(data) - self.hedge_config.lookback_period):
                window = data.iloc[i : i + self.hedge_config.lookback_period]
                signal = self.generate_signal(window)
                if signal:
                    signals.append(signal)

            # Рассчитываем метрики
            if len(signals) < self.hedge_config.min_trades:
                return {}

            # Расчет прибыли
            profits = []
            for signal in signals:
                if signal.direction == "long":
                    profit = (
                        signal.take_profit - signal.entry_price
                    ) / signal.entry_price
                else:
                    profit = (
                        signal.entry_price - signal.take_profit
                    ) / signal.entry_price
                profits.append(profit)

            # Расчет метрик
            returns = pd.Series(profits)
            sharpe_ratio = (
                returns.mean() / returns.std() * (252**0.5) if len(returns) > 1 else 0
            )
            sortino_ratio = (
                returns.mean() / returns[returns < 0].std() * (252**0.5)
                if len(returns[returns < 0]) > 0
                else 0
            )
            win_rate = (
                len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0
            )

            return {
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "win_rate": win_rate,
                "total_trades": len(signals),
                "avg_profit": returns.mean(),
            }

        except Exception as e:
            logger.error(f"Error in _test_parameters: {str(e)}")
            return {}
