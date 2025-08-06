import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import talib
except ImportError:
    talib = None
from loguru import logger
import pandas as pd
from shared.numpy_utils import np

from .base_strategy import BaseStrategy, Signal

warnings.filterwarnings("ignore")


@dataclass
class StatisticalArbitrageConfig:
    """Конфигурация стратегии статистического арбитража"""

    # Основные параметры
    lookback_period: int = 20  # Период для анализа
    z_score_threshold: float = 2.0  # Порог Z-score
    min_correlation: float = 0.7  # Минимальная корреляция
    min_cointegration: float = 0.05  # Минимальный p-value для коинтеграции
    min_half_life: int = 5  # Минимальный период полураспада
    max_half_life: int = 50  # Максимальный период полураспада
    # Параметры управления рисками
    max_position_size: float = 0.1  # Максимальный размер позиции
    stop_loss: float = 0.02  # Стоп-лосс
    take_profit: float = 0.04  # Тейк-профит
    trailing_stop: bool = True  # Использовать трейлинг-стоп
    trailing_stop_activation: float = 0.01  # Активация трейлинг-стопа
    trailing_stop_distance: float = 0.005  # Дистанция трейлинг-стопа
    # Параметры анализа
    min_trades: int = 10  # Минимальное количество сделок
    confidence_threshold: float = 0.7  # Порог уверенности
    min_spread: float = 0.001  # Минимальный спред
    max_spread: float = 0.05  # Максимальный спред
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
    # Адаптивные параметры
    adaptive_z_score: bool = True  # Адаптивный Z-score
    adaptive_correlation: bool = True  # Адаптивная корреляция
    adaptive_volatility: bool = True  # Адаптивная волатильность
    adaptive_position_sizing: bool = True  # Адаптивный размер позиции
    adaptive_stop_loss: bool = True  # Адаптивный стоп-лосс
    adaptive_take_profit: bool = True  # Адаптивный тейк-профит
    # Параметры для адаптации
    adaptation_window: int = 100  # Окно для адаптации
    adaptation_threshold: float = 0.1  # Порог для адаптации
    adaptation_speed: float = 0.1  # Скорость адаптации
    adaptation_method: str = "ewm"  # Метод адаптации (ewm, kalman, particle)


class StatisticalArbitrageStrategy(BaseStrategy):
    """Стратегия статистического арбитража"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии
        """
        super().__init__(config)
        
        # Устанавливаем конфигурацию для этого класса
        if isinstance(self.config, dict):
            self.arb_config = StatisticalArbitrageConfig(**self.config)
        else:
            self.arb_config = StatisticalArbitrageConfig()
        self._initialize_indicators()
        self._setup_optimization()
        self._setup_monitoring()

    def _initialize_indicators(self) -> None:
        """Инициализация индикаторов"""
        self.indicators = {
            "sma": lambda x: talib.SMA(x, timeperiod=self.arb_config.lookback_period),
            "ema": lambda x: talib.EMA(x, timeperiod=self.arb_config.lookback_period),
            "rsi": lambda x: talib.RSI(x, timeperiod=self.arb_config.lookback_period),
            "macd": lambda x: talib.MACD(
                x, fastperiod=12, slowperiod=26, signalperiod=9
            ),
            "bollinger": lambda x: talib.BBANDS(
                x, timeperiod=self.arb_config.lookback_period, nbdevup=2, nbdevdn=2
            ),
            "atr": lambda x: talib.ATR(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.arb_config.lookback_period,
            ),
            "adx": lambda x: talib.ADX(
                x["high"],
                x["low"],
                x["close"],
                timeperiod=self.arb_config.lookback_period,
            ),
        }

    def _setup_optimization(self) -> None:
        """Настройка оптимизации"""
        self.last_optimization = datetime.now()
        self.optimization_results = {}

    def _setup_monitoring(self) -> None:
        """Настройка мониторинга"""
        self.monitoring_data: Dict[str, Any] = {
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
            # Анализ коинтеграции
            cointegration = self._analyze_cointegration(data)
            # Анализ корреляции
            correlation = self._analyze_correlation(data)
            # Анализ волатильности
            volatility = self._analyze_volatility(data)
            # Анализ спреда
            spread = self._analyze_spread(data)
            # Расчет Z-score
            z_score = self._calculate_z_score(data)
            # Определение режима рынка
            regime = self._determine_market_regime(data, indicators)
            # Анализ настроений
            sentiment = (
                self._analyze_sentiment(data) if self.arb_config.use_sentiment else None
            )
            # Расчет метрик риска
            risk_metrics = self.calculate_risk_metrics(data)
            return {
                "indicators": indicators,
                "cointegration": cointegration,
                "correlation": correlation,
                "volatility": volatility,
                "spread": spread,
                "z_score": z_score,
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
            # Проверка условий для арбитража
            if not self._check_arbitrage_conditions(analysis):
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

    def _check_arbitrage_conditions(self, analysis: Dict[str, Any]) -> bool:
        """
        Проверка условий для арбитража.
        Args:
            analysis: Результаты анализа
        Returns:
            bool: True если условия выполнены
        """
        try:
            # Проверка коинтеграции
            if analysis["cointegration"]["p_value"] > self.arb_config.min_cointegration:
                return False
            # Проверка корреляции
            if self.arb_config.use_correlation:
                if analysis["correlation"] < self.arb_config.min_correlation:
                    return False
            # Проверка волатильности
            if self.arb_config.use_volatility:
                if analysis["volatility"] < self.arb_config.volatility_regime_threshold:
                    return False
            # Проверка спреда
            spread = analysis["spread"]
            if (
                spread < self.arb_config.min_spread
                or spread > self.arb_config.max_spread
            ):
                return False
            # Проверка Z-score
            z_score = analysis["z_score"]
            if z_score is not None and abs(z_score) < self.arb_config.z_score_threshold:
                return False
            # Проверка режима рынка
            if self.arb_config.use_regime:
                if analysis["regime"] == "unknown":
                    return False
            # Проверка настроений
            if self.arb_config.use_sentiment and analysis["sentiment"]:
                if analysis["sentiment"] < self.arb_config.sentiment_threshold:
                    return False
            return True
        except Exception as e:
            logger.error(f"Error in _check_arbitrage_conditions: {str(e)}")
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
            # Получаем Z-score
            z_score = analysis["z_score"]
            # Определение направления
            if z_score < -self.arb_config.z_score_threshold:
                return "long"
            elif z_score > self.arb_config.z_score_threshold:
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
            atr_value = analysis["indicators"]["atr"].iloc[-1] if "atr" in analysis["indicators"] and len(analysis["indicators"]["atr"]) > 0 else 0.0
            # Базовый стоп-лосс
            if direction == "long":
                stop_loss = entry_price * (1 - self.arb_config.stop_loss)
            else:
                stop_loss = entry_price * (1 + self.arb_config.stop_loss)
            # Корректировка на волатильность
            if self.arb_config.use_volatility:
                volatility = analysis.get("volatility", 0.0)
                stop_loss = stop_loss * (1 - volatility)
            return stop_loss
        except Exception as e:
            logger.error(f"Error in _calculate_stop_loss: {str(e)}")
            return entry_price * (1 - self.arb_config.stop_loss)

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
                take_profit = entry_price * (1 + self.arb_config.take_profit)
            else:
                take_profit = entry_price * (1 - self.arb_config.take_profit)
            # Корректировка на волатильность
            if self.arb_config.use_volatility:
                volatility = analysis.get("volatility", 0.0)
                take_profit = take_profit * (1 + volatility)
            return take_profit
        except Exception as e:
            logger.error(f"Error in _calculate_take_profit: {str(e)}")
            return entry_price * (1 + self.arb_config.take_profit)

    def _calculate_position_size(
        self, entry_price: float, stop_loss: float, analysis: Dict[str, Any]
    ) -> float:
        """
        Расчет размера позиции с учетом адаптивных параметров.
        Args:
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            analysis: Результаты анализа
        Returns:
            float: Размер позиции
        """
        try:
            # Получаем адаптивные параметры
            adaptive_params = self._calculate_adaptive_parameters(
                pd.DataFrame(analysis)
            )
            # Базовый размер позиции
            risk_amount = self.arb_config.test_balance * self.arb_config.risk_per_trade
            risk_per_unit = abs(entry_price - stop_loss)
            position_size = risk_amount / risk_per_unit
            # Корректировка на уверенность
            confidence = self._calculate_confidence(analysis)
            position_size *= confidence
            # Корректировка на волатильность
            if self.arb_config.adaptive_volatility:
                volatility = adaptive_params["volatility"]
                position_size *= 1 - volatility
            # Корректировка на корреляцию
            if self.arb_config.adaptive_correlation:
                correlation = adaptive_params["correlation"]
                position_size *= correlation
            # Корректировка на Z-score
            if self.arb_config.adaptive_z_score:
                z_score = adaptive_params["z_score"]
                position_size *= 1 - abs(z_score) / 3.0
            # Применение метода расчета размера позиции
            if self.arb_config.position_sizing == "kelly":
                position_size = self._apply_kelly_criterion(position_size, analysis)
            elif self.arb_config.position_sizing == "optimal_f":
                position_size = self._apply_optimal_f(position_size, analysis)
            elif self.arb_config.position_sizing == "risk_parity":
                position_size = self._apply_risk_parity(position_size, analysis)
            # Ограничение размера позиции
            position_size = min(
                position_size,
                self.arb_config.test_balance * self.arb_config.max_position_size,
            )
            return position_size
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _apply_kelly_criterion(
        self, position_size: float, analysis: Dict[str, Any]
    ) -> float:
        """
        Применение критерия Келли.
        Args:
            position_size: Базовый размер позиции
            analysis: Результаты анализа
        Returns:
            float: Скорректированный размер позиции
        """
        try:
            # Расчет вероятности успеха
            win_rate = analysis.get("win_rate", 0.5)
            # Расчет соотношения риск/доходность
            avg_profit = analysis.get("avg_profit", 0.02)
            avg_loss = analysis.get("avg_loss", 0.01)
            risk_reward = abs(avg_profit / avg_loss) if avg_loss != 0 else 1.0
            # Расчет критерия Келли
            kelly = win_rate - ((1 - win_rate) / risk_reward)
            # Ограничение критерия Келли
            kelly = max(min(kelly, 0.5), 0.0)
            return float(position_size * kelly)
        except Exception as e:
            logger.error(f"Error applying Kelly criterion: {str(e)}")
            return float(position_size * 0.5)

    def _apply_optimal_f(self, position_size: float, analysis: Dict[str, Any]) -> float:
        """
        Применение оптимального f.
        Args:
            position_size: Базовый размер позиции
            analysis: Результаты анализа
        Returns:
            float: Скорректированный размер позиции
        """
        try:
            # Расчет оптимального f
            returns = analysis.get("returns", pd.Series())
            if len(returns) < 2:
                return float(position_size * 0.5)
            # Расчет оптимального f методом Ньютона
            f = 0.5
            for _ in range(10):
                f_derivative = sum(returns / (1 + f * returns))
                f_second_derivative = sum(-(returns**2) / (1 + f * returns) ** 2)
                f = f - f_derivative / f_second_derivative
            # Ограничение оптимального f
            f = max(min(f, 0.5), 0.0)
            return float(position_size * f)
        except Exception as e:
            logger.error(f"Error applying optimal f: {str(e)}")
            return float(position_size * 0.5)

    def _apply_risk_parity(
        self, position_size: float, analysis: Dict[str, Any]
    ) -> float:
        """
        Применение риск-парити.
        Args:
            position_size: Базовый размер позиции
            analysis: Результаты анализа
        Returns:
            float: Скорректированный размер позиции
        """
        try:
            # Расчет волатильности
            volatility = analysis.get("volatility", 0.0)
            if volatility == 0:
                return float(position_size * 0.5)
            # Расчет размера позиции по риск-парити
            risk_parity_size = 1.0 / volatility
            # Ограничение размера позиции
            risk_parity_size = max(min(risk_parity_size, 1.0), 0.0)
            return float(position_size * risk_parity_size)
        except Exception as e:
            logger.error(f"Error applying risk parity: {str(e)}")
            return float(position_size * 0.5)

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
            # Коинтеграция
            cointegration = analysis["cointegration"]
            factors.append(1 - cointegration["p_value"])
            # Корреляция
            if self.arb_config.use_correlation:
                correlation = analysis["correlation"]
                factors.append(correlation)
            # Волатильность
            if self.arb_config.use_volatility:
                volatility = analysis["volatility"]
                factors.append(1 - volatility)
            # Z-score
            z_score = analysis["z_score"]
            factors.append(
                min(abs(z_score) / (2 * self.arb_config.z_score_threshold), 1)
            )
            # Режим рынка
            if self.arb_config.use_regime:
                regime = analysis["regime"]
                if regime == "trending":
                    factors.append(0.8)
                elif regime == "sideways":
                    factors.append(0.6)
                else:
                    factors.append(0.4)
            # Настроения
            if self.arb_config.use_sentiment and analysis["sentiment"]:
                sentiment = analysis["sentiment"]
                factors.append(sentiment)
            # Расчет средней уверенности
            if factors:
                confidence = sum(factors) / len(factors)
                return float(max(min(confidence, 1.0), 0.0))
            return 0.5
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
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
            if signal.confidence < self.arb_config.confidence_threshold:
                return False
            # Проверка размера позиции
            if signal.volume <= 0:
                return False
            # Проверка стоп-лосса
            if (
                signal.direction == "long"
                and signal.stop_loss is not None
                and signal.entry_price is not None
                and float(signal.stop_loss) >= float(signal.entry_price)
            ):
                return False
            if (
                signal.direction == "short"
                and signal.stop_loss is not None
                and signal.entry_price is not None
                and float(signal.stop_loss) <= float(signal.entry_price)
            ):
                return False
            # Проверка тейк-профита
            if signal.take_profit is not None and signal.entry_price is not None:
                if (
                    signal.direction == "long"
                    and float(signal.take_profit) <= float(signal.entry_price)
                ):
                    return False
                if (
                    signal.direction == "short"
                    and float(signal.take_profit) >= float(signal.entry_price)
                ):
                    return False
            return True
        except Exception as e:
            logger.error(f"Error in _validate_signal: {str(e)}")
            return False

    def _analyze_cointegration(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ коинтеграции.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с результатами анализа коинтеграции
        """
        try:
            # Расчет спреда
            spread = data["close"].diff().dropna()
            # Тест на коинтеграцию
            from statsmodels.tsa.stattools import adfuller

            adf_result = adfuller(spread)
            # Расчет периода полураспада
            half_life = self._calculate_half_life(spread)
            return {
                "p_value": adf_result[1],
                "test_statistic": adf_result[0],
                "half_life": half_life,
                "is_cointegrated": adf_result[1] < self.arb_config.min_cointegration,
            }
        except Exception as e:
            logger.error(f"Error in _analyze_cointegration: {str(e)}")
            return {
                "p_value": 1.0,
                "test_statistic": 0.0,
                "half_life": 0,
                "is_cointegrated": False,
            }

    def _calculate_half_life(self, spread: pd.Series) -> int:
        """
        Расчет периода полураспада.
        Args:
            spread: Series со спредом
        Returns:
            int: Период полураспада
        """
        try:
            # Регрессия
            spread_lag = spread.shift(1)
            spread_ret = spread - spread_lag
            spread_lag = spread_lag.dropna()
            spread_ret = spread_ret.dropna()
            # Расчет коэффициента
            spread_lag_array = spread_lag.to_numpy()
            spread_ret_array = spread_ret.to_numpy()
            beta = np.cov(spread_lag_array, spread_ret_array)[0, 1] / np.var(spread_lag_array)
            # Расчет периода полураспада
            half_life = int(np.log(2) / beta)
            # Ограничение периода
            half_life = max(
                min(half_life, self.arb_config.max_half_life),
                self.arb_config.min_half_life,
            )
            return half_life
        except Exception as e:
            logger.error(f"Error in _calculate_half_life: {str(e)}")
            return self.arb_config.min_half_life

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

    def _calculate_z_score(self, data: pd.DataFrame) -> float:
        """
        Расчет Z-score.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            float: Z-score
        """
        try:
            # Расчет спреда
            spread = data["close"].diff().dropna()
            # Расчет Z-score
            z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
            return z_score
        except Exception as e:
            logger.error(f"Error in _calculate_z_score: {str(e)}")
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
            if adx > self.arb_config.trend_threshold:
                return "trending"
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
            if not self.arb_config.use_sentiment:
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
        Оптимизация параметров стратегии с использованием расширенных методов.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с оптимизированными параметрами
        """
        try:
            # Проверка необходимости оптимизации
            if (
                datetime.now() - self.last_optimization
            ).total_seconds() < self.arb_config.reoptimization_interval * 3600:
                return self.optimization_results
            # Параметры для оптимизации
            params = {
                "z_score_threshold": np.arange(1.5, 3.0, 0.1),
                "min_correlation": np.arange(0.5, 0.9, 0.1),
                "min_cointegration": np.arange(0.01, 0.1, 0.01),
                "stop_loss": np.arange(0.01, 0.05, 0.005),
                "take_profit": np.arange(0.02, 0.1, 0.01),
                "lookback_period": np.arange(10, 50, 5),
                "adaptation_window": np.arange(50, 200, 25),
                "adaptation_speed": np.arange(0.05, 0.2, 0.05),
            }
            # Оптимизация с использованием различных методов
            results = {}
            # Генетическая оптимизация
            genetic_results = self._genetic_optimization(data, params)
            results["genetic"] = genetic_results
            # Байесовская оптимизация
            bayesian_results = self._bayesian_optimization(data, params)
            results["bayesian"] = bayesian_results
            # Оптимизация по сетке
            grid_results = self._grid_optimization(data, params)
            results["grid"] = grid_results
            # Выбор лучших параметров
            best_params = self._select_best_parameters(results)
            # Сохранение результатов
            self.optimization_results = best_params
            self.last_optimization = datetime.now()
            # Обновление параметров
            for param, value in best_params.items():
                setattr(self.arb_config, param, value)
            return best_params
        except Exception as e:
            logger.error(f"Error in optimize: {str(e)}")
            return {}

    def _genetic_optimization(
        self, data: pd.DataFrame, params: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Генетическая оптимизация параметров.
        Args:
            data: DataFrame с OHLCV данными
            params: Словарь с параметрами для оптимизации
        Returns:
            Dict с результатами оптимизации
        """
        try:
            from deap import algorithms, base, creator, tools

            # Создание классов для генетического алгоритма
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)
            # Инициализация генетического алгоритма
            toolbox = base.Toolbox()
            # Регистрация атрибутов
            for param, values in params.items():
                toolbox.register(f"attr_{param}", np.random.choice, values)
            # Регистрация создания особи
            toolbox.register(
                "individual",
                tools.initCycle,
                creator.Individual,
                [getattr(toolbox, f"attr_{param}") for param in params.keys()],
                n=1,
            )
            # Регистрация создания популяции
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            # Регистрация операторов
            toolbox.register(
                "evaluate",
                lambda ind: self._evaluate_parameters(
                    data, dict(zip(params.keys(), ind))
                ),
            )
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
            toolbox.register("select", tools.selTournament, tournsize=3)
            # Создание начальной популяции
            pop = toolbox.population(n=50)
            # Запуск генетического алгоритма
            result, logbook = algorithms.eaSimple(
                pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=20, verbose=False
            )
            # Выбор лучшей особи
            best_individual = tools.selBest(result, k=1)[0]
            return dict(zip(params.keys(), best_individual))
        except Exception as e:
            logger.error(f"Error in genetic optimization: {str(e)}")
            return {}

    def _bayesian_optimization(
        self, data: pd.DataFrame, params: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Байесовская оптимизация.
        Args:
            data: DataFrame с данными
            params: Параметры для оптимизации
        Returns:
            Dict с результатами оптимизации
        """
        try:
            # Проверяем доступность skopt
            try:
                from skopt import gp_minimize
                from skopt.space import Integer, Real
            except ImportError:
                logger.warning("skopt not available, skipping Bayesian optimization")
                return {"method": "bayesian", "success": False, "error": "skopt not available"}
            
            # Определение пространства поиска
            space = []
            for param_name, param_range in params.items():
                if param_name.endswith("_period"):
                    space.append(Integer(param_range[0], param_range[1], name=param_name))
                else:
                    space.append(Real(param_range[0], param_range[1], name=param_name))
            
            def objective(x):
                # Преобразование параметров
                param_dict = {space[i].name: x[i] for i in range(len(x))}
                # Оценка параметров
                return -self._evaluate_parameters(data, param_dict)
            
            # Запуск оптимизации
            result = gp_minimize(
                objective,
                space,
                n_calls=50,
                random_state=42,
                n_initial_points=10,
            )
            
            return {
                "method": "bayesian",
                "success": True,
                "best_params": {space[i].name: result.x[i] for i in range(len(result.x))},
                "best_score": -result.fun,
                "n_iterations": result.nit,
            }
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {str(e)}")
            return {"method": "bayesian", "success": False, "error": str(e)}

    def _grid_optimization(
        self, data: pd.DataFrame, params: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Оптимизация по сетке параметров.
        Args:
            data: DataFrame с OHLCV данными
            params: Словарь с параметрами для оптимизации
        Returns:
            Dict с результатами оптимизации
        """
        try:
            from itertools import product

            # Создание сетки параметров
            param_grid = [
                dict(zip(params.keys(), v)) for v in product(*params.values())
            ]
            # Ограничение размера сетки
            if len(param_grid) > 1000:
                import random
                param_grid = random.sample(param_grid, 1000)
            # Оценка параметров
            results = []
            for param_dict in param_grid:
                score = self._evaluate_parameters(data, param_dict)
                results.append((param_dict, score))
            # Выбор лучших параметров
            best_params = max(results, key=lambda x: x[1])[0]
            return best_params
        except Exception as e:
            logger.error(f"Error in grid optimization: {str(e)}")
            return {}

    def _evaluate_parameters(self, data: pd.DataFrame, params: Dict[str, Any]) -> float:
        """
        Оценка параметров стратегии.
        Args:
            data: DataFrame с OHLCV данными
            params: Словарь с параметрами
        Returns:
            float: Оценка параметров
        """
        try:
            # Обновление параметров
            for param, value in params.items():
                setattr(self.arb_config, param, value)
            # Тестирование параметров
            results = self._test_parameters(data)
            # Расчет оценки
            if not results:
                return float("-inf")
            # Взвешенная оценка метрик
            score = (
                results.get("sharpe_ratio", 0) * 0.3
                + results.get("sortino_ratio", 0) * 0.2
                + results.get("win_rate", 0) * 0.2
                + results.get("profit_factor", 0) * 0.2
                + (1 - results.get("max_drawdown", 1)) * 0.1
            )
            return score
        except Exception as e:
            logger.error(f"Error evaluating parameters: {str(e)}")
            return float("-inf")

    def _select_best_parameters(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Выбор лучших параметров из результатов различных методов оптимизации.
        Args:
            results: Словарь с результатами оптимизации
        Returns:
            Dict с лучшими параметрами
        """
        try:
            # Оценка результатов
            scores = {}
            for method, params in results.items():
                if not params:
                    continue
                score = self._evaluate_parameters(pd.DataFrame(), params)
                scores[method] = score
            # Выбор лучшего метода
            if not scores:
                return {}
            best_method = max(scores.items(), key=lambda x: x[1])[0]
            return results[best_method]
        except Exception as e:
            logger.error(f"Error selecting best parameters: {str(e)}")
            return {}

    def _calculate_adaptive_parameters(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Расчет адаптивных параметров.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с адаптивными параметрами
        """
        try:
            window = self.arb_config.adaptation_window
            returns = data["close"].pct_change().dropna()
            # Адаптивный Z-score
            if self.arb_config.adaptive_z_score:
                z_score = self._calculate_adaptive_z_score(returns, window)
            else:
                z_score = self.arb_config.z_score_threshold
            # Адаптивная корреляция
            if self.arb_config.adaptive_correlation:
                correlation = self._calculate_adaptive_correlation(returns, window)
            else:
                correlation = self.arb_config.min_correlation
            # Адаптивная волатильность
            if self.arb_config.adaptive_volatility:
                volatility = self._calculate_adaptive_volatility(returns, window)
            else:
                volatility = self._analyze_volatility(data)
            return {
                "z_score": z_score,
                "correlation": correlation,
                "volatility": volatility,
            }
        except Exception as e:
            logger.error(f"Error calculating adaptive parameters: {str(e)}")
            return {
                "z_score": self.arb_config.z_score_threshold,
                "correlation": self.arb_config.min_correlation,
                "volatility": 0.0,
            }

    def _calculate_adaptive_z_score(self, returns: pd.Series, window: int) -> float:
        """
        Расчет адаптивного Z-score.
        Args:
            returns: Series с доходностями
            window: Размер окна
        Returns:
            float: Адаптивный Z-score
        """
        try:
            if len(returns) < window:
                return float(self.arb_config.z_score_threshold)
            # Расчет скользящего среднего и стандартного отклонения
            mean = returns.ewm(span=window, adjust=False).mean()
            std = returns.ewm(span=window, adjust=False).std()
            # Расчет адаптивного Z-score
            z_score = abs((returns.iloc[-1] - mean.iloc[-1]) / std.iloc[-1])
            # Ограничение Z-score
            z_score = max(min(z_score, 3.0), 1.0)
            return float(z_score)
        except Exception as e:
            logger.error(f"Error calculating adaptive Z-score: {str(e)}")
            return float(self.arb_config.z_score_threshold)

    def _calculate_adaptive_correlation(self, returns: pd.Series, window: int) -> float:
        """
        Расчет адаптивной корреляции.
        Args:
            returns: Series с доходностями
            window: Размер окна
        Returns:
            float: Адаптивная корреляция
        """
        try:
            if len(returns) < window:
                return float(self.arb_config.min_correlation)
            # Расчет скользящей корреляции
            correlation = returns.rolling(window=window).corr(returns.shift(1))
            # Ограничение корреляции
            correlation = max(min(correlation.iloc[-1], 0.9), 0.5)
            return float(correlation)
        except Exception as e:
            logger.error(f"Error calculating adaptive correlation: {str(e)}")
            return float(self.arb_config.min_correlation)

    def _calculate_adaptive_volatility(self, returns: pd.Series, window: int) -> float:
        """
        Расчет адаптивной волатильности.
        Args:
            returns: Series с доходностями
            window: Размер окна
        Returns:
            float: Адаптивная волатильность
        """
        try:
            if len(returns) < window:
                return 0.0
            # Расчет скользящей волатильности
            volatility = returns.ewm(span=window, adjust=False).std()
            # Ограничение волатильности
            volatility = max(min(volatility.iloc[-1], 0.1), 0.01)
            return float(volatility)
        except Exception as e:
            logger.error(f"Error calculating adaptive volatility: {str(e)}")
            return 0.0

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет расширенных метрик риска.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с метриками риска
        """
        try:
            returns = data["close"].pct_change().dropna()
            if len(returns) < 2:
                return {}
            # Базовые метрики
            volatility = returns.std()
            max_drawdown = self._calculate_max_drawdown(data)
            # Расчет Value at Risk
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            # Расчет Conditional Value at Risk
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            # Расчет метрик риска
            sharpe_ratio = (
                returns.mean() / volatility * (252**0.5) if volatility != 0 else 0
            )
            sortino_ratio = (
                returns.mean() / returns[returns < 0].std() * (252**0.5)
                if len(returns[returns < 0]) > 0
                else 0
            )
            calmar_ratio = (
                returns.mean() / max_drawdown * (252**0.5) if max_drawdown != 0 else 0
            )
            # Расчет метрик распределения
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            # Расчет метрик просадки
            drawdown_duration = self._calculate_drawdown_duration(data)
            recovery_factor = float(returns.sum()) / max_drawdown if max_drawdown != 0 else 0
            return {
                "volatility": volatility,
                "max_drawdown": max_drawdown,
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
                "cvar_99": cvar_99,
                "sharpe_ratio": sharpe_ratio,
                "sortino_ratio": sortino_ratio,
                "calmar_ratio": calmar_ratio,
                "skewness": skewness,
                "kurtosis": kurtosis,
                "drawdown_duration": drawdown_duration,
                "recovery_factor": recovery_factor,
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {}

    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """
        Расчет максимальной просадки.
        Args:
            data: DataFrame с данными
        Returns:
            float: Максимальная просадка
        """
        try:
            # Расчет максимальной просадки
            cumulative_returns = (data["close"] / data["close"].iloc[0]) - 1
            max_drawdown = cumulative_returns.cummax().cummin().max()
            return float(max_drawdown)
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _calculate_drawdown_duration(self, data: pd.DataFrame) -> int:
        """
        Расчет продолжительности просадки.
        Args:
            data: DataFrame с данными
        Returns:
            int: Продолжительность просадки
        """
        try:
            # Расчет просадки
            cumulative_returns = (data["close"] / data["close"].iloc[0]) - 1
            drawdown = cumulative_returns.cummax().cummin()
            # Расчет продолжительности просадки
            drawdown_duration = 0
            max_duration = 0
            for value in drawdown:
                if value < 0:
                    drawdown_duration += 1
                    max_duration = max(max_duration, drawdown_duration)
                else:
                    drawdown_duration = 0
            return int(max_duration)
        except Exception as e:
            logger.error(f"Error calculating drawdown duration: {str(e)}")
            return 0

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
            for i in range(len(data) - self.arb_config.lookback_period):
                window = data.iloc[i : i + self.arb_config.lookback_period]
                signal = self.generate_signal(window)
                if signal:
                    signals.append(signal)
            # Рассчитываем метрики
            if len(signals) < self.arb_config.min_trades:
                return {}
            # Расчет прибыли
            profits = []
            for signal in signals:
                if signal.direction == "long":
                    if signal.take_profit is not None and signal.entry_price is not None:
                        profit = (
                            signal.take_profit - signal.entry_price
                        ) / signal.entry_price
                    else:
                        profit = 0.0
                else:
                    if signal.entry_price is not None and signal.take_profit is not None:
                        profit = (
                            signal.entry_price - signal.take_profit
                        ) / signal.entry_price
                    else:
                        profit = 0.0
                profits.append(profit)
            # Расчет метрик
            returns = pd.Series(profits)
            sharpe_ratio = (
                returns.mean() / returns.std() * (252**0.5) if len(returns) > 1 else 0
            )
            negative_returns = returns[returns < 0]
            sortino_ratio = (
                returns.mean() / negative_returns.std() * (252**0.5)
                if len(negative_returns) > 0
                else 0
            )
            positive_returns = returns[returns > 0]
            win_rate = (
                len(positive_returns) / len(returns) if len(returns) > 0 else 0
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
