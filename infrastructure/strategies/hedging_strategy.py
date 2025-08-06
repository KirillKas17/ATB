import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from shared.decimal_utils import TradingDecimal, to_trading_decimal

# Проверка наличия talib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("TA-Lib not available. Using pandas/numpy alternatives.")
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
    # Общие параметры
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"
    # Дополнительные настройки
    additional: Dict[str, Any] = field(default_factory=dict)

class HedgingStrategy(BaseStrategy):
    """Стратегия хеджирования"""
    def __init__(self, config: Optional[Union[Dict[str, Any], HedgingConfig]] = None):
        """
        Инициализация стратегии хеджирования.
        Args:
            config: Словарь с параметрами стратегии или объект HedgingConfig
        """
        # Преобразуем конфигурацию в словарь для базового класса
        if isinstance(config, HedgingConfig):
            config_dict = {
                "hedge_ratio": config.hedge_ratio,
                "correlation_threshold": config.correlation_threshold,
                "volatility_threshold": config.volatility_threshold,
                "min_spread": config.min_spread,
                "max_spread": config.max_spread,
                "max_position_size": config.max_position_size,
                "stop_loss": config.stop_loss,
                "take_profit": config.take_profit,
                "trailing_stop": config.trailing_stop,
                "trailing_stop_activation": config.trailing_stop_activation,
                "trailing_stop_distance": config.trailing_stop_distance,
                "lookback_period": config.lookback_period,
                "min_trades": config.min_trades,
                "confidence_threshold": config.confidence_threshold,
                "optimization_period": config.optimization_period,
                "optimization_metric": config.optimization_metric,
                "reoptimization_interval": config.reoptimization_interval,
                "use_volume": config.use_volume,
                "use_volatility": config.use_volatility,
                "use_correlation": config.use_correlation,
                "use_regime": config.use_regime,
                "use_sentiment": config.use_sentiment,
                "regime_threshold": config.regime_threshold,
                "trend_threshold": config.trend_threshold,
                "volatility_regime_threshold": config.volatility_regime_threshold,
                "sentiment_threshold": config.sentiment_threshold,
                "sentiment_lookback": config.sentiment_lookback,
                "risk_per_trade": config.risk_per_trade,
                "max_drawdown": config.max_drawdown,
                "position_sizing": config.position_sizing,
                "monitor_interval": config.monitor_interval,
                "alert_threshold": config.alert_threshold,
                "log_level": config.log_level,
                "log_interval": config.log_interval,
                "test_mode": config.test_mode,
                "test_balance": config.test_balance,
                "symbols": config.symbols,
                "timeframes": config.timeframes,
                "log_dir": config.log_dir,
                "additional": config.additional,
            }
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
        
        super().__init__(config_dict)
        
        # Устанавливаем конфигурацию
        if isinstance(config, HedgingConfig):
            self._config = config
        elif isinstance(config, dict):
            self._config = HedgingConfig(**config)
        else:
            self._config = HedgingConfig()
            
        self._initialize_indicators()
        self._setup_optimization()
        self._setup_monitoring()

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, Optional[str]]:
        """
        Валидация входных данных.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Tuple[bool, Optional[str]]: (валидность, сообщение об ошибке)
        """
        try:
            if data is None or data.empty:
                return False, "Data is None or empty"
            
            required_columns = ["open", "high", "low", "close", "volume"]
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return False, f"Missing columns: {missing_columns}"
            
            if len(data) < self._config.lookback_period:
                return False, f"Insufficient data: {len(data)} < {self._config.lookback_period}"
            
            # Проверка на NaN значения
            for col in required_columns:
                if data[col].isna().any():
                    return False, f"NaN values found in column: {col}"
            
            # Проверка на отрицательные цены
            for col in ["open", "high", "low", "close"]:
                if (data[col] <= 0).any():
                    return False, f"Non-positive values found in column: {col}"
            
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def calculate_risk_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Расчет метрик риска.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict[str, float]: Словарь с метриками риска
        """
        try:
            returns = data["close"].pct_change().dropna()
            if len(returns) == 0:
                return {
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                    "var_95": 0.0
                }
            
            volatility = float(returns.std()) if returns.std() is not None and not pd.isna(returns.std()) else 0.0
            mean_return = float(returns.mean()) if returns.mean() is not None and not pd.isna(returns.mean()) else 0.0
            
            # Sharpe ratio
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0.0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = float(drawdown.min()) if drawdown.min() is not None and not pd.isna(drawdown.min()) else 0.0
            
            # Value at Risk (95%)
            var_95 = float(returns.quantile(0.05)) if returns.quantile(0.05) is not None and not pd.isna(returns.quantile(0.05)) else 0.0
            
            return {
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "var_95": var_95
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0
            }

    def _initialize_indicators(self) -> None:
        """Инициализация индикаторов"""
        if TALIB_AVAILABLE:
            self.indicators = {
                "sma": lambda x: talib.SMA(x, timeperiod=self._config.lookback_period),
                "ema": lambda x: talib.EMA(x, timeperiod=self._config.lookback_period),
                "rsi": lambda x: talib.RSI(x, timeperiod=self._config.lookback_period),
                "macd": lambda x: talib.MACD(
                    x, fastperiod=12, slowperiod=26, signalperiod=9
                ),
                "bollinger": lambda x: talib.BBANDS(
                    x, timeperiod=self._config.lookback_period, nbdevup=2, nbdevdn=2
                ),
                "atr": lambda x: talib.ATR(
                    x["high"],
                    x["low"],
                    x["close"],
                    timeperiod=self._config.lookback_period,
                ),
                "adx": lambda x: talib.ADX(
                    x["high"],
                    x["low"],
                    x["close"],
                    timeperiod=self._config.lookback_period,
                ),
            }
        else:
            # Альтернативные реализации с pandas/numpy
            self.indicators = {
                "sma": lambda x: x.rolling(window=self._config.lookback_period).mean(),
                "ema": lambda x: x.ewm(span=self._config.lookback_period).mean(),
                "rsi": lambda x: self._calculate_rsi(x, self._config.lookback_period),
                "macd": lambda x: self._calculate_macd(x),
                "bollinger": lambda x: self._calculate_bollinger_bands(x, self._config.lookback_period),
                "atr": lambda x: self._calculate_atr(x, self._config.lookback_period),
                "adx": lambda x: self._calculate_adx(x, self._config.lookback_period),
            }

    def _setup_optimization(self) -> None:
        """Настройка оптимизации"""
        self.last_optimization = datetime.now()
        self.optimization_results: dict[str, Any] = {}

    def _setup_monitoring(self) -> None:
        """Настройка мониторинга"""
        self.monitoring_data: Dict[str, Any] = {
            "positions": [],
            "trades": [],
            "metrics": {},
            "alerts": [],
        }

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
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
                if self._config.use_sentiment
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
            # Проверка условий хеджирования
            if not self._check_hedging_conditions(analysis):
                return None
            # Определение направления сигнала
            direction = self._calculate_signal_direction(data, analysis)
            if not direction:
                return None
            # Расчет цены входа
                        try:
                from shared.signal_validator import get_safe_price
                entry_price = get_safe_price(data["close"], -1, "entry_price")
            except (ValueError, ImportError):
                entry_price = data["close"].iloc[-1] if len(data['close']) > 0 else None
                entry_price = float(entry_price) if entry_price is not None and not pd.isna(entry_price) else None
                if entry_price is None or entry_price <= 0:
                return None
            # Расчет стоп-лосса
            stop_loss = self._calculate_stop_loss(entry_price, direction, analysis)
            # Расчет тейк-профита
            take_profit = self._calculate_take_profit(entry_price, direction, analysis)
            # Расчет размера позиции
            position_size = self._calculate_position_size(entry_price, stop_loss, analysis)
            # Расчет уверенности
            confidence = self._calculate_confidence(analysis)
            # Создание сигнала
            signal = Signal(
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volume=position_size,
                confidence=confidence,
                timestamp=datetime.now(),
            )
            # Валидация сигнала
            if not self._validate_signal(signal):
                return None
            return signal
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_hedging_conditions(self, analysis: Dict[str, Any]) -> bool:
        """Проверка условий для хеджирования."""
        try:
            # Проверка корреляции
            correlation = analysis.get("correlation", 0.0)
            correlation = float(correlation) if correlation is not None and not pd.isna(correlation) else 0.0
            if correlation < self._config.correlation_threshold:
                return False
            # Проверка волатильности
            volatility = analysis.get("volatility", 0.0)
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            if volatility < self._config.volatility_threshold:
                return False
            # Проверка спреда
            spread = analysis.get("spread", 0.0)
            spread = float(spread) if spread is not None and not pd.isna(spread) else 0.0
            if not (self._config.min_spread <= spread <= self._config.max_spread):
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking hedging conditions: {str(e)}")
            return False

    def _calculate_signal_direction(
        self, data: pd.DataFrame, analysis: Dict[str, Any]
    ) -> Optional[str]:
        """Расчет направления сигнала."""
        try:
            # Получаем индикаторы
            indicators = analysis.get("indicators", {})
            if not indicators:
                return None
            # Анализируем тренд
            trend = analysis.get("regime", "neutral")
            # Анализируем волатильность
            volatility = analysis.get("volatility", 0.0)
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            # Анализируем спред
            spread = analysis.get("spread", 0.0)
            spread = float(spread) if spread is not None and not pd.isna(spread) else 0.0
            # Определяем направление на основе анализа
            if trend == "uptrend" and volatility > self._config.volatility_threshold:
                return "long"
            elif trend == "downtrend" and volatility > self._config.volatility_threshold:
                return "short"
            return None
        except Exception as e:
            logger.error(f"Error calculating signal direction: {str(e)}")
            return None

    def _calculate_stop_loss(
        self, entry_price: float, direction: str, analysis: Dict[str, Any]
    ) -> float:
        """Расчет стоп-лосса."""
        try:
            stop_loss_pct = self._config.stop_loss
            if direction == "long":
                # Используем Decimal для точных расчетов
        entry_decimal = to_trading_decimal(entry_price)
        stop_loss_decimal = TradingDecimal.calculate_stop_loss(
            entry_decimal, "long", to_trading_decimal(stop_loss_pct * 100)
        )
        return float(stop_loss_decimal)
            else:
                # Используем Decimal для точных расчетов (short позиция)
                stop_loss_decimal = TradingDecimal.calculate_stop_loss(
                    entry_decimal, "short", to_trading_decimal(stop_loss_pct * 100)
                )
                return float(stop_loss_decimal)
        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price * 0.98  # Fallback

    def _calculate_take_profit(
        self, entry_price: float, direction: str, analysis: Dict[str, Any]
    ) -> float:
        """Расчет тейк-профита."""
        try:
            take_profit_pct = self._config.take_profit
            if direction == "long":
                # Используем Decimal для точных расчетов
            entry_decimal = to_trading_decimal(entry_price)
            take_profit_decimal = TradingDecimal.calculate_take_profit(
                entry_decimal, "long", to_trading_decimal(take_profit_pct * 100)
            )
            return float(take_profit_decimal)
            else:
                # Используем Decimal для точных расчетов (short позиция)
                take_profit_decimal = TradingDecimal.calculate_take_profit(
                    entry_decimal, "short", to_trading_decimal(take_profit_pct * 100)
                )
                return float(take_profit_decimal)
        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return entry_price * 1.02  # Fallback

    def _calculate_position_size(
        self, entry_price: float, stop_loss: float, analysis: Dict[str, Any]
    ) -> float:
        """Расчет размера позиции."""
        try:
            # Базовый размер на основе риска
            risk_amount = self._config.risk_per_trade
            risk_per_unit = abs(entry_price - stop_loss)
            if risk_per_unit == 0:
                position_size = self._config.max_position_size
            else:
                position_size = risk_amount / risk_per_unit
            # Ограничиваем размер позиции
            return min(position_size, self._config.max_position_size)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.0

    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """Расчет уверенности в сигнале."""
        try:
            # Базовая уверенность
            confidence = 0.5
            # Корректировка на волатильность
            volatility = analysis.get("volatility", 0.0)
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            vol_confidence = min(1.0, volatility / self._config.volatility_threshold)
            confidence += vol_confidence * 0.2
            # Корректировка на корреляцию
            correlation = analysis.get("correlation", 0.0)
            correlation = float(correlation) if correlation is not None and not pd.isna(correlation) else 0.0
            corr_confidence = min(1.0, correlation)
            confidence += corr_confidence * 0.2
            # Корректировка на режим рынка
            regime = analysis.get("regime", "neutral")
            if regime in ["trending", "volatile"]:
                confidence += 0.1
            return min(1.0, confidence)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

    def _validate_signal(self, signal: Signal) -> bool:
        """Валидация сигнала."""
        try:
            # Проверка уверенности
            if signal.confidence < self._config.confidence_threshold:
                return False
            # Проверка цены входа
            if signal.entry_price <= 0:
                return False
            # Проверка стоп-лосса
            if signal.stop_loss and signal.stop_loss <= 0:
                return False
            # Проверка тейк-профита
            if signal.take_profit and signal.take_profit <= 0:
                return False
            # Проверка объема
            if signal.volume and signal.volume <= 0:
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating signal: {str(e)}")
            return False

    def _analyze_correlation(self, data: pd.DataFrame) -> float:
        """Анализ корреляции."""
        try:
            # Простой анализ корреляции на основе цены и объема
            if len(data) < 2:
                return 0.0
            price_changes = data["close"].pct_change().dropna()
            volume_changes = data["volume"].pct_change().dropna()
            if len(price_changes) < 2 or len(volume_changes) < 2:
                return 0.0
            correlation = price_changes.corr(volume_changes)
            return float(correlation) if correlation is not None and not pd.isna(correlation) else 0.0
        except Exception as e:
            logger.error(f"Error analyzing correlation: {str(e)}")
            return 0.0

    def _analyze_volatility(self, data: pd.DataFrame) -> float:
        """Анализ волатильности."""
        try:
            returns = data["close"].pct_change().dropna()
            volatility = returns.std()
            return float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
            return 0.0

    def _analyze_spread(self, data: pd.DataFrame) -> float:
        """Анализ спреда."""
        try:
            spread = (data["high"] - data["low"]) / data["low"]
            spread_mean = spread.mean()
            return float(spread_mean) if spread_mean is not None and not pd.isna(spread_mean) else 0.0
        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return 0.0

    def _determine_market_regime(
        self, data: pd.DataFrame, indicators: Dict[str, Any]
    ) -> str:
        """Определение режима рынка."""
        try:
            # Анализ тренда
            if "adx" in indicators:
                adx_series = indicators["adx"]
                if isinstance(adx_series, pd.Series) and len(adx_series) > 0:
                    adx_value = adx_series.iloc[-1]
                    adx_value = float(adx_value) if adx_value is not None and not pd.isna(adx_value) else 0.0
                    if adx_value > self._config.trend_threshold:
                        return "trending"
            # Анализ волатильности
            volatility = self._analyze_volatility(data)
            if volatility > self._config.volatility_regime_threshold:
                return "volatile"
            # Анализ диапазона
            high_max = data["high"].max()
            low_min = data["low"].min()
            close_mean = data["close"].mean()
            if high_max is not None and not pd.isna(high_max) and low_min is not None and not pd.isna(low_min) and close_mean is not None and not pd.isna(close_mean) and close_mean > 0:
                price_range = (high_max - low_min) / close_mean
                if price_range < 0.02:
                    return "ranging"
            return "neutral"
        except Exception as e:
            logger.error(f"Error determining market regime: {str(e)}")
            return "neutral"

    def _analyze_sentiment(self, data: pd.DataFrame) -> Optional[float]:
        """Анализ настроений рынка."""
        try:
            if not self._config.use_sentiment:
                return None
            # Простой анализ настроений на основе объема и цены
            if len(data) < self._config.sentiment_lookback:
                return None
            recent_data = data.tail(self._config.sentiment_lookback) if len(data) >= self._config.sentiment_lookback else data.copy()
            close_last = recent_data["close"].iloc[-1]
            close_first = recent_data["close"].iloc[0]
            volume_last = recent_data["volume"].iloc[-1]
            volume_first = recent_data["volume"].iloc[0]
            
            if close_last is not None and not pd.isna(close_last) and close_first is not None and not pd.isna(close_first) and close_first > 0:
                price_trend = (close_last / close_first) - 1
            else:
                price_trend = 0.0
                
            if volume_last is not None and not pd.isna(volume_last) and volume_first is not None and not pd.isna(volume_first) and volume_first > 0:
                volume_trend = (volume_last / volume_first) - 1
            else:
                volume_trend = 0.0
                
            sentiment = (price_trend + volume_trend) / 2
            return float(sentiment) if sentiment is not None and not pd.isna(sentiment) else 0.0
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return None

    def optimize(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Оптимизация параметров стратегии."""
        try:
            # Проверяем, нужно ли оптимизировать
            if (
                datetime.now() - self.last_optimization
            ).total_seconds() < self._config.reoptimization_interval * 3600:
                return self.optimization_results
            # Тестируем различные параметры
            test_results = self._test_parameters(data)
            # Находим лучшие параметры
            best_metric = max(test_results.values())
            best_params = [k for k, v in test_results.items() if v == best_metric][0]
            # Обновляем конфигурацию
            if isinstance(best_params, (list, tuple)) and len(best_params) >= 5:
                hedge_ratio, corr_thresh, vol_thresh, sl, tp = best_params
            else:
                # Используем значения по умолчанию
                hedge_ratio, corr_thresh, vol_thresh, sl, tp = 1.0, 0.7, 0.02, 0.02, 0.04
            self._config.hedge_ratio = hedge_ratio
            self._config.correlation_threshold = corr_thresh
            self._config.volatility_threshold = vol_thresh
            self._config.stop_loss = sl
            self._config.take_profit = tp
            # Сохраняем результаты
            self.optimization_results = {
                "best_params": best_params,
                "best_metric": best_metric,
                "metric_name": self._config.optimization_metric,
                "timestamp": datetime.now()
            }
            self.last_optimization = datetime.now()
            return self.optimization_results
        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            return {}

    def _test_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """Тестирование различных параметров."""
        # Простая реализация для примера
        return {"default": 0.5}

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # type: ignore[operator]
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # type: ignore[operator]
            # Защита от деления на ноль
            rs = gain / loss.where(loss != 0, 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Заполняем NaN значения
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD."""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет полос Боллинджера."""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            return upper, sma, lower
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            zeros = pd.Series([0] * len(prices), index=prices.index)
            return zeros, zeros, zeros

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ATR."""
        try:
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr.fillna(0)  # Заполняем NaN значения
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series([0] * len(data), index=data.index)

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ADX."""
        try:
            # Упрощенная реализация ADX
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            adx = true_range.rolling(window=period).mean()
            return adx.fillna(0)  # Заполняем NaN значения
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.Series([0] * len(data), index=data.index)