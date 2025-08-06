import pandas as pd
from shared.numpy_utils import np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from loguru import logger

from .base_strategy import BaseStrategy, Signal


@dataclass
class RegimeConfig:
    """Конфигурация адаптивной стратегии"""

    min_confidence: float = 0.7
    max_drawdown: float = 0.1
    adaptation_rate: float = 0.1
    atr_period: int = 14
    atr_multiplier: float = 2.0
    volume_ma_period: int = 20
    risk_per_trade: float = 0.02
    max_position_size: float = 0.1
    trailing_stop: bool = True
    trailing_step: float = 0.002
    partial_close: bool = True
    partial_close_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    partial_close_sizes: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.4])
    max_trades: Optional[int] = None
    min_volume: float = 0.0
    min_volatility: float = 0.0
    max_spread: float = 0.01
    symbols: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=lambda: ["1h"])
    log_dir: str = "logs"
    use_volume: bool = True
    use_volatility: bool = True
    use_correlation: bool = True
    use_regime: bool = True
    # Параметры для разных режимов
    trend_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "bb_period": 20,
            "bb_std": 2.0,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "adx_period": 14,
            "adx_threshold": 25.0,
        }
    )
    volatility_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "rsi_period": 10,
            "rsi_overbought": 75,
            "rsi_oversold": 25,
            "bb_period": 15,
            "bb_std": 2.5,
            "macd_fast": 8,
            "macd_slow": 21,
            "macd_signal": 5,
            "adx_period": 10,
            "adx_threshold": 30.0,
        }
    )
    sideways_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "rsi_period": 20,
            "rsi_overbought": 65,
            "rsi_oversold": 35,
            "bb_period": 25,
            "bb_std": 1.8,
            "macd_fast": 16,
            "macd_slow": 32,
            "macd_signal": 9,
            "adx_period": 20,
            "adx_threshold": 20.0,
        }
    )


class RegimeAdaptiveStrategy(BaseStrategy):
    """Адаптивная стратегия, меняющая параметры в зависимости от режима рынка (расширенная)"""

    def __init__(self, config: Optional[Union[Dict[str, Any], RegimeConfig]] = None):
        """
        Инициализация стратегии.
        Args:
            config: Словарь с параметрами стратегии или объект конфигурации
        """
        if isinstance(config, RegimeConfig):
            super().__init__(asdict(config))
            self._config = config
        elif isinstance(config, dict):
            super().__init__(config)
            self._config = RegimeConfig(**config)
        else:
            super().__init__(None)
            self._config = RegimeConfig()
        self.position: Optional[str] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trailing_stop: Optional[float] = None
        self.partial_closes: List[float] = []
        self._setup_logger()

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
            
            if len(data) < 50:  # Минимальное количество данных для анализа
                return False, f"Insufficient data: {len(data)} < 50"
            
            # Проверка на NaN значения
            for col in required_columns:
                try:
                    if data[col].isna().any():
                        return False, f"NaN values found in column: {col}"
                except (AttributeError, TypeError):
                    return False, f"Invalid data type in column: {col}"
            
            # Проверка на отрицательные цены
            for col in ["open", "high", "low", "close"]:
                try:
                    if (data[col] <= 0).any():
                        return False, f"Non-positive values found in column: {col}"
                except (AttributeError, TypeError):
                    return False, f"Invalid data type in column: {col}"
            
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

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
            # Определение режима рынка
            regime = self.detect_regime(data)
            # Получение параметров для режима
            params = self._get_regime_params(regime)
            # Расчет индикаторов
            indicators = self._calculate_indicators(data, params)
            # Анализ состояния рынка
            market_state = self._analyze_market_state(data, indicators, regime)
            return {
                "regime": regime,
                "indicators": indicators,
                "market_state": market_state,
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
            indicators = analysis["indicators"]
            market_state = analysis["market_state"]
            regime = analysis["regime"]
            # Проверяем базовые условия
            if not self._check_basic_conditions(data, indicators):
                return None
            # Генерируем сигнал
            signal = self._generate_trading_signal(data, indicators, market_state, regime)
            if signal:
                self._update_position_state(signal, data)
            return signal
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _check_basic_conditions(
        self, data: pd.DataFrame, indicators: Dict[str, float]
    ) -> bool:
        """
        Проверка базовых условий для торговли.
        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
        Returns:
            bool: Результат проверки
        """
        try:
            # Проверка объема
            volume = data["volume"].iloc[-1]
            volume = float(volume) if volume is not None and not pd.isna(volume) else 0.0
            if volume < self._config.min_volume:
                return False
                
            # Проверка волатильности
            volatility = indicators.get("volatility", 0.0)
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            if volatility < self._config.min_volatility:
                return False
                
            # Проверка спреда
            try:
                if len(data) > 0 and all(col in data.columns for col in ["high", "low", "close"]):
                    high = data["high"].iloc[-1]
                    low = data["low"].iloc[-1]
                    close = data["close"].iloc[-1]
                    if high is not None and not pd.isna(high) and low is not None and not pd.isna(low) and close is not None and not pd.isna(close) and close > 0:
                        spread = (high - low) / close
                        spread = float(spread) if spread is not None and not pd.isna(spread) else 0.0
                        if spread > self._config.max_spread:
                            return False
            except (IndexError, KeyError, TypeError, ZeroDivisionError):
                pass
                
            return True
        except Exception as e:
            logger.error(f"Error checking basic conditions: {str(e)}")
            return False

    def _generate_trading_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Генерация торгового сигнала.
        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            regime: Режим рынка
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            if self.position is None:
                return self._generate_entry_signal(
                    data, indicators, market_state, regime
                )
            else:
                return self._generate_exit_signal(
                    data, indicators, market_state, regime
                )
        except Exception as e:
            logger.error(f"Error generating trading signal: {str(e)}")
            return None

    def _generate_entry_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Генерация сигнала на вход в позицию.
        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            regime: Режим рынка
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            try:
                if len(data) > 0 and "close" in data.columns:
                    current_price = data["close"].iloc[-1]
                    current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0
                else:
                    return None
            except (IndexError, KeyError, TypeError):
                return None
                
            if current_price <= 0:
                return None
            params = self._get_regime_params(regime)
            # Проверяем условия для длинной позиции
            if (
                indicators["rsi"] < params["rsi_oversold"]
                and current_price < indicators["bb_lower"]
                and indicators["macd"] > indicators["macd_signal"]
                and indicators["adx"] > params["adx_threshold"]
            ):
                # Рассчитываем размер позиции
                atr_value = indicators.get("atr", 0.0)
                atr_value = float(atr_value) if atr_value is not None and not pd.isna(atr_value) else 0.0
                volume = self._calculate_position_size(current_price, atr_value)
                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price - atr_value * self._config.atr_multiplier
                take_profit = current_price + atr_value * self._config.atr_multiplier * 2
                return Signal(
                    direction="long",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=self._calculate_confidence(indicators, regime),
                    timestamp=data.index[-1],
                    metadata={
                        "regime": regime,
                        "indicators": indicators,
                        "market_state": market_state,
                    },
                )
            # Проверяем условия для короткой позиции
            elif (
                indicators["rsi"] > params["rsi_overbought"]
                and current_price > indicators["bb_upper"]
                and indicators["macd"] < indicators["macd_signal"]
                and indicators["adx"] > params["adx_threshold"]
            ):
                # Рассчитываем размер позиции
                atr_value = indicators.get("atr", 0.0)
                atr_value = float(atr_value) if atr_value is not None and not pd.isna(atr_value) else 0.0
                volume = self._calculate_position_size(current_price, atr_value)
                # Устанавливаем стоп-лосс и тейк-профит
                stop_loss = current_price + atr_value * self._config.atr_multiplier
                take_profit = current_price - atr_value * self._config.atr_multiplier * 2
                return Signal(
                    direction="short",
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    volume=volume,
                    confidence=self._calculate_confidence(indicators, regime),
                    timestamp=data.index[-1],
                    metadata={
                        "regime": regime,
                        "indicators": indicators,
                        "market_state": market_state,
                    },
                )
            return None
        except Exception as e:
            logger.error(f"Error generating entry signal: {str(e)}")
            return None

    def _generate_exit_signal(
        self,
        data: pd.DataFrame,
        indicators: Dict[str, float],
        market_state: Dict[str, Any],
        regime: str,
    ) -> Optional[Signal]:
        """
        Генерация сигнала на выход из позиции.
        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            market_state: Состояние рынка
            regime: Режим рынка
        Returns:
            Optional[Signal] с сигналом или None
        """
        try:
            try:
                if len(data) > 0 and "close" in data.columns:
                    current_price = data["close"].iloc[-1]
                    current_price = float(current_price) if current_price is not None and not pd.isna(current_price) else 0.0
                else:
                    return None
            except (IndexError, KeyError, TypeError):
                return None
                
            if current_price <= 0:
                return None
            # Проверяем стоп-лосс и тейк-профит
            if self.position == "long":
                if current_price <= (self.stop_loss or 0) or current_price >= (self.take_profit or float('inf')):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss_or_take_profit",
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )
            elif self.position == "short":
                if current_price >= (self.stop_loss or float('inf')) or current_price <= (self.take_profit or 0):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "stop_loss_or_take_profit",
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )
            # Проверяем трейлинг-стоп
            if self._config.trailing_stop and self.trailing_stop:
                atr_value = indicators.get("atr", 0.0)
                atr_value = float(atr_value) if atr_value is not None and not pd.isna(atr_value) else 0.0
                if self.position == "long" and current_price > self.trailing_stop:
                    self.trailing_stop = current_price * (1 - self._config.trailing_step)
                elif self.position == "short" and current_price < self.trailing_stop:
                    self.trailing_stop = current_price * (1 + self._config.trailing_step)
                
                # Проверка трейлинг-стопа
                if (self.position == "long" and current_price <= (self.trailing_stop or 0)) or (self.position == "short" and current_price >= (self.trailing_stop or float('inf'))):
                    return Signal(
                        direction="close",
                        entry_price=current_price,
                        timestamp=data.index[-1],
                        confidence=1.0,
                        metadata={
                            "reason": "trailing_stop",
                            "indicators": indicators,
                            "market_state": market_state,
                        },
                    )
            # Частичное закрытие позиций
            if self._config.partial_close and self.position:
                for i, level in enumerate(self._config.partial_close_levels):
                    if self.position == "long" and current_price >= (self.take_profit or 0) * level:
                        return Signal(
                            direction="partial_close",
                            entry_price=current_price,
                            volume=self._config.partial_close_sizes[i],
                            timestamp=data.index[-1],
                            confidence=1.0,
                            metadata={
                                "reason": "partial_close",
                                "level": level,
                                "indicators": indicators,
                                "market_state": market_state,
                            },
                        )
                    elif self.position == "short" and current_price <= (self.take_profit or float('inf')) * level:
                        return Signal(
                            direction="partial_close",
                            entry_price=current_price,
                            volume=self._config.partial_close_sizes[i],
                            timestamp=data.index[-1],
                            confidence=1.0,
                            metadata={
                                "reason": "partial_close",
                                "level": level,
                                "indicators": indicators,
                                "market_state": market_state,
                            },
                        )
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
                self.trailing_stop = signal.entry_price
                self.partial_closes = []
            elif signal.direction == "close":
                self.position = None
                self.stop_loss = None
                self.take_profit = None
                self.trailing_stop = None
                self.partial_closes = []
            elif signal.direction == "partial_close":
                if signal.volume is not None:
                    self.partial_closes.append(signal.volume)
        except Exception as e:
            logger.error(f"Error updating position state: {str(e)}")

    def detect_regime(self, data: pd.DataFrame) -> str:
        """
        Определение режима рынка.
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            str: Режим рынка ('trend', 'volatility', 'sideways')
        """
        try:
            # Расчет волатильности
            returns = data["close"].pct_change()
            volatility = returns.std()
            volatility = float(volatility) if volatility is not None and not pd.isna(volatility) else 0.0
            # Расчет тренда
            sma_20 = data["close"].rolling(window=20).mean()
            sma_50 = data["close"].rolling(window=50).mean()
            sma_20_last = sma_20.iloc[-1]
            sma_50_last = sma_50.iloc[-1]
            if sma_20_last is not None and not pd.isna(sma_20_last) and sma_50_last is not None and not pd.isna(sma_50_last) and sma_50_last > 0:
                trend_strength = abs(sma_20_last - sma_50_last) / sma_50_last
            else:
                trend_strength = 0.0
            # Правильный расчет ADX согласно стандартной формуле
            adx_last = self._calculate_adx_correct(data)
            adx_last = float(adx_last) if adx_last is not None and not pd.isna(adx_last) else 0.0
            # Используем адаптивные пороги для определения режима
            from shared.adaptive_thresholds import AdaptiveThresholds
            adaptive_thresholds = AdaptiveThresholds()
            
            trend_threshold = adaptive_thresholds.get_adaptive_trend_threshold(data)
            volatility_regime = adaptive_thresholds.get_volatility_regime(data)
            
            # Определение режима с адаптивными порогами
            if trend_strength > trend_threshold and adx_last > 25:  # type: ignore[operator]
                return "trend"
            elif volatility_regime == 'high':
                return "volatility"
            else:
                return "sideways"
        except Exception as e:
            logger.error(f"Error detecting regime: {str(e)}")
            return "sideways"

    def _calculate_adx_correct(self, data: pd.DataFrame, period: int = 14) -> float:
        """
        Правильный расчет Average Directional Index (ADX).
        
        Args:
            data: DataFrame с OHLCV данными
            period: Период для расчета (по умолчанию 14)
            
        Returns:
            float: Значение ADX
        """
        try:
            if len(data) < period + 1:
                return 0.0
                
            # Расчет True Range (TR)
            high = data["high"]
            low = data["low"]
            close = data["close"]
            
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = np.abs(high - prev_close)
            tr3 = np.abs(low - prev_close)
            
            # True Range = максимум из трех значений
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            
            # Расчет Directional Movement (DM)
            plus_dm = high.diff()
            minus_dm = -low.diff()
            
            # Оставляем только положительные движения
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
            
            # Сглаживание по методу Wilder (как экспоненциальная скользящая средняя)
            alpha = 1.0 / period
            
            # Сглаженный TR
            atr = tr.ewm(alpha=alpha, adjust=False).mean()
            
            # Сглаженные DM
            plus_di_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
            minus_di_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()
            
            # Directional Indicators (DI)
            plus_di = 100 * (plus_di_smooth / atr)
            minus_di = 100 * (minus_di_smooth / atr)
            
            # Directional Index (DX)
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            dx = dx.fillna(0)  # Заменяем NaN на 0
            
            # ADX = сглаженный DX
            adx = dx.ewm(alpha=alpha, adjust=False).mean()
            
            return float(adx.iloc[-1]) if len(adx) > 0 and not pd.isna(adx.iloc[-1]) else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return 0.0

    def _get_regime_params(self, regime: str) -> Dict[str, Any]:
        """
        Получение параметров для режима.
        Args:
            regime: Режим рынка
        Returns:
            Dict с параметрами
        """
        try:
            if regime == "trend":
                return self._config.trend_params
            elif regime == "volatility":
                return self._config.volatility_params
            else:
                return self._config.sideways_params
        except Exception as e:
            logger.error(f"Error getting regime parameters: {str(e)}")
            return self._config.sideways_params

    def _calculate_indicators(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Расчет индикаторов.
        Args:
            data: DataFrame с OHLCV данными
            params: Параметры индикаторов
        Returns:
            Dict с значениями индикаторов
        """
        try:
            # RSI
            delta = data["close"].diff()
            gain = (
                (delta.where(delta > 0, 0)).rolling(window=params["rsi_period"]).mean()  # type: ignore[operator]
            )
            loss = (
                (-delta.where(delta < 0, 0)).rolling(window=params["rsi_period"]).mean()  # type: ignore[operator]
            )
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            # Bollinger Bands
            bb_middle = data["close"].rolling(window=params["bb_period"]).mean()
            bb_std = data["close"].rolling(window=params["bb_period"]).std()
            bb_upper = bb_middle + (bb_std * params["bb_std"])
            bb_lower = bb_middle - (bb_std * params["bb_std"])
            # MACD
            ema_fast = data["close"].ewm(span=params["macd_fast"]).mean()
            ema_slow = data["close"].ewm(span=params["macd_slow"]).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=params["macd_signal"]).mean()
            # ATR
            high_low = data["high"] - data["low"]
            high_close = np.abs(data["high"] - data["close"].shift())
            low_close = np.abs(data["low"] - data["close"].shift())
            ranges = pd.DataFrame({
                'high_low': high_low,
                'high_close': high_close,
                'low_close': low_close
            })
            true_range = ranges.max(axis=1)
            atr = true_range.rolling(window=self._config.atr_period).mean()
            # ADX
            plus_dm = data["high"].diff()
            minus_dm = data["low"].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            tr = true_range
            plus_di = 100 * (
                plus_dm.rolling(window=params["adx_period"]).mean()
                / tr.rolling(window=params["adx_period"]).mean()
            )
            minus_di = 100 * (
                minus_dm.rolling(window=params["adx_period"]).mean()
                / tr.rolling(window=params["adx_period"]).mean()
            )
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=params["adx_period"]).mean()
            # Волатильность
            returns = data["close"].pct_change()
            volatility = returns.rolling(window=20).std()
            # Volume MA
            volume_ma = (
                data["volume"].rolling(window=self._config.volume_ma_period).mean()
            )
            return {
                "rsi": float(rsi.iloc[-1]) if rsi.iloc[-1] is not None and not pd.isna(rsi.iloc[-1]) else 50.0,
                "bb_upper": float(bb_upper.iloc[-1]) if bb_upper.iloc[-1] is not None and not pd.isna(bb_upper.iloc[-1]) else 0.0,
                "bb_middle": float(bb_middle.iloc[-1]) if bb_middle.iloc[-1] is not None and not pd.isna(bb_middle.iloc[-1]) else 0.0,
                "bb_lower": float(bb_lower.iloc[-1]) if bb_lower.iloc[-1] is not None and not pd.isna(bb_lower.iloc[-1]) else 0.0,
                "macd": float(macd.iloc[-1]) if macd.iloc[-1] is not None and not pd.isna(macd.iloc[-1]) else 0.0,
                "macd_signal": float(macd_signal.iloc[-1]) if macd_signal.iloc[-1] is not None and not pd.isna(macd_signal.iloc[-1]) else 0.0,
                "atr": float(atr.iloc[-1]) if atr.iloc[-1] is not None and not pd.isna(atr.iloc[-1]) else 0.0,
                "adx": float(adx.iloc[-1]) if adx.iloc[-1] is not None and not pd.isna(adx.iloc[-1]) else 0.0,
                "plus_di": float(plus_di.iloc[-1]) if plus_di.iloc[-1] is not None and not pd.isna(plus_di.iloc[-1]) else 0.0,
                "minus_di": float(minus_di.iloc[-1]) if minus_di.iloc[-1] is not None and not pd.isna(minus_di.iloc[-1]) else 0.0,
                "volatility": float(volatility.iloc[-1]) if volatility.iloc[-1] is not None and not pd.isna(volatility.iloc[-1]) else 0.0,
                "volume_ma": float(volume_ma.iloc[-1]) if volume_ma.iloc[-1] is not None and not pd.isna(volume_ma.iloc[-1]) else 0.0,
            }
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return {}

    def _analyze_market_state(
        self, data: pd.DataFrame, indicators: Dict[str, float], regime: str
    ) -> Dict[str, Any]:
        """
        Анализ состояния рынка.
        Args:
            data: DataFrame с OHLCV данными
            indicators: Значения индикаторов
            regime: Режим рынка
        Returns:
            Dict с состоянием рынка
        """
        try:
            # Тренд
            trend = "up" if indicators["macd"] > indicators["macd_signal"] else "down"
            # Сила тренда
            trend_strength = "strong" if indicators["adx"] > 25 else "weak"
            # Волатильность
            volatility = "high" if indicators["volatility"] > 0.015 else "low"
            # Объем
            current_volume = data["volume"].iloc[-1]
            current_volume = float(current_volume) if current_volume is not None and not pd.isna(current_volume) else 0.0
            volume = "high" if current_volume > indicators["volume_ma"] else "low"
            # Перекупленность/перепроданность
            overbought = indicators["rsi"] > 30
            oversold = indicators["rsi"] < 70
            return {
                "regime": regime,
                "trend": trend,
                "trend_strength": trend_strength,
                "volatility": volatility,
                "volume": volume,
                "overbought": overbought,
                "oversold": oversold,
            }
        except Exception as e:
            logger.error(f"Error analyzing market state: {str(e)}")
            return {}

    def _calculate_position_size(self, price: float, atr: float) -> float:
        """
        Расчет размера позиции.
        Args:
            price: Текущая цена
            atr: ATR
        Returns:
            float: Размер позиции
        """
        try:
            # Базовый размер позиции
            base_size = self._config.risk_per_trade
            # Корректировка на волатильность
            volatility_factor = 1 / (1 + atr / price) if price > 0 else 1.0
            # Корректировка на максимальный размер
            position_size = base_size * volatility_factor
            position_size = min(position_size, self._config.max_position_size)
            return float(position_size)
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return float(self._config.risk_per_trade)

    def _calculate_confidence(self, indicators: Dict[str, float], regime: str) -> float:
        """
        Расчет уверенности в сигнале.
        Args:
            indicators: Значения индикаторов
            regime: Режим рынка
        Returns:
            float: Уверенность (0-1)
        """
        try:
            # Нормализация индикаторов
            rsi_conf = 1 - abs(indicators["rsi"] - 50) / 50
            macd_signal = indicators.get("macd_signal", 0.0)
            if macd_signal != 0:
                macd_conf = abs(indicators["macd"] - macd_signal) / abs(macd_signal)
            else:
                macd_conf = 0.0
            adx_conf = indicators["adx"] / 100
            bb_middle = indicators.get("bb_middle", 0.0)
            if bb_middle > 0:
                bb_conf = 1 - abs(bb_middle - indicators["bb_lower"]) / bb_middle
            else:
                bb_conf = 0.0
            # Взвешенная сумма в зависимости от режима
            if regime == "trend":
                confidence = (
                    0.3 * macd_conf + 0.3 * adx_conf + 0.2 * rsi_conf + 0.2 * bb_conf
                )
            elif regime == "volatility":
                confidence = (
                    0.3 * bb_conf + 0.3 * rsi_conf + 0.2 * macd_conf + 0.2 * adx_conf
                )
            else:  # sideways
                confidence = (
                    0.3 * rsi_conf + 0.3 * bb_conf + 0.2 * macd_conf + 0.2 * adx_conf
                )
            return min(max(confidence, 0), 1)
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
