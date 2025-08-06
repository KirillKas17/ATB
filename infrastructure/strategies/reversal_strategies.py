import pandas as pd
from shared.numpy_utils import np
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
import pandas as pd

from loguru import logger
from shared.signal_validator import get_safe_price
from shared.decimal_utils import TradingDecimal, to_trading_decimal

from .base_strategy import BaseStrategy, Signal as BaseSignal


@dataclass
class ReversalConfig:
    """Конфигурация стратегии разворота тренда"""

    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volume_threshold: float = 1.5
    min_swing_points: int = 3
    swing_threshold: float = 0.02
    confirmation_candles: int = 3
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    trailing_stop: bool = True
    trailing_step: float = 0.02
    partial_close: bool = True
    partial_close_levels: List[float] = field(default_factory=lambda: [0.5, 0.3, 0.2])
    risk_per_trade: float = 0.01
    max_trades: Optional[int] = None
    log_dir: str = "logs"


class ReversalStrategy(BaseStrategy):
    """Стратегия разворота тренда"""

    def __init__(self, config: Optional[Union[Dict[str, Any], ReversalConfig]] = None):
        if isinstance(config, ReversalConfig):
            super().__init__(asdict(config))
            self._config = config
        elif isinstance(config, dict):
            super().__init__(config)
            self._config = ReversalConfig(**config)
        else:
            super().__init__(None)
            self._config = ReversalConfig()
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
            
            min_periods = max(
                self._config.rsi_period,
                self._config.macd_slow,
                self._config.confirmation_candles,
                50  # Минимальное количество данных для анализа
            )
            if len(data) < min_periods:
                return False, f"Insufficient data: {len(data)} < {min_periods}"
            
            # Проверка на NaN значения
            for col in required_columns:
                if pd.isna(data[col]).any():
                    return False, f"NaN values found in column: {col}"
            
            # Проверка на отрицательные цены
            for col in ["open", "high", "low", "close"]:
                if (data[col] <= 0).any():
                    return False, f"Non-positive values found in column: {col}"
            
            return True, None
        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _setup_logger(self) -> None:
        """Настройка логгера"""
        logger.add(
            f"{self._config.log_dir}/reversal_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def analyze(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Анализ рыночных данных для поиска разворотов тренда
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Dict с результатами анализа
        """
        try:
            is_valid, error = self.validate_data(data)
            if not is_valid:
                raise ValueError(f"Invalid data: {error}")
            
            # Рассчитываем индикаторы
            rsi = self._calculate_rsi(data["close"])
            macd, signal, hist = self._calculate_macd(data["close"])
            atr = self._calculate_atr(data)
            swing_points = self._find_swing_points(data)
            # Анализируем объемы
            volume_analysis = self._analyze_volume(data["volume"])
            # Определяем тренд
            trend = self._determine_trend(data, swing_points)
            # Ищем потенциальные развороты
            reversals = self._find_potential_reversals(
                data, rsi, macd, signal, swing_points, volume_analysis
            )
            # Формируем сигналы
            signals = self._generate_signals(data, reversals, atr, trend)
            return {
                "rsi": rsi,
                "macd": macd,
                "signal": signal,
                "histogram": hist,
                "atr": atr,
                "swing_points": swing_points,
                "volume_analysis": volume_analysis,
                "trend": trend,
                "reversals": reversals,
                "signals": signals,
            }
        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
            return {}

    def _calculate_rsi(self, prices: pd.Series, period: Optional[int] = None) -> pd.Series:
        """Расчет RSI"""
        period = period or self._config.rsi_period
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean() 
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean() 
        # Защита от деления на ноль
        rs = gain / loss.where(loss != 0, 1e-10)  # Заменяем 0 на очень маленькое число
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD"""
        exp1 = prices.ewm(span=self._config.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self._config.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self._config.macd_signal, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Расчет ATR"""
        high = data["high"]
        low = data["low"]
        close = data["close"]
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _find_swing_points(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        """Поиск точек разворота"""
        highs = []
        lows = []
        for i in range(2, len(data) - 2):
            # Поиск локальных максимумов
            if (
                data["high"].iloc[i] > data["high"].iloc[i - 1] 
                and data["high"].iloc[i] > data["high"].iloc[i - 2] 
                and data["high"].iloc[i] > data["high"].iloc[i + 1] 
                and data["high"].iloc[i] > data["high"].iloc[i + 2] 
            ):
                highs.append(i)
            # Поиск локальных минимумов
            if (
                data["low"].iloc[i] < data["low"].iloc[i - 1] 
                and data["low"].iloc[i] < data["low"].iloc[i - 2] 
                and data["low"].iloc[i] < data["low"].iloc[i + 1] 
                and data["low"].iloc[i] < data["low"].iloc[i + 2] 
            ):
                lows.append(i)
        return {"highs": highs, "lows": lows}

    def _analyze_volume(self, volume: pd.Series) -> Dict[str, Any]:
        """Анализ объемов"""
        avg_volume = volume.rolling(window=20).mean()
        # Защита от деления на ноль
        volume_ratio = volume / avg_volume.where(avg_volume != 0, 1.0)
        return {
            "average": avg_volume,
            "ratio": volume_ratio,
            "is_high": volume_ratio > self._config.volume_threshold,
        }

    def _safe_get_price(self, price_series: pd.Series, index: int) -> float:
        """Безопасное получение цены с проверками"""
        try:
            price = price_series.iloc[index]
            if price is not None and not pd.isna(price) and price > 0:
                return float(price)
            # Если цена некорректна, пытаемся получить последнюю корректную цену
            for i in range(index, max(0, index - 10), -1):
                fallback_price = price_series.iloc[i]
                if fallback_price is not None and not pd.isna(fallback_price) and fallback_price > 0:
                    logger.warning(f"Using fallback price {fallback_price} instead of invalid price at index {index}")
                    return float(fallback_price)
            # Если и это не помогло, используем среднюю цену за последние данные
            recent_prices = price_series.dropna()
            if len(recent_prices) > 0:
                avg_price = recent_prices.mean()
                logger.warning(f"Using average price {avg_price} as last resort for index {index}")
                return float(avg_price)
            else:
                raise ValueError(f"Cannot find any valid price data around index {index}")
        except Exception as e:
            logger.error(f"Critical error getting price at index {index}: {str(e)}")
            raise ValueError(f"Invalid price data at index {index}")

    def _determine_trend(self, data: pd.DataFrame, swing_points: Dict[str, List[int]]) -> str:
        """Определение текущего тренда"""
        if len(swing_points["highs"]) < 2 or len(swing_points["lows"]) < 2:
            return "sideways"
        # Анализируем последние точки разворота
        last_highs = swing_points["highs"][-2:]
        last_lows = swing_points["lows"][-2:]
        # Проверяем последовательность максимумов и минимумов
        if last_highs[-1] > last_highs[-2] and last_lows[-1] > last_lows[-2]:
            return "uptrend"
        elif last_highs[-1] < last_highs[-2] and last_lows[-1] < last_lows[-2]:
            return "downtrend"
        else:
            return "sideways"

    def _find_potential_reversals(
        self,
        data: pd.DataFrame,
        rsi: pd.Series,
        macd: pd.Series,
        signal: pd.Series,
        swing_points: Dict[str, List[int]],
        volume_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Поиск потенциальных разворотов"""
        reversals = []
        for i in range(1, len(data)):  # Начинаем с 1 для безопасного доступа к i-1
            # Проверяем условия перепроданности/перекупленности
            rsi_value = rsi.iloc[i] 
            rsi_value = float(rsi_value) if rsi_value is not None and not pd.isna(rsi_value) else 50.0 
            is_oversold = rsi_value < self._config.rsi_oversold
            is_overbought = rsi_value > self._config.rsi_overbought
            
            # Проверяем пересечение MACD
            macd_prev = macd.iloc[i - 1] 
            macd_prev = float(macd_prev) if macd_prev is not None and not pd.isna(macd_prev) else 0.0
            macd_curr = macd.iloc[i]
            macd_curr = float(macd_curr) if macd_curr is not None and not pd.isna(macd_curr) else 0.0
            signal_prev = signal.iloc[i - 1]
            signal_prev = float(signal_prev) if signal_prev is not None and not pd.isna(signal_prev) else 0.0
            signal_curr = signal.iloc[i]
            signal_curr = float(signal_curr) if signal_curr is not None and not pd.isna(signal_curr) else 0.0
            
            macd_bullish = macd_prev < signal_prev and macd_curr > signal_curr
            macd_bearish = macd_prev > signal_prev and macd_curr < signal_curr
            
            # Проверяем объем
            volume_high = volume_analysis["is_high"].iloc[i] 
            volume_high = bool(volume_high) if volume_high is not None and not pd.isna(volume_high) else False 
            
            # Проверяем точки разворота
            is_swing_high = i in swing_points["highs"]
            is_swing_low = i in swing_points["lows"]
            
            # Формируем сигнал разворота
            if is_oversold and macd_bullish and volume_high and is_swing_low:
                reversals.append(
                    {
                        "index": i,
                        "type": "bullish",
                        "price": self._safe_get_price(data["close"], i), 
                        "strength": self._calculate_reversal_strength(
                            data, i, "bullish"
                        ),
                    }
                )
            elif is_overbought and macd_bearish and volume_high and is_swing_high:
                reversals.append(
                    {
                        "index": i,
                        "type": "bearish",
                        "price": self._safe_get_price(data["close"], i), 
                        "strength": self._calculate_reversal_strength(
                            data, i, "bearish"
                        ),
                    }
                )
        return reversals

    def _calculate_reversal_strength(
        self, data: pd.DataFrame, index: int, reversal_type: str
    ) -> float:
        """Расчет силы разворота"""
        try:
            strength = 0.0
            try:
                close_price = get_safe_price(data["close"], index, "close_price")
            except ValueError:
                return 0.0  # Не можем получить корректную цену
            
            # Анализируем свечи
            if reversal_type == "bullish":
                # Для бычьего разворота
                open_price = data["open"].iloc[index]
                high_price = data["high"].iloc[index]
                low_price = data["low"].iloc[index]
                
                open_price = float(open_price) if open_price is not None and not pd.isna(open_price) else close_price 
                high_price = float(high_price) if high_price is not None and not pd.isna(high_price) else close_price 
                low_price = float(low_price) if low_price is not None and not pd.isna(low_price) else close_price 
                
                body_size = close_price - open_price
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                
                # Учитываем размер тела и тени
                strength += abs(body_size) / close_price
                strength += lower_shadow / close_price
                strength -= upper_shadow / close_price
            else:
                # Для медвежьего разворота
                open_price = data["open"].iloc[index]
                high_price = data["high"].iloc[index]
                low_price = data["low"].iloc[index]
                
                open_price = float(open_price) if open_price is not None and not pd.isna(open_price) else close_price 
                high_price = float(high_price) if high_price is not None and not pd.isna(high_price) else close_price 
                low_price = float(low_price) if low_price is not None and not pd.isna(low_price) else close_price 
                
                body_size = open_price - close_price
                upper_shadow = high_price - max(open_price, close_price)
                lower_shadow = min(open_price, close_price) - low_price
                
                # Учитываем размер тела и тени
                strength += abs(body_size) / close_price
                strength += upper_shadow / close_price
                strength -= lower_shadow / close_price
            
            return max(0.0, strength)
        except Exception as e:
            logger.error(f"Error calculating reversal strength: {str(e)}")
            return 0.0

    def _generate_signals(
        self, data: pd.DataFrame, reversals: List[Dict[str, Any]], atr: pd.Series, trend: str
    ) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов"""
        signals = []
        for reversal in reversals:
            index = reversal["index"]
            # Пропускаем, если недостаточно данных для подтверждения
            if index < self._config.confirmation_candles:
                continue
            # Проверяем подтверждение разворота
            if not self._confirm_reversal(data, index, reversal["type"]):
                continue
            # Рассчитываем уровни входа и выхода
            try:
                entry_price = get_safe_price(data["close"], index, "entry_price")
            except ValueError:
                continue  # Пропускаем, если нет корректной цены
                
            atr_value = atr.iloc[index] 
            atr_value = float(atr_value) if atr_value is not None and not pd.isna(atr_value) else 0.0 
            
            if reversal["type"] == "bullish":
                # Используем Decimal для точных расчетов
                entry_decimal = to_trading_decimal(entry_price)
                atr_decimal = to_trading_decimal(atr_value)
                stop_multiplier = to_trading_decimal(self._config.stop_loss_atr_multiplier)
                take_multiplier = to_trading_decimal(self._config.take_profit_atr_multiplier)
                
                stop_distance = atr_decimal * stop_multiplier
                take_distance = atr_decimal * take_multiplier
                
                stop_loss = float(entry_decimal - stop_distance)
                take_profit = float(entry_decimal + take_distance)
                direction = "long"
            else:
                # Используем Decimal для точных расчетов (short позиция)
                stop_loss = float(entry_decimal + stop_distance)
                take_profit = float(entry_decimal - take_distance)
                direction = "short"
            
            signals.append(
                {
                    "timestamp": data.index[index],
                    "type": reversal["type"],
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "strength": reversal["strength"],
                    "atr": atr_value,
                    "trend": trend,
                }
            )
        return signals

    def _confirm_reversal(
        self, data: pd.DataFrame, index: int, reversal_type: str
    ) -> bool:
        """Подтверждение разворота"""
        try:
            # Проверяем последние свечи
            for i in range(index - self._config.confirmation_candles + 1, index + 1):
                if i < 0 or i >= len(data):
                    continue
                    
                close_price = data["close"].iloc[i]
                open_price = data["open"].iloc[i]
                
                close_price = float(close_price) if close_price is not None and not pd.isna(close_price) else 0.0 
                open_price = float(open_price) if open_price is not None and not pd.isna(open_price) else 0.0 
                
                if reversal_type == "bullish":
                    # Для бычьего разворота проверяем закрытие выше открытия
                    if close_price <= open_price:
                        return False
                else:
                    # Для медвежьего разворота проверяем закрытие ниже открытия
                    if close_price >= open_price:
                        return False
            return True
        except Exception as e:
            logger.error(f"Error confirming reversal: {str(e)}")
            return False

    def generate_signal(self, data: pd.DataFrame) -> Optional[BaseSignal]:
        """
        Генерация торгового сигнала
        Args:
            data: DataFrame с OHLCV данными
        Returns:
            Optional[BaseSignal]: Торговый сигнал или None
        """
        try:
            # Проверяем наличие необходимых данных
            min_periods = max(
                self._config.rsi_period,
                self._config.macd_slow,
                self._config.confirmation_candles,
            )
            if len(data) < min_periods:
                return None
            
            # Анализируем данные
            analysis = self.analyze(data)
            if not analysis:
                return None
            
            # Получаем последний сигнал
            signals = analysis.get("signals", [])
            if signals:
                signal_data = signals[-1]
                return BaseSignal(
                    direction=signal_data["direction"],
                    entry_price=signal_data["entry_price"],
                    stop_loss=signal_data["stop_loss"],
                    take_profit=signal_data["take_profit"],
                    volume=None,
                    confidence=signal_data.get("strength", 0.5),
                    timestamp=pd.Timestamp.now(), 
                    metadata={
                        "type": signal_data["type"],
                        "atr": signal_data["atr"],
                        "trend": signal_data["trend"],
                    },
                )
            return None
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
