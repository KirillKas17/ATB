from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy, StrategyConfig


@dataclass
class ReversalConfig(StrategyConfig):
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
    partial_close_levels: List[float] = None
    risk_per_trade: float = 0.01
    max_trades: Optional[int] = None
    log_dir: str = "logs"


class ReversalStrategy(BaseStrategy):
    """Стратегия разворота тренда"""

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.config = ReversalConfig(**(config or {}))
        self._setup_logger()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/reversal_strategy_{{time}}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def analyze(self, data: pd.DataFrame) -> Dict:
        """
        Анализ рыночных данных для поиска разворотов тренда

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с результатами анализа
        """
        try:
            # Проверяем наличие необходимых данных
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Missing required columns in data")

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

    def _calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Расчет RSI"""
        period = period or self.config.rsi_period
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD"""
        exp1 = prices.ewm(span=self.config.macd_fast, adjust=False).mean()
        exp2 = prices.ewm(span=self.config.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.config.macd_signal, adjust=False).mean()
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

    def _analyze_volume(self, volume: pd.Series) -> Dict:
        """Анализ объемов"""
        avg_volume = volume.rolling(window=20).mean()
        volume_ratio = volume / avg_volume
        return {
            "average": avg_volume,
            "ratio": volume_ratio,
            "is_high": volume_ratio > self.config.volume_threshold,
        }

    def _determine_trend(self, data: pd.DataFrame, swing_points: Dict) -> str:
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
        swing_points: Dict,
        volume_analysis: Dict,
    ) -> List[Dict]:
        """Поиск потенциальных разворотов"""
        reversals = []

        for i in range(len(data)):
            # Проверяем условия перепроданности/перекупленности
            is_oversold = rsi.iloc[i] < self.config.rsi_oversold
            is_overbought = rsi.iloc[i] > self.config.rsi_overbought

            # Проверяем пересечение MACD
            macd_cross_up = (
                macd.iloc[i - 1] < signal.iloc[i - 1] and macd.iloc[i] > signal.iloc[i]
            )
            macd_cross_down = (
                macd.iloc[i - 1] > signal.iloc[i - 1] and macd.iloc[i] < signal.iloc[i]
            )

            # Проверяем объем
            high_volume = volume_analysis["is_high"].iloc[i]

            # Проверяем точки разворота
            is_swing_high = i in swing_points["highs"]
            is_swing_low = i in swing_points["lows"]

            # Формируем сигнал разворота
            if is_oversold and macd_cross_up and high_volume and is_swing_low:
                reversals.append(
                    {
                        "index": i,
                        "type": "bullish",
                        "price": data["close"].iloc[i],
                        "strength": self._calculate_reversal_strength(
                            data, i, "bullish"
                        ),
                    }
                )
            elif is_overbought and macd_cross_down and high_volume and is_swing_high:
                reversals.append(
                    {
                        "index": i,
                        "type": "bearish",
                        "price": data["close"].iloc[i],
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
        strength = 0.0

        # Анализируем свечи
        if reversal_type == "bullish":
            # Для бычьего разворота
            body_size = data["close"].iloc[index] - data["open"].iloc[index]
            upper_shadow = data["high"].iloc[index] - max(
                data["open"].iloc[index], data["close"].iloc[index]
            )
            lower_shadow = (
                min(data["open"].iloc[index], data["close"].iloc[index])
                - data["low"].iloc[index]
            )

            # Учитываем размер тела и тени
            strength += abs(body_size) / data["close"].iloc[index]
            strength += lower_shadow / data["close"].iloc[index]
            strength -= upper_shadow / data["close"].iloc[index]

        else:
            # Для медвежьего разворота
            body_size = data["open"].iloc[index] - data["close"].iloc[index]
            upper_shadow = data["high"].iloc[index] - max(
                data["open"].iloc[index], data["close"].iloc[index]
            )
            lower_shadow = (
                min(data["open"].iloc[index], data["close"].iloc[index])
                - data["low"].iloc[index]
            )

            # Учитываем размер тела и тени
            strength += abs(body_size) / data["close"].iloc[index]
            strength += upper_shadow / data["close"].iloc[index]
            strength -= lower_shadow / data["close"].iloc[index]

        return strength

    def _generate_signals(
        self, data: pd.DataFrame, reversals: List[Dict], atr: pd.Series, trend: str
    ) -> List[Dict]:
        """Генерация торговых сигналов"""
        signals = []

        for reversal in reversals:
            index = reversal["index"]

            # Пропускаем, если недостаточно данных для подтверждения
            if index < self.config.confirmation_candles:
                continue

            # Проверяем подтверждение разворота
            if not self._confirm_reversal(data, index, reversal["type"]):
                continue

            # Рассчитываем уровни входа и выхода
            entry_price = data["close"].iloc[index]
            atr_value = atr.iloc[index]

            if reversal["type"] == "bullish":
                stop_loss = entry_price - (
                    atr_value * self.config.stop_loss_atr_multiplier
                )
                take_profit = entry_price + (
                    atr_value * self.config.take_profit_atr_multiplier
                )
                direction = "long"
            else:
                stop_loss = entry_price + (
                    atr_value * self.config.stop_loss_atr_multiplier
                )
                take_profit = entry_price - (
                    atr_value * self.config.take_profit_atr_multiplier
                )
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
        # Проверяем последние свечи
        for i in range(index - self.config.confirmation_candles + 1, index + 1):
            if reversal_type == "bullish":
                # Для бычьего разворота проверяем закрытие выше открытия
                if data["close"].iloc[i] <= data["open"].iloc[i]:
                    return False
            else:
                # Для медвежьего разворота проверяем закрытие ниже открытия
                if data["close"].iloc[i] >= data["open"].iloc[i]:
                    return False

        return True

    def generate_signal(self, data: pd.DataFrame) -> Optional[Dict]:
        """
        Генерация торгового сигнала

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с сигналом или None
        """
        try:
            # Проверяем наличие необходимых данных
            if len(data) < max(
                self.config.rsi_period,
                self.config.macd_slow,
                self.config.confirmation_candles,
            ):
                return None

            # Анализируем данные
            analysis = self.analyze(data)

            # Получаем последний сигнал
            if analysis.get("signals"):
                return analysis["signals"][-1]

            return None

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
