from typing import Dict, Optional, Tuple, Any, List

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy
from utils.indicators import (
    calculate_adx,
    calculate_atr,
    calculate_ema,
    calculate_liquidity_zones,
    calculate_macd,
    calculate_market_structure,
    calculate_rsi,
    detect_impulse_candle,
    detect_inner_bar,
)


class TrendStrategy(BaseStrategy):
    """Базовый класс для трендовых стратегий"""

    def __init__(
        self,
        ema_fast: int = 20,
        ema_medium: int = 50,
        ema_slow: int = 200,
        adx_threshold: float = 25.0,
        atr_period: int = 14,
        risk_reward: float = 2.0,
    ):
        """
        Инициализация стратегии.

        Args:
            ema_fast: Период быстрой EMA
            ema_medium: Период средней EMA
            ema_slow: Период медленной EMA
            adx_threshold: Порог ADX для подтверждения тренда
            atr_period: Период ATR
            risk_reward: Соотношение риск/доходность
        """
        super().__init__()
        self.ema_fast = ema_fast
        self.ema_medium = ema_medium
        self.ema_slow = ema_slow
        self.adx_threshold = adx_threshold
        self.atr_period = atr_period
        self.risk_reward = risk_reward
        self.config = {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9}

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов"""
        # EMA
        data["ema_fast"] = calculate_ema(data["close"], self.ema_fast)
        data["ema_medium"] = calculate_ema(data["close"], self.ema_medium)
        data["ema_slow"] = calculate_ema(data["close"], self.ema_slow)

        # MACD
        macd, signal, hist = calculate_macd(data["close"])
        data["macd"] = macd
        data["macd_signal"] = signal
        data["macd_hist"] = hist

        # ADX
        data["adx"] = calculate_adx(data)

        # ATR
        data["atr"] = calculate_atr(data, self.atr_period)

        return data

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Расчет MACD"""
        exp1 = prices.ewm(span=self.config["macd_fast"], adjust=False).mean()
        exp2 = prices.ewm(span=self.config["macd_slow"], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.config["macd_signal"], adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def _check_trend_strength(self, data: pd.DataFrame) -> bool:
        """Проверка силы тренда"""
        try:
            macd, signal, hist = self._calculate_macd(data["close"])
            # Преобразуем в список для безопасной итерации
            macd_values = list(macd)
            signal_values = list(signal)

            # Проверяем пересечение MACD
            if len(macd_values) < 2 or len(signal_values) < 2:
                return False

            return (
                macd_values[-2] < signal_values[-2] and macd_values[-1] > signal_values[-1]
            ) or (macd_values[-2] > signal_values[-2] and macd_values[-1] < signal_values[-1])
        except Exception as e:
            logger.error(f"Error checking trend strength: {str(e)}")
            return False

    def _calculate_stop_loss(
        self, data: pd.DataFrame, entry_price: float, position_type: str
    ) -> float:
        """Расширенный расчет уровня стоп-лосса.

        Args:
            data: DataFrame с рыночными данными
            entry_price: Цена входа
            position_type: Тип позиции ('long' или 'short')

        Returns:
            float: Уровень стоп-лосса
        """
        try:
            # Расчет ATR
            atr = float(calculate_atr(data, self.atr_period).iloc[-1])

            # Расчет волатильности
            volatility = float(data["close"].pct_change().rolling(20).std().iloc[-1])

            # Расчет структуры рынка
            support_resistance = [float(level) for level in calculate_market_structure(data)]
            liquidity_zones = [float(level) for level in calculate_liquidity_zones(data)]

            # Расчет импульса
            rsi = float(calculate_rsi(data["close"], 14).iloc[-1])
            macd, signal, hist = calculate_macd(data["close"])

            # Расчет тренда
            ema_20 = calculate_ema(data["close"], 20)
            ema_50 = calculate_ema(data["close"], 50)
            trend_strength = float(abs(ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1])

            # Расчет объема
            volume_ma = float(data["volume"].rolling(20).mean().iloc[-1])
            volume_ratio = float(data["volume"].iloc[-1] / volume_ma)

            # Базовый стоп на основе ATR
            base_stop = atr * 2

            # Корректировка на волатильность
            volatility_multiplier = 1 + volatility * 10

            # Корректировка на силу тренда
            trend_multiplier = 1 + trend_strength * 5

            # Корректировка на объем
            volume_multiplier = 1 + (volume_ratio - 1) * 0.5

            # Корректировка на импульс
            momentum_multiplier = 1 + (abs(rsi - 50) / 50) * 0.5

            # Расчет финального множителя
            final_multiplier = float(
                volatility_multiplier * trend_multiplier * volume_multiplier * momentum_multiplier
            )

            # Расчет стопа
            stop_distance = base_stop * final_multiplier

            # Поиск ближайшего уровня поддержки/сопротивления
            if position_type == "long":
                # Для длинной позиции ищем ближайший уровень поддержки
                support_levels = [level for level in support_resistance if level < entry_price]
                if support_levels:
                    nearest_support = max(support_levels)
                    stop_distance = min(stop_distance, entry_price - nearest_support)

                # Проверка ликвидности
                liquidity_levels = [level for level in liquidity_zones if level < entry_price]
                if liquidity_levels:
                    nearest_liquidity = max(liquidity_levels)
                    stop_distance = min(stop_distance, entry_price - nearest_liquidity)

                return float(entry_price - stop_distance)

            else:
                # Для короткой позиции ищем ближайший уровень сопротивления
                resistance_levels = [level for level in support_resistance if level > entry_price]
                if resistance_levels:
                    nearest_resistance = min(resistance_levels)
                    stop_distance = min(stop_distance, nearest_resistance - entry_price)

                # Проверка ликвидности
                liquidity_levels = [level for level in liquidity_zones if level > entry_price]
                if liquidity_levels:
                    nearest_liquidity = min(liquidity_levels)
                    stop_distance = min(stop_distance, nearest_liquidity - entry_price)

                return float(entry_price + stop_distance)

        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return float(entry_price * 0.99 if position_type == "long" else entry_price * 1.01)

    def _calculate_take_profit(
        self, data: pd.DataFrame, entry_price: float, stop_loss: float, position_type: str
    ) -> float:
        """Расчет тейк-профита"""
        try:
            # Расчет ATR
            atr = float(calculate_atr(data, self.atr_period).iloc[-1])

            # Расчет волатильности
            volatility = float(data["close"].pct_change().rolling(20).std().iloc[-1])

            # Расчет структуры рынка
            support_resistance = [float(level) for level in calculate_market_structure(data)]
            liquidity_zones = [float(level) for level in calculate_liquidity_zones(data)]

            # Расчет импульса
            rsi = float(calculate_rsi(data["close"], 14).iloc[-1])
            macd, signal, hist = calculate_macd(data["close"])

            # Расчет диапазона
            high = float(data["high"].rolling(20).max().iloc[-1])
            low = float(data["low"].rolling(20).min().iloc[-1])
            range_size = float((high - low) / low)

            # Расчет объема
            volume_ma = float(data["volume"].rolling(20).mean().iloc[-1])
            volume_ratio = float(data["volume"].iloc[-1] / volume_ma)

            # Расчет риска
            risk = float(abs(entry_price - stop_loss))

            # Базовое соотношение риск/прибыль
            base_rr = 2.0  # Большее соотношение для тренда

            # Корректировка на волатильность
            volatility_multiplier = 1 + volatility * 5  # Больший множитель для тренда

            # Корректировка на диапазон
            range_multiplier = 1 + range_size * 2

            # Корректировка на объем
            volume_multiplier = 1 + (volume_ratio - 1) * 0.3

            # Корректировка на импульс
            momentum_multiplier = 1 + (abs(rsi - 50) / 50) * 0.3

            # Расчет финального множителя
            final_multiplier = float(
                volatility_multiplier * range_multiplier * volume_multiplier * momentum_multiplier
            )

            # Расчет тейка
            take_profit_distance = float(risk * base_rr * final_multiplier)

            # Поиск ближайшего уровня поддержки/сопротивления
            if position_type == "long":
                # Для длинной позиции ищем ближайший уровень сопротивления
                resistance_levels = [
                    float(level)
                    for level in support_resistance
                    if float(level) > float(entry_price)
                ]
                if resistance_levels:
                    nearest_resistance = min(resistance_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(nearest_resistance - entry_price)
                    )

                # Проверка ликвидности
                liquidity_levels = [
                    float(level) for level in liquidity_zones if float(level) > float(entry_price)
                ]
                if liquidity_levels:
                    nearest_liquidity = min(liquidity_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(nearest_liquidity - entry_price)
                    )

                return float(entry_price + take_profit_distance)

            else:
                # Для короткой позиции ищем ближайший уровень поддержки
                support_levels = [
                    float(level)
                    for level in support_resistance
                    if float(level) < float(entry_price)
                ]
                if support_levels:
                    nearest_support = max(support_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(entry_price - nearest_support)
                    )

                # Проверка ликвидности
                liquidity_levels = [
                    float(level) for level in liquidity_zones if float(level) < float(entry_price)
                ]
                if liquidity_levels:
                    nearest_liquidity = max(liquidity_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(entry_price - nearest_liquidity)
                    )

                return float(entry_price - take_profit_distance)

        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return float(entry_price * 1.02 if position_type == "long" else entry_price * 0.98)

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ рыночных данных для определения тренда

        Args:
            data: Dict с OHLCV данными

        Returns:
            Dict с результатами анализа
        """
        try:
            # Преобразуем данные в DataFrame
            df = pd.DataFrame(data)

            # Проверяем наличие необходимых данных
            required_columns = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                raise ValueError("Missing required columns in data")

            # Рассчитываем индикаторы
            df = self._calculate_indicators(df)

            # Определяем тренд
            trend = "up" if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1] else "down"

            # Анализируем силу тренда
            trend_strength = self._check_trend_strength(df)

            # Ищем точки входа
            entry_points = []
            if trend_strength:
                if trend == "up":
                    entry_points.append(
                        {
                            "type": "long",
                            "price": float(df["close"].iloc[-1]),
                            "stop_loss": self._calculate_stop_loss(
                                df, float(df["close"].iloc[-1]), "long"
                            ),
                            "take_profit": self._calculate_take_profit(
                                df, float(df["close"].iloc[-1]), float(df["close"].iloc[-1]), "long"
                            ),
                        }
                    )
                else:
                    entry_points.append(
                        {
                            "type": "short",
                            "price": float(df["close"].iloc[-1]),
                            "stop_loss": self._calculate_stop_loss(
                                df, float(df["close"].iloc[-1]), "short"
                            ),
                            "take_profit": self._calculate_take_profit(
                                df,
                                float(df["close"].iloc[-1]),
                                float(df["close"].iloc[-1]),
                                "short",
                            ),
                        }
                    )

            return {
                "trend": trend,
                "trend_strength": trend_strength,
                "entry_points": entry_points,
                "indicators": {
                    "ema_fast": float(df["ema_fast"].iloc[-1]),
                    "ema_slow": float(df["ema_slow"].iloc[-1]),
                    "adx": float(df["adx"].iloc[-1]),
                    "atr": float(df["atr"].iloc[-1]),
                },
            }

        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
            return {}


def trend_strategy_ema_macd(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе EMA и MACD.

    Args:
        data: Рыночные данные

    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = TrendStrategy()
        data = strategy._calculate_indicators(data)

        # Проверка силы тренда
        if not strategy._check_trend_strength(data):
            return None

        # Определение тренда по EMA
        ema_trend = (
            data["ema_fast"].iloc[-1] > data["ema_medium"].iloc[-1] > data["ema_slow"].iloc[-1]
        )

        # Проверка кроссовера MACD
        macd_cross_up = (
            data["macd"].iloc[-2] < data["macd_signal"].iloc[-2]
            and data["macd"].iloc[-1] > data["macd_signal"].iloc[-1]
        )
        macd_cross_down = (
            data["macd"].iloc[-2] > data["macd_signal"].iloc[-2]
            and data["macd"].iloc[-1] < data["macd_signal"].iloc[-1]
        )

        # Генерация сигнала
        if ema_trend and macd_cross_up:
            side = "buy"
        elif not ema_trend and macd_cross_down:
            side = "sell"
        else:
            return None

        # Расчет стоп-лосса и тейк-профита
        stop_loss = strategy._calculate_stop_loss(data, data["close"].iloc[-1], side)
        take_profit = strategy._calculate_take_profit(data, data["close"].iloc[-1], stop_loss, side)

        return {
            "side": side,
            "entry_price": data["close"].iloc[-1],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }

    except Exception as e:
        logger.error(f"Error in EMA-MACD strategy: {str(e)}")
        return None


def trend_strategy_price_action(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе Price Action.

    Args:
        data: Рыночные данные

    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = TrendStrategy()
        data = strategy._calculate_indicators(data)

        # Проверка силы тренда
        if not strategy._check_trend_strength(data):
            return None

        # Определение тренда
        trend = "up" if data["ema_fast"].iloc[-1] > data["ema_slow"].iloc[-1] else "down"

        # Проверка паттернов
        last_candle = data.iloc[-1]
        prev_candle = data.iloc[-2]

        # Проверка импульсной свечи
        is_impulse = detect_impulse_candle(last_candle)

        # Проверка внутреннего бара
        is_inner = detect_inner_bar(last_candle, prev_candle)

        # Проверка дивергенции MACD
        macd_divergence = (
            trend == "up"
            and data["close"].iloc[-1] > data["close"].iloc[-2]
            and data["macd"].iloc[-1] < data["macd"].iloc[-2]
        ) or (
            trend == "down"
            and data["close"].iloc[-1] < data["close"].iloc[-2]
            and data["macd"].iloc[-1] > data["macd"].iloc[-2]
        )

        # Генерация сигнала
        if trend == "up" and is_impulse and not is_inner and macd_divergence:
            side = "buy"
        elif trend == "down" and is_impulse and not is_inner and macd_divergence:
            side = "sell"
        else:
            return None

        # Расчет стоп-лосса и тейк-профита
        stop_loss = strategy._calculate_stop_loss(data, data["close"].iloc[-1], side)
        take_profit = strategy._calculate_take_profit(data, data["close"].iloc[-1], stop_loss, side)

        return {
            "side": side,
            "entry_price": data["close"].iloc[-1],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }

    except Exception as e:
        logger.error(f"Error in Price Action strategy: {str(e)}")
        return None
