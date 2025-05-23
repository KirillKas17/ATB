from typing import Dict, Optional

import pandas as pd
from loguru import logger

from strategies.base_strategy import BaseStrategy
from utils.indicators import (
    calculate_atr,
    calculate_bollinger_bands,
    calculate_liquidity_zones,
    calculate_macd,
    calculate_market_structure,
    calculate_obv,
    calculate_rsi,
    calculate_stochastic,
)


class SidewaysStrategy(BaseStrategy):
    """Базовый класс для стратегий флэта"""

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        stoch_k: int = 14,
        stoch_d: int = 3,
        obv_threshold: float = 1.5,
    ):
        """
        Инициализация стратегии.

        Args:
            bb_period: Период Bollinger Bands
            bb_std: Стандартное отклонение для BB
            rsi_period: Период RSI
            stoch_k: Период %K для Stochastic
            stoch_d: Период %D для Stochastic
            obv_threshold: Порог для OBV
        """
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.obv_threshold = obv_threshold

    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Расчет индикаторов"""
        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(
            data["close"], self.bb_period, self.bb_std
        )
        data["bb_upper"] = upper
        data["bb_middle"] = middle
        data["bb_lower"] = lower

        # RSI
        data["rsi"] = calculate_rsi(data["close"], self.rsi_period)

        # Stochastic
        k, d = calculate_stochastic(
            data["high"], data["low"], data["close"], self.stoch_k, self.stoch_d
        )
        data["stoch_k"] = k
        data["stoch_d"] = d

        # OBV
        data["obv"] = calculate_obv(data["close"], data["volume"])

        return data

    def _check_volume(self, data: pd.DataFrame) -> bool:
        """Проверка объема"""
        obv_change = data["obv"].iloc[-1] / data["obv"].iloc[-2]
        return obv_change > self.obv_threshold

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
            atr = calculate_atr(data, 14).iloc[-1]

            # Расчет волатильности
            volatility = data["close"].pct_change().rolling(20).std().iloc[-1]

            # Расчет структуры рынка
            support_resistance = calculate_market_structure(data)
            liquidity_zones = calculate_liquidity_zones(data)

            # Расчет импульса
            rsi = calculate_rsi(data["close"], 14).iloc[-1]
            macd, signal, hist = calculate_macd(data["close"])

            # Расчет диапазона
            high = data["high"].rolling(20).max()
            low = data["low"].rolling(20).min()
            range_size = (high - low) / low

            # Расчет объема
            volume_ma = data["volume"].rolling(20).mean()
            volume_ratio = data["volume"].iloc[-1] / volume_ma.iloc[-1]

            # Базовый стоп на основе ATR
            base_stop = atr * 1.5  # Меньший множитель для боковика

            # Корректировка на волатильность
            volatility_multiplier = 1 + volatility * 5  # Меньший множитель для боковика

            # Корректировка на диапазон
            range_multiplier = 1 + range_size.iloc[-1] * 2

            # Корректировка на объем
            volume_multiplier = 1 + (volume_ratio - 1) * 0.3

            # Корректировка на импульс
            momentum_multiplier = 1 + (abs(rsi - 50) / 50) * 0.3

            # Расчет финального множителя
            final_multiplier = (
                volatility_multiplier
                * range_multiplier
                * volume_multiplier
                * momentum_multiplier
            )

            # Расчет стопа
            stop_distance = base_stop * final_multiplier

            # Поиск ближайшего уровня поддержки/сопротивления
            if position_type == "long":
                # Для длинной позиции ищем ближайший уровень поддержки
                support_levels = [
                    level for level in support_resistance if level < entry_price
                ]
                if support_levels:
                    nearest_support = max(support_levels)
                    stop_distance = min(stop_distance, entry_price - nearest_support)

                # Проверка ликвидности
                liquidity_levels = [
                    level for level in liquidity_zones if level < entry_price
                ]
                if liquidity_levels:
                    nearest_liquidity = max(liquidity_levels)
                    stop_distance = min(stop_distance, entry_price - nearest_liquidity)

                return entry_price - stop_distance

            else:
                # Для короткой позиции ищем ближайший уровень сопротивления
                resistance_levels = [
                    level for level in support_resistance if level > entry_price
                ]
                if resistance_levels:
                    nearest_resistance = min(resistance_levels)
                    stop_distance = min(stop_distance, nearest_resistance - entry_price)

                # Проверка ликвидности
                liquidity_levels = [
                    level for level in liquidity_zones if level > entry_price
                ]
                if liquidity_levels:
                    nearest_liquidity = min(liquidity_levels)
                    stop_distance = min(stop_distance, nearest_liquidity - entry_price)

                return entry_price + stop_distance

        except Exception as e:
            logger.error(f"Error calculating stop loss: {str(e)}")
            return entry_price * 0.99 if position_type == "long" else entry_price * 1.01

    def _calculate_take_profit(
        self,
        data: pd.DataFrame,
        entry_price: float,
        stop_loss: float,
        position_type: str,
    ) -> float:
        """Расширенный расчет уровня тейк-профита.

        Args:
            data: DataFrame с рыночными данными
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            position_type: Тип позиции ('long' или 'short')

        Returns:
            float: Уровень тейк-профита
        """
        try:
            # Расчет ATR
            float(calculate_atr(data, 14).iloc[-1])

            # Расчет волатильности
            volatility = float(data["close"].pct_change().rolling(20).std().iloc[-1])

            # Расчет структуры рынка
            support_resistance = [
                float(level) for level in calculate_market_structure(data)
            ]
            liquidity_zones = [
                float(level) for level in calculate_liquidity_zones(data)
            ]

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
            base_rr = 1.5  # Меньшее соотношение для боковика

            # Корректировка на волатильность
            volatility_multiplier = 1 + volatility * 3  # Меньший множитель для боковика

            # Корректировка на диапазон
            range_multiplier = 1 + range_size

            # Корректировка на объем
            volume_multiplier = 1 + (volume_ratio - 1) * 0.2

            # Корректировка на импульс
            momentum_multiplier = 1 + (abs(rsi - 50) / 50) * 0.2

            # Расчет финального множителя
            final_multiplier = float(
                volatility_multiplier
                * range_multiplier
                * volume_multiplier
                * momentum_multiplier
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
                    float(level)
                    for level in liquidity_zones
                    if float(level) > float(entry_price)
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
                    float(level)
                    for level in liquidity_zones
                    if float(level) < float(entry_price)
                ]
                if liquidity_levels:
                    nearest_liquidity = max(liquidity_levels)
                    take_profit_distance = min(
                        take_profit_distance, float(entry_price - nearest_liquidity)
                    )

                return float(entry_price - take_profit_distance)

        except Exception as e:
            logger.error(f"Error calculating take profit: {str(e)}")
            return float(
                entry_price * 1.01 if position_type == "long" else entry_price * 0.99
            )


def sideways_strategy_bb_rsi(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе Bollinger Bands и RSI.

    Args:
        data: Рыночные данные

    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = SidewaysStrategy()
        data = strategy._calculate_indicators(data)

        # Проверка объема
        if not strategy._check_volume(data):
            return None

        # Проверка RSI
        rsi = data["rsi"].iloc[-1]
        rsi_prev = data["rsi"].iloc[-2]

        # Проверка отскока от границ BB
        price = data["close"].iloc[-1]
        upper = data["bb_upper"].iloc[-1]
        lower = data["bb_lower"].iloc[-1]

        # Генерация сигнала
        if price <= lower and rsi < 30 and rsi > rsi_prev:
            side = "buy"
        elif price >= upper and rsi > 70 and rsi < rsi_prev:
            side = "sell"
        else:
            return None

        # Расчет стоп-лосса и тейк-профита
        stop_loss = strategy._calculate_stop_loss(data, price, side)
        take_profit = strategy._calculate_take_profit(data, price, stop_loss, side)

        return {
            "side": side,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }

    except Exception as e:
        logger.error(f"Error in BB-RSI strategy: {str(e)}")
        return None


def sideways_strategy_stoch_obv(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе Stochastic и OBV.

    Args:
        data: Рыночные данные

    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = SidewaysStrategy()
        data = strategy._calculate_indicators(data)

        # Проверка объема
        if not strategy._check_volume(data):
            return None

        # Проверка Stochastic
        k = data["stoch_k"].iloc[-1]
        d = data["stoch_d"].iloc[-1]
        k_prev = data["stoch_k"].iloc[-2]
        d_prev = data["stoch_d"].iloc[-2]

        # Проверка кроссовера Stochastic
        stoch_cross_up = k_prev < d_prev and k > d
        stoch_cross_down = k_prev > d_prev and k < d

        # Проверка перекупленности/перепроданности
        oversold = k < 20 and d < 20
        overbought = k > 80 and d > 80

        # Генерация сигнала
        if stoch_cross_up and oversold:
            side = "buy"
        elif stoch_cross_down and overbought:
            side = "sell"
        else:
            return None

        # Расчет стоп-лосса и тейк-профита
        stop_loss = strategy._calculate_stop_loss(data, data["close"].iloc[-1], side)
        take_profit = strategy._calculate_take_profit(
            data, data["close"].iloc[-1], stop_loss, side
        )

        return {
            "side": side,
            "entry_price": data["close"].iloc[-1],
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }

    except Exception as e:
        logger.error(f"Error in Stochastic-OBV strategy: {str(e)}")
        return None
