from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
from loguru import logger

from strategies.base_strategy import BaseStrategy, Signal
from utils.indicators import (
    calculate_fractals,
    calculate_order_book_imbalance,
    calculate_volume_delta,
    calculate_vwap,
    detect_volume_spike,
)


class ManipulationStrategy(BaseStrategy):
    """Базовый класс для стратегий манипуляций"""

    def __init__(
        self,
        volume_delta_threshold: float = 1.5,
        fractal_period: int = 5,
        vwap_period: int = 20,
        imbalance_threshold: float = 0.7,
        log_dir: str = "logs/manipulation",
    ):
        """
        Инициализация стратегии.

        Args:
            volume_delta_threshold: Порог для Volume Delta
            fractal_period: Период для фракталов
            vwap_period: Период VWAP
            imbalance_threshold: Порог для дисбаланса стакана
            log_dir: Директория для логов
        """
        super().__init__({"log_dir": log_dir})
        self.volume_delta_threshold = volume_delta_threshold
        self.fractal_period = fractal_period
        self.vwap_period = vwap_period
        self.imbalance_threshold = imbalance_threshold

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Расчет индикаторов"""
        # Volume Delta
        data["volume_delta"] = calculate_volume_delta(data)

        # Order Book Imbalance (если доступен)
        if "order_book" in data.columns:
            data["imbalance"] = calculate_order_book_imbalance(data["order_book"])

        # Fractals
        upper_fractals, lower_fractals = calculate_fractals(
            data["high"], data["low"], self.fractal_period
        )
        data["upper_fractals"] = upper_fractals
        data["lower_fractals"] = lower_fractals

        # VWAP
        data["vwap"] = calculate_vwap(data)

        return data

    def _check_volume_spike(self, data: pd.DataFrame) -> bool:
        """Проверка всплеска объема"""
        return detect_volume_spike(data["volume"], self.volume_delta_threshold)

    def _check_imbalance(self, data: pd.DataFrame) -> bool:
        """Проверка дисбаланса стакана"""
        if "imbalance" in data.columns:
            return abs(data["imbalance"].iloc[-1]) > self.imbalance_threshold
        return False

    def _calculate_stop_loss(self, data: pd.DataFrame, side: str) -> float:
        """Расчет стоп-лосса"""
        if side == "buy":
            return float(data["lower_fractals"].iloc[-1])
        else:
            return float(data["upper_fractals"].iloc[-1])

    def _calculate_take_profit(self, data: pd.DataFrame, side: str) -> float:
        """Расчет тейк-профита"""
        if side == "buy":
            return float(data["vwap"].iloc[-1])
        else:
            return float(data["vwap"].iloc[-1])

    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Анализ рыночных данных для выявления манипуляций

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

            # Анализируем объемы
            volume_analysis = self.analyze_volume_profile(df)

            # Анализируем спред
            spread_analysis = self._analyze_spread(df)

            # Анализируем паттерны
            patterns = self._analyze_patterns(df)

            # Анализируем аномалии
            anomalies = self._analyze_anomalies(df)

            return {
                "volume_analysis": volume_analysis,
                "spread_analysis": spread_analysis,
                "patterns": patterns,
                "anomalies": anomalies,
            }

        except Exception as e:
            logger.error(f"Error in analyze: {str(e)}")
            return {}

    def _analyze_spread(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ спреда"""
        try:
            spread = (data["high"] - data["low"]) / data["low"]
            spread_ma = spread.rolling(20).mean()
            spread_std = spread.rolling(20).std()

            return {
                "current_spread": float(spread.iloc[-1]),
                "spread_ma": float(spread_ma.iloc[-1]),
                "spread_std": float(spread_std.iloc[-1]),
                "is_anomaly": bool(spread.iloc[-1] > spread_ma.iloc[-1] + 2 * spread_std.iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return {"current_spread": 0.0, "spread_ma": 0.0, "spread_std": 0.0, "is_anomaly": False}

    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ паттернов"""
        try:
            # Расчет базовых метрик
            body = data["close"] - data["open"]
            upper_shadow = data["high"] - data[["open", "close"]].max(axis=1)
            lower_shadow = data[["open", "close"]].min(axis=1) - data["low"]

            # Определение паттернов
            doji = abs(body) <= 0.1 * (data["high"] - data["low"])
            hammer = (lower_shadow > 2 * abs(body)) & (upper_shadow < abs(body))
            shooting_star = (upper_shadow > 2 * abs(body)) & (lower_shadow < abs(body))

            return {
                "doji": bool(doji.iloc[-1]),
                "hammer": bool(hammer.iloc[-1]),
                "shooting_star": bool(shooting_star.iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error analyzing patterns: {str(e)}")
            return {"doji": False, "hammer": False, "shooting_star": False}

    def _analyze_anomalies(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ аномалий"""
        try:
            # Расчет волатильности
            volatility = data["close"].pct_change().rolling(20).std()

            # Расчет объема
            volume_ma = data["volume"].rolling(20).mean()
            volume_std = data["volume"].rolling(20).std()

            # Определение аномалий
            price_anomaly = abs(data["close"].pct_change().iloc[-1]) > 2 * volatility.iloc[-1]
            volume_anomaly = data["volume"].iloc[-1] > volume_ma.iloc[-1] + 2 * volume_std.iloc[-1]

            return {
                "price_anomaly": bool(price_anomaly),
                "volume_anomaly": bool(volume_anomaly),
                "volatility": float(volatility.iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error analyzing anomalies: {str(e)}")
            return {"price_anomaly": False, "volume_anomaly": False, "volatility": 0.0}

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

            # Проверка всплеска объема
            if not analysis["volume_analysis"]:
                return None

            # Проверка дисбаланса стакана
            if not analysis["spread_analysis"]:
                return None

            # Проверка фракталов
            price = float(data["close"].iloc[-1])
            upper_fractal = float(data["upper_fractals"].iloc[-1])
            lower_fractal = float(data["lower_fractals"].iloc[-1])

            # Проверка Volume Delta
            volume_delta = float(data["volume_delta"].iloc[-1])

            # Генерация сигнала
            if price < lower_fractal and volume_delta > self.volume_delta_threshold:
                side = "buy"  # Захват стопов на продажу
            elif price > upper_fractal and volume_delta < -self.volume_delta_threshold:
                side = "sell"  # Захват стопов на покупку
            else:
                return None

            # Расчет стоп-лосса и тейк-профита
            stop_loss = self._calculate_stop_loss(data, side)
            take_profit = self._calculate_take_profit(data, side)

            return {
                "action": side,
                "entry_price": price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "amount": 1.0,  # Фиксированный размер позиции
                "confidence": 0.8,  # Высокая уверенность в сигнале
            }

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def detect_manipulation(self, data: pd.DataFrame) -> bool:
        """
        Определение манипуляции на рынке.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            bool: True если обнаружена манипуляция
        """
        try:
            # Проверка всплеска объема
            volume_spike = self._check_volume_spike(data)

            # Проверка дисбаланса стакана
            imbalance = self._check_imbalance(data)

            # Проверка фракталов
            data = self._calculate_indicators(data)
            upper_fractal = float(data["upper_fractals"].iloc[-1])
            lower_fractal = float(data["lower_fractals"].iloc[-1])
            price = float(data["close"].iloc[-1])

            # Проверка Volume Delta
            volume_delta = float(data["volume_delta"].iloc[-1])

            return (
                volume_spike
                and imbalance
                and (
                    (price < lower_fractal and volume_delta > self.volume_delta_threshold)
                    or (price > upper_fractal and volume_delta < -self.volume_delta_threshold)
                )
            )

        except Exception as e:
            logger.error(f"Error detecting manipulation: {str(e)}")
            return False

    def analyze_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Анализ профиля объема.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            Dict с результатами анализа
        """
        try:
            # Расчет Volume Delta
            data = self._calculate_indicators(data)
            volume_delta = data["volume_delta"]

            # Анализ дисбаланса
            imbalance = self._check_imbalance(data)

            # Анализ тренда объема
            volume_ma = data["volume"].rolling(window=20).mean()
            volume_trend = "up" if data["volume"].iloc[-1] > volume_ma.iloc[-1] else "down"

            # Поиск аномалий
            volume_std = data["volume"].rolling(window=20).std()
            volume_anomalies = data["volume"] > (volume_ma + 2 * volume_std)

            # Преобразуем в список для безопасной итерации
            volume_anomalies_list = list(volume_anomalies)

            return {
                "volume_imbalance": bool(imbalance),
                "volume_trend": volume_trend,
                "volume_anomalies": int(sum(volume_anomalies_list)),
                "volume_delta": float(volume_delta.iloc[-1]),
            }

        except Exception as e:
            logger.error(f"Error analyzing volume profile: {str(e)}")
            return {
                "volume_imbalance": False,
                "volume_trend": "unknown",
                "volume_anomalies": 0,
                "volume_delta": 0.0,
            }

    def detect_pump_and_dump(self, data: pd.DataFrame) -> bool:
        """
        Определение памп-энд-дампа.

        Args:
            data: DataFrame с OHLCV данными

        Returns:
            bool: True если обнаружен памп-энд-дамп
        """
        try:
            # Расчет индикаторов
            data = self._calculate_indicators(data)

            # Проверка всплеска объема
            volume_spike = self._check_volume_spike(data)

            # Проверка резкого роста цены
            price_change = float(
                (data["close"].iloc[-1] - data["close"].iloc[-20]) / data["close"].iloc[-20]
            )
            sharp_price_increase = price_change > 0.1  # 10% рост

            # Проверка последующего падения
            price_drop = float(
                (data["close"].iloc[-1] - data["close"].iloc[-5]) / data["close"].iloc[-5]
            )
            sharp_price_drop = price_drop < -0.05  # 5% падение

            return volume_spike and sharp_price_increase and sharp_price_drop

        except Exception as e:
            logger.error(f"Error detecting pump and dump: {str(e)}")
            return False


def manipulation_strategy_stop_hunt(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе захвата ликвидности.

    Args:
        data: Рыночные данные

    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = ManipulationStrategy()
        data = strategy._calculate_indicators(data)

        # Проверка всплеска объема
        if not strategy._check_volume_spike(data):
            return None

        # Проверка дисбаланса стакана
        if not strategy._check_imbalance(data):
            return None

        # Проверка фракталов
        price = data["close"].iloc[-1]
        upper_fractal = data["upper_fractals"].iloc[-1]
        lower_fractal = data["lower_fractals"].iloc[-1]

        # Проверка Volume Delta
        volume_delta = data["volume_delta"].iloc[-1]

        # Генерация сигнала
        if price < lower_fractal and volume_delta > strategy.volume_delta_threshold:
            side = "buy"  # Захват стопов на продажу
        elif price > upper_fractal and volume_delta < -strategy.volume_delta_threshold:
            side = "sell"  # Захват стопов на покупку
        else:
            return None

        # Расчет стоп-лосса и тейк-профита
        stop_loss = strategy._calculate_stop_loss(data, side)
        take_profit = strategy._calculate_take_profit(data, side)

        return {
            "side": side,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }

    except Exception as e:
        logger.error(f"Error in stop hunt strategy: {str(e)}")
        return None


def manipulation_strategy_fake_breakout(data: pd.DataFrame) -> Optional[Dict]:
    """
    Стратегия на основе ложных пробоев.

    Args:
        data: Рыночные данные

    Returns:
        Optional[Dict]: Сигнал на вход
    """
    try:
        strategy = ManipulationStrategy()
        data = strategy._calculate_indicators(data)

        # Проверка всплеска объема
        if not strategy._check_volume_spike(data):
            return None

        # Проверка дисбаланса стакана
        if not strategy._check_imbalance(data):
            return None

        # Проверка ложного пробоя
        price = data["close"].iloc[-1]
        vwap = data["vwap"].iloc[-1]
        upper_fractal = data["upper_fractals"].iloc[-1]
        lower_fractal = data["lower_fractals"].iloc[-1]

        # Проверка Volume Delta
        volume_delta = data["volume_delta"].iloc[-1]

        # Генерация сигнала
        if price > upper_fractal and volume_delta > strategy.volume_delta_threshold:
            side = "sell"  # Ложный пробой вверх
        elif price < lower_fractal and volume_delta < -strategy.volume_delta_threshold:
            side = "buy"  # Ложный пробой вниз
        else:
            return None

        # Расчет стоп-лосса и тейк-профита
        stop_loss = strategy._calculate_stop_loss(data, side)
        take_profit = strategy._calculate_take_profit(data, side)

        return {
            "side": side,
            "entry_price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "amount": 1.0,  # Фиксированный размер позиции
        }

    except Exception as e:
        logger.error(f"Error in fake breakout strategy: {str(e)}")
        return None
