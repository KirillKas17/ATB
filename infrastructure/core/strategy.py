"""
Модуль торговых стратегий.
Содержит базовые классы и реализации различных торговых стратегий
для автоматической торговли на криптовалютных биржах.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
import pandas as pd
import numpy as np
import pickle

from domain.entities.market import MarketData  # исправлен импорт

# Type aliases для pandas
DataFrame = pd.DataFrame
Series = pd.Series


@dataclass
class Signal:
    """Торговый сигнал."""

    pair: str
    action: str  # buy, sell
    price: float
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Преобразование в словарь.
        Returns:
            Dict[str, Any]: Словарь с данными
        """
        return {
            "pair": self.pair,
            "action": self.action,
            "price": self.price,
            "size": self.size,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
        }


class CoreStrategy:
    """Базовый класс для торговых стратегий."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация стратегии.
        Args:
            config: Конфигурация стратегии
        """
        self.config = config
        self.parameters = config.get("parameters", {})
        self.metrics = config.get("metrics", {})
        self.last_update = None
        self.pair = str(config.get("pair", "BTC/USD"))
        self.timeframe = str(config.get("timeframe", "1h"))
        self.trend_threshold = float(self.parameters.get("trend_threshold", 0.0))
        self.position_size = float(self.parameters.get("position_size", 1.0))
        self.stop_loss = float(self.parameters.get("stop_loss", 0.02))
        self.take_profit = float(self.parameters.get("take_profit", 0.04))
        self.model = None
        logger.info(f"CoreStrategy initialized for {self.pair}")

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Генерация торговых сигналов на основе рыночных данных.
        Args:
            market_data: Рыночные данные
        Returns:
            List[Signal]: Список сигналов
        """
        try:
            if (
                not market_data
                or not hasattr(market_data, "close")
                or not hasattr(market_data.close, "__len__")
                or len(market_data.close) < 20
            ):
                return []
            signals = []
            if hasattr(market_data.close, 'iloc'):
                current_price_raw = market_data.close.iloc[-1]
                # Безопасное преобразование Price в float
                if hasattr(current_price_raw, 'value'):
                    current_price = float(current_price_raw.value)
                else:
                    current_price = float(current_price_raw)
            else:
                # Fallback если iloc недоступен
                current_price_raw = market_data.close
                if hasattr(current_price_raw, 'value'):
                    current_price = float(current_price_raw.value)
                else:
                    current_price = float(str(current_price_raw))
            # Расчет EMA
            ema_period = self.parameters.get("ema_period", 14)
            if hasattr(market_data.close, 'tolist'):
                ema = self._calculate_ema(market_data.close.tolist(), ema_period)
            else:
                # Fallback если tolist недоступен
                ema = self._calculate_ema([float(market_data.close)], ema_period)
            # Получение параметров
            trend_threshold = self.parameters.get("trend_threshold", 0.01)
            position_size = self.parameters.get("position_size", 1.0)
            stop_loss_pct = self.parameters.get("stop_loss", 0.02)
            take_profit_pct = self.parameters.get("take_profit", 0.04)
            # Анализ тренда
            price_change = (current_price - ema) / ema
            # Генерация сигналов
            if price_change > trend_threshold:
                # Сигнал на покупку
                signals.append(
                    Signal(
                        pair=str(self.pair),
                        action="buy",
                        price=float(current_price.value) if hasattr(current_price, 'value') else float(current_price),
                        size=position_size,
                        stop_loss=float(current_price.value * (1 - stop_loss_pct)) if hasattr(current_price, 'value') else float(current_price) * (1 - stop_loss_pct),
                        take_profit=float(current_price.value * (1 + take_profit_pct)) if hasattr(current_price, 'value') else float(current_price) * (1 + take_profit_pct),
                        metadata={
                            "strategy": "CoreStrategy",
                            "ema": ema,
                            "price_change": price_change,
                            "confidence": min(abs(price_change) / trend_threshold, 1.0),
                        },
                    )
                )
            elif price_change < -trend_threshold:
                # Сигнал на продажу
                signals.append(
                    Signal(
                        pair=str(self.pair),
                        action="sell",
                        price=float(current_price.value) if hasattr(current_price, 'value') else float(current_price),
                        size=position_size,
                        stop_loss=float(current_price.value * (1 + stop_loss_pct)) if hasattr(current_price, 'value') else float(current_price) * (1 + stop_loss_pct),
                        take_profit=float(current_price.value * (1 - take_profit_pct)) if hasattr(current_price, 'value') else float(current_price) * (1 - take_profit_pct),
                        metadata={
                            "strategy": "CoreStrategy",
                            "ema": ema,
                            "price_change": price_change,
                            "confidence": min(abs(price_change) / trend_threshold, 1.0),
                        },
                    )
                )
            logger.debug(f"Generated {len(signals)} signals for {self.pair}")
            return signals
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []

    def update(self, market_data: MarketData) -> None:
        """
        Обновление состояния стратегии. Должен быть реализован в подклассе.
        """
        # Базовая реализация - сохранение последних данных
        self.last_market_data = market_data
        self.last_update_time = datetime.now()

    def get_parameters(self) -> Dict[str, Any]:
        """
        Получение параметров стратегии.
        """
        return self.parameters.copy()

    def set_parameters(self, parameters: Union[Dict[str, Any], pd.DataFrame]) -> None:
        """
        Установка параметров стратегии.
        Args:
            parameters: Новые параметры
        """
        if isinstance(parameters, pd.DataFrame):
            if hasattr(parameters, 'to_dict'):
                params_dict = parameters.to_dict(orient="records")[0]
            else:
                # Fallback если to_dict недоступен
                if hasattr(parameters, 'iloc') and callable(parameters.iloc):
                    iloc_result: pd.Series = parameters.iloc[0]
                    if hasattr(iloc_result, 'to_dict') and callable(iloc_result.to_dict):
                        params_dict = iloc_result.to_dict()
                    else:
                        params_dict = {}
                else:
                    params_dict = {}
            self.parameters = {str(k): v for k, v in params_dict.items()}
        else:
            self.parameters = {str(k): v for k, v in parameters.items()}

    def get_state(self) -> Dict[str, Any]:
        """
        Получение состояния стратегии.
        Returns:
            Dict[str, Any]: Состояние стратегии
        """
        return {
            "pair": str(self.pair),
            "timeframe": str(self.timeframe),
            "parameters": self.parameters,
        }

    def save_state(self, filename: str) -> None:
        """
        Сохранение состояния стратегии.
        Args:
            filename: Имя файла
        """
        state = self.get_state()
        with open(filename, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Strategy state saved to {filename}")

    def load_state(self, filename: str) -> None:
        """
        Загрузка состояния стратегии из файла.
        Args:
            filename: Имя файла
        """
        try:
            # Проверка существования файла
            if not os.path.exists(filename):
                logger.error(f"State file not found: {filename}")
                return
            # Безопасная загрузка с проверкой размера файла
            file_size = os.path.getsize(filename)
            if file_size > 100 * 1024 * 1024:  # Максимум 100MB
                logger.error(f"State file too large: {file_size} bytes")
                return
            with open(filename, "rb") as f:
                state = pickle.load(f)
            # Проверка структуры данных
            required_keys = ["pair", "timeframe", "parameters"]
            if not all(key in state for key in required_keys):
                logger.error(f"Invalid state file structure: {filename}")
                return
            self.pair = state["pair"]
            self.timeframe = state["timeframe"]
            self.parameters = state["parameters"]
            logger.info(f"Strategy state loaded from {filename}")
        except Exception as e:
            logger.error(f"Error loading strategy state: {e}")

    def analyze(self, data: Optional[pd.DataFrame] = None) -> List[Signal]:
        """
        Анализ данных и генерация сигналов.
        Args:
            data: Данные для анализа
        Returns:
            List[Signal]: Список сигналов
        """
        if data is None or data.empty:
            return []
        signals = []
        close_data = data["close"]
        if hasattr(close_data, 'iloc') and callable(close_data.iloc):
            iloc_result: pd.Series = close_data.iloc[-1]  # type: ignore[index]
            if callable(iloc_result):
                current_price = float(iloc_result())
            else:
                current_price = float(iloc_result)
        else:
            current_price = float(close_data)
        ema = self._calculate_ema(data["close"].tolist())
        if current_price > ema * (1 + (self.trend_threshold or 0)):
            signals.append(
                Signal(
                    pair=str(self.pair),  # type: ignore
                    action="buy",
                    price=current_price,
                    size=self.position_size or 1.0,
                    stop_loss=current_price * (1 - (self.stop_loss or 0.02)),
                    take_profit=current_price * (1 + (self.take_profit or 0.04)),
                )
            )
        elif current_price < ema * (1 - (self.trend_threshold or 0)):
            signals.append(
                Signal(
                    pair=str(self.pair),  # type: ignore
                    action="sell",
                    price=current_price,
                    size=self.position_size or 1.0,
                    stop_loss=current_price * (1 + (self.stop_loss or 0.02)),
                    take_profit=current_price * (1 - (self.take_profit or 0.04)),
                )
            )
        return signals

    def update_parameters(self, df: pd.DataFrame) -> None:
        """
        Обновляет параметры стратегии из DataFrame.
        Args:
            df: DataFrame с параметрами
        """
        if df is None or df.empty:
            return
        params_dict = df.iloc[0].to_dict()  # type: ignore
        self.parameters = {str(k): v for k, v in params_dict.items()}
        logger.info(f"Updated strategy parameters: {self.parameters}")

    def _calculate_ema(
        self, prices: List[float], period: Optional[int] = None
    ) -> float:
        """
        Расчет экспоненциальной скользящей средней.
        Args:
            prices: Список цен
            period: Период расчета
        Returns:
            float: Значение EMA
        """
        if not prices:
            return 0.0
        period = period or 14
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]  # type: ignore[arg-type]

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Обучение стратегии.
        Args:
            X: Признаки
            y: Целевая переменная (опционально)
        Raises:
            ValueError: Если модель не инициализирована
        """
        try:
            if self.model is None:
                raise ValueError("Model is not initialized")
            if y is not None:
                self.model.fit(X, y)
            else:
                self.model.fit(X)
            logger.info("Strategy fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting strategy: {str(e)}")
            raise


class MovingAverageCrossover(CoreStrategy):
    """Стратегия на основе пересечения скользящих средних."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация стратегии.
        Args:
            config: Конфигурация стратегии
        """
        super().__init__(config)
        # Параметры стратегии
        self.fast_period = self.parameters.get("fast_period", 10)
        self.slow_period = self.parameters.get("slow_period", 20)
        self.signal_period = self.parameters.get("signal_period", 9)
        # Состояние стратегии
        self.fast_ma: List[float] = []
        self.slow_ma: List[float] = []
        self.signal_ma: List[float] = []
        logger.info("MovingAverageCrossover strategy initialized")

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Генерация торговых сигналов.
        Args:
            market_data: Рыночные данные
        Returns:
            List[Signal]: Список сигналов
        """
        signals = []
        # Обновление скользящих средних
        self.fast_ma.append(float(market_data.close.value))  # Преобразуем в float
        self.slow_ma.append(float(market_data.close.value))  # Преобразуем в float
        self.signal_ma.append(float(market_data.close.value))  # Преобразуем в float
        if len(self.fast_ma) > self.fast_period:
            self.fast_ma.pop(0)
        if len(self.slow_ma) > self.slow_period:
            self.slow_ma.pop(0)
        if len(self.signal_ma) > self.signal_period:
            self.signal_ma.pop(0)
        # Расчет значений
        if (
            len(self.fast_ma) == self.fast_period
            and len(self.slow_ma) == self.slow_period
        ):
            fast_value = np.mean(self.fast_ma)
            slow_value = np.mean(self.slow_ma)
            # Генерация сигналов
            if fast_value > slow_value and len(self.signal_ma) == self.signal_period:
                signal_value = np.mean(self.signal_ma)
                if fast_value > signal_value:
                    signals.append(
                        Signal(
                            pair=str(market_data.symbol),  # Преобразуем в str
                            action="buy",
                            price=float(market_data.close.value),  # Преобразуем в float
                            size=1.0,
                            stop_loss=float(float(market_data.close.value) * 0.95),  # Преобразуем в float
                            take_profit=float(float(market_data.close.value) * 1.05),  # Преобразуем в float
                        )
                    )
            elif fast_value < slow_value and len(self.signal_ma) == self.signal_period:
                signal_value = np.mean(self.signal_ma)
                if fast_value < signal_value:
                    signals.append(
                        Signal(
                            pair=str(market_data.symbol),  # Преобразуем в str
                            action="sell",
                            price=float(market_data.close.value),  # Преобразуем в float
                            size=1.0,
                            stop_loss=float(float(market_data.close.value) * 1.05),  # Преобразуем в float
                            take_profit=float(float(market_data.close.value) * 0.95),  # Преобразуем в float
                        )
                    )
        return signals


class RSIStrategy(CoreStrategy):
    """Стратегия на основе индикатора RSI."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация стратегии.
        Args:
            config: Конфигурация стратегии
        """
        super().__init__(config)
        # Параметры стратегии
        self.period = self.parameters.get("period", 14)
        self.overbought = self.parameters.get("overbought", 70)
        self.oversold = self.parameters.get("oversold", 30)
        # Состояние стратегии
        self.prices: List[float] = []
        self.rsi_values: List[float] = []
        logger.info("RSI strategy initialized")

    def _calculate_rsi(self, prices: List[float], period: int) -> float:
        """
        Расчет значения RSI.
        Args:
            prices: Список цен
            period: Период расчета
        Returns:
            float: Значение RSI
        """
        if len(prices) < period + 1:
            return 50.0
        # Расчет изменений
        deltas = np.diff(prices)
        # Разделение на положительные и отрицательные изменения
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        # Расчет средних значений
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)  # Явное приведение к float

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Генерация торговых сигналов.
        Args:
            market_data: Рыночные данные
        Returns:
            List[Signal]: Список сигналов
        """
        signals = []
        # Обновление цен
        self.prices.append(float(market_data.close.value))  # Преобразуем в float
        if len(self.prices) > self.period + 1:
            self.prices.pop(0)
        # Расчет RSI
        if len(self.prices) == self.period + 1:
            rsi = self._calculate_rsi(self.prices, self.period)
            self.rsi_values.append(rsi)
            # Генерация сигналов
            if rsi < self.oversold:
                signals.append(
                    Signal(
                        pair=str(market_data.symbol),  # Преобразуем в str
                        action="buy",
                        price=float(market_data.close.value),  # Преобразуем в float
                        size=1.0,
                        stop_loss=float(float(market_data.close.value) * 0.95),  # Преобразуем в float
                        take_profit=float(float(market_data.close.value) * 1.05),  # Преобразуем в float
                    )
                )
            elif rsi > self.overbought:
                signals.append(
                    Signal(
                        pair=str(market_data.symbol),  # Преобразуем в str
                        action="sell",
                        price=float(market_data.close.value),  # Преобразуем в float
                        size=1.0,
                        stop_loss=float(float(market_data.close.value) * 1.05),  # Преобразуем в float
                        take_profit=float(float(market_data.close.value) * 0.95),  # Преобразуем в float
                    )
                )
        return signals


class MACDStrategy(CoreStrategy):
    """Стратегия на основе индикатора MACD."""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация стратегии.
        Args:
            config: Конфигурация стратегии
        """
        super().__init__(config)
        # Параметры стратегии
        self.fast_period = self.parameters.get("fast_period", 12)
        self.slow_period = self.parameters.get("slow_period", 26)
        self.signal_period = self.parameters.get("signal_period", 9)
        # Состояние стратегии
        self.prices: List[float] = []
        self.macd_values: List[float] = []
        self.signal_values: List[float] = []
        logger.info("MACD strategy initialized")

    def _calculate_ema(
        self, prices: List[float], period: Optional[int] = None
    ) -> float:
        """Расчет EMA."""
        if not prices:
            return 0.0
        period = period or 14  # Значение по умолчанию
        multiplier = 2.0 / (period + 1)
        ema = float(prices[0])
        for price in prices[1:]:
            ema = (float(price) if hasattr(price, "__float__") else 0.0 - ema) * multiplier + ema
        return ema

    def _calculate_macd(self, prices: List[float]) -> Tuple[float, float, float]:
        """
        Расчет значений MACD.
        Args:
            prices: Список цен
        Returns:
            Tuple[float, float, float]: (MACD, Signal, Histogram)
        """
        if len(prices) < self.slow_period:
            return 0.0, 0.0, 0.0
        # Расчет быстрой и медленной EMA
        fast_ema = self._calculate_ema(prices[-self.fast_period :], self.fast_period)
        slow_ema = self._calculate_ema(prices[-self.slow_period :], self.slow_period)
        # Расчет MACD
        macd = fast_ema - slow_ema
        # Расчет сигнальной линии
        self.macd_values.append(macd)
        if len(self.macd_values) > self.signal_period:
            self.macd_values.pop(0)
        signal = self._calculate_ema(self.macd_values, self.signal_period)
        # Расчет гистограммы
        histogram = macd - signal
        return macd, signal, histogram

    def generate_signals(self, market_data: MarketData) -> List[Signal]:
        """
        Генерация торговых сигналов.
        Args:
            market_data: Рыночные данные
        Returns:
            List[Signal]: Список сигналов
        """
        signals = []
        # Обновление цен
        self.prices.append(float(market_data.close.value))  # Преобразуем в float
        if len(self.prices) > self.slow_period:
            self.prices.pop(0)
        # Расчет MACD
        if len(self.prices) == self.slow_period:
            macd, signal, histogram = self._calculate_macd(self.prices)
            # Генерация сигналов
            if (
                histogram > 0
                and len(self.signal_values) > 0
                and self.signal_values[-1] <= 0
            ):
                signals.append(
                    Signal(
                        pair=str(market_data.symbol),  # Преобразуем в str
                        action="buy",
                        price=float(market_data.close.value),  # Преобразуем в float
                        size=1.0,
                        stop_loss=float(float(market_data.close.value) * 0.95),  # Преобразуем в float
                        take_profit=float(float(market_data.close.value) * 1.05),  # Преобразуем в float
                    )
                )
            elif (
                histogram < 0
                and len(self.signal_values) > 0
                and self.signal_values[-1] >= 0
            ):
                signals.append(
                    Signal(
                        pair=str(market_data.symbol),  # Преобразуем в str
                        action="sell",
                        price=float(market_data.close.value),  # Преобразуем в float
                        size=1.0,
                        stop_loss=float(float(market_data.close.value) * 1.05),  # Преобразуем в float
                        take_profit=float(float(market_data.close.value) * 0.95),  # Преобразуем в float
                    )
                )
            self.signal_values.append(histogram)
            if len(self.signal_values) > self.signal_period:
                self.signal_values.pop(0)
        return signals
