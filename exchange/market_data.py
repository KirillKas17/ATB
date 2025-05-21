import asyncio
import json
from asyncio import Lock
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import backoff
import numpy as np
import pandas as pd
import websockets
from loguru import logger

from utils.technical import (
    adx,
    atr,
    bollinger_bands,
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ichimoku,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    calculate_stochastic,
    calculate_vwap,
    cci,
    ema,
    fractals,
    get_significant_levels,
    keltner_channels,
    macd,
    order_imbalance,
    rsi,
    sma,
    stoch_rsi,
    volume_delta,
    vwap,
    wma,
)


@dataclass
class MarketDataConfig:
    """Конфигурация для работы с рыночными данными"""

    update_interval: int = 60  # секунд
    max_candles: int = 1000
    cache_ttl: int = 300  # секунд
    indicators_update_interval: int = 60  # секунд
    min_volume_threshold: float = 1000.0
    significant_levels_threshold: float = 0.02  # 2%
    max_retries: int = 3
    retry_delay: int = 1  # секунд


class MarketData:
    def __init__(
        self,
        symbol: str,
        interval: str,
        max_candles: int = 1000,
        update_interval: int = 1,
        data_dir=None,
    ):
        """Инициализация обработчика рыночных данных"""
        self.symbol = symbol
        self.interval = interval
        self.max_candles = max_candles
        self.update_interval = update_interval
        self.data_dir = data_dir

        # Данные
        self.df = pd.DataFrame()
        self.indicators = {}
        self.last_update = None

        # Кэш и состояние
        self.lock = Lock()
        self.cache_ttl = 60  # секунд
        self.websocket = None
        self.is_connected = False

        # Метрики
        self.volume_profile = defaultdict(float)
        self.price_levels = defaultdict(int)
        self.volatility = 0.0
        self.trend_strength = 0.0
        self.market_regime = "neutral"  # bullish, bearish, neutral

        # События
        self.on_update = None
        self.on_alert = None

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()

    @backoff.on_exception(backoff.expo, (ConnectionError, TimeoutError), max_tries=3)
    async def start(self):
        """Запуск обработчика с повторными попытками"""
        try:
            # Подключение к WebSocket
            self.websocket = await websockets.connect(f"wss://stream.bybit.com/v5/public/spot")
            self.is_connected = True

            # Подписка на канал
            await self._subscribe()

            # Запуск обработки сообщений
            asyncio.create_task(self._handle_messages())

            logger.info(f"Market data started for {self.symbol}")
        except Exception as e:
            logger.error(f"Error starting market data: {str(e)}")
            raise

    async def stop(self):
        """Корректная остановка обработчика"""
        try:
            if self.websocket:
                await self.websocket.close()
                self.is_connected = False

            logger.info(f"Market data stopped for {self.symbol}")
        except Exception as e:
            logger.error(f"Error stopping market data: {str(e)}")
            raise

    async def _subscribe(self):
        """Подписка на канал"""
        try:
            subscribe_message = {
                "op": "subscribe",
                "args": [f"kline.{self.interval}.{self.symbol}"],
            }
            await self.websocket.send(json.dumps(subscribe_message))
        except Exception as e:
            logger.error(f"Error subscribing to channel: {str(e)}")
            raise

    async def _handle_messages(self):
        """Обработка входящих сообщений"""
        try:
            while self.is_connected:
                try:
                    message = await self.websocket.recv()
                    data = json.loads(message)

                    if "data" in data:
                        await self._process_candle(data["data"])

                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    await self._reconnect()
                except Exception as e:
                    logger.error(f"Error handling message: {str(e)}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in message handler: {str(e)}")
            raise

    async def _reconnect(self):
        """Переподключение к WebSocket"""
        try:
            self.is_connected = False
            await asyncio.sleep(5)  # Задержка перед переподключением

            self.websocket = await websockets.connect(f"wss://stream.bybit.com/v5/public/spot")
            self.is_connected = True

            await self._subscribe()
            logger.info("WebSocket reconnected")
        except Exception as e:
            logger.error(f"Error reconnecting: {str(e)}")
            raise

    async def _process_candle(self, candle_data: Dict):
        """Обработка свечи"""
        try:
            async with self.lock:
                # Создание новой свечи
                new_candle = pd.DataFrame(
                    [
                        {
                            "timestamp": pd.to_datetime(candle_data["timestamp"], unit="ms"),
                            "open": float(candle_data["open"]),
                            "high": float(candle_data["high"]),
                            "low": float(candle_data["low"]),
                            "close": float(candle_data["close"]),
                            "volume": float(candle_data["volume"]),
                        }
                    ]
                )

                # Обновление DataFrame
                self.df = pd.concat([self.df, new_candle]).tail(self.max_candles)

                # Обновление индикаторов
                await self._update_indicators()

                # Обновление метрик
                await self._update_metrics()

                # Проверка сигналов
                await self._check_signals()

                # Вызов обработчика обновления
                if self.on_update:
                    await self.on_update(self.df, self.indicators)

                self.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error processing candle: {str(e)}")
            raise

    @lru_cache(maxsize=100)
    async def _update_indicators(self):
        """Обновление индикаторов с кэшированием"""
        try:
            # Технические индикаторы
            self.indicators["rsi"] = calculate_rsi(self.df["close"])
            self.indicators["macd"] = calculate_macd(self.df["close"])
            self.indicators["bb"] = calculate_bollinger_bands(self.df["close"])
            self.indicators["atr"] = calculate_atr(self.df)
            self.indicators["ichimoku"] = calculate_ichimoku(self.df)
            self.indicators["stochastic"] = calculate_stochastic(self.df)
            self.indicators["adx"] = calculate_adx(self.df)
            self.indicators["obv"] = calculate_obv(self.df)
            self.indicators["vwap"] = calculate_vwap(self.df)

            # Дополнительные индикаторы
            self.indicators["sma"] = self.df["close"].rolling(window=20).mean()
            self.indicators["ema"] = self.df["close"].ewm(span=20).mean()
            self.indicators["momentum"] = self.df["close"].pct_change(periods=10)

        except Exception as e:
            logger.error(f"Error updating indicators: {str(e)}")
            raise

    async def _update_metrics(self):
        """Обновление метрик рынка"""
        try:
            # Обновление профиля объема
            for i, row in self.df.iterrows():
                price_level = round(row["close"], 2)
                self.volume_profile[price_level] += row["volume"]
                self.price_levels[price_level] += 1

            # Расчет волатильности
            returns = self.df["close"].pct_change()
            self.volatility = returns.std() * np.sqrt(252)  # Годовая волатильность

            # Расчет силы тренда
            adx = self.indicators["adx"]
            self.trend_strength = adx.iloc[-1] if not adx.empty else 0

            # Определение режима рынка
            sma_20 = self.indicators["sma"].iloc[-1]
            sma_50 = self.df["close"].rolling(window=50).mean().iloc[-1]
            rsi = self.indicators["rsi"].iloc[-1]

            if sma_20 > sma_50 and rsi > 50:
                self.market_regime = "bullish"
            elif sma_20 < sma_50 and rsi < 50:
                self.market_regime = "bearish"
            else:
                self.market_regime = "neutral"

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            raise

    async def _check_signals(self):
        """Проверка торговых сигналов"""
        try:
            if len(self.df) < 50:  # Минимальное количество свечей
                return

            # Получение последних значений
            last_close = self.df["close"].iloc[-1]
            last_rsi = self.indicators["rsi"].iloc[-1]
            last_macd = self.indicators["macd"]["macd"].iloc[-1]
            last_signal = self.indicators["macd"]["signal"].iloc[-1]
            last_bb_upper = self.indicators["bb"]["upper"].iloc[-1]
            last_bb_lower = self.indicators["bb"]["lower"].iloc[-1]

            # Проверка сигналов
            signals = []

            # RSI
            if last_rsi > 70:
                signals.append(("overbought", "RSI"))
            elif last_rsi < 30:
                signals.append(("oversold", "RSI"))

            # MACD
            if last_macd > last_signal:
                signals.append(("bullish", "MACD"))
            elif last_macd < last_signal:
                signals.append(("bearish", "MACD"))

            # Bollinger Bands
            if last_close > last_bb_upper:
                signals.append(("overbought", "BB"))
            elif last_close < last_bb_lower:
                signals.append(("oversold", "BB"))

            # Вызов обработчика сигналов
            if signals and self.on_alert:
                await self.on_alert(signals)

        except Exception as e:
            logger.error(f"Error checking signals: {str(e)}")
            raise

    async def get_indicators(self) -> Dict:
        """Получение индикаторов"""
        try:
            async with self.lock:
                if not self.indicators:
                    await self._update_indicators()
                return self.indicators
        except Exception as e:
            logger.error(f"Error getting indicators: {str(e)}")
            raise

    async def get_metrics(self) -> Dict:
        """Получение метрик рынка"""
        try:
            async with self.lock:
                return {
                    "volatility": self.volatility,
                    "trend_strength": self.trend_strength,
                    "market_regime": self.market_regime,
                    "volume_profile": dict(self.volume_profile),
                    "price_levels": dict(self.price_levels),
                }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            raise

    async def get_support_resistance(self) -> Dict[str, List[float]]:
        """Расчет уровней поддержки и сопротивления"""
        try:
            # Нахождение локальных минимумов и максимумов
            window = 20
            highs = self.df["high"].rolling(window=window, center=True).max()
            lows = self.df["low"].rolling(window=window, center=True).min()

            # Фильтрация уровней
            support_levels = []
            resistance_levels = []

            for i in range(window, len(self.df) - window):
                if self.df["low"].iloc[i] == lows.iloc[i]:
                    support_levels.append(self.df["low"].iloc[i])
                if self.df["high"].iloc[i] == highs.iloc[i]:
                    resistance_levels.append(self.df["high"].iloc[i])

            # Группировка близких уровней
            def group_levels(levels, threshold=0.001):
                if not levels:
                    return []
                levels = sorted(levels)
                grouped = []
                current_group = [levels[0]]

                for level in levels[1:]:
                    if (level - current_group[-1]) / current_group[-1] <= threshold:
                        current_group.append(level)
                    else:
                        grouped.append(sum(current_group) / len(current_group))
                        current_group = [level]

                grouped.append(sum(current_group) / len(current_group))
                return grouped

            return {
                "support": group_levels(support_levels),
                "resistance": group_levels(resistance_levels),
            }

        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            raise


class DataCache:
    """Кэш для хранения рыночных данных с оптимизацией"""

    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = []
        self.lock = Lock()

    async def add(self, symbol: str, data: pd.DataFrame):
        """Добавление данных в кэш с очисткой старых"""
        async with self.lock:
            now = datetime.now()

            # Очистка устаревших данных
            self._cleanup_old_data(now)

            # Проверка размера кэша
            if len(self.timestamps) >= self.max_size:
                oldest = self.timestamps.pop(0)
                del self.cache[oldest]

            self.cache[now] = (symbol, data)
            self.timestamps.append(now)

    def _cleanup_old_data(self, now: datetime):
        """Очистка устаревших данных"""
        while self.timestamps and (now - self.timestamps[0]).seconds > self.ttl:
            oldest = self.timestamps.pop(0)
            del self.cache[oldest]

    async def get(self, symbol: str, lookback: int = 100) -> Optional[pd.DataFrame]:
        """Получение данных из кэша с фильтрацией"""
        async with self.lock:
            data = []
            now = datetime.now()

            for ts in reversed(self.timestamps):
                if (now - ts).seconds > self.ttl:
                    continue

                sym, df = self.cache[ts]
                if sym == symbol:
                    data.append(df)
                    if len(data) >= lookback:
                        break

            if not data:
                return None

            result = pd.concat(data).drop_duplicates()
            return result.sort_index()

    async def clear(self):
        """Очистка кэша"""
        async with self.lock:
            self.cache.clear()
            self.timestamps.clear()
