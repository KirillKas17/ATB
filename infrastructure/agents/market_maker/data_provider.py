"""
Провайдер данных для Market Maker агента.
Включает:
- Получение рыночных данных
- Кэширование данных
- Расчет технических индикаторов
- Обработку ошибок
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4
import time
from shared.numpy_utils import np
import pandas as pd

from domain.types import Symbol
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume as Quantity
from domain.value_objects.currency import Currency


@dataclass
class MarketData:
    """Структура рыночных данных."""

    symbol: Symbol
    timestamp: Timestamp
    open: Price
    high: Price
    low: Price
    close: Price
    volume: Quantity
    bid: Optional[Price] = None
    ask: Optional[Price] = None


@dataclass
class OrderBookLevel:
    """Уровень ордербука."""

    price: Price
    quantity: Quantity
    side: str  # 'bid' или 'ask'


@dataclass
class OrderBook:
    """Структура ордербука."""

    symbol: Symbol
    timestamp: Timestamp
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]

    def get_best_bid(self) -> Optional[Price]:
        """Получить лучшую цену покупки."""
        if self.bids:
            return max(bid.price for bid in self.bids)
        return None

    def get_best_ask(self) -> Optional[Price]:
        """Получить лучшую цену продажи."""
        if self.asks:
            return min(ask.price for ask in self.asks)
        return None

    def get_spread(self) -> Optional[Decimal]:
        """Получить спред."""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        if best_bid and best_ask:
            return Decimal(str(best_ask - best_bid))
        return None


class DataProvider(ABC):
    """Базовый класс для провайдера данных."""

    def __init__(self, cache_ttl: int = 60):
        self.cache_ttl = cache_ttl
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}

    @abstractmethod
    async def get_market_data(self, symbol: Symbol) -> MarketData:
        """Получить рыночные данные."""

    @abstractmethod
    async def get_order_book(self, symbol: Symbol) -> OrderBook:
        """Получить ордербук."""

    @abstractmethod
    async def get_recent_trades(
        self, symbol: Symbol, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получить последние сделки."""

    def _is_cache_valid(self, key: str) -> bool:
        """Проверить валидность кэша."""
        if key not in self._cache_timestamps:
            return False
        return time.time() - self._cache_timestamps[key] < self.cache_ttl

    def _cache_get(self, key: str) -> Optional[Any]:
        """Получить данные из кэша."""
        if self._is_cache_valid(key):
            return self._cache.get(key)
        return None

    def _cache_set(self, key: str, value: Any) -> None:
        """Сохранить данные в кэш."""
        self._cache[key] = value
        self._cache_timestamps[key] = time.time()


class MarketDataProvider(DataProvider):
    """Провайдер рыночных данных с кэшированием."""

    def __init__(self, cache_ttl: int = 60):
        super().__init__(cache_ttl)
        self._historical_data: Dict[Symbol, pd.DataFrame] = {}

    async def get_market_data(self, symbol: Symbol) -> MarketData:
        """Получить рыночные данные."""
        cache_key = f"market_data_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data and isinstance(cached_data, MarketData):
            return cached_data
        # Симуляция получения данных
        data = await self._fetch_market_data(symbol)
        self._cache_set(cache_key, data)
        return data

    async def get_order_book(self, symbol: Symbol) -> OrderBook:
        """Получить ордербук."""
        cache_key = f"orderbook_{symbol}"
        cached_data = self._cache_get(cache_key)
        if cached_data and isinstance(cached_data, OrderBook):
            return cached_data
        # Симуляция получения ордербука
        orderbook = await self._fetch_order_book(symbol)
        self._cache_set(cache_key, orderbook)
        return orderbook

    async def get_recent_trades(
        self, symbol: Symbol, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получить последние сделки."""
        cache_key = f"trades_{symbol}_{limit}"
        cached_data = self._cache_get(cache_key)
        if cached_data and isinstance(cached_data, list):
            return cached_data
        # Симуляция получения сделок
        trades = await self._fetch_recent_trades(symbol, limit)
        self._cache_set(cache_key, trades)
        return trades

    def get_current_price(self, symbol: Symbol) -> Optional[Decimal]:
        """Получить текущую цену."""
        # Используем кэшированные данные или генерируем
        cache_key = f"current_price_{symbol}"
        cached_price = self._cache_get(cache_key)
        if cached_price:
            if hasattr(cached_price, 'value'):
                return Decimal(str(cached_price.value))
            return Decimal(str(cached_price))
        # Генерируем случайную цену
        base_price = 500.0 + np.random.normal(0, 10)
        price = Price(Decimal(str(base_price)), Currency.USDT)
        self._cache_set(cache_key, price)
        return price.value

    def get_volatility(self, symbol: Symbol, window: int = 20) -> float:
        """Получить волатильность."""
        # Используем исторические данные для расчета
        if symbol in self._historical_data:
            df = self._historical_data[symbol]
            if len(df) >= window:
                returns = df["close"].pct_change().dropna()
                return float(returns.tail(window).std())
        # Возвращаем случайную волатильность
        return float(np.random.uniform(0.01, 0.05))

    def get_volume(self, symbol: Symbol, window: int = 20) -> float:
        """Получить объем."""
        # Используем исторические данные
        if symbol in self._historical_data:
            df = self._historical_data[symbol]
            if len(df) >= window:
                return float(df["volume"].tail(window).mean())
        # Возвращаем случайный объем
        return float(np.random.uniform(1000, 10000))

    async def _fetch_market_data(self, symbol: Symbol) -> MarketData:
        """Получить рыночные данные с биржи."""
        # Симуляция задержки сети
        await asyncio.sleep(0.01)
        # Генерируем случайные данные
        base_price = float(np.random.uniform(100, 1000))
        change = np.random.normal(0, 0.02)
        return MarketData(
            symbol=symbol,
            timestamp=Timestamp.now(),
            open=Price(Decimal(str(base_price)), Currency.USDT),
            high=Price(Decimal(str(base_price * (1 + abs(change)))), Currency.USDT),
            low=Price(Decimal(str(base_price * (1 - abs(change)))), Currency.USDT),
            close=Price(Decimal(str(base_price * (1 + change))), Currency.USDT),
            volume=Quantity(Decimal(str(np.random.randint(1000, 10000))), Currency.USDT),
            bid=Price(Decimal(str(base_price * 0.999)), Currency.USDT),
            ask=Price(Decimal(str(base_price * 1.001)), Currency.USDT),
        )

    async def _fetch_order_book(self, symbol: Symbol) -> OrderBook:
        """Получить ордербук с биржи."""
        # Симуляция задержки сети
        await asyncio.sleep(0.01)
        current_price = self.get_current_price(symbol)
        if current_price is None:
            base_price = 500.0  # Значение по умолчанию
        else:
            base_price = float(current_price)
        # Генерируем уровни ордербука
        bids = []
        asks = []
        for i in range(10):
            bid_price = base_price * (1 - 0.001 * (i + 1))
            ask_price = base_price * (1 + 0.001 * (i + 1))
            bids.append(
                OrderBookLevel(
                    price=Price(Decimal(str(bid_price)), Currency.USDT),
                    quantity=Quantity(Decimal(str(np.random.uniform(0.1, 10))), Currency.USDT),
                    side="bid",
                )
            )
            asks.append(
                OrderBookLevel(
                    price=Price(Decimal(str(ask_price)), Currency.USDT),
                    quantity=Quantity(Decimal(str(np.random.uniform(0.1, 10))), Currency.USDT),
                    side="ask",
                )
            )
        return OrderBook(symbol=symbol, timestamp=Timestamp.now(), bids=bids, asks=asks)

    async def _fetch_recent_trades(
        self, symbol: Symbol, limit: int
    ) -> List[Dict[str, Any]]:
        """Получить последние сделки с биржи."""
        # Симуляция задержки сети
        await asyncio.sleep(0.01)
        current_price = self.get_current_price(symbol)
        if current_price is None:
            base_price = 500.0  # Значение по умолчанию
        else:
            base_price = float(current_price)
        trades = []
        for _ in range(limit):
            price_change = np.random.normal(0, 0.001)
            trade_price = base_price * (1 + price_change)
            trades.append(
                {
                    "id": str(uuid4()),
                    "price": Price(Decimal(str(trade_price)), Currency.USDT),
                    "quantity": Quantity(Decimal(str(np.random.uniform(0.01, 1))), Currency.USDT),
                    "side": np.random.choice(["buy", "sell"]),
                    "timestamp": Timestamp.now().to_iso(),
                }
            )
        return trades

    def update_historical_data(self, symbol: Symbol, data: pd.DataFrame) -> None:
        """Обновить исторические данные."""
        self._historical_data[symbol] = data

    def get_historical_data(self, symbol: Symbol) -> Optional[pd.DataFrame]:
        """Получить исторические данные."""
        return self._historical_data.get(symbol)


class TechnicalIndicators:
    """Калькулятор технических индикаторов."""

    @staticmethod
    def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
        """Рассчитать простую скользящую среднюю."""
        return prices.rolling(window=window).mean()

    @staticmethod
    def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
        """Рассчитать экспоненциальную скользящую среднюю."""
        return prices.ewm(span=window).mean()

    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Рассчитать RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()  # type: ignore
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()  # type: ignore
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_bollinger_bands(
        prices: pd.Series, window: int = 20, std_dev: int = 2
    ) -> Dict[str, pd.Series]:
        """Рассчитать полосы Боллинджера."""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band,
        }

    @staticmethod
    def calculate_atr(
        high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
    ) -> pd.Series:
        """Рассчитать Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()


# Фабричная функция для создания провайдера данных
def create_market_data_provider(cache_ttl: int = 60) -> MarketDataProvider:
    """Создать провайдер рыночных данных."""
    return MarketDataProvider(cache_ttl=cache_ttl)
