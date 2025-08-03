# -*- coding: utf-8 -*-
"""BingX WebSocket client for real-time market data."""
import asyncio
import json
import time
from decimal import Decimal
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None  # type: ignore
from loguru import logger

from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from shared.models.orderbook import OrderBookUpdate


class BingXWebSocketClient:
    """WebSocket клиент для BingX биржи."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "wss://open-api-swap.bingx.com/swap-market"
        self.websocket: Optional[Any] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        self.ping_interval = 30.0
        self.last_ping = 0.0
        self.subscribed_symbols: set = set()
        self.callback: Optional[Callable[[OrderBookUpdate], Awaitable[None]]] = None

    async def connect(self) -> bool:
        """Подключение к WebSocket BingX."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available. Install websockets package.")
            return False
        try:
            logger.info(f"Connecting to BingX WebSocket: {self.base_url}")
            self.websocket = await websockets.connect(
                self.base_url, ping_interval=20, ping_timeout=10, close_timeout=10
            )
            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully connected to BingX WebSocket")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to BingX WebSocket: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Отключение от WebSocket."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        self.subscribed_symbols.clear()
        logger.info("Disconnected from BingX WebSocket")

    async def subscribe(self, symbol: str) -> bool:
        """Подписка на ордербук и сделки для символа."""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected to BingX WebSocket")
            return False
        try:
            # Нормализуем символ для BingX (BTCUSDT -> BTC-USDT)
            formatted_symbol = self._normalize_symbol(symbol)
            # Подписка на ордербук
            depth_subscription = {
                "id": f"depth_sub_{symbol}",
                "event": "subscribe",
                "topic": "market.depth",
                "params": {"symbol": formatted_symbol},
            }
            # Подписка на сделки
            trade_subscription = {
                "id": f"trade_sub_{symbol}",
                "event": "subscribe",
                "topic": "market.trade",
                "params": {"symbol": formatted_symbol},
            }
            await self.websocket.send(json.dumps(depth_subscription))
            await self.websocket.send(json.dumps(trade_subscription))
            self.subscribed_symbols.add(symbol)
            logger.info(f"Subscribed to {symbol} on BingX")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol} on BingX: {e}")
            return False

    async def unsubscribe(self, symbol: str) -> bool:
        """Отписка от символа."""
        if not self.is_connected or not self.websocket:
            return False
        try:
            formatted_symbol = self._normalize_symbol(symbol)
            unsubscribe_msg = {
                "id": f"unsub_{symbol}",
                "event": "unsubscribe",
                "topic": "market.depth",
                "params": {"symbol": formatted_symbol},
            }
            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.subscribed_symbols.discard(symbol)
            logger.info(f"Unsubscribed from {symbol} on BingX")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {symbol} on BingX: {e}")
            return False

    async def listen(self, callback: Callable[[OrderBookUpdate], Awaitable[None]]) -> None:
        """Прослушивание сообщений от WebSocket."""
        self.callback = callback
        while self.is_connected:
            try:
                if not self.websocket:
                    if not await self.connect():
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                # Проверяем ping
                current_time = time.time()
                if current_time - self.last_ping > self.ping_interval:
                    await self._send_ping()
                    self.last_ping = current_time
                # Получаем сообщение
                if self.websocket is not None:
                    message = await asyncio.wait_for(self.websocket.recv(), timeout=5.0)
                else:
                    continue
                # Обрабатываем сообщение
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except websockets.exceptions.ConnectionClosed:
                logger.warning(
                    "BingX WebSocket connection closed, attempting reconnect..."
                )
                self.is_connected = False
                await self._reconnect()
            except Exception as e:
                logger.error(f"Error in BingX WebSocket listener: {e}")
                await asyncio.sleep(1.0)

    async def _handle_message(self, message: str) -> None:
        """Обработка входящего сообщения."""
        try:
            data = json.loads(message)
            # Обрабатываем pong
            if data.get("event") == "pong":
                return
            # Обрабатываем подписку
            if data.get("event") == "subscribe":
                logger.info(f"Successfully subscribed to BingX: {data}")
                return
            # Обрабатываем данные ордербука
            if data.get("topic") == "market.depth":
                order_book_update = self._parse_order_book_update(data)
                if order_book_update and self.callback:
                    await self.callback(order_book_update)
            # Обрабатываем данные сделок
            elif data.get("topic") == "market.trade":
                # Можно добавить обработку сделок если нужно
                pass
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse BingX message: {e}")
        except Exception as e:
            logger.error(f"Error handling BingX message: {e}")

    def _parse_order_book_update(
        self, data: Dict[str, Any]
    ) -> Optional[OrderBookUpdate]:
        """Парсинг обновления ордербука из сообщения BingX."""
        try:
            # BingX формат данных:
            # {
            #   "topic": "market.depth",
            #   "data": {
            #     "symbol": "BTC-USDT",
            #     "bids": [["50000.0", "1.5"], ...],
            #     "asks": [["50010.0", "2.0"], ...],
            #     "timestamp": 1640995200000
            #   }
            # }
            if "data" not in data:
                return None
            order_book_data = data["data"]
            symbol = order_book_data.get("symbol", "UNKNOWN")
            # Конвертируем обратно в стандартный формат
            normalized_symbol = self._denormalize_symbol(symbol)
            # Получаем timestamp
            timestamp_ms = order_book_data.get("timestamp", int(time.time() * 1000))
            timestamp = Timestamp.from_unix_ms(timestamp_ms)
            # Определяем валюту
            base_currency = Currency.from_string("BTC") or Currency.USD
            quote_currency = Currency.from_string("USD") or Currency.USD
            # Парсим bids и asks
            bids_data = order_book_data.get("bids", [])
            asks_data = order_book_data.get("asks", [])
            bids = self._parse_price_volume_pairs(
                bids_data, base_currency, quote_currency
            )
            asks = self._parse_price_volume_pairs(
                asks_data, base_currency, quote_currency
            )
            return OrderBookUpdate(
                exchange="bingx",
                symbol=normalized_symbol,
                bids=bids,
                asks=asks,
                timestamp=timestamp,
                sequence_id=timestamp_ms,
            )
        except Exception as e:
            logger.error(f"Error parsing BingX order book update: {e}")
            return None

    def _parse_price_volume_pairs(
        self, data: List[List[str]], base_currency: Currency, quote_currency: Currency
    ) -> List[Tuple[Price, Volume]]:
        """Парсинг пар цена-объем."""
        result = []
        for price_str, volume_str in data:
            try:
                price = Price(Decimal(price_str), base_currency, quote_currency)
                volume = Volume(Decimal(volume_str), quote_currency)
                result.append((price, volume))
            except Exception as e:
                logger.warning(f"Error parsing price-volume pair: {e}")
        return result

    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа для BingX (BTCUSDT -> BTC-USDT)."""
        if "USDT" in symbol:
            return symbol.replace("USDT", "-USDT")
        return symbol

    def _denormalize_symbol(self, symbol: str) -> str:
        """Денормализация символа из BingX (BTC-USDT -> BTCUSDT)."""
        return symbol.replace("-USDT", "USDT")

    async def _send_ping(self) -> None:
        """Отправка ping для поддержания соединения."""
        if self.websocket:
            try:
                ping_msg = {"event": "ping"}
                await self.websocket.send(json.dumps(ping_msg))
            except Exception as e:
                logger.error(f"Failed to send ping to BingX: {e}")

    async def _reconnect(self) -> None:
        """Переподключение к WebSocket."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached for BingX")
            return
        self.reconnect_attempts += 1
        logger.info(
            f"Attempting to reconnect to BingX (attempt {self.reconnect_attempts})"
        )
        await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)
        if await self.connect():
            # Переподписываемся на символы
            for symbol in self.subscribed_symbols.copy():
                await self.subscribe(symbol)

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса подключения."""
        return {
            "exchange": "bingx",
            "is_connected": self.is_connected,
            "subscribed_symbols": list(self.subscribed_symbols),
            "reconnect_attempts": self.reconnect_attempts,
            "last_ping": self.last_ping,
        }
