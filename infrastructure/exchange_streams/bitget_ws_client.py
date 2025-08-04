# -*- coding: utf-8 -*-
"""Bitget WebSocket client for real-time market data."""

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
    websockets = None

from loguru import logger

from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from shared.models.orderbook import OrderBookUpdate


class BitgetWebSocketClient:
    """WebSocket клиент для Bitget биржи."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "wss://ws.bitget.com/spot/v1/stream"
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
        """Подключение к WebSocket Bitget."""
        try:
            logger.info(f"Connecting to Bitget WebSocket: {self.base_url}")

            self.websocket = await websockets.connect(
                self.base_url, ping_interval=20, ping_timeout=10, close_timeout=10
            )

            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully connected to Bitget WebSocket")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Bitget WebSocket: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """Отключение от WebSocket."""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        self.subscribed_symbols.clear()
        logger.info("Disconnected from Bitget WebSocket")

    async def subscribe(self, symbol: str) -> bool:
        """Подписка на ордербук и сделки для символа."""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected to Bitget WebSocket")
            return False

        try:
            # Нормализуем символ для Bitget (BTCUSDT -> BTCUSDT_SPBL)
            formatted_symbol = self._normalize_symbol(symbol)

            # Подписка на ордербук (depth5) и сделки
            subscription = {
                "op": "subscribe",
                "args": [
                    f"spot/depth5:{formatted_symbol}",
                    f"spot/match:{formatted_symbol}",
                ],
            }

            await self.websocket.send(json.dumps(subscription))
            self.subscribed_symbols.add(symbol)
            logger.info(f"Subscribed to {symbol} on Bitget")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol} on Bitget: {e}")
            return False

    async def unsubscribe(self, symbol: str) -> bool:
        """Отписка от символа."""
        if not self.is_connected or not self.websocket:
            return False

        try:
            formatted_symbol = self._normalize_symbol(symbol)

            unsubscribe_msg = {
                "op": "unsubscribe",
                "args": [
                    f"spot/depth5:{formatted_symbol}",
                    f"spot/match:{formatted_symbol}",
                ],
            }

            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.subscribed_symbols.discard(symbol)
            logger.info(f"Unsubscribed from {symbol} on Bitget")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe from {symbol} on Bitget: {e}")
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
                    "Bitget WebSocket connection closed, attempting reconnect..."
                )
                self.is_connected = False
                await self._reconnect()
            except Exception as e:
                logger.error(f"Error in Bitget WebSocket listener: {e}")
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
                logger.info(f"Successfully subscribed to Bitget: {data}")
                return

            # Обрабатываем данные ордербука
            if "data" in data and "arg" in data:
                arg = data["arg"]
                if "depth5" in arg.get("channel", ""):
                    order_book_update = self._parse_order_book_update(data)
                    if order_book_update and self.callback:
                        await self.callback(order_book_update)

            # Обрабатываем данные сделок
            elif "data" in data and "arg" in data:
                arg = data["arg"]
                if "match" in arg.get("channel", ""):
                    # Можно добавить обработку сделок если нужно
                    pass

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Bitget message: {e}")
        except Exception as e:
            logger.error(f"Error handling Bitget message: {e}")

    def _parse_order_book_update(
        self, data: Dict[str, Any]
    ) -> Optional[OrderBookUpdate]:
        """Парсинг обновления ордербука из сообщения Bitget."""
        try:
            # Bitget формат данных:
            # {
            #   "arg": {
            #     "channel": "spot/depth5",
            #     "instId": "BTCUSDT_SPBL"
            #   },
            #   "data": [
            #     {
            #       "asks": [["50010.0", "2.0", "0", "3"], ...],
            #       "bids": [["50000.0", "1.5", "0", "2"], ...],
            #       "ts": "1640995200000"
            #     }
            #   ]
            # }

            if "data" not in data or "arg" not in data:
                return None

            order_book_data = data["data"][0] if data["data"] else {}
            arg = data["arg"]
            symbol = arg.get("instId", "UNKNOWN")

            # Конвертируем обратно в стандартный формат
            normalized_symbol = self._denormalize_symbol(symbol)

            # Получаем timestamp
            timestamp_str = order_book_data.get("ts", str(int(time.time() * 1000)))
            timestamp_ms = int(timestamp_str)
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
                exchange="bitget",
                symbol=normalized_symbol,
                bids=bids,
                asks=asks,
                timestamp=timestamp,
                sequence_id=timestamp_ms,
            )

        except Exception as e:
            logger.error(f"Error parsing Bitget order book update: {e}")
            return None

    def _parse_price_volume_pairs(
        self, data: List[List[str]], base_currency: Currency, quote_currency: Currency
    ) -> List[Tuple[Price, Volume]]:
        """Парсинг пар цена-объем."""
        result = []
        for price_str, volume_str, *_ in data:
            try:
                price = Price(Decimal(price_str), base_currency, quote_currency)
                volume = Volume(Decimal(volume_str), quote_currency)
                result.append((price, volume))
            except Exception as e:
                logger.warning(f"Error parsing price-volume pair: {e}")
        return result

    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа для Bitget (BTCUSDT -> BTCUSDT_SPBL)."""
        # Bitget использует суффикс _SPBL для спотовой торговли
        if not symbol.endswith("_SPBL"):
            return f"{symbol}_SPBL"
        return symbol

    def _denormalize_symbol(self, symbol: str) -> str:
        """Денормализация символа из Bitget (BTCUSDT_SPBL -> BTCUSDT)."""
        return symbol.replace("_SPBL", "")

    async def _send_ping(self) -> None:
        """Отправка ping для поддержания соединения."""
        if self.websocket:
            try:
                ping_msg = {"op": "ping"}
                await self.websocket.send(json.dumps(ping_msg))
            except Exception as e:
                logger.error(f"Failed to send ping to Bitget: {e}")

    async def _reconnect(self) -> None:
        """Переподключение к WebSocket."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached for Bitget")
            return

        self.reconnect_attempts += 1
        logger.info(
            f"Attempting to reconnect to Bitget (attempt {self.reconnect_attempts})"
        )

        await asyncio.sleep(self.reconnect_delay * self.reconnect_attempts)

        if await self.connect():
            # Переподписываемся на символы
            for symbol in self.subscribed_symbols.copy():
                await self.subscribe(symbol)

    def get_status(self) -> Dict[str, Any]:
        """Получение статуса подключения."""
        return {
            "exchange": "bitget",
            "is_connected": self.is_connected,
            "subscribed_symbols": list(self.subscribed_symbols),
            "reconnect_attempts": self.reconnect_attempts,
            "last_ping": self.last_ping,
        }
