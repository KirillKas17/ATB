# -*- coding: utf-8 -*-
"""Base connector for exchange market data."""
import asyncio
import json
from abc import ABC, abstractmethod
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncGenerator, Dict, Optional, Union

from loguru import logger

from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.volume import Volume
from shared.models.orderbook import OrderBookUpdate

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    websockets = None
    WEBSOCKETS_AVAILABLE = False


class BaseExchangeConnector(ABC):
    """
    Базовый класс для коннекторов бирж с промышленной реализацией WebSocket.
    """

    exchange_name: str
    api_key: Optional[str]
    api_secret: Optional[str]
    websocket_url: str
    is_connected: bool
    reconnect_attempts: int
    max_reconnect_attempts: int
    reconnect_delay: float
    _websocket: Optional[Any]

    def __init__(
        self,
        exchange_name: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.exchange_name = exchange_name
        self.api_key = api_key
        self.api_secret = api_secret
        self.websocket_url = ""
        self.is_connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.reconnect_delay = 1.0
        self._websocket = None

    @property
    def websocket(self) -> Optional[Any]:
        """Получение WebSocket объекта."""
        return self._websocket

    @abstractmethod
    def get_websocket_url(self, symbol: str) -> str:
        """Получение URL для WebSocket подключения."""

    @abstractmethod
    def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
        """Получение сообщения для подписки на ордербук."""

    @abstractmethod
    def parse_order_book_update(
        self, message: Dict[str, Any]
    ) -> Optional[OrderBookUpdate]:
        """Парсинг обновления ордербука из сообщения биржи."""

    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа (базовая реализация)."""
        return symbol.lower()

    async def connect(self, symbol: str) -> bool:
        """
        Подключение к WebSocket биржи.
        """
        if not WEBSOCKETS_AVAILABLE:
            logger.error("WebSockets not available. Установите websockets.")
            return False
        try:
            self.websocket_url = self.get_websocket_url(symbol)
            logger.info(
                f"Connecting to {self.exchange_name} WebSocket: {self.websocket_url}"
            )
            self._websocket = await websockets.connect(self.websocket_url)
            self.is_connected = True
            self.reconnect_attempts = 0
            # Отправляем сообщение подписки
            sub_msg = self.get_subscription_message(symbol)
            await self._websocket.send(json.dumps(sub_msg))
            logger.info(f"Subscribed to {symbol} order book on {self.exchange_name}")
            return True
        except Exception as e:
            logger.error(f"Ошибка в base_connector: {e}")
            self.is_connected = False
            return False

    async def disconnect(self) -> None:
        """
        Отключение от WebSocket.
        """
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"Error closing websocket: {e}")
        self.is_connected = False
        self._websocket = None
        logger.info(f"Disconnected from {self.exchange_name}")

    async def stream_order_book(
        self, symbol: str, callback: Optional[Any] = None
    ) -> AsyncGenerator[OrderBookUpdate, None]:
        """
        Поток обновлений ордербука (реализация через WebSocket).
        """
        while True:
            try:
                if not self.is_connected:
                    if not await self.connect(symbol):
                        await asyncio.sleep(self.reconnect_delay)
                        continue
                assert self._websocket is not None
                message = await self._websocket.recv()
                data = json.loads(message)
                update = self.parse_order_book_update(data)
                if update:
                    if callback:
                        await callback(update)
                    yield update
            except (asyncio.CancelledError, GeneratorExit):
                break
            except Exception as e:
                logger.error(f"Error in {self.exchange_name} stream: {e}")
                self.is_connected = False
                await asyncio.sleep(self.reconnect_delay)

    def _normalize_currency(self, currency: Union[Currency, str]) -> Optional[Currency]:
        """Нормализация валюты."""
        if isinstance(currency, str):
            return Currency.from_string(currency) or Currency.USD
        elif isinstance(currency, Currency):
            return currency
        else:
            # Обработка других типов валют
            try:
                return Currency(currency)
            except (ValueError, TypeError, AttributeError):
                logger.error(f"Invalid currency type: {type(currency)}")
                return None

    def _parse_price_volume_pairs(
        self,
        data: list,
        base_currency: Union[Currency, str],
        quote_currency: Union[Currency, str],
    ) -> list[tuple[Price, Volume]]:
        """
        Парсинг пар цена-объем.
        """
        # Нормализация валют
        normalized_base_currency = self._normalize_currency(base_currency)
        normalized_quote_currency = self._normalize_currency(quote_currency)
        
        if normalized_base_currency is None or normalized_quote_currency is None:
            return []
        
        # Обработка данных
        result = []
        for price_str, volume_str in data:
            try:
                # Проверяем, что строки содержат только цифры и точку
                if not str(price_str).replace(".", "").replace("-", "").isdigit():
                    logger.warning(f"Invalid price format: {price_str}")
                    continue
                if not str(volume_str).replace(".", "").replace("-", "").isdigit():
                    logger.warning(f"Invalid volume format: {volume_str}")
                    continue
                price = Decimal(str(price_str))
                volume = Decimal(str(volume_str))
                result.append(
                    (
                        Price(price, normalized_base_currency, normalized_quote_currency),
                        Volume(volume, normalized_quote_currency),
                    )
                )
            except (ValueError, TypeError, InvalidOperation):
                logger.warning(f"Invalid price/volume data: {price_str}, {volume_str}")
                continue
            return result

    def _get_current_timestamp(self) -> Timestamp:
        """
        Получение текущего времени.
        """
        return Timestamp.now()
