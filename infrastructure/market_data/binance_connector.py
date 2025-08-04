# -*- coding: utf-8 -*-
"""Binance exchange connector for market data."""

import time
from typing import Any, Dict, Optional

from loguru import logger

from domain.type_definitions.value_object_types import CurrencyCode
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp
from shared.models.orderbook import OrderBookUpdate

from .base_connector import BaseExchangeConnector


class BinanceConnector(BaseExchangeConnector):
    """Коннектор для Binance с промышленной реализацией."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__("binance", api_key, api_secret)
        self.base_url = "wss://stream.binance.com:9443/ws"

    def get_websocket_url(self, symbol: str) -> str:
        """Получение URL для WebSocket подключения."""
        normalized_symbol = self._normalize_symbol(symbol)
        return f"wss://stream.binance.com:9443/ws/{normalized_symbol}@depth20@100ms"

    def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
        """Получение сообщения для подписки на ордербук."""
        return {
            "method": "SUBSCRIBE",
            "params": [f"{symbol.lower()}@depth20@100ms"],
            "id": 1,
        }

    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа для Binance."""
        return symbol.lower()

    def parse_order_book_update(
        self, message: Dict[str, Any]
    ) -> Optional[OrderBookUpdate]:
        """Парсинг обновления ордербука из сообщения Binance."""
        try:
            # Извлекаем данные из структуры Binance
            if "data" not in message:
                return None

            data = message["data"]
            if "e" not in data or data["e"] != "depthUpdate":
                return None

            # Извлекаем символ
            symbol = data.get("s", "UNKNOWN")

            # Извлекаем timestamp
            timestamp = Timestamp.from_unix_ms(data.get("E", int(time.time() * 1000)))

            # Извлекаем sequence_id
            sequence_id = data.get("u")

            # Извлекаем bids и asks
            bids_data = data.get("b", [])
            asks_data = data.get("a", [])

            # Если нет данных bids или asks, возвращаем None
            if not bids_data and not asks_data:
                return None

            # Парсим bids и asks
            base_currency = Currency(CurrencyCode("BTC"))
            quote_currency = Currency(CurrencyCode("USD"))

            bids = self._parse_price_volume_pairs(
                bids_data, base_currency, quote_currency
            )
            asks = self._parse_price_volume_pairs(
                asks_data, base_currency, quote_currency
            )

            return OrderBookUpdate(
                exchange=self.exchange_name,
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=timestamp,
                sequence_id=sequence_id,
                meta={"source": "binance"},
            )

        except Exception as e:
            logger.error(f"Error parsing Binance order book update: {e}")
            return None
