# -*- coding: utf-8 -*-
"""Coinbase exchange connector for market data."""
from typing import Any, Dict, Optional

from loguru import logger

from domain.type_definitions.value_object_types import CurrencyCode
from domain.value_objects.currency import Currency
from domain.value_objects.timestamp import Timestamp
from shared.models.orderbook import OrderBookUpdate

from .base_connector import BaseExchangeConnector


class CoinbaseConnector(BaseExchangeConnector):
    """Коннектор для Coinbase с промышленной реализацией."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None) -> None:
        super().__init__("coinbase", api_key, api_secret)
        self.base_url = "wss://ws-feed.exchange.coinbase.com"

    def get_websocket_url(self, symbol: str) -> str:
        """Получение URL для WebSocket подключения."""
        return self.base_url

    def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
        """Получение сообщения для подписки на ордербук."""
        return {"type": "subscribe", "product_ids": [symbol], "channels": ["level2"]}

    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа для Coinbase."""
        return symbol.upper()

    def parse_order_book_update(
        self, message: Dict[str, Any]
    ) -> Optional[OrderBookUpdate]:
        """Парсинг обновления ордербука из сообщения Coinbase."""
        try:
            # Coinbase отправляет данные в формате:
            # {
            #   "type": "snapshot",
            #   "product_id": "BTC-USD",
            #   "bids": [["price", "size"], ...],
            #   "asks": [["price", "size"], ...],
            #   "sequence": 123456
            # }
            # Проверяем тип сообщения
            if message.get("type") not in ["snapshot", "l2update"]:
                return None
            # Извлекаем символ
            symbol = message.get("product_id", "UNKNOWN")
            # Извлекаем timestamp
            time_str = message.get("time")
            if time_str:
                timestamp = Timestamp.from_iso(time_str.replace("Z", "+00:00"))
            else:
                timestamp = self._get_current_timestamp()
            # Извлекаем sequence_id
            sequence_id = message.get("sequence")
            # Извлекаем bids и asks
            bids_data = message.get("bids", [])
            asks_data = message.get("asks", [])
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
                meta={"source": "coinbase"},
            )
        except Exception as e:
            logger.error(f"Error parsing Coinbase order book update: {e}")
            return None
