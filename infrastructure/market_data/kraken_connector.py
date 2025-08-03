# -*- coding: utf-8 -*-
"""Kraken exchange connector for market data."""
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from domain.types.value_object_types import CurrencyCode
from domain.value_objects.currency import Currency
from shared.models.orderbook import OrderBookUpdate

from .base_connector import BaseExchangeConnector


class KrakenConnector(BaseExchangeConnector):
    """Коннектор для Kraken с промышленной реализацией."""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        super().__init__("kraken", api_key, api_secret)
        self.base_url = "wss://ws.kraken.com"
        self._current_symbol: Optional[str] = None

    def get_websocket_url(self, symbol: str) -> str:
        """Получение URL для WebSocket подключения."""
        self._current_symbol = symbol
        return self.base_url

    def get_subscription_message(self, symbol: str) -> Dict[str, Any]:
        """Получение сообщения для подписки на ордербук."""
        return {
            "event": "subscribe",
            "pair": [symbol],
            "subscription": {"name": "book", "depth": 10},
        }

    def _convert_symbol_to_kraken_format(self, symbol: str) -> str:
        """Преобразование символа в формат Kraken."""
        # Простое преобразование, можно расширить
        if "BTC" in symbol:
            return "XBT/USD"
        elif "ETH" in symbol:
            return "ETH/USD"
        else:
            return symbol.replace("USDT", "/USD")

    def _normalize_symbol(self, symbol: str) -> str:
        """Нормализация символа для Kraken."""
        return symbol.upper()

    def parse_order_book_update(
        self, message: Union[Dict[str, Any], List[Any]]
    ) -> Optional[OrderBookUpdate]:
        """Парсинг обновления ордербука из сообщения Kraken."""
        try:
            # Kraken отправляет данные в формате:
            # [channelID, {
            #   "a": [["price", "volume", "timestamp"], ...],
            #   "b": [["price", "volume", "timestamp"], ...]
            # }, channel_name, pair]
            # Проверяем структуру сообщения
            if not isinstance(message, list) or len(message) < 4:
                return None
            # Извлекаем данные
            data = message[1]
            symbol = message[3]
            # Извлекаем timestamp
            timestamp = self._get_current_timestamp()
            # Извлекаем bids и asks
            bids_data = data.get("b", [])
            asks_data = data.get("a", [])
            # Если нет данных bids или asks, возвращаем None
            if not bids_data and not asks_data:
                return None
            # Парсим bids и asks (убираем timestamp из данных)
            base_currency = Currency(CurrencyCode("BTC"))
            quote_currency = Currency(CurrencyCode("USD"))
            # Обрабатываем данные Kraken (цена, объем, timestamp)
            bids = []
            for bid in bids_data:
                if len(bid) >= 2:
                    bids.append([bid[0], bid[1]])  # Только цена и объем
            asks = []
            for ask in asks_data:
                if len(ask) >= 2:
                    asks.append([ask[0], ask[1]])  # Только цена и объем
            bids_parsed = self._parse_price_volume_pairs(
                bids, base_currency, quote_currency
            )
            asks_parsed = self._parse_price_volume_pairs(
                asks, base_currency, quote_currency
            )
            return OrderBookUpdate(
                exchange=self.exchange_name,
                symbol=symbol,
                bids=bids_parsed,
                asks=asks_parsed,
                timestamp=timestamp,
                sequence_id=None,
                meta={"source": "kraken"},
            )
        except Exception as e:
            logger.error(f"Error parsing Kraken order book update: {e}")
            return None
