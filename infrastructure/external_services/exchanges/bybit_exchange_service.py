"""
Сервис для Bybit - Production Ready
"""

import json
from datetime import datetime
from typing import Any, Dict

from domain.types.external_service_types import ExchangeName

from .base_exchange_service import BaseExchangeService
from .config import ExchangeServiceConfig


class BybitExchangeService(BaseExchangeService):
    """Реализация сервиса для Bybit."""

    def __init__(self, config: ExchangeServiceConfig):
        super().__init__(config)
        self._exchange_name = ExchangeName("bybit")

    def _get_websocket_url(self) -> str:
        """Получить URL для WebSocket Bybit."""
        if self.config.credentials.testnet:
            return "wss://testnet.stream.bybit.com/v5/public/spot"
        return "wss://stream.bybit.com/v5/public/spot"

    async def _subscribe_websocket_channels(self) -> None:
        """Подписка на WebSocket каналы Bybit."""
        if not self.websocket:
            return
        else:
            # Подписка на тикеры
            subscribe_message = {
                "op": "subscribe",
                "args": ["tickers.BTCUSDT", "tickers.ETHUSDT"],
            }
            await self.websocket.send(json.dumps(subscribe_message))

    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Обработка WebSocket сообщения Bybit."""
        if "topic" in data and "data" in data:
            topic = data["topic"]
            if topic.startswith("tickers."):
                await self._process_ticker_data(data["data"])

    async def _process_ticker_data(self, ticker_data: Dict[str, Any]) -> None:
        """Обработка данных тикера."""
        symbol = ticker_data.get("symbol", "")
        cache_key = f"ticker_{symbol}"
        ticker_info = {
            "symbol": symbol,
            "price": float(ticker_data.get("lastPrice", 0)),
            "volume": float(ticker_data.get("volume24h", 0)),
            "change": float(ticker_data.get("price24hPcnt", 0)),
            "high": float(ticker_data.get("highPrice24h", 0)),
            "low": float(ticker_data.get("lowPrice24h", 0)),
            "timestamp": datetime.now().timestamp(),
        }
        await self.cache.set(cache_key, ticker_info)
