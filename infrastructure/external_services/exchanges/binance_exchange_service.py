"""
Сервис для Binance - Production Ready
"""

import json
from datetime import datetime
from typing import Any, Dict

from domain.type_definitions.external_service_types import ExchangeName

from .base_exchange_service import BaseExchangeService
from .config import ExchangeServiceConfig


class BinanceExchangeService(BaseExchangeService):
    """Реализация сервиса для Binance."""

    def __init__(self, config: ExchangeServiceConfig) -> None:
        super().__init__(config)
        self._exchange_name = ExchangeName("binance")

    def _get_websocket_url(self) -> str:
        """Получить URL для WebSocket Binance."""
        if self.config.credentials.testnet:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"

    async def _subscribe_websocket_channels(self) -> None:
        """Подписка на WebSocket каналы Binance."""
        if not self.websocket:
            return
        else:
            # Подписка на тикеры
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": ["btcusdt@ticker", "ethusdt@ticker"],
                "id": 1,
            }
            await self.websocket.send(json.dumps(subscribe_message))

    async def _process_websocket_message(self, data: Dict[str, Any]) -> None:
        """Обработка WebSocket сообщения Binance."""
        if "s" in data and "c" in data:  # symbol и close price
            symbol = data["s"]
            cache_key = f"ticker_{symbol}"
            ticker_info = {
                "symbol": symbol,
                "price": float(data.get("c", 0)),
                "volume": float(data.get("v", 0)),
                "change": float(data.get("P", 0)),
                "high": float(data.get("h", 0)),
                "low": float(data.get("l", 0)),
                "timestamp": datetime.now().timestamp(),
            }
            await self.cache.set(cache_key, ticker_info) 