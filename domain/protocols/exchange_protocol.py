"""
Протоколы для работы с биржами.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Protocol, runtime_checkable
from uuid import UUID

import websockets

from domain.entities.order import Order, OrderSide, OrderType
from domain.entities.position import Position
from domain.entities.trade import Trade
from domain.exceptions.protocol_exceptions import (
    ConnectionError,
    ExchangeRateLimitError,
    OrderNotFoundError,
    ProtocolError,
)
from domain.type_definitions import OrderId, PortfolioId, Symbol
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume

from domain.entities.market import MarketData, OrderBook
from domain.entities.account import Balance
from domain.exceptions import (
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderError,
    NetworkError,
    TimeoutError,
)


@dataclass
class ExchangeConfig:
    """Конфигурация биржи для domain слоя."""
    api_key: str
    api_secret: str
    sandbox: bool = True
    timeout: int = 30
    rate_limit: int = 100
    retry_attempts: int = 3


__all__ = [
    "ExchangeProtocol",
    "ExchangeConfig",
    "ConnectionStatus",
    "MarketData",
    "OrderBook",
    "Trade",
    "Order",
    "Position",
    "Balance",
    "ExchangeError",
    "ExchangeRateLimitError",
    "InsufficientFundsError",
    "OrderNotFoundError",
    "InvalidOrderError",
    "NetworkError",
    "TimeoutError",
]


class ConnectionStatus(Enum):
    """Статус подключения к бирже."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@runtime_checkable
class ExchangeProtocol(Protocol):
    """Протокол для работы с биржей."""
    
    async def create_order(self, order: Order) -> Dict[str, Any]:
        """Создание ордера."""
        ...
    
    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        ...
    
    async def fetch_order(self, order_id: str) -> Dict[str, Any]:
        """Получение информации об ордере."""
        ...
    
    async def fetch_open_orders(self) -> List[Dict[str, Any]]:
        """Получение открытых ордеров."""
        ...
    
    async def fetch_balance(self) -> Dict[str, Any]:
        """Получение баланса."""
        ...
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение тикера."""
        ...
    
    async def fetch_order_book(self, symbol: str) -> Dict[str, Any]:
        """Получение ордербука."""
        ...
