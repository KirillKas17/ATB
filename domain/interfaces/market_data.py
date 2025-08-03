"""
Протокол для работы с рыночными данными.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from decimal import Decimal

class MarketDataProtocol(Protocol):
    """Протокол для работы с рыночными данными."""
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение текущего тикера."""
        ...
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """Получение стакана заявок."""
        ...
    
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Получение последних сделок."""
        ...
    
    async def get_klines(self, symbol: str, interval: str, 
                        limit: int = 100, 
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Получение исторических данных (свечи)."""
        ...
    
    async def get_24hr_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение 24-часовой статистики."""
        ...
    
    async def get_price(self, symbol: str) -> Decimal:
        """Получение текущей цены."""
        ...
    
    async def get_symbols(self) -> List[str]:
        """Получение списка доступных символов."""
        ...
    
    async def subscribe_ticker(self, symbol: str, callback) -> None:
        """Подписка на обновления тикера."""
        ...
    
    async def subscribe_orderbook(self, symbol: str, callback) -> None:
        """Подписка на обновления стакана."""
        ...
    
    async def subscribe_trades(self, symbol: str, callback) -> None:
        """Подписка на обновления сделок."""
        ...
    
    async def unsubscribe(self, symbol: str, stream_type: str) -> None:
        """Отписка от потока данных."""
        ...

class BaseMarketDataProvider(ABC):
    """Базовый класс для провайдера рыночных данных."""
    
    def __init__(self, exchange: str):
        self.exchange = exchange
        self._subscriptions = {}
        self._connected = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Подключение к источнику данных."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Отключение от источника данных."""
        pass
    
    @property
    def is_connected(self) -> bool:
        """Статус подключения."""
        return self._connected
    
    @property
    def exchange_name(self) -> str:
        """Название биржи."""
        return self.exchange