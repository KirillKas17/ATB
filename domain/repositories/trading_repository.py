"""
Репозиторий торговых операций.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Protocol, Any, Union
from uuid import UUID

from domain.entities.order import Order, OrderStatus
from domain.entities.trading import Position, Trade, TradingSession
from domain.protocols.repository_protocol import RepositoryProtocol
from domain.types import EntityId, OrderId, PositionId
from domain.value_objects.money import Money


class TradingRepository(RepositoryProtocol):
    """Репозиторий для работы с торговыми операциями."""

    @abstractmethod
    async def save_order(self, order: Order) -> Order:
        """Сохранить ордер"""

    @abstractmethod
    async def get_order(self, order_id: EntityId) -> Optional[Order]:
        """Получить ордер по ID"""

    @abstractmethod
    async def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Получить ордера по статусу"""

    @abstractmethod
    async def get_orders_by_trading_pair(self, trading_pair: str) -> List[Order]:
        """Получить ордера по торговой паре"""

    @abstractmethod
    async def get_active_orders(self) -> List[Order]:
        """Получить активные ордера"""

    @abstractmethod
    async def update_order(self, order: Order) -> Order:
        """Обновить ордер"""

    @abstractmethod
    async def delete_order(self, order_id: EntityId) -> bool:
        """Удалить ордер"""

    @abstractmethod
    async def save_trade(self, trade: Trade) -> Trade:
        """Сохранить сделку"""

    @abstractmethod
    async def get_trade(self, trade_id: EntityId) -> Optional[Trade]:
        """Получить сделку по ID"""

    @abstractmethod
    async def get_trades_by_order(self, order_id: EntityId) -> List[Trade]:
        """Получить сделки по ордеру"""

    @abstractmethod
    async def get_trades_by_trading_pair(
        self,
        trading_pair: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Trade]:
        """Получить сделки по торговой паре"""

    @abstractmethod
    async def save_trading_session(self, session: TradingSession) -> TradingSession:
        """Сохранить торговую сессию"""

    @abstractmethod
    async def get_trading_session(
        self, session_id: EntityId
    ) -> Optional[TradingSession]:
        """Получить торговую сессию по ID"""

    @abstractmethod
    async def get_active_trading_session(self) -> Optional[TradingSession]:
        """Получить активную торговую сессию"""

    @abstractmethod
    async def update_trading_session(self, session: TradingSession) -> TradingSession:
        """Обновить торговую сессию"""

    @abstractmethod
    async def get_balance(self, account_id: Optional[str] = None) -> Dict[str, "Money"]:
        """Получить баланс аккаунта"""


class OrderRepository(Protocol):
    """Протокол репозитория ордеров."""

    async def save(self, order: Order) -> bool:
        """Сохранение ордера."""
        ...

    async def get_by_id(self, order_id: UUID) -> Optional[Order]:
        """Получение ордера по ID."""
        ...

    async def get_by_symbol(
        self, symbol: str, limit: Optional[int] = None
    ) -> List[Order]:
        """Получение ордеров по символу."""
        ...

    async def get_active(self, symbol: Optional[str] = None) -> List[Order]:
        """Получение активных ордеров."""
        ...

    async def update(self, order: Order) -> bool:
        """Обновление ордера."""
        ...

    async def delete(self, order_id: UUID) -> bool:
        """Удаление ордера."""
        ...


class TradeRepository(Protocol):
    """Протокол репозитория сделок."""

    async def save(self, trade: Trade) -> bool:
        """Сохранение сделки."""
        ...

    async def get_by_id(self, trade_id: UUID) -> Optional[Trade]:
        """Получение сделки по ID."""
        ...

    async def get_by_symbol(
        self, symbol: str, limit: Optional[int] = None
    ) -> List[Trade]:
        """Получение сделок по символу."""
        ...

    async def get_by_date_range(
        self, start_date: datetime, end_date: datetime, symbol: Optional[str] = None
    ) -> List[Trade]:
        """Получение сделок по диапазону дат."""
        ...

    async def get_by_order(self, order_id: UUID) -> List[Trade]:
        """Получение сделок по ордеру."""
        ...


class PositionRepository(Protocol):
    """Протокол репозитория позиций."""

    async def save(self, position: Position) -> bool:
        """Сохранение позиции."""
        ...

    async def get_by_id(self, position_id: UUID) -> Optional[Position]:
        """Получение позиции по ID."""
        ...

    async def get_by_symbol(self, symbol: str) -> List[Position]:
        """Получение позиций по символу."""
        ...

    async def get_active(self, symbol: Optional[str] = None) -> List[Position]:
        """Получение активных позиций."""
        ...

    async def update(self, position: Position) -> bool:
        """Обновление позиции."""
        ...

    async def delete(self, position_id: UUID) -> bool:
        """Удаление позиции."""
        ...


class InMemoryTradingRepository(TradingRepository):
    """In-memory реализация репозитория торговых сущностей"""

    def __init__(self) -> None:
        """Инициализация репозитория."""
        self._orders: Dict[OrderId, Order] = {}
        self._trades: Dict[EntityId, Trade] = {}
        self._positions: Dict[PositionId, Position] = {}
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._metrics: Dict[str, Any] = {}

    async def save(self, entity: Order) -> Order:
        """Сохранить ордер."""
        self._orders[OrderId(entity.id)] = entity
        return entity

    async def get_by_id(self, entity_id: Any) -> Optional[Order]:
        """Получить ордер по ID."""
        return self._orders.get(OrderId(entity_id))

    async def update(self, entity: Order) -> Order:
        """Обновить ордер."""
        if OrderId(entity.id) in self._orders:
            self._orders[OrderId(entity.id)] = entity
        return entity

    async def delete(self, entity_id: Any) -> bool:
        """Удалить ордер."""
        if OrderId(entity_id) in self._orders:
            del self._orders[OrderId(entity_id)]
            return True
        return False
