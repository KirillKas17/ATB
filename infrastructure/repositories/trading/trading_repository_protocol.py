"""
Протокол торгового репозитория.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from domain.entities.order import Order, OrderId, OrderSide, OrderStatus
from domain.entities.position import Position, PositionId
from domain.entities.trading import Trade
from domain.entities.account import Account
from domain.entities.trading_pair import TradingPair
from domain.type_definitions import Symbol
from domain.value_objects.money import Money


class TradingRepositoryProtocol(ABC):
    """Протокол торгового репозитория."""

    @abstractmethod
    async def add_order(self, order: Order) -> "RepositoryResult":
        """Добавление ордера."""
        pass

    @abstractmethod
    async def get_order(self, order_id: Union[str, UUID]) -> Optional[Order]:
        """Получение ордера по ID."""
        pass

    @abstractmethod
    async def update_order(
        self, order_id: Union[str, UUID], updates: Dict[str, Any]
    ) -> "RepositoryResult":
        """Обновление ордера."""
        pass

    @abstractmethod
    async def delete_order(self, order_id: Union[str, UUID]) -> "RepositoryResult":
        """Удаление ордера."""
        pass

    @abstractmethod
    async def list_orders(
        self,
        account_id: Optional[Union[str, UUID]] = None,
        trading_pair_id: Optional[Union[str, UUID]] = None,
        status: Optional[OrderStatus] = None,
        side: Optional[OrderSide] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> "RepositoryResult":
        """Список ордеров с фильтрацией."""
        pass

    @abstractmethod
    async def add_position(self, position: Position) -> "RepositoryResult":
        """Добавление позиции."""
        pass

    @abstractmethod
    async def get_position(self, position_id: Union[str, UUID]) -> "RepositoryResult":
        """Получение позиции по ID."""
        pass

    @abstractmethod
    async def update_position(
        self, position_id: Union[str, UUID], updates: Dict[str, Any]
    ) -> "RepositoryResult":
        """Обновление позиции."""
        pass

    @abstractmethod
    async def delete_position(self, position_id: Union[str, UUID]) -> "RepositoryResult":
        """Удаление позиции."""
        pass

    @abstractmethod
    async def list_positions(
        self,
        account_id: Optional[Union[str, UUID]] = None,
        trading_pair_id: Optional[Union[str, UUID]] = None,
        side: Optional[OrderSide] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> "RepositoryResult":
        """Список позиций с фильтрацией."""
        pass

    @abstractmethod
    async def save_trade(self, trade: Trade) -> Trade:
        """Сохранение сделки."""
        pass

    @abstractmethod
    async def get_trades_by_order(self, order_id: OrderId) -> List[Trade]:
        """Получение сделок по ордеру."""
        pass

    @abstractmethod
    async def get_trades_by_symbol(
        self,
        symbol: Symbol,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """Получение сделок по символу."""
        pass

    @abstractmethod
    async def get_trade(
        self, 
        symbol: Optional[Symbol] = None, 
        limit: int = 100,
        account_id: Optional[str] = None
    ) -> List[Trade]:
        """Получение сделок."""
        pass

    @abstractmethod
    async def get_balance(self, account_id: Optional[str] = None) -> Dict[str, Money]:
        """Получение баланса."""
        pass

    @abstractmethod
    async def save_account(self, account: Account) -> Account:
        """Сохранение аккаунта."""
        pass

    @abstractmethod
    async def get_account(self, account_id: str) -> Optional[Account]:
        """Получение аккаунта."""
        pass

    @abstractmethod
    async def update_account_balance(
        self, 
        account_id: str, 
        currency: str, 
        amount: Money
    ) -> bool:
        """Обновление баланса аккаунта."""
        pass

    @abstractmethod
    async def get_trading_metrics(self) -> "RepositoryResult":
        """Получение торговых метрик."""
        pass

    @abstractmethod
    async def clear_all_data(self) -> "RepositoryResult":
        """Очистка всех данных."""
        pass


class RepositoryResult:
    """Результат операции репозитория."""
    
    def __init__(self, success: bool, data: Any = None, error_message: str = "") -> None:
        self.success = success
        self.data = data
        self.error_message = error_message