"""
Репозиторий для работы с ордерами.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trading_pair import TradingPair
from domain.repositories.base_repository import BaseRepository, RepositoryError
from domain.type_definitions import OrderId
from domain.type_definitions.repository_types import EntityId, QueryOptions, QueryFilter


class OrderRepository(BaseRepository[Order]):
    """
    Репозиторий для работы с ордерами.
    Предоставляет методы для сохранения, получения, обновления и удаления ордеров,
    а также для поиска ордеров по различным критериям.
    """

    @abstractmethod
    async def save(self, order: Order) -> Order:
        """
        Сохранение ордера
        Args:
            order: Ордер для сохранения
        Returns:
            Сохраненный ордер
        Raises:
            RepositoryError: При ошибке сохранения
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: EntityId) -> Optional[Order]:
        """
        Получение ордера по ID
        Args:
            entity_id: ID ордера
        Returns:
            Ордер или None если не найден
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def get_by_trading_pair(
        self, trading_pair: TradingPair, status: Optional[OrderStatus] = None
    ) -> List[Order]:
        """
        Получение ордеров по торговой паре
        Args:
            trading_pair: Торговая пара
            status: Статус ордеров (опционально)
        Returns:
            Список ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def get_active_orders(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """
        Получение активных ордеров
        Args:
            trading_pair: Торговая пара (опционально)
        Returns:
            Список активных ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def get_orders_by_status(
        self, status: OrderStatus, limit: Optional[int] = None
    ) -> List[Order]:
        """
        Получение ордеров по статусу
        Args:
            status: Статус ордеров
            limit: Максимальное количество
        Returns:
            Список ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def get_orders_by_side(
        self, side: OrderSide, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """
        Получение ордеров по стороне
        Args:
            side: Сторона ордеров
            trading_pair: Торговая пара (опционально)
        Returns:
            Список ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def get_orders_by_type(
        self, order_type: OrderType, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        """
        Получение ордеров по типу
        Args:
            order_type: Тип ордеров
            trading_pair: Торговая пара (опционально)
        Returns:
            Список ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def get_orders_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[Order]:
        """
        Получение ордеров по диапазону дат
        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            trading_pair: Торговая пара (опционально)
        Returns:
            Список ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    @abstractmethod
    async def update(self, order: Order) -> Order:
        """
        Обновление ордера
        Args:
            order: Ордер для обновления
        Returns:
            Обновленный ордер
        Raises:
            RepositoryError: При ошибке обновления
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """
        Удаление ордера
        Args:
            entity_id: ID ордера
        Returns:
            True если удален, False если не найден
        Raises:
            RepositoryError: При ошибке удаления
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """
        Проверка существования ордера
        Args:
            entity_id: ID ордера
        Returns:
            True если существует, False если нет
        Raises:
            RepositoryError: При ошибке проверки
        """
        pass

    @abstractmethod
    async def count(
        self,
        filters: Optional[List[QueryFilter]] = None,
    ) -> int:
        """
        Подсчет количества ордеров
        Args:
            filters: Критерии фильтрации
        Returns:
            Количество ордеров
        Raises:
            RepositoryError: При ошибке подсчета
        """
        pass

    @abstractmethod
    async def get_all(
        self,
        options: Optional[QueryOptions] = None,
    ) -> List[Order]:
        """
        Получение всех ордеров с фильтрацией и пагинацией
        Args:
            options: Опции запроса
        Returns:
            Список ордеров
        Raises:
            RepositoryError: При ошибке получения
        """
        pass

    # Реализация базовых методов из BaseRepository
    async def find_by_id(self, id: EntityId) -> Optional[Order]:
        """Алиас для get_by_id для совместимости с BaseRepository."""
        return await self.get_by_id(id)

    async def find_all(
        self, options: Optional[QueryOptions] = None
    ) -> List[Order]:
        """Алиас для get_all для совместимости с BaseRepository."""
        return await self.get_all(options=options)

    async def find_by_criteria(
        self,
        filters: List[QueryFilter],
        options: Optional[QueryOptions] = None,
    ) -> List[Order]:
        """Алиас для get_all с фильтрами для совместимости с BaseRepository."""
        return await self.get_all(options=options)


class InMemoryOrderRepository(OrderRepository):
    """In-memory реализация репозитория ордеров для тестирования."""

    def __init__(self) -> None:
        self._orders: Dict[EntityId, Order] = {}

    async def save(self, order: Order) -> Order:
        self._orders[EntityId(order.id)] = order
        return order

    async def get_by_id(self, entity_id: EntityId) -> Optional[Order]:
        return self._orders.get(EntityId(entity_id))

    async def get_by_trading_pair(
        self, trading_pair: TradingPair, status: Optional[OrderStatus] = None
    ) -> List[Order]:
        orders = [
            order
            for order in self._orders.values()
            if order.trading_pair == trading_pair
        ]
        if status:
            orders = [order for order in orders if order.status == status]
        return orders

    async def get_active_orders(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        active_statuses = [
            OrderStatus.PENDING,
            OrderStatus.OPEN,
            OrderStatus.PARTIALLY_FILLED,
        ]
        orders = [
            order for order in self._orders.values() if order.status in active_statuses
        ]
        if trading_pair:
            orders = [order for order in orders if order.trading_pair == trading_pair]
        return orders

    async def get_orders_by_status(
        self, status: OrderStatus, limit: Optional[int] = None
    ) -> List[Order]:
        orders = [order for order in self._orders.values() if order.status == status]
        if limit:
            orders = orders[:limit]
        return orders

    async def get_orders_by_side(
        self, side: OrderSide, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        orders = [order for order in self._orders.values() if order.side == side]
        if trading_pair:
            orders = [order for order in orders if order.trading_pair == trading_pair]
        return orders

    async def get_orders_by_type(
        self, order_type: OrderType, trading_pair: Optional[TradingPair] = None
    ) -> List[Order]:
        orders = [
            order for order in self._orders.values() if order.order_type == order_type
        ]
        if trading_pair:
            orders = [order for order in orders if order.trading_pair == trading_pair]
        return orders

    async def get_orders_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[Order]:
        orders = [
            order
            for order in self._orders.values()
            if start_date <= order.created_at.to_datetime().replace(tzinfo=None) <= end_date
        ]
        if trading_pair:
            orders = [order for order in orders if order.trading_pair == trading_pair]
        return orders

    async def update(self, order: Order) -> Order:
        eid = EntityId(order.id)
        if eid not in self._orders:
            raise RepositoryError(f"Order {order.id} not found")
        self._orders[eid] = order
        return order

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        eid = EntityId(UUID(entity_id) if isinstance(entity_id, str) else entity_id)
        if eid in self._orders:
            del self._orders[eid]
            return True
        return False

    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        eid = EntityId(UUID(entity_id) if isinstance(entity_id, str) else entity_id)
        return eid in self._orders

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        if not filters:
            return len(self._orders)
        filtered_orders = list(self._orders.values())
        for filter in filters:
            if hasattr(filter, 'apply'):
                filtered_orders = [order for order in filtered_orders if filter.apply(order)]
        return len(filtered_orders)

    async def get_all(
        self, options: Optional[QueryOptions] = None
    ) -> List[Order]:
        orders = list(self._orders.values())
        # Применяем фильтры
        if options:
            for filter in getattr(options, 'filters', []):
                if hasattr(filter, 'apply'):
                    orders = [order for order in orders if filter.apply(order)]
        # Сортировка
        if options and getattr(options, 'sort_orders', []):
            for sort in options.sort_orders:
                orders.sort(key=lambda x: getattr(x, sort.field, 0), reverse=(sort.direction == 'desc'))
        # Пагинация
        if options and getattr(options, 'pagination', None):
            pag = options.pagination
            offset = getattr(pag, 'offset', 0) or 0
            limit = getattr(pag, 'limit', None)
            orders = orders[offset:]
            if limit is not None:
                orders = orders[:limit]
        return orders
