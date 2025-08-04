"""
Брокерские интерфейсы для исполнения ордеров.
Включает:
- Абстрактные классы брокеров
- Протоколы для взаимодействия с биржами
- Реализации для различных бирж
- Обработку ошибок и повторные попытки
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger

from domain.type_definitions import Symbol
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume as Quantity
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.currency import Currency


class OrderType(Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Стороны ордера."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Статусы ордера."""

    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class OrderRequest:
    """Запрос на создание ордера."""

    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price] = None
    stop_price: Optional[Price] = None
    time_in_force: str = "GTC"
    client_order_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OrderResponse:
    """Ответ на создание ордера."""

    order_id: str
    client_order_id: Optional[str]
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    quantity: Quantity
    price: Optional[Price]
    status: OrderStatus
    timestamp: Timestamp
    metadata: Dict[str, Any]


@dataclass
class OrderUpdate:
    """Обновление статуса ордера."""

    order_id: str
    status: OrderStatus
    filled_quantity: Quantity
    remaining_quantity: Quantity
    average_price: Optional[Price]
    timestamp: Timestamp
    metadata: Dict[str, Any]


@dataclass
class Trade:
    """Информация о сделке."""

    trade_id: str
    order_id: str
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    price: Price
    timestamp: Timestamp
    fee: Optional[Decimal]
    fee_currency: Optional[str]
    metadata: Dict[str, Any]


@runtime_checkable
class BrokerProtocol(Protocol):
    """Протокол для брокерского интерфейса."""

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Разместить ордер."""
        ...

    async def cancel_order(self, order_id: str, symbol: Symbol) -> bool:
        """Отменить ордер."""
        ...

    async def get_order_status(self, order_id: str, symbol: Symbol) -> OrderResponse:
        """Получить статус ордера."""
        ...

    async def get_open_orders(
        self, symbol: Optional[Symbol] = None
    ) -> List[OrderResponse]:
        """Получить открытые ордера."""
        ...

    async def get_trades(self, order_id: str) -> List[Trade]:
        """Получить сделки по ордеру."""
        ...

    async def get_account_info(self) -> Dict[str, Any]:
        """Получить информацию об аккаунте."""
        ...


class BaseBroker(ABC):
    """Базовый класс для брокерских интерфейсов."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", "unknown")
        self.api_key = config.get("api_key")
        self.secret_key = config.get("secret_key")
        self.testnet = config.get("testnet", False)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)
        self.max_retries = config.get("max_retries", 3)

    @abstractmethod
    async def _place_order_impl(self, request: OrderRequest) -> OrderResponse:
        """Реализация размещения ордера."""
        pass

    @abstractmethod
    async def _cancel_order_impl(self, order_id: str, symbol: Symbol) -> bool:
        """Реализация отмены ордера."""
        pass

    @abstractmethod
    async def _get_order_status_impl(
        self, order_id: str, symbol: Symbol
    ) -> OrderResponse:
        """Реализация получения статуса ордера."""
        pass

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Разместить ордер с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit_delay)
                return await self._place_order_impl(request)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)  # Экспоненциальная задержка
        raise Exception("Max retries exceeded")

    async def cancel_order(self, order_id: str, symbol: Symbol) -> bool:
        """Отменить ордер с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit_delay)
                return await self._cancel_order_impl(order_id, symbol)
            except Exception as e:
                logger.warning(
                    f"Cancel attempt {attempt + 1} failed for {self.name}: {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        return False

    async def get_order_status(self, order_id: str, symbol: Symbol) -> OrderResponse:
        """Получить статус ордера с повторными попытками."""
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit_delay)
                return await self._get_order_status_impl(order_id, symbol)
            except Exception as e:
                logger.warning(
                    f"Status check attempt {attempt + 1} failed for {self.name}: {e}"
                )
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2**attempt)
        raise Exception("Max retries exceeded")

    async def get_open_orders(
        self, symbol: Optional[Symbol] = None
    ) -> List[OrderResponse]:
        """Получить открытые ордера."""
        # Базовая реализация - должна быть переопределена
        return []

    async def get_trades(self, order_id: str) -> List[Trade]:
        """Получить сделки по ордеру."""
        # Базовая реализация - должна быть переопределена
        return []

    async def get_account_info(self) -> Dict[str, Any]:
        """Получить информацию об аккаунте."""
        # Базовая реализация - должна быть переопределена
        return {}


class MockBroker(BaseBroker):
    """Мок-брокер для тестирования."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        self.orders: Dict[str, OrderResponse] = {}
        self.order_counter = 0

    async def _place_order_impl(self, request: OrderRequest) -> OrderResponse:
        """Мок-реализация размещения ордера."""
        self.order_counter += 1
        order_id = f"mock_order_{self.order_counter}"
        response = OrderResponse(
            order_id=order_id,
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            status=OrderStatus.PENDING,
            timestamp=Timestamp.now(),
            metadata=request.metadata or {},
        )
        self.orders[order_id] = response
        logger.info(f"Mock order placed: {order_id}")
        return response

    async def _cancel_order_impl(self, order_id: str, symbol: Symbol) -> bool:
        """Мок-реализация отмены ордера."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            logger.info(f"Mock order cancelled: {order_id}")
            return True
        return False

    async def _get_order_status_impl(
        self, order_id: str, symbol: Symbol
    ) -> OrderResponse:
        """Мок-реализация получения статуса ордера."""
        if order_id in self.orders:
            return self.orders[order_id]
        raise ValueError(f"Order not found: {order_id}")

    async def get_open_orders(
        self, symbol: Optional[Symbol] = None
    ) -> List[OrderResponse]:
        """Получить открытые ордера."""
        open_orders = [
            order
            for order in self.orders.values()
            if order.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
        ]
        if symbol:
            open_orders = [order for order in open_orders if order.symbol == symbol]
        return open_orders


class BybitBroker(BaseBroker):
    """Брокер для Bybit."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Здесь должна быть инициализация Bybit API клиента
        logger.info(
            f"Bybit broker initialized for {'testnet' if self.testnet else 'mainnet'}"
        )

    async def _place_order_impl(self, request: OrderRequest) -> OrderResponse:
        """Реализация размещения ордера для Bybit."""
        # Здесь должна быть реальная интеграция с Bybit API
        logger.info(
            f"Placing Bybit order: {request.symbol} {request.side.value} {request.quantity}"
        )
        # Мок-реализация
        return OrderResponse(
            order_id=f"bybit_{uuid.uuid4()}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            status=OrderStatus.PENDING,
            timestamp=Timestamp.now(),
            metadata=request.metadata or {},
        )

    async def _cancel_order_impl(self, order_id: str, symbol: Symbol) -> bool:
        """Реализация отмены ордера для Bybit."""
        logger.info(f"Cancelling Bybit order: {order_id}")
        # Здесь должна быть реальная интеграция с Bybit API
        return True

    async def _get_order_status_impl(
        self, order_id: str, symbol: Symbol
    ) -> OrderResponse:
        """Реализация получения статуса ордера для Bybit."""
        logger.info(f"Getting Bybit order status: {order_id}")
        # Здесь должна быть реальная интеграция с Bybit API
        # Возвращаем мок-ответ
        return OrderResponse(
            order_id=order_id,
            client_order_id=None,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(Decimal("1.0")),
            price=Price(Decimal("100.0"), Currency.USDT),
            status=OrderStatus.FILLED,
            timestamp=Timestamp.now(),
            metadata={},
        )


class BinanceBroker(BaseBroker):
    """Брокер для Binance."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Здесь должна быть инициализация Binance API клиента
        logger.info(
            f"Binance broker initialized for {'testnet' if self.testnet else 'mainnet'}"
        )

    async def _place_order_impl(self, request: OrderRequest) -> OrderResponse:
        """Реализация размещения ордера для Binance."""
        logger.info(
            f"Placing Binance order: {request.symbol} {request.side.value} {request.quantity}"
        )
        # Мок-реализация
        return OrderResponse(
            order_id=f"binance_{uuid.uuid4()}",
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            order_type=request.order_type,
            quantity=request.quantity,
            price=request.price,
            status=OrderStatus.PENDING,
            timestamp=Timestamp.now(),
            metadata=request.metadata or {},
        )

    async def _cancel_order_impl(self, order_id: str, symbol: Symbol) -> bool:
        """Реализация отмены ордера для Binance."""
        logger.info(f"Cancelling Binance order: {order_id}")
        return True

    async def _get_order_status_impl(
        self, order_id: str, symbol: Symbol
    ) -> OrderResponse:
        """Реализация получения статуса ордера для Binance."""
        logger.info(f"Getting Binance order status: {order_id}")
        return OrderResponse(
            order_id=order_id,
            client_order_id=None,
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Quantity(Decimal("1.0")),
            price=Price(Decimal("100.0"), Currency.USDT),
            status=OrderStatus.FILLED,
            timestamp=Timestamp.now(),
            metadata={},
        )


class BrokerFactory:
    """Фабрика для создания брокеров."""

    _brokers: Dict[str, type[BaseBroker]] = {
        "mock": MockBroker,
        "bybit": BybitBroker,
        "binance": BinanceBroker,
    }

    @classmethod
    def create_broker(cls, broker_type: str, config: Dict[str, Any]) -> BaseBroker:
        """Создать брокера по типу."""
        if broker_type not in cls._brokers:
            raise ValueError(f"Unknown broker type: {broker_type}")
        broker_class = cls._brokers[broker_type]
        return broker_class(config)

    @classmethod
    def register_broker(cls, name: str, broker_class: type[BaseBroker]) -> None:
        """Зарегистрировать новый тип брокера."""
        cls._brokers[name] = broker_class

    @classmethod
    def list_available_brokers(cls) -> List[str]:
        """Получить список доступных брокеров."""
        return list(cls._brokers.keys())


# Утилиты для работы с брокерами
def validate_order_request(request: OrderRequest) -> bool:
    """Валидация запроса на создание ордера."""
    if request.quantity.value <= 0:
        return False
    if request.order_type == OrderType.LIMIT and request.price is None:
        return False
    if request.order_type == OrderType.STOP and request.stop_price is None:
        return False
    if request.order_type == OrderType.STOP_LIMIT and (
        request.price is None or request.stop_price is None
    ):
        return False
    return True


def calculate_order_value(order: OrderResponse) -> Decimal:
    """Рассчитать стоимость ордера."""
    if order.price is None:
        return Decimal("0")
    return order.price.value * order.quantity.value
