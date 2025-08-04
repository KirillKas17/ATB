"""
Типы для модуля исполнения ордеров.

Включает:
- Типизированные словари
- Перечисления
- Протоколы
- Новые типы
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    runtime_checkable,
)
from uuid import uuid4

# Новые типы
OrderId = NewType("OrderId", str)
ClientOrderId = NewType("ClientOrderId", str)
TradeId = NewType("TradeId", str)
ExecutionId = NewType("ExecutionId", str)


# Перечисления
class OrderType(Enum):
    """Типы ордеров."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"


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
    PENDING_CANCEL = "pending_cancel"


class TimeInForce(Enum):
    """Время действия ордера."""

    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    DAY = "DAY"  # Day Order


class ExecutionType(Enum):
    """Типы исполнения."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class BrokerType(Enum):
    """Типы брокеров."""

    MOCK = "mock"
    BYBIT = "bybit"
    BINANCE = "binance"
    KUCOIN = "kucoin"
    OKX = "okx"


# Типизированные словари
class OrderRequestDict(TypedDict, total=False):
    """Словарь запроса на создание ордера."""

    symbol: str
    side: str
    order_type: str
    quantity: str
    price: Optional[str]
    stop_price: Optional[str]
    time_in_force: str
    client_order_id: Optional[str]
    metadata: Dict[str, Any]


class OrderResponseDict(TypedDict, total=False):
    """Словарь ответа на создание ордера."""

    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: str
    price: Optional[str]
    status: str
    timestamp: str
    metadata: Dict[str, Any]


class OrderUpdateDict(TypedDict, total=False):
    """Словарь обновления ордера."""

    order_id: str
    status: str
    filled_quantity: str
    remaining_quantity: str
    average_price: Optional[str]
    timestamp: str
    metadata: Dict[str, Any]


class TradeDict(TypedDict, total=False):
    """Словарь сделки."""

    trade_id: str
    order_id: str
    symbol: str
    side: str
    quantity: str
    price: str
    timestamp: str
    fee: Optional[str]
    fee_currency: Optional[str]
    metadata: Dict[str, Any]


class ExecutionConfigDict(TypedDict, total=False):
    """Словарь конфигурации исполнения."""

    max_retries: int
    retry_delay: float
    timeout: float
    enable_logging: bool
    enable_metrics: bool
    rate_limit_delay: float
    max_concurrent_orders: int


class BrokerConfigDict(TypedDict, total=False):
    """Словарь конфигурации брокера."""

    name: str
    api_key: str
    secret_key: str
    testnet: bool
    rate_limit_delay: float
    max_retries: int
    timeout: float
    additional_params: Dict[str, Any]


class ExecutionResultDict(TypedDict, total=False):
    """Словарь результата исполнения."""

    success: bool
    order_id: Optional[str]
    error_message: Optional[str]
    execution_time: float
    trades: List[TradeDict]
    metadata: Dict[str, Any]


class OrderExecutionStateDict(TypedDict, total=False):
    """Словарь состояния исполнения ордера."""

    order_id: str
    request: OrderRequestDict
    response: Optional[OrderResponseDict]
    status: str
    created_at: str
    updated_at: str
    retry_count: int
    trades: List[TradeDict]
    metadata: Dict[str, Any]


# Протоколы
@runtime_checkable
class OrderExecutionCallback(Protocol):
    """Протокол для callback'ов исполнения ордеров."""

    async def on_order_placed(
        self, order_id: OrderId, response: OrderResponseDict
    ) -> None:
        """Вызывается при размещении ордера."""
        ...

    async def on_order_updated(
        self, order_id: OrderId, update: OrderUpdateDict
    ) -> None:
        """Вызывается при обновлении ордера."""
        ...

    async def on_order_filled(self, order_id: OrderId, trades: List[TradeDict]) -> None:
        """Вызывается при исполнении ордера."""
        ...

    async def on_order_cancelled(self, order_id: OrderId) -> None:
        """Вызывается при отмене ордера."""
        ...

    async def on_order_failed(self, order_id: OrderId, error: str) -> None:
        """Вызывается при ошибке ордера."""
        ...


@runtime_checkable
class OrderValidator(Protocol):
    """Протокол для валидации ордеров."""

    def validate_order_request(self, request: OrderRequestDict) -> bool:
        """Валидировать запрос на создание ордера."""
        ...

    def get_validation_errors(self, request: OrderRequestDict) -> List[str]:
        """Получить ошибки валидации."""
        ...


@runtime_checkable
class OrderTransformer(Protocol):
    """Протокол для трансформации ордеров."""

    def transform_request(self, request: OrderRequestDict) -> OrderRequestDict:
        """Трансформировать запрос."""
        ...

    def transform_response(self, response: OrderResponseDict) -> OrderResponseDict:
        """Трансформировать ответ."""
        ...


# Dataclass'ы
@dataclass
class OrderMetrics:
    """Метрики ордера."""

    order_id: OrderId
    placement_time: float
    execution_time: float
    total_time: float
    retry_count: int
    success: bool
    error_count: int
    latency_ms: float


@dataclass
class ExecutionMetrics:
    """Метрики исполнения."""

    total_orders: int
    successful_orders: int
    failed_orders: int
    success_rate: float
    average_execution_time: float
    total_volume: Decimal
    total_value: Decimal
    average_latency: float


@dataclass
class BrokerMetrics:
    """Метрики брокера."""

    broker_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    rate_limit_hits: int
    error_rate: float


# Константы
DEFAULT_EXECUTION_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "timeout": 30.0,
    "enable_logging": True,
    "enable_metrics": True,
    "rate_limit_delay": 0.1,
    "max_concurrent_orders": 10,
}

DEFAULT_BROKER_CONFIG = {
    "testnet": True,
    "rate_limit_delay": 0.1,
    "max_retries": 3,
    "timeout": 30.0,
    "additional_params": {},
}


# Валидаторы
def validate_order_request_dict(request: OrderRequestDict) -> bool:
    """Валидировать словарь запроса на создание ордера."""
    required_fields = ["symbol", "side", "order_type", "quantity"]

    # Проверка наличия всех обязательных полей
    for field in required_fields:
        if field not in request:
            return False

    # Проверка типов
    if not isinstance(request["symbol"], str):
        return False

    if request["side"] not in [side.value for side in OrderSide]:
        return False

    if request["order_type"] not in [order_type.value for order_type in OrderType]:
        return False

    try:
        quantity = Decimal(request["quantity"])
        if quantity <= 0:
            return False
    except (ValueError, TypeError):
        return False

    # Проверка цены для лимитных ордеров
    if request["order_type"] in ["limit", "stop_limit", "take_profit_limit"]:
        if "price" not in request or request["price"] is None:
            return False
        try:
            price = Decimal(request["price"])
            if price <= 0:
                return False
        except (ValueError, TypeError):
            return False

    return True


def get_order_request_errors(request: OrderRequestDict) -> List[str]:
    """Получить ошибки валидации запроса на создание ордера."""
    errors = []

    if "symbol" not in request:
        errors.append("Symbol is required")
    elif not isinstance(request["symbol"], str):
        errors.append("Symbol must be a string")

    if "side" not in request:
        errors.append("Side is required")
    elif request["side"] not in [side.value for side in OrderSide]:
        errors.append(f"Invalid side: {request['side']}")

    if "order_type" not in request:
        errors.append("Order type is required")
    elif request["order_type"] not in [order_type.value for order_type in OrderType]:
        errors.append(f"Invalid order type: {request['order_type']}")

    if "quantity" not in request:
        errors.append("Quantity is required")
    else:
        try:
            quantity = Decimal(request["quantity"])
            if quantity <= 0:
                errors.append("Quantity must be positive")
        except (ValueError, TypeError):
            errors.append("Invalid quantity format")

    # Проверка цены для лимитных ордеров
    if request.get("order_type") in ["limit", "stop_limit", "take_profit_limit"]:
        if "price" not in request or request["price"] is None:
            errors.append("Price is required for limit orders")
        else:
            try:
                price = Decimal(request["price"])
                if price <= 0:
                    errors.append("Price must be positive")
            except (ValueError, TypeError):
                errors.append("Invalid price format")

    return errors


# Утилиты
def create_order_id() -> OrderId:
    """Создать новый ID ордера."""
    return OrderId(str(uuid4()))


def create_client_order_id(prefix: str = "client") -> ClientOrderId:
    """Создать новый клиентский ID ордера."""
    return ClientOrderId(f"{prefix}_{uuid4()}")


def create_trade_id() -> TradeId:
    """Создать новый ID сделки."""
    return TradeId(str(uuid4()))


def create_execution_id() -> ExecutionId:
    """Создать новый ID исполнения."""
    return ExecutionId(str(uuid4()))


def calculate_order_value(quantity: str, price: Optional[str]) -> Decimal:
    """Рассчитать стоимость ордера."""
    if price is None:
        return Decimal("0")

    try:
        return Decimal(quantity) * Decimal(price)
    except (ValueError, TypeError):
        return Decimal("0")


def format_order_request(request: OrderRequestDict) -> str:
    """Форматировать запрос на создание ордера для логирования."""
    return f"{request.get('side', 'unknown')} {request.get('quantity', '0')} {request.get('symbol', 'unknown')} @ {request.get('price', 'market')}"


def format_execution_result(result: ExecutionResultDict) -> str:
    """Форматировать результат исполнения для логирования."""
    if result.get("success", False):
        return f"Order {result.get('order_id', 'unknown')} executed successfully in {result.get('execution_time', 0):.2f}s"
    else:
        return f"Order execution failed: {result.get('error_message', 'unknown error')}"
