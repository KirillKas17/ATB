"""
Протоколы для use cases приложения.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from domain.entities.market import MarketData
from domain.entities.order import Order, OrderStatus
from domain.entities.position import Position
from domain.entities.portfolio import Portfolio
from domain.entities.signal import Signal
from domain.entities.strategy import Strategy
from domain.entities.trade import Trade
from domain.entities.trading_pair import TradingPair
from domain.entities.trading_session import TradingSession
from domain.types import (
    OrderId,
    PortfolioId,
    PositionId,
    PriceValue,
    StrategyId,
    Symbol,
    VolumeValue,
    RiskMetrics,
)
from domain.value_objects import Money, Price, Volume

# Request/Response модели
class CreateOrderRequest:
    """Запрос на создание ордера."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        symbol: Symbol,
        side: str,
        order_type: str,
        quantity: Volume,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
    ):
        self.portfolio_id = portfolio_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price


class CreateOrderResponse:
    """Ответ на создание ордера."""

    def __init__(self, order: Order, success: bool, message: str = ""):
        self.order = order
        self.success = success
        self.message = message


class CancelOrderRequest:
    """Запрос на отмену ордера."""

    def __init__(self, order_id: OrderId):
        self.order_id = order_id


class CancelOrderResponse:
    """Ответ на отмену ордера."""

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


class GetOrdersRequest:
    """Запрос на получение ордеров."""

    def __init__(
        self,
        portfolio_id: Optional[PortfolioId] = None,
        status: Optional[OrderStatus] = None,
        limit: Optional[int] = None,
    ):
        self.portfolio_id = portfolio_id
        self.status = status
        self.limit = limit


class GetOrdersResponse:
    """Ответ с ордерами."""

    def __init__(self, orders: List[Order]):
        self.orders = orders


class CreatePositionRequest:
    """Запрос на создание позиции."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        symbol: Symbol,
        side: str,
        quantity: Volume,
        entry_price: Price,
    ):
        self.portfolio_id = portfolio_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.entry_price = entry_price


class CreatePositionResponse:
    """Ответ на создание позиции."""

    def __init__(self, position: Position, success: bool, message: str = ""):
        self.position = position
        self.success = success
        self.message = message


class UpdatePositionRequest:
    """Запрос на обновление позиции."""

    def __init__(
        self,
        position_id: PositionId,
        current_price: Price,
        unrealized_pnl: Money,
    ):
        self.position_id = position_id
        self.current_price = current_price
        self.unrealized_pnl = unrealized_pnl


class UpdatePositionResponse:
    """Ответ на обновление позиции."""

    def __init__(self, position: Position, success: bool, message: str = ""):
        self.position = position
        self.success = success
        self.message = message


class ClosePositionRequest:
    """Запрос на закрытие позиции."""

    def __init__(
        self,
        position_id: PositionId,
        close_price: Price,
        close_quantity: Volume,
        realized_pnl: Money,
    ):
        self.position_id = position_id
        self.close_price = close_price
        self.close_quantity = close_quantity
        self.realized_pnl = realized_pnl


class ClosePositionResponse:
    """Ответ на закрытие позиции."""

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


class GetPositionsRequest:
    """Запрос на получение позиций."""

    def __init__(
        self,
        portfolio_id: Optional[PortfolioId] = None,
        symbol: Optional[Symbol] = None,
    ):
        self.portfolio_id = portfolio_id
        self.symbol = symbol


class GetPositionsResponse:
    """Ответ с позициями."""

    def __init__(self, positions: List[Position]):
        self.positions = positions


class PositionMetrics:
    """Метрики позиции."""

    def __init__(
        self,
        unrealized_pnl: Money,
        realized_pnl: Money,
        margin_used: Money,
        leverage: float,
    ):
        self.unrealized_pnl = unrealized_pnl
        self.realized_pnl = realized_pnl
        self.margin_used = margin_used
        self.leverage = leverage


class RiskAssessmentRequest:
    """Запрос на оценку риска."""

    def __init__(self, portfolio_id: PortfolioId):
        self.portfolio_id = portfolio_id


class RiskAssessmentResponse:
    """Ответ с оценкой риска."""

    def __init__(self, risk_level: float, recommendations: List[str]):
        self.risk_level = risk_level
        self.recommendations = recommendations


class RiskLimitRequest:
    """Запрос на установку лимитов риска."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        max_risk_per_trade: float,
        max_daily_loss: float,
        max_portfolio_risk: float,
    ):
        self.portfolio_id = portfolio_id
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_loss = max_daily_loss
        self.max_portfolio_risk = max_portfolio_risk


class RiskLimitResponse:
    """Ответ на установку лимитов риска."""

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


class CreateTradingPairRequest:
    """Запрос на создание торговой пары."""

    def __init__(
        self,
        symbol: Symbol,
        base_currency: str,
        quote_currency: str,
        exchange: str,
    ):
        self.symbol = symbol
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        self.exchange = exchange


class CreateTradingPairResponse:
    """Ответ на создание торговой пары."""

    def __init__(self, trading_pair: TradingPair, success: bool, message: str = ""):
        self.trading_pair = trading_pair
        self.success = success
        self.message = message


class UpdateTradingPairRequest:
    """Запрос на обновление торговой пары."""

    def __init__(
        self,
        pair_id: str,
        is_active: bool,
        min_order_size: Volume,
        max_order_size: Volume,
        price_precision: int,
        volume_precision: int,
    ):
        self.pair_id = pair_id
        self.is_active = is_active
        self.min_order_size = min_order_size
        self.max_order_size = max_order_size
        self.price_precision = price_precision
        self.volume_precision = volume_precision


class UpdateTradingPairResponse:
    """Ответ на обновление торговой пары."""

    def __init__(self, trading_pair: TradingPair, success: bool, message: str = ""):
        self.trading_pair = trading_pair
        self.success = success
        self.message = message


class GetTradingPairsRequest:
    """Запрос на получение торговых пар."""

    def __init__(
        self,
        exchange: Optional[str] = None,
        is_active: Optional[bool] = None,
    ):
        self.exchange = exchange
        self.is_active = is_active


class GetTradingPairsResponse:
    """Ответ с торговыми парами."""

    def __init__(self, trading_pairs: List[TradingPair]):
        self.trading_pairs = trading_pairs


class ExecuteStrategyRequest:
    """Запрос на выполнение стратегии."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        strategy_id: StrategyId,
        parameters: Dict[str, Any],
    ):
        self.portfolio_id = portfolio_id
        self.strategy_id = strategy_id
        self.parameters = parameters


class ExecuteStrategyResponse:
    """Ответ на выполнение стратегии."""

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


class ProcessSignalRequest:
    """Запрос на обработку сигнала."""

    def __init__(self, signal: Signal):
        self.signal = signal


class ProcessSignalResponse:
    """Ответ на обработку сигнала."""

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


class PortfolioRebalanceRequest:
    """Запрос на ребалансировку портфеля."""

    def __init__(
        self,
        portfolio_id: PortfolioId,
        target_weights: Dict[Symbol, float],
    ):
        self.portfolio_id = portfolio_id
        self.target_weights = target_weights


class PortfolioRebalanceResponse:
    """Ответ на ребалансировку портфеля."""

    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


# Протоколы use cases
@runtime_checkable
class OrderManagementUseCase(Protocol):
    """Протокол для управления ордерами."""

    @abstractmethod
    async def create_order(self, request: CreateOrderRequest) -> CreateOrderResponse:
        """Создание нового ордера."""
        ...

    @abstractmethod
    async def cancel_order(self, request: CancelOrderRequest) -> CancelOrderResponse:
        """Отмена ордера."""
        ...

    @abstractmethod
    async def get_orders(self, request: GetOrdersRequest) -> GetOrdersResponse:
        """Получение списка ордеров."""
        ...

    @abstractmethod
    async def get_order_by_id(self, order_id: OrderId) -> Optional[Order]:
        """Получение ордера по ID."""
        ...

    @abstractmethod
    async def update_order_status(
        self,
        order_id: OrderId,
        status: OrderStatus,
        filled_amount: Optional[VolumeValue] = None,
    ) -> bool:
        """Обновление статуса ордера."""
        ...

    @abstractmethod
    async def validate_order(
        self, request: CreateOrderRequest
    ) -> tuple[bool, List[str]]:
        """Валидация ордера."""
        ...


@runtime_checkable
class PositionManagementUseCase(Protocol):
    """Протокол для управления позициями."""

    @abstractmethod
    async def create_position(
        self, request: CreatePositionRequest
    ) -> CreatePositionResponse:
        """Создание новой позиции."""
        ...

    @abstractmethod
    async def update_position(
        self, request: UpdatePositionRequest
    ) -> UpdatePositionResponse:
        """Обновление позиции."""
        ...

    @abstractmethod
    async def close_position(
        self, request: ClosePositionRequest
    ) -> ClosePositionResponse:
        """Закрытие позиции."""
        ...

    @abstractmethod
    async def get_positions(self, request: GetPositionsRequest) -> GetPositionsResponse:
        """Получение списка позиций."""
        ...

    @abstractmethod
    async def get_position_by_id(self, position_id: PositionId) -> Optional[Position]:
        """Получение позиции по ID."""
        ...

    @abstractmethod
    async def get_position_by_symbol(
        self, portfolio_id: PortfolioId, symbol: Symbol
    ) -> Optional[Position]:
        """Получение позиции по символу."""
        ...

    @abstractmethod
    async def calculate_position_metrics(
        self, position: Position, current_price: PriceValue
    ) -> PositionMetrics:
        """Расчет метрик позиции."""
        ...

    @abstractmethod
    async def process_trade(self, trade: Trade) -> bool:
        """Обработка исполненной сделки."""
        ...


@runtime_checkable
class RiskManagementUseCase(Protocol):
    """Протокол для управления рисками."""

    @abstractmethod
    async def assess_risk(
        self, request: RiskAssessmentRequest
    ) -> RiskAssessmentResponse:
        """Оценка риска портфеля."""
        ...

    @abstractmethod
    async def set_risk_limits(self, request: RiskLimitRequest) -> RiskLimitResponse:
        """Установка лимитов риска."""
        ...

    @abstractmethod
    async def get_risk_metrics(self, portfolio_id: PortfolioId) -> RiskMetrics:
        """Получение метрик риска."""
        ...

    @abstractmethod
    async def validate_order_risk(
        self, order_request: CreateOrderRequest
    ) -> tuple[bool, List[str]]:
        """Валидация риска ордера."""
        ...

    @abstractmethod
    async def calculate_position_risk(
        self, position: Position, market_data: MarketData
    ) -> Dict[str, Any]:
        """Расчет риска позиции."""
        ...


@runtime_checkable
class TradingPairManagementUseCase(Protocol):
    """Протокол для управления торговыми парами."""

    @abstractmethod
    async def create_trading_pair(
        self, request: CreateTradingPairRequest
    ) -> CreateTradingPairResponse:
        """Создание торговой пары."""
        ...

    @abstractmethod
    async def update_trading_pair(
        self, request: UpdateTradingPairRequest
    ) -> UpdateTradingPairResponse:
        """Обновление торговой пары."""
        ...

    @abstractmethod
    async def get_trading_pairs(
        self, request: GetTradingPairsRequest
    ) -> GetTradingPairsResponse:
        """Получение торговых пар."""
        ...

    @abstractmethod
    async def get_trading_pair_by_id(self, pair_id: str) -> Optional[TradingPair]:
        """Получение торговой пары по ID."""
        ...

    @abstractmethod
    async def get_trading_pair_by_symbol(
        self, exchange: str, symbol: Symbol
    ) -> Optional[TradingPair]:
        """Получение торговой пары по символу."""
        ...


@runtime_checkable
class TradingOrchestratorUseCase(Protocol):
    """Протокол для торгового оркестратора."""

    @abstractmethod
    async def execute_strategy(
        self, request: ExecuteStrategyRequest
    ) -> ExecuteStrategyResponse:
        """Выполнение торговой стратегии."""
        ...

    @abstractmethod
    async def process_signal(
        self, request: ProcessSignalRequest
    ) -> ProcessSignalResponse:
        """Обработка торгового сигнала."""
        ...

    @abstractmethod
    async def rebalance_portfolio(
        self, request: PortfolioRebalanceRequest
    ) -> PortfolioRebalanceResponse:
        """Ребалансировка портфеля."""
        ...

    @abstractmethod
    async def start_trading_session(
        self, portfolio_id: PortfolioId, strategy_id: StrategyId
    ) -> TradingSession:
        """Запуск торговой сессии."""
        ...

    @abstractmethod
    async def stop_trading_session(self, session_id: str) -> bool:
        """Остановка торговой сессии."""
        ...

    @abstractmethod
    async def get_trading_session(self, session_id: str) -> Optional[TradingSession]:
        """Получение торговой сессии."""
        ...
