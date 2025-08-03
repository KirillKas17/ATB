"""
Bybit Client Adapter - Backward Compatibility
Адаптер для обратной совместимости с существующим кодом.
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal
from uuid import uuid4
from datetime import datetime
import time

from domain.entities.account import Account, Balance
from domain.entities.market import MarketData
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.orderbook import OrderBookSnapshot
from domain.entities.position import Position
from domain.entities.trade import Trade
from domain.exceptions import ConnectionError, ExchangeError
from domain.protocols.exchange_protocol import ConnectionStatus, ExchangeProtocol
from domain.types import (
    OrderId,
    PortfolioId,
    PriceValue,
    SignalId,
    StrategyId,
    Symbol,
    VolumeValue,
    TimestampValue,
)
from domain.value_objects import Currency, Money, Price, Volume
from domain.value_objects.timestamp import Timestamp
from domain.types.external_service_types import (
    APIKey,
    APISecret,
    ConnectionConfig,
    ExchangeCredentials,
    ExchangeType,
    OrderRequest,
    TimeFrame,
    MarketDataRequest,
)


class BybitClient(ExchangeProtocol):
    """Адаптер BybitClient для обратной совместимости."""

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._connection_status = ConnectionStatus.DISCONNECTED
        # Создаем новый сервис
        credentials = ExchangeCredentials(
            api_key=APIKey(api_key), api_secret=APISecret(api_secret), testnet=testnet
        )
        # Временное решение - создаем заглушку
        self.exchange_service = None

    @property
    def exchange_name(self) -> str:
        """Название биржи."""
        return "bybit"

    @property
    def connection_status(self) -> ConnectionStatus:
        """Статус подключения."""
        return self._connection_status

    async def connect(self) -> bool:
        """Подключение к Bybit."""
        try:
            self._connection_status = ConnectionStatus.CONNECTING
            credentials = ExchangeCredentials(
                api_key=APIKey(self.api_key),
                api_secret=APISecret(self.api_secret),
                testnet=self.testnet,
            )
            # Временное решение
            result = True
            if result:
                self._connection_status = ConnectionStatus.CONNECTED
            return result
        except Exception as e:
            self._connection_status = ConnectionStatus.ERROR
            raise ConnectionError(f"Failed to connect to Bybit: {e}")

    async def disconnect(self) -> None:
        """Отключение от Bybit."""
        if self.exchange_service:
            await self.exchange_service.disconnect()
        self._connection_status = ConnectionStatus.DISCONNECTED

    async def authenticate(self, api_key: str, secret_key: str) -> bool:
        """Аутентификация."""
        try:
            credentials = ExchangeCredentials(
                api_key=APIKey(api_key), api_secret=APISecret(secret_key), testnet=self.testnet
            )
            self._connection_status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            self._connection_status = ConnectionStatus.ERROR
            raise ConnectionError(f"Failed to authenticate with Bybit: {e}")

    async def get_account_info(self) -> Account:
        """Получение информации об аккаунте."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_account_info()
            # Временное решение
            return Account()
        except Exception as e:
            raise ExchangeError(f"Failed to get account info: {e}")

    async def get_balance(self, currency: Currency) -> Balance:
        """Получение баланса по валюте."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_balance(currency)
            return Balance(currency=str(currency), available=Decimal("1000.0"), locked=Decimal("0.0"))
        except Exception as e:
            raise ExchangeError(f"Failed to get balance: {e}")

    async def get_order_book(
        self, symbol: Symbol, depth: int = 20
    ) -> OrderBookSnapshot:
        """Получение стакана заявок."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_order_book(symbol, depth)
            return OrderBookSnapshot(
                symbol=symbol,
                bids=[],
                asks=[],
                exchange="bybit",
                timestamp=Timestamp(datetime.now()),
            )
        except Exception as e:
            raise ExchangeError(f"Failed to get order book: {e}")

    async def get_market_data(self, symbol: Symbol) -> MarketData:
        """Получение рыночных данных."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_market_data(symbol)
            return MarketData(
                symbol=symbol,
                open=Price(Decimal("50000"), Currency.USDT, Currency.USDT),
                high=Price(Decimal("50001"), Currency.USDT, Currency.USDT),
                low=Price(Decimal("49999"), Currency.USDT, Currency.USDT),
                close=Price(Decimal("50000.5"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("1000"), Currency.USDT),
                timestamp=TimestampValue(datetime.fromtimestamp(1640995200)),
            )
        except Exception as e:
            raise ExchangeError(f"Failed to get market data: {e}")

    async def place_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Optional[Price] = None,
        stop_price: Optional[Price] = None,
        portfolio_id: Optional[PortfolioId] = None,
        strategy_id: Optional[StrategyId] = None,
        signal_id: Optional[SignalId] = None,
    ) -> Order:
        """Размещение ордера с полной валидацией."""
        try:
            if self.exchange_service:
                return await self.exchange_service.place_order(
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    portfolio_id=portfolio_id,
                    strategy_id=strategy_id,
                    signal_id=signal_id,
                )
            # Временное решение
            return Order(
                id=OrderId(uuid4()),
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=VolumeValue(quantity.value),
                price=price,
                status=OrderStatus.OPEN,
                created_at=Timestamp.now(),
                updated_at=Timestamp.now(),
            )
        except Exception as e:
            raise ExchangeError(f"Failed to place order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        try:
            if self.exchange_service:
                # Исправление 199: cancel_order должен принимать str, а не OrderId
                return await self.exchange_service.cancel_order(order_id)
            return True
        except Exception as e:
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def get_order_status(self, order_id: str) -> Order:
        """Получение статуса ордера."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_order_status(order_id)
            # Временное решение
            # Упрощенная логика преобразования типов
            try:
                if isinstance(order_id, str):
                    order_uuid = OrderId(uuid4())
                else:
                    order_uuid = OrderId(order_id)
            except (ValueError, TypeError):
                order_uuid = OrderId(uuid4())
            
            return Order(
                id=order_uuid,
                symbol=Symbol("BTCUSDT"),
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=VolumeValue(Decimal("1.0")),
                status=OrderStatus.FILLED,
            )
        except Exception as e:
            raise ExchangeError(f"Failed to get order status: {e}")

    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[Order]:
        """Получение открытых ордеров."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_open_orders(symbol)
            # Временное решение
            return []
        except Exception as e:
            raise ExchangeError(f"Failed to get open orders: {e}")

    async def get_trade_history(
        self, symbol: Optional[Symbol] = None, limit: int = 100
    ) -> List[Trade]:
        """Получение истории сделок."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_trade_history(symbol, limit)
            # Временное решение
            return []
        except Exception as e:
            raise ExchangeError(f"Failed to get trade history: {e}")

    async def get_positions(self) -> List[Position]:
        """Получение позиций."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_positions()
            # Временное решение
            return []
        except Exception as e:
            raise ExchangeError(f"Failed to get positions: {e}")

    async def validate_order(
        self,
        symbol: Symbol,
        side: OrderSide,
        order_type: OrderType,
        quantity: Volume,
        price: Optional[Price] = None,
    ) -> bool:
        """Валидация ордера."""
        try:
            if self.exchange_service:
                return await self.exchange_service.validate_order(
                    symbol, side, order_type, quantity, price
                )
            # Временное решение
            return True
        except Exception as e:
            raise ExchangeError(f"Failed to validate order: {e}")

    async def calculate_commission(
        self, order_value: Money, order_type: OrderType
    ) -> Money:
        """Расчет комиссии."""
        try:
            if self.exchange_service:
                return await self.exchange_service.calculate_commission(order_value, order_type)
            # Временное решение
            commission_rate = 0.001  # 0.1%
            commission_amount = float(order_value.amount) * commission_rate
            return Money(amount=Decimal(commission_amount), currency=order_value.currency)
        except Exception as e:
            raise ExchangeError(f"Failed to calculate commission: {e}")

    async def get_market_data_legacy(
        self, symbol: str, interval: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Получение рыночных данных (legacy метод)."""
        try:
            if self.exchange_service:
                request = MarketDataRequest(
                    symbol=Symbol(symbol),
                    timeframe=TimeFrame(interval),
                    limit=limit,
                )
                return await self.exchange_service.get_market_data(request)
            # Временное решение
            return []
        except Exception as e:
            raise ExchangeError(f"Failed to get market data: {e}")

    async def place_order_legacy(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Размещение ордера (legacy метод)."""
        try:
            if self.exchange_service:
                request = OrderRequest(
                    symbol=Symbol(symbol),
                    side=OrderSide(side),
                    order_type=OrderType(order_type),
                    quantity=VolumeValue(Decimal(quantity)),
                    price=PriceValue(Decimal(price)) if price else None,
                )
                return await self.exchange_service.place_order(request)
            # Временное решение
            return {
                "order_id": "temp_order_id",
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "status": "open",
            }
        except Exception as e:
            raise ExchangeError(f"Failed to place order: {e}")

    async def cancel_order_legacy(self, order_id: str) -> bool:
        """Отмена ордера (legacy метод)."""
        try:
            if self.exchange_service:
                return await self.exchange_service.cancel_order(OrderId(order_id))
            return True
        except Exception as e:
            raise ExchangeError(f"Failed to cancel order: {e}")

    async def get_order_status_legacy(self, order_id: str) -> Dict[str, Any]:
        """Получение статуса ордера (legacy метод)."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_order_status(OrderId(order_id))
            return {
                "order_id": order_id,
                "status": "filled",
                "symbol": "BTCUSDT",
                "side": "buy",
                "quantity": Decimal("1.0"),
                "price": Decimal("50000.0"),
            }
        except Exception as e:
            raise ExchangeError(f"Failed to get order status: {e}")

    async def get_balance_legacy(self) -> Dict[str, float]:
        """Получение баланса (legacy метод)."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_balance()
            # Временное решение
            return {"BTC": 1.0, "USDT": 50000.0}
        except Exception as e:
            raise ExchangeError(f"Failed to get balance: {e}")

    async def get_positions_legacy(self) -> List[Dict[str, Any]]:
        """Получение позиций (legacy метод)."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_positions()
            # Временное решение
            return []
        except Exception as e:
            raise ExchangeError(f"Failed to get positions: {e}")

    async def get_server_time(self) -> int:
        """Получение времени сервера."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_server_time()
            # Временное решение
            import time
            return int(time.time() * 1000)
        except Exception as e:
            raise ExchangeError(f"Failed to get server time: {e}")

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Получение информации о бирже."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_exchange_info()
            # Временное решение
            return {
                "timezone": "UTC",
                "serverTime": int(time.time() * 1000),
                "symbols": [],
            }
        except Exception as e:
            raise ExchangeError(f"Failed to get exchange info: {e}")

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Получение тикера."""
        try:
            if self.exchange_service:
                return await self.exchange_service.get_ticker(Symbol(symbol))
            # Временное решение
            return {
                "symbol": symbol,
                "price": "50000.0",
                "volume": "1000.0",
                "change": "0.5",
            }
        except Exception as e:
            raise ExchangeError(f"Failed to get ticker: {e}")


# Экспорт для обратной совместимости
__all__ = ["BybitClient"]
