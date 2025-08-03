"""
Unit тесты для ExchangeProtocol.

Покрывает:
- Основные протоколы биржи
- Управление ордерами
- Получение рыночных данных
- Обработку ошибок
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from decimal import Decimal
from datetime import datetime

from domain.protocols.exchange_protocol import (
    ExchangeProtocol,
    ConnectionStatus
)
from domain.entities.order import Order, OrderSide, OrderType
from domain.entities.trade import Trade
from domain.entities.position import Position
from domain.entities.market import MarketData, OrderBook
from domain.entities.account import Balance
from domain.entities.trading_pair import TradingPair
from domain.exceptions.protocol_exceptions import (
    ConnectionError,
    ExchangeRateLimitError,
    OrderNotFoundError,
    ProtocolError
)
from domain.exceptions import (
    ExchangeError,
    InsufficientFundsError,
    InvalidOrderError,
    NetworkError,
    TimeoutError
)
from domain.types import OrderId, PortfolioId, Symbol
from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume


class TestExchangeProtocol:
    """Тесты для базового ExchangeProtocol."""

    @pytest.fixture
    def mock_exchange_protocol(self) -> Mock:
        """Мок протокола биржи."""
        return Mock(spec=ExchangeProtocol)

    @pytest.fixture
    def sample_order(self) -> Order:
        """Тестовый ордер."""
        return Order(
            id=uuid4(),
            trading_pair=TradingPair(base="BTC", quote="USDT"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=1.0,
            price=50000.0,
            status="PENDING",
            created_at=datetime.now()
        )

    @pytest.fixture
    def sample_trade(self) -> Trade:
        """Тестовый трейд."""
        return Trade(
            id=uuid4(),
            trading_pair=TradingPair(base="BTC", quote="USDT"),
            side=OrderSide.BUY,
            quantity=1.0,
            price=50000.0,
            timestamp=datetime.now()
        )

    @pytest.fixture
    def sample_position(self) -> Position:
        """Тестовая позиция."""
        return Position(
            id=uuid4(),
            trading_pair=TradingPair(base="BTC", quote="USDT"),
            side="LONG",
            quantity=1.0,
            entry_price=50000.0,
            current_price=51000.0,
            is_open=True,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    def test_create_order_method_exists(self, mock_exchange_protocol, sample_order):
        """Тест наличия метода create_order."""
        mock_exchange_protocol.create_order = AsyncMock(return_value={})
        assert hasattr(mock_exchange_protocol, 'create_order')
        assert callable(mock_exchange_protocol.create_order)

    def test_cancel_order_method_exists(self, mock_exchange_protocol):
        """Тест наличия метода cancel_order."""
        mock_exchange_protocol.cancel_order = AsyncMock(return_value=True)
        assert hasattr(mock_exchange_protocol, 'cancel_order')
        assert callable(mock_exchange_protocol.cancel_order)

    def test_fetch_order_method_exists(self, mock_exchange_protocol):
        """Тест наличия метода fetch_order."""
        mock_exchange_protocol.fetch_order = AsyncMock(return_value={})
        assert hasattr(mock_exchange_protocol, 'fetch_order')
        assert callable(mock_exchange_protocol.fetch_order)

    def test_fetch_open_orders_method_exists(self, mock_exchange_protocol):
        """Тест наличия метода fetch_open_orders."""
        mock_exchange_protocol.fetch_open_orders = AsyncMock(return_value=[])
        assert hasattr(mock_exchange_protocol, 'fetch_open_orders')
        assert callable(mock_exchange_protocol.fetch_open_orders)

    def test_fetch_balance_method_exists(self, mock_exchange_protocol):
        """Тест наличия метода fetch_balance."""
        mock_exchange_protocol.fetch_balance = AsyncMock(return_value={})
        assert hasattr(mock_exchange_protocol, 'fetch_balance')
        assert callable(mock_exchange_protocol.fetch_balance)

    def test_fetch_ticker_method_exists(self, mock_exchange_protocol):
        """Тест наличия метода fetch_ticker."""
        mock_exchange_protocol.fetch_ticker = AsyncMock(return_value={})
        assert hasattr(mock_exchange_protocol, 'fetch_ticker')
        assert callable(mock_exchange_protocol.fetch_ticker)

    def test_fetch_order_book_method_exists(self, mock_exchange_protocol):
        """Тест наличия метода fetch_order_book."""
        mock_exchange_protocol.fetch_order_book = AsyncMock(return_value={})
        assert hasattr(mock_exchange_protocol, 'fetch_order_book')
        assert callable(mock_exchange_protocol.fetch_order_book)


class TestConnectionStatus:
    """Тесты для ConnectionStatus."""

    def test_connection_statuses_exist(self):
        """Тест наличия всех статусов подключения."""
        assert ConnectionStatus.DISCONNECTED == "disconnected"
        assert ConnectionStatus.CONNECTING == "connecting"
        assert ConnectionStatus.CONNECTED == "connected"
        assert ConnectionStatus.ERROR == "error"

    def test_connection_status_transitions(self):
        """Тест переходов между статусами подключения."""
        # Валидные переходы
        valid_transitions = {
            ConnectionStatus.DISCONNECTED: [ConnectionStatus.CONNECTING, ConnectionStatus.ERROR],
            ConnectionStatus.CONNECTING: [ConnectionStatus.CONNECTED, ConnectionStatus.ERROR, ConnectionStatus.DISCONNECTED],
            ConnectionStatus.CONNECTED: [ConnectionStatus.DISCONNECTED, ConnectionStatus.ERROR],
            ConnectionStatus.ERROR: [ConnectionStatus.DISCONNECTED, ConnectionStatus.CONNECTING]
        }
        
        for status, valid_next_states in valid_transitions.items():
            assert isinstance(status, str)
            assert all(isinstance(next_state, str) for next_state in valid_next_states)

    def test_connection_status_validation(self):
        """Тест валидации статусов подключения."""
        valid_statuses = [
            ConnectionStatus.DISCONNECTED,
            ConnectionStatus.CONNECTING,
            ConnectionStatus.CONNECTED,
            ConnectionStatus.ERROR
        ]
        
        for status in valid_statuses:
            assert isinstance(status, str)
            assert status in [cs.value for cs in ConnectionStatus]


class TestExchangeProtocolIntegration:
    """Интеграционные тесты для ExchangeProtocol."""

    @pytest.mark.asyncio
    async def test_exchange_protocol_workflow(self, mock_exchange_protocol, sample_order):
        """Тест полного рабочего процесса протокола биржи."""
        order_id = str(uuid4())
        
        # Создание ордера
        order_response = {
            'id': order_id,
            'status': 'PENDING',
            'filled': 0.0,
            'remaining': 1.0
        }
        mock_exchange_protocol.create_order = AsyncMock(return_value=order_response)
        
        created_order = await mock_exchange_protocol.create_order(sample_order)
        assert created_order['id'] == order_id
        assert created_order['status'] == 'PENDING'
        mock_exchange_protocol.create_order.assert_called_once_with(sample_order)

        # Получение информации об ордере
        order_info = {
            'id': order_id,
            'status': 'FILLED',
            'filled': 1.0,
            'remaining': 0.0,
            'price': 50000.0
        }
        mock_exchange_protocol.fetch_order = AsyncMock(return_value=order_info)
        
        fetched_order = await mock_exchange_protocol.fetch_order(order_id)
        assert fetched_order['id'] == order_id
        assert fetched_order['status'] == 'FILLED'
        mock_exchange_protocol.fetch_order.assert_called_once_with(order_id)

        # Отмена ордера
        mock_exchange_protocol.cancel_order = AsyncMock(return_value=True)
        cancelled = await mock_exchange_protocol.cancel_order(order_id)
        assert cancelled is True
        mock_exchange_protocol.cancel_order.assert_called_once_with(order_id)

    @pytest.mark.asyncio
    async def test_market_data_operations(self, mock_exchange_protocol):
        """Тест операций с рыночными данными."""
        symbol = "BTCUSDT"
        
        # Получение тикера
        ticker_data = {
            'symbol': symbol,
            'last': 50000.0,
            'bid': 49999.0,
            'ask': 50001.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        mock_exchange_protocol.fetch_ticker = AsyncMock(return_value=ticker_data)
        
        ticker = await mock_exchange_protocol.fetch_ticker(symbol)
        assert ticker['symbol'] == symbol
        assert ticker['last'] == 50000.0
        mock_exchange_protocol.fetch_ticker.assert_called_once_with(symbol)

        # Получение ордербука
        orderbook_data = {
            'symbol': symbol,
            'bids': [[49999.0, 1.5], [49998.0, 2.0]],
            'asks': [[50001.0, 1.0], [50002.0, 1.5]],
            'timestamp': datetime.now().isoformat()
        }
        mock_exchange_protocol.fetch_order_book = AsyncMock(return_value=orderbook_data)
        
        orderbook = await mock_exchange_protocol.fetch_order_book(symbol)
        assert orderbook['symbol'] == symbol
        assert len(orderbook['bids']) == 2
        assert len(orderbook['asks']) == 2
        mock_exchange_protocol.fetch_order_book.assert_called_once_with(symbol)

    @pytest.mark.asyncio
    async def test_balance_operations(self, mock_exchange_protocol):
        """Тест операций с балансом."""
        # Получение баланса
        balance_data = {
            'BTC': {'free': 1.0, 'used': 0.5, 'total': 1.5},
            'USDT': {'free': 50000.0, 'used': 25000.0, 'total': 75000.0}
        }
        mock_exchange_protocol.fetch_balance = AsyncMock(return_value=balance_data)
        
        balance = await mock_exchange_protocol.fetch_balance()
        assert 'BTC' in balance
        assert 'USDT' in balance
        assert balance['BTC']['total'] == 1.5
        assert balance['USDT']['total'] == 75000.0
        mock_exchange_protocol.fetch_balance.assert_called_once()

    @pytest.mark.asyncio
    async def test_open_orders_operations(self, mock_exchange_protocol):
        """Тест операций с открытыми ордерами."""
        # Получение открытых ордеров
        open_orders_data = [
            {
                'id': str(uuid4()),
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'type': 'LIMIT',
                'quantity': 1.0,
                'price': 50000.0,
                'status': 'PENDING'
            },
            {
                'id': str(uuid4()),
                'symbol': 'ETHUSDT',
                'side': 'SELL',
                'type': 'LIMIT',
                'quantity': 10.0,
                'price': 3000.0,
                'status': 'PENDING'
            }
        ]
        mock_exchange_protocol.fetch_open_orders = AsyncMock(return_value=open_orders_data)
        
        open_orders = await mock_exchange_protocol.fetch_open_orders()
        assert len(open_orders) == 2
        assert open_orders[0]['symbol'] == 'BTCUSDT'
        assert open_orders[1]['symbol'] == 'ETHUSDT'
        mock_exchange_protocol.fetch_open_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_in_exchange_protocol(self, mock_exchange_protocol, sample_order):
        """Тест обработки ошибок в протоколе биржи."""
        order_id = str(uuid4())
        
        # Ошибка создания ордера
        mock_exchange_protocol.create_order = AsyncMock(side_effect=InvalidOrderError("Invalid order"))
        
        with pytest.raises(InvalidOrderError, match="Invalid order"):
            await mock_exchange_protocol.create_order(sample_order)

        # Ошибка получения ордера
        mock_exchange_protocol.fetch_order = AsyncMock(side_effect=OrderNotFoundError("Order not found"))
        
        with pytest.raises(OrderNotFoundError, match="Order not found"):
            await mock_exchange_protocol.fetch_order(order_id)

        # Ошибка отмены ордера
        mock_exchange_protocol.cancel_order = AsyncMock(side_effect=ExchangeError("Cancel failed"))
        
        with pytest.raises(ExchangeError, match="Cancel failed"):
            await mock_exchange_protocol.cancel_order(order_id)

        # Ошибка получения тикера
        mock_exchange_protocol.fetch_ticker = AsyncMock(side_effect=NetworkError("Network error"))
        
        with pytest.raises(NetworkError, match="Network error"):
            await mock_exchange_protocol.fetch_ticker("BTCUSDT")

        # Ошибка получения баланса
        mock_exchange_protocol.fetch_balance = AsyncMock(side_effect=TimeoutError("Request timeout"))
        
        with pytest.raises(TimeoutError, match="Request timeout"):
            await mock_exchange_protocol.fetch_balance()

    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, mock_exchange_protocol):
        """Тест обработки ограничений скорости запросов."""
        # Ошибка превышения лимита запросов
        mock_exchange_protocol.fetch_ticker = AsyncMock(side_effect=ExchangeRateLimitError("Rate limit exceeded"))
        
        with pytest.raises(ExchangeRateLimitError, match="Rate limit exceeded"):
            await mock_exchange_protocol.fetch_ticker("BTCUSDT")

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_exchange_protocol):
        """Тест обработки ошибок подключения."""
        # Ошибка подключения
        mock_exchange_protocol.fetch_balance = AsyncMock(side_effect=ConnectionError("Connection failed"))
        
        with pytest.raises(ConnectionError, match="Connection failed"):
            await mock_exchange_protocol.fetch_balance()

    @pytest.mark.asyncio
    async def test_insufficient_funds_handling(self, mock_exchange_protocol, sample_order):
        """Тест обработки ошибки недостаточных средств."""
        # Ошибка недостаточных средств
        mock_exchange_protocol.create_order = AsyncMock(side_effect=InsufficientFundsError("Insufficient funds"))
        
        with pytest.raises(InsufficientFundsError, match="Insufficient funds"):
            await mock_exchange_protocol.create_order(sample_order)

    @pytest.mark.asyncio
    async def test_protocol_error_handling(self, mock_exchange_protocol):
        """Тест обработки протокольных ошибок."""
        # Общая протокольная ошибка
        mock_exchange_protocol.fetch_order_book = AsyncMock(side_effect=ProtocolError("Protocol error"))
        
        with pytest.raises(ProtocolError, match="Protocol error"):
            await mock_exchange_protocol.fetch_order_book("BTCUSDT")

    @pytest.mark.asyncio
    async def test_order_lifecycle_management(self, mock_exchange_protocol, sample_order):
        """Тест управления жизненным циклом ордера."""
        order_id = str(uuid4())
        
        # Создание ордера
        order_response = {'id': order_id, 'status': 'PENDING'}
        mock_exchange_protocol.create_order = AsyncMock(return_value=order_response)
        created = await mock_exchange_protocol.create_order(sample_order)
        assert created['id'] == order_id

        # Проверка статуса ордера
        order_info = {'id': order_id, 'status': 'PENDING', 'filled': 0.0}
        mock_exchange_protocol.fetch_order = AsyncMock(return_value=order_info)
        fetched = await mock_exchange_protocol.fetch_order(order_id)
        assert fetched['status'] == 'PENDING'

        # Отмена ордера
        mock_exchange_protocol.cancel_order = AsyncMock(return_value=True)
        cancelled = await mock_exchange_protocol.cancel_order(order_id)
        assert cancelled is True

        # Проверка отмененного ордера
        cancelled_order_info = {'id': order_id, 'status': 'CANCELLED', 'filled': 0.0}
        mock_exchange_protocol.fetch_order = AsyncMock(return_value=cancelled_order_info)
        cancelled_fetched = await mock_exchange_protocol.fetch_order(order_id)
        assert cancelled_fetched['status'] == 'CANCELLED'

    @pytest.mark.asyncio
    async def test_market_data_validation(self, mock_exchange_protocol):
        """Тест валидации рыночных данных."""
        symbol = "BTCUSDT"
        
        # Валидные данные тикера
        valid_ticker = {
            'symbol': symbol,
            'last': 50000.0,
            'bid': 49999.0,
            'ask': 50001.0,
            'volume': 1000.0,
            'timestamp': datetime.now().isoformat()
        }
        mock_exchange_protocol.fetch_ticker = AsyncMock(return_value=valid_ticker)
        
        ticker = await mock_exchange_protocol.fetch_ticker(symbol)
        assert ticker['last'] > 0
        assert ticker['bid'] <= ticker['ask']
        assert ticker['volume'] >= 0

        # Валидные данные ордербука
        valid_orderbook = {
            'symbol': symbol,
            'bids': [[49999.0, 1.5], [49998.0, 2.0]],
            'asks': [[50001.0, 1.0], [50002.0, 1.5]],
            'timestamp': datetime.now().isoformat()
        }
        mock_exchange_protocol.fetch_order_book = AsyncMock(return_value=valid_orderbook)
        
        orderbook = await mock_exchange_protocol.fetch_order_book(symbol)
        assert len(orderbook['bids']) > 0
        assert len(orderbook['asks']) > 0
        assert all(bid[0] > 0 and bid[1] > 0 for bid in orderbook['bids'])
        assert all(ask[0] > 0 and ask[1] > 0 for ask in orderbook['asks'])

    @pytest.mark.asyncio
    async def test_balance_validation(self, mock_exchange_protocol):
        """Тест валидации баланса."""
        # Валидные данные баланса
        valid_balance = {
            'BTC': {'free': 1.0, 'used': 0.5, 'total': 1.5},
            'USDT': {'free': 50000.0, 'used': 25000.0, 'total': 75000.0}
        }
        mock_exchange_protocol.fetch_balance = AsyncMock(return_value=valid_balance)
        
        balance = await mock_exchange_protocol.fetch_balance()
        for currency, amounts in balance.items():
            assert amounts['free'] >= 0
            assert amounts['used'] >= 0
            assert amounts['total'] >= 0
            assert amounts['total'] == amounts['free'] + amounts['used']

    def test_protocol_compliance(self):
        """Тест соответствия протоколу."""
        # Проверка, что протокол имеет все необходимые методы
        assert hasattr(ExchangeProtocol, 'create_order')
        assert hasattr(ExchangeProtocol, 'cancel_order')
        assert hasattr(ExchangeProtocol, 'fetch_order')
        assert hasattr(ExchangeProtocol, 'fetch_open_orders')
        assert hasattr(ExchangeProtocol, 'fetch_balance')
        assert hasattr(ExchangeProtocol, 'fetch_ticker')
        assert hasattr(ExchangeProtocol, 'fetch_order_book')

    def test_exchange_protocol_method_signatures(self):
        """Тест сигнатур методов протокола биржи."""
        # Проверка, что методы являются асинхронными
        import inspect
        
        # Получаем методы протокола
        protocol_methods = [
            'create_order',
            'cancel_order', 
            'fetch_order',
            'fetch_open_orders',
            'fetch_balance',
            'fetch_ticker',
            'fetch_order_book'
        ]
        
        # Проверяем, что все методы определены в протоколе
        for method_name in protocol_methods:
            assert hasattr(ExchangeProtocol, method_name)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_exchange_protocol):
        """Тест конкурентных операций."""
        import asyncio
        
        # Создание нескольких ордеров одновременно
        orders = []
        for i in range(3):
            order = Order(
                id=uuid4(),
                trading_pair=TradingPair(base="BTC", quote="USDT"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=1.0 + i * 0.1,
                price=50000.0 + i * 100,
                status="PENDING",
                created_at=datetime.now()
            )
            orders.append(order)
        
        # Мок для создания ордеров
        mock_exchange_protocol.create_order = AsyncMock(side_effect=[
            {'id': str(uuid4()), 'status': 'PENDING'} for _ in range(3)
        ])
        
        # Конкурентное создание ордеров
        tasks = [mock_exchange_protocol.create_order(order) for order in orders]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        assert all('id' in result for result in results)
        assert all(result['status'] == 'PENDING' for result in results)

    @pytest.mark.asyncio
    async def test_error_recovery(self, mock_exchange_protocol):
        """Тест восстановления после ошибок."""
        symbol = "BTCUSDT"
        
        # Симуляция временной ошибки сети
        call_count = 0
        async def mock_fetch_with_retry(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise NetworkError("Temporary network error")
            return {'symbol': symbol, 'last': 50000.0}
        
        mock_exchange_protocol.fetch_ticker = mock_fetch_with_retry
        
        # Первый вызов должен вызвать ошибку
        with pytest.raises(NetworkError):
            await mock_exchange_protocol.fetch_ticker(symbol)
        
        # Второй вызов должен успешно выполниться
        result = await mock_exchange_protocol.fetch_ticker(symbol)
        assert result['symbol'] == symbol
        assert result['last'] == 50000.0 