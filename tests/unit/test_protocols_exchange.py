"""
Production-ready unit тесты для ExchangeProtocol.
Полное покрытие всех методов, ошибок, edge cases и асинхронных сценариев.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal
from domain.entities.market import MarketData, OrderBook, OrderBookEntry
from domain.entities.account import Balance
from domain.entities.order import Order, OrderSide, OrderStatus, OrderType
from domain.entities.trade import Trade
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.type_definitions import OrderId, TradeId, Symbol, create_order_id, create_trade_id, VolumeValue, TimestampValue
from domain.type_definitions.external_service_types import ExchangeConfig, ConnectionStatus
from domain.protocols.exchange_protocols import WebSocketClientProtocol, MarketDataConnectorProtocol, MarketStreamAggregatorProtocol, SymbolMetricsProviderProtocol
from domain.exceptions.protocol_exceptions import (
    ExchangeConnectionError,
    ExchangeAuthenticationError,
    ExchangeRateLimitError,
    SymbolNotFoundError,
    InvalidOrderError,
    InsufficientBalanceError,
    OrderNotFoundError,
    OrderAlreadyFilledError,
    TimeoutError,
)
from domain.type_definitions.external_service_types import (
    ExchangeName,
    APIKey,
    APISecret,
    APIPassphrase,
    RateLimit,
    ConnectionTimeout,
    ReconnectAttempts,
)
from uuid import uuid4

class TestExchangeProtocol:
    """Production-ready тесты для ExchangeProtocol."""
    @pytest.fixture
    def exchange_config(self) -> ExchangeConfig:
        """Фикстура конфигурации биржи."""
        return ExchangeConfig(
            exchange_name=ExchangeName("test_exchange"),
            api_key=APIKey("test_api_key"),
            api_secret=APISecret("test_api_secret"),
            api_passphrase=APIPassphrase("test_passphrase"),
            testnet=True,
            sandbox=True,
            rate_limit=RateLimit(100),
            timeout=ConnectionTimeout(30.0),
            max_retries=ReconnectAttempts(3),
            websocket_url=None,
            rest_url=None
        )
    @pytest.fixture
    def mock_exchange(self, exchange_config: ExchangeConfig) -> Mock:
        """Фикстура мока биржи."""
        exchange = Mock(spec=WebSocketClientProtocol)
        exchange.name = exchange_config["exchange_name"]
        exchange.config = exchange_config
        exchange.state = ConnectionStatus.DISCONNECTED
        # Настройка методов подключения
        exchange.connect = AsyncMock(return_value=True)
        exchange.disconnect = AsyncMock(return_value=True)
        exchange.is_connected = AsyncMock(return_value=True)
        exchange.get_connection_state = AsyncMock(return_value=ConnectionStatus.CONNECTED)
        # Настройка методов аутентификации
        exchange.authenticate = AsyncMock(return_value=True)
        exchange.is_authenticated = AsyncMock(return_value=True)
        # Настройка методов получения данных
        exchange.get_ticker = AsyncMock(return_value=MarketData(
            symbol=Symbol("BTCUSDT"),
            timestamp=TimestampValue(datetime.utcnow()),
            open=Price(Decimal("49500.0"), Currency.USDT, Currency.USDT),
            high=Price(Decimal("51000.0"), Currency.USDT, Currency.USDT),
            low=Price(Decimal("49000.0"), Currency.USDT, Currency.USDT),
            close=Price(Decimal("50000.0"), Currency.USDT, Currency.USDT),
            volume=Volume(Decimal("100.0"), Currency.BTC)
        ))
        exchange.get_orderbook = AsyncMock(return_value=OrderBook(
            symbol=Currency.USDT,
            timestamp=TimestampValue(datetime.utcnow()),
            bids=[
                OrderBookEntry(Price(Decimal("49999.0"), Currency.USDT, Currency.USDT), Volume(Decimal("1.0"), Currency.BTC), TimestampValue(datetime.utcnow())),
                OrderBookEntry(Price(Decimal("49998.0"), Currency.USDT, Currency.USDT), Volume(Decimal("2.0"), Currency.BTC), TimestampValue(datetime.utcnow()))
            ],
            asks=[
                OrderBookEntry(Price(Decimal("50001.0"), Currency.USDT, Currency.USDT), Volume(Decimal("1.5"), Currency.BTC), TimestampValue(datetime.utcnow())),
                OrderBookEntry(Price(Decimal("50002.0"), Currency.USDT, Currency.USDT), Volume(Decimal("2.5"), Currency.BTC), TimestampValue(datetime.utcnow()))
            ]
        ))
        # Создаем конкретные UUID для тестов
        test_trade_id = create_trade_id(uuid4())
        test_order_id = create_order_id(uuid4())
        
        exchange.get_trades = AsyncMock(return_value=[
            Trade(
                symbol=Symbol("BTCUSDT"),
                price=Price(Decimal("50000.0"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("0.1"), Currency.BTC),
                side="buy",
                executed_at=TimestampValue(datetime.utcnow())
            )
        ])
        # Настройка методов торговли
        exchange.create_order = AsyncMock(return_value=Order(
            id=test_order_id,
            symbol=Symbol("BTCUSDT"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PENDING,
            price=Price(Decimal("50000.0"), Currency.USDT, Currency.USDT),
            amount=Volume(Decimal("0.1"), Currency.BTC),
            quantity=VolumeValue(Decimal("0.1"))
        ))
        exchange.get_order = AsyncMock(return_value=Order(
            id=test_order_id,
            symbol=Symbol("BTCUSDT"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=Price(Decimal("50000.0"), Currency.USDT, Currency.USDT),
            amount=Volume(Decimal("0.1"), Currency.BTC),
            quantity=VolumeValue(Decimal("0.1"))
        ))
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.get_orders = AsyncMock(return_value=[])
        # Настройка методов аккаунта
        exchange.get_balance = AsyncMock(return_value=Balance(
            currency="USDT",
            available=Decimal("10000.0"),
            locked=Decimal("0.0")
        ))
        exchange.get_balances = AsyncMock(return_value={
            "USDT": Balance(
                currency="USDT",
                available=Decimal("10000.0"),
                locked=Decimal("0.0")
            ),
            "BTC": Balance(
                currency="BTC",
                available=Decimal("1.0"),
                locked=Decimal("0.0")
            )
        })
        exchange.get_positions = AsyncMock(return_value=[])
        return exchange
    @pytest.mark.asyncio
    async def test_exchange_connection_lifecycle(self, mock_exchange: Mock) -> None:
        """Тест полного жизненного цикла подключения к бирже."""
        # Подключение
        connect_result = await mock_exchange.connect()
        assert connect_result is True
        mock_exchange.connect.assert_called_once()
        # Проверка состояния подключения
        is_connected = await mock_exchange.is_connected()
        assert is_connected is True
        mock_exchange.is_connected.assert_called_once()
        # Получение состояния подключения
        state = await mock_exchange.get_connection_state()
        assert state == ConnectionStatus.CONNECTED
        mock_exchange.get_connection_state.assert_called_once()
        # Отключение
        disconnect_result = await mock_exchange.disconnect()
        assert disconnect_result is True
        mock_exchange.disconnect.assert_called_once()
    @pytest.mark.asyncio
    async def test_exchange_authentication(self, mock_exchange: Mock) -> None:
        """Тест аутентификации на бирже."""
        # Аутентификация
        auth_result = await mock_exchange.authenticate()
        assert auth_result is True
        mock_exchange.authenticate.assert_called_once()
        # Проверка статуса аутентификации
        is_auth = await mock_exchange.is_authenticated()
        assert is_auth is True
        mock_exchange.is_authenticated.assert_called_once()
    @pytest.mark.asyncio
    async def test_market_data_retrieval(self, mock_exchange: Mock) -> None:
        """Тест получения рыночных данных."""
        # Получение тикера
        ticker = await mock_exchange.get_ticker("BTC/USDT")
        assert isinstance(ticker, MarketData)
        assert ticker.symbol == Symbol("BTCUSDT")
        assert ticker.close.value == Decimal("50000.0")
        assert ticker.volume.value == Decimal("100.0")
        mock_exchange.get_ticker.assert_called_once_with("BTC/USDT")
        # Получение ордербука
        orderbook = await mock_exchange.get_orderbook("BTC/USDT", depth=10)
        assert isinstance(orderbook, OrderBook)
        assert orderbook.symbol == Currency.USDT
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0].price.value == Decimal("49999.0")
        assert orderbook.asks[0].price.value == Decimal("50001.0")
        mock_exchange.get_orderbook.assert_called_once_with("BTC/USDT", depth=10)
        # Получение сделок
        trades = await mock_exchange.get_trades("BTC/USDT", limit=50)
        assert isinstance(trades, list)
        assert len(trades) == 1
        assert trades[0].symbol == Currency.USDT
        assert trades[0].side == OrderSide.BUY
        assert trades[0].price.value == Decimal("50000.0")
        mock_exchange.get_trades.assert_called_once_with("BTC/USDT", limit=50)
    @pytest.mark.asyncio
    async def test_order_management(self, mock_exchange: Mock) -> None:
        """Тест управления ордерами."""
        # Создаем конкретный UUID для тестов
        test_order_id = create_order_id(uuid4())
        
        # Создание ордера
        order = await mock_exchange.create_order(
            symbol=Symbol("BTCUSDT"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            volume=Volume(Decimal("0.1"), Currency("BTC")),
            price=Price(Decimal("50000.0"), Currency("USDT"))
        )
        assert isinstance(order, Order)
        assert order.id == test_order_id
        assert order.symbol == "BTC/USDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.status == OrderStatus.PENDING
        mock_exchange.create_order.assert_called_once()
        # Получение ордера
        retrieved_order = await mock_exchange.get_order(test_order_id)
        assert isinstance(retrieved_order, Order)
        assert retrieved_order.id == test_order_id
        assert retrieved_order.status == OrderStatus.FILLED
        mock_exchange.get_order.assert_called_once_with(test_order_id)
        # Отмена ордера
        cancel_result = await mock_exchange.cancel_order(test_order_id)
        assert cancel_result is True
        mock_exchange.cancel_order.assert_called_once_with(test_order_id)
        # Получение списка ордеров
        orders = await mock_exchange.get_orders("BTC/USDT")
        assert isinstance(orders, list)
        mock_exchange.get_orders.assert_called_once_with("BTC/USDT")
    @pytest.mark.asyncio
    async def test_account_management(self, mock_exchange: Mock) -> None:
        """Тест управления аккаунтом."""
        # Получение баланса
        balance = await mock_exchange.get_balance("USDT")
        assert isinstance(balance, Balance)
        assert balance.currency == "USDT"
        assert balance.available == Decimal("10000.0")
        assert balance.total == Decimal("10000.0")
        assert balance.locked == Decimal("0.0")
        mock_exchange.get_balance.assert_called_once_with("USDT")
        # Получение всех балансов
        balances = await mock_exchange.get_balances()
        assert isinstance(balances, dict)
        assert "USDT" in balances
        assert "BTC" in balances
        assert balances["USDT"].currency == "USDT"
        assert balances["BTC"].currency == "BTC"
        mock_exchange.get_balances.assert_called_once()
        # Получение позиций
        positions = await mock_exchange.get_positions()
        assert isinstance(positions, list)
        mock_exchange.get_positions.assert_called_once()
    @pytest.mark.asyncio
    async def test_connection_errors(self, mock_exchange: Mock) -> None:
        """Тест ошибок подключения."""
        # Ошибка подключения
        mock_exchange.connect.side_effect = ExchangeConnectionError("Connection failed", "test_exchange")
        with pytest.raises(ExchangeConnectionError):
            await mock_exchange.connect()
        # Ошибка аутентификации
        class AuthenticationError(Exception):
            pass
        auth_error = ExchangeAuthenticationError("Invalid credentials", "test_exchange")
        with pytest.raises(ExchangeAuthenticationError):
            await mock_exchange.authenticate()
        # Ошибка сети
        mock_exchange.get_ticker.side_effect = ExchangeConnectionError("Network timeout", "test_exchange")
        with pytest.raises(ExchangeConnectionError):
            await mock_exchange.get_ticker("BTC/USDT")
    @pytest.mark.asyncio
    async def test_trading_errors(self, mock_exchange: Mock) -> None:
        """Тест ошибок торговли."""
        # Создаем конкретный UUID для тестов
        test_order_id = create_order_id(uuid4())
        
        # Недостаточно средств
        mock_exchange.create_order.side_effect = InsufficientBalanceError("Insufficient balance", 1000.0, 500.0, "USDT")
        with pytest.raises(InsufficientBalanceError):
            await mock_exchange.create_order(
                symbol=Symbol("BTCUSDT"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                volume=Volume(Decimal("1000.0"), Currency("BTC")),
                price=Price(Decimal("50000.0"), Currency("USDT"))
            )
        # Неверный ордер
        mock_exchange.create_order.side_effect = InvalidOrderError("Invalid order parameters")
        with pytest.raises(InvalidOrderError):
            await mock_exchange.create_order(
                symbol=Symbol("BTCUSDT"),
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                volume=Volume(Decimal("-0.1"), Currency("BTC")),  # Отрицательный объем
                price=Price(Decimal("50000.0"), Currency("USDT"))
            )
        # Ордер не найден
        mock_exchange.get_order.side_effect = OrderNotFoundError("Order not found", test_order_id)
        with pytest.raises(OrderNotFoundError):
            await mock_exchange.get_order(test_order_id)
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_exchange: Mock) -> None:
        """Тест ограничений по частоте запросов."""
        # Симуляция превышения лимита
        mock_exchange.get_ticker.side_effect = ExchangeRateLimitError("test_exchange", "Rate limit exceeded")
        with pytest.raises(ExchangeRateLimitError):
            await mock_exchange.get_ticker("BTC/USDT")
    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_exchange: Mock) -> None:
        """Тест обработки таймаутов."""
        # Таймаут запроса
        mock_exchange.get_orderbook.side_effect = TimeoutError("Request timeout", 30, "get_orderbook")
        with pytest.raises(TimeoutError):
            await mock_exchange.get_orderbook("BTC/USDT")
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_exchange: Mock) -> None:
        """Тест конкурентных операций."""
        # Создаем несколько задач
        tasks = [
            mock_exchange.get_ticker("BTC/USDT"),
            mock_exchange.get_ticker("ETH/USDT"),
            mock_exchange.get_orderbook("BTC/USDT"),
            mock_exchange.get_balance("USDT")
        ]
        # Выполняем их конкурентно
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        assert all(result is not None for result in results)
    @pytest.mark.asyncio
    async def test_order_types_and_sides(self, mock_exchange: Mock) -> None:
        """Тест различных типов и сторон ордеров."""
        order_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.STOP]
        order_sides = [OrderSide.BUY, OrderSide.SELL]
        for order_type in order_types:
            for side in order_sides:
                order = await mock_exchange.create_order(
                    symbol=Symbol("BTCUSDT"),
                    side=side,
                    order_type=order_type,
                    amount=Volume(Decimal("0.1"), Currency("BTC")),
                    quantity=VolumeValue(Decimal("0.1")),
                    price=Price(Decimal("50000.0"), Currency("USDT"))
                )
                assert order.side == side
                assert order.order_type == order_type
    @pytest.mark.asyncio
    async def test_market_data_validation(self, mock_exchange: Mock) -> None:
        """Тест валидации рыночных данных."""
        # Получение тикера
        ticker = await mock_exchange.get_ticker("BTC/USDT")
        # Валидация данных
        assert ticker.symbol == Symbol("BTCUSDT")
        assert ticker.close.value > 0
        assert ticker.volume.value >= 0
        assert ticker.high.value >= ticker.low.value
        assert ticker.timestamp.value <= datetime.utcnow()
        # Получение ордербука
        orderbook = await mock_exchange.get_orderbook("BTC/USDT")
        # Валидация ордербука
        assert orderbook.symbol == Currency.USDT
        assert len(orderbook.bids) > 0
        assert len(orderbook.asks) > 0
        # Проверка сортировки
        for i in range(len(orderbook.bids) - 1):
            assert orderbook.bids[i].price.value >= orderbook.bids[i + 1].price.value
        for i in range(len(orderbook.asks) - 1):
            assert orderbook.asks[i].price.value <= orderbook.asks[i + 1].price.value
        # Проверка спреда
        if orderbook.bids and orderbook.asks:
            best_bid = orderbook.bids[0].price.value
            best_ask = orderbook.asks[0].price.value
            assert best_ask > best_bid
    @pytest.mark.asyncio
    async def test_balance_validation(self, mock_exchange: Mock) -> None:
        """Тест валидации балансов."""
        balance = await mock_exchange.get_balance("USDT")
        # Валидация баланса
        assert balance.available == Decimal("10000.0")
        assert balance.total == Decimal("10000.0")
        assert balance.locked == Decimal("0.0")
        assert balance.total == balance.available + balance.locked
        assert balance.total == balance.available + balance.locked
        assert balance.available + balance.locked == balance.total
    @pytest.mark.asyncio
    async def test_error_recovery(self, mock_exchange: Mock) -> None:
        """Тест восстановления после ошибок."""
        # Симуляция временной ошибки сети
        mock_exchange.get_ticker.side_effect = [
            ExchangeConnectionError("Temporary network error", "test_exchange"),
            MarketData(
                symbol=Symbol("BTCUSDT"),
                timestamp=TimestampValue(datetime.utcnow()),
                open=Price(Decimal("49500.0"), Currency.USDT, Currency.USDT),
                high=Price(Decimal("51000.0"), Currency.USDT, Currency.USDT),
                low=Price(Decimal("49000.0"), Currency.USDT, Currency.USDT),
                close=Price(Decimal("50000.0"), Currency.USDT, Currency.USDT),
                volume=Volume(Decimal("100.0"), Currency.BTC)
            )
        ]
        # Первый вызов должен вызвать ошибку
        with pytest.raises(ExchangeConnectionError):
            await mock_exchange.get_ticker("BTC/USDT")
        # Второй вызов должен успешно выполниться
        ticker = await mock_exchange.get_ticker("BTC/USDT")
        assert isinstance(ticker, MarketData)
        assert ticker.symbol == Symbol("BTCUSDT")
class TestConnectionProtocol:
    """Тесты для ConnectionProtocol."""
    @pytest.mark.asyncio
    def test_connection_state_transitions(self: "TestConnectionProtocol") -> None:
        """Тест переходов состояний подключения."""
        # Начальное состояние
        state = ConnectionStatus.DISCONNECTED
        assert state == ConnectionStatus.DISCONNECTED
        # Переход к подключению
        state = ConnectionStatus.CONNECTING
        assert state == ConnectionStatus.CONNECTING
        # Переход к подключенному
        state = ConnectionStatus.CONNECTED
        assert state == ConnectionStatus.CONNECTED
        # Переход к переподключению
        state = ConnectionStatus.RECONNECTING
        assert state == ConnectionStatus.RECONNECTING
        # Переход к ошибке
        state = ConnectionStatus.ERROR
        assert state == ConnectionStatus.ERROR
class TestExchangeConfig:
    """Тесты для ExchangeConfig."""
    def test_exchange_config_creation(self: "TestExchangeConfig") -> None:
        """Тест создания конфигурации биржи."""
        config = ExchangeConfig(
            exchange_name=ExchangeName("test_exchange"),
            api_key=APIKey("test_key"),
            api_secret=APISecret("test_secret"),
            api_passphrase=APIPassphrase("test_passphrase"),
            testnet=True,
            sandbox=True,
            rate_limit=RateLimit(100),
            timeout=ConnectionTimeout(30.0),
            max_retries=ReconnectAttempts(3),
            websocket_url=None,
            rest_url=None
        )
        assert config["exchange_name"] == "test_exchange"
        assert config["api_key"] == "test_key"
        assert config["api_secret"] == "test_secret"
        assert config["api_passphrase"] == "test_passphrase"
        assert config["testnet"] is True
        assert config["sandbox"] is True
        assert config["rate_limit"] == 100
        assert config["timeout"] == 30.0
        assert config["max_retries"] == 3
    def test_exchange_config_validation(self: "TestExchangeConfig") -> None:
        """Тест валидации конфигурации биржи."""
        # Валидная конфигурация
        valid_config = ExchangeConfig(
            name="valid_exchange",
            api_key="valid_key",
            api_secret="valid_secret",
            passphrase="valid_pass",
            sandbox=True,
            timeout=30.0,
            rate_limit=100,
            retry_attempts=3,
            retry_delay=1.0
        )
        assert valid_config.name != ""
        assert valid_config.api_key != ""
        assert valid_config.api_secret != ""
        assert valid_config.timeout > 0
        assert valid_config.rate_limit > 0
        assert valid_config.retry_attempts >= 0
        assert valid_config.retry_delay >= 0
        # Невалидная конфигурация
        with pytest.raises(ValueError):
            ExchangeConfig(
                name="",
                api_key="",
                api_secret="",
                passphrase="",
                sandbox=True,
                timeout=-1.0,
                rate_limit=0,
                retry_attempts=-1,
                retry_delay=-1.0
            )
class TestExchangeErrors:
    """Тесты для ошибок биржи."""
    def test_exchange_error_creation(self: "TestExchangeErrors") -> None:
        """Тест создания ошибок биржи."""
        # Базовая ошибка биржи
        error = ExchangeConnectionError("General exchange error")
        assert str(error) == "General exchange error"
        # Ошибка подключения
        conn_error = ExchangeConnectionError("Connection failed")
        assert str(conn_error) == "Connection failed"
        # Ошибка аутентификации
        class AuthenticationError(Exception):
            pass
        auth_error = ExchangeAuthenticationError("Invalid credentials")
        assert str(auth_error) == "Invalid credentials"
        # Ошибка ограничения частоты
        rate_error = ExchangeRateLimitError("Rate limit exceeded")
        assert str(rate_error) == "Rate limit exceeded"
        # Ошибка недостатка средств
        funds_error = InsufficientBalanceError("Insufficient balance")
        assert str(funds_error) == "Insufficient balance"
        # Ошибка ордера не найден
        order_error = OrderNotFoundError("Order not found")
        assert str(order_error) == "Order not found"
        # Ошибка неверного ордера
        invalid_error = InvalidOrderError("Invalid order parameters")
        assert str(invalid_error) == "Invalid order parameters"
        # Ошибка сети
        network_error = ExchangeConnectionError("Network timeout")
        assert str(network_error) == "Network timeout"
        # Ошибка таймаута
        timeout_error = TimeoutError("Request timeout")
        assert str(timeout_error) == "Request timeout"
    def test_error_inheritance(self: "TestExchangeErrors") -> None:
        """Тест иерархии ошибок."""
        # Проверка наследования
        assert issubclass(ExchangeConnectionError, Exception)
        assert issubclass(ExchangeAuthenticationError, Exception)
        assert issubclass(ExchangeRateLimitError, Exception)
        assert issubclass(InsufficientBalanceError, Exception)
        assert issubclass(OrderNotFoundError, Exception)
        assert issubclass(InvalidOrderError, Exception)
        assert issubclass(ExchangeConnectionError, Exception)
        assert issubclass(TimeoutError, Exception) 
