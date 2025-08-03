"""
Интеграционные тесты для application слоя.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock
from uuid import uuid4
from decimal import Decimal
from datetime import datetime
from application.use_cases.manage_orders import DefaultOrderManagementUseCase
from application.services.market_data_service import MarketDataService
from application.services.trading_service import DefaultTradingService
from application.types import CreateOrderRequest, CancelOrderRequest, GetOrdersRequest
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.order import OrderType, OrderSide, OrderStatus
from domain.types import PortfolioId, OrderId, TradingPair, Symbol
from domain.entities.trade import Trade
from domain.entities.trading_session import TradingSession
from domain.entities.trading_pair import TradingPair as DomainTradingPair

class TestApplicationIntegration:
    """Интеграционные тесты для application слоя."""
    @pytest.fixture
    def mock_repositories(self) -> tuple[Mock, Mock, Mock, Mock, Mock]:
        """Создает mock репозитории для интеграционных тестов."""
        order_repo = Mock()
        position_repo = Mock()
        portfolio_repo = Mock()
        session_repo = Mock()
        market_repo = Mock()
        # Настройка order_repo
        order_repo.create = AsyncMock()
        order_repo.get_by_id = AsyncMock()
        order_repo.update = AsyncMock()
        order_repo.get_by_portfolio_id = AsyncMock()
        # Настройка position_repo
        position_repo.create = AsyncMock()
        position_repo.get_by_id = AsyncMock()
        position_repo.update = AsyncMock()
        position_repo.get_by_symbol = AsyncMock()
        # Настройка portfolio_repo
        portfolio_repo.get_by_id = AsyncMock()
        # Настройка session_repo
        session_repo.create = AsyncMock()
        session_repo.get_by_id = AsyncMock()
        session_repo.update = AsyncMock()
        # Настройка market_repo
        market_repo.get_market_data = AsyncMock()
        market_repo.get_market_state = AsyncMock()
        market_repo.get_volume_profile = AsyncMock()
        market_repo.get_market_regime_analysis = AsyncMock()
        return order_repo, position_repo, portfolio_repo, session_repo, market_repo
    @pytest.fixture
    def use_cases_and_services(self, mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService]:
        """Создает экземпляры use cases и сервисов."""
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        order_management = DefaultOrderManagementUseCase(order_repo, portfolio_repo, position_repo)
        market_data_service = MarketDataService(market_repo)
        # Убираем лишний параметр session_repo из DefaultTradingService
        trading_service = DefaultTradingService(order_repo, position_repo, portfolio_repo)
        return order_management, market_data_service, trading_service
    @pytest.mark.asyncio
    async def test_complete_order_workflow(self, use_cases_and_services: tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService], mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> None:
        """Тест полного workflow создания и управления ордером."""
        order_management, market_data_service, trading_service = use_cases_and_services
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        # Настройка mock'ов
        portfolio_id = PortfolioId(uuid4())
        order_id = OrderId(uuid4())
        portfolio = Mock()
        portfolio.available_balance = Money(Decimal("10000"), Currency.USD)
        portfolio_repo.get_by_id.return_value = portfolio
        mock_order = Mock()
        mock_order.id = order_id
        mock_order.status = OrderStatus.PENDING
        mock_order.update_status = Mock()
        order_repo.create.return_value = mock_order
        order_repo.get_by_id.return_value = mock_order
        # 1. Создание ордера через use case
        create_request = CreateOrderRequest(
            portfolio_id=portfolio_id,
            trading_pair=DomainTradingPair("BTC", "USD"),  # type: ignore[call-arg, arg-type]
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            volume=Volume(Decimal("0.1"), Currency.BTC),
            price=Price(Decimal("50000"), Currency.USD, Currency.BTC)
        )
        create_result = await order_management.create_order(create_request)
        assert create_result.success is True
        assert create_result.order is not None
        # 2. Получение рыночных данных через сервис
        market_data = [
            {"timestamp": "2024-01-01T00:00:00", "close": "50000", "volume": "1000"},
            {"timestamp": "2024-01-01T01:00:00", "close": "51000", "volume": "1200"}
        ]
        market_repo.get_market_data.return_value = market_data
        market_summary = {
            "symbol": "BTC/USD",
            "last_price": "50000",
            "price_change": "1000",
            "volume": "10000"
        }
        market_repo.get_market_state.return_value = market_summary
        market_data_result = await market_data_service.get_market_data("BTC/USD", "1h", limit=100)
        assert market_data_result == market_data
        market_summary_result = await market_data_service.get_market_state("BTC/USD")
        assert market_summary_result == market_summary
        # 3. Исполнение ордера через trading service
        execute_result = await trading_service.execute_order(
            order_id=order_id,
            execution_price=Price(Decimal("50000"), Currency.USD, Currency.BTC),
            execution_quantity=Volume(Decimal("0.1"), Currency.BTC)
        )
        assert isinstance(execute_result, Trade)
        assert execute_result.order_id == order_id
        # 4. Отмена ордера через use case
        cancel_request = CancelOrderRequest(
            order_id=order_id,
            portfolio_id=portfolio_id
        )
        cancel_result = await order_management.cancel_order(cancel_request)
        assert cancel_result.cancelled is True
        # Проверяем вызовы репозиториев
        portfolio_repo.get_by_id.assert_called_with(portfolio_id)
        order_repo.create.assert_called()
        order_repo.get_by_id.assert_called_with(order_id)
        order_repo.update.assert_called()
        market_repo.get_market_data.assert_called_with(
            symbol="BTC/USD", 
            timeframe="1h", 
            start_time=None,
            end_time=None,
            limit=100
        )
        market_repo.get_market_state.assert_called_with("BTC/USD")
    @pytest.mark.asyncio
    async def test_market_data_integration(self, use_cases_and_services: tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService], mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> None:
        """Тест интеграции рыночных данных."""
        order_management, market_data_service, trading_service = use_cases_and_services
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        symbol = "BTC/USD"
        timeframe = "1h"
        # Настройка mock'ов для рыночных данных
        market_data = [
            {"timestamp": "2024-01-01T00:00:00", "close": "50000", "volume": "1000"},
            {"timestamp": "2024-01-01T01:00:00", "close": "51000", "volume": "1200"},
            {"timestamp": "2024-01-01T02:00:00", "close": "52000", "volume": "1100"}
        ]
        market_summary = {
            "symbol": symbol,
            "last_price": "52000",
            "price_change": "2000",
            "volume": "10000",
            "high": "52000",
            "low": "50000"
        }
        volume_profile = {
            "symbol": symbol,
            "poc_price": "51000",
            "total_volume": "10000",
            "volume_profile": {"51000": "5000"}
        }
        market_regime = {
            "symbol": symbol,
            "regime": "trending",
            "volatility": "20.5",
            "trend_strength": "75.0"
        }
        market_repo.get_market_data.return_value = market_data
        market_repo.get_market_state.return_value = market_summary
        market_repo.get_volume_profile.return_value = volume_profile
        market_repo.get_market_regime_analysis.return_value = market_regime
        # Получение различных типов рыночных данных
        data_result = await market_data_service.get_market_data(symbol, timeframe, limit=100)
        assert data_result == market_data
        assert len(data_result) == 3
        summary_result = await market_data_service.get_market_state(symbol)
        assert summary_result == market_summary
        assert summary_result is not None and hasattr(summary_result, "last_price") and summary_result.last_price == "52000"
        profile_result = await market_data_service.get_volume_profile(symbol, timeframe)
        assert profile_result == volume_profile
        assert profile_result is not None and profile_result["poc_price"] == "51000"
        regime_result = await market_data_service.get_market_regime_analysis(symbol, timeframe)
        assert regime_result == market_regime
        assert regime_result["regime"] == "trending"
        # Проверяем вызовы репозитория
        assert market_repo.get_market_data.call_count == 1
        assert market_repo.get_market_state.call_count == 1
        assert market_repo.get_volume_profile.call_count == 1
        assert market_repo.get_market_regime_analysis.call_count == 1
    @pytest.mark.asyncio
    async def test_trading_session_integration(self, use_cases_and_services: tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService], mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> None:
        """Тест интеграции торговых сессий."""
        order_management, market_data_service, trading_service = use_cases_and_services
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        portfolio_id = PortfolioId(uuid4())
        session_id = uuid4()
        # Настройка mock'ов
        portfolio = Mock()
        portfolio.available_balance = Money(Decimal("10000"), Currency.USD)
        portfolio_repo.get_by_id.return_value = portfolio
        mock_session = Mock()
        mock_session.id = session_id
        mock_session.status = "ACTIVE"
        mock_session.update = Mock()
        session_repo.create.return_value = mock_session
        session_repo.get_by_id.return_value = mock_session
        mock_orders = [
            Mock(symbol="BTC/USD", status=OrderStatus.FILLED),
            Mock(symbol="ETH/USD", status=OrderStatus.PENDING)
        ]
        order_repo.get_by_portfolio_id.return_value = mock_orders
        # 1. Создание торговой сессии
        session_result = await trading_service.start_trading_session(portfolio_id)
        assert isinstance(session_result, TradingSession)
        assert session_result.portfolio_id == portfolio_id
        # 2. Получение ордеров портфеля
        orders_result = await trading_service.get_active_orders(portfolio_id)
        assert isinstance(orders_result, list)
        # 3. Завершение торговой сессии
        end_result = await trading_service.end_trading_session(session_id)
        assert isinstance(end_result, TradingSession)
        # Проверяем вызовы репозиториев
        session_repo.create.assert_called_once()
        session_repo.get_by_id.assert_called_with(session_id)
        session_repo.update.assert_called()
        order_repo.get_by_portfolio_id.assert_called_with(portfolio_id)
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, use_cases_and_services: tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService], mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> None:
        """Тест обработки ошибок в интеграции."""
        order_management, market_data_service, trading_service = use_cases_and_services
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        # Тест с несуществующим портфелем
        portfolio_repo.get_by_id.return_value = None
        create_request = CreateOrderRequest(
            portfolio_id=PortfolioId(uuid4()),
            trading_pair=DomainTradingPair("BTC", "USD"),  # type: ignore[call-arg, arg-type]
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            volume=Volume(Decimal("0.1"), Currency.BTC),
            price=Price(Decimal("50000"), Currency.USD, Currency.BTC)
        )
        create_result = await order_management.create_order(create_request)
        assert create_result.success is False
        assert "Portfolio not found" in create_result.message
        # Тест с недостаточными средствами
        poor_portfolio = Mock()
        poor_portfolio.available_balance = Money(Decimal("100"), Currency.USD)
        portfolio_repo.get_by_id.return_value = poor_portfolio
        create_result = await order_management.create_order(create_request)
        assert create_result.success is False
        assert "Insufficient funds" in create_result.message
        # Тест с несуществующим ордером
        order_repo.get_by_id.return_value = None
        cancel_request = CancelOrderRequest(
            order_id=OrderId(uuid4()),
            portfolio_id=PortfolioId(uuid4())
        )
        cancel_result = await order_management.cancel_order(cancel_request)
        assert cancel_result.cancelled is False
        assert "Order not found" in cancel_result.message
    @pytest.mark.asyncio
    async def test_data_consistency_integration(self, use_cases_and_services: tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService], mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> None:
        """Тест согласованности данных в интеграции."""
        order_management, market_data_service, trading_service = use_cases_and_services
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        portfolio_id = PortfolioId(uuid4())
        order_id = OrderId(uuid4())
        symbol = "BTC/USD"
        # Настройка согласованных данных
        portfolio = Mock()
        portfolio.available_balance = Money(Decimal("10000"), Currency.USD)
        portfolio_repo.get_by_id.return_value = portfolio
        mock_order = Mock()
        mock_order.id = order_id
        mock_order.portfolio_id = portfolio_id
        mock_order.symbol = symbol
        mock_order.status = OrderStatus.PENDING
        mock_order.update_status = Mock()
        order_repo.create.return_value = mock_order
        order_repo.get_by_id.return_value = mock_order
        market_summary = {
            "symbol": symbol,
            "last_price": "50000",
            "volume": "10000"
        }
        market_repo.get_market_state.return_value = market_summary
        # Создание ордера
        create_request = CreateOrderRequest(
            portfolio_id=portfolio_id,
            trading_pair=DomainTradingPair("BTC", "USD"),  # type: ignore[call-arg, arg-type]
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            volume=Volume(Decimal("0.1"), Currency.BTC),
            price=Price(Decimal("50000"), Currency.USD, Currency.BTC)
        )
        create_result = await order_management.create_order(create_request)
        assert create_result.success is True
        # Проверка согласованности данных
        get_request = GetOrdersRequest(
            portfolio_id=portfolio_id,
            trading_pair=DomainTradingPair("BTC", "USD"),  # type: ignore[call-arg, arg-type]
            status=OrderStatus.PENDING
        )
        get_result = await order_management.get_orders(get_request)
        assert get_result.success is True
        assert len(get_result.orders) == 1
        assert get_result.orders[0].symbol == symbol
        # Получение рыночных данных для того же символа
        market_data_result = await market_data_service.get_market_state(symbol)
        assert market_data_result is not None and hasattr(market_data_result, "symbol") and market_data_result.symbol == symbol
        assert market_data_result is not None and hasattr(market_data_result, "last_price") and market_data_result.last_price == "50000"
    @pytest.mark.asyncio
    async def test_performance_integration(self, use_cases_and_services: tuple[DefaultOrderManagementUseCase, MarketDataService, DefaultTradingService], mock_repositories: tuple[Mock, Mock, Mock, Mock, Mock]) -> None:
        """Тест производительности в интеграции."""
        order_management, market_data_service, trading_service = use_cases_and_services
        order_repo, position_repo, portfolio_repo, session_repo, market_repo = mock_repositories
        portfolio_id = PortfolioId(uuid4())
        # Настройка mock'ов для тестирования производительности
        portfolio = Mock()
        portfolio.available_balance = Money(Decimal("100000"), Currency.USD)
        portfolio_repo.get_by_id.return_value = portfolio
        mock_orders = [
            Mock(symbol="BTC/USD", status=OrderStatus.FILLED, realized_pnl=Decimal("100")),
            Mock(symbol="ETH/USD", status=OrderStatus.FILLED, realized_pnl=Decimal("200")),
            Mock(symbol="ADA/USD", status=OrderStatus.FILLED, realized_pnl=Decimal("-50"))
        ]
        order_repo.get_by_portfolio_id.return_value = mock_orders
        # Расчет торговой статистики
        stats_result = await trading_service.get_trading_statistics()
        assert "total_trades" in stats_result
        assert "winning_trades" in stats_result
        assert "losing_trades" in stats_result
        assert "total_pnl" in stats_result
        assert "win_rate" in stats_result
        # Расчет метрик риска
        risk_result = await trading_service.get_risk_metrics()
        assert "sharpe_ratio" in risk_result
        assert "max_drawdown" in risk_result
        assert "profit_factor" in risk_result
        assert "var_95" in risk_result
        assert isinstance(risk_result["sharpe_ratio"], (int, float))
        assert isinstance(risk_result["max_drawdown"], (int, float))
        assert isinstance(risk_result["profit_factor"], (int, float))
        assert isinstance(risk_result["var_95"], (int, float)) 
