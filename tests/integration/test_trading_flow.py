#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Интеграционные тесты торгового потока.
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from unittest.mock import AsyncMock, Mock

from application.use_cases.manage_orders import OrderManagementUseCase
from application.use_cases.manage_positions import PositionManagementUseCase
from application.use_cases.manage_risk import RiskManagementUseCase
from domain.entities.order import Order, OrderSide, OrderType
from domain.entities.position import Position
from domain.value_objects.money import Money
from domain.value_objects.volume import Volume
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from infrastructure.external_services.bybit_client import BybitClient


class TestTradingFlow:
    """Интеграционные тесты торгового потока."""

    @pytest.fixture
    def mock_exchange(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для мока биржи."""
        exchange = Mock(spec=BybitClient)
        exchange.create_order = AsyncMock(return_value={"id": "test_order_123", "status": "pending"})
        exchange.cancel_order = AsyncMock(return_value=True)
        exchange.fetch_order = AsyncMock(return_value={"id": "test_order_123", "status": "filled"})
        exchange.fetch_balance = AsyncMock(return_value={"USDT": {"free": 1000.0, "used": 0.0}})
        return exchange

    @pytest.fixture
    def mock_portfolio_repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для мока репозитория портфеля."""
        repo = Mock()
        repo.get_by_account_id = AsyncMock(return_value=None)
        repo.save = AsyncMock(return_value=True)
        repo.update = AsyncMock(return_value=True)
        return repo

    @pytest.fixture
    def mock_order_repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для мока репозитория ордеров."""
        repo = Mock()
        repo.save = AsyncMock(return_value=True)
        repo.get_by_id = AsyncMock(return_value=None)
        repo.update = AsyncMock(return_value=True)
        return repo

    @pytest.fixture
    def mock_position_repository(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для мока репозитория позиций."""
        repo = Mock()
        repo.save = AsyncMock(return_value=True)
        repo.get_by_trading_pair = AsyncMock(return_value=None)
        repo.update = AsyncMock(return_value=True)
        return repo

    @pytest.fixture
    def manage_orders_use_case(self, mock_exchange, mock_order_repository) -> Any:
        """Фикстура для use case управления ордерами."""
        return OrderManagementUseCase(exchange=mock_exchange, order_repository=mock_order_repository)

    @pytest.fixture
    def manage_positions_use_case(self, mock_position_repository) -> Any:
        """Фикстура для use case управления позициями."""
        return PositionManagementUseCase(position_repository=mock_position_repository)

    @pytest.fixture
    def manage_risk_use_case(self, mock_portfolio_repository) -> Any:
        """Фикстура для use case управления рисками."""
        return RiskManagementUseCase(portfolio_repository=mock_portfolio_repository)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_buy_order_flow(
        self,
        manage_orders_use_case,
        manage_positions_use_case,
        manage_risk_use_case,
        mock_exchange,
        mock_portfolio_repository,
    ) -> None:
        """Тест полного потока покупки."""
        # Arrange
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        # Mock portfolio
        portfolio = Mock()
        portfolio.total_equity = Money(Decimal("10000"), Currency.USD)
        portfolio.free_margin = Money(Decimal("10000"), Currency.USD)
        mock_portfolio_repository.get_by_account_id.return_value = portfolio

        # Act - Создание ордера
        order_result = await manage_orders_use_case.create_order(order)

        # Assert - Проверка создания ордера
        assert order_result["id"] == "test_order_123"
        assert order_result["status"] == "pending"
        mock_exchange.create_order.assert_called_once()

        # Act - Заполнение ордера
        mock_exchange.fetch_order.return_value = {
            "id": "test_order_123",
            "status": "filled",
            "filled": "0.001",
            "price": "50000",
        }

        filled_order = await manage_orders_use_case.get_order("test_order_123")

        # Assert - Проверка заполнения
        assert filled_order["status"] == "filled"
        assert filled_order["filled"] == "0.001"

        # Act - Создание позиции
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("50000"), Currency.USD),
        )

        position_result = await manage_positions_use_case.create_position(position)

        # Assert - Проверка создания позиции
        assert position_result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_sell_order_flow(
        self, manage_orders_use_case, manage_positions_use_case, mock_exchange, mock_position_repository
    ) -> None:
        """Тест полного потока продажи."""
        # Arrange - Существующая позиция
        existing_position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("51000"), Currency.USD),
        )
        mock_position_repository.get_by_trading_pair.return_value = existing_position

        # Act - Создание ордера продажи
        sell_order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("51000")),
        )

        order_result = await manage_orders_use_case.create_order(sell_order)

        # Assert - Проверка создания ордера продажи
        assert order_result["id"] == "test_order_123"
        mock_exchange.create_order.assert_called_once()

        # Act - Заполнение ордера продажи
        mock_exchange.fetch_order.return_value = {
            "id": "test_order_123",
            "status": "filled",
            "filled": "0.001",
            "price": "51000",
        }

        filled_order = await manage_orders_use_case.get_order("test_order_123")

        # Assert - Проверка заполнения
        assert filled_order["status"] == "filled"

        # Act - Закрытие позиции
        close_result = await manage_positions_use_case.close_position("BTCUSDT", Decimal("0.001"), Decimal("51000"))

        # Assert - Проверка закрытия позиции
        assert close_result is True

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_risk_management_integration(
        self, manage_orders_use_case, manage_risk_use_case, mock_exchange, mock_portfolio_repository
    ) -> None:
        """Тест интеграции управления рисками."""
        # Arrange
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.1")),  # Большой объем
            price=Price(Decimal("50000")),
        )

        # Mock portfolio с ограниченными средствами
        portfolio = Mock()
        portfolio.total_equity = Money(Decimal("1000"), Currency.USD)  # Мало средств
        portfolio.free_margin = Money(Decimal("1000"), Currency.USD)
        mock_portfolio_repository.get_by_account_id.return_value = portfolio

        # Act - Проверка риска
        risk_assessment = await manage_risk_use_case.assess_order_risk(order)

        # Assert - Проверка оценки риска
        assert risk_assessment["is_allowed"] is False
        assert "insufficient_funds" in risk_assessment["reasons"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_order_cancellation_flow(self, manage_orders_use_case, mock_exchange, mock_order_repository) -> None:
        """Тест потока отмены ордера."""
        # Arrange
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        # Act - Создание ордера
        order_result = await manage_orders_use_case.create_order(order)
        order_id = order_result["id"]

        # Act - Отмена ордера
        cancel_result = await manage_orders_use_case.cancel_order(order_id)

        # Assert - Проверка отмены
        assert cancel_result is True
        mock_exchange.cancel_order.assert_called_once_with(order_id)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_market_order_flow(self, manage_orders_use_case, mock_exchange, mock_order_repository) -> None:
        """Тест потока рыночного ордера."""
        # Arrange
        market_order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
        )

        # Mock текущую цену
        mock_exchange.fetch_ticker.return_value = {"last": 50000.0, "bid": 49999.0, "ask": 50001.0}

        # Act - Создание рыночного ордера
        order_result = await manage_orders_use_case.create_order(market_order)

        # Assert - Проверка создания рыночного ордера
        assert order_result["id"] == "test_order_123"
        mock_exchange.create_order.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_position_pnl_calculation(self, manage_positions_use_case, mock_position_repository) -> None:
        """Тест расчета PnL позиции."""
        # Arrange
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("51000"), Currency.USD),
        )

        # Act - Создание позиции
        await manage_positions_use_case.create_position(position)

        # Act - Обновление цены
        updated_position = await manage_positions_use_case.update_position_price("BTCUSDT", Decimal("51000"))

        # Assert - Проверка PnL
        assert updated_position.unrealized_pnl.value == Decimal("10")  # (51000-50000)*0.001

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(
        self, manage_orders_use_case, mock_exchange, mock_order_repository
    ) -> None:
        """Тест обработки ошибок в интеграции."""
        # Arrange - Ошибка биржи
        mock_exchange.create_order.side_effect = Exception("Exchange error")

        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        # Act & Assert - Проверка обработки ошибки
        with pytest.raises(Exception, match="Exchange error"):
            await manage_orders_use_case.create_order(order)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_order_handling(
        self, manage_orders_use_case, mock_exchange, mock_order_repository
    ) -> None:
        """Тест обработки конкурентных ордеров."""
        # Arrange
        orders = [
            Order(
                trading_pair="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Volume(Decimal("0.001")),
                price=Price(Decimal("50000")),
            ),
            Order(
                trading_pair="ETHUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Volume(Decimal("0.01")),
                price=Price(Decimal("3000")),
            ),
        ]

        # Act - Создание нескольких ордеров
        results = []
        for order in orders:
            result = await manage_orders_use_case.create_order(order)
            results.append(result)

        # Assert - Проверка создания всех ордеров
        assert len(results) == 2
        assert all(result["id"] == "test_order_123" for result in results)
        assert mock_exchange.create_order.call_count == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_portfolio_balance_integration(
        self, manage_orders_use_case, manage_risk_use_case, mock_exchange, mock_portfolio_repository
    ) -> None:
        """Тест интеграции с балансом портфеля."""
        # Arrange
        portfolio = Mock()
        portfolio.total_equity = Money(Decimal("10000"), Currency.USD)
        portfolio.free_margin = Money(Decimal("5000"), Currency.USD)  # Ограниченные средства
        mock_portfolio_repository.get_by_account_id.return_value = portfolio

        # Mock баланс биржи
        mock_exchange.fetch_balance.return_value = {
            "USDT": {"free": 5000.0, "used": 0.0},
            "BTC": {"free": 0.0, "used": 0.0},
        }

        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.1")),  # Большой объем
            price=Price(Decimal("50000")),
        )

        # Act - Проверка риска
        risk_assessment = await manage_risk_use_case.assess_order_risk(order)

        # Assert - Проверка оценки риска
        assert risk_assessment["is_allowed"] is False
        assert "insufficient_balance" in risk_assessment["reasons"]

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_order_status_tracking(self, manage_orders_use_case, mock_exchange, mock_order_repository) -> None:
        """Тест отслеживания статуса ордера."""
        # Arrange
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )

        # Act - Создание ордера
        order_result = await manage_orders_use_case.create_order(order)
        order_id = order_result["id"]

        # Act - Проверка статуса
        mock_exchange.fetch_order.return_value = {
            "id": order_id,
            "status": "partially_filled",
            "filled": "0.0005",
            "remaining": "0.0005",
        }

        status = await manage_orders_use_case.get_order_status(order_id)

        # Assert - Проверка статуса
        assert status["status"] == "partially_filled"
        assert status["filled"] == "0.0005"
        assert status["remaining"] == "0.0005"
