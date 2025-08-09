#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
E2E тесты полной торговой сессии.
"""
import asyncio
import pytest
from unittest.mock import Mock, patch
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from unittest.mock import AsyncMock, Mock, patch
from application.di_container import DIContainer
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


class TestCompleteTradingSession:
    """E2E тесты полной торговой сессии."""

    @pytest.fixture
    def mock_di_container(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для мока DI контейнера."""
        container = Mock(spec=DIContainer)
        # Mock сервисов
        container.get = Mock()
        # Mock биржи
        mock_exchange = Mock(spec=BybitClient)
        mock_exchange.create_order = AsyncMock(return_value={"id": "e2e_order_123", "status": "pending"})
        mock_exchange.cancel_order = AsyncMock(return_value=True)
        mock_exchange.fetch_order = AsyncMock(return_value={"id": "e2e_order_123", "status": "filled"})
        mock_exchange.fetch_balance = AsyncMock(return_value={"USDT": {"free": 10000.0, "used": 0.0}})
        mock_exchange.fetch_ticker = AsyncMock(return_value={"last": 50000.0, "bid": 49999.0, "ask": 50001.0})
        # Mock репозиториев
        mock_order_repo = Mock()
        mock_order_repo.save = AsyncMock(return_value=True)
        mock_order_repo.get_by_id = AsyncMock(return_value=None)
        mock_order_repo.update = AsyncMock(return_value=True)
        mock_position_repo = Mock()
        mock_position_repo.save = AsyncMock(return_value=True)
        mock_position_repo.get_by_trading_pair = AsyncMock(return_value=None)
        mock_position_repo.update = AsyncMock(return_value=True)
        mock_portfolio_repo = Mock()
        mock_portfolio_repo.get_by_account_id = AsyncMock(return_value=None)
        mock_portfolio_repo.save = AsyncMock(return_value=True)
        mock_portfolio_repo.update = AsyncMock(return_value=True)
        # Настройка возвращаемых значений
        container.get.side_effect = lambda service_type: {
            BybitClient: mock_exchange,
            "order_repository": mock_order_repo,
            "position_repository": mock_position_repo,
            "portfolio_repository": mock_portfolio_repo,
        }.get(service_type, Mock())
        return container

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с тестовыми рыночными данными."""
        return {
            "symbol": "BTCUSDT",
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": "2024-01-01T00:00:00Z",
            "open": 49900.0,
            "high": 50100.0,
            "low": 49800.0,
            "close": 50000.0,
        }

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_trading_session_buy_sell(self, mock_di_container, sample_market_data) -> None:
        """Тест полной торговой сессии: покупка и продажа."""
        # Arrange
        with patch("application.di_container.DIContainer", return_value=mock_di_container):
            # Создание use cases
            manage_orders = OrderManagementUseCase(
                exchange=mock_di_container.get(BybitClient), order_repository=mock_di_container.get("order_repository")
            )
            manage_positions = PositionManagementUseCase(
                position_repository=mock_di_container.get("position_repository")
            )
            manage_risk = RiskManagementUseCase(portfolio_repository=mock_di_container.get("portfolio_repository"))
        # Act - Шаг 1: Анализ рынка и принятие решения о покупке
        market_analysis = await self._analyze_market(sample_market_data)
        assert market_analysis["should_trade"] is True
        assert market_analysis["action"] == "buy"
        # Act - Шаг 2: Проверка рисков
        buy_order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )
        risk_assessment = await manage_risk.assess_order_risk(buy_order)
        assert risk_assessment["is_allowed"] is True
        # Act - Шаг 3: Создание ордера покупки
        buy_result = await manage_orders.create_order(buy_order)
        assert buy_result["id"] == "e2e_order_123"
        assert buy_result["status"] == "pending"
        # Act - Шаг 4: Мониторинг исполнения ордера
        order_status = await manage_orders.get_order_status(buy_result["id"])
        assert order_status["status"] in ["pending", "partially_filled", "filled"]
        # Act - Шаг 5: Создание позиции после исполнения
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("50000"), Currency.USD),
        )
        position_result = await manage_positions.create_position(position)
        assert position_result is True
        # Act - Шаг 6: Мониторинг позиции и принятие решения о продаже
        updated_market_data = {**sample_market_data, "price": 51000.0}  # Рост цены
        sell_analysis = await self._analyze_market(updated_market_data)
        assert sell_analysis["should_trade"] is True
        assert sell_analysis["action"] == "sell"
        # Act - Шаг 7: Создание ордера продажи
        sell_order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("51000")),
        )
        sell_result = await manage_orders.create_order(sell_order)
        assert sell_result["id"] == "e2e_order_123"
        # Act - Шаг 8: Закрытие позиции
        close_result = await manage_positions.close_position("BTCUSDT", Decimal("0.001"), Decimal("51000"))
        assert close_result is True
        # Act - Шаг 9: Расчет PnL
        pnl = await self._calculate_session_pnl(50000, 51000, Decimal("0.001"))
        assert pnl == 10.0  # (51000-50000)*0.001

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_trading_session_with_stop_loss(self, mock_di_container, sample_market_data) -> None:
        """Тест торговой сессии со стоп-лоссом."""
        # Arrange
        with patch("application.di_container.DIContainer", return_value=mock_di_container):
            manage_orders = OrderManagementUseCase(
                exchange=mock_di_container.get(BybitClient), order_repository=mock_di_container.get("order_repository")
            )
            manage_positions = PositionManagementUseCase(
                position_repository=mock_di_container.get("position_repository")
            )
        # Act - Шаг 1: Создание позиции
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("50000"), Currency.USD),
        )
        await manage_positions.create_position(position)
        # Act - Шаг 2: Падение цены и срабатывание стоп-лосса
        falling_market_data = {**sample_market_data, "price": 47500.0}  # Падение на 5%
        # Проверка стоп-лосса
        stop_loss_triggered = await self._check_stop_loss(position, Decimal("47500"), Decimal("0.05"))  # 5% стоп-лосс
        assert stop_loss_triggered is True
        # Act - Шаг 3: Автоматическое закрытие позиции
        stop_loss_order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Volume(Decimal("0.001")),
        )
        stop_result = await manage_orders.create_order(stop_loss_order)
        assert stop_result["id"] == "e2e_order_123"
        # Act - Шаг 4: Расчет убытка
        loss = await self._calculate_session_pnl(50000, 47500, Decimal("0.001"))
        assert loss == -25.0  # (47500-50000)*0.001

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_trading_session_with_take_profit(self, mock_di_container, sample_market_data) -> None:
        """Тест торговой сессии с тейк-профитом."""
        # Arrange
        with patch("application.di_container.DIContainer", return_value=mock_di_container):
            manage_orders = OrderManagementUseCase(
                exchange=mock_di_container.get(BybitClient), order_repository=mock_di_container.get("order_repository")
            )
            manage_positions = PositionManagementUseCase(
                position_repository=mock_di_container.get("position_repository")
            )
        # Act - Шаг 1: Создание позиции
        position = Position(
            trading_pair="BTCUSDT",
            side="long",
            quantity=Volume(Decimal("0.001")),
            average_price=Money(Decimal("50000"), Currency.USD),
            current_price=Money(Decimal("50000"), Currency.USD),
        )
        await manage_positions.create_position(position)
        # Act - Шаг 2: Рост цены и срабатывание тейк-профита
        rising_market_data = {**sample_market_data, "price": 55000.0}  # Рост на 10%
        # Проверка тейк-профита
        take_profit_triggered = await self._check_take_profit(
            position, Decimal("55000"), Decimal("0.10")  # 10% тейк-профит
        )
        assert take_profit_triggered is True
        # Act - Шаг 3: Автоматическое закрытие позиции
        take_profit_order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("55000")),
        )
        tp_result = await manage_orders.create_order(take_profit_order)
        assert tp_result["id"] == "e2e_order_123"
        # Act - Шаг 4: Расчет прибыли
        profit = await self._calculate_session_pnl(50000, 55000, Decimal("0.001"))
        assert profit == 50.0  # (55000-50000)*0.001

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_trading_session_error_handling(self, mock_di_container, sample_market_data) -> None:
        """Тест обработки ошибок в торговой сессии."""
        # Arrange - Настройка ошибки биржи
        mock_exchange = mock_di_container.get(BybitClient)
        mock_exchange.create_order.side_effect = Exception("Network error")
        with patch("application.di_container.DIContainer", return_value=mock_di_container):
            manage_orders = OrderManagementUseCase(
                exchange=mock_exchange, order_repository=mock_di_container.get("order_repository")
            )
        # Act & Assert - Проверка обработки ошибки
        order = Order(
            trading_pair="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Volume(Decimal("0.001")),
            price=Price(Decimal("50000")),
        )
        with pytest.raises(Exception, match="Network error"):
            await manage_orders.create_order(order)
        # Act - Проверка восстановления после ошибки
        mock_exchange.create_order.side_effect = None  # Сброс ошибки
        mock_exchange.create_order.return_value = {"id": "recovery_order", "status": "pending"}
        recovery_result = await manage_orders.create_order(order)
        assert recovery_result["id"] == "recovery_order"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_trading_session_performance_metrics(self, mock_di_container, sample_market_data) -> None:
        """Тест метрик производительности торговой сессии."""
        # Arrange
        with patch("application.di_container.DIContainer", return_value=mock_di_container):
            manage_orders = OrderManagementUseCase(
                exchange=mock_di_container.get(BybitClient), order_repository=mock_di_container.get("order_repository")
            )
        # Act - Выполнение нескольких сделок
        trades = []
        for i in range(5):
            order = Order(
                trading_pair="BTCUSDT",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=Volume(Decimal("0.001")),
                price=Price(Decimal("50000")),
            )
            result = await manage_orders.create_order(order)
            trades.append(result)
        # Act - Расчет метрик
        metrics = await self._calculate_performance_metrics(trades)
        # Assert - Проверка метрик
        assert metrics["total_trades"] == 5
        assert metrics["success_rate"] == 1.0  # Все ордера успешны в моке
        assert "average_execution_time" in metrics
        assert "total_volume" in metrics

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_trading_session_concurrent_orders(self, mock_di_container, sample_market_data) -> None:
        """Тест конкурентных ордеров в торговой сессии."""
        # Arrange
        with patch("application.di_container.DIContainer", return_value=mock_di_container):
            manage_orders = OrderManagementUseCase(
                exchange=mock_di_container.get(BybitClient), order_repository=mock_di_container.get("order_repository")
            )
        # Act - Создание конкурентных ордеров
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
            Order(
                trading_pair="ADAUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Volume(Decimal("100")),
            ),
        ]
        # Выполнение ордеров конкурентно
        tasks = [manage_orders.create_order(order) for order in orders]
        results = await asyncio.gather(*tasks)
        # Assert - Проверка результатов
        assert len(results) == 3
        assert all(result["id"] == "e2e_order_123" for result in results)

    # Вспомогательные методы
    async def _analyze_market(self, market_data: dict) -> dict:
        """Анализ рыночных данных."""
        # Упрощенная логика анализа
        price = market_data["price"]
        if price > 50000:
            return {"should_trade": True, "action": "sell"}
        elif price < 50000:
            return {"should_trade": True, "action": "buy"}
        else:
            return {"should_trade": False, "action": "hold"}

    async def _check_stop_loss(self, position: Position, current_price: Decimal, stop_loss_pct: Decimal) -> bool:
        """Проверка срабатывания стоп-лосса."""
        entry_price = position.entry_price.amount
        stop_loss_price = entry_price * (1 - stop_loss_pct)
        return current_price <= stop_loss_price

    async def _check_take_profit(self, position: Position, current_price: Decimal, take_profit_pct: Decimal) -> bool:
        """Проверка срабатывания тейк-профита."""
        entry_price = position.entry_price.amount
        take_profit_price = entry_price * (1 + take_profit_pct)
        return current_price >= take_profit_price

    async def _calculate_session_pnl(self, entry_price: int, exit_price: int, quantity: Decimal) -> float:
        """Расчет PnL сессии."""
        return float((exit_price - entry_price) * quantity)

    async def _calculate_performance_metrics(self, trades: list) -> dict:
        """Расчет метрик производительности."""
        return {
            "total_trades": len(trades),
            "success_rate": 1.0,
            "average_execution_time": 0.1,
            "total_volume": len(trades) * 0.001,
        }
