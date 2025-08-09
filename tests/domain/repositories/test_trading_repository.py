"""
Unit тесты для domain/repositories/trading_repository.py.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

from domain.repositories.trading_repository import TradingRepository
from domain.entities.trade import Trade
from domain.entities.order import Order, OrderType, OrderSide
from domain.exceptions.base_exceptions import EntityNotFoundError, ValidationError


class TestTradingRepository:
    """Тесты для TradingRepository."""

    @pytest.fixture
    def repository(self):
        """Создание репозитория."""
        return TradingRepository()

    @pytest.fixture
    def sample_trade_data(self) -> Dict[str, Any]:
        """Тестовые данные сделки."""
        return {
            "id": "trade_001",
            "symbol": "BTCUSD",
            "side": "BUY",
            "quantity": Decimal("1.0"),
            "price": Decimal("50000.00"),
            "timestamp": datetime.now(timezone.utc),
        }

    @pytest.fixture
    def sample_order_data(self) -> Dict[str, Any]:
        """Тестовые данные заказа."""
        return {
            "id": "order_001",
            "symbol": "BTCUSD",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("1.0"),
            "price": Decimal("50000.00"),
            "status": "PENDING",
            "timestamp": datetime.now(timezone.utc),
        }

    @pytest.fixture
    def sample_trades(self) -> List[Trade]:
        """Тестовые сделки."""
        return [
            Trade(
                id="trade_001",
                symbol="BTCUSD",
                side="BUY",
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                timestamp=datetime.now(timezone.utc),
            ),
            Trade(
                id="trade_002",
                symbol="BTCUSD",
                side="SELL",
                quantity=Decimal("0.5"),
                price=Decimal("52000.00"),
                timestamp=datetime.now(timezone.utc),
            ),
            Trade(
                id="trade_003",
                symbol="ETHUSD",
                side="BUY",
                quantity=Decimal("10.0"),
                price=Decimal("3500.00"),
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    @pytest.fixture
    def sample_orders(self) -> List[Order]:
        """Тестовые заказы."""
        return [
            Order(
                id="order_001",
                symbol="BTCUSD",
                side=OrderSide.BUY,
                type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                status="PENDING",
                timestamp=datetime.now(timezone.utc),
            ),
            Order(
                id="order_002",
                symbol="BTCUSD",
                side=OrderSide.SELL,
                type=OrderType.MARKET,
                quantity=Decimal("0.5"),
                price=None,
                status="FILLED",
                timestamp=datetime.now(timezone.utc),
            ),
            Order(
                id="order_003",
                symbol="ETHUSD",
                side=OrderSide.BUY,
                type=OrderType.STOP_LOSS,
                quantity=Decimal("10.0"),
                price=Decimal("3500.00"),
                status="CANCELLED",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

    def test_add_trade(self, repository, sample_trades):
        """Тест добавления сделки."""
        trade = sample_trades[0]

        result = repository.add_trade(trade)

        assert result == trade
        assert repository.get_trade_by_id("trade_001") == trade
        assert len(repository.get_all_trades()) == 1

    def test_get_trade_by_id_existing(self, repository, sample_trades):
        """Тест получения существующей сделки по ID."""
        trade = sample_trades[0]
        repository.add_trade(trade)

        result = repository.get_trade_by_id("trade_001")

        assert result == trade
        assert result.id == "trade_001"
        assert result.symbol == "BTCUSD"
        assert result.side == "BUY"

    def test_get_trade_by_id_not_found(self, repository):
        """Тест получения несуществующей сделки по ID."""
        with pytest.raises(EntityNotFoundError, match="Trade with id trade_999 not found"):
            repository.get_trade_by_id("trade_999")

    def test_get_trades_by_symbol(self, repository, sample_trades):
        """Тест получения сделок по символу."""
        for trade in sample_trades:
            repository.add_trade(trade)

        btc_trades = repository.get_trades_by_symbol("BTCUSD")
        eth_trades = repository.get_trades_by_symbol("ETHUSD")

        assert len(btc_trades) == 2
        assert len(eth_trades) == 1
        assert all(trade.symbol == "BTCUSD" for trade in btc_trades)
        assert all(trade.symbol == "ETHUSD" for trade in eth_trades)

    def test_get_trades_by_side(self, repository, sample_trades):
        """Тест получения сделок по стороне."""
        for trade in sample_trades:
            repository.add_trade(trade)

        buy_trades = repository.get_trades_by_side("BUY")
        sell_trades = repository.get_trades_by_side("SELL")

        assert len(buy_trades) == 2
        assert len(sell_trades) == 1
        assert all(trade.side == "BUY" for trade in buy_trades)
        assert all(trade.side == "SELL" for trade in sell_trades)

    def test_get_trades_by_price_range(self, repository, sample_trades):
        """Тест получения сделок по диапазону цен."""
        for trade in sample_trades:
            repository.add_trade(trade)

        # Сделки с ценой от 40000 до 55000
        trades_in_range = repository.get_trades_by_price_range(
            min_price=Decimal("40000.00"), max_price=Decimal("55000.00")
        )

        assert len(trades_in_range) == 2
        assert all(Decimal("40000.00") <= trade.price <= Decimal("55000.00") for trade in trades_in_range)

    def test_get_trades_by_quantity_range(self, repository, sample_trades):
        """Тест получения сделок по диапазону количества."""
        for trade in sample_trades:
            repository.add_trade(trade)

        # Сделки с количеством от 0.1 до 5.0
        trades_in_range = repository.get_trades_by_quantity_range(
            min_quantity=Decimal("0.1"), max_quantity=Decimal("5.0")
        )

        assert len(trades_in_range) == 2
        assert all(Decimal("0.1") <= trade.quantity <= Decimal("5.0") for trade in trades_in_range)

    def test_get_trades_by_timestamp_range(self, repository, sample_trades):
        """Тест получения сделок по диапазону времени."""
        for trade in sample_trades:
            repository.add_trade(trade)

        # Получаем сделки за последний час
        end_time = datetime.now(timezone.utc)
        start_time = end_time.replace(hour=end_time.hour - 1)

        trades_in_range = repository.get_trades_by_timestamp_range(start_time, end_time)

        assert len(trades_in_range) == 3  # Все сделки созданы в последний час

    def test_get_trades_by_date(self, repository, sample_trades):
        """Тест получения сделок по дате."""
        for trade in sample_trades:
            repository.add_trade(trade)

        today = datetime.now(timezone.utc).date()
        trades_today = repository.get_trades_by_date(today)

        assert len(trades_today) == 3  # Все сделки созданы сегодня

    def test_get_trade_statistics(self, repository, sample_trades):
        """Тест получения статистики сделок."""
        for trade in sample_trades:
            repository.add_trade(trade)

        stats = repository.get_trade_statistics()

        assert isinstance(stats, dict)
        assert "total_trades" in stats
        assert "total_volume" in stats
        assert "average_price" in stats
        assert "total_value" in stats

        assert stats["total_trades"] == 3
        assert stats["total_volume"] == Decimal("11.5")  # 1.0 + 0.5 + 10.0

    def test_get_trades_by_symbol_and_side(self, repository, sample_trades):
        """Тест получения сделок по символу и стороне."""
        for trade in sample_trades:
            repository.add_trade(trade)

        btc_buy_trades = repository.get_trades_by_symbol_and_side("BTCUSD", "BUY")
        btc_sell_trades = repository.get_trades_by_symbol_and_side("BTCUSD", "SELL")

        assert len(btc_buy_trades) == 1
        assert len(btc_sell_trades) == 1
        assert btc_buy_trades[0].side == "BUY"
        assert btc_sell_trades[0].side == "SELL"

    def test_add_order(self, repository, sample_orders):
        """Тест добавления заказа."""
        order = sample_orders[0]

        result = repository.add_order(order)

        assert result == order
        assert repository.get_order_by_id("order_001") == order
        assert len(repository.get_all_orders()) == 1

    def test_get_order_by_id_existing(self, repository, sample_orders):
        """Тест получения существующего заказа по ID."""
        order = sample_orders[0]
        repository.add_order(order)

        result = repository.get_order_by_id("order_001")

        assert result == order
        assert result.id == "order_001"
        assert result.symbol == "BTCUSD"
        assert result.side == OrderSide.BUY

    def test_get_order_by_id_not_found(self, repository):
        """Тест получения несуществующего заказа по ID."""
        with pytest.raises(EntityNotFoundError, match="Order with id order_999 not found"):
            repository.get_order_by_id("order_999")

    def test_get_orders_by_symbol(self, repository, sample_orders):
        """Тест получения заказов по символу."""
        for order in sample_orders:
            repository.add_order(order)

        btc_orders = repository.get_orders_by_symbol("BTCUSD")
        eth_orders = repository.get_orders_by_symbol("ETHUSD")

        assert len(btc_orders) == 2
        assert len(eth_orders) == 1
        assert all(order.symbol == "BTCUSD" for order in btc_orders)
        assert all(order.symbol == "ETHUSD" for order in eth_orders)

    def test_get_orders_by_status(self, repository, sample_orders):
        """Тест получения заказов по статусу."""
        for order in sample_orders:
            repository.add_order(order)

        pending_orders = repository.get_orders_by_status("PENDING")
        filled_orders = repository.get_orders_by_status("FILLED")
        cancelled_orders = repository.get_orders_by_status("CANCELLED")

        assert len(pending_orders) == 1
        assert len(filled_orders) == 1
        assert len(cancelled_orders) == 1
        assert all(order.status == "PENDING" for order in pending_orders)
        assert all(order.status == "FILLED" for order in filled_orders)
        assert all(order.status == "CANCELLED" for order in cancelled_orders)

    def test_get_orders_by_type(self, repository, sample_orders):
        """Тест получения заказов по типу."""
        for order in sample_orders:
            repository.add_order(order)

        limit_orders = repository.get_orders_by_type(OrderType.LIMIT)
        market_orders = repository.get_orders_by_type(OrderType.MARKET)
        stop_orders = repository.get_orders_by_type(OrderType.STOP_LOSS)

        assert len(limit_orders) == 1
        assert len(market_orders) == 1
        assert len(stop_orders) == 1
        assert all(order.type == OrderType.LIMIT for order in limit_orders)
        assert all(order.type == OrderType.MARKET for order in market_orders)
        assert all(order.type == OrderType.STOP_LOSS for order in stop_orders)

    def test_get_orders_by_side(self, repository, sample_orders):
        """Тест получения заказов по стороне."""
        for order in sample_orders:
            repository.add_order(order)

        buy_orders = repository.get_orders_by_side(OrderSide.BUY)
        sell_orders = repository.get_orders_by_side(OrderSide.SELL)

        assert len(buy_orders) == 2
        assert len(sell_orders) == 1
        assert all(order.side == OrderSide.BUY for order in buy_orders)
        assert all(order.side == OrderSide.SELL for order in sell_orders)

    def test_update_order_status(self, repository, sample_orders):
        """Тест обновления статуса заказа."""
        order = sample_orders[0]
        repository.add_order(order)

        repository.update_order_status("order_001", "FILLED")

        updated_order = repository.get_order_by_id("order_001")
        assert updated_order.status == "FILLED"

    def test_update_order_status_not_found(self, repository):
        """Тест обновления статуса несуществующего заказа."""
        with pytest.raises(EntityNotFoundError, match="Order with id order_999 not found"):
            repository.update_order_status("order_999", "FILLED")

    def test_get_pending_orders(self, repository, sample_orders):
        """Тест получения ожидающих заказов."""
        for order in sample_orders:
            repository.add_order(order)

        pending_orders = repository.get_pending_orders()

        assert len(pending_orders) == 1
        assert all(order.status == "PENDING" for order in pending_orders)

    def test_get_filled_orders(self, repository, sample_orders):
        """Тест получения исполненных заказов."""
        for order in sample_orders:
            repository.add_order(order)

        filled_orders = repository.get_filled_orders()

        assert len(filled_orders) == 1
        assert all(order.status == "FILLED" for order in filled_orders)

    def test_get_cancelled_orders(self, repository, sample_orders):
        """Тест получения отмененных заказов."""
        for order in sample_orders:
            repository.add_order(order)

        cancelled_orders = repository.get_cancelled_orders()

        assert len(cancelled_orders) == 1
        assert all(order.status == "CANCELLED" for order in cancelled_orders)

    def test_get_orders_by_price_range(self, repository, sample_orders):
        """Тест получения заказов по диапазону цен."""
        for order in sample_orders:
            repository.add_order(order)

        # Заказы с ценой от 40000 до 55000
        orders_in_range = repository.get_orders_by_price_range(
            min_price=Decimal("40000.00"), max_price=Decimal("55000.00")
        )

        assert len(orders_in_range) == 2
        assert all(
            Decimal("40000.00") <= order.price <= Decimal("55000.00")
            for order in orders_in_range
            if order.price is not None
        )

    def test_get_orders_by_quantity_range(self, repository, sample_orders):
        """Тест получения заказов по диапазону количества."""
        for order in sample_orders:
            repository.add_order(order)

        # Заказы с количеством от 0.1 до 5.0
        orders_in_range = repository.get_orders_by_quantity_range(
            min_quantity=Decimal("0.1"), max_quantity=Decimal("5.0")
        )

        assert len(orders_in_range) == 2
        assert all(Decimal("0.1") <= order.quantity <= Decimal("5.0") for order in orders_in_range)

    def test_get_orders_by_timestamp_range(self, repository, sample_orders):
        """Тест получения заказов по диапазону времени."""
        for order in sample_orders:
            repository.add_order(order)

        # Получаем заказы за последний час
        end_time = datetime.now(timezone.utc)
        start_time = end_time.replace(hour=end_time.hour - 1)

        orders_in_range = repository.get_orders_by_timestamp_range(start_time, end_time)

        assert len(orders_in_range) == 3  # Все заказы созданы в последний час

    def test_get_order_statistics(self, repository, sample_orders):
        """Тест получения статистики заказов."""
        for order in sample_orders:
            repository.add_order(order)

        stats = repository.get_order_statistics()

        assert isinstance(stats, dict)
        assert "total_orders" in stats
        assert "pending_orders" in stats
        assert "filled_orders" in stats
        assert "cancelled_orders" in stats

        assert stats["total_orders"] == 3
        assert stats["pending_orders"] == 1
        assert stats["filled_orders"] == 1
        assert stats["cancelled_orders"] == 1

    def test_get_trades_for_order(self, repository, sample_trades, sample_orders):
        """Тест получения сделок для заказа."""
        for trade in sample_trades:
            repository.add_trade(trade)
        for order in sample_orders:
            repository.add_order(order)

        # Связываем сделки с заказом
        repository.link_trade_to_order("trade_001", "order_001")
        repository.link_trade_to_order("trade_002", "order_002")

        trades_for_order = repository.get_trades_for_order("order_001")

        assert len(trades_for_order) == 1
        assert trades_for_order[0].id == "trade_001"

    def test_get_orders_for_trade(self, repository, sample_trades, sample_orders):
        """Тест получения заказов для сделки."""
        for trade in sample_trades:
            repository.add_trade(trade)
        for order in sample_orders:
            repository.add_order(order)

        # Связываем сделки с заказом
        repository.link_trade_to_order("trade_001", "order_001")

        orders_for_trade = repository.get_orders_for_trade("trade_001")

        assert len(orders_for_trade) == 1
        assert orders_for_trade[0].id == "order_001"

    def test_get_trading_summary(self, repository, sample_trades, sample_orders):
        """Тест получения сводки торговли."""
        for trade in sample_trades:
            repository.add_trade(trade)
        for order in sample_orders:
            repository.add_order(order)

        summary = repository.get_trading_summary()

        assert isinstance(summary, dict)
        assert "total_trades" in summary
        assert "total_orders" in summary
        assert "total_volume" in summary
        assert "total_value" in summary
        assert "buy_trades" in summary
        assert "sell_trades" in summary

        assert summary["total_trades"] == 3
        assert summary["total_orders"] == 3
        assert summary["buy_trades"] == 2
        assert summary["sell_trades"] == 1

    def test_get_trading_performance(self, repository, sample_trades):
        """Тест получения производительности торговли."""
        for trade in sample_trades:
            repository.add_trade(trade)

        performance = repository.get_trading_performance()

        assert isinstance(performance, dict)
        assert "total_pnl" in performance
        assert "win_rate" in performance
        assert "average_trade_size" in performance
        assert "total_trades" in performance

    def test_exists_trade(self, repository, sample_trades):
        """Тест проверки существования сделки."""
        trade = sample_trades[0]
        repository.add_trade(trade)

        assert repository.exists_trade("trade_001") is True
        assert repository.exists_trade("trade_999") is False

    def test_exists_order(self, repository, sample_orders):
        """Тест проверки существования заказа."""
        order = sample_orders[0]
        repository.add_order(order)

        assert repository.exists_order("order_001") is True
        assert repository.exists_order("order_999") is False

    def test_delete_trade(self, repository, sample_trades):
        """Тест удаления сделки."""
        trade = sample_trades[0]
        repository.add_trade(trade)

        result = repository.delete_trade("trade_001")

        assert result == trade
        with pytest.raises(EntityNotFoundError):
            repository.get_trade_by_id("trade_001")
        assert len(repository.get_all_trades()) == 0

    def test_delete_order(self, repository, sample_orders):
        """Тест удаления заказа."""
        order = sample_orders[0]
        repository.add_order(order)

        result = repository.delete_order("order_001")

        assert result == order
        with pytest.raises(EntityNotFoundError):
            repository.get_order_by_id("order_001")
        assert len(repository.get_all_orders()) == 0
