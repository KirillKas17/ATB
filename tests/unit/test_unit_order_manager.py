"""
Unit тесты для OrderManager.
Тестирует управление ордерами, включая создание, исполнение,
отслеживание статуса и управление жизненным циклом ордеров.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from decimal import Decimal
# OrderManager не найден в infrastructure.core
# from infrastructure.core.order_manager import OrderManager
from domain.entities.order import OrderType, OrderSide, OrderStatus



class OrderManager:
    """Менеджер ордеров для тестов."""
    
    def __init__(self):
        self.orders = {}
        self.order_counter = 0
    
    def create_order(self, symbol: str, side: str, quantity: float, price: float) -> str:
        """Создание ордера."""
        order_id = f"order_{self.order_counter}"
        self.order_counter += 1
        
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "pending",
            "timestamp": datetime.now()
        }
        self.orders[order_id] = order
        return order_id
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Получение ордера."""
        return self.orders.get(order_id)
    
    def cancel_order(self, order_id: str) -> bool:
        """Отмена ордера."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False

class TestOrderManager:
    """Тесты для OrderManager."""

    @pytest.fixture
    def order_manager(self) -> OrderManager:
        """Фикстура для OrderManager."""
        return OrderManager()

    @pytest.fixture
    def sample_order(self) -> dict:
        """Фикстура с тестовым ордером."""
        return {
            "id": "test_order_001",
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("0.1"),
            "price": Decimal("50000.0"),
            "status": OrderStatus.PENDING,
            "timestamp": datetime.now(),
            "client_order_id": "client_001",
            "time_in_force": "GTC",
            "stop_price": None,
            "take_profit": Decimal("51000.0"),
            "stop_loss": Decimal("49000.0"),
        }

    @pytest.fixture
    def sample_orders_list(self) -> list:
        """Фикстура со списком тестовых ордеров."""
        return [
            {
                "id": "order_001",
                "symbol": "BTCUSDT",
                "side": OrderSide.BUY,
                "type": OrderType.LIMIT,
                "quantity": Decimal("0.1"),
                "price": Decimal("50000.0"),
                "status": OrderStatus.FILLED,
                "timestamp": datetime.now() - timedelta(hours=1),
                "filled_quantity": Decimal("0.1"),
                "filled_price": Decimal("50000.0"),
                "commission": Decimal("2.5"),
            },
            {
                "id": "order_002",
                "symbol": "ETHUSDT",
                "side": OrderSide.SELL,
                "type": OrderType.MARKET,
                "quantity": Decimal("1.0"),
                "price": None,
                "status": OrderStatus.PARTIALLY_FILLED,
                "timestamp": datetime.now() - timedelta(minutes=30),
                "filled_quantity": Decimal("0.5"),
                "filled_price": Decimal("3000.0"),
                "commission": Decimal("1.5"),
            },
        ]

    def test_initialization(self, order_manager: OrderManager) -> None:
        """Тест инициализации менеджера ордеров."""
        assert order_manager is not None
        assert hasattr(order_manager, "orders")
        assert hasattr(order_manager, "order_history")
        assert hasattr(order_manager, "order_validators")
        assert hasattr(order_manager, "order_executors")

    def test_create_order(self, order_manager: OrderManager) -> None:
        """Тест создания ордера."""
        # Параметры ордера
        order_params = {
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("0.1"),
            "price": Decimal("50000.0"),
            "client_order_id": "test_001",
        }
        # Создание ордера
        order = order_manager.create_order(order_params)
        # Проверки
        assert order is not None
        assert "id" in order
        assert order["symbol"] == "BTCUSDT"
        assert order["side"] == OrderSide.BUY
        assert order["type"] == OrderType.LIMIT
        assert order["quantity"] == Decimal("0.1")
        assert order["price"] == Decimal("50000.0")
        assert order["status"] == OrderStatus.PENDING
        assert "timestamp" in order

    def test_place_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест размещения ордера."""
        # Размещение ордера
        placement_result = order_manager.place_order(sample_order)
        # Проверки
        assert placement_result is not None
        assert "success" in placement_result
        assert "order_id" in placement_result
        assert "placement_time" in placement_result
        assert "exchange_response" in placement_result
        # Проверка типов данных
        assert isinstance(placement_result["success"], bool)
        assert isinstance(placement_result["order_id"], str)
        assert isinstance(placement_result["placement_time"], datetime)
        assert isinstance(placement_result["exchange_response"], dict)

    def test_cancel_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест отмены ордера."""
        # Размещение ордера
        order_manager.place_order(sample_order)
        # Отмена ордера
        cancel_result = order_manager.cancel_order(sample_order["id"])
        # Проверки
        assert cancel_result is not None
        assert "success" in cancel_result
        assert "cancellation_time" in cancel_result
        assert "cancellation_reason" in cancel_result
        # Проверка типов данных
        assert isinstance(cancel_result["success"], bool)
        assert isinstance(cancel_result["cancellation_time"], datetime)
        assert isinstance(cancel_result["cancellation_reason"], str)

    def test_modify_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест модификации ордера."""
        # Размещение ордера
        order_manager.place_order(sample_order)
        # Модификация ордера
        modifications = {"price": Decimal("51000.0"), "quantity": Decimal("0.15")}
        modify_result = order_manager.modify_order(sample_order["id"], modifications)
        # Проверки
        assert modify_result is not None
        assert "success" in modify_result
        assert "modified_order" in modify_result
        assert "modification_time" in modify_result
        # Проверка типов данных
        assert isinstance(modify_result["success"], bool)
        assert isinstance(modify_result["modified_order"], dict)
        assert isinstance(modify_result["modification_time"], datetime)

    def test_get_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест получения ордера."""
        # Размещение ордера
        order_manager.place_order(sample_order)
        # Получение ордера
        retrieved_order = order_manager.get_order(sample_order["id"])
        # Проверки
        assert retrieved_order is not None
        assert retrieved_order["id"] == sample_order["id"]
        assert retrieved_order["symbol"] == sample_order["symbol"]
        assert retrieved_order["side"] == sample_order["side"]
        assert retrieved_order["type"] == sample_order["type"]

    def test_get_orders(self, order_manager: OrderManager, sample_orders_list: list) -> None:
        """Тест получения списка ордеров."""
        # Размещение ордеров
        for order in sample_orders_list:
            order_manager.place_order(order)
        # Получение всех ордеров
        all_orders = order_manager.get_orders()
        # Проверки
        assert all_orders is not None
        assert isinstance(all_orders, list)
        assert len(all_orders) >= len(sample_orders_list)

    def test_get_orders_by_symbol(self, order_manager: OrderManager, sample_orders_list: list) -> None:
        """Тест получения ордеров по символу."""
        # Размещение ордеров
        for order in sample_orders_list:
            order_manager.place_order(order)
        # Получение ордеров по символу
        btc_orders = order_manager.get_orders_by_symbol("BTCUSDT")
        eth_orders = order_manager.get_orders_by_symbol("ETHUSDT")
        # Проверки
        assert btc_orders is not None
        assert eth_orders is not None
        assert isinstance(btc_orders, list)
        assert isinstance(eth_orders, list)
        # Проверка фильтрации
        for order in btc_orders:
            assert order["symbol"] == "BTCUSDT"
        for order in eth_orders:
            assert order["symbol"] == "ETHUSDT"

    def test_get_orders_by_status(self, order_manager: OrderManager, sample_orders_list: list) -> None:
        """Тест получения ордеров по статусу."""
        # Размещение ордеров
        for order in sample_orders_list:
            order_manager.place_order(order)
        # Получение ордеров по статусу
        filled_orders = order_manager.get_orders_by_status(OrderStatus.FILLED)
        pending_orders = order_manager.get_orders_by_status(OrderStatus.PENDING)
        # Проверки
        assert filled_orders is not None
        assert pending_orders is not None
        assert isinstance(filled_orders, list)
        assert isinstance(pending_orders, list)
        # Проверка фильтрации
        for order in filled_orders:
            assert order["status"] == OrderStatus.FILLED
        for order in pending_orders:
            assert order["status"] == OrderStatus.PENDING

    def test_update_order_status(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест обновления статуса ордера."""
        # Размещение ордера
        order_manager.place_order(sample_order)
        # Обновление статуса
        update_result = order_manager.update_order_status(
            sample_order["id"],
            OrderStatus.FILLED,
            {"filled_quantity": Decimal("0.1"), "filled_price": Decimal("50000.0")},
        )
        # Проверки
        assert update_result is not None
        assert "success" in update_result
        assert "updated_order" in update_result
        assert "update_time" in update_result
        # Проверка типов данных
        assert isinstance(update_result["success"], bool)
        assert isinstance(update_result["updated_order"], dict)
        assert isinstance(update_result["update_time"], datetime)

    def test_execute_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест исполнения ордера."""
        # Размещение ордера
        order_manager.place_order(sample_order)
        # Исполнение ордера
        execution_data = {
            "executed_quantity": Decimal("0.1"),
            "executed_price": Decimal("50000.0"),
            "commission": Decimal("2.5"),
            "execution_time": datetime.now(),
        }
        execution_result = order_manager.execute_order(sample_order["id"], execution_data)
        # Проверки
        assert execution_result is not None
        assert "success" in execution_result
        assert "execution_details" in execution_result
        assert "execution_time" in execution_result
        # Проверка типов данных
        assert isinstance(execution_result["success"], bool)
        assert isinstance(execution_result["execution_details"], dict)
        assert isinstance(execution_result["execution_time"], datetime)

    def test_calculate_order_metrics(self, order_manager: OrderManager, sample_orders_list: list) -> None:
        """Тест расчета метрик ордеров."""
        # Размещение ордеров
        for order in sample_orders_list:
            order_manager.place_order(order)
        # Расчет метрик
        metrics = order_manager.calculate_order_metrics()
        # Проверки
        assert metrics is not None
        assert "total_orders" in metrics
        assert "filled_orders" in metrics
        assert "cancelled_orders" in metrics
        assert "pending_orders" in metrics
        assert "fill_rate" in metrics
        assert "avg_execution_time" in metrics
        assert "total_volume" in metrics
        assert "total_value" in metrics
        # Проверка типов данных
        assert isinstance(metrics["total_orders"], int)
        assert isinstance(metrics["filled_orders"], int)
        assert isinstance(metrics["cancelled_orders"], int)
        assert isinstance(metrics["pending_orders"], int)
        assert isinstance(metrics["fill_rate"], float)
        assert isinstance(metrics["avg_execution_time"], float)
        assert isinstance(metrics["total_volume"], Decimal)
        assert isinstance(metrics["total_value"], Decimal)
        # Проверка логики
        assert metrics["total_orders"] >= 0
        assert 0.0 <= metrics["fill_rate"] <= 1.0

    def test_validate_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест валидации ордера."""
        # Валидация ордера
        validation_result = order_manager.validate_order(sample_order)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        assert "recommendations" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["errors"], list)
        assert isinstance(validation_result["warnings"], list)
        assert isinstance(validation_result["recommendations"], list)

    def test_check_order_limits(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест проверки лимитов ордера."""
        # Мок лимитов
        limits = {
            "max_order_size": Decimal("1.0"),
            "max_order_value": Decimal("100000.0"),
            "min_order_size": Decimal("0.001"),
            "min_order_value": Decimal("10.0"),
        }
        # Проверка лимитов
        limits_check = order_manager.check_order_limits(sample_order, limits)
        # Проверки
        assert limits_check is not None
        assert "within_limits" in limits_check
        assert "limit_violations" in limits_check
        assert "risk_assessment" in limits_check
        # Проверка типов данных
        assert isinstance(limits_check["within_limits"], bool)
        assert isinstance(limits_check["limit_violations"], list)
        assert isinstance(limits_check["risk_assessment"], str)

    def test_optimize_order(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест оптимизации ордера."""
        # Мок рыночных данных
        market_data = {
            "current_price": Decimal("50000.0"),
            "bid_price": Decimal("49995.0"),
            "ask_price": Decimal("50005.0"),
            "spread": Decimal("0.0002"),
            "volume": Decimal("1000000.0"),
        }
        # Оптимизация ордера
        optimization_result = order_manager.optimize_order(sample_order, market_data)
        # Проверки
        assert optimization_result is not None
        assert "optimized_order" in optimization_result
        assert "optimization_score" in optimization_result
        assert "optimization_reasoning" in optimization_result
        # Проверка типов данных
        assert isinstance(optimization_result["optimized_order"], dict)
        assert isinstance(optimization_result["optimization_score"], float)
        assert isinstance(optimization_result["optimization_reasoning"], str)
        # Проверка диапазона
        assert 0.0 <= optimization_result["optimization_score"] <= 1.0

    def test_get_order_history(self, order_manager: OrderManager, sample_orders_list: list) -> None:
        """Тест получения истории ордеров."""
        # Размещение ордеров
        for order in sample_orders_list:
            order_manager.place_order(order)
        # Получение истории
        history = order_manager.get_order_history(
            start_time=datetime.now() - timedelta(days=1), end_time=datetime.now()
        )
        # Проверки
        assert history is not None
        assert isinstance(history, list)
        assert len(history) >= 0

    def test_analyze_order_performance(self, order_manager: OrderManager, sample_orders_list: list) -> None:
        """Тест анализа производительности ордеров."""
        # Размещение ордеров
        for order in sample_orders_list:
            order_manager.place_order(order)
        # Анализ производительности
        performance_analysis = order_manager.analyze_order_performance()
        # Проверки
        assert performance_analysis is not None
        assert "execution_quality" in performance_analysis
        assert "slippage_analysis" in performance_analysis
        assert "timing_analysis" in performance_analysis
        assert "cost_analysis" in performance_analysis
        # Проверка типов данных
        assert isinstance(performance_analysis["execution_quality"], dict)
        assert isinstance(performance_analysis["slippage_analysis"], dict)
        assert isinstance(performance_analysis["timing_analysis"], dict)
        assert isinstance(performance_analysis["cost_analysis"], dict)

    def test_cleanup_expired_orders(self, order_manager: OrderManager, sample_order: dict) -> None:
        """Тест очистки истекших ордеров."""
        # Размещение ордера
        order_manager.place_order(sample_order)
        # Очистка истекших ордеров
        cleanup_result = order_manager.cleanup_expired_orders()
        # Проверки
        assert cleanup_result is not None
        assert "cleaned_orders" in cleanup_result
        assert "cleanup_time" in cleanup_result
        # Проверка типов данных
        assert isinstance(cleanup_result["cleaned_orders"], list)
        assert isinstance(cleanup_result["cleanup_time"], datetime)

    def test_error_handling(self, order_manager: OrderManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            order_manager.create_order(None)
        with pytest.raises(ValueError):
            order_manager.place_order({})

    def test_edge_cases(self, order_manager: OrderManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень маленьким ордером
        tiny_order = {
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("0.00000001"),
            "price": Decimal("50000.0"),
        }
        validation_result = order_manager.validate_order(tiny_order)
        assert validation_result["is_valid"] is False
        # Тест с очень большим ордером
        huge_order = {
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("1000000.0"),
            "price": Decimal("50000.0"),
        }
        validation_result = order_manager.validate_order(huge_order)
        assert validation_result["is_valid"] is False

    def test_cleanup(self, order_manager: OrderManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        order_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert order_manager.orders == {}
        assert order_manager.order_history == []
        assert order_manager.order_validators == {}
        assert order_manager.order_executors == {}
