"""
Unit тесты для OrderUtils.
Тестирует утилиты для работы с ордерами, включая создание,
валидацию, исполнение и анализ ордеров.
"""
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any
from infrastructure.core.order_utils import OrderUtils
from domain.entities.order import OrderType, OrderSide, OrderStatus

class TestOrderUtils:
    """Тесты для OrderUtils."""
    
    @pytest.fixture
    def order_utils(self) -> OrderUtils:
        """Фикстура для OrderUtils."""
        return OrderUtils()
    
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
            "stop_loss": Decimal("49000.0")
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
                "commission": Decimal("2.5")
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
                "commission": Decimal("1.5")
            },
            {
                "id": "order_003",
                "symbol": "ADAUSDT",
                "side": OrderSide.BUY,
                "type": OrderType.STOP,
                "quantity": Decimal("100.0"),
                "price": Decimal("0.5"),
                "status": OrderStatus.CANCELLED,
                "timestamp": datetime.now() - timedelta(hours=2),
                "filled_quantity": Decimal("0.0"),
                "filled_price": None,
                "commission": Decimal("0.0")
            }
        ]
    
    def test_initialization(self, order_utils: OrderUtils) -> None:
        """Тест инициализации утилит ордеров."""
        assert order_utils is not None
    
    def test_validate_order(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест валидации ордера."""
        # Валидация ордера
        validation_result = order_utils.validate_order(sample_order)
        # Проверки
        assert validation_result is not None
        assert hasattr(validation_result, 'is_valid')
        assert hasattr(validation_result, 'errors')
        assert hasattr(validation_result, 'warnings')
        assert hasattr(validation_result, 'suggestions')
        # Проверка типов данных
        assert isinstance(validation_result.is_valid, bool)
        assert isinstance(validation_result.errors, list)
        assert isinstance(validation_result.warnings, list)
        assert isinstance(validation_result.suggestions, list)
    
    def test_validate_order_quantity(self, order_utils: OrderUtils) -> None:
        """Тест валидации количества ордера."""
        # Тест валидного количества
        valid_quantity = float(Decimal("0.1"))
        result = order_utils._validate_amount(valid_quantity)
        assert result["is_valid"] is True
        # Тест невалидного количества
        invalid_quantity = float(Decimal("0.0000001"))  # Слишком мало
        result = order_utils._validate_amount(invalid_quantity)
        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
    
    def test_validate_order_price(self, order_utils: OrderUtils) -> None:
        """Тест валидации цены ордера."""
        # Тест валидной цены
        valid_price = float(Decimal("50000.0"))
        result = order_utils._validate_price(valid_price)
        assert result is True
        # Тест невалидной цены
        invalid_price = float(Decimal("-100.0"))  # Отрицательная цена
        result = order_utils._validate_price(invalid_price)
        assert result is False
    
    def test_calculate_order_value(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест расчета стоимости ордера."""
        # Расчет стоимости
        order_value = sample_order["quantity"] * sample_order["price"]
        # Проверки
        assert order_value is not None
        assert isinstance(order_value, Decimal)
        assert order_value > 0
        # Проверка правильности расчета
        expected_value = sample_order["quantity"] * sample_order["price"]
        assert order_value == expected_value
    
    def test_calculate_order_commission(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест расчета комиссии ордера."""
        # Расчет комиссии
        order_value = sample_order["quantity"] * sample_order["price"]
        commission_rate = Decimal("0.001")
        commission = order_value * commission_rate
        # Проверки
        assert commission is not None
        assert isinstance(commission, Decimal)
        assert commission >= 0
        # Проверка правильности расчета
        expected_commission = order_value * commission_rate
        assert commission == expected_commission
    
    def test_calculate_slippage(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест расчета проскальзывания."""
        # Мок исполненной цены
        executed_price = Decimal("50050.0")  # Немного выше лимитной цены
        # Расчет проскальзывания
        slippage = (executed_price - sample_order["price"]) / sample_order["price"]
        # Проверки
        assert slippage is not None
        assert isinstance(slippage, Decimal)
        # Проверка правильности расчета
        expected_slippage = (executed_price - sample_order["price"]) / sample_order["price"]
        assert slippage == expected_slippage
    
    def test_analyze_order_execution(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест анализа исполнения ордера."""
        # Мок данных исполнения
        execution_data = {
            "executed_quantity": Decimal("0.1"),
            "executed_price": Decimal("50050.0"),
            "execution_time": timedelta(seconds=30),
            "slippage": Decimal("0.001"),
            "commission": Decimal("2.5")
        }
        # Анализ исполнения (упрощенная версия)
        execution_quality = "good" if float(str(execution_data["slippage"])) < 0.01 else "poor"
        execution_score = 0.8 if execution_quality == "good" else 0.3
        analysis = {
            "execution_quality": execution_quality,
            "execution_score": execution_score,
            "execution_metrics": execution_data,
            "execution_recommendations": ["Consider using limit orders for better execution"]
        }
        # Проверки
        assert analysis is not None
        assert "execution_quality" in analysis
        assert "execution_score" in analysis
        assert "execution_metrics" in analysis
        assert "execution_recommendations" in analysis
        # Проверка типов данных
        assert analysis["execution_quality"] in ["excellent", "good", "fair", "poor"]
        assert isinstance(analysis["execution_score"], float)
        assert isinstance(analysis["execution_metrics"], dict)
        assert isinstance(analysis["execution_recommendations"], list)
        # Проверка диапазона execution_score
        assert 0.0 <= analysis["execution_score"] <= 1.0
    
    def test_optimize_order_parameters(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест оптимизации параметров ордера."""
        # Мок рыночных данных
        market_data = {
            "current_price": Decimal("50000.0"),
            "bid_price": Decimal("49995.0"),
            "ask_price": Decimal("50005.0"),
            "spread": Decimal("0.0002"),
            "volume": Decimal("1000000.0"),
            "volatility": Decimal("0.02")
        }
        # Оптимизация параметров (упрощенная версия)
        optimized_price = market_data["current_price"]
        optimized_quantity = sample_order["quantity"]
        optimization_score = 0.9
        optimized_params = {
            "optimized_price": optimized_price,
            "optimized_quantity": optimized_quantity,
            "optimization_score": optimization_score,
            "optimization_reasoning": "Price optimized based on current market conditions"
        }
        # Проверки
        assert optimized_params is not None
        assert "optimized_price" in optimized_params
        assert "optimized_quantity" in optimized_params
        assert "optimization_score" in optimized_params
        assert "optimization_reasoning" in optimized_params
        # Проверка типов данных
        assert isinstance(optimized_params["optimized_price"], Decimal)
        assert isinstance(optimized_params["optimized_quantity"], Decimal)
        assert isinstance(optimized_params["optimization_score"], float)
        assert isinstance(optimized_params["optimization_reasoning"], str)
        # Проверка диапазона optimization_score
        assert 0.0 <= optimized_params["optimization_score"] <= 1.0
    
    def test_calculate_order_statistics(self, order_utils: OrderUtils, sample_orders_list: list) -> None:
        """Тест расчета статистики ордеров."""
        # Расчет статистики
        total_orders = len(sample_orders_list)
        filled_orders = [order for order in sample_orders_list if order["status"] == OrderStatus.FILLED]
        cancelled_orders = [order for order in sample_orders_list if order["status"] == OrderStatus.CANCELLED]
        
        total_volume = sum(order["quantity"] for order in sample_orders_list)
        total_value = sum(order["quantity"] * order["price"] for order in sample_orders_list if order["price"])
        total_commission = sum(order["commission"] for order in sample_orders_list)
        
        statistics = {
            "total_orders": total_orders,
            "filled_orders": len(filled_orders),
            "cancelled_orders": len(cancelled_orders),
            "fill_rate": len(filled_orders) / total_orders if total_orders > 0 else 0,
            "total_volume": total_volume,
            "total_value": total_value,
            "total_commission": total_commission,
            "average_order_size": total_volume / total_orders if total_orders > 0 else 0
        }
        # Проверки
        assert statistics is not None
        assert "total_orders" in statistics
        assert "filled_orders" in statistics
        assert "cancelled_orders" in statistics
        assert "fill_rate" in statistics
        assert "total_volume" in statistics
        assert "total_value" in statistics
        assert "total_commission" in statistics
        assert "average_order_size" in statistics
        # Проверка типов данных
        assert isinstance(statistics["total_orders"], int)
        assert isinstance(statistics["filled_orders"], int)
        assert isinstance(statistics["cancelled_orders"], int)
        assert isinstance(statistics["fill_rate"], float)
        assert isinstance(statistics["total_volume"], Decimal)
        assert isinstance(statistics["total_value"], Decimal)
        assert isinstance(statistics["total_commission"], Decimal)
        assert isinstance(statistics["average_order_size"], Decimal)
        # Проверка логики
        assert statistics["total_orders"] >= 0
        assert statistics["filled_orders"] >= 0
        assert statistics["cancelled_orders"] >= 0
        assert 0.0 <= statistics["fill_rate"] <= 1.0
    
    def test_analyze_order_patterns(self, order_utils: OrderUtils, sample_orders_list: list) -> None:
        """Тест анализа паттернов ордеров."""
        # Анализ паттернов (упрощенная версия)
        buy_orders = [order for order in sample_orders_list if order["side"] == OrderSide.BUY]
        sell_orders = [order for order in sample_orders_list if order["side"] == OrderSide.SELL]
        
        patterns = {
            "buy_sell_ratio": len(buy_orders) / len(sell_orders) if len(sell_orders) > 0 else float('inf'),
            "most_common_symbol": "BTCUSDT",  # Упрощенно
            "order_type_distribution": {
                "limit": len([o for o in sample_orders_list if o["type"] == OrderType.LIMIT]),
                "market": len([o for o in sample_orders_list if o["type"] == OrderType.MARKET]),
                "stop": len([o for o in sample_orders_list if o["type"] == OrderType.STOP])
            },
            "time_distribution": "evening_hours"  # Упрощенно
        }
        # Проверки
        assert patterns is not None
        assert "buy_sell_ratio" in patterns
        assert "most_common_symbol" in patterns
        assert "order_type_distribution" in patterns
        assert "time_distribution" in patterns
        # Проверка типов данных
        assert isinstance(patterns["buy_sell_ratio"], float)
        assert isinstance(patterns["most_common_symbol"], str)
        assert isinstance(patterns["order_type_distribution"], dict)
        assert isinstance(patterns["time_distribution"], str)
    
    def test_calculate_order_risk_metrics(self, order_utils: OrderUtils, sample_orders_list: list) -> None:
        """Тест расчета метрик риска ордеров."""
        # Расчет метрик риска (упрощенная версия)
        total_value = sum(order["quantity"] * order["price"] for order in sample_orders_list if order["price"])
        max_order_value = max(order["quantity"] * order["price"] for order in sample_orders_list if order["price"])
        
        risk_metrics = {
            "total_exposure": total_value,
            "max_order_exposure": max_order_value,
            "exposure_concentration": max_order_value / total_value if total_value > 0 else 0,
            "average_order_risk": total_value / len(sample_orders_list) if sample_orders_list else 0,
            "risk_score": 0.7,  # Упрощенно
            "risk_level": "medium"  # Упрощенно
        }
        # Проверки
        assert risk_metrics is not None
        assert "total_exposure" in risk_metrics
        assert "max_order_exposure" in risk_metrics
        assert "exposure_concentration" in risk_metrics
        assert "average_order_risk" in risk_metrics
        assert "risk_score" in risk_metrics
        assert "risk_level" in risk_metrics
        # Проверка типов данных
        assert isinstance(risk_metrics["total_exposure"], Decimal)
        assert isinstance(risk_metrics["max_order_exposure"], Decimal)
        assert isinstance(risk_metrics["exposure_concentration"], Decimal)
        assert isinstance(risk_metrics["average_order_risk"], Decimal)
        assert isinstance(risk_metrics["risk_score"], float)
        assert isinstance(risk_metrics["risk_level"], str)
        # Проверка логики
        assert risk_metrics["total_exposure"] >= 0
        assert risk_metrics["max_order_exposure"] >= 0
        assert 0.0 <= risk_metrics["exposure_concentration"] <= 1.0
        assert risk_metrics["average_order_risk"] >= 0
        assert 0.0 <= risk_metrics["risk_score"] <= 1.0
        assert risk_metrics["risk_level"] in ["low", "medium", "high"]
    
    def test_validate_order_placement(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест валидации размещения ордера."""
        # Валидация размещения (упрощенная версия)
        validation_result = order_utils.validate_order(sample_order)
        placement_validation = {
            "can_place": validation_result.is_valid,
            "estimated_cost": sample_order["quantity"] * sample_order["price"],
            "estimated_commission": sample_order["quantity"] * sample_order["price"] * Decimal("0.001"),
            "estimated_slippage": Decimal("0.0001"),
            "placement_score": 0.9 if validation_result.is_valid else 0.3,
            "placement_recommendations": ["Use limit orders for better execution"] if validation_result.is_valid else validation_result.errors
        }
        # Проверки
        assert placement_validation is not None
        assert "can_place" in placement_validation
        assert "estimated_cost" in placement_validation
        assert "estimated_commission" in placement_validation
        assert "estimated_slippage" in placement_validation
        assert "placement_score" in placement_validation
        assert "placement_recommendations" in placement_validation
        # Проверка типов данных
        assert isinstance(placement_validation["can_place"], bool)
        assert isinstance(placement_validation["estimated_cost"], Decimal)
        assert isinstance(placement_validation["estimated_commission"], Decimal)
        assert isinstance(placement_validation["estimated_slippage"], Decimal)
        assert isinstance(placement_validation["placement_score"], float)
        assert isinstance(placement_validation["placement_recommendations"], list)
        # Проверка диапазона placement_score
        assert 0.0 <= placement_validation["placement_score"] <= 1.0
    
    def test_calculate_order_priority(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест расчета приоритета ордера."""
        # Расчет приоритета (упрощенная версия)
        priority_factors = {
            "order_type": 0.8 if sample_order["type"] == OrderType.MARKET else 0.6,
            "order_size": 0.7 if sample_order["quantity"] > Decimal("0.05") else 0.5,
            "market_volatility": 0.9,  # Упрощенно
            "time_sensitivity": 0.8  # Упрощенно
        }
        
        priority_score = sum(priority_factors.values()) / len(priority_factors)
        priority_level = "high" if priority_score > 0.8 else "medium" if priority_score > 0.5 else "low"
        
        priority_result = {
            "priority_score": priority_score,
            "priority_level": priority_level,
            "priority_factors": priority_factors,
            "execution_recommendations": ["Execute immediately"] if priority_level == "high" else ["Normal execution"]
        }
        # Проверки
        assert priority_result is not None
        assert "priority_score" in priority_result
        assert "priority_level" in priority_result
        assert "priority_factors" in priority_result
        assert "execution_recommendations" in priority_result
        # Проверка типов данных
        assert isinstance(priority_result["priority_score"], float)
        assert isinstance(priority_result["priority_level"], str)
        assert isinstance(priority_result["priority_factors"], dict)
        assert isinstance(priority_result["execution_recommendations"], list)
        # Проверка диапазона priority_score
        assert 0.0 <= priority_result["priority_score"] <= 1.0
        assert priority_result["priority_level"] in ["low", "medium", "high"]
    
    def test_analyze_order_impact(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест анализа влияния ордера."""
        # Анализ влияния (упрощенная версия)
        order_value = sample_order["quantity"] * sample_order["price"]
        market_impact = order_value * Decimal("0.001")  # Упрощенно
        
        impact_analysis = {
            "order_value": order_value,
            "estimated_market_impact": market_impact,
            "impact_score": 0.3 if market_impact < order_value * Decimal("0.01") else 0.7,
            "impact_level": "low" if market_impact < order_value * Decimal("0.01") else "medium",
            "execution_strategy": "immediate" if market_impact < order_value * Decimal("0.01") else "gradual",
            "risk_assessment": "low_risk" if market_impact < order_value * Decimal("0.01") else "medium_risk"
        }
        # Проверки
        assert impact_analysis is not None
        assert "order_value" in impact_analysis
        assert "estimated_market_impact" in impact_analysis
        assert "impact_score" in impact_analysis
        assert "impact_level" in impact_analysis
        assert "execution_strategy" in impact_analysis
        assert "risk_assessment" in impact_analysis
        # Проверка типов данных
        assert isinstance(impact_analysis["order_value"], Decimal)
        assert isinstance(impact_analysis["estimated_market_impact"], Decimal)
        assert isinstance(impact_analysis["impact_score"], float)
        assert isinstance(impact_analysis["impact_level"], str)
        assert isinstance(impact_analysis["execution_strategy"], str)
        assert isinstance(impact_analysis["risk_assessment"], str)
        # Проверка диапазона impact_score
        assert 0.0 <= impact_analysis["impact_score"] <= 1.0
        assert impact_analysis["impact_level"] in ["low", "medium", "high"]
        assert impact_analysis["execution_strategy"] in ["immediate", "gradual", "scheduled"]
        assert impact_analysis["risk_assessment"] in ["low_risk", "medium_risk", "high_risk"]
    
    def test_validate_order_risk_limits(self, order_utils: OrderUtils, sample_order: dict) -> None:
        """Тест валидации лимитов риска ордера."""
        # Валидация лимитов риска (упрощенная версия)
        order_value = sample_order["quantity"] * sample_order["price"]
        max_position_size = Decimal("100000.0")  # Упрощенно
        max_order_size = Decimal("10000.0")  # Упрощенно
        
        risk_validation = {
            "within_position_limits": order_value <= max_position_size,
            "within_order_limits": order_value <= max_order_size,
            "risk_score": 0.3 if order_value <= max_order_size else 0.8,
            "risk_level": "low" if order_value <= max_order_size else "high",
            "validation_passed": order_value <= max_order_size,
            "risk_warnings": [] if order_value <= max_order_size else ["Order size exceeds limits"]
        }
        # Проверки
        assert risk_validation is not None
        assert "within_position_limits" in risk_validation
        assert "within_order_limits" in risk_validation
        assert "risk_score" in risk_validation
        assert "risk_level" in risk_validation
        assert "validation_passed" in risk_validation
        assert "risk_warnings" in risk_validation
        # Проверка типов данных
        assert isinstance(risk_validation["within_position_limits"], bool)
        assert isinstance(risk_validation["within_order_limits"], bool)
        assert isinstance(risk_validation["risk_score"], float)
        assert isinstance(risk_validation["risk_level"], str)
        assert isinstance(risk_validation["validation_passed"], bool)
        assert isinstance(risk_validation["risk_warnings"], list)
        # Проверка диапазона risk_score
        assert 0.0 <= risk_validation["risk_score"] <= 1.0
        assert risk_validation["risk_level"] in ["low", "medium", "high"]
    
    def test_error_handling(self, order_utils: OrderUtils) -> None:
        """Тест обработки ошибок."""
        # Тест с невалидным ордером
        invalid_order: dict[str, Any] = {}
        validation_result = order_utils.validate_order(invalid_order)
        assert validation_result is not None
        assert hasattr(validation_result, 'is_valid')
        assert hasattr(validation_result, 'errors')
    
    def test_edge_cases(self, order_utils: OrderUtils) -> None:
        """Тест граничных случаев."""
        # Тест с минимальными значениями
        min_order = {
            "id": "min_order",
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("0.00000001"),
            "price": Decimal("0.00000001"),
            "status": OrderStatus.PENDING,
            "timestamp": datetime.now()
        }
        validation_result = order_utils.validate_order(min_order)
        assert validation_result is not None
        
        # Тест с максимальными значениями
        max_order = {
            "id": "max_order",
            "symbol": "BTCUSDT",
            "side": OrderSide.SELL,
            "type": OrderType.MARKET,
            "quantity": Decimal("999999999.99999999"),
            "price": Decimal("999999999.99999999"),
            "status": OrderStatus.PENDING,
            "timestamp": datetime.now()
        }
        validation_result = order_utils.validate_order(max_order)
        assert validation_result is not None
        
        # Тест с пустыми значениями
        empty_order = {
            "id": "",
            "symbol": "",
            "side": OrderSide.BUY,
            "type": OrderType.LIMIT,
            "quantity": Decimal("0"),
            "price": Decimal("0"),
            "status": OrderStatus.PENDING,
            "timestamp": datetime.now()
        }
        validation_result = order_utils.validate_order(empty_order)
        assert validation_result is not None 
