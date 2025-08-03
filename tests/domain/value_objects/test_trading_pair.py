"""
Unit тесты для domain/value_objects/trading_pair.py.
"""

import pytest
from typing import Dict, Any
from decimal import Decimal

from domain.value_objects.trading_pair import TradingPair
from domain.exceptions.base_exceptions import ValidationError


class TestTradingPair:
    """Тесты для TradingPair."""
    
    @pytest.fixture
    def sample_data(self) -> Dict[str, Any]:
        """Тестовые данные."""
        return {
            "base_currency": "BTC",
            "quote_currency": "USD",
            "min_order_size": Decimal("0.001"),
            "max_order_size": Decimal("100.0"),
            "price_precision": 2,
            "quantity_precision": 6,
            "min_notional": Decimal("10.00"),
            "tick_size": Decimal("0.01"),
            "step_size": Decimal("0.000001"),
            "is_active": True,
            "trading_fee": Decimal("0.001"),
            "maker_fee": Decimal("0.0005"),
            "taker_fee": Decimal("0.001")
        }
    
    def test_creation(self, sample_data):
        """Тест создания TradingPair."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"],
            is_active=sample_data["is_active"],
            trading_fee=sample_data["trading_fee"],
            maker_fee=sample_data["maker_fee"],
            taker_fee=sample_data["taker_fee"]
        )
        
        assert trading_pair.base_currency == sample_data["base_currency"]
        assert trading_pair.quote_currency == sample_data["quote_currency"]
        assert trading_pair.min_order_size == sample_data["min_order_size"]
        assert trading_pair.max_order_size == sample_data["max_order_size"]
        assert trading_pair.price_precision == sample_data["price_precision"]
        assert trading_pair.quantity_precision == sample_data["quantity_precision"]
        assert trading_pair.min_notional == sample_data["min_notional"]
        assert trading_pair.tick_size == sample_data["tick_size"]
        assert trading_pair.step_size == sample_data["step_size"]
        assert trading_pair.is_active == sample_data["is_active"]
        assert trading_pair.trading_fee == sample_data["trading_fee"]
        assert trading_pair.maker_fee == sample_data["maker_fee"]
        assert trading_pair.taker_fee == sample_data["taker_fee"]
    
    def test_validation_empty_base_currency(self, sample_data):
        """Тест валидации пустой базовой валюты."""
        with pytest.raises(ValidationError, match="Base currency cannot be empty"):
            TradingPair(
                base_currency="",
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_empty_quote_currency(self, sample_data):
        """Тест валидации пустой котируемой валюты."""
        with pytest.raises(ValidationError, match="Quote currency cannot be empty"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency="",
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_same_currencies(self, sample_data):
        """Тест валидации одинаковых валют."""
        with pytest.raises(ValidationError, match="Base and quote currencies cannot be the same"):
            TradingPair(
                base_currency="BTC",
                quote_currency="BTC",
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_negative_min_order_size(self, sample_data):
        """Тест валидации отрицательного минимального размера заказа."""
        with pytest.raises(ValidationError, match="Min order size cannot be negative"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=Decimal("-0.001"),
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_min_greater_than_max_order_size(self, sample_data):
        """Тест валидации когда min > max order size."""
        with pytest.raises(ValidationError, match="Min order size cannot be greater than max order size"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=Decimal("200.0"),
                max_order_size=Decimal("100.0"),
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_invalid_price_precision(self, sample_data):
        """Тест валидации некорректной точности цены."""
        with pytest.raises(ValidationError, match="Price precision must be between 0 and 8"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=10,
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_invalid_quantity_precision(self, sample_data):
        """Тест валидации некорректной точности количества."""
        with pytest.raises(ValidationError, match="Quantity precision must be between 0 and 8"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=10,
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_negative_min_notional(self, sample_data):
        """Тест валидации отрицательного минимального номинала."""
        with pytest.raises(ValidationError, match="Min notional cannot be negative"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=Decimal("-10.00"),
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"]
            )
    
    def test_validation_negative_tick_size(self, sample_data):
        """Тест валидации отрицательного tick size."""
        with pytest.raises(ValidationError, match="Tick size cannot be negative"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=Decimal("-0.01"),
                step_size=sample_data["step_size"]
            )
    
    def test_validation_negative_step_size(self, sample_data):
        """Тест валидации отрицательного step size."""
        with pytest.raises(ValidationError, match="Step size cannot be negative"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=Decimal("-0.000001")
            )
    
    def test_validation_negative_fees(self, sample_data):
        """Тест валидации отрицательных комиссий."""
        with pytest.raises(ValidationError, match="Trading fee cannot be negative"):
            TradingPair(
                base_currency=sample_data["base_currency"],
                quote_currency=sample_data["quote_currency"],
                min_order_size=sample_data["min_order_size"],
                max_order_size=sample_data["max_order_size"],
                price_precision=sample_data["price_precision"],
                quantity_precision=sample_data["quantity_precision"],
                min_notional=sample_data["min_notional"],
                tick_size=sample_data["tick_size"],
                step_size=sample_data["step_size"],
                trading_fee=Decimal("-0.001")
            )
    
    def test_symbol_property(self, sample_data):
        """Тест свойства symbol."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair.symbol == "BTCUSD"
    
    def test_symbol_property_with_separator(self, sample_data):
        """Тест свойства symbol с разделителем."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair.symbol_with_separator("/") == "BTC/USD"
        assert trading_pair.symbol_with_separator("-") == "BTC-USD"
    
    def test_is_valid_order_size(self, sample_data):
        """Тест проверки валидности размера заказа."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair.is_valid_order_size(Decimal("0.5")) is True
        assert trading_pair.is_valid_order_size(Decimal("0.0001")) is False  # Меньше min
        assert trading_pair.is_valid_order_size(Decimal("200.0")) is False   # Больше max
    
    def test_is_valid_price(self, sample_data):
        """Тест проверки валидности цены."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair.is_valid_price(Decimal("50000.00")) is True
        assert trading_pair.is_valid_price(Decimal("50000.001")) is False  # Не соответствует tick_size
    
    def test_is_valid_quantity(self, sample_data):
        """Тест проверки валидности количества."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair.is_valid_quantity(Decimal("0.5")) is True
        assert trading_pair.is_valid_quantity(Decimal("0.0000001")) is False  # Не соответствует step_size
    
    def test_is_valid_notional(self, sample_data):
        """Тест проверки валидности номинала."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair.is_valid_notional(Decimal("50.00")) is True
        assert trading_pair.is_valid_notional(Decimal("5.00")) is False  # Меньше min_notional
    
    def test_round_price(self, sample_data):
        """Тест округления цены."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        rounded_price = trading_pair.round_price(Decimal("50000.123"))
        assert rounded_price == Decimal("50000.12")
    
    def test_round_quantity(self, sample_data):
        """Тест округления количества."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        rounded_quantity = trading_pair.round_quantity(Decimal("0.123456789"))
        assert rounded_quantity == Decimal("0.123457")
    
    def test_calculate_fee(self, sample_data):
        """Тест расчета комиссии."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"],
            trading_fee=sample_data["trading_fee"]
        )
        
        fee = trading_pair.calculate_fee(Decimal("1000.00"))
        assert fee == Decimal("1.00")  # 1000 * 0.001
    
    def test_calculate_maker_fee(self, sample_data):
        """Тест расчета maker комиссии."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"],
            maker_fee=sample_data["maker_fee"]
        )
        
        fee = trading_pair.calculate_maker_fee(Decimal("1000.00"))
        assert fee == Decimal("0.50")  # 1000 * 0.0005
    
    def test_calculate_taker_fee(self, sample_data):
        """Тест расчета taker комиссии."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"],
            taker_fee=sample_data["taker_fee"]
        )
        
        fee = trading_pair.calculate_taker_fee(Decimal("1000.00"))
        assert fee == Decimal("1.00")  # 1000 * 0.001
    
    def test_to_dict(self, sample_data):
        """Тест сериализации в словарь."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"],
            is_active=sample_data["is_active"],
            trading_fee=sample_data["trading_fee"],
            maker_fee=sample_data["maker_fee"],
            taker_fee=sample_data["taker_fee"]
        )
        
        result = trading_pair.to_dict()
        
        assert result["base_currency"] == sample_data["base_currency"]
        assert result["quote_currency"] == sample_data["quote_currency"]
        assert result["symbol"] == "BTCUSD"
        assert result["min_order_size"] == str(sample_data["min_order_size"])
        assert result["max_order_size"] == str(sample_data["max_order_size"])
        assert result["price_precision"] == sample_data["price_precision"]
        assert result["quantity_precision"] == sample_data["quantity_precision"]
        assert result["min_notional"] == str(sample_data["min_notional"])
        assert result["tick_size"] == str(sample_data["tick_size"])
        assert result["step_size"] == str(sample_data["step_size"])
        assert result["is_active"] == sample_data["is_active"]
        assert result["trading_fee"] == str(sample_data["trading_fee"])
        assert result["maker_fee"] == str(sample_data["maker_fee"])
        assert result["taker_fee"] == str(sample_data["taker_fee"])
    
    def test_from_dict(self, sample_data):
        """Тест десериализации из словаря."""
        data = {
            "base_currency": sample_data["base_currency"],
            "quote_currency": sample_data["quote_currency"],
            "min_order_size": str(sample_data["min_order_size"]),
            "max_order_size": str(sample_data["max_order_size"]),
            "price_precision": sample_data["price_precision"],
            "quantity_precision": sample_data["quantity_precision"],
            "min_notional": str(sample_data["min_notional"]),
            "tick_size": str(sample_data["tick_size"]),
            "step_size": str(sample_data["step_size"]),
            "is_active": sample_data["is_active"],
            "trading_fee": str(sample_data["trading_fee"]),
            "maker_fee": str(sample_data["maker_fee"]),
            "taker_fee": str(sample_data["taker_fee"])
        }
        
        trading_pair = TradingPair.from_dict(data)
        
        assert trading_pair.base_currency == sample_data["base_currency"]
        assert trading_pair.quote_currency == sample_data["quote_currency"]
        assert trading_pair.min_order_size == sample_data["min_order_size"]
        assert trading_pair.max_order_size == sample_data["max_order_size"]
        assert trading_pair.price_precision == sample_data["price_precision"]
        assert trading_pair.quantity_precision == sample_data["quantity_precision"]
        assert trading_pair.min_notional == sample_data["min_notional"]
        assert trading_pair.tick_size == sample_data["tick_size"]
        assert trading_pair.step_size == sample_data["step_size"]
        assert trading_pair.is_active == sample_data["is_active"]
        assert trading_pair.trading_fee == sample_data["trading_fee"]
        assert trading_pair.maker_fee == sample_data["maker_fee"]
        assert trading_pair.taker_fee == sample_data["taker_fee"]
    
    def test_equality(self, sample_data):
        """Тест равенства объектов."""
        trading_pair1 = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        trading_pair2 = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        assert trading_pair1 == trading_pair2
        assert hash(trading_pair1) == hash(trading_pair2)
    
    def test_str_representation(self, sample_data):
        """Тест строкового представления."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        expected = "BTCUSD (BTC/USD)"
        assert str(trading_pair) == expected
    
    def test_repr_representation(self, sample_data):
        """Тест repr представления."""
        trading_pair = TradingPair(
            base_currency=sample_data["base_currency"],
            quote_currency=sample_data["quote_currency"],
            min_order_size=sample_data["min_order_size"],
            max_order_size=sample_data["max_order_size"],
            price_precision=sample_data["price_precision"],
            quantity_precision=sample_data["quantity_precision"],
            min_notional=sample_data["min_notional"],
            tick_size=sample_data["tick_size"],
            step_size=sample_data["step_size"]
        )
        
        expected = f"TradingPair(base_currency='BTC', quote_currency='USD', symbol='BTCUSD')"
        assert repr(trading_pair) == expected 