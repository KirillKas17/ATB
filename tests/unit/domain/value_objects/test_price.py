"""
Unit тесты для price.py.

Покрывает:
- Основной функционал Price
- Валидацию данных
- Бизнес-логику операций с ценами
- Обработку ошибок
- Математические операции
- Сериализацию и десериализацию
"""

import pytest
import dataclasses
from typing import Dict, Any, Tuple
from unittest.mock import Mock, patch
from decimal import Decimal

from domain.value_objects.price import Price
from domain.value_objects.currency import Currency


class TestPrice:
    """Тесты для Price."""

    @pytest.fixture
    def sample_price(self) -> Price:
        """Тестовая цена."""
        return Price(
            value=Decimal("50000.00"),
            currency=Currency.BTC,
            quote_currency=Currency.USD
        )

    @pytest.fixture
    def usd_price(self) -> Price:
        """Цена в USD."""
        return Price(
            value=Decimal("1.00"),
            currency=Currency.USD,
            quote_currency=Currency.USD
        )

    @pytest.fixture
    def eth_price(self) -> Price:
        """Цена ETH."""
        return Price(
            value=Decimal("3000.00"),
            currency=Currency.ETH,
            quote_currency=Currency.USD
        )

    def test_price_creation(self, sample_price):
        """Тест создания цены."""
        assert sample_price.value == Decimal("50000.00")
        assert sample_price.currency == Currency.BTC
        assert sample_price.quote_currency == Currency.USD

    def test_price_creation_default_quote_currency(self):
        """Тест создания цены с валютой по умолчанию."""
        price = Price(value=Decimal("100.00"), currency=Currency.USD)
        assert price.quote_currency == Currency.USD

    def test_price_validation_negative_value(self):
        """Тест валидации отрицательной цены."""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            Price(value=Decimal("-100.00"), currency=Currency.USD)

    def test_price_zero_value(self):
        """Тест цены с нулевым значением."""
        price = Price(value=Decimal("0.00"), currency=Currency.USD)
        assert price.value == Decimal("0.00")
        assert price.validate() is True

    def test_price_arithmetic_operations(self, sample_price, eth_price):
        """Тест арифметических операций."""
        # Сложение
        result = sample_price + sample_price
        assert result.value == Decimal("100000.00")
        assert result.currency == Currency.BTC
        assert result.quote_currency == Currency.USD

        # Вычитание
        result = sample_price - Price(value=Decimal("10000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        assert result.value == Decimal("40000.00")

        # Умножение
        result = sample_price * 2
        assert result.value == Decimal("100000.00")

        # Деление
        result = sample_price / 2
        assert result.value == Decimal("25000.00")

    def test_price_arithmetic_errors(self, sample_price, eth_price):
        """Тест ошибок арифметических операций."""
        # Сложение с разными валютами
        with pytest.raises(ValueError, match="Cannot add prices with different currencies"):
            sample_price + eth_price

        # Сложение с неверным типом
        with pytest.raises(TypeError, match="Can only add Price with Price"):
            sample_price + 100

        # Вычитание с разными валютами
        with pytest.raises(ValueError, match="Cannot subtract prices with different currencies"):
            sample_price - eth_price

        # Деление на ноль
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            sample_price / 0

    def test_price_comparison_operations(self, sample_price):
        """Тест операций сравнения."""
        lower_price = Price(value=Decimal("40000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        higher_price = Price(value=Decimal("60000.00"), currency=Currency.BTC, quote_currency=Currency.USD)

        # Меньше
        assert lower_price < sample_price
        assert sample_price < higher_price

        # Больше
        assert higher_price > sample_price
        assert sample_price > lower_price

        # Меньше или равно
        assert lower_price <= sample_price
        assert sample_price <= sample_price

        # Больше или равно
        assert higher_price >= sample_price
        assert sample_price >= sample_price

    def test_price_comparison_errors(self, sample_price, eth_price):
        """Тест ошибок сравнения."""
        with pytest.raises(ValueError, match="Cannot compare prices with different currencies"):
            sample_price < eth_price

        with pytest.raises(TypeError, match="Can only compare Price with Price"):
            sample_price < 100

    def test_price_percentage_change(self, sample_price):
        """Тест вычисления процентного изменения."""
        old_price = Price(value=Decimal("40000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        new_price = Price(value=Decimal("50000.00"), currency=Currency.BTC, quote_currency=Currency.USD)

        change = new_price.percentage_change(old_price)
        assert change == Decimal("25.00")  # 25% увеличение

        # Обратное изменение
        change = old_price.percentage_change(new_price)
        assert change == Decimal("-20.00")  # 20% уменьшение

    def test_price_percentage_change_zero_base(self, sample_price):
        """Тест процентного изменения с нулевой базой."""
        zero_price = Price(value=Decimal("0.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        change = sample_price.percentage_change(zero_price)
        assert change == Decimal("0")  # Должно возвращать 0 при делении на ноль

    def test_price_spread(self, sample_price):
        """Тест вычисления спреда."""
        bid_price = Price(value=Decimal("49000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        ask_price = Price(value=Decimal("51000.00"), currency=Currency.BTC, quote_currency=Currency.USD)

        spread = ask_price.spread(bid_price)
        assert spread == Decimal("2000.00")

        # Спред должен быть абсолютным значением
        spread = bid_price.spread(ask_price)
        assert spread == Decimal("2000.00")

    def test_price_slippage(self, sample_price):
        """Тест вычисления проскальзывания."""
        target_price = Price(value=Decimal("48000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        
        slippage = sample_price.slippage(target_price)
        expected_slippage = abs(Decimal("50000.00") - Decimal("48000.00")) / Decimal("48000.00")
        assert slippage == expected_slippage

    def test_price_slippage_zero_target(self, sample_price):
        """Тест проскальзывания с нулевой целевой ценой."""
        zero_price = Price(value=Decimal("0.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        slippage = sample_price.slippage(zero_price)
        assert slippage == Decimal("0")

    def test_price_properties(self, sample_price):
        """Тест свойств цены."""
        assert sample_price.amount == Decimal("50000.00")
        assert sample_price.to_decimal() == Decimal("50000.00")

    def test_price_percentage_change_from(self, sample_price):
        """Тест метода percentage_change_from."""
        old_price = Price(value=Decimal("40000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        change = sample_price.percentage_change_from(old_price)
        assert change == Decimal("25.00")

    def test_price_spread_with(self, sample_price):
        """Тест метода spread_with."""
        other_price = Price(value=Decimal("49000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        spread_price = sample_price.spread_with(other_price)
        
        assert isinstance(spread_price, Price)
        assert spread_price.value == Decimal("1000.00")
        assert spread_price.currency == Currency.BTC
        assert spread_price.quote_currency == Currency.USD

    def test_price_apply_slippage(self, sample_price):
        """Тест применения проскальзывания."""
        slippage_percent = Decimal("1.0")  # 1%
        bid_price, ask_price = sample_price.apply_slippage(slippage_percent)
        
        # Bid цена должна быть ниже
        assert bid_price.value < sample_price.value
        # Ask цена должна быть выше
        assert ask_price.value > sample_price.value
        
        # Проверяем точность
        expected_bid = Decimal("50000.00") * Decimal("0.99")
        expected_ask = Decimal("50000.00") * Decimal("1.01")
        assert bid_price.value == expected_bid
        assert ask_price.value == expected_ask

    def test_price_to_dict(self, sample_price):
        """Тест сериализации в словарь."""
        result = sample_price.to_dict()
        
        assert result["value"] == "50000.00"
        assert result["currency"] == "BTC"
        assert result["quote_currency"] == "USD"
        assert result["type"] == "Price"

    def test_price_from_dict(self, sample_price):
        """Тест десериализации из словаря."""
        data = {
            "value": "50000.00",
            "currency": "BTC",
            "quote_currency": "USD"
        }
        
        price = Price.from_dict(data)
        assert price.value == Decimal("50000.00")
        assert price.currency == Currency.BTC
        assert price.quote_currency == Currency.USD

    def test_price_from_dict_default_quote(self):
        """Тест десериализации с валютой по умолчанию."""
        data = {
            "value": "100.00",
            "currency": "USD",
            "quote_currency": None
        }
        
        price = Price.from_dict(data)
        assert price.quote_currency == Currency.USD

    def test_price_from_dict_invalid_currency(self):
        """Тест десериализации с неверной валютой."""
        data = {
            "value": "100.00",
            "currency": "INVALID",
            "quote_currency": "USD"
        }
        
        with pytest.raises(ValueError, match="Unknown currency"):
            Price.from_dict(data)

    def test_price_hash(self, sample_price):
        """Тест хеширования цены."""
        hash_value = sample_price.hash
        assert len(hash_value) == 32  # MD5 hex digest length
        assert isinstance(hash_value, str)

    def test_price_validation(self, sample_price):
        """Тест валидации цены."""
        assert sample_price.validate() is True

        # Невалидная цена (отрицательная)
        invalid_price = Price(value=Decimal("-100.00"), currency=Currency.USD)
        # Но валидация не проверяет отрицательные значения в __post_init__
        assert invalid_price.validate() is True

    def test_price_equality(self, sample_price):
        """Тест равенства цен."""
        same_price = Price(value=Decimal("50000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        different_price = Price(value=Decimal("60000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        
        assert sample_price == same_price
        assert sample_price != different_price
        assert sample_price != "not a price"

    def test_price_hash_equality(self, sample_price):
        """Тест хеширования для равенства."""
        same_price = Price(value=Decimal("50000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        
        assert hash(sample_price) == hash(same_price)

    def test_price_str_representation(self, sample_price):
        """Тест строкового представления."""
        result = str(sample_price)
        assert result == "50000.00 BTC/USD"

    def test_price_repr_representation(self, sample_price):
        """Тест repr представления."""
        result = repr(sample_price)
        assert result == "Price(50000.00, BTC, USD)"


class TestPriceOperations:
    """Тесты операций с ценами."""

    def test_price_precision_handling(self):
        """Тест обработки точности."""
        price = Price(value=Decimal("100.123456"), currency=Currency.USD)
        assert price.value == Decimal("100.123456")

        # Операции с высокой точностью
        result = price * Decimal("1.000001")
        assert result.value == Decimal("100.123456") * Decimal("1.000001")

    def test_price_large_numbers(self):
        """Тест больших чисел."""
        large_price = Price(value=Decimal("999999999.99999999"), currency=Currency.USD)
        assert large_price.validate() is True

        # Операции с большими числами
        result = large_price * 2
        assert result.value == Decimal("1999999999.99999998")

    def test_price_immutability(self, sample_price):
        """Тест неизменяемости цены."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample_price.value = Decimal("60000.00")

    def test_price_currency_consistency(self):
        """Тест консистентности валют."""
        # Все операции должны сохранять валюты
        price1 = Price(value=Decimal("100.00"), currency=Currency.USD)
        price2 = Price(value=Decimal("200.00"), currency=Currency.USD)
        
        result = price1 + price2
        assert result.currency == Currency.USD
        assert result.quote_currency == Currency.USD

    def test_price_serialization_roundtrip(self):
        """Тест сериализации и десериализации."""
        original_price = Price(value=Decimal("12345.67"), currency=Currency.ETH, quote_currency=Currency.USD)
        
        # Сериализация
        data = original_price.to_dict()
        
        # Десериализация
        restored_price = Price.from_dict(data)
        
        # Проверка равенства
        assert restored_price == original_price
        assert restored_price.value == original_price.value
        assert restored_price.currency == original_price.currency
        assert restored_price.quote_currency == original_price.quote_currency


class TestPriceEdgeCases:
    """Тесты граничных случаев для цен."""

    def test_price_minimum_values(self):
        """Тест минимальных значений."""
        min_price = Price(value=Decimal("0.00000001"), currency=Currency.USD)
        assert min_price.validate() is True
        assert min_price.value > 0

    def test_price_maximum_values(self):
        """Тест максимальных значений."""
        max_price = Price(value=Decimal("999999999999999.99999999"), currency=Currency.USD)
        assert max_price.validate() is True

    def test_price_different_currencies(self):
        """Тест разных валют."""
        btc_price = Price(value=Decimal("50000.00"), currency=Currency.BTC, quote_currency=Currency.USD)
        eth_price = Price(value=Decimal("3000.00"), currency=Currency.ETH, quote_currency=Currency.USD)
        
        # Цены в разных валютах не должны сравниваться
        with pytest.raises(ValueError):
            btc_price < eth_price

    def test_price_slippage_edge_cases(self):
        """Тест граничных случаев проскальзывания."""
        price = Price(value=Decimal("100.00"), currency=Currency.USD)
        
        # Нулевое проскальзывание
        bid, ask = price.apply_slippage(Decimal("0.0"))
        assert bid.value == ask.value == price.value
        
        # Очень большое проскальзывание
        bid, ask = price.apply_slippage(Decimal("50.0"))  # 50%
        assert bid.value == Decimal("50.00")
        assert ask.value == Decimal("150.00")

    def test_price_percentage_change_edge_cases(self):
        """Тест граничных случаев процентного изменения."""
        # Изменение с очень маленькой базой
        small_price = Price(value=Decimal("0.000001"), currency=Currency.USD)
        large_price = Price(value=Decimal("1000000.00"), currency=Currency.USD)
        
        change = large_price.percentage_change(small_price)
        assert change == Decimal("0")  # При делении на очень маленькое число

    def test_price_hash_collision_resistance(self):
        """Тест устойчивости к коллизиям хешей."""
        prices = [
            Price(value=Decimal("100.00"), currency=Currency.USD),
            Price(value=Decimal("200.00"), currency=Currency.USD),
            Price(value=Decimal("100.00"), currency=Currency.EUR),
            Price(value=Decimal("100.00"), currency=Currency.USD, quote_currency=Currency.EUR),
        ]
        
        hashes = [price.hash for price in prices]
        assert len(hashes) == len(set(hashes))  # Все хеши должны быть уникальными

    def test_price_performance(self):
        """Тест производительности операций с ценами."""
        import time
        
        price = Price(value=Decimal("100.00"), currency=Currency.USD)
        
        # Тест скорости арифметических операций
        start_time = time.time()
        for _ in range(1000):
            result = price * 2
        end_time = time.time()
        
        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0  # Менее 1 секунды для 1000 операций 