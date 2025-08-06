#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Price Value Object.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.exceptions import ValidationError


class TestPrice:
    """Тесты для Price Value Object."""

    @pytest.fixture
    def usd_currency(self) -> Currency:
        """Фикстура USD валюты."""
        return Currency("USD")

    @pytest.fixture
    def btc_currency(self) -> Currency:
        """Фикстура BTC валюты."""
        return Currency("BTC")

    @pytest.fixture
    def sample_price_btc(self, usd_currency: Currency) -> Price:
        """Фикстура цены BTC в USD."""
        return Price(value=Decimal("45000.50"), currency=usd_currency)

    @pytest.fixture
    def sample_price_eth(self, usd_currency: Currency) -> Price:
        """Фикстура цены ETH в USD."""
        return Price(value=Decimal("3200.75"), currency=usd_currency)

    def test_price_creation_valid(self, usd_currency: Currency) -> None:
        """Тест создания валидной цены."""
        price = Price(value=Decimal("100.50"), currency=usd_currency)
        
        assert price.value == Decimal("100.50")
        assert price.currency == usd_currency
        assert str(price) == "100.50 USD"

    def test_price_creation_zero(self, usd_currency: Currency) -> None:
        """Тест создания нулевой цены."""
        price = Price(value=Decimal("0"), currency=usd_currency)
        
        assert price.value == Decimal("0")
        assert price.is_zero()
        assert not price.is_positive()

    def test_price_creation_negative_invalid(self, usd_currency: Currency) -> None:
        """Тест создания отрицательной цены (недопустимо)."""
        with pytest.raises(ValidationError, match="Price cannot be negative"):
            Price(value=Decimal("-10.50"), currency=usd_currency)

    def test_price_creation_too_large(self, usd_currency: Currency) -> None:
        """Тест создания слишком большой цены."""
        with pytest.raises(ValidationError):
            Price(value=Decimal("1000000000000"), currency=usd_currency)

    def test_price_creation_invalid_precision(self, usd_currency: Currency) -> None:
        """Тест создания цены с невалидной точностью."""
        with pytest.raises(ValidationError):
            Price(value=Decimal("100.123456789012345"), currency=usd_currency)

    def test_price_comparison_same_currency(self, usd_currency: Currency) -> None:
        """Тест сравнения цен в одной валюте."""
        price1 = Price(value=Decimal("100"), currency=usd_currency)
        price2 = Price(value=Decimal("200"), currency=usd_currency)
        price3 = Price(value=Decimal("100"), currency=usd_currency)
        
        assert price1 < price2
        assert price2 > price1
        assert price1 == price3
        assert price1 <= price2
        assert price2 >= price1

    def test_price_comparison_different_currency(self, sample_price_btc: Price, btc_currency: Currency) -> None:
        """Тест сравнения цен в разных валютах."""
        eth_price = Price(value=Decimal("3000"), currency=btc_currency)
        
        with pytest.raises(ValidationError, match="Cannot compare prices with different currencies"):
            sample_price_btc < eth_price

    def test_price_arithmetic_operations(self, sample_price_btc: Price) -> None:
        """Тест арифметических операций."""
        # Умножение на коэффициент
        doubled = sample_price_btc * Decimal("2")
        assert doubled.value == sample_price_btc.value * 2
        assert doubled.currency == sample_price_btc.currency
        
        # Деление на коэффициент
        halved = sample_price_btc / Decimal("2")
        assert halved.value == sample_price_btc.value / 2
        assert halved.currency == sample_price_btc.currency

    def test_price_division_by_zero(self, sample_price_btc: Price) -> None:
        """Тест деления на ноль."""
        with pytest.raises(ValidationError, match="Cannot divide by zero"):
            sample_price_btc / Decimal("0")

    def test_price_percentage_change(self, usd_currency: Currency) -> None:
        """Тест расчета процентного изменения."""
        old_price = Price(value=Decimal("100"), currency=usd_currency)
        new_price = Price(value=Decimal("110"), currency=usd_currency)
        
        change = new_price.percentage_change_from(old_price)
        assert change == Decimal("10.0")  # 10% рост
        
        # Обратный тест
        reverse_change = old_price.percentage_change_from(new_price)
        assert reverse_change == Decimal("-9.090909090909091")  # ~-9.09% падение

    def test_price_percentage_change_different_currency(self, sample_price_btc: Price, btc_currency: Currency) -> None:
        """Тест расчета процентного изменения для разных валют."""
        eth_price = Price(value=Decimal("3000"), currency=btc_currency)
        
        with pytest.raises(ValidationError, match="Cannot calculate percentage change"):
            sample_price_btc.percentage_change_from(eth_price)

    def test_price_apply_percentage_change(self, sample_price_btc: Price) -> None:
        """Тест применения процентного изменения."""
        # Рост на 10%
        increased = sample_price_btc.apply_percentage_change(Decimal("10"))
        expected_value = sample_price_btc.value * Decimal("1.10")
        assert increased.value == expected_value
        
        # Падение на 15%
        decreased = sample_price_btc.apply_percentage_change(Decimal("-15"))
        expected_value = sample_price_btc.value * Decimal("0.85")
        assert decreased.value == expected_value

    def test_price_spread_calculation(self, usd_currency: Currency) -> None:
        """Тест расчета спреда."""
        bid_price = Price(value=Decimal("100"), currency=usd_currency)
        ask_price = Price(value=Decimal("102"), currency=usd_currency)
        
        spread = ask_price.calculate_spread(bid_price)
        assert spread.value == Decimal("2")
        assert spread.currency == usd_currency
        
        # Спред в процентах
        spread_pct = ask_price.calculate_spread_percentage(bid_price)
        assert spread_pct == Decimal("2.0")  # 2%

    def test_price_round_to_tick_size(self, usd_currency: Currency) -> None:
        """Тест округления до размера тика."""
        price = Price(value=Decimal("123.456789"), currency=usd_currency)
        
        # Округление до 0.01
        rounded = price.round_to_tick_size(Decimal("0.01"))
        assert rounded.value == Decimal("123.46")
        
        # Округление до 0.1
        rounded = price.round_to_tick_size(Decimal("0.1"))
        assert rounded.value == Decimal("123.5")
        
        # Округление до 1
        rounded = price.round_to_tick_size(Decimal("1"))
        assert rounded.value == Decimal("123")

    def test_price_market_impact(self, sample_price_btc: Price) -> None:
        """Тест расчета рыночного воздействия."""
        # Воздействие покупки (цена растет)
        buy_impact = sample_price_btc.apply_market_impact(
            side="BUY", 
            impact_percentage=Decimal("0.1")
        )
        expected = sample_price_btc.value * Decimal("1.001")
        assert buy_impact.value == expected
        
        # Воздействие продажи (цена падает)
        sell_impact = sample_price_btc.apply_market_impact(
            side="SELL", 
            impact_percentage=Decimal("0.1")
        )
        expected = sample_price_btc.value * Decimal("0.999")
        assert sell_impact.value == expected

    def test_price_slippage_application(self, sample_price_btc: Price) -> None:
        """Тест применения проскальзывания."""
        slippage = Decimal("0.5")  # 0.5%
        
        # Покупка с проскальзыванием (цена выше)
        buy_price = sample_price_btc.apply_slippage("BUY", slippage)
        expected = sample_price_btc.value * Decimal("1.005")
        assert buy_price.value == expected
        
        # Продажа с проскальзыванием (цена ниже)
        sell_price = sample_price_btc.apply_slippage("SELL", slippage)
        expected = sample_price_btc.value * Decimal("0.995")
        assert sell_price.value == expected

    def test_price_support_resistance_levels(self, sample_price_btc: Price) -> None:
        """Тест расчета уровней поддержки и сопротивления."""
        # Уровень поддержки (ниже текущей цены)
        support = sample_price_btc.calculate_support_level(Decimal("5"))  # 5% ниже
        expected = sample_price_btc.value * Decimal("0.95")
        assert support.value == expected
        
        # Уровень сопротивления (выше текущей цены)
        resistance = sample_price_btc.calculate_resistance_level(Decimal("8"))  # 8% выше
        expected = sample_price_btc.value * Decimal("1.08")
        assert resistance.value == expected

    def test_price_volatility_range(self, sample_price_btc: Price) -> None:
        """Тест расчета диапазона волатильности."""
        volatility = Decimal("15")  # 15% волатильность
        
        lower_bound, upper_bound = sample_price_btc.calculate_volatility_range(volatility)
        
        expected_lower = sample_price_btc.value * Decimal("0.85")
        expected_upper = sample_price_btc.value * Decimal("1.15")
        
        assert lower_bound.value == expected_lower
        assert upper_bound.value == expected_upper

    def test_price_serialization(self, sample_price_btc: Price) -> None:
        """Тест сериализации цены."""
        # Тест to_dict
        data = sample_price_btc.to_dict()
        assert data["value"] == "45000.50"
        assert data["currency"] == "USD"
        
        # Тест from_dict
        restored = Price.from_dict(data)
        assert restored == sample_price_btc

    def test_price_from_dict_invalid(self) -> None:
        """Тест десериализации с невалидными данными."""
        with pytest.raises(ValidationError):
            Price.from_dict({"value": "invalid", "currency": "USD"})
        
        with pytest.raises(ValidationError):
            Price.from_dict({"value": "100", "currency": "INVALID"})

    def test_price_hash_consistency(self, usd_currency: Currency) -> None:
        """Тест консистентности хэширования."""
        price1 = Price(value=Decimal("100"), currency=usd_currency)
        price2 = Price(value=Decimal("100"), currency=usd_currency)
        
        assert hash(price1) == hash(price2)
        assert price1 == price2

    def test_price_immutability(self, sample_price_btc: Price) -> None:
        """Тест неизменяемости объекта Price."""
        original_value = sample_price_btc.value
        original_currency = sample_price_btc.currency
        
        # Все операции должны возвращать новые объекты
        doubled = sample_price_btc * Decimal("2")
        rounded = sample_price_btc.round_to_tick_size(Decimal("0.01"))
        
        # Оригинальный объект не должен измениться
        assert sample_price_btc.value == original_value
        assert sample_price_btc.currency == original_currency
        assert doubled is not sample_price_btc
        assert rounded is not sample_price_btc

    def test_price_trading_calculations(self, sample_price_btc: Price) -> None:
        """Тест торговых расчетов."""
        quantity = Decimal("0.5")  # 0.5 BTC
        
        # Расчет общей стоимости
        total_value = sample_price_btc.calculate_total_value(quantity)
        expected = sample_price_btc.value * quantity
        assert total_value.value == expected
        
        # Расчет количества по сумме
        investment = sample_price_btc.currency.create_money(Decimal("10000"))
        calculated_quantity = sample_price_btc.calculate_quantity_for_amount(investment)
        expected_qty = investment.amount / sample_price_btc.value
        assert calculated_quantity == expected_qty

    def test_price_stop_loss_take_profit(self, sample_price_btc: Price) -> None:
        """Тест расчета стоп-лосса и тейк-профита."""
        # Длинная позиция
        entry_price = sample_price_btc
        
        # Стоп-лосс на 5% ниже
        stop_loss = entry_price.calculate_stop_loss("LONG", Decimal("5"))
        expected_sl = entry_price.value * Decimal("0.95")
        assert stop_loss.value == expected_sl
        
        # Тейк-профит на 10% выше
        take_profit = entry_price.calculate_take_profit("LONG", Decimal("10"))
        expected_tp = entry_price.value * Decimal("1.10")
        assert take_profit.value == expected_tp
        
        # Короткая позиция
        # Стоп-лосс на 5% выше
        stop_loss_short = entry_price.calculate_stop_loss("SHORT", Decimal("5"))
        expected_sl_short = entry_price.value * Decimal("1.05")
        assert stop_loss_short.value == expected_sl_short
        
        # Тейк-профит на 10% ниже
        take_profit_short = entry_price.calculate_take_profit("SHORT", Decimal("10"))
        expected_tp_short = entry_price.value * Decimal("0.90")
        assert take_profit_short.value == expected_tp_short

    def test_price_technical_levels(self, sample_price_btc: Price) -> None:
        """Тест технических уровней."""
        # Психологический уровень (округление до тысяч)
        psychological = sample_price_btc.get_psychological_level(1000)
        assert psychological.value == Decimal("45000")  # Ближайший уровень 1000
        
        # Фибоначчи уровни
        high = Price(value=Decimal("50000"), currency=sample_price_btc.currency)
        low = Price(value=Decimal("40000"), currency=sample_price_btc.currency)
        
        fib_levels = sample_price_btc.calculate_fibonacci_levels(high, low)
        assert len(fib_levels) == 7  # 0%, 23.6%, 38.2%, 50%, 61.8%, 78.6%, 100%
        assert fib_levels[0].value == low.value  # 0% = low
        assert fib_levels[-1].value == high.value  # 100% = high

    def test_price_validation_comprehensive(self, usd_currency: Currency) -> None:
        """Комплексная валидация цены."""
        # Валидация типов
        with pytest.raises(ValidationError):
            Price(value="invalid", currency=usd_currency)  # type: ignore
        
        with pytest.raises(ValidationError):
            Price(value=None, currency=usd_currency)  # type: ignore
        
        with pytest.raises(ValidationError):
            Price(value=Decimal("100"), currency=None)  # type: ignore
        
        # Валидация значений
        with pytest.raises(ValidationError):
            Price(value=Decimal("inf"), currency=usd_currency)
        
        with pytest.raises(ValidationError):
            Price(value=Decimal("nan"), currency=usd_currency)

    def test_price_performance_critical_operations(self, sample_price_btc: Price) -> None:
        """Тест производительности критических операций."""
        # Множественные вычисления
        result = sample_price_btc
        for i in range(100):
            multiplier = Decimal(str(1 + (i * 0.01)))
            result = result * multiplier
        
        assert result.value > sample_price_btc.value
        assert result.currency == sample_price_btc.currency
        
        # Проверка создания множества объектов
        prices = [
            Price(value=Decimal(str(45000 + i)), currency=sample_price_btc.currency)
            for i in range(1000)
        ]
        assert len(prices) == 1000
        assert all(isinstance(p, Price) for p in prices)

    def test_price_edge_cases(self, usd_currency: Currency) -> None:
        """Тест граничных случаев."""
        # Очень маленькая цена
        tiny_price = Price(value=Decimal("0.00000001"), currency=usd_currency)
        assert tiny_price.is_positive()
        
        # Максимально допустимая цена
        max_price = Price(value=Price.MAX_VALUE, currency=usd_currency)
        assert max_price.is_positive()
        
        # Точность до 8 знаков
        precise_price = Price(value=Decimal("123.12345678"), currency=usd_currency)
        assert len(str(precise_price.value).split('.')[-1]) <= 8 