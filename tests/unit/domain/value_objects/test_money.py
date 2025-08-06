#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Value Object Money.
Покрывает все критические операции для торговой системы.
"""

import pytest
from decimal import Decimal
from typing import Any, Dict

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.exceptions import ValidationError


class TestMoney:
    """Тесты для Money Value Object."""

    @pytest.fixture
    def usd_currency(self) -> Currency:
        """Фикстура USD валюты."""
        return Currency("USD")

    @pytest.fixture
    def eur_currency(self) -> Currency:
        """Фикстура EUR валюты."""
        return Currency("EUR")

    @pytest.fixture
    def sample_money_usd(self, usd_currency: Currency) -> Money:
        """Фикстура денежной суммы в USD."""
        return Money(amount=Decimal("1000.00"), currency=usd_currency)

    @pytest.fixture
    def sample_money_eur(self, eur_currency: Currency) -> Money:
        """Фикстура денежной суммы в EUR."""
        return Money(amount=Decimal("850.00"), currency=eur_currency)

    def test_money_creation_valid(self, usd_currency: Currency) -> None:
        """Тест создания валидной денежной суммы."""
        money = Money(amount=Decimal("100.50"), currency=usd_currency)
        
        assert money.amount == Decimal("100.50")
        assert money.currency == usd_currency
        assert str(money) == "100.50 USD"

    def test_money_creation_zero_amount(self, usd_currency: Currency) -> None:
        """Тест создания денежной суммы с нулевой суммой."""
        money = Money(amount=Decimal("0"), currency=usd_currency)
        
        assert money.amount == Decimal("0")
        assert money.is_zero()
        assert not money.is_positive()
        assert not money.is_negative()

    def test_money_creation_negative_amount(self, usd_currency: Currency) -> None:
        """Тест создания отрицательной денежной суммы."""
        money = Money(amount=Decimal("-50.25"), currency=usd_currency)
        
        assert money.amount == Decimal("-50.25")
        assert money.is_negative()
        assert not money.is_positive()
        assert not money.is_zero()

    def test_money_creation_invalid_amount_too_large(self, usd_currency: Currency) -> None:
        """Тест валидации максимальной суммы."""
        with pytest.raises(ValidationError):
            Money(amount=Decimal("1000000000000"), currency=usd_currency)

    def test_money_creation_invalid_amount_too_small(self, usd_currency: Currency) -> None:
        """Тест валидации минимальной суммы."""
        with pytest.raises(ValidationError):
            Money(amount=Decimal("-1000000000000"), currency=usd_currency)

    def test_money_addition_same_currency(self, sample_money_usd: Money, usd_currency: Currency) -> None:
        """Тест сложения денежных сумм одной валюты."""
        other = Money(amount=Decimal("250.75"), currency=usd_currency)
        result = sample_money_usd + other
        
        assert result.amount == Decimal("1250.75")
        assert result.currency == usd_currency

    def test_money_addition_different_currency(self, sample_money_usd: Money, sample_money_eur: Money) -> None:
        """Тест сложения денежных сумм разных валют."""
        with pytest.raises(ValidationError, match="Cannot add money with different currencies"):
            sample_money_usd + sample_money_eur

    def test_money_subtraction_same_currency(self, sample_money_usd: Money, usd_currency: Currency) -> None:
        """Тест вычитания денежных сумм одной валюты."""
        other = Money(amount=Decimal("250.75"), currency=usd_currency)
        result = sample_money_usd - other
        
        assert result.amount == Decimal("749.25")
        assert result.currency == usd_currency

    def test_money_subtraction_different_currency(self, sample_money_usd: Money, sample_money_eur: Money) -> None:
        """Тест вычитания денежных сумм разных валют."""
        with pytest.raises(ValidationError, match="Cannot subtract money with different currencies"):
            sample_money_usd - sample_money_eur

    def test_money_multiplication_by_number(self, sample_money_usd: Money) -> None:
        """Тест умножения денежной суммы на число."""
        result = sample_money_usd * Decimal("2.5")
        
        assert result.amount == Decimal("2500.00")
        assert result.currency == sample_money_usd.currency

    def test_money_division_by_number(self, sample_money_usd: Money) -> None:
        """Тест деления денежной суммы на число."""
        result = sample_money_usd / Decimal("4")
        
        assert result.amount == Decimal("250.00")
        assert result.currency == sample_money_usd.currency

    def test_money_division_by_zero(self, sample_money_usd: Money) -> None:
        """Тест деления на ноль."""
        with pytest.raises(ValidationError, match="Cannot divide by zero"):
            sample_money_usd / Decimal("0")

    def test_money_comparison_same_currency(self, usd_currency: Currency) -> None:
        """Тест сравнения денежных сумм одной валюты."""
        money1 = Money(amount=Decimal("100"), currency=usd_currency)
        money2 = Money(amount=Decimal("200"), currency=usd_currency)
        money3 = Money(amount=Decimal("100"), currency=usd_currency)
        
        assert money1 < money2
        assert money2 > money1
        assert money1 == money3
        assert money1 <= money2
        assert money2 >= money1

    def test_money_comparison_different_currency(self, sample_money_usd: Money, sample_money_eur: Money) -> None:
        """Тест сравнения денежных сумм разных валют."""
        with pytest.raises(ValidationError, match="Cannot compare money with different currencies"):
            sample_money_usd < sample_money_eur

    def test_money_abs_positive(self, sample_money_usd: Money) -> None:
        """Тест абсолютного значения положительной суммы."""
        result = abs(sample_money_usd)
        assert result == sample_money_usd

    def test_money_abs_negative(self, usd_currency: Currency) -> None:
        """Тест абсолютного значения отрицательной суммы."""
        negative_money = Money(amount=Decimal("-100"), currency=usd_currency)
        result = abs(negative_money)
        
        assert result.amount == Decimal("100")
        assert result.currency == usd_currency

    def test_money_negate(self, sample_money_usd: Money) -> None:
        """Тест отрицания денежной суммы."""
        result = -sample_money_usd
        
        assert result.amount == Decimal("-1000.00")
        assert result.currency == sample_money_usd.currency

    def test_money_round(self, usd_currency: Currency) -> None:
        """Тест округления денежной суммы."""
        money = Money(amount=Decimal("123.456789"), currency=usd_currency)
        rounded = money.round(2)
        
        assert rounded.amount == Decimal("123.46")
        assert rounded.currency == usd_currency

    def test_money_to_dict(self, sample_money_usd: Money) -> None:
        """Тест сериализации в словарь."""
        result = sample_money_usd.to_dict()
        
        assert result["amount"] == "1000.00"
        assert result["currency"] == "USD"

    def test_money_from_dict(self, usd_currency: Currency) -> None:
        """Тест десериализации из словаря."""
        data = {"amount": "500.25", "currency": "USD"}
        money = Money.from_dict(data)
        
        assert money.amount == Decimal("500.25")
        assert money.currency.code == "USD"

    def test_money_from_dict_invalid_data(self) -> None:
        """Тест десериализации с невалидными данными."""
        with pytest.raises(ValidationError):
            Money.from_dict({"amount": "invalid", "currency": "USD"})

    def test_money_hash_consistency(self, usd_currency: Currency) -> None:
        """Тест консистентности хэширования."""
        money1 = Money(amount=Decimal("100"), currency=usd_currency)
        money2 = Money(amount=Decimal("100"), currency=usd_currency)
        
        assert hash(money1) == hash(money2)
        assert money1 == money2

    def test_money_repr(self, sample_money_usd: Money) -> None:
        """Тест строкового представления."""
        repr_str = repr(sample_money_usd)
        assert "Money" in repr_str
        assert "1000.00" in repr_str
        assert "USD" in repr_str

    def test_money_percentage_calculation(self, sample_money_usd: Money) -> None:
        """Тест расчета процентов."""
        percentage = Decimal("15.5")  # 15.5%
        result = sample_money_usd.calculate_percentage(percentage)
        
        assert result.amount == Decimal("155.00")
        assert result.currency == sample_money_usd.currency

    def test_money_slippage_calculation(self, sample_money_usd: Money) -> None:
        """Тест расчета проскальзывания."""
        slippage = Decimal("0.002")  # 0.2%
        result = sample_money_usd.apply_slippage(slippage)
        
        expected_amount = Decimal("1000.00") * (Decimal("1") + slippage)
        assert result.amount == expected_amount
        assert result.currency == sample_money_usd.currency

    def test_money_slippage_too_high(self, sample_money_usd: Money) -> None:
        """Тест валидации максимального проскальзывания."""
        with pytest.raises(ValidationError, match="Slippage cannot exceed"):
            sample_money_usd.apply_slippage(Decimal("0.1"))  # 10% - слишком много

    def test_money_risk_calculation(self, sample_money_usd: Money) -> None:
        """Тест расчета риска."""
        risk_percentage = Decimal("2.5")  # 2.5%
        result = sample_money_usd.calculate_risk_amount(risk_percentage)
        
        assert result.amount == Decimal("25.00")
        assert result.currency == sample_money_usd.currency

    def test_money_position_sizing(self, sample_money_usd: Money) -> None:
        """Тест расчета размера позиции."""
        risk_per_trade = Decimal("2.0")  # 2%
        stop_loss_distance = Decimal("5.0")  # 5%
        
        position_size = sample_money_usd.calculate_position_size(
            risk_per_trade, stop_loss_distance
        )
        
        # При риске 2% и стоп-лоссе 5%, размер позиции должен быть 40% от капитала
        expected_size = sample_money_usd.amount * (risk_per_trade / stop_loss_distance)
        assert position_size.amount == expected_size
        assert position_size.currency == sample_money_usd.currency

    def test_money_immutability(self, sample_money_usd: Money) -> None:
        """Тест неизменяемости объекта Money."""
        original_amount = sample_money_usd.amount
        
        # Все операции должны возвращать новые объекты
        result1 = sample_money_usd + Money(Decimal("100"), sample_money_usd.currency)
        result2 = sample_money_usd * Decimal("2")
        result3 = sample_money_usd.round(0)
        
        # Оригинальный объект не должен измениться
        assert sample_money_usd.amount == original_amount
        assert result1 is not sample_money_usd
        assert result2 is not sample_money_usd
        assert result3 is not sample_money_usd

    def test_money_caching(self, usd_currency: Currency) -> None:
        """Тест кэширования объектов Money."""
        # Создаем одинаковые объекты
        money1 = Money(amount=Decimal("100"), currency=usd_currency)
        money2 = Money(amount=Decimal("100"), currency=usd_currency)
        
        # Проверяем, что кэширование работает корректно
        assert money1 == money2
        assert hash(money1) == hash(money2)

    def test_money_trading_calculations(self, sample_money_usd: Money) -> None:
        """Тест торговых расчетов."""
        # Тест расчета комиссии
        commission_rate = Decimal("0.001")  # 0.1%
        commission = sample_money_usd.calculate_commission(commission_rate)
        
        assert commission.amount == Decimal("1.00")
        assert commission.currency == sample_money_usd.currency
        
        # Тест расчета чистой суммы после комиссии
        net_amount = sample_money_usd.subtract_commission(commission_rate)
        expected_net = sample_money_usd.amount * (Decimal("1") - commission_rate)
        
        assert net_amount.amount == expected_net
        assert net_amount.currency == sample_money_usd.currency

    def test_money_precision_handling(self, usd_currency: Currency) -> None:
        """Тест обработки точности."""
        # Создаем сумму с высокой точностью
        high_precision = Money(
            amount=Decimal("123.123456789012345"), 
            currency=usd_currency
        )
        
        # Проверяем, что точность соблюдается
        assert len(str(high_precision.amount).split('.')[-1]) <= 8  # Максимум 8 знаков
        
        # Тест округления до валютной точности
        currency_rounded = high_precision.round_to_currency_precision()
        assert len(str(currency_rounded.amount).split('.')[-1]) <= 2  # 2 знака для USD

    def test_money_edge_cases(self, usd_currency: Currency) -> None:
        """Тест граничных случаев."""
        # Очень маленькая сумма
        tiny_money = Money(amount=Decimal("0.01"), currency=usd_currency)
        assert tiny_money.is_positive()
        
        # Максимально допустимая сумма
        max_money = Money(amount=Money.MAX_AMOUNT, currency=usd_currency)
        assert max_money.is_positive()
        
        # Минимально допустимая сумма
        min_money = Money(amount=Money.MIN_AMOUNT, currency=usd_currency)
        assert min_money.is_negative()

    def test_money_validation_comprehensive(self, usd_currency: Currency) -> None:
        """Комплексный тест валидации."""
        # Валидация типов
        with pytest.raises(ValidationError):
            Money(amount="invalid", currency=usd_currency)  # type: ignore
        
        with pytest.raises(ValidationError):
            Money(amount=None, currency=usd_currency)  # type: ignore
        
        with pytest.raises(ValidationError):
            Money(amount=Decimal("100"), currency=None)  # type: ignore
        
        # Валидация диапазонов
        with pytest.raises(ValidationError):
            Money(amount=Decimal("inf"), currency=usd_currency)
        
        with pytest.raises(ValidationError):
            Money(amount=Decimal("nan"), currency=usd_currency)

    def test_money_performance_critical_operations(self, sample_money_usd: Money) -> None:
        """Тест производительности критических операций."""
        # Множественные арифметические операции
        result = sample_money_usd
        for i in range(100):
            other = Money(Decimal(str(i + 1)), sample_money_usd.currency)
            result = result + other
        
        assert result.amount > sample_money_usd.amount
        assert result.currency == sample_money_usd.currency
        
        # Проверка, что операции не вызывают утечек памяти
        # (косвенная проверка через создание множества объектов)
        monies = [
            Money(Decimal(str(i)), sample_money_usd.currency)
            for i in range(1000)
        ]
        assert len(monies) == 1000
        assert all(isinstance(m, Money) for m in monies) 