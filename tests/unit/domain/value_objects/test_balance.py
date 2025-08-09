"""
Unit тесты для balance.py.

Покрывает:
- Основной функционал Balance
- Валидацию данных
- Бизнес-логику операций с балансом
- Обработку ошибок
- Сериализацию и десериализацию
"""

import pytest
import dataclasses
from typing import Dict, Any
from unittest.mock import Mock, patch
from decimal import Decimal

from domain.value_objects.balance import Balance
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency


class TestBalance:
    """Тесты для Balance."""

    @pytest.fixture
    def sample_balance(self) -> Balance:
        """Тестовый баланс."""
        return Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("200.00"), currency=Currency.USD),
        )

    @pytest.fixture
    def btc_balance(self) -> Balance:
        """Баланс в BTC."""
        return Balance(
            currency=Currency.BTC,
            free=Money(amount=Decimal("1.5"), currency=Currency.BTC),
            used=Money(amount=Decimal("0.5"), currency=Currency.BTC),
        )

    def test_balance_creation(self, sample_balance):
        """Тест создания баланса."""
        assert sample_balance.currency == Currency.USD
        assert sample_balance.free.amount == Decimal("1000.00")
        assert sample_balance.used.amount == Decimal("200.00")
        assert sample_balance.free.currency == Currency.USD
        assert sample_balance.used.currency == Currency.USD

    def test_balance_total_property(self, sample_balance):
        """Тест свойства total."""
        total = sample_balance.total
        assert total.amount == Decimal("1200.00")
        assert total.currency == Currency.USD

    def test_balance_available_property(self, sample_balance):
        """Тест свойства available."""
        available = sample_balance.available
        assert available == sample_balance.free
        assert available.amount == Decimal("1000.00")

    def test_balance_validation_currency_mismatch_free(self):
        """Тест валидации несоответствия валюты для free."""
        with pytest.raises(ValueError, match="Валюта free должна совпадать с currency"):
            Balance(
                currency=Currency.USD,
                free=Money(amount=Decimal("1000.00"), currency=Currency.EUR),
                used=Money(amount=Decimal("200.00"), currency=Currency.USD),
            )

    def test_balance_validation_currency_mismatch_used(self):
        """Тест валидации несоответствия валюты для used."""
        with pytest.raises(ValueError, match="Валюта used должна совпадать с currency"):
            Balance(
                currency=Currency.USD,
                free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
                used=Money(amount=Decimal("200.00"), currency=Currency.EUR),
            )

    def test_balance_validation_negative_free(self):
        """Тест валидации отрицательного free баланса."""
        with pytest.raises(ValueError, match="Free баланс не может быть отрицательным"):
            Balance(
                currency=Currency.USD,
                free=Money(amount=Decimal("-100.00"), currency=Currency.USD),
                used=Money(amount=Decimal("200.00"), currency=Currency.USD),
            )

    def test_balance_validation_negative_used(self):
        """Тест валидации отрицательного used баланса."""
        with pytest.raises(ValueError, match="Used баланс не может быть отрицательным"):
            Balance(
                currency=Currency.USD,
                free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
                used=Money(amount=Decimal("-200.00"), currency=Currency.USD),
            )

    def test_balance_can_afford_sufficient_funds(self, sample_balance):
        """Тест проверки достаточности средств - достаточно."""
        amount = Money(amount=Decimal("500.00"), currency=Currency.USD)
        assert sample_balance.can_afford(amount) is True

    def test_balance_can_afford_insufficient_funds(self, sample_balance):
        """Тест проверки достаточности средств - недостаточно."""
        amount = Money(amount=Decimal("1500.00"), currency=Currency.USD)
        assert sample_balance.can_afford(amount) is False

    def test_balance_can_afford_exact_amount(self, sample_balance):
        """Тест проверки достаточности средств - точная сумма."""
        amount = Money(amount=Decimal("1000.00"), currency=Currency.USD)
        assert sample_balance.can_afford(amount) is True

    def test_balance_can_afford_different_currency(self, sample_balance):
        """Тест проверки достаточности средств - другая валюта."""
        amount = Money(amount=Decimal("500.00"), currency=Currency.EUR)
        assert sample_balance.can_afford(amount) is False

    def test_balance_reserve_success(self, sample_balance):
        """Тест успешного резервирования средств."""
        amount = Money(amount=Decimal("300.00"), currency=Currency.USD)
        new_balance = sample_balance.reserve(amount)

        assert new_balance.free.amount == Decimal("700.00")
        assert new_balance.used.amount == Decimal("500.00")
        assert new_balance.total.amount == Decimal("1200.00")
        assert new_balance.currency == Currency.USD

    def test_balance_reserve_insufficient_funds(self, sample_balance):
        """Тест резервирования при недостатке средств."""
        amount = Money(amount=Decimal("1500.00"), currency=Currency.USD)
        with pytest.raises(ValueError, match="Недостаточно средств"):
            sample_balance.reserve(amount)

    def test_balance_reserve_different_currency(self, sample_balance):
        """Тест резервирования в другой валюте."""
        amount = Money(amount=Decimal("300.00"), currency=Currency.EUR)
        with pytest.raises(ValueError, match="Недостаточно средств"):
            sample_balance.reserve(amount)

    def test_balance_release_success(self, sample_balance):
        """Тест успешного освобождения средств."""
        amount = Money(amount=Decimal("100.00"), currency=Currency.USD)
        new_balance = sample_balance.release(amount)

        assert new_balance.free.amount == Decimal("1100.00")
        assert new_balance.used.amount == Decimal("100.00")
        assert new_balance.total.amount == Decimal("1200.00")
        assert new_balance.currency == Currency.USD

    def test_balance_release_different_currency(self, sample_balance):
        """Тест освобождения средств в другой валюте."""
        amount = Money(amount=Decimal("100.00"), currency=Currency.EUR)
        with pytest.raises(ValueError, match="Валюта должна совпадать"):
            sample_balance.release(amount)

    def test_balance_release_too_much(self, sample_balance):
        """Тест освобождения больше чем зарезервировано."""
        amount = Money(amount=Decimal("300.00"), currency=Currency.USD)
        with pytest.raises(ValueError, match="Нельзя освободить больше чем зарезервировано"):
            sample_balance.release(amount)

    def test_balance_add_success(self, sample_balance):
        """Тест успешного добавления средств."""
        amount = Money(amount=Decimal("500.00"), currency=Currency.USD)
        new_balance = sample_balance.add(amount)

        assert new_balance.free.amount == Decimal("1500.00")
        assert new_balance.used.amount == Decimal("200.00")
        assert new_balance.total.amount == Decimal("1700.00")
        assert new_balance.currency == Currency.USD

    def test_balance_add_different_currency(self, sample_balance):
        """Тест добавления средств в другой валюте."""
        amount = Money(amount=Decimal("500.00"), currency=Currency.EUR)
        with pytest.raises(ValueError, match="Валюта должна совпадать"):
            sample_balance.add(amount)

    def test_balance_subtract_success(self, sample_balance):
        """Тест успешного вычитания средств."""
        amount = Money(amount=Decimal("300.00"), currency=Currency.USD)
        new_balance = sample_balance.subtract(amount)

        assert new_balance.free.amount == Decimal("700.00")
        assert new_balance.used.amount == Decimal("200.00")
        assert new_balance.total.amount == Decimal("900.00")
        assert new_balance.currency == Currency.USD

    def test_balance_subtract_different_currency(self, sample_balance):
        """Тест вычитания средств в другой валюте."""
        amount = Money(amount=Decimal("300.00"), currency=Currency.EUR)
        with pytest.raises(ValueError, match="Валюта должна совпадать"):
            sample_balance.subtract(amount)

    def test_balance_subtract_insufficient_funds(self, sample_balance):
        """Тест вычитания при недостатке средств."""
        amount = Money(amount=Decimal("1500.00"), currency=Currency.USD)
        with pytest.raises(ValueError, match="Недостаточно средств"):
            sample_balance.subtract(amount)

    def test_balance_subtract_exact_amount(self, sample_balance):
        """Тест вычитания точной суммы."""
        amount = Money(amount=Decimal("1000.00"), currency=Currency.USD)
        new_balance = sample_balance.subtract(amount)

        assert new_balance.free.amount == Decimal("0.00")
        assert new_balance.used.amount == Decimal("200.00")
        assert new_balance.total.amount == Decimal("200.00")

    def test_balance_to_dict(self, sample_balance):
        """Тест сериализации в словарь."""
        result = sample_balance.to_dict()

        assert result["currency"] == "USD"
        assert result["free"]["amount"] == "1000.00"
        assert result["free"]["currency"] == "USD"
        assert result["used"]["amount"] == "200.00"
        assert result["used"]["currency"] == "USD"
        assert result["total"]["amount"] == "1200.00"
        assert result["total"]["currency"] == "USD"

    def test_balance_from_dict(self, sample_balance):
        """Тест десериализации из словаря."""
        data = {
            "currency": "USD",
            "free": {"amount": "1000.00", "currency": "USD"},
            "used": {"amount": "200.00", "currency": "USD"},
        }

        balance = Balance.from_dict(data)

        assert balance.currency == Currency.USD
        assert balance.free.amount == Decimal("1000.00")
        assert balance.used.amount == Decimal("200.00")
        assert balance.total.amount == Decimal("1200.00")

    def test_balance_str_representation(self, sample_balance):
        """Тест строкового представления."""
        result = str(sample_balance)
        expected = "Balance(USD: free=Money(1000.00, USD), used=Money(200.00, USD), total=Money(1200.00, USD))"
        assert result == expected

    def test_balance_repr_representation(self, sample_balance):
        """Тест repr представления."""
        result = repr(sample_balance)
        expected = "Balance(USD: free=Money(1000.00, USD), used=Money(200.00, USD), total=Money(1200.00, USD))"
        assert result == expected


class TestBalanceOperations:
    """Тесты операций с балансом."""

    def test_balance_zero_values(self):
        """Тест баланса с нулевыми значениями."""
        balance = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("0.00"), currency=Currency.USD),
            used=Money(amount=Decimal("0.00"), currency=Currency.USD),
        )

        assert balance.total.amount == Decimal("0.00")
        assert balance.can_afford(Money(amount=Decimal("0.01"), currency=Currency.USD)) is False
        assert balance.can_afford(Money(amount=Decimal("0.00"), currency=Currency.USD)) is True

    def test_balance_large_numbers(self):
        """Тест баланса с большими числами."""
        balance = Balance(
            currency=Currency.BTC,
            free=Money(amount=Decimal("1000000.12345678"), currency=Currency.BTC),
            used=Money(amount=Decimal("500000.87654321"), currency=Currency.BTC),
        )

        assert balance.total.amount == Decimal("1500001.00000000")
        assert balance.can_afford(Money(amount=Decimal("1000000.12345678"), currency=Currency.BTC)) is True
        assert balance.can_afford(Money(amount=Decimal("1000000.12345679"), currency=Currency.BTC)) is False

    def test_balance_precision_handling(self):
        """Тест обработки точности."""
        balance = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("100.123456"), currency=Currency.USD),
            used=Money(amount=Decimal("50.654321"), currency=Currency.USD),
        )

        assert balance.total.amount == Decimal("150.777777")

        # Добавление с высокой точностью
        new_balance = balance.add(Money(amount=Decimal("0.000001"), currency=Currency.USD))
        assert new_balance.free.amount == Decimal("100.123457")

    def test_balance_immutability(self):
        """Тест неизменяемости баланса."""
        balance = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("200.00"), currency=Currency.USD),
        )

        # Попытка изменения должна вызвать ошибку
        with pytest.raises(dataclasses.FrozenInstanceError):
            balance.currency = Currency.EUR

    def test_balance_equality(self):
        """Тест равенства балансов."""
        balance1 = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("200.00"), currency=Currency.USD),
        )

        balance2 = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("200.00"), currency=Currency.USD),
        )

        balance3 = Balance(
            currency=Currency.EUR,
            free=Money(amount=Decimal("1000.00"), currency=Currency.EUR),
            used=Money(amount=Decimal("200.00"), currency=Currency.EUR),
        )

        assert balance1 == balance2
        assert balance1 != balance3
        assert balance1 != "not a balance"

    def test_balance_hash_consistency(self):
        """Тест консистентности хеширования."""
        balance1 = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("200.00"), currency=Currency.USD),
        )

        balance2 = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("200.00"), currency=Currency.USD),
        )

        assert hash(balance1) == hash(balance2)


class TestBalanceEdgeCases:
    """Тесты граничных случаев для баланса."""

    def test_balance_maximum_values(self):
        """Тест максимальных значений."""
        max_amount = Decimal("999999999999999.99999999")
        balance = Balance(
            currency=Currency.USD,
            free=Money(amount=max_amount, currency=Currency.USD),
            used=Money(amount=Decimal("0.00"), currency=Currency.USD),
        )

        assert balance.total.amount == max_amount
        assert balance.can_afford(Money(amount=max_amount, currency=Currency.USD)) is True

    def test_balance_minimum_values(self):
        """Тест минимальных значений."""
        min_amount = Decimal("0.00000001")
        balance = Balance(
            currency=Currency.USD,
            free=Money(amount=min_amount, currency=Currency.USD),
            used=Money(amount=Decimal("0.00"), currency=Currency.USD),
        )

        assert balance.total.amount == min_amount
        assert balance.can_afford(Money(amount=min_amount, currency=Currency.USD)) is True
        assert balance.can_afford(Money(amount=Decimal("0.00000002"), currency=Currency.USD)) is False

    def test_balance_currency_specific_behavior(self):
        """Тест поведения для разных валют."""
        # BTC с 8 знаками после запятой
        btc_balance = Balance(
            currency=Currency.BTC,
            free=Money(amount=Decimal("1.12345678"), currency=Currency.BTC),
            used=Money(amount=Decimal("0.87654321"), currency=Currency.BTC),
        )

        # USD с 2 знаками после запятой
        usd_balance = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("1000.12"), currency=Currency.USD),
            used=Money(amount=Decimal("200.34"), currency=Currency.USD),
        )

        assert btc_balance.total.amount == Decimal("2.00000000")
        assert usd_balance.total.amount == Decimal("1200.46")

    def test_balance_serialization_roundtrip(self):
        """Тест сериализации и десериализации."""
        original_balance = Balance(
            currency=Currency.BTC,
            free=Money(amount=Decimal("1.5"), currency=Currency.BTC),
            used=Money(amount=Decimal("0.5"), currency=Currency.BTC),
        )

        # Сериализация
        data = original_balance.to_dict()

        # Десериализация
        restored_balance = Balance.from_dict(data)

        # Проверка равенства
        assert restored_balance == original_balance
        assert restored_balance.currency == original_balance.currency
        assert restored_balance.free.amount == original_balance.free.amount
        assert restored_balance.used.amount == original_balance.used.amount
        assert restored_balance.total.amount == original_balance.total.amount

    def test_balance_performance(self):
        """Тест производительности операций с балансом."""
        import time

        balance = Balance(
            currency=Currency.USD,
            free=Money(amount=Decimal("10000.00"), currency=Currency.USD),
            used=Money(amount=Decimal("2000.00"), currency=Currency.USD),
        )

        # Тест скорости операций
        start_time = time.time()
        for _ in range(1000):
            balance.can_afford(Money(amount=Decimal("5000.00"), currency=Currency.USD))
        end_time = time.time()

        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0  # Менее 1 секунды для 1000 операций
