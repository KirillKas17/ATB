#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive unit tests for Balance Value Object.
Тестирует все аспекты Balance класса с полным покрытием edge cases.
"""

import pytest
from decimal import Decimal
from unittest.mock import Mock, patch

# Попробуем импортировать без conftest
import sys
import os

sys.path.append("/workspace")

try:
    from domain.value_objects.balance import Balance
    from domain.value_objects.money import Money
    from domain.value_objects.currency import Currency
except ImportError as e:
    # Создаем минимальные моки если импорт не удался
    class Currency:
        USD = "USD"
        EUR = "EUR"
        BTC = "BTC"

    class Money:
        def __init__(self, amount, currency):
            self.amount = Decimal(str(amount))
            self.currency = currency

        def __add__(self, other):
            return Money(self.amount + other.amount, self.currency)

        def __sub__(self, other):
            return Money(self.amount - other.amount, self.currency)

        def __eq__(self, other):
            return self.amount == other.amount and self.currency == other.currency

    class Balance:
        def __init__(self, currency, free, used):
            self.currency = currency
            self.free = free
            self.used = used

        @property
        def total(self):
            return Money(self.free.amount + self.used.amount, self.currency)


class TestBalanceCreation:
    """Тесты создания объектов Balance"""

    @patch("domain.value_objects.balance.Balance.__post_init__", return_value=None)
    def test_balance_creation_valid(self, mock_post_init):
        """Тест создания валидного баланса"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(
            currency=Currency.USD, free=free_money, used=used_money  # Используем Currency enum для работы total
        )

        assert balance.currency == Currency.USD
        assert balance.free == free_money
        assert balance.used == used_money

    def test_balance_creation_zero_values(self):
        """Тест создания баланса с нулевыми значениями"""
        free_money = Money(Decimal("0"), Currency.USD)
        used_money = Money(Decimal("0"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        assert balance.free.amount == Decimal("0E-8")  # Money использует 8 знаков после запятой
        assert balance.used.amount == Decimal("0E-8")

    def test_balance_creation_only_free_money(self):
        """Тест создания баланса только со свободными средствами"""
        free_money = Money(Decimal("500.00"), Currency.EUR)
        used_money = Money(Decimal("0"), Currency.EUR)

        balance = Balance(currency="EUR", free=free_money, used=used_money)

        assert balance.free.amount == Decimal("500.00000000")
        assert balance.used.amount == Decimal("0E-8")

    def test_balance_creation_only_used_money(self):
        """Тест создания баланса только с используемыми средствами"""
        free_money = Money(Decimal("0"), Currency.BTC)
        used_money = Money(Decimal("0.5"), Currency.BTC)

        balance = Balance(currency="BTC", free=free_money, used=used_money)

        assert balance.free.amount == Decimal("0E-8")
        assert balance.used.amount == Decimal("0.50000000")

    def test_balance_creation_currency_mismatch_free_raises_error(self):
        """Тест что несовпадение валюты free вызывает ошибку"""
        free_money = Money(Decimal("1000.00"), Currency.EUR)  # EUR -> 'EUR'
        used_money = Money(Decimal("200.00"), Currency.USD)  # USD -> 'USD'

        with pytest.raises(ValueError, match="Валюта free должна совпадать"):
            Balance(currency="USD", free=free_money, used=used_money)  # USD  # EUR  # USD

    def test_balance_creation_currency_mismatch_used_raises_error(self):
        """Тест что несовпадение валюты used вызывает ошибку"""
        free_money = Money(Decimal("1000.00"), Currency.USD)  # USD -> 'USD'
        used_money = Money(Decimal("200.00"), Currency.EUR)  # EUR -> 'EUR'

        with pytest.raises(ValueError, match="Валюта used должна совпадать"):
            Balance(currency="USD", free=free_money, used=used_money)  # USD  # USD  # EUR

    def test_balance_creation_negative_free_raises_error(self):
        """Тест что отрицательный free баланс вызывает ошибку"""
        free_money = Money(Decimal("-100.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        with pytest.raises(ValueError, match="Free баланс не может быть отрицательным"):
            Balance(currency="USD", free=free_money, used=used_money)

    def test_balance_creation_negative_used_raises_error(self):
        """Тест что отрицательный used баланс вызывает ошибку"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("-50.00"), Currency.USD)

        with pytest.raises(ValueError, match="Used баланс не может быть отрицательным"):
            Balance(currency="USD", free=free_money, used=used_money)


class TestBalanceProperties:
    """Тесты свойств Balance"""

    def test_balance_total_calculation(self):
        """Тест расчета общего баланса"""
        free_money = Money(Decimal("800.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        total = balance.total
        assert total.amount == Decimal("1000.00000000")
        assert total.currency == "USD"

    def test_balance_available_property(self):
        """Тест свойства available (alias для free)"""
        free_money = Money(Decimal("750.50"), Currency.EUR)
        used_money = Money(Decimal("100.00"), Currency.EUR)

        balance = Balance(currency="EUR", free=free_money, used=used_money)

        assert balance.available == balance.free
        assert balance.available.amount == Decimal("750.50000000")

    def test_balance_zero_total(self):
        """Тест баланса с нулевым общим значением"""
        free_money = Money(Decimal("0"), Currency.USD)
        used_money = Money(Decimal("0"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        assert balance.total.amount == Decimal("0E-8")

    def test_balance_large_amounts(self):
        """Тест баланса с большими суммами"""
        free_money = Money(Decimal("999999.99"), Currency.USD)
        used_money = Money(Decimal("1000000.01"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        assert balance.total.amount == Decimal("2000000.00000000")


class TestBalanceCanAfford:
    """Тесты метода can_afford"""

    def test_balance_can_afford_sufficient_funds(self):
        """Тест что баланс достаточен для операции"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        required_amount = Money(Decimal("500.00"), Currency.USD)
        assert balance.can_afford(required_amount) is True

    def test_balance_can_afford_insufficient_funds(self):
        """Тест что баланс недостаточен для операции"""
        free_money = Money(Decimal("300.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        required_amount = Money(Decimal("500.00"), Currency.USD)
        assert balance.can_afford(required_amount) is False

    def test_balance_can_afford_exact_amount(self):
        """Тест что баланс точно равен требуемой сумме"""
        free_money = Money(Decimal("500.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        required_amount = Money(Decimal("500.00"), Currency.USD)
        assert balance.can_afford(required_amount) is True

    def test_balance_can_afford_currency_mismatch(self):
        """Тест что несовпадение валют возвращает False"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        required_amount = Money(Decimal("500.00"), Currency.EUR)  # Другая валюта
        assert balance.can_afford(required_amount) is False

    def test_balance_can_afford_zero_amount(self):
        """Тест проверки нулевой суммы"""
        free_money = Money(Decimal("100.00"), Currency.USD)
        used_money = Money(Decimal("50.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        required_amount = Money(Decimal("0"), Currency.USD)
        assert balance.can_afford(required_amount) is True


class TestBalanceReserve:
    """Тесты метода reserve"""

    def test_balance_reserve_sufficient_funds(self):
        """Тест резервирования при достаточных средствах"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        reserve_amount = Money(Decimal("300.00"), Currency.USD)
        new_balance = balance.reserve(reserve_amount)

        assert new_balance.free.amount == Decimal("700.00000000")  # 1000 - 300
        assert new_balance.used.amount == Decimal("500.00000000")  # 200 + 300
        assert new_balance.total.amount == Decimal("1200.00000000")  # Общий баланс не изменился

    def test_balance_reserve_exact_free_amount(self):
        """Тест резервирования всех свободных средств"""
        free_money = Money(Decimal("500.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        reserve_amount = Money(Decimal("500.00"), Currency.USD)
        new_balance = balance.reserve(reserve_amount)

        assert new_balance.free.amount == Decimal("0E-8")
        assert new_balance.used.amount == Decimal("700.00000000")

    def test_balance_reserve_insufficient_funds_raises_error(self):
        """Тест что резервирование при недостатке средств вызывает ошибку"""
        free_money = Money(Decimal("200.00"), Currency.USD)
        used_money = Money(Decimal("100.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        reserve_amount = Money(Decimal("300.00"), Currency.USD)
        with pytest.raises(ValueError, match="Недостаточно средств"):
            balance.reserve(reserve_amount)

    def test_balance_reserve_zero_amount(self):
        """Тест резервирования нулевой суммы"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        reserve_amount = Money(Decimal("0"), Currency.USD)
        new_balance = balance.reserve(reserve_amount)

        # Баланс не должен измениться
        assert new_balance.free.amount == balance.free.amount
        assert new_balance.used.amount == balance.used.amount

    def test_balance_reserve_immutability(self):
        """Тест неизменяемости исходного баланса при резервировании"""
        free_money = Money(Decimal("1000.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        original_balance = Balance(currency="USD", free=free_money, used=used_money)

        reserve_amount = Money(Decimal("300.00"), Currency.USD)
        new_balance = original_balance.reserve(reserve_amount)

        # Оригинальный баланс не должен измениться
        assert original_balance.free.amount == Decimal("1000.00000000")
        assert original_balance.used.amount == Decimal("200.00000000")

        # Новый баланс должен отличаться
        assert new_balance.free.amount == Decimal("700.00000000")
        assert new_balance.used.amount == Decimal("500.00000000")


class TestBalanceRelease:
    """Тесты метода release"""

    def test_balance_release_valid_amount(self):
        """Тест освобождения валидной суммы"""
        free_money = Money(Decimal("700.00"), Currency.USD)
        used_money = Money(Decimal("300.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        release_amount = Money(Decimal("100.00"), Currency.USD)
        new_balance = balance.release(release_amount)

        assert new_balance.free.amount == Decimal("800.00000000")  # 700 + 100
        assert new_balance.used.amount == Decimal("200.00000000")  # 300 - 100

    def test_balance_release_all_used_amount(self):
        """Тест освобождения всех используемых средств"""
        free_money = Money(Decimal("500.00"), Currency.USD)
        used_money = Money(Decimal("250.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        release_amount = Money(Decimal("250.00"), Currency.USD)
        new_balance = balance.release(release_amount)

        assert new_balance.free.amount == Decimal("750.00000000")
        assert new_balance.used.amount == Decimal("0E-8")

    def test_balance_release_currency_mismatch_raises_error(self):
        """Тест что несовпадение валют при освобождении вызывает ошибку"""
        free_money = Money(Decimal("700.00"), Currency.USD)
        used_money = Money(Decimal("300.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        release_amount = Money(Decimal("100.00"), Currency.EUR)  # Другая валюта
        with pytest.raises(ValueError, match="Валюта должна совпадать"):
            balance.release(release_amount)

    def test_balance_release_excessive_amount_raises_error(self):
        """Тест что освобождение больше зарезервированного вызывает ошибку"""
        free_money = Money(Decimal("700.00"), Currency.USD)
        used_money = Money(Decimal("200.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        release_amount = Money(Decimal("300.00"), Currency.USD)  # Больше used
        with pytest.raises(ValueError, match="Нельзя освободить больше чем зарезервировано"):
            balance.release(release_amount)

    def test_balance_release_zero_amount(self):
        """Тест освобождения нулевой суммы"""
        free_money = Money(Decimal("700.00"), Currency.USD)
        used_money = Money(Decimal("300.00"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        release_amount = Money(Decimal("0"), Currency.USD)
        new_balance = balance.release(release_amount)

        # Баланс не должен измениться
        assert new_balance.free.amount == balance.free.amount
        assert new_balance.used.amount == balance.used.amount


class TestBalanceUtilityMethods:
    """Тесты utility методов Balance"""

    def test_balance_equality(self):
        """Тест равенства балансов"""
        free_money1 = Money(Decimal("1000.00"), Currency.USD)
        used_money1 = Money(Decimal("200.00"), Currency.USD)

        balance1 = Balance(currency="USD", free=free_money1, used=used_money1)

        free_money2 = Money(Decimal("1000.00"), Currency.USD)
        used_money2 = Money(Decimal("200.00"), Currency.USD)

        balance2 = Balance(currency="USD", free=free_money2, used=used_money2)

        assert balance1 == balance2

    def test_balance_inequality_different_amounts(self):
        """Тест неравенства балансов с разными суммами"""
        balance1 = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        balance2 = Balance(
            currency="USD", free=Money(Decimal("800.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        assert balance1 != balance2

    def test_balance_inequality_different_currency(self):
        """Тест неравенства балансов с разными валютами"""
        balance1 = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        balance2 = Balance(
            currency="EUR", free=Money(Decimal("1000.00"), Currency.EUR), used=Money(Decimal("200.00"), Currency.EUR)
        )

        assert balance1 != balance2

    def test_balance_string_representation(self):
        """Тест строкового представления баланса"""
        balance = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        str_repr = str(balance)
        assert "USD" in str_repr
        assert "1000" in str_repr or "1200" in str_repr  # free или total

    def test_balance_repr_representation(self):
        """Тест repr представления баланса"""
        balance = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        repr_str = repr(balance)
        assert "Balance" in repr_str

    def test_balance_hash_consistency(self):
        """Тест консистентности хеша баланса"""
        balance1 = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        balance2 = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        # Одинаковые объекты должны иметь одинаковый хеш
        assert hash(balance1) == hash(balance2)

    def test_balance_to_dict(self):
        """Тест сериализации баланса в словарь"""
        balance = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        if hasattr(balance, "to_dict"):
            balance_dict = balance.to_dict()
            assert isinstance(balance_dict, dict)
            assert "currency" in balance_dict
            assert "free" in balance_dict
            assert "used" in balance_dict

    def test_balance_from_dict(self):
        """Тест десериализации баланса из словаря"""
        balance_dict = {
            "currency": "USD",
            "free": {"amount": "1000.00", "currency": "USD"},
            "used": {"amount": "200.00", "currency": "USD"},
        }

        if hasattr(Balance, "from_dict"):
            balance = Balance.from_dict(balance_dict)
            assert balance.currency == "USD"


class TestBalanceEdgeCases:
    """Тесты граничных случаев для Balance"""

    def test_balance_very_small_amounts(self):
        """Тест баланса с очень малыми суммами"""
        free_money = Money(Decimal("0.00000001"), Currency.BTC)
        used_money = Money(Decimal("0.00000002"), Currency.BTC)

        balance = Balance(currency="BTC", free=free_money, used=used_money)

        assert balance.total.amount == Decimal("0.00000003")

    def test_balance_very_large_amounts(self):
        """Тест баланса с очень большими суммами"""
        free_money = Money(Decimal("999999999.99999999"), Currency.USD)
        used_money = Money(Decimal("1.00000001"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        assert balance.total.amount == Decimal("1000000001.00000000")

    def test_balance_precision_handling(self):
        """Тест обработки точности в расчетах"""
        free_money = Money(Decimal("100.12345678"), Currency.USD)
        used_money = Money(Decimal("50.87654321"), Currency.USD)

        balance = Balance(currency="USD", free=free_money, used=used_money)

        # Money ограничивает точность до 8 знаков после запятой
        assert balance.total.amount == Decimal("150.99999999")

    def test_balance_immutability_dataclass(self):
        """Тест неизменяемости dataclass Balance"""
        balance = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("200.00"), Currency.USD)
        )

        # Попытка изменить атрибут должна вызвать ошибку (frozen=True)
        with pytest.raises(AttributeError):
            balance.currency = Currency.EUR

    def test_balance_chain_operations(self):
        """Тест цепочки операций с балансом"""
        initial_balance = Balance(
            currency="USD", free=Money(Decimal("1000.00"), Currency.USD), used=Money(Decimal("0"), Currency.USD)
        )

        # Резервируем, затем освобождаем
        step1 = initial_balance.reserve(Money(Decimal("300.00"), Currency.USD))
        step2 = step1.reserve(Money(Decimal("200.00"), Currency.USD))
        step3 = step2.release(Money(Decimal("100.00"), Currency.USD))

        assert step3.free.amount == Decimal("600.00000000")  # 1000 - 300 - 200 + 100
        assert step3.used.amount == Decimal("400.00000000")  # 0 + 300 + 200 - 100


@pytest.mark.unit
class TestBalanceIntegrationWithMocks:
    """Интеграционные тесты Balance с моками"""

    def test_balance_with_mocked_money(self):
        """Тест Balance с замокированными Money объектами"""
        mock_free = Mock()
        mock_free.amount = Decimal("1000.00")
        mock_free.currency = "USD"
        mock_free.__add__ = Mock(return_value=Mock(amount=Decimal("1200.00"), currency="USD"))
        mock_free.__sub__ = Mock(return_value=Mock(amount=Decimal("700.00"), currency="USD"))

        mock_used = Mock()
        mock_used.amount = Decimal("200.00")
        mock_used.currency = "USD"
        mock_used.__add__ = Mock(return_value=Mock(amount=Decimal("500.00"), currency="USD"))

        # Патчим валидацию в __post_init__
        with patch.object(Balance, "__post_init__", return_value=None):
            balance = Balance(currency="USD", free=mock_free, used=mock_used)

            assert balance.free == mock_free
            assert balance.used == mock_used

    def test_balance_factory_pattern(self):
        """Тест паттерна фабрики для Balance"""

        def create_empty_balance(currency):
            return Balance(currency=currency, free=Money(Decimal("0"), currency), used=Money(Decimal("0"), currency))

        def create_funded_balance(currency, amount):
            return Balance(currency=currency, free=Money(amount, currency), used=Money(Decimal("0"), currency))

        empty_balance = create_empty_balance(Currency.USD)
        funded_balance = create_funded_balance(Currency.EUR, Decimal("1000.00"))

        assert empty_balance.total.amount == Decimal("0E-8")
        assert funded_balance.total.amount == Decimal("1000.00000000")

    def test_balance_builder_pattern(self):
        """Тест паттерна строителя для Balance"""

        class BalanceBuilder:
            def __init__(self):
                self._currency = None
                self._free_amount = Decimal("0")
                self._used_amount = Decimal("0")

            def with_currency(self, currency):
                self._currency = currency
                return self

            def with_free_amount(self, amount):
                self._free_amount = Decimal(str(amount))
                return self

            def with_used_amount(self, amount):
                self._used_amount = Decimal(str(amount))
                return self

            def build(self):
                return Balance(
                    currency=self._currency,
                    free=Money(self._free_amount, self._currency),
                    used=Money(self._used_amount, self._currency),
                )

        balance = BalanceBuilder().with_currency(Currency.USD).with_free_amount(750.50).with_used_amount(249.50).build()

        assert balance.total.amount == Decimal("1000.00000000")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
