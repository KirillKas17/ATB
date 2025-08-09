"""
Тесты для сущностей Account и Balance.
"""

import pytest
from decimal import Decimal
from datetime import datetime
from typing import Any

from domain.entities.account import Account, Balance, AccountProtocol, BalanceProtocol


class TestAccount:
    """Тесты для сущности Account."""

    @pytest.fixture
    def sample_balances(self) -> list[Balance]:
        """Фикстура с примерами балансов."""
        return [
            Balance(currency="BTC", available=Decimal("1.5"), locked=Decimal("0.1")),
            Balance(currency="ETH", available=Decimal("10.0"), locked=Decimal("2.0")),
            Balance(currency="USDT", available=Decimal("1000.0"), locked=Decimal("100.0")),
        ]

    @pytest.fixture
    def account(self, sample_balances: list[Balance]) -> Account:
        """Фикстура с валидным аккаунтом."""
        return Account(
            account_id="test_account_001",
            exchange_name="test_exchange",
            balances=sample_balances,
            is_active=True,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            updated_at=datetime(2024, 1, 1, 12, 0, 0),
            metadata={"user_id": "user_123", "tier": "premium"},
        )

    def test_account_creation(self, account: Account) -> None:
        """Тест создания аккаунта."""
        assert account.account_id == "test_account_001"
        assert account.exchange_name == "test_exchange"
        assert len(account.balances) == 3
        assert account.is_active is True
        assert account.metadata["user_id"] == "user_123"
        assert account.metadata["tier"] == "premium"

    def test_account_default_values(self: "TestAccount") -> None:
        """Тест значений по умолчанию."""
        account = Account()
        assert account.account_id is not None
        assert account.exchange_name == ""
        assert account.balances == []
        assert account.is_active is True
        assert isinstance(account.created_at, datetime)
        assert isinstance(account.updated_at, datetime)
        assert account.metadata == {}

    def test_get_balance_existing(self, account: Account) -> None:
        """Тест получения существующего баланса."""
        balance = account.get_balance("BTC")
        assert balance is not None
        assert balance.currency == "BTC"
        assert balance.available == Decimal("1.5")
        assert balance.locked == Decimal("0.1")

    def test_get_balance_non_existing(self, account: Account) -> None:
        """Тест получения несуществующего баланса."""
        balance = account.get_balance("LTC")
        assert balance is None

    def test_get_balance_empty_account(self: "TestAccount") -> None:
        """Тест получения баланса из пустого аккаунта."""
        account = Account()
        balance = account.get_balance("BTC")
        assert balance is None

    def test_has_sufficient_balance_sufficient(self, account: Account) -> None:
        """Тест проверки достаточности баланса - достаточно."""
        assert account.has_sufficient_balance("BTC", Decimal("1.0")) is True
        assert account.has_sufficient_balance("BTC", Decimal("1.5")) is True

    def test_has_sufficient_balance_insufficient(self, account: Account) -> None:
        """Тест проверки достаточности баланса - недостаточно."""
        assert account.has_sufficient_balance("BTC", Decimal("2.0")) is False
        assert account.has_sufficient_balance("BTC", Decimal("1.6")) is False

    def test_has_sufficient_balance_exact(self, account: Account) -> None:
        """Тест проверки достаточности баланса - точно."""
        assert account.has_sufficient_balance("BTC", Decimal("1.5")) is True

    def test_has_sufficient_balance_no_balance(self, account: Account) -> None:
        """Тест проверки достаточности баланса - нет баланса."""
        assert account.has_sufficient_balance("LTC", Decimal("1.0")) is False

    def test_has_sufficient_balance_zero_amount(self, account: Account) -> None:
        """Тест проверки достаточности баланса - нулевая сумма."""
        assert account.has_sufficient_balance("BTC", Decimal("0")) is True

    def test_to_dict(self, account: Account) -> None:
        """Тест преобразования в словарь."""
        data = account.to_dict()
        assert data["account_id"] == "test_account_001"
        assert data["exchange_name"] == "test_exchange"
        assert len(data["balances"]) == 3
        assert data["is_active"] is True
        assert data["created_at"] == "2024-01-01T12:00:00"
        assert data["updated_at"] == "2024-01-01T12:00:00"
        assert data["metadata"]["user_id"] == "user_123"
        assert data["metadata"]["tier"] == "premium"

    def test_from_dict(self, account: Account) -> None:
        """Тест создания из словаря."""
        data = account.to_dict()
        restored_account = Account.from_dict(data)

        assert restored_account.account_id == account.account_id
        assert restored_account.exchange_name == account.exchange_name
        assert len(restored_account.balances) == len(account.balances)
        assert restored_account.is_active == account.is_active
        assert restored_account.metadata == account.metadata

    def test_from_dict_with_defaults(self: "TestAccount") -> None:
        """Тест создания из словаря с значениями по умолчанию."""
        data = {
            "account_id": "test_account_002",
            "exchange_name": "test_exchange",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T12:00:00",
        }
        account = Account.from_dict(data)

        assert account.account_id == "test_account_002"
        assert account.exchange_name == "test_exchange"
        assert account.balances == []
        assert account.is_active is True
        assert account.metadata == {}

    def test_account_protocol_compliance(self, account: Account) -> None:
        """Тест соответствия протоколу AccountProtocol."""
        assert isinstance(account, AccountProtocol)

        # Проверяем наличие всех атрибутов
        assert hasattr(account, "account_id")
        assert hasattr(account, "exchange_name")
        assert hasattr(account, "balances")
        assert hasattr(account, "is_active")
        assert hasattr(account, "created_at")
        assert hasattr(account, "updated_at")
        assert hasattr(account, "metadata")

        # Проверяем наличие всех методов
        assert hasattr(account, "get_balance")
        assert hasattr(account, "has_sufficient_balance")
        assert hasattr(account, "to_dict")
        assert hasattr(account, "from_dict")


class TestBalance:
    """Тесты для сущности Balance."""

    @pytest.fixture
    def balance(self) -> Balance:
        """Фикстура с валидным балансом."""
        return Balance(
            currency="BTC",
            available=Decimal("1.5"),
            locked=Decimal("0.1"),
        )

    def test_balance_creation(self, balance: Balance) -> None:
        """Тест создания баланса."""
        assert balance.currency == "BTC"
        assert balance.available == Decimal("1.5")
        assert balance.locked == Decimal("0.1")

    def test_balance_default_values(self: "TestBalance") -> None:
        """Тест значений по умолчанию."""
        balance = Balance(currency="ETH")
        assert balance.currency == "ETH"
        assert balance.available == Decimal("0")
        assert balance.locked == Decimal("0")

    def test_total_property(self, balance: Balance) -> None:
        """Тест свойства total."""
        assert balance.total == Decimal("1.6")  # 1.5 + 0.1

    def test_total_property_zero_values(self: "TestBalance") -> None:
        """Тест свойства total с нулевыми значениями."""
        balance = Balance(currency="ETH")
        assert balance.total == Decimal("0")

    def test_total_property_negative_values(self: "TestBalance") -> None:
        """Тест свойства total с отрицательными значениями."""
        balance = Balance(
            currency="ETH",
            available=Decimal("-1.0"),
            locked=Decimal("-0.5"),
        )
        assert balance.total == Decimal("-1.5")

    def test_to_dict(self, balance: Balance) -> None:
        """Тест преобразования в словарь."""
        data = balance.to_dict()
        assert data["currency"] == "BTC"
        assert data["available"] == "1.5"
        assert data["locked"] == "0.1"
        assert data["total"] == "1.6"

    def test_from_dict(self, balance: Balance) -> None:
        """Тест создания из словаря."""
        data = balance.to_dict()
        restored_balance = Balance.from_dict(data)

        assert restored_balance.currency == balance.currency
        assert restored_balance.available == balance.available
        assert restored_balance.locked == balance.locked

    def test_from_dict_with_string_values(self: "TestBalance") -> None:
        """Тест создания из словаря со строковыми значениями."""
        data = {
            "currency": "ETH",
            "available": "10.5",
            "locked": "2.0",
        }
        balance = Balance.from_dict(data)

        assert balance.currency == "ETH"
        assert balance.available == Decimal("10.5")
        assert balance.locked == Decimal("2.0")

    def test_balance_protocol_compliance(self, balance: Balance) -> None:
        """Тест соответствия протоколу BalanceProtocol."""
        assert isinstance(balance, BalanceProtocol)

        # Проверяем наличие всех атрибутов
        assert hasattr(balance, "currency")
        assert hasattr(balance, "available")
        assert hasattr(balance, "locked")

        # Проверяем наличие всех методов и свойств
        assert hasattr(balance, "total")
        assert hasattr(balance, "to_dict")
        assert hasattr(balance, "from_dict")

    def test_balance_equality(self: "TestBalance") -> None:
        """Тест равенства балансов."""
        balance1 = Balance(currency="BTC", available=Decimal("1.0"), locked=Decimal("0.1"))
        balance2 = Balance(currency="BTC", available=Decimal("1.0"), locked=Decimal("0.1"))
        balance3 = Balance(currency="ETH", available=Decimal("1.0"), locked=Decimal("0.1"))

        assert balance1 == balance2
        assert balance1 != balance3

    def test_balance_string_representation(self, balance: Balance) -> None:
        """Тест строкового представления."""
        str_repr = str(balance)
        assert "Balance" in str_repr
        assert "BTC" in str_repr

    def test_balance_repr_representation(self, balance: Balance) -> None:
        """Тест представления для отладки."""
        repr_str = repr(balance)
        assert "Balance" in repr_str
        assert "currency='BTC'" in repr_str
        assert "available=Decimal('1.5')" in repr_str
        assert "locked=Decimal('0.1')" in repr_str


class TestAccountBalanceIntegration:
    """Тесты интеграции Account и Balance."""

    def test_account_with_multiple_balances(self: "TestAccountBalanceIntegration") -> None:
        """Тест аккаунта с несколькими балансами."""
        balances = [
            Balance(currency="BTC", available=Decimal("1.0"), locked=Decimal("0.1")),
            Balance(currency="ETH", available=Decimal("10.0"), locked=Decimal("2.0")),
            Balance(currency="USDT", available=Decimal("1000.0"), locked=Decimal("100.0")),
        ]

        account = Account(
            account_id="test_account",
            exchange_name="test_exchange",
            balances=balances,
        )

        assert len(account.balances) == 3
        assert account.get_balance("BTC").available == Decimal("1.0")
        assert account.get_balance("ETH").available == Decimal("10.0")
        assert account.get_balance("USDT").available == Decimal("1000.0")

    def test_account_balance_operations(self: "TestAccountBalanceIntegration") -> None:
        """Тест операций с балансами аккаунта."""
        account = Account(
            account_id="test_account",
            exchange_name="test_exchange",
            balances=[
                Balance(currency="BTC", available=Decimal("2.0"), locked=Decimal("0.5")),
            ],
        )

        # Проверяем достаточность баланса
        assert account.has_sufficient_balance("BTC", Decimal("1.0")) is True
        assert account.has_sufficient_balance("BTC", Decimal("2.0")) is True
        assert account.has_sufficient_balance("BTC", Decimal("2.5")) is False

        # Проверяем общий баланс
        btc_balance = account.get_balance("BTC")
        assert btc_balance.total == Decimal("2.5")  # 2.0 + 0.5

    def test_account_serialization_with_balances(self: "TestAccountBalanceIntegration") -> None:
        """Тест сериализации аккаунта с балансами."""
        account = Account(
            account_id="test_account",
            exchange_name="test_exchange",
            balances=[
                Balance(currency="BTC", available=Decimal("1.0"), locked=Decimal("0.1")),
                Balance(currency="ETH", available=Decimal("10.0"), locked=Decimal("2.0")),
            ],
            metadata={"user_id": "user_123"},
        )

        data = account.to_dict()
        restored_account = Account.from_dict(data)

        assert restored_account.account_id == account.account_id
        assert restored_account.exchange_name == account.exchange_name
        assert len(restored_account.balances) == len(account.balances)
        assert restored_account.metadata == account.metadata

        # Проверяем, что балансы восстановлены корректно
        btc_balance = restored_account.get_balance("BTC")
        assert btc_balance.available == Decimal("1.0")
        assert btc_balance.locked == Decimal("0.1")

        eth_balance = restored_account.get_balance("ETH")
        assert eth_balance.available == Decimal("10.0")
        assert eth_balance.locked == Decimal("2.0")
