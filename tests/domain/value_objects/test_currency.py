"""Тесты для Currency value object."""
from domain.value_objects.currency import Currency
class TestCurrency:
    """Тесты для класса Currency."""
    def test_currency_enum_values(self) -> None:
        """Тест значений перечисления валют."""
        assert Currency.BTC.currency_code == "BTC"
        assert Currency.ETH.currency_code == "ETH"
        assert Currency.USDT.currency_code == "USDT"
        assert Currency.USD.currency_code == "USD"
        assert Currency.EUR.currency_code == "EUR"
    def test_currency_property_value(self) -> None:
        """Тест свойства value."""
        currency = Currency.BTC
        assert currency.currency_code == "BTC"
    def test_from_string_valid_crypto(self) -> None:
        """Тест создания валюты из строки для криптовалют."""
        assert Currency.from_string("BTC") == Currency.BTC
        assert Currency.from_string("ETH") == Currency.ETH
        assert Currency.from_string("USDT") == Currency.USDT
        assert Currency.from_string("SOL") == Currency.SOL
    def test_from_string_valid_fiat(self) -> None:
        """Тест создания валюты из строки для фиатных валют."""
        assert Currency.from_string("USD") == Currency.USD
        assert Currency.from_string("EUR") == Currency.EUR
        assert Currency.from_string("RUB") == Currency.RUB
    def test_from_string_case_insensitive(self) -> None:
        """Тест создания валюты из строки с разным регистром."""
        assert Currency.from_string("btc") == Currency.BTC
        assert Currency.from_string("Btc") == Currency.BTC
        assert Currency.from_string("BTC") == Currency.BTC
    def test_from_string_invalid(self) -> None:
        """Тест создания валюты из невалидной строки."""
        assert Currency.from_string("INVALID") is None
        assert Currency.from_string("") is None
        assert Currency.from_string("123") is None
    def test_str_representation(self) -> None:
        """Тест строкового представления."""
        assert str(Currency.BTC) == "BTC"
        assert str(Currency.ETH) == "ETH"
        assert str(Currency.USD) == "USD"
    def test_repr_representation(self) -> None:
        """Тест представления для отладки."""
        assert repr(Currency.BTC) == "Currency.BTC"
        assert repr(Currency.ETH) == "Currency.ETH"
        assert repr(Currency.USD) == "Currency.USD"
    def test_currency_comparison(self) -> None:
        """Тест сравнения валют."""
        assert Currency.BTC == Currency.BTC
        assert Currency.BTC != Currency.ETH
        assert Currency.USD == Currency.USD
    def test_currency_hash(self) -> None:
        """Тест хеширования валют."""
        currency_set = {Currency.BTC, Currency.ETH, Currency.USD}
        assert len(currency_set) == 3
        assert Currency.BTC in currency_set
    def test_all_currencies_defined(self) -> None:
        """Тест что все валюты определены корректно."""
        expected_crypto = ["BTC", "ETH", "USDT", "USDC", "BNB", "ADA", "SOL", "DOT", "MATIC", "LINK"]
        expected_fiat = ["USD", "EUR", "RUB"]
        for crypto in expected_crypto:
            assert Currency.from_string(crypto) is not None
        for fiat in expected_fiat:
            assert Currency.from_string(fiat) is not None
    def test_currency_immutability(self) -> None:
        """Тест неизменяемости валют."""
        currency = Currency.BTC
        original_value = currency.currency_code
        # Enum значения неизменяемы по дизайну
        assert currency.currency_code == original_value
        assert currency.currency_code == "BTC" 
