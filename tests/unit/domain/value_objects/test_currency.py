#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Currency Value Object.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any

from domain.value_objects.currency import Currency
from domain.exceptions import ValidationError


class TestCurrency:
    """Тесты для Currency Value Object."""

    @pytest.fixture
    def valid_currency_codes(self) -> list[str]:
        """Валидные коды валют."""
        return ["USD", "EUR", "BTC", "ETH", "USDT", "BUSD"]

    @pytest.fixture
    def invalid_currency_codes(self) -> list[str]:
        """Невалидные коды валют."""
        return ["", "US", "USDD", "123", "usd", None]

    def test_currency_creation_valid(self, valid_currency_codes: list[str]) -> None:
        """Тест создания валидных валют."""
        for code in valid_currency_codes:
            currency = Currency(code)
            assert currency.code == code
            assert str(currency) == code

    def test_currency_creation_invalid(self, invalid_currency_codes: list[str]) -> None:
        """Тест создания невалидных валют."""
        for code in invalid_currency_codes:
            with pytest.raises(ValidationError):
                Currency(code)

    def test_currency_equality(self) -> None:
        """Тест равенства валют."""
        usd1 = Currency("USD")
        usd2 = Currency("USD")
        eur = Currency("EUR")

        assert usd1 == usd2
        assert usd1 != eur
        assert hash(usd1) == hash(usd2)
        assert hash(usd1) != hash(eur)

    def test_currency_immutability(self) -> None:
        """Тест неизменяемости валюты."""
        currency = Currency("USD")
        original_code = currency.code

        # Попытка изменения через атрибут должна вызвать ошибку
        with pytest.raises(AttributeError):
            currency.code = "EUR"  # type: ignore

        assert currency.code == original_code

    def test_currency_serialization(self) -> None:
        """Тест сериализации валюты."""
        currency = Currency("BTC")
        
        # Тест to_dict
        data = currency.to_dict()
        assert data["code"] == "BTC"
        
        # Тест from_dict
        restored = Currency.from_dict(data)
        assert restored == currency

    def test_currency_string_representations(self) -> None:
        """Тест строковых представлений."""
        currency = Currency("ETH")
        
        assert str(currency) == "ETH"
        assert repr(currency) == "Currency(ETH)"

    def test_currency_comparison_operations(self) -> None:
        """Тест операций сравнения (лексикографическое)."""
        btc = Currency("BTC")
        eth = Currency("ETH")
        usd = Currency("USD")

        # BTC < ETH < USD (алфавитный порядок)
        assert btc < eth < usd
        assert usd > eth > btc
        assert btc <= eth <= usd
        assert usd >= eth >= btc

    def test_currency_hash_consistency(self) -> None:
        """Тест консистентности хэширования."""
        currency1 = Currency("USD")
        currency2 = Currency("USD")
        
        # Одинаковые валюты должны иметь одинаковый хэш
        assert hash(currency1) == hash(currency2)
        
        # Можно использовать в множествах и словарях
        currency_set = {currency1, currency2}
        assert len(currency_set) == 1

    def test_currency_validation_comprehensive(self) -> None:
        """Комплексная валидация валюты."""
        # Тест null/None
        with pytest.raises(ValidationError):
            Currency(None)  # type: ignore
            
        # Тест пустой строки
        with pytest.raises(ValidationError):
            Currency("")
            
        # Тест слишком короткого кода
        with pytest.raises(ValidationError):
            Currency("U")
            
        # Тест слишком длинного кода
        with pytest.raises(ValidationError):
            Currency("USDTT")
            
        # Тест невалидных символов
        with pytest.raises(ValidationError):
            Currency("US$")
            
        # Тест нижнего регистра
        with pytest.raises(ValidationError):
            Currency("usd")

    def test_currency_is_fiat(self) -> None:
        """Тест определения фиатной валюты."""
        usd = Currency("USD")
        eur = Currency("EUR")
        btc = Currency("BTC")
        
        assert usd.is_fiat() is True
        assert eur.is_fiat() is True
        assert btc.is_fiat() is False

    def test_currency_is_crypto(self) -> None:
        """Тест определения криптовалюты."""
        btc = Currency("BTC")
        eth = Currency("ETH")
        usd = Currency("USD")
        
        assert btc.is_crypto() is True
        assert eth.is_crypto() is True
        assert usd.is_crypto() is False

    def test_currency_is_stablecoin(self) -> None:
        """Тест определения стейблкоина."""
        usdt = Currency("USDT")
        busd = Currency("BUSD")
        btc = Currency("BTC")
        
        assert usdt.is_stablecoin() is True
        assert busd.is_stablecoin() is True
        assert btc.is_stablecoin() is False

    def test_currency_decimal_places(self) -> None:
        """Тест количества десятичных знаков."""
        usd = Currency("USD")
        btc = Currency("BTC")
        
        assert usd.decimal_places() == 2  # Фиат: 2 знака
        assert btc.decimal_places() == 8  # Крипто: 8 знаков

    def test_currency_symbol(self) -> None:
        """Тест символа валюты."""
        usd = Currency("USD")
        eur = Currency("EUR")
        btc = Currency("BTC")
        
        assert usd.symbol() == "$"
        assert eur.symbol() == "€"
        assert btc.symbol() == "₿"

    def test_currency_conversion_compatibility(self) -> None:
        """Тест совместимости для конвертации."""
        usd = Currency("USD")
        eur = Currency("EUR")
        btc = Currency("BTC")
        
        # Фиат к фиат - совместимо
        assert usd.is_conversion_compatible(eur) is True
        
        # Крипто к крипто - совместимо
        eth = Currency("ETH")
        assert btc.is_conversion_compatible(eth) is True
        
        # Фиат к крипто - ограниченно совместимо
        assert usd.is_conversion_compatible(btc) is True
        
        # Стейблкоины совместимы со всем
        usdt = Currency("USDT")
        assert usdt.is_conversion_compatible(usd) is True
        assert usdt.is_conversion_compatible(btc) is True

    def test_currency_risk_level(self) -> None:
        """Тест уровня риска валюты."""
        usd = Currency("USD")
        btc = Currency("BTC")
        usdt = Currency("USDT")
        
        # Фиат - низкий риск
        assert usd.risk_level() == "LOW"
        
        # Стейблкоин - низкий/средний риск
        assert usdt.risk_level() == "MEDIUM"
        
        # Крипто - высокий риск
        assert btc.risk_level() == "HIGH"

    def test_currency_market_hours(self) -> None:
        """Тест рыночных часов."""
        usd = Currency("USD")
        btc = Currency("BTC")
        
        # Фиат торгуется в рабочие часы
        assert usd.is_market_open() is not None
        
        # Крипто торгуется 24/7
        assert btc.is_market_open() is True

    def test_currency_pair_creation(self) -> None:
        """Тест создания валютных пар."""
        btc = Currency("BTC")
        usd = Currency("USD")
        
        pair = btc.create_pair_with(usd)
        assert pair == "BTCUSD"
        
        reverse_pair = usd.create_pair_with(btc)
        assert reverse_pair == "USDBTC"

    def test_currency_performance_critical_operations(self) -> None:
        """Тест производительности критических операций."""
        # Создание множества валют
        currencies = [Currency("USD") for _ in range(1000)]
        assert len(currencies) == 1000
        assert all(c.code == "USD" for c in currencies)
        
        # Операции сравнения
        usd = Currency("USD")
        for _ in range(1000):
            assert usd == Currency("USD")
            assert hash(usd) == hash(Currency("USD"))

    def test_currency_edge_cases(self) -> None:
        """Тест граничных случаев."""
        # Минимальная длина (3 символа)
        min_currency = Currency("BTC")
        assert len(min_currency.code) == 3
        
        # Максимальная длина (4 символа)
        max_currency = Currency("USDT")
        assert len(max_currency.code) == 4
        
        # Специальные символы в названии (только буквы разрешены)
        with pytest.raises(ValidationError):
            Currency("BT1")  # Цифра
            
        with pytest.raises(ValidationError):
            Currency("BT-")  # Дефис

    def test_currency_caching_behavior(self) -> None:
        """Тест поведения кэширования."""
        # Создание одинаковых валют
        usd1 = Currency("USD")
        usd2 = Currency("USD")
        
        # Должны быть равны и иметь одинаковый хэш
        assert usd1 == usd2
        assert hash(usd1) == hash(usd2)
        
        # Могут быть использованы в качестве ключей
        currency_dict = {usd1: "US Dollar", usd2: "US Dollar"}
        assert len(currency_dict) == 1  # Один ключ 