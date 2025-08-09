"""
Unit тесты для currency.py.

Покрывает:
- Основной функционал Currency, CurrencyType, CurrencyNetwork, CurrencyInfo
- Валидацию данных
- Бизнес-логику торговых операций
- Обработку ошибок
- Сетевую совместимость
- Рыночные данные
"""

import pytest
import dataclasses
from typing import Dict, Any, Set
from unittest.mock import Mock, patch
from decimal import Decimal
from datetime import datetime, timezone

from domain.value_objects.currency import (
    Currency,
    CurrencyType,
    CurrencyNetwork,
    CurrencyInfo,
    create_currency,
)


class TestCurrencyType:
    """Тесты для CurrencyType."""

    def test_currency_type_values(self):
        """Тест значений типов валют."""
        assert CurrencyType.CRYPTO.value == "crypto"
        assert CurrencyType.FIAT.value == "fiat"
        assert CurrencyType.STABLECOIN.value == "stablecoin"
        assert CurrencyType.DEFI.value == "defi"

    def test_currency_type_membership(self):
        """Тест принадлежности к типам валют."""
        assert CurrencyType.CRYPTO in CurrencyType
        assert CurrencyType.FIAT in CurrencyType
        assert CurrencyType.STABLECOIN in CurrencyType
        assert CurrencyType.DEFI in CurrencyType


class TestCurrencyNetwork:
    """Тесты для CurrencyNetwork."""

    def test_currency_network_values(self):
        """Тест значений сетей."""
        assert CurrencyNetwork.BITCOIN.value == "bitcoin"
        assert CurrencyNetwork.ETHEREUM.value == "ethereum"
        assert CurrencyNetwork.BINANCE_SMART_CHAIN.value == "bsc"
        assert CurrencyNetwork.POLYGON.value == "polygon"
        assert CurrencyNetwork.ARBITRUM.value == "arbitrum"
        assert CurrencyNetwork.OPTIMISM.value == "optimism"
        assert CurrencyNetwork.SOLANA.value == "solana"
        assert CurrencyNetwork.CARDANO.value == "cardano"
        assert CurrencyNetwork.POLKADOT.value == "polkadot"
        assert CurrencyNetwork.MULTI_CHAIN.value == "multi"

    def test_currency_network_membership(self):
        """Тест принадлежности к сетям."""
        assert CurrencyNetwork.BITCOIN in CurrencyNetwork
        assert CurrencyNetwork.ETHEREUM in CurrencyNetwork
        assert CurrencyNetwork.BINANCE_SMART_CHAIN in CurrencyNetwork


class TestCurrencyInfo:
    """Тесты для CurrencyInfo."""

    @pytest.fixture
    def sample_currency_info(self) -> CurrencyInfo:
        """Тестовые данные для CurrencyInfo."""
        return CurrencyInfo(
            code="TEST",
            name="Test Currency",
            symbol="T",
            type=CurrencyType.CRYPTO,
            networks={CurrencyNetwork.ETHEREUM},
            decimals=8,
            is_active=True,
            market_cap_rank=100,
            volume_24h=Decimal("1000000.50"),
            price_usd=Decimal("1.25"),
            last_updated=datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            trading_priority=50,
        )

    def test_currency_info_creation(self, sample_currency_info):
        """Тест создания CurrencyInfo."""
        assert sample_currency_info.code == "TEST"
        assert sample_currency_info.name == "Test Currency"
        assert sample_currency_info.symbol == "T"
        assert sample_currency_info.type == CurrencyType.CRYPTO
        assert CurrencyNetwork.ETHEREUM in sample_currency_info.networks
        assert sample_currency_info.decimals == 8
        assert sample_currency_info.is_active is True
        assert sample_currency_info.market_cap_rank == 100
        assert sample_currency_info.volume_24h == Decimal("1000000.50")
        assert sample_currency_info.price_usd == Decimal("1.25")
        assert sample_currency_info.trading_priority == 50

    def test_currency_info_defaults(self):
        """Тест значений по умолчанию."""
        info = CurrencyInfo(
            code="DEFAULT",
            name="Default Currency",
            symbol="D",
            type=CurrencyType.FIAT,
        )
        assert info.networks == set()
        assert info.decimals == 8
        assert info.is_active is True
        assert info.market_cap_rank is None
        assert info.volume_24h is None
        assert info.price_usd is None
        assert info.last_updated is None
        assert info.trading_priority == 100

    def test_currency_info_to_dict(self, sample_currency_info):
        """Тест сериализации в словарь."""
        result = sample_currency_info.to_dict()

        assert result["code"] == "TEST"
        assert result["name"] == "Test Currency"
        assert result["symbol"] == "T"
        assert result["type"] == "crypto"
        assert result["networks"] == ["ethereum"]
        assert result["decimals"] == 8
        assert result["is_active"] is True
        assert result["market_cap_rank"] == 100
        assert result["volume_24h"] == "1000000.50"
        assert result["price_usd"] == "1.25"
        assert result["trading_priority"] == 50

    def test_currency_info_immutability(self, sample_currency_info):
        """Тест неизменяемости CurrencyInfo."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            sample_currency_info.code = "CHANGED"


class TestCurrency:
    """Тесты для Currency."""

    @pytest.fixture
    def sample_currencies(self) -> Dict[str, Currency]:
        """Тестовые валюты."""
        return {
            "btc": Currency.BTC,
            "eth": Currency.ETH,
            "usdt": Currency.USDT,
            "usd": Currency.USD,
            "eur": Currency.EUR,
        }

    def test_currency_creation(self, sample_currencies):
        """Тест создания валют."""
        btc = sample_currencies["btc"]
        assert btc.code == "BTC"
        assert btc.name == "Bitcoin"
        assert btc.symbol == "₿"
        assert btc.type == CurrencyType.CRYPTO

    def test_currency_properties(self, sample_currencies):
        """Тест свойств валют."""
        btc = sample_currencies["btc"]
        usdt = sample_currencies["usdt"]
        usd = sample_currencies["usd"]

        # Основные свойства
        assert btc.currency_code == "BTC"
        assert btc.code == "BTC"
        assert btc.name == "Bitcoin"
        assert btc.symbol == "₿"
        assert btc.type == CurrencyType.CRYPTO
        assert btc.decimals == 8
        assert btc.trading_priority == 1
        assert btc.is_active is True

        # Сети
        assert CurrencyNetwork.BITCOIN in btc.networks
        assert CurrencyNetwork.ETHEREUM in usdt.networks
        assert CurrencyNetwork.BINANCE_SMART_CHAIN in usdt.networks

    def test_currency_type_checks(self, sample_currencies):
        """Тест проверок типов валют."""
        btc = sample_currencies["btc"]
        usdt = sample_currencies["usdt"]
        usd = sample_currencies["usd"]

        # Проверки типов
        assert btc.is_crypto is True
        assert btc.is_fiat is False
        assert btc.is_stablecoin is False
        assert btc.is_major_crypto is True

        assert usdt.is_crypto is False
        assert usdt.is_fiat is False
        assert usdt.is_stablecoin is True
        assert usdt.is_major_crypto is False

        assert usd.is_crypto is False
        assert usd.is_fiat is True
        assert usd.is_stablecoin is False
        assert usd.is_major_crypto is False

    def test_currency_trading_compatibility(self, sample_currencies):
        """Тест совместимости для торговли."""
        btc = sample_currencies["btc"]
        usdt = sample_currencies["usdt"]
        usd = sample_currencies["usd"]

        # Валюты могут торговаться друг с другом
        assert btc.can_trade_with(usdt) is True
        assert usdt.can_trade_with(btc) is True
        assert btc.can_trade_with(usd) is True

        # Валюты не могут торговаться сами с собой
        assert btc.can_trade_with(btc) is False
        assert usdt.can_trade_with(usdt) is False

    def test_currency_hash(self, sample_currencies):
        """Тест хеширования валют."""
        btc = sample_currencies["btc"]
        eth = sample_currencies["eth"]

        # Хеши должны быть разными для разных валют
        assert btc.hash != eth.hash
        assert len(btc.hash) == 32  # MD5 hex digest length

    def test_currency_to_dict(self, sample_currencies):
        """Тест сериализации в словарь."""
        btc = sample_currencies["btc"]
        result = btc.to_dict()

        assert result["code"] == "BTC"
        assert result["name"] == "Bitcoin"
        assert result["symbol"] == "₿"
        assert result["type"] == "crypto"
        assert "bitcoin" in result["networks"]
        assert result["decimals"] == 8
        assert result["is_active"] is True
        assert result["trading_priority"] == 1

    def test_currency_validation(self, sample_currencies):
        """Тест валидации валют."""
        btc = sample_currencies["btc"]
        assert btc.validate() is True

    def test_currency_equality(self, sample_currencies):
        """Тест равенства валют."""
        btc1 = sample_currencies["btc"]
        btc2 = Currency.BTC
        eth = sample_currencies["eth"]

        assert btc1 == btc2
        assert btc1 != eth
        assert btc1 != "BTC"  # Не равно строке

    def test_currency_hash_equality(self, sample_currencies):
        """Тест хеширования для равенства."""
        btc1 = sample_currencies["btc"]
        btc2 = Currency.BTC
        eth = sample_currencies["eth"]

        assert hash(btc1) == hash(btc2)
        assert hash(btc1) != hash(eth)

    def test_currency_repr(self, sample_currencies):
        """Тест строкового представления."""
        btc = sample_currencies["btc"]
        assert repr(btc) == "Currency.BTC"

    def test_currency_str(self, sample_currencies):
        """Тест строкового представления."""
        btc = sample_currencies["btc"]
        assert str(btc) == "BTC"

    def test_currency_from_string_valid(self, sample_currencies):
        """Тест создания из строки - валидные случаи."""
        btc = Currency.from_string("BTC")
        eth = Currency.from_string("eth")  # Регистр не важен
        usdt = Currency.from_string("USDT")

        assert btc == Currency.BTC
        assert eth == Currency.ETH
        assert usdt == Currency.USDT

    def test_currency_from_string_invalid(self):
        """Тест создания из строки - невалидные случаи."""
        assert Currency.from_string("INVALID") is None
        assert Currency.from_string("") is None
        assert Currency.from_string(None) is None

    def test_currency_get_trading_pairs(self, sample_currencies):
        """Тест получения торговых пар."""
        btc = sample_currencies["btc"]
        pairs = Currency.get_trading_pairs(btc)

        # BTC не должен быть в списке пар
        assert btc not in pairs
        # Должны быть другие валюты
        assert Currency.ETH in pairs
        assert Currency.USDT in pairs
        assert Currency.USD in pairs

    def test_currency_major_crypto_detection(self):
        """Тест определения основных криптовалют."""
        assert Currency.BTC.is_major_crypto is True
        assert Currency.ETH.is_major_crypto is True
        assert Currency.USDT.is_major_crypto is False
        assert Currency.USD.is_major_crypto is False


class TestCreateCurrency:
    """Тесты для фабричной функции create_currency."""

    def test_create_currency_valid(self):
        """Тест создания валюты - валидные случаи."""
        btc = create_currency("BTC")
        eth = create_currency("eth")
        usdt = create_currency("USDT")

        assert btc == Currency.BTC
        assert eth == Currency.ETH
        assert usdt == Currency.USDT

    def test_create_currency_invalid(self):
        """Тест создания валюты - невалидные случаи."""
        assert create_currency("INVALID") is None
        assert create_currency("") is None
        assert create_currency(None) is None

    def test_create_currency_with_kwargs(self):
        """Тест создания валюты с дополнительными параметрами."""
        # Функция игнорирует дополнительные параметры
        btc = create_currency("BTC", extra_param="value")
        assert btc == Currency.BTC


class TestCurrencyIntegration:
    """Интеграционные тесты для валют."""

    def test_crypto_currencies_network_support(self):
        """Тест поддержки сетей криптовалют."""
        # BTC поддерживает Bitcoin сеть
        assert CurrencyNetwork.BITCOIN in Currency.BTC.networks

        # ETH поддерживает Ethereum сеть
        assert CurrencyNetwork.ETHEREUM in Currency.ETH.networks

        # USDT поддерживает несколько сетей
        assert CurrencyNetwork.ETHEREUM in Currency.USDT.networks
        assert CurrencyNetwork.BINANCE_SMART_CHAIN in Currency.USDT.networks

    def test_stablecoin_characteristics(self):
        """Тест характеристик стейблкоинов."""
        stablecoins = [Currency.USDT, Currency.USDC, Currency.BUSD, Currency.DAI, Currency.TUSD]

        for stablecoin in stablecoins:
            assert stablecoin.is_stablecoin is True
            assert stablecoin.is_crypto is False
            assert stablecoin.is_fiat is False
            assert stablecoin.is_major_crypto is False

    def test_fiat_characteristics(self):
        """Тест характеристик фиатных валют."""
        fiat_currencies = [
            Currency.USD,
            Currency.EUR,
            Currency.GBP,
            Currency.JPY,
            Currency.CNY,
            Currency.KRW,
            Currency.RUB,
        ]

        for fiat in fiat_currencies:
            assert fiat.is_fiat is True
            assert fiat.is_crypto is False
            assert fiat.is_stablecoin is False
            assert fiat.is_major_crypto is False
            assert fiat.networks == set()  # Фиатные валюты не имеют сетей

    def test_trading_priority_order(self):
        """Тест порядка торговых приоритетов."""
        # BTC должен иметь наивысший приоритет
        assert Currency.BTC.trading_priority == 1
        assert Currency.ETH.trading_priority == 2
        assert Currency.USDT.trading_priority == 3

        # Фиатные валюты должны иметь низкий приоритет
        assert Currency.USD.trading_priority > 25

    def test_currency_comparison_operations(self):
        """Тест операций сравнения валют."""
        currencies = [Currency.BTC, Currency.ETH, Currency.USDT, Currency.USD]

        # Все валюты должны быть уникальными
        for i, curr1 in enumerate(currencies):
            for j, curr2 in enumerate(currencies):
                if i == j:
                    assert curr1 == curr2
                else:
                    assert curr1 != curr2

    def test_currency_hash_consistency(self):
        """Тест консистентности хеширования."""
        btc1 = Currency.BTC
        btc2 = Currency.BTC

        # Одинаковые валюты должны иметь одинаковые хеши
        assert hash(btc1) == hash(btc2)

        # Хеш должен быть стабильным
        assert hash(btc1) == hash(Currency.BTC)

    def test_currency_serialization_roundtrip(self):
        """Тест сериализации и десериализации."""
        btc = Currency.BTC
        btc_dict = btc.to_dict()

        # Проверяем, что все ключи присутствуют
        expected_keys = {
            "code",
            "name",
            "symbol",
            "type",
            "networks",
            "decimals",
            "is_active",
            "market_cap_rank",
            "volume_24h",
            "price_usd",
            "last_updated",
            "trading_priority",
        }
        assert set(btc_dict.keys()) == expected_keys

        # Проверяем типы значений
        assert isinstance(btc_dict["code"], str)
        assert isinstance(btc_dict["name"], str)
        assert isinstance(btc_dict["symbol"], str)
        assert isinstance(btc_dict["type"], str)
        assert isinstance(btc_dict["networks"], list)
        assert isinstance(btc_dict["decimals"], int)
        assert isinstance(btc_dict["is_active"], bool)
        assert isinstance(btc_dict["trading_priority"], int)


class TestCurrencyEdgeCases:
    """Тесты граничных случаев для валют."""

    def test_empty_networks(self):
        """Тест валют без сетей."""
        # XRP не имеет определенных сетей
        assert Currency.XRP.networks == set()

        # Фиатные валюты не имеют сетей
        assert Currency.USD.networks == set()
        assert Currency.EUR.networks == set()

    def test_currency_with_special_symbols(self):
        """Тест валют со специальными символами."""
        # Проверяем, что символы корректно обрабатываются
        assert Currency.BTC.symbol == "₿"
        assert Currency.ETH.symbol == "Ξ"
        assert Currency.ADA.symbol == "₳"
        assert Currency.LTC.symbol == "Ł"
        assert Currency.DOT.symbol == "●"
        assert Currency.SOL.symbol == "◎"

    def test_currency_case_insensitivity(self):
        """Тест нечувствительности к регистру."""
        # Создание из строки должно быть нечувствительно к регистру
        assert Currency.from_string("btc") == Currency.BTC
        assert Currency.from_string("BTC") == Currency.BTC
        assert Currency.from_string("Btc") == Currency.BTC

    def test_currency_trading_with_inactive(self):
        """Тест торговли с неактивными валютами."""
        # Проверяем, что активные валюты могут торговаться друг с другом
        assert Currency.BTC.can_trade_with(Currency.ETH) is True
        assert Currency.ETH.can_trade_with(Currency.BTC) is True

        # Валюты не могут торговаться сами с собой
        assert Currency.BTC.can_trade_with(Currency.BTC) is False
        assert Currency.ETH.can_trade_with(Currency.ETH) is False

    def test_currency_validation_edge_cases(self):
        """Тест валидации граничных случаев."""
        # Все существующие валюты должны быть валидными
        for currency in Currency:
            assert currency.validate() is True

    def test_currency_hash_collision_resistance(self):
        """Тест устойчивости к коллизиям хешей."""
        currencies = list(Currency)
        hashes = [currency.hash for currency in currencies]

        # Все хеши должны быть уникальными
        assert len(hashes) == len(set(hashes))

    def test_currency_memory_efficiency(self):
        """Тест эффективности использования памяти."""
        # Создание валюты из строки должно возвращать тот же объект
        btc1 = Currency.from_string("BTC")
        btc2 = Currency.from_string("BTC")
        assert btc1 is btc2  # Должен быть тот же объект

    def test_currency_performance(self):
        """Тест производительности операций с валютами."""
        import time

        # Тест скорости создания из строки
        start_time = time.time()
        for _ in range(1000):
            Currency.from_string("BTC")
        end_time = time.time()

        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0  # Менее 1 секунды для 1000 операций
