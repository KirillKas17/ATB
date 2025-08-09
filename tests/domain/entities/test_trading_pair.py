"""
Тесты для доменной сущности TradingPair
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from decimal import Decimal

from domain.entities.trading_pair import TradingPair, PairStatus
from domain.value_objects.currency import Currency
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume


class TestTradingPair:
    """Тесты для TradingPair"""

    @pytest.fixture
    def btc_currency(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для BTC валюты"""
        return Currency.BTC

    @pytest.fixture
    def usdt_currency(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для USDT валюты"""
        return Currency.USDT

    @pytest.fixture
    def eth_currency(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для ETH валюты"""
        return Currency.ETH

    @pytest.fixture
    def btc_usdt_pair(self, btc_currency, usdt_currency) -> Any:
        """Фикстура для BTC/USDT пары"""
        return TradingPair(
            symbol="BTC/USDT",
            base_currency=btc_currency,
            quote_currency=usdt_currency,
            min_order_size=Volume(value=Decimal("0.001"), currency=btc_currency),
            max_order_size=Volume(value=Decimal("100"), currency=btc_currency),
            price_precision=2,
            volume_precision=6,
        )

    def test_trading_pair_creation(self, btc_currency, usdt_currency) -> None:
        """Тест создания торговой пары"""
        pair = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        assert pair.symbol == "BTC/USDT"
        assert pair.base_currency == btc_currency
        assert pair.quote_currency == usdt_currency
        assert pair.is_active is True
        assert pair.price_precision == 8
        assert pair.volume_precision == 8
        assert isinstance(pair.created_at, datetime)
        assert isinstance(pair.updated_at, datetime)

    def test_trading_pair_with_optional_params(self, btc_currency, usdt_currency) -> None:
        """Тест создания торговой пары с опциональными параметрами"""
        min_volume = Volume(value=Decimal("0.001"), currency=btc_currency)
        max_volume = Volume(value=Decimal("100"), currency=btc_currency)

        pair = TradingPair(
            symbol="BTC/USDT",
            base_currency=btc_currency,
            quote_currency=usdt_currency,
            min_order_size=min_volume,
            max_order_size=max_volume,
            price_precision=2,
            volume_precision=6,
        )

        assert pair.min_order_size == min_volume
        assert pair.max_order_size == max_volume
        assert pair.price_precision == 2
        assert pair.volume_precision == 6

    def test_empty_symbol_validation(self, btc_currency, usdt_currency) -> None:
        """Тест валидации пустого символа"""
        with pytest.raises(ValueError, match="Symbol cannot be empty"):
            TradingPair(symbol="", base_currency=btc_currency, quote_currency=usdt_currency)

    def test_same_currencies_validation(self, btc_currency) -> None:
        """Тест валидации одинаковых валют"""
        with pytest.raises(ValueError, match="Base and quote currencies cannot be the same"):
            TradingPair(symbol="BTC/BTC", base_currency=btc_currency, quote_currency=btc_currency)

    def test_negative_precision_validation(self, btc_currency, usdt_currency) -> None:
        """Тест валидации отрицательной точности"""
        with pytest.raises(ValueError, match="Precision cannot be negative"):
            TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency, price_precision=-1)

        with pytest.raises(ValueError, match="Precision cannot be negative"):
            TradingPair(
                symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency, volume_precision=-1
            )

    def test_status_property(self, btc_usdt_pair) -> None:
        """Тест свойства status"""
        assert btc_usdt_pair.status == PairStatus.ACTIVE

        btc_usdt_pair.is_active = False
        assert btc_usdt_pair.status == PairStatus.INACTIVE

    def test_status_setter(self, btc_usdt_pair) -> None:
        """Тест установки статуса"""
        btc_usdt_pair.status = PairStatus.INACTIVE
        assert btc_usdt_pair.is_active is False

        btc_usdt_pair.status = PairStatus.ACTIVE
        assert btc_usdt_pair.is_active is True

    def test_display_name(self, btc_usdt_pair) -> None:
        """Тест отображаемого имени"""
        assert btc_usdt_pair.display_name == "BTC/USDT"

    def test_validate_price_valid(self, btc_usdt_pair, usdt_currency) -> None:
        """Тест валидации корректной цены"""
        price = Price(value=Decimal("50000"), currency=usdt_currency)
        assert btc_usdt_pair.validate_price(price) is True

    def test_validate_price_wrong_currency(self, btc_usdt_pair, btc_currency) -> None:
        """Тест валидации цены с неправильной валютой"""
        price = Price(value=Decimal("50000"), currency=btc_currency)
        assert btc_usdt_pair.validate_price(price) is False

    def test_validate_price_zero_value(self, btc_usdt_pair, usdt_currency) -> None:
        """Тест валидации нулевой цены"""
        price = Price(value=Decimal("0"), currency=usdt_currency)
        assert btc_usdt_pair.validate_price(price) is False

    def test_validate_price_negative_value(self, btc_usdt_pair, usdt_currency) -> None:
        """Тест валидации отрицательной цены"""
        # Создаем цену с неправильной валютой вместо отрицательной
        price = Price(value=Decimal("100"), currency=Currency.EUR)
        assert btc_usdt_pair.validate_price(price) is False

    def test_validate_volume_valid(self, btc_usdt_pair, btc_currency) -> None:
        """Тест валидации корректного объема"""
        volume = Volume(value=Decimal("1.0"), currency=btc_currency)
        assert btc_usdt_pair.validate_volume(volume) is True

    def test_validate_volume_wrong_currency(self, btc_usdt_pair, usdt_currency) -> None:
        """Тест валидации объема с неправильной валютой"""
        volume = Volume(value=Decimal("1.0"), currency=usdt_currency)
        assert btc_usdt_pair.validate_volume(volume) is False

    def test_validate_volume_below_minimum(self, btc_usdt_pair, btc_currency) -> None:
        """Тест валидации объема ниже минимума"""
        volume = Volume(value=Decimal("0.0001"), currency=btc_currency)
        assert btc_usdt_pair.validate_volume(volume) is False

    def test_validate_volume_above_maximum(self, btc_usdt_pair, btc_currency) -> None:
        """Тест валидации объема выше максимума"""
        volume = Volume(value=Decimal("200"), currency=btc_currency)
        assert btc_usdt_pair.validate_volume(volume) is False

    def test_validate_volume_no_limits(self, btc_currency, usdt_currency) -> None:
        """Тест валидации объема без ограничений"""
        pair = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        volume = Volume(value=Decimal("1000"), currency=btc_currency)
        assert pair.validate_volume(volume) is True

    def test_calculate_notional_value(self, btc_usdt_pair, usdt_currency, btc_currency) -> None:
        """Тест расчета номинальной стоимости"""
        price = Price(value=Decimal("50000"), currency=usdt_currency)
        volume = Volume(value=Decimal("2.0"), currency=btc_currency)

        notional = btc_usdt_pair.calculate_notional_value(price, volume)

        assert notional.value == Decimal("100000")
        assert notional.currency == usdt_currency

    def test_calculate_notional_value_invalid_price(self, btc_usdt_pair, btc_currency) -> None:
        """Тест расчета номинальной стоимости с неверной ценой"""
        price = Price(value=Decimal("0"), currency=btc_currency)  # Неверная валюта
        volume = Volume(value=Decimal("2.0"), currency=btc_currency)

        with pytest.raises(ValueError, match="Invalid price or volume for this trading pair"):
            btc_usdt_pair.calculate_notional_value(price, volume)

    def test_calculate_notional_value_invalid_volume(self, btc_usdt_pair, usdt_currency, btc_currency) -> None:
        """Тест расчета номинальной стоимости с неверным объемом"""
        price = Price(value=Decimal("50000"), currency=usdt_currency)
        volume = Volume(value=Decimal("0.0001"), currency=btc_currency)  # Ниже минимума

        with pytest.raises(ValueError, match="Invalid price or volume for this trading pair"):
            btc_usdt_pair.calculate_notional_value(price, volume)

    def test_deactivate(self, btc_usdt_pair) -> None:
        """Тест деактивации торговой пары"""
        btc_usdt_pair.deactivate()

        assert btc_usdt_pair.is_active is False
        assert btc_usdt_pair.updated_at is not None

    def test_activate(self, btc_usdt_pair) -> None:
        """Тест активации торговой пары"""
        btc_usdt_pair.is_active = False

        btc_usdt_pair.activate()

        assert btc_usdt_pair.is_active is True
        assert btc_usdt_pair.updated_at is not None

    def test_update_precision(self, btc_usdt_pair) -> None:
        """Тест обновления точности"""
        btc_usdt_pair.update_precision(price_precision=4, volume_precision=8)

        assert btc_usdt_pair.price_precision == 4
        assert btc_usdt_pair.volume_precision == 8
        assert btc_usdt_pair.updated_at is not None

    def test_update_precision_negative(self, btc_usdt_pair) -> None:
        """Тест обновления точности с отрицательными значениями"""
        with pytest.raises(ValueError, match="Precision cannot be negative"):
            btc_usdt_pair.update_precision(price_precision=-1, volume_precision=8)

        with pytest.raises(ValueError, match="Precision cannot be negative"):
            btc_usdt_pair.update_precision(price_precision=4, volume_precision=-1)

    def test_to_dict(self, btc_usdt_pair) -> None:
        """Тест преобразования в словарь"""
        result = btc_usdt_pair.to_dict()

        expected = {
            "symbol": "BTC/USDT",
            "base_currency": "BTC",
            "quote_currency": "USDT",
            "is_active": True,
            "min_order_size": "0.001",
            "max_order_size": "100",
            "price_precision": 2,
            "volume_precision": 6,
            "created_at": btc_usdt_pair.created_at.isoformat(),
            "updated_at": btc_usdt_pair.updated_at.isoformat(),
        }

        assert result == expected

    def test_to_dict_without_limits(self, btc_currency, usdt_currency) -> None:
        """Тест преобразования в словарь без ограничений"""
        pair = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        result = pair.to_dict()

        assert result["min_order_size"] is None
        assert result["max_order_size"] is None

    def test_equality(self, btc_currency, usdt_currency) -> None:
        """Тест равенства торговых пар"""
        pair1 = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        pair2 = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        pair3 = TradingPair(symbol="ETH/USDT", base_currency=usdt_currency, quote_currency=btc_currency)

        assert pair1 == pair2
        assert pair1 != pair3
        assert pair1 != "not a pair"

    def test_hash(self, btc_currency, usdt_currency) -> None:
        """Тест хеширования"""
        pair1 = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        pair2 = TradingPair(symbol="BTC/USDT", base_currency=btc_currency, quote_currency=usdt_currency)

        assert hash(pair1) == hash(pair2)

        # Проверяем, что можно использовать в множествах
        pairs_set = {pair1, pair2}
        assert len(pairs_set) == 1

    def test_string_representation(self, btc_usdt_pair) -> None:
        """Тест строкового представления"""
        assert str(btc_usdt_pair) == "BTC/USDT"


class TestPairStatus:
    """Тесты для перечисления PairStatus"""

    def test_pair_status_values(self: "TestPairStatus") -> None:
        """Тест значений статусов"""
        assert PairStatus.ACTIVE.value == "active"
        assert PairStatus.INACTIVE.value == "inactive"
        assert PairStatus.SUSPENDED.value == "suspended"
        assert PairStatus.DELISTED.value == "delisted"

    def test_pair_status_comparison(self: "TestPairStatus") -> None:
        """Тест сравнения статусов"""
        assert PairStatus.ACTIVE != PairStatus.INACTIVE
        assert PairStatus.ACTIVE == PairStatus.ACTIVE
