"""
Тесты для Value Objects с проверкой типизации и функциональности.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from datetime import datetime, timezone

from domain.value_objects.currency import Currency
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.percentage import Percentage
from domain.value_objects.timestamp import Timestamp
from domain.value_objects.signal import Signal
from domain.value_objects.signal_type import SignalType
from domain.value_objects.signal_strength import SignalStrength
from domain.value_objects.trading_pair import TradingPair
from domain.value_objects.factory import factory


class TestCurrency:
    """Тесты для Currency value object."""

    def test_currency_creation(self) -> None:
        """Тест создания валюты."""
        currency = Currency.BTC
        assert currency.currency_code == "BTC"
        assert currency.code == "BTC"
        assert str(currency) == "BTC"

    def test_currency_from_string(self) -> None:
        """Тест создания валюты из строки."""
        currency = Currency.from_string("ETH")
        assert currency == Currency.ETH
        
        currency = Currency.from_string("eth")
        assert currency == Currency.ETH
        
        currency = Currency.from_string("invalid")
        assert currency is None

    def test_currency_properties(self) -> None:
        """Тест свойств валюты."""
        assert Currency.USDT.is_stablecoin
        assert Currency.BTC.is_major_crypto
        assert Currency.USD.is_fiat
        assert Currency.BTC.is_crypto

    def test_currency_trading_priority(self) -> None:
        """Тест приоритета торговли."""
        assert Currency.BTC.trading_priority < Currency.ETH.trading_priority
        assert Currency.USDT.trading_priority < Currency.ADA.trading_priority

    def test_currency_trading_pairs(self) -> None:
        """Тест получения торговых пар."""
        pairs = Currency.get_trading_pairs(Currency.BTC)
        assert len(pairs) > 0
        assert Currency.USDT in pairs

    def test_currency_can_trade_with(self) -> None:
        """Тест возможности торговли между валютами."""
        assert Currency.BTC.can_trade_with(Currency.USDT)
        assert not Currency.BTC.can_trade_with(Currency.BTC)


class TestMoney:
    """Тесты для Money value object."""

    def test_money_creation(self) -> None:
        """Тест создания денежной суммы."""
        money = Money(100, Currency.USD)
        assert money.amount == Decimal("100")
        assert money.currency == Currency.USD
        assert str(money) == "100 USD"

    def test_money_arithmetic(self) -> None:
        """Тест арифметических операций."""
        money1 = Money(100, Currency.USD)
        money2 = Money(50, Currency.USD)
        
        result = money1 + money2
        assert result.amount == Decimal("150")
        assert result.currency == Currency.USD
        
        result = money1 - money2
        assert result.amount == Decimal("50")
        
        result = money1 * 2
        assert result.amount == Decimal("200")
        
        result = money1 / 2
        assert result.amount == Decimal("50")

    def test_money_comparison(self) -> None:
        """Тест сравнения денежных сумм."""
        money1 = Money(100, Currency.USD)
        money2 = Money(50, Currency.USD)
        
        assert money1 > money2
        assert money2 < money1
        assert money1 >= money2
        assert money2 <= money1

    def test_money_validation(self) -> None:
        """Тест валидации денежных сумм."""
        with pytest.raises(ValueError):
            Money(100, "INVALID")

    def test_money_serialization(self) -> None:
        """Тест сериализации."""
        money = Money(100.50, Currency.USD)
        data = money.to_dict()
        reconstructed = Money.from_dict(data)
        assert money == reconstructed

    def test_money_percentage_operations(self) -> None:
        """Тест процентных операций."""
        money = Money(100, Currency.USD)
        
        result = money.apply_percentage(Decimal("10"))
        assert result.amount == Decimal("10")
        
        result = money.increase_by_percentage(Decimal("10"))
        assert result.amount == Decimal("110")
        
        result = money.decrease_by_percentage(Decimal("10"))
        assert result.amount == Decimal("90")


class TestPrice:
    """Тесты для Price value object."""

    def test_price_creation(self) -> None:
        """Тест создания цены."""
        price = Price(50000, Currency.USD)
        assert price.amount == Decimal("50000")
        assert price.currency == Currency.USD

    def test_price_validation(self) -> None:
        """Тест валидации цены."""
        with pytest.raises(ValueError):
            Price(-100, Currency.USD)

    def test_price_percentage_change(self) -> None:
        """Тест расчета процентного изменения."""
        price1 = Price(100, Currency.USD)
        price2 = Price(110, Currency.USD)
        
        change = price2.percentage_change_from(price1)
        assert change == Decimal("10")

    def test_price_spread(self) -> None:
        """Тест расчета спреда."""
        price1 = Price(100, Currency.USD)
        price2 = Price(102, Currency.USD)
        
        spread = price1.spread_with(price2)
        assert spread.amount == Decimal("2")

    def test_price_slippage(self) -> None:
        """Тест применения проскальзывания."""
        price = Price(100, Currency.USD)
        buy_price, sell_price = price.apply_slippage(Decimal("1"))
        
        assert buy_price.amount < price.amount  # Цена покупки меньше (99)
        assert sell_price.amount > price.amount  # Цена продажи больше (101)


class TestVolume:
    """Тесты для Volume value object."""

    def test_volume_creation(self) -> None:
        """Тест создания объема."""
        volume = Volume(1000)
        assert volume.value == Decimal("1000")
        assert str(volume) == "1000.00000000"

    def test_volume_validation(self) -> None:
        """Тест валидации объема."""
        with pytest.raises(ValueError):
            Volume(-100)

    def test_volume_arithmetic(self) -> None:
        """Тест арифметических операций."""
        volume1 = Volume(1000)
        volume2 = Volume(500)
        
        result = volume1 + volume2
        assert result.value == Decimal("1500")
        
        result = volume1 - volume2
        assert result.value == Decimal("500")

    def test_volume_percentage(self) -> None:
        """Тест процентных операций."""
        volume = Volume(1000)
        total = Volume(2000)
        
        percentage = volume.percentage_of(total)
        assert percentage == Decimal("50")


class TestPercentage:
    """Тесты для Percentage value object."""

    def test_percentage_creation(self) -> None:
        """Тест создания процента."""
        percentage = Percentage(5.5)
        assert percentage.value == Decimal("5.5")
        assert str(percentage) == "5.50%"

    def test_percentage_fraction(self) -> None:
        """Тест преобразования в долю."""
        percentage = Percentage(50)
        assert percentage.to_fraction() == Decimal("0.5")

    def test_percentage_apply_to(self) -> None:
        """Тест применения процента к значению."""
        percentage = Percentage(10)
        result = percentage.apply_to(100)
        assert result == Decimal("10")

    def test_percentage_compound(self) -> None:
        """Тест сложного процента."""
        p1 = Percentage(10)
        p2 = Percentage(20)
        compound = p1.compound_with(p2)
        assert compound.value > Decimal("30")  # Сложный процент больше простого

    def test_percentage_annualize(self) -> None:
        """Тест годового процента."""
        percentage = Percentage(10)
        annual = percentage.annualize(30)  # 30 дней
        assert annual.value > percentage.value


class TestTimestamp:
    """Тесты для Timestamp value object."""

    def test_timestamp_creation(self) -> None:
        """Тест создания временной метки."""
        now = datetime.now(timezone.utc)
        timestamp = Timestamp(now)
        assert timestamp.value == now

    def test_timestamp_now(self) -> None:
        """Тест создания текущего времени."""
        timestamp = Timestamp.now()
        assert timestamp.is_now()

    def test_timestamp_arithmetic(self) -> None:
        """Тест арифметических операций."""
        timestamp = Timestamp.now()
        
        future = timestamp.add_hours(1)
        assert future > timestamp
        
        past = timestamp.subtract_hours(1)
        assert past < timestamp

    def test_timestamp_difference(self) -> None:
        """Тест разности времени."""
        t1 = Timestamp.now()
        t2 = t1.add_hours(1)
        
        diff = t2.time_difference(t1)
        assert diff == 3600  # 1 час в секундах

    def test_timestamp_rounding(self) -> None:
        """Тест округления времени."""
        timestamp = Timestamp.now()
        
        rounded_minute = timestamp.round_to_minute()
        assert rounded_minute.value.second == 0
        
        rounded_hour = timestamp.round_to_hour()
        assert rounded_hour.value.minute == 0

    def test_timestamp_trading_hours(self) -> None:
        """Тест торговых часов."""
        # Создаем время в торговые часы
        trading_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        timestamp = Timestamp(trading_time)
        assert timestamp.is_trading_hours()


class TestSignal:
    """Тесты для Signal value object."""

    def test_signal_creation(self) -> None:
        """Тест создания сигнала."""
        from domain.value_objects.trading_pair import TradingPair
        from domain.value_objects.currency import Currency
        
        trading_pair = TradingPair(Currency.BTC, Currency.USDT)
        signal = Signal(
            direction=SignalType.BUY,
            signal_type=SignalType.BUY,
            strength=Decimal("0.8"),
            confidence=Decimal("0.7"),
            trading_pair=trading_pair
        )
        assert signal.signal_type == SignalType.BUY
        assert signal.is_buy_signal
        assert signal.is_strong_signal

    def test_signal_factory_methods(self) -> None:
        """Тест фабричных методов."""
        from domain.value_objects.trading_pair import TradingPair
        from domain.value_objects.currency import Currency
        
        trading_pair = TradingPair(Currency.BTC, Currency.USDT)
        
        buy_signal = Signal.create_buy_signal(trading_pair)
        assert buy_signal.is_buy_signal
        
        sell_signal = Signal.create_sell_signal(trading_pair)
        assert sell_signal.is_sell_signal
        
        hold_signal = Signal.create_hold_signal(trading_pair)
        assert hold_signal.is_hold_signal

    def test_signal_properties(self) -> None:
        """Тест свойств сигнала."""
        from domain.value_objects.trading_pair import TradingPair
        from domain.value_objects.currency import Currency
        
        trading_pair = TradingPair(Currency.BTC, Currency.USDT)
        signal = Signal(
            direction=SignalType.BUY,
            signal_type=SignalType.BUY,
            strength=Decimal("0.5"),
            confidence=Decimal("0.6"),
            trading_pair=trading_pair
        )
        
        assert signal.is_trading_signal
        assert not signal.is_weak_signal

    def test_signal_scoring(self) -> None:
        """Тест скоринга сигнала."""
        from domain.value_objects.trading_pair import TradingPair
        from domain.value_objects.currency import Currency
        
        trading_pair = TradingPair(Currency.BTC, Currency.USDT)
        signal = Signal(
            direction=SignalType.BUY,
            signal_type=SignalType.BUY,
            strength=Decimal("0.8"),
            confidence=Decimal("0.8"),
            trading_pair=trading_pair
        )
        
        score = signal.get_combined_score()
        assert 0 < score <= 1


class TestTradingPair:
    """Тесты для TradingPair value object."""

    def test_trading_pair_creation(self) -> None:
        """Тест создания торговой пары."""
        pair = TradingPair(Currency.BTC, Currency.USDT)
        assert pair.base_currency == Currency.BTC
        assert pair.quote_currency == Currency.USDT
        assert pair.symbol == "BTCUSDT"

    def test_trading_pair_validation(self) -> None:
        """Тест валидации торговой пары."""
        with pytest.raises(ValueError):
            TradingPair(Currency.BTC, Currency.BTC)

    def test_trading_pair_properties(self) -> None:
        """Тест свойств торговой пары."""
        pair = TradingPair(Currency.BTC, Currency.USDT)
        
        assert pair.is_crypto_pair
        assert pair.is_stablecoin_pair
        assert pair.is_major_pair

    def test_trading_pair_from_symbol(self) -> None:
        """Тест создания из символа."""
        pair = TradingPair.from_symbol("BTCUSDT")
        assert pair.base_currency == Currency.BTC
        assert pair.quote_currency == Currency.USDT

    def test_trading_pair_reverse(self) -> None:
        """Тест обратной пары."""
        pair = TradingPair(Currency.BTC, Currency.USDT)
        reverse = pair.get_reverse_pair()
        assert reverse.base_currency == Currency.USDT
        assert reverse.quote_currency == Currency.BTC


class TestFactory:
    """Тесты для фабрики value objects."""

    def test_factory_money_creation(self) -> None:
        """Тест создания Money через фабрику."""
        money = factory.create_money(100, Currency.USD)
        assert isinstance(money, Money)
        assert money.amount == Decimal("100")

    def test_factory_price_creation(self) -> None:
        """Тест создания Price через фабрику."""
        price = factory.create_price(50000, Currency.USD)
        assert isinstance(price, Price)
        assert price.amount == Decimal("50000")

    def test_factory_volume_creation(self) -> None:
        """Тест создания Volume через фабрику."""
        volume = factory.create_volume(1000)
        assert isinstance(volume, Volume)
        assert volume.value == Decimal("1000")

    def test_factory_percentage_creation(self) -> None:
        """Тест создания Percentage через фабрику."""
        percentage = factory.create_percentage("10.5%")
        assert isinstance(percentage, Percentage)
        assert percentage.value == Decimal("10.5")

    def test_factory_timestamp_creation(self) -> None:
        """Тест создания Timestamp через фабрику."""
        timestamp = factory.create_timestamp("2024-01-15T10:00:00Z")
        assert isinstance(timestamp, Timestamp)

    def test_factory_trading_pair_creation(self) -> None:
        """Тест создания TradingPair через фабрику."""
        pair = factory.create_trading_pair("BTC", "USDT")
        assert isinstance(pair, TradingPair)
        assert pair.base_currency == Currency.BTC

    def test_factory_signal_creation(self) -> None:
        """Тест создания Signal через фабрику."""
        signal = factory.create_signal("BUY", Timestamp.now(), "STRONG")
        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.BUY

    def test_factory_serialization(self) -> None:
        """Тест сериализации через фабрику."""
        money = Money(100, Currency.USD)
        data = factory.to_dict(money)
        reconstructed = factory.from_dict(data)
        assert money == reconstructed

    def test_factory_validation(self) -> None:
        """Тест валидации через фабрику."""
        money = Money(100, Currency.USD)
        assert factory.validate(money)


class TestIntegration:
    """Интеграционные тесты."""

    def test_value_objects_workflow(self) -> None:
        """Тест рабочего процесса с value objects."""
        # Создаем торговую пару
        pair = TradingPair(Currency.BTC, Currency.USDT)
        
        # Создаем цену и объем
        price = Price(50000, Currency.USD)
        volume = Volume(1.5)
        
        # Создаем временную метку
        timestamp = Timestamp.now()
        
        # Создаем сигнал
        signal = Signal(
            direction=SignalType.BUY,
            signal_type=SignalType.BUY,
            strength=Decimal("0.8"),
            confidence=Decimal("0.7"),
            trading_pair=pair,
            price=price.amount,
            volume=volume.value
        )
        
        # Проверяем, что все работает вместе
        assert signal.is_buy_signal
        assert signal.price == price.amount
        assert signal.volume == volume.value
        # Проверяем, что timestamp создан (не проверяем точное время из-за задержки)
        assert signal.timestamp is not None

    def test_serialization_workflow(self) -> None:
        """Тест рабочего процесса сериализации."""
        # Создаем объекты
        money = Money(100, Currency.USD)
        price = Price(50000, Currency.USD)
        volume = Volume(1.5)
        timestamp = Timestamp.now()
        
        # Сериализуем
        objects = [money, price, volume, timestamp]
        serialized = [obj.to_dict() for obj in objects]
        
        # Десериализуем
        deserialized = [factory.from_dict(data) for data in serialized]
        
        # Проверяем равенство
        for original, reconstructed in zip(objects, deserialized):
            assert original == reconstructed 
