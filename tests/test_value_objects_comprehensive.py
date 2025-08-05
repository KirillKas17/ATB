"""
Комплексные тесты для промышленных Value Objects с расширенной функциональностью для алготрейдинга.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from domain.value_objects import (
    Currency, CurrencyType, CurrencyNetwork,
    Money, MoneyConfig,
    Price, PriceConfig,
    Volume,
    Percentage,
    Timestamp,
    Signal, SignalType, SignalStrength,
    TradingPair,
    ValueObjectFactory, factory
)
from domain.type_definitions.value_object_types import (
    ValidationResult, PrecisionMode, PrecisionConfig,
    ValueObjectEvent, ValueObjectEventData, ValueObjectMetrics
)
class TestCurrency:
    """Тесты для Currency value object."""
    def test_currency_creation(self: "TestCurrency") -> None:
        """Тест создания валюты."""
        currency = Currency(Currency.BTC)
        assert currency.code == "BTC"
        assert currency.name == "Bitcoin"
        assert currency.type == CurrencyType.CRYPTO
        assert CurrencyNetwork.BITCOIN in currency.networks
    def test_currency_validation(self: "TestCurrency") -> None:
        """Тест валидации валюты."""
        # Валидная валюта
        currency = Currency(Currency.ETH)
        validation = currency.validate()
        assert validation.is_valid
        # Невалидная валюта (слишком короткий код)
        with pytest.raises(ValueError):
            Currency("A")
    def test_currency_caching(self: "TestCurrency") -> None:
        """Тест кэширования валют."""
        currency1 = Currency(Currency.USDT)
        currency2 = Currency(Currency.USDT)
        assert currency1 is currency2  # Должны быть одним объектом
    def test_currency_trading_methods(self: "TestCurrency") -> None:
        """Тест торговых методов валюты."""
        currency = Currency(Currency.BTC)
        assert currency.is_crypto()
        assert not currency.is_fiat()
        assert not currency.is_stablecoin()
        assert currency.supports_network(CurrencyNetwork.BITCOIN)
        assert currency.get_trading_precision() == 8
        assert currency.get_min_order_size() == Decimal("0.0001")
        assert currency.is_valid_for_trading()
    def test_currency_risk_analysis(self: "TestCurrency") -> None:
        """Тест анализа рисков валюты."""
        btc = Currency(Currency.BTC)
        usdt = Currency(Currency.USDT)
        assert btc.get_risk_score() > usdt.get_risk_score()
        assert usdt.get_liquidity_score() >= btc.get_liquidity_score()
class TestMoney:
    """Тесты для Money value object."""
    def test_money_creation(self: "TestMoney") -> None:
        """Тест создания денежной суммы."""
        money = Money(Decimal("100.50"), Currency.USD)
        assert money.amount == Decimal("100.50")
        assert money.currency.code == "USD"
        assert str(money) == "100.50 USD"
    def test_money_validation(self: "TestMoney") -> None:
        """Тест валидации денежной суммы."""
        # Валидная сумма
        money = Money(Decimal("100"), Currency.USD)
        validation = money.validate()
        assert validation.is_valid
        # Невалидная сумма (отрицательная в строгом режиме)
        config = MoneyConfig(allow_negative=False)
        with pytest.raises(ValueError):
            Money(Decimal("-100"), Currency.USD, config)
    def test_money_arithmetic(self: "TestMoney") -> None:
        """Тест арифметических операций."""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("50"), Currency.USD)
        # Сложение
        result = money1 + money2
        assert result.amount == Decimal("150")
        assert result.currency == money1.currency
        # Вычитание
        result = money1 - money2
        assert result.amount == Decimal("50")
        # Умножение
        result = money1 * 2
        assert result.amount == Decimal("200")
        # Деление
        result = money1 / 2
        assert result.amount == Decimal("50")
    def test_money_currency_conversion(self: "TestMoney") -> None:
        """Тест конвертации валют."""
        usd_money = Money(Decimal("100"), Currency.USD)
        eur_money = usd_money.convert_to(Currency.EUR, Decimal("0.85"))
        assert eur_money.currency.code == "EUR"
        assert eur_money.amount == Decimal("85.00")
    def test_money_trading_methods(self: "TestMoney") -> None:
        """Тест торговых методов."""
        money = Money(Decimal("1000"), Currency.USD)
        assert money.is_valid_for_trading()
        assert money.get_position_risk_score() <= 10
        assert money.get_margin_requirement(Decimal("2")).amount == Decimal("500")
        assert money.get_stop_loss_amount(Decimal("0.1")).amount == Decimal("100")
    def test_money_caching(self: "TestMoney") -> None:
        """Тест кэширования денежных сумм."""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("100"), Currency.USD)
        assert money1 is money2  # Должны быть одним объектом
class TestPrice:
    """Тесты для Price value object."""
    def test_price_creation(self: "TestPrice") -> None:
        """Тест создания цены."""
        price = Price(Decimal("50000"), Currency.BTC, Currency.USD)
        assert price.amount == Decimal("50000")
        assert price.base_currency.code == "BTC"
        assert price.quote_currency.code == "USD"
        assert str(price) == "50000 BTC/USD"
    def test_price_validation(self: "TestPrice") -> None:
        """Тест валидации цены."""
        # Валидная цена
        price = Price(Decimal("50000"), Currency.BTC, Currency.USD)
        validation = price.validate()
        assert validation.is_valid
        # Невалидная цена (нулевая)
        with pytest.raises(ValueError):
            Price(Decimal("0"), Currency.BTC, Currency.USD)
    def test_price_arithmetic(self: "TestPrice") -> None:
        """Тест арифметических операций."""
        price1 = Price(Decimal("50000"), Currency.BTC, Currency.USD)
        price2 = Price(Decimal("1000"), Currency.BTC, Currency.USD)
        # Сложение
        result = price1 + price2
        assert result.amount == Decimal("51000")
        # Вычитание
        result = price1 - price2
        assert result.amount == Decimal("49000")
        # Умножение
        result = price1 * 2
        assert result.amount == Decimal("100000")
        # Деление
        result = price1 / 2
        assert result.amount == Decimal("25000")
    def test_price_trading_methods(self: "TestPrice") -> None:
        """Тест торговых методов."""
        price = Price(Decimal("50000"), Currency.BTC, Currency.USD)
        assert price.is_valid_for_trading()
        # Bid/Ask цены
        bid_price = price.get_bid_price(Decimal("0.001"))
        ask_price = price.get_ask_price(Decimal("0.001"))
        assert bid_price.amount < price.amount
        assert ask_price.amount > price.amount
        # Спред
        spread = price.get_spread(bid_price, ask_price)
        assert spread > 0
    def test_price_analysis(self: "TestPrice") -> None:
        """Тест аналитических методов."""
        price1 = Price(Decimal("50000"), Currency.BTC, Currency.USD)
        price2 = Price(Decimal("51000"), Currency.BTC, Currency.USD)
        # Направление тренда
        trend = price2.get_trend_direction(price1)
        assert trend == "up"
        # Изменение цены
        change = price2.get_price_change_percentage(price1)
        assert change == Decimal("0.02")  # 2%
class TestVolume:
    """Тесты для Volume value object."""
    def test_volume_creation(self: "TestVolume") -> None:
        """Тест создания объема."""
        volume = Volume(Decimal("1000"), Currency.BTC)
        assert volume.amount == Decimal("1000")
        assert volume.currency.code == "BTC"
    def test_volume_validation(self: "TestVolume") -> None:
        """Тест валидации объема."""
        # Валидный объем
        volume = Volume(Decimal("1000"), Currency.BTC)
        assert volume.amount > 0
        # Невалидный объем (отрицательный)
        with pytest.raises(ValueError):
            Volume(Decimal("-1000"), Currency.BTC)
    def test_volume_arithmetic(self: "TestVolume") -> None:
        """Тест арифметических операций."""
        volume1 = Volume(Decimal("1000"), Currency.BTC)
        volume2 = Volume(Decimal("500"), Currency.BTC)
        # Сложение
        result = volume1 + volume2
        assert result.amount == Decimal("1500")
        # Вычитание
        result = volume1 - volume2
        assert result.amount == Decimal("500")
        # Умножение
        result = volume1 * 2
        assert result.amount == Decimal("2000")
    def test_volume_liquidity_analysis(self: "TestVolume") -> None:
        """Тест анализа ликвидности."""
        high_volume = Volume(Decimal("2000000"), Currency.BTC)
        low_volume = Volume(Decimal("1000"), Currency.BTC)
        assert high_volume.get_liquidity_level() == "HIGH"
        assert low_volume.get_liquidity_level() == "VERY_LOW"
        assert high_volume.is_liquid()
        assert not low_volume.is_liquid()
class TestPercentage:
    """Тесты для Percentage value object."""
    def test_percentage_creation(self: "TestPercentage") -> None:
        """Тест создания процента."""
        percentage = Percentage(Decimal("5.5"))
        assert percentage.amount == Decimal("5.5")
        assert str(percentage) == "5.5%"
    def test_percentage_validation(self: "TestPercentage") -> None:
        """Тест валидации процента."""
        # Валидный процент
        percentage = Percentage(Decimal("5.5"))
        assert percentage.amount > 0
        # Невалидный процент (слишком большой)
        with pytest.raises(ValueError):
            Percentage(Decimal("15000"))
    def test_percentage_arithmetic(self: "TestPercentage") -> None:
        """Тест арифметических операций."""
        p1 = Percentage(Decimal("10"))
        p2 = Percentage(Decimal("5"))
        # Сложение
        result = p1 + p2
        assert result.amount == Decimal("15")
        # Вычитание
        result = p1 - p2
        assert result.amount == Decimal("5")
        # Умножение
        result = p1 * 2
        assert result.amount == Decimal("20")
    def test_percentage_risk_analysis(self: "TestPercentage") -> None:
        """Тест анализа рисков."""
        low_risk = Percentage(Decimal("2"))
        high_risk = Percentage(Decimal("30"))
        assert low_risk.get_risk_level() == "LOW"
        assert high_risk.get_risk_level() == "HIGH"
        assert low_risk.is_acceptable_risk()
        assert not high_risk.is_acceptable_risk()
class TestTimestamp:
    """Тесты для Timestamp value object."""
    def test_timestamp_creation(self: "TestTimestamp") -> None:
        """Тест создания временной метки."""
        now = datetime.now(timezone.utc)
        timestamp = Timestamp(now)
        assert timestamp.value == now
        assert timestamp.to_iso() == now.isoformat()
    def test_timestamp_validation(self: "TestTimestamp") -> None:
        """Тест валидации временной метки."""
        # Валидная метка
        timestamp = Timestamp("2023-01-01T00:00:00+00:00")
        assert timestamp.value.tzinfo is not None
        # Невалидная метка
        with pytest.raises(ValueError):
            Timestamp("invalid-date")
    def test_timestamp_operations(self: "TestTimestamp") -> None:
        """Тест временных операций."""
        timestamp = Timestamp("2023-01-01T00:00:00+00:00")
        # Добавление времени
        future = timestamp.add_days(1)
        assert future.value > timestamp.value
        # Разность времени
        diff = future.time_difference_days(timestamp)
        assert diff == 1.0
    def test_timestamp_trading_analysis(self: "TestTimestamp") -> None:
        """Тест торгового анализа времени."""
        timestamp = Timestamp("2023-01-02T10:00:00+00:00")  # Понедельник 10:00
        assert timestamp.is_weekday()
        assert not timestamp.is_weekend()
        assert timestamp.is_trading_hours()
class TestSignal:
    """Тесты для Signal value object."""
    def test_signal_creation(self: "TestSignal") -> None:
        """Тест создания сигнала."""
        timestamp = Timestamp.now()
        signal = Signal(
            signal_type=SignalType.BUY,
            timestamp=timestamp,
            strength=SignalStrength.STRONG,
            confidence=Percentage(Decimal("80"))
        )
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.is_buy_signal()
        assert signal.is_strong_signal()
    def test_signal_validation(self: "TestSignal") -> None:
        """Тест валидации сигнала."""
        timestamp = Timestamp.now()
        # Валидный сигнал
        signal = Signal(SignalType.BUY, timestamp)
        assert signal.signal_id is not None
        # Невалидный сигнал (без временной метки)
        with pytest.raises(ValueError):
            Signal(SignalType.BUY, None)
    def test_signal_analysis(self: "TestSignal") -> None:
        """Тест анализа сигнала."""
        timestamp = Timestamp.now()
        signal = Signal(
            signal_type=SignalType.BUY,
            timestamp=timestamp,
            strength=SignalStrength.STRONG,
            confidence=Percentage(Decimal("80"))
        )
        assert signal.get_risk_level() in ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
        assert signal.get_trading_recommendation() is not None
        assert signal.get_combined_score() > 0
    def test_signal_factory_methods(self: "TestSignal") -> None:
        """Тест фабричных методов сигналов."""
        timestamp = Timestamp.now()
        # Buy сигнал
        buy_signal = Signal.create_buy_signal(
            timestamp=timestamp,
            strength=SignalStrength.STRONG,
            price=Decimal("50000")
        )
        assert buy_signal.is_buy_signal()
        # Sell сигнал
        sell_signal = Signal.create_sell_signal(
            timestamp=timestamp,
            strength=SignalStrength.MODERATE
        )
        assert sell_signal.is_sell_signal()
class TestTradingPair:
    """Тесты для TradingPair value object."""
    def test_trading_pair_creation(self: "TestTradingPair") -> None:
        """Тест создания торговой пары."""
        pair = TradingPair(Currency.BTC, Currency.USD)
        assert pair.base_currency.code == "BTC"
        assert pair.quote_currency.code == "USD"
        assert pair.symbol == "BTCUSD"
    def test_trading_pair_validation(self: "TestTradingPair") -> None:
        """Тест валидации торговой пары."""
        # Валидная пара
        pair = TradingPair(Currency.BTC, Currency.USD)
        assert pair.is_active
        # Невалидная пара (одинаковые валюты)
        with pytest.raises(ValueError):
            TradingPair(Currency.BTC, Currency.BTC)
    def test_trading_pair_analysis(self: "TestTradingPair") -> None:
        """Тест анализа торговой пары."""
        btc_usdt = TradingPair(Currency.BTC, Currency.USDT)
        btc_usd = TradingPair(Currency.BTC, Currency.USD)
        assert btc_usdt.is_crypto_pair()
        assert btc_usd.is_crypto_fiat_pair()
        assert btc_usdt.is_stablecoin_pair()
        assert btc_usdt.get_trading_priority() < btc_usd.get_trading_priority()
    def test_trading_pair_factory_methods(self: "TestTradingPair") -> None:
        """Тест фабричных методов торговых пар."""
        btc_usdt = TradingPair.create_btc_usdt()
        assert btc_usdt.base_currency.code == "BTC"
        assert btc_usdt.quote_currency.code == "USDT"
        eth_usdt = TradingPair.create_eth_usdt()
        assert eth_usdt.base_currency.code == "ETH"
        assert eth_usdt.quote_currency.code == "USDT"
class TestValueObjectFactory:
    """Тесты для ValueObjectFactory."""
    def test_factory_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания фабрики."""
        factory = ValueObjectFactory()
        assert factory is not None
        assert len(factory.get_registered_types()) > 0
    def test_factory_money_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания Money через фабрику."""
        money = factory.create_money(Decimal("100"), Currency.USD)
        assert isinstance(money, Money)
        assert money.amount == Decimal("100")
        assert money.currency.code == "USD"
    def test_factory_price_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания Price через фабрику."""
        price = factory.create_price(Decimal("50000"), Currency.BTC, Currency.USD)
        assert isinstance(price, Price)
        assert price.amount == Decimal("50000")
        assert price.base_currency.code == "BTC"
        assert price.quote_currency.code == "USD"
    def test_factory_volume_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания Volume через фабрику."""
        volume = factory.create_volume(Decimal("1000"), Currency.BTC)
        assert isinstance(volume, Volume)
        assert volume.amount == Decimal("1000")
        assert volume.currency.code == "BTC"
    def test_factory_percentage_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания Percentage через фабрику."""
        percentage = factory.create_percentage(Decimal("5.5"))
        assert isinstance(percentage, Percentage)
        assert percentage.amount == Decimal("5.5")
    def test_factory_timestamp_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания Timestamp через фабрику."""
        timestamp = factory.create_timestamp("2023-01-01T00:00:00+00:00")
        assert isinstance(timestamp, Timestamp)
        assert timestamp.value.year == 2023
    def test_factory_trading_pair_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания TradingPair через фабрику."""
        pair = factory.create_trading_pair(Currency.BTC, Currency.USD)
        assert isinstance(pair, TradingPair)
        assert pair.base_currency.code == "BTC"
        assert pair.quote_currency.code == "USD"
    def test_factory_signal_creation(self: "TestValueObjectFactory") -> None:
        """Тест создания Signal через фабрику."""
        timestamp = Timestamp.now()
        signal = factory.create_signal(
            SignalType.BUY,
            timestamp,
            SignalStrength.STRONG
        )
        assert isinstance(signal, Signal)
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
    def test_factory_serialization(self: "TestValueObjectFactory") -> None:
        """Тест сериализации через фабрику."""
        money = Money(Decimal("100"), Currency.USD)
        data = factory.to_dict(money)
        assert data["type"] == "Money"
        assert data["amount"] == "100"
        assert data["currency"] == "USD"
        # Десериализация
        restored_money = factory.from_dict(data)
        assert isinstance(restored_money, Money)
        assert restored_money.amount == money.amount
        assert restored_money.currency == money.currency
    def test_factory_caching(self: "TestValueObjectFactory") -> None:
        """Тест кэширования фабрики."""
        money1 = factory.create_money(Decimal("100"), Currency.USD)
        money2 = factory.create_money(Decimal("100"), Currency.USD)
        assert money1 is money2  # Должны быть одним объектом
        stats = factory.get_cache_stats()
        assert stats["cache_size"] > 0
    def test_factory_performance_stats(self: "TestValueObjectFactory") -> None:
        """Тест статистики производительности фабрики."""
        # Создаем несколько объектов
        for _ in range(10):
            factory.create_money(Decimal("100"), Currency.USD)
        stats = factory.get_performance_stats()
        assert stats["success_count"] > 0
        assert "average_creation_time" in stats
class TestValueObjectTypes:
    """Тесты для типов value objects."""
    def test_validation_result(self: "TestValueObjectTypes") -> None:
        """Тест ValidationResult."""
        result = ValidationResult(is_valid=True)
        assert bool(result) is True
        result = ValidationResult(is_valid=False, errors=["Error"])
        assert bool(result) is False
        assert len(result.errors) == 1
    def test_precision_config(self: "TestValueObjectTypes") -> None:
        """Тест PrecisionConfig."""
        config = PrecisionConfig(
            mode=PrecisionMode.TRADING,
            decimal_places=8,
            rounding_mode="ROUND_HALF_UP"
        )
        assert config.mode == PrecisionMode.TRADING
        assert config.decimal_places == 8
        assert config.get_rounding_mode() == ROUND_HALF_UP
    def test_value_object_events(self: "TestValueObjectTypes") -> None:
        """Тест событий value objects."""
        event = ValueObjectEvent.CREATED
        assert event.name == "CREATED"
        event_data = ValueObjectEventData(
            event_type=ValueObjectEvent.CREATED,
            value_object=Money(Decimal("100"), Currency.USD),
            timestamp=datetime.now(timezone.utc)
        )
        assert event_data.event_type == ValueObjectEvent.CREATED
    def test_value_object_metrics(self: "TestValueObjectTypes") -> None:
        """Тест метрик value objects."""
        metrics = ValueObjectMetrics()
        metrics.update_creation_time(100.0)
        metrics.update_validation_time(50.0)
        stats = metrics.to_dict()
        assert stats["total_created"] == 1
        assert stats["total_validated"] == 1
        assert stats["average_creation_time_ms"] == 100.0
class TestIntegration:
    """Интеграционные тесты."""
    def test_trading_scenario(self: "TestIntegration") -> None:
        """Тест торгового сценария."""
        # Создаем торговую пару
        pair = TradingPair(Currency.BTC, Currency.USD)
        # Создаем цену
        price = Price(Decimal("50000"), Currency.BTC, Currency.USD)
        # Создаем объем
        volume = Volume(Decimal("1"), Currency.BTC)
        # Создаем сигнал
        timestamp = Timestamp.now()
        signal = Signal(
            signal_type=SignalType.BUY,
            timestamp=timestamp,
            strength=SignalStrength.STRONG,
            confidence=Percentage(Decimal("80")),
            price=price.amount,
            volume=volume.amount
        )
        # Проверяем валидность
        assert pair.is_valid_for_trading()
        assert price.is_valid_for_trading()
        assert signal.is_buy_signal()
        assert signal.is_strong_signal()
    def test_risk_management_scenario(self: "TestIntegration") -> None:
        """Тест сценария риск-менеджмента."""
        # Создаем позицию
        position_size = Money(Decimal("10000"), Currency.USD)
        # Рассчитываем риск
        risk_score = position_size.get_position_risk_score()
        margin_requirement = position_size.get_margin_requirement(Decimal("2"))
        stop_loss = position_size.get_stop_loss_amount(Decimal("0.1"))
        assert 1 <= risk_score <= 10
        assert margin_requirement.amount == Decimal("5000")
        assert stop_loss.amount == Decimal("1000")
    def test_performance_optimization(self: "TestIntegration") -> None:
        """Тест оптимизации производительности."""
        # Очищаем кэш
        Currency.clear_cache()
        Money.clear_cache()
        Price.clear_cache()
        # Создаем множество объектов
        currencies = []
        money_objects = []
        prices = []
        for i in range(100):
            currency = Currency(Currency.BTC)
            money = Money(Decimal(str(i)), currency)
            price = Price(Decimal(str(i * 1000)), currency, Currency.USD)
            currencies.append(currency)
            money_objects.append(money)
            prices.append(price)
        # Проверяем кэширование
        currency_stats = Currency.get_cache_stats()
        money_stats = Money.get_cache_stats()
        price_stats = Price.get_cache_stats()
        assert currency_stats["currencies"] > 0
        assert money_stats["money_objects"] > 0
        assert price_stats["price_objects"] > 0
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
