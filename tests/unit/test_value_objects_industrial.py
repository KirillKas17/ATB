"""
Комплексные тесты для промышленной реализации Value Objects.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from datetime import datetime, timezone
from domain.value_objects import (
    Currency, Money, Price, Volume, Percentage, Timestamp, 
    Signal, SignalType, SignalStrength, TradingPair
)
from domain.value_objects.factory import ValueObjectFactory, factory
from domain.type_definitions.value_object_types import ValidationContext
class TestCurrencyIndustrial:
    """Тесты для промышленной реализации Currency."""
    def test_currency_creation(self) -> None:
        """Тест создания валюты."""
        currency = Currency.BTC
        assert currency.currency_code == "BTC"
        assert currency.code == "BTC"
        assert currency.is_crypto
        assert not currency.is_fiat
        assert not currency.is_stablecoin
    def test_stablecoin_properties(self) -> None:
        """Тест свойств стейблкоинов."""
        usdt = Currency.USDT
        assert usdt.is_stablecoin
        assert usdt.is_crypto
        assert not usdt.is_fiat
        assert usdt.trading_priority < 10
    def test_fiat_properties(self) -> None:
        """Тест свойств фиатных валют."""
        usd = Currency.USD
        assert usd.is_fiat
        assert not usd.is_crypto
        assert not usd.is_stablecoin
        assert usd.trading_priority > 50
    def test_volatility_rating(self) -> None:
        """Тест рейтинга волатильности."""
        btc = Currency.BTC
        usdt = Currency.USDT
        usd = Currency.USD
        assert btc.volatility_rating > usdt.volatility_rating
        assert usdt.volatility_rating > usd.volatility_rating
    def test_liquidity_rating(self) -> None:
        """Тест рейтинга ликвидности."""
        btc = Currency.BTC
        usdt = Currency.USDT
        ada = Currency.ADA
        assert btc.liquidity_rating > ada.liquidity_rating
        assert usdt.liquidity_rating > ada.liquidity_rating
    def test_risk_score(self) -> None:
        """Тест оценки риска."""
        btc = Currency.BTC
        usdt = Currency.USDT
        ada = Currency.ADA
        assert usdt.get_risk_score() < btc.get_risk_score()
        assert btc.get_risk_score() < ada.get_risk_score()
    def test_trading_recommendation(self) -> None:
        """Тест торговых рекомендаций."""
        usdt = Currency.USDT
        btc = Currency.BTC
        ada = Currency.ADA
        assert usdt.get_trading_recommendation() == "SAFE_FOR_NEWBIES"
        assert btc.get_trading_recommendation() in ["MODERATE_RISK", "SAFE_FOR_NEWBIES"]
        assert ada.get_trading_recommendation() in ["HIGH_RISK", "VERY_HIGH_RISK"]
    def test_from_string(self) -> None:
        """Тест создания из строки."""
        currency = Currency.from_string("BTC")
        assert currency == Currency.BTC
        invalid_currency = Currency.from_string("INVALID")
        assert invalid_currency is None
    def test_get_trading_pairs(self) -> None:
        """Тест получения торговых пар."""
        btc_pairs = Currency.get_trading_pairs(Currency.BTC)
        assert len(btc_pairs) > 0
        assert all(pair != Currency.BTC for pair in btc_pairs)
        assert btc_pairs[0].trading_priority <= btc_pairs[-1].trading_priority
    def test_can_trade_with(self) -> None:
        """Тест возможности торговли."""
        assert Currency.BTC.can_trade_with(Currency.USDT)
        assert Currency.USDT.can_trade_with(Currency.BTC)
        assert not Currency.BTC.can_trade_with(Currency.BTC)
        assert Currency.USD.can_trade_with(Currency.BTC)
    def test_serialization(self) -> None:
        """Тест сериализации."""
        currency = Currency.BTC
        data = currency.to_dict()
        assert data["value"] == "BTC"
        assert data["type"] == "Currency"
        assert data["is_crypto"] is True
        assert data["is_fiat"] is False
class TestMoneyIndustrial:
    """Тесты для промышленной реализации Money."""
    def test_money_creation(self) -> None:
        """Тест создания денежной суммы."""
        money = Money(Decimal("100.50"), Currency.USD)
        assert money.amount == Decimal("100.50")
        assert money.currency == Currency.USD
        assert str(money) == "100.50 USD"
    def test_money_validation(self) -> None:
        """Тест валидации денежных сумм."""
        # Валидные суммы
        Money(Decimal("100"), Currency.USD)
        Money(Decimal("0"), Currency.USD)
        # Невалидные суммы
        with pytest.raises(ValueError):
            Money(Decimal("NaN"), Currency.USD)
        with pytest.raises(ValueError):
            Money(Decimal("inf"), Currency.USD)
    def test_money_arithmetic(self) -> None:
        """Тест арифметических операций."""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("50"), Currency.USD)
        # Сложение
        result = money1 + money2
        assert result.amount == Decimal("150")
        assert result.currency == Currency.USD
        # Вычитание
        result = money1 - money2
        assert result.amount == Decimal("50")
        # Умножение
        result = money1 * 2
        assert result.amount == Decimal("200")
        # Деление
        result = money1 / 2
        assert result.amount == Decimal("50")
    def test_money_comparison(self) -> None:
        """Тест сравнения денежных сумм."""
        money1 = Money(Decimal("100"), Currency.USD)
        money2 = Money(Decimal("50"), Currency.USD)
        money3 = Money(Decimal("100"), Currency.USD)
        assert money1 > money2
        assert money2 < money1
        assert money1 == money3
        assert money1 >= money3
        assert money2 <= money1
    def test_money_currency_mismatch(self) -> None:
        """Тест несоответствия валют."""
        usd_money = Money(Decimal("100"), Currency.USD)
        eur_money = Money(Decimal("100"), Currency.EUR)
        with pytest.raises(ValueError):
            usd_money + eur_money
        with pytest.raises(ValueError):
            usd_money - eur_money
        with pytest.raises(ValueError):
            usd_money < eur_money
    def test_money_percentage_operations(self) -> None:
        """Тест процентных операций."""
        money = Money(Decimal("100"), Currency.USD)
        # Процент от суммы
        percentage = money.percentage_of(Money(Decimal("200"), Currency.USD))
        assert percentage == Decimal("50")
        # Применение процента
        result = money.apply_percentage(Decimal("10"))
        assert result.amount == Decimal("10")
        # Увеличение на процент
        result = money.increase_by_percentage(Decimal("10"))
        assert result.amount == Decimal("110")
        # Уменьшение на процент
        result = money.decrease_by_percentage(Decimal("10"))
        assert result.amount == Decimal("90")
    def test_money_trading_methods(self) -> None:
        """Тест торговых методов."""
        money = Money(Decimal("100"), Currency.USD)
        # Применение проскальзывания
        buy_price, sell_price = money.apply_slippage(Decimal("1"))  # 1%
        assert buy_price.amount > money.amount
        assert sell_price.amount < money.amount
        # Расчет комиссий
        fee = money.calculate_fees(Decimal("0.1"))  # 0.1%
        assert fee.amount == Decimal("0.1")
        # Чистая сумма после комиссий
        net_amount = money.calculate_net_amount(Decimal("0.1"))
        assert net_amount.amount == Decimal("99.9")
    def test_money_risk_management(self) -> None:
        """Тест риск-менеджмента."""
        small_money = Money(Decimal("0.001"), Currency.USD)
        large_money = Money(Decimal("1000000"), Currency.USD)
        normal_money = Money(Decimal("1000"), Currency.USD)
        assert not small_money.is_valid_position_size()
        assert not large_money.is_valid_position_size()
        assert normal_money.is_valid_position_size()
        assert small_money.get_position_risk_level() == "TOO_SMALL"
        assert large_money.get_position_risk_level() == "TOO_LARGE"
        assert normal_money.get_position_risk_level() in ["LOW", "MODERATE"]
    def test_money_serialization(self) -> None:
        """Тест сериализации."""
        money = Money(Decimal("100.50"), Currency.USD)
        data = money.to_dict()
        assert data["amount"] == "100.50"
        assert data["currency"] == "USD"
        assert data["type"] == "Money"
        # Десериализация
        reconstructed = Money.from_dict(data)
        assert reconstructed == money
class TestPriceIndustrial:
    """Тесты для промышленной реализации Price."""
    def test_price_creation(self) -> None:
        """Тест создания цены."""
        price = Price(Decimal("50000"), Currency.USD)
        assert price.amount == Decimal("50000")
        assert price.currency == Currency.USD
        assert str(price) == "50000 USD"
    def test_price_validation(self) -> None:
        """Тест валидации цен."""
        # Валидные цены
        Price(Decimal("100"), Currency.USD)
        Price(Decimal("0"), Currency.USD)
        # Невалидные цены
        with pytest.raises(ValueError):
            Price(Decimal("-100"), Currency.USD)
        with pytest.raises(ValueError):
            Price(Decimal("NaN"), Currency.USD)
    def test_price_arithmetic(self) -> None:
        """Тест арифметических операций."""
        price1 = Price(Decimal("100"), Currency.USD)
        price2 = Price(Decimal("50"), Currency.USD)
        # Сложение
        result = price1 + price2
        assert result.amount == Decimal("150")
        # Вычитание
        result = price1 - price2
        assert result.amount == Decimal("50")
        # Умножение
        result = price1 * 2
        assert result.amount == Decimal("200")
        # Деление
        result = price1 / 2
        assert result.amount == Decimal("50")
    def test_price_analysis(self) -> None:
        """Тест анализа цен."""
        current_price = Price(Decimal("100"), Currency.USD)
        previous_price = Price(Decimal("90"), Currency.USD)
        # Процентное изменение
        change = current_price.percentage_change_from(previous_price)
        assert change == Decimal("11.11111111111111111111111111")
        # Изменение цены
        price_change = current_price.price_change_from(previous_price)
        assert price_change.amount == Decimal("10")
        # Спред
        other_price = Price(Decimal("101"), Currency.USD)
        spread = current_price.spread_with(other_price)
        assert spread.amount == Decimal("1")
        spread_percentage = current_price.spread_percentage_with(other_price)
        assert spread_percentage == Decimal("1")
    def test_price_trading_methods(self) -> None:
        """Тест торговых методов."""
        price = Price(Decimal("100"), Currency.USD)
        # Применение проскальзывания
        buy_price, sell_price = price.apply_slippage(Decimal("1"))  # 1%
        assert buy_price.amount > price.amount
        assert sell_price.amount < price.amount
        # Расчет маржи
        margin = price.calculate_margin(Decimal("10"))  # 10x leverage
        assert margin.amount == Decimal("10")
        # Расчет P&L
        entry_price = Price(Decimal("90"), Currency.USD)
        pnl = price.calculate_pnl(entry_price, Decimal("1"), True)  # Long position
        assert pnl == Decimal("10")
        # Расчет ROI
        roi = price.calculate_roi(entry_price, True)
        assert roi == Decimal("11.11111111111111111111111111")
    def test_price_technical_analysis(self) -> None:
        """Тест технического анализа."""
        current_price = Price(Decimal("100"), Currency.USD)
        previous_price = Price(Decimal("90"), Currency.USD)
        # Волатильность
        assert current_price.is_volatile(previous_price)
        # Направление тренда
        trend = current_price.get_trend_direction(previous_price)
        assert trend == "UP"
        # Уровни поддержки/сопротивления
        assert current_price.is_support_level()
        assert current_price.is_resistance_level()
class TestVolumeIndustrial:
    """Тесты для промышленной реализации Volume."""
    def test_volume_creation(self) -> None:
        """Тест создания объема."""
        volume = Volume(Decimal("1000"), Currency.USD)
        assert volume.value == Decimal("1000")
        assert volume.currency == Currency.USD
        assert str(volume) == "1000.00000000 USD"
    def test_volume_validation(self) -> None:
        """Тест валидации объемов."""
        # Валидные объемы
        Volume(Decimal("1000"), Currency.USD)
        Volume(Decimal("0"), Currency.USD)
        # Невалидные объемы
        with pytest.raises(ValueError):
            Volume(Decimal("-1000"), Currency.USD)
        with pytest.raises(ValueError):
            Volume(Decimal("NaN"), Currency.USD)
    def test_volume_arithmetic(self) -> None:
        """Тест арифметических операций."""
        volume1 = Volume(Decimal("1000"), Currency.USD)
        volume2 = Volume(Decimal("500"), Currency.USD)
        # Сложение
        result = volume1 + volume2
        assert result.value == Decimal("1500")
        # Вычитание
        result = volume1 - volume2
        assert result.value == Decimal("500")
        # Умножение
        result = volume1 * 2
        assert result.value == Decimal("2000")
        # Деление
        result = volume1 / 2
        assert result.value == Decimal("500")
    def test_volume_liquidity_analysis(self) -> None:
        """Тест анализа ликвидности."""
        high_volume = Volume(Decimal("1000000"), Currency.USD)
        medium_volume = Volume(Decimal("100000"), Currency.USD)
        low_volume = Volume(Decimal("10000"), Currency.USD)
        assert high_volume.get_liquidity_level() == "HIGH"
        assert medium_volume.get_liquidity_level() == "MEDIUM"
        assert low_volume.get_liquidity_level() == "LOW"
        assert high_volume.is_liquid()
        assert medium_volume.is_liquid()
        assert not low_volume.is_liquid()
        assert high_volume.get_liquidity_score() > medium_volume.get_liquidity_score()
        assert medium_volume.get_liquidity_score() > low_volume.get_liquidity_score()
    def test_volume_trading_methods(self) -> None:
        """Тест торговых методов."""
        volume = Volume(Decimal("1000"), Currency.USD)
        total_volume = Volume(Decimal("10000"), Currency.USD)
        # Проверка торговли
        assert volume.is_tradable()
        # Рекомендация
        recommendation = volume.get_trading_recommendation()
        assert recommendation in ["SAFE_TO_TRADE", "CAUTION_ADVISED", "HIGH_RISK", "AVOID_TRADING"]
        # Влияние на рынок
        impact = volume.calculate_market_impact(total_volume)
        assert impact == Decimal("10")
        # Значимость объема
        assert volume.is_significant_volume(Decimal("1"))
        assert not volume.is_significant_volume(Decimal("50"))
class TestPercentageIndustrial:
    """Тесты для промышленной реализации Percentage."""
    def test_percentage_creation(self) -> None:
        """Тест создания процента."""
        percentage = Percentage(Decimal("5.5"))
        assert percentage.value == Decimal("5.5")
        assert str(percentage) == "5.50%"
    def test_percentage_validation(self) -> None:
        """Тест валидации процентов."""
        # Валидные проценты
        Percentage(Decimal("5"))
        Percentage(Decimal("-5"))
        Percentage(Decimal("0"))
        # Невалидные проценты
        with pytest.raises(ValueError):
            Percentage(Decimal("10001"))  # Превышает максимум
        with pytest.raises(ValueError):
            Percentage(Decimal("-10001"))  # Ниже минимума
    def test_percentage_arithmetic(self) -> None:
        """Тест арифметических операций."""
        p1 = Percentage(Decimal("10"))
        p2 = Percentage(Decimal("5"))
        # Сложение
        result = p1 + p2
        assert result.value == Decimal("15")
        # Вычитание
        result = p1 - p2
        assert result.value == Decimal("5")
        # Умножение
        result = p1 * 2
        assert result.value == Decimal("20")
        # Деление
        result = p1 / 2
        assert result.value == Decimal("5")
    def test_percentage_conversions(self) -> None:
        """Тест преобразований."""
        percentage = Percentage(Decimal("25"))
        # В долю
        fraction = percentage.to_fraction()
        assert fraction == Decimal("0.25")
        # Применение к значению
        result = percentage.apply_to(Decimal("100"))
        assert result == Decimal("25")
        # Увеличение значения
        result = percentage.increase_by(Decimal("100"))
        assert result == Decimal("125")
        # Уменьшение значения
        result = percentage.decrease_by(Decimal("100"))
        assert result == Decimal("75")
    def test_percentage_risk_analysis(self) -> None:
        """Тест анализа рисков."""
        low_risk = Percentage(Decimal("5"))
        medium_risk = Percentage(Decimal("20"))
        high_risk = Percentage(Decimal("50"))
        very_high_risk = Percentage(Decimal("80"))
        assert low_risk.get_risk_level() == "LOW"
        assert medium_risk.get_risk_level() == "MEDIUM"
        assert high_risk.get_risk_level() == "HIGH"
        assert very_high_risk.get_risk_level() == "VERY_HIGH"
        assert not low_risk.is_high_risk()
        assert high_risk.is_high_risk()
        assert low_risk.is_acceptable_risk(Decimal("10"))
        assert not high_risk.is_acceptable_risk(Decimal("10"))
    def test_percentage_return_analysis(self) -> None:
        """Тест анализа доходности."""
        excellent = Percentage(Decimal("25"))
        good = Percentage(Decimal("15"))
        moderate = Percentage(Decimal("8"))
        poor = Percentage(Decimal("2"))
        loss = Percentage(Decimal("-5"))
        assert excellent.get_return_rating() == "EXCELLENT"
        assert good.get_return_rating() == "GOOD"
        assert moderate.get_return_rating() == "MODERATE"
        assert poor.get_return_rating() == "POOR"
        assert loss.get_return_rating() == "LOSS"
        assert excellent.is_profitable()
        assert not loss.is_profitable()
        assert excellent.is_significant_return(Decimal("10"))
        assert not poor.is_significant_return(Decimal("10"))
    def test_percentage_trading_analysis(self) -> None:
        """Тест торгового анализа."""
        growth_rate = Percentage(Decimal("10"))
        # Сложный рост
        compound_growth = growth_rate.calculate_compound_growth(3)
        assert compound_growth.value > Decimal("30")
        # Периоды до достижения цели
        target = Percentage(Decimal("50"))
        periods = growth_rate.calculate_break_even_periods(target)
        assert periods > 0
        # Сила сигнала
        strong_signal = Percentage(Decimal("60"))
        weak_signal = Percentage(Decimal("10"))
        assert strong_signal.get_trading_signal_strength() == "STRONG"
        assert weak_signal.get_trading_signal_strength() == "WEAK"
class TestTimestampIndustrial:
    """Тесты для промышленной реализации Timestamp."""
    def test_timestamp_creation(self) -> None:
        """Тест создания временной метки."""
        now = datetime.now(timezone.utc)
        timestamp = Timestamp(now)
        assert timestamp.value == now
        assert str(timestamp) == now.isoformat()
    def test_timestamp_parsing(self) -> None:
        """Тест парсинга различных форматов."""
        # ISO строка
        iso_str = "2023-01-01T12:00:00+00:00"
        timestamp = Timestamp(iso_str)
        assert timestamp.to_iso() == iso_str
        # Unix timestamp
        unix_ts = 1672574400
        timestamp = Timestamp(unix_ts)
        assert timestamp.to_unix() == unix_ts
        # Unix timestamp в миллисекундах
        unix_ms = 1672574400000
        timestamp = Timestamp(unix_ms)
        assert timestamp.to_unix_ms() == unix_ms
    def test_timestamp_validation(self) -> None:
        """Тест валидации временных меток."""
        # Валидные метки
        Timestamp(datetime.now(timezone.utc))
        Timestamp("2023-01-01T12:00:00+00:00")
        # Невалидные метки
        with pytest.raises(ValueError):
            Timestamp("invalid-date")
    def test_timestamp_arithmetic(self) -> None:
        """Тест арифметических операций."""
        timestamp = Timestamp("2023-01-01T12:00:00+00:00")
        # Добавление времени
        result = timestamp.add_hours(2)
        assert result.time_difference_hours(timestamp) == 2
        result = timestamp.add_days(1)
        assert result.time_difference_days(timestamp) == 1
        # Вычитание времени
        result = timestamp.subtract_hours(2)
        assert timestamp.time_difference_hours(result) == 2
    def test_timestamp_comparison(self) -> None:
        """Тест сравнения временных меток."""
        earlier = Timestamp("2023-01-01T12:00:00+00:00")
        later = Timestamp("2023-01-01T14:00:00+00:00")
        assert earlier < later
        assert later > earlier
        assert earlier <= later
        assert later >= earlier
    def test_timestamp_trading_analysis(self) -> None:
        """Тест торгового анализа."""
        # Создаем временную метку в торговые часы
        trading_time = Timestamp("2023-01-02T10:00:00+00:00")  # Понедельник 10:00
        weekend_time = Timestamp("2023-01-07T10:00:00+00:00")  # Суббота 10:00
        assert trading_time.get_trading_session() == "REGULAR"
        assert weekend_time.get_trading_session() == "CLOSED"
        assert trading_time.is_market_open()
        assert not weekend_time.is_market_open()
        # Время до открытия/закрытия
        if trading_time.is_market_open():
            close_time = trading_time.get_time_until_market_close()
            assert close_time is not None
            assert close_time > 0
    def test_timestamp_utility_methods(self) -> None:
        """Тест утилитарных методов."""
        now = Timestamp.now()
        past = Timestamp("2023-01-01T12:00:00+00:00")
        future = Timestamp("2030-01-01T12:00:00+00:00")
        assert past.is_past()
        assert future.is_future()
        assert now.is_now(tolerance_seconds=5)
        # Возраст
        age = past.get_age_in_seconds()
        assert age > 0
        # Релевантность
        assert past.is_expired(max_age_seconds=1)
        assert not now.is_expired(max_age_seconds=3600)
class TestSignalIndustrial:
    """Тесты для промышленной реализации Signal."""
    def test_signal_creation(self) -> None:
        """Тест создания сигнала."""
        timestamp = Timestamp.now()
        signal = Signal(
            signal_type=SignalType.BUY,
            timestamp=timestamp,
            strength=SignalStrength.STRONG
        )
        assert signal.signal_type == SignalType.BUY
        assert signal.timestamp == timestamp
        assert signal.strength == SignalStrength.STRONG
        assert signal.is_trading_signal
        assert signal.is_buy_signal
    def test_signal_validation(self) -> None:
        """Тест валидации сигналов."""
        timestamp = Timestamp.now()
        # Валидные сигналы
        Signal(SignalType.BUY, timestamp, SignalStrength.STRONG)
        Signal(SignalType.SELL, timestamp, SignalStrength.WEAK)
        # Невалидные сигналы
        with pytest.raises(ValueError):
            Signal("INVALID", timestamp, SignalStrength.STRONG)
    def test_signal_properties(self) -> None:
        """Тест свойств сигналов."""
        timestamp = Timestamp.now()
        buy_signal = Signal(SignalType.BUY, timestamp, SignalStrength.STRONG)
        sell_signal = Signal(SignalType.SELL, timestamp, SignalStrength.WEAK)
        hold_signal = Signal(SignalType.HOLD, timestamp, SignalStrength.MODERATE)
        assert buy_signal.is_buy_signal
        assert sell_signal.is_sell_signal
        assert hold_signal.is_hold_signal
        assert buy_signal.is_strong_signal
        assert not sell_signal.is_strong_signal
        assert not hold_signal.is_strong_signal
    def test_signal_scoring(self) -> None:
        """Тест скоринга сигналов."""
        timestamp = Timestamp.now()
        confidence = Percentage(Decimal("80"))
        signal = Signal(
            signal_type=SignalType.BUY,
            timestamp=timestamp,
            strength=SignalStrength.STRONG,
            confidence=confidence
        )
        # Базовый скор
        combined_score = signal.get_combined_score()
        assert 0 <= combined_score <= 1
        # Комплексный скор
        comprehensive_score = signal.get_comprehensive_score()
        assert 0 <= comprehensive_score <= 1
    def test_signal_lifecycle(self) -> None:
        """Тест жизненного цикла сигналов."""
        recent_timestamp = Timestamp.now()
        old_timestamp = Timestamp("2023-01-01T12:00:00+00:00")
        recent_signal = Signal(SignalType.BUY, recent_timestamp)
        old_signal = Signal(SignalType.BUY, old_timestamp)
        assert recent_signal.is_recent()
        assert not old_signal.is_recent()
        assert not recent_signal.is_expired()
        assert old_signal.is_expired()
    def test_signal_risk_analysis(self) -> None:
        """Тест анализа рисков."""
        timestamp = Timestamp.now()
        error_signal = Signal(SignalType.ERROR, timestamp)
        alert_signal = Signal(SignalType.ALERT, timestamp)
        buy_signal = Signal(SignalType.BUY, timestamp, SignalStrength.STRONG)
        weak_buy_signal = Signal(SignalType.BUY, timestamp, SignalStrength.WEAK)
        assert error_signal.get_risk_level() == "VERY_HIGH"
        assert alert_signal.get_risk_level() == "HIGH"
        assert buy_signal.get_risk_level() == "MEDIUM"
        assert weak_buy_signal.get_risk_level() == "HIGH"
    def test_signal_trading_recommendations(self) -> None:
        """Тест торговых рекомендаций."""
        timestamp = Timestamp.now()
        strong_buy = Signal(SignalType.BUY, timestamp, SignalStrength.STRONG)
        weak_buy = Signal(SignalType.BUY, timestamp, SignalStrength.WEAK)
        expired_signal = Signal(SignalType.BUY, Timestamp("2023-01-01T12:00:00+00:00"))
        assert strong_buy.get_trading_recommendation() in ["CONSIDER_BUY", "WATCH_FOR_BUY"]
        assert weak_buy.get_trading_recommendation() == "WAIT_FOR_CONFIRMATION"
        assert expired_signal.get_trading_recommendation() == "IGNORE_EXPIRED"
    def test_signal_metadata(self) -> None:
        """Тест метаданных сигналов."""
        timestamp = Timestamp.now()
        signal = Signal(SignalType.BUY, timestamp)
        # Добавление метаданных
        signal_with_meta = signal.add_metadata("source", "technical_analysis")
        assert signal_with_meta.metadata["source"] == "technical_analysis"
        assert signal.metadata == {}  # Оригинальный сигнал не изменился
    def test_signal_factory_methods(self) -> None:
        """Тест фабричных методов."""
        timestamp = Timestamp.now()
        buy_signal = Signal.create_buy_signal(timestamp, SignalStrength.STRONG)
        sell_signal = Signal.create_sell_signal(timestamp, SignalStrength.MODERATE)
        hold_signal = Signal.create_hold_signal(timestamp)
        assert buy_signal.is_buy_signal
        assert sell_signal.is_sell_signal
        assert hold_signal.is_hold_signal
class TestTradingPairIndustrial:
    """Тесты для промышленной реализации TradingPair."""
    def test_trading_pair_creation(self) -> None:
        """Тест создания торговой пары."""
        pair = TradingPair(Currency.BTC, Currency.USDT, "BTCUSDT")
        assert pair.base_currency == Currency.BTC
        assert pair.quote_currency == Currency.USDT
        assert pair.symbol == "BTCUSDT"
        assert str(pair) == "BTCUSDT"
    def test_trading_pair_validation(self) -> None:
        """Тест валидации торговых пар."""
        # Валидные пары
        TradingPair(Currency.BTC, Currency.USDT)
        TradingPair(Currency.ETH, Currency.USD)
        # Невалидные пары
        with pytest.raises(ValueError):
            TradingPair(Currency.BTC, Currency.BTC)  # Одинаковые валюты
        with pytest.raises(ValueError):
            TradingPair(Currency.BTC, Currency.INVALID)  # Несуществующая валюта
    def test_trading_pair_properties(self) -> None:
        """Тест свойств торговых пар."""
        crypto_pair = TradingPair(Currency.BTC, Currency.ETH)
        fiat_pair = TradingPair(Currency.USD, Currency.EUR)
        crypto_fiat_pair = TradingPair(Currency.BTC, Currency.USD)
        stablecoin_pair = TradingPair(Currency.BTC, Currency.USDT)
        major_pair = TradingPair(Currency.BTC, Currency.USDT, "BTCUSDT")
        assert crypto_pair.is_crypto_pair
        assert fiat_pair.is_fiat_pair
        assert crypto_fiat_pair.is_crypto_fiat_pair
        assert stablecoin_pair.is_stablecoin_pair
        assert major_pair.is_major_pair
    def test_trading_pair_priority(self) -> None:
        """Тест приоритетов торговых пар."""
        major_pair = TradingPair(Currency.BTC, Currency.USDT, "BTCUSDT")
        stablecoin_pair = TradingPair(Currency.BTC, Currency.USDC)
        crypto_fiat_pair = TradingPair(Currency.BTC, Currency.USD)
        crypto_pair = TradingPair(Currency.BTC, Currency.ETH)
        assert major_pair.get_trading_priority() < stablecoin_pair.get_trading_priority()
        assert stablecoin_pair.get_trading_priority() < crypto_fiat_pair.get_trading_priority()
        assert crypto_fiat_pair.get_trading_priority() < crypto_pair.get_trading_priority()
    def test_trading_pair_configuration(self) -> None:
        """Тест конфигурации торговых пар."""
        btc_usdt = TradingPair(Currency.BTC, Currency.USDT, "BTCUSDT")
        assert btc_usdt.get_min_order_size() == Decimal("0.001")
        assert btc_usdt.get_price_precision() == 2
        assert btc_usdt.get_volume_precision() == 6
    def test_trading_pair_analysis(self) -> None:
        """Тест анализа торговых пар."""
        major_pair = TradingPair(Currency.BTC, Currency.USDT, "BTCUSDT")
        alt_pair = TradingPair(Currency.ADA, Currency.USDT)
        # Ликвидность
        assert major_pair.get_liquidity_rating() > alt_pair.get_liquidity_rating()
        # Волатильность
        btc_volatility = major_pair.get_volatility_rating()
        ada_volatility = alt_pair.get_volatility_rating()
        assert 0 <= btc_volatility <= 1
        assert 0 <= ada_volatility <= 1
        # Риск
        major_risk = major_pair.get_risk_score()
        alt_risk = alt_pair.get_risk_score()
        assert 0 <= major_risk <= 1
        assert 0 <= alt_risk <= 1
        assert major_risk < alt_risk
    def test_trading_pair_recommendations(self) -> None:
        """Тест торговых рекомендаций."""
        major_pair = TradingPair(Currency.BTC, Currency.USDT, "BTCUSDT")
        alt_pair = TradingPair(Currency.ADA, Currency.USDT)
        major_rec = major_pair.get_trading_recommendation()
        alt_rec = alt_pair.get_trading_recommendation()
        assert major_rec in ["SAFE_FOR_NEWBIES", "MODERATE_RISK"]
        assert alt_rec in ["HIGH_RISK", "VERY_HIGH_RISK"]
    def test_trading_pair_operations(self) -> None:
        """Тест операций с торговыми парами."""
        pair = TradingPair(Currency.BTC, Currency.USDT)
        # Обратная пара
        reverse_pair = pair.get_reverse_pair()
        assert reverse_pair.base_currency == Currency.USDT
        assert reverse_pair.quote_currency == Currency.BTC
        # Популярные пары
        common_pairs = pair.get_common_pairs()
        assert len(common_pairs) > 0
        assert all(p.base_currency == Currency.BTC for p in common_pairs)
    def test_trading_pair_factory_methods(self) -> None:
        """Тест фабричных методов."""
        btc_usdt = TradingPair.create_btc_usdt()
        eth_usdt = TradingPair.create_eth_usdt()
        btc_usd = TradingPair.create_btc_usd()
        assert btc_usdt.symbol == "BTCUSDT"
        assert eth_usdt.symbol == "ETHUSDT"
        assert btc_usd.symbol == "BTCUSD"
    def test_trading_pair_from_symbol(self) -> None:
        """Тест создания из символа."""
        pair = TradingPair.from_symbol("BTCUSDT")
        assert pair.base_currency == Currency.BTC
        assert pair.quote_currency == Currency.USDT
        with pytest.raises(ValueError):
            TradingPair.from_symbol("INVALID")
class TestValueObjectFactoryIndustrial:
    """Тесты для промышленной фабрики Value Objects."""
    def test_factory_creation(self) -> None:
        """Тест создания фабрики."""
        factory = ValueObjectFactory()
        assert len(factory.get_registered_types()) > 0
    def test_money_creation(self) -> None:
        """Тест создания Money через фабрику."""
        money = factory.create_money(100.50, "USD")
        assert money.amount == Decimal("100.50")
        assert money.currency == Currency.USD
        money = factory.create_money("1000", Currency.EUR)
        assert money.amount == Decimal("1000")
        assert money.currency == Currency.EUR
    def test_price_creation(self) -> None:
        """Тест создания Price через фабрику."""
        price = factory.create_price(50000, "USD")
        assert price.amount == Decimal("50000")
        assert price.currency == Currency.USD
    def test_volume_creation(self) -> None:
        """Тест создания Volume через фабрику."""
        volume = factory.create_volume(1000, "USD")
        assert volume.value == Decimal("1000")
        assert volume.currency == Currency.USD
    def test_percentage_creation(self) -> None:
        """Тест создания Percentage через фабрику."""
        percentage = factory.create_percentage("5.5")
        assert percentage.value == Decimal("5.5")
        percentage = factory.create_percentage(10)
        assert percentage.value == Decimal("10")
    def test_timestamp_creation(self) -> None:
        """Тест создания Timestamp через фабрику."""
        timestamp = factory.create_timestamp("2023-01-01T12:00:00+00:00")
        assert timestamp.to_iso() == "2023-01-01T12:00:00+00:00"
        timestamp = factory.create_timestamp(1672574400)
        assert timestamp.to_unix() == 1672574400
    def test_trading_pair_creation(self) -> None:
        """Тест создания TradingPair через фабрику."""
        pair = factory.create_trading_pair("BTC", "USDT", "BTCUSDT")
        assert pair.base_currency == Currency.BTC
        assert pair.quote_currency == Currency.USDT
        assert pair.symbol == "BTCUSDT"
    def test_signal_creation(self) -> None:
        """Тест создания Signal через фабрику."""
        timestamp = Timestamp.now()
        signal = factory.create_signal("BUY", timestamp, "STRONG")
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
    def test_validation_context(self) -> None:
        """Тест контекста валидации."""
        context = ValidationContext(
            strict_mode=True,
            allow_negative=False,
            allow_zero=True,
            max_precision=2
        )
        # Строгий режим
        with pytest.raises(ValueError):
            factory.create_money(-100, "USD", context)
        # Обычный режим
        money = factory.create_money(100, "USD")
        assert money.amount == Decimal("100")
    def test_serialization(self) -> None:
        """Тест сериализации через фабрику."""
        money = Money(Decimal("100"), Currency.USD)
        data = factory.to_dict(money)
        assert data["type"] == "Money"
        assert data["amount"] == "100"
        assert data["currency"] == "USD"
        reconstructed = factory.from_dict(data)
        assert reconstructed == money
    def test_caching(self) -> None:
        """Тест кэширования."""
        # Создаем объект
        money1 = factory.get_cached_or_create(
            "test_money_100_usd",
            factory.create_money,
            100,
            "USD"
        )
        # Получаем из кэша
        money2 = factory.get_cached_or_create(
            "test_money_100_usd",
            factory.create_money,
            100,
            "USD"
        )
        assert money1 is money2  # Тот же объект из кэша
        # Статистика кэша
        stats = factory.get_cache_stats()
        assert stats["size"] > 0
    def test_error_handling(self) -> None:
        """Тест обработки ошибок."""
        # Невалидная валюта
        with pytest.raises(ValueError):
            factory.create_money(100, "INVALID")
        # Невалидный символ
        with pytest.raises(ValueError):
            factory.create_trading_pair("INVALID", "USDT")
        # Статистика ошибок
        stats = factory.get_performance_stats()
        assert stats["error_count"] > 0
    def test_performance_monitoring(self) -> None:
        """Тест мониторинга производительности."""
        # Создаем несколько объектов
        for i in range(10):
            factory.create_money(i, "USD")
        stats = factory.get_performance_stats()
        assert stats["total_operations"] >= 10
        assert stats["success_count"] >= 10
        assert stats["success_rate"] > 0
    def test_registration(self) -> None:
        """Тест регистрации новых типов."""
        factory = ValueObjectFactory()
        original_count = len(factory.get_registered_types())
        # Регистрируем новый тип (если бы был)
        # factory.register("CustomType", CustomValueObject)
        # assert len(factory.get_registered_types()) == original_count + 1
if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 
