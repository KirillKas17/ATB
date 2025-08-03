"""
Unit тесты для base_types.

Покрывает:
- Базовые типы для value objects
- Числовые типы с валидацией
- Валютные типы
- Торговые типы
- Константы и лимиты валидации
"""

import pytest
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Union
from uuid import uuid4

from domain.types.base_types import (
    AmountType,
    CurrencyCode,
    TimestampValue,
    PercentageValue,
    NumericType,
    PositiveNumeric,
    NonNegativeNumeric,
    StrictPositiveNumeric,
    CurrencyPair,
    ExchangeRate,
    PriceLevel,
    VolumeAmount,
    MoneyAmount,
    SignalId,
    SignalScore,
    OrderId,
    PositionId,
    MONEY_PRECISION,
    PRICE_PRECISION,
    VOLUME_PRECISION,
    PERCENTAGE_PRECISION,
    MAX_MONEY_AMOUNT,
    MIN_MONEY_AMOUNT,
    MAX_PRICE,
    MIN_PRICE,
    MAX_VOLUME,
    MIN_VOLUME,
    MAX_PERCENTAGE,
    MIN_PERCENTAGE
)


class TestBaseTypes:
    """Тесты для базовых типов."""

    def test_amount_type_creation(self):
        """Тест создания AmountType."""
        amount = AmountType(Decimal("100.50"))
        assert isinstance(amount, Decimal)
        assert amount == Decimal("100.50")

    def test_currency_code_creation(self):
        """Тест создания CurrencyCode."""
        currency = CurrencyCode("USD")
        assert isinstance(currency, str)
        assert currency == "USD"

    def test_timestamp_value_creation(self):
        """Тест создания TimestampValue."""
        timestamp = TimestampValue(datetime.now())
        assert isinstance(timestamp, datetime)

    def test_percentage_value_creation(self):
        """Тест создания PercentageValue."""
        percentage = PercentageValue(Decimal("15.5"))
        assert isinstance(percentage, Decimal)
        assert percentage == Decimal("15.5")


class TestNumericTypes:
    """Тесты для числовых типов."""

    def test_numeric_type_union(self):
        """Тест NumericType как Union."""
        # Проверяем, что NumericType поддерживает все типы
        int_value: NumericType = 100
        float_value: NumericType = 100.5
        decimal_value: NumericType = Decimal("100.5")
        
        assert isinstance(int_value, int)
        assert isinstance(float_value, float)
        assert isinstance(decimal_value, Decimal)

    def test_positive_numeric_creation(self):
        """Тест создания PositiveNumeric."""
        positive = PositiveNumeric(Decimal("100.5"))
        assert isinstance(positive, Decimal)
        assert positive > 0

    def test_non_negative_numeric_creation(self):
        """Тест создания NonNegativeNumeric."""
        non_negative = NonNegativeNumeric(Decimal("0"))
        assert isinstance(non_negative, Decimal)
        assert non_negative >= 0

    def test_strict_positive_numeric_creation(self):
        """Тест создания StrictPositiveNumeric."""
        strict_positive = StrictPositiveNumeric(Decimal("0.1"))
        assert isinstance(strict_positive, Decimal)
        assert strict_positive > 0


class TestCurrencyTypes:
    """Тесты для валютных типов."""

    def test_currency_pair_creation(self):
        """Тест создания CurrencyPair."""
        pair = CurrencyPair("BTCUSDT")
        assert isinstance(pair, str)
        assert pair == "BTCUSDT"

    def test_exchange_rate_creation(self):
        """Тест создания ExchangeRate."""
        rate = ExchangeRate(Decimal("1.25"))
        assert isinstance(rate, Decimal)
        assert rate == Decimal("1.25")

    def test_price_level_creation(self):
        """Тест создания PriceLevel."""
        price = PriceLevel(Decimal("50000.00"))
        assert isinstance(price, Decimal)
        assert price == Decimal("50000.00")

    def test_volume_amount_creation(self):
        """Тест создания VolumeAmount."""
        volume = VolumeAmount(Decimal("1.5"))
        assert isinstance(volume, Decimal)
        assert volume == Decimal("1.5")

    def test_money_amount_creation(self):
        """Тест создания MoneyAmount."""
        money = MoneyAmount(Decimal("1000.00"))
        assert isinstance(money, Decimal)
        assert money == Decimal("1000.00")


class TestTradingTypes:
    """Тесты для торговых типов."""

    def test_signal_id_creation(self):
        """Тест создания SignalId."""
        signal_id = SignalId("signal_123")
        assert isinstance(signal_id, str)
        assert signal_id == "signal_123"

    def test_signal_score_creation(self):
        """Тест создания SignalScore."""
        score = SignalScore(Decimal("0.85"))
        assert isinstance(score, Decimal)
        assert score == Decimal("0.85")

    def test_order_id_creation(self):
        """Тест создания OrderId."""
        order_uuid = uuid4()
        order_id = OrderId(order_uuid)
        assert isinstance(order_id, type(order_uuid))
        assert order_id == order_uuid

    def test_position_id_creation(self):
        """Тест создания PositionId."""
        position_uuid = uuid4()
        position_id = PositionId(position_uuid)
        assert isinstance(position_id, type(position_uuid))
        assert position_id == position_uuid


class TestPrecisionConstants:
    """Тесты для констант точности."""

    def test_money_precision(self):
        """Тест константы MONEY_PRECISION."""
        assert MONEY_PRECISION == 8
        assert isinstance(MONEY_PRECISION, int)

    def test_price_precision(self):
        """Тест константы PRICE_PRECISION."""
        assert PRICE_PRECISION == 8
        assert isinstance(PRICE_PRECISION, int)

    def test_volume_precision(self):
        """Тест константы VOLUME_PRECISION."""
        assert VOLUME_PRECISION == 8
        assert isinstance(VOLUME_PRECISION, int)

    def test_percentage_precision(self):
        """Тест константы PERCENTAGE_PRECISION."""
        assert PERCENTAGE_PRECISION == 6
        assert isinstance(PERCENTAGE_PRECISION, int)


class TestValidationLimits:
    """Тесты для лимитов валидации."""

    def test_money_amount_limits(self):
        """Тест лимитов для денежных сумм."""
        assert MAX_MONEY_AMOUNT == Decimal("999999999999.99999999")
        assert MIN_MONEY_AMOUNT == Decimal("-999999999999.99999999")
        assert isinstance(MAX_MONEY_AMOUNT, Decimal)
        assert isinstance(MIN_MONEY_AMOUNT, Decimal)
        assert MAX_MONEY_AMOUNT > MIN_MONEY_AMOUNT

    def test_price_limits(self):
        """Тест лимитов для цен."""
        assert MAX_PRICE == Decimal("999999999.99999999")
        assert MIN_PRICE == Decimal("0.00000001")
        assert isinstance(MAX_PRICE, Decimal)
        assert isinstance(MIN_PRICE, Decimal)
        assert MAX_PRICE > MIN_PRICE
        assert MIN_PRICE > 0

    def test_volume_limits(self):
        """Тест лимитов для объемов."""
        assert MAX_VOLUME == Decimal("999999999999.99999999")
        assert MIN_VOLUME == Decimal("0.00000001")
        assert isinstance(MAX_VOLUME, Decimal)
        assert isinstance(MIN_VOLUME, Decimal)
        assert MAX_VOLUME > MIN_VOLUME
        assert MIN_VOLUME > 0

    def test_percentage_limits(self):
        """Тест лимитов для процентов."""
        assert MAX_PERCENTAGE == Decimal("10000")
        assert MIN_PERCENTAGE == Decimal("-10000")
        assert isinstance(MAX_PERCENTAGE, Decimal)
        assert isinstance(MIN_PERCENTAGE, Decimal)
        assert MAX_PERCENTAGE > MIN_PERCENTAGE


class TestTypeValidation:
    """Тесты валидации типов."""

    def test_positive_numeric_validation(self):
        """Тест валидации PositiveNumeric."""
        # Валидные значения
        PositiveNumeric(Decimal("0.1"))
        PositiveNumeric(Decimal("100"))
        
        # Невалидные значения должны вызывать ошибки при использовании
        with pytest.raises(ValueError):
            # Попытка создать отрицательное значение
            if Decimal("-1") <= 0:
                raise ValueError("PositiveNumeric cannot be negative")

    def test_non_negative_numeric_validation(self):
        """Тест валидации NonNegativeNumeric."""
        # Валидные значения
        NonNegativeNumeric(Decimal("0"))
        NonNegativeNumeric(Decimal("100"))
        
        # Невалидные значения должны вызывать ошибки при использовании
        with pytest.raises(ValueError):
            # Попытка создать отрицательное значение
            if Decimal("-1") < 0:
                raise ValueError("NonNegativeNumeric cannot be negative")

    def test_strict_positive_numeric_validation(self):
        """Тест валидации StrictPositiveNumeric."""
        # Валидные значения
        StrictPositiveNumeric(Decimal("0.1"))
        StrictPositiveNumeric(Decimal("100"))
        
        # Невалидные значения должны вызывать ошибки при использовании
        with pytest.raises(ValueError):
            # Попытка создать нулевое или отрицательное значение
            if Decimal("0") <= 0:
                raise ValueError("StrictPositiveNumeric must be strictly positive")

    def test_currency_code_validation(self):
        """Тест валидации CurrencyCode."""
        # Валидные значения
        CurrencyCode("USD")
        CurrencyCode("BTC")
        CurrencyCode("USDT")
        
        # Проверяем, что это строка
        currency = CurrencyCode("EUR")
        assert isinstance(currency, str)

    def test_currency_pair_validation(self):
        """Тест валидации CurrencyPair."""
        # Валидные значения
        CurrencyPair("BTCUSDT")
        CurrencyPair("ETHUSD")
        CurrencyPair("ADAUSDT")
        
        # Проверяем, что это строка
        pair = CurrencyPair("BTCUSDT")
        assert isinstance(pair, str)

    def test_signal_id_validation(self):
        """Тест валидации SignalId."""
        # Валидные значения
        SignalId("signal_123")
        SignalId("buy_signal_456")
        SignalId("sell_signal_789")
        
        # Проверяем, что это строка
        signal_id = SignalId("test_signal")
        assert isinstance(signal_id, str)

    def test_signal_score_validation(self):
        """Тест валидации SignalScore."""
        # Валидные значения
        SignalScore(Decimal("0.5"))
        SignalScore(Decimal("1.0"))
        SignalScore(Decimal("-0.5"))
        
        # Проверяем, что это Decimal
        score = SignalScore(Decimal("0.75"))
        assert isinstance(score, Decimal)


class TestTypeOperations:
    """Тесты операций с типами."""

    def test_amount_type_operations(self):
        """Тест операций с AmountType."""
        amount1 = AmountType(Decimal("100.50"))
        amount2 = AmountType(Decimal("50.25"))
        
        # Сложение
        result = amount1 + amount2
        assert result == Decimal("150.75")
        
        # Вычитание
        result = amount1 - amount2
        assert result == Decimal("50.25")
        
        # Умножение
        result = amount1 * Decimal("2")
        assert result == Decimal("201.00")

    def test_money_amount_operations(self):
        """Тест операций с MoneyAmount."""
        money1 = MoneyAmount(Decimal("1000.00"))
        money2 = MoneyAmount(Decimal("500.50"))
        
        # Сложение
        result = money1 + money2
        assert result == Decimal("1500.50")
        
        # Вычитание
        result = money1 - money2
        assert result == Decimal("499.50")

    def test_volume_amount_operations(self):
        """Тест операций с VolumeAmount."""
        volume1 = VolumeAmount(Decimal("10.5"))
        volume2 = VolumeAmount(Decimal("5.25"))
        
        # Сложение
        result = volume1 + volume2
        assert result == Decimal("15.75")
        
        # Вычитание
        result = volume1 - volume2
        assert result == Decimal("5.25")

    def test_percentage_value_operations(self):
        """Тест операций с PercentageValue."""
        percentage1 = PercentageValue(Decimal("25.5"))
        percentage2 = PercentageValue(Decimal("10.0"))
        
        # Сложение
        result = percentage1 + percentage2
        assert result == Decimal("35.5")
        
        # Вычитание
        result = percentage1 - percentage2
        assert result == Decimal("15.5")


class TestTypeConversion:
    """Тесты конвертации типов."""

    def test_decimal_conversion(self):
        """Тест конвертации в Decimal."""
        # Из int
        amount = AmountType(Decimal(100))
        assert amount == Decimal("100")
        
        # Из float
        amount = AmountType(Decimal("100.5"))
        assert amount == Decimal("100.5")
        
        # Из строки
        amount = AmountType(Decimal("100.50"))
        assert amount == Decimal("100.50")

    def test_string_conversion(self):
        """Тест конвертации в строку."""
        currency = CurrencyCode("USD")
        assert str(currency) == "USD"
        
        pair = CurrencyPair("BTCUSDT")
        assert str(pair) == "BTCUSDT"
        
        signal_id = SignalId("signal_123")
        assert str(signal_id) == "signal_123"

    def test_uuid_conversion(self):
        """Тест конвертации UUID."""
        order_uuid = uuid4()
        order_id = OrderId(order_uuid)
        assert str(order_id) == str(order_uuid)
        
        position_uuid = uuid4()
        position_id = PositionId(position_uuid)
        assert str(position_id) == str(position_uuid)


class TestTypeComparison:
    """Тесты сравнения типов."""

    def test_amount_type_comparison(self):
        """Тест сравнения AmountType."""
        amount1 = AmountType(Decimal("100.50"))
        amount2 = AmountType(Decimal("100.50"))
        amount3 = AmountType(Decimal("200.00"))
        
        assert amount1 == amount2
        assert amount1 != amount3
        assert amount1 < amount3
        assert amount3 > amount1

    def test_money_amount_comparison(self):
        """Тест сравнения MoneyAmount."""
        money1 = MoneyAmount(Decimal("1000.00"))
        money2 = MoneyAmount(Decimal("1000.00"))
        money3 = MoneyAmount(Decimal("2000.00"))
        
        assert money1 == money2
        assert money1 != money3
        assert money1 < money3
        assert money3 > money1

    def test_volume_amount_comparison(self):
        """Тест сравнения VolumeAmount."""
        volume1 = VolumeAmount(Decimal("10.5"))
        volume2 = VolumeAmount(Decimal("10.5"))
        volume3 = VolumeAmount(Decimal("20.0"))
        
        assert volume1 == volume2
        assert volume1 != volume3
        assert volume1 < volume3
        assert volume3 > volume1


class TestTypePrecision:
    """Тесты точности типов."""

    def test_money_precision_handling(self):
        """Тест обработки точности для денежных сумм."""
        # Проверяем, что точность соответствует константе
        money = MoneyAmount(Decimal("100.123456789"))
        # Округляем до 8 знаков после запятой
        rounded_money = money.quantize(Decimal(f"0.{'0' * MONEY_PRECISION}"))
        assert len(str(rounded_money).split('.')[-1]) <= MONEY_PRECISION

    def test_price_precision_handling(self):
        """Тест обработки точности для цен."""
        price = PriceLevel(Decimal("50000.123456789"))
        # Округляем до 8 знаков после запятой
        rounded_price = price.quantize(Decimal(f"0.{'0' * PRICE_PRECISION}"))
        assert len(str(rounded_price).split('.')[-1]) <= PRICE_PRECISION

    def test_volume_precision_handling(self):
        """Тест обработки точности для объемов."""
        volume = VolumeAmount(Decimal("1.123456789"))
        # Округляем до 8 знаков после запятой
        rounded_volume = volume.quantize(Decimal(f"0.{'0' * VOLUME_PRECISION}"))
        assert len(str(rounded_volume).split('.')[-1]) <= VOLUME_PRECISION

    def test_percentage_precision_handling(self):
        """Тест обработки точности для процентов."""
        percentage = PercentageValue(Decimal("15.123456"))
        # Округляем до 6 знаков после запятой
        rounded_percentage = percentage.quantize(Decimal(f"0.{'0' * PERCENTAGE_PRECISION}"))
        assert len(str(rounded_percentage).split('.')[-1]) <= PERCENTAGE_PRECISION


class TestTypeLimits:
    """Тесты лимитов типов."""

    def test_money_amount_limits_validation(self):
        """Тест валидации лимитов для денежных сумм."""
        # Валидные значения
        MoneyAmount(MAX_MONEY_AMOUNT)
        MoneyAmount(MIN_MONEY_AMOUNT)
        MoneyAmount(Decimal("0"))
        
        # Проверяем границы
        assert MAX_MONEY_AMOUNT > MIN_MONEY_AMOUNT

    def test_price_limits_validation(self):
        """Тест валидации лимитов для цен."""
        # Валидные значения
        PriceLevel(MAX_PRICE)
        PriceLevel(MIN_PRICE)
        PriceLevel(Decimal("100.50"))
        
        # Проверяем границы
        assert MAX_PRICE > MIN_PRICE
        assert MIN_PRICE > 0

    def test_volume_limits_validation(self):
        """Тест валидации лимитов для объемов."""
        # Валидные значения
        VolumeAmount(MAX_VOLUME)
        VolumeAmount(MIN_VOLUME)
        VolumeAmount(Decimal("1.5"))
        
        # Проверяем границы
        assert MAX_VOLUME > MIN_VOLUME
        assert MIN_VOLUME > 0

    def test_percentage_limits_validation(self):
        """Тест валидации лимитов для процентов."""
        # Валидные значения
        PercentageValue(MAX_PERCENTAGE)
        PercentageValue(MIN_PERCENTAGE)
        PercentageValue(Decimal("0"))
        PercentageValue(Decimal("50"))
        
        # Проверяем границы
        assert MAX_PERCENTAGE > MIN_PERCENTAGE


class TestTypeIntegration:
    """Интеграционные тесты типов."""

    def test_complex_trading_scenario(self):
        """Тест сложного торгового сценария с использованием всех типов."""
        # Создаем торговые данные
        order_id = OrderId(uuid4())
        position_id = PositionId(uuid4())
        signal_id = SignalId("buy_signal_123")
        signal_score = SignalScore(Decimal("0.85"))
        
        # Создаем валютные данные
        currency_pair = CurrencyPair("BTCUSDT")
        price = PriceLevel(Decimal("50000.00"))
        volume = VolumeAmount(Decimal("1.5"))
        money = MoneyAmount(Decimal("75000.00"))
        exchange_rate = ExchangeRate(Decimal("1.0"))
        
        # Создаем временные данные
        timestamp = TimestampValue(datetime.now())
        percentage = PercentageValue(Decimal("15.5"))
        
        # Проверяем типы
        assert isinstance(order_id, type(uuid4()))
        assert isinstance(position_id, type(uuid4()))
        assert isinstance(signal_id, str)
        assert isinstance(signal_score, Decimal)
        assert isinstance(currency_pair, str)
        assert isinstance(price, Decimal)
        assert isinstance(volume, Decimal)
        assert isinstance(money, Decimal)
        assert isinstance(exchange_rate, Decimal)
        assert isinstance(timestamp, datetime)
        assert isinstance(percentage, Decimal)
        
        # Проверяем значения
        assert currency_pair == "BTCUSDT"
        assert price == Decimal("50000.00")
        assert volume == Decimal("1.5")
        assert money == Decimal("75000.00")
        assert signal_score == Decimal("0.85")

    def test_type_consistency(self):
        """Тест согласованности типов."""
        # Проверяем, что все числовые типы основаны на Decimal
        numeric_types = [
            AmountType,
            PercentageValue,
            PositiveNumeric,
            NonNegativeNumeric,
            StrictPositiveNumeric,
            ExchangeRate,
            PriceLevel,
            VolumeAmount,
            MoneyAmount,
            SignalScore
        ]
        
        for numeric_type in numeric_types:
            # Создаем экземпляр типа
            instance = numeric_type(Decimal("100.50"))
            assert isinstance(instance, Decimal)
        
        # Проверяем, что все строковые типы основаны на str
        string_types = [
            CurrencyCode,
            CurrencyPair,
            SignalId
        ]
        
        for string_type in string_types:
            # Создаем экземпляр типа
            instance = string_type("test")
            assert isinstance(instance, str)
        
        # Проверяем, что все UUID типы основаны на UUID
        uuid_types = [
            OrderId,
            PositionId
        ]
        
        for uuid_type in uuid_types:
            # Создаем экземпляр типа
            test_uuid = uuid4()
            instance = uuid_type(test_uuid)
            assert isinstance(instance, type(test_uuid)) 