"""
Unit тесты для domain.value_objects.volume

Покрывает:
- Создание и валидацию Volume объектов
- Арифметические операции
- Сравнения
- Сериализация/десериализация
- Округление и форматирование
"""

import pytest
from decimal import Decimal
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency


class TestVolume:
    """Тесты для Volume value object"""

    def test_volume_creation_valid(self):
        """Тест создания Volume с валидными данными"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        assert volume.amount == Decimal("1000.50")
        assert volume.currency == Currency.USD

    def test_volume_creation_from_string(self):
        """Тест создания Volume из строки"""
        volume = Volume.from_string("1000.50", "USD")
        assert volume.amount == Decimal("1000.50")
        assert volume.currency == Currency.USD

    def test_volume_creation_from_float(self):
        """Тест создания Volume из float"""
        volume = Volume.from_float(1000.50, Currency.USD)
        assert volume.amount == Decimal("1000.50")
        assert volume.currency == Currency.USD

    def test_volume_creation_invalid_amount(self):
        """Тест создания Volume с невалидной суммой"""
        with pytest.raises(ValueError):
            Volume(amount=Decimal("-100"), currency=Currency.USD)

    def test_volume_creation_zero_amount(self):
        """Тест создания Volume с нулевой суммой"""
        volume = Volume(amount=Decimal("0"), currency=Currency.USD)
        assert volume.amount == Decimal("0")
        assert volume.currency == Currency.USD

    def test_volume_addition(self):
        """Тест сложения Volume объектов"""
        volume1 = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        volume2 = Volume(amount=Decimal("500.25"), currency=Currency.USD)
        result = volume1 + volume2
        assert result.amount == Decimal("1500.75")
        assert result.currency == Currency.USD

    def test_volume_subtraction(self):
        """Тест вычитания Volume объектов"""
        volume1 = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        volume2 = Volume(amount=Decimal("300.25"), currency=Currency.USD)
        result = volume1 - volume2
        assert result.amount == Decimal("700.25")
        assert result.currency == Currency.USD

    def test_volume_subtraction_negative_result(self):
        """Тест вычитания с отрицательным результатом"""
        volume1 = Volume(amount=Decimal("500.00"), currency=Currency.USD)
        volume2 = Volume(amount=Decimal("1000.00"), currency=Currency.USD)
        with pytest.raises(ValueError):
            volume1 - volume2

    def test_volume_multiplication(self):
        """Тест умножения Volume на число"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        result = volume * Decimal("2.5")
        assert result.amount == Decimal("2501.25")
        assert result.currency == Currency.USD

    def test_volume_division(self):
        """Тест деления Volume на число"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        result = volume / Decimal("2")
        assert result.amount == Decimal("500.25")
        assert result.currency == Currency.USD

    def test_volume_division_by_zero(self):
        """Тест деления Volume на ноль"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        with pytest.raises(ValueError):
            volume / Decimal("0")

    def test_volume_comparison(self):
        """Тест сравнения Volume объектов"""
        volume1 = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        volume2 = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        volume3 = Volume(amount=Decimal("2000.00"), currency=Currency.USD)

        assert volume1 == volume2
        assert volume1 != volume3
        assert volume1 < volume3
        assert volume3 > volume1
        assert volume1 <= volume2
        assert volume1 >= volume2

    def test_volume_percentage_change(self):
        """Тест расчета процентного изменения"""
        old_volume = Volume(amount=Decimal("1000.00"), currency=Currency.USD)
        new_volume = Volume(amount=Decimal("1500.00"), currency=Currency.USD)
        change = new_volume.percentage_change(old_volume)
        assert change == Decimal("50.00")

    def test_volume_percentage_change_decrease(self):
        """Тест расчета процентного изменения (уменьшение)"""
        old_volume = Volume(amount=Decimal("1000.00"), currency=Currency.USD)
        new_volume = Volume(amount=Decimal("500.00"), currency=Currency.USD)
        change = new_volume.percentage_change(old_volume)
        assert change == Decimal("-50.00")

    def test_volume_rounding(self):
        """Тест округления Volume"""
        volume = Volume(amount=Decimal("1000.567"), currency=Currency.USD)
        rounded = volume.round_to_currency_precision()
        assert rounded.amount == Decimal("1000.57")
        assert rounded.currency == Currency.USD

    def test_volume_formatting(self):
        """Тест форматирования Volume"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        formatted = volume.format()
        assert "1000.50" in formatted
        assert "USD" in formatted

    def test_volume_to_dict(self):
        """Тест сериализации Volume в словарь"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        data = volume.to_dict()
        assert data["amount"] == "1000.50"
        assert data["currency"] == "USD"

    def test_volume_from_dict(self):
        """Тест десериализации Volume из словаря"""
        data = {"amount": "1000.50", "currency": "USD"}
        volume = Volume.from_dict(data)
        assert volume.amount == Decimal("1000.50")
        assert volume.currency == Currency.USD

    def test_volume_zero(self):
        """Тест создания нулевого Volume"""
        zero_usd = Volume.zero(Currency.USD)
        assert zero_usd.amount == Decimal("0")
        assert zero_usd.currency == Currency.USD

    def test_volume_is_zero(self):
        """Тест проверки на ноль"""
        zero_volume = Volume.zero(Currency.USD)
        non_zero_volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)

        assert zero_volume.is_zero() is True
        assert non_zero_volume.is_zero() is False

    def test_volume_hash(self):
        """Тест хеширования Volume"""
        volume1 = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        volume2 = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        volume3 = Volume(amount=Decimal("2000.00"), currency=Currency.USD)

        assert hash(volume1) == hash(volume2)
        assert hash(volume1) != hash(volume3)

    def test_volume_repr(self):
        """Тест строкового представления"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        repr_str = repr(volume)
        assert "Volume" in repr_str
        assert "1000.50" in repr_str
        assert "USD" in repr_str

    def test_volume_str(self):
        """Тест строкового представления для пользователя"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        str_repr = str(volume)
        assert "1000.50" in str_repr
        assert "USD" in str_repr

    def test_volume_validation_precision(self):
        """Тест валидации точности"""
        with pytest.raises(ValueError):
            Volume(amount=Decimal("1000.123456"), currency=Currency.USD)

    def test_volume_validation_max_amount(self):
        """Тест валидации максимальной суммы"""
        with pytest.raises(ValueError):
            Volume(amount=Decimal("999999999.99"), currency=Currency.USD)

    def test_volume_with_amount(self):
        """Тест создания нового Volume с измененной суммой"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        new_volume = volume.with_amount(Decimal("2000.00"))
        assert new_volume.amount == Decimal("2000.00")
        assert new_volume.currency == Currency.USD
        assert volume.amount == Decimal("1000.50")  # оригинал не изменился

    def test_volume_with_currency(self):
        """Тест создания нового Volume с измененной валютой"""
        volume = Volume(amount=Decimal("1000.50"), currency=Currency.USD)
        new_volume = volume.with_currency(Currency.EUR)
        assert new_volume.amount == Decimal("1000.50")
        assert new_volume.currency == Currency.EUR
        assert volume.currency == Currency.USD  # оригинал не изменился
