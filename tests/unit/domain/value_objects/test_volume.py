"""
Unit тесты для volume.py.

Покрывает:
- Основной функционал Volume
- Валидацию данных
- Бизнес-логику операций с объемами
- Обработку ошибок
- Анализ ликвидности
- Сериализацию и десериализацию
"""

import pytest
import dataclasses
from typing import Dict, Any, Union
from unittest.mock import Mock, patch
from decimal import Decimal, ROUND_HALF_UP

from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.volume_config import VolumeConfig


class TestVolume:
    """Тесты для Volume."""

    @pytest.fixture
    def sample_volume(self) -> Volume:
        """Тестовый объем."""
        return Volume(
            value=Decimal("1000.00"),
            currency=Currency.USD
        )

    @pytest.fixture
    def btc_volume(self) -> Volume:
        """Объем в BTC."""
        return Volume(
            value=Decimal("1.5"),
            currency=Currency.BTC
        )

    @pytest.fixture
    def large_volume(self) -> Volume:
        """Большой объем."""
        return Volume(
            value=Decimal("1000000.00"),
            currency=Currency.USD
        )

    def test_volume_creation(self, sample_volume):
        """Тест создания объема."""
        assert sample_volume.value == Decimal("1000.00")
        assert sample_volume.currency == Currency.USD
        assert sample_volume.amount == Decimal("1000.00")

    def test_volume_creation_without_currency(self):
        """Тест создания объема без валюты."""
        volume = Volume(value=Decimal("100.00"))
        assert volume.value == Decimal("100.00")
        assert volume.currency is None

    def test_volume_creation_with_config(self):
        """Тест создания объема с конфигурацией."""
        config = VolumeConfig()
        volume = Volume(value=Decimal("100.00"), currency=Currency.USD, config=config)
        assert volume.value == Decimal("100.00")
        assert volume.currency == Currency.USD

    def test_volume_validation_negative_value(self):
        """Тест валидации отрицательного объема."""
        with pytest.raises(ValueError, match="Volume cannot be negative"):
            Volume(value=Decimal("-100.00"))

    def test_volume_validation_zero_value(self):
        """Тест валидации нулевого объема."""
        volume = Volume(value=Decimal("0.00"))
        assert volume.value == Decimal("0.00")
        assert volume.validate() is True

    def test_volume_validation_max_value(self):
        """Тест валидации максимального значения."""
        max_value = Volume.MAX_VOLUME
        volume = Volume(value=max_value)
        assert volume.validate() is True

    def test_volume_validation_exceeds_max(self):
        """Тест валидации превышения максимального значения."""
        with pytest.raises(ValueError, match="Volume cannot exceed"):
            Volume(value=Volume.MAX_VOLUME + Decimal("1"))

    def test_volume_validation_invalid_type(self):
        """Тест валидации неверного типа."""
        with pytest.raises(ValueError, match="Invalid value type"):
            Volume(value="invalid")

    def test_volume_properties(self, sample_volume):
        """Тест свойств объема."""
        assert sample_volume.value == Decimal("1000.00")
        assert sample_volume.amount == Decimal("1000.00")
        assert sample_volume.currency == Currency.USD

    def test_volume_hash(self, sample_volume):
        """Тест хеширования объема."""
        hash_value = sample_volume.hash
        assert len(hash_value) == 32  # MD5 hex digest length
        assert isinstance(hash_value, str)

    def test_volume_validation(self, sample_volume):
        """Тест валидации объема."""
        assert sample_volume.validate() is True

    def test_volume_equality(self, sample_volume):
        """Тест равенства объемов."""
        same_volume = Volume(value=Decimal("1000.00"), currency=Currency.USD)
        different_volume = Volume(value=Decimal("2000.00"), currency=Currency.USD)
        
        assert sample_volume == same_volume
        assert sample_volume != different_volume
        assert sample_volume != "not a volume"

    def test_volume_hash_equality(self, sample_volume):
        """Тест хеширования для равенства."""
        same_volume = Volume(value=Decimal("1000.00"), currency=Currency.USD)
        assert hash(sample_volume) == hash(same_volume)

    def test_volume_arithmetic_operations(self, sample_volume):
        """Тест арифметических операций."""
        # Сложение
        result = sample_volume + Volume(value=Decimal("500.00"), currency=Currency.USD)
        assert result.value == Decimal("1500.00")
        assert result.currency == Currency.USD

        # Сложение с числом
        result = sample_volume + 500
        assert result.value == Decimal("1500.00")

        # Вычитание
        result = sample_volume - Volume(value=Decimal("300.00"), currency=Currency.USD)
        assert result.value == Decimal("700.00")

        # Умножение
        result = sample_volume * 2
        assert result.value == Decimal("2000.00")

        # Деление на число
        result = sample_volume / 2
        assert result.value == Decimal("500.00")

        # Деление на объем
        result = sample_volume / Volume(value=Decimal("100.00"), currency=Currency.USD)
        assert result == Decimal("10.00")

    def test_volume_arithmetic_errors(self, sample_volume):
        """Тест ошибок арифметических операций."""
        # Вычитание большего объема
        with pytest.raises(ValueError, match="Volume difference cannot be negative"):
            sample_volume - Volume(value=Decimal("2000.00"), currency=Currency.USD)

        # Деление на ноль
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            sample_volume / 0

        # Деление на нулевой объем
        with pytest.raises(ValueError, match="Cannot divide by zero volume"):
            sample_volume / Volume(value=Decimal("0.00"), currency=Currency.USD)

    def test_volume_comparison_operations(self, sample_volume):
        """Тест операций сравнения."""
        smaller_volume = Volume(value=Decimal("500.00"), currency=Currency.USD)
        larger_volume = Volume(value=Decimal("2000.00"), currency=Currency.USD)

        # Меньше
        assert smaller_volume < sample_volume
        assert sample_volume < larger_volume

        # Больше
        assert larger_volume > sample_volume
        assert sample_volume > smaller_volume

        # Меньше или равно
        assert smaller_volume <= sample_volume
        assert sample_volume <= sample_volume

        # Больше или равно
        assert larger_volume >= sample_volume
        assert sample_volume >= sample_volume

    def test_volume_comparison_errors(self, sample_volume):
        """Тест ошибок сравнения."""
        with pytest.raises(TypeError, match="Can only compare Volume with Volume"):
            sample_volume < 100

    def test_volume_round(self, sample_volume):
        """Тест округления объема."""
        volume = Volume(value=Decimal("1000.123456"), currency=Currency.USD)
        rounded = volume.round(2)
        assert rounded.value == Decimal("1000.12")

    def test_volume_percentage_of(self, sample_volume):
        """Тест вычисления процента от общего объема."""
        total_volume = Volume(value=Decimal("5000.00"), currency=Currency.USD)
        percentage = sample_volume.percentage_of(total_volume)
        assert percentage == Decimal("20.00")  # 20%

    def test_volume_min_max(self, sample_volume):
        """Тест методов min и max."""
        other_volume = Volume(value=Decimal("2000.00"), currency=Currency.USD)
        
        min_volume = sample_volume.min(other_volume)
        assert min_volume.value == Decimal("1000.00")
        
        max_volume = sample_volume.max(other_volume)
        assert max_volume.value == Decimal("2000.00")

    def test_volume_is_within_range(self, sample_volume):
        """Тест проверки нахождения в диапазоне."""
        min_volume = Volume(value=Decimal("500.00"), currency=Currency.USD)
        max_volume = Volume(value=Decimal("2000.00"), currency=Currency.USD)
        
        assert sample_volume.is_within_range(min_volume, max_volume) is True
        
        # Вне диапазона
        assert sample_volume.is_within_range(min_volume, Volume(value=Decimal("800.00"), currency=Currency.USD)) is False

    def test_volume_percentage_operations(self, sample_volume):
        """Тест операций с процентами."""
        # Применение процента
        result = sample_volume.apply_percentage(Decimal("50.0"))
        assert result.value == Decimal("500.00")

        # Увеличение на процент
        result = sample_volume.increase_by_percentage(Decimal("20.0"))
        assert result.value == Decimal("1200.00")

        # Уменьшение на процент
        result = sample_volume.decrease_by_percentage(Decimal("30.0"))
        assert result.value == Decimal("700.00")

    def test_volume_liquidity_analysis(self, sample_volume, large_volume):
        """Тест анализа ликвидности."""
        # Уровень ликвидности
        assert sample_volume.get_liquidity_level() == "low"
        assert large_volume.get_liquidity_level() == "high"

        # Проверка ликвидности
        assert sample_volume.is_liquid() is False
        assert large_volume.is_liquid() is True

        # Оценка ликвидности
        score = sample_volume.get_liquidity_score()
        assert 0 <= score <= 1

    def test_volume_trading_analysis(self, sample_volume):
        """Тест торгового анализа."""
        # Проверка торговости
        assert sample_volume.is_tradable() is True

        # Рекомендации по торговле
        recommendation = sample_volume.get_trading_recommendation()
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0

    def test_volume_market_impact(self, sample_volume):
        """Тест расчета рыночного влияния."""
        total_market_volume = Volume(value=Decimal("100000.00"), currency=Currency.USD)
        impact = sample_volume.calculate_market_impact(total_market_volume)
        assert impact == Decimal("0.01")  # 1%

    def test_volume_significance(self, sample_volume):
        """Тест проверки значимости объема."""
        # Значимый объем
        assert sample_volume.is_significant_volume(Decimal("0.001")) is True
        
        # Незначимый объем
        small_volume = Volume(value=Decimal("1.00"), currency=Currency.USD)
        assert small_volume.is_significant_volume(Decimal("0.1")) is False

    def test_volume_to_dict(self, sample_volume):
        """Тест сериализации в словарь."""
        result = sample_volume.to_dict()
        
        assert result["value"] == "1000.00"
        assert result["currency"] == "USD"
        assert result["type"] == "Volume"

    def test_volume_from_dict(self, sample_volume):
        """Тест десериализации из словаря."""
        data = {
            "value": "1000.00",
            "currency": "USD",
            "type": "Volume"
        }
        
        volume = Volume.from_dict(data)
        assert volume.value == Decimal("1000.00")
        assert volume.currency == Currency.USD

    def test_volume_from_dict_no_currency(self):
        """Тест десериализации без валюты."""
        data = {
            "value": "1000.00",
            "currency": None,
            "type": "Volume"
        }
        
        volume = Volume.from_dict(data)
        assert volume.value == Decimal("1000.00")
        assert volume.currency is None

    def test_volume_factory_methods(self):
        """Тест фабричных методов."""
        # Zero volume
        zero_volume = Volume.zero(Currency.USD)
        assert zero_volume.value == Decimal("0.00")
        assert zero_volume.currency == Currency.USD

        # From string
        string_volume = Volume.from_string("1000.50", Currency.USD)
        assert string_volume.value == Decimal("1000.50")
        assert string_volume.currency == Currency.USD

        # From float
        float_volume = Volume.from_float(1000.50, Currency.USD)
        assert float_volume.value == Decimal("1000.50")
        assert float_volume.currency == Currency.USD

        # From int
        int_volume = Volume.from_int(1000, Currency.USD)
        assert int_volume.value == Decimal("1000")
        assert int_volume.currency == Currency.USD

    def test_volume_conversion_methods(self, sample_volume):
        """Тест методов конвертации."""
        # To float
        float_value = sample_volume.to_float()
        assert float_value == 1000.0
        assert isinstance(float_value, float)

        # To decimal
        decimal_value = sample_volume.to_decimal()
        assert decimal_value == Decimal("1000.00")
        assert isinstance(decimal_value, Decimal)

    def test_volume_copy(self, sample_volume):
        """Тест копирования объема."""
        copied_volume = sample_volume.copy()
        assert copied_volume == sample_volume
        assert copied_volume is not sample_volume

    def test_volume_str_representation(self, sample_volume):
        """Тест строкового представления."""
        result = str(sample_volume)
        assert "1000.00" in result
        assert "USD" in result

    def test_volume_repr_representation(self, sample_volume):
        """Тест repr представления."""
        result = repr(sample_volume)
        assert "Volume" in result
        assert "1000.00" in result
        assert "USD" in result


class TestVolumeOperations:
    """Тесты операций с объемами."""

    def test_volume_precision_handling(self):
        """Тест обработки точности."""
        volume = Volume(value=Decimal("100.123456"), currency=Currency.USD)
        assert volume.value == Decimal("100.123456")

        # Округление
        rounded = volume.round(2)
        assert rounded.value == Decimal("100.12")

    def test_volume_large_numbers(self):
        """Тест больших чисел."""
        large_volume = Volume(value=Decimal("999999999.99999999"), currency=Currency.USD)
        assert large_volume.validate() is True

        # Операции с большими числами
        result = large_volume * 2
        assert result.value == Decimal("1999999999.99999998")

    def test_volume_currency_consistency(self):
        """Тест консистентности валют."""
        # При сложении объемов с разными валютами
        volume1 = Volume(value=Decimal("100.00"), currency=Currency.USD)
        volume2 = Volume(value=Decimal("200.00"))  # Без валюты
        
        result = volume1 + volume2
        assert result.currency == Currency.USD

    def test_volume_serialization_roundtrip(self):
        """Тест сериализации и десериализации."""
        original_volume = Volume(value=Decimal("12345.67"), currency=Currency.ETH)
        
        # Сериализация
        data = original_volume.to_dict()
        
        # Десериализация
        restored_volume = Volume.from_dict(data)
        
        # Проверка равенства
        assert restored_volume == original_volume
        assert restored_volume.value == original_volume.value
        assert restored_volume.currency == original_volume.currency


class TestVolumeLiquidityAnalysis:
    """Тесты анализа ликвидности."""

    def test_liquidity_thresholds(self):
        """Тест порогов ликвидности."""
        # Низкая ликвидность
        low_volume = Volume(value=Decimal("5000.00"), currency=Currency.USD)
        assert low_volume.get_liquidity_level() == "low"
        assert low_volume.is_liquid() is False

        # Средняя ликвидность
        medium_volume = Volume(value=Decimal("50000.00"), currency=Currency.USD)
        assert medium_volume.get_liquidity_level() == "medium"

        # Высокая ликвидность
        high_volume = Volume(value=Decimal("500000.00"), currency=Currency.USD)
        assert high_volume.get_liquidity_level() == "high"
        assert high_volume.is_liquid() is True

    def test_liquidity_score_calculation(self):
        """Тест расчета оценки ликвидности."""
        volumes = [
            Volume(value=Decimal("1000.00"), currency=Currency.USD),   # Низкая
            Volume(value=Decimal("50000.00"), currency=Currency.USD),  # Средняя
            Volume(value=Decimal("500000.00"), currency=Currency.USD), # Высокая
        ]
        
        scores = [volume.get_liquidity_score() for volume in volumes]
        
        # Оценки должны быть в диапазоне [0, 1]
        for score in scores:
            assert 0 <= score <= 1
        
        # Оценки должны быть упорядочены по ликвидности
        assert scores[0] < scores[1] < scores[2]

    def test_trading_recommendations(self):
        """Тест торговых рекомендаций."""
        # Очень маленький объем
        tiny_volume = Volume(value=Decimal("0.0001"), currency=Currency.USD)
        recommendation = tiny_volume.get_trading_recommendation()
        assert "small" in recommendation.lower() or "avoid" in recommendation.lower()

        # Нормальный объем
        normal_volume = Volume(value=Decimal("1000.00"), currency=Currency.USD)
        recommendation = normal_volume.get_trading_recommendation()
        assert len(recommendation) > 0

        # Очень большой объем
        huge_volume = Volume(value=Decimal("10000000.00"), currency=Currency.USD)
        recommendation = huge_volume.get_trading_recommendation()
        assert "large" in recommendation.lower() or "impact" in recommendation.lower()


class TestVolumeEdgeCases:
    """Тесты граничных случаев для объемов."""

    def test_volume_minimum_values(self):
        """Тест минимальных значений."""
        min_volume = Volume(value=Decimal("0.00000001"), currency=Currency.USD)
        assert min_volume.validate() is True
        assert min_volume.value > 0

    def test_volume_maximum_values(self):
        """Тест максимальных значений."""
        max_volume = Volume(value=Volume.MAX_VOLUME, currency=Currency.USD)
        assert max_volume.validate() is True

    def test_volume_nan_infinite_handling(self):
        """Тест обработки NaN и бесконечности."""
        import math
        
        # NaN
        with pytest.raises(ValueError, match="Volume cannot be NaN"):
            Volume(value=Decimal("NaN"))
        
        # Бесконечность
        with pytest.raises(ValueError, match="Volume cannot be infinite"):
            Volume(value=Decimal("Infinity"))

    def test_volume_market_impact_edge_cases(self):
        """Тест граничных случаев рыночного влияния."""
        # Нулевой общий объем рынка
        volume = Volume(value=Decimal("1000.00"), currency=Currency.USD)
        total_market = Volume(value=Decimal("0.00"), currency=Currency.USD)
        
        impact = volume.calculate_market_impact(total_market)
        assert impact == Decimal("1.0")  # 100% влияние

        # Очень маленький общий объем
        small_market = Volume(value=Decimal("1.00"), currency=Currency.USD)
        impact = volume.calculate_market_impact(small_market)
        assert impact == Decimal("1000.0")  # 100000% влияние

    def test_volume_significance_edge_cases(self):
        """Тест граничных случаев значимости."""
        # Нулевой порог
        volume = Volume(value=Decimal("1000.00"), currency=Currency.USD)
        assert volume.is_significant_volume(Decimal("0.0")) is True

        # Очень высокий порог
        assert volume.is_significant_volume(Decimal("1.0")) is False

    def test_volume_hash_collision_resistance(self):
        """Тест устойчивости к коллизиям хешей."""
        volumes = [
            Volume(value=Decimal("100.00"), currency=Currency.USD),
            Volume(value=Decimal("200.00"), currency=Currency.USD),
            Volume(value=Decimal("100.00"), currency=Currency.EUR),
            Volume(value=Decimal("100.00")),  # Без валюты
        ]
        
        hashes = [volume.hash for volume in volumes]
        assert len(hashes) == len(set(hashes))  # Все хеши должны быть уникальными

    def test_volume_performance(self):
        """Тест производительности операций с объемами."""
        import time
        
        volume = Volume(value=Decimal("1000.00"), currency=Currency.USD)
        
        # Тест скорости арифметических операций
        start_time = time.time()
        for _ in range(1000):
            result = volume * 2
        end_time = time.time()
        
        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0  # Менее 1 секунды для 1000 операций 