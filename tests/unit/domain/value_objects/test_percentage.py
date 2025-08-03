"""
Unit тесты для percentage.py.

Покрывает:
- Основной функционал Percentage
- Валидацию данных
- Бизнес-логику операций с процентами
- Обработку ошибок
- Анализ рисков и доходности
- Сериализацию и десериализацию
"""

import pytest
import dataclasses
from typing import Dict, Any, Union
from unittest.mock import Mock, patch
from decimal import Decimal, ROUND_HALF_UP

from domain.value_objects.percentage import Percentage
from domain.value_objects.percentage_config import PercentageConfig


class TestPercentage:
    """Тесты для Percentage."""

    @pytest.fixture
    def sample_percentage(self) -> Percentage:
        """Тестовый процент."""
        return Percentage(value=Decimal("25.50"))

    @pytest.fixture
    def negative_percentage(self) -> Percentage:
        """Отрицательный процент."""
        return Percentage(value=Decimal("-10.25"))

    @pytest.fixture
    def high_percentage(self) -> Percentage:
        """Высокий процент."""
        return Percentage(value=Decimal("75.00"))

    def test_percentage_creation(self, sample_percentage):
        """Тест создания процента."""
        assert sample_percentage.value == Decimal("25.50")

    def test_percentage_creation_with_config(self):
        """Тест создания процента с конфигурацией."""
        config = PercentageConfig()
        percentage = Percentage(value=Decimal("50.00"), config=config)
        assert percentage.value == Decimal("50.00")

    def test_percentage_validation_positive_value(self):
        """Тест валидации положительного значения."""
        percentage = Percentage(value=Decimal("100.00"))
        assert percentage.value == Decimal("100.00")
        assert percentage.validate() is True

    def test_percentage_validation_negative_value(self):
        """Тест валидации отрицательного значения."""
        percentage = Percentage(value=Decimal("-50.00"))
        assert percentage.value == Decimal("-50.00")
        assert percentage.validate() is True

    def test_percentage_validation_zero_value(self):
        """Тест валидации нулевого значения."""
        percentage = Percentage(value=Decimal("0.00"))
        assert percentage.value == Decimal("0.00")
        assert percentage.validate() is True

    def test_percentage_validation_exceeds_max(self):
        """Тест валидации превышения максимального значения."""
        with pytest.raises(ValueError, match="Percentage cannot exceed"):
            Percentage(value=Decimal("10001.00"))

    def test_percentage_validation_below_min(self):
        """Тест валидации значения ниже минимального."""
        with pytest.raises(ValueError, match="Percentage cannot be less than"):
            Percentage(value=Decimal("-10001.00"))

    def test_percentage_validation_invalid_type(self):
        """Тест валидации неверного типа."""
        with pytest.raises(ValueError, match="Invalid value type"):
            Percentage(value="invalid")

    def test_percentage_properties(self, sample_percentage):
        """Тест свойств процента."""
        assert sample_percentage.value == Decimal("25.50")

    def test_percentage_hash(self, sample_percentage):
        """Тест хеширования процента."""
        hash_value = sample_percentage.hash
        assert len(hash_value) == 32  # MD5 hex digest length
        assert isinstance(hash_value, str)

    def test_percentage_validation(self, sample_percentage):
        """Тест валидации процента."""
        assert sample_percentage.validate() is True

    def test_percentage_equality(self, sample_percentage):
        """Тест равенства процентов."""
        same_percentage = Percentage(value=Decimal("25.50"))
        different_percentage = Percentage(value=Decimal("50.00"))
        
        assert sample_percentage == same_percentage
        assert sample_percentage != different_percentage
        assert sample_percentage != "not a percentage"

    def test_percentage_hash_equality(self, sample_percentage):
        """Тест хеширования для равенства."""
        same_percentage = Percentage(value=Decimal("25.50"))
        assert hash(sample_percentage) == hash(same_percentage)

    def test_percentage_arithmetic_operations(self, sample_percentage):
        """Тест арифметических операций."""
        # Сложение
        result = sample_percentage + Percentage(value=Decimal("10.00"))
        assert result.value == Decimal("35.50")

        # Сложение с числом
        result = sample_percentage + 10
        assert result.value == Decimal("35.50")

        # Вычитание
        result = sample_percentage - Percentage(value=Decimal("5.00"))
        assert result.value == Decimal("20.50")

        # Умножение
        result = sample_percentage * 2
        assert result.value == Decimal("51.00")

        # Деление на число
        result = sample_percentage / 2
        assert result.value == Decimal("12.75")

        # Деление на процент
        result = sample_percentage / Percentage(value=Decimal("5.00"))
        assert result == Decimal("5.10")

    def test_percentage_arithmetic_errors(self, sample_percentage):
        """Тест ошибок арифметических операций."""
        # Деление на ноль
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            sample_percentage / 0

        # Деление на нулевой процент
        with pytest.raises(ValueError, match="Cannot divide by zero percentage"):
            sample_percentage / Percentage(value=Decimal("0.00"))

    def test_percentage_comparison_operations(self, sample_percentage):
        """Тест операций сравнения."""
        smaller_percentage = Percentage(value=Decimal("10.00"))
        larger_percentage = Percentage(value=Decimal("50.00"))

        # Меньше
        assert smaller_percentage < sample_percentage
        assert sample_percentage < larger_percentage

        # Больше
        assert larger_percentage > sample_percentage
        assert sample_percentage > smaller_percentage

        # Меньше или равно
        assert smaller_percentage <= sample_percentage
        assert sample_percentage <= sample_percentage

        # Больше или равно
        assert larger_percentage >= sample_percentage
        assert sample_percentage >= sample_percentage

    def test_percentage_comparison_errors(self, sample_percentage):
        """Тест ошибок сравнения."""
        with pytest.raises(TypeError, match="Can only compare Percentage with Percentage"):
            sample_percentage < 100

    def test_percentage_round(self, sample_percentage):
        """Тест округления процента."""
        percentage = Percentage(value=Decimal("25.123456"))
        rounded = percentage.round(2)
        assert rounded.value == Decimal("25.12")

    def test_percentage_conversion_methods(self, sample_percentage):
        """Тест методов конвертации."""
        # To float
        float_value = sample_percentage.to_float()
        assert float_value == 25.5
        assert isinstance(float_value, float)

        # To decimal
        decimal_value = sample_percentage.to_decimal()
        assert decimal_value == Decimal("25.50")
        assert isinstance(decimal_value, Decimal)

        # To fraction
        fraction_value = sample_percentage.to_fraction()
        assert fraction_value == Decimal("0.2550")
        assert isinstance(fraction_value, Decimal)

    def test_percentage_application_methods(self, sample_percentage):
        """Тест методов применения процента."""
        base_value = Decimal("1000.00")

        # Применение процента
        result = sample_percentage.apply_to(base_value)
        assert result == Decimal("255.00")  # 25.5% от 1000

        # Увеличение на процент
        result = sample_percentage.increase_by(base_value)
        assert result == Decimal("1255.00")  # 1000 + 25.5%

        # Уменьшение на процент
        result = sample_percentage.decrease_by(base_value)
        assert result == Decimal("745.00")  # 1000 - 25.5%

    def test_percentage_range_check(self, sample_percentage):
        """Тест проверки нахождения в диапазоне."""
        assert sample_percentage.is_within_range(Decimal("0.00"), Decimal("50.00")) is True
        assert sample_percentage.is_within_range(Decimal("30.00"), Decimal("50.00")) is False

    def test_percentage_min_max(self, sample_percentage):
        """Тест методов min и max."""
        other_percentage = Percentage(value=Decimal("50.00"))
        
        min_percentage = sample_percentage.min(other_percentage)
        assert min_percentage.value == Decimal("25.50")
        
        max_percentage = sample_percentage.max(other_percentage)
        assert max_percentage.value == Decimal("50.00")

    def test_percentage_compound_operations(self, sample_percentage):
        """Тест составных операций."""
        other_percentage = Percentage(value=Decimal("10.00"))
        
        # Составной процент
        compound = sample_percentage.compound_with(other_percentage)
        expected = Decimal("25.50") + Decimal("10.00") + (Decimal("25.50") * Decimal("10.00") / Decimal("100"))
        assert compound.value == expected

    def test_percentage_annualization(self, sample_percentage):
        """Тест годового расчета."""
        # Годовой процент для 30 дней
        annualized = sample_percentage.annualize(30)
        expected = Decimal("25.50") * (Decimal("365") / Decimal("30"))
        assert annualized.value == expected

    def test_percentage_risk_analysis(self, sample_percentage, high_percentage):
        """Тест анализа рисков."""
        # Уровень риска
        assert sample_percentage.get_risk_level() == "medium"
        assert high_percentage.get_risk_level() == "high"

        # Проверка высокого риска
        assert sample_percentage.is_high_risk() is False
        assert high_percentage.is_high_risk() is True

        # Приемлемый риск
        assert sample_percentage.is_acceptable_risk(Decimal("30")) is True
        assert high_percentage.is_acceptable_risk(Decimal("30")) is False

    def test_percentage_return_analysis(self, sample_percentage, negative_percentage):
        """Тест анализа доходности."""
        # Рейтинг доходности
        assert sample_percentage.get_return_rating() == "excellent"
        assert negative_percentage.get_return_rating() == "poor"

        # Прибыльность
        assert sample_percentage.is_profitable() is True
        assert negative_percentage.is_profitable() is False

        # Значимая доходность
        assert sample_percentage.is_significant_return(Decimal("5")) is True
        assert negative_percentage.is_significant_return(Decimal("5")) is False

    def test_percentage_compound_growth(self, sample_percentage):
        """Тест расчета сложного роста."""
        # Сложный рост за 3 периода
        growth = sample_percentage.calculate_compound_growth(3)
        expected = ((Decimal("1") + Decimal("25.50") / Decimal("100")) ** 3 - Decimal("1")) * Decimal("100")
        assert growth.value == expected

    def test_percentage_break_even(self, sample_percentage):
        """Тест расчета периода окупаемости."""
        target_return = Percentage(value=Decimal("100.00"))
        periods = sample_percentage.calculate_break_even_periods(target_return)
        assert periods > 0

    def test_percentage_trading_signal(self, sample_percentage):
        """Тест силы торгового сигнала."""
        signal_strength = sample_percentage.get_trading_signal_strength()
        assert isinstance(signal_strength, str)
        assert len(signal_strength) > 0

    def test_percentage_to_dict(self, sample_percentage):
        """Тест сериализации в словарь."""
        result = sample_percentage.to_dict()
        
        assert result["value"] == "25.50"
        assert result["type"] == "Percentage"

    def test_percentage_from_dict(self, sample_percentage):
        """Тест десериализации из словаря."""
        data = {
            "value": "25.50",
            "type": "Percentage"
        }
        
        percentage = Percentage.from_dict(data)
        assert percentage.value == Decimal("25.50")

    def test_percentage_factory_methods(self):
        """Тест фабричных методов."""
        # Zero percentage
        zero_percentage = Percentage.zero()
        assert zero_percentage.value == Decimal("0.00")

        # From decimal
        decimal_percentage = Percentage.from_decimal(Decimal("50.00"))
        assert decimal_percentage.value == Decimal("50.00")

        # From float
        float_percentage = Percentage.from_float(50.0)
        assert float_percentage.value == Decimal("50.00")

        # From string
        string_percentage = Percentage.from_string("50.00")
        assert string_percentage.value == Decimal("50.00")

        # From fraction
        fraction_percentage = Percentage.from_fraction(Decimal("0.50"))
        assert fraction_percentage.value == Decimal("50.00")

    def test_percentage_copy(self, sample_percentage):
        """Тест копирования процента."""
        copied_percentage = sample_percentage.copy()
        assert copied_percentage == sample_percentage
        assert copied_percentage is not sample_percentage

    def test_percentage_str_representation(self, sample_percentage):
        """Тест строкового представления."""
        result = str(sample_percentage)
        assert "25.50%" in result

    def test_percentage_repr_representation(self, sample_percentage):
        """Тест repr представления."""
        result = repr(sample_percentage)
        assert "Percentage" in result
        assert "25.50" in result


class TestPercentageOperations:
    """Тесты операций с процентами."""

    def test_percentage_precision_handling(self):
        """Тест обработки точности."""
        percentage = Percentage(value=Decimal("25.123456"))
        assert percentage.value == Decimal("25.123456")

        # Округление
        rounded = percentage.round(2)
        assert rounded.value == Decimal("25.12")

    def test_percentage_large_numbers(self):
        """Тест больших чисел."""
        large_percentage = Percentage(value=Decimal("9999.99"))
        assert large_percentage.validate() is True

        # Операции с большими числами
        result = large_percentage * 2
        # Должно вызвать ошибку валидации
        with pytest.raises(ValueError):
            result.validate()

    def test_percentage_serialization_roundtrip(self):
        """Тест сериализации и десериализации."""
        original_percentage = Percentage(value=Decimal("123.45"))
        
        # Сериализация
        data = original_percentage.to_dict()
        
        # Десериализация
        restored_percentage = Percentage.from_dict(data)
        
        # Проверка равенства
        assert restored_percentage == original_percentage
        assert restored_percentage.value == original_percentage.value


class TestPercentageRiskAnalysis:
    """Тесты анализа рисков."""

    def test_risk_thresholds(self):
        """Тест порогов риска."""
        # Низкий риск
        low_risk = Percentage(value=Decimal("3.00"))
        assert low_risk.get_risk_level() == "low"
        assert low_risk.is_high_risk() is False

        # Средний риск
        medium_risk = Percentage(value=Decimal("25.00"))
        assert medium_risk.get_risk_level() == "medium"
        assert medium_risk.is_high_risk() is False

        # Высокий риск
        high_risk = Percentage(value=Decimal("75.00"))
        assert high_risk.get_risk_level() == "high"
        assert high_risk.is_high_risk() is True

    def test_acceptable_risk_calculation(self):
        """Тест расчета приемлемого риска."""
        percentages = [
            Percentage(value=Decimal("5.00")),   # Низкий риск
            Percentage(value=Decimal("25.00")),  # Средний риск
            Percentage(value=Decimal("75.00")),  # Высокий риск
        ]
        
        # Проверка с разными порогами
        assert percentages[0].is_acceptable_risk(Decimal("10")) is True
        assert percentages[1].is_acceptable_risk(Decimal("30")) is True
        assert percentages[2].is_acceptable_risk(Decimal("30")) is False


class TestPercentageReturnAnalysis:
    """Тесты анализа доходности."""

    def test_return_ratings(self):
        """Тест рейтингов доходности."""
        # Плохая доходность
        poor_return = Percentage(value=Decimal("-10.00"))
        assert poor_return.get_return_rating() == "poor"
        assert poor_return.is_profitable() is False

        # Умеренная доходность
        moderate_return = Percentage(value=Decimal("3.00"))
        assert moderate_return.get_return_rating() == "moderate"
        assert moderate_return.is_profitable() is True

        # Хорошая доходность
        good_return = Percentage(value=Decimal("15.00"))
        assert good_return.get_return_rating() == "good"
        assert good_return.is_profitable() is True

        # Отличная доходность
        excellent_return = Percentage(value=Decimal("25.00"))
        assert excellent_return.get_return_rating() == "excellent"
        assert excellent_return.is_profitable() is True

    def test_significant_return_calculation(self):
        """Тест расчета значимой доходности."""
        # Значимая доходность
        significant = Percentage(value=Decimal("10.00"))
        assert significant.is_significant_return(Decimal("5")) is True

        # Незначимая доходность
        insignificant = Percentage(value=Decimal("3.00"))
        assert insignificant.is_significant_return(Decimal("5")) is False


class TestPercentageEdgeCases:
    """Тесты граничных случаев для процентов."""

    def test_percentage_minimum_values(self):
        """Тест минимальных значений."""
        min_percentage = Percentage(value=Decimal("-10000.00"))
        assert min_percentage.validate() is True

    def test_percentage_maximum_values(self):
        """Тест максимальных значений."""
        max_percentage = Percentage(value=Decimal("10000.00"))
        assert max_percentage.validate() is True

    def test_percentage_nan_infinite_handling(self):
        """Тест обработки NaN и бесконечности."""
        import math
        
        # NaN
        with pytest.raises(ValueError, match="Percentage cannot be NaN"):
            Percentage(value=Decimal("NaN"))
        
        # Бесконечность
        with pytest.raises(ValueError, match="Percentage cannot be infinite"):
            Percentage(value=Decimal("Infinity"))

    def test_percentage_compound_growth_edge_cases(self):
        """Тест граничных случаев сложного роста."""
        # Нулевой процент
        zero_percentage = Percentage(value=Decimal("0.00"))
        growth = zero_percentage.calculate_compound_growth(10)
        assert growth.value == Decimal("0.00")

        # Отрицательный процент
        negative_percentage = Percentage(value=Decimal("-10.00"))
        growth = negative_percentage.calculate_compound_growth(3)
        assert growth.value < 0

    def test_percentage_break_even_edge_cases(self):
        """Тест граничных случаев окупаемости."""
        # Нулевой процент
        zero_percentage = Percentage(value=Decimal("0.00"))
        target_return = Percentage(value=Decimal("10.00"))
        periods = zero_percentage.calculate_break_even_periods(target_return)
        assert periods == -1  # Невозможно достичь

        # Отрицательный процент
        negative_percentage = Percentage(value=Decimal("-5.00"))
        periods = negative_percentage.calculate_break_even_periods(target_return)
        assert periods == -1  # Невозможно достичь

    def test_percentage_hash_collision_resistance(self):
        """Тест устойчивости к коллизиям хешей."""
        percentages = [
            Percentage(value=Decimal("10.00")),
            Percentage(value=Decimal("20.00")),
            Percentage(value=Decimal("-10.00")),
            Percentage(value=Decimal("0.00")),
        ]
        
        hashes = [percentage.hash for percentage in percentages]
        assert len(hashes) == len(set(hashes))  # Все хеши должны быть уникальными

    def test_percentage_performance(self):
        """Тест производительности операций с процентами."""
        import time
        
        percentage = Percentage(value=Decimal("25.00"))
        
        # Тест скорости арифметических операций
        start_time = time.time()
        for _ in range(1000):
            result = percentage * 2
        end_time = time.time()
        
        # Операция должна выполняться быстро
        assert end_time - start_time < 1.0  # Менее 1 секунды для 1000 операций 