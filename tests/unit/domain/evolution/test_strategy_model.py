"""
Unit тесты для domain/evolution/strategy_model.py
"""

import pytest
from decimal import Decimal
from datetime import datetime
from uuid import UUID, uuid4
from typing import Dict, Any

from domain.evolution.strategy_model import (
    EvolutionStatus,
    IndicatorType,
    FilterType,
    IndicatorConfig,
    FilterConfig,
    EntryRule,
    ExitRule,
    StrategyCandidate,
    EvolutionContext,
)
from domain.types.technical_types import SignalType
from domain.types.strategy_types import StrategyType


class TestEvolutionStatus:
    """Тесты для EvolutionStatus enum."""

    def test_enum_values(self):
        """Проверка значений enum."""
        assert EvolutionStatus.GENERATED.value == "generated"
        assert EvolutionStatus.TESTING.value == "testing"
        assert EvolutionStatus.EVALUATED.value == "evaluated"
        assert EvolutionStatus.APPROVED.value == "approved"
        assert EvolutionStatus.REJECTED.value == "rejected"
        assert EvolutionStatus.ARCHIVED.value == "archived"
        assert EvolutionStatus.DEPRECATED.value == "deprecated"


class TestIndicatorType:
    """Тесты для IndicatorType enum."""

    def test_enum_values(self):
        """Проверка значений enum."""
        assert IndicatorType.TREND.value == "trend"
        assert IndicatorType.MOMENTUM.value == "momentum"
        assert IndicatorType.VOLATILITY.value == "volatility"
        assert IndicatorType.VOLUME.value == "volume"
        assert IndicatorType.SUPPORT_RESISTANCE.value == "support_resistance"
        assert IndicatorType.OSCILLATOR.value == "oscillator"
        assert IndicatorType.CUSTOM.value == "custom"


class TestFilterType:
    """Тесты для FilterType enum."""

    def test_enum_values(self):
        """Проверка значений enum."""
        assert FilterType.VOLATILITY.value == "volatility"
        assert FilterType.VOLUME.value == "volume"
        assert FilterType.TREND.value == "trend"
        assert FilterType.TIME.value == "time"
        assert FilterType.CORRELATION.value == "correlation"
        assert FilterType.MARKET_REGIME.value == "market_regime"
        assert FilterType.CUSTOM.value == "custom"


class TestIndicatorConfig:
    """Тесты для IndicatorConfig."""

    def test_creation_with_defaults(self):
        """Создание с значениями по умолчанию."""
        config = IndicatorConfig()
        assert isinstance(config.id, UUID)
        assert config.name == ""
        assert config.indicator_type == IndicatorType.TREND
        assert config.parameters == {}
        assert config.weight == Decimal("1.0")
        assert config.is_active is True

    def test_creation_with_custom_values(self):
        """Создание с пользовательскими значениями."""
        config = IndicatorConfig(
            name="RSI",
            indicator_type=IndicatorType.OSCILLATOR,
            parameters={"period": 14},
            weight=Decimal("0.8"),
            is_active=False,
        )
        assert config.name == "RSI"
        assert config.indicator_type == IndicatorType.OSCILLATOR
        assert config.parameters == {"period": 14}
        assert config.weight == Decimal("0.8")
        assert config.is_active is False

    def test_post_init_converts_weight_to_decimal(self):
        """Проверка конвертации weight в Decimal."""
        config = IndicatorConfig(weight=0.5)
        assert config.weight == Decimal("0.5")

    def test_get_parameter(self):
        """Получение параметра."""
        config = IndicatorConfig(parameters={"period": 14, "overbought": 70})
        assert config.get_parameter("period") == 14
        assert config.get_parameter("overbought") == 70
        assert config.get_parameter("nonexistent", "default") == "default"

    def test_set_parameter(self):
        """Установка параметра."""
        config = IndicatorConfig()
        config.set_parameter("period", 20)
        assert config.parameters["period"] == 20

    def test_validate_parameters_valid(self):
        """Валидация корректных параметров."""
        config = IndicatorConfig(
            name="RSI",
            weight=Decimal("0.8"),
            parameters={"period": 14},
        )
        errors = config.validate_parameters()
        assert len(errors) == 0

    def test_validate_parameters_missing_name(self):
        """Валидация с отсутствующим именем."""
        config = IndicatorConfig(name="")
        errors = config.validate_parameters()
        assert "Indicator name is required" in errors

    def test_validate_parameters_negative_weight(self):
        """Валидация с отрицательным весом."""
        config = IndicatorConfig(name="RSI", weight=Decimal("-0.1"))
        errors = config.validate_parameters()
        assert "Indicator weight cannot be negative" in errors

    def test_validate_parameters_excessive_weight(self):
        """Валидация с чрезмерным весом."""
        config = IndicatorConfig(name="RSI", weight=Decimal("15.0"))
        errors = config.validate_parameters()
        assert "Indicator weight cannot exceed 10" in errors

    def test_validate_parameters_trend_indicator(self):
        """Валидация параметров трендового индикатора."""
        config = IndicatorConfig(
            name="SMA",
            indicator_type=IndicatorType.TREND,
            parameters={"period": 0},
        )
        errors = config.validate_parameters()
        assert "Trend indicator period must be positive" in errors

    def test_validate_parameters_momentum_indicator(self):
        """Валидация параметров индикатора импульса."""
        config = IndicatorConfig(
            name="RSI",
            indicator_type=IndicatorType.MOMENTUM,
            parameters={"period": -1},
        )
        errors = config.validate_parameters()
        assert "Momentum indicator period must be positive" in errors

    def test_validate_parameters_volatility_indicator(self):
        """Валидация параметров индикатора волатильности."""
        config = IndicatorConfig(
            name="Bollinger",
            indicator_type=IndicatorType.VOLATILITY,
            parameters={"period": 0, "std_dev": -1},
        )
        errors = config.validate_parameters()
        assert "Volatility indicator period must be positive" in errors
        assert "Standard deviation must be positive" in errors

    def test_to_dict(self):
        """Преобразование в словарь."""
        config = IndicatorConfig(
            name="RSI",
            indicator_type=IndicatorType.OSCILLATOR,
            parameters={"period": 14},
            weight=Decimal("0.8"),
            is_active=False,
        )
        data = config.to_dict()
        assert data["name"] == "RSI"
        assert data["indicator_type"] == "oscillator"
        assert data["parameters"] == {"period": 14}
        assert data["weight"] == "0.8"
        assert data["is_active"] is False
        assert "id" in data

    def test_from_dict(self):
        """Создание из словаря."""
        data = {
            "id": str(uuid4()),
            "name": "RSI",
            "indicator_type": "oscillator",
            "parameters": {"period": 14},
            "weight": "0.8",
            "is_active": False,
        }
        config = IndicatorConfig.from_dict(data)
        assert config.name == "RSI"
        assert config.indicator_type == IndicatorType.OSCILLATOR
        assert config.parameters == {"period": 14}
        assert config.weight == Decimal("0.8")
        assert config.is_active is False

    def test_clone(self):
        """Клонирование индикатора."""
        original = IndicatorConfig(
            name="RSI",
            indicator_type=IndicatorType.OSCILLATOR,
            parameters={"period": 14},
            weight=Decimal("0.8"),
        )
        cloned = original.clone()
        assert cloned.id != original.id
        assert cloned.name == original.name
        assert cloned.indicator_type == original.indicator_type
        assert cloned.parameters == original.parameters
        assert cloned.weight == original.weight
        assert cloned.is_active == original.is_active


class TestFilterConfig:
    """Тесты для FilterConfig."""

    def test_creation_with_defaults(self):
        """Создание с значениями по умолчанию."""
        config = FilterConfig()
        assert isinstance(config.id, UUID)
        assert config.name == ""
        assert config.filter_type == FilterType.VOLATILITY
        assert config.parameters == {}
        assert config.threshold == Decimal("0.5")
        assert config.is_active is True

    def test_creation_with_custom_values(self):
        """Создание с пользовательскими значениями."""
        config = FilterConfig(
            name="Volume Filter",
            filter_type=FilterType.VOLUME,
            parameters={"min_volume": 1000},
            threshold=Decimal("0.7"),
            is_active=False,
        )
        assert config.name == "Volume Filter"
        assert config.filter_type == FilterType.VOLUME
        assert config.parameters == {"min_volume": 1000}
        assert config.threshold == Decimal("0.7")
        assert config.is_active is False

    def test_post_init_converts_threshold_to_decimal(self):
        """Проверка конвертации threshold в Decimal."""
        config = FilterConfig(threshold=0.3)
        assert config.threshold == Decimal("0.3")

    def test_get_parameter(self):
        """Получение параметра."""
        config = FilterConfig(parameters={"min_volume": 1000, "spike_threshold": 2.0})
        assert config.get_parameter("min_volume") == 1000
        assert config.get_parameter("spike_threshold") == 2.0
        assert config.get_parameter("nonexistent", "default") == "default"

    def test_set_parameter(self):
        """Установка параметра."""
        config = FilterConfig()
        config.set_parameter("min_volume", 2000)
        assert config.parameters["min_volume"] == 2000

    def test_validate_parameters_valid(self):
        """Валидация корректных параметров."""
        config = FilterConfig(
            name="Volume Filter",
            threshold=Decimal("0.7"),
            parameters={"min_volume": 1000},
        )
        errors = config.validate_parameters()
        assert len(errors) == 0

    def test_validate_parameters_missing_name(self):
        """Валидация с отсутствующим именем."""
        config = FilterConfig(name="")
        errors = config.validate_parameters()
        assert "Filter name is required" in errors

    def test_validate_parameters_negative_threshold(self):
        """Валидация с отрицательным порогом."""
        config = FilterConfig(name="Filter", threshold=Decimal("-0.1"))
        errors = config.validate_parameters()
        assert "Filter threshold cannot be negative" in errors

    def test_validate_parameters_excessive_threshold(self):
        """Валидация с чрезмерным порогом."""
        config = FilterConfig(name="Filter", threshold=Decimal("1.5"))
        errors = config.validate_parameters()
        assert "Filter threshold cannot exceed 1" in errors

    def test_validate_parameters_volatility_filter(self):
        """Валидация параметров фильтра волатильности."""
        config = FilterConfig(
            name="ATR Filter",
            filter_type=FilterType.VOLATILITY,
            parameters={"min_atr": 0.02, "max_atr": 0.01},
        )
        errors = config.validate_parameters()
        assert "Min ATR must be less than max ATR" in errors

    def test_validate_parameters_volume_filter(self):
        """Валидация параметров фильтра объема."""
        config = FilterConfig(
            name="Volume Filter",
            filter_type=FilterType.VOLUME,
            parameters={"min_volume": 0, "spike_threshold": -1},
        )
        errors = config.validate_parameters()
        assert "Min volume must be positive" in errors
        assert "Spike threshold must be positive" in errors

    def test_validate_parameters_time_filter(self):
        """Валидация параметров временного фильтра."""
        config = FilterConfig(
            name="Time Filter",
            filter_type=FilterType.TIME,
            parameters={"start_hour": 25, "end_hour": -1},
        )
        errors = config.validate_parameters()
        assert "Hours must be between 0 and 23" in errors

    def test_to_dict(self):
        """Преобразование в словарь."""
        config = FilterConfig(
            name="Volume Filter",
            filter_type=FilterType.VOLUME,
            parameters={"min_volume": 1000},
            threshold=Decimal("0.7"),
            is_active=False,
        )
        data = config.to_dict()
        assert data["name"] == "Volume Filter"
        assert data["filter_type"] == "volume"
        assert data["parameters"] == {"min_volume": 1000}
        assert data["threshold"] == "0.7"
        assert data["is_active"] is False
        assert "id" in data

    def test_from_dict(self):
        """Создание из словаря."""
        data = {
            "id": str(uuid4()),
            "name": "Volume Filter",
            "filter_type": "volume",
            "parameters": {"min_volume": 1000},
            "threshold": "0.7",
            "is_active": False,
        }
        config = FilterConfig.from_dict(data)
        assert config.name == "Volume Filter"
        assert config.filter_type == FilterType.VOLUME
        assert config.parameters == {"min_volume": 1000}
        assert config.threshold == Decimal("0.7")
        assert config.is_active is False

    def test_clone(self):
        """Клонирование фильтра."""
        original = FilterConfig(
            name="Volume Filter",
            filter_type=FilterType.VOLUME,
            parameters={"min_volume": 1000},
            threshold=Decimal("0.7"),
        )
        cloned = original.clone()
        assert cloned.id != original.id
        assert cloned.name == original.name
        assert cloned.filter_type == original.filter_type
        assert cloned.parameters == original.parameters
        assert cloned.threshold == original.threshold
        assert cloned.is_active == original.is_active


class TestEntryRule:
    """Тесты для EntryRule."""

    def test_creation_with_defaults(self):
        """Создание с значениями по умолчанию."""
        rule = EntryRule()
        assert isinstance(rule.id, UUID)
        assert rule.conditions == []
        assert rule.signal_type == SignalType.BUY
        assert rule.confidence_threshold == Decimal("0.7")
        assert rule.volume_ratio == Decimal("1.0")
        assert rule.is_active is True

    def test_creation_with_custom_values(self):
        """Создание с пользовательскими значениями."""
        conditions = [{"indicator": "RSI", "condition": "above", "value": 70}]
        rule = EntryRule(
            conditions=conditions,
            signal_type=SignalType.SELL,
            confidence_threshold=Decimal("0.8"),
            volume_ratio=Decimal("1.5"),
            is_active=False,
        )
        assert rule.conditions == conditions
        assert rule.signal_type == SignalType.SELL
        assert rule.confidence_threshold == Decimal("0.8")
        assert rule.volume_ratio == Decimal("1.5")
        assert rule.is_active is False

    def test_post_init_converts_decimals(self):
        """Проверка конвертации в Decimal."""
        rule = EntryRule(confidence_threshold=0.6, volume_ratio=1.2)
        assert rule.confidence_threshold == Decimal("0.6")
        assert rule.volume_ratio == Decimal("1.2")

    def test_add_condition(self):
        """Добавление условия."""
        rule = EntryRule()
        condition = {"indicator": "RSI", "condition": "above", "value": 70}
        rule.add_condition(condition)
        assert len(rule.conditions) == 1
        assert rule.conditions[0] == condition

    def test_remove_condition_valid_index(self):
        """Удаление условия по валидному индексу."""
        rule = EntryRule()
        condition1 = {"indicator": "RSI", "condition": "above", "value": 70}
        condition2 = {"indicator": "MACD", "condition": "cross", "value": 0}
        rule.add_condition(condition1)
        rule.add_condition(condition2)
        rule.remove_condition(0)
        assert len(rule.conditions) == 1
        assert rule.conditions[0] == condition2

    def test_remove_condition_invalid_index(self):
        """Удаление условия по невалидному индексу."""
        rule = EntryRule()
        condition = {"indicator": "RSI", "condition": "above", "value": 70}
        rule.add_condition(condition)
        rule.remove_condition(5)  # Несуществующий индекс
        assert len(rule.conditions) == 1  # Условие не удалено

    def test_validate_conditions_valid(self):
        """Валидация корректных условий."""
        conditions = [{"indicator": "RSI", "condition": "above", "value": 70}]
        rule = EntryRule(conditions=conditions)
        errors = rule.validate_conditions()
        assert len(errors) == 0

    def test_validate_conditions_empty(self):
        """Валидация пустых условий."""
        rule = EntryRule(conditions=[])
        errors = rule.validate_conditions()
        assert "At least one condition is required" in errors

    def test_validate_conditions_invalid_threshold(self):
        """Валидация с невалидным порогом уверенности."""
        rule = EntryRule(confidence_threshold=Decimal("1.5"))
        errors = rule.validate_conditions()
        assert "Confidence threshold must be between 0 and 1" in errors

    def test_validate_conditions_negative_volume_ratio(self):
        """Валидация с отрицательным соотношением объема."""
        rule = EntryRule(volume_ratio=Decimal("-0.1"))
        errors = rule.validate_conditions()
        assert "Volume ratio cannot be negative" in errors

    def test_validate_conditions_missing_indicator(self):
        """Валидация с отсутствующим индикатором."""
        conditions = [{"condition": "above", "value": 70}]
        rule = EntryRule(conditions=conditions)
        errors = rule.validate_conditions()
        assert "Condition 0: indicator is required" in errors

    def test_validate_conditions_missing_condition(self):
        """Валидация с отсутствующим типом условия."""
        conditions = [{"indicator": "RSI", "value": 70}]
        rule = EntryRule(conditions=conditions)
        errors = rule.validate_conditions()
        assert "Condition 0: condition type is required" in errors

    def test_validate_parameters_alias(self):
        """Проверка алиаса validate_parameters."""
        rule = EntryRule()
        errors1 = rule.validate_conditions()
        errors2 = rule.validate_parameters()
        assert errors1 == errors2

    def test_to_dict(self):
        """Преобразование в словарь."""
        conditions = [{"indicator": "RSI", "condition": "above", "value": 70}]
        rule = EntryRule(
            conditions=conditions,
            signal_type=SignalType.SELL,
            confidence_threshold=Decimal("0.8"),
            volume_ratio=Decimal("1.5"),
            is_active=False,
        )
        data = rule.to_dict()
        assert data["conditions"] == conditions
        assert data["signal_type"] == "sell"
        assert data["confidence_threshold"] == "0.8"
        assert data["volume_ratio"] == "1.5"
        assert data["is_active"] is False
        assert "id" in data

    def test_from_dict(self):
        """Создание из словаря."""
        conditions = [{"indicator": "RSI", "condition": "above", "value": 70}]
        data = {
            "id": str(uuid4()),
            "conditions": conditions,
            "signal_type": "sell",
            "confidence_threshold": "0.8",
            "volume_ratio": "1.5",
            "is_active": False,
        }
        rule = EntryRule.from_dict(data)
        assert rule.conditions == conditions
        assert rule.signal_type == SignalType.SELL
        assert rule.confidence_threshold == Decimal("0.8")
        assert rule.volume_ratio == Decimal("1.5")
        assert rule.is_active is False

    def test_clone(self):
        """Клонирование правила входа."""
        conditions = [{"indicator": "RSI", "condition": "above", "value": 70}]
        original = EntryRule(
            conditions=conditions,
            signal_type=SignalType.SELL,
            confidence_threshold=Decimal("0.8"),
            volume_ratio=Decimal("1.5"),
        )
        cloned = original.clone()
        assert cloned.id != original.id
        assert cloned.conditions == original.conditions
        assert cloned.signal_type == original.signal_type
        assert cloned.confidence_threshold == original.confidence_threshold
        assert cloned.volume_ratio == original.volume_ratio
        assert cloned.is_active == original.is_active


class TestExitRule:
    """Тесты для ExitRule."""

    def test_creation_with_defaults(self):
        """Создание с значениями по умолчанию."""
        rule = ExitRule()
        assert isinstance(rule.id, UUID)
        assert rule.conditions == []
        assert rule.signal_type == SignalType.SELL
        assert rule.stop_loss_pct == Decimal("0.02")
        assert rule.take_profit_pct == Decimal("0.04")
        assert rule.trailing_stop is False
        assert rule.trailing_distance == Decimal("0.01")
        assert rule.is_active is True

    def test_creation_with_custom_values(self):
        """Создание с пользовательскими значениями."""
        conditions = [{"indicator": "RSI", "condition": "below", "value": 30}]
        rule = ExitRule(
            conditions=conditions,
            signal_type=SignalType.BUY,
            stop_loss_pct=Decimal("0.03"),
            take_profit_pct=Decimal("0.06"),
            trailing_stop=True,
            trailing_distance=Decimal("0.02"),
            is_active=False,
        )
        assert rule.conditions == conditions
        assert rule.signal_type == SignalType.BUY
        assert rule.stop_loss_pct == Decimal("0.03")
        assert rule.take_profit_pct == Decimal("0.06")
        assert rule.trailing_stop is True
        assert rule.trailing_distance == Decimal("0.02")
        assert rule.is_active is False

    def test_post_init_converts_decimals(self):
        """Проверка конвертации в Decimal."""
        rule = ExitRule(
            stop_loss_pct=0.025,
            take_profit_pct=0.05,
            trailing_distance=0.015,
        )
        assert rule.stop_loss_pct == Decimal("0.025")
        assert rule.take_profit_pct == Decimal("0.05")
        assert rule.trailing_distance == Decimal("0.015")

    def test_add_condition(self):
        """Добавление условия."""
        rule = ExitRule()
        condition = {"indicator": "RSI", "condition": "below", "value": 30}
        rule.add_condition(condition)
        assert len(rule.conditions) == 1
        assert rule.conditions[0] == condition

    def test_remove_condition_valid_index(self):
        """Удаление условия по валидному индексу."""
        rule = ExitRule()
        condition1 = {"indicator": "RSI", "condition": "below", "value": 30}
        condition2 = {"indicator": "MACD", "condition": "cross", "value": 0}
        rule.add_condition(condition1)
        rule.add_condition(condition2)
        rule.remove_condition(0)
        assert len(rule.conditions) == 1
        assert rule.conditions[0] == condition2

    def test_remove_condition_invalid_index(self):
        """Удаление условия по невалидному индексу."""
        rule = ExitRule()
        condition = {"indicator": "RSI", "condition": "below", "value": 30}
        rule.add_condition(condition)
        rule.remove_condition(5)  # Несуществующий индекс
        assert len(rule.conditions) == 1  # Условие не удалено

    def test_validate_parameters_valid(self):
        """Валидация корректных параметров."""
        rule = ExitRule()
        errors = rule.validate_parameters()
        assert len(errors) == 0

    def test_validate_parameters_negative_stop_loss(self):
        """Валидация с отрицательным стоп-лоссом."""
        rule = ExitRule(stop_loss_pct=Decimal("-0.01"))
        errors = rule.validate_parameters()
        assert "Stop loss percentage cannot be negative" in errors

    def test_validate_parameters_negative_take_profit(self):
        """Валидация с отрицательным тейк-профитом."""
        rule = ExitRule(take_profit_pct=Decimal("-0.01"))
        errors = rule.validate_parameters()
        assert "Take profit percentage cannot be negative" in errors

    def test_validate_parameters_negative_trailing_distance(self):
        """Валидация с отрицательным trailing distance."""
        rule = ExitRule(trailing_stop=True, trailing_distance=Decimal("-0.01"))
        errors = rule.validate_parameters()
        assert "Trailing distance cannot be negative" in errors

    def test_validate_parameters_missing_indicator(self):
        """Валидация с отсутствующим индикатором."""
        conditions = [{"condition": "below", "value": 30}]
        rule = ExitRule(conditions=conditions)
        errors = rule.validate_parameters()
        assert "Condition 0: indicator is required" in errors

    def test_validate_parameters_missing_condition(self):
        """Валидация с отсутствующим типом условия."""
        conditions = [{"indicator": "RSI", "value": 30}]
        rule = ExitRule(conditions=conditions)
        errors = rule.validate_parameters()
        assert "Condition 0: condition type is required" in errors

    def test_to_dict(self):
        """Преобразование в словарь."""
        conditions = [{"indicator": "RSI", "condition": "below", "value": 30}]
        rule = ExitRule(
            conditions=conditions,
            signal_type=SignalType.BUY,
            stop_loss_pct=Decimal("0.03"),
            take_profit_pct=Decimal("0.06"),
            trailing_stop=True,
            trailing_distance=Decimal("0.02"),
            is_active=False,
        )
        data = rule.to_dict()
        assert data["conditions"] == conditions
        assert data["signal_type"] == "buy"
        assert data["stop_loss_pct"] == "0.03"
        assert data["take_profit_pct"] == "0.06"
        assert data["trailing_stop"] is True
        assert data["trailing_distance"] == "0.02"
        assert data["is_active"] is False
        assert "id" in data

    def test_from_dict(self):
        """Создание из словаря."""
        conditions = [{"indicator": "RSI", "condition": "below", "value": 30}]
        data = {
            "id": str(uuid4()),
            "conditions": conditions,
            "signal_type": "buy",
            "stop_loss_pct": "0.03",
            "take_profit_pct": "0.06",
            "trailing_stop": True,
            "trailing_distance": "0.02",
            "is_active": False,
        }
        rule = ExitRule.from_dict(data)
        assert rule.conditions == conditions
        assert rule.signal_type == SignalType.BUY
        assert rule.stop_loss_pct == Decimal("0.03")
        assert rule.take_profit_pct == Decimal("0.06")
        assert rule.trailing_stop is True
        assert rule.trailing_distance == Decimal("0.02")
        assert rule.is_active is False

    def test_clone(self):
        """Клонирование правила выхода."""
        conditions = [{"indicator": "RSI", "condition": "below", "value": 30}]
        original = ExitRule(
            conditions=conditions,
            signal_type=SignalType.BUY,
            stop_loss_pct=Decimal("0.03"),
            take_profit_pct=Decimal("0.06"),
            trailing_stop=True,
            trailing_distance=Decimal("0.02"),
        )
        cloned = original.clone()
        assert cloned.id != original.id
        assert cloned.conditions == original.conditions
        assert cloned.signal_type == original.signal_type
        assert cloned.stop_loss_pct == original.stop_loss_pct
        assert cloned.take_profit_pct == original.take_profit_pct
        assert cloned.trailing_stop == original.trailing_stop
        assert cloned.trailing_distance == original.trailing_distance
        assert cloned.is_active == original.is_active


class TestStrategyCandidate:
    """Тесты для StrategyCandidate."""

    def test_creation_with_defaults(self):
        """Создание с значениями по умолчанию."""
        candidate = StrategyCandidate()
        assert isinstance(candidate.id, UUID)
        assert candidate.name == ""
        assert candidate.description == ""
        assert candidate.strategy_type == StrategyType.TREND
        assert candidate.status == EvolutionStatus.GENERATED
        assert candidate.indicators == []
        assert candidate.filters == []
        assert candidate.entry_rules == []
        assert candidate.exit_rules == []
        assert candidate.position_size_pct == Decimal("0.1")
        assert candidate.max_positions == 3
        assert candidate.min_holding_time == 60
        assert candidate.max_holding_time == 86400
        assert candidate.generation == 0
        assert candidate.parent_ids == []
        assert candidate.mutation_count == 0
        assert isinstance(candidate.created_at, datetime)
        assert isinstance(candidate.updated_at, datetime)
        assert candidate.metadata == {}

    def test_creation_with_custom_values(self):
        """Создание с пользовательскими значениями."""
        candidate = StrategyCandidate(
            name="Test Strategy",
            description="Test Description",
            strategy_type=StrategyType.MEAN_REVERSION,
            status=EvolutionStatus.TESTING,
            position_size_pct=Decimal("0.2"),
            max_positions=5,
            min_holding_time=120,
            max_holding_time=172800,
            generation=2,
            mutation_count=3,
        )
        assert candidate.name == "Test Strategy"
        assert candidate.description == "Test Description"
        assert candidate.strategy_type == StrategyType.MEAN_REVERSION
        assert candidate.status == EvolutionStatus.TESTING
        assert candidate.position_size_pct == Decimal("0.2")
        assert candidate.max_positions == 5
        assert candidate.min_holding_time == 120
        assert candidate.max_holding_time == 172800
        assert candidate.generation == 2
        assert candidate.mutation_count == 3

    def test_post_init_converts_position_size_to_decimal(self):
        """Проверка конвертации position_size_pct в Decimal."""
        candidate = StrategyCandidate(position_size_pct=0.15)
        assert candidate.position_size_pct == Decimal("0.15")

    def test_add_indicator(self):
        """Добавление индикатора."""
        candidate = StrategyCandidate()
        indicator = IndicatorConfig(name="RSI")
        original_updated_at = candidate.updated_at
        candidate.add_indicator(indicator)
        assert len(candidate.indicators) == 1
        assert candidate.indicators[0] == indicator
        assert candidate.updated_at > original_updated_at

    def test_add_filter(self):
        """Добавление фильтра."""
        candidate = StrategyCandidate()
        filter_config = FilterConfig(name="Volume Filter")
        original_updated_at = candidate.updated_at
        candidate.add_filter(filter_config)
        assert len(candidate.filters) == 1
        assert candidate.filters[0] == filter_config
        assert candidate.updated_at > original_updated_at

    def test_add_entry_rule(self):
        """Добавление правила входа."""
        candidate = StrategyCandidate()
        rule = EntryRule()
        original_updated_at = candidate.updated_at
        candidate.add_entry_rule(rule)
        assert len(candidate.entry_rules) == 1
        assert candidate.entry_rules[0] == rule
        assert candidate.updated_at > original_updated_at

    def test_add_exit_rule(self):
        """Добавление правила выхода."""
        candidate = StrategyCandidate()
        rule = ExitRule()
        original_updated_at = candidate.updated_at
        candidate.add_exit_rule(rule)
        assert len(candidate.exit_rules) == 1
        assert candidate.exit_rules[0] == rule
        assert candidate.updated_at > original_updated_at

    def test_get_active_indicators(self):
        """Получение активных индикаторов."""
        candidate = StrategyCandidate()
        active_indicator = IndicatorConfig(name="RSI", is_active=True)
        inactive_indicator = IndicatorConfig(name="MACD", is_active=False)
        candidate.add_indicator(active_indicator)
        candidate.add_indicator(inactive_indicator)
        active_indicators = candidate.get_active_indicators()
        assert len(active_indicators) == 1
        assert active_indicators[0] == active_indicator

    def test_get_active_filters(self):
        """Получение активных фильтров."""
        candidate = StrategyCandidate()
        active_filter = FilterConfig(name="Volume Filter", is_active=True)
        inactive_filter = FilterConfig(name="Time Filter", is_active=False)
        candidate.add_filter(active_filter)
        candidate.add_filter(inactive_filter)
        active_filters = candidate.get_active_filters()
        assert len(active_filters) == 1
        assert active_filters[0] == active_filter

    def test_get_active_entry_rules(self):
        """Получение активных правил входа."""
        candidate = StrategyCandidate()
        active_rule = EntryRule(is_active=True)
        inactive_rule = EntryRule(is_active=False)
        candidate.add_entry_rule(active_rule)
        candidate.add_entry_rule(inactive_rule)
        active_rules = candidate.get_active_entry_rules()
        assert len(active_rules) == 1
        assert active_rules[0] == active_rule

    def test_get_active_exit_rules(self):
        """Получение активных правил выхода."""
        candidate = StrategyCandidate()
        active_rule = ExitRule(is_active=True)
        inactive_rule = ExitRule(is_active=False)
        candidate.add_exit_rule(active_rule)
        candidate.add_exit_rule(inactive_rule)
        active_rules = candidate.get_active_exit_rules()
        assert len(active_rules) == 1
        assert active_rules[0] == active_rule

    def test_update_status(self):
        """Обновление статуса."""
        candidate = StrategyCandidate()
        original_updated_at = candidate.updated_at
        candidate.update_status(EvolutionStatus.APPROVED)
        assert candidate.status == EvolutionStatus.APPROVED
        assert candidate.updated_at > original_updated_at

    def test_increment_generation(self):
        """Увеличение поколения."""
        candidate = StrategyCandidate(generation=1)
        original_updated_at = candidate.updated_at
        candidate.increment_generation()
        assert candidate.generation == 2
        assert candidate.updated_at > original_updated_at

    def test_add_parent(self):
        """Добавление родительской стратегии."""
        candidate = StrategyCandidate()
        parent_id = uuid4()
        original_updated_at = candidate.updated_at
        candidate.add_parent(parent_id)
        assert parent_id in candidate.parent_ids
        assert candidate.updated_at > original_updated_at

    def test_add_parent_duplicate(self):
        """Добавление дублирующейся родительской стратегии."""
        candidate = StrategyCandidate()
        parent_id = uuid4()
        candidate.add_parent(parent_id)
        candidate.add_parent(parent_id)  # Дубликат
        assert candidate.parent_ids.count(parent_id) == 1

    def test_increment_mutation_count(self):
        """Увеличение счетчика мутаций."""
        candidate = StrategyCandidate(mutation_count=2)
        original_updated_at = candidate.updated_at
        candidate.increment_mutation_count()
        assert candidate.mutation_count == 3
        assert candidate.updated_at > original_updated_at

    def test_validate_configuration_valid(self):
        """Валидация корректной конфигурации."""
        candidate = StrategyCandidate(name="Test Strategy")
        errors = candidate.validate_configuration()
        assert len(errors) == 0

    def test_validate_configuration_missing_name(self):
        """Валидация с отсутствующим именем."""
        candidate = StrategyCandidate(name="")
        errors = candidate.validate_configuration()
        assert "Strategy name is required" in errors

    def test_validate_configuration_invalid_position_size(self):
        """Валидация с невалидным размером позиции."""
        candidate = StrategyCandidate(position_size_pct=Decimal("1.5"))
        errors = candidate.validate_configuration()
        assert "Position size percentage must be between 0 and 1" in errors

    def test_validate_configuration_invalid_max_positions(self):
        """Валидация с невалидным максимальным количеством позиций."""
        candidate = StrategyCandidate(max_positions=0)
        errors = candidate.validate_configuration()
        assert "Max positions must be positive" in errors

    def test_validate_configuration_invalid_holding_time(self):
        """Валидация с невалидным временем удержания."""
        candidate = StrategyCandidate(min_holding_time=-1)
        errors = candidate.validate_configuration()
        assert "Min holding time cannot be negative" in errors

    def test_validate_configuration_invalid_max_holding_time(self):
        """Валидация с невалидным максимальным временем удержания."""
        candidate = StrategyCandidate(min_holding_time=100, max_holding_time=50)
        errors = candidate.validate_configuration()
        assert "Max holding time must be greater than min holding time" in errors

    def test_validate_configuration_with_invalid_indicators(self):
        """Валидация с невалидными индикаторами."""
        candidate = StrategyCandidate(name="Test Strategy")
        invalid_indicator = IndicatorConfig(name="")  # Невалидный
        candidate.add_indicator(invalid_indicator)
        errors = candidate.validate_configuration()
        assert "Indicator name is required" in errors

    def test_validate_configuration_with_invalid_filters(self):
        """Валидация с невалидными фильтрами."""
        candidate = StrategyCandidate(name="Test Strategy")
        invalid_filter = FilterConfig(name="")  # Невалидный
        candidate.add_filter(invalid_filter)
        errors = candidate.validate_configuration()
        assert "Filter name is required" in errors

    def test_validate_configuration_with_invalid_entry_rules(self):
        """Валидация с невалидными правилами входа."""
        candidate = StrategyCandidate(name="Test Strategy")
        invalid_rule = EntryRule(conditions=[])  # Невалидный
        candidate.add_entry_rule(invalid_rule)
        errors = candidate.validate_configuration()
        assert "At least one condition is required" in errors

    def test_validate_configuration_with_invalid_exit_rules(self):
        """Валидация с невалидными правилами выхода."""
        candidate = StrategyCandidate(name="Test Strategy")
        invalid_rule = ExitRule(stop_loss_pct=Decimal("-0.01"))  # Невалидный
        candidate.add_exit_rule(invalid_rule)
        errors = candidate.validate_configuration()
        assert "Stop loss percentage cannot be negative" in errors

    def test_get_complexity_score(self):
        """Получение оценки сложности."""
        candidate = StrategyCandidate()
        # Добавляем компоненты для увеличения сложности
        candidate.add_indicator(IndicatorConfig(name="RSI", parameters={"period": 14}))
        candidate.add_filter(FilterConfig(name="Volume", parameters={"min_volume": 1000}))
        candidate.add_entry_rule(EntryRule())
        candidate.add_exit_rule(ExitRule())
        
        complexity = candidate.get_complexity_score()
        assert complexity > 0
        assert isinstance(complexity, float)

    def test_clone(self):
        """Клонирование стратегии."""
        original = StrategyCandidate(
            name="Test Strategy",
            description="Test Description",
            strategy_type=StrategyType.MEAN_REVERSION,
            status=EvolutionStatus.TESTING,
            position_size_pct=Decimal("0.2"),
            max_positions=5,
            generation=2,
            mutation_count=3,
        )
        # Добавляем компоненты
        original.add_indicator(IndicatorConfig(name="RSI"))
        original.add_filter(FilterConfig(name="Volume"))
        original.add_entry_rule(EntryRule())
        original.add_exit_rule(ExitRule())
        original.add_parent(uuid4())

        cloned = original.clone()
        assert cloned.id != original.id
        assert cloned.name == original.name
        assert cloned.description == original.description
        assert cloned.strategy_type == original.strategy_type
        assert cloned.status == original.status
        assert cloned.position_size_pct == original.position_size_pct
        assert cloned.max_positions == original.max_positions
        assert cloned.generation == original.generation
        assert cloned.parent_ids == original.parent_ids
        assert cloned.mutation_count == original.mutation_count
        assert len(cloned.indicators) == len(original.indicators)
        assert len(cloned.filters) == len(original.filters)
        assert len(cloned.entry_rules) == len(original.entry_rules)
        assert len(cloned.exit_rules) == len(original.exit_rules)
        assert cloned.updated_at > original.updated_at

    def test_to_dict(self):
        """Преобразование в словарь."""
        candidate = StrategyCandidate(
            name="Test Strategy",
            description="Test Description",
            strategy_type=StrategyType.MEAN_REVERSION,
            status=EvolutionStatus.TESTING,
            position_size_pct=Decimal("0.2"),
            max_positions=5,
            generation=2,
            mutation_count=3,
        )
        # Добавляем компоненты
        candidate.add_indicator(IndicatorConfig(name="RSI"))
        candidate.add_filter(FilterConfig(name="Volume"))
        candidate.add_entry_rule(EntryRule())
        candidate.add_exit_rule(ExitRule())
        candidate.add_parent(uuid4())

        data = candidate.to_dict()
        assert data["name"] == "Test Strategy"
        assert data["description"] == "Test Description"
        assert data["strategy_type"] == "mean_reversion"
        assert data["status"] == "testing"
        assert data["position_size_pct"] == "0.2"
        assert data["max_positions"] == 5
        assert data["generation"] == 2
        assert data["mutation_count"] == 3
        assert "id" in data
        assert "indicators" in data
        assert "filters" in data
        assert "entry_rules" in data
        assert "exit_rules" in data
        assert "parent_ids" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "metadata" in data

    def test_from_dict(self):
        """Создание из словаря."""
        data = {
            "id": str(uuid4()),
            "name": "Test Strategy",
            "description": "Test Description",
            "strategy_type": "mean_reversion",
            "status": "testing",
            "indicators": [],
            "filters": [],
            "entry_rules": [],
            "exit_rules": [],
            "position_size_pct": "0.2",
            "max_positions": 5,
            "min_holding_time": 60,
            "max_holding_time": 86400,
            "generation": 2,
            "parent_ids": [],
            "mutation_count": 3,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
        }
        candidate = StrategyCandidate.from_dict(data)
        assert candidate.name == "Test Strategy"
        assert candidate.description == "Test Description"
        assert candidate.strategy_type == StrategyType.MEAN_REVERSION
        assert candidate.status == EvolutionStatus.TESTING
        assert candidate.position_size_pct == Decimal("0.2")
        assert candidate.max_positions == 5
        assert candidate.generation == 2
        assert candidate.mutation_count == 3


class TestEvolutionContext:
    """Тесты для EvolutionContext."""

    def test_creation_with_defaults(self):
        """Создание с значениями по умолчанию."""
        context = EvolutionContext()
        assert isinstance(context.id, UUID)
        assert context.name == ""
        assert context.description == ""
        assert context.population_size == 50
        assert context.generations == 100
        assert context.mutation_rate == Decimal("0.1")
        assert context.crossover_rate == Decimal("0.8")
        assert context.elite_size == 5
        assert context.min_accuracy == Decimal("0.82")
        assert context.min_profitability == Decimal("0.05")
        assert context.max_drawdown == Decimal("0.15")
        assert context.min_sharpe == Decimal("1.0")
        assert context.max_indicators == 10
        assert context.max_filters == 5
        assert context.max_entry_rules == 3
        assert context.max_exit_rules == 3
        assert isinstance(context.created_at, datetime)
        assert isinstance(context.updated_at, datetime)
        assert context.metadata == {}

    def test_creation_with_custom_values(self):
        """Создание с пользовательскими значениями."""
        context = EvolutionContext(
            name="Test Evolution",
            description="Test Description",
            population_size=100,
            generations=200,
            mutation_rate=Decimal("0.15"),
            crossover_rate=Decimal("0.9"),
            elite_size=10,
            min_accuracy=Decimal("0.85"),
            min_profitability=Decimal("0.08"),
            max_drawdown=Decimal("0.12"),
            min_sharpe=Decimal("1.5"),
            max_indicators=15,
            max_filters=8,
            max_entry_rules=5,
            max_exit_rules=5,
        )
        assert context.name == "Test Evolution"
        assert context.description == "Test Description"
        assert context.population_size == 100
        assert context.generations == 200
        assert context.mutation_rate == Decimal("0.15")
        assert context.crossover_rate == Decimal("0.9")
        assert context.elite_size == 10
        assert context.min_accuracy == Decimal("0.85")
        assert context.min_profitability == Decimal("0.08")
        assert context.max_drawdown == Decimal("0.12")
        assert context.min_sharpe == Decimal("1.5")
        assert context.max_indicators == 15
        assert context.max_filters == 8
        assert context.max_entry_rules == 5
        assert context.max_exit_rules == 5

    def test_post_init_converts_decimals(self):
        """Проверка конвертации в Decimal."""
        context = EvolutionContext(
            mutation_rate=0.12,
            crossover_rate=0.85,
            min_accuracy=0.83,
            min_profitability=0.06,
            max_drawdown=0.13,
            min_sharpe=1.2,
        )
        assert context.mutation_rate == Decimal("0.12")
        assert context.crossover_rate == Decimal("0.85")
        assert context.min_accuracy == Decimal("0.83")
        assert context.min_profitability == Decimal("0.06")
        assert context.max_drawdown == Decimal("0.13")
        assert context.min_sharpe == Decimal("1.2")

    def test_validate_configuration_valid(self):
        """Валидация корректной конфигурации."""
        context = EvolutionContext(name="Test Evolution")
        errors = context.validate_configuration()
        assert len(errors) == 0

    def test_validate_configuration_missing_name(self):
        """Валидация с отсутствующим именем."""
        context = EvolutionContext(name="")
        errors = context.validate_configuration()
        assert "Evolution context name is required" in errors

    def test_validate_configuration_invalid_population_size(self):
        """Валидация с невалидным размером популяции."""
        context = EvolutionContext(name="Test", population_size=0)
        errors = context.validate_configuration()
        assert "Population size must be positive" in errors

    def test_validate_configuration_invalid_generations(self):
        """Валидация с невалидным количеством поколений."""
        context = EvolutionContext(name="Test", generations=0)
        errors = context.validate_configuration()
        assert "Number of generations must be positive" in errors

    def test_validate_configuration_invalid_mutation_rate(self):
        """Валидация с невалидной частотой мутаций."""
        context = EvolutionContext(name="Test", mutation_rate=Decimal("1.5"))
        errors = context.validate_configuration()
        assert "Mutation rate must be between 0 and 1" in errors

    def test_validate_configuration_invalid_crossover_rate(self):
        """Валидация с невалидной частотой скрещивания."""
        context = EvolutionContext(name="Test", crossover_rate=Decimal("-0.1"))
        errors = context.validate_configuration()
        assert "Crossover rate must be between 0 and 1" in errors

    def test_validate_configuration_invalid_elite_size(self):
        """Валидация с невалидным размером элиты."""
        context = EvolutionContext(name="Test", population_size=10, elite_size=15)
        errors = context.validate_configuration()
        assert "Elite size must be between 0 and population size" in errors

    def test_validate_configuration_invalid_min_accuracy(self):
        """Валидация с невалидной минимальной точностью."""
        context = EvolutionContext(name="Test", min_accuracy=Decimal("1.5"))
        errors = context.validate_configuration()
        assert "Min accuracy must be between 0 and 1" in errors

    def test_validate_configuration_invalid_min_profitability(self):
        """Валидация с невалидной минимальной прибыльностью."""
        context = EvolutionContext(name="Test", min_profitability=Decimal("-0.01"))
        errors = context.validate_configuration()
        assert "Min profitability cannot be negative" in errors

    def test_validate_configuration_invalid_max_drawdown(self):
        """Валидация с невалидным максимальным просадкой."""
        context = EvolutionContext(name="Test", max_drawdown=Decimal("1.5"))
        errors = context.validate_configuration()
        assert "Max drawdown must be between 0 and 1" in errors

    def test_validate_configuration_invalid_min_sharpe(self):
        """Валидация с невалидным минимальным коэффициентом Шарпа."""
        context = EvolutionContext(name="Test", min_sharpe=Decimal("-0.1"))
        errors = context.validate_configuration()
        assert "Min Sharpe ratio cannot be negative" in errors

    def test_to_dict(self):
        """Преобразование в словарь."""
        context = EvolutionContext(
            name="Test Evolution",
            description="Test Description",
            population_size=100,
            generations=200,
            mutation_rate=Decimal("0.15"),
            crossover_rate=Decimal("0.9"),
            elite_size=10,
            min_accuracy=Decimal("0.85"),
            min_profitability=Decimal("0.08"),
            max_drawdown=Decimal("0.12"),
            min_sharpe=Decimal("1.5"),
            max_indicators=15,
            max_filters=8,
            max_entry_rules=5,
            max_exit_rules=5,
        )
        data = context.to_dict()
        assert data["name"] == "Test Evolution"
        assert data["description"] == "Test Description"
        assert data["population_size"] == 100
        assert data["generations"] == 200
        assert data["mutation_rate"] == "0.15"
        assert data["crossover_rate"] == "0.9"
        assert data["elite_size"] == 10
        assert data["min_accuracy"] == "0.85"
        assert data["min_profitability"] == "0.08"
        assert data["max_drawdown"] == "0.12"
        assert data["min_sharpe"] == "1.5"
        assert data["max_indicators"] == 15
        assert data["max_filters"] == 8
        assert data["max_entry_rules"] == 5
        assert data["max_exit_rules"] == 5
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data
        assert "metadata" in data

    def test_from_dict(self):
        """Создание из словаря."""
        data = {
            "id": str(uuid4()),
            "name": "Test Evolution",
            "description": "Test Description",
            "population_size": 100,
            "generations": 200,
            "mutation_rate": "0.15",
            "crossover_rate": "0.9",
            "elite_size": 10,
            "min_accuracy": "0.85",
            "min_profitability": "0.08",
            "max_drawdown": "0.12",
            "min_sharpe": "1.5",
            "max_indicators": 15,
            "max_filters": 8,
            "max_entry_rules": 5,
            "max_exit_rules": 5,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": {},
        }
        context = EvolutionContext.from_dict(data)
        assert context.name == "Test Evolution"
        assert context.description == "Test Description"
        assert context.population_size == 100
        assert context.generations == 200
        assert context.mutation_rate == Decimal("0.15")
        assert context.crossover_rate == Decimal("0.9")
        assert context.elite_size == 10
        assert context.min_accuracy == Decimal("0.85")
        assert context.min_profitability == Decimal("0.08")
        assert context.max_drawdown == Decimal("0.12")
        assert context.min_sharpe == Decimal("1.5")
        assert context.max_indicators == 15
        assert context.max_filters == 8
        assert context.max_entry_rules == 5
        assert context.max_exit_rules == 5 