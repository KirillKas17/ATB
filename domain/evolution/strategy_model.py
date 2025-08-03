"""
Доменная модель для эволюции стратегий.
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast
from dataclasses import dataclass, field
from decimal import Decimal
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from domain.types import (
    EntryCondition,
    ExitCondition,
    IndicatorParameters,
    FilterParameters,
)
from domain.types.technical_types import SignalType
from domain.types.strategy_types import StrategyType

logger = logging.getLogger(__name__)


class EvolutionStatus(Enum):
    """Статусы эволюции стратегии."""

    GENERATED = "generated"
    TESTING = "testing"
    EVALUATED = "evaluated"
    APPROVED = "approved"
    REJECTED = "rejected"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class IndicatorType(Enum):
    """Типы индикаторов."""

    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    OSCILLATOR = "oscillator"
    CUSTOM = "custom"


class FilterType(Enum):
    """Типы фильтров."""

    VOLATILITY = "volatility"
    VOLUME = "volume"
    TREND = "trend"
    TIME = "time"
    CORRELATION = "correlation"
    MARKET_REGIME = "market_regime"
    CUSTOM = "custom"


@dataclass
class IndicatorConfig:
    """Конфигурация индикатора."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    indicator_type: IndicatorType = IndicatorType.TREND
    parameters: Dict[str, Any] = field(default_factory=dict)
    weight: Decimal = Decimal("1.0")
    is_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.weight, Decimal):
            self.weight = Decimal(str(self.weight))  # type: ignore[unreachable]

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Получить параметр индикатора."""
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any) -> None:
        """Установить параметр индикатора."""
        self.parameters[key] = value

    def validate_parameters(self) -> List[str]:
        """Валидировать параметры индикатора."""
        errors = []

        if not self.name:
            errors.append("Indicator name is required")

        if self.weight < Decimal("0"):
            errors.append("Indicator weight cannot be negative")

        if self.weight > Decimal("10"):
            errors.append("Indicator weight cannot exceed 10")

        # Валидация специфичных параметров по типу индикатора
        if self.indicator_type == IndicatorType.TREND:
            if "period" in self.parameters and self.parameters["period"] <= 0:
                errors.append("Trend indicator period must be positive")

        elif self.indicator_type == IndicatorType.MOMENTUM:
            if "period" in self.parameters and self.parameters["period"] <= 0:
                errors.append("Momentum indicator period must be positive")

        elif self.indicator_type == IndicatorType.VOLATILITY:
            if "period" in self.parameters and self.parameters["period"] <= 0:
                errors.append("Volatility indicator period must be positive")
            if "std_dev" in self.parameters and self.parameters["std_dev"] <= 0:
                errors.append("Standard deviation must be positive")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "indicator_type": self.indicator_type.value,
            "parameters": self.parameters,
            "weight": str(self.weight),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndicatorConfig":
        """Создать из словаря."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            indicator_type=IndicatorType(data["indicator_type"]),
            parameters=data["parameters"],
            weight=Decimal(data["weight"]),
            is_active=data["is_active"],
        )

    def clone(self) -> "IndicatorConfig":
        """Создать копию индикатора."""
        return IndicatorConfig(
            id=uuid4(),
            name=self.name,
            indicator_type=self.indicator_type,
            parameters=self.parameters.copy(),
            weight=self.weight,
            is_active=self.is_active,
        )


@dataclass
class FilterConfig:
    """Конфигурация фильтра."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    filter_type: FilterType = FilterType.VOLATILITY
    parameters: Dict[str, Any] = field(default_factory=dict)
    threshold: Decimal = Decimal("0.5")
    is_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.threshold, Decimal):
            self.threshold = Decimal(str(self.threshold))  # type: ignore[unreachable]

    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Получить параметр фильтра."""
        return self.parameters.get(key, default)

    def set_parameter(self, key: str, value: Any) -> None:
        """Установить параметр фильтра."""
        self.parameters[key] = value

    def validate_parameters(self) -> List[str]:
        """Валидировать параметры фильтра."""
        errors = []

        if not self.name:
            errors.append("Filter name is required")

        if self.threshold < Decimal("0"):
            errors.append("Filter threshold cannot be negative")

        if self.threshold > Decimal("1"):
            errors.append("Filter threshold cannot exceed 1")

        # Валидация специфичных параметров по типу фильтра
        if self.filter_type == FilterType.VOLATILITY:
            if "min_atr" in self.parameters and "max_atr" in self.parameters:
                if self.parameters["min_atr"] >= self.parameters["max_atr"]:
                    errors.append("Min ATR must be less than max ATR")

        elif self.filter_type == FilterType.VOLUME:
            if "min_volume" in self.parameters and self.parameters["min_volume"] <= 0:
                errors.append("Min volume must be positive")
            if (
                "spike_threshold" in self.parameters
                and self.parameters["spike_threshold"] <= 0
            ):
                errors.append("Spike threshold must be positive")

        elif self.filter_type == FilterType.TIME:
            if "start_hour" in self.parameters and "end_hour" in self.parameters:
                if not (
                    0 <= self.parameters["start_hour"] <= 23
                    and 0 <= self.parameters["end_hour"] <= 23
                ):
                    errors.append("Hours must be between 0 and 23")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "filter_type": self.filter_type.value,
            "parameters": self.parameters,
            "threshold": str(self.threshold),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        """Создать из словаря."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            filter_type=FilterType(data["filter_type"]),
            parameters=data["parameters"],
            threshold=Decimal(data["threshold"]),
            is_active=data["is_active"],
        )

    def clone(self) -> "FilterConfig":
        """Создать копию фильтра."""
        return FilterConfig(
            id=uuid4(),
            name=self.name,
            filter_type=self.filter_type,
            parameters=self.parameters.copy(),
            threshold=self.threshold,
            is_active=self.is_active,
        )


@dataclass
class EntryRule:
    """Правило входа в позицию."""

    id: UUID = field(default_factory=uuid4)
    conditions: List[EntryCondition] = field(default_factory=list)
    signal_type: SignalType = SignalType.BUY
    confidence_threshold: Decimal = Decimal("0.7")
    volume_ratio: Decimal = Decimal("1.0")
    is_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.confidence_threshold, Decimal):
            self.confidence_threshold = Decimal(str(self.confidence_threshold))  # type: ignore[unreachable]
            
        if not isinstance(self.volume_ratio, Decimal):
            self.volume_ratio = Decimal(str(self.volume_ratio))  # type: ignore[unreachable]

    def add_condition(self, condition: EntryCondition) -> None:
        """Добавить условие."""
        self.conditions.append(condition)

    def remove_condition(self, condition_index: int) -> None:
        """Удалить условие по индексу."""
        if 0 <= condition_index < len(self.conditions):
            self.conditions.pop(condition_index)

    def validate_conditions(self) -> List[str]:
        """Валидировать условия."""
        errors = []

        if not self.conditions:
            errors.append("At least one condition is required")

        if self.confidence_threshold < Decimal(
            "0"
        ) or self.confidence_threshold > Decimal("1"):
            errors.append("Confidence threshold must be between 0 and 1")

        if self.volume_ratio < Decimal("0"):
            errors.append("Volume ratio cannot be negative")

        # Валидация условий
        for i, condition in enumerate(self.conditions):
            if not condition.get("indicator"):
                errors.append(f"Condition {i}: indicator is required")
            if not condition.get("condition"):
                errors.append(f"Condition {i}: condition type is required")

        return errors

    def validate_parameters(self) -> List[str]:
        """Валидировать параметры (алиас для validate_conditions)."""
        return self.validate_conditions()

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "conditions": self.conditions,
            "signal_type": self.signal_type.value,
            "confidence_threshold": str(self.confidence_threshold),
            "volume_ratio": str(self.volume_ratio),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntryRule":
        """Создать из словаря."""
        return cls(
            id=UUID(data["id"]),
            conditions=data["conditions"],
            signal_type=SignalType(data["signal_type"]),
            confidence_threshold=Decimal(data["confidence_threshold"]),
            volume_ratio=Decimal(data["volume_ratio"]),
            is_active=data["is_active"],
        )

    def clone(self) -> "EntryRule":
        """Создать копию правила входа."""
        return EntryRule(
            id=uuid4(),
            conditions=self.conditions.copy(),
            signal_type=self.signal_type,
            confidence_threshold=self.confidence_threshold,
            volume_ratio=self.volume_ratio,
            is_active=self.is_active,
        )


@dataclass
class ExitRule:
    """Правило выхода из позиции."""

    id: UUID = field(default_factory=uuid4)
    conditions: List[ExitCondition] = field(default_factory=list)
    signal_type: SignalType = SignalType.SELL
    stop_loss_pct: Decimal = Decimal("0.02")
    take_profit_pct: Decimal = Decimal("0.04")
    trailing_stop: bool = False
    trailing_distance: Decimal = Decimal("0.01")
    is_active: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))  # type: ignore[unreachable]
            
        if not isinstance(self.take_profit_pct, Decimal):
            self.take_profit_pct = Decimal(str(self.take_profit_pct))  # type: ignore[unreachable]
            
        if not isinstance(self.trailing_distance, Decimal):
            self.trailing_distance = Decimal(str(self.trailing_distance))  # type: ignore[unreachable]

    def add_condition(self, condition: ExitCondition) -> None:
        """Добавить условие."""
        self.conditions.append(condition)

    def remove_condition(self, condition_index: int) -> None:
        """Удалить условие по индексу."""
        if 0 <= condition_index < len(self.conditions):
            self.conditions.pop(condition_index)

    def validate_parameters(self) -> List[str]:
        """Валидировать параметры."""
        errors = []

        if self.stop_loss_pct < Decimal("0"):
            errors.append("Stop loss percentage cannot be negative")

        if self.take_profit_pct < Decimal("0"):
            errors.append("Take profit percentage cannot be negative")

        if self.trailing_stop and self.trailing_distance < Decimal("0"):
            errors.append("Trailing distance cannot be negative")

        # Валидация условий
        for i, condition in enumerate(self.conditions):
            if not condition.get("indicator"):
                errors.append(f"Condition {i}: indicator is required")
            if not condition.get("condition"):
                errors.append(f"Condition {i}: condition type is required")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "conditions": self.conditions,
            "signal_type": self.signal_type.value,
            "stop_loss_pct": str(self.stop_loss_pct),
            "take_profit_pct": str(self.take_profit_pct),
            "trailing_stop": self.trailing_stop,
            "trailing_distance": str(self.trailing_distance),
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExitRule":
        """Создать из словаря."""
        return cls(
            id=UUID(data["id"]),
            conditions=data["conditions"],
            signal_type=SignalType(data["signal_type"]),
            stop_loss_pct=Decimal(data["stop_loss_pct"]),
            take_profit_pct=Decimal(data["take_profit_pct"]),
            trailing_stop=data["trailing_stop"],
            trailing_distance=Decimal(data["trailing_distance"]),
            is_active=data["is_active"],
        )

    def clone(self) -> "ExitRule":
        """Создать копию правила выхода."""
        return ExitRule(
            id=uuid4(),
            conditions=self.conditions.copy(),
            signal_type=self.signal_type,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            trailing_stop=self.trailing_stop,
            trailing_distance=self.trailing_distance,
            is_active=self.is_active,
        )


@dataclass
class StrategyCandidate:
    """Кандидат стратегии для эволюции."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    strategy_type: StrategyType = StrategyType.TREND
    status: EvolutionStatus = EvolutionStatus.GENERATED

    # Конфигурация стратегии
    indicators: List[IndicatorConfig] = field(default_factory=list)
    filters: List[FilterConfig] = field(default_factory=list)
    entry_rules: List[EntryRule] = field(default_factory=list)
    exit_rules: List[ExitRule] = field(default_factory=list)

    # Параметры исполнения
    position_size_pct: Decimal = Decimal("0.1")
    max_positions: int = 3
    min_holding_time: int = 60  # секунды
    max_holding_time: int = 86400  # секунды (24 часа)

    # Метаданные
    generation: int = 0
    parent_ids: List[UUID] = field(default_factory=list)
    mutation_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))  # type: ignore[unreachable]

    def add_indicator(self, indicator: IndicatorConfig) -> None:
        """Добавить индикатор."""
        self.indicators.append(indicator)
        self.updated_at = datetime.now()

    def add_filter(self, filter_config: FilterConfig) -> None:
        """Добавить фильтр."""
        self.filters.append(filter_config)
        self.updated_at = datetime.now()

    def add_entry_rule(self, rule: EntryRule) -> None:
        """Добавить правило входа."""
        self.entry_rules.append(rule)
        self.updated_at = datetime.now()

    def add_exit_rule(self, rule: ExitRule) -> None:
        """Добавить правило выхода."""
        self.exit_rules.append(rule)
        self.updated_at = datetime.now()

    def get_active_indicators(self) -> List[IndicatorConfig]:
        """Получить активные индикаторы."""
        return [ind for ind in self.indicators if ind.is_active]

    def get_active_filters(self) -> List[FilterConfig]:
        """Получить активные фильтры."""
        return [filt for filt in self.filters if filt.is_active]

    def get_active_entry_rules(self) -> List[EntryRule]:
        """Получить активные правила входа."""
        return [rule for rule in self.entry_rules if rule.is_active]

    def get_active_exit_rules(self) -> List[ExitRule]:
        """Получить активные правила выхода."""
        return [rule for rule in self.exit_rules if rule.is_active]

    def update_status(self, status: EvolutionStatus) -> None:
        """Обновить статус."""
        self.status = status
        self.updated_at = datetime.now()

    def increment_generation(self) -> None:
        """Увеличить поколение."""
        self.generation += 1
        self.updated_at = datetime.now()

    def add_parent(self, parent_id: UUID) -> None:
        """Добавить родительскую стратегию."""
        if parent_id not in self.parent_ids:
            self.parent_ids.append(parent_id)
            self.updated_at = datetime.now()

    def increment_mutation_count(self) -> None:
        """Увеличить счетчик мутаций."""
        self.mutation_count += 1
        self.updated_at = datetime.now()

    def validate_configuration(self) -> List[str]:
        """Валидировать конфигурацию стратегии."""
        errors = []

        if not self.name:
            errors.append("Strategy name is required")

        if self.position_size_pct <= Decimal("0") or self.position_size_pct > Decimal(
            "1"
        ):
            errors.append("Position size percentage must be between 0 and 1")

        if self.max_positions <= 0:
            errors.append("Max positions must be positive")

        if self.min_holding_time < 0:
            errors.append("Min holding time cannot be negative")

        if self.max_holding_time <= self.min_holding_time:
            errors.append("Max holding time must be greater than min holding time")

        # Валидировать индикаторы
        for indicator in self.indicators:
            errors.extend(indicator.validate_parameters())

        # Валидировать фильтры
        for filter_config in self.filters:
            errors.extend(filter_config.validate_parameters())

        # Валидировать правила входа
        for rule in self.entry_rules:
            errors.extend(rule.validate_conditions())

        # Валидировать правила выхода
        for exit_rule in self.exit_rules:
            errors.extend(exit_rule.validate_parameters())

        return errors

    def get_complexity_score(self) -> float:
        """Получить оценку сложности стратегии."""
        complexity = 0.0

        # Сложность индикаторов
        complexity += len(self.indicators) * 0.1

        # Сложность фильтров
        complexity += len(self.filters) * 0.15

        # Сложность правил
        complexity += len(self.entry_rules) * 0.2
        complexity += len(self.exit_rules) * 0.2

        # Сложность параметров
        for indicator in self.indicators:
            complexity += len(indicator.parameters) * 0.05

        for filter_config in self.filters:
            complexity += len(filter_config.parameters) * 0.05

        return complexity

    def clone(self) -> "StrategyCandidate":
        """Создать копию стратегии."""
        candidate = StrategyCandidate(
            id=uuid4(),
            name=self.name,
            description=self.description,
            strategy_type=self.strategy_type,
            status=self.status,
            position_size_pct=self.position_size_pct,
            max_positions=self.max_positions,
            min_holding_time=self.min_holding_time,
            max_holding_time=self.max_holding_time,
            generation=self.generation,
            parent_ids=self.parent_ids.copy(),
            mutation_count=self.mutation_count,
            created_at=self.created_at,
            updated_at=datetime.now(),
            metadata=self.metadata.copy(),
        )

        # Клонировать индикаторы
        for indicator in self.indicators:
            candidate.indicators.append(indicator.clone())

        # Клонировать фильтры
        for filter_config in self.filters:
            candidate.filters.append(filter_config.clone())

        # Клонировать правила входа
        for entry_rule in self.entry_rules:
            cloned_entry_rule = entry_rule.clone()
            candidate.entry_rules.append(cloned_entry_rule)

        # Клонировать правила выхода
        for exit_rule in self.exit_rules:
            cloned_exit_rule: ExitRule = exit_rule.clone()
            candidate.exit_rules.append(cloned_exit_rule)

        return candidate

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "strategy_type": self.strategy_type.value,
            "status": self.status.value,
            "indicators": [ind.to_dict() for ind in self.indicators],
            "filters": [filt.to_dict() for filt in self.filters],
            "entry_rules": [rule.to_dict() for rule in self.entry_rules],
            "exit_rules": [rule.to_dict() for rule in self.exit_rules],
            "position_size_pct": str(self.position_size_pct),
            "max_positions": self.max_positions,
            "min_holding_time": self.min_holding_time,
            "max_holding_time": self.max_holding_time,
            "generation": self.generation,
            "parent_ids": [str(pid) for pid in self.parent_ids],
            "mutation_count": self.mutation_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyCandidate":
        """Создать из словаря."""
        candidate = cls(
            id=UUID(data["id"]),
            name=data["name"],
            description=data["description"],
            strategy_type=StrategyType(data["strategy_type"]),
            status=EvolutionStatus(data["status"]),
            position_size_pct=Decimal(data["position_size_pct"]),
            max_positions=data["max_positions"],
            min_holding_time=data["min_holding_time"],
            max_holding_time=data["max_holding_time"],
            generation=data["generation"],
            parent_ids=[UUID(pid) for pid in data["parent_ids"]],
            mutation_count=data["mutation_count"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data["metadata"],
        )

        # Восстановить индикаторы
        for ind_data in data["indicators"]:
            indicator = IndicatorConfig.from_dict(ind_data)
            candidate.indicators.append(indicator)

        # Восстановить фильтры
        for filt_data in data["filters"]:
            filter_config = FilterConfig.from_dict(filt_data)
            candidate.filters.append(filter_config)

        # Восстановить правила входа
        for rule_data in data["entry_rules"]:
            entry_rule = EntryRule.from_dict(rule_data)
            candidate.entry_rules.append(entry_rule)

        # Восстановить правила выхода
        for rule_data in data["exit_rules"]:
            exit_rule = ExitRule.from_dict(rule_data)
            candidate.exit_rules.append(exit_rule)

        return candidate


@dataclass
class EvolutionContext:
    """Контекст эволюции стратегий."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""

    # Параметры эволюции
    population_size: int = 50
    generations: int = 100
    mutation_rate: Decimal = Decimal("0.1")
    crossover_rate: Decimal = Decimal("0.8")
    elite_size: int = 5

    # Критерии отбора
    min_accuracy: Decimal = Decimal("0.82")
    min_profitability: Decimal = Decimal("0.05")
    max_drawdown: Decimal = Decimal("0.15")
    min_sharpe: Decimal = Decimal("1.0")

    # Ограничения
    max_indicators: int = 10
    max_filters: int = 5
    max_entry_rules: int = 3
    max_exit_rules: int = 3

    # Метаданные
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.mutation_rate, Decimal):
            self.mutation_rate = Decimal(str(self.mutation_rate))  # type: ignore[unreachable]
            
        if not isinstance(self.crossover_rate, Decimal):
            self.crossover_rate = Decimal(str(self.crossover_rate))  # type: ignore[unreachable]
            
        if not isinstance(self.min_accuracy, Decimal):
            self.min_accuracy = Decimal(str(self.min_accuracy))  # type: ignore[unreachable]
            
        if not isinstance(self.min_profitability, Decimal):
            self.min_profitability = Decimal(str(self.min_profitability))  # type: ignore[unreachable]
            
        if not isinstance(self.max_drawdown, Decimal):
            self.max_drawdown = Decimal(str(self.max_drawdown))  # type: ignore[unreachable]
            
        if not isinstance(self.min_sharpe, Decimal):
            self.min_sharpe = Decimal(str(self.min_sharpe))  # type: ignore[unreachable]

    def validate_configuration(self) -> List[str]:
        """Валидировать конфигурацию эволюции."""
        errors = []

        if not self.name:
            errors.append("Evolution context name is required")

        if self.population_size <= 0:
            errors.append("Population size must be positive")

        if self.generations <= 0:
            errors.append("Number of generations must be positive")

        if self.mutation_rate < Decimal("0") or self.mutation_rate > Decimal("1"):
            errors.append("Mutation rate must be between 0 and 1")

        if self.crossover_rate < Decimal("0") or self.crossover_rate > Decimal("1"):
            errors.append("Crossover rate must be between 0 and 1")

        if self.elite_size < 0 or self.elite_size > self.population_size:
            errors.append("Elite size must be between 0 and population size")

        if self.min_accuracy < Decimal("0") or self.min_accuracy > Decimal("1"):
            errors.append("Min accuracy must be between 0 and 1")

        if self.min_profitability < Decimal("0"):
            errors.append("Min profitability cannot be negative")

        if self.max_drawdown < Decimal("0") or self.max_drawdown > Decimal("1"):
            errors.append("Max drawdown must be between 0 and 1")

        if self.min_sharpe < Decimal("0"):
            errors.append("Min Sharpe ratio cannot be negative")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "population_size": self.population_size,
            "generations": self.generations,
            "mutation_rate": str(self.mutation_rate),
            "crossover_rate": str(self.crossover_rate),
            "elite_size": self.elite_size,
            "min_accuracy": str(self.min_accuracy),
            "min_profitability": str(self.min_profitability),
            "max_drawdown": str(self.max_drawdown),
            "min_sharpe": str(self.min_sharpe),
            "max_indicators": self.max_indicators,
            "max_filters": self.max_filters,
            "max_entry_rules": self.max_entry_rules,
            "max_exit_rules": self.max_exit_rules,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvolutionContext":
        """Создать из словаря."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            description=data["description"],
            population_size=data["population_size"],
            generations=data["generations"],
            mutation_rate=Decimal(data["mutation_rate"]),
            crossover_rate=Decimal(data["crossover_rate"]),
            elite_size=data["elite_size"],
            min_accuracy=Decimal(data["min_accuracy"]),
            min_profitability=Decimal(data["min_profitability"]),
            max_drawdown=Decimal(data["max_drawdown"]),
            min_sharpe=Decimal(data["min_sharpe"]),
            max_indicators=data["max_indicators"],
            max_filters=data["max_filters"],
            max_entry_rules=data["max_entry_rules"],
            max_exit_rules=data["max_exit_rules"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data["metadata"],
        )
