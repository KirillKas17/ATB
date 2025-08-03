# -*- coding: utf-8 -*-
"""
Генератор стратегий для эволюционного алгоритма.
"""

import random
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, cast
from uuid import uuid4

from loguru import logger

from domain.entities.strategy import StrategyType
from domain.types.technical_types import SignalType
from domain.types.strategy_types import StrategyType as StrategyTypeFromTypes
from domain.evolution.strategy_model import (
    EntryCondition,
    EntryRule,
    ExitCondition,
    ExitRule,
    FilterConfig,
    FilterType,
    IndicatorConfig,
    IndicatorType,
    StrategyCandidate,
)
from domain.evolution.strategy_types import EvolutionContext  # type: ignore[import-not-found]
from domain.types.evolution_types import FilterParameters, IndicatorParameters


class StrategyGenerator:
    """Генератор стратегий для эволюционного алгоритма."""

    def __init__(self, context: EvolutionContext):
        self.context = context
        self.generation_count = 0

        # Шаблоны индикаторов
        self.indicator_templates: Dict[IndicatorType, List[Dict[str, Any]]] = {
            IndicatorType.TREND: [
                {"name": "SMA", "params": {"period": [10, 20, 50, 100, 200]}},
                {"name": "EMA", "params": {"period": [12, 26, 50, 100]}},
                {"name": "MACD", "params": {"fast": [12], "slow": [26], "signal": [9]}},
                {"name": "ADX", "params": {"period": [14, 20, 30]}},
                {
                    "name": "ParabolicSAR",
                    "params": {"acceleration": [0.02, 0.05], "maximum": [0.2, 0.3]},
                },
            ],
            IndicatorType.MOMENTUM: [
                {"name": "RSI", "params": {"period": [14, 21, 30]}},
                {
                    "name": "Stochastic",
                    "params": {"k_period": [14], "d_period": [3, 5]},
                },
                {"name": "WilliamsR", "params": {"period": [14, 21]}},
                {"name": "CCI", "params": {"period": [14, 20]}},
                {"name": "ROC", "params": {"period": [10, 14, 20]}},
            ],
            IndicatorType.VOLATILITY: [
                {
                    "name": "BollingerBands",
                    "params": {"period": [20], "std_dev": [2, 2.5]},
                },
                {"name": "ATR", "params": {"period": [14, 20]}},
                {
                    "name": "KeltnerChannel",
                    "params": {"period": [20], "multiplier": [2, 3]},
                },
                {"name": "DonchianChannel", "params": {"period": [20, 30]}},
            ],
            IndicatorType.VOLUME: [
                {"name": "VolumeSMA", "params": {"period": [20, 30]}},
                {"name": "OBV", "params": {}},
                {"name": "VWAP", "params": {"period": [14, 20]}},
                {"name": "MoneyFlowIndex", "params": {"period": [14, 20]}},
            ],
            IndicatorType.OSCILLATOR: [
                {
                    "name": "StochRSI",
                    "params": {"period": [14], "k_period": [3, 5], "d_period": [3, 5]},
                },
                {
                    "name": "UltimateOscillator",
                    "params": {"period1": [7], "period2": [14], "period3": [28]},
                },
                {"name": "AwesomeOscillator", "params": {}},
                {"name": "ChaikinOscillator", "params": {"fast": [3], "slow": [10]}},
            ],
        }
        # Шаблоны фильтров
        self.filter_templates: Dict[FilterType, List[Dict[str, Any]]] = {
            FilterType.VOLATILITY: [
                {
                    "name": "VolatilityFilter",
                    "params": {"min_atr": [0.01, 0.02], "max_atr": [0.05, 0.1]},
                },
                {
                    "name": "BBWidthFilter",
                    "params": {"min_width": [0.02, 0.03], "max_width": [0.08, 0.12]},
                },
            ],
            FilterType.VOLUME: [
                {"name": "VolumeFilter", "params": {"min_volume": [1000000, 5000000]}},
                {
                    "name": "VolumeSpikeFilter",
                    "params": {"spike_threshold": [1.5, 2.0, 3.0]},
                },
            ],
            FilterType.TREND: [
                {"name": "TrendStrengthFilter", "params": {"min_adx": [20, 25, 30]}},
                {"name": "TrendDirectionFilter", "params": {"trend_period": [20, 50]}},
            ],
            FilterType.TIME: [
                {
                    "name": "TimeFilter",
                    "params": {"start_hour": [9, 10], "end_hour": [16, 17]},
                },
                {"name": "DayOfWeekFilter", "params": {"excluded_days": [[5, 6], [6]]}},
            ],
        }
        # Шаблоны правил входа
        self.entry_rule_templates = [
            {
                "conditions": ["trend_up", "volume_high"],
                "signal_type": SignalType.BUY,
            },
            {
                "conditions": ["momentum_positive", "volatility_low"],
                "signal_type": SignalType.BUY,
            },
            {
                "conditions": ["breakout_up", "volume_spike"],
                "signal_type": SignalType.BUY,
            },
        ]
        # Шаблоны правил выхода
        self.exit_rule_templates = [
            {
                "conditions": ["trend_reversal"],
                "stop_loss_pct": [0.02, 0.03, 0.05],
                "take_profit_pct": [0.04, 0.06, 0.08],
                "trailing_stop": True,
                "trailing_distance": [0.01, 0.015, 0.02],
            },
            {
                "conditions": ["time_expired"],
                "stop_loss_pct": [0.01, 0.02],
                "take_profit_pct": [0.03, 0.05],
                "trailing_stop": False,
            },
        ]

    def generate_random_strategy(
        self, name_prefix: str = "Evolved"
    ) -> StrategyCandidate:
        """Генерация случайной стратегии."""
        candidate = StrategyCandidate(
            name=f"{name_prefix}_{self.generation_count}_{random.randint(1000, 9999)}",
            strategy_type=random.choice(list(StrategyTypeFromTypes)),
            generation=self.generation_count,
        )

        # Добавляем случайные индикаторы
        indicator_count = random.randint(1, self.context.max_indicators)
        self._add_random_indicators(candidate, indicator_count)

        # Добавляем случайные фильтры
        filter_count = random.randint(0, self.context.max_filters)
        self._add_random_filters(candidate, filter_count)

        # Добавляем случайные правила входа
        entry_rule_count = random.randint(1, self.context.max_entry_rules)
        self._add_random_entry_rules(candidate, entry_rule_count)

        # Добавляем случайные правила выхода
        exit_rule_count = random.randint(1, self.context.max_exit_rules)
        self._add_random_exit_rules(candidate, exit_rule_count)

        return candidate

    def _add_random_indicators(self, candidate: StrategyCandidate, count: int) -> None:
        """Добавить случайные индикаторы."""
        for _ in range(count):
            indicator_type = random.choice(list(IndicatorType))
            template = random.choice(self.indicator_templates[indicator_type])
            # Преобразуем параметры в правильный тип
            parameters = self._randomize_parameters(template["params"])
            indicator = IndicatorConfig(
                id=uuid4(),
                name=template["name"],
                indicator_type=indicator_type,
                parameters=parameters,
                weight=Decimal(str(random.uniform(0.5, 2.0))),
                is_active=True,
            )
            candidate.add_indicator(indicator)

    def _add_random_filters(self, candidate: StrategyCandidate, count: int) -> None:
        """Добавить случайные фильтры."""
        for _ in range(count):
            filter_type = random.choice(list(FilterType))
            template = random.choice(self.filter_templates[filter_type])
            # Преобразуем параметры в правильный тип
            parameters = self._randomize_parameters(template["params"])
            filter_config = FilterConfig(
                id=uuid4(),
                name=template["name"],
                filter_type=filter_type,
                parameters=parameters,
                threshold=Decimal(str(random.uniform(0.3, 0.8))),
                is_active=True,
            )
            candidate.add_filter(filter_config)

    def _add_random_entry_rules(self, candidate: StrategyCandidate, count: int) -> None:
        """Добавить случайные правила входа."""
        for _ in range(count):
            template = random.choice(self.entry_rule_templates)
            rule = EntryRule(
                conditions=cast(List[EntryCondition], template["conditions"]),
                signal_type=cast(SignalType, template["signal_type"]),
                confidence_threshold=Decimal(str(random.uniform(0.6, 0.9))),
                volume_ratio=Decimal(str(random.uniform(0.8, 1.5))),
            )
            candidate.add_entry_rule(rule)

    def _add_random_exit_rules(self, candidate: StrategyCandidate, count: int) -> None:
        """Добавить случайные правила выхода."""
        for _ in range(count):
            template = random.choice(self.exit_rule_templates)
            rule = ExitRule(
                conditions=cast(List[ExitCondition], template["conditions"]),
                signal_type=SignalType.SELL,
                stop_loss_pct=Decimal(str(random.choice(cast(List[float], template["stop_loss_pct"])))),
                take_profit_pct=Decimal(
                    str(random.choice(cast(List[float], template["take_profit_pct"])))
                ),
                trailing_stop=cast(bool, template.get("trailing_stop", False)),
                trailing_distance=(
                    Decimal(
                        str(random.choice(cast(List[float], template.get("trailing_distance", [0.01]))))
                    )
                    if template.get("trailing_stop")
                    else Decimal("0.01")
                ),
            )
            candidate.add_exit_rule(rule)

    def _randomize_parameters(
        self, param_template: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Случайно выбрать параметры из шаблона."""
        params = {}
        for key, values in param_template.items():
            if values:
                params[key] = random.choice(values)
        return params

    def generate_population(
        self, size: Optional[int] = None
    ) -> List[StrategyCandidate]:
        """Сгенерировать популяцию стратегий."""
        population_size = size or self.context.population_size
        population = []
        for _ in range(population_size):
            strategy = self.generate_random_strategy()
            population.append(strategy)
        return population

    def generate_from_parents(
        self, parents: List[StrategyCandidate], num_children: int
    ) -> List[StrategyCandidate]:
        """Сгенерировать потомков от родителей."""
        children = []
        for _ in range(num_children):
            if len(parents) >= 2:
                # Скрещивание двух родителей
                parent1, parent2 = random.sample(parents, 2)
                child = self._crossover_strategies(parent1, parent2)
            else:
                # Мутация одного родителя
                parent = random.choice(parents)
                child = self._mutate_strategy(parent)
            children.append(child)
        return children

    def _crossover_strategies(
        self, parent1: StrategyCandidate, parent2: StrategyCandidate
    ) -> StrategyCandidate:
        """Скрещивание двух стратегий."""
        child = StrategyCandidate(
            name=f"Child_{self.generation_count}_{random.randint(1000, 9999)}",
            strategy_type=random.choice([parent1.strategy_type, parent2.strategy_type]),
            generation=self.generation_count,
            parent_ids=[parent1.id, parent2.id],
        )

        # Скрещивание индикаторов
        self._crossover_indicators(parent1, parent2, child)

        # Скрещивание фильтров
        self._crossover_filters(parent1, parent2, child)

        # Скрещивание правил входа
        self._crossover_entry_rules(parent1, parent2, child)

        # Скрещивание правил выхода
        self._crossover_exit_rules(parent1, parent2, child)

        return child

    def _crossover_indicators(
        self,
        parent1: StrategyCandidate,
        parent2: StrategyCandidate,
        child: StrategyCandidate,
    ) -> None:
        """Скрещивание индикаторов."""
        # Берем индикаторы от обоих родителей
        all_indicators = parent1.indicators + parent2.indicators
        # Выбираем случайное количество
        num_indicators = random.randint(1, min(len(all_indicators), self.context.max_indicators))
        selected_indicators = random.sample(all_indicators, num_indicators)
        child.indicators = selected_indicators

    def _crossover_filters(
        self,
        parent1: StrategyCandidate,
        parent2: StrategyCandidate,
        child: StrategyCandidate,
    ) -> None:
        """Скрещивание фильтров."""
        # Берем фильтры от обоих родителей
        all_filters = parent1.filters + parent2.filters
        # Выбираем случайное количество
        num_filters = random.randint(0, min(len(all_filters), self.context.max_filters))
        selected_filters = random.sample(all_filters, num_filters)
        child.filters = selected_filters

    def _crossover_entry_rules(
        self,
        parent1: StrategyCandidate,
        parent2: StrategyCandidate,
        child: StrategyCandidate,
    ) -> None:
        """Скрещивание правил входа."""
        # Берем правила от обоих родителей
        all_rules = parent1.entry_rules + parent2.entry_rules
        # Выбираем случайное количество
        num_rules = random.randint(1, min(len(all_rules), self.context.max_entry_rules))
        selected_rules = random.sample(all_rules, num_rules)
        child.entry_rules = selected_rules

    def _crossover_exit_rules(
        self,
        parent1: StrategyCandidate,
        parent2: StrategyCandidate,
        child: StrategyCandidate,
    ) -> None:
        """Скрещивание правил выхода."""
        # Берем правила от обоих родителей
        all_rules = parent1.exit_rules + parent2.exit_rules
        # Выбираем случайное количество
        num_rules = random.randint(1, min(len(all_rules), self.context.max_exit_rules))
        selected_rules = random.sample(all_rules, num_rules)
        child.exit_rules = selected_rules

    def _mutate_strategy(self, parent: StrategyCandidate) -> StrategyCandidate:
        """Мутация стратегии."""
        child = StrategyCandidate(
            name=f"Mutant_{self.generation_count}_{random.randint(1000, 9999)}",
            strategy_type=parent.strategy_type,
            generation=self.generation_count,
            parent_ids=[parent.id],
            mutation_count=parent.mutation_count + 1,
        )

        # Копируем компоненты
        child.indicators = parent.indicators.copy()
        child.filters = parent.filters.copy()
        child.entry_rules = parent.entry_rules.copy()
        child.exit_rules = parent.exit_rules.copy()

        # Мутируем компоненты
        self._mutate_indicators(child)
        self._mutate_filters(child)
        self._mutate_entry_rules(child)
        self._mutate_exit_rules(child)
        self._mutate_execution_parameters(child)

        return child

    def _mutate_indicators(self, candidate: StrategyCandidate) -> None:
        """Мутация индикаторов."""
        for indicator in candidate.indicators:
            if random.random() < self.context.mutation_rate:
                # Мутируем параметры
                if hasattr(indicator, 'parameters') and indicator.parameters:
                    # Преобразуем в dict для мутации
                    params_dict = indicator.parameters
                    mutated_params = self._mutate_parameters(params_dict)
                    # Создаем новый объект параметров
                    indicator.parameters = mutated_params

    def _mutate_filters(self, candidate: StrategyCandidate) -> None:
        """Мутация фильтров."""
        for filter_config in candidate.filters:
            if random.random() < self.context.mutation_rate:
                # Мутируем параметры
                if hasattr(filter_config, 'parameters') and filter_config.parameters:
                    # Преобразуем в dict для мутации
                    params_dict = filter_config.parameters
                    mutated_params = self._mutate_parameters(params_dict)
                    # Создаем новый объект параметров
                    filter_config.parameters = mutated_params

    def _mutate_entry_rules(self, candidate: StrategyCandidate) -> None:
        """Мутировать правила входа."""
        if not candidate.entry_rules:
            return
        mutation_type = random.choice(["add", "remove", "modify"])
        if (
            mutation_type == "add"
            and len(candidate.entry_rules) < self.context.max_entry_rules
        ):
            # Добавить новое правило
            template = random.choice(self.entry_rule_templates)
            rule = EntryRule(
                conditions=cast(List[EntryCondition], template["conditions"]),
                signal_type=cast(SignalType, template["signal_type"]),
                confidence_threshold=Decimal(str(random.uniform(0.6, 0.9))),
                volume_ratio=Decimal(str(random.uniform(0.8, 1.5))),
            )
            candidate.add_entry_rule(rule)
        elif mutation_type == "remove" and len(candidate.entry_rules) > 1:
            # Удалить случайное правило
            candidate.entry_rules.pop(random.randrange(len(candidate.entry_rules)))
        elif mutation_type == "modify":
            # Модифицировать случайное правило
            rule = random.choice(candidate.entry_rules)
            rule.confidence_threshold = Decimal(str(random.uniform(0.6, 0.9)))
            rule.volume_ratio = Decimal(str(random.uniform(0.8, 1.5)))

    def _mutate_exit_rules(self, candidate: StrategyCandidate) -> None:
        """Мутировать правила выхода."""
        if not candidate.exit_rules:
            return
        mutation_type = random.choice(["add", "remove", "modify"])
        if (
            mutation_type == "add"
            and len(candidate.exit_rules) < self.context.max_exit_rules
        ):
            # Добавить новое правило
            template = random.choice(self.exit_rule_templates)
            rule = ExitRule(
                conditions=cast(List[ExitCondition], template["conditions"]),
                signal_type=SignalType.SELL,
                stop_loss_pct=Decimal(str(random.choice(cast(List[float], template["stop_loss_pct"])))),
                take_profit_pct=Decimal(
                    str(random.choice(cast(List[float], template["take_profit_pct"])))
                ),
                trailing_stop=cast(bool, template.get("trailing_stop", False)),
                trailing_distance=(
                    Decimal(
                        str(random.choice(cast(List[float], template.get("trailing_distance", [0.01]))))
                    )
                    if template.get("trailing_stop")
                    else Decimal("0.01")
                ),
            )
            candidate.add_exit_rule(rule)
        elif mutation_type == "remove" and len(candidate.exit_rules) > 1:
            # Удалить случайное правило
            candidate.exit_rules.pop(random.randrange(len(candidate.exit_rules)))
        elif mutation_type == "modify":
            # Модифицировать случайное правило
            rule = random.choice(candidate.exit_rules)
            rule.stop_loss_pct = Decimal(str(random.uniform(0.01, 0.1)))
            rule.take_profit_pct = Decimal(str(random.uniform(0.02, 0.2)))

    def _mutate_execution_parameters(self, candidate: StrategyCandidate) -> None:
        """Мутировать параметры исполнения."""
        candidate.position_size_pct = Decimal(str(random.uniform(0.05, 0.3)))
        candidate.max_positions = random.randint(1, 5)
        candidate.min_holding_time = random.randint(30, 3600)
        candidate.max_holding_time = random.randint(3600, 86400)

    def _mutate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Мутировать параметры."""
        mutated_params = parameters.copy()
        for key, value in mutated_params.items():
            if isinstance(value, (int, float)):
                # Добавить случайное изменение ±20%
                change = random.uniform(0.8, 1.2)
                if isinstance(value, int):
                    mutated_params[key] = int(value * change)
                else:
                    mutated_params[key] = value * change
        return mutated_params

    def get_generation_count(self) -> int:
        """Получить номер поколения."""
        return self.generation_count

    def reset_generation_count(self) -> None:
        """Сбросить счетчик поколений."""
        self.generation_count = 0
