import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from uuid import uuid4
from datetime import datetime, timedelta
from domain.strategies import get_strategy_registry
from domain.strategies.exceptions import StrategyNotFoundError, StrategyDuplicateError
from domain.entities.strategy import StrategyType
from domain.type_definitions import StrategyId, TradingPair, ConfidenceLevel, RiskLevel

class MockStrategy:
    def __init__(self, strategy_id, name, strategy_type, trading_pairs, parameters, risk_level, confidence_threshold) -> Any:
        self._strategy_id = strategy_id
        self._name = name
        self._strategy_type = strategy_type
        self._trading_pairs = trading_pairs
        self._parameters = parameters
        self._risk_level = risk_level
        self._confidence_threshold = confidence_threshold
        self._status = "active"
        self._created_at = datetime.now()
        self._execution_count = 0
        self._success_count = 0
    
    def get_strategy_id(self) -> Any:
        return self._strategy_id
    
    def get_name(self) -> Any:
        return self._name
    
    def get_strategy_type(self) -> Any:
        return self._strategy_type
    
    def get_trading_pairs(self) -> Any:
        return self._trading_pairs
    
    def get_parameters(self) -> Any:
        return self._parameters
    
    def get_risk_level(self) -> Any:
        return self._risk_level
    
    def get_confidence_threshold(self) -> Any:
        return self._confidence_threshold
    
    def get_status(self) -> Any:
        return self._status
    
    def activate(self) -> Any:
        self._status = "active"
    
    def deactivate(self) -> Any:
        self._status = "inactive"

    def test_registry_register_strategy() -> None:
    registry = get_strategy_registry()
    strategy_id = StrategyId(uuid4())
    
    strategy = MockStrategy(
        strategy_id=strategy_id,
        name="Test Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
    )
    
    registered_id = registry.register_strategy(
        strategy=strategy,
        name="Test Strategy",
        tags=["trend", "test"],
        priority=1
    )
    
    assert registered_id == strategy_id
    assert registry.get_strategy(strategy_id) is not None

    def test_registry_get_strategy_not_found() -> None:
    registry = get_strategy_registry()
    strategy_id = StrategyId(uuid4())
    
    with pytest.raises(StrategyNotFoundError):
        registry.get_strategy(strategy_id)

    def test_registry_duplicate_registration() -> None:
    registry = get_strategy_registry()
    strategy_id = StrategyId(uuid4())
    
    strategy = MockStrategy(
        strategy_id=strategy_id,
        name="Test Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
    )
    
    # Первая регистрация
    registry.register_strategy(strategy=strategy, name="Test Strategy")
    
    # Попытка повторной регистрации
    with pytest.raises(StrategyDuplicateError):
        registry.register_strategy(strategy=strategy, name="Test Strategy")

    def test_registry_search_strategies() -> None:
    registry = get_strategy_registry()
    
    # Создаем несколько стратегий
    strategies = []
    for i in range(3):
        strategy = MockStrategy(
            strategy_id=StrategyId(uuid4()),
            name=f"Strategy {i}",
            strategy_type=StrategyType.TREND_FOLLOWING,
            trading_pairs=[TradingPair(f"PAIR{i}/USDT")],
            parameters={"param": i},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        registry.register_strategy(
            strategy=strategy,
            name=f"Strategy {i}",
            tags=["trend", f"test_{i}"],
            priority=i
        )
        strategies.append(strategy)
    
    # Поиск по тегу
    trend_strategies = registry.search_strategies(tags=["trend"])
    assert len(trend_strategies) == 3
    
    # Поиск по типу
    trend_type_strategies = registry.search_strategies(strategy_type=StrategyType.TREND_FOLLOWING)
    assert len(trend_type_strategies) == 3
    
    # Поиск по имени
    strategy_0 = registry.search_strategies(name="Strategy 0")
    assert len(strategy_0) == 1
    assert strategy_0[0].get_name() == "Strategy 0"

    def test_registry_update_metrics() -> None:
    registry = get_strategy_registry()
    strategy_id = StrategyId(uuid4())
    
    strategy = MockStrategy(
        strategy_id=strategy_id,
        name="Test Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
    )
    
    registry.register_strategy(strategy=strategy, name="Test Strategy")
    
    # Обновляем метрики
    registry.update_strategy_metrics(
        strategy_id=strategy_id,
        execution_count=10,
        success_count=7
    )
    
    metrics = registry.get_strategy_metrics(strategy_id)
    assert metrics is not None
    assert metrics.execution_count == 10
    assert metrics.success_count == 7
    assert metrics.success_rate == 0.7

    def test_registry_get_statistics() -> None:
    registry = get_strategy_registry()
    
    # Создаем стратегии разных типов
    strategy_types = [
        StrategyType.TREND_FOLLOWING,
        StrategyType.MEAN_REVERSION,
        StrategyType.BREAKOUT
    ]
    
    for i, strategy_type in enumerate(strategy_types):
        strategy = MockStrategy(
            strategy_id=StrategyId(uuid4()),
            name=f"Strategy {i}",
            strategy_type=strategy_type,
            trading_pairs=[TradingPair(f"PAIR{i}/USDT")],
            parameters={"param": i},
            risk_level=RiskLevel(Decimal("0.5")),
            confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        registry.register_strategy(strategy=strategy, name=f"Strategy {i}")
    
    stats = registry.get_registry_statistics()
    assert stats.total_strategies == 3
    assert stats.active_strategies == 3
    assert len(stats.strategy_type_distribution) == 3
    assert stats.strategy_type_distribution[StrategyType.TREND_FOLLOWING] == 1
    assert stats.strategy_type_distribution[StrategyType.MEAN_REVERSION] == 1
    assert stats.strategy_type_distribution[StrategyType.BREAKOUT] == 1

    def test_registry_remove_strategy() -> None:
    registry = get_strategy_registry()
    strategy_id = StrategyId(uuid4())
    
    strategy = MockStrategy(
        strategy_id=strategy_id,
        name="Test Strategy",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=[TradingPair("BTC/USDT")],
        parameters={"sma_period": 20},
        risk_level=RiskLevel(Decimal("0.5")),
        confidence_threshold=ConfidenceLevel(Decimal("0.7"))
    )
    
    registry.register_strategy(strategy=strategy, name="Test Strategy")
    
    # Удаляем стратегию
    registry.remove_strategy(strategy_id)
    
    # Проверяем, что стратегия удалена
    with pytest.raises(StrategyNotFoundError):
        registry.get_strategy(strategy_id)
    
    stats = registry.get_registry_statistics()
    assert stats.total_strategies == 0 
