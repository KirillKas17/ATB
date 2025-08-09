import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from domain.strategies import get_strategy_factory
from domain.strategies.exceptions import StrategyCreationError


class DummyStrategy:
    def __init__(self, name, **kwargs) -> Any:
        self._name = name
        self._params = kwargs

    def get_name(self) -> Any:
        return self._name

    def get_params(self) -> Any:
        return self._params


def test_factory_creates_strategy() -> None:
    factory = get_strategy_factory()
    strategy = factory.create_strategy(
        name="trend_following",
        trading_pairs=["BTC/USDT"],
        parameters={"sma_period": 20},
        risk_level="medium",
        confidence_threshold=Decimal("0.7"),
    )
    assert hasattr(strategy, "get_name")
    assert hasattr(strategy, "trading_pairs")
    assert hasattr(strategy, "parameters")
    assert getattr(strategy, "trading_pairs") == ["BTC/USDT"]
    assert getattr(strategy, "parameters")["sma_period"] == 20


def test_factory_raises_on_invalid_type() -> None:
    factory = get_strategy_factory()
    with pytest.raises(StrategyCreationError):
        factory.create_strategy(
            name="nonexistent",
            trading_pairs=["BTC/USDT"],
            parameters={},
            risk_level="low",
            confidence_threshold=Decimal("0.5"),
        )


def test_factory_registers_custom_strategy() -> None:
    factory = get_strategy_factory()

    class CustomStrategy(DummyStrategy):
        pass

    factory.register_strategy_type("custom", CustomStrategy)
    strategy = factory.create_strategy(
        name="custom", trading_pairs=["ETH/USDT"], parameters={"foo": 1}, risk_level="high", confidence_threshold=0.9
    )
    assert isinstance(strategy, CustomStrategy)
    assert strategy.get_name() == "custom"
    assert strategy.get_params()["foo"] == 1
