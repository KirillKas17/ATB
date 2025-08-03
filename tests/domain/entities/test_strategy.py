"""
Заглушка test_strategy.py. Оригинальный код утерян. Для корректной работы mypy и импорта.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime
from decimal import Decimal
from domain.entities.strategy import Strategy, StrategyType, StrategyStatus
from domain.entities.signal import Signal, SignalType
from domain.entities.strategy_parameters import StrategyParameters
def test_strategy_creation() -> None:
    strategy = Strategy(
        name="Test Strategy",
        description="Test Description",
        strategy_type=StrategyType.TREND_FOLLOWING,
        trading_pairs=["BTC/USDT"],
    )
    assert strategy.name == "Test Strategy"
    assert strategy.strategy_type == StrategyType.TREND_FOLLOWING
    assert strategy.status == StrategyStatus.ACTIVE
    assert "BTC/USDT" in strategy.trading_pairs
    assert strategy.is_active
def test_strategy_add_signal() -> None:
    strategy = Strategy(trading_pairs=["BTC/USDT"])
    signal = Signal(
        strategy_id=strategy.id,
        trading_pair="BTC/USDT",
        signal_type=SignalType.BUY,
        confidence=Decimal("0.8"),
        timestamp=datetime.now(),
    )
    strategy.add_signal(signal)
    assert len(strategy.signals) == 1
    assert strategy.signals[0].signal_type == SignalType.BUY
def test_strategy_get_latest_signal() -> None:
    strategy = Strategy(trading_pairs=["BTC/USDT"])
    now = datetime.now()
    s1 = Signal(strategy_id=strategy.id, trading_pair="BTC/USDT", signal_type=SignalType.BUY, confidence=Decimal("0.7"), timestamp=now)
    s2 = Signal(strategy_id=strategy.id, trading_pair="BTC/USDT", signal_type=SignalType.SELL, confidence=Decimal("0.6"), timestamp=now.replace(second=now.second+1))
    strategy.add_signal(s1)
    strategy.add_signal(s2)
    latest = strategy.get_latest_signal("BTC/USDT")
    assert latest.signal_type == SignalType.SELL
def test_strategy_update_status() -> None:
    strategy = Strategy()
    strategy.update_status(StrategyStatus.PAUSED)
    assert strategy.status == StrategyStatus.PAUSED
def test_strategy_parameters() -> None:
    params = StrategyParameters()
    params.set_parameter("window", 20)
    assert params.get_parameter("window") == 20
    params.update_parameters({"threshold": 0.05})
    assert params.get_parameter("threshold") == 0.05
def test_strategy_save_and_load_state(tmp_path) -> None:
    strategy = Strategy(name="SaveTest", trading_pairs=["BTC/USDT"])
    strategy.parameters.set_parameter("window", 10)
    file = tmp_path / "state.pkl"
    assert strategy.save_state(str(file))
    loaded = Strategy(name="Loaded", trading_pairs=["BTC/USDT"])
    assert loaded.load_state(str(file))
    assert loaded.name == "SaveTest"
    assert loaded.parameters.get_parameter("window") == 10
def test_strategy_should_execute_signal() -> None:
    strategy = Strategy(trading_pairs=["BTC/USDT"])
    signal = Signal(strategy_id=strategy.id, trading_pair="BTC/USDT", signal_type=SignalType.BUY, confidence=Decimal("0.8"), timestamp=datetime.now())
    assert strategy.should_execute_signal(signal)
    strategy.is_active = False
    assert not strategy.should_execute_signal(signal)
def test_strategy_to_dict() -> None:
    strategy = Strategy(name="DictTest", trading_pairs=["BTC/USDT"])
    d = strategy.to_dict()
    assert d["name"] == "DictTest"
    assert "BTC/USDT" in d["trading_pairs"]
def test_strategy_generate_signals_errors() -> None:
    strategy = Strategy(trading_pairs=["BTC/USDT"])
    strategy.is_active = False
    with pytest.raises(Exception):
        strategy.generate_signals("BTC/USDT")
    strategy.is_active = True
    with pytest.raises(ValueError):
        strategy.generate_signals("ETH/USDT") 
