import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from core.optimizer import OptimizationResult, Optimizer
class MockStrategy:
    def __init__(self, params) -> Any:
        self.params = params
    def generate_signals(self, data) -> Any:
        return []
@pytest.fixture
def sample_data() -> Any:
    dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="1H")
    data = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 101,
            "low": np.random.randn(len(dates)).cumsum() + 99,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, len(dates)),
        },
        index=dates,
    )
    return data
@pytest.fixture
def config() -> Any:
    return {
        "optimization": {"parallel": True, "max_workers": 4, "timeout": 300},
        "backtest": {"initial_balance": 10000, "commission": 0.001},
    }
@pytest.fixture
def param_grid() -> Any:
    return {"param1": [1, 2, 3], "param2": ["a", "b"]}
def test_optimizer_initialization(config) -> None:
    optimizer = Optimizer(config)
    assert optimizer.config == config
    assert optimizer.results == []
def test_generate_param_combinations(config, param_grid) -> None:
    optimizer = Optimizer(config)
    combinations = optimizer._generate_param_combinations(param_grid)
    assert len(combinations) == 6  # 3 * 2 combinations
    assert all(isinstance(combo, dict) for combo in combinations)
    assert all("param1" in combo and "param2" in combo for combo in combinations)
def test_optimize_sequential(config, sample_data, param_grid) -> None:
    optimizer = Optimizer(config)
    strategy = MockStrategy({})
    results = optimizer.optimize(sample_data, strategy, param_grid)
    assert isinstance(results, list)
    assert all(isinstance(r, OptimizationResult) for r in results)
    assert len(results) == 6  # Should match number of parameter combinations
def test_get_best_parameters(config, sample_data, param_grid) -> None:
    optimizer = Optimizer(config)
    strategy = MockStrategy({})
    optimizer.optimize(sample_data, strategy, param_grid)
    best_params = optimizer.get_best_parameters()
    assert isinstance(best_params, dict)
    assert "param1" in best_params
    assert "param2" in best_params
def test_get_optimization_statistics(config, sample_data, param_grid) -> None:
    optimizer = Optimizer(config)
    strategy = MockStrategy({})
    optimizer.optimize(sample_data, strategy, param_grid)
    stats = optimizer.get_optimization_statistics()
    assert isinstance(stats, dict)
    assert "total_combinations" in stats
    assert "best_metrics" in stats
    assert "parameter_importance" in stats
def test_evaluate_parameters(config, sample_data) -> None:
    optimizer = Optimizer(config)
    strategy = MockStrategy({"param1": 1, "param2": "a"})
    result = optimizer._evaluate_parameters(
        sample_data, strategy, {"param1": 1, "param2": "a"}
    )
    assert isinstance(result, OptimizationResult)
    assert result.parameters == {"param1": 1, "param2": "a"}
    assert isinstance(result.metrics, dict)
    assert isinstance(result.trades, list)
def test_sort_results(config) -> None:
    optimizer = Optimizer(config)
    # Add some mock results
    optimizer.results = [
        OptimizationResult(
            parameters={"param1": 1},
            metrics={"sharpe_ratio": 0.5},
            trades=[],
            metadata={},
        ),
        OptimizationResult(
            parameters={"param1": 2},
            metrics={"sharpe_ratio": 1.5},
            trades=[],
            metadata={},
        ),
    ]
    optimizer._sort_results()
    assert (
        optimizer.results[0].metrics["sharpe_ratio"]
        > optimizer.results[1].metrics["sharpe_ratio"]
    )
