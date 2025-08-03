from unittest.mock import Mock
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from simulation.backtest_explainer import BacktestExplainer
from simulation.backtester import Backtester
from simulation.market_simulator import MarketSimulator
# Фикстуры
@pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
    data = pd.DataFrame(
        {
            "open": np.random.normal(100, 1, 100),
            "high": np.random.normal(101, 1, 100),
            "low": np.random.normal(99, 1, 100),
            "close": np.random.normal(100, 1, 100),
            "volume": np.random.normal(1000, 100, 100),
        },
        index=dates,
    )
    return data
@pytest.fixture
def mock_strategy() -> Any:
    """Фикстура с тестовой стратегией"""
    strategy = Mock()
    strategy.generate_signals = Mock(
        return_value=[
            {"symbol": "BTCUSDT", "side": "long", "quantity": 1.0, "confidence": 0.8}
        ]
    )
    return strategy
@pytest.fixture
def backtester() -> Any:
    """Фикстура с бэктестером"""
    return Backtester(
        config={
            "initial_balance": 10000.0,
            "commission": 0.001,
            "slippage": 0.001,
            "max_position_size": 0.1,
            "risk_per_trade": 0.02,
        }
    )
@pytest.fixture
def market_simulator() -> Any:
    """Фикстура с симулятором рынка"""
    return MarketSimulator(volatility=0.02, trend_strength=0.5, noise_level=0.01)
@pytest.fixture
def backtest_explainer() -> Any:
    """Фикстура с объяснителем бэктеста"""
    return BacktestExplainer(min_trades=10, confidence_threshold=0.7)
# Тесты для Backtester
class TestBacktester:
    @pytest.mark.asyncio
    async def test_run_backtest(self, backtester, mock_strategy, mock_market_data) -> None:
        """Тест запуска бэктеста"""
        results = await backtester.run_backtest(
            symbol="BTCUSDT", strategy=mock_strategy, data=mock_market_data
        )
        assert isinstance(results, dict)
        assert "trades" in results
        assert "equity_curve" in results
        assert "metrics" in results
    def test_calculate_metrics(self, backtester, mock_market_data) -> None:
        """Тест расчета метрик"""
        trades = [
            {"entry_price": 100, "exit_price": 105, "size": 1, "pnl": 5},
            {"entry_price": 105, "exit_price": 100, "size": 1, "pnl": -5},
        ]
        metrics = backtester._calculate_metrics()
        assert isinstance(metrics, dict)
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
    @pytest.mark.asyncio
    async def test_initialize(self, backtester) -> None:
        """Тест инициализации"""
        await backtester.initialize()
        assert backtester.trading_controller is not None
    def test_save_load_results(self, backtester, tmp_path) -> None:
        """Тест сохранения и загрузки результатов"""
        # Подготовка тестовых данных
        backtester.results = {
            "trades": [{"entry_price": 100, "exit_price": 105, "size": 1, "pnl": 5}],
            "equity_curve": [10000, 10005],
            "metrics": {"win_rate": 1.0, "profit_factor": 2.0},
        }
        # Сохранение
        save_path = tmp_path / "test_results.json"
        backtester.save_results(str(save_path))
        assert save_path.exists()
        # Загрузка
        backtester.results = {}
        backtester.load_results(str(save_path))
        assert backtester.results["trades"][0]["pnl"] == 5
        assert backtester.results["metrics"]["win_rate"] == 1.0
# Тесты для MarketSimulator
class TestMarketSimulator:
    def test_simulate_market(self, mock_market_data) -> None:
        """Тест симуляции рынка"""
        simulator = MarketSimulator()
        simulated_data = simulator.simulate(
            initial_price=100, volatility=0.02, trend=0.001, periods=100
        )
        assert isinstance(simulated_data, pd.DataFrame)
        assert len(simulated_data) == 100
        assert all(
            col in simulated_data.columns
            for col in ["open", "high", "low", "close", "volume"]
        )
# Тесты для BacktestExplainer
class TestBacktestExplainer:
    def test_generate_recommendations(self, backtest_explainer, mock_market_data) -> None:
        """Тест генерации рекомендаций"""
        backtest_results = {
            "trades": [
                {"entry_price": 100, "exit_price": 105, "size": 1, "pnl": 5},
                {"entry_price": 105, "exit_price": 100, "size": 1, "pnl": -5},
            ],
            "metrics": {
                "win_rate": 0.5,
                "profit_factor": 1.0,
                "max_drawdown": 0.05,
                "sharpe_ratio": 0.0,
            },
        }
        recommendations = backtest_explainer.generate_recommendations(
            results=backtest_results, market_data=mock_market_data
        )
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, dict) for rec in recommendations)
        assert all("type" in rec and "description" in rec for rec in recommendations)
