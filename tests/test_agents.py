import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import yaml
from infrastructure.agents.market_regime.agent import MarketRegime, MarketRegimeAgent
from infrastructure.agents.meta_controller.agent import BayesianMetaController, MetaControllerAgent, PairManager
from infrastructure.agents.order_executor.agent_order_executor import OrderExecutorAgent
from infrastructure.agents.portfolio.agent_portfolio import PortfolioAgent
from infrastructure.agents.risk.agent import RiskAgent
from infrastructure.core.types import TradeDecision
from infrastructure.external_services.bybit_client import BybitClient, BybitConfig
from infrastructure.external_services.market_data import MarketData
from shared.logging import setup_logger

logger = setup_logger(__name__)


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
    strategy.generate_signal.return_value = {
        "action": "buy",
        "confidence": 0.8,
        "stop_loss": 95,
        "take_profit": 105,
    }
    return strategy


@pytest.fixture
def risk_agent() -> Any:
    """Фикстура с агентом управления рисками"""
    return RiskAgent(
        config={
            "max_position_size": 1.0,
            "max_portfolio_risk": 0.02,
            "var_confidence": 0.95,
            "max_leverage": 3.0,
            "min_leverage": 1.0,
            "kelly_threshold": 0.5,
            "drawdown_threshold": 0.1,
            "volatility_lookback": 20,
            "equity_lookback": 100,
            "risk_per_trade": 0.02,
        }
    )


@pytest.fixture
def portfolio_agent() -> Any:
    """Фикстура с агентом управления портфелем"""
    return PortfolioAgent(
        config={
            "max_correlation": 0.7,
            "min_diversification": 3,
            "max_position_size": 0.1,
            "rebalance_threshold": 0.1,
            "rebalance_interval": 24,
        }
    )


@pytest.fixture
def market_regime_agent() -> Any:
    """Фикстура с агентом определения режима рынка"""
    return MarketRegimeAgent(
        config={
            "window_size": 20,
            "volatility_threshold": 0.02,
            "trend_threshold": 0.01,
            "regime_change_threshold": 0.05,
        }
    )


@pytest.fixture
def bybit_config() -> Any:
    """Фикстура с конфигурацией Bybit"""
    return BybitConfig(api_key="test_key", api_secret="test_secret", testnet=True)


@pytest.fixture
async def bybit_client(bybit_config) -> Any:
    """Фикстура с клиентом Bybit"""
    client = BybitClient(config=bybit_config)
    await client.initialize()
    return client


@pytest.fixture
async def market_data() -> Any:
    """Фикстура с рыночными данными"""
    data = MarketData(
        symbol="BTC/USDT",
        interval="1m",
        exchange="bybit",
        api_key="test_key",
        api_secret="test_secret",
    )
    await data.initialize()
    return data


@pytest.fixture
async def order_executor_agent(bybit_client) -> Any:
    """Фикстура с агентом исполнения ордеров"""
    agent = OrderExecutorAgent(client=bybit_client)
    await agent.initialize()
    return agent


@pytest.fixture
def mock_config() -> Any:
    return {
        "min_win_rate": 0.7,
        "min_profit_factor": 1.5,
        "min_sharpe": 1.0,
        "max_drawdown": 0.2,
        "min_trades": 100,
        "retrain_threshold": 0.1,
        "evaluation_period": 30,
        "confidence_threshold": 0.8,
    }


@pytest.fixture
async def meta_controller_agent() -> Any:
    """Фикстура с мета-контроллером"""
    agent = MetaControllerAgent(
        config={
            "min_win_rate": 0.55,
            "min_profit_factor": 1.5,
            "min_sharpe": 1.0,
            "min_trades": 100,
            "retrain_interval": 24,
            "max_drawdown": 0.15,
            "confidence_threshold": 0.7,
        }
    )
    await agent.initialize()
    return agent


@pytest.fixture
async def pair_manager() -> Any:
    """Фикстура с менеджером пар"""
    manager = PairManager(config_path="config/allowed_pairs.yaml")
    await manager.initialize()
    return manager


@pytest.fixture
async def bayesian_meta_controller() -> Any:
    """Фикстура с байесовским мета-контроллером"""
    controller = BayesianMetaController(
        config={
            "min_confidence": 0.7,
            "min_signals": 2,
            "max_position_size": 1.0,
            "history_size": 100,
            "update_interval": 3600,
        }
    )
    await controller.initialize()
    return controller


# Тесты для RiskAgent
class TestRiskAgent:
    def test_calculate_position_size(self, risk_agent, mock_market_data) -> None:
        """Тест расчета размера позиции"""
        account_balance = 10000
        current_price = 100
        volatility = 0.02
        position_size = risk_agent.calculate_position_size(
            account_balance=account_balance,
            current_price=current_price,
            volatility=volatility,
        )
        assert 0 < position_size <= 1.0
        assert isinstance(position_size, float)

    def test_adjust_leverage(self, risk_agent, mock_market_data) -> None:
        """Тест корректировки плеча"""
        base_leverage = 2.0
        volatility = 0.03
        adjusted_leverage = risk_agent.adjust_leverage(base_leverage=base_leverage, volatility=volatility)
        assert 0 < adjusted_leverage <= 3.0
        assert isinstance(adjusted_leverage, float)

    def test_calculate_stop_loss_take_profit(self, risk_agent, mock_market_data) -> None:
        """Тест расчета стоп-лосса и тейк-профита"""
        entry_price = 100
        direction = "buy"
        volatility = 0.02
        sl, tp = risk_agent.calculate_stop_loss_take_profit(
            entry_price=entry_price, direction=direction, volatility=volatility
        )
        assert sl < entry_price
        assert tp > entry_price
