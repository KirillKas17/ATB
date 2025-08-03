import json
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator_asyncio
import yaml
from agents.agent_market_regime import MarketRegime, MarketRegimeAgent
from agents.agent_meta_controller import (BayesianMetaController,
                                          MetaControllerAgent, PairManager)
from agents.agent_order_executor import OrderExecutorAgent
from agents.agent_portfolio import PortfolioAgent
from agents.agent_risk import RiskAgent
from core.types import TradeDecision
from exchange.bybit_client import BybitClient, BybitConfig
from exchange.market_data import MarketData
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
@pytest_asyncio.fixture
async def bybit_client(bybit_config) -> Any:
    """Фикстура с клиентом Bybit"""
    client = BybitClient(config=bybit_config)
    await client.initialize()
    return client
@pytest_asyncio.fixture
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
@pytest_asyncio.fixture
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
@pytest_asyncio.fixture
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
@pytest_asyncio.fixture
async def pair_manager() -> Any:
    """Фикстура с менеджером пар"""
    manager = PairManager(config_path="config/allowed_pairs.yaml")
    await manager.initialize()
    return manager
@pytest_asyncio.fixture
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
        adjusted_leverage = risk_agent.adjust_leverage(
            base_leverage=base_leverage, volatility=volatility
        )
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
        assert isinstance(sl, float)
        assert isinstance(tp, float)
# Тесты для PortfolioAgent
class TestPortfolioAgent:
    def test_get_portfolio_allocation(self, portfolio_agent, mock_market_data) -> None:
        """Тест получения распределения портфеля"""
        positions = {
            "BTC/USDT": {"size": 0.5, "pnl": 0.1},
            "ETH/USDT": {"size": 0.3, "pnl": -0.05},
        }
        allocation = portfolio_agent.get_portfolio_allocation(positions)
        assert isinstance(allocation, dict)
        assert sum(allocation.values()) <= 1.0
    def test_rebalance_portfolio(self, portfolio_agent, mock_market_data) -> None:
        """Тест ребалансировки портфеля"""
        current_positions = {"BTC/USDT": 0.7, "ETH/USDT": 0.3}
        target_allocation = {"BTC/USDT": 0.5, "ETH/USDT": 0.5}
        rebalance_orders = portfolio_agent.rebalance_portfolio(
            current_positions=current_positions, target_allocation=target_allocation
        )
        assert isinstance(rebalance_orders, list)
        assert len(rebalance_orders) > 0
# Тесты для MarketRegimeAgent
class TestMarketRegimeAgent:
    def test_detect_regime(self, market_regime_agent, mock_market_data) -> None:
        """Тест определения рыночного режима"""
        regime, confidence = market_regime_agent.detect_regime(mock_market_data)
        assert isinstance(regime, MarketRegime)
        assert 0 <= confidence <= 1
    def test_switch_regime_logic(self, market_regime_agent, mock_market_data) -> None:
        """Тест логики переключения режимов"""
        # Первое определение режима
        regime1, conf1 = market_regime_agent.detect_regime(mock_market_data)
        # Изменяем данные для смены режима
        mock_market_data["close"] *= 1.1  # Сильный тренд вверх
        # Второе определение режима
        regime2, conf2 = market_regime_agent.detect_regime(mock_market_data)
        assert regime1 != regime2 or conf1 != conf2
# Тесты для OrderExecutorAgent
class TestOrderExecutorAgent:
    def test_place_order(self, order_executor_agent) -> None:
        """Тест размещения ордера"""
        order = order_executor_agent.place_order(
            symbol="BTC/USDT", side="buy", type="limit", amount=0.1, price=50000
        )
        assert isinstance(order, dict)
        assert "order_id" in order
    def test_cancel_order(self, order_executor_agent) -> None:
        """Тест отмены ордера"""
        result = order_executor_agent.cancel_order(
            symbol="BTC/USDT", order_id="test_order_id"
        )
        assert isinstance(result, bool)
# Тесты для MetaControllerAgent
@pytest.mark.asyncio
async def test_meta_controller_initialization(meta_controller_agent) -> None:
    """Тест инициализации мета-контроллера"""
    assert meta_controller_agent.config is not None
    assert meta_controller_agent.market_regime_agent is not None
    assert meta_controller_agent.risk_agent is not None
    assert meta_controller_agent.whales_agent is not None
    assert meta_controller_agent.news_agent is not None
    assert meta_controller_agent.market_maker_agent is not None
    assert meta_controller_agent.strategies == {}
    assert meta_controller_agent.active_strategies == {}
    assert meta_controller_agent.last_retrain == {}
@pytest.mark.asyncio
async def test_meta_controller_evaluate_strategies(meta_controller_agent) -> None:
    """Тест оценки стратегий"""
    symbol = "BTC/USDT"
    result = await meta_controller_agent.evaluate_strategies(symbol)
    assert result is None  # Должен вернуть None, так как нет активных стратегий
@pytest.mark.asyncio
async def test_meta_controller_retrain_if_needed(meta_controller_agent) -> None:
    """Тест ретрейнинга стратегий"""
    symbol = "BTC/USDT"
    await meta_controller_agent.retrain_if_needed(symbol)
    assert symbol in meta_controller_agent.last_retrain
def test_activate_best_strategy(meta_controller_agent) -> None:
    """Тест активации лучшей стратегии"""
    pair = "BTC/USDT"
    result = meta_controller_agent.activate_best_strategy(pair)
    assert isinstance(result, bool)
def test_is_pair_ready_to_trade(meta_controller_agent) -> None:
    """Тест проверки готовности пары к торговле"""
    pair = "BTC/USDT"
    result = meta_controller_agent.is_pair_ready_to_trade(pair)
    assert isinstance(result, bool)
def test_pair_manager_initialization(pair_manager) -> None:
    """Тест инициализации менеджера пар"""
    assert pair_manager.config_path == "config/allowed_pairs.yaml"
    assert isinstance(pair_manager.allowed_pairs, dict)
def test_create_base_strategy(pair_manager) -> None:
    """Тест создания базовой стратегии"""
    pair = "BTC/USDT"
    pair_manager.create_base_strategy(pair)
    assert pair in pair_manager.allowed_pairs
    assert "strategy" in pair_manager.allowed_pairs[pair]
    assert "indicators" in pair_manager.allowed_pairs[pair]
def test_create_base_indicators(pair_manager) -> None:
    """Тест создания базовых индикаторов"""
    pair = "BTC/USDT"
    pair_manager.create_base_indicators(pair)
    assert pair in pair_manager.allowed_pairs
    assert "indicators" in pair_manager.allowed_pairs[pair]
    assert isinstance(pair_manager.allowed_pairs[pair]["indicators"], dict)
def test_init_pair_structure(pair_manager) -> None:
    """Тест инициализации структуры пары"""
    pair = "BTC/USDT"
    pair_manager.init_pair_structure(pair)
    assert pair in pair_manager.allowed_pairs
    assert "status" in pair_manager.allowed_pairs[pair]
    assert "strategy" in pair_manager.allowed_pairs[pair]
    assert "indicators" in pair_manager.allowed_pairs[pair]
def test_get_pair_status(pair_manager) -> None:
    """Тест получения статуса пары"""
    pair = "BTC/USDT"
    pair_manager.init_pair_structure(pair)
    status = pair_manager.get_pair_status(pair)
    assert isinstance(status, dict)
    assert "status" in status
    assert "strategy" in status
    assert "indicators" in status
def test_get_active_pairs(pair_manager) -> None:
    """Тест получения активных пар"""
    pair = "BTC/USDT"
    pair_manager.init_pair_structure(pair)
    active_pairs = pair_manager.get_active_pairs()
    assert isinstance(active_pairs, list)
    assert pair in active_pairs
def test_update_pair_status(pair_manager) -> None:
    """Тест обновления статуса пары"""
    pair = "BTC/USDT"
    pair_manager.init_pair_structure(pair)
    status_updates = {
        "status": "active",
        "strategy": "trend_following",
        "indicators": {"rsi": True, "macd": True},
    }
    pair_manager.update_pair_status(pair, status_updates)
    assert pair_manager.allowed_pairs[pair]["status"] == "active"
    assert pair_manager.allowed_pairs[pair]["strategy"] == "trend_following"
    assert pair_manager.allowed_pairs[pair]["indicators"]["rsi"] is True
    assert pair_manager.allowed_pairs[pair]["indicators"]["macd"] is True
def test_check_pair_requirements(pair_manager) -> None:
    """Тест проверки требований пары"""
    pair = "BTC/USDT"
    pair_manager.init_pair_structure(pair)
    requirements = pair_manager.check_pair_requirements(pair)
    assert isinstance(requirements, dict)
    assert "has_strategy" in requirements
    assert "has_indicators" in requirements
    assert "is_active" in requirements
def test_strategy_and_config_init() -> None:
    """Test initialization of strategy and config files."""
    from agents.agent_meta_controller import PairManager
    pair = "TESTPAIR2"
    manager = PairManager()
    manager.init_pair_structure(pair)
    # Check strategy profile
    strategy_path = Path("data/pairs") / pair / "strategy_profile.json"
    assert strategy_path.exists()
    with open(strategy_path) as f:
        strategy = json.load(f)
        assert "entry_signal" in strategy
        assert "exit_signal" in strategy
        assert "stop_loss" in strategy
        assert "take_profit" in strategy
        assert "confidence_score" in strategy
        assert strategy["regime"] == "trend"
    # Check indicators config
    config_path = Path("data/pairs") / pair / "indicators_config.yaml"
    assert config_path.exists()
    with open(config_path) as f:
        config = yaml.safe_load(f)
        assert "timeframes" in config
        assert "indicators" in config
        assert config["timeframes"]["main"] == "1h"
        assert config["timeframes"]["confirm"] == "4h"
        assert config["timeframes"]["entry"] == "15m"
        assert len(config["indicators"]) == 4
        assert any(ind["name"] == "EMA" for ind in config["indicators"])
        assert any(ind["name"] == "RSI" for ind in config["indicators"])
        assert any(ind["name"] == "ATR" for ind in config["indicators"])
        assert any(ind["name"] == "OBV" for ind in config["indicators"])
    # Check meta status
    meta_path = Path("data/pairs") / pair / "meta_status.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
        assert meta["WR"] == 0.0
        assert meta["is_trade_ready"] is False
        assert meta["is_trained"] is False
        assert meta["strategy_defined"] is True
    # Cleanup
    import shutil
    shutil.rmtree(Path("data/pairs") / pair)
def test_bayesian_meta_controller_initialization(bayesian_meta_controller) -> None:
    """Тест инициализации байесовского мета-контроллера"""
    assert bayesian_meta_controller.config is not None
    assert bayesian_meta_controller.config["min_confidence"] == 0.7
    assert bayesian_meta_controller.config["min_signals"] == 2
    assert bayesian_meta_controller.config["max_position_size"] == 1.0
    assert bayesian_meta_controller.config["history_size"] == 100
    assert bayesian_meta_controller.config["update_interval"] == 3600

def test_aggregate_signals(bayesian_meta_controller) -> None:
    """Тест агрегации сигналов"""
    signals = [
        {
            "action": "buy",
            "confidence": 0.8,
            "source": "strategy1",
            "timestamp": datetime.now(),
        },
        {
            "action": "buy",
            "confidence": 0.7,
            "source": "strategy2",
            "timestamp": datetime.now(),
        },
    ]
    decision = bayesian_meta_controller.aggregate_signals(signals)
    assert isinstance(decision, TradeDecision)
    assert decision.action in ["buy", "sell", "hold"]
    assert 0 <= decision.confidence <= 1
    assert isinstance(decision.position_size, float)
    assert isinstance(decision.stop_loss, float)
    assert isinstance(decision.take_profit, float)
    assert isinstance(decision.source, str)
    assert isinstance(decision.timestamp, datetime)
    assert isinstance(decision.explanation, str)

def test_evaluate_strategy_performance(bayesian_meta_controller) -> None:
    """Тест оценки производительности стратегий"""
    performance = bayesian_meta_controller.evaluate_strategy_performance()
    assert isinstance(performance, dict)
    for strategy, metrics in performance.items():
        assert isinstance(strategy, str)
        assert isinstance(metrics, dict)
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
