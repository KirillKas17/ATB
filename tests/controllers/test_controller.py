import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.controllers.base import BaseController
from core.controllers.market_controller import MarketController
from core.controllers.order_controller import OrderController
from core.controllers.position_controller import PositionController
from core.controllers.risk_controller import RiskController
from core.controllers.trading_controller import TradingController
from core.types import TradingMode, TradingPair
from utils.logger import setup_logger

logger = setup_logger(__name__)


@pytest.fixture
def mock_config():
    return {
        "exchange": {"name": "binance", "api_key": "test_key", "api_secret": "test_secret"},
        "trading": {"pairs": ["BTC/USDT", "ETH/USDT"], "update_interval": 1, "signal_interval": 1},
        "min_confidence": 0.6,
        "high_confidence": 0.85,
        "min_volume": 0.01,
        "normal_volume": 0.1,
        "max_volume": 0.5,
        "min_leverage": 1,
        "normal_leverage": 3,
        "max_leverage": 5,
    }


@pytest.fixture
def mock_exchange():
    return AsyncMock()


@pytest.fixture
def trading_controller(mock_exchange, mock_config):
    return TradingController(mock_exchange, mock_config)


class TestTradingController:
    """Тесты для TradingController"""

    def test_initialization(self, trading_controller, mock_config):
        """Тест инициализации"""
        assert trading_controller.config == mock_config
        assert isinstance(trading_controller.order_controller, OrderController)
        assert isinstance(trading_controller.position_controller, PositionController)
        assert isinstance(trading_controller.market_controller, MarketController)
        assert isinstance(trading_controller.risk_controller, RiskController)

    @pytest.mark.asyncio
    async def test_start(self, trading_controller):
        """Тест запуска контроллера"""
        await trading_controller.start()
        assert len(trading_controller.monitoring_tasks) > 0

    @pytest.mark.asyncio
    async def test_stop(self, trading_controller):
        """Тест остановки контроллера"""
        await trading_controller.start()
        await trading_controller.stop()
        assert len(trading_controller.monitoring_tasks) == 0

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, trading_controller):
        """Тест отмены всех ордеров"""
        await trading_controller.cancel_all_orders()
        trading_controller.order_controller.cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_all_positions(self, trading_controller):
        """Тест закрытия всех позиций"""
        await trading_controller.close_all_positions()
        trading_controller.position_controller.close_all_positions.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitor_market(self, trading_controller):
        """Тест мониторинга рынка"""
        await trading_controller._monitor_market()
        trading_controller.market_controller.get_ohlcv.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_positions(self, trading_controller):
        """Тест мониторинга позиций"""
        await trading_controller._monitor_positions()
        trading_controller.position_controller.update_positions.assert_called()
        trading_controller.risk_controller.update_risk_metrics.assert_called()

    @pytest.mark.asyncio
    async def test_monitor_orders(self, trading_controller):
        """Тест мониторинга ордеров"""
        await trading_controller._monitor_orders()
        trading_controller.order_controller.get_open_orders.assert_called()
        trading_controller.order_controller.clear_invalid_orders.assert_called()

    def test_validate_config(self, trading_controller):
        """Тест валидации конфигурации"""
        assert trading_controller._validate_config()

        # Тест с невалидной конфигурацией
        trading_controller.config = {}
        assert not trading_controller._validate_config()


@pytest.fixture
def controller(mock_config):
    with patch(
        "core.controllers.trading_controller.ConfigLoader.load_config", return_value=mock_config
    ):
        return TradingController()


def test_controller_initialization(controller, mock_config):
    """Тест инициализации контроллера"""
    assert controller.config == mock_config
    assert controller.mode == TradingMode.PAUSED
    assert isinstance(controller.trading_pairs, dict)
    assert isinstance(controller.decision_history, list)


@pytest.mark.asyncio
async def test_start(controller):
    """Тест запуска контроллера"""
    with patch("core.controllers.trading_controller.Exchange.connect") as mock_connect:
        await controller.start()
        assert controller.is_running
        mock_connect.assert_called_once()


@pytest.mark.asyncio
async def test_stop(controller):
    """Тест остановки контроллера"""
    controller.is_running = True
    with patch("core.controllers.trading_controller.Exchange.disconnect") as mock_disconnect:
        await controller.stop()
        assert not controller.is_running
        mock_disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_set_mode(controller):
    """Тест установки режима работы"""
    with patch(
        "core.controllers.trading_controller.TradingController._start_trading_mode"
    ) as mock_start:
        await controller.set_mode(TradingMode.TRADING)
        assert controller.mode == TradingMode.TRADING
        mock_start.assert_called_once()


def test_load_trading_pairs(controller):
    """Тест загрузки торговых пар"""
    pairs = ["BTC/USDT", "ETH/USDT"]
    controller.config["trading"]["pairs"] = pairs
    controller._load_trading_pairs()
    assert len(controller.trading_pairs) == len(pairs)
    for pair in pairs:
        assert pair in controller.trading_pairs
        assert isinstance(controller.trading_pairs[pair], TradingPair)


@pytest.mark.asyncio
async def test_monitor_market(controller):
    """Тест мониторинга рынка"""
    controller.is_running = True
    with patch("core.controllers.trading_controller.Exchange.get_market_data") as mock_data:
        mock_data.return_value = {"price": 50000}
        await controller._monitor_market()
        mock_data.assert_called()


@pytest.mark.asyncio
async def test_process_signals(controller):
    """Тест обработки сигналов"""
    controller.is_running = True
    controller.mode = TradingMode.TRADING
    with patch(
        "core.controllers.trading_controller.TradingController._collect_signals"
    ) as mock_signals:
        mock_signals.return_value = [{"action": "buy", "confidence": 0.8}]
        await controller._process_signals()
        mock_signals.assert_called()


def test_calculate_levels(controller):
    """Тест расчета уровней"""
    symbol = "BTC/USDT"
    direction = "long"
    entry_price = 50000
    confidence = 0.8

    stop_loss, take_profit = controller._calculate_levels(
        symbol=symbol, direction=direction, entry_price=entry_price, confidence=confidence
    )

    assert isinstance(stop_loss, float)
    assert isinstance(take_profit, float)
    assert stop_loss < entry_price
    assert take_profit > entry_price


def test_log_decision_tree(controller):
    """Тест логирования дерева решений"""
    decision = TradingDecision(
        symbol="BTC/USDT",
        action="open",
        direction="long",
        volume=1.0,
        leverage=1,
        confidence=0.8,
        stop_loss=49000,
        take_profit=51000,
        timestamp=datetime.now(),
        metadata={},
    )

    signal = {"action": "buy", "confidence": 0.8}
    market_context = {"trend": "up", "volatility": 0.1}

    controller._log_decision_tree(decision, signal, market_context)
    # Проверяем, что логирование прошло без ошибок


@pytest.fixture
def market_context():
    """Фикстура для рыночного контекста"""
    return {
        "price": 50000.0,
        "regime": "trend",
        "whale_activity": True,
        "cvd": 100.0,
        "delta_volume": 50.0,
    }


@pytest.fixture
def high_confidence_signal():
    """Фикстура для сигнала с высокой уверенностью"""
    return {"direction": "long", "confidence": 0.9, "source": "ml_model"}


@pytest.fixture
def low_confidence_signal():
    """Фикстура для сигнала с низкой уверенностью"""
    return {"direction": "long", "confidence": 0.5, "source": "ml_model"}


@pytest.fixture
def normal_confidence_signal():
    """Фикстура для сигнала со средней уверенностью"""
    return {"direction": "long", "confidence": 0.75, "source": "ml_model"}


class TestTradingController:
    """Тесты для TradingController"""

    def test_initialization(self, controller):
        """Тест инициализации"""
        assert controller.config["min_confidence"] == 0.6
        assert controller.config["high_confidence"] == 0.85
        assert controller.config["min_volume"] == 0.01
        assert controller.config["normal_volume"] == 0.1
        assert controller.config["max_volume"] == 0.5
        assert controller.config["min_leverage"] == 1
        assert controller.config["normal_leverage"] == 3
        assert controller.config["max_leverage"] == 5

    async def test_decide_action_high_confidence(
        self, controller, high_confidence_signal, market_context
    ):
        """Тест принятия решения при высокой уверенности"""
        decision = await controller.decide_action(
            symbol="BTCUSDT", signal=high_confidence_signal, market_context=market_context
        )

        assert decision.action == "open"
        assert decision.volume == controller.config["max_volume"]
        assert decision.leverage == controller.config["max_leverage"]
        assert decision.direction == "long"
        assert decision.confidence == 0.9
        assert decision.stop_loss < market_context["price"]
        assert decision.take_profit > market_context["price"]

    async def test_decide_action_low_confidence(
        self, controller, low_confidence_signal, market_context
    ):
        """Тест принятия решения при низкой уверенности"""
        decision = await controller.decide_action(
            symbol="BTCUSDT", signal=low_confidence_signal, market_context=market_context
        )

        assert decision.action == "hold"
        assert decision.volume == 0.0
        assert decision.leverage == controller.config["min_leverage"]

    async def test_decide_action_normal_confidence(
        self, controller, normal_confidence_signal, market_context
    ):
        """Тест принятия решения при средней уверенности"""
        decision = await controller.decide_action(
            symbol="BTCUSDT", signal=normal_confidence_signal, market_context=market_context
        )

        assert decision.action == "open"
        assert decision.volume == controller.config["normal_volume"]
        assert decision.leverage == controller.config["normal_leverage"]

    async def test_calculate_levels_long(self, controller):
        """Тест расчета уровней для длинной позиции"""
        stop_loss, take_profit = controller._calculate_levels(
            symbol="BTCUSDT", direction="long", entry_price=50000.0, confidence=0.9
        )

        assert stop_loss < 50000.0
        assert take_profit > 50000.0
        assert (take_profit - 50000.0) / (50000.0 - stop_loss) >= controller.config[
            "risk_reward_ratio"
        ]

    async def test_calculate_levels_short(self, controller):
        """Тест расчета уровней для короткой позиции"""
        stop_loss, take_profit = controller._calculate_levels(
            symbol="BTCUSDT", direction="short", entry_price=50000.0, confidence=0.9
        )

        assert stop_loss > 50000.0
        assert take_profit < 50000.0
        assert (50000.0 - take_profit) / (stop_loss - 50000.0) >= controller.config[
            "risk_reward_ratio"
        ]

    async def test_log_decision_tree(
        self, controller, high_confidence_signal, market_context, tmp_path
    ):
        """Тест логирования дерева решений"""
        # Установка временной директории для логов
        controller.log_dir = tmp_path

        # Принятие решения
        decision = await controller.decide_action(
            symbol="BTCUSDT", signal=high_confidence_signal, market_context=market_context
        )

        # Проверка наличия файла лога
        log_file = tmp_path / "decision_tree_BTCUSDT.jsonl"
        assert log_file.exists()

        # Проверка содержимого лога
        with open(log_file) as f:
            log_entry = json.loads(f.readline())

            assert log_entry["symbol"] == "BTCUSDT"
            assert log_entry["decision"]["action"] == decision.action
            assert log_entry["decision"]["direction"] == decision.direction
            assert log_entry["decision"]["volume"] == decision.volume
            assert log_entry["decision"]["leverage"] == decision.leverage
            assert log_entry["decision"]["confidence"] == decision.confidence
            assert "signal" in log_entry
            assert "market_context" in log_entry
            assert "metadata" in log_entry

    async def test_decision_history(self, controller, high_confidence_signal, market_context):
        """Тест истории решений"""
        # Принятие нескольких решений
        decision1 = await controller.decide_action(
            symbol="BTCUSDT", signal=high_confidence_signal, market_context=market_context
        )

        decision2 = await controller.decide_action(
            symbol="ETHUSDT", signal=high_confidence_signal, market_context=market_context
        )

        # Проверка получения всей истории
        history = controller.get_decision_history()
        assert len(history) == 2

        # Проверка получения истории по символу
        btc_history = controller.get_decision_history("BTCUSDT")
        assert len(btc_history) == 1
        assert btc_history[0].symbol == "BTCUSDT"

        # Очистка истории по символу
        controller.clear_history("BTCUSDT")
        assert len(controller.get_decision_history("BTCUSDT")) == 0
        assert len(controller.get_decision_history("ETHUSDT")) == 1

        # Очистка всей истории
        controller.clear_history()
        assert len(controller.get_decision_history()) == 0
