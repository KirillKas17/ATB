from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from unittest.mock import Mock, patch
from infrastructure.core.controllers.trading_controller import TradingController
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from domain.entities import Order, Position
else:
    try:
        from domain.entities import Order, Position
    except ImportError:
        # Создаем заглушки для тестирования
        class Order:
            def __init__(
                self,
                id: str,
                pair: str,
                type: str,
                side: str,
                price: float,
                size: float,
                status: str,
                timestamp: datetime,
            ) -> Any:
                self.id = id
                self.pair = pair
                self.type = type
                self.side = side
                self.price = price
                self.size = size
                self.status = status
                self.timestamp = timestamp

        class Position:
            def __init__(
                self,
                pair: str,
                side: str,
                size: float,
                entry_price: float,
                current_price: float,
                pnl: float,
                leverage: float,
                entry_time: datetime,
            ) -> Any:
                self.pair = pair
                self.side = side
                self.size = size
                self.entry_price = entry_price
                self.current_price = current_price
                self.pnl = pnl
                self.leverage = leverage
                self.entry_time = entry_time


@pytest.fixture
def mock_exchange() -> Any:
    exchange = AsyncMock()
    exchange.get_balance = Mock(
        return_value={
            "BTC": {"free": 1.0, "used": 0.0, "total": 1.0},
            "USDT": {"free": 10000.0, "used": 0.0, "total": 10000.0},
        }
    )
    exchange.get_ticker = Mock(return_value={"last": 50000.0, "bid": 49900.0, "ask": 50100.0, "volume": 100.0})
    exchange.create_order = Mock(
        return_value={
            "id": "test_order",
            "symbol": "BTC/USDT",
            "type": "market",
            "side": "sell",
            "amount": 0.1,
            "price": 50000.0,
            "status": "closed",
        }
    )
    exchange.cancel_order = Mock(return_value={"id": "test_order", "status": "canceled"})
    exchange.get_markets = Mock(return_value=[{"symbol": "BTC/USDT", "active": True}])
    return exchange


@pytest.fixture
def config() -> Any:
    return {
        "trading_pairs": ["BTC/USDT"],
        "risk_limits": {
            "max_position_size": 1.0,
            "max_daily_loss": 1000.0,
            "max_leverage": 3.0,
        },
        "order_settings": {"default_amount": 0.1, "default_leverage": 1.0},
        "market_update_interval": 60,
        "position_update_interval": 60,
        "order_update_interval": 60,
    }


@pytest.fixture
def controller(mock_exchange, config) -> Any:
    return TradingController(mock_exchange, config)


@pytest.mark.asyncio
async def test_start_trading(controller) -> None:
    """Тест запуска торговли"""
    result = await controller.start_trading("manual")
    assert result is True
    assert controller.is_trading is True
    assert controller.trading_mode == "manual"


@pytest.mark.asyncio
async def test_stop_trading(controller) -> None:
    """Тест остановки торговли"""
    await controller.start_trading("manual")
    result = await controller.stop_trading()
    assert result is True
    assert controller.is_trading is False


@pytest.mark.asyncio
async def test_get_trading_status(controller) -> None:
    """Тест получения статуса торговли"""
    result = await controller.get_trading_status()
    
    assert "is_trading" in result
    assert "trading_mode" in result
    assert "open_positions" in result
    assert "account_balance" in result
    assert "total_pnl" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_place_trade(controller) -> None:
    """Тест размещения торговой операции"""
    result = await controller.place_trade("BTC/USDT", "buy", 0.1)
    
    assert "success" in result
    assert "order" in result or "errors" in result


@pytest.mark.asyncio
async def test_close_position(controller) -> None:
    """Тест закрытия позиции"""
    # Мокаем get_position чтобы вернуть позицию
    position_data = {
        "symbol": "BTC/USDT",
        "side": "buy",
        "size": 0.1,
        "entry_price": 50000.0
    }
    
    with patch.object(controller.position_controller, 'get_position', return_value=position_data):
        with patch.object(controller.position_controller, 'close_position', return_value={"id": "123", "status": "closed"}):
            result = await controller.close_position("BTC/USDT")
            
            assert "success" in result
            assert result["success"] is True


@pytest.mark.asyncio
async def test_get_risk_report(controller) -> None:
    """Тест получения отчета о рисках"""
    result = await controller.get_risk_report()
    
    assert "portfolio_risk" in result
    assert "alerts" in result
    assert "positions_count" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_execute_strategy(controller) -> None:
    """Тест выполнения стратегии"""
    await controller.start_trading("manual")
    result = await controller.execute_strategy("test_strategy", "BTC/USDT", {})
    
    assert "success" in result
    assert "strategy" in result
    assert "symbol" in result


@pytest.mark.asyncio
async def test_monitor_positions(controller) -> None:
    """Тест мониторинга позиций"""
    result = await controller.monitor_positions()
    
    assert isinstance(result, list)
    # Проверяем, что возвращается список действий


@pytest.mark.asyncio
async def test_execute_strategy_not_trading(controller) -> None:
    """Тест выполнения стратегии когда торговля не активна"""
    result = await controller.execute_strategy("test_strategy", "BTC/USDT", {})
    
    assert result["success"] is False
    assert "Trading is not active" in result["errors"]


@pytest.mark.asyncio
async def test_close_position_not_found(controller) -> None:
    """Тест закрытия несуществующей позиции"""
    with patch.object(controller.position_controller, 'get_position', return_value=None):
        result = await controller.close_position("BTC/USDT")
        
        assert result["success"] is False
        assert "No position found" in result["errors"][0]


@pytest.mark.asyncio
async def test_place_trade_with_limit_order(controller) -> None:
    """Тест размещения лимитного ордера"""
    result = await controller.place_trade("BTC/USDT", "buy", 0.1, "limit", 50000.0)
    
    assert "success" in result
    assert "order" in result or "errors" in result


def test_controller_initialization(controller) -> None:
    """Тест инициализации контроллера"""
    assert controller.is_trading is False
    assert controller.trading_mode == "manual"
    assert controller.market_controller is not None
    assert controller.order_controller is not None
    assert controller.position_controller is not None
    assert controller.risk_controller is not None


