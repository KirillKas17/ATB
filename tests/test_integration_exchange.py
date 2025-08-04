from unittest.mock import AsyncMock, patch
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pytest
from unittest.mock import Mock, patch
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator_asyncio
from core.exchange import Exchange
from domain.entities.order import Order
from exchange.account_manager import AccountManager
from exchange.bybit_client import BybitClient, BybitConfig
from exchange.market_data import MarketData
from exchange.order_manager import OrderManager
# Фикстуры
@pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
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
async def order_manager(bybit_client) -> Any:
    """Фикстура с менеджером ордеров"""
    manager = OrderManager(client=bybit_client)
    await manager.initialize()
    return manager
@pytest_asyncio.fixture
async def account_manager(bybit_client) -> Any:
    """Фикстура с менеджером аккаунта"""
    manager = AccountManager(client=bybit_client)
    await manager.initialize()
    return manager
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
# Тесты для BybitClient
class TestBybitClient:
    @pytest.mark.asyncio
    async def test_create_order(self, bybit_client) -> None:
        """Тест создания ордера"""
        with patch("ccxt.bybit") as mock_exchange:
            mock_exchange.create_order = AsyncMock(return_value={"id": "123"})
            result = await bybit_client.create_order(
                symbol="BTC/USDT",
                order_type="limit",
                side="buy",
                amount=0.1,
                price=50000,
            )
            assert isinstance(result, dict)
            assert "id" in result
    @pytest.mark.asyncio
    async def test_cancel_order(self, bybit_client) -> None:
        """Тест отмены ордера"""
        order_id = "123"
        symbol = "BTC/USDT"
        with patch("ccxt.bybit") as mock_exchange:
            mock_exchange.cancel_order = AsyncMock(return_value={"status": "cancelled"})
            result = await bybit_client.cancel_order(order_id, symbol)
            assert isinstance(result, dict)
            assert result["status"] == "cancelled"
    @pytest.mark.asyncio
    async def test_get_balance(self, bybit_client) -> None:
        """Тест получения баланса"""
        with patch("ccxt.bybit") as mock_exchange:
            mock_exchange.fetch_balance = AsyncMock(
                return_value={"USDT": {"free": 1000, "used": 0, "total": 1000}}
            )
            result = await bybit_client.get_balance()
            assert isinstance(result, dict)
            assert "USDT" in result
# Тесты для OrderManager
class TestOrderManager:
    @pytest.mark.asyncio
    async def test_create_entry_order(self, order_manager) -> None:
        """Тест создания входного ордера"""
        order = await order_manager.create_entry_order(
            symbol="BTC/USDT",
            side="buy",
            amount=0.1,
            price=50000,
            stop_loss=49000,
            take_profit=51000,
            confidence=0.8,
        )
        assert isinstance(order, Order)
        assert order.symbol == "BTC/USDT"
        assert order.side == "buy"
        assert order.amount == 0.1
        assert order.price == 50000
    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager) -> None:
        """Тест отмены ордера"""
        with patch.object(
            order_manager.client,
            "cancel_order",
            AsyncMock(return_value={"status": "cancelled"}),
        ):
            result = await order_manager.cancel_order("test_order_id")
            assert result is True
# Тесты для AccountManager
class TestAccountManager:
    def test_get_account_info(self, account_manager) -> None:
        """Тест получения информации об аккаунте"""
        info = account_manager.get_account_info()
        assert isinstance(info, dict)
        assert "balance" in info
        assert "positions" in info
    def test_set_leverage(self, account_manager) -> None:
        """Тест установки плеча"""
        result = account_manager.set_leverage(symbol="BTC/USDT", leverage=10)
        assert isinstance(result, bool)
# Тесты для MarketData
class TestMarketData:
    @pytest.mark.asyncio
    async def test_get_klines(self, market_data) -> None:
        """Тест получения свечей"""
        with patch.object(
            market_data.client,
            "get_klines",
            AsyncMock(
                return_value=[
                    [1625097600000, 35000, 36000, 34000, 35500, 100],
                    [1625097900000, 35500, 36500, 35000, 36000, 150],
                ]
            ),
        ):
            data = await market_data.get_klines(
                symbol="BTC/USDT", interval="1m", limit=100
            )
            assert isinstance(data, list)
            assert len(data) > 0
    @pytest.mark.asyncio
    async def test_get_orderbook(self, market_data) -> None:
        """Тест получения стакана"""
        with patch.object(
            market_data.client,
            "get_orderbook",
            AsyncMock(
                return_value={
                    "bids": [[35000, 1.5], [34900, 2.0]],
                    "asks": [[35100, 1.0], [35200, 2.5]],
                }
            ),
        ):
            orderbook = await market_data.get_orderbook(symbol="BTC/USDT", depth=20)
            assert isinstance(orderbook, dict)
            assert "bids" in orderbook
            assert "asks" in orderbook
def test_exchange_initialization() -> None:
    """Тест инициализации базового класса Exchange"""
    with pytest.raises(TypeError):
        Exchange()
def test_error_handling(bybit_client) -> None:
    """Тест обработки ошибок"""
    # Тест ошибки при получении рыночных данных
    with patch("ccxt.bybit") as mock_exchange:
        mock_exchange.fetch_ohlcv = AsyncMock(side_effect=Exception("API Error"))
        with pytest.raises(Exception):
            bybit_client.get_ohlcv("BTC/USDT", "1h")
    # Тест ошибки при создании ордера
    with patch("ccxt.bybit") as mock_exchange:
        mock_exchange.create_order = AsyncMock(side_effect=Exception("API Error"))
        with pytest.raises(Exception):
            bybit_client.place_order(
                {
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "type": "limit",
                    "amount": 0.1,
                    "price": 50000,
                }
            )
    # Тест ошибки при отмене ордера
    with patch("ccxt.bybit") as mock_exchange:
        mock_exchange.cancel_order = AsyncMock(side_effect=Exception("API Error"))
        with pytest.raises(Exception):
            bybit_client.cancel_order("12345", "BTC/USDT")
def test_data_validation(bybit_client) -> None:
    """Тест валидации данных"""
    # Тест с некорректной парой
    with pytest.raises(ValueError):
        bybit_client.get_ohlcv("INVALID", "1h")
    # Тест с некорректным типом ордера
    with pytest.raises(ValueError):
        bybit_client.place_order(
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "invalid",
                "amount": 0.1,
                "price": 50000,
            }
        )
    # Тест с некорректной стороной ордера
    with pytest.raises(ValueError):
        bybit_client.place_order(
            {
                "symbol": "BTC/USDT",
                "side": "invalid",
                "type": "limit",
                "amount": 0.1,
                "price": 50000,
            }
        )
    # Тест с некорректным размером ордера
    with pytest.raises(ValueError):
        bybit_client.place_order(
            {
                "symbol": "BTC/USDT",
                "side": "buy",
                "type": "limit",
                "amount": -0.1,
                "price": 50000,
            }
        )
def test_rate_limiting(bybit_client) -> None:
    """Тест ограничения частоты запросов"""
    # Имитация превышения лимита запросов
    with patch("ccxt.bybit") as mock_exchange:
        mock_exchange.fetch_ohlcv = AsyncMock(
            side_effect=Exception("Too many requests")
        )
        # Проверка повторных попыток
        with pytest.raises(Exception):
            bybit_client.get_ohlcv("BTC/USDT", "1h")
        # Проверка, что было сделано несколько попыток
        assert mock_exchange.fetch_ohlcv.call_count > 1
