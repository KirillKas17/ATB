import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator_asyncio
from core.market_state import MarketState
from dashboard.api import app
from dashboard.dashboard import Dashboard
from fastapi.testclient import TestClient
from ml.model_selector import ModelSelector
from shared.logging import setup_logger
logger = setup_logger(__name__)
# Константы для тестов
SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
TEST_CONFIG = {
    "symbols": SYMBOLS,
    "interval": "1m",
    "max_positions": 10,
    "max_position_size": 1.0,
    "max_drawdown": 0.1,
    "stop_loss": 0.02,
    "take_profit": 0.05,
    "trailing_stop": 0.01,
    "position_timeout": 24,
}
@pytest.fixture
def mock_market_state() -> Any:
    """Фикстура с моком MarketState"""
    mock = Mock(spec=MarketState)
    mock.get_pairs = Mock(return_value=SYMBOLS)
    mock.get_pair_status = Mock(
        return_value={
            "symbol": "BTC/USDT",
            "status": "active",
            "last_update": datetime.now().isoformat(),
        }
    )
    mock.get_all_pairs_status = Mock(
        return_value={
            symbol: {
                "symbol": symbol,
                "status": "active",
                "last_update": datetime.now().isoformat(),
            }
            for symbol in SYMBOLS
        }
    )
    return mock
@pytest.fixture
def mock_model_selector() -> Any:
    """Фикстура с моком ModelSelector"""
    mock = Mock(spec=ModelSelector)
    mock.get_model = Mock(return_value=Mock())
    mock.get_model_metrics = Mock(
        return_value={"accuracy": 0.85, "precision": 0.82, "recall": 0.80, "f1": 0.81}
    )
    return mock
@pytest_asyncio.fixture
async def dashboard(mock_market_state, mock_model_selector) -> Any:
    """Фикстура с экземпляром дашборда"""
    dashboard = Dashboard()
    dashboard.market_state = mock_market_state
    dashboard.model_selector = mock_model_selector
    await dashboard.init(TEST_CONFIG)
    return dashboard
@pytest.fixture
def client(dashboard) -> Any:
    """Фикстура с тестовым клиентом FastAPI"""
    app.dependency_overrides = {"get_dashboard": lambda: dashboard}
    return TestClient(app)
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
class TestDashboard:
    def test_get_symbols(self, dashboard) -> None:
        """Тест получения списка символов"""
        symbols = dashboard.get_symbols()
        assert isinstance(symbols, list)
        assert len(symbols) == len(SYMBOLS)
        assert all(s in symbols for s in SYMBOLS)
    def test_get_status(self, dashboard) -> None:
        """Тест получения статуса"""
        status = dashboard.get_status()
        assert isinstance(status, dict)
        assert "status" in status
        assert "timestamp" in status
        assert "version" in status
    @pytest.mark.asyncio
    async def test_start_and_stop_bot(self, dashboard) -> None:
        """Тест запуска и остановки бота"""
        # Запуск бота
        await dashboard.start_bot()
        assert dashboard.is_running()
        # Остановка бота
        await dashboard.stop_bot()
        assert not dashboard.is_running()
    @pytest.mark.asyncio
    async def test_start_training(self, dashboard) -> None:
        """Тест запуска обучения"""
        with patch("dashboard.dashboard.TrainingManager") as mock_training:
            mock_training.return_value.train = AsyncMock()
            await dashboard.start_training()
            mock_training.return_value.train.assert_called_once()
    def test_get_pair_status(self, dashboard) -> None:
        """Тест получения статуса пары"""
        status = dashboard.get_pair_status("BTC/USDT")
        assert isinstance(status, dict)
        assert "symbol" in status
        assert "status" in status
        assert "last_update" in status
    def test_get_all_pairs_status(self, dashboard) -> None:
        """Тест получения статуса всех пар"""
        statuses = dashboard.get_all_pairs_status()
        assert isinstance(statuses, dict)
        assert len(statuses) == len(SYMBOLS)
        for symbol in SYMBOLS:
            assert symbol in statuses
    def test_get_correlations(self, dashboard, mock_market_data) -> None:
        """Тест получения корреляций"""
        with patch(
            "dashboard.dashboard.MarketData.get_historical_data",
            return_value=mock_market_data,
        ):
            correlations = dashboard.get_correlations()
            assert isinstance(correlations, dict)
            assert len(correlations) == len(SYMBOLS)
    def test_get_pair_correlations(self, dashboard, mock_market_data) -> None:
        """Тест получения корреляций для пары"""
        with patch(
            "dashboard.dashboard.MarketData.get_historical_data",
            return_value=mock_market_data,
        ):
            correlations = dashboard.get_pair_correlations("BTC/USDT")
            assert isinstance(correlations, dict)
            assert "BTC/USDT" in correlations
    def test_pair_status(self, dashboard) -> None:
        """Тест статуса пары"""
        status = dashboard.pair_status("BTC/USDT")
        assert isinstance(status, dict)
        assert "symbol" in status
        assert "status" in status
        assert "last_update" in status
    @pytest.mark.asyncio
    async def test_high_load_async(self, dashboard) -> None:
        """Тест асинхронной обработки высокой нагрузки"""
        async with dashboard as d:
            assert d.is_running()
            # Выполняем несколько асинхронных операций
            tasks = [
                d.get_pair_status("BTC/USDT"),
                d.get_pair_status("ETH/USDT"),
                d.get_pair_status("BNB/USDT"),
            ]
            results = await asyncio.gather(*tasks)
            assert len(results) == 3
            assert all(isinstance(r, dict) for r in results)
class TestDashboardAPI:
    """Тесты API дашборда"""
    def test_get_pairs(self, client) -> None:
        """Тест получения списка пар"""
        response = client.get("/api/pairs")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == len(SYMBOLS)
        assert all(s in data for s in SYMBOLS)
    def test_get_pair_status(self, client) -> None:
        """Тест получения статуса пары"""
        response = client.get("/api/pairs/BTC_USDT/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "symbol" in data
        assert "status" in data
        assert "last_update" in data
    def test_get_all_pairs_status(self, client) -> None:
        """Тест получения статуса всех пар"""
        response = client.get("/api/pairs/status")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == len(SYMBOLS)
        for symbol in SYMBOLS:
            assert symbol.replace("/", "_") in data
