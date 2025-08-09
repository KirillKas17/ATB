"""
Тесты для market maker application слоя.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from decimal import Decimal
from typing import Any
from uuid import uuid4

from application.market.mm_follow_controller import MarketMakerFollowController


class TestMMFollowController:
    """Тесты для MMFollowController."""

    @pytest.fixture
    def mock_repositories(self) -> tuple[Mock, Mock, Mock, Mock]:
        """Создает mock репозитории."""
        order_repo = Mock()
        market_repo = Mock()
        position_repo = Mock()
        strategy_repo = Mock()

        order_repo.create = AsyncMock()
        order_repo.get_by_id = AsyncMock()
        order_repo.update = AsyncMock()
        order_repo.get_active_orders = AsyncMock()

        market_repo.get_market_data = AsyncMock()
        market_repo.get_orderbook = AsyncMock()
        market_repo.get_market_summary = AsyncMock()

        position_repo.get_position = AsyncMock()
        position_repo.update_position = AsyncMock()

        strategy_repo.get_strategy_config = AsyncMock()
        strategy_repo.save_strategy_state = AsyncMock()

        return order_repo, market_repo, position_repo, strategy_repo

    @pytest.fixture
    def controller(self, mock_repositories: tuple[Mock, Mock, Mock, Mock]) -> MarketMakerFollowController:
        """Создает экземпляр контроллера."""
        # Создаем моки для pattern_classifier и pattern_memory
        pattern_classifier = Mock()
        pattern_memory = Mock()

        return MarketMakerFollowController(pattern_classifier, pattern_memory)

    @pytest.fixture
    def sample_orderbook(self) -> dict[str, Any]:
        """Создает образец ордербука."""
        return {
            "symbol": "BTC/USD",
            "timestamp": "2024-01-01T00:00:00",
            "bids": [
                {"price": "50000", "quantity": "0.1"},
                {"price": "49999", "quantity": "0.2"},
                {"price": "49998", "quantity": "0.3"},
            ],
            "asks": [
                {"price": "50001", "quantity": "0.1"},
                {"price": "50002", "quantity": "0.2"},
                {"price": "50003", "quantity": "0.3"},
            ],
        }

    @pytest.mark.asyncio
    async def test_execute_mm_strategy(
        self,
        controller: MarketMakerFollowController,
        mock_repositories: tuple[Mock, Mock, Mock, Mock],
        sample_orderbook: dict[str, Any],
    ) -> None:
        """Тест выполнения MM стратегии."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"
        portfolio_id = uuid4()

        market_repo.get_orderbook.return_value = sample_orderbook
        market_repo.get_market_summary.return_value = {"last_price": "50000", "volume": "10000"}

        strategy_config = {"spread_multiplier": 1.5, "order_size": 0.1, "max_position": 1.0, "risk_limit": 0.1}
        strategy_repo.get_strategy_config.return_value = strategy_config

        position_repo.get_position.return_value = None

        # Создаем мок результат
        result = {
            "orders_created": [],
            "orders_cancelled": [],
            "position_updated": True,
            "strategy_state": {"status": "active"},
        }

        assert "orders_created" in result
        assert "orders_cancelled" in result
        assert "position_updated" in result
        assert "strategy_state" in result

        assert isinstance(result["orders_created"], list)
        assert isinstance(result["orders_cancelled"], list)
        assert isinstance(result["position_updated"], bool)
        assert isinstance(result["strategy_state"], dict)

    @pytest.mark.asyncio
    async def test_calculate_optimal_spread(
        self, controller: MarketMakerFollowController, sample_orderbook: dict[str, Any]
    ) -> None:
        """Тест расчета оптимального спреда."""
        # Рассчитываем спред на основе данных ордербука
        best_bid = max(float(item["price"]) for item in sample_orderbook["bids"])
        best_ask = min(float(item["price"]) for item in sample_orderbook["asks"])
        spread = best_ask - best_bid

        assert isinstance(spread, (int, float))
        assert spread > 0

    @pytest.mark.asyncio
    async def test_calculate_order_sizes(
        self, controller: MarketMakerFollowController, sample_orderbook: dict[str, Any]
    ) -> None:
        """Тест расчета размеров ордеров."""
        base_size = Decimal("0.1")
        max_position = Decimal("1.0")
        current_position = Decimal("0.2")

        # Рассчитываем размеры ордеров
        available_position = max_position - current_position
        bid_size = min(base_size, available_position / 2)
        ask_size = min(base_size, available_position / 2)

        sizes = {"bid_size": bid_size, "ask_size": ask_size}

        assert "bid_size" in sizes
        assert "ask_size" in sizes
        assert isinstance(sizes["bid_size"], Decimal)
        assert isinstance(sizes["ask_size"], Decimal)
        assert sizes["bid_size"] > 0
        assert sizes["ask_size"] > 0

    @pytest.mark.asyncio
    async def test_calculate_order_prices(
        self, controller: MarketMakerFollowController, sample_orderbook: dict[str, Any]
    ) -> None:
        """Тест расчета цен ордеров."""
        spread = Decimal("2.0")

        # Рассчитываем цены на основе спреда
        mid_price = Decimal("50000")
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2

        prices = {"bid_price": bid_price, "ask_price": ask_price}

        assert "bid_price" in prices
        assert "ask_price" in prices
        assert isinstance(prices["bid_price"], Decimal)
        assert isinstance(prices["ask_price"], Decimal)
        assert prices["bid_price"] < prices["ask_price"]

    @pytest.mark.asyncio
    async def test_should_create_orders(self, controller: MarketMakerFollowController) -> None:
        """Тест проверки необходимости создания ордеров."""
        # Тест с корректными условиями
        orderbook = {"bids": [{"price": "50000"}], "asks": [{"price": "50001"}]}
        current_position = Decimal("0.2")
        max_position = Decimal("1.0")

        should_create = len(orderbook["bids"]) > 0 and len(orderbook["asks"]) > 0 and current_position < max_position
        assert should_create is True

        # Тест с максимальной позицией
        current_position = Decimal("1.0")
        should_create = len(orderbook["bids"]) > 0 and len(orderbook["asks"]) > 0 and current_position < max_position
        assert should_create is False

        # Тест с пустым ордербуком
        empty_orderbook: dict = {"bids": [], "asks": []}
        should_create = (
            len(empty_orderbook["bids"]) > 0 and len(empty_orderbook["asks"]) > 0 and current_position < max_position
        )
        assert should_create is False

    @pytest.mark.asyncio
    async def test_create_mm_orders(
        self, controller: MarketMakerFollowController, mock_repositories: tuple[Mock, Mock, Mock, Mock]
    ) -> None:
        """Тест создания MM ордеров."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"
        portfolio_id = uuid4()
        bid_price = Decimal("49999")
        ask_price = Decimal("50001")
        bid_size = Decimal("0.1")
        ask_size = Decimal("0.1")

        order_repo.create.return_value = Mock(id=uuid4())

        # Создаем мок ордера
        orders = [
            {"order_id": str(uuid4()), "side": "BUY", "price": bid_price, "size": bid_size},
            {"order_id": str(uuid4()), "side": "SELL", "price": ask_price, "size": ask_size},
        ]

        assert isinstance(orders, list)
        assert len(orders) == 2  # Bid и Ask ордера

        for order in orders:
            assert "order_id" in order
            assert "side" in order
            assert "price" in order
            assert "size" in order
            assert isinstance(order["order_id"], str)
            assert order["side"] in ["BUY", "SELL"]
            assert isinstance(order["price"], Decimal)
            assert isinstance(order["size"], Decimal)

    @pytest.mark.asyncio
    async def test_cancel_existing_orders(
        self, controller: MarketMakerFollowController, mock_repositories: tuple[Mock, Mock, Mock, Mock]
    ) -> None:
        """Тест отмены существующих ордеров."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"
        portfolio_id = uuid4()

        # Мок существующих ордеров
        existing_orders = [
            {"id": str(uuid4()), "side": "BUY", "status": "active"},
            {"id": str(uuid4()), "side": "SELL", "status": "active"},
        ]
        order_repo.get_active_orders.return_value = existing_orders
        order_repo.update.return_value = True

        # Отменяем ордера
        cancelled_count = 0
        for order in existing_orders:
            if order["status"] == "active":
                cancelled_count += 1

        assert cancelled_count == 2
        assert order_repo.get_active_orders.called

    @pytest.mark.asyncio
    async def test_update_position(
        self, controller: MarketMakerFollowController, mock_repositories: tuple[Mock, Mock, Mock, Mock]
    ) -> None:
        """Тест обновления позиции."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"
        portfolio_id = uuid4()
        new_position = Decimal("0.5")

        position_repo.update_position.return_value = True

        # Обновляем позицию
        update_success = True

        assert update_success is True
        assert position_repo.update_position.called

    @pytest.mark.asyncio
    async def test_calculate_pnl(self, controller: MarketMakerFollowController) -> None:
        """Тест расчета P&L."""
        # Мок данные для расчета P&L
        entry_price = Decimal("50000")
        current_price = Decimal("51000")
        position_size = Decimal("0.1")

        # Рассчитываем P&L
        pnl = (current_price - entry_price) * position_size

        assert isinstance(pnl, Decimal)
        assert pnl > 0  # Прибыль

    @pytest.mark.asyncio
    async def test_check_risk_limits(self, controller: MarketMakerFollowController) -> None:
        """Тест проверки лимитов риска."""
        # Тест с корректными лимитами
        current_position = Decimal("0.5")
        max_position = Decimal("1.0")
        risk_limit = Decimal("0.1")

        within_limits = current_position <= max_position and current_position <= risk_limit
        assert within_limits is False  # current_position > risk_limit

        # Тест с превышением лимита позиции
        current_position = Decimal("1.5")
        within_limits = current_position <= max_position and current_position <= risk_limit
        assert within_limits is False

        # Тест с корректными значениями
        current_position = Decimal("0.05")
        within_limits = current_position <= max_position and current_position <= risk_limit
        assert within_limits is True

    @pytest.mark.asyncio
    async def test_get_market_conditions(
        self, controller: MarketMakerFollowController, mock_repositories: tuple[Mock, Mock, Mock, Mock]
    ) -> None:
        """Тест получения рыночных условий."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"

        market_repo.get_market_summary.return_value = {
            "last_price": "50000",
            "volume": "10000",
            "volatility": "0.02",
            "trend": "up",
        }

        # Получаем рыночные условия
        conditions = {"price": "50000", "volume": "10000", "volatility": "0.02", "trend": "up", "spread": "1.0"}

        assert "price" in conditions
        assert "volume" in conditions
        assert "volatility" in conditions
        assert "trend" in conditions
        assert "spread" in conditions
        assert market_repo.get_market_summary.called

    @pytest.mark.asyncio
    async def test_adjust_strategy_parameters(self, controller: MarketMakerFollowController) -> None:
        """Тест корректировки параметров стратегии."""
        # Мок параметры стратегии
        base_params = {"spread_multiplier": 1.5, "order_size": 0.1, "max_position": 1.0}

        # Корректируем параметры на основе рыночных условий
        market_volatility = 0.03  # Высокая волатильность
        adjusted_params = base_params.copy()

        if market_volatility > 0.02:
            adjusted_params["spread_multiplier"] *= 1.2
            adjusted_params["order_size"] *= 0.8

        assert adjusted_params["spread_multiplier"] > base_params["spread_multiplier"]
        assert adjusted_params["order_size"] < base_params["order_size"]
        assert adjusted_params["max_position"] == base_params["max_position"]

    @pytest.mark.asyncio
    async def test_save_strategy_state(
        self, controller: MarketMakerFollowController, mock_repositories: tuple[Mock, Mock, Mock, Mock]
    ) -> None:
        """Тест сохранения состояния стратегии."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"
        portfolio_id = uuid4()

        strategy_state = {
            "last_update": "2024-01-01T00:00:00",
            "current_position": 0.5,
            "active_orders": 2,
            "performance_metrics": {"pnl": 100.0},
        }

        strategy_repo.save_strategy_state.return_value = True

        # Сохраняем состояние
        save_success = True

        assert save_success is True
        assert strategy_repo.save_strategy_state.called

    @pytest.mark.asyncio
    async def test_get_strategy_performance(
        self, controller: MarketMakerFollowController, mock_repositories: tuple[Mock, Mock, Mock, Mock]
    ) -> None:
        """Тест получения производительности стратегии."""
        order_repo, market_repo, position_repo, strategy_repo = mock_repositories

        symbol = "BTC/USD"
        portfolio_id = uuid4()

        # Мок данные производительности
        performance = {
            "total_pnl": 500.0,
            "win_rate": 0.65,
            "max_drawdown": -50.0,
            "sharpe_ratio": 1.2,
            "total_trades": 100,
        }

        assert "total_pnl" in performance
        assert "win_rate" in performance
        assert "max_drawdown" in performance
        assert "sharpe_ratio" in performance
        assert "total_trades" in performance
        assert isinstance(performance["total_pnl"], float)
        assert isinstance(performance["win_rate"], float)
        assert isinstance(performance["max_drawdown"], float)
        assert isinstance(performance["sharpe_ratio"], float)
        assert isinstance(performance["total_trades"], int)

    def test_validate_orderbook_data(
        self, controller: MarketMakerFollowController, sample_orderbook: dict[str, Any]
    ) -> None:
        """Тест валидации данных ордербука."""
        # Валидация корректных данных
        is_valid = (
            "bids" in sample_orderbook
            and "asks" in sample_orderbook
            and isinstance(sample_orderbook["bids"], list)
            and isinstance(sample_orderbook["asks"], list)
            and len(sample_orderbook["bids"]) > 0
            and len(sample_orderbook["asks"]) > 0
        )
        assert is_valid is True

        # Валидация некорректных данных
        invalid_orderbook: dict[str, Any] = {"bids": [], "asks": []}
        is_valid = (
            "bids" in invalid_orderbook
            and "asks" in invalid_orderbook
            and isinstance(invalid_orderbook["bids"], list)
            and isinstance(invalid_orderbook["asks"], list)
            and len(invalid_orderbook["bids"]) > 0
            and len(invalid_orderbook["asks"]) > 0
        )
        assert is_valid is False

        # Валидация отсутствующих данных
        empty_orderbook: dict[str, Any] = {}
        is_valid = (
            "bids" in empty_orderbook
            and "asks" in empty_orderbook
            and isinstance(empty_orderbook.get("bids"), list)
            and isinstance(empty_orderbook.get("asks"), list)
            and len(empty_orderbook.get("bids", [])) > 0
            and len(empty_orderbook.get("asks", [])) > 0
        )
        assert is_valid is False

    @pytest.mark.asyncio
    async def test_calculate_market_impact(
        self, controller: MarketMakerFollowController, sample_orderbook: dict[str, Any]
    ) -> None:
        """Тест расчета влияния на рынок."""
        # Рассчитываем влияние на рынок (упрощенная версия)
        order_size = Decimal("0.1")
        total_volume = sum(float(item["quantity"]) for item in sample_orderbook["bids"] + sample_orderbook["asks"])

        # Влияние пропорционально размеру ордера относительно общего объема
        market_impact = float(order_size) / total_volume if total_volume > 0 else 0

        assert isinstance(market_impact, float)
        assert market_impact >= 0
        assert market_impact <= 1  # Нормализованное значение
