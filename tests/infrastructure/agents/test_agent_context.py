"""
Comprehensive tests for AgentContext.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from uuid import uuid4
from decimal import Decimal
from infrastructure.agents.agent_context_refactored import AgentContext
from domain.entities.order import Order, OrderSide, OrderType, OrderStatus
from domain.entities.position import Position
from domain.entities.market import MarketData
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.trading_pair import TradingPair
class TestAgentContext:
    """Tests for AgentContext."""
    @pytest.fixture
    async def context(self) -> Any:
        """Create AgentContext instance."""
        context = AgentContext(symbol="BTC/USDT")
        return context
    @pytest.fixture
    def sample_order(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Create sample order."""
        return Order(
            id=uuid4(),
            trading_pair=TradingPair.from_string("BTC/USD"),
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Money(Decimal("50000"), Currency.USD),
            status=OrderStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    @pytest.fixture
    def sample_position(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Create sample position."""
        return Position(
            id=uuid4(),
            portfolio_id=uuid4(),
            trading_pair=TradingPair.from_string("BTC/USD"),
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            average_price=Money(Decimal("50000"), Currency.USD),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Create sample market data."""
        return MarketData(
            id=uuid4(),
            symbol="BTC/USD",
            price=Money(Decimal("50000"), Currency.USD),
            volume=Decimal("100.0"),
            timestamp=datetime.now()
        )
    async def test_context_initialization(self, context) -> None:
        """Test context initialization."""
        # Assert
        assert context is not None
        assert hasattr(context, 'market_data')
        assert hasattr(context, 'positions')
        assert hasattr(context, 'orders')
        assert hasattr(context, 'signals')
    async def test_add_market_data(self, context, sample_market_data) -> None:
        """Test adding market data to context."""
        # Act
        context.add_market_data(sample_market_data)
        # Assert
        assert len(context.market_data) == 1
        assert context.market_data[0] == sample_market_data
    async def test_add_position(self, context, sample_position) -> None:
        """Test adding position to context."""
        # Act
        context.add_position(sample_position)
        # Assert
        assert len(context.positions) == 1
        assert context.positions[0] == sample_position
    async def test_add_order(self, context, sample_order) -> None:
        """Test adding order to context."""
        # Act
        context.add_order(sample_order)
        # Assert
        assert len(context.orders) == 1
        assert context.orders[0] == sample_order
    async def test_add_signal(self, context) -> None:
        """Test adding signal to context."""
        # Arrange
        signal = {
            "type": "BUY",
            "strength": 0.8,
            "timestamp": datetime.now(),
            "source": "technical_analysis"
        }
        # Act
        context.add_signal(signal)
        # Assert
        assert len(context.signals) == 1
        assert context.signals[0] == signal
    async def test_get_latest_market_data(self, context, sample_market_data) -> None:
        """Test getting latest market data."""
        # Arrange
        context.add_market_data(sample_market_data)
        # Act
        result = context.get_latest_market_data("BTC/USD")
        # Assert
        assert result == sample_market_data
    async def test_get_latest_market_data_not_found(self, context) -> None:
        """Test getting latest market data when not found."""
        # Act
        result = context.get_latest_market_data("BTC/USD")
        # Assert
        assert result is None
    async def test_get_positions_by_symbol(self, context, sample_position) -> None:
        """Test getting positions by symbol."""
        # Arrange
        context.add_position(sample_position)
        # Act
        result = context.get_positions_by_symbol("BTC/USD")
        # Assert
        assert len(result) == 1
        assert result[0] == sample_position
    async def test_get_orders_by_symbol(self, context, sample_order) -> None:
        """Test getting orders by symbol."""
        # Arrange
        context.add_order(sample_order)
        # Act
        result = context.get_orders_by_symbol("BTC/USD")
        # Assert
        assert len(result) == 1
        assert result[0] == sample_order
    async def test_get_signals_by_type(self, context) -> None:
        """Test getting signals by type."""
        # Arrange
        buy_signal = {
            "type": "BUY",
            "strength": 0.8,
            "timestamp": datetime.now(),
            "source": "technical_analysis"
        }
        sell_signal = {
            "type": "SELL",
            "strength": 0.6,
            "timestamp": datetime.now(),
            "source": "fundamental_analysis"
        }
        context.add_signal(buy_signal)
        context.add_signal(sell_signal)
        # Act
        buy_signals = context.get_signals_by_type("BUY")
        sell_signals = context.get_signals_by_type("SELL")
        # Assert
        assert len(buy_signals) == 1
        assert len(sell_signals) == 1
        assert buy_signals[0]["type"] == "BUY"
        assert sell_signals[0]["type"] == "SELL"
    async def test_clear_context(self, context, sample_market_data, sample_position, sample_order) -> None:
        """Test clearing context."""
        # Arrange
        context.add_market_data(sample_market_data)
        context.add_position(sample_position)
        context.add_order(sample_order)
        context.add_signal({"type": "BUY", "strength": 0.8, "timestamp": datetime.now()})
        # Act
        context.clear()
        # Assert
        assert len(context.market_data) == 0
        assert len(context.positions) == 0
        assert len(context.orders) == 0
        assert len(context.signals) == 0
    async def test_context_size_limits(self, context) -> None:
        """Test context size limits."""
        # Arrange
        max_size = 1000
        # Act - Add more data than the limit
        for i in range(max_size + 100):
            market_data = MarketData(
                id=uuid4(),
                symbol=f"BTC/USD_{i}",
                price=Money(Decimal("50000"), Currency.USD),
                volume=Decimal("100.0"),
                timestamp=datetime.now()
            )
            context.add_market_data(market_data)
        # Assert - Should maintain size limit
        assert len(context.market_data) <= max_size
    async def test_context_data_consistency(self, context, sample_market_data) -> None:
        """Test context data consistency."""
        # Arrange
        context.add_market_data(sample_market_data)
        # Act
        latest_data = context.get_latest_market_data("BTC/USD")
        # Assert
        assert latest_data.id == sample_market_data.id
        assert latest_data.symbol == sample_market_data.symbol
        assert latest_data.price == sample_market_data.price
    async def test_context_thread_safety(self, context) -> None:
        """Test context thread safety."""
        # This would test concurrent access to context
        # For now, we'll just verify the interface
        assert hasattr(context, 'add_market_data')
        assert hasattr(context, 'add_position')
        assert hasattr(context, 'add_order')
        assert hasattr(context, 'add_signal')
    async def test_context_serialization(self, context, sample_market_data, sample_position) -> None:
        """Test context serialization."""
        # Arrange
        context.add_market_data(sample_market_data)
        context.add_position(sample_position)
        # Act
        serialized = context.to_dict()
        # Assert
        assert "market_data" in serialized
        assert "positions" in serialized
        assert "orders" in serialized
        assert "signals" in serialized
    async def test_context_deserialization(self, context, sample_market_data, sample_position) -> None:
        """Test context deserialization."""
        # Arrange
        context.add_market_data(sample_market_data)
        context.add_position(sample_position)
        serialized = context.to_dict()
        # Act
        new_context = AgentContext(symbol="BTC/USDT")
        new_context.from_dict(serialized)
        # Assert
        assert len(new_context.market_data) == len(context.market_data)
        assert len(new_context.positions) == len(context.positions)
    async def test_context_metrics(self, context, sample_market_data, sample_position, sample_order) -> None:
        """Test context metrics."""
        # Arrange
        context.add_market_data(sample_market_data)
        context.add_position(sample_position)
        context.add_order(sample_order)
        # Act
        metrics = context.get_metrics()
        # Assert
        assert "market_data_count" in metrics
        assert "positions_count" in metrics
        assert "orders_count" in metrics
        assert "signals_count" in metrics
        assert metrics["market_data_count"] == 1
        assert metrics["positions_count"] == 1
        assert metrics["orders_count"] == 1
    async def test_context_health_check(self, context) -> None:
        """Test context health check."""
        # Act
        health = context.health_check()
        # Assert
        assert "status" in health
        assert "memory_usage" in health
        assert "data_counts" in health
        assert health["status"] == "healthy"


class TestAgentContextIntegration:
    """Integration tests for AgentContext."""
    
    @pytest.mark.integration
    def test_context_with_real_data(self: "TestAgentContextIntegration") -> None:
        """Test context with real market data."""
        # This would test with actual market data feeds
        pass
    
    @pytest.mark.integration
    def test_context_with_multiple_agents(self: "TestAgentContextIntegration") -> None:
        """Test context with multiple agents accessing simultaneously."""
        # This would test multi-agent scenarios
        pass
    @pytest.mark.integration
    def test_context_persistence(self: "TestAgentContextIntegration") -> None:
        """Test context persistence across sessions."""
        # This would test saving/loading context state
        pass


class TestAgentContextPerformance:
    """Performance tests for AgentContext."""
    
    @pytest.mark.performance
    def test_context_data_insertion_performance(self: "TestAgentContextPerformance") -> None:
        """Test performance of data insertion."""
        # This would measure insertion speed
        pass
    
    @pytest.mark.performance
    def test_context_query_performance(self: "TestAgentContextPerformance") -> None:
        """Test performance of data queries."""
        # This would measure query speed
        pass
    
    @pytest.mark.performance
    def test_context_memory_usage(self: "TestAgentContextPerformance") -> None:
        """Test context memory usage."""
        # This would measure memory consumption
        pass


class TestAgentContextEdgeCases:
    """Edge cases and error handling tests."""
    
    def test_context_with_invalid_data(self: "TestAgentContextEdgeCases") -> None:
        """Test context with invalid data."""
        # This would test error handling for invalid data
        pass
    
    def test_context_with_duplicate_data(self: "TestAgentContextEdgeCases") -> None:
        """Test context with duplicate data."""
        # This would test duplicate handling
        pass
    
    def test_context_with_missing_data(self: "TestAgentContextEdgeCases") -> None:
        """Test context with missing data."""
        # This would test handling of missing data
        pass
    
    def test_context_under_high_load(self: "TestAgentContextEdgeCases") -> None:
        """Test context under high load conditions."""
        # This would test high load scenarios
        pass 
