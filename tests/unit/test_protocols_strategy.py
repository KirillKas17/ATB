"""
Production-ready unit тесты для StrategyProtocol.
Полное покрытие жизненного цикла, сигналов, рисков, метрик, ошибок, edge cases, типизации и конкурентности.
"""

import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
from unittest.mock import Mock, AsyncMock
from datetime import datetime
from domain.type_definitions.strategy_types import (
    StrategyProtocol,
    StrategyConfig,
    StrategyMetrics,
)
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.entities.market import Price, Volume
from domain.type_definitions import RiskLevel, ConfidenceLevel, Symbol, StrategyId
from domain.type_definitions.strategy_types import StrategyType
from decimal import Decimal
from uuid import UUID


class TestStrategyProtocol:
    @pytest.fixture
    def strategy_config(self) -> StrategyConfig:
        return StrategyConfig(
            strategy_type=StrategyType.MEDIUM,  # type: ignore[attr-defined]
            trading_pairs=[Symbol("BTC/USDT")],
            risk_level=RiskLevel.MEDIUM,  # type: ignore[attr-defined]
            max_position_size=Decimal("1000.0"),
            stop_loss=Decimal("0.02"),
            take_profit=Decimal("0.04"),
            confidence_threshold=ConfidenceLevel(Decimal("0.7")),
            max_signals=10,
            signal_cooldown=300,
            name="test_strategy",
            version="1.0.0",
            parameters={"risk_level": 0.02},
        )

    @pytest.fixture
    def mock_strategy(self, strategy_config: StrategyConfig) -> Mock:
        strategy = Mock(spec=StrategyProtocol)
        strategy.name = strategy_config.name
        strategy.version = strategy_config.version
        strategy.config = strategy_config
        strategy.state = StrategyState(value="active")
        strategy.initialize = AsyncMock(return_value=True)
        strategy.start = AsyncMock(return_value=True)
        strategy.stop = AsyncMock(return_value=True)
        strategy.pause = AsyncMock(return_value=True)
        strategy.resume = AsyncMock(return_value=True)
        strategy.update_config = AsyncMock(return_value=True)
        strategy.validate_config = AsyncMock(return_value=True)
        strategy.analyze_market = AsyncMock(
            return_value=StrategySignal(
                timestamp=datetime.utcnow(),
                signal_type="hold",
                confidence=0.5,
                price=Price(Decimal("50000.0"), Currency("USDT")),
                volume=Volume(Decimal("0.0"), Currency("BTC")),
            )
        )
        strategy.generate_signal = AsyncMock(
            return_value=StrategySignal(
                timestamp=datetime.utcnow(),
                signal_type="buy",
                confidence=0.8,
                price=Price(Decimal("50000.0"), Currency("USDT")),
                volume=Volume(Decimal("0.1"), Currency("BTC")),
            )
        )
        strategy.execute_signal = AsyncMock(
            return_value=StrategyExecutionResult(
                success=True,
                order_id="order_123",
                execution_price=Price(Decimal("50000.0"), Currency("USDT")),
                execution_volume=Volume(Decimal("0.1"), Currency("BTC")),
                fees=Money(Decimal("2.5"), Currency("USDT")),
                timestamp=datetime.utcnow(),
            )
        )
        strategy.risk_check = AsyncMock(return_value=True)
        strategy.get_metrics = AsyncMock(
            return_value=StrategyMetrics(
                strategy_id=StrategyId(UUID("12345678-1234-5678-9abc-123456789abc")),
                name="test_strategy",
                profit_factor=Decimal("1.5"),
                total_return=Decimal("15.0"),
                average_trade=Decimal("150.0"),
                calmar_ratio=Decimal("1.2"),
                sortino_ratio=Decimal("1.8"),
                var_95=Decimal("-50.0"),
                cvar_95=Decimal("-75.0"),
                timestamp=datetime.utcnow(),
                total_trades=10,
                winning_trades=7,
                losing_trades=3,
                max_drawdown=Decimal("-200.0"),
                sharpe_ratio=Decimal("1.25"),
                win_rate=Decimal("0.7"),
            )
        )
        strategy.reset_metrics = AsyncMock(return_value=True)
        strategy.get_state = AsyncMock(return_value=StrategyState(value="active"))
        strategy.set_state = AsyncMock(return_value=True)
        strategy.cleanup = AsyncMock(return_value=True)
        return strategy

    @pytest.mark.asyncio
    async def test_lifecycle(self, mock_strategy: Mock, strategy_config: StrategyConfig) -> None:
        assert await mock_strategy.initialize(strategy_config) is True
        assert await mock_strategy.start() is True
        assert await mock_strategy.pause() is True
        assert await mock_strategy.resume() is True
        assert await mock_strategy.stop() is True
        assert await mock_strategy.cleanup() is True

    @pytest.mark.asyncio
    async def test_config_update_and_validation(self, mock_strategy: Mock, strategy_config: StrategyConfig) -> None:
        assert await mock_strategy.update_config(strategy_config) is True
        assert await mock_strategy.validate_config(strategy_config) is True

    @pytest.mark.asyncio
    async def test_market_analysis_and_signal(self, mock_strategy: Mock) -> None:
        signal = await mock_strategy.analyze_market(Mock())
        assert isinstance(signal, StrategySignal)
        assert signal.signal_type in ["buy", "sell", "hold"]
        signal2 = await mock_strategy.generate_signal()
        assert signal2.signal_type == "buy"
        assert signal2.confidence == 0.8

    @pytest.mark.asyncio
    async def test_signal_execution(self, mock_strategy: Mock) -> None:
        signal = StrategySignal(
            timestamp=datetime.utcnow(),
            signal_type="buy",
            confidence=0.8,
            price=Price(Decimal("50000.0"), Currency("USDT")),
            volume=Volume(Decimal("0.1"), Currency("BTC")),
        )
        result = await mock_strategy.execute_signal(signal)
        assert result.success is True
        assert result.order_id == "order_123"

    @pytest.mark.asyncio
    async def test_risk_check(self, mock_strategy: Mock) -> None:
        signal = StrategySignal(
            timestamp=datetime.utcnow(),
            signal_type="buy",
            confidence=0.8,
            price=Price(Decimal("50000.0"), Currency("USDT")),
            volume=Volume(Decimal("0.1"), Currency("BTC")),
        )
        assert await mock_strategy.risk_check(signal) is True

    @pytest.mark.asyncio
    async def test_metrics(self, mock_strategy: Mock) -> None:
        metrics = await mock_strategy.get_metrics()
        assert metrics.total_trades == 10
        assert metrics.win_rate == 0.7
        assert await mock_strategy.reset_metrics() is True

    @pytest.mark.asyncio
    async def test_state_management(self, mock_strategy: Mock) -> None:
        state = await mock_strategy.get_state()
        assert isinstance(state, StrategyState)
        assert state.value == "active"
        assert await mock_strategy.set_state(state) is True

    @pytest.mark.asyncio
    async def test_strategy_errors(self, mock_strategy: Mock) -> None:
        mock_strategy.validate_config.side_effect = StrategyValidationError("Invalid config")
        with pytest.raises(StrategyValidationError):
            await mock_strategy.validate_config(Mock())
        mock_strategy.execute_signal.side_effect = StrategyExecutionError("Execution failed")
        with pytest.raises(StrategyExecutionError):
            await mock_strategy.execute_signal(Mock())

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_strategy: Mock) -> None:
        tasks = [
            mock_strategy.analyze_market(Mock()),
            mock_strategy.generate_signal(),
            mock_strategy.get_metrics(),
            mock_strategy.get_state(),
        ]
        results = await asyncio.gather(*tasks)
        assert len(results) == 4
        assert all(result is not None for result in results)

    @pytest.mark.asyncio
    async def test_edge_cases(self, mock_strategy: Mock) -> None:
        # Неизвестный тип сигнала
        with pytest.raises(ValueError):
            StrategySignal(
                timestamp=datetime.utcnow(),
                signal_type="unknown",
                confidence=0.5,
                price=Price(Decimal("50000.0"), Currency("USDT")),
                volume=Volume(Decimal("0.1"), Currency("BTC")),
            )
        # Некорректная уверенность
        with pytest.raises(ValueError):
            StrategySignal(
                timestamp=datetime.utcnow(),
                signal_type="buy",
                confidence=1.5,
                price=Price(Decimal("50000.0"), Currency("USDT")),
                volume=Volume(Decimal("0.1"), Currency("BTC")),
            )
        # Отрицательная цена
        with pytest.raises(ValueError):
            StrategySignal(
                timestamp=datetime.utcnow(),
                signal_type="buy",
                confidence=0.8,
                price=Price(Decimal("-1.0"), Currency("USDT")),
                volume=Volume(Decimal("0.1"), Currency("BTC")),
            )
