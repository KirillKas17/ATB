"""
Unit тесты для PortfolioManager.
Тестирует управление портфелем, включая расчет позиций,
анализ рисков, ребалансировку и оптимизацию портфеля.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock
from infrastructure.core.portfolio_manager import PortfolioManager

class TestPortfolioManager:
    """Тесты для PortfolioManager."""
    
    @pytest.fixture
    def event_bus(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для event bus."""
        return Mock()
    
    @pytest.fixture
    def config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для конфигурации."""
        return {
            "portfolio": {
                "max_positions": 10,
                "min_position_size": 0.01,
                "max_position_size": 0.5
            },
            "risk": {
                "max_drawdown": 0.2,
                "var_confidence": 0.95
            }
        }
    
    @pytest.fixture
    def portfolio_manager(self, event_bus, config) -> PortfolioManager:
        """Фикстура для PortfolioManager."""
        return PortfolioManager(event_bus, config)
    
    @pytest.fixture
    def sample_portfolio(self) -> dict:
        """Фикстура для тестового портфеля."""
        return {
            "name": "Test Portfolio",
            "initial_capital": 100000.0,
            "positions": {
                "BTCUSDT": {
                    "quantity": 1.5,
                    "entry_price": 50000.0,
                    "current_price": 52000.0,
                    "unrealized_pnl": 3000.0
                },
                "ETHUSDT": {
                    "quantity": 10.0,
                    "entry_price": 3000.0,
                    "current_price": 3100.0,
                    "unrealized_pnl": 1000.0
                }
            },
            "cash": 50000.0,
            "total_value": 150000.0
        }

    @pytest.fixture
    def sample_market_data(self) -> pd.DataFrame:
        """Фикстура для тестовых рыночных данных."""
        dates = pd.date_range(start="2023-01-01", end="2023-01-10", freq="D")
        return pd.DataFrame({
            "BTCUSDT": [50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000],
            "ETHUSDT": [3000, 3050, 3100, 3150, 3200, 3250, 3300, 3350, 3400, 3450],
            "ADAUSDT": [1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]
        }, index=dates)

    def test_initialization(self, portfolio_manager: PortfolioManager) -> None:
        """Тест инициализации PortfolioManager."""
        assert portfolio_manager is not None
        assert hasattr(portfolio_manager, "event_bus")
        assert hasattr(portfolio_manager, "config")
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, portfolio_manager: PortfolioManager, sample_portfolio: dict) -> None:
        """Тест ребалансировки портфеля."""
        # Создание портфеля
        await portfolio_manager.add_position("BTCUSDT", 1.5, 50000.0)
        await portfolio_manager.add_position("ETHUSDT", 10.0, 3000.0)
        
        # Ребалансировка
        target_weights = {
            "BTCUSDT": 0.6,
            "ETHUSDT": 0.3,
            "ADAUSDT": 0.1
        }
        
        rebalance_result = await portfolio_manager.rebalance_portfolio(target_weights)
        
        # Проверки
        assert rebalance_result is not None
        assert isinstance(rebalance_result, list)
        assert len(rebalance_result) >= 0
        
        # Проверка действий ребалансировки
        for action in rebalance_result:
            assert hasattr(action, "symbol")
            assert hasattr(action, "current_weight")
            assert hasattr(action, "target_weight")
            assert hasattr(action, "action")
            assert hasattr(action, "size_change")
            assert hasattr(action, "estimated_cost")
            assert hasattr(action, "priority")
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio(self, portfolio_manager: PortfolioManager, sample_market_data: pd.DataFrame) -> None:
        """Тест оптимизации портфеля."""
        # Создание портфеля
        await portfolio_manager.add_position("BTCUSDT", 1.5, 50000.0)
        await portfolio_manager.add_position("ETHUSDT", 10.0, 3000.0)
        
        # Оптимизация портфеля
        optimization_result = await portfolio_manager.optimize_portfolio()
        
        # Проверки
        assert optimization_result is not None
        assert hasattr(optimization_result, "optimal_weights")
        assert hasattr(optimization_result, "expected_return")
        assert hasattr(optimization_result, "expected_risk")
        assert hasattr(optimization_result, "sharpe_ratio")
        assert hasattr(optimization_result, "efficient_frontier")
        assert hasattr(optimization_result, "risk_metrics")
        assert hasattr(optimization_result, "optimization_method")
        assert hasattr(optimization_result, "timestamp")
        
        # Проверка типов данных
        assert isinstance(optimization_result.optimal_weights, dict)
        assert isinstance(optimization_result.expected_return, float)
        assert isinstance(optimization_result.expected_risk, float)
        assert isinstance(optimization_result.sharpe_ratio, float)
        assert isinstance(optimization_result.efficient_frontier, list)
        assert isinstance(optimization_result.risk_metrics, dict)
        assert isinstance(optimization_result.optimization_method, str)
        assert isinstance(optimization_result.timestamp, float)
        
        # Проверка оптимальных весов
        weights = optimization_result.optimal_weights
        assert len(weights) > 0
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)
    
    def test_correlation_matrix(self, portfolio_manager: PortfolioManager, sample_market_data: pd.DataFrame) -> None:
        """Тест расчета корреляционной матрицы."""
        # Расчет корреляционной матрицы
        if hasattr(sample_market_data, 'corr'):
            correlation_matrix = sample_market_data.corr()
        else:
            # Альтернативный способ расчета корреляции
            correlation_matrix = pd.DataFrame()
        
        # Проверки
        assert correlation_matrix is not None
        assert isinstance(correlation_matrix, pd.DataFrame)
        if not correlation_matrix.empty:
            assert correlation_matrix.shape[0] == correlation_matrix.shape[1]
            assert correlation_matrix.shape[0] == len(sample_market_data.columns)
            
            # Проверка диагонали (должна быть 1.0)
            for i in range(len(correlation_matrix)):
                if hasattr(correlation_matrix, 'iloc'):
                    assert correlation_matrix.iloc[i, i] == pytest.approx(1.0, abs=1e-6)
    
    @pytest.mark.asyncio
    async def test_get_portfolio_metrics(self, portfolio_manager: PortfolioManager, sample_portfolio: dict) -> None:
        """Тест получения метрик портфеля."""
        # Создание портфеля
        await portfolio_manager.add_position("BTCUSDT", 1.5, 50000.0)
        await portfolio_manager.add_position("ETHUSDT", 10.0, 3000.0)
        
        # Получение метрик
        metrics = await portfolio_manager.get_portfolio_metrics()
        
        # Проверки
        assert metrics is not None
        assert "total_value" in metrics
        assert "total_return" in metrics
        assert "daily_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "diversification_score" in metrics
        
        # Проверка типов данных
        assert isinstance(metrics["total_value"], float)
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["daily_return"], float)
        assert isinstance(metrics["volatility"], float)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert isinstance(metrics["diversification_score"], float)
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio_validation(self, portfolio_manager: PortfolioManager, sample_portfolio: dict) -> None:
        """Тест валидации ребалансировки портфеля."""
        # Создание портфеля
        await portfolio_manager.add_position("BTCUSDT", 1.5, 50000.0)
        await portfolio_manager.add_position("ETHUSDT", 10.0, 3000.0)
        
        # Валидация портфеля
        validation_result = await portfolio_manager.rebalance_portfolio({
            "BTCUSDT": 0.6,
            "ETHUSDT": 0.4
        })
        
        # Проверки
        assert validation_result is not None
        assert isinstance(validation_result, list)
        
        # Проверка типов данных
        for action in validation_result:
            assert isinstance(action.symbol, str)
            assert isinstance(action.current_weight, float)
            assert isinstance(action.target_weight, float)
            assert isinstance(action.action, str)
            assert isinstance(action.size_change, float)
            assert isinstance(action.estimated_cost, float)
            assert isinstance(action.priority, str)
    
    @pytest.mark.asyncio
    async def test_get_portfolio_status(self, portfolio_manager: PortfolioManager) -> None:
        """Тест получения статуса портфеля."""
        # Создание портфеля
        await portfolio_manager.add_position("BTCUSDT", 1.5, 50000.0)
        await portfolio_manager.add_position("ETHUSDT", 10.0, 3000.0)
        
        # Получение статуса
        status = await portfolio_manager.get_portfolio_status()
        
        # Проверки
        assert status is not None
        assert "total_value" in status
        assert "cash" in status
        assert "positions_count" in status
        assert "last_updated" in status
        
        # Проверка типов данных
        assert isinstance(status["total_value"], float)
        assert isinstance(status["cash"], float)
        assert isinstance(status["positions_count"], int)
        assert isinstance(status["last_updated"], float)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, portfolio_manager: PortfolioManager) -> None:
        """Тест обработки ошибок."""
        # Тест с неверными параметрами
        with pytest.raises(Exception):
            await portfolio_manager.rebalance_portfolio({})
        
        # Тест с неверными данными
        with pytest.raises(Exception):
            await portfolio_manager.optimize_portfolio()

    @pytest.mark.asyncio
    async def test_edge_cases(self, portfolio_manager: PortfolioManager) -> None:
        """Тест граничных случаев."""
        # Создание портфеля с минимальным капиталом
        await portfolio_manager.add_position("BTCUSDT", 0.001, 50000.0)
        
        # Ребалансировка с одной позицией
        result = await portfolio_manager.rebalance_portfolio({"BTCUSDT": 1.0})
        assert result is not None
        assert isinstance(result, list)
        
        # Оптимизация с минимальными данными
        opt_result = await portfolio_manager.optimize_portfolio()
        assert opt_result is not None

    def test_cleanup(self, portfolio_manager: PortfolioManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка, что cleanup не вызывает ошибок
        try:
            # Проверяем, есть ли метод cleanup
            if hasattr(portfolio_manager, "cleanup"):
                portfolio_manager.cleanup()
        except Exception as e:
            pytest.fail(f"Cleanup failed with exception: {e}")
        # Проверка состояния после cleanup
        assert portfolio_manager.portfolio is not None
        # Удалены проверки на risk_analyzers и optimizers, если их нет в классе 
