"""
Unit тесты для BacktestManager.
Тестирует управление бэктестингом, включая выполнение бэктестов,
анализ результатов, оптимизацию параметров и генерацию отчетов.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from uuid import uuid4
from domain.types import StrategyId, PortfolioId
from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from infrastructure.core.backtest_manager import BacktestManager


class TestBacktestManager:
    """Тесты для BacktestManager."""
    
    @pytest.fixture
    def backtest_manager(self) -> BacktestManager:
        """Фикстура для BacktestManager."""
        return BacktestManager()
    
    @pytest.fixture
    def sample_historical_data(self) -> pd.DataFrame:
        """Фикстура с историческими данными."""
        dates = pd.DatetimeIndex(pd.date_range('2023-01-01', periods=1000, freq='1H'))
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.uniform(45000, 55000, 1000),
            'high': np.random.uniform(46000, 56000, 1000),
            'low': np.random.uniform(44000, 54000, 1000),
            'close': np.random.uniform(45000, 55000, 1000),
            'volume': np.random.uniform(1000000, 5000000, 1000)
        }, index=dates)
        # Создание более реалистичных данных
        data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 1000, 1000)
        data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 1000, 1000)
        return data
    
    @pytest.fixture
    def sample_strategy_config(self) -> dict:
        """Фикстура с конфигурацией стратегии."""
        return {
            "strategy_name": "Test Strategy",
            "strategy_type": "trend_following",
            "parameters": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_short": 10,
                "ma_long": 50,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "position_size": 0.1
            },
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
            "initial_capital": Decimal("10000.0"),
            "commission": Decimal("0.001")
        }
    
    def test_initialization(self, backtest_manager: BacktestManager) -> None:
        """Тест инициализации менеджера бэктестинга."""
        assert backtest_manager is not None
        assert hasattr(backtest_manager, 'active_backtests')
        assert hasattr(backtest_manager, 'results_cache')
    
    @pytest.mark.asyncio
    async def test_run_backtest(self, backtest_manager: BacktestManager, sample_strategy_config: dict) -> None:
        """Тест выполнения бэктеста."""
        # Подготовка данных для бэктеста
        strategy_id = StrategyId(uuid4())
        portfolio_id = PortfolioId(uuid4())
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 31)
        initial_balance = Money(Decimal("10000.0"), Currency.USD)
        symbols = ["BTCUSDT"]
        
        # Выполнение бэктеста
        backtest_result = await backtest_manager.run_backtest(
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            symbols=symbols,
            parameters=sample_strategy_config["parameters"]
        )
        
        # Проверки
        assert backtest_result is not None
        assert "backtest_id" in backtest_result
        assert "strategy_id" in backtest_result
        assert "portfolio_id" in backtest_result
        assert "total_return" in backtest_result
        assert "equity_curve" in backtest_result
        assert "orders" in backtest_result
        assert "trades" in backtest_result
        assert "metrics" in backtest_result
        
        # Проверка типов данных
        assert isinstance(backtest_result["backtest_id"], str)
        assert isinstance(backtest_result["strategy_id"], str)
        assert isinstance(backtest_result["portfolio_id"], str)
        assert isinstance(backtest_result["total_return"], Decimal)
        assert isinstance(backtest_result["equity_curve"], list)
        assert isinstance(backtest_result["orders"], list)
        assert isinstance(backtest_result["trades"], list)
        assert isinstance(backtest_result["metrics"], dict)
    
    def test_get_backtest_status(self, backtest_manager: BacktestManager) -> None:
        """Тест получения статуса бэктеста."""
        # Создаем тестовый backtest_id
        test_backtest_id = uuid4()
        
        # Проверяем статус несуществующего бэктеста
        status = backtest_manager.get_backtest_status(test_backtest_id)
        assert status is None
    
    def test_get_backtest_results(self, backtest_manager: BacktestManager) -> None:
        """Тест получения результатов бэктеста."""
        # Создаем тестовый backtest_id
        test_backtest_id = uuid4()
        
        # Проверяем результаты несуществующего бэктеста
        results = backtest_manager.get_backtest_results(test_backtest_id)
        assert results is None
    
    def test_stop_backtest(self, backtest_manager: BacktestManager) -> None:
        """Тест остановки бэктеста."""
        # Создаем тестовый backtest_id
        test_backtest_id = uuid4()
        
        # Пытаемся остановить несуществующий бэктест
        result = backtest_manager.stop_backtest(test_backtest_id)
        assert result is False
    
    def test_clear_backtest(self, backtest_manager: BacktestManager) -> None:
        """Тест очистки бэктеста."""
        # Создаем тестовый backtest_id
        test_backtest_id = uuid4()
        
        # Очищаем несуществующий бэктест
        result = backtest_manager.clear_backtest(test_backtest_id)
        assert result is True
    
    @pytest.mark.asyncio
    async def test_error_handling(self, backtest_manager: BacktestManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(Exception):
            await backtest_manager.run_backtest(
                strategy_id=StrategyId(uuid4()),
                portfolio_id=PortfolioId(uuid4()),
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 1),  # Одинаковые даты
                initial_balance=Money(Decimal("10000.0"), Currency.USD),
                symbols=[]  # Пустой список символов
            )
    
    @pytest.mark.asyncio
    async def test_edge_cases(self, backtest_manager: BacktestManager) -> None:
        """Тест граничных случаев."""
        # Тест с очень коротким периодом
        strategy_id = StrategyId(uuid4())
        portfolio_id = PortfolioId(uuid4())
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 2)  # Один день
        initial_balance = Money(Decimal("1000.0"), Currency.USD)
        symbols = ["BTCUSDT"]
        
        # Эти функции должны обрабатывать короткие периоды
        backtest_result = await backtest_manager.run_backtest(
            strategy_id=strategy_id,
            portfolio_id=portfolio_id,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            symbols=symbols
        )
        assert backtest_result is not None
    
    def test_cleanup(self, backtest_manager: BacktestManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        backtest_manager.clear_backtest(uuid4())
        
        # Проверка, что кэши пусты
        assert len(backtest_manager.active_backtests) == 0
        assert len(backtest_manager.results_cache) == 0 
