"""
Тесты для доменного сервиса стратегий.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.strategy_service import StrategyService, IStrategyService
from domain.type_definitions.ml_types import StrategyResult, StrategyPerformance, StrategyConfig
class TestStrategyService:
    """Тесты для сервиса стратегий."""
    @pytest.fixture
    def strategy_service(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура сервиса стратегий."""
        return StrategyService()
    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с примерными рыночными данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 10, 100),
            'bollinger_upper': np.random.uniform(51000, 52000, 100),
            'bollinger_lower': np.random.uniform(49000, 50000, 100)
        }, index=dates)
    @pytest.fixture
    def sample_strategy_config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с конфигурацией стратегии."""
        return {
            "strategy_type": "trend_following",
            "entry_threshold": 0.7,
            "exit_threshold": 0.3,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "position_size": 0.1,
            "max_positions": 3
        }
    def test_strategy_service_initialization(self, strategy_service) -> None:
        """Тест инициализации сервиса."""
        assert strategy_service is not None
        assert isinstance(strategy_service, IStrategyService)
        assert hasattr(strategy_service, 'config')
        assert isinstance(strategy_service.config, dict)
    def test_strategy_service_config_defaults(self, strategy_service) -> None:
        """Тест конфигурации по умолчанию."""
        config = strategy_service.config
        assert "default_strategy" in config
        assert "risk_management" in config
        assert "position_sizing" in config
        assert "backtest_settings" in config
        assert isinstance(config["default_strategy"], str)
        assert isinstance(config["risk_management"], dict)
    def test_execute_strategy_valid_data(self, strategy_service, sample_market_data, sample_strategy_config) -> None:
        """Тест выполнения стратегии с валидными данными."""
        result = strategy_service.execute_strategy(sample_market_data, sample_strategy_config)
        assert isinstance(result, dict)
        assert "strategy_name" in result
        assert "signals" in result
        assert "performance" in result
        assert "positions" in result
        assert "execution_time" in result
        assert isinstance(result["strategy_name"], str)
        assert isinstance(result["signals"], list)
        assert isinstance(result["performance"], dict)
        assert isinstance(result["positions"], list)
        assert isinstance(result["execution_time"], float)
        assert result["strategy_name"] == "trend_following"
        assert result["execution_time"] >= 0.0
    def test_execute_strategy_empty_data(self, strategy_service, sample_strategy_config) -> None:
        """Тест выполнения стратегии с пустыми данными."""
        empty_data = pd.DataFrame()
        result = strategy_service.execute_strategy(empty_data, sample_strategy_config)
        assert isinstance(result, dict)
        assert result["strategy_name"] == "trend_following"
        assert len(result["signals"]) == 0
        assert len(result["positions"]) == 0
    def test_execute_strategy_empty_config(self, strategy_service, sample_market_data) -> None:
        """Тест выполнения стратегии с пустой конфигурацией."""
        empty_config = {}
        result = strategy_service.execute_strategy(sample_market_data, empty_config)
        assert isinstance(result, dict)
        assert "strategy_name" in result
        assert "signals" in result
        assert "performance" in result
    def test_backtest_strategy(self, strategy_service, sample_market_data, sample_strategy_config) -> None:
        """Тест бэктестинга стратегии."""
        backtest_result = strategy_service.backtest_strategy(sample_market_data, sample_strategy_config)
        assert isinstance(backtest_result, dict)
        assert "total_return" in backtest_result
        assert "sharpe_ratio" in backtest_result
        assert "max_drawdown" in backtest_result
        assert "win_rate" in backtest_result
        assert "trades" in backtest_result
        assert "equity_curve" in backtest_result
        assert isinstance(backtest_result["total_return"], float)
        assert isinstance(backtest_result["sharpe_ratio"], float)
        assert isinstance(backtest_result["max_drawdown"], float)
        assert isinstance(backtest_result["win_rate"], float)
        assert isinstance(backtest_result["trades"], list)
        assert isinstance(backtest_result["equity_curve"], pd.Series)
        assert backtest_result["win_rate"] >= 0.0 and backtest_result["win_rate"] <= 1.0
        assert backtest_result["max_drawdown"] <= 0.0
    def test_backtest_strategy_empty_data(self, strategy_service, sample_strategy_config) -> None:
        """Тест бэктестинга стратегии с пустыми данными."""
        empty_data = pd.DataFrame()
        backtest_result = strategy_service.backtest_strategy(empty_data, sample_strategy_config)
        assert isinstance(backtest_result, dict)
        assert backtest_result["total_return"] == 0.0
        assert backtest_result["sharpe_ratio"] == 0.0
        assert backtest_result["max_drawdown"] == 0.0
        assert backtest_result["win_rate"] == 0.0
        assert len(backtest_result["trades"]) == 0
    def test_optimize_strategy_parameters(self, strategy_service, sample_market_data, sample_strategy_config) -> None:
        """Тест оптимизации параметров стратегии."""
        optimization_result = strategy_service.optimize_strategy_parameters(sample_market_data, sample_strategy_config)
        assert isinstance(optimization_result, dict)
        assert "best_params" in optimization_result
        assert "best_performance" in optimization_result
        assert "optimization_history" in optimization_result
        assert "optimization_time" in optimization_result
        assert isinstance(optimization_result["best_params"], dict)
        assert isinstance(optimization_result["best_performance"], float)
        assert isinstance(optimization_result["optimization_history"], list)
        assert isinstance(optimization_result["optimization_time"], float)
        assert optimization_result["best_performance"] >= 0.0
        assert optimization_result["optimization_time"] >= 0.0
    def test_compare_strategies(self, strategy_service, sample_market_data) -> None:
        """Тест сравнения стратегий."""
        strategies = [
            {"strategy_type": "trend_following", "entry_threshold": 0.7},
            {"strategy_type": "mean_reversion", "entry_threshold": 0.3},
            {"strategy_type": "momentum", "entry_threshold": 0.8}
        ]
        comparison_result = strategy_service.compare_strategies(sample_market_data, strategies)
        assert isinstance(comparison_result, dict)
        assert "strategies" in comparison_result
        assert "performance_comparison" in comparison_result
        assert "best_strategy" in comparison_result
        assert isinstance(comparison_result["strategies"], list)
        assert isinstance(comparison_result["performance_comparison"], pd.DataFrame)
        assert isinstance(comparison_result["best_strategy"], str)
        assert len(comparison_result["strategies"]) == 3
        assert len(comparison_result["performance_comparison"]) == 3
    def test_validate_strategy_config(self, strategy_service, sample_strategy_config) -> None:
        """Тест валидации конфигурации стратегии."""
        is_valid = strategy_service.validate_strategy_config(sample_strategy_config)
        assert isinstance(is_valid, bool)
        assert is_valid == True
    def test_validate_strategy_config_invalid(self, strategy_service) -> None:
        """Тест валидации невалидной конфигурации стратегии."""
        invalid_config = {
            "strategy_type": "invalid_type",
            "entry_threshold": 1.5,
            "stop_loss": -0.1
        }
        is_valid = strategy_service.validate_strategy_config(invalid_config)
        assert isinstance(is_valid, bool)
        assert is_valid == False
    def test_get_strategy_performance_metrics(self, strategy_service, sample_market_data, sample_strategy_config) -> None:
        """Тест получения метрик производительности стратегии."""
        strategy_result = strategy_service.execute_strategy(sample_market_data, sample_strategy_config)
        metrics = strategy_service.get_strategy_performance_metrics(strategy_result)
        assert isinstance(metrics, dict)
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "calmar_ratio" in metrics
        assert isinstance(metrics["total_return"], float)
        assert isinstance(metrics["sharpe_ratio"], float)
        assert isinstance(metrics["max_drawdown"], float)
        assert isinstance(metrics["win_rate"], float)
        assert isinstance(metrics["profit_factor"], float)
        assert isinstance(metrics["calmar_ratio"], float)
        assert metrics["win_rate"] >= 0.0 and metrics["win_rate"] <= 1.0
        assert metrics["max_drawdown"] <= 0.0
    def test_generate_strategy_report(self, strategy_service, sample_market_data, sample_strategy_config) -> None:
        """Тест генерации отчета по стратегии."""
        strategy_result = strategy_service.execute_strategy(sample_market_data, sample_strategy_config)
        report = strategy_service.generate_strategy_report(strategy_result)
        assert isinstance(report, dict)
        assert "summary" in report
        assert "performance_metrics" in report
        assert "trades_analysis" in report
        assert "risk_analysis" in report
        assert "recommendations" in report
        assert isinstance(report["summary"], str)
        assert isinstance(report["performance_metrics"], dict)
        assert isinstance(report["trades_analysis"], dict)
        assert isinstance(report["risk_analysis"], dict)
        assert isinstance(report["recommendations"], list)
    def test_calculate_position_size(self, strategy_service, sample_strategy_config) -> None:
        """Тест расчета размера позиции."""
        account_balance = 10000.0
        risk_per_trade = 0.02
        position_size = strategy_service.calculate_position_size(account_balance, risk_per_trade, sample_strategy_config)
        assert isinstance(position_size, float)
        assert position_size >= 0.0
        assert position_size <= account_balance
    def test_calculate_position_size_zero_balance(self, strategy_service, sample_strategy_config) -> None:
        """Тест расчета размера позиции с нулевым балансом."""
        account_balance = 0.0
        risk_per_trade = 0.02
        position_size = strategy_service.calculate_position_size(account_balance, risk_per_trade, sample_strategy_config)
        assert isinstance(position_size, float)
        assert position_size == 0.0
    def test_strategy_service_error_handling(self, strategy_service) -> None:
        """Тест обработки ошибок в сервисе."""
        with pytest.raises(Exception):
            strategy_service.execute_strategy(None, {})
        with pytest.raises(Exception):
            strategy_service.backtest_strategy("invalid_data", {})
    def test_strategy_service_performance(self, strategy_service, sample_market_data, sample_strategy_config) -> None:
        """Тест производительности сервиса."""
        import time
        start_time = time.time()
        for _ in range(3):
            strategy_service.execute_strategy(sample_market_data, sample_strategy_config)
        end_time = time.time()
        assert (end_time - start_time) < 10.0
    def test_strategy_service_config_customization(self: "TestStrategyService") -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {
            "default_strategy": "custom_strategy",
            "risk_management": {
                "max_risk_per_trade": 0.02,
                "max_portfolio_risk": 0.1
            },
            "position_sizing": {
                "method": "kelly_criterion",
                "max_position_size": 0.2
            }
        }
        service = StrategyService(custom_config)
        assert service.config["default_strategy"] == "custom_strategy"
        assert service.config["risk_management"]["max_risk_per_trade"] == 0.02
        assert service.config["position_sizing"]["method"] == "kelly_criterion" 
