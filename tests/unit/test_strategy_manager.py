"""
Unit тесты для StrategyManager.
Тестирует управление стратегиями, включая создание, активацию,
мониторинг и оптимизацию торговых стратегий.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
# StrategyManager не найден в infrastructure.core
# from infrastructure.core.strategy_manager import StrategyManager



class StrategyManager:
    """Менеджер стратегий для тестов."""
    
    def __init__(self):
        self.strategies = {}
        self.active_strategies = []
    
    def add_strategy(self, name: str, strategy_config: Dict[str, Any]) -> bool:
        """Добавление стратегии."""
        self.strategies[name] = strategy_config
        return True
    
    def activate_strategy(self, name: str) -> bool:
        """Активация стратегии."""
        if name in self.strategies:
            self.active_strategies.append(name)
            return True
        return False
    
    def get_active_strategies(self) -> List[str]:
        """Получение активных стратегий."""
        return self.active_strategies.copy()

class TestStrategyManager:
    """Тесты для StrategyManager."""

    @pytest.fixture
    def strategy_manager(self) -> StrategyManager:
        """Фикстура для StrategyManager."""
        return StrategyManager()

    @pytest.fixture
    def sample_strategy(self) -> dict:
        """Фикстура с тестовой стратегией."""
        return {
            "id": "strategy_001",
            "name": "Test Strategy",
            "type": "trend_following",
            "status": "active",
            "parameters": {
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "ma_short": 10,
                "ma_long": 50,
                "stop_loss": 0.02,
                "take_profit": 0.05,
                "position_size": 0.1,
            },
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "timeframe": "1h",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "performance_metrics": {"total_return": 0.15, "sharpe_ratio": 1.5, "max_drawdown": -0.05, "win_rate": 0.65},
        }

    @pytest.fixture
    def sample_strategies_list(self) -> list:
        """Фикстура со списком тестовых стратегий."""
        return [
            {
                "id": "strategy_001",
                "name": "Trend Following",
                "type": "trend_following",
                "status": "active",
                "parameters": {"ma_short": 10, "ma_long": 50},
                "symbols": ["BTCUSDT"],
                "timeframe": "1h",
                "created_at": datetime.now() - timedelta(days=1),
            },
            {
                "id": "strategy_002",
                "name": "Mean Reversion",
                "type": "mean_reversion",
                "status": "inactive",
                "parameters": {"rsi_period": 14, "rsi_overbought": 70},
                "symbols": ["ETHUSDT"],
                "timeframe": "4h",
                "created_at": datetime.now() - timedelta(hours=6),
            },
        ]

    def test_initialization(self, strategy_manager: StrategyManager) -> None:
        """Тест инициализации менеджера стратегий."""
        assert strategy_manager is not None
        assert hasattr(strategy_manager, "strategies")
        assert hasattr(strategy_manager, "strategy_validators")
        assert hasattr(strategy_manager, "strategy_monitors")
        assert hasattr(strategy_manager, "performance_trackers")

    def test_create_strategy(self, strategy_manager: StrategyManager) -> None:
        """Тест создания стратегии."""
        # Параметры стратегии
        strategy_params = {
            "name": "Test Strategy",
            "type": "trend_following",
            "parameters": {"rsi_period": 14, "ma_short": 10, "ma_long": 50},
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
        }
        # Создание стратегии
        strategy = strategy_manager.create_strategy(strategy_params)
        # Проверки
        assert strategy is not None
        assert "id" in strategy
        assert strategy["name"] == "Test Strategy"
        assert strategy["type"] == "trend_following"
        assert strategy["status"] == "inactive"
        assert "created_at" in strategy
        assert "updated_at" in strategy

    def test_activate_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест активации стратегии."""
        # Создание стратегии
        strategy_manager.create_strategy(sample_strategy)
        # Активация стратегии
        activation_result = strategy_manager.activate_strategy(sample_strategy["id"])
        # Проверки
        assert activation_result is not None
        assert "success" in activation_result
        assert "activation_time" in activation_result
        assert "strategy_status" in activation_result
        # Проверка типов данных
        assert isinstance(activation_result["success"], bool)
        assert isinstance(activation_result["activation_time"], datetime)
        assert isinstance(activation_result["strategy_status"], str)

    def test_deactivate_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест деактивации стратегии."""
        # Создание и активация стратегии
        strategy_manager.create_strategy(sample_strategy)
        strategy_manager.activate_strategy(sample_strategy["id"])
        # Деактивация стратегии
        deactivation_result = strategy_manager.deactivate_strategy(sample_strategy["id"])
        # Проверки
        assert deactivation_result is not None
        assert "success" in deactivation_result
        assert "deactivation_time" in deactivation_result
        assert "strategy_status" in deactivation_result
        # Проверка типов данных
        assert isinstance(deactivation_result["success"], bool)
        assert isinstance(deactivation_result["deactivation_time"], datetime)
        assert isinstance(deactivation_result["strategy_status"], str)

    def test_update_strategy_parameters(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест обновления параметров стратегии."""
        # Создание стратегии
        strategy_manager.create_strategy(sample_strategy)
        # Обновление параметров
        new_parameters = {"rsi_period": 21, "ma_short": 15, "ma_long": 60}
        update_result = strategy_manager.update_strategy_parameters(sample_strategy["id"], new_parameters)
        # Проверки
        assert update_result is not None
        assert "success" in update_result
        assert "updated_strategy" in update_result
        assert "update_time" in update_result
        # Проверка типов данных
        assert isinstance(update_result["success"], bool)
        assert isinstance(update_result["updated_strategy"], dict)
        assert isinstance(update_result["update_time"], datetime)

    def test_get_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест получения стратегии."""
        # Создание стратегии
        strategy_manager.create_strategy(sample_strategy)
        # Получение стратегии
        retrieved_strategy = strategy_manager.get_strategy(sample_strategy["id"])
        # Проверки
        assert retrieved_strategy is not None
        assert retrieved_strategy["id"] == sample_strategy["id"]
        assert retrieved_strategy["name"] == sample_strategy["name"]
        assert retrieved_strategy["type"] == sample_strategy["type"]

    def test_get_strategies(self, strategy_manager: StrategyManager, sample_strategies_list: list) -> None:
        """Тест получения списка стратегий."""
        # Создание стратегий
        for strategy in sample_strategies_list:
            strategy_manager.create_strategy(strategy)
        # Получение всех стратегий
        all_strategies = strategy_manager.get_strategies()
        # Проверки
        assert all_strategies is not None
        assert isinstance(all_strategies, list)
        assert len(all_strategies) >= len(sample_strategies_list)

    def test_get_strategies_by_type(self, strategy_manager: StrategyManager, sample_strategies_list: list) -> None:
        """Тест получения стратегий по типу."""
        # Создание стратегий
        for strategy in sample_strategies_list:
            strategy_manager.create_strategy(strategy)
        # Получение стратегий по типу
        trend_strategies = strategy_manager.get_strategies_by_type("trend_following")
        mean_reversion_strategies = strategy_manager.get_strategies_by_type("mean_reversion")
        # Проверки
        assert trend_strategies is not None
        assert mean_reversion_strategies is not None
        assert isinstance(trend_strategies, list)
        assert isinstance(mean_reversion_strategies, list)
        # Проверка фильтрации
        for strategy in trend_strategies:
            assert strategy["type"] == "trend_following"
        for strategy in mean_reversion_strategies:
            assert strategy["type"] == "mean_reversion"

    def test_get_strategies_by_status(self, strategy_manager: StrategyManager, sample_strategies_list: list) -> None:
        """Тест получения стратегий по статусу."""
        # Создание стратегий
        for strategy in sample_strategies_list:
            strategy_manager.create_strategy(strategy)
        # Активация первой стратегии
        strategy_manager.activate_strategy(sample_strategies_list[0]["id"])
        # Получение стратегий по статусу
        active_strategies = strategy_manager.get_strategies_by_status("active")
        inactive_strategies = strategy_manager.get_strategies_by_status("inactive")
        # Проверки
        assert active_strategies is not None
        assert inactive_strategies is not None
        assert isinstance(active_strategies, list)
        assert isinstance(inactive_strategies, list)
        # Проверка фильтрации
        for strategy in active_strategies:
            assert strategy["status"] == "active"
        for strategy in inactive_strategies:
            assert strategy["status"] == "inactive"

    def test_validate_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест валидации стратегии."""
        # Валидация стратегии
        validation_result = strategy_manager.validate_strategy(sample_strategy)
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_errors" in validation_result
        assert "validation_warnings" in validation_result
        assert "validation_score" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_errors"], list)
        assert isinstance(validation_result["validation_warnings"], list)
        assert isinstance(validation_result["validation_score"], float)
        # Проверка диапазона
        assert 0.0 <= validation_result["validation_score"] <= 1.0

    def test_backtest_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест бэктестинга стратегии."""
        # Мок исторических данных
        historical_data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 1000),
                "high": np.random.uniform(46000, 56000, 1000),
                "low": np.random.uniform(44000, 54000, 1000),
                "close": np.random.uniform(45000, 55000, 1000),
                "volume": np.random.uniform(1000000, 5000000, 1000),
            }
        )
        # Бэктестинг стратегии
        backtest_result = strategy_manager.backtest_strategy(sample_strategy, historical_data)
        # Проверки
        assert backtest_result is not None
        assert "total_return" in backtest_result
        assert "sharpe_ratio" in backtest_result
        assert "max_drawdown" in backtest_result
        assert "win_rate" in backtest_result
        assert "trades" in backtest_result
        assert "equity_curve" in backtest_result
        # Проверка типов данных
        assert isinstance(backtest_result["total_return"], float)
        assert isinstance(backtest_result["sharpe_ratio"], float)
        assert isinstance(backtest_result["max_drawdown"], float)
        assert isinstance(backtest_result["win_rate"], float)
        assert isinstance(backtest_result["trades"], list)
        assert isinstance(backtest_result["equity_curve"], pd.Series)

    def test_optimize_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест оптимизации стратегии."""
        # Мок исторических данных
        historical_data = pd.DataFrame({"close": np.random.uniform(45000, 55000, 1000)})
        # Параметры для оптимизации
        param_ranges = {"rsi_period": (10, 20), "ma_short": (5, 15), "ma_long": (40, 60)}
        # Оптимизация стратегии
        optimization_result = strategy_manager.optimize_strategy(sample_strategy, param_ranges, historical_data)
        # Проверки
        assert optimization_result is not None
        assert "best_parameters" in optimization_result
        assert "best_performance" in optimization_result
        assert "optimization_history" in optimization_result
        assert "optimization_score" in optimization_result
        # Проверка типов данных
        assert isinstance(optimization_result["best_parameters"], dict)
        assert isinstance(optimization_result["best_performance"], dict)
        assert isinstance(optimization_result["optimization_history"], list)
        assert isinstance(optimization_result["optimization_score"], float)
        # Проверка диапазона
        assert 0.0 <= optimization_result["optimization_score"] <= 1.0

    def test_monitor_strategy_performance(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест мониторинга производительности стратегии."""
        # Создание и активация стратегии
        strategy_manager.create_strategy(sample_strategy)
        strategy_manager.activate_strategy(sample_strategy["id"])
        # Мониторинг производительности
        performance_monitoring = strategy_manager.monitor_strategy_performance(sample_strategy["id"])
        # Проверки
        assert performance_monitoring is not None
        assert "current_performance" in performance_monitoring
        assert "performance_trend" in performance_monitoring
        assert "risk_metrics" in performance_monitoring
        assert "alerts" in performance_monitoring
        # Проверка типов данных
        assert isinstance(performance_monitoring["current_performance"], dict)
        assert isinstance(performance_monitoring["performance_trend"], str)
        assert isinstance(performance_monitoring["risk_metrics"], dict)
        assert isinstance(performance_monitoring["alerts"], list)

    def test_calculate_strategy_metrics(self, strategy_manager: StrategyManager, sample_strategies_list: list) -> None:
        """Тест расчета метрик стратегий."""
        # Создание стратегий
        for strategy in sample_strategies_list:
            strategy_manager.create_strategy(strategy)
        # Расчет метрик
        metrics = strategy_manager.calculate_strategy_metrics()
        # Проверки
        assert metrics is not None
        assert "total_strategies" in metrics
        assert "active_strategies" in metrics
        assert "inactive_strategies" in metrics
        assert "avg_performance" in metrics
        assert "best_strategy" in metrics
        assert "worst_strategy" in metrics
        # Проверка типов данных
        assert isinstance(metrics["total_strategies"], int)
        assert isinstance(metrics["active_strategies"], int)
        assert isinstance(metrics["inactive_strategies"], int)
        assert isinstance(metrics["avg_performance"], float)
        assert isinstance(metrics["best_strategy"], str)
        assert isinstance(metrics["worst_strategy"], str)
        # Проверка логики
        assert metrics["total_strategies"] >= 0
        assert metrics["active_strategies"] + metrics["inactive_strategies"] == metrics["total_strategies"]

    def test_compare_strategies(self, strategy_manager: StrategyManager, sample_strategies_list: list) -> None:
        """Тест сравнения стратегий."""
        # Создание стратегий
        for strategy in sample_strategies_list:
            strategy_manager.create_strategy(strategy)
        # Сравнение стратегий
        comparison_result = strategy_manager.compare_strategies([s["id"] for s in sample_strategies_list])
        # Проверки
        assert comparison_result is not None
        assert "comparison_metrics" in comparison_result
        assert "performance_ranking" in comparison_result
        assert "risk_ranking" in comparison_result
        assert "recommendations" in comparison_result
        # Проверка типов данных
        assert isinstance(comparison_result["comparison_metrics"], dict)
        assert isinstance(comparison_result["performance_ranking"], list)
        assert isinstance(comparison_result["risk_ranking"], list)
        assert isinstance(comparison_result["recommendations"], list)

    def test_delete_strategy(self, strategy_manager: StrategyManager, sample_strategy: dict) -> None:
        """Тест удаления стратегии."""
        # Создание стратегии
        strategy_manager.create_strategy(sample_strategy)
        # Удаление стратегии
        delete_result = strategy_manager.delete_strategy(sample_strategy["id"])
        # Проверки
        assert delete_result is not None
        assert "success" in delete_result
        assert "deletion_time" in delete_result
        # Проверка типов данных
        assert isinstance(delete_result["success"], bool)
        assert isinstance(delete_result["deletion_time"], datetime)
        # Проверка, что стратегия удалена
        retrieved_strategy = strategy_manager.get_strategy(sample_strategy["id"])
        assert retrieved_strategy is None

    def test_error_handling(self, strategy_manager: StrategyManager) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            strategy_manager.create_strategy(None)
        with pytest.raises(ValueError):
            strategy_manager.activate_strategy("invalid_id")

    def test_edge_cases(self, strategy_manager: StrategyManager) -> None:
        """Тест граничных случаев."""
        # Тест с пустыми параметрами
        empty_strategy = {
            "name": "Empty Strategy",
            "type": "trend_following",
            "parameters": {},
            "symbols": [],
            "timeframe": "1h",
        }
        validation_result = strategy_manager.validate_strategy(empty_strategy)
        assert validation_result["is_valid"] is False
        # Тест с очень большими параметрами
        large_strategy = {
            "name": "Large Strategy",
            "type": "trend_following",
            "parameters": {"rsi_period": 1000, "ma_short": 1000, "ma_long": 10000},
            "symbols": ["BTCUSDT"],
            "timeframe": "1h",
        }
        validation_result = strategy_manager.validate_strategy(large_strategy)
        assert validation_result["is_valid"] is False

    def test_cleanup(self, strategy_manager: StrategyManager) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        strategy_manager.cleanup()
        # Проверка, что ресурсы освобождены
        assert strategy_manager.strategies == {}
        assert strategy_manager.strategy_validators == {}
        assert strategy_manager.strategy_monitors == {}
        assert strategy_manager.performance_trackers == {}
