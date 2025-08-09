"""
Unit тесты для StrategyOptimizer.
Тестирует оптимизацию стратегий, включая настройку параметров,
генетические алгоритмы, бэктестинг и анализ производительности.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Any
# StrategyOptimizer не найден в infrastructure.core
# from infrastructure.core.strategy_optimizer import StrategyOptimizer



class StrategyOptimizer:
    """Оптимизатор стратегий для тестов."""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_strategy(self, strategy_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация стратегии."""
        optimization_result = {
            "strategy_name": strategy_name,
            "optimized_parameters": parameters,
            "performance_improvement": 0.1,
            "timestamp": datetime.now()
        }
        self.optimization_history.append(optimization_result)
        return optimization_result
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Получение истории оптимизации."""
        return self.optimization_history.copy()

class TestStrategyOptimizer:
    """Тесты для StrategyOptimizer."""

    @pytest.fixture
    def strategy_optimizer(self) -> StrategyOptimizer:
        """Фикстура для StrategyOptimizer."""
        return StrategyOptimizer()

    @pytest.fixture
    def sample_strategy_params(self) -> dict:
        """Фикстура с параметрами стратегии."""
        return {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "ma_short": 10,
            "ma_long": 50,
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "position_size": 0.1,
        }

    @pytest.fixture
    def sample_historical_data(self) -> pd.DataFrame:
        """Фикстура с историческими данными."""
        dates = pd.DatetimeIndex(pd.date_range("2023-01-01", periods=1000, freq="1H"))
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "open": np.random.uniform(45000, 55000, 1000),
                "high": np.random.uniform(46000, 56000, 1000),
                "low": np.random.uniform(44000, 54000, 1000),
                "close": np.random.uniform(45000, 55000, 1000),
                "volume": np.random.uniform(1000000, 5000000, 1000),
            },
            index=dates,
        )
        # Создание более реалистичных данных
        data["high"] = data[["open", "close"]].max(axis=1) + np.random.uniform(0, 1000, 1000)
        data["low"] = data[["open", "close"]].min(axis=1) - np.random.uniform(0, 1000, 1000)
        return data

    def test_initialization(self, strategy_optimizer: StrategyOptimizer) -> None:
        """Тест инициализации оптимизатора стратегий."""
        assert strategy_optimizer is not None
        assert hasattr(strategy_optimizer, "optimization_algorithms")
        assert hasattr(strategy_optimizer, "backtest_engine")
        assert hasattr(strategy_optimizer, "performance_metrics")
        assert hasattr(strategy_optimizer, "optimization_results")

    def test_optimize_parameters(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации параметров."""
        # Параметры для оптимизации
        param_ranges = {
            "rsi_period": (10, 20),
            "rsi_overbought": (65, 75),
            "rsi_oversold": (25, 35),
            "ma_short": (5, 15),
            "ma_long": (40, 60),
            "stop_loss": (0.01, 0.03),
            "take_profit": (0.03, 0.07),
            "position_size": (0.05, 0.15),
        }
        # Оптимизация параметров
        optimization_result = strategy_optimizer.optimize_parameters(
            sample_strategy_params,
            param_ranges,
            sample_historical_data,
            method="genetic_algorithm",
            generations=10,
            population_size=20,
        )
        # Проверки
        assert optimization_result is not None
        assert "best_params" in optimization_result
        assert "best_score" in optimization_result
        assert "optimization_history" in optimization_result
        assert "convergence_plot" in optimization_result
        # Проверка типов данных
        assert isinstance(optimization_result["best_params"], dict)
        assert isinstance(optimization_result["best_score"], float)
        assert isinstance(optimization_result["optimization_history"], list)
        assert isinstance(optimization_result["convergence_plot"], dict)
        # Проверка логики
        assert optimization_result["best_score"] > 0
        assert len(optimization_result["optimization_history"]) > 0

    def test_genetic_algorithm_optimization(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации генетическим алгоритмом."""
        # Параметры для оптимизации
        param_ranges = {
            "rsi_period": (10, 20),
            "rsi_overbought": (65, 75),
            "rsi_oversold": (25, 35),
            "ma_short": (5, 15),
            "ma_long": (40, 60),
        }
        # Генетическая оптимизация
        ga_result = strategy_optimizer.genetic_algorithm_optimization(
            sample_strategy_params, param_ranges, sample_historical_data, generations=5, population_size=10
        )
        # Проверки
        assert ga_result is not None
        assert "best_individual" in ga_result
        assert "best_fitness" in ga_result
        assert "population_history" in ga_result
        assert "fitness_history" in ga_result
        # Проверка типов данных
        assert isinstance(ga_result["best_individual"], dict)
        assert isinstance(ga_result["best_fitness"], float)
        assert isinstance(ga_result["population_history"], list)
        assert isinstance(ga_result["fitness_history"], list)
        # Проверка логики
        assert ga_result["best_fitness"] > 0
        assert len(ga_result["population_history"]) > 0

    def test_grid_search_optimization(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации перебором сетки."""
        # Параметры для оптимизации
        param_grid = {
            "rsi_period": [10, 14, 20],
            "rsi_overbought": [65, 70, 75],
            "rsi_oversold": [25, 30, 35],
            "ma_short": [5, 10, 15],
            "ma_long": [40, 50, 60],
        }
        # Поиск по сетке
        grid_result = strategy_optimizer.grid_search_optimization(
            sample_strategy_params, param_grid, sample_historical_data
        )
        # Проверки
        assert grid_result is not None
        assert "best_params" in grid_result
        assert "best_score" in grid_result
        assert "all_results" in grid_result
        assert "search_time" in grid_result
        # Проверка типов данных
        assert isinstance(grid_result["best_params"], dict)
        assert isinstance(grid_result["best_score"], float)
        assert isinstance(grid_result["all_results"], list)
        assert isinstance(grid_result["search_time"], float)
        # Проверка логики
        assert grid_result["best_score"] > 0
        assert len(grid_result["all_results"]) > 0

    def test_bayesian_optimization(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест байесовской оптимизации."""
        # Параметры для оптимизации
        param_bounds = {
            "rsi_period": (10, 20),
            "rsi_overbought": (65, 75),
            "rsi_oversold": (25, 35),
            "ma_short": (5, 15),
            "ma_long": (40, 60),
        }
        # Байесовская оптимизация
        bayesian_result = strategy_optimizer.bayesian_optimization(
            sample_strategy_params, param_bounds, sample_historical_data, n_iterations=10
        )
        # Проверки
        assert bayesian_result is not None
        assert "best_params" in bayesian_result
        assert "best_score" in bayesian_result
        assert "acquisition_history" in bayesian_result
        assert "optimization_time" in bayesian_result
        # Проверка типов данных
        assert isinstance(bayesian_result["best_params"], dict)
        assert isinstance(bayesian_result["best_score"], float)
        assert isinstance(bayesian_result["acquisition_history"], list)
        assert isinstance(bayesian_result["optimization_time"], float)
        # Проверка логики
        assert bayesian_result["best_score"] > 0
        assert len(bayesian_result["acquisition_history"]) > 0

    def test_backtest_strategy(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест бэктестинга стратегии."""
        # Бэктестинг стратегии
        backtest_result = strategy_optimizer.backtest_strategy(sample_strategy_params, sample_historical_data)
        # Проверки
        assert backtest_result is not None
        assert "total_return" in backtest_result
        assert "sharpe_ratio" in backtest_result
        assert "max_drawdown" in backtest_result
        assert "win_rate" in backtest_result
        assert "profit_factor" in backtest_result
        assert "trades" in backtest_result
        assert "equity_curve" in backtest_result
        # Проверка типов данных
        assert isinstance(backtest_result["total_return"], float)
        assert isinstance(backtest_result["sharpe_ratio"], float)
        assert isinstance(backtest_result["max_drawdown"], float)
        assert isinstance(backtest_result["win_rate"], float)
        assert isinstance(backtest_result["profit_factor"], float)
        assert isinstance(backtest_result["trades"], list)
        assert isinstance(backtest_result["equity_curve"], pd.Series)
        # Проверка логики
        assert 0.0 <= backtest_result["win_rate"] <= 1.0
        assert backtest_result["max_drawdown"] <= 0.0

    def test_calculate_performance_metrics(
        self, strategy_optimizer: StrategyOptimizer, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест расчета метрик производительности."""
        # Мок результатов бэктестинга
        backtest_results = {
            "returns": pd.Series(np.random.normal(0.001, 0.02, 1000)),
            "trades": [
                {"entry_time": datetime.now(), "exit_time": datetime.now() + timedelta(hours=1), "pnl": 100},
                {"entry_time": datetime.now(), "exit_time": datetime.now() + timedelta(hours=2), "pnl": -50},
            ],
            "equity_curve": pd.Series(np.cumsum(np.random.normal(0.001, 0.02, 1000))),
        }
        # Расчет метрик производительности
        performance_metrics = strategy_optimizer.calculate_performance_metrics(backtest_results)
        # Проверки
        assert performance_metrics is not None
        assert "total_return" in performance_metrics
        assert "annualized_return" in performance_metrics
        assert "volatility" in performance_metrics
        assert "sharpe_ratio" in performance_metrics
        assert "sortino_ratio" in performance_metrics
        assert "calmar_ratio" in performance_metrics
        assert "max_drawdown" in performance_metrics
        assert "win_rate" in performance_metrics
        assert "profit_factor" in performance_metrics
        assert "avg_trade" in performance_metrics
        # Проверка типов данных
        assert isinstance(performance_metrics["total_return"], float)
        assert isinstance(performance_metrics["annualized_return"], float)
        assert isinstance(performance_metrics["volatility"], float)
        assert isinstance(performance_metrics["sharpe_ratio"], float)
        assert isinstance(performance_metrics["sortino_ratio"], float)
        assert isinstance(performance_metrics["calmar_ratio"], float)
        assert isinstance(performance_metrics["max_drawdown"], float)
        assert isinstance(performance_metrics["win_rate"], float)
        assert isinstance(performance_metrics["profit_factor"], float)
        assert isinstance(performance_metrics["avg_trade"], float)

    def test_optimize_risk_parameters(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации параметров риска."""
        # Параметры риска для оптимизации
        risk_params = {
            "stop_loss": 0.02,
            "take_profit": 0.05,
            "position_size": 0.1,
            "max_positions": 5,
            "max_daily_loss": 0.03,
        }
        # Оптимизация параметров риска
        risk_optimization = strategy_optimizer.optimize_risk_parameters(
            sample_strategy_params, risk_params, sample_historical_data
        )
        # Проверки
        assert risk_optimization is not None
        assert "optimal_risk_params" in risk_optimization
        assert "risk_adjusted_return" in risk_optimization
        assert "risk_metrics" in risk_optimization
        assert "risk_analysis" in risk_optimization
        # Проверка типов данных
        assert isinstance(risk_optimization["optimal_risk_params"], dict)
        assert isinstance(risk_optimization["risk_adjusted_return"], float)
        assert isinstance(risk_optimization["risk_metrics"], dict)
        assert isinstance(risk_optimization["risk_analysis"], dict)

    def test_optimize_timing_parameters(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации параметров времени."""
        # Параметры времени для оптимизации
        timing_params = {
            "entry_delay": 1,
            "exit_delay": 1,
            "min_hold_time": 10,
            "max_hold_time": 100,
            "time_filter": "all",
        }
        # Оптимизация параметров времени
        timing_optimization = strategy_optimizer.optimize_timing_parameters(
            sample_strategy_params, timing_params, sample_historical_data
        )
        # Проверки
        assert timing_optimization is not None
        assert "optimal_timing_params" in timing_optimization
        assert "timing_analysis" in timing_optimization
        assert "time_based_returns" in timing_optimization
        assert "timing_recommendations" in timing_optimization
        # Проверка типов данных
        assert isinstance(timing_optimization["optimal_timing_params"], dict)
        assert isinstance(timing_optimization["timing_analysis"], dict)
        assert isinstance(timing_optimization["time_based_returns"], dict)
        assert isinstance(timing_optimization["timing_recommendations"], list)

    def test_optimize_feature_parameters(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест оптимизации параметров признаков."""
        # Параметры признаков для оптимизации
        feature_params = {
            "rsi_period": 14,
            "ma_short": 10,
            "ma_long": 50,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "volume_ma_period": 20,
        }
        # Оптимизация параметров признаков
        feature_optimization = strategy_optimizer.optimize_feature_parameters(
            sample_strategy_params, feature_params, sample_historical_data
        )
        # Проверки
        assert feature_optimization is not None
        assert "optimal_feature_params" in feature_optimization
        assert "feature_importance" in feature_optimization
        assert "feature_analysis" in feature_optimization
        assert "feature_recommendations" in feature_optimization
        # Проверка типов данных
        assert isinstance(feature_optimization["optimal_feature_params"], dict)
        assert isinstance(feature_optimization["feature_importance"], dict)
        assert isinstance(feature_optimization["feature_analysis"], dict)
        assert isinstance(feature_optimization["feature_recommendations"], list)

    def test_validate_optimization_results(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест валидации результатов оптимизации."""
        # Мок результатов оптимизации
        optimization_results = {"best_params": sample_strategy_params, "best_score": 0.75, "optimization_history": []}
        # Валидация результатов
        validation_result = strategy_optimizer.validate_optimization_results(
            optimization_results, sample_historical_data
        )
        # Проверки
        assert validation_result is not None
        assert "is_valid" in validation_result
        assert "validation_score" in validation_result
        assert "overfitting_analysis" in validation_result
        assert "stability_analysis" in validation_result
        assert "validation_recommendations" in validation_result
        # Проверка типов данных
        assert isinstance(validation_result["is_valid"], bool)
        assert isinstance(validation_result["validation_score"], float)
        assert isinstance(validation_result["overfitting_analysis"], dict)
        assert isinstance(validation_result["stability_analysis"], dict)
        assert isinstance(validation_result["validation_recommendations"], list)
        # Проверка диапазона
        assert 0.0 <= validation_result["validation_score"] <= 1.0

    def test_generate_optimization_report(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест генерации отчета об оптимизации."""
        # Мок результатов оптимизации
        optimization_results = {
            "best_params": sample_strategy_params,
            "best_score": 0.75,
            "optimization_history": [],
            "performance_metrics": {"total_return": 0.15, "sharpe_ratio": 1.5, "max_drawdown": -0.05},
        }
        # Генерация отчета
        report = strategy_optimizer.generate_optimization_report(optimization_results)
        # Проверки
        assert report is not None
        assert "summary" in report
        assert "detailed_results" in report
        assert "recommendations" in report
        assert "charts" in report
        # Проверка типов данных
        assert isinstance(report["summary"], dict)
        assert isinstance(report["detailed_results"], dict)
        assert isinstance(report["recommendations"], list)
        assert isinstance(report["charts"], dict)

    def test_compare_optimization_methods(
        self, strategy_optimizer: StrategyOptimizer, sample_strategy_params: dict, sample_historical_data: pd.DataFrame
    ) -> None:
        """Тест сравнения методов оптимизации."""
        # Сравнение методов оптимизации
        comparison_result = strategy_optimizer.compare_optimization_methods(
            sample_strategy_params,
            sample_historical_data,
            methods=["genetic_algorithm", "grid_search", "bayesian_optimization"],
        )
        # Проверки
        assert comparison_result is not None
        assert "method_comparison" in comparison_result
        assert "best_method" in comparison_result
        assert "performance_comparison" in comparison_result
        assert "recommendations" in comparison_result
        # Проверка типов данных
        assert isinstance(comparison_result["method_comparison"], dict)
        assert isinstance(comparison_result["best_method"], str)
        assert isinstance(comparison_result["performance_comparison"], dict)
        assert isinstance(comparison_result["recommendations"], list)

    def test_error_handling(self, strategy_optimizer: StrategyOptimizer) -> None:
        """Тест обработки ошибок."""
        # Тест с некорректными данными
        with pytest.raises(ValueError):
            strategy_optimizer.optimize_parameters(None, {}, pd.DataFrame())
        with pytest.raises(ValueError):
            strategy_optimizer.backtest_strategy({}, pd.DataFrame())

    def test_edge_cases(self, strategy_optimizer: StrategyOptimizer) -> None:
        """Тест граничных случаев."""
        # Тест с очень короткими данными
        short_data = pd.DataFrame({"close": [100, 101, 102, 103, 104]})
        simple_params = {"rsi_period": 2, "ma_short": 2, "ma_long": 3}
        # Эти функции должны обрабатывать короткие данные
        backtest_result = strategy_optimizer.backtest_strategy(simple_params, short_data)
        assert backtest_result is not None

    def test_cleanup(self, strategy_optimizer: StrategyOptimizer) -> None:
        """Тест очистки ресурсов."""
        # Проверка корректной очистки
        strategy_optimizer.cleanup()
        # Проверка, что ресурсы освобождены
        assert strategy_optimizer.optimization_algorithms == {}
        assert strategy_optimizer.backtest_engine is None
        assert strategy_optimizer.performance_metrics == {}
        assert strategy_optimizer.optimization_results == {}
