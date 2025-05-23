import unittest

import numpy as np
import pandas as pd

from .backtest import Backtest
from .base_strategy import BaseStrategy
from .optimizer import StrategyOptimizer
from .visualization import Visualizer


class TestStrategy(unittest.TestCase):
    """Тесты для торговых стратегий"""

    def setUp(self):
        """Подготовка тестовых данных"""
        # Создаем тестовые данные
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="1H")
        np.random.seed(42)

        # Генерируем случайные цены
        prices = np.random.normal(100, 1, len(dates))
        prices = np.cumsum(prices - 100) + 100

        # Создаем DataFrame
        self.data = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.001,
                "low": prices * 0.999,
                "close": prices,
                "volume": np.random.randint(1000, 10000, len(dates)),
            },
            index=dates,
        )

        # Создаем тестовую стратегию
        self.strategy = BaseStrategy()

        # Создаем оптимизатор
        self.optimizer = StrategyOptimizer()

        # Создаем бэктестер
        self.backtest = Backtest()

        # Создаем визуализатор
        self.visualizer = Visualizer()

    def test_strategy_initialization(self):
        """Тест инициализации стратегии"""
        self.assertIsNotNone(self.strategy)
        self.assertIsNotNone(self.strategy.config)

    def test_generate_signal(self):
        """Тест генерации сигналов"""
        signal = self.strategy.generate_signal(self.data)
        self.assertIsNotNone(signal)
        self.assertIn(signal.direction, ["long", "short", "close", None])

    def test_optimizer(self):
        """Тест оптимизатора"""
        # Создаем сетку параметров
        param_grid = {"window": [10, 20, 30], "threshold": [0.01, 0.02, 0.03]}

        # Запускаем оптимизацию
        best_params = self.optimizer.optimize(self.strategy, self.data, param_grid)

        self.assertIsNotNone(best_params)
        self.assertIsInstance(best_params, dict)

    def test_backtest(self):
        """Тест бэктеста"""
        # Запускаем бэктест
        results = self.backtest.run(self.strategy, self.data)

        self.assertIsNotNone(results)
        self.assertIsInstance(results, dict)
        self.assertIn("n_trades", results)
        self.assertIn("total_profit", results)
        self.assertIn("win_rate", results)

    def test_visualizer(self):
        """Тест визуализатора"""
        # Запускаем бэктест
        results = self.backtest.run(self.strategy, self.data)

        # Проверяем наличие trades и equity_curve
        if self.backtest.trades is None:
            self.backtest.trades = []
        if self.backtest.equity_curve is None:
            self.backtest.equity_curve = pd.Series()

        # Создаем отчет
        self.visualizer.create_report(
            self.data,
            self.backtest.trades,
            self.backtest.equity_curve,
            results,
            "test_report",
        )

    def test_strategy_parameters(self):
        """Тест параметров стратегии"""
        # Проверяем параметры по умолчанию
        self.assertIsNotNone(self.strategy.config.window)
        self.assertIsNotNone(self.strategy.config.threshold)

        # Изменяем параметры
        self.strategy.config.window = 20
        self.strategy.config.threshold = 0.02

        self.assertEqual(self.strategy.config.window, 20)
        self.assertEqual(self.strategy.config.threshold, 0.02)

    def test_data_validation(self):
        """Тест валидации данных"""
        # Проверяем наличие необходимых колонок
        required_columns = ["open", "high", "low", "close", "volume"]
        for column in required_columns:
            self.assertIn(column, self.data.columns)

        # Проверяем отсутствие пропусков
        self.assertFalse(self.data.isnull().any().any())

    def test_signal_validation(self):
        """Тест валидации сигналов"""
        # Генерируем сигналы
        signals = []
        for i in range(len(self.data)):
            signal = self.strategy.generate_signal(self.data.iloc[: i + 1])
            if signal:
                signals.append(signal)

        # Проверяем валидность сигналов
        for signal in signals:
            self.assertIn(signal.direction, ["long", "short", "close"])
            self.assertIsNotNone(signal.entry_price)
            self.assertIsNotNone(signal.stop_loss)
            self.assertIsNotNone(signal.take_profit)

    def test_optimization_validation(self):
        """Тест валидации оптимизации"""
        # Создаем сетку параметров
        param_grid = {"window": [10, 20, 30], "threshold": [0.01, 0.02, 0.03]}

        # Запускаем оптимизацию
        best_params = self.optimizer.optimize(self.strategy, self.data, param_grid)

        # Проверяем валидность параметров
        self.assertIn("window", best_params)
        self.assertIn("threshold", best_params)
        self.assertIn(best_params["window"], param_grid["window"])
        self.assertIn(best_params["threshold"], param_grid["threshold"])

    def test_backtest_validation(self):
        """Тест валидации бэктеста"""
        # Запускаем бэктест
        results = self.backtest.run(self.strategy, self.data)

        # Проверяем валидность результатов
        self.assertGreaterEqual(results["n_trades"], 0)
        self.assertGreaterEqual(results["win_rate"], 0)
        self.assertLessEqual(results["win_rate"], 1)
        self.assertGreaterEqual(results["profit_factor"], 0)
        self.assertGreaterEqual(results["sharpe_ratio"], 0)
        self.assertGreaterEqual(results["sortino_ratio"], 0)
        self.assertGreaterEqual(results["max_drawdown"], 0)
        self.assertLessEqual(results["max_drawdown"], 1)

    def test_visualization_validation(self):
        """Тест валидации визуализации"""
        # Запускаем бэктест
        results = self.backtest.run(self.strategy, self.data)

        # Проверяем наличие trades и equity_curve
        if self.backtest.trades is None:
            self.backtest.trades = []
        if self.backtest.equity_curve is None:
            self.backtest.equity_curve = pd.Series()

        # Создаем отчет
        self.visualizer.create_report(
            self.data,
            self.backtest.trades,
            self.backtest.equity_curve,
            results,
            "test_report",
        )

        # Проверяем наличие файлов
        import os

        self.assertTrue(os.path.exists("plots"))
        self.assertTrue(os.path.exists("logs"))

    def test_error_handling(self):
        """Тест обработки ошибок"""
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        with self.assertRaises(Exception):
            self.strategy.generate_signal(empty_data)

        # Тест с некорректными данными
        invalid_data = self.data.copy()
        invalid_data["close"] = "invalid"
        with self.assertRaises(Exception):
            self.strategy.generate_signal(invalid_data)

        # Тест с некорректными параметрами
        with self.assertRaises(Exception):
            self.strategy.config.window = -1

        # Тест с некорректной сеткой параметров
        invalid_param_grid = {"window": ["invalid"], "threshold": ["invalid"]}
        with self.assertRaises(Exception):
            self.optimizer.optimize(self.strategy, self.data, invalid_param_grid)

    def test_performance(self):
        """Тест производительности"""
        import time

        # Тест производительности генерации сигналов
        start_time = time.time()
        for i in range(len(self.data)):
            self.strategy.generate_signal(self.data.iloc[: i + 1])
        signal_time = time.time() - start_time

        # Тест производительности оптимизации
        start_time = time.time()
        param_grid = {"window": [10, 20, 30], "threshold": [0.01, 0.02, 0.03]}
        self.optimizer.optimize(self.strategy, self.data, param_grid)
        optimization_time = time.time() - start_time

        # Тест производительности бэктеста
        start_time = time.time()
        self.backtest.run(self.strategy, self.data)
        backtest_time = time.time() - start_time

        # Проверяем, что время выполнения в разумных пределах
        self.assertLess(signal_time, 10)  # Менее 10 секунд
        self.assertLess(optimization_time, 60)  # Менее 60 секунд
        self.assertLess(backtest_time, 30)  # Менее 30 секунд

    def test_memory_usage(self):
        """Тест использования памяти"""
        import os

        import psutil

        # Измеряем использование памяти до тестов
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss

        # Запускаем тесты
        self.test_strategy_initialization()
        self.test_generate_signal()
        self.test_optimizer()
        self.test_backtest()
        self.test_visualizer()

        # Измеряем использование памяти после тестов
        memory_after = process.memory_info().rss
        memory_used = memory_after - memory_before

        # Проверяем, что использование памяти в разумных пределах
        self.assertLess(memory_used, 100 * 1024 * 1024)  # Менее 100 МБ


if __name__ == "__main__":
    unittest.main()
