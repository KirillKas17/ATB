from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger
from scikit_optimize import BayesSearchCV, gp_minimize
from scikit_optimize.space import Categorical, Integer, Real

from core.strategy import Strategy


@dataclass
class OptimizationConfig:
    """Конфигурация оптимизации"""

    # Параметры оптимизации
    optimization_method: str = "grid"  # Метод оптимизации (grid/random/bayesian)
    n_trials: int = 100  # Количество попыток
    n_jobs: int = -1  # Количество процессов (-1 = все доступные)
    metric: str = "sharpe"  # Метрика оптимизации (sharpe/sortino/max_drawdown)
    min_trades: int = 10  # Минимальное количество сделок
    min_profit: float = 0.0  # Минимальная прибыль

    # Параметры валидации
    validation_method: str = "walk_forward"  # Метод валидации
    validation_window: int = 1000  # Размер окна валидации
    validation_step: int = 100  # Шаг валидации

    # Параметры логирования
    log_dir: str = "logs"
    save_results: bool = True  # Сохранять результаты


class StrategyOptimizer:
    """Оптимизатор параметров стратегий"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация оптимизатора.

        Args:
            config: Словарь с параметрами оптимизации
        """
        self.config = OptimizationConfig(**config) if config else OptimizationConfig()
        self._setup_logger()
        self.optimizer = None  # Будет инициализирован в методе optimize

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def optimize(
        self, strategy: Strategy, data: pd.DataFrame, param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Оптимизация параметров стратегии.

        Args:
            strategy: Стратегия для оптимизации
            data: DataFrame с данными
            param_grid: Сетка параметров для оптимизации

        Returns:
            Dict с оптимальными параметрами
        """
        try:
            if self.config.optimization_method == "grid":
                return self._grid_search(strategy, data, param_grid)
            elif self.config.optimization_method == "random":
                return self._random_search(strategy, data, param_grid)
            elif self.config.optimization_method == "bayesian":
                return self._bayesian_optimization(strategy, data, param_grid)
            else:
                raise ValueError(f"Unknown optimization method: {self.config.optimization_method}")

        except Exception as e:
            logger.error(f"Error optimizing strategy: {str(e)}")
            return {}

    def _grid_search(
        self, strategy: Strategy, data: pd.DataFrame, param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Поиск по сетке параметров.

        Args:
            strategy: Стратегия для оптимизации
            data: DataFrame с данными
            param_grid: Сетка параметров

        Returns:
            Dict с оптимальными параметрами
        """
        try:
            # Генерируем все комбинации параметров
            param_combinations = list(product(*param_grid.values()))
            param_names = list(param_grid.keys())

            # Запускаем оптимизацию в параллельном режиме
            with ProcessPoolExecutor(max_workers=self.config.n_jobs) as executor:
                futures = []
                for params in param_combinations:
                    param_dict = dict(zip(param_names, params))
                    futures.append(
                        executor.submit(self._evaluate_params, strategy, data, param_dict, None)
                    )

                # Собираем результаты
                results = []
                for future in futures:
                    result = future.result()
                    if result:
                        results.append(result)

            # Выбираем лучший результат
            if not results:
                return {}

            best_result = max(results, key=lambda x: x["score"])
            return best_result["params"]

        except Exception as e:
            logger.error(f"Error in grid search: {str(e)}")
            return {}

    def _random_search(
        self, strategy: Strategy, data: pd.DataFrame, param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Случайный поиск параметров.

        Args:
            strategy: Стратегия для оптимизации
            data: DataFrame с данными
            param_grid: Сетка параметров

        Returns:
            Dict с оптимальными параметрами
        """
        try:
            results = []
            for _ in range(self.config.n_trials):
                # Генерируем случайные параметры
                params = {}
                for param_name, param_values in param_grid.items():
                    params[param_name] = np.random.choice(param_values)

                # Оцениваем параметры
                result = self._evaluate_params(strategy, data, params, None)
                if result:
                    results.append(result)

            # Выбираем лучший результат
            if not results:
                return {}

            best_result = max(results, key=lambda x: x["score"])
            return best_result["params"]

        except Exception as e:
            logger.error(f"Error in random search: {str(e)}")
            return {}

    def _bayesian_optimization(
        self, strategy: Strategy, data: pd.DataFrame, param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Байесовская оптимизация параметров.

        Args:
            strategy: Стратегия для оптимизации
            data: DataFrame с данными
            param_grid: Сетка параметров

        Returns:
            Dict с оптимальными параметрами
        """
        try:
            from skopt import Optimizer
            from skopt.space import Categorical, Integer, Real

            # Преобразуем сетку параметров в пространство поиска
            search_space = []
            param_names = []
            for param_name, param_values in param_grid.items():
                param_names.append(param_name)
                if isinstance(param_values[0], int):
                    search_space.append(Integer(min(param_values), max(param_values)))
                elif isinstance(param_values[0], float):
                    search_space.append(Real(min(param_values), max(param_values)))
                else:
                    search_space.append(Categorical(param_values))

            # Создаем оптимизатор
            optimizer = Optimizer(
                dimensions=search_space,
                base_estimator="gp",
                n_initial_points=10,
                acq_func="gp_hedge",
                n_jobs=self.config.n_jobs,
            )

            # Функция для оценки параметров
            def objective(params):
                param_dict = dict(zip(param_names, params))
                result = self._evaluate_params(strategy, data, param_dict, None)
                if result is None:
                    return -np.inf
                return result["score"]

            # Запускаем оптимизацию
            for _ in range(self.config.n_trials):
                next_params = optimizer.ask()
                score = objective(next_params)
                optimizer.tell(next_params, score)

            # Получаем лучшие параметры
            best_params = dict(zip(param_names, optimizer.Xi[np.argmax(optimizer.yi)]))
            return best_params

        except Exception as e:
            logger.error(f"Error in bayesian optimization: {str(e)}")
            return {}

    def _evaluate_params(
        self,
        strategy: Strategy,
        data: pd.DataFrame,
        params: Dict[str, Any],
        y: Optional[pd.Series] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Оценка параметров стратегии.

        Args:
            strategy: Стратегия для оптимизации
            data: DataFrame с данными
            params: Параметры для оценки
            y: Целевая переменная (опционально)

        Returns:
            Dict с результатами оценки или None
        """
        try:
            # Устанавливаем параметры
            strategy.config = type(strategy.config)(**params)

            # Запускаем бэктест
            results = self._run_backtest(strategy, data, y)

            # Проверяем минимальные требования
            if results["n_trades"] < self.config.min_trades:
                return None

            if float(results["total_profit"]) < self.config.min_profit:
                return None

            # Рассчитываем метрику
            score = self._calculate_metric(results)

            return {"params": params, "results": results, "score": score}

        except Exception as e:
            logger.error(f"Error evaluating parameters: {str(e)}")
            return None

    def _run_backtest(
        self, strategy: Strategy, data: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Запуск бэктеста стратегии.

        Args:
            strategy: Стратегия для тестирования
            data: DataFrame с данными
            y: Целевая переменная (опционально)

        Returns:
            Dict с результатами бэктеста
        """
        try:
            # Разделяем данные на обучающую и тестовую выборки
            if self.config.validation_method == "walk_forward":
                train_data = data.iloc[: -self.config.validation_window]
                test_data = data.iloc[-self.config.validation_window :]
                if y is not None:
                    train_y = y.iloc[: -self.config.validation_window]
                    test_y = y.iloc[-self.config.validation_window :]
                else:
                    train_y = None
                    test_y = None
            else:
                split_idx = int(len(data) * 0.8)
                train_data = data.iloc[:split_idx]
                test_data = data.iloc[split_idx:]
                if y is not None:
                    train_y = y.iloc[:split_idx]
                    test_y = y.iloc[split_idx:]
                else:
                    train_y = None
                    test_y = None

            # Обучаем стратегию
            if train_y is not None:
                strategy.fit(train_data, y=train_y)
            else:
                strategy.fit(train_data)

            # Тестируем на тестовой выборке
            signals = strategy.analyze(test_data)
            trades = self._simulate_trades(signals, test_data)

            # Рассчитываем метрики
            returns = pd.Series([float(trade["profit"]) for trade in trades])
            total_profit = float(returns.sum())
            n_trades = len(trades)
            win_rate = len(returns[returns > 0]) / n_trades if n_trades > 0 else 0
            max_drawdown = self._calculate_max_drawdown(returns)

            return {
                "total_profit": total_profit,
                "n_trades": n_trades,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "returns": returns,
                "trades": trades,
            }

        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            return {
                "total_profit": 0,
                "n_trades": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "returns": pd.Series(),
                "trades": [],
            }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        Расчет максимальной просадки.

        Args:
            returns: Series с доходностями

        Returns:
            float: Максимальная просадка
        """
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return abs(drawdown.min())
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0

    def _calculate_metric(self, results: Dict[str, Any]) -> float:
        """
        Расчет метрики оптимизации.

        Args:
            results: Результаты бэктеста

        Returns:
            float: Значение метрики
        """
        try:
            if self.config.metric == "sharpe":
                returns = results["returns"]
                if len(returns) < 2:
                    return 0.0
                return returns.mean() / returns.std() * np.sqrt(252)
            elif self.config.metric == "sortino":
                returns = results["returns"]
                if len(returns) < 2:
                    return 0.0
                downside = returns[returns < 0]
                if len(downside) == 0:
                    return 0.0
                return returns.mean() / downside.std() * np.sqrt(252)
            elif self.config.metric == "max_drawdown":
                return -results["max_drawdown"]
            else:
                raise ValueError(f"Unknown metric: {self.config.metric}")
        except Exception as e:
            logger.error(f"Error calculating metric: {str(e)}")
            return 0.0

    def _get_scoring_function(self):
        """Получение функции оценки для оптимизатора."""
        if self.config.metric == "sharpe":
            return lambda x: x.mean() / x.std() * np.sqrt(252)
        elif self.config.metric == "sortino":
            return lambda x: x.mean() / x[x < 0].std() * np.sqrt(252)
        elif self.config.metric == "max_drawdown":
            return lambda x: -self._calculate_max_drawdown(x)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")

    def _simulate_trades(self, signals: List[Any], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Симуляция сделок по сигналам.

        Args:
            signals: Список сигналов
            data: DataFrame с данными

        Returns:
            List[Dict[str, Any]]: Список сделок
        """
        trades = []
        position = None

        for signal in signals:
            if signal.action == "buy" and position is None:
                position = {
                    "entry_price": signal.price,
                    "entry_time": data.index[-1],
                    "size": signal.size,
                    "stop_loss": signal.stop_loss,
                    "take_profit": signal.take_profit,
                }
            elif signal.action == "sell" and position is not None:
                profit = (signal.price - position["entry_price"]) * position["size"]
                trades.append(
                    {
                        "entry_price": position["entry_price"],
                        "exit_price": signal.price,
                        "entry_time": position["entry_time"],
                        "exit_time": data.index[-1],
                        "size": position["size"],
                        "profit": profit,
                    }
                )
                position = None

        return trades

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Обучение оптимизатора.

        Args:
            X: Признаки
            y: Целевая переменная
        """
        if self.optimizer is not None:
            self.optimizer.fit(X, y)
        else:
            logger.warning("Optimizer is not initialized")

    def update_parameters(self, parameters):
        if isinstance(parameters, pd.DataFrame):
            parameters = parameters.to_dict(orient="records")[0]
        # ... существующий код ...
