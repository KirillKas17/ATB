import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from .base_strategy import BaseStrategy


@dataclass
class ParallelConfig:
    """Конфигурация параллельной обработки"""

    # Параметры параллельной обработки
    n_jobs: int = -1  # Количество процессов (-1 = все доступные)
    backend: str = "process"  # Бэкенд (process/thread)
    chunk_size: int = 1000  # Размер чанка данных
    max_workers: int = None  # Максимальное количество воркеров

    # Параметры логирования
    log_dir: str = "logs"  # Директория для логов


class ParallelProcessor:
    """Параллельный процессор для обработки данных"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Инициализация процессора.

        Args:
            config: Словарь с параметрами параллельной обработки
        """
        self.config = ParallelConfig(**config) if config else ParallelConfig()
        self._setup_logger()
        self._setup_executor()

    def _setup_logger(self):
        """Настройка логгера"""
        logger.add(
            f"{self.config.log_dir}/parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
        )

    def _setup_executor(self):
        """Настройка исполнителя"""
        if self.config.backend == "process":
            self.executor = ProcessPoolExecutor(
                max_workers=self.config.max_workers or multiprocessing.cpu_count()
            )
        else:
            self.executor = ThreadPoolExecutor(
                max_workers=self.config.max_workers or multiprocessing.cpu_count() * 2
            )

    def process_data(
        self, data: pd.DataFrame, func: callable, *args, **kwargs
    ) -> pd.DataFrame:
        """
        Параллельная обработка данных.

        Args:
            data: DataFrame с данными
            func: Функция для обработки
            *args: Позиционные аргументы
            **kwargs: Именованные аргументы

        Returns:
            DataFrame с обработанными данными
        """
        try:
            # Разбиваем данные на чанки
            chunks = np.array_split(data, len(data) // self.config.chunk_size + 1)

            # Создаем частичную функцию
            partial_func = partial(func, *args, **kwargs)

            # Запускаем обработку в параллельном режиме
            results = list(self.executor.map(partial_func, chunks))

            # Объединяем результаты
            return pd.concat(results)

        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return pd.DataFrame()

    def process_strategy(self, strategy: BaseStrategy, data: pd.DataFrame) -> List[Any]:
        """
        Параллельная обработка стратегии.

        Args:
            strategy: Стратегия для обработки
            data: DataFrame с данными

        Returns:
            List с результатами обработки
        """
        try:
            # Разбиваем данные на чанки
            chunks = np.array_split(data, len(data) // self.config.chunk_size + 1)

            # Запускаем обработку в параллельном режиме
            results = list(self.executor.map(strategy.generate_signal, chunks))

            # Фильтруем None результаты
            return [r for r in results if r is not None]

        except Exception as e:
            logger.error(f"Error processing strategy: {str(e)}")
            return []

    def process_multiple_strategies(
        self, strategies: List[BaseStrategy], data: pd.DataFrame
    ) -> Dict[str, List[Any]]:
        """
        Параллельная обработка нескольких стратегий.

        Args:
            strategies: Список стратегий
            data: DataFrame с данными

        Returns:
            Dict с результатами обработки
        """
        try:
            results = {}

            # Запускаем обработку в параллельном режиме
            futures = []
            for strategy in strategies:
                future = self.executor.submit(self.process_strategy, strategy, data)
                futures.append((strategy.__class__.__name__, future))

            # Собираем результаты
            for name, future in futures:
                results[name] = future.result()

            return results

        except Exception as e:
            logger.error(f"Error processing multiple strategies: {str(e)}")
            return {}

    def process_with_optimization(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """
        Параллельная обработка с оптимизацией.

        Args:
            strategy: Стратегия для обработки
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
            futures = []
            for params in param_combinations:
                param_dict = dict(zip(param_names, params))
                future = self.executor.submit(
                    self._evaluate_params, strategy, data, param_dict
                )
                futures.append(future)

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
            logger.error(f"Error processing with optimization: {str(e)}")
            return {}

    def _evaluate_params(
        self, strategy: BaseStrategy, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Оценка параметров стратегии.

        Args:
            strategy: Стратегия для оценки
            data: DataFrame с данными
            params: Параметры для оценки

        Returns:
            Dict с результатами оценки или None
        """
        try:
            # Устанавливаем параметры
            strategy.config = type(strategy.config)(**params)

            # Запускаем обработку
            results = self.process_strategy(strategy, data)

            # Рассчитываем метрики
            if not results:
                return None

            # Рассчитываем score
            score = self._calculate_score(results)

            return {"params": params, "results": results, "score": score}

        except Exception as e:
            logger.error(f"Error evaluating parameters: {str(e)}")
            return None

    def _calculate_score(self, results: List[Any]) -> float:
        """
        Расчет score для результатов.

        Args:
            results: Список результатов

        Returns:
            float: Score
        """
        try:
            # Рассчитываем базовые метрики
            n_trades = len(results)
            if n_trades == 0:
                return 0.0

            profits = [r.profit for r in results if hasattr(r, "profit")]
            if not profits:
                return 0.0

            # Рассчитываем score
            win_rate = len([p for p in profits if p > 0]) / len(profits)
            avg_profit = np.mean(profits)
            std_profit = np.std(profits)

            # Комбинируем метрики
            score = win_rate * avg_profit / (std_profit + 1e-6)

            return score

        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            return 0.0

    def close(self):
        """Закрытие исполнителя"""
        try:
            self.executor.shutdown(wait=True)

        except Exception as e:
            logger.error(f"Error closing executor: {str(e)}")

    def __enter__(self):
        """Контекстный менеджер: вход"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Контекстный менеджер: выход"""
        self.close()
