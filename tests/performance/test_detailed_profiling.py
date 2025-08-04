"""
Детальное профилирование производительности системы.
Выявляет медленные функции и узкие места в коде.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
import time
import cProfile
import pstats
import io
from pstats import SortKey
import tracemalloc
import psutil
import gc
from typing import Dict, List, Any, Tuple, Callable
from contextlib import contextmanager
from functools import wraps
import warnings
from datetime import datetime
import json
import os

# Импорты для профилирования
from infrastructure.core.feature_engineering import FeatureEngineer, FeatureConfig
from infrastructure.strategies.trend_strategies import TrendStrategy


class PerformanceProfiler:
    """Профилировщик производительности с детальной аналитикой."""
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
        self.function_calls = {}
        self.bottlenecks = []
        
    @contextmanager
    def profile_function(self, function_name: str):
        """Контекстный менеджер для профилирования функции."""
        # Замеряем память до выполнения
        tracemalloc.start()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Замеряем время
        start_time = time.perf_counter()
        start_cpu = time.process_time()
        
        try:
            yield
        finally:
            # Финальные замеры
            end_time = time.perf_counter()
            end_cpu = time.process_time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Сохраняем данные
            execution_time = end_time - start_time
            cpu_time = end_cpu - start_cpu
            memory_used = end_memory - start_memory
            peak_memory = peak / 1024 / 1024  # MB
            
            self.timing_data[function_name] = {
                'wall_time': execution_time,
                'cpu_time': cpu_time,
                'memory_used': memory_used,
                'peak_memory': peak_memory,
                'current_memory': current / 1024 / 1024,
                'timestamp': datetime.now().isoformat()
            }
            
            # Определяем узкие места
            if execution_time > 1.0:  # Функции, выполняющиеся дольше 1 секунды
                self.bottlenecks.append({
                    'function': function_name,
                    'time': execution_time,
                    'issue': 'slow_execution'
                })
            
            if memory_used > 100:  # Функции, использующие больше 100 MB
                self.bottlenecks.append({
                    'function': function_name,
                    'memory': memory_used,
                    'issue': 'high_memory_usage'
                })

    def profile_with_cprofile(self, func: Callable, *args, **kwargs) -> Tuple[Any, str]:
        """Профилирование с помощью cProfile."""
        pr = cProfile.Profile()
        pr.enable()
        
        result = func(*args, **kwargs)
        
        pr.disable()
        
        # Анализируем результаты
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.sort_stats(SortKey.CUMULATIVE)
        ps.print_stats(20)  # Топ 20 функций
        
        profile_output = s.getvalue()
        return result, profile_output

    def get_performance_report(self) -> Dict[str, Any]:
        """Генерирует отчет о производительности."""
        return {
            'timing_data': self.timing_data,
            'memory_data': self.memory_data,
            'bottlenecks': self.bottlenecks,
            'summary': self._generate_summary()
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Генерирует сводку по производительности."""
        if not self.timing_data:
            return {}
        
        times = [data['wall_time'] for data in self.timing_data.values()]
        memories = [data['memory_used'] for data in self.timing_data.values()]
        
        return {
            'total_functions_tested': len(self.timing_data),
            'avg_execution_time': np.mean(times),
            'max_execution_time': np.max(times),
            'min_execution_time': np.min(times),
            'avg_memory_usage': np.mean(memories),
            'max_memory_usage': np.max(memories),
            'bottlenecks_count': len(self.bottlenecks),
            'slow_functions': [b for b in self.bottlenecks if b.get('issue') == 'slow_execution'],
            'memory_heavy_functions': [b for b in self.bottlenecks if b.get('issue') == 'high_memory_usage']
        }

    def save_report(self, filename: str = None):
        """Сохраняет отчет в файл."""
        if filename is None:
            filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_performance_report()
        
        os.makedirs('tests/reports', exist_ok=True)
        filepath = os.path.join('tests/reports', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Performance report saved to: {filepath}")
        return filepath


class TestDetailedProfiling:
    """Детальные тесты профилирования производительности."""

    @pytest.fixture
    def profiler(self) -> PerformanceProfiler:
        """Создает экземпляр профилировщика."""
        return PerformanceProfiler()

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """Создает большой датасет для тестирования."""
        np.random.seed(42)
        size = 50000
        
        dates = pd.date_range('2020-01-01', periods=size, freq='1min')
        
        # Генерируем реалистичные данные
        base_price = 100.0
        prices = []
        
        for i in range(size):
            price_change = np.random.normal(0, 0.001) * base_price
            base_price = max(base_price + price_change, 0.01)
            
            high = base_price * (1 + abs(np.random.normal(0, 0.002)))
            low = base_price * (1 - abs(np.random.normal(0, 0.002)))
            open_price = base_price + np.random.normal(0, 0.0005) * base_price
            close = base_price
            volume = np.random.exponential(1000)
            
            prices.append({
                'open': max(open_price, 0.01),
                'high': max(high, max(open_price, close, 0.01)),
                'low': min(low, min(open_price, close)),
                'close': max(close, 0.01),
                'volume': max(volume, 1)
            })
        
        return pd.DataFrame(prices, index=dates)

    def test_feature_engineering_profiling(self, profiler: PerformanceProfiler, large_dataset: pd.DataFrame):
        """Профилирование инженерии признаков."""
        config = FeatureConfig(
            use_technical_indicators=True,
            use_statistical_features=True,
            use_time_features=True,
            ema_periods=[5, 10, 20, 50],
            rsi_periods=[14, 21],
            rolling_windows=[5, 10, 20, 50]
        )
        engineer = FeatureEngineer(config=config)
        
        # Профилируем полную генерацию признаков
        with profiler.profile_function("feature_engineering_full"):
            features = engineer.generate_features(large_dataset)
        
        # Детальное профилирование отдельных этапов
        with profiler.profile_function("technical_indicators"):
            temp_config = FeatureConfig(use_technical_indicators=True, use_statistical_features=False, use_time_features=False)
            temp_engineer = FeatureEngineer(config=temp_config)
            tech_features = temp_engineer.generate_features(large_dataset)
        
        with profiler.profile_function("statistical_features"):
            temp_config = FeatureConfig(use_technical_indicators=False, use_statistical_features=True, use_time_features=False)
            temp_engineer = FeatureEngineer(config=temp_config)
            stat_features = temp_engineer.generate_features(large_dataset)
        
        with profiler.profile_function("time_features"):
            temp_config = FeatureConfig(use_technical_indicators=False, use_statistical_features=False, use_time_features=True)
            temp_engineer = FeatureEngineer(config=temp_config)
            time_features = temp_engineer.generate_features(large_dataset)
        
        # Проверяем результаты
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        print(f"Generated {len(features.columns)} features from {len(large_dataset)} records")
        
        # Анализируем производительность
        timing_data = profiler.timing_data
        slowest_stage = max(timing_data.items(), key=lambda x: x[1]['wall_time'])
        print(f"Slowest stage: {slowest_stage[0]} - {slowest_stage[1]['wall_time']:.2f}s")

    def test_strategy_profiling(self, profiler: PerformanceProfiler, large_dataset: pd.DataFrame):
        """Профилирование торговых стратегий."""
        strategy_config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(strategy_config)
        
        # Профилируем генерацию сигналов
        with profiler.profile_function("strategy_signal_generation"):
            signals = strategy.generate_signals(large_dataset)
        
        # Профилируем отдельные компоненты стратегии
        with profiler.profile_function("strategy_initialization"):
            new_strategy = TrendStrategy(strategy_config)
        
        # Тестируем на разных размерах данных
        data_sizes = [1000, 5000, 10000, 25000]
        for size in data_sizes:
            subset = large_dataset.iloc[:size]
            with profiler.profile_function(f"strategy_signals_{size}"):
                subset_signals = strategy.generate_signals(subset)
        
        assert isinstance(signals, pd.DataFrame)
        print(f"Generated signals for {len(large_dataset)} records")

    def test_pandas_operations_profiling(self, profiler: PerformanceProfiler, large_dataset: pd.DataFrame):
        """Профилирование операций pandas."""
        
        # Тестируем различные операции pandas
        with profiler.profile_function("pandas_rolling_mean"):
            rolling_means = large_dataset['close'].rolling(window=20).mean()
        
        with profiler.profile_function("pandas_ewm"):
            ewm_values = large_dataset['close'].ewm(span=20).mean()
        
        with profiler.profile_function("pandas_pct_change"):
            pct_changes = large_dataset['close'].pct_change()
        
        with profiler.profile_function("pandas_groupby"):
            grouped = large_dataset.groupby(large_dataset.index.hour).mean()
        
        with profiler.profile_function("pandas_merge"):
            df1 = large_dataset[['open', 'close']]
            df2 = large_dataset[['high', 'low', 'volume']]
            merged = pd.merge(df1, df2, left_index=True, right_index=True)
        
        with profiler.profile_function("pandas_fillna"):
            filled = large_dataset.fillna(method='ffill')
        
        print("Pandas operations profiling completed")

    def test_numpy_operations_profiling(self, profiler: PerformanceProfiler, large_dataset: pd.DataFrame):
        """Профилирование операций numpy."""
        data_array = large_dataset['close'].values
        
        with profiler.profile_function("numpy_mean"):
            mean_val = np.mean(data_array)
        
        with profiler.profile_function("numpy_std"):
            std_val = np.std(data_array)
        
        with profiler.profile_function("numpy_correlate"):
            corr = np.correlate(data_array[:-1], data_array[1:], mode='valid')
        
        with profiler.profile_function("numpy_fft"):
            fft_result = np.fft.fft(data_array)
        
        with profiler.profile_function("numpy_gradient"):
            gradient = np.gradient(data_array)
        
        print("Numpy operations profiling completed")

    def test_memory_intensive_operations(self, profiler: PerformanceProfiler):
        """Тестирование операций, интенсивно использующих память."""
        
        # Создание больших массивов
        with profiler.profile_function("large_array_creation"):
            large_array = np.random.random((10000, 1000))
        
        # Операции с большими массивами
        with profiler.profile_function("large_array_operations"):
            result = np.dot(large_array, large_array.T)
        
        # Создание больших DataFrame
        with profiler.profile_function("large_dataframe_creation"):
            large_df = pd.DataFrame(np.random.random((50000, 100)))
        
        # Операции с большими DataFrame
        with profiler.profile_function("large_dataframe_operations"):
            corr_matrix = large_df.corr()
        
        print("Memory intensive operations profiling completed")

    def test_io_operations_profiling(self, profiler: PerformanceProfiler, large_dataset: pd.DataFrame):
        """Профилирование операций ввода-вывода."""
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, 'test_data.csv')
            parquet_path = os.path.join(temp_dir, 'test_data.parquet')
            
            # Профилируем сохранение в CSV
            with profiler.profile_function("save_to_csv"):
                large_dataset.to_csv(csv_path)
            
            # Профилируем чтение из CSV
            with profiler.profile_function("read_from_csv"):
                csv_data = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            
            # Профилируем сохранение в Parquet
            with profiler.profile_function("save_to_parquet"):
                large_dataset.to_parquet(parquet_path)
            
            # Профилируем чтение из Parquet
            with profiler.profile_function("read_from_parquet"):
                parquet_data = pd.read_parquet(parquet_path)
        
        print("I/O operations profiling completed")

    def test_algorithmic_complexity(self, profiler: PerformanceProfiler):
        """Тестирование алгоритмической сложности операций."""
        data_sizes = [1000, 2000, 5000, 10000, 20000]
        
        for size in data_sizes:
            data = np.random.random(size)
            
            # O(n) операции
            with profiler.profile_function(f"linear_sum_{size}"):
                result = np.sum(data)
            
            # O(n log n) операции
            with profiler.profile_function(f"sort_{size}"):
                sorted_data = np.sort(data)
            
            # O(n²) операции (для небольших размеров)
            if size <= 5000:
                with profiler.profile_function(f"nested_loop_{size}"):
                    result_matrix = np.outer(data[:100], data[:100])
        
        print("Algorithmic complexity profiling completed")

    def test_cprofile_detailed_analysis(self, profiler: PerformanceProfiler, large_dataset: pd.DataFrame):
        """Детальный анализ с помощью cProfile."""
        config = FeatureConfig(use_technical_indicators=True)
        engineer = FeatureEngineer(config=config)
        
        # Профилируем с помощью cProfile
        result, profile_output = profiler.profile_with_cprofile(
            engineer.generate_features, large_dataset
        )
        
        # Анализируем вывод cProfile
        lines = profile_output.split('\n')
        slow_functions = []
        
        for line in lines[5:25]:  # Пропускаем заголовки, берем топ 20
            if line.strip() and 'function calls' not in line:
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        cumtime = float(parts[3])
                        function_name = ' '.join(parts[5:])
                        if cumtime > 0.01:  # Функции дольше 0.01 секунды
                            slow_functions.append({
                                'function': function_name,
                                'cumulative_time': cumtime,
                                'calls': int(parts[0]) if parts[0].isdigit() else 0
                            })
                except (ValueError, IndexError):
                    continue
        
        # Сохраняем детальный профиль
        profiler.function_calls['detailed_profile'] = {
            'slow_functions': slow_functions,
            'full_output': profile_output
        }
        
        print(f"CProfile analysis completed. Found {len(slow_functions)} slow functions.")
        
        # Выводим топ медленных функций
        if slow_functions:
            print("\nTop slow functions:")
            for func in slow_functions[:5]:
                print(f"  {func['function']}: {func['cumulative_time']:.3f}s ({func['calls']} calls)")

    def test_generate_performance_report(self, profiler: PerformanceProfiler):
        """Генерирует финальный отчет о производительности."""
        report = profiler.get_performance_report()
        
        print("\n" + "="*50)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*50)
        
        summary = report['summary']
        if summary:
            print(f"Total functions tested: {summary['total_functions_tested']}")
            print(f"Average execution time: {summary['avg_execution_time']:.3f}s")
            print(f"Max execution time: {summary['max_execution_time']:.3f}s")
            print(f"Average memory usage: {summary['avg_memory_usage']:.2f}MB")
            print(f"Max memory usage: {summary['max_memory_usage']:.2f}MB")
            print(f"Bottlenecks found: {summary['bottlenecks_count']}")
        
        if profiler.bottlenecks:
            print("\nBOTTLENECKS DETECTED:")
            for bottleneck in profiler.bottlenecks:
                if bottleneck.get('issue') == 'slow_execution':
                    print(f"  SLOW: {bottleneck['function']} - {bottleneck['time']:.3f}s")
                elif bottleneck.get('issue') == 'high_memory_usage':
                    print(f"  MEMORY: {bottleneck['function']} - {bottleneck['memory']:.2f}MB")
        
        # Сохраняем отчет
        filepath = profiler.save_report()
        print(f"\nDetailed report saved to: {filepath}")
        
        assert isinstance(report, dict)
        assert 'timing_data' in report
        assert 'bottlenecks' in report


class TestPerformanceRegression:
    """Тесты регрессии производительности."""
    
    @pytest.fixture
    def baseline_performance(self) -> Dict[str, float]:
        """Базовые показатели производительности."""
        return {
            'feature_engineering_small': 0.5,  # секунды для 1k записей
            'feature_engineering_medium': 2.0,  # секунды для 10k записей
            'strategy_signals_small': 0.1,     # секунды для 1k записей
            'strategy_signals_medium': 0.5,    # секунды для 10k записей
            'pandas_operations': 0.05,         # секунды для базовых операций
        }

    def test_performance_regression_feature_engineering(self, baseline_performance: Dict[str, float]):
        """Тест регрессии производительности для инженерии признаков."""
        # Малый датасет
        small_data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 1000),
            'high': np.random.uniform(100, 102, 1000),
            'low': np.random.uniform(98, 100, 1000),
            'close': np.random.uniform(99, 101, 1000),
            'volume': np.random.uniform(1000, 5000, 1000)
        })
        
        config = FeatureConfig(use_technical_indicators=True)
        engineer = FeatureEngineer(config=config)
        
        start_time = time.time()
        features = engineer.generate_features(small_data)
        execution_time = time.time() - start_time
        
        baseline = baseline_performance['feature_engineering_small']
        regression_threshold = baseline * 1.5  # Допускаем 50% ухудшение
        
        print(f"Feature engineering (1k records): {execution_time:.3f}s (baseline: {baseline}s)")
        
        assert execution_time < regression_threshold, \
            f"Performance regression detected: {execution_time:.3f}s > {regression_threshold:.3f}s"

    def test_performance_regression_strategy(self, baseline_performance: Dict[str, float]):
        """Тест регрессии производительности для стратегий."""
        small_data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 1000),
            'high': np.random.uniform(100, 102, 1000),
            'low': np.random.uniform(98, 100, 1000),
            'close': np.random.uniform(99, 101, 1000),
            'volume': np.random.uniform(1000, 5000, 1000)
        })
        
        strategy = TrendStrategy({'ema_fast': 12, 'ema_slow': 26})
        
        start_time = time.time()
        signals = strategy.generate_signals(small_data)
        execution_time = time.time() - start_time
        
        baseline = baseline_performance['strategy_signals_small']
        regression_threshold = baseline * 1.5
        
        print(f"Strategy signals (1k records): {execution_time:.3f}s (baseline: {baseline}s)")
        
        assert execution_time < regression_threshold, \
            f"Performance regression detected: {execution_time:.3f}s > {regression_threshold:.3f}s"


class TestPerformanceOptimizations:
    """Тесты для проверки оптимизаций производительности."""
    
    def test_vectorized_vs_loop_operations(self):
        """Сравнение векторизованных операций с циклами."""
        data = np.random.random(100000)
        
        # Метод с циклом (медленный)
        start_time = time.time()
        result_loop = []
        for i in range(len(data) - 1):
            result_loop.append((data[i+1] - data[i]) / data[i])
        loop_time = time.time() - start_time
        
        # Векторизованный метод (быстрый)
        start_time = time.time()
        result_vectorized = np.diff(data) / data[:-1]
        vectorized_time = time.time() - start_time
        
        speedup = loop_time / vectorized_time
        print(f"Loop method: {loop_time:.4f}s")
        print(f"Vectorized method: {vectorized_time:.4f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        # Векторизованный метод должен быть минимум в 10 раз быстрее
        assert speedup > 10, f"Insufficient speedup: {speedup:.1f}x"
        
        # Результаты должны быть одинаковыми (с учетом точности)
        np.testing.assert_array_almost_equal(result_loop, result_vectorized, decimal=10)

    def test_memory_efficient_operations(self):
        """Тестирование эффективности использования памяти."""
        size = 1000000
        
        # Неэффективный метод - создание промежуточных копий
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        data = np.random.random(size)
        temp1 = data * 2
        temp2 = temp1 + 1
        temp3 = temp2 / 3
        result_inefficient = np.sqrt(temp3)
        
        peak_memory_inefficient = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used_inefficient = peak_memory_inefficient - start_memory
        
        # Очищаем память
        del data, temp1, temp2, temp3, result_inefficient
        gc.collect()
        
        # Эффективный метод - операции на месте
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        data = np.random.random(size)
        data *= 2
        data += 1
        data /= 3
        np.sqrt(data, out=data)
        result_efficient = data
        
        peak_memory_efficient = psutil.Process().memory_info().rss / 1024 / 1024
        memory_used_efficient = peak_memory_efficient - start_memory
        
        memory_savings = memory_used_inefficient - memory_used_efficient
        
        print(f"Inefficient memory usage: {memory_used_inefficient:.2f}MB")
        print(f"Efficient memory usage: {memory_used_efficient:.2f}MB")
        print(f"Memory savings: {memory_savings:.2f}MB")
        
        # Эффективный метод должен использовать меньше памяти
        assert memory_used_efficient < memory_used_inefficient, \
            "Efficient method should use less memory"

    def test_caching_effectiveness(self):
        """Тестирование эффективности кэширования."""
        from functools import lru_cache
        
        # Функция без кэширования
        def expensive_calculation_no_cache(n):
            time.sleep(0.01)  # Симулируем дорогую операцию
            return n * n
        
        # Функция с кэшированием
        @lru_cache(maxsize=128)
        def expensive_calculation_with_cache(n):
            time.sleep(0.01)  # Симулируем дорогую операцию
            return n * n
        
        # Тестируем без кэша
        start_time = time.time()
        for _ in range(5):
            for i in range(10):
                expensive_calculation_no_cache(i)
        no_cache_time = time.time() - start_time
        
        # Тестируем с кэшем
        start_time = time.time()
        for _ in range(5):
            for i in range(10):
                expensive_calculation_with_cache(i)
        cache_time = time.time() - start_time
        
        speedup = no_cache_time / cache_time
        
        print(f"Without cache: {no_cache_time:.3f}s")
        print(f"With cache: {cache_time:.3f}s")
        print(f"Speedup: {speedup:.1f}x")
        
        # Кэширование должно дать значительное ускорение
        assert speedup > 3, f"Insufficient caching speedup: {speedup:.1f}x"