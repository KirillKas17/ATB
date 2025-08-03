"""
Специальная система для детального измерения времени выполнения функций.
Выявляет конкретные функции, которые затрачивают больше времени, чем ожидается.
"""

import pytest
import pandas as pd
import numpy as np
import time
import functools
import inspect
from typing import Dict, List, Any, Callable
from collections import defaultdict
import warnings


class FunctionTimer:
    """Система для измерения времени выполнения функций."""
    
    def __init__(self):
        self.timing_data = defaultdict(list)
        self.function_stats = {}
        self.thresholds = {}
        
    def set_threshold(self, function_name: str, max_time: float):
        """Устанавливает порог времени выполнения для функции."""
        self.thresholds[function_name] = max_time
    
    def time_function(self, func: Callable) -> Callable:
        """Декоратор для измерения времени выполнения функции."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            
            execution_time = end_time - start_time
            func_name = f"{func.__module__}.{func.__name__}"
            
            self.timing_data[func_name].append({
                'time': execution_time,
                'args_count': len(args),
                'kwargs_count': len(kwargs),
                'timestamp': time.time()
            })
            
            # Проверяем пороги
            threshold = self.thresholds.get(func_name)
            if threshold and execution_time > threshold:
                print(f"⚠ WARNING: {func_name} took {execution_time:.4f}s (threshold: {threshold:.4f}s)")
            
            return result
        return wrapper
    
    def patch_module_functions(self, module, function_names: List[str]):
        """Применяет декоратор ко всем указанным функциям модуля."""
        for func_name in function_names:
            if hasattr(module, func_name):
                original_func = getattr(module, func_name)
                if callable(original_func):
                    timed_func = self.time_function(original_func)
                    setattr(module, func_name, timed_func)
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Анализирует производительность всех измеренных функций."""
        analysis = {}
        
        for func_name, timings in self.timing_data.items():
            times = [t['time'] for t in timings]
            
            if times:
                analysis[func_name] = {
                    'call_count': len(times),
                    'total_time': sum(times),
                    'avg_time': np.mean(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': np.std(times),
                    'p95_time': np.percentile(times, 95),
                    'p99_time': np.percentile(times, 99),
                    'threshold': self.thresholds.get(func_name),
                    'threshold_violations': sum(1 for t in times if self.thresholds.get(func_name) and t > self.thresholds.get(func_name, float('inf')))
                }
        
        return analysis
    
    def get_slow_functions(self, min_time: float = 0.001) -> List[Dict[str, Any]]:
        """Возвращает функции, которые выполняются медленнее указанного времени."""
        analysis = self.analyze_performance()
        slow_functions = []
        
        for func_name, stats in analysis.items():
            if stats['avg_time'] > min_time:
                slow_functions.append({
                    'function': func_name,
                    'avg_time': stats['avg_time'],
                    'max_time': stats['max_time'],
                    'call_count': stats['call_count'],
                    'total_time': stats['total_time']
                })
        
        return sorted(slow_functions, key=lambda x: x['total_time'], reverse=True)
    
    def find_performance_regressions(self, baseline: Dict[str, float]) -> List[Dict[str, Any]]:
        """Находит регрессии производительности по сравнению с базовой линией."""
        analysis = self.analyze_performance()
        regressions = []
        
        for func_name, stats in analysis.items():
            baseline_time = baseline.get(func_name)
            if baseline_time and stats['avg_time'] > baseline_time * 1.5:  # 50% ухудшение
                regressions.append({
                    'function': func_name,
                    'current_time': stats['avg_time'],
                    'baseline_time': baseline_time,
                    'regression_factor': stats['avg_time'] / baseline_time
                })
        
        return sorted(regressions, key=lambda x: x['regression_factor'], reverse=True)


class TestFunctionTiming:
    """Тесты для детального измерения времени выполнения функций."""
    
    @pytest.fixture
    def timer(self) -> FunctionTimer:
        """Создает таймер функций."""
        return FunctionTimer()
    
    @pytest.fixture
    def test_data(self) -> pd.DataFrame:
        """Создает тестовые данные."""
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.uniform(99, 101, 10000),
            'high': np.random.uniform(100, 102, 10000),
            'low': np.random.uniform(98, 100, 10000),
            'close': np.random.uniform(99, 101, 10000),
            'volume': np.random.uniform(1000, 5000, 10000)
        })
    
    def test_pandas_function_timing(self, timer: FunctionTimer, test_data: pd.DataFrame):
        """Измеряет время выполнения отдельных функций pandas."""
        
        # Устанавливаем пороги для операций pandas
        timer.set_threshold('rolling_mean', 0.01)  # 10ms
        timer.set_threshold('groupby_operation', 0.02)  # 20ms
        timer.set_threshold('merge_operation', 0.015)  # 15ms
        timer.set_threshold('sort_operation', 0.02)  # 20ms
        
        print(f"\nTiming pandas functions on {len(test_data)} records...")
        
        # Тестируем отдельные операции с измерением времени
        @timer.time_function
        def rolling_mean(data):
            return data['close'].rolling(window=20).mean()
        
        @timer.time_function
        def ewm_calculation(data):
            return data['close'].ewm(span=12).mean()
        
        @timer.time_function
        def groupby_operation(data):
            return data.groupby(data.index % 100).mean()
        
        @timer.time_function
        def merge_operation(data):
            df1 = data[['open', 'close']]
            df2 = data[['high', 'low']]
            return pd.merge(df1, df2, left_index=True, right_index=True)
        
        @timer.time_function
        def sort_operation(data):
            return data.sort_values('volume')
        
        @timer.time_function
        def correlation_calculation(data):
            return data[['open', 'high', 'low', 'close']].corr()
        
        # Выполняем функции несколько раз для получения статистики
        for i in range(5):
            rolling_mean(test_data)
            ewm_calculation(test_data)
            groupby_operation(test_data)
            merge_operation(test_data)
            sort_operation(test_data)
            correlation_calculation(test_data)
        
        # Анализируем результаты
        analysis = timer.analyze_performance()
        
        print("\nPandas Functions Performance Analysis:")
        print("-" * 60)
        print(f"{'Function':<30} {'Avg (ms)':<10} {'Max (ms)':<10} {'Calls':<8} {'Violations':<10}")
        print("-" * 60)
        
        for func_name, stats in sorted(analysis.items(), key=lambda x: x[1]['avg_time'], reverse=True):
            avg_ms = stats['avg_time'] * 1000
            max_ms = stats['max_time'] * 1000
            violations = stats.get('threshold_violations', 0)
            
            print(f"{func_name.split('.')[-1]:<30} {avg_ms:<10.2f} {max_ms:<10.2f} {stats['call_count']:<8} {violations:<10}")
        
        # Находим медленные функции
        slow_functions = timer.get_slow_functions(min_time=0.005)  # Функции медленнее 5ms
        
        if slow_functions:
            print(f"\nSlow Functions (>5ms average):")
            for func in slow_functions:
                print(f"  {func['function'].split('.')[-1]}: {func['avg_time']*1000:.2f}ms avg, {func['total_time']*1000:.2f}ms total")
        
        assert len(analysis) > 0
    
    def test_numpy_function_timing(self, timer: FunctionTimer, test_data: pd.DataFrame):
        """Измеряет время выполнения функций numpy."""
        
        data_array = test_data['close'].values
        
        # Устанавливаем пороги для операций numpy
        timer.set_threshold('numpy_sort', 0.001)  # 1ms
        timer.set_threshold('numpy_fft', 0.005)  # 5ms
        timer.set_threshold('numpy_correlation', 0.002)  # 2ms
        
        print(f"\nTiming numpy functions on {len(data_array)} elements...")
        
        @timer.time_function
        def numpy_mean(data):
            return np.mean(data)
        
        @timer.time_function
        def numpy_std(data):
            return np.std(data)
        
        @timer.time_function
        def numpy_sort(data):
            return np.sort(data)
        
        @timer.time_function
        def numpy_fft(data):
            return np.fft.fft(data)
        
        @timer.time_function
        def numpy_correlation(data):
            return np.correlate(data[:-1], data[1:], mode='valid')
        
        @timer.time_function
        def numpy_percentiles(data):
            return np.percentile(data, [25, 50, 75, 95, 99])
        
        # Выполняем функции несколько раз
        for i in range(10):
            numpy_mean(data_array)
            numpy_std(data_array)
            numpy_sort(data_array.copy())  # copy для избежания побочных эффектов
            numpy_fft(data_array)
            numpy_correlation(data_array)
            numpy_percentiles(data_array)
        
        # Анализируем результаты
        analysis = timer.analyze_performance()
        
        print("\nNumpy Functions Performance Analysis:")
        print("-" * 60)
        print(f"{'Function':<25} {'Avg (μs)':<10} {'Max (μs)':<10} {'P95 (μs)':<10} {'Calls':<8}")
        print("-" * 60)
        
        numpy_funcs = {k: v for k, v in analysis.items() if 'numpy_' in k}
        for func_name, stats in sorted(numpy_funcs.items(), key=lambda x: x[1]['avg_time'], reverse=True):
            avg_us = stats['avg_time'] * 1_000_000
            max_us = stats['max_time'] * 1_000_000
            p95_us = stats['p95_time'] * 1_000_000
            
            print(f"{func_name.split('.')[-1]:<25} {avg_us:<10.0f} {max_us:<10.0f} {p95_us:<10.0f} {stats['call_count']:<8}")
        
        assert len(numpy_funcs) > 0
    
    def test_custom_function_performance(self, timer: FunctionTimer):
        """Тестирует производительность пользовательских функций."""
        
        print("\nTesting custom function performance...")
        
        # Устанавливаем строгие пороги для демонстрации
        timer.set_threshold('efficient_calculation', 0.0001)  # 0.1ms
        timer.set_threshold('inefficient_calculation', 0.001)  # 1ms
        
        @timer.time_function
        def efficient_calculation(n):
            """Эффективная реализация."""
            return np.sum(np.arange(n))
        
        @timer.time_function
        def inefficient_calculation(n):
            """Неэффективная реализация."""
            total = 0
            for i in range(n):
                total += i
            return total
        
        @timer.time_function
        def memory_intensive_operation(size):
            """Операция, интенсивно использующая память."""
            array = np.random.random((size, size))
            return np.sum(array)
        
        # Тестируем различные размеры данных
        sizes = [1000, 5000, 10000]
        
        for size in sizes:
            efficient_calculation(size)
            inefficient_calculation(size)
            
            # Для memory-intensive операций используем меньшие размеры
            if size <= 1000:
                memory_intensive_operation(min(size // 10, 100))
        
        # Анализируем результаты
        analysis = timer.analyze_performance()
        
        print("\nCustom Functions Performance Analysis:")
        print("-" * 70)
        
        custom_funcs = {k: v for k, v in analysis.items() if any(name in k for name in ['efficient', 'inefficient', 'memory_intensive'])}
        
        for func_name, stats in custom_funcs.items():
            short_name = func_name.split('.')[-1]
            avg_time = stats['avg_time']
            threshold = stats.get('threshold')
            violations = stats.get('threshold_violations', 0)
            
            status = "✓" if violations == 0 else f"❌ ({violations} violations)"
            threshold_str = f"{threshold*1000:.1f}ms" if threshold else "N/A"
            
            print(f"{short_name:<25}: {avg_time*1000:>6.2f}ms avg (threshold: {threshold_str:<8}) {status}")
        
        # Сравниваем эффективные и неэффективные реализации
        efficient_times = analysis.get('test_function_timing.efficient_calculation', {}).get('avg_time', 0)
        inefficient_times = analysis.get('test_function_timing.inefficient_calculation', {}).get('avg_time', 0)
        
        if efficient_times and inefficient_times:
            speedup = inefficient_times / efficient_times
            print(f"\nEfficient implementation is {speedup:.1f}x faster than inefficient one")
        
        assert len(custom_funcs) > 0
    
    def test_performance_regression_detection(self, timer: FunctionTimer):
        """Тестирует обнаружение регрессий производительности."""
        
        print("\nTesting performance regression detection...")
        
        # Базовая линия производительности (гипотетические значения)
        baseline = {
            'test_function_timing.stable_function': 0.001,  # 1ms
            'test_function_timing.regressed_function': 0.002,  # 2ms
            'test_function_timing.improved_function': 0.005,  # 5ms
        }
        
        @timer.time_function
        def stable_function():
            """Функция со стабильной производительностью."""
            time.sleep(0.001)  # Симулируем 1ms работы
            return "stable"
        
        @timer.time_function
        def regressed_function():
            """Функция с регрессией производительности."""
            time.sleep(0.004)  # Симулируем 4ms работы (было 2ms)
            return "regressed"
        
        @timer.time_function
        def improved_function():
            """Функция с улучшенной производительностью."""
            time.sleep(0.002)  # Симулируем 2ms работы (было 5ms)
            return "improved"
        
        # Выполняем функции
        for _ in range(3):
            stable_function()
            regressed_function()
            improved_function()
        
        # Находим регрессии
        regressions = timer.find_performance_regressions(baseline)
        
        print("\nPerformance Regression Analysis:")
        print("-" * 50)
        
        if regressions:
            for regression in regressions:
                func_name = regression['function'].split('.')[-1]
                current = regression['current_time'] * 1000
                baseline_time = regression['baseline_time'] * 1000
                factor = regression['regression_factor']
                
                print(f"❌ {func_name}: {current:.1f}ms (was {baseline_time:.1f}ms) - {factor:.1f}x slower")
        else:
            print("✓ No performance regressions detected")
        
        # Показываем все результаты для сравнения
        analysis = timer.analyze_performance()
        
        print(f"\nAll Functions vs Baseline:")
        print("-" * 40)
        
        for func_name, stats in analysis.items():
            short_name = func_name.split('.')[-1]
            current_time = stats['avg_time'] * 1000
            baseline_time = baseline.get(func_name, 0) * 1000
            
            if baseline_time > 0:
                ratio = current_time / baseline_time
                status = "✓" if ratio <= 1.5 else "❌"
                print(f"{status} {short_name:<20}: {current_time:>6.1f}ms (baseline: {baseline_time:>6.1f}ms, ratio: {ratio:.1f}x)")
        
        assert len(analysis) > 0
    
    def test_generate_timing_report(self, timer: FunctionTimer):
        """Генерирует детальный отчет о времени выполнения функций."""
        
        if not timer.timing_data:
            print("No timing data available - running a quick test...")
            
            @timer.time_function
            def quick_test():
                time.sleep(0.001)
                return "test"
            
            quick_test()
        
        analysis = timer.analyze_performance()
        
        print("\n" + "="*70)
        print("FUNCTION TIMING DETAILED REPORT")
        print("="*70)
        
        if not analysis:
            print("No function timing data available")
            return
        
        # Общая статистика
        total_functions = len(analysis)
        total_calls = sum(stats['call_count'] for stats in analysis.values())
        total_time = sum(stats['total_time'] for stats in analysis.values())
        
        print(f"Total functions analyzed: {total_functions}")
        print(f"Total function calls: {total_calls}")
        print(f"Total execution time: {total_time*1000:.1f}ms")
        print(f"Average time per call: {(total_time/total_calls)*1000:.2f}ms")
        
        # Детальная статистика по функциям
        print(f"\nDETAILED FUNCTION STATISTICS:")
        print("-" * 70)
        print(f"{'Function':<25} {'Calls':<8} {'Total(ms)':<10} {'Avg(ms)':<10} {'Max(ms)':<10} {'P95(ms)':<10}")
        print("-" * 70)
        
        for func_name, stats in sorted(analysis.items(), key=lambda x: x[1]['total_time'], reverse=True):
            short_name = func_name.split('.')[-1][:24]
            calls = stats['call_count']
            total_ms = stats['total_time'] * 1000
            avg_ms = stats['avg_time'] * 1000
            max_ms = stats['max_time'] * 1000
            p95_ms = stats['p95_time'] * 1000
            
            print(f"{short_name:<25} {calls:<8} {total_ms:<10.1f} {avg_ms:<10.2f} {max_ms:<10.2f} {p95_ms:<10.2f}")
        
        # Выявляем проблемные функции
        print(f"\nPROBLEMATIC FUNCTIONS:")
        print("-" * 40)
        
        problems_found = False
        
        # Функции с высоким разбросом времени выполнения
        high_variance_funcs = [
            (name, stats) for name, stats in analysis.items()
            if stats['std_time'] > stats['avg_time'] * 0.5  # Стандартное отклонение > 50% от среднего
        ]
        
        if high_variance_funcs:
            problems_found = True
            print("High variance functions (inconsistent performance):")
            for func_name, stats in high_variance_funcs:
                short_name = func_name.split('.')[-1]
                variance_ratio = stats['std_time'] / stats['avg_time']
                print(f"  ⚠ {short_name}: variance ratio {variance_ratio:.1f}")
        
        # Функции с нарушениями порогов
        threshold_violations = [
            (name, stats) for name, stats in analysis.items()
            if stats.get('threshold_violations', 0) > 0
        ]
        
        if threshold_violations:
            problems_found = True
            print("Functions exceeding thresholds:")
            for func_name, stats in threshold_violations:
                short_name = func_name.split('.')[-1]
                violations = stats['threshold_violations']
                total_calls = stats['call_count']
                print(f"  ❌ {short_name}: {violations}/{total_calls} calls exceeded threshold")
        
        # Медленные функции
        slow_funcs = [
            (name, stats) for name, stats in analysis.items()
            if stats['avg_time'] > 0.01  # Медленнее 10ms
        ]
        
        if slow_funcs:
            problems_found = True
            print("Slow functions (>10ms average):")
            for func_name, stats in slow_funcs:
                short_name = func_name.split('.')[-1]
                avg_ms = stats['avg_time'] * 1000
                print(f"  🐌 {short_name}: {avg_ms:.1f}ms average")
        
        if not problems_found:
            print("✓ No significant performance problems detected!")
        
        # Рекомендации по оптимизации
        print(f"\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)
        
        recommendations = []
        
        for func_name, stats in analysis.items():
            short_name = func_name.split('.')[-1]
            
            if stats['avg_time'] > 0.01:
                recommendations.append(f"Optimize {short_name} - average time > 10ms")
            
            if stats.get('threshold_violations', 0) > stats['call_count'] * 0.1:
                recommendations.append(f"Investigate {short_name} - frequent threshold violations")
            
            if stats['std_time'] > stats['avg_time']:
                recommendations.append(f"Stabilize {short_name} - high performance variance")
        
        if recommendations:
            for i, rec in enumerate(recommendations[:10], 1):  # Топ 10 рекомендаций
                print(f"{i}. {rec}")
        else:
            print("No specific recommendations - performance is good!")
        
        print(f"\nReport generated at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        assert isinstance(analysis, dict)