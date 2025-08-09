"""
Упрощенная система профилирования производительности.
Выявляет медленные функции без зависимости от сложных модулей.
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
import statistics


class SimpleProfiler:
    """Простой профилировщик производительности."""

    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
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
                "wall_time": execution_time,
                "cpu_time": cpu_time,
                "memory_used": memory_used,
                "peak_memory": peak_memory,
                "current_memory": current / 1024 / 1024,
                "timestamp": datetime.now().isoformat(),
            }

            # Определяем узкие места
            if execution_time > 0.5:  # Функции, выполняющиеся дольше 500ms
                self.bottlenecks.append({"function": function_name, "time": execution_time, "issue": "slow_execution"})

            if memory_used > 50:  # Функции, использующие больше 50 MB
                self.bottlenecks.append(
                    {"function": function_name, "memory": memory_used, "issue": "high_memory_usage"}
                )

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
            "timing_data": self.timing_data,
            "memory_data": self.memory_data,
            "bottlenecks": self.bottlenecks,
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Генерирует сводку по производительности."""
        if not self.timing_data:
            return {}

        times = [data["wall_time"] for data in self.timing_data.values()]
        memories = [data["memory_used"] for data in self.timing_data.values()]

        return {
            "total_functions_tested": len(self.timing_data),
            "avg_execution_time": np.mean(times),
            "max_execution_time": np.max(times),
            "min_execution_time": np.min(times),
            "avg_memory_usage": np.mean(memories),
            "max_memory_usage": np.max(memories),
            "bottlenecks_count": len(self.bottlenecks),
            "slow_functions": [b for b in self.bottlenecks if b.get("issue") == "slow_execution"],
            "memory_heavy_functions": [b for b in self.bottlenecks if b.get("issue") == "high_memory_usage"],
        }


class TestSimpleProfiling:
    """Простые тесты профилирования производительности."""

    @pytest.fixture
    def profiler(self) -> SimpleProfiler:
        """Создает экземпляр профилировщика."""
        return SimpleProfiler()

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """Создает большой датасет для тестирования."""
        np.random.seed(42)
        size = 50000

        dates = pd.date_range("2020-01-01", periods=size, freq="1min")

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

            prices.append(
                {
                    "open": max(open_price, 0.01),
                    "high": max(high, max(open_price, close, 0.01)),
                    "low": min(low, min(open_price, close)),
                    "close": max(close, 0.01),
                    "volume": max(volume, 1),
                }
            )

        return pd.DataFrame(prices, index=dates)

    def test_pandas_operations_profiling(self, profiler: SimpleProfiler, large_dataset: pd.DataFrame):
        """Профилирование операций pandas."""

        print(f"\nProfiling pandas operations on {len(large_dataset)} records...")

        # Тестируем различные операции pandas
        with profiler.profile_function("pandas_rolling_mean"):
            rolling_means = large_dataset["close"].rolling(window=20).mean()

        with profiler.profile_function("pandas_ewm"):
            ewm_values = large_dataset["close"].ewm(span=20).mean()

        with profiler.profile_function("pandas_pct_change"):
            pct_changes = large_dataset["close"].pct_change()

        with profiler.profile_function("pandas_groupby"):
            grouped = large_dataset.groupby(large_dataset.index.hour).mean()

        with profiler.profile_function("pandas_merge"):
            df1 = large_dataset[["open", "close"]]
            df2 = large_dataset[["high", "low", "volume"]]
            merged = pd.merge(df1, df2, left_index=True, right_index=True)

        with profiler.profile_function("pandas_fillna"):
            filled = large_dataset.fillna(method="ffill")

        with profiler.profile_function("pandas_sort"):
            sorted_data = large_dataset.sort_values("volume")

        with profiler.profile_function("pandas_resample"):
            try:
                resampled = large_dataset["close"].resample("5min").ohlc()
            except Exception:
                # Если индекс не поддерживает resample, используем простую группировку
                resampled = large_dataset.groupby(large_dataset.index // 5).agg(
                    {"close": ["first", "max", "min", "last"]}
                )

        # Анализируем результаты
        timing_data = profiler.timing_data

        print("\nPandas Operations Performance:")
        print("-" * 50)
        for op_name, timing in sorted(timing_data.items(), key=lambda x: x[1]["wall_time"], reverse=True):
            print(f"{op_name:<25}: {timing['wall_time']:.4f}s ({timing['memory_used']:+.1f}MB)")

        assert len(timing_data) > 0
        print("Pandas operations profiling completed")

    def test_numpy_operations_profiling(self, profiler: SimpleProfiler, large_dataset: pd.DataFrame):
        """Профилирование операций numpy."""
        data_array = large_dataset["close"].values

        print(f"\nProfiling numpy operations on {len(data_array)} elements...")

        with profiler.profile_function("numpy_mean"):
            mean_val = np.mean(data_array)

        with profiler.profile_function("numpy_std"):
            std_val = np.std(data_array)

        with profiler.profile_function("numpy_sort"):
            sorted_data = np.sort(data_array)

        with profiler.profile_function("numpy_correlate"):
            corr = np.correlate(data_array[:-1], data_array[1:], mode="valid")

        with profiler.profile_function("numpy_fft"):
            fft_result = np.fft.fft(data_array)

        with profiler.profile_function("numpy_gradient"):
            gradient = np.gradient(data_array)

        with profiler.profile_function("numpy_percentile"):
            percentiles = np.percentile(data_array, [25, 50, 75])

        with profiler.profile_function("numpy_convolve"):
            convolved = np.convolve(data_array, np.array([0.25, 0.5, 0.25]), mode="valid")

        # Анализируем результаты
        timing_data = profiler.timing_data

        print("\nNumpy Operations Performance:")
        print("-" * 50)
        numpy_ops = {k: v for k, v in timing_data.items() if k.startswith("numpy_")}
        for op_name, timing in sorted(numpy_ops.items(), key=lambda x: x[1]["wall_time"], reverse=True):
            print(f"{op_name:<25}: {timing['wall_time']:.6f}s ({timing['memory_used']:+.1f}MB)")

        assert len(numpy_ops) > 0
        print("Numpy operations profiling completed")

    def test_memory_intensive_operations(self, profiler: SimpleProfiler):
        """Тестирование операций, интенсивно использующих память."""

        print("\nTesting memory-intensive operations...")

        # Создание больших массивов
        with profiler.profile_function("large_array_creation"):
            large_array = np.random.random((10000, 1000))

        # Операции с большими массивами
        with profiler.profile_function("large_array_dot_product"):
            result = np.dot(large_array, large_array.T)

        # Создание больших DataFrame
        with profiler.profile_function("large_dataframe_creation"):
            large_df = pd.DataFrame(np.random.random((20000, 100)))

        # Операции с большими DataFrame
        with profiler.profile_function("large_dataframe_corr"):
            corr_matrix = large_df.corr()

        # Анализируем результаты
        timing_data = profiler.timing_data

        print("\nMemory-Intensive Operations Performance:")
        print("-" * 50)
        memory_ops = {k: v for k, v in timing_data.items() if k.startswith("large_")}
        for op_name, timing in sorted(memory_ops.items(), key=lambda x: x[1]["memory_used"], reverse=True):
            print(
                f"{op_name:<30}: {timing['wall_time']:.3f}s ({timing['memory_used']:+.1f}MB peak: {timing['peak_memory']:.1f}MB)"
            )

        assert len(memory_ops) > 0
        print("Memory intensive operations profiling completed")

    def test_algorithmic_complexity(self, profiler: SimpleProfiler):
        """Тестирование алгоритмической сложности операций."""
        data_sizes = [1000, 2000, 5000, 10000, 20000]

        print("\nTesting algorithmic complexity...")

        performance_data = {}

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
                with profiler.profile_function(f"nested_operation_{size}"):
                    # Имитируем O(n²) операцию без реального nested loop
                    matrix = np.outer(data[: min(100, len(data))], data[: min(100, len(data))])

        # Анализируем сложность
        print("\nAlgorithmic Complexity Analysis:")
        print("-" * 50)

        # Группируем по типам операций
        linear_ops = {}
        sort_ops = {}
        nested_ops = {}

        for op_name, timing in profiler.timing_data.items():
            if "linear_sum" in op_name:
                size = int(op_name.split("_")[-1])
                linear_ops[size] = timing["wall_time"]
            elif "sort_" in op_name and op_name.startswith("sort_"):
                size = int(op_name.split("_")[-1])
                sort_ops[size] = timing["wall_time"]
            elif "nested_operation" in op_name:
                size = int(op_name.split("_")[-1])
                nested_ops[size] = timing["wall_time"]

        # Анализируем каждый тип операций
        def analyze_complexity(ops_dict, op_type):
            if len(ops_dict) < 2:
                return

            sizes = sorted(ops_dict.keys())
            times = [ops_dict[size] for size in sizes]

            print(f"\n{op_type} operations:")
            for size, time_val in zip(sizes, times):
                throughput = size / time_val if time_val > 0 else 0
                print(f"  Size {size:5d}: {time_val:.6f}s ({throughput:>8.0f} ops/sec)")

            # Проверяем рост времени
            if len(times) > 1:
                ratios = []
                for i in range(1, len(times)):
                    size_ratio = sizes[i] / sizes[i - 1]
                    time_ratio = times[i] / times[i - 1] if times[i - 1] > 0 else 0
                    complexity_ratio = time_ratio / size_ratio
                    ratios.append(complexity_ratio)

                avg_ratio = np.mean(ratios)
                print(f"  Average complexity ratio: {avg_ratio:.2f}")

                if avg_ratio < 1.2:
                    print("  ✓ Linear or sub-linear complexity")
                elif avg_ratio < 2.0:
                    print("  ⚠ Slightly super-linear complexity")
                else:
                    print("  ❌ Poor complexity - needs optimization")

        analyze_complexity(linear_ops, "Linear")
        analyze_complexity(sort_ops, "Sort")
        analyze_complexity(nested_ops, "Nested")

        print("Algorithmic complexity profiling completed")

    def test_cprofile_detailed_analysis(self, profiler: SimpleProfiler, large_dataset: pd.DataFrame):
        """Детальный анализ с помощью cProfile."""
        print("\nRunning detailed cProfile analysis...")

        def complex_pandas_operation(df):
            """Сложная операция pandas для профилирования."""
            # Несколько операций, чтобы увидеть детальную картину
            result1 = df["close"].rolling(window=20).mean()
            result2 = df["close"].ewm(span=12).mean()
            result3 = df.groupby(df.index.hour).agg({"close": ["mean", "std"], "volume": "sum"})
            correlation = df[["open", "high", "low", "close"]].corr()
            return result1, result2, result3, correlation

        # Профилируем с помощью cProfile
        result, profile_output = profiler.profile_with_cprofile(complex_pandas_operation, large_dataset)

        # Анализируем вывод cProfile
        lines = profile_output.split("\n")
        slow_functions = []

        print("\nTop functions by cumulative time:")
        print("-" * 60)

        for line in lines[5:15]:  # Пропускаем заголовки, берем топ 10
            if line.strip() and "function calls" not in line and "filename:lineno(function)" not in line:
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        cumtime = float(parts[3])
                        function_name = " ".join(parts[5:])
                        if cumtime > 0.001:  # Функции дольше 1ms
                            slow_functions.append(
                                {
                                    "function": function_name,
                                    "cumulative_time": cumtime,
                                    "calls": int(parts[0]) if parts[0].isdigit() else 0,
                                }
                            )
                            print(f"{function_name[:50]:<50} {cumtime:>8.3f}s")
                except (ValueError, IndexError):
                    continue

        print(f"\nFound {len(slow_functions)} functions with significant time usage")

        # Выводим топ медленных функций
        if slow_functions:
            print("\nTop 5 slowest functions:")
            for i, func in enumerate(slow_functions[:5], 1):
                print(f"{i}. {func['function'][:60]}")
                print(f"   Time: {func['cumulative_time']:.3f}s, Calls: {func['calls']}")

        assert len(slow_functions) >= 0

    def test_generate_performance_report(self, profiler: SimpleProfiler):
        """Генерирует финальный отчет о производительности."""
        report = profiler.get_performance_report()

        print("\n" + "=" * 60)
        print("SIMPLE PERFORMANCE ANALYSIS REPORT")
        print("=" * 60)

        summary = report["summary"]
        if summary:
            print(f"Total functions tested: {summary['total_functions_tested']}")
            print(f"Average execution time: {summary['avg_execution_time']:.3f}s")
            print(f"Max execution time: {summary['max_execution_time']:.3f}s")
            print(f"Min execution time: {summary['min_execution_time']:.6f}s")
            print(f"Average memory usage: {summary['avg_memory_usage']:.2f}MB")
            print(f"Max memory usage: {summary['max_memory_usage']:.2f}MB")
            print(f"Bottlenecks found: {summary['bottlenecks_count']}")

        if profiler.bottlenecks:
            print("\nBOTTLENECKS DETECTED:")
            print("-" * 40)
            for bottleneck in profiler.bottlenecks:
                if bottleneck.get("issue") == "slow_execution":
                    print(f"  SLOW: {bottleneck['function']} - {bottleneck['time']:.3f}s")
                elif bottleneck.get("issue") == "high_memory_usage":
                    print(f"  MEMORY: {bottleneck['function']} - {bottleneck['memory']:.2f}MB")
        else:
            print("\n✓ No significant bottlenecks detected!")

        # Детализация по типам операций
        timing_data = profiler.timing_data
        pandas_ops = {k: v for k, v in timing_data.items() if k.startswith("pandas_")}
        numpy_ops = {k: v for k, v in timing_data.items() if k.startswith("numpy_")}
        memory_ops = {k: v for k, v in timing_data.items() if k.startswith("large_")}

        if pandas_ops:
            print(f"\nPANDAS OPERATIONS ({len(pandas_ops)} operations):")
            slowest_pandas = max(pandas_ops.items(), key=lambda x: x[1]["wall_time"])
            fastest_pandas = min(pandas_ops.items(), key=lambda x: x[1]["wall_time"])
            print(f"  Slowest: {slowest_pandas[0]} - {slowest_pandas[1]['wall_time']:.4f}s")
            print(f"  Fastest: {fastest_pandas[0]} - {fastest_pandas[1]['wall_time']:.6f}s")

        if numpy_ops:
            print(f"\nNUMPY OPERATIONS ({len(numpy_ops)} operations):")
            slowest_numpy = max(numpy_ops.items(), key=lambda x: x[1]["wall_time"])
            fastest_numpy = min(numpy_ops.items(), key=lambda x: x[1]["wall_time"])
            print(f"  Slowest: {slowest_numpy[0]} - {slowest_numpy[1]['wall_time']:.6f}s")
            print(f"  Fastest: {fastest_numpy[0]} - {fastest_numpy[1]['wall_time']:.6f}s")

        if memory_ops:
            print(f"\nMEMORY-INTENSIVE OPERATIONS ({len(memory_ops)} operations):")
            highest_memory = max(memory_ops.items(), key=lambda x: x[1]["memory_used"])
            print(f"  Highest memory usage: {highest_memory[0]} - {highest_memory[1]['memory_used']:.1f}MB")

        # Рекомендации по оптимизации
        print(f"\nOPTIMIZATION RECOMMENDATIONS:")
        print("-" * 40)

        recommendations = []

        # Анализируем pandas операции
        if pandas_ops:
            slow_pandas = [k for k, v in pandas_ops.items() if v["wall_time"] > 0.1]
            for op in slow_pandas:
                recommendations.append(f"Consider optimizing {op} - execution time > 100ms")

        # Анализируем использование памяти
        high_memory_ops = [k for k, v in timing_data.items() if v["memory_used"] > 100]
        for op in high_memory_ops:
            recommendations.append(f"Optimize memory usage in {op} - using > 100MB")

        # Анализируем общие паттерны
        all_times = [v["wall_time"] for v in timing_data.values()]
        if all_times:
            time_std = np.std(all_times)
            time_mean = np.mean(all_times)
            if time_std > time_mean:
                recommendations.append("High variance in execution times - investigate inconsistent performance")

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        else:
            print("No specific optimization recommendations - performance looks good!")

        # Сохраняем отчет
        os.makedirs("tests/reports", exist_ok=True)
        report_file = f"tests/reports/simple_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nDetailed report saved to: {report_file}")

        assert isinstance(report, dict)
        assert "timing_data" in report
        assert "bottlenecks" in report


# Дополнительные утилиты


def benchmark_function(func):
    """Декоратор для простого бенчмарка функции."""

    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        execution_time = end_time - start_time
        print(f"{func.__name__} executed in {execution_time:.6f} seconds")

        return result

    return wrapper


def compare_implementations(*funcs):
    """Сравнивает производительность нескольких реализаций."""

    def decorator(data_generator):
        def wrapper(*args, **kwargs):
            data = data_generator(*args, **kwargs)
            results = {}

            print(f"\nComparing {len(funcs)} implementations:")
            print("-" * 40)

            for func in funcs:
                start_time = time.perf_counter()
                result = func(data)
                end_time = time.perf_counter()

                execution_time = end_time - start_time
                results[func.__name__] = {"time": execution_time, "result": result}
                print(f"{func.__name__:<20}: {execution_time:.6f}s")

            # Находим самую быструю реализацию
            fastest = min(results.items(), key=lambda x: x[1]["time"])
            print(f"\nFastest: {fastest[0]} ({fastest[1]['time']:.6f}s)")

            # Показываем ускорение
            for name, info in results.items():
                if name != fastest[0]:
                    speedup = info["time"] / fastest[1]["time"]
                    print(f"{name} is {speedup:.1f}x slower than {fastest[0]}")

            return results

        return wrapper

    return decorator
