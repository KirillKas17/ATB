"""
Комплексная система бенчмарков для отслеживания производительности.
Создает базовые показатели и отслеживает регрессии производительности.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import statistics
import warnings
from dataclasses import dataclass, asdict
import hashlib

# Импорты для бенчмарков
from infrastructure.core.feature_engineering import FeatureEngineer, FeatureConfig
from infrastructure.strategies.trend_strategies import TrendStrategy


@dataclass
class BenchmarkResult:
    """Результат бенчмарка."""

    name: str
    execution_time: float
    memory_usage: float
    throughput: float
    cpu_usage: float
    timestamp: str
    data_size: int
    parameters: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь."""
        return asdict(self)


@dataclass
class BenchmarkBaseline:
    """Базовая линия производительности."""

    name: str
    avg_execution_time: float
    max_execution_time: float
    avg_memory_usage: float
    max_memory_usage: float
    avg_throughput: float
    min_throughput: float
    confidence_interval: Tuple[float, float]
    sample_count: int
    last_updated: str


class BenchmarkRunner:
    """Исполнитель бенчмарков с анализом производительности."""

    def __init__(self, baseline_file: str = "tests/benchmarks/baselines.json"):
        self.baseline_file = baseline_file
        self.baselines: Dict[str, BenchmarkBaseline] = {}
        self.results: List[BenchmarkResult] = []
        self.load_baselines()

    def load_baselines(self):
        """Загружает базовые показатели из файла."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, "r") as f:
                    data = json.load(f)
                    self.baselines = {name: BenchmarkBaseline(**baseline_data) for name, baseline_data in data.items()}
            except Exception as e:
                print(f"Error loading baselines: {e}")

    def save_baselines(self):
        """Сохраняет базовые показатели в файл."""
        os.makedirs(os.path.dirname(self.baseline_file), exist_ok=True)

        baseline_data = {name: asdict(baseline) for name, baseline in self.baselines.items()}

        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

    def run_benchmark(self, name: str, func, *args, **kwargs) -> BenchmarkResult:
        """Выполняет бенчмарк функции."""
        import psutil
        import gc

        # Подготовка
        gc.collect()
        process = psutil.Process()

        # Замеры до выполнения
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = process.cpu_percent()
        start_time = time.perf_counter()

        # Выполнение функции
        result = func(*args, **kwargs)

        # Замеры после выполнения
        end_time = time.perf_counter()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = process.cpu_percent()

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = (start_cpu + end_cpu) / 2

        # Рассчитываем пропускную способность
        data_size = self._estimate_data_size(args, kwargs)
        throughput = data_size / execution_time if execution_time > 0 else 0

        benchmark_result = BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_usage=memory_usage,
            throughput=throughput,
            cpu_usage=cpu_usage,
            timestamp=datetime.now().isoformat(),
            data_size=data_size,
            parameters=self._extract_parameters(args, kwargs),
        )

        self.results.append(benchmark_result)
        return benchmark_result

    def _estimate_data_size(self, args, kwargs) -> int:
        """Оценивает размер обрабатываемых данных."""
        total_size = 0

        for arg in args:
            if isinstance(arg, pd.DataFrame):
                total_size += len(arg)
            elif isinstance(arg, (list, tuple, np.ndarray)):
                total_size += len(arg)

        for value in kwargs.values():
            if isinstance(value, pd.DataFrame):
                total_size += len(value)
            elif isinstance(value, (list, tuple, np.ndarray)):
                total_size += len(value)

        return max(total_size, 1)  # Минимум 1 для избежания деления на ноль

    def _extract_parameters(self, args, kwargs) -> Dict[str, Any]:
        """Извлекает параметры для анализа."""
        params = {}

        # Добавляем размеры DataFrame
        for i, arg in enumerate(args):
            if isinstance(arg, pd.DataFrame):
                params[f"dataframe_{i}_shape"] = arg.shape

        # Добавляем значимые параметры из kwargs
        for key, value in kwargs.items():
            if isinstance(value, (int, float, str, bool)):
                params[key] = value
            elif isinstance(value, pd.DataFrame):
                params[f"{key}_shape"] = value.shape

        return params

    def update_baseline(self, benchmark_name: str, results: List[BenchmarkResult]):
        """Обновляет базовую линию на основе результатов."""
        if not results:
            return

        execution_times = [r.execution_time for r in results]
        memory_usages = [r.memory_usage for r in results]
        throughputs = [r.throughput for r in results]

        # Рассчитываем доверительный интервал
        mean_time = statistics.mean(execution_times)
        if len(execution_times) > 1:
            stdev_time = statistics.stdev(execution_times)
            confidence_interval = (
                mean_time - 1.96 * stdev_time / np.sqrt(len(execution_times)),
                mean_time + 1.96 * stdev_time / np.sqrt(len(execution_times)),
            )
        else:
            confidence_interval = (mean_time, mean_time)

        baseline = BenchmarkBaseline(
            name=benchmark_name,
            avg_execution_time=mean_time,
            max_execution_time=max(execution_times),
            avg_memory_usage=statistics.mean(memory_usages),
            max_memory_usage=max(memory_usages),
            avg_throughput=statistics.mean(throughputs),
            min_throughput=min(throughputs),
            confidence_interval=confidence_interval,
            sample_count=len(results),
            last_updated=datetime.now().isoformat(),
        )

        self.baselines[benchmark_name] = baseline
        self.save_baselines()

    def check_regression(self, result: BenchmarkResult, threshold: float = 1.5) -> Optional[str]:
        """Проверяет наличие регрессии производительности."""
        if result.name not in self.baselines:
            return None

        baseline = self.baselines[result.name]

        # Проверяем время выполнения
        if result.execution_time > baseline.avg_execution_time * threshold:
            return f"Performance regression detected: {result.execution_time:.3f}s > {baseline.avg_execution_time * threshold:.3f}s (baseline * {threshold})"

        # Проверяем пропускную способность
        if result.throughput < baseline.avg_throughput / threshold:
            return f"Throughput regression detected: {result.throughput:.0f} < {baseline.avg_throughput / threshold:.0f} (baseline / {threshold})"

        # Проверяем использование памяти
        if result.memory_usage > baseline.avg_memory_usage * (threshold + 0.5):  # Более мягкий порог для памяти
            return f"Memory usage regression detected: {result.memory_usage:.1f}MB > {baseline.avg_memory_usage * (threshold + 0.5):.1f}MB"

        return None

    def generate_report(self) -> Dict[str, Any]:
        """Генерирует отчет о бенчмарках."""
        if not self.results:
            return {"error": "No benchmark results available"}

        # Группируем результаты по именам
        grouped_results = {}
        for result in self.results:
            if result.name not in grouped_results:
                grouped_results[result.name] = []
            grouped_results[result.name].append(result)

        report = {
            "summary": {
                "total_benchmarks": len(grouped_results),
                "total_runs": len(self.results),
                "report_time": datetime.now().isoformat(),
            },
            "benchmarks": {},
            "regressions": [],
        }

        # Анализируем каждый бенчмарк
        for name, results in grouped_results.items():
            execution_times = [r.execution_time for r in results]
            memory_usages = [r.memory_usage for r in results]
            throughputs = [r.throughput for r in results]

            benchmark_info = {
                "runs": len(results),
                "avg_execution_time": statistics.mean(execution_times),
                "min_execution_time": min(execution_times),
                "max_execution_time": max(execution_times),
                "avg_memory_usage": statistics.mean(memory_usages),
                "avg_throughput": statistics.mean(throughputs),
                "latest_run": results[-1].to_dict(),
            }

            # Проверяем регрессии
            for result in results:
                regression_msg = self.check_regression(result)
                if regression_msg:
                    report["regressions"].append(
                        {"benchmark": name, "message": regression_msg, "timestamp": result.timestamp}
                    )

            report["benchmarks"][name] = benchmark_info

        return report


class TestBenchmarkSuite:
    """Комплексная система бенчмарков."""

    @pytest.fixture
    def benchmark_runner(self) -> BenchmarkRunner:
        """Создает исполнитель бенчмарков."""
        return BenchmarkRunner()

    @pytest.fixture
    def small_dataset(self) -> pd.DataFrame:
        """Малый датасет для быстрых бенчмарков."""
        np.random.seed(42)
        size = 1000

        return pd.DataFrame(
            {
                "open": np.random.uniform(99, 101, size),
                "high": np.random.uniform(100, 102, size),
                "low": np.random.uniform(98, 100, size),
                "close": np.random.uniform(99, 101, size),
                "volume": np.random.uniform(1000, 5000, size),
            }
        )

    @pytest.fixture
    def medium_dataset(self) -> pd.DataFrame:
        """Средний датасет для стандартных бенчмарков."""
        np.random.seed(42)
        size = 10000

        return pd.DataFrame(
            {
                "open": np.random.uniform(99, 101, size),
                "high": np.random.uniform(100, 102, size),
                "low": np.random.uniform(98, 100, size),
                "close": np.random.uniform(99, 101, size),
                "volume": np.random.uniform(1000, 5000, size),
            }
        )

    @pytest.fixture
    def large_dataset(self) -> pd.DataFrame:
        """Большой датасет для стресс-тестов."""
        np.random.seed(42)
        size = 50000

        return pd.DataFrame(
            {
                "open": np.random.uniform(99, 101, size),
                "high": np.random.uniform(100, 102, size),
                "low": np.random.uniform(98, 100, size),
                "close": np.random.uniform(99, 101, size),
                "volume": np.random.uniform(1000, 5000, size),
            }
        )

    def test_feature_engineering_benchmarks(
        self,
        benchmark_runner: BenchmarkRunner,
        small_dataset: pd.DataFrame,
        medium_dataset: pd.DataFrame,
        large_dataset: pd.DataFrame,
    ):
        """Бенчмарки инженерии признаков для различных размеров данных."""

        configs = {
            "minimal": FeatureConfig(
                use_technical_indicators=True,
                use_statistical_features=False,
                use_time_features=False,
                ema_periods=[12, 26],
                rsi_periods=[14],
            ),
            "standard": FeatureConfig(
                use_technical_indicators=True,
                use_statistical_features=True,
                use_time_features=False,
                ema_periods=[5, 10, 20, 50],
                rsi_periods=[14, 21],
            ),
            "comprehensive": FeatureConfig(
                use_technical_indicators=True,
                use_statistical_features=True,
                use_time_features=True,
                ema_periods=[5, 10, 20, 50],
                rsi_periods=[14, 21],
                rolling_windows=[5, 10, 20, 50],
            ),
        }

        datasets = {"small": small_dataset, "medium": medium_dataset, "large": large_dataset}

        # Выполняем бенчмарки для всех комбинаций
        for config_name, config in configs.items():
            for dataset_name, dataset in datasets.items():
                engineer = FeatureEngineer(config=config)

                benchmark_name = f"feature_engineering_{config_name}_{dataset_name}"

                # Выполняем несколько прогонов для статистики
                results = []
                for run in range(3):
                    result = benchmark_runner.run_benchmark(benchmark_name, engineer.generate_features, dataset)
                    results.append(result)

                # Обновляем базовую линию
                benchmark_runner.update_baseline(benchmark_name, results)

                # Проверяем последний результат на регрессии
                regression_msg = benchmark_runner.check_regression(results[-1])
                if regression_msg:
                    print(f"REGRESSION in {benchmark_name}: {regression_msg}")

                # Выводим статистику
                avg_time = statistics.mean([r.execution_time for r in results])
                avg_throughput = statistics.mean([r.throughput for r in results])
                print(f"{benchmark_name}: {avg_time:.3f}s avg, {avg_throughput:.0f} records/sec")

    def test_strategy_benchmarks(
        self,
        benchmark_runner: BenchmarkRunner,
        small_dataset: pd.DataFrame,
        medium_dataset: pd.DataFrame,
        large_dataset: pd.DataFrame,
    ):
        """Бенчмарки торговых стратегий."""

        strategy_configs = {
            "fast_trend": {"ema_fast": 5, "ema_slow": 15},
            "standard_trend": {"ema_fast": 12, "ema_slow": 26},
            "slow_trend": {"ema_fast": 21, "ema_slow": 50},
        }

        datasets = {"small": small_dataset, "medium": medium_dataset, "large": large_dataset}

        for strategy_name, strategy_config in strategy_configs.items():
            for dataset_name, dataset in datasets.items():
                strategy = TrendStrategy(strategy_config)

                benchmark_name = f"strategy_{strategy_name}_{dataset_name}"

                # Выполняем бенчмарк
                results = []
                for run in range(3):
                    result = benchmark_runner.run_benchmark(benchmark_name, strategy.generate_signals, dataset)
                    results.append(result)

                # Обновляем базовую линию
                benchmark_runner.update_baseline(benchmark_name, results)

                # Проверяем регрессии
                regression_msg = benchmark_runner.check_regression(results[-1])
                if regression_msg:
                    print(f"REGRESSION in {benchmark_name}: {regression_msg}")

                # Статистика
                avg_time = statistics.mean([r.execution_time for r in results])
                avg_throughput = statistics.mean([r.throughput for r in results])
                print(f"{benchmark_name}: {avg_time:.3f}s avg, {avg_throughput:.0f} records/sec")

    def test_pandas_operations_benchmarks(self, benchmark_runner: BenchmarkRunner, medium_dataset: pd.DataFrame):
        """Бенчмарки основных операций pandas."""

        operations = {
            "rolling_mean_20": lambda df: df["close"].rolling(window=20).mean(),
            "rolling_std_20": lambda df: df["close"].rolling(window=20).std(),
            "ewm_12": lambda df: df["close"].ewm(span=12).mean(),
            "pct_change": lambda df: df["close"].pct_change(),
            "groupby_hour_mean": lambda df: df.groupby(df.index.hour if hasattr(df.index, "hour") else 0).mean(),
            "resample_5min_ohlc": lambda df: (
                df["close"].resample("5min").ohlc() if hasattr(df.index, "freq") else df["close"].head(100)
            ),
            "fillna_ffill": lambda df: df.fillna(method="ffill"),
            "sort_by_volume": lambda df: df.sort_values("volume"),
        }

        for op_name, operation in operations.items():
            benchmark_name = f"pandas_{op_name}"

            # Выполняем бенчмарк
            results = []
            for run in range(5):  # Больше прогонов для операций pandas
                try:
                    result = benchmark_runner.run_benchmark(benchmark_name, operation, medium_dataset)
                    results.append(result)
                except Exception as e:
                    print(f"Error in {op_name}: {e}")
                    continue

            if results:
                # Обновляем базовую линию
                benchmark_runner.update_baseline(benchmark_name, results)

                # Проверяем регрессии
                regression_msg = benchmark_runner.check_regression(results[-1])
                if regression_msg:
                    print(f"REGRESSION in {benchmark_name}: {regression_msg}")

                # Статистика
                avg_time = statistics.mean([r.execution_time for r in results])
                print(f"{benchmark_name}: {avg_time:.6f}s avg")

    def test_numpy_operations_benchmarks(self, benchmark_runner: BenchmarkRunner, medium_dataset: pd.DataFrame):
        """Бенчмарки операций numpy."""

        data_array = medium_dataset["close"].values

        operations = {
            "mean": lambda arr: np.mean(arr),
            "std": lambda arr: np.std(arr),
            "sort": lambda arr: np.sort(arr),
            "argsort": lambda arr: np.argsort(arr),
            "diff": lambda arr: np.diff(arr),
            "gradient": lambda arr: np.gradient(arr),
            "percentile_quartiles": lambda arr: np.percentile(arr, [25, 50, 75]),
            "correlate_lag1": lambda arr: np.correlate(arr[:-1], arr[1:], mode="valid"),
            "fft": lambda arr: np.fft.fft(arr),
            "convolve_smooth": lambda arr: np.convolve(arr, np.array([0.25, 0.5, 0.25]), mode="valid"),
        }

        for op_name, operation in operations.items():
            benchmark_name = f"numpy_{op_name}"

            # Выполняем бенчмарк
            results = []
            for run in range(10):  # Больше прогонов для быстрых numpy операций
                try:
                    result = benchmark_runner.run_benchmark(
                        benchmark_name, operation, data_array.copy()  # Копируем для избежания побочных эффектов
                    )
                    results.append(result)
                except Exception as e:
                    print(f"Error in {op_name}: {e}")
                    continue

            if results:
                # Обновляем базовую линию
                benchmark_runner.update_baseline(benchmark_name, results)

                # Проверяем регрессии
                regression_msg = benchmark_runner.check_regression(results[-1])
                if regression_msg:
                    print(f"REGRESSION in {benchmark_name}: {regression_msg}")

                # Статистика
                avg_time = statistics.mean([r.execution_time for r in results])
                print(f"{benchmark_name}: {avg_time:.6f}s avg")

    def test_scaling_benchmarks(self, benchmark_runner: BenchmarkRunner):
        """Бенчмарки масштабируемости для различных размеров данных."""

        sizes = [1000, 2000, 5000, 10000, 20000, 50000]

        config = FeatureConfig(use_technical_indicators=True)
        engineer = FeatureEngineer(config=config)

        scaling_results = []

        for size in sizes:
            # Создаем датасет нужного размера
            np.random.seed(42)
            dataset = pd.DataFrame(
                {
                    "open": np.random.uniform(99, 101, size),
                    "high": np.random.uniform(100, 102, size),
                    "low": np.random.uniform(98, 100, size),
                    "close": np.random.uniform(99, 101, size),
                    "volume": np.random.uniform(1000, 5000, size),
                }
            )

            benchmark_name = f"scaling_feature_engineering_{size}"

            # Выполняем бенчмарк
            result = benchmark_runner.run_benchmark(benchmark_name, engineer.generate_features, dataset)

            scaling_results.append(
                {
                    "size": size,
                    "time": result.execution_time,
                    "throughput": result.throughput,
                    "memory": result.memory_usage,
                }
            )

            print(f"Size {size}: {result.execution_time:.3f}s, {result.throughput:.0f} records/sec")

        # Анализируем масштабируемость
        self._analyze_scaling_performance(scaling_results)

    def _analyze_scaling_performance(self, results: List[Dict[str, Any]]):
        """Анализирует производительность масштабирования."""
        if len(results) < 3:
            return

        sizes = [r["size"] for r in results]
        times = [r["time"] for r in results]

        # Проверяем линейность роста времени
        # Рассчитываем коэффициент корреляции между размером и временем
        size_normalized = [(s - min(sizes)) / (max(sizes) - min(sizes)) for s in sizes]
        time_normalized = [(t - min(times)) / (max(times) - min(times)) for t in times]

        correlation = np.corrcoef(size_normalized, time_normalized)[0, 1]

        print(f"\nScaling Analysis:")
        print(f"Size-Time correlation: {correlation:.3f}")

        if correlation > 0.95:
            print("✓ Excellent linear scaling")
        elif correlation > 0.85:
            print("⚠ Good scaling with some overhead")
        elif correlation > 0.7:
            print("⚠ Moderate scaling - investigate bottlenecks")
        else:
            print("❌ Poor scaling - significant optimization needed")

        # Проверяем рост времени
        time_ratios = []
        for i in range(1, len(results)):
            size_ratio = results[i]["size"] / results[i - 1]["size"]
            time_ratio = results[i]["time"] / results[i - 1]["time"]
            complexity_ratio = time_ratio / size_ratio
            time_ratios.append(complexity_ratio)

        avg_complexity = np.mean(time_ratios)

        print(f"Average complexity ratio: {avg_complexity:.2f}")

        if avg_complexity < 1.2:
            print("✓ Sub-linear or linear complexity")
        elif avg_complexity < 2.0:
            print("⚠ Slightly super-linear complexity")
        else:
            print("❌ Poor algorithmic complexity detected")

    def test_generate_comprehensive_benchmark_report(self, benchmark_runner: BenchmarkRunner):
        """Генерирует комплексный отчет по всем бенчмаркам."""
        report = benchmark_runner.generate_report()

        print("\n" + "=" * 70)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("=" * 70)

        # Общая статистика
        summary = report.get("summary", {})
        print(f"Total benchmarks: {summary.get('total_benchmarks', 0)}")
        print(f"Total runs: {summary.get('total_runs', 0)}")
        print(f"Report generated: {summary.get('report_time', 'Unknown')}")

        # Результаты бенчмарков
        benchmarks = report.get("benchmarks", {})
        if benchmarks:
            print(f"\nBENCHMARK RESULTS:")
            print("-" * 70)
            print(f"{'Benchmark':<40} {'Avg Time (s)':<12} {'Throughput':<12} {'Memory (MB)':<12}")
            print("-" * 70)

            for name, info in sorted(benchmarks.items()):
                avg_time = info.get("avg_execution_time", 0)
                throughput = info.get("avg_throughput", 0)
                memory = info.get("avg_memory_usage", 0)

                print(f"{name[:39]:<40} {avg_time:<12.4f} {throughput:<12.0f} {memory:<12.1f}")

        # Регрессии
        regressions = report.get("regressions", [])
        if regressions:
            print(f"\nPERFORMANCE REGRESSIONS DETECTED:")
            print("-" * 70)
            for regression in regressions:
                print(f"❌ {regression['benchmark']}: {regression['message']}")
        else:
            print(f"\n✓ No performance regressions detected!")

        # Топ самых медленных бенчмарков
        if benchmarks:
            slowest = sorted(benchmarks.items(), key=lambda x: x[1].get("avg_execution_time", 0), reverse=True)[:5]

            print(f"\nTOP 5 SLOWEST BENCHMARKS:")
            print("-" * 40)
            for i, (name, info) in enumerate(slowest, 1):
                time_val = info.get("avg_execution_time", 0)
                print(f"{i}. {name}: {time_val:.4f}s")

        # Сохраняем отчет в файл
        report_file = f"tests/reports/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs(os.path.dirname(report_file), exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nDetailed report saved to: {report_file}")

        assert isinstance(report, dict)
        assert "summary" in report
        assert "benchmarks" in report


class TestPerformanceMonitoring:
    """Тесты для мониторинга производительности в реальном времени."""

    def test_continuous_performance_monitoring(self):
        """Непрерывный мониторинг производительности системы."""
        import psutil
        import threading
        import time
        from collections import deque

        # Буфер для хранения метрик
        metrics_buffer = deque(maxlen=100)
        monitoring_active = threading.Event()
        monitoring_active.set()

        def monitor_system():
            """Мониторинг системных ресурсов."""
            while monitoring_active.is_set():
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()

                metrics_buffer.append(
                    {
                        "timestamp": time.time(),
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory.percent,
                        "memory_available": memory.available / 1024 / 1024,  # MB
                    }
                )

                time.sleep(0.1)

        # Запускаем мониторинг в отдельном потоке
        monitor_thread = threading.Thread(target=monitor_system)
        monitor_thread.start()

        # Выполняем некоторые операции под мониторингом
        config = FeatureConfig(use_technical_indicators=True)
        engineer = FeatureEngineer(config=config)

        # Создаем данные
        np.random.seed(42)
        dataset = pd.DataFrame(
            {
                "open": np.random.uniform(99, 101, 10000),
                "high": np.random.uniform(100, 102, 10000),
                "low": np.random.uniform(98, 100, 10000),
                "close": np.random.uniform(99, 101, 10000),
                "volume": np.random.uniform(1000, 5000, 10000),
            }
        )

        # Выполняем операции
        start_time = time.time()
        for i in range(5):
            features = engineer.generate_features(dataset)
            time.sleep(0.5)  # Небольшая пауза
        end_time = time.time()

        # Останавливаем мониторинг
        monitoring_active.clear()
        monitor_thread.join()

        # Анализируем собранные метрики
        if metrics_buffer:
            cpu_values = [m["cpu_percent"] for m in metrics_buffer]
            memory_values = [m["memory_percent"] for m in metrics_buffer]

            avg_cpu = np.mean(cpu_values)
            max_cpu = np.max(cpu_values)
            avg_memory = np.mean(memory_values)
            max_memory = np.max(memory_values)

            print(f"\nSystem Performance During Test:")
            print(f"Duration: {end_time - start_time:.1f}s")
            print(f"CPU Usage - Average: {avg_cpu:.1f}%, Peak: {max_cpu:.1f}%")
            print(f"Memory Usage - Average: {avg_memory:.1f}%, Peak: {max_memory:.1f}%")

            # Проверяем на проблемы с производительностью
            if max_cpu > 90:
                print("⚠ High CPU usage detected!")
            if max_memory > 85:
                print("⚠ High memory usage detected!")

            assert len(metrics_buffer) > 0
            assert avg_cpu >= 0
            assert avg_memory >= 0
