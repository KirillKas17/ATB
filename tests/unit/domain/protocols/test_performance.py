"""
Unit тесты для domain/protocols/performance.py.

Покрывает:
- PerformanceProfiler
- BenchmarkRunner
- PerformanceOptimizer
- Декораторы профилирования
- Утилиты производительности
- Обработку ошибок
"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Tuple
from shared.numpy_utils import np
from datetime import datetime, timedelta

from domain.protocols.performance import (
    PerformanceMetric,
    PerformanceProtocol,
    PerformanceProfile,
    BenchmarkResult,
    OptimizationSuggestion,
    PerformanceProfiler,
    BenchmarkRunner,
    PerformanceOptimizer,
    profile_performance,
    benchmark_performance,
    optimize_performance,
    get_performance_report,
    optimize_slow_functions,
    run_performance_benchmarks,
    get_system_performance,
    monitor_performance_continuously,
)


class TestPerformanceMetric:
    """Тесты для PerformanceMetric."""

    def test_enum_values(self):
        """Тест значений enum."""
        assert PerformanceMetric.EXECUTION_TIME.value == "execution_time"
        assert PerformanceMetric.MEMORY_USAGE.value == "memory_usage"
        assert PerformanceMetric.CPU_USAGE.value == "cpu_usage"
        assert PerformanceMetric.CALL_COUNT.value == "call_count"
        assert PerformanceMetric.ERROR_RATE.value == "error_rate"
        assert PerformanceMetric.THROUGHPUT.value == "throughput"
        assert PerformanceMetric.LATENCY.value == "latency"


class TestPerformanceProfile:
    """Тесты для PerformanceProfile."""

    def test_creation(self):
        """Тест создания профиля."""
        timestamp = datetime.now()
        profile = PerformanceProfile(
            function_name="test_func",
            total_calls=100,
            total_time=1.5,
            avg_time=0.015,
            min_time=0.001,
            std_time=0.005,
            p95_time=0.025,
            p99_time=0.035,
            memory_peak=1024,
            memory_avg=512,
            cpu_usage=25.5,
            timestamp=timestamp,
        )

        assert profile.function_name == "test_func"
        assert profile.total_calls == 100
        assert profile.total_time == 1.5
        assert profile.avg_time == 0.015
        assert profile.min_time == 0.001
        assert profile.std_time == 0.005
        assert profile.p95_time == 0.025
        assert profile.p99_time == 0.035
        assert profile.memory_peak == 1024
        assert profile.memory_avg == 512
        assert profile.cpu_usage == 25.5
        assert profile.timestamp == timestamp


class TestBenchmarkResult:
    """Тесты для BenchmarkResult."""

    def test_creation(self):
        """Тест создания результата бенчмарка."""
        timestamp = datetime.now()
        result = BenchmarkResult(
            name="test_benchmark",
            iterations=1000,
            total_time=2.5,
            avg_time=0.0025,
            min_time=0.001,
            max_time=0.005,
            std_time=0.001,
            memory_usage=2048,
            cpu_usage=30.0,
            throughput=400.0,
            timestamp=timestamp,
        )

        assert result.name == "test_benchmark"
        assert result.iterations == 1000
        assert result.total_time == 2.5
        assert result.avg_time == 0.0025
        assert result.min_time == 0.001
        assert result.max_time == 0.005
        assert result.std_time == 0.001
        assert result.memory_usage == 2048
        assert result.cpu_usage == 30.0
        assert result.throughput == 400.0
        assert result.timestamp == timestamp


class TestOptimizationSuggestion:
    """Тесты для OptimizationSuggestion."""

    def test_creation(self):
        """Тест создания предложения по оптимизации."""
        suggestion = OptimizationSuggestion(
            function_name="slow_function",
            issue_type="memory_leak",
            description="Функция использует слишком много памяти",
            impact="high",
            suggestion="Использовать генераторы вместо списков",
            priority=8,
        )

        assert suggestion.function_name == "slow_function"
        assert suggestion.issue_type == "memory_leak"
        assert suggestion.description == "Функция использует слишком много памяти"
        assert suggestion.impact == "high"
        assert suggestion.suggestion == "Использовать генераторы вместо списков"
        assert suggestion.priority == 8


class TestPerformanceProfiler:
    """Тесты для PerformanceProfiler."""

    @pytest.fixture
    def profiler(self):
        """Фикстура профилировщика."""
        return PerformanceProfiler()

    def test_initialization(self, profiler):
        """Тест инициализации."""
        assert profiler.profiles == {}
        assert isinstance(profiler.start_time, datetime)

    @patch('domain.protocols.performance.tracemalloc')
    def test_profile_function_context_manager(self, mock_tracemalloc, profiler):
        """Тест контекстного менеджера профилирования."""
        mock_tracemalloc.start.return_value = None
        mock_tracemalloc.stop.return_value = None
        mock_tracemalloc.get_traced_memory.return_value = (1024, 2048)

        with profiler.profile_function("test_func") as profile_id:
            time.sleep(0.01)  # Имитация работы функции

        assert profile_id in profiler.profiles
        profile = profiler.profiles[profile_id]
        assert profile.function_name == "test_func"
        assert profile.total_calls == 1
        assert profile.total_time > 0

    @patch('domain.protocols.performance.tracemalloc')
    async def test_save_profile(self, mock_tracemalloc, profiler):
        """Тест сохранения профиля."""
        mock_tracemalloc.get_traced_memory.return_value = (1024, 2048)
        
        profile = PerformanceProfile(
            function_name="test_func",
            total_calls=1,
            total_time=0.1,
            avg_time=0.1,
            min_time=0.1,
            std_time=0.0,
            p95_time=0.1,
            p99_time=0.1,
            memory_peak=1024,
            memory_avg=512,
            cpu_usage=10.0,
            timestamp=datetime.now(),
        )

        await profiler._save_profile(profile)
        # Проверяем, что профиль сохранен
        assert len(profiler.profiles) > 0

    @patch('domain.protocols.performance.psutil')
    def test_get_memory_usage_with_psutil(self, mock_psutil, profiler):
        """Тест получения использования памяти с psutil."""
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024)
        mock_psutil.Process.return_value = mock_process

        memory_usage = profiler._get_memory_usage()
        assert memory_usage > 0

    def test_get_memory_usage_without_psutil(self, profiler):
        """Тест получения использования памяти без psutil."""
        with patch('domain.protocols.performance.PSUTIL_AVAILABLE', False):
            memory_usage = profiler._get_memory_usage()
            assert memory_usage >= 0

    async def test_get_function_profile(self, profiler):
        """Тест получения профиля функции."""
        # Создаем тестовый профиль
        profile = PerformanceProfile(
            function_name="test_func",
            total_calls=1,
            total_time=0.1,
            avg_time=0.1,
            min_time=0.1,
            std_time=0.0,
            p95_time=0.1,
            p99_time=0.1,
            memory_peak=1024,
            memory_avg=512,
            cpu_usage=10.0,
            timestamp=datetime.now(),
        )
        profiler.profiles["test_id"] = profile

        result = await profiler.get_function_profile("test_func", hours=24)
        assert result is not None
        assert result.function_name == "test_func"

    async def test_get_all_profiles(self, profiler):
        """Тест получения всех профилей."""
        # Создаем тестовые профили
        profile1 = PerformanceProfile(
            function_name="func1",
            total_calls=1,
            total_time=0.1,
            avg_time=0.1,
            min_time=0.1,
            std_time=0.0,
            p95_time=0.1,
            p99_time=0.1,
            memory_peak=1024,
            memory_avg=512,
            cpu_usage=10.0,
            timestamp=datetime.now(),
        )
        profile2 = PerformanceProfile(
            function_name="func2",
            total_calls=1,
            total_time=0.2,
            avg_time=0.2,
            min_time=0.2,
            std_time=0.0,
            p95_time=0.2,
            p99_time=0.2,
            memory_peak=2048,
            memory_avg=1024,
            cpu_usage=20.0,
            timestamp=datetime.now(),
        )
        profiler.profiles["id1"] = profile1
        profiler.profiles["id2"] = profile2

        results = await profiler.get_all_profiles(hours=24)
        assert len(results) == 2
        assert "func1" in results
        assert "func2" in results


class TestBenchmarkRunner:
    """Тесты для BenchmarkRunner."""

    @pytest.fixture
    def benchmark_runner(self):
        """Фикстура бенчмарк раннера."""
        return BenchmarkRunner()

    def test_initialization(self, benchmark_runner):
        """Тест инициализации."""
        assert benchmark_runner.benchmarks == {}

    @patch('domain.protocols.performance.psutil')
    async def test_benchmark_function(self, mock_psutil, benchmark_runner):
        """Тест бенчмарка функции."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_psutil.Process.return_value = mock_process

        def test_func(x: int) -> int:
            return x * 2

        result = await benchmark_runner.benchmark_function(
            test_func, "test_benchmark", 5, iterations=10, warmup_iterations=2
        )

        assert result.name == "test_benchmark"
        assert result.iterations == 10
        assert result.total_time > 0
        assert result.avg_time > 0
        assert result.min_time > 0
        assert result.max_time > 0
        assert result.memory_usage > 0
        assert result.cpu_usage > 0
        assert result.throughput > 0

    @patch('domain.protocols.performance.psutil')
    async def test_benchmark_async_function(self, mock_psutil, benchmark_runner):
        """Тест бенчмарка асинхронной функции."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_psutil.Process.return_value = mock_process

        async def async_test_func(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        result = await benchmark_runner.benchmark_function(
            async_test_func, "async_test_benchmark", 5, iterations=5, warmup_iterations=1
        )

        assert result.name == "async_test_benchmark"
        assert result.iterations == 5
        assert result.total_time > 0

    @patch('domain.protocols.performance.psutil')
    async def test_compare_functions(self, mock_psutil, benchmark_runner):
        """Тест сравнения функций."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_psutil.Process.return_value = mock_process

        def func1(x: int) -> int:
            return x * 2

        def func2(x: int) -> int:
            return x + x

        functions = [(func1, "func1"), (func2, "func2")]
        results = await benchmark_runner.compare_functions(functions, 5, iterations=5)

        assert len(results) == 2
        assert "func1" in results
        assert "func2" in results
        assert isinstance(results["func1"], BenchmarkResult)
        assert isinstance(results["func2"], BenchmarkResult)

    async def test_get_benchmark_history(self, benchmark_runner):
        """Тест получения истории бенчмарков."""
        # Создаем тестовый результат
        result = BenchmarkResult(
            name="test_benchmark",
            iterations=10,
            total_time=0.1,
            avg_time=0.01,
            min_time=0.005,
            max_time=0.02,
            std_time=0.005,
            memory_usage=1024,
            cpu_usage=25.0,
            throughput=100.0,
            timestamp=datetime.now(),
        )
        benchmark_runner.benchmarks["test_benchmark"] = [result]

        history = await benchmark_runner.get_benchmark_history("test_benchmark", hours=24)
        assert len(history) == 1
        assert history[0].name == "test_benchmark"


class TestPerformanceOptimizer:
    """Тесты для PerformanceOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Фикстура оптимизатора."""
        return PerformanceOptimizer()

    async def test_analyze_function_performance(self, optimizer):
        """Тест анализа производительности функции."""
        def slow_function(n: int) -> List[int]:
            return [i for i in range(n)]

        suggestions = await optimizer.analyze_function_performance("slow_function")
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, OptimizationSuggestion) for s in suggestions)

    async def test_optimize_function(self, optimizer):
        """Тест оптимизации функции."""
        def original_func(x: int) -> int:
            return x * 2

        optimized_func = await optimizer.optimize_function(original_func, "test_func")
        
        # Проверяем, что функция работает
        result = optimized_func(5)
        assert result == 10

    def test_apply_caching_async(self, optimizer):
        """Тест применения кэширования для асинхронной функции."""
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        cached_func = optimizer._apply_caching(async_func)
        assert callable(cached_func)

    def test_apply_caching_sync(self, optimizer):
        """Тест применения кэширования для синхронной функции."""
        def sync_func(x: int) -> int:
            return x * 2

        cached_func = optimizer._apply_caching(sync_func)
        assert callable(cached_func)

    def test_apply_memory_optimization_async(self, optimizer):
        """Тест применения оптимизации памяти для асинхронной функции."""
        async def async_func(x: int) -> int:
            await asyncio.sleep(0.001)
            return x * 2

        optimized_func = optimizer._apply_memory_optimization(async_func)
        assert callable(optimized_func)

    def test_apply_memory_optimization_sync(self, optimizer):
        """Тест применения оптимизации памяти для синхронной функции."""
        def sync_func(x: int) -> int:
            return x * 2

        optimized_func = optimizer._apply_memory_optimization(sync_func)
        assert callable(optimized_func)


class TestPerformanceDecorators:
    """Тесты для декораторов производительности."""

    @patch('domain.protocols.performance.PerformanceProfiler')
    def test_profile_performance_decorator(self, mock_profiler_class):
        """Тест декоратора профилирования."""
        mock_profiler = Mock()
        mock_profiler_class.return_value = mock_profiler

        @profile_performance("test_function")
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

    @patch('domain.protocols.performance.BenchmarkRunner')
    def test_benchmark_performance_decorator(self, mock_benchmark_class):
        """Тест декоратора бенчмаркинга."""
        mock_runner = Mock()
        mock_benchmark_class.return_value = mock_runner

        @benchmark_performance("test_benchmark", iterations=10)
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10

    @patch('domain.protocols.performance.PerformanceOptimizer')
    def test_optimize_performance_decorator(self, mock_optimizer_class):
        """Тест декоратора оптимизации."""
        mock_optimizer = Mock()
        mock_optimizer_class.return_value = mock_optimizer

        @optimize_performance("test_optimization")
        def test_func(x: int) -> int:
            return x * 2

        result = test_func(5)
        assert result == 10


class TestPerformanceUtilities:
    """Тесты для утилит производительности."""

    @patch('domain.protocols.performance.PerformanceProfiler')
    @patch('domain.protocols.performance.BenchmarkRunner')
    async def test_get_performance_report(self, mock_benchmark_class, mock_profiler_class):
        """Тест получения отчета о производительности."""
        mock_profiler = Mock()
        mock_profiler.get_all_profiles = AsyncMock(return_value={})
        mock_profiler_class.return_value = mock_profiler

        mock_runner = Mock()
        mock_runner.get_benchmark_history = AsyncMock(return_value=[])
        mock_benchmark_class.return_value = mock_runner

        report = await get_performance_report(hours=24)
        assert isinstance(report, dict)
        assert "profiles" in report
        assert "benchmarks" in report
        assert "system_metrics" in report

    @patch('domain.protocols.performance.PerformanceOptimizer')
    async def test_optimize_slow_functions(self, mock_optimizer_class):
        """Тест оптимизации медленных функций."""
        mock_optimizer = Mock()
        mock_optimizer.analyze_function_performance = AsyncMock(
            return_value=[OptimizationSuggestion(
                function_name="slow_func",
                issue_type="performance",
                description="Function is slow",
                impact="medium",
                suggestion="Optimize algorithm",
                priority=5
            )]
        )
        mock_optimizer_class.return_value = mock_optimizer

        with patch('domain.protocols.performance.PerformanceProfiler') as mock_profiler_class:
            mock_profiler = Mock()
            mock_profiler.get_all_profiles = AsyncMock(return_value={
                "slow_func": PerformanceProfile(
                    function_name="slow_func",
                    total_calls=100,
                    total_time=10.0,
                    avg_time=0.1,
                    min_time=0.05,
                    std_time=0.02,
                    p95_time=0.15,
                    p99_time=0.2,
                    memory_peak=1024,
                    memory_avg=512,
                    cpu_usage=50.0,
                    timestamp=datetime.now(),
                )
            })
            mock_profiler_class.return_value = mock_profiler

            suggestions = await optimize_slow_functions(threshold=0.05)
            assert isinstance(suggestions, dict)
            assert "slow_func" in suggestions

    @patch('domain.protocols.performance.BenchmarkRunner')
    async def test_run_performance_benchmarks(self, mock_benchmark_class):
        """Тест запуска бенчмарков производительности."""
        mock_runner = Mock()
        mock_runner.benchmark_function = AsyncMock(return_value=BenchmarkResult(
            name="test_benchmark",
            iterations=10,
            total_time=0.1,
            avg_time=0.01,
            min_time=0.005,
            max_time=0.02,
            std_time=0.005,
            memory_usage=1024,
            cpu_usage=25.0,
            throughput=100.0,
            timestamp=datetime.now(),
        ))
        mock_benchmark_class.return_value = mock_runner

        def test_func(x: int) -> int:
            return x * 2

        results = await run_performance_benchmarks()
        assert isinstance(results, dict)

    @patch('domain.protocols.performance.psutil')
    def test_get_system_performance(self, mock_psutil):
        """Тест получения системной производительности."""
        mock_cpu_percent = Mock(return_value=25.0)
        mock_virtual_memory = Mock(return_value=Mock(percent=50.0, used=1024*1024*1024))
        mock_disk_usage = Mock(return_value=Mock(percent=30.0))
        
        mock_psutil.cpu_percent = mock_cpu_percent
        mock_psutil.virtual_memory = mock_virtual_memory
        mock_psutil.disk_usage = mock_disk_usage

        metrics = get_system_performance()
        assert isinstance(metrics, dict)
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "disk_usage" in metrics

    def test_monitor_performance_continuously(self):
        """Тест непрерывного мониторинга производительности."""
        monitor = monitor_performance_continuously(interval=1)
        assert callable(monitor)


class TestPerformanceProtocol:
    """Тесты для PerformanceProtocol."""

    def test_protocol_definition(self):
        """Тест определения протокола."""
        # Проверяем, что протокол определен корректно
        assert hasattr(PerformanceProtocol, '__call__')
        
        # Проверяем методы протокола
        methods = PerformanceProtocol.__dict__.get('__annotations__', {})
        assert 'start_profiling' in methods
        assert 'stop_profiling' in methods
        assert 'get_performance_metrics' in methods
        assert 'benchmark_function' in methods
        assert 'optimize_function' in methods


class TestPerformanceIntegration:
    """Интеграционные тесты производительности."""

    @patch('domain.protocols.performance.psutil')
    async def test_full_performance_workflow(self, mock_psutil):
        """Тест полного рабочего процесса производительности."""
        mock_process = Mock()
        mock_process.cpu_percent.return_value = 25.0
        mock_process.memory_info.return_value = Mock(rss=1024 * 1024)
        mock_psutil.Process.return_value = mock_process

        # Создаем компоненты
        profiler = PerformanceProfiler()
        benchmark_runner = BenchmarkRunner()
        optimizer = PerformanceOptimizer()

        # Тестируем функцию
        def test_function(n: int) -> List[int]:
            return [i * 2 for i in range(n)]

        # Профилируем
        with profiler.profile_function("test_function") as profile_id:
            result = test_function(1000)

        assert len(result) == 1000
        assert profile_id in profiler.profiles

        # Бенчмаркаем
        benchmark_result = await benchmark_runner.benchmark_function(
            test_function, "test_benchmark", 1000, iterations=10
        )

        assert benchmark_result.name == "test_benchmark"
        assert benchmark_result.iterations == 10
        assert benchmark_result.total_time > 0

        # Анализируем производительность
        suggestions = await optimizer.analyze_function_performance("test_function")
        assert isinstance(suggestions, list)

        # Оптимизируем
        optimized_func = await optimizer.optimize_function(test_function, "test_function")
        optimized_result = optimized_func(1000)
        assert len(optimized_result) == 1000

    async def test_error_handling(self):
        """Тест обработки ошибок."""
        profiler = PerformanceProfiler()
        benchmark_runner = BenchmarkRunner()
        optimizer = PerformanceOptimizer()

        # Тест с некорректной функцией
        def invalid_function():
            raise ValueError("Test error")

        # Профилирование должно обработать ошибку
        with pytest.raises(ValueError):
            with profiler.profile_function("invalid_function"):
                invalid_function()

        # Бенчмарк должен обработать ошибку
        with pytest.raises(ValueError):
            await benchmark_runner.benchmark_function(
                invalid_function, "invalid_benchmark", iterations=1
            )

        # Анализ должен вернуть предложения по оптимизации
        suggestions = await optimizer.analyze_function_performance("invalid_function")
        assert isinstance(suggestions, list) 