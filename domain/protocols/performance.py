"""
Система анализа производительности протоколов - промышленный уровень.
Этот модуль содержит инструменты для анализа производительности:
- Профилирование функций
- Анализ памяти
- Бенчмаркинг
- Оптимизация производительности
- Метрики производительности
- Анализ узких мест
"""

import asyncio
import functools
import logging
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    Generator,
)
from uuid import uuid4
from contextlib import contextmanager
from shared.numpy_utils import np

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None
try:
    import memory_profiler  # type: ignore

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False
    memory_profiler = None


class PerformanceMetric(Enum):
    """Метрики производительности."""

    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    CALL_COUNT = "call_count"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"
    LATENCY = "latency"


class PerformanceProtocol(Protocol):
    """Протокол производительности."""

    async def start_profiling(self) -> None:
        """Запуск профилирования."""
        ...

    async def stop_profiling(self) -> None:
        """Остановка профилирования."""
        ...

    async def get_performance_metrics(self) -> Dict[str, float]:
        """Получение метрик производительности."""
        ...

    async def benchmark_function(
        self, func: Callable, name: str, iterations: int = 1000
    ) -> Dict[str, Any]:
        """Бенчмарк функции."""
        ...

    async def optimize_function(self, func: Callable, name: str) -> Callable:
        """Оптимизация функции."""
        ...


@dataclass
class PerformanceProfile:
    """Профиль производительности."""

    function_name: str
    total_calls: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    p95_time: float
    p99_time: float
    memory_peak: int
    memory_avg: int
    cpu_usage: float
    timestamp: datetime


@dataclass
class BenchmarkResult:
    """Результат бенчмарка."""

    name: str
    iterations: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    std_time: float
    memory_usage: int
    cpu_usage: float
    throughput: float
    timestamp: datetime


@dataclass
class OptimizationSuggestion:
    """Предложение по оптимизации."""

    function_name: str
    issue_type: str
    description: str
    impact: str
    suggestion: str
    priority: int  # 1-10, где 10 - высший приоритет


class PerformanceProfiler:
    """Профилировщик производительности."""

    def __init__(self) -> None:
        self._profiles: Dict[str, List[PerformanceProfile]] = {}
        self._current_profiles: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    @contextmanager
    def profile_function(self, function_name: str) -> Generator[str, None, None]:
        """Контекстный менеджер для профилирования функции."""
        profile_id = str(uuid4())
        # Начинаем профилирование
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_cpu = psutil.cpu_percent()
        # Запускаем tracemalloc
        tracemalloc.start()
        try:
            yield profile_id
        finally:
            # Останавливаем профилирование
            end_time = time.time()
            end_memory = self._get_memory_usage()
            end_cpu = psutil.cpu_percent()
            # Получаем статистику памяти
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            # Создаем профиль
            profile = PerformanceProfile(
                function_name=function_name,
                total_calls=1,
                total_time=end_time - start_time,
                avg_time=end_time - start_time,
                min_time=end_time - start_time,
                max_time=end_time - start_time,
                std_time=0.0,
                p95_time=end_time - start_time,
                p99_time=end_time - start_time,
                memory_peak=peak,
                memory_avg=current,
                cpu_usage=(start_cpu + end_cpu) / 2,
                timestamp=datetime.now(),
            )
            # Сохраняем профиль
            asyncio.create_task(self._save_profile(profile))

    async def _save_profile(self, profile: PerformanceProfile) -> None:
        """Сохранить профиль."""
        async with self._lock:
            if profile.function_name not in self._profiles:
                self._profiles[profile.function_name] = []
            self._profiles[profile.function_name].append(profile)
            # Ограничиваем количество профилей
            if len(self._profiles[profile.function_name]) > 1000:
                self._profiles[profile.function_name] = self._profiles[
                    profile.function_name
                ][-500:]

    def _get_memory_usage(self) -> int:
        """Получить использование памяти."""
        try:
            process = psutil.Process()
            return int(process.memory_info().rss)
        except Exception:
            return 0

    async def get_function_profile(
        self, function_name: str, hours: int = 24
    ) -> Optional[PerformanceProfile]:
        """Получить профиль функции."""
        async with self._lock:
            if function_name not in self._profiles:
                return None
            profiles = self._profiles[function_name]
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_profiles = [p for p in profiles if p.timestamp >= cutoff_time]
            if not recent_profiles:
                return None
            # Агрегируем профили
            total_calls = len(recent_profiles)
            total_time = sum(p.total_time for p in recent_profiles)
            times = [p.total_time for p in recent_profiles]
            return PerformanceProfile(
                function_name=function_name,
                total_calls=total_calls,
                total_time=total_time,
                avg_time=float(np.mean(times)),
                min_time=float(np.min(times)),
                max_time=float(np.max(times)),
                std_time=float(np.std(times)),
                p95_time=float(np.percentile(times, 95)),
                p99_time=float(np.percentile(times, 99)),
                memory_peak=max(p.memory_peak for p in recent_profiles),
                memory_avg=int(np.mean([p.memory_avg for p in recent_profiles])),
                cpu_usage=float(np.mean([p.cpu_usage for p in recent_profiles])),
                timestamp=datetime.now(),
            )

    async def get_all_profiles(self, hours: int = 24) -> Dict[str, PerformanceProfile]:
        """Получить все профили."""
        profiles = {}
        for function_name in self._profiles.keys():
            profile = await self.get_function_profile(function_name, hours)
            if profile:
                profiles[function_name] = profile
        return profiles


class BenchmarkRunner:
    """Запуск бенчмарков."""

    def __init__(self) -> None:
        self._results: Dict[str, List[BenchmarkResult]] = {}
        self._lock = asyncio.Lock()

    async def benchmark_function(
        self,
        func: Callable,
        name: str,
        iterations: int = 1000,
        warmup_iterations: int = 100,
        *args: Any,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Запустить бенчмарк функции."""
        # Разогрев
        for _ in range(warmup_iterations):
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
        # Основной бенчмарк
        times: List[float] = []
        memory_usage: List[int] = []
        cpu_usage: List[float] = []
        start_memory = self._get_memory_usage()
        for _ in range(iterations):
            iteration_start = time.time()
            iteration_cpu_start = psutil.cpu_percent()
            if asyncio.iscoroutinefunction(func):
                await func(*args, **kwargs)
            else:
                func(*args, **kwargs)
            iteration_time = time.time() - iteration_start
            iteration_cpu = psutil.cpu_percent() - iteration_cpu_start
            times.append(iteration_time)
            cpu_usage.append(iteration_cpu)
        end_memory = self._get_memory_usage()
        memory_used = end_memory - start_memory
        # Вычисляем статистику
        total_time = sum(times)
        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)
        std_time = np.std(times)
        avg_cpu = np.mean(cpu_usage)
        throughput = iterations / total_time if total_time > 0 else 0
        result = BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time=total_time,
            avg_time=float(avg_time),
            min_time=float(min_time),
            max_time=float(max_time),
            std_time=float(std_time),
            memory_usage=memory_used,
            cpu_usage=float(avg_cpu),
            throughput=float(throughput),
            timestamp=datetime.now(),
        )
        # Сохраняем результат
        async with self._lock:
            if name not in self._results:
                self._results[name] = []
            self._results[name].append(result)
        return result

    def _get_memory_usage(self) -> int:
        """Получить использование памяти."""
        try:
            process = psutil.Process()
            return int(process.memory_info().rss)
        except Exception:
            return 0

    async def compare_functions(
        self,
        functions: List[Tuple[Callable, str]],
        iterations: int = 1000,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, BenchmarkResult]:
        """Сравнить производительность функций."""
        results = {}
        for func, name in functions:
            result = await self.benchmark_function(
                func, name, iterations, *args, **kwargs
            )
            results[name] = result
        return results

    async def get_benchmark_history(
        self, name: str, hours: int = 24
    ) -> List[BenchmarkResult]:
        """Получить историю бенчмарков."""
        async with self._lock:
            if name not in self._results:
                return []
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [r for r in self._results[name] if r.timestamp >= cutoff_time]


class PerformanceOptimizer:
    """Оптимизатор производительности."""

    def __init__(self) -> None:
        self._profiler = PerformanceProfiler()
        self._benchmark_runner = BenchmarkRunner()
        self._suggestions: List[OptimizationSuggestion] = []

    async def analyze_function_performance(
        self, function_name: str
    ) -> List[OptimizationSuggestion]:
        """Анализировать производительность функции."""
        profile = await self._profiler.get_function_profile(function_name)
        if not profile:
            return []
        suggestions = []
        # Анализ времени выполнения
        if profile.avg_time > 1.0:  # Больше 1 секунды
            suggestions.append(
                OptimizationSuggestion(
                    function_name=function_name,
                    issue_type="slow_execution",
                    description=f"Function takes {profile.avg_time:.3f}s on average",
                    impact="high",
                    suggestion="Consider caching, parallelization, or algorithm optimization",
                    priority=8,
                )
            )
        # Анализ памяти
        if profile.memory_peak > 100 * 1024 * 1024:  # Больше 100MB
            suggestions.append(
                OptimizationSuggestion(
                    function_name=function_name,
                    issue_type="high_memory_usage",
                    description=f"Function uses {profile.memory_peak / 1024 / 1024:.1f}MB peak memory",
                    impact="medium",
                    suggestion="Consider memory pooling, lazy loading, or streaming",
                    priority=6,
                )
            )
        # Анализ CPU
        if profile.cpu_usage > 80:  # Больше 80% CPU
            suggestions.append(
                OptimizationSuggestion(
                    function_name=function_name,
                    issue_type="high_cpu_usage",
                    description=f"Function uses {profile.cpu_usage:.1f}% CPU on average",
                    impact="medium",
                    suggestion="Consider async operations, caching, or algorithm optimization",
                    priority=7,
                )
            )
        # Анализ вариативности
        if profile.std_time > profile.avg_time * 0.5:  # Высокая вариативность
            suggestions.append(
                OptimizationSuggestion(
                    function_name=function_name,
                    issue_type="high_variability",
                    description=f"Function has high execution time variability (std: {profile.std_time:.3f}s)",
                    impact="low",
                    suggestion="Investigate external dependencies or resource contention",
                    priority=4,
                )
            )
        return suggestions

    async def optimize_function(self, func: Callable, name: str) -> Callable:
        """Оптимизировать функцию."""
        # Применяем кэширование
        optimized_func = self._apply_caching(func)
        # Применяем оптимизацию памяти
        optimized_func = self._apply_memory_optimization(optimized_func)
        return optimized_func

    def _apply_caching(self, func: Callable) -> Callable:
        """Применить кэширование к функции."""
        cache: Dict[str, Any] = {}

        @functools.wraps(func)
        async def cached_async_func(*args: Any, **kwargs: Any) -> Any:
            key = str((args, sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = await func(*args, **kwargs)
            cache[key] = result
            return result

        @functools.wraps(func)
        def cached_sync_func(*args: Any, **kwargs: Any) -> Any:
            key = str((args, sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            return result

        return (
            cached_async_func if asyncio.iscoroutinefunction(func) else cached_sync_func
        )

    def _apply_memory_optimization(self, func: Callable) -> Callable:
        """Применить оптимизацию памяти к функции."""

        @functools.wraps(func)
        async def optimized_async_func(*args: Any, **kwargs: Any) -> Any:
            # Принудительная сборка мусора перед вызовом
            import gc

            gc.collect()
            result = await func(*args, **kwargs)
            # Принудительная сборка мусора после вызова
            gc.collect()
            return result

        @functools.wraps(func)
        def optimized_sync_func(*args: Any, **kwargs: Any) -> Any:
            # Принудительная сборка мусора перед вызовом
            import gc

            gc.collect()
            result = func(*args, **kwargs)
            # Принудительная сборка мусора после вызова
            gc.collect()
            return result

        return (
            optimized_async_func
            if asyncio.iscoroutinefunction(func)
            else optimized_sync_func
        )


# Глобальные экземпляры
performance_profiler = PerformanceProfiler()
benchmark_runner = BenchmarkRunner()
performance_optimizer = PerformanceOptimizer()


# Декораторы для производительности
def profile_performance(function_name: Optional[str] = None) -> Callable:
    """Декоратор для профилирования производительности."""

    def decorator(func: Callable) -> Callable:
        name = function_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with performance_profiler.profile_function(name):
                return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with performance_profiler.profile_function(name):
                return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def benchmark_performance(name: Optional[str] = None, iterations: int = 1000) -> Callable:
    """Декоратор для бенчмаркинга производительности."""

    def decorator(func: Callable) -> Callable:
        benchmark_name = name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Запускаем бенчмарк в фоне
            asyncio.create_task(
                benchmark_runner.benchmark_function(
                    func, benchmark_name, iterations, *args, **kwargs
                )
            )
            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Для синхронных функций запускаем бенчмарк синхронно
            asyncio.run(
                benchmark_runner.benchmark_function(
                    func, benchmark_name, iterations, *args, **kwargs
                )
            )
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def optimize_performance(name: Optional[str] = None) -> Callable:
    """Декоратор для автоматической оптимизации производительности."""

    def decorator(func: Callable) -> Callable:
        optimization_name = name or f"{func.__module__}.{func.__name__}"
        # Оптимизируем функцию
        optimized_func = asyncio.run(
            performance_optimizer.optimize_function(func, optimization_name)
        )
        return optimized_func

    return decorator


# Функции для анализа производительности
async def get_performance_report(hours: int = 24) -> Dict[str, Any]:
    """Получить отчет о производительности."""
    profiles = await performance_profiler.get_all_profiles(hours)
    # Анализируем медленные функции
    slow_functions = [
        name for name, profile in profiles.items() if profile.avg_time > 1.0
    ]
    # Анализируем функции с высоким использованием памяти
    memory_intensive_functions = [
        name
        for name, profile in profiles.items()
        if profile.memory_peak > 100 * 1024 * 1024
    ]
    # Анализируем функции с высоким использованием CPU
    cpu_intensive_functions = [
        name for name, profile in profiles.items() if profile.cpu_usage > 80
    ]
    return {
        "total_functions_profiled": len(profiles),
        "slow_functions": slow_functions,
        "memory_intensive_functions": memory_intensive_functions,
        "cpu_intensive_functions": cpu_intensive_functions,
        "profiles": {
            name: {
                "avg_time": profile.avg_time,
                "memory_peak": profile.memory_peak,
                "cpu_usage": profile.cpu_usage,
                "total_calls": profile.total_calls,
            }
            for name, profile in profiles.items()
        },
        "timestamp": datetime.now().isoformat(),
    }


async def optimize_slow_functions(
    threshold: float = 1.0,
) -> Dict[str, List[OptimizationSuggestion]]:
    """Оптимизировать медленные функции."""
    profiles = await performance_profiler.get_all_profiles()
    optimizations = {}
    for name, profile in profiles.items():
        if profile.avg_time > threshold:
            suggestions = await performance_optimizer.analyze_function_performance(name)
            if suggestions:
                optimizations[name] = suggestions
    return optimizations


async def run_performance_benchmarks() -> Dict[str, BenchmarkResult]:
    """Запустить бенчмарки производительности."""
    # Здесь можно добавить стандартные бенчмарки для протоколов
    return {}


# Утилиты для мониторинга производительности
def get_system_performance() -> Dict[str, Any]:
    """Получить производительность системы."""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available": memory.available,
            "memory_total": memory.total,
            "disk_percent": disk.percent,
            "disk_free": disk.free,
            "disk_total": disk.total,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"error": str(e), "timestamp": datetime.now().isoformat()}


def monitor_performance_continuously(interval: int = 60) -> Any:
    """Мониторить производительность непрерывно."""

    async def monitor() -> None:
        while True:
            try:
                system_perf = get_system_performance()
                perf_report = await get_performance_report()
                # Логируем производительность
                logging.info(f"System Performance: {system_perf}")
                logging.info(f"Performance Report: {perf_report}")
                await asyncio.sleep(interval)
            except Exception as e:
                logging.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(interval)

    return asyncio.create_task(monitor())
