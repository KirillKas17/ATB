"""
Модуль оптимизации производительности - МАКСИМАЛЬНАЯ СКОРОСТЬ
"""

import asyncio
import time
import gc
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import weakref
from loguru import logger
import numpy as np


@dataclass
class PerformanceMetrics:
    """Метрики производительности."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    execution_time: float = 0.0
    throughput: float = 0.0
    cache_hit_rate: float = 0.0
    gc_collections: int = 0
    active_threads: int = 0
    async_tasks: int = 0


class PerformanceOptimizer:
    """Оптимизатор производительности системы."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.thread_pool = ThreadPoolExecutor(max_workers=psutil.cpu_count())
        self.process_pool = ProcessPoolExecutor(max_workers=psutil.cpu_count())
        self.cache_stats = {"hits": 0, "misses": 0}
        self.execution_history = []
        self.optimization_enabled = True
        
        # Оптимизированные настройки
        self._setup_gc_optimization()
        self._setup_asyncio_optimization()
        
        logger.info("PerformanceOptimizer инициализирован с максимальными настройками")
    
    def _setup_gc_optimization(self):
        """Настройка оптимального garbage collection."""
        # Увеличиваем пороги GC для лучшей производительности
        gc.set_threshold(1000, 15, 15)
        
        # Отключаем автоматический GC для критических секций
        self.gc_disabled = False
        
        logger.info("Garbage collection оптимизирован")
    
    def _setup_asyncio_optimization(self):
        """Настройка оптимального asyncio."""
        try:
            # Используем более быстрый event loop если доступен
            import uvloop
            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("Использую uvloop для максимальной производительности")
        except ImportError:
            logger.info("uvloop недоступен, используется стандартный event loop")
    
    def performance_monitor(self, func: Callable) -> Callable:
        """Декоратор для мониторинга производительности функций."""
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                result = await func(*args, **kwargs)
                self.cache_stats["hits"] += 1
                return result
            except Exception as e:
                self.cache_stats["misses"] += 1
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory
                
                self.execution_history.append({
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "memory_delta": memory_delta,
                    "timestamp": time.time()
                })
                
                # Логируем только медленные функции
                if execution_time > 0.1:
                    logger.debug(f"{func.__name__} выполнена за {execution_time:.3f}с, память: {memory_delta:+.1f}MB")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                self.cache_stats["hits"] += 1
                return result
            except Exception as e:
                self.cache_stats["misses"] += 1
                raise
            finally:
                execution_time = time.time() - start_time
                if execution_time > 0.1:
                    logger.debug(f"{func.__name__} выполнена за {execution_time:.3f}с")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    @lru_cache(maxsize=1000)
    def cached_calculation(self, key: str, *args) -> Any:
        """Кэшированные вычисления для повторяющихся операций."""
        # Здесь могут быть сложные вычисления
        return f"cached_result_for_{key}"
    
    async def parallel_execution(self, tasks: List[Callable], use_processes: bool = False) -> List[Any]:
        """Параллельное выполнение задач."""
        if not self.optimization_enabled:
            return [await task() if asyncio.iscoroutinefunction(task) else task() for task in tasks]
        
        if use_processes:
            # Используем ProcessPoolExecutor для CPU-intensive задач
            loop = asyncio.get_event_loop()
            futures = [loop.run_in_executor(self.process_pool, task) for task in tasks if not asyncio.iscoroutinefunction(task)]
            async_tasks = [task() for task in tasks if asyncio.iscoroutinefunction(task)]
            
            results = []
            if futures:
                results.extend(await asyncio.gather(*futures))
            if async_tasks:
                results.extend(await asyncio.gather(*async_tasks))
            return results
        else:
            # Используем ThreadPoolExecutor для I/O-intensive задач
            if all(asyncio.iscoroutinefunction(task) for task in tasks):
                return await asyncio.gather(*[task() for task in tasks])
            else:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(self.thread_pool, task) if not asyncio.iscoroutinefunction(task) else task()
                    for task in tasks
                ]
                return await asyncio.gather(*futures)
    
    def optimize_memory(self):
        """Оптимизация использования памяти."""
        if not self.optimization_enabled:
            return
        
        # Принудительная сборка мусора
        collected = gc.collect()
        
        # Компактификация памяти
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except:
            pass  # Не критично если не удалось
        
        logger.debug(f"Память оптимизирована, собрано {collected} объектов")
    
    def get_system_metrics(self) -> PerformanceMetrics:
        """Получение текущих метрик системы."""
        process = psutil.Process()
        
        self.metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
        self.metrics.memory_usage = process.memory_percent()
        self.metrics.active_threads = threading.active_count()
        self.metrics.gc_collections = sum(gc.get_stats()[i]['collections'] for i in range(len(gc.get_stats())))
        
        # Расчёт cache hit rate
        total_cache_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_cache_ops > 0:
            self.metrics.cache_hit_rate = self.cache_stats["hits"] / total_cache_ops * 100
        
        # Расчёт throughput (операций в секунду)
        if len(self.execution_history) > 1:
            recent_executions = [h for h in self.execution_history if time.time() - h["timestamp"] < 60]
            self.metrics.throughput = len(recent_executions) / 60 if recent_executions else 0
        
        return self.metrics
    
    def auto_optimize(self):
        """Автоматическая оптимизация на основе текущих метрик."""
        metrics = self.get_system_metrics()
        
        # Оптимизация памяти при высоком использовании
        if metrics.memory_usage > 80:
            self.optimize_memory()
            logger.info("Автоматическая оптимизация памяти выполнена")
        
        # Управление GC при высокой загрузке CPU
        if metrics.cpu_usage > 90 and not self.gc_disabled:
            gc.disable()
            self.gc_disabled = True
            logger.info("GC временно отключён для снижения CPU нагрузки")
        elif metrics.cpu_usage < 50 and self.gc_disabled:
            gc.enable()
            self.gc_disabled = False
            logger.info("GC снова включён")
        
        # Очистка истории выполнения
        if len(self.execution_history) > 10000:
            self.execution_history = self.execution_history[-5000:]
            logger.debug("История выполнения очищена")
    
    def create_optimized_cache(self, maxsize: int = 1000) -> Callable:
        """Создание оптимизированного кэша."""
        def cache_decorator(func):
            cache = {}
            weak_cache = weakref.WeakValueDictionary()
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = str(args) + str(sorted(kwargs.items()))
                
                # Проверяем обычный кэш
                if key in cache:
                    self.cache_stats["hits"] += 1
                    return cache[key]
                
                # Проверяем weak кэш
                if key in weak_cache:
                    result = weak_cache[key]
                    if result is not None:
                        self.cache_stats["hits"] += 1
                        return result
                
                # Вычисляем результат
                result = func(*args, **kwargs)
                self.cache_stats["misses"] += 1
                
                # Сохраняем в кэш
                if len(cache) < maxsize:
                    cache[key] = result
                else:
                    # Используем weak кэш если основной переполнен
                    try:
                        weak_cache[key] = result
                    except TypeError:
                        pass  # Некоторые объекты нельзя добавить в weak cache
                
                return result
            
            return wrapper
        return cache_decorator
    
    async def benchmark_function(self, func: Callable, iterations: int = 1000) -> Dict[str, float]:
        """Бенчмарк функции для измерения производительности."""
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
            
            times.append(time.perf_counter() - start)
        
        times = np.array(times)
        
        return {
            "mean_time": float(np.mean(times)),
            "median_time": float(np.median(times)),
            "std_time": float(np.std(times)),
            "min_time": float(np.min(times)),
            "max_time": float(np.max(times)),
            "total_time": float(np.sum(times)),
            "ops_per_second": iterations / float(np.sum(times))
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Генерация отчёта о производительности."""
        metrics = self.get_system_metrics()
        
        # Анализ функций по времени выполнения
        if self.execution_history:
            recent_history = [h for h in self.execution_history if time.time() - h["timestamp"] < 300]
            
            if recent_history:
                func_stats = {}
                for record in recent_history:
                    func_name = record["function"]
                    if func_name not in func_stats:
                        func_stats[func_name] = {"times": [], "memory": []}
                    func_stats[func_name]["times"].append(record["execution_time"])
                    func_stats[func_name]["memory"].append(record.get("memory_delta", 0))
                
                slowest_functions = sorted(
                    [(name, np.mean(stats["times"])) for name, stats in func_stats.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            else:
                slowest_functions = []
        else:
            slowest_functions = []
        
        return {
            "system_metrics": {
                "cpu_usage": metrics.cpu_usage,
                "memory_usage": metrics.memory_usage,
                "active_threads": metrics.active_threads,
                "gc_collections": metrics.gc_collections,
                "cache_hit_rate": metrics.cache_hit_rate,
                "throughput": metrics.throughput
            },
            "optimization_status": {
                "gc_disabled": self.gc_disabled,
                "optimization_enabled": self.optimization_enabled,
                "thread_pool_workers": self.thread_pool._max_workers,
                "process_pool_workers": self.process_pool._max_workers
            },
            "performance_analysis": {
                "slowest_functions": slowest_functions,
                "total_executions": len(self.execution_history),
                "cache_stats": self.cache_stats.copy()
            },
            "recommendations": self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Генерация рекомендаций по оптимизации."""
        recommendations = []
        
        if metrics.cpu_usage > 80:
            recommendations.append("Рассмотрите увеличение количества CPU или оптимизацию алгоритмов")
        
        if metrics.memory_usage > 70:
            recommendations.append("Высокое использование памяти - рекомендуется оптимизация кэширования")
        
        if metrics.cache_hit_rate < 80:
            recommendations.append("Низкий hit rate кэша - рассмотрите увеличение размера кэша")
        
        if metrics.active_threads > psutil.cpu_count() * 4:
            recommendations.append("Слишком много активных потоков - рассмотрите использование async/await")
        
        if metrics.throughput < 10:
            recommendations.append("Низкая пропускная способность - проверьте узкие места в коде")
        
        if not recommendations:
            recommendations.append("Система работает оптимально!")
        
        return recommendations
    
    async def cleanup(self):
        """Очистка ресурсов оптимизатора."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        if self.gc_disabled:
            gc.enable()
        
        logger.info("PerformanceOptimizer очищен")


# Глобальный экземпляр оптимизатора
performance_optimizer = PerformanceOptimizer()


# Удобные декораторы
def optimized(func: Callable) -> Callable:
    """Декоратор для автоматической оптимизации функции."""
    return performance_optimizer.performance_monitor(func)


def cached(maxsize: int = 1000) -> Callable:
    """Декоратор для кэширования результатов функции."""
    return performance_optimizer.create_optimized_cache(maxsize)


async def parallel_execute(*tasks) -> List[Any]:
    """Быстрое параллельное выполнение задач."""
    return await performance_optimizer.parallel_execution(list(tasks))


def get_performance_stats() -> Dict[str, Any]:
    """Быстрое получение статистики производительности."""
    return performance_optimizer.get_performance_report()


# Автоматическая оптимизация каждые 60 секунд
async def auto_optimization_loop():
    """Фоновый цикл автоматической оптимизации."""
    while True:
        try:
            performance_optimizer.auto_optimize()
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Ошибка в автоматической оптимизации: {e}")
            await asyncio.sleep(60)


def start_auto_optimization():
    """Запуск автоматической оптимизации."""
    asyncio.create_task(auto_optimization_loop())
    logger.info("Автоматическая оптимизация производительности запущена")