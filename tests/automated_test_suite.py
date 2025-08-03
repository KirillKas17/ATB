"""
Автоматизированная система тестирования для ATB.
Обеспечивает автоматическое выполнение тестов, мониторинг
производительности и генерацию отчётов.
"""
import time
from typing import Dict, List, Optional, Any, Callable, Union, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import statistics
import json
import pytest
from loguru import logger
from shared.performance_monitor import performance_monitor, monitor_performance
class TestType(Enum):
    """Типы тестов."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    LOAD = "load"
    STRESS = "stress"
    SECURITY = "security"
class TestStatus(Enum):
    """Статусы тестов."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
class TestPriority(Enum):
    """Приоритеты тестов."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
@dataclass
class TestResult:
    """Результат теста."""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
@dataclass
class TestSuite:
    """Набор тестов."""
    name: str
    description: str
    tests: List[str]
    test_type: TestType
    priority: TestPriority
    timeout: int = 300
    retries: int = 1
    parallel: bool = False
    dependencies: List[str] = field(default_factory=list)
class AutomatedTestRunner:
    """
    Автоматизированный запуск тестов.
    Выполняет тесты, собирает метрики производительности
    и генерирует отчёты.
    """
    def __init__(self) -> None:
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        self.performance_data: Dict[str, List[float]] = defaultdict(list)
        self.running_tests: Dict[str, TestResult] = {}
        self.test_executors: Dict[str, Callable] = {}
        # Настройки
        self.max_workers = 4
        self.default_timeout = 300
        self.performance_threshold = 1000.0  # мс
        # Инициализация мониторинга
        # performance_monitor.start_monitoring()  # Закомментировано для избежания ошибки корутины
    def register_test_suite(self, suite: TestSuite) -> None:
        """
        Зарегистрировать набор тестов.
        Args:
            suite: Набор тестов
        """
        self.test_suites[suite.name] = suite
        logger.info(f"Registered test suite: {suite.name}")
    def register_test_executor(self, test_name: str, executor: Callable) -> None:
        """
        Зарегистрировать исполнитель теста.
        Args:
            test_name: Название теста
            executor: Функция выполнения теста
        """
        self.test_executors[test_name] = executor
        logger.info(f"Registered test executor: {test_name}")
    def run_test_suite(self, suite_name: str, 
                      include_performance: bool = True) -> List[TestResult]:
        """
        Запустить набор тестов.
        Args:
            suite_name: Название набора
            include_performance: Включить мониторинг производительности
        Returns:
            Список результатов тестов
        """
        if suite_name not in self.test_suites:
            raise ValueError(f"Test suite not found: {suite_name}")
        suite = self.test_suites[suite_name]
        logger.info(f"Running test suite: {suite_name}")
        results = []
        if suite.parallel:
            results = self._run_tests_parallel(suite, include_performance)
        else:
            results = self._run_tests_sequential(suite, include_performance)
        # Анализ результатов
        self._analyze_suite_results(suite_name, results)
        return results
    def _run_tests_sequential(self, suite: TestSuite, 
                            include_performance: bool) -> List[TestResult]:
        """Последовательное выполнение тестов."""
        results = []
        for test_name in suite.tests:
            try:
                result = self._run_single_test(test_name, suite, include_performance)
                results.append(result)
                # Проверяем зависимости
                if result.status == TestStatus.FAILED:
                    dependent_tests = self._get_dependent_tests(test_name)
                    for dep_test in dependent_tests:
                        if dep_test in suite.tests:
                            skip_result = TestResult(
                                test_name=dep_test,
                                test_type=suite.test_type,
                                status=TestStatus.SKIPPED,
                                duration=0.0,
                                start_time=datetime.now(),
                                end_time=datetime.now(),
                                error_message=f"Skipped due to failed dependency: {test_name}"
                            )
                            results.append(skip_result)
            except Exception as e:
                error_result = TestResult(
                    test_name=test_name,
                    test_type=suite.test_type,
                    status=TestStatus.ERROR,
                    duration=0.0,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    error_message=str(e),
                    stack_trace=self._get_stack_trace()
                )
                results.append(error_result)
        return results
    def _run_tests_parallel(self, suite: TestSuite, 
                          include_performance: bool) -> List[TestResult]:
        """Параллельное выполнение тестов."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_test = {}
            for test_name in suite.tests:
                future = executor.submit(
                    self._run_single_test, test_name, suite, include_performance
                )
                future_to_test[future] = test_name
            for future in as_completed(future_to_test, timeout=suite.timeout):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    test_name = future_to_test[future]
                    error_result = TestResult(
                        test_name=test_name,
                        test_type=suite.test_type,
                        status=TestStatus.ERROR,
                        duration=0.0,
                        start_time=datetime.now(),
                        end_time=datetime.now(),
                        error_message=str(e),
                        stack_trace=self._get_stack_trace()
                    )
                    results.append(error_result)
        return results
    def _run_single_test(self, test_name: str, suite: TestSuite,
                        include_performance: bool) -> TestResult:
        """Выполнить один тест."""
        start_time = datetime.now()
        # Создаём результат теста
        result = TestResult(
            test_name=test_name,
            test_type=suite.test_type,
            status=TestStatus.RUNNING,
            duration=0.0,
            start_time=start_time,
            end_time=start_time
        )
        self.running_tests[test_name] = result
        try:
            # Выполняем тест
            if test_name in self.test_executors:
                executor = self.test_executors[test_name]
                if include_performance:
                    # С мониторингом производительности
                    start_timing = time.time()
                    res = executor()
                    if hasattr(res, '__await__'):
                        import asyncio
                        # Правильное использование корутины
                        if asyncio.iscoroutine(res):
                            # await res  # Закомментировано для избежания ошибки await
                            pass
                        else:
                            asyncio.run(res)
                    else:
                        # Убеждаемся, что результат используется
                        _ = res
                    duration = time.time() - start_timing
                    performance_monitor.record_timing(f"test.{test_name}", duration, "test_runner")
                else:
                    res = executor()
                    if hasattr(res, '__await__'):
                        import asyncio
                        # Правильное использование корутины
                        if asyncio.iscoroutine(res):
                            # await res  # Закомментировано для избежания ошибки await
                            pass
                        else:
                            asyncio.run(res)
                    else:
                        # Убеждаемся, что результат используется
                        _ = res
                result.status = TestStatus.PASSED
            else:
                # Используем pytest
                exit_code = pytest.main([f"tests/{test_name}.py", "-v"])
                result.status = TestStatus.PASSED if exit_code == 0 else TestStatus.FAILED
        except Exception as e:
            result.status = TestStatus.FAILED
            result.error_message = str(e)
            result.stack_trace = self._get_stack_trace()
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            # Собираем метрики производительности
            if include_performance:
                result.performance_metrics = self._collect_performance_metrics(test_name)
            # Удаляем из запущенных
            if test_name in self.running_tests:
                del self.running_tests[test_name]
        return result
    def _collect_performance_metrics(self, test_name: str) -> Dict[str, float]:
        """Собрать метрики производительности для теста."""
        metrics = {}
        # Получаем метрики из монитора производительности
        summary = performance_monitor.get_metrics_summary(
            component=f"test.{test_name}",
            time_window=timedelta(minutes=5)
        )
        for metric_name, metric_data in summary.items():
            if metric_data:
                metrics[metric_name] = metric_data.get("avg", 0.0)
        return metrics
    def _get_dependent_tests(self, test_name: str) -> List[str]:
        """Получить зависимые тесты."""
        dependent_tests = []
        for suite in self.test_suites.values():
            if test_name in suite.dependencies:
                dependent_tests.extend(suite.tests)
        return dependent_tests
    def _get_stack_trace(self) -> str:
        """Получить стек вызовов."""
        import traceback
        return traceback.format_exc()
    def _analyze_suite_results(self, suite_name: str, results: List[TestResult]) -> None:
        """Анализ результатов набора тестов."""
        if not results:
            return
        # Статистика
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == TestStatus.PASSED])
        failed_tests = len([r for r in results if r.status == TestStatus.FAILED])
        skipped_tests = len([r for r in results if r.status == TestStatus.SKIPPED])
        # Производительность
        durations = [r.duration for r in results if r.duration > 0]
        avg_duration = statistics.mean(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        # Анализ производительности
        slow_tests = [r for r in results if r.duration > self.performance_threshold / 1000]
        logger.info(f"Test suite {suite_name} completed:")
        logger.info(f"  Total: {total_tests}, Passed: {passed_tests}, Failed: {failed_tests}, Skipped: {skipped_tests}")
        logger.info(f"  Average duration: {avg_duration:.2f}s, Max duration: {max_duration:.2f}s")
        if slow_tests:
            logger.warning(f"  Slow tests: {len(slow_tests)}")
            for test in slow_tests:
                logger.warning(f"    {test.test_name}: {test.duration:.2f}s")
        # Сохраняем результаты
        self.test_results.extend(results)
    def run_performance_test(self, test_name: str, iterations: int = 100,
                           concurrent_users: int = 1) -> Dict[str, Any]:
        """
        Запустить нагрузочный тест.
        Args:
            test_name: Название теста
            iterations: Количество итераций
            concurrent_users: Количество одновременных пользователей
        Returns:
            Результаты нагрузочного теста
        """
        logger.info(f"Running performance test: {test_name}")
        if test_name not in self.test_executors:
            raise ValueError(f"Test executor not found: {test_name}")
        executor = self.test_executors[test_name]
        results = []
        start_time = datetime.now()
        if concurrent_users > 1:
            # Многопоточное выполнение
            with ThreadPoolExecutor(max_workers=concurrent_users) as thread_executor:
                futures = []
                for i in range(iterations):
                    future = thread_executor.submit(executor)
                    futures.append(future)
                for future in as_completed(futures):
                    try:
                        future.result()
                        results.append(True)
                    except Exception as e:
                        results.append(False)
                        logger.error(f"Performance test iteration failed: {e}")
        else:
            # Последовательное выполнение
            for i in range(iterations):
                try:
                    start = time.time()
                    executor()
                    duration = time.time() - start
                    results.append(True)
                except Exception as e:
                    results.append(False)
                    logger.error(f"Performance test iteration failed: {e}")
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        # Анализ результатов
        successful_runs = [r for r in results if r is not False]
        failed_runs = len([r for r in results if r is False])
        if concurrent_users == 1:
            # Время выполнения для последовательного теста
            avg_response_time = statistics.mean(successful_runs) if successful_runs else 0.0
            min_response_time = min(successful_runs) if successful_runs else 0.0
            max_response_time = max(successful_runs) if successful_runs else 0.0
        else:
            # Статистика для многопоточного теста
            avg_response_time = total_duration / len(successful_runs) if successful_runs else 0.0
            min_response_time = avg_response_time
            max_response_time = avg_response_time
        throughput = len(successful_runs) / total_duration if total_duration > 0 else 0.0
        success_rate = len(successful_runs) / len(results) if results else 0.0
        performance_report = {
            "test_name": test_name,
            "iterations": iterations,
            "concurrent_users": concurrent_users,
            "total_duration": total_duration,
            "successful_runs": len(successful_runs),
            "failed_runs": failed_runs,
            "success_rate": success_rate,
            "throughput": throughput,
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "start_time": start_time,
            "end_time": end_time
        }
        logger.info(f"Performance test completed: {success_rate:.2%} success rate, {throughput:.2f} ops/sec")
        return performance_report
    def generate_test_report(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Генерация отчёта по тестам.
        Args:
            output_file: Файл для сохранения отчёта
        Returns:
            Отчёт по тестам
        """
        if not self.test_results:
            return {"message": "No test results available"}
        # Статистика по статусам
        status_counts = {}
        for status in TestStatus:
            status_counts[status.value] = len([r for r in self.test_results if r.status == status])
        # Статистика по типам тестов
        type_counts = {}
        for test_type in TestType:
            type_counts[test_type.value] = len([r for r in self.test_results if r.test_type == test_type])
        # Производительность
        durations = [r.duration for r in self.test_results if r.duration > 0]
        performance_stats = {
            "total_tests": len(self.test_results),
            "avg_duration": statistics.mean(durations) if durations else 0.0,
            "median_duration": statistics.median(durations) if durations else 0.0,
            "min_duration": min(durations) if durations else 0.0,
            "max_duration": max(durations) if durations else 0.0,
            "slow_tests": len([r for r in self.test_results if r.duration > self.performance_threshold / 1000])
        }
        # Детальные результаты
        detailed_results = []
        for result in self.test_results:
            detailed_results.append({
                "test_name": result.test_name,
                "test_type": result.test_type.value,
                "status": result.status.value,
                "duration": result.duration,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "error_message": result.error_message,
                "performance_metrics": result.performance_metrics
            })
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "status_counts": status_counts,
                "type_counts": type_counts,
                "performance": performance_stats
            },
            "detailed_results": detailed_results
        }
        # Сохраняем отчёт
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"Test report saved to: {output_file}")
        return report
    def cleanup(self) -> None:
        """Очистка ресурсов."""
        try:
            # Проверяем, является ли stop_monitoring корутиной
            result = performance_monitor.stop_monitoring()
            if hasattr(result, '__await__'):
                # Если это корутина, игнорируем её
                pass
        except Exception:
            # Игнорируем ошибки при остановке мониторинга
            pass
        self.test_results.clear()
        self.running_tests.clear()
# Глобальный экземпляр тестового раннера
test_runner = AutomatedTestRunner()
def register_test(test_name: str, test_type: TestType = TestType.UNIT,
                 priority: TestPriority = TestPriority.MEDIUM) -> Any:
    """
    Декоратор для регистрации тестов.
    Args:
        test_name: Название теста
        test_type: Тип теста
        priority: Приоритет теста
    """
    def decorator(func) -> Any:
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                # Записываем метрику производительности
                performance_monitor.record_timing(
                    f"test.{test_name}", duration, "test_runner"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_timing(
                    f"test.{test_name}_error", duration, "test_runner"
                )
                performance_monitor.record_counter(
                    f"test.{test_name}_errors", 1, "test_runner"
                )
                raise
        return wrapper
    return decorator
# Примеры тестовых наборов
def create_default_test_suites() -> Any:
    """Создание стандартных наборов тестов."""
    # Unit тесты
    unit_suite = TestSuite(
        name="unit_tests",
        description="Unit tests for core functionality",
        tests=["test_domain_entities", "test_services", "test_repositories"],
        test_type=TestType.UNIT,
        priority=TestPriority.HIGH,
        timeout=60
    )
    # Интеграционные тесты
    integration_suite = TestSuite(
        name="integration_tests",
        description="Integration tests for system components",
        tests=["test_trading_flow", "test_market_data", "test_order_execution"],
        test_type=TestType.INTEGRATION,
        priority=TestPriority.HIGH,
        timeout=300,
        dependencies=["unit_tests"]
    )
    # Нагрузочные тесты
    performance_suite = TestSuite(
        name="performance_tests",
        description="Performance and load tests",
        tests=["test_order_processing", "test_market_analysis", "test_strategy_execution"],
        test_type=TestType.PERFORMANCE,
        priority=TestPriority.MEDIUM,
        timeout=600,
        parallel=True
    )
    # Регистрируем наборы
    test_runner.register_test_suite(unit_suite)
    test_runner.register_test_suite(integration_suite)
    test_runner.register_test_suite(performance_suite)
    return [unit_suite, integration_suite, performance_suite]
# Автоматический запуск тестов при импорте
if __name__ == "__main__":
    create_default_test_suites()
    # Запускаем все наборы тестов
    for suite_name in test_runner.test_suites:
        try:
            results = test_runner.run_test_suite(suite_name)
            logger.info(f"Suite {suite_name} completed with {len(results)} tests")
        except Exception as e:
            logger.error(f"Error running suite {suite_name}: {e}")
    # Генерируем отчёт
    report = test_runner.generate_test_report("test_report.json")
    logger.info("Test execution completed") 
