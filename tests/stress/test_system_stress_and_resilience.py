#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Стресс-тесты системной устойчивости и отказоустойчивости.
Тестирование экстремальных нагрузок и восстановления после сбоев.
"""
import asyncio
import pytest
import time
import threading
import multiprocessing
import random
import gc
import resource
import psutil
import os
import signal
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import AsyncMock, Mock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import statistics


class TestSystemStressAndResilience:
    """Стресс-тесты системной устойчивости."""

    @pytest.fixture
    def stress_config(self):
        """Конфигурация стресс-тестов."""
        return {
            'max_concurrent_operations': 1000,
            'stress_duration_seconds': 300,      # 5 минут
            'memory_stress_mb': 1000,           # 1GB стресс памяти
            'cpu_stress_threads': multiprocessing.cpu_count(),
            'network_stress_connections': 500,
            'disk_stress_operations': 10000,
            'failure_injection_rate': 0.05,     # 5% отказов
            'recovery_timeout_seconds': 30
        }

    @pytest.fixture
    def system_monitor(self):
        """Монитор системных ресурсов."""
        class SystemMonitor:
            def __init__(self):
                self.process = psutil.Process()
                self.initial_memory = self.process.memory_info().rss / 1024 / 1024
                self.initial_cpu = self.process.cpu_percent()
                self.peak_memory = self.initial_memory
                self.peak_cpu = self.initial_cpu
                self.monitoring = False
                self.metrics = []
            
            def start_monitoring(self):
                """Запуск мониторинга."""
                self.monitoring = True
                threading.Thread(target=self._monitor_loop, daemon=True).start()
            
            def stop_monitoring(self):
                """Остановка мониторинга."""
                self.monitoring = False
            
            def _monitor_loop(self):
                """Цикл мониторинга."""
                while self.monitoring:
                    current_memory = self.process.memory_info().rss / 1024 / 1024
                    current_cpu = self.process.cpu_percent()
                    
                    self.peak_memory = max(self.peak_memory, current_memory)
                    self.peak_cpu = max(self.peak_cpu, current_cpu)
                    
                    self.metrics.append({
                        'timestamp': time.time(),
                        'memory_mb': current_memory,
                        'cpu_percent': current_cpu,
                        'threads_count': self.process.num_threads()
                    })
                    
                    time.sleep(0.1)  # Мониторинг каждые 100ms
            
            def get_stats(self) -> Dict:
                """Получение статистики."""
                if not self.metrics:
                    return {}
                
                memory_values = [m['memory_mb'] for m in self.metrics]
                cpu_values = [m['cpu_percent'] for m in self.metrics]
                
                return {
                    'peak_memory_mb': self.peak_memory,
                    'peak_cpu_percent': self.peak_cpu,
                    'avg_memory_mb': statistics.mean(memory_values),
                    'avg_cpu_percent': statistics.mean(cpu_values),
                    'memory_increase_mb': self.peak_memory - self.initial_memory,
                    'samples_count': len(self.metrics)
                }
        
        return SystemMonitor()

    def test_extreme_concurrent_load_stress(self, stress_config, system_monitor):
        """Стресс-тест экстремальной параллельной нагрузки."""
        system_monitor.start_monitoring()
        
        # Результаты операций
        operation_results = {
            'completed': 0,
            'failed': 0,
            'timeouts': 0,
            'errors': []
        }
        
        results_lock = threading.Lock()
        
        def stress_operation(operation_id: int) -> Dict:
            """Стрессовая операция."""
            try:
                start_time = time.time()
                
                # Симуляция сложной операции
                # 1. Обработка данных
                data = [Decimal(str(random.randint(1, 100000))) for _ in range(1000)]
                result = sum(data) / len(data)
                
                # 2. Симуляция I/O операций
                time.sleep(random.uniform(0.001, 0.01))  # 1-10ms
                
                # 3. Симуляция сетевых операций
                network_delay = random.uniform(0.002, 0.02)  # 2-20ms
                time.sleep(network_delay)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                # Симуляция случайных отказов
                if random.random() < stress_config['failure_injection_rate']:
                    raise Exception(f"Simulated failure in operation {operation_id}")
                
                with results_lock:
                    operation_results['completed'] += 1
                
                return {
                    'operation_id': operation_id,
                    'status': 'SUCCESS',
                    'execution_time': execution_time,
                    'result': float(result)
                }
                
            except Exception as e:
                with results_lock:
                    operation_results['failed'] += 1
                    operation_results['errors'].append(str(e))
                
                return {
                    'operation_id': operation_id,
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        # Запуск стресс-теста
        start_time = time.time()
        max_workers = stress_config['max_concurrent_operations']
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Создание большого количества операций
            futures = []
            for i in range(max_workers * 5):  # 5x больше операций чем потоков
                future = executor.submit(stress_operation, i)
                futures.append(future)
            
            # Ожидание завершения с таймаутом
            completed_futures = []
            for future in futures:
                try:
                    result = future.result(timeout=1.0)  # 1 секунда таймаут
                    completed_futures.append(result)
                except Exception as e:
                    with results_lock:
                        operation_results['timeouts'] += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        system_monitor.stop_monitoring()
        system_stats = system_monitor.get_stats()
        
        # Анализ результатов
        total_operations = sum([
            operation_results['completed'],
            operation_results['failed'],
            operation_results['timeouts']
        ])
        
        success_rate = operation_results['completed'] / total_operations if total_operations > 0 else 0
        throughput = operation_results['completed'] / total_time
        
        # Проверки стресс-теста
        assert total_operations > 0
        assert success_rate >= 0.8  # Минимум 80% успешных операций
        assert throughput >= 100     # Минимум 100 операций/сек
        assert system_stats['peak_memory_mb'] < 2000  # Максимум 2GB памяти
        assert len(operation_results['errors']) < total_operations * 0.3  # < 30% ошибок

    def test_memory_pressure_stress(self, stress_config, system_monitor):
        """Стресс-тест давления на память."""
        system_monitor.start_monitoring()
        
        # Создание большого объема данных в памяти
        memory_pools = []
        
        try:
            # Постепенное увеличение использования памяти
            target_memory_mb = stress_config['memory_stress_mb']
            chunk_size_mb = 50  # 50MB чанки
            chunks_count = target_memory_mb // chunk_size_mb
            
            for i in range(chunks_count):
                # Создание чанка данных
                chunk_data = []
                for j in range(50000):  # ~50MB данных
                    market_record = {
                        'timestamp': datetime.now(),
                        'price': Decimal(str(50000 + j)),
                        'volume': Decimal(str(random.uniform(0.1, 100.0))),
                        'order_id': f'order_{i}_{j}',
                        'metadata': f'metadata_string_{j}' * 10  # Дополнительная память
                    }
                    chunk_data.append(market_record)
                
                memory_pools.append(chunk_data)
                
                # Проверка использования памяти
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                if current_memory > target_memory_mb * 2:  # Защита от чрезмерного потребления
                    break
                
                # Небольшая пауза между чанками
                time.sleep(0.1)
            
            # Операции с данными в условиях высокого потребления памяти
            operations_count = 1000
            successful_operations = 0
            
            for i in range(operations_count):
                try:
                    # Выбор случайного пула данных
                    if memory_pools:
                        pool = random.choice(memory_pools)
                        if pool:
                            # Операции с данными
                            sample_data = random.sample(pool, min(100, len(pool)))
                            prices = [record['price'] for record in sample_data]
                            avg_price = sum(prices) / len(prices)
                            
                            # Дополнительные вычисления
                            volatility = statistics.stdev(prices) if len(prices) > 1 else Decimal('0')
                            
                            successful_operations += 1
                
                except MemoryError:
                    # Ожидаемая ошибка при нехватке памяти
                    break
                except Exception:
                    # Другие ошибки
                    continue
            
            # Постепенная очистка памяти
            for i in range(len(memory_pools)):
                if i % 5 == 0:  # Очистка каждого 5-го пула
                    memory_pools[i] = None
                    gc.collect()
                    time.sleep(0.01)
            
        finally:
            # Принудительная очистка памяти
            memory_pools.clear()
            gc.collect()
            
            system_monitor.stop_monitoring()
        
        system_stats = system_monitor.get_stats()
        
        # Проверки памяти
        assert successful_operations >= operations_count * 0.5  # Минимум 50% операций
        assert system_stats['memory_increase_mb'] < stress_config['memory_stress_mb'] * 2
        
        # Проверка освобождения памяти
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_cleanup_ratio = (system_stats['peak_memory_mb'] - final_memory) / system_stats['peak_memory_mb']
        assert memory_cleanup_ratio >= 0.7  # Минимум 70% памяти освобождено

    def test_cpu_intensive_stress(self, stress_config, system_monitor):
        """Стресс-тест CPU-интенсивных операций."""
        system_monitor.start_monitoring()
        
        # CPU-интенсивные вычисления
        def cpu_intensive_worker(worker_id: int, duration_seconds: int) -> Dict:
            """CPU-интенсивная работа."""
            start_time = time.time()
            operations_count = 0
            
            while (time.time() - start_time) < duration_seconds:
                # Комплексные математические вычисления
                # 1. Расчет технических индикаторов
                prices = [Decimal(str(50000 + i * random.uniform(-100, 100))) for i in range(1000)]
                
                # SMA
                sma_period = 20
                sma = sum(prices[-sma_period:]) / sma_period
                
                # EMA
                ema = prices[0]
                multiplier = Decimal('2') / (sma_period + 1)
                for price in prices[1:]:
                    ema = (price * multiplier) + (ema * (1 - multiplier))
                
                # RSI
                gains = []
                losses = []
                for i in range(1, min(len(prices), sma_period + 1)):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                    else:
                        losses.append(abs(change))
                
                avg_gain = sum(gains) / len(gains) if gains else Decimal('0')
                avg_loss = sum(losses) / len(losses) if losses else Decimal('0')
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = Decimal('100')
                
                # 2. Статистические вычисления
                mean_price = sum(prices) / len(prices)
                variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
                std_dev = variance ** Decimal('0.5')
                
                # 3. Корреляционный анализ
                prices2 = [p * Decimal(str(random.uniform(0.95, 1.05))) for p in prices]
                correlation = self._calculate_correlation(prices, prices2)
                
                operations_count += 1
                
                # Небольшая пауза для предотвращения полной блокировки
                if operations_count % 100 == 0:
                    time.sleep(0.001)
            
            end_time = time.time()
            
            return {
                'worker_id': worker_id,
                'duration': end_time - start_time,
                'operations_completed': operations_count,
                'operations_per_second': operations_count / (end_time - start_time)
            }
        
        # Запуск CPU-интенсивных работников
        stress_duration = 30  # 30 секунд стресса
        workers_count = stress_config['cpu_stress_threads']
        
        with ProcessPoolExecutor(max_workers=workers_count) as executor:
            futures = []
            for i in range(workers_count):
                future = executor.submit(cpu_intensive_worker, i, stress_duration)
                futures.append(future)
            
            # Сбор результатов
            worker_results = []
            for future in futures:
                try:
                    result = future.result(timeout=stress_duration + 10)
                    worker_results.append(result)
                except Exception as e:
                    # Логирование ошибок воркеров
                    pass
        
        system_monitor.stop_monitoring()
        system_stats = system_monitor.get_stats()
        
        # Анализ результатов CPU стресса
        if worker_results:
            total_operations = sum(w['operations_completed'] for w in worker_results)
            avg_ops_per_second = statistics.mean(w['operations_per_second'] for w in worker_results)
            
            # Проверки CPU стресса
            assert len(worker_results) >= workers_count * 0.8  # Минимум 80% воркеров завершились
            assert total_operations > 0
            assert avg_ops_per_second >= 10  # Минимум 10 операций/сек на воркер
            assert system_stats['peak_cpu_percent'] >= 50  # CPU использование >= 50%

    def _calculate_correlation(self, series1: List[Decimal], series2: List[Decimal]) -> Decimal:
        """Расчет корреляции между двумя рядами."""
        if len(series1) != len(series2) or len(series1) < 2:
            return Decimal('0')
        
        n = len(series1)
        mean1 = sum(series1) / n
        mean2 = sum(series2) / n
        
        numerator = sum((x - mean1) * (y - mean2) for x, y in zip(series1, series2))
        
        sum_sq_x = sum((x - mean1) ** 2 for x in series1)
        sum_sq_y = sum((y - mean2) ** 2 for y in series2)
        
        denominator = (sum_sq_x * sum_sq_y) ** Decimal('0.5')
        
        return numerator / denominator if denominator != 0 else Decimal('0')

    def test_network_stress_simulation(self, stress_config):
        """Стресс-тест сетевых операций."""
        # Симуляция множественных сетевых соединений
        network_operations = []
        failed_operations = 0
        successful_operations = 0
        
        async def network_operation(operation_id: int) -> Dict:
            """Симуляция сетевой операции."""
            try:
                # Симуляция различных типов сетевых операций
                operation_type = random.choice(['API_CALL', 'WEBSOCKET', 'DATABASE', 'CACHE'])
                
                # Различные латентности для разных типов операций
                latencies = {
                    'API_CALL': random.uniform(0.01, 0.1),      # 10-100ms
                    'WEBSOCKET': random.uniform(0.001, 0.01),   # 1-10ms
                    'DATABASE': random.uniform(0.005, 0.05),    # 5-50ms
                    'CACHE': random.uniform(0.0001, 0.001)      # 0.1-1ms
                }
                
                await asyncio.sleep(latencies[operation_type])
                
                # Симуляция данных ответа
                response_data = {
                    'operation_id': operation_id,
                    'type': operation_type,
                    'status': 'SUCCESS',
                    'data_size': random.randint(100, 10000),  # размер ответа в байтах
                    'latency': latencies[operation_type]
                }
                
                # Симуляция сетевых сбоев
                if random.random() < 0.02:  # 2% сбоев
                    raise Exception(f"Network timeout for operation {operation_id}")
                
                return response_data
                
            except Exception as e:
                return {
                    'operation_id': operation_id,
                    'status': 'FAILED',
                    'error': str(e)
                }
        
        async def run_network_stress():
            """Запуск сетевого стресс-теста."""
            nonlocal failed_operations, successful_operations
            
            # Создание множественных параллельных операций
            operations_count = stress_config['network_stress_connections']
            tasks = []
            
            for i in range(operations_count):
                task = network_operation(i)
                tasks.append(task)
            
            # Выполнение всех операций параллельно
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Обработка результатов
            for result in results:
                if isinstance(result, Exception):
                    failed_operations += 1
                elif isinstance(result, dict):
                    if result.get('status') == 'SUCCESS':
                        successful_operations += 1
                    else:
                        failed_operations += 1
                    network_operations.append(result)
        
        # Запуск асинхронного стресс-теста
        start_time = time.time()
        asyncio.run(run_network_stress())
        end_time = time.time()
        
        total_time = end_time - start_time
        total_operations = successful_operations + failed_operations
        
        # Анализ сетевого стресса
        if network_operations:
            successful_ops = [op for op in network_operations if op.get('status') == 'SUCCESS']
            if successful_ops:
                avg_latency = statistics.mean(op['latency'] for op in successful_ops)
                max_latency = max(op['latency'] for op in successful_ops)
                
                # Группировка по типам операций
                operation_types = {}
                for op in successful_ops:
                    op_type = op['type']
                    if op_type not in operation_types:
                        operation_types[op_type] = []
                    operation_types[op_type].append(op['latency'])
        
        # Проверки сетевого стресса
        assert total_operations > 0
        success_rate = successful_operations / total_operations
        assert success_rate >= 0.9  # Минимум 90% успешных операций
        assert total_time < 60  # Операции должны завершиться за 60 секунд
        
        if network_operations:
            throughput = successful_operations / total_time
            assert throughput >= 50  # Минимум 50 операций/сек

    def test_disk_io_stress(self, stress_config):
        """Стресс-тест дисковых операций."""
        import tempfile
        import os
        
        # Создание временной директории для тестов
        temp_dir = tempfile.mkdtemp(prefix='stress_test_')
        
        try:
            # Параметры дискового стресса
            operations_count = stress_config['disk_stress_operations']
            file_size_kb = 100  # 100KB файлы
            
            disk_operations = {
                'writes': 0,
                'reads': 0,
                'deletes': 0,
                'write_errors': 0,
                'read_errors': 0,
                'delete_errors': 0
            }
            
            # Данные для записи
            test_data = 'x' * (file_size_kb * 1024)  # 100KB данных
            
            # Дисковые операции
            file_paths = []
            
            start_time = time.time()
            
            # Фаза записи
            for i in range(operations_count // 3):
                file_path = os.path.join(temp_dir, f'test_file_{i}.txt')
                try:
                    with open(file_path, 'w') as f:
                        f.write(test_data)
                        f.flush()
                        os.fsync(f.fileno())  # Принудительная синхронизация
                    
                    file_paths.append(file_path)
                    disk_operations['writes'] += 1
                    
                except Exception:
                    disk_operations['write_errors'] += 1
            
            # Фаза чтения
            for file_path in file_paths[:operations_count // 3]:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        assert len(content) == len(test_data)
                    
                    disk_operations['reads'] += 1
                    
                except Exception:
                    disk_operations['read_errors'] += 1
            
            # Фаза удаления
            for file_path in file_paths[:operations_count // 3]:
                try:
                    os.remove(file_path)
                    disk_operations['deletes'] += 1
                    
                except Exception:
                    disk_operations['delete_errors'] += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Анализ дисковых операций
            total_operations = (disk_operations['writes'] + 
                              disk_operations['reads'] + 
                              disk_operations['deletes'])
            
            total_errors = (disk_operations['write_errors'] + 
                           disk_operations['read_errors'] + 
                           disk_operations['delete_errors'])
            
            # Проверки дискового стресса
            assert total_operations > 0
            assert total_time > 0
            
            operations_per_second = total_operations / total_time
            error_rate = total_errors / (total_operations + total_errors) if (total_operations + total_errors) > 0 else 0
            
            assert operations_per_second >= 50   # Минимум 50 операций/сек
            assert error_rate <= 0.05           # Максимум 5% ошибок
            
        finally:
            # Очистка временных файлов
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception:
                pass

    def test_cascade_failure_recovery(self, stress_config):
        """Тест восстановления после каскадных отказов."""
        # Симуляция системы с взаимозависимыми компонентами
        class SystemComponent:
            def __init__(self, name: str, dependencies: List[str] = None):
                self.name = name
                self.dependencies = dependencies or []
                self.status = 'HEALTHY'
                self.failure_count = 0
                self.recovery_attempts = 0
                self.last_failure_time = None
            
            def is_healthy(self) -> bool:
                return self.status == 'HEALTHY'
            
            def fail(self, reason: str = 'Unknown'):
                self.status = 'FAILED'
                self.failure_count += 1
                self.last_failure_time = time.time()
            
            def attempt_recovery(self) -> bool:
                """Попытка восстановления компонента."""
                self.recovery_attempts += 1
                
                # Симуляция успешности восстановления (90% вероятность)
                if random.random() < 0.9:
                    self.status = 'HEALTHY'
                    return True
                else:
                    return False
        
        # Создание системных компонентов
        components = {
            'database': SystemComponent('database'),
            'cache': SystemComponent('cache', ['database']),
            'api_gateway': SystemComponent('api_gateway', ['database', 'cache']),
            'order_service': SystemComponent('order_service', ['api_gateway', 'database']),
            'market_data': SystemComponent('market_data', ['api_gateway']),
            'risk_service': SystemComponent('risk_service', ['order_service', 'market_data']),
            'trading_engine': SystemComponent('trading_engine', ['order_service', 'risk_service'])
        }
        
        def check_component_health(component_name: str) -> bool:
            """Проверка здоровья компонента с учетом зависимостей."""
            component = components[component_name]
            
            if not component.is_healthy():
                return False
            
            # Проверка зависимостей
            for dep_name in component.dependencies:
                if dep_name in components and not components[dep_name].is_healthy():
                    return False
            
            return True
        
        def trigger_cascade_failure():
            """Запуск каскадного отказа."""
            # Начинаем с отказа базы данных
            components['database'].fail('Connection timeout')
            
            # Каскадные отказы зависимых компонентов
            cascade_order = ['cache', 'api_gateway', 'order_service', 'market_data', 'risk_service', 'trading_engine']
            
            for comp_name in cascade_order:
                if not check_component_health(comp_name):
                    components[comp_name].fail(f'Dependency failure')
                    time.sleep(0.1)  # Задержка каскада
        
        def attempt_system_recovery() -> Dict:
            """Попытка восстановления системы."""
            recovery_log = []
            
            # Порядок восстановления (обратный каскаду)
            recovery_order = ['database', 'cache', 'api_gateway', 'order_service', 'market_data', 'risk_service', 'trading_engine']
            
            for comp_name in recovery_order:
                component = components[comp_name]
                
                if not component.is_healthy():
                    # Проверка готовности зависимостей
                    dependencies_ready = all(
                        check_component_health(dep) for dep in component.dependencies
                    )
                    
                    if dependencies_ready:
                        success = component.attempt_recovery()
                        recovery_log.append({
                            'component': comp_name,
                            'recovery_attempt': component.recovery_attempts,
                            'success': success,
                            'timestamp': time.time()
                        })
                        
                        if success:
                            # Небольшая пауза после успешного восстановления
                            time.sleep(0.05)
            
            return recovery_log
        
        # Тестирование каскадного отказа и восстановления
        start_time = time.time()
        
        # 1. Проверка исходного состояния
        initial_health = {name: comp.is_healthy() for name, comp in components.items()}
        assert all(initial_health.values()), "All components should be healthy initially"
        
        # 2. Запуск каскадного отказа
        trigger_cascade_failure()
        
        # Проверка отказа системы
        post_failure_health = {name: check_component_health(name) for name, comp in components.items()}
        failed_components = [name for name, healthy in post_failure_health.items() if not healthy]
        
        assert len(failed_components) >= 5, "Cascade failure should affect multiple components"
        
        # 3. Циклы восстановления
        recovery_cycles = []
        max_recovery_attempts = 10
        
        for cycle in range(max_recovery_attempts):
            recovery_log = attempt_system_recovery()
            recovery_cycles.append(recovery_log)
            
            # Проверка текущего состояния системы
            current_health = {name: check_component_health(name) for name, comp in components.items()}
            healthy_components = [name for name, healthy in current_health.items() if healthy]
            
            if len(healthy_components) == len(components):
                # Полное восстановление
                break
            
            time.sleep(0.1)  # Пауза между циклами восстановления
        
        end_time = time.time()
        recovery_time = end_time - start_time
        
        # Анализ восстановления
        final_health = {name: check_component_health(name) for name, comp in components.items()}
        recovered_components = [name for name, healthy in final_health.items() if healthy]
        
        total_recovery_attempts = sum(comp.recovery_attempts for comp in components.values())
        total_failures = sum(comp.failure_count for comp in components.values())
        
        # Проверки восстановления
        assert len(recovered_components) >= len(components) * 0.8  # Минимум 80% компонентов восстановлено
        assert recovery_time < stress_config['recovery_timeout_seconds']  # Восстановление в пределах таймаута
        assert total_recovery_attempts > 0  # Были попытки восстановления
        assert len(recovery_cycles) <= max_recovery_attempts  # Не превышен лимит циклов

    def test_resource_exhaustion_handling(self, stress_config, system_monitor):
        """Тест обработки исчерпания ресурсов."""
        system_monitor.start_monitoring()
        
        resource_exhaustion_tests = []
        
        # 1. Тест исчерпания дескрипторов файлов
        def test_file_descriptors_exhaustion():
            """Тест исчерпания файловых дескрипторов."""
            import tempfile
            
            open_files = []
            max_files = 100  # Ограниченное количество для теста
            
            try:
                for i in range(max_files * 2):  # Пытаемся открыть больше чем лимит
                    try:
                        temp_file = tempfile.NamedTemporaryFile(delete=False)
                        open_files.append(temp_file)
                    except OSError as e:
                        # Ожидаемая ошибка при исчерпании дескрипторов
                        return {
                            'test': 'file_descriptors',
                            'status': 'HANDLED',
                            'files_opened': len(open_files),
                            'error': str(e)
                        }
            
            finally:
                # Очистка файлов
                for f in open_files:
                    try:
                        f.close()
                        os.unlink(f.name)
                    except:
                        pass
            
            return {
                'test': 'file_descriptors',
                'status': 'COMPLETED',
                'files_opened': len(open_files)
            }
        
        # 2. Тест исчерпания потоков
        def test_thread_exhaustion():
            """Тест исчерпания потоков."""
            threads = []
            max_threads = 200  # Ограниченное количество
            
            def dummy_thread_work():
                time.sleep(1)  # Короткая работа
            
            try:
                for i in range(max_threads):
                    try:
                        thread = threading.Thread(target=dummy_thread_work)
                        thread.start()
                        threads.append(thread)
                    except RuntimeError as e:
                        # Ожидаемая ошибка при исчерпании потоков
                        return {
                            'test': 'threads',
                            'status': 'HANDLED',
                            'threads_created': len(threads),
                            'error': str(e)
                        }
            
            finally:
                # Ожидание завершения потоков
                for thread in threads:
                    try:
                        thread.join(timeout=2)
                    except:
                        pass
            
            return {
                'test': 'threads',
                'status': 'COMPLETED',
                'threads_created': len(threads)
            }
        
        # 3. Тест исчерпания памяти (контролируемый)
        def test_controlled_memory_exhaustion():
            """Контролируемый тест исчерпания памяти."""
            memory_chunks = []
            chunk_size = 10 * 1024 * 1024  # 10MB чанки
            max_memory_mb = 500  # Максимум 500MB для теста
            max_chunks = max_memory_mb // 10
            
            try:
                for i in range(max_chunks):
                    try:
                        # Создание чанка памяти
                        chunk = bytearray(chunk_size)
                        memory_chunks.append(chunk)
                        
                        # Проверка текущего использования памяти
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        if current_memory > max_memory_mb * 2:
                            break
                            
                    except MemoryError as e:
                        return {
                            'test': 'memory',
                            'status': 'HANDLED',
                            'chunks_allocated': len(memory_chunks),
                            'memory_mb': len(memory_chunks) * 10,
                            'error': str(e)
                        }
            
            finally:
                # Очистка памяти
                memory_chunks.clear()
                gc.collect()
            
            return {
                'test': 'memory',
                'status': 'COMPLETED',
                'chunks_allocated': len(memory_chunks),
                'memory_mb': max_chunks * 10
            }
        
        # Запуск тестов исчерпания ресурсов
        resource_tests = [
            test_file_descriptors_exhaustion,
            test_thread_exhaustion,
            test_controlled_memory_exhaustion
        ]
        
        for test_func in resource_tests:
            try:
                result = test_func()
                resource_exhaustion_tests.append(result)
            except Exception as e:
                resource_exhaustion_tests.append({
                    'test': test_func.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        system_monitor.stop_monitoring()
        system_stats = system_monitor.get_stats()
        
        # Проверки тестов исчерпания ресурсов
        assert len(resource_exhaustion_tests) == len(resource_tests)
        
        # Все тесты должны завершиться корректно (обработать исчерпание или завершиться)
        for test_result in resource_exhaustion_tests:
            assert test_result['status'] in ['HANDLED', 'COMPLETED']
        
        # Система должна остаться стабильной после тестов
        assert system_stats['peak_memory_mb'] < 2000  # Разумное потребление памяти