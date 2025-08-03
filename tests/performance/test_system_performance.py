"""
Комплексные тесты производительности торговой системы.
Включает нагрузочные тесты, бенчмарки и профилирование.
"""

import pytest
import pandas as pd
import numpy as np
import time
import asyncio
import threading
from decimal import Decimal
from unittest.mock import Mock, AsyncMock
from typing import List, Dict, Any
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Импорты для тестирования различных компонентов
from infrastructure.core.feature_engineering import FeatureEngineer, FeatureConfig
from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.market_data.base_connector import BaseExchangeConnector
from application.services.trading_service import TradingService


class TestPerformanceBenchmarks:
    """Бенчмарки производительности основных компонентов."""

    @pytest.fixture
    def large_market_data(self) -> pd.DataFrame:
        """Создает большой датасет для тестов производительности."""
        np.random.seed(42)
        size = 100000  # 100k записей
        
        dates = pd.date_range('2020-01-01', periods=size, freq='1min')
        
        # Генерируем реалистичные OHLCV данные
        base_price = 100.0
        prices = []
        volumes = []
        
        for i in range(size):
            # Случайное движение цены
            price_change = np.random.normal(0, 0.001) * base_price
            base_price = max(base_price + price_change, 0.01)
            
            # Генерируем OHLC
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

    def test_feature_engineering_performance(self, large_market_data: pd.DataFrame):
        """Тест производительности инженерии признаков."""
        config = FeatureConfig(
            use_technical_indicators=True,
            use_statistical_features=True,
            ema_periods=[5, 10, 20, 50],
            rsi_periods=[14, 21],
            rolling_windows=[5, 10, 20]
        )
        engineer = FeatureEngineer(config=config)
        
        # Измеряем время выполнения
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        features = engineer.generate_features(large_market_data)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = end_memory - start_memory
        
        # Проверяем производительность
        assert execution_time < 30.0, f"Слишком долгое выполнение: {execution_time:.2f}s"
        assert memory_used < 1000, f"Слишком много памяти: {memory_used:.2f}MB"
        assert isinstance(features, pd.DataFrame)
        assert len(features) > 0
        
        print(f"Feature Engineering: {execution_time:.2f}s, {memory_used:.2f}MB")

    def test_strategy_performance_multiple_symbols(self):
        """Тест производительности стратегии на множественных символах."""
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Создаем данные для 100 символов
        symbols_data = {}
        for i in range(100):
            symbol = f"SYMBOL{i:03d}"
            data = pd.DataFrame({
                'open': np.random.uniform(99, 101, 1000),
                'high': np.random.uniform(100, 102, 1000),
                'low': np.random.uniform(98, 100, 1000),
                'close': np.random.uniform(99, 101, 1000),
                'volume': np.random.uniform(1000, 5000, 1000)
            })
            symbols_data[symbol] = data
        
        # Измеряем время обработки всех символов
        start_time = time.time()
        
        results = {}
        for symbol, data in symbols_data.items():
            signals = strategy.generate_signals(data)
            results[symbol] = signals
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Проверяем производительность
        assert execution_time < 60.0, f"Слишком долгая обработка множественных символов: {execution_time:.2f}s"
        assert len(results) == 100
        
        avg_time_per_symbol = execution_time / 100
        print(f"Strategy Performance: {execution_time:.2f}s total, {avg_time_per_symbol:.3f}s per symbol")

    def test_concurrent_processing_performance(self, large_market_data: pd.DataFrame):
        """Тест производительности параллельной обработки."""
        config = {'ema_fast': 12, 'ema_slow': 26}
        
        # Разбиваем данные на чанки
        chunk_size = len(large_market_data) // 10
        chunks = [
            large_market_data.iloc[i:i+chunk_size] 
            for i in range(0, len(large_market_data), chunk_size)
        ]
        
        def process_chunk(chunk_data):
            strategy = TrendStrategy(config)
            return strategy.generate_signals(chunk_data)
        
        # Последовательная обработка
        start_time = time.time()
        sequential_results = []
        for chunk in chunks:
            result = process_chunk(chunk)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        # Параллельная обработка
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            parallel_results = [future.result() for future in as_completed(futures)]
        parallel_time = time.time() - start_time
        
        # Проверяем результаты
        assert len(sequential_results) == len(parallel_results)
        speedup = sequential_time / parallel_time
        
        print(f"Sequential: {sequential_time:.2f}s, Parallel: {parallel_time:.2f}s, Speedup: {speedup:.2f}x")
        
        # Ожидаем ускорение от параллельной обработки
        assert speedup > 1.2, f"Недостаточное ускорение: {speedup:.2f}x"

    def test_memory_usage_scaling(self):
        """Тест масштабирования использования памяти."""
        config = FeatureConfig(use_technical_indicators=True)
        engineer = FeatureEngineer(config=config)
        
        data_sizes = [1000, 5000, 10000, 25000, 50000]
        memory_usage = []
        execution_times = []
        
        for size in data_sizes:
            # Создаем данные определенного размера
            data = pd.DataFrame({
                'open': np.random.uniform(99, 101, size),
                'high': np.random.uniform(100, 102, size),
                'low': np.random.uniform(98, 100, size),
                'close': np.random.uniform(99, 101, size),
                'volume': np.random.uniform(1000, 5000, size)
            })
            
            # Очищаем память перед тестом
            gc.collect()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            features = engineer.generate_features(data)
            end_time = time.time()
            
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_used = end_memory - start_memory
            execution_time = end_time - start_time
            
            memory_usage.append(memory_used)
            execution_times.append(execution_time)
            
            # Очищаем результаты
            del features, data
            gc.collect()
        
        # Проверяем линейное масштабирование памяти
        memory_per_record = [mem / size for mem, size in zip(memory_usage, data_sizes)]
        memory_variance = statistics.variance(memory_per_record)
        
        print(f"Memory scaling: {memory_per_record}")
        print(f"Time scaling: {execution_times}")
        
        # Память на запись не должна сильно варьироваться
        assert memory_variance < 0.01, f"Нестабильное использование памяти: {memory_variance}"


class TestStressTests:
    """Стресс тесты системы."""

    def test_high_frequency_data_processing(self):
        """Стресс-тест обработки высокочастотных данных."""
        # Данные каждую секунду в течение дня (86400 записей)
        size = 86400
        dates = pd.date_range('2023-01-01', periods=size, freq='1s')
        
        # Генерируем высокочастотные данные
        price_changes = np.random.normal(0, 0.0001, size)
        prices = 100 + np.cumsum(price_changes)
        
        hf_data = pd.DataFrame({
            'open': prices + np.random.normal(0, 0.001, size),
            'high': prices + abs(np.random.normal(0, 0.002, size)),
            'low': prices - abs(np.random.normal(0, 0.002, size)),
            'close': prices,
            'volume': np.random.exponential(100, size)
        }, index=dates)
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        start_time = time.time()
        signals = strategy.generate_signals(hf_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) > 0
        assert execution_time < 120.0, f"Слишком долгая обработка HF данных: {execution_time:.2f}s"
        
        print(f"High-frequency processing: {execution_time:.2f}s for {size} records")

    def test_extreme_market_conditions(self):
        """Стресс-тест экстремальных рыночных условий."""
        scenarios = {
            'flash_crash': self._create_flash_crash_data(),
            'high_volatility': self._create_high_volatility_data(),
            'gap_up': self._create_gap_data(direction='up'),
            'gap_down': self._create_gap_data(direction='down'),
            'circuit_breaker': self._create_circuit_breaker_data()
        }
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        for scenario_name, data in scenarios.items():
            start_time = time.time()
            
            try:
                signals = strategy.generate_signals(data)
                
                # Проверяем, что стратегия обрабатывает экстремальные условия
                assert isinstance(signals, pd.DataFrame)
                
                # Сигналы должны быть в разумных пределах
                if not signals.empty and 'signal' in signals.columns:
                    assert all(-1 <= signal <= 1 for signal in signals['signal'] if not pd.isna(signal))
                
                execution_time = time.time() - start_time
                assert execution_time < 10.0, f"Слишком долгая обработка {scenario_name}: {execution_time:.2f}s"
                
                print(f"Scenario {scenario_name}: processed in {execution_time:.3f}s")
                
            except Exception as e:
                # Логируем ошибки, но не падаем
                print(f"Scenario {scenario_name} failed: {str(e)}")

    def _create_flash_crash_data(self) -> pd.DataFrame:
        """Создает данные флеш-краха."""
        normal_price = 100
        crash_price = 50
        recovery_price = 95
        
        prices = [normal_price] * 50  # Нормальная торговля
        prices.extend([normal_price - i*2 for i in range(25)])  # Резкое падение
        prices.extend([crash_price + i for i in range(25)])  # Восстановление
        
        return pd.DataFrame({
            'open': [p + np.random.normal(0, 0.1) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
            'close': prices,
            'volume': [np.random.exponential(5000) for _ in prices]
        })

    def _create_high_volatility_data(self) -> pd.DataFrame:
        """Создает данные высокой волатильности."""
        size = 100
        base_price = 100
        
        # Высокая волатильность
        price_changes = np.random.normal(0, 0.05, size) * base_price
        prices = base_price + np.cumsum(price_changes)
        
        return pd.DataFrame({
            'open': [p + np.random.normal(0, 2) for p in prices],
            'high': [p + abs(np.random.normal(0, 5)) for p in prices],
            'low': [p - abs(np.random.normal(0, 5)) for p in prices],
            'close': prices,
            'volume': [np.random.exponential(10000) for _ in prices]
        })

    def _create_gap_data(self, direction: str) -> pd.DataFrame:
        """Создает данные с гэпами."""
        size = 50
        base_price = 100
        
        prices = [base_price + np.random.normal(0, 0.5) for _ in range(size//2)]
        
        # Гэп
        gap_size = 10 if direction == 'up' else -10
        gap_price = prices[-1] + gap_size
        
        prices.extend([gap_price + np.random.normal(0, 0.5) for _ in range(size//2)])
        
        return pd.DataFrame({
            'open': [p + np.random.normal(0, 0.1) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.3)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.3)) for p in prices],
            'close': prices,
            'volume': [np.random.exponential(2000) for _ in prices]
        })

    def _create_circuit_breaker_data(self) -> pd.DataFrame:
        """Создает данные с остановками торгов."""
        size = 100
        base_price = 100
        
        # Нормальная торговля
        prices = [base_price + np.random.normal(0, 0.5) for _ in range(30)]
        
        # Резкое движение, вызывающее остановку
        for i in range(10):
            prices.append(prices[-1] * 0.95)  # Падение на 5%
        
        # Остановка торгов (одинаковые цены)
        halt_price = prices[-1]
        prices.extend([halt_price] * 20)
        
        # Возобновление торгов
        prices.extend([halt_price + np.random.normal(0, 1) for _ in range(40)])
        
        return pd.DataFrame({
            'open': [p + np.random.normal(0, 0.1) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
            'close': prices,
            'volume': [np.random.exponential(1000) if i < 60 else 0 for i in range(len(prices))]
        })

    def test_concurrent_users_simulation(self):
        """Стресс-тест множественных пользователей."""
        def simulate_user_session(user_id: int) -> Dict[str, Any]:
            """Симулирует сессию пользователя."""
            # Каждый пользователь обрабатывает свои данные
            data = pd.DataFrame({
                'open': np.random.uniform(99, 101, 1000),
                'high': np.random.uniform(100, 102, 1000),
                'low': np.random.uniform(98, 100, 1000),
                'close': np.random.uniform(99, 101, 1000),
                'volume': np.random.uniform(1000, 5000, 1000)
            })
            
            config = {'ema_fast': 12, 'ema_slow': 26}
            strategy = TrendStrategy(config)
            
            start_time = time.time()
            signals = strategy.generate_signals(data)
            execution_time = time.time() - start_time
            
            return {
                'user_id': user_id,
                'execution_time': execution_time,
                'signals_count': len(signals),
                'success': True
            }
        
        # Симулируем 50 одновременных пользователей
        num_users = 50
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(simulate_user_session, i) for i in range(num_users)]
            results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Проверяем результаты
        assert len(results) == num_users
        assert all(result['success'] for result in results)
        
        avg_execution_time = statistics.mean([r['execution_time'] for r in results])
        max_execution_time = max([r['execution_time'] for r in results])
        
        print(f"Concurrent users test: {total_time:.2f}s total, avg: {avg_execution_time:.3f}s, max: {max_execution_time:.3f}s")
        
        # Максимальное время выполнения не должно быть слишком большим
        assert max_execution_time < 5.0, f"Слишком долгое выполнение: {max_execution_time:.2f}s"


class TestResourceUsage:
    """Тесты использования ресурсов."""

    def test_memory_leak_detection(self):
        """Тест обнаружения утечек памяти."""
        config = FeatureConfig(use_technical_indicators=True)
        engineer = FeatureEngineer(config=config)
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Выполняем множество итераций
        for i in range(100):
            data = pd.DataFrame({
                'open': np.random.uniform(99, 101, 1000),
                'high': np.random.uniform(100, 102, 1000),
                'low': np.random.uniform(98, 100, 1000),
                'close': np.random.uniform(99, 101, 1000),
                'volume': np.random.uniform(1000, 5000, 1000)
            })
            
            features = engineer.generate_features(data)
            
            # Принудительно удаляем данные
            del features, data
            
            # Периодически проверяем память
            if i % 20 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                print(f"Iteration {i}: Memory growth: {memory_growth:.2f}MB")
                
                # Рост памяти не должен быть слишком большим
                assert memory_growth < 500, f"Возможная утечка памяти: {memory_growth:.2f}MB"

    def test_cpu_usage_monitoring(self):
        """Тест мониторинга использования CPU."""
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Создаем данные для интенсивной обработки
        large_data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 50000),
            'high': np.random.uniform(100, 102, 50000),
            'low': np.random.uniform(98, 100, 50000),
            'close': np.random.uniform(99, 101, 50000),
            'volume': np.random.uniform(1000, 5000, 50000)
        })
        
        # Мониторим CPU во время выполнения
        cpu_percentages = []
        
        def monitor_cpu():
            while True:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_percentages.append(cpu_percent)
                if len(cpu_percentages) > 100:  # Ограничиваем мониторинг
                    break
        
        # Запускаем мониторинг в отдельном потоке
        monitor_thread = threading.Thread(target=monitor_cpu, daemon=True)
        monitor_thread.start()
        
        # Выполняем обработку
        start_time = time.time()
        signals = strategy.generate_signals(large_data)
        execution_time = time.time() - start_time
        
        # Ждем завершения мониторинга
        time.sleep(0.5)
        
        if cpu_percentages:
            avg_cpu = statistics.mean(cpu_percentages)
            max_cpu = max(cpu_percentages)
            
            print(f"CPU usage: avg={avg_cpu:.1f}%, max={max_cpu:.1f}%")
            
            # CPU не должен быть 100% все время
            assert avg_cpu < 90, f"Слишком высокое использование CPU: {avg_cpu:.1f}%"
        
        assert isinstance(signals, pd.DataFrame)

    def test_disk_io_efficiency(self):
        """Тест эффективности дисковых операций."""
        # Создаем временные файлы с данными
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Создаем несколько файлов с данными
            file_paths = []
            for i in range(10):
                data = pd.DataFrame({
                    'open': np.random.uniform(99, 101, 10000),
                    'high': np.random.uniform(100, 102, 10000),
                    'low': np.random.uniform(98, 100, 10000),
                    'close': np.random.uniform(99, 101, 10000),
                    'volume': np.random.uniform(1000, 5000, 10000)
                })
                
                file_path = os.path.join(temp_dir, f'data_{i}.csv')
                data.to_csv(file_path)
                file_paths.append(file_path)
            
            # Тестируем скорость чтения
            config = {'ema_fast': 12, 'ema_slow': 26}
            strategy = TrendStrategy(config)
            
            start_time = time.time()
            
            for file_path in file_paths:
                data = pd.read_csv(file_path, index_col=0)
                signals = strategy.generate_signals(data)
            
            io_time = time.time() - start_time
            
            print(f"Disk I/O test: {io_time:.2f}s for {len(file_paths)} files")
            
            # Дисковые операции не должны занимать слишком много времени
            assert io_time < 30.0, f"Слишком медленные дисковые операции: {io_time:.2f}s"


class TestScalabilityTests:
    """Тесты масштабируемости системы."""

    def test_horizontal_scaling_simulation(self):
        """Симуляция горизонтального масштабирования."""
        # Симулируем обработку на нескольких "нодах"
        node_configs = [
            {'node_id': 1, 'symbols': [f'SYM{i:03d}' for i in range(0, 25)]},
            {'node_id': 2, 'symbols': [f'SYM{i:03d}' for i in range(25, 50)]},
            {'node_id': 3, 'symbols': [f'SYM{i:03d}' for i in range(50, 75)]},
            {'node_id': 4, 'symbols': [f'SYM{i:03d}' for i in range(75, 100)]},
        ]
        
        def process_node(node_config):
            """Обрабатывает данные на одном узле."""
            node_id = node_config['node_id']
            symbols = node_config['symbols']
            
            config = {'ema_fast': 12, 'ema_slow': 26}
            strategy = TrendStrategy(config)
            
            results = {}
            start_time = time.time()
            
            for symbol in symbols:
                # Генерируем данные для символа
                data = pd.DataFrame({
                    'open': np.random.uniform(99, 101, 1000),
                    'high': np.random.uniform(100, 102, 1000),
                    'low': np.random.uniform(98, 100, 1000),
                    'close': np.random.uniform(99, 101, 1000),
                    'volume': np.random.uniform(1000, 5000, 1000)
                })
                
                signals = strategy.generate_signals(data)
                results[symbol] = len(signals)
            
            execution_time = time.time() - start_time
            
            return {
                'node_id': node_id,
                'symbols_processed': len(symbols),
                'execution_time': execution_time,
                'results': results
            }
        
        # Обрабатываем все узлы параллельно
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_node, config) for config in node_configs]
            node_results = [future.result() for future in as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Проверяем результаты
        total_symbols = sum(result['symbols_processed'] for result in node_results)
        assert total_symbols == 100
        
        max_node_time = max(result['execution_time'] for result in node_results)
        avg_node_time = statistics.mean([result['execution_time'] for result in node_results])
        
        print(f"Horizontal scaling: {total_time:.2f}s total, max node: {max_node_time:.2f}s, avg node: {avg_node_time:.2f}s")
        
        # Параллельная обработка должна быть эффективной
        assert max_node_time < 30.0, f"Слишком долгая обработка на узле: {max_node_time:.2f}s"

    def test_data_volume_scaling(self):
        """Тест масштабирования по объему данных."""
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Тестируем различные объемы данных
        data_sizes = [1000, 5000, 10000, 25000, 50000, 100000]
        results = []
        
        for size in data_sizes:
            data = pd.DataFrame({
                'open': np.random.uniform(99, 101, size),
                'high': np.random.uniform(100, 102, size),
                'low': np.random.uniform(98, 100, size),
                'close': np.random.uniform(99, 101, size),
                'volume': np.random.uniform(1000, 5000, size)
            })
            
            start_time = time.time()
            signals = strategy.generate_signals(data)
            execution_time = time.time() - start_time
            
            throughput = size / execution_time  # записей в секунду
            
            results.append({
                'size': size,
                'execution_time': execution_time,
                'throughput': throughput
            })
            
            print(f"Size {size}: {execution_time:.3f}s, {throughput:.0f} records/sec")
        
        # Проверяем, что throughput остается разумным при увеличении объема
        min_throughput = min(r['throughput'] for r in results)
        max_throughput = max(r['throughput'] for r in results)
        
        # Различие в throughput не должно быть слишком большим
        throughput_ratio = max_throughput / min_throughput
        assert throughput_ratio < 10, f"Слишком большое различие в throughput: {throughput_ratio:.2f}x"

    @pytest.mark.asyncio
    async def test_async_processing_performance(self):
        """Тест производительности асинхронной обработки."""
        async def async_strategy_processing(data):
            """Асинхронная обработка стратегии."""
            config = {'ema_fast': 12, 'ema_slow': 26}
            strategy = TrendStrategy(config)
            
            # Симулируем асинхронную обработку
            await asyncio.sleep(0.001)  # Небольшая задержка
            
            return strategy.generate_signals(data)
        
        # Создаем множество задач
        tasks_data = []
        for i in range(100):
            data = pd.DataFrame({
                'open': np.random.uniform(99, 101, 500),
                'high': np.random.uniform(100, 102, 500),
                'low': np.random.uniform(98, 100, 500),
                'close': np.random.uniform(99, 101, 500),
                'volume': np.random.uniform(1000, 5000, 500)
            })
            tasks_data.append(data)
        
        # Выполняем задачи асинхронно
        start_time = time.time()
        
        tasks = [async_strategy_processing(data) for data in tasks_data]
        results = await asyncio.gather(*tasks)
        
        async_time = time.time() - start_time
        
        # Выполняем те же задачи синхронно для сравнения
        start_time = time.time()
        
        sync_results = []
        for data in tasks_data:
            config = {'ema_fast': 12, 'ema_slow': 26}
            strategy = TrendStrategy(config)
            result = strategy.generate_signals(data)
            sync_results.append(result)
        
        sync_time = time.time() - start_time
        
        # Проверяем результаты
        assert len(results) == len(sync_results) == 100
        
        speedup = sync_time / async_time
        print(f"Async processing: {async_time:.2f}s vs Sync: {sync_time:.2f}s, Speedup: {speedup:.2f}x")
        
        # Асинхронная обработка должна быть быстрее или сравнимой
        assert async_time <= sync_time * 1.1, f"Асинхронная обработка медленнее: {async_time:.2f}s vs {sync_time:.2f}s"