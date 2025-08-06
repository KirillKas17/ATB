#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance тесты для высокочастотной торговли.
"""

import pytest
import asyncio
import time
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch
import concurrent.futures
import statistics
import psutil
import os

from application.orchestration.trading_orchestrator import TradingOrchestrator
from infrastructure.external_services.bybit_client import BybitClient
from domain.entities.order import Order, OrderSide, OrderType
from domain.strategies.quantum_arbitrage_strategy import QuantumArbitrageStrategy


class TestHighFrequencyTradingPerformance:
    """Performance тесты для высокочастотной торговли."""

    @pytest.fixture
    async def hft_trading_orchestrator(self) -> TradingOrchestrator:
        """Фикстура оркестратора, оптимизированного для HFT."""
        config = {
            "risk_limit": Decimal("0.01"),
            "max_open_positions": 100,
            "execution_timeout": 1,  # 1 секунда для HFT
            "slippage_tolerance": Decimal("0.001"),
            "latency_optimization": True,
            "batch_processing": True,
            "memory_pooling": True
        }
        return TradingOrchestrator(**config)

    @pytest.fixture
    async def optimized_exchange_client(self) -> BybitClient:
        """Оптимизированный мок биржевого клиента."""
        client = AsyncMock(spec=BybitClient)
        
        # Симулируем очень быстрые ответы (< 1ms)
        async def fast_place_order(*args, **kwargs):
            await asyncio.sleep(0.0001)  # 0.1ms latency
            return {"order_id": f"hft_order_{time.time_ns()}", "status": "pending"}
        
        async def fast_get_ticker(*args, **kwargs):
            await asyncio.sleep(0.00005)  # 0.05ms latency
            return {
                "symbol": "BTCUSDT",
                "price": Decimal("45000.00") + Decimal(str(time.time() % 1)),
                "bid": Decimal("44999.50"),
                "ask": Decimal("45000.50")
            }
        
        client.place_limit_order = fast_place_order
        client.place_market_order = fast_place_order
        client.get_ticker = fast_get_ticker
        
        return client

    @pytest.mark.asyncio
    async def test_single_order_latency(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест латентности размещения одного ордера."""
        
        hft_trading_orchestrator.add_exchange_client("bybit", optimized_exchange_client)
        
        signal = {
            "symbol": "BTCUSDT",
            "side": "BUY",
            "quantity": Decimal("0.001"),
            "price": Decimal("45000.00"),
            "strategy_id": "latency_test"
        }
        
        # Измеряем латентность
        start_time = time.perf_counter()
        result = await hft_trading_orchestrator.execute_trade_signal(signal)
        end_time = time.perf_counter()
        
        latency_ms = (end_time - start_time) * 1000
        
        # Для HFT латентность должна быть < 5ms
        assert latency_ms < 5.0
        assert result["status"] == "success"
        
        print(f"Single order latency: {latency_ms:.2f}ms")

    @pytest.mark.asyncio
    async def test_order_throughput(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест пропускной способности ордеров."""
        
        hft_trading_orchestrator.add_exchange_client("bybit", optimized_exchange_client)
        
        # Создаем множество ордеров
        orders_count = 1000
        signals = []
        for i in range(orders_count):
            signals.append({
                "symbol": "BTCUSDT",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": Decimal("0.001"),
                "price": Decimal("45000.00") + Decimal(str(i % 100)),
                "strategy_id": f"throughput_test_{i}"
            })
        
        # Измеряем пропускную способность
        start_time = time.perf_counter()
        
        # Выполняем ордера параллельно
        tasks = [
            hft_trading_orchestrator.execute_trade_signal(signal)
            for signal in signals
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        successful_orders = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        orders_per_second = successful_orders / total_time
        
        # Для HFT должно быть > 500 ордеров в секунду
        assert orders_per_second > 500
        assert successful_orders > orders_count * 0.95  # 95% success rate
        
        print(f"Order throughput: {orders_per_second:.0f} orders/second")
        print(f"Success rate: {(successful_orders/orders_count)*100:.1f}%")

    @pytest.mark.asyncio
    async def test_concurrent_strategy_performance(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест производительности с множественными стратегиями."""
        
        hft_trading_orchestrator.add_exchange_client("bybit", optimized_exchange_client)
        
        # Создаем несколько стратегий
        strategies = []
        for i in range(10):
            strategy = QuantumArbitrageStrategy(
                min_arbitrage_threshold=Decimal("0.001"),
                exchanges=["bybit"],
                symbols=[f"SYMBOL{i}USDT"],
                quantum_states=8
            )
            strategies.append(strategy)
            hft_trading_orchestrator.add_strategy(strategy)
        
        # Генерируем сигналы от всех стратегий
        all_signals = []
        for i, strategy in enumerate(strategies):
            for j in range(50):  # 50 сигналов на стратегию
                all_signals.append({
                    "symbol": f"SYMBOL{i}USDT",
                    "side": "BUY" if j % 2 == 0 else "SELL",
                    "quantity": Decimal("0.001"),
                    "strategy_id": strategy.strategy_id,
                    "confidence": 0.8
                })
        
        # Измеряем производительность
        start_time = time.perf_counter()
        results = await hft_trading_orchestrator.process_multiple_signals_concurrent(all_signals)
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        signals_per_second = len(all_signals) / total_time
        
        # Проверяем производительность
        assert signals_per_second > 200  # > 200 сигналов в секунду
        assert len(results) == len(all_signals)
        
        print(f"Multi-strategy throughput: {signals_per_second:.0f} signals/second")

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест использования памяти под нагрузкой."""
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        hft_trading_orchestrator.add_exchange_client("bybit", optimized_exchange_client)
        
        # Создаем интенсивную нагрузку
        for round_num in range(10):
            signals = []
            for i in range(100):
                signals.append({
                    "symbol": "BTCUSDT",
                    "side": "BUY" if i % 2 == 0 else "SELL",
                    "quantity": Decimal("0.001"),
                    "price": Decimal("45000.00"),
                    "strategy_id": f"memory_test_{round_num}_{i}"
                })
            
            # Выполняем batch
            await hft_trading_orchestrator.process_signals_batch(signals)
            
            # Принудительная сборка мусора
            import gc
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Увеличение памяти должно быть разумным (< 100MB)
        assert memory_increase < 100
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{memory_increase:.1f}MB)")

    @pytest.mark.asyncio
    async def test_latency_distribution(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест распределения латентности."""
        
        hft_trading_orchestrator.add_exchange_client("bybit", optimized_exchange_client)
        
        latencies = []
        
        # Измеряем латентность для множества ордеров
        for i in range(100):
            signal = {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": Decimal("0.001"),
                "price": Decimal("45000.00"),
                "strategy_id": f"latency_dist_{i}"
            }
            
            start_time = time.perf_counter()
            await hft_trading_orchestrator.execute_trade_signal(signal)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Анализируем статистику латентности
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]
        
        # Требования для HFT
        assert mean_latency < 2.0    # Средняя латентность < 2ms
        assert median_latency < 1.5  # Медианная латентность < 1.5ms
        assert p95_latency < 5.0     # 95-й перцентиль < 5ms
        assert p99_latency < 10.0    # 99-й перцентиль < 10ms
        
        print(f"Latency stats (ms): mean={mean_latency:.2f}, median={median_latency:.2f}, "
              f"p95={p95_latency:.2f}, p99={p99_latency:.2f}")

    @pytest.mark.asyncio
    async def test_order_book_processing_speed(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест скорости обработки orderbook данных."""
        
        # Создаем большой orderbook
        orderbook_data = {
            "symbol": "BTCUSDT",
            "bids": [[Decimal(f"44{900 + i}.{j:02d}"), Decimal(f"{i}.{j}")] 
                    for i in range(100) for j in range(10)],
            "asks": [[Decimal(f"45{000 + i}.{j:02d}"), Decimal(f"{i}.{j}")] 
                    for i in range(100) for j in range(10)],
            "timestamp": int(time.time() * 1000)
        }
        
        # Измеряем скорость обработки
        processing_times = []
        
        for _ in range(50):
            start_time = time.perf_counter()
            analysis = await hft_trading_orchestrator.analyze_orderbook(orderbook_data)
            end_time = time.perf_counter()
            
            processing_time_ms = (end_time - start_time) * 1000
            processing_times.append(processing_time_ms)
        
        avg_processing_time = statistics.mean(processing_times)
        
        # Обработка orderbook должна быть < 1ms
        assert avg_processing_time < 1.0
        assert analysis is not None
        
        print(f"Orderbook processing time: {avg_processing_time:.3f}ms")

    @pytest.mark.asyncio
    async def test_arbitrage_detection_speed(
        self,
        hft_trading_orchestrator: TradingOrchestrator
    ) -> None:
        """Тест скорости обнаружения арбитража."""
        
        # Создаем данные с нескольких бирж
        exchange_data = {
            "binance": {
                "symbol": "BTCUSDT",
                "bid": Decimal("44999.50"),
                "ask": Decimal("45000.50"),
                "timestamp": int(time.time() * 1000)
            },
            "bybit": {
                "symbol": "BTCUSDT",
                "bid": Decimal("45024.50"),
                "ask": Decimal("45025.50"),
                "timestamp": int(time.time() * 1000)
            },
            "okx": {
                "symbol": "BTCUSDT",
                "bid": Decimal("45009.50"),
                "ask": Decimal("45010.50"),
                "timestamp": int(time.time() * 1000)
            }
        }
        
        detection_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            arbitrage_opportunities = await hft_trading_orchestrator.detect_arbitrage_opportunities(
                exchange_data
            )
            end_time = time.perf_counter()
            
            detection_time_ms = (end_time - start_time) * 1000
            detection_times.append(detection_time_ms)
        
        avg_detection_time = statistics.mean(detection_times)
        
        # Обнаружение арбитража должно быть < 0.5ms
        assert avg_detection_time < 0.5
        assert len(arbitrage_opportunities) > 0
        
        print(f"Arbitrage detection time: {avg_detection_time:.3f}ms")

    @pytest.mark.asyncio
    async def test_stress_concurrent_connections(
        self,
        hft_trading_orchestrator: TradingOrchestrator
    ) -> None:
        """Стресс-тест множественных соединений."""
        
        # Создаем множество клиентов бирж
        clients = {}
        for i in range(10):
            client = AsyncMock(spec=BybitClient)
            client.place_limit_order.return_value = {
                "order_id": f"stress_order_{i}_{time.time_ns()}",
                "status": "pending"
            }
            clients[f"exchange_{i}"] = client
            hft_trading_orchestrator.add_exchange_client(f"exchange_{i}", client)
        
        # Генерируем сигналы для всех бирж
        stress_signals = []
        for i in range(500):  # 500 сигналов
            exchange_name = f"exchange_{i % 10}"
            stress_signals.append({
                "symbol": "BTCUSDT",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": Decimal("0.001"),
                "price": Decimal("45000.00"),
                "exchange": exchange_name,
                "strategy_id": f"stress_test_{i}"
            })
        
        # Выполняем стресс-тест
        start_time = time.perf_counter()
        
        # Максимальная параллельность
        semaphore = asyncio.Semaphore(100)
        
        async def execute_with_semaphore(signal):
            async with semaphore:
                return await hft_trading_orchestrator.execute_trade_signal(signal)
        
        results = await asyncio.gather(
            *[execute_with_semaphore(signal) for signal in stress_signals],
            return_exceptions=True
        )
        
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        successful_results = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
        
        # Проверяем результаты стресс-теста
        success_rate = successful_results / len(stress_signals)
        throughput = len(stress_signals) / total_time
        
        assert success_rate > 0.95  # 95% успешность
        assert throughput > 100     # > 100 операций в секунду
        
        print(f"Stress test: {throughput:.0f} ops/sec, {success_rate*100:.1f}% success")

    @pytest.mark.asyncio
    async def test_real_time_price_feed_processing(
        self,
        hft_trading_orchestrator: TradingOrchestrator
    ) -> None:
        """Тест обработки real-time ценовых потоков."""
        
        # Симулируем высокочастотный поток цен
        async def price_feed_generator():
            base_price = Decimal("45000.00")
            for i in range(1000):
                yield {
                    "symbol": "BTCUSDT",
                    "price": base_price + Decimal(str((i % 100) - 50)) * Decimal("0.01"),
                    "timestamp": int(time.time() * 1000) + i,
                    "volume": Decimal(f"{100 + (i % 50)}.0")
                }
                await asyncio.sleep(0.001)  # 1000 updates per second
        
        processed_count = 0
        signals_generated = 0
        
        start_time = time.perf_counter()
        
        async for price_update in price_feed_generator():
            # Обрабатываем обновление цены
            result = await hft_trading_orchestrator.process_price_update(price_update)
            processed_count += 1
            
            if result.get("signal_generated"):
                signals_generated += 1
            
            # Останавливаемся после обработки определенного количества
            if processed_count >= 500:
                break
        
        end_time = time.perf_counter()
        
        processing_rate = processed_count / (end_time - start_time)
        
        # Должны обрабатывать > 800 обновлений в секунду
        assert processing_rate > 800
        assert signals_generated > 0
        
        print(f"Price feed processing: {processing_rate:.0f} updates/sec, "
              f"{signals_generated} signals generated")

    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(
        self,
        hft_trading_orchestrator: TradingOrchestrator,
        optimized_exchange_client: BybitClient
    ) -> None:
        """Тест использования CPU под нагрузкой."""
        
        hft_trading_orchestrator.add_exchange_client("bybit", optimized_exchange_client)
        
        # Начальное измерение CPU
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # Создаем интенсивную нагрузку
        intensive_signals = []
        for i in range(2000):
            intensive_signals.append({
                "symbol": "BTCUSDT",
                "side": "BUY" if i % 2 == 0 else "SELL",
                "quantity": Decimal("0.001"),
                "price": Decimal("45000.00"),
                "strategy_id": f"cpu_test_{i}"
            })
        
        # Выполняем под нагрузкой
        start_time = time.time()
        
        tasks = [
            hft_trading_orchestrator.execute_trade_signal(signal)
            for signal in intensive_signals
        ]
        
        await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Измеряем CPU после нагрузки
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        execution_time = end_time - start_time
        orders_per_second = len(intensive_signals) / execution_time
        
        # CPU не должен превышать 80% при интенсивной нагрузке
        assert cpu_percent_after < 80
        assert orders_per_second > 300  # Минимальная производительность
        
        print(f"CPU usage: {cpu_percent_before:.1f}% -> {cpu_percent_after:.1f}%")
        print(f"Performance under load: {orders_per_second:.0f} orders/sec")

    @pytest.mark.asyncio
    async def test_network_latency_simulation(
        self,
        hft_trading_orchestrator: TradingOrchestrator
    ) -> None:
        """Тест с симуляцией сетевой латентности."""
        
        # Создаем клиент с различной латентностью
        variable_latency_client = AsyncMock(spec=BybitClient)
        
        async def variable_latency_order(*args, **kwargs):
            # Симулируем переменную латентность (1-10ms)
            import random
            latency = random.uniform(0.001, 0.01)
            await asyncio.sleep(latency)
            return {"order_id": f"var_latency_{time.time_ns()}", "status": "pending"}
        
        variable_latency_client.place_limit_order = variable_latency_order
        hft_trading_orchestrator.add_exchange_client("variable_exchange", variable_latency_client)
        
        # Тестируем адаптацию к переменной латентности
        signals = []
        for i in range(100):
            signals.append({
                "symbol": "BTCUSDT",
                "side": "BUY",
                "quantity": Decimal("0.001"),
                "price": Decimal("45000.00"),
                "strategy_id": f"latency_adapt_{i}"
            })
        
        start_time = time.perf_counter()
        results = await asyncio.gather(
            *[hft_trading_orchestrator.execute_trade_signal(signal) for signal in signals]
        )
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        successful_orders = sum(1 for r in results if r.get("status") == "success")
        
        # Система должна адаптироваться к переменной латентности
        success_rate = successful_orders / len(signals)
        assert success_rate > 0.90  # 90% успешность даже при переменной латентности
        
        print(f"Variable latency performance: {success_rate*100:.1f}% success rate")

    def test_performance_benchmarks_summary(self) -> None:
        """Сводка performance бенчмарков."""
        
        benchmarks = {
            "Single Order Latency": "< 5ms",
            "Order Throughput": "> 500 orders/sec",
            "Multi-Strategy Throughput": "> 200 signals/sec", 
            "Memory Usage Increase": "< 100MB under load",
            "Mean Latency": "< 2ms",
            "P95 Latency": "< 5ms",
            "P99 Latency": "< 10ms",
            "Orderbook Processing": "< 1ms",
            "Arbitrage Detection": "< 0.5ms",
            "Stress Test Throughput": "> 100 ops/sec",
            "Price Feed Processing": "> 800 updates/sec",
            "CPU Usage Under Load": "< 80%",
            "Success Rate Variable Latency": "> 90%"
        }
        
        print("\n=== HFT PERFORMANCE BENCHMARKS ===")
        for benchmark, target in benchmarks.items():
            print(f"{benchmark}: {target}")
        
        # Все бенчмарки должны проходить для production-ready системы
        assert len(benchmarks) == 13  # Проверяем, что все бенчмарки учтены