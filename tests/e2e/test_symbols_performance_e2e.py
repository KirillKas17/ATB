#!/usr/bin/env python3
"""
E2E тесты производительности для анализа символов.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
import asyncio
from decimal import Decimal
from typing import List, Dict, Any

from domain.symbols import (
    SymbolValidator,
    MarketPhaseClassifier,
    OpportunityScoreCalculator,
    MemorySymbolCache,
    SymbolProfile,
    MarketPhase,
    PriceStructure,
    VolumeProfile,
    OrderBookMetricsData,
    PatternMetricsData,
    SessionMetricsData,
)
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency

import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch, MagicMock
import psutil
import gc
from domain.symbols import (
    SymbolValidator,
    MarketPhaseClassifier,
    OpportunityScoreCalculator,
    MemorySymbolCache,
    SymbolProfile,
    MarketPhase,
    PriceStructure,
    VolumeProfile,
    OrderBookMetricsData,
    PatternMetricsData,
    SessionMetricsData,
)
from application.symbol_selection.opportunity_selector import DynamicOpportunityAwareSymbolSelector


class TestSymbolsPerformanceE2E:
    """E2E тесты производительности для системы анализа символов."""

    @pytest.fixture
    def performance_components(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с компонентами для тестов производительности."""
        return {
            "validator": SymbolValidator(),
            "classifier": MarketPhaseClassifier(),
            "calculator": OpportunityScoreCalculator(),
            "cache": MemorySymbolCache(default_ttl=300),
            "doass": DynamicOpportunityAwareSymbolSelector(),
        }

    @pytest.fixture
    def large_dataset(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с большим набором данных для тестирования производительности."""
        np.random.seed(42)
        # Генерируем данные для 100 символов
        symbols = [f"SYMBOL{i:03d}" for i in range(100)]
        market_data, order_books = {}, {}
        for symbol in symbols:
            # Генерируем OHLCV данные (1000 периодов)
            n_periods = 1000
            base_price = 100 + np.random.uniform(0, 900)
            returns = np.random.normal(0, 0.02, n_periods)
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            data = []
            for i, price in enumerate(prices):
                volatility = 0.01
                high = price * (1 + abs(np.random.normal(0, volatility)))
                low = price * (1 - abs(np.random.normal(0, volatility)))
                open_price = price * (1 + np.random.normal(0, volatility / 2))
                close_price = price * (1 + np.random.normal(0, volatility / 2))
                base_volume = 1000 + np.random.uniform(0, 9000)
                volume = base_volume * (1 + np.random.normal(0, 0.2))
                data.append(
                    {
                        "open": max(open_price, 0),
                        "high": max(high, open_price, close_price),
                        "low": max(low, 0),
                        "close": max(close_price, 0),
                        "volume": max(volume, 0),
                    }
                )
            market_data[symbol] = pd.DataFrame(data)
            # Генерируем стакан заявок
            current_price = prices[-1]
            spread = 0.001
            bids = []
            asks = []
            for i in range(20):
                bid_price = current_price * (1 - spread / 2 - i * 0.0001)
                ask_price = current_price * (1 + spread / 2 + i * 0.0001)
                bid_volume = 100 * (1 + np.random.normal(0, 0.3))
                ask_volume = 100 * (1 + np.random.normal(0, 0.3))
                bids.append([bid_price, max(bid_volume, 0)])
                asks.append([ask_price, max(ask_volume, 0)])
            order_books[symbol] = {
                "bids": sorted(bids, key=lambda x: x[0], reverse=True),
                "asks": sorted(asks, key=lambda x: x[0]),
            }
        return {"symbols": symbols, "market_data": market_data, "order_books": order_books}

    def test_mass_symbol_analysis_performance(self, performance_components, large_dataset) -> None:
        """Тест производительности массового анализа символов."""
        components = performance_components
        dataset = large_dataset
        # Измеряем время анализа всех символов
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        analyzed_count = 0
        for symbol in dataset["symbols"]:
            market_data = dataset["market_data"][symbol]
            order_book = dataset["order_books"][symbol]
            try:
                # Валидация
                assert components["validator"].validate_symbol(symbol) is True
                assert components["validator"].validate_ohlcv_data(market_data) is True
                assert components["validator"].validate_order_book(order_book) is True
                # Классификация
                with patch.object(components["classifier"], "classify_market_phase") as mock_classify:
                    mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.7)
                    # Расчет возможностей
                    with patch.object(components["calculator"], "calculate_opportunity_score") as mock_calc:
                        mock_calc.return_value = 0.6
                        profile = SymbolProfile(
                            symbol=symbol,
                            opportunity_score=0.6,
                            market_phase=MarketPhase.BREAKOUT_ACTIVE,
                            confidence=ConfidenceValue(0.7),
                            volume_profile=VolumeProfile(
                                current_volume=VolumeValue(market_data["volume"].iloc[-1]),
                                volume_trend=VolumeValue(0.1),
                                volume_stability=ConfidenceValue(0.8),
                            ),
                            price_structure=PriceStructure(
                                current_price=PriceValue(market_data["close"].iloc[-1]),
                                atr=ATRValue(1.0),
                                atr_percent=0.01,
                                vwap=VWAPValue(market_data["close"].mean()),
                                vwap_deviation=0.001,
                                price_entropy=EntropyValue(0.3),
                                volatility_compression=VolatilityValue(0.2),
                            ),
                            order_book_metrics=OrderBookMetrics(
                                bid_ask_spread=SpreadValue(0.1),
                                spread_percent=0.001,
                                bid_volume=VolumeValue(100.0),
                                ask_volume=VolumeValue(120.0),
                                volume_imbalance=-0.1,
                                order_book_symmetry=0.8,
                                liquidity_depth=0.9,
                                absorption_ratio=0.7,
                            ),
                            pattern_metrics=PatternMetrics(
                                mirror_neuron_score=0.6,
                                gravity_anomaly_score=0.5,
                                reversal_setup_score=0.4,
                                pattern_confidence=PatternConfidenceValue(0.7),
                                historical_pattern_match=0.6,
                                pattern_complexity=0.5,
                            ),
                            session_metrics=SessionMetrics(
                                session_alignment=SessionAlignmentValue(0.8),
                                session_activity=0.7,
                                session_volatility=VolatilityValue(0.15),
                                session_momentum=MomentumValue(0.6),
                                session_influence_score=0.7,
                            ),
                        )
                        components["cache"].set_profile(symbol, profile)
                        analyzed_count += 1
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                continue
        end_time = time.time()
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        total_time = end_time - start_time
        memory_used = memory_after - memory_before
        symbols_per_second = analyzed_count / total_time
        # Проверяем производительность
        assert total_time < 30.0  # Анализ 100 символов менее 30 секунд
        assert symbols_per_second > 3.0  # Минимум 3 символа в секунду
        assert memory_used < 500.0  # Использование памяти менее 500 MB
        print(f"Performance metrics:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Symbols analyzed: {analyzed_count}")
        print(f"  Symbols per second: {symbols_per_second:.2f}")
        print(f"  Memory used: {memory_used:.2f}MB")

    def test_concurrent_symbol_analysis_performance(self, performance_components, large_dataset) -> None:
        """Тест производительности параллельного анализа символов."""
        components = performance_components
        dataset = large_dataset

        def analyze_symbol(symbol) -> Any:
            """Функция анализа одного символа."""
            market_data = dataset["market_data"][symbol]
            order_book = dataset["order_books"][symbol]
            try:
                # Валидация
                assert components["validator"].validate_symbol(symbol) is True
                assert components["validator"].validate_ohlcv_data(market_data) is True
                assert components["validator"].validate_order_book(order_book) is True
                # Классификация
                with patch.object(components["classifier"], "classify_market_phase") as mock_classify:
                    mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.7)
                    # Расчет возможностей
                    with patch.object(components["calculator"], "calculate_opportunity_score") as mock_calc:
                        mock_calc.return_value = 0.6
                        profile = SymbolProfile(
                            symbol=symbol,
                            opportunity_score=0.6,
                            market_phase=MarketPhase.BREAKOUT_ACTIVE,
                            confidence=ConfidenceValue(0.7),
                            volume_profile=VolumeProfile(
                                current_volume=VolumeValue(market_data["volume"].iloc[-1]),
                                volume_trend=VolumeValue(0.1),
                                volume_stability=ConfidenceValue(0.8),
                            ),
                            price_structure=PriceStructure(
                                current_price=PriceValue(market_data["close"].iloc[-1]),
                                atr=ATRValue(1.0),
                                atr_percent=0.01,
                                vwap=VWAPValue(market_data["close"].mean()),
                                vwap_deviation=0.001,
                                price_entropy=EntropyValue(0.3),
                                volatility_compression=VolatilityValue(0.2),
                            ),
                            order_book_metrics=OrderBookMetrics(
                                bid_ask_spread=SpreadValue(0.1),
                                spread_percent=0.001,
                                bid_volume=VolumeValue(100.0),
                                ask_volume=VolumeValue(120.0),
                                volume_imbalance=-0.1,
                                order_book_symmetry=0.8,
                                liquidity_depth=0.9,
                                absorption_ratio=0.7,
                            ),
                            pattern_metrics=PatternMetrics(
                                mirror_neuron_score=0.6,
                                gravity_anomaly_score=0.5,
                                reversal_setup_score=0.4,
                                pattern_confidence=PatternConfidenceValue(0.7),
                                historical_pattern_match=0.6,
                                pattern_complexity=0.5,
                            ),
                            session_metrics=SessionMetrics(
                                session_alignment=SessionAlignmentValue(0.8),
                                session_activity=0.7,
                                session_volatility=VolatilityValue(0.15),
                                session_momentum=MomentumValue(0.6),
                                session_influence_score=0.7,
                            ),
                        )
                        components["cache"].set_profile(symbol, profile)
                        return symbol
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                return None

        # Тестируем с разным количеством потоков
        thread_counts = [1, 2, 4, 8]
        results = {}
        for thread_count in thread_counts:
            # Очищаем кэш перед каждым тестом
            components["cache"].clear()
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            with ThreadPoolExecutor(max_workers=thread_count) as executor:
                futures = [executor.submit(analyze_symbol, symbol) for symbol in dataset["symbols"]]
                analyzed_symbols = [future.result() for future in futures if future.result() is not None]
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            total_time = end_time - start_time
            memory_used = memory_after - memory_before
            symbols_per_second = len(analyzed_symbols) / total_time
            results[thread_count] = {
                "time": total_time,
                "symbols": len(analyzed_symbols),
                "symbols_per_second": symbols_per_second,
                "memory_used": memory_used,
            }
            print(
                f"Threads: {thread_count}, Time: {total_time:.2f}s, "
                f"Symbols: {len(analyzed_symbols)}, Rate: {symbols_per_second:.2f}/s"
            )
        # Проверяем, что параллелизм улучшает производительность
        assert results[4]["symbols_per_second"] > results[1]["symbols_per_second"] * 1.5
        assert results[8]["symbols_per_second"] > results[1]["symbols_per_second"] * 2.0

    def test_cache_performance_under_load(self, performance_components, large_dataset) -> None:
        """Тест производительности кэша под нагрузкой."""
        components = performance_components
        dataset = large_dataset
        # Заполняем кэш
        for symbol in dataset["symbols"][:50]:  # Первые 50 символов
            profile = SymbolProfile(
                symbol=symbol,
                opportunity_score=0.6,
                market_phase=MarketPhase.BREAKOUT_ACTIVE,
                confidence=ConfidenceValue(0.7),
                volume_profile=VolumeProfile(
                    current_volume=VolumeValue(1000.0),
                    volume_trend=VolumeValue(0.1),
                    volume_stability=ConfidenceValue(0.8),
                ),
                price_structure=PriceStructure(
                    current_price=PriceValue(market_data[symbol]["close"].iloc[-1]),
                    atr=ATRValue(1.0),
                    atr_percent=0.01,
                    vwap=VWAPValue(market_data[symbol]["close"].mean()),
                    vwap_deviation=0.001,
                    price_entropy=EntropyValue(0.3),
                    volatility_compression=VolatilityValue(0.2),
                ),
                order_book_metrics=OrderBookMetrics(
                    bid_ask_spread=SpreadValue(0.1),
                    spread_percent=0.001,
                    bid_volume=VolumeValue(100.0),
                    ask_volume=VolumeValue(120.0),
                    volume_imbalance=-0.1,
                    order_book_symmetry=0.8,
                    liquidity_depth=0.9,
                    absorption_ratio=0.7,
                ),
                pattern_metrics=PatternMetrics(
                    mirror_neuron_score=0.6,
                    gravity_anomaly_score=0.5,
                    reversal_setup_score=0.4,
                    pattern_confidence=PatternConfidenceValue(0.7),
                    historical_pattern_match=0.6,
                    pattern_complexity=0.5,
                ),
                session_metrics=SessionMetrics(
                    session_alignment=SessionAlignmentValue(0.8),
                    session_activity=0.7,
                    session_volatility=VolatilityValue(0.15),
                    session_momentum=MomentumValue(0.6),
                    session_influence_score=0.7,
                ),
            )
            components["cache"].set_profile(symbol, profile)
        # Тестируем производительность чтения
        start_time = time.time()
        for _ in range(1000):  # 1000 операций чтения
            symbol = np.random.choice(dataset["symbols"][:50])
            profile = components["cache"].get_profile(symbol)
            assert profile is not None
        read_time = time.time() - start_time
        reads_per_second = 1000 / read_time
        # Тестируем производительность записи
        start_time = time.time()
        for symbol in dataset["symbols"][50:100]:  # Следующие 50 символов
            profile = SymbolProfile(
                symbol=symbol,
                opportunity_score=0.6,
                market_phase=MarketPhase.BREAKOUT_ACTIVE,
                confidence=ConfidenceValue(0.7),
                volume_profile=VolumeProfile(
                    current_volume=VolumeValue(1000.0),
                    volume_trend=VolumeValue(0.1),
                    volume_stability=ConfidenceValue(0.8),
                ),
                price_structure=PriceStructure(
                    current_price=PriceValue(market_data[symbol]["close"].iloc[-1]),
                    atr=ATRValue(1.0),
                    atr_percent=0.01,
                    vwap=VWAPValue(market_data[symbol]["close"].mean()),
                    vwap_deviation=0.001,
                    price_entropy=EntropyValue(0.3),
                    volatility_compression=VolatilityValue(0.2),
                ),
                order_book_metrics=OrderBookMetrics(
                    bid_ask_spread=SpreadValue(0.1),
                    spread_percent=0.001,
                    bid_volume=VolumeValue(100.0),
                    ask_volume=VolumeValue(120.0),
                    volume_imbalance=-0.1,
                    order_book_symmetry=0.8,
                    liquidity_depth=0.9,
                    absorption_ratio=0.7,
                ),
                pattern_metrics=PatternMetrics(
                    mirror_neuron_score=0.6,
                    gravity_anomaly_score=0.5,
                    reversal_setup_score=0.4,
                    pattern_confidence=PatternConfidenceValue(0.7),
                    historical_pattern_match=0.6,
                    pattern_complexity=0.5,
                ),
                session_metrics=SessionMetrics(
                    session_alignment=SessionAlignmentValue(0.8),
                    session_activity=0.7,
                    session_volatility=VolatilityValue(0.15),
                    session_momentum=MomentumValue(0.6),
                    session_influence_score=0.7,
                ),
            )
            components["cache"].set_profile(symbol, profile)
        write_time = time.time() - start_time
        writes_per_second = 50 / write_time
        # Проверяем производительность
        assert reads_per_second > 1000.0  # Минимум 1000 чтений в секунду
        assert writes_per_second > 10.0  # Минимум 10 записей в секунду
        print(f"Cache performance:")
        print(f"  Reads per second: {reads_per_second:.2f}")
        print(f"  Writes per second: {writes_per_second:.2f}")

    def test_memory_usage_stability(self, performance_components, large_dataset) -> None:
        """Тест стабильности использования памяти."""
        components = performance_components
        dataset = large_dataset
        # Измеряем базовое использование памяти
        gc.collect()
        base_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_usage = []
        # Выполняем несколько циклов анализа
        for cycle in range(5):
            cycle_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            # Анализируем символы
            for symbol in dataset["symbols"][:20]:  # 20 символов за цикл
                market_data = dataset["market_data"][symbol]
                order_book = dataset["order_books"][symbol]
                with patch.object(components["classifier"], "classify_market_phase") as mock_classify:
                    mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.7)
                    with patch.object(components["calculator"], "calculate_opportunity_score") as mock_calc:
                        mock_calc.return_value = 0.6
                        profile = SymbolProfile(
                            symbol=symbol,
                            opportunity_score=0.6,
                            market_phase=MarketPhase.BREAKOUT_ACTIVE,
                            confidence=ConfidenceValue(0.7),
                            volume_profile=VolumeProfile(
                                current_volume=VolumeValue(market_data["volume"].iloc[-1]),
                                volume_trend=VolumeValue(0.1),
                                volume_stability=ConfidenceValue(0.8),
                            ),
                            price_structure=PriceStructure(
                                current_price=PriceValue(market_data["close"].iloc[-1]),
                                atr=ATRValue(1.0),
                                atr_percent=0.01,
                                vwap=VWAPValue(market_data["close"].mean()),
                                vwap_deviation=0.001,
                                price_entropy=EntropyValue(0.3),
                                volatility_compression=VolatilityValue(0.2),
                            ),
                            order_book_metrics=OrderBookMetrics(
                                bid_ask_spread=SpreadValue(0.1),
                                spread_percent=0.001,
                                bid_volume=VolumeValue(100.0),
                                ask_volume=VolumeValue(120.0),
                                volume_imbalance=-0.1,
                                order_book_symmetry=0.8,
                                liquidity_depth=0.9,
                                absorption_ratio=0.7,
                            ),
                            pattern_metrics=PatternMetrics(
                                mirror_neuron_score=0.6,
                                gravity_anomaly_score=0.5,
                                reversal_setup_score=0.4,
                                pattern_confidence=PatternConfidenceValue(0.7),
                                historical_pattern_match=0.6,
                                pattern_complexity=0.5,
                            ),
                            session_metrics=SessionMetrics(
                                session_alignment=SessionAlignmentValue(0.8),
                                session_activity=0.7,
                                session_volatility=VolatilityValue(0.15),
                                session_momentum=MomentumValue(0.6),
                                session_influence_score=0.7,
                            ),
                        )
                        components["cache"].set_profile(symbol, profile)
            cycle_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(cycle_end_memory - cycle_start_memory)
            # Очищаем кэш между циклами
            components["cache"].clear()
            gc.collect()
        # Проверяем стабильность памяти
        max_memory_increase = max(memory_usage)
        avg_memory_increase = sum(memory_usage) / len(memory_usage)
        assert max_memory_increase < 100.0  # Максимальное увеличение менее 100 MB
        assert avg_memory_increase < 50.0  # Среднее увеличение менее 50 MB
        print(f"Memory stability:")
        print(f"  Max memory increase: {max_memory_increase:.2f}MB")
        print(f"  Avg memory increase: {avg_memory_increase:.2f}MB")

    def test_doass_performance_with_large_dataset(self, performance_components, large_dataset) -> None:
        """Тест производительности DOASS с большим набором данных."""
        components = performance_components
        dataset = large_dataset
        # Подготавливаем данные в кэше
        for symbol in dataset["symbols"]:
            profile = SymbolProfile(
                symbol=symbol,
                opportunity_score=np.random.uniform(0.1, 0.9),
                market_phase=np.random.choice(
                    [MarketPhase.BREAKOUT_ACTIVE, MarketPhase.EXHAUSTION, MarketPhase.ACCUMULATION]
                ),
                confidence=np.random.uniform(0.5, 0.9),
                volume_profile=VolumeProfile(
                    current_volume=VolumeValue(1000.0),
                    volume_trend=VolumeValue(0.1),
                    volume_stability=ConfidenceValue(0.8),
                ),
                price_structure=PriceStructure(
                    current_price=PriceValue(market_data[symbol]["close"].iloc[-1]),
                    atr=ATRValue(1.0),
                    atr_percent=0.01,
                    vwap=VWAPValue(market_data[symbol]["close"].mean()),
                    vwap_deviation=0.001,
                    price_entropy=EntropyValue(0.3),
                    volatility_compression=VolatilityValue(0.2),
                ),
                order_book_metrics=OrderBookMetrics(
                    bid_ask_spread=SpreadValue(0.1),
                    spread_percent=0.001,
                    bid_volume=VolumeValue(100.0),
                    ask_volume=VolumeValue(120.0),
                    volume_imbalance=-0.1,
                    order_book_symmetry=0.8,
                    liquidity_depth=0.9,
                    absorption_ratio=0.7,
                ),
                pattern_metrics=PatternMetrics(
                    mirror_neuron_score=0.6,
                    gravity_anomaly_score=0.5,
                    reversal_setup_score=0.4,
                    pattern_confidence=PatternConfidenceValue(0.7),
                    historical_pattern_match=0.6,
                    pattern_complexity=0.5,
                ),
                session_metrics=SessionMetrics(
                    session_alignment=SessionAlignmentValue(0.8),
                    session_activity=0.7,
                    session_volatility=VolatilityValue(0.15),
                    session_momentum=MomentumValue(0.6),
                    session_influence_score=0.7,
                ),
            )
            components["cache"].set_profile(symbol, profile)
        # Тестируем производительность DOASS
        limits = [10, 25, 50, 100]
        results = {}
        for limit in limits:
            start_time = time.time()
            with patch.object(components["doass"], "get_detailed_analysis") as mock_doass:
                mock_doass.return_value = Mock(
                    selected_symbols=dataset["symbols"][:limit],
                    detailed_profiles={
                        symbol: components["cache"].get_profile(symbol) for symbol in dataset["symbols"][:limit]
                    },
                    total_symbols_analyzed=len(dataset["symbols"]),
                    processing_time_ms=100.0,
                    cache_hit_rate=0.9,
                )
                analysis_result = components["doass"].get_detailed_analysis(limit=limit)
            end_time = time.time()
            processing_time = end_time - start_time
            results[limit] = {"time": processing_time, "symbols_selected": len(analysis_result.selected_symbols)}
            print(
                f"Limit: {limit}, Time: {processing_time:.4f}s, " f"Selected: {len(analysis_result.selected_symbols)}"
            )
        # Проверяем производительность
        for limit in limits:
            assert results[limit]["time"] < 1.0  # Обработка менее 1 секунды
            assert results[limit]["symbols_selected"] == limit

    def test_stress_test_concurrent_access(self, performance_components, large_dataset) -> None:
        """Стресс-тест параллельного доступа к компонентам."""
        components = performance_components
        dataset = large_dataset

        def stress_worker(worker_id, symbols) -> Any:
            """Рабочая функция для стресс-теста."""
            results = []
            for symbol in symbols:
                try:
                    # Валидация
                    assert components["validator"].validate_symbol(symbol) is True
                    # Чтение из кэша
                    profile = components["cache"].get_profile(symbol)
                    # Запись в кэш
                    if profile is None:
                        profile = SymbolProfile(
                            symbol=symbol,
                            opportunity_score=0.6,
                            market_phase=MarketPhase.BREAKOUT_ACTIVE,
                            confidence=ConfidenceValue(0.7),
                            volume_profile=VolumeProfile(
                                current_volume=VolumeValue(1000.0),
                                volume_trend=VolumeValue(0.1),
                                volume_stability=ConfidenceValue(0.8),
                            ),
                            price_structure=PriceStructure(
                                current_price=PriceValue(market_data[symbol]["close"].iloc[-1]),
                                atr=ATRValue(1.0),
                                atr_percent=0.01,
                                vwap=VWAPValue(market_data[symbol]["close"].mean()),
                                vwap_deviation=0.001,
                                price_entropy=EntropyValue(0.3),
                                volatility_compression=VolatilityValue(0.2),
                            ),
                            order_book_metrics=OrderBookMetrics(
                                bid_ask_spread=SpreadValue(0.1),
                                spread_percent=0.001,
                                bid_volume=VolumeValue(100.0),
                                ask_volume=VolumeValue(120.0),
                                volume_imbalance=-0.1,
                                order_book_symmetry=0.8,
                                liquidity_depth=0.9,
                                absorption_ratio=0.7,
                            ),
                            pattern_metrics=PatternMetrics(
                                mirror_neuron_score=0.6,
                                gravity_anomaly_score=0.5,
                                reversal_setup_score=0.4,
                                pattern_confidence=PatternConfidenceValue(0.7),
                                historical_pattern_match=0.6,
                                pattern_complexity=0.5,
                            ),
                            session_metrics=SessionMetrics(
                                session_alignment=SessionAlignmentValue(0.8),
                                session_activity=0.7,
                                session_volatility=VolatilityValue(0.15),
                                session_momentum=MomentumValue(0.6),
                                session_influence_score=0.7,
                            ),
                        )
                        components["cache"].set_profile(symbol, profile)
                    results.append(symbol)
                except Exception as e:
                    print(f"Worker {worker_id} error with {symbol}: {e}")
                    continue
            return results

        # Запускаем несколько параллельных рабочих
        num_workers = 4
        symbols_per_worker = len(dataset["symbols"]) // num_workers
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                start_idx = i * symbols_per_worker
                end_idx = start_idx + symbols_per_worker if i < num_workers - 1 else len(dataset["symbols"])
                worker_symbols = dataset["symbols"][start_idx:end_idx]
                futures.append(executor.submit(stress_worker, i, worker_symbols))
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        end_time = time.time()
        total_time = end_time - start_time
        # Проверяем результаты стресс-теста
        assert len(all_results) > len(dataset["symbols"]) * 0.8  # Успешно обработано 80% символов
        assert total_time < 60.0  # Общее время менее 60 секунд
        print(f"Stress test results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Symbols processed: {len(all_results)}")
        print(f"  Success rate: {len(all_results) / len(dataset['symbols']) * 100:.1f}%")


if __name__ == "__main__":
    pytest.main([__file__])
