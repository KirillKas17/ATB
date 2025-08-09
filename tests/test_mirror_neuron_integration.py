# -*- coding: utf-8 -*-
"""Интеграционные тесты для системы Mirror Neuron Signal Detection."""
import time
import logging
from shared.numpy_utils import np
import pandas as pd
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.strategy_advisor.mirror_map_builder import MirrorMap, MirrorMapBuilder, MirrorMapConfig
from domain.intelligence.mirror_detector import MirrorDetector, MirrorSignal

logger = logging.getLogger(__name__)


class TestMirrorNeuronIntegration:
    """Интеграционные тесты для системы Mirror Neuron Signal."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = MirrorMapConfig(
            min_correlation=0.3,
            max_p_value=0.05,
            min_confidence=0.7,
            max_lag=5,
            correlation_method="pearson",
            normalize_data=True,
            remove_trend=True,
            min_cluster_size=2,
            max_cluster_size=10,
            update_interval=3600,
            parallel_processing=True,
            max_workers=4,
        )
        self.builder = MirrorMapBuilder(self.config)
        self.detector = MirrorDetector(min_correlation=0.3, max_p_value=0.05, min_confidence=0.7, max_lag=5)

    def create_correlated_data(self, n_assets: int = 5, periods: int = 500) -> tuple:
        """Создание коррелированных данных для тестирования."""
        np.random.seed(42)
        assets = [f"ASSET_{i}" for i in range(n_assets)]
        price_data: dict[str, pd.Series] = {}
        # Создаем базовый тренд
        base_trend = np.linspace(100, 110, periods)
        for i, asset in enumerate(assets):
            # Добавляем случайный шум
            noise = np.random.normal(0, 2, periods)
            # Добавляем корреляции между активами
            if i > 0:
                # Каждый актив коррелирует с предыдущим с лагом
                lag = i % 3 + 1  # Лаг от 1 до 3
                if lag < periods:
                    prev_asset = assets[i - 1]
                    if prev_asset in price_data:
                        # Исправление: добавляем проверку типа перед индексированием
                        prev_series = price_data[prev_asset]
                        if hasattr(prev_series, "iloc") and len(prev_series) > lag:
                            # Исправление: правильное обращение к Series
                            if callable(prev_series.iloc):
                                series_data = prev_series.iloc()
                            else:
                                series_data = prev_series.iloc
                            if len(series_data) > lag:
                                noise[lag:] += series_data[:-lag] * 0.3
            prices = base_trend + noise
            price_data[asset] = pd.Series(prices, index=pd.date_range("2024-01-01", periods=periods, freq="H"))
        return assets, price_data

    def test_end_to_end_mirror_detection(self: "TestMirrorNeuronIntegration") -> None:
        """Тест полного цикла обнаружения зеркальных сигналов."""
        logger.info("=== Testing End-to-End Mirror Detection ===")
        # Создаем данные
        assets, price_data = self.create_correlated_data(n_assets=4, periods=300)
        # Обнаруживаем зеркальные сигналы
        signals = []
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    signal = self.detector.detect_mirror_signal(
                        asset1,
                        asset2,
                        price_data[asset1],
                        price_data[asset2],
                        max_lag=5,
                    )
                    if signal:
                        signals.append(signal)
        logger.info(f"Detected {len(signals)} mirror signals")
        # Проверяем результаты
        assert len(signals) > 0
        for signal in signals:
            assert isinstance(signal, MirrorSignal)
            assert signal.asset1 in assets
            assert signal.asset2 in assets
            assert signal.asset1 != signal.asset2
            assert abs(signal.correlation) >= self.config.min_correlation
            assert signal.p_value <= self.config.max_p_value
            assert signal.confidence >= self.config.min_confidence

    def test_end_to_end_mirror_map_building(self: "TestMirrorNeuronIntegration") -> None:
        """Тест полного цикла построения карты зеркальных зависимостей."""
        logger.info("=== Testing End-to-End Mirror Map Building ===")
        # Создаем данные
        assets, price_data = self.create_correlated_data(n_assets=6, periods=400)
        # Строим карту
        start_time = time.time()
        mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        build_time = time.time() - start_time
        logger.info(f"Mirror map built in {build_time:.2f} seconds")
        # Проверяем результаты
        assert isinstance(mirror_map, MirrorMap)
        assert len(mirror_map.assets) > 0
        assert mirror_map.correlation_matrix is not None
        assert mirror_map.correlation_matrix.correlation_matrix.shape[0] == len(mirror_map.assets)
        # Проверяем, что есть зеркальные зависимости
        total_dependencies = sum(len(mirrors) for mirrors in mirror_map.mirror_map.values())
        logger.info(f"Total mirror dependencies: {total_dependencies}")
        # Проверяем кластеры
        assert len(mirror_map.clusters) >= 0
        for cluster in mirror_map.clusters:
            assert len(cluster) >= self.config.min_cluster_size
            assert len(cluster) <= self.config.max_cluster_size

    def test_mirror_map_caching(self: "TestMirrorNeuronIntegration") -> None:
        """Тест кэширования карты зеркальных зависимостей."""
        logger.info("=== Testing Mirror Map Caching ===")
        assets, price_data = self.create_correlated_data(n_assets=3, periods=200)
        # Первое построение
        start_time = time.time()
        mirror_map1 = self.builder.build_mirror_map(assets, price_data, force_rebuild=False)
        first_build_time = time.time() - start_time
        # Второе построение (должно использовать кэш)
        start_time = time.time()
        mirror_map2 = self.builder.build_mirror_map(assets, price_data, force_rebuild=False)
        second_build_time = time.time() - start_time
        logger.info(f"First build: {first_build_time:.3f}s, Second build: {second_build_time:.3f}s")
        # Проверяем, что кэш работает
        assert mirror_map1 is mirror_map2
        assert second_build_time < first_build_time  # Кэш должен быть быстрее

    def test_parallel_vs_sequential_processing(self: "TestMirrorNeuronIntegration") -> None:
        """Тест сравнения параллельной и последовательной обработки."""
        logger.info("=== Testing Parallel vs Sequential Processing ===")
        assets, price_data = self.create_correlated_data(n_assets=8, periods=300)
        # Параллельная обработка
        parallel_config = MirrorMapConfig(
            min_correlation=0.3,
            max_p_value=0.05,
            max_lag=5,
            parallel_processing=True,
            max_workers=4,
        )
        parallel_builder = MirrorMapBuilder(parallel_config)
        start_time = time.time()
        parallel_map = parallel_builder.build_mirror_map(assets, price_data, force_rebuild=True)
        parallel_time = time.time() - start_time
        # Последовательная обработка
        sequential_config = MirrorMapConfig(min_correlation=0.3, max_p_value=0.05, max_lag=5, parallel_processing=False)
        sequential_builder = MirrorMapBuilder(sequential_config)
        start_time = time.time()
        sequential_map = sequential_builder.build_mirror_map(assets, price_data, force_rebuild=True)
        sequential_time = time.time() - start_time
        logger.info(f"Parallel: {parallel_time:.3f}s, Sequential: {sequential_time:.3f}s")
        logger.info(f"Speedup: {sequential_time / parallel_time:.2f}x")
        # Проверяем, что результаты одинаковые
        assert len(parallel_map.assets) == len(sequential_map.assets)
        assert len(parallel_map.mirror_map) == len(sequential_map.mirror_map)

    @pytest.mark.asyncio
    async def test_async_mirror_map_building(self: "TestMirrorNeuronIntegration") -> None:
        """Тест асинхронного построения карты."""
        logger.info("=== Testing Async Mirror Map Building ===")
        assets, price_data = self.create_correlated_data(n_assets=5, periods=250)
        # Асинхронное построение
        start_time = time.time()
        mirror_map = await self.builder.build_mirror_map_async(assets, price_data, force_rebuild=True)
        async_time = time.time() - start_time
        logger.info(f"Async build time: {async_time:.3f}s")
        # Проверяем результат
        assert isinstance(mirror_map, MirrorMap)
        assert len(mirror_map.assets) == 5
        assert mirror_map.correlation_matrix is not None

    def test_mirror_map_with_different_correlation_methods(self: "TestMirrorNeuronIntegration") -> None:
        """Тест карты с различными методами корреляции."""
        logger.info("=== Testing Different Correlation Methods ===")
        assets, price_data = self.create_correlated_data(n_assets=4, periods=200)
        methods = ["pearson", "spearman", "kendall"]
        results = {}
        for method in methods:
            config = MirrorMapConfig(
                min_correlation=0.3,
                max_p_value=0.05,
                max_lag=5,
                correlation_method=method,
            )
            builder = MirrorMapBuilder(config)
            mirror_map = builder.build_mirror_map(assets, price_data, force_rebuild=True)
            total_dependencies = sum(len(mirrors) for mirrors in mirror_map.mirror_map.values())
            results[method] = {
                "dependencies": total_dependencies,
                "clusters": len(mirror_map.clusters),
            }
        logger.info(f"Results by correlation method: {results}")
        # Проверяем, что все методы работают
        for method, result in results.items():
            assert result["dependencies"] >= 0
            assert result["clusters"] >= 0

    def test_mirror_map_with_different_lag_settings(self: "TestMirrorNeuronIntegration") -> None:
        """Тест карты с различными настройками лага."""
        logger.info("=== Testing Different Lag Settings ===")
        assets, price_data = self.create_correlated_data(n_assets=4, periods=300)
        lag_settings = [1, 3, 5, 10]
        results = {}
        for max_lag in lag_settings:
            config = MirrorMapConfig(min_correlation=0.3, max_p_value=0.05, max_lag=max_lag)
            builder = MirrorMapBuilder(config)
            mirror_map = builder.build_mirror_map(assets, price_data, force_rebuild=True)
            total_dependencies = sum(len(mirrors) for mirrors in mirror_map.mirror_map.values())
            max_lag_found = 0
            if mirror_map.correlation_matrix:
                max_lag_found = np.max(np.abs(mirror_map.correlation_matrix.lag_matrix))
            results[max_lag] = {
                "dependencies": total_dependencies,
                "max_lag_found": max_lag_found,
            }
        logger.info(f"Results by lag setting: {results}")
        # Проверяем, что больший лаг может найти больше зависимостей
        assert results[10]["dependencies"] >= results[1]["dependencies"]

    def test_mirror_map_cluster_analysis(self: "TestMirrorNeuronIntegration") -> None:
        """Тест анализа кластеров в карте."""
        logger.info("=== Testing Cluster Analysis ===")
        assets, price_data = self.create_correlated_data(n_assets=8, periods=400)
        mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        # Анализ кластеров
        analysis = self.builder.analyze_mirror_clusters(mirror_map)
        logger.info(f"Cluster analysis: {analysis}")
        # Проверяем результаты анализа
        assert analysis["total_clusters"] == len(mirror_map.clusters)
        assert analysis["largest_cluster"] >= 0
        assert analysis["average_cluster_size"] >= 0
        if mirror_map.clusters:
            assert analysis["largest_cluster"] <= self.config.max_cluster_size
            assert analysis["average_cluster_size"] <= self.config.max_cluster_size
        # Проверяем детали кластеров
        assert len(analysis["cluster_details"]) == len(mirror_map.clusters)
        for cluster_detail in analysis["cluster_details"]:
            assert "cluster_id" in cluster_detail
            assert "assets" in cluster_detail
            assert "size" in cluster_detail
            assert "average_correlation" in cluster_detail
            assert "average_lag" in cluster_detail

    def test_mirror_map_strategy_integration(self: "TestMirrorNeuronIntegration") -> None:
        """Тест интеграции карты с торговыми стратегиями."""
        logger.info("=== Testing Strategy Integration ===")
        assets, price_data = self.create_correlated_data(n_assets=6, periods=300)
        mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        # Симуляция торговой стратегии
        strategy_results = {}
        for asset in assets[:3]:  # Тестируем первые 3 актива
            mirror_assets = self.builder.get_mirror_assets_for_strategy(mirror_map, asset, min_correlation=0.3)
            strategy_results[asset] = {
                "mirror_assets": len(mirror_assets),
                "total_correlation": sum(abs(corr) for _, corr, _ in mirror_assets),
                "avg_lag": (np.mean([lag for _, _, lag in mirror_assets]) if mirror_assets else 0),
            }
        logger.info(f"Strategy integration results: {strategy_results}")
        # Проверяем результаты
        for asset, result in strategy_results.items():
            assert result["mirror_assets"] >= 0
            assert result["total_correlation"] >= 0
            assert result["avg_lag"] >= 0

    def test_mirror_map_performance_scaling(self: "TestMirrorNeuronIntegration") -> None:
        """Тест масштабирования производительности."""
        logger.info("=== Testing Performance Scaling ===")
        asset_counts = [3, 5, 8, 10]
        performance_results = {}
        for n_assets in asset_counts:
            assets, price_data = self.create_correlated_data(n_assets=n_assets, periods=200)
            start_time = time.time()
            mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
            build_time = time.time() - start_time
            total_dependencies = sum(len(mirrors) for mirrors in mirror_map.mirror_map.values())
            performance_results[n_assets] = {
                "build_time": build_time,
                "dependencies": total_dependencies,
                "clusters": len(mirror_map.clusters),
            }
        logger.info(f"Performance scaling results: {performance_results}")
        # Проверяем, что время построения растет с количеством активов
        times = [result["build_time"] for result in performance_results.values()]
        assert times[-1] >= times[0]  # Больше активов = больше времени

    def test_mirror_map_error_handling(self: "TestMirrorNeuronIntegration") -> None:
        """Тест обработки ошибок в карте."""
        logger.info("=== Testing Error Handling ===")
        # Тест с некорректными данными
        assets = ["BTC", "ETH", "ADA"]
        invalid_price_data = {
            "BTC": pd.Series([1, 2, 3]),  # Недостаточно данных
            "ETH": pd.Series([np.nan, np.nan, np.nan]),  # Все NaN
            "ADA": pd.Series([1, 2, 3, 4, 5]),  # Недостаточно данных
        }
        # Должно обработать ошибки gracefully
        mirror_map = self.builder.build_mirror_map(assets, invalid_price_data, force_rebuild=True)
        assert isinstance(mirror_map, MirrorMap)
        assert len(mirror_map.assets) == 0  # Нет валидных активов
        assert len(mirror_map.mirror_map) == 0

    def test_mirror_map_configuration_updates(self: "TestMirrorNeuronIntegration") -> None:
        """Тест обновления конфигурации карты."""
        logger.info("=== Testing Configuration Updates ===")
        assets, price_data = self.create_correlated_data(n_assets=4, periods=200)
        # Первая конфигурация
        config1 = MirrorMapConfig(min_correlation=0.5, max_p_value=0.01, max_lag=3)  # Высокий порог
        builder1 = MirrorMapBuilder(config1)
        mirror_map1 = builder1.build_mirror_map(assets, price_data, force_rebuild=True)
        # Вторая конфигурация
        config2 = MirrorMapConfig(min_correlation=0.2, max_p_value=0.1, max_lag=5)  # Низкий порог
        builder2 = MirrorMapBuilder(config2)
        mirror_map2 = builder2.build_mirror_map(assets, price_data, force_rebuild=True)
        # Считаем зависимости
        deps1 = sum(len(mirrors) for mirrors in mirror_map1.mirror_map.values())
        deps2 = sum(len(mirrors) for mirrors in mirror_map2.mirror_map.values())
        logger.info(f"Dependencies with high threshold: {deps1}")
        logger.info(f"Dependencies with low threshold: {deps2}")
        # Низкий порог должен найти больше зависимостей
        assert deps2 >= deps1

    def test_mirror_map_statistics(self: "TestMirrorNeuronIntegration") -> None:
        """Тест статистики карты."""
        logger.info("=== Testing Mirror Map Statistics ===")
        assets, price_data = self.create_correlated_data(n_assets=5, periods=250)
        mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        # Получаем статистику
        stats = self.builder.get_mirror_map_statistics()
        logger.info(f"Mirror map statistics: {stats}")
        # Проверяем структуру статистики
        assert "config" in stats
        assert "detector_stats" in stats
        assert "cache_info" in stats
        assert "mirror_map_info" in stats
        # Проверяем информацию о карте
        map_info = stats["mirror_map_info"]
        assert map_info["total_assets"] == len(mirror_map.assets)
        assert map_info["assets_with_dependencies"] == len(mirror_map.mirror_map)
        assert map_info["total_dependencies"] >= 0
        assert map_info["total_clusters"] == len(mirror_map.clusters)

    def test_mirror_map_concurrent_access(self: "TestMirrorNeuronIntegration") -> None:
        """Тест конкурентного доступа к карте."""
        logger.info("=== Testing Concurrent Access ===")
        import queue
        import threading

        assets, price_data = self.create_correlated_data(n_assets=4, periods=200)
        # Создаем карту
        mirror_map = self.builder.build_mirror_map(assets, price_data, force_rebuild=True)
        # Создаем очередь для результатов
        results = queue.Queue()

        def access_mirror_map(thread_id) -> Any:
            """Функция для доступа к карте в отдельном потоке."""
            try:
                # Получаем зеркальные активы для каждого актива
                for asset in assets:
                    mirror_assets = mirror_map.get_mirror_assets(asset)
                    correlation = mirror_map.get_correlation(asset, assets[0]) if assets else 0.0
                    lag = mirror_map.get_lag(asset, assets[0]) if assets else 0
                    results.put((thread_id, asset, len(mirror_assets), correlation, lag))
            except Exception as e:
                results.put((thread_id, f"Error: {e}", 0, 0.0, 0))

        # Запускаем несколько потоков
        threads = []
        for i in range(5):
            thread = threading.Thread(target=access_mirror_map, args=(i,))
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Собираем результаты
        thread_results = []
        while not results.empty():
            thread_results.append(results.get())
        assert len(thread_results) == 5 * len(assets)  # 5 потоков * количество активов
        # Проверяем, что все результаты корректны
        for thread_id, asset, mirror_count, correlation, lag in thread_results:
            assert isinstance(thread_id, int)
            assert isinstance(asset, str)
            assert isinstance(mirror_count, int)
            assert isinstance(correlation, float)
            assert isinstance(lag, int)


if __name__ == "__main__":
    pytest.main([__file__])
