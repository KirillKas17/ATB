"""
Тесты для доменного сервиса анализа ликвидности.
"""
import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.liquidity_analyzer import LiquidityAnalyzer, ILiquidityAnalyzer
from domain.type_definitions.ml_types import LiquidityAnalysisResult, LiquidityZone, LiquiditySweep
class TestLiquidityAnalyzer:
    """Тесты для сервиса анализа ликвидности."""
    @pytest.fixture
    def liquidity_analyzer(self) -> Any:
        """Фикстура сервиса анализа ликвидности."""
        return LiquidityAnalyzer()
    @pytest.fixture
    def sample_market_data(self) -> Any:
        """Фикстура с примерными рыночными данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'vwap': np.random.uniform(50000, 51000, 100)
        }, index=dates)
    @pytest.fixture
    def sample_order_book(self) -> Any:
        """Фикстура с примерным ордербуком."""
        return {
            "bids": [
                {"price": "50000.0", "quantity": "1.5"},
                {"price": "49999.0", "quantity": "2.0"},
                {"price": "49998.0", "quantity": "1.0"},
                {"price": "49997.0", "quantity": "3.0"},
                {"price": "49996.0", "quantity": "1.5"},
            ],
            "asks": [
                {"price": "50001.0", "quantity": "1.0"},
                {"price": "50002.0", "quantity": "2.5"},
                {"price": "50003.0", "quantity": "1.5"},
                {"price": "50004.0", "quantity": "2.0"},
                {"price": "50005.0", "quantity": "1.0"},
            ],
            "timestamp": "2024-01-01T12:00:00Z"
        }
    def test_liquidity_analyzer_initialization(self, liquidity_analyzer) -> None:
        """Тест инициализации сервиса."""
        assert liquidity_analyzer is not None
        assert isinstance(liquidity_analyzer, ILiquidityAnalyzer)
        assert hasattr(liquidity_analyzer, 'config')
        assert isinstance(liquidity_analyzer.config, dict)
    def test_liquidity_analyzer_config_defaults(self, liquidity_analyzer) -> None:
        """Тест конфигурации по умолчанию."""
        config = liquidity_analyzer.config
        assert "liquidity_zone_size" in config
        assert "volume_threshold" in config
        assert "sweep_threshold" in config
        assert "lookback_period" in config
        assert isinstance(config["liquidity_zone_size"], float)
        assert isinstance(config["volume_threshold"], (int, float))
    def test_analyze_liquidity_valid_data(self, liquidity_analyzer, sample_market_data, sample_order_book) -> None:
        """Тест анализа ликвидности с валидными данными."""
        result = liquidity_analyzer.analyze_liquidity(sample_market_data, sample_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert "liquidity_score" in result
        assert "confidence" in result
        assert "volume_score" in result
        assert "order_book_score" in result
        assert "volatility_score" in result
        assert isinstance(result["liquidity_score"], float)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["volume_score"], float)
        assert isinstance(result["order_book_score"], float)
        assert isinstance(result["volatility_score"], float)
        # Проверяем логику
        assert result["liquidity_score"] >= 0.0 and result["liquidity_score"] <= 1.0
        assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0
        assert result["volume_score"] >= 0.0 and result["volume_score"] <= 1.0
        assert result["order_book_score"] >= 0.0 and result["order_book_score"] <= 1.0
        assert result["volatility_score"] >= 0.0 and result["volatility_score"] <= 1.0
    def test_analyze_liquidity_empty_market_data(self, liquidity_analyzer, sample_order_book) -> None:
        """Тест анализа ликвидности с пустыми рыночными данными."""
        empty_market_data = pd.DataFrame()
        result = liquidity_analyzer.analyze_liquidity(empty_market_data, sample_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["liquidity_score"] == 0.0
        assert result["confidence"] == 0.0
        assert result["volume_score"] == 0.0
        assert result["order_book_score"] >= 0.0  # Может быть рассчитан из ордербука
        assert result["volatility_score"] == 0.0
    def test_analyze_liquidity_empty_order_book(self, liquidity_analyzer, sample_market_data) -> None:
        """Тест анализа ликвидности с пустым ордербуком."""
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        result = liquidity_analyzer.analyze_liquidity(sample_market_data, empty_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["liquidity_score"] >= 0.0  # Может быть рассчитан из рыночных данных
        assert result["confidence"] >= 0.0
        assert result["volume_score"] >= 0.0  # Может быть рассчитан из объема
        assert result["order_book_score"] == 0.0
        assert result["volatility_score"] >= 0.0
    def test_analyze_liquidity_both_empty(self, liquidity_analyzer) -> None:
        """Тест анализа ликвидности с пустыми данными."""
        empty_market_data = pd.DataFrame()
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        result = liquidity_analyzer.analyze_liquidity(empty_market_data, empty_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        assert result["liquidity_score"] == 0.0
        assert result["confidence"] == 0.0
        assert result["volume_score"] == 0.0
        assert result["order_book_score"] == 0.0
        assert result["volatility_score"] == 0.0
    def test_identify_liquidity_zones_valid_data(self, liquidity_analyzer, sample_market_data) -> None:
        """Тест идентификации зон ликвидности с валидными данными."""
        zones = liquidity_analyzer.identify_liquidity_zones(sample_market_data)
        assert isinstance(zones, list)
        # Проверяем, что все зоны имеют правильную структуру
        for zone in zones:
            assert isinstance(zone, dict)
            assert "price" in zone
            assert "type" in zone
            assert "strength" in zone
            assert "volume" in zone
            assert "touches" in zone
            assert isinstance(zone["price"], float)
            assert isinstance(zone["type"], str)
            assert isinstance(zone["strength"], float)
            assert isinstance(zone["volume"], float)
            assert isinstance(zone["touches"], int)
            assert zone["type"] in ["support", "resistance", "neutral"]
            assert zone["strength"] >= 0.0 and zone["strength"] <= 1.0
            assert zone["volume"] >= 0.0
            assert zone["touches"] >= 0
    def test_identify_liquidity_zones_empty_data(self, liquidity_analyzer) -> None:
        """Тест идентификации зон ликвидности с пустыми данными."""
        empty_data = pd.DataFrame()
        zones = liquidity_analyzer.identify_liquidity_zones(empty_data)
        assert isinstance(zones, list)
        assert len(zones) == 0
    def test_identify_liquidity_zones_insufficient_data(self, liquidity_analyzer) -> None:
        """Тест идентификации зон ликвидности с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'high': [50000, 50001],
            'low': [49999, 50000],
            'volume': [1000, 1001]
        })
        zones = liquidity_analyzer.identify_liquidity_zones(insufficient_data)
        assert isinstance(zones, list)
        # При недостатке данных должно быть мало зон или их не должно быть
        assert len(zones) <= 2
    def test_detect_liquidity_sweeps_valid_data(self, liquidity_analyzer, sample_market_data) -> None:
        """Тест обнаружения сметаний ликвидности с валидными данными."""
        sweeps = liquidity_analyzer.detect_liquidity_sweeps(sample_market_data)
        assert isinstance(sweeps, list)
        # Проверяем, что все сметания имеют правильную структуру
        for sweep in sweeps:
            assert isinstance(sweep, LiquiditySweep)
            assert "timestamp" in sweep
            assert "price" in sweep
            assert "type" in sweep
            assert "confidence" in sweep
            assert isinstance(sweep["price"], float)
            assert isinstance(sweep["type"], str)
            assert isinstance(sweep["confidence"], float)
            assert sweep["type"] in ["sweep_high", "sweep_low"]
            assert sweep["confidence"] >= 0.0 and sweep["confidence"] <= 1.0
            assert sweep["price"] > 0.0
    def test_detect_liquidity_sweeps_empty_data(self, liquidity_analyzer) -> None:
        """Тест обнаружения сметаний ликвидности с пустыми данными."""
        empty_data = pd.DataFrame()
        sweeps = liquidity_analyzer.detect_liquidity_sweeps(empty_data)
        assert isinstance(sweeps, list)
        assert len(sweeps) == 0
    def test_detect_liquidity_sweeps_insufficient_data(self, liquidity_analyzer) -> None:
        """Тест обнаружения сметаний ликвидности с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'high': [50000, 50001, 50002],
            'low': [49999, 50000, 50001],
            'volume': [1000, 1001, 1002]
        })
        sweeps = liquidity_analyzer.detect_liquidity_sweeps(insufficient_data)
        assert isinstance(sweeps, list)
        # При недостатке данных должно быть мало сметаний или их не должно быть
        assert len(sweeps) <= 3
    def test_calculate_volume_profile(self, liquidity_analyzer, sample_market_data) -> None:
        """Тест расчета профиля объема."""
        profile = liquidity_analyzer.calculate_volume_profile(sample_market_data)
        assert isinstance(profile, dict)
        assert "poc" in profile  # Point of Control
        assert "value_area" in profile
        assert "volume_distribution" in profile
        assert isinstance(profile["poc"], float)
        assert isinstance(profile["value_area"], list)
        assert isinstance(profile["volume_distribution"], dict)
        assert profile["poc"] > 0.0
        assert len(profile["value_area"]) >= 2  # Минимум верхняя и нижняя граница
    def test_calculate_volume_profile_empty_data(self, liquidity_analyzer) -> None:
        """Тест расчета профиля объема с пустыми данными."""
        empty_data = pd.DataFrame()
        profile = liquidity_analyzer.calculate_volume_profile(empty_data)
        assert isinstance(profile, dict)
        assert profile["poc"] == 0.0
        assert profile["value_area"] == [0.0, 0.0]
        assert profile["volume_distribution"] == {}
    def test_identify_support_resistance_levels(self, liquidity_analyzer, sample_market_data) -> None:
        """Тест идентификации уровней поддержки и сопротивления."""
        levels = liquidity_analyzer.identify_support_resistance_levels(sample_market_data)
        assert isinstance(levels, dict)
        assert "support_levels" in levels
        assert "resistance_levels" in levels
        assert isinstance(levels["support_levels"], list)
        assert isinstance(levels["resistance_levels"], list)
        # Проверяем структуру уровней поддержки
        for level in levels["support_levels"]:
            assert isinstance(level, dict)
            assert "price" in level
            assert "strength" in level
            assert "volume" in level
            assert isinstance(level["price"], float)
            assert isinstance(level["strength"], float)
            assert isinstance(level["volume"], float)
            assert level["price"] > 0.0
            assert level["strength"] >= 0.0 and level["strength"] <= 1.0
            assert level["volume"] >= 0.0
        # Проверяем структуру уровней сопротивления
        for level in levels["resistance_levels"]:
            assert isinstance(level, dict)
            assert "price" in level
            assert "strength" in level
            assert "volume" in level
            assert isinstance(level["price"], float)
            assert isinstance(level["strength"], float)
            assert isinstance(level["volume"], float)
            assert level["price"] > 0.0
            assert level["strength"] >= 0.0 and level["strength"] <= 1.0
            assert level["volume"] >= 0.0
    def test_identify_support_resistance_levels_empty_data(self, liquidity_analyzer) -> None:
        """Тест идентификации уровней поддержки и сопротивления с пустыми данными."""
        empty_data = pd.DataFrame()
        levels = liquidity_analyzer.identify_support_resistance_levels(empty_data)
        assert isinstance(levels, dict)
        assert levels["support_levels"] == []
        assert levels["resistance_levels"] == []
    def test_calculate_liquidity_metrics(self, liquidity_analyzer, sample_market_data, sample_order_book) -> None:
        """Тест расчета метрик ликвидности."""
        metrics = liquidity_analyzer.calculate_liquidity_metrics(sample_market_data, sample_order_book)
        assert isinstance(metrics, dict)
        assert "bid_ask_spread" in metrics
        assert "order_book_depth" in metrics
        assert "volume_liquidity" in metrics
        assert "price_impact" in metrics
        assert "liquidity_score" in metrics
        assert isinstance(metrics["bid_ask_spread"], float)
        assert isinstance(metrics["order_book_depth"], float)
        assert isinstance(metrics["volume_liquidity"], float)
        assert isinstance(metrics["price_impact"], float)
        assert isinstance(metrics["liquidity_score"], float)
        assert metrics["bid_ask_spread"] >= 0.0
        assert metrics["order_book_depth"] >= 0.0
        assert metrics["volume_liquidity"] >= 0.0
        assert metrics["price_impact"] >= 0.0
        assert metrics["liquidity_score"] >= 0.0 and metrics["liquidity_score"] <= 1.0
    def test_calculate_liquidity_metrics_empty_data(self, liquidity_analyzer) -> None:
        """Тест расчета метрик ликвидности с пустыми данными."""
        empty_market_data = pd.DataFrame()
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        metrics = liquidity_analyzer.calculate_liquidity_metrics(empty_market_data, empty_order_book)
        assert isinstance(metrics, dict)
        assert metrics["bid_ask_spread"] == 0.0
        assert metrics["order_book_depth"] == 0.0
        assert metrics["volume_liquidity"] == 0.0
        assert metrics["price_impact"] == 0.0
        assert metrics["liquidity_score"] == 0.0
    def test_detect_liquidity_clusters(self, liquidity_analyzer, sample_market_data) -> None:
        """Тест обнаружения кластеров ликвидности."""
        clusters = liquidity_analyzer.detect_liquidity_clusters(sample_market_data)
        assert isinstance(clusters, list)
        # Проверяем, что все кластеры имеют правильную структуру
        for cluster in clusters:
            assert isinstance(cluster, dict)
            assert "center_price" in cluster
            assert "volume" in cluster
            assert "density" in cluster
            assert "strength" in cluster
            assert isinstance(cluster["center_price"], float)
            assert isinstance(cluster["volume"], float)
            assert isinstance(cluster["density"], float)
            assert isinstance(cluster["strength"], float)
            assert cluster["center_price"] > 0.0
            assert cluster["volume"] >= 0.0
            assert cluster["density"] >= 0.0
            assert cluster["strength"] >= 0.0 and cluster["strength"] <= 1.0
    def test_detect_liquidity_clusters_empty_data(self, liquidity_analyzer) -> None:
        """Тест обнаружения кластеров ликвидности с пустыми данными."""
        empty_data = pd.DataFrame()
        clusters = liquidity_analyzer.detect_liquidity_clusters(empty_data)
        assert isinstance(clusters, list)
        assert len(clusters) == 0
    def test_analyze_order_book_imbalance(self, liquidity_analyzer, sample_order_book) -> None:
        """Тест анализа дисбаланса ордербука."""
        imbalance = liquidity_analyzer.analyze_order_book_imbalance(sample_order_book)
        assert isinstance(imbalance, dict)
        assert "bid_volume" in imbalance
        assert "ask_volume" in imbalance
        assert "imbalance_ratio" in imbalance
        assert "imbalance_direction" in imbalance
        assert isinstance(imbalance["bid_volume"], float)
        assert isinstance(imbalance["ask_volume"], float)
        assert isinstance(imbalance["imbalance_ratio"], float)
        assert isinstance(imbalance["imbalance_direction"], str)
        assert imbalance["bid_volume"] >= 0.0
        assert imbalance["ask_volume"] >= 0.0
        assert imbalance["imbalance_ratio"] >= 0.0
        assert imbalance["imbalance_direction"] in ["bid_heavy", "ask_heavy", "balanced"]
    def test_analyze_order_book_imbalance_empty_order_book(self, liquidity_analyzer) -> None:
        """Тест анализа дисбаланса ордербука с пустым ордербуком."""
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        imbalance = liquidity_analyzer.analyze_order_book_imbalance(empty_order_book)
        assert isinstance(imbalance, dict)
        assert imbalance["bid_volume"] == 0.0
        assert imbalance["ask_volume"] == 0.0
        assert imbalance["imbalance_ratio"] == 0.0
        assert imbalance["imbalance_direction"] == "balanced"
    def test_calculate_market_impact(self, liquidity_analyzer, sample_market_data, sample_order_book) -> None:
        """Тест расчета рыночного воздействия."""
        impact = liquidity_analyzer.calculate_market_impact(sample_market_data, sample_order_book)
        assert isinstance(impact, dict)
        assert "buy_impact" in impact
        assert "sell_impact" in impact
        assert "average_impact" in impact
        assert "impact_curve" in impact
        assert isinstance(impact["buy_impact"], float)
        assert isinstance(impact["sell_impact"], float)
        assert isinstance(impact["average_impact"], float)
        assert isinstance(impact["impact_curve"], dict)
        assert impact["buy_impact"] >= 0.0
        assert impact["sell_impact"] >= 0.0
        assert impact["average_impact"] >= 0.0
    def test_calculate_market_impact_empty_data(self, liquidity_analyzer) -> None:
        """Тест расчета рыночного воздействия с пустыми данными."""
        empty_market_data = pd.DataFrame()
        empty_order_book = {"bids": [], "asks": [], "timestamp": "2024-01-01T12:00:00Z"}
        impact = liquidity_analyzer.calculate_market_impact(empty_market_data, empty_order_book)
        assert isinstance(impact, dict)
        assert impact["buy_impact"] == 0.0
        assert impact["sell_impact"] == 0.0
        assert impact["average_impact"] == 0.0
        assert impact["impact_curve"] == {}
    def test_liquidity_analyzer_error_handling(self, liquidity_analyzer) -> None:
        """Тест обработки ошибок в сервисе."""
        # Тест с None данными
        with pytest.raises(Exception):
            liquidity_analyzer.analyze_liquidity(None, None)
        # Тест с невалидным типом данных
        with pytest.raises(Exception):
            liquidity_analyzer.identify_liquidity_zones("invalid_data")
        # Тест с невалидным ордербуком
        invalid_order_book = {"invalid": "data"}
        result = liquidity_analyzer.analyze_liquidity(pd.DataFrame(), invalid_order_book)
        # Проверяем структуру результата вместо isinstance
        assert isinstance(result, dict)
        # Должен вернуть безопасные значения
    def test_liquidity_analyzer_performance(self, liquidity_analyzer, sample_market_data, sample_order_book) -> None:
        """Тест производительности сервиса."""
        import time
        start_time = time.time()
        for _ in range(50):  # Меньше итераций, так как анализ ликвидности сложнее
            liquidity_analyzer.analyze_liquidity(sample_market_data, sample_order_book)
        end_time = time.time()
        # Проверяем, что 50 операций выполняются менее чем за 2 секунды
        assert (end_time - start_time) < 2.0
    def test_liquidity_analyzer_thread_safety(self, liquidity_analyzer, sample_market_data, sample_order_book) -> None:
        """Тест потокобезопасности сервиса."""
        import threading
        import queue
        results = queue.Queue()
        def analyze_liquidity() -> Any:
            try:
                result = liquidity_analyzer.analyze_liquidity(sample_market_data, sample_order_book)
                results.put(result)
            except Exception as e:
                results.put(e)
        # Запускаем несколько потоков одновременно
        threads = []
        for _ in range(5):  # Меньше потоков для сложных операций
            thread = threading.Thread(target=analyze_liquidity)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все результаты корректны
        for _ in range(5):
            result = results.get()
            # Проверяем структуру результата вместо isinstance
            assert isinstance(result, dict)
            assert "liquidity_score" in result
    def test_liquidity_analyzer_config_customization(self) -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {
            "liquidity_zone_size": 0.01,
            "volume_threshold": 50000,
            "sweep_threshold": 0.03,
            "lookback_period": 50
        }
        analyzer = LiquidityAnalyzer(custom_config)
        assert analyzer.config["liquidity_zone_size"] == 0.01
        assert analyzer.config["volume_threshold"] == 50000
        assert analyzer.config["sweep_threshold"] == 0.03
        assert analyzer.config["lookback_period"] == 50
    def test_liquidity_analyzer_integration_with_market_data(self, liquidity_analyzer, sample_market_data, sample_order_book) -> None:
        """Тест интеграции с рыночными данными."""
        # Полный анализ ликвидности
        analysis = liquidity_analyzer.analyze_liquidity(sample_market_data, sample_order_book)
        # Идентификация зон
        zones = liquidity_analyzer.identify_liquidity_zones(sample_market_data)
        # Обнаружение сметаний
        sweeps = liquidity_analyzer.detect_liquidity_sweeps(sample_market_data)
        # Проверяем согласованность результатов
        # Проверяем структуру результата вместо isinstance
        assert isinstance(analysis, dict)
        assert isinstance(zones, list)
        assert isinstance(sweeps, list)
        # Если есть зоны ликвидности, анализ должен показать некоторую ликвидность
        if len(zones) > 0:
            assert analysis["liquidity_score"] > 0.0
        # Если есть сметания, уверенность должна быть положительной
        if len(sweeps) > 0:
            assert analysis["confidence"] > 0.0 
