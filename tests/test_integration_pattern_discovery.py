import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.pattern_discovery import (Pattern, PatternConfig,
                                               PatternDiscovery)
from shared.logging import setup_logger
logger = setup_logger(__name__)
    @pytest.fixture
def mock_market_data() -> Any:
    """Фикстура с тестовыми рыночными данными"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
    data = pd.DataFrame(
        {
            "open": np.random.normal(100, 1, 100),
            "high": np.random.normal(101, 1, 100),
            "low": np.random.normal(99, 1, 100),
            "close": np.random.normal(100, 1, 100),
            "volume": np.random.normal(1000, 100, 100),
        },
        index=dates,
    )
    return data
    @pytest.fixture
def pattern_config() -> Any:
    """Фикстура с конфигурацией паттернов"""
    return PatternConfig(
        min_pattern_length=5,
        max_pattern_length=20,
        min_confidence=0.7,
        min_support=0.1,
        max_patterns=100,
        clustering_method="dbscan",
        min_cluster_size=3,
        pattern_types=["candle", "price", "volume"],
        feature_columns=["open", "high", "low", "close", "volume"],
        window_sizes=[5, 10, 20],
        similarity_threshold=0.8,
        technical_indicators=["RSI", "MACD", "BB"],
        volume_threshold=1.5,
        price_threshold=0.02,
        trend_window=20,
    )
    @pytest.fixture
def pattern_discovery(pattern_config) -> Any:
    """Фикстура с экземпляром PatternDiscovery"""
    return PatternDiscovery(config=pattern_config)
class TestPatternDiscovery:
    def test_discover_patterns(self, pattern_discovery, mock_market_data) -> None:
        """Тест обнаружения паттернов"""
        patterns = pattern_discovery.discover_patterns(mock_market_data)
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(
            p.confidence >= pattern_discovery.config.min_confidence for p in patterns
        )
        assert all(p.support >= pattern_discovery.config.min_support for p in patterns)
    def test_prepare_features(self, pattern_discovery, mock_market_data) -> None:
        """Тест подготовки признаков"""
        features = pattern_discovery.prepare_features(mock_market_data)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 100
        assert all(
            col in features.columns for col in pattern_discovery.config.feature_columns
        )
        assert "RSI" in features.columns
        assert "MACD" in features.columns
        assert "MACD_signal" in features.columns
        assert "MACD_hist" in features.columns
        assert "BB_upper" in features.columns
        assert "BB_middle" in features.columns
        assert "BB_lower" in features.columns
    def test_cluster_patterns(self, pattern_discovery, mock_market_data) -> None:
        """Тест кластеризации паттернов"""
        # Тест с пустым списком паттернов
        empty_clusters = pattern_discovery.cluster_patterns([])
        assert isinstance(empty_clusters, dict)
        assert len(empty_clusters) == 0
        # Тест с реальными паттернами
        patterns = pattern_discovery.discover_patterns(mock_market_data)
        clusters = pattern_discovery.cluster_patterns(patterns)
        assert isinstance(clusters, dict)
        assert len(clusters) > 0
        assert all(isinstance(cluster, list) for cluster in clusters.values())
        assert all(
            isinstance(p, Pattern) for cluster in clusters.values() for p in cluster
        )
    def test_find_candle_patterns(self, pattern_discovery, mock_market_data) -> None:
        """Тест поиска свечных паттернов"""
        # Тест с пустыми данными
        empty_patterns = pattern_discovery.find_candle_patterns(pd.DataFrame())
        assert isinstance(empty_patterns, list)
        assert len(empty_patterns) == 0
        # Тест с реальными данными
        patterns = pattern_discovery.find_candle_patterns(mock_market_data)
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(p.pattern_type == "candle" for p in patterns)
        assert all("pattern_name" in p.metadata for p in patterns)
    def test_find_price_patterns(self, pattern_discovery, mock_market_data) -> None:
        """Тест поиска ценовых паттернов"""
        # Тест с пустыми данными
        empty_patterns = pattern_discovery.find_price_patterns(pd.DataFrame())
        assert isinstance(empty_patterns, list)
        assert len(empty_patterns) == 0
        # Тест с данными без изменений цены
        flat_data = pd.DataFrame({"close": [100] * 100, "volume": [1000] * 100})
        flat_patterns = pattern_discovery.find_price_patterns(flat_data)
        assert isinstance(flat_patterns, list)
        assert len(flat_patterns) == 0
        # Тест с реальными данными
        patterns = pattern_discovery.find_price_patterns(mock_market_data)
        assert isinstance(patterns, list)
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(p.pattern_type == "price" for p in patterns)
        assert all("price_change" in p.metadata for p in patterns)
        assert all(
            abs(p.metadata["price_change"]) >= pattern_discovery.config.price_threshold
            for p in patterns
        )
    def test_find_volume_patterns(self, pattern_discovery, mock_market_data) -> None:
        """Тест поиска паттернов объема"""
        # Тест с пустыми данными
        empty_patterns = pattern_discovery.find_volume_patterns(pd.DataFrame())
        assert isinstance(empty_patterns, list)
        assert len(empty_patterns) == 0
        # Тест с данными без аномальных объемов
        flat_data = pd.DataFrame({"close": [100] * 100, "volume": [1000] * 100})
        flat_patterns = pattern_discovery.find_volume_patterns(flat_data)
        assert isinstance(flat_patterns, list)
        assert len(flat_patterns) == 0
        # Тест с реальными данными
        patterns = pattern_discovery.find_volume_patterns(mock_market_data)
        assert isinstance(patterns, list)
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(p.pattern_type == "volume" for p in patterns)
        assert all("volume_change" in p.metadata for p in patterns)
        assert all(
            p.metadata["volume_change"] >= pattern_discovery.config.volume_threshold
            for p in patterns
        )
    def test_evaluate_pattern(self, pattern_discovery, mock_market_data) -> None:
        """Тест оценки паттерна"""
        # Тест с минимальным паттерном
        min_pattern = Pattern(
            pattern_type="candle",
            start_idx=0,
            end_idx=pattern_discovery.config.min_pattern_length,
            features=mock_market_data.iloc[
                0 : pattern_discovery.config.min_pattern_length
            ].values,
            confidence=0.7,
            support=0.1,
            metadata={"pattern_name": "test_pattern"},
        )
        min_score = pattern_discovery.evaluate_pattern(min_pattern, mock_market_data)
        assert isinstance(min_score, float)
        assert 0 <= min_score <= 1
        # Тест с максимальным паттерном
        max_pattern = Pattern(
            pattern_type="candle",
            start_idx=0,
            end_idx=pattern_discovery.config.max_pattern_length,
            features=mock_market_data.iloc[
                0 : pattern_discovery.config.max_pattern_length
            ].values,
            confidence=1.0,
            support=1.0,
            metadata={"pattern_name": "test_pattern"},
        )
        max_score = pattern_discovery.evaluate_pattern(max_pattern, mock_market_data)
        assert isinstance(max_score, float)
        assert 0 <= max_score <= 1
        assert (
            max_score >= min_score
        )  # Максимальный паттерн должен иметь не меньшую оценку
        # Проверка обновления атрибутов паттерна
        assert max_pattern.trend is not None
        assert max_pattern.volume_profile is not None
        assert isinstance(max_pattern.technical_indicators, dict)
        assert all(
            isinstance(v, float) for v in max_pattern.technical_indicators.values()
        )
    def test_save_and_load_patterns(
        self, pattern_discovery, mock_market_data, tmp_path
    ) -> None:
        """Тест сохранения и загрузки паттернов"""
        patterns = pattern_discovery.discover_patterns(mock_market_data)
        # Оцениваем паттерны перед сохранением
        for pattern in patterns:
            pattern_discovery.evaluate_pattern(pattern, mock_market_data)
        file_path = tmp_path / "patterns.json"
        # Сохранение паттернов
        pattern_discovery.save_patterns(patterns, str(file_path))
        assert file_path.exists()
        # Загрузка паттернов
        loaded_patterns = pattern_discovery.load_patterns(str(file_path))
        assert isinstance(loaded_patterns, list)
        assert len(loaded_patterns) == len(patterns)
        assert all(isinstance(p, Pattern) for p in loaded_patterns)
        assert all(p.trend is not None for p in loaded_patterns)
        assert all(p.volume_profile is not None for p in loaded_patterns)
        assert all(isinstance(p.technical_indicators, dict) for p in loaded_patterns)
    def test_error_handling(self, pattern_discovery) -> None:
        """Тест обработки ошибок"""
        # Тест с пустыми данными
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            pattern_discovery.discover_patterns(empty_data)
        # Тест с некорректными данными
        invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
        with pytest.raises(ValueError):
            pattern_discovery.discover_patterns(invalid_data)
        # Тест с некорректной конфигурацией
        invalid_configs = [
            PatternConfig(  # min_pattern_length >= max_pattern_length
                min_pattern_length=20,
                max_pattern_length=10,
                min_confidence=0.7,
                min_support=0.1,
                max_patterns=100,
                clustering_method="dbscan",
                min_cluster_size=3,
                pattern_types=[],
                feature_columns=[],
                window_sizes=[5, 10, 20],
                similarity_threshold=0.8,
            ),
            PatternConfig(  # пустой pattern_types
                min_pattern_length=5,
                max_pattern_length=20,
                min_confidence=0.7,
                min_support=0.1,
                max_patterns=100,
                clustering_method="dbscan",
                min_cluster_size=3,
                pattern_types=[],
                feature_columns=["open", "high", "low", "close", "volume"],
                window_sizes=[5, 10, 20],
                similarity_threshold=0.8,
            ),
            PatternConfig(  # пустой feature_columns
                min_pattern_length=5,
                max_pattern_length=20,
                min_confidence=0.7,
                min_support=0.1,
                max_patterns=100,
                clustering_method="dbscan",
                min_cluster_size=3,
                pattern_types=["candle"],
                feature_columns=[],
                window_sizes=[5, 10, 20],
                similarity_threshold=0.8,
            ),
        ]
        for config in invalid_configs:
            with pytest.raises(ValueError):
                PatternDiscovery(config=config)
        # Тест с некорректными типами паттернов
        invalid_pattern_type_config = PatternConfig(
            min_pattern_length=5,
            max_pattern_length=20,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=["invalid_type"],
            feature_columns=["open", "high", "low", "close", "volume"],
            window_sizes=[5, 10, 20],
            similarity_threshold=0.8,
        )
        discovery = PatternDiscovery(config=invalid_pattern_type_config)
        with pytest.raises(AttributeError):
            discovery.discover_patterns(
                pd.DataFrame(
                    {
                        "open": [1, 2, 3],
                        "high": [2, 3, 4],
                        "low": [0, 1, 2],
                        "close": [1, 2, 3],
                        "volume": [100, 200, 300],
                    }
                )
            )
    def test_pattern_combining(self, pattern_discovery, mock_market_data) -> None:
        """Тест комбинирования паттернов"""
        patterns = pattern_discovery.discover_patterns(mock_market_data)
        combined_patterns = pattern_discovery.combine_patterns(patterns)
        assert isinstance(combined_patterns, list)
        assert len(combined_patterns) > 0
        assert all(isinstance(p, Pattern) for p in combined_patterns)
        # Проверяем метаданные только для комбинированных паттернов
        for p in combined_patterns:
            if "original_patterns" in p.metadata:
                assert "combined_from" in p.metadata
                assert len(p.metadata["original_patterns"]) > 1
    def test_technical_indicators(self, pattern_discovery, mock_market_data) -> None:
        """Тест расчета технических индикаторов"""
        indicators = pattern_discovery._calculate_technical_indicators(mock_market_data)
        assert isinstance(indicators, dict)
        assert "RSI" in indicators
        assert "MACD" in indicators
        assert "MACD_signal" in indicators
        assert all(isinstance(v, float) for v in indicators.values())
    def test_trend_calculation(self, pattern_discovery, mock_market_data) -> None:
        """Тест расчета тренда"""
        # Тест с реальными данными
        trend = pattern_discovery._calculate_trend(mock_market_data)
        assert isinstance(trend, float)
        assert not np.isnan(trend)
        assert not np.isinf(trend)
        # Тест с данными без изменений
        flat_data = pd.DataFrame({"close": [100] * 100})
        flat_trend = pattern_discovery._calculate_trend(flat_data)
        assert isinstance(flat_trend, float)
        assert not np.isnan(flat_trend)
        assert not np.isinf(flat_trend)
        assert abs(flat_trend) < 0.1  # Тренд должен быть близок к нулю
        # Тест с восходящим трендом
        up_data = pd.DataFrame({"close": np.linspace(100, 200, 100)})
        up_trend = pattern_discovery._calculate_trend(up_data)
        assert up_trend > 0
        # Тест с нисходящим трендом
        down_data = pd.DataFrame({"close": np.linspace(200, 100, 100)})
        down_trend = pattern_discovery._calculate_trend(down_data)
        assert down_trend < 0
    def test_volume_profile(self, pattern_discovery, mock_market_data) -> None:
        """Тест расчета профиля объема"""
        # Тест с реальными данными
        profile = pattern_discovery._calculate_volume_profile(mock_market_data)
        assert isinstance(profile, float)
        assert not np.isnan(profile)
        assert not np.isinf(profile)
        assert profile > 0
        # Тест с постоянным объемом
        flat_data = pd.DataFrame({"volume": [1000] * 100})
        flat_profile = pattern_discovery._calculate_volume_profile(flat_data)
        assert isinstance(flat_profile, float)
        assert not np.isnan(flat_profile)
        assert not np.isinf(flat_profile)
        assert flat_profile > 0
        # Тест с возрастающим объемом
        up_data = pd.DataFrame({"volume": np.linspace(1000, 2000, 100)})
        up_profile = pattern_discovery._calculate_volume_profile(up_data)
        assert (
            up_profile > flat_profile
        )  # Профиль должен быть выше при возрастающем объеме
        # Тест с убывающим объемом
        down_data = pd.DataFrame({"volume": np.linspace(2000, 1000, 100)})
        down_profile = pattern_discovery._calculate_volume_profile(down_data)
        assert (
            down_profile > flat_profile
        )  # Профиль должен быть выше при убывающем объеме
    def test_pattern_ranking(self, pattern_discovery, mock_market_data) -> None:
        """Тест ранжирования паттернов"""
        # Тест с пустым списком
        empty_ranked = pattern_discovery._rank_patterns([])
        assert isinstance(empty_ranked, list)
        assert len(empty_ranked) == 0
        # Тест с реальными паттернами
        patterns = pattern_discovery.discover_patterns(mock_market_data)
        ranked_patterns = pattern_discovery._rank_patterns(patterns)
        assert len(ranked_patterns) == len(patterns)
        assert all(
            ranked_patterns[i].confidence * ranked_patterns[i].support
            >= ranked_patterns[i + 1].confidence * ranked_patterns[i + 1].support
            for i in range(len(ranked_patterns) - 1)
        )
    def test_pattern_filtering(self, pattern_discovery, mock_market_data) -> None:
        """Тест фильтрации паттернов"""
        # Тест с пустым списком
        empty_filtered = pattern_discovery._filter_patterns([])
        assert isinstance(empty_filtered, list)
        assert len(empty_filtered) == 0
        # Тест с реальными паттернами
        patterns = pattern_discovery.discover_patterns(mock_market_data)
        filtered_patterns = pattern_discovery._filter_patterns(patterns)
        assert all(
            p.confidence >= pattern_discovery.config.min_confidence
            for p in filtered_patterns
        )
        assert all(
            p.support >= pattern_discovery.config.min_support for p in filtered_patterns
        )
        # Тест с паттернами ниже порога
        low_confidence_patterns = [
            Pattern(
                pattern_type="candle",
                start_idx=0,
                end_idx=10,
                features=np.zeros((10, 5)),
                confidence=pattern_discovery.config.min_confidence - 0.1,
                support=pattern_discovery.config.min_support,
                metadata={"pattern_name": "test_pattern"},
            )
        ]
        filtered_low = pattern_discovery._filter_patterns(low_confidence_patterns)
        assert len(filtered_low) == 0
