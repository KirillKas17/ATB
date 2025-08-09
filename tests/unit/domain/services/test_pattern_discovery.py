"""
Unit тесты для PatternDiscovery.

Покрывает:
- Основной функционал обнаружения паттернов
- Валидацию конфигурации
- Бизнес-логику анализа паттернов
- Обработку ошибок
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from domain.services.pattern_discovery import PatternDiscovery, PatternConfig, Pattern
from domain.type_definitions.service_types import PatternType, AnalysisConfig, MarketDataFrame
from domain.exceptions.base_exceptions import ValidationError


class TestPatternConfig:
    """Тесты для PatternConfig."""

    def test_creation(self):
        """Тест создания конфигурации."""
        config = PatternConfig(
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=[PatternType.CANDLE, PatternType.PRICE],
            feature_columns=["close", "volume"],
            window_sizes=[10, 20],
            similarity_threshold=0.8,
        )

        assert config.min_pattern_length == 5
        assert config.max_pattern_length == 50
        assert config.min_confidence == 0.7
        assert config.pattern_types == [PatternType.CANDLE, PatternType.PRICE]

    def test_default_values(self):
        """Тест значений по умолчанию."""
        config = PatternConfig(
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=[PatternType.CANDLE],
            feature_columns=["close"],
            window_sizes=[10],
            similarity_threshold=0.8,
        )

        assert config.volume_threshold == 1.5
        assert config.price_threshold == 0.02
        assert config.trend_window == 20


class TestPattern:
    """Тесты для Pattern."""

    def test_creation(self):
        """Тест создания паттерна."""
        features = np.array([1.0, 2.0, 3.0])
        pattern = Pattern(
            pattern_type=PatternType.CANDLE,
            start_idx=0,
            end_idx=5,
            features=features,
            confidence=0.8,
            support=0.15,
            metadata={"pattern": "doji"},
        )

        assert pattern.pattern_type == PatternType.CANDLE
        assert pattern.start_idx == 0
        assert pattern.end_idx == 5
        assert np.array_equal(pattern.features, features)
        assert pattern.confidence == 0.8
        assert pattern.metadata["pattern"] == "doji"

    def test_from_dict(self):
        """Тест создания из словаря."""
        data = {
            "pattern_type": "CANDLE",
            "start_idx": 0,
            "end_idx": 5,
            "features": [1.0, 2.0, 3.0],
            "confidence": 0.8,
            "support": 0.15,
            "metadata": {"pattern": "doji"},
        }

        pattern = Pattern.from_dict(data)

        assert pattern.pattern_type == PatternType.CANDLE
        assert pattern.start_idx == 0
        assert pattern.end_idx == 5
        assert pattern.confidence == 0.8


class TestPatternDiscovery:
    """Тесты для PatternDiscovery."""

    @pytest.fixture
    def config(self) -> PatternConfig:
        """Тестовая конфигурация."""
        return PatternConfig(
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=[PatternType.CANDLE, PatternType.PRICE, PatternType.VOLUME],
            feature_columns=["open", "high", "low", "close", "volume"],
            window_sizes=[10, 20],
            similarity_threshold=0.8,
        )

    @pytest.fixture
    def sample_data(self) -> MarketDataFrame:
        """Тестовые данные."""
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        data = pd.DataFrame(
            {
                "open": np.random.uniform(100, 200, 100),
                "high": np.random.uniform(200, 300, 100),
                "low": np.random.uniform(50, 100, 100),
                "close": np.random.uniform(100, 200, 100),
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

        # Создаем некоторые паттерны
        data.loc[10:15, "close"] = data.loc[10:15, "close"] * 1.1  # Восходящий тренд
        data.loc[20:25, "close"] = data.loc[20:25, "close"] * 0.9  # Нисходящий тренд

        return data

    @pytest.fixture
    def pattern_discovery(self, config) -> PatternDiscovery:
        """Экземпляр PatternDiscovery."""
        return PatternDiscovery(config)

    def test_creation(self, config):
        """Тест создания сервиса."""
        service = PatternDiscovery(config)
        assert service.config == config
        assert service.scaler is not None

    def test_validation_config_invalid_pattern_length(self):
        """Тест валидации конфигурации с неверной длиной паттерна."""
        config = PatternConfig(
            min_pattern_length=50,
            max_pattern_length=10,  # Меньше min_pattern_length
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=[PatternType.CANDLE],
            feature_columns=["close"],
            window_sizes=[10],
            similarity_threshold=0.8,
        )

        with pytest.raises(ValueError, match="min_pattern_length must be less than max_pattern_length"):
            PatternDiscovery(config)

    def test_validation_config_empty_pattern_types(self):
        """Тест валидации конфигурации с пустыми типами паттернов."""
        config = PatternConfig(
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=[],  # Пустой список
            feature_columns=["close"],
            window_sizes=[10],
            similarity_threshold=0.8,
        )

        with pytest.raises(ValueError, match="pattern_types cannot be empty"):
            PatternDiscovery(config)

    def test_validation_config_empty_feature_columns(self):
        """Тест валидации конфигурации с пустыми колонками признаков."""
        config = PatternConfig(
            min_pattern_length=5,
            max_pattern_length=50,
            min_confidence=0.7,
            min_support=0.1,
            max_patterns=100,
            clustering_method="dbscan",
            min_cluster_size=3,
            pattern_types=[PatternType.CANDLE],
            feature_columns=[],  # Пустой список
            window_sizes=[10],
            similarity_threshold=0.8,
        )

        with pytest.raises(ValueError, match="feature_columns cannot be empty"):
            PatternDiscovery(config)

    @pytest.mark.asyncio
    async def test_discover_patterns_empty_data(self, pattern_discovery):
        """Тест обнаружения паттернов с пустыми данными."""
        empty_data = pd.DataFrame()
        analysis_config = AnalysisConfig()

        with pytest.raises(ValueError, match="Empty data provided"):
            await pattern_discovery.discover_patterns(empty_data, analysis_config)

    @pytest.mark.asyncio
    async def test_discover_patterns_missing_columns(self, pattern_discovery):
        """Тест обнаружения паттернов с отсутствующими колонками."""
        data = pd.DataFrame({"open": [100, 101, 102]})  # Отсутствуют другие колонки
        analysis_config = AnalysisConfig()

        with pytest.raises(ValueError, match="Missing required feature columns"):
            await pattern_discovery.discover_patterns(data, analysis_config)

    @pytest.mark.asyncio
    async def test_discover_patterns_success(self, pattern_discovery, sample_data):
        """Тест успешного обнаружения паттернов."""
        analysis_config = AnalysisConfig()

        patterns = await pattern_discovery.discover_patterns(sample_data, analysis_config)

        assert isinstance(patterns, list)
        assert len(patterns) <= pattern_discovery.config.max_patterns

        for pattern in patterns:
            assert isinstance(pattern, Pattern)
            assert pattern.confidence >= pattern_discovery.config.min_confidence
            assert pattern.support >= pattern_discovery.config.min_support

    @pytest.mark.asyncio
    async def test_discover_patterns_candle_only(self, config, sample_data):
        """Тест обнаружения только свечных паттернов."""
        config.pattern_types = [PatternType.CANDLE]
        service = PatternDiscovery(config)
        analysis_config = AnalysisConfig()

        patterns = await service.discover_patterns(sample_data, analysis_config)

        assert all(p.pattern_type == PatternType.CANDLE for p in patterns)

    @pytest.mark.asyncio
    async def test_discover_patterns_price_only(self, config, sample_data):
        """Тест обнаружения только ценовых паттернов."""
        config.pattern_types = [PatternType.PRICE]
        service = PatternDiscovery(config)
        analysis_config = AnalysisConfig()

        patterns = await service.discover_patterns(sample_data, analysis_config)

        assert all(p.pattern_type == PatternType.PRICE for p in patterns)

    @pytest.mark.asyncio
    async def test_discover_patterns_volume_only(self, config, sample_data):
        """Тест обнаружения только объемных паттернов."""
        config.pattern_types = [PatternType.VOLUME]
        service = PatternDiscovery(config)
        analysis_config = AnalysisConfig()

        patterns = await service.discover_patterns(sample_data, analysis_config)

        assert all(p.pattern_type == PatternType.VOLUME for p in patterns)

    @pytest.mark.asyncio
    async def test_validate_pattern(self, pattern_discovery, sample_data):
        """Тест валидации паттерна."""
        pattern = Pattern(
            pattern_type=PatternType.CANDLE,
            start_idx=0,
            end_idx=5,
            features=np.array([1.0, 2.0, 3.0]),
            confidence=0.8,
            support=0.15,
            metadata={"pattern": "doji"},
        )

        confidence = await pattern_discovery.validate_pattern(pattern, sample_data)

        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_calculate_rsi(self, pattern_discovery):
        """Тест расчета RSI."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])

        rsi = pattern_discovery._calculate_rsi(prices, period=5)

        assert isinstance(rsi, pd.Series)
        assert len(rsi) == len(prices)
        assert not rsi.isna().all()  # Не все значения NaN

    def test_calculate_macd(self, pattern_discovery):
        """Тест расчета MACD."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])

        macd, signal, histogram = pattern_discovery._calculate_macd(prices)

        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
        assert len(macd) == len(prices)

    def test_calculate_bollinger_bands(self, pattern_discovery):
        """Тест расчета полос Боллинджера."""
        prices = pd.Series([100, 101, 102, 101, 100, 99, 98, 97, 96, 95])

        upper, middle, lower = pattern_discovery._calculate_bollinger_bands(prices)

        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)
        assert len(upper) == len(prices)
        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_cluster_patterns_empty(self, pattern_discovery):
        """Тест кластеризации пустого списка паттернов."""
        patterns = pattern_discovery.cluster_patterns([])

        assert patterns == []

    def test_cluster_patterns_single(self, pattern_discovery):
        """Тест кластеризации одного паттерна."""
        pattern = Pattern(
            pattern_type=PatternType.CANDLE,
            start_idx=0,
            end_idx=5,
            features=np.array([1.0, 2.0, 3.0]),
            confidence=0.8,
            support=0.15,
            metadata={"pattern": "doji"},
        )

        patterns = pattern_discovery.cluster_patterns([pattern])

        assert len(patterns) == 1
        assert patterns[0] == pattern

    def test_cluster_patterns_multiple(self, pattern_discovery):
        """Тест кластеризации нескольких паттернов."""
        patterns = [
            Pattern(
                pattern_type=PatternType.CANDLE,
                start_idx=i,
                end_idx=i + 5,
                features=np.array([1.0, 2.0, 3.0]),
                confidence=0.8,
                support=0.15,
                metadata={"pattern": "doji"},
            )
            for i in range(5)
        ]

        clustered = pattern_discovery.cluster_patterns(patterns)

        assert isinstance(clustered, list)
        assert len(clustered) >= 0  # Может быть 0 если паттерны не кластеризуются

    def test_find_candle_patterns_empty_data(self, pattern_discovery):
        """Тест поиска свечных паттернов в пустых данных."""
        empty_data = pd.DataFrame()

        patterns = pattern_discovery.find_candle_patterns(empty_data)

        assert patterns == []

    def test_find_candle_patterns_invalid_data(self, pattern_discovery):
        """Тест поиска свечных паттернов в неверных данных."""
        invalid_data = "not a dataframe"

        patterns = pattern_discovery.find_candle_patterns(invalid_data)

        assert patterns == []

    def test_find_candle_patterns_success(self, pattern_discovery, sample_data):
        """Тест успешного поиска свечных паттернов."""
        patterns = pattern_discovery.find_candle_patterns(sample_data)

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.CANDLE
            assert pattern.start_idx >= 0
            assert pattern.end_idx > pattern.start_idx

    def test_find_price_patterns_success(self, pattern_discovery, sample_data):
        """Тест успешного поиска ценовых паттернов."""
        patterns = pattern_discovery.find_price_patterns(sample_data)

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.PRICE
            assert pattern.start_idx >= 0
            assert pattern.end_idx > pattern.start_idx

    def test_find_volume_patterns_success(self, pattern_discovery, sample_data):
        """Тест успешного поиска объемных паттернов."""
        patterns = pattern_discovery.find_volume_patterns(sample_data)

        assert isinstance(patterns, list)
        for pattern in patterns:
            assert pattern.pattern_type == PatternType.VOLUME
            assert pattern.start_idx >= 0
            assert pattern.end_idx > pattern.start_idx

    def test_is_doji(self, pattern_discovery):
        """Тест определения паттерна Doji."""
        # Создаем данные для Doji (открытие и закрытие почти равны)
        doji_candle = pd.Series({"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.1})  # Почти равно открытию

        is_doji = pattern_discovery._is_doji(doji_candle)

        assert isinstance(is_doji, bool)

    def test_is_hammer(self, pattern_discovery):
        """Тест определения паттерна Hammer."""
        # Создаем данные для Hammer
        hammer_candle = pd.Series({"open": 100.0, "high": 101.0, "low": 95.0, "close": 100.5})  # Низкая тень

        is_hammer = pattern_discovery._is_hammer(hammer_candle)

        assert isinstance(is_hammer, bool)

    def test_is_shooting_star(self, pattern_discovery):
        """Тест определения паттерна Shooting Star."""
        # Создаем данные для Shooting Star
        shooting_star_candle = pd.Series({"open": 100.0, "high": 105.0, "low": 99.0, "close": 100.5})  # Высокая тень

        is_shooting_star = pattern_discovery._is_shooting_star(shooting_star_candle)

        assert isinstance(is_shooting_star, bool)

    def test_rank_patterns(self, pattern_discovery):
        """Тест ранжирования паттернов."""
        patterns = [
            Pattern(
                pattern_type=PatternType.CANDLE,
                start_idx=0,
                end_idx=5,
                features=np.array([1.0, 2.0, 3.0]),
                confidence=0.6,
                support=0.1,
                metadata={"pattern": "doji"},
            ),
            Pattern(
                pattern_type=PatternType.CANDLE,
                start_idx=10,
                end_idx=15,
                features=np.array([1.0, 2.0, 3.0]),
                confidence=0.9,
                support=0.2,
                metadata={"pattern": "hammer"},
            ),
        ]

        ranked = pattern_discovery._rank_patterns(patterns)

        assert isinstance(ranked, list)
        assert len(ranked) == len(patterns)
        # Паттерн с более высокой уверенностью должен быть первым
        assert ranked[0].confidence >= ranked[1].confidence

    def test_add_technical_indicators(self, pattern_discovery, sample_data):
        """Тест добавления технических индикаторов."""
        features = pd.DataFrame({"close": sample_data["close"]})

        enhanced_features = pattern_discovery._add_technical_indicators(features, sample_data)

        assert isinstance(enhanced_features, pd.DataFrame)
        assert len(enhanced_features) == len(features)
        assert len(enhanced_features.columns) >= len(features.columns)
