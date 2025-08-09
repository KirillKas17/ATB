"""
Тесты для PatternMemory
"""

import pytest
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from domain.memory.pattern_memory import (
    PatternMemory,
    PatternSnapshot,
    PatternOutcome,
    PredictionResult,
    MarketFeatures,
    OutcomeType,
    PatternMatcher,
    PatternPredictor,
    SQLitePatternMemoryRepository,
    PatternMemoryConfig,
    PredictionDirection,
)
from domain.intelligence.market_pattern_recognizer import PatternType
from domain.value_objects.timestamp import Timestamp


class TestMarketFeatures:
    """Тесты для MarketFeatures"""

    @pytest.fixture
    def sample_features(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для образцовых рыночных характеристик"""
        return MarketFeatures(
            price=50000.0,
            price_change_1m=0.001,
            price_change_5m=0.005,
            price_change_15m=0.01,
            volatility=0.02,
            volume=1000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.5,
            spread=1.0,
            spread_change=0.1,
            bid_volume=500.0,
            ask_volume=600.0,
            order_book_imbalance=0.1,
            depth_absorption=0.3,
            entropy=0.5,
            gravity=0.7,
            latency=0.001,
            correlation=0.8,
            whale_signal=0.6,
            mm_signal=0.4,
            external_sync=True,
        )

    def test_market_features_creation(self, sample_features) -> None:
        """Тест создания MarketFeatures"""
        assert sample_features.price == 50000.0
        assert sample_features.price_change_1m == 0.001
        assert sample_features.volume == 1000.0
        assert sample_features.spread == 1.0
        assert sample_features.whale_signal == 0.6
        assert sample_features.external_sync is True

    def test_market_features_to_vector(self, sample_features) -> None:
        """Тест преобразования в вектор"""
        vector = sample_features.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 16  # Количество характеристик в векторе
        assert vector[0] == 0.001  # price_change_1m
        assert vector[1] == 0.005  # price_change_5m
        assert vector[15] == 0.4  # mm_signal


class TestPatternSnapshot:
    """Тесты для PatternSnapshot"""

    @pytest.fixture
    def sample_snapshot(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для образцового снимка паттерна"""
        features = MarketFeatures(
            price=50000.0,
            price_change_1m=0.001,
            price_change_5m=0.005,
            price_change_15m=0.01,
            volatility=0.02,
            volume=1000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.5,
            spread=1.0,
            spread_change=0.1,
            bid_volume=500.0,
            ask_volume=600.0,
            order_book_imbalance=0.1,
            depth_absorption=0.3,
            entropy=0.5,
            gravity=0.7,
            latency=0.001,
            correlation=0.8,
            whale_signal=0.6,
            mm_signal=0.4,
            external_sync=True,
        )
        return PatternSnapshot(
            pattern_id="pattern_001",
            timestamp=Timestamp.now(),
            symbol="BTC/USDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.85,
            strength=2.5,
            direction="up",
            features=features,
            metadata={"strategy": "momentum", "timeframe": "1m"},
        )

    def test_pattern_snapshot_creation(self, sample_snapshot) -> None:
        """Тест создания PatternSnapshot"""
        assert sample_snapshot.pattern_id == "pattern_001"
        assert sample_snapshot.symbol == "BTC/USDT"
        assert sample_snapshot.pattern_type == PatternType.WHALE_ABSORPTION
        assert sample_snapshot.confidence == 0.85
        assert sample_snapshot.strength == 2.5
        assert sample_snapshot.direction == "up"
        assert sample_snapshot.metadata["strategy"] == "momentum"

    def test_pattern_snapshot_to_dict(self, sample_snapshot) -> None:
        """Тест преобразования в словарь"""
        result = sample_snapshot.to_dict()
        assert result["pattern_id"] == "pattern_001"
        assert result["symbol"] == "BTC/USDT"
        assert result["pattern_type"] == "whale_absorption"
        assert result["confidence"] == 0.85
        assert result["strength"] == 2.5
        assert result["direction"] == "up"
        assert result["features"]["price"] == 50000.0
        assert result["metadata"]["strategy"] == "momentum"

    def test_pattern_snapshot_from_dict(self, sample_snapshot) -> None:
        """Тест создания из словаря"""
        data = sample_snapshot.to_dict()
        snapshot = PatternSnapshot.from_dict(data)
        assert snapshot.pattern_id == "pattern_001"
        assert snapshot.symbol == "BTC/USDT"
        assert snapshot.pattern_type == PatternType.WHALE_ABSORPTION
        assert snapshot.confidence == 0.85
        assert snapshot.features.price == 50000.0


class TestPatternOutcome:
    """Тесты для PatternOutcome"""

    @pytest.fixture
    def sample_outcome(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для образцового исхода паттерна"""
        return PatternOutcome(
            pattern_id="pattern_001",
            symbol="BTC/USDT",
            outcome_type=OutcomeType.PROFITABLE,
            timestamp=Timestamp.now(),
            price_change_percent=2.5,
            volume_change_percent=15.0,
            duration_minutes=30,
            max_profit_percent=3.2,
            max_loss_percent=-0.8,
            final_return_percent=2.5,
            volatility_during=0.025,
            volume_profile="increasing",
            market_regime="trending",
            metadata={"strategy": "momentum", "risk_level": "medium"},
        )

    def test_pattern_outcome_creation(self, sample_outcome) -> None:
        """Тест создания PatternOutcome"""
        assert sample_outcome.pattern_id == "pattern_001"
        assert sample_outcome.symbol == "BTC/USDT"
        assert sample_outcome.outcome_type == OutcomeType.PROFITABLE
        assert sample_outcome.price_change_percent == 2.5
        assert sample_outcome.volume_change_percent == 15.0
        assert sample_outcome.duration_minutes == 30
        assert sample_outcome.max_profit_percent == 3.2
        assert sample_outcome.volume_profile == "increasing"
        assert sample_outcome.market_regime == "trending"

    def test_pattern_outcome_to_dict(self, sample_outcome) -> None:
        """Тест преобразования в словарь"""
        result = sample_outcome.to_dict()
        assert result["pattern_id"] == "pattern_001"
        assert result["symbol"] == "BTC/USDT"
        assert result["outcome_type"] == "profitable"
        assert result["price_change_percent"] == 2.5
        assert result["volume_change_percent"] == 15.0
        assert result["duration_minutes"] == 30
        assert result["volume_profile"] == "increasing"
        assert result["market_regime"] == "trending"

    def test_pattern_outcome_from_dict(self, sample_outcome) -> None:
        """Тест создания из словаря"""
        data = sample_outcome.to_dict()
        outcome = PatternOutcome.from_dict(data)
        assert outcome.pattern_id == "pattern_001"
        assert outcome.symbol == "BTC/USDT"
        assert outcome.outcome_type == OutcomeType.PROFITABLE
        assert outcome.price_change_percent == 2.5
        assert outcome.volume_profile == "increasing"


class TestPredictionResult:
    """Тесты для PredictionResult"""

    @pytest.fixture
    def sample_prediction(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для образцового прогноза"""
        return PredictionResult(
            pattern_id="pattern_001",
            symbol="BTC/USDT",
            confidence=0.75,
            predicted_direction="up",
            predicted_return_percent=2.0,
            predicted_duration_minutes=25,
            predicted_volatility=0.02,
            similar_cases_count=15,
            success_rate=0.73,
            avg_return=1.8,
            avg_duration=28,
            metadata={"strategy": "momentum", "timeframe": "1m"},
        )

    def test_prediction_result_creation(self, sample_prediction) -> None:
        """Тест создания PredictionResult"""
        assert sample_prediction.pattern_id == "pattern_001"
        assert sample_prediction.symbol == "BTC/USDT"
        assert sample_prediction.confidence == 0.75
        assert sample_prediction.predicted_direction == "up"
        assert sample_prediction.predicted_return_percent == 2.0
        assert sample_prediction.similar_cases_count == 15
        assert sample_prediction.success_rate == 0.73

    def test_prediction_result_to_dict(self, sample_prediction) -> None:
        """Тест преобразования в словарь"""
        result = sample_prediction.to_dict()
        assert result["pattern_id"] == "pattern_001"
        assert result["symbol"] == "BTC/USDT"
        assert result["confidence"] == 0.75
        assert result["predicted_direction"] == "up"
        assert result["predicted_return_percent"] == 2.0
        assert result["similar_cases_count"] == 15
        assert result["success_rate"] == 0.73


class TestPatternMemory:
    """Тесты для PatternMemory."""

    @pytest.fixture
    def pattern_memory(self, tmp_path) -> Any:
        """Создать экземпляр PatternMemory для тестов."""
        return PatternMemory(
            storage_path=tmp_path / "pattern_memory", max_patterns=1000, similarity_threshold=0.8, cleanup_days=30
        )

    @pytest.fixture
    def sample_features(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Образец рыночных признаков."""
        return MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )

    def test_pattern_memory_initialization(self, pattern_memory) -> None:
        """Тест инициализации памяти паттернов."""
        assert pattern_memory is not None
        assert pattern_memory.max_patterns == 1000

    def test_save_snapshot(self, pattern_memory, sample_features) -> None:
        """Тест сохранения снимка паттерна."""
        snapshot = PatternSnapshot(
            pattern_id="test1",
            timestamp=Timestamp.from_datetime(datetime.now()),
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=sample_features,
            metadata={},
        )
        success = pattern_memory.save_snapshot(snapshot)
        assert success
        # Проверяем, что снимок сохранен
        stats = pattern_memory.get_pattern_statistics("BTCUSDT")
        assert stats["total_patterns"] == 1

    def test_save_outcome(self, pattern_memory) -> None:
        """Тест сохранения результата паттерна."""
        outcome = PatternOutcome(
            pattern_id="test1",
            symbol="BTCUSDT",
            outcome_type=OutcomeType.PROFITABLE,
            timestamp=Timestamp.from_datetime(datetime.now()),
            price_change_percent=2.0,
            volume_change_percent=10.0,
            duration_minutes=30,
            max_profit_percent=3.0,
            max_loss_percent=-1.0,
            final_return_percent=2.0,
            volatility_during=0.05,
            volume_profile="increasing",
            market_regime="trending",
            metadata={},
        )
        success = pattern_memory.save_outcome(outcome)
        assert success
        # Проверяем, что результат сохранен
        stats = pattern_memory.get_pattern_statistics("BTCUSDT")
        assert stats["total_outcomes"] == 1

    def test_calculate_similarity(self, pattern_memory) -> None:
        """Тест расчета схожести паттернов."""
        features1 = MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )
        features2 = MarketFeatures(
            price=101.0,
            price_change_1m=0.011,
            price_change_5m=0.021,
            price_change_15m=0.031,
            volatility=0.051,
            volume=1001000.0,
            volume_change_1m=0.101,
            volume_change_5m=0.201,
            volume_sma_ratio=1.01,
            spread=0.0011,
            spread_change=0.00011,
            bid_volume=500500.0,
            ask_volume=500500.0,
            order_book_imbalance=0.001,
            depth_absorption=0.501,
            entropy=0.701,
            gravity=0.301,
            latency=10.1,
            correlation=0.801,
            whale_signal=0.201,
            mm_signal=0.101,
            external_sync=True,
        )
        similarity = pattern_memory.calculate_similarity(features1, features2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.9  # Должны быть очень похожи

    def test_get_statistics(self, pattern_memory) -> None:
        """Тест получения статистики."""
        stats = pattern_memory.get_pattern_statistics("BTCUSDT")
        assert isinstance(stats, dict)
        assert "total_patterns" in stats
        assert "total_outcomes" in stats
        assert "avg_confidence" in stats
        assert "avg_strength" in stats

    def test_save_and_get_pattern_data(self: "TestPatternMemory") -> None:
        """Тест сохранения и получения данных паттерна."""
        features = MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )
        snapshot = PatternSnapshot(
            pattern_id="test1",
            timestamp=Timestamp.from_datetime(datetime.now()),
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=features,
            metadata={"test": "data"},
        )
        # Сохраняем снимок
        success = pattern_memory.save_pattern_data("test1", snapshot)
        assert success
        # Получаем статистику
        stats = pattern_memory.get_pattern_statistics("BTCUSDT")
        assert stats["total_patterns"] == 1
        assert stats["avg_confidence"] == 0.8
        assert stats["avg_strength"] == 0.7

    def test_match_snapshot(self: "TestPatternMemory") -> None:
        """Тест сопоставления снимка."""
        features = MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )
        snapshot = PatternSnapshot(
            pattern_id="test1",
            timestamp=Timestamp.from_datetime(datetime.now()),
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=features,
            metadata={},
        )
        outcome = PatternOutcome(
            pattern_id="test1",
            symbol="BTCUSDT",
            outcome_type=OutcomeType.PROFITABLE,
            timestamp=Timestamp.from_datetime(datetime.now()),
            price_change_percent=2.0,
            volume_change_percent=10.0,
            duration_minutes=30,
            max_profit_percent=3.0,
            max_loss_percent=-1.0,
            final_return_percent=2.0,
            volatility_during=0.05,
            volume_profile="increasing",
            market_regime="trending",
            metadata={},
        )
        # Сохраняем данные
        pattern_memory.save_pattern_data("test1", snapshot)
        pattern_memory.update_pattern_outcome("test1", outcome)
        # Ищем похожие паттерны
        prediction = pattern_memory.match_snapshot(features, "BTCUSDT")
        # Должен найти похожий паттерн
        assert prediction is not None
        assert prediction.symbol == "BTCUSDT"
        assert prediction.similar_cases_count > 0

    def test_get_memory_statistics(self: "TestPatternMemory") -> None:
        """Тест получения статистики памяти."""
        stats = pattern_memory.get_memory_statistics()
        assert isinstance(stats.total_snapshots, int)
        assert isinstance(stats.total_outcomes, int)
        assert isinstance(stats.avg_confidence, float)
        assert isinstance(stats.avg_success_rate, float)

    def test_cleanup_old_patterns(self: "TestPatternMemory") -> None:
        """Тест очистки старых паттернов."""
        # Создаем старый паттерн
        old_timestamp = Timestamp.from_datetime(datetime.now() - timedelta(days=40))
        features = MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )
        old_snapshot = PatternSnapshot(
            pattern_id="old_test",
            timestamp=old_timestamp,
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=features,
            metadata={},
        )
        pattern_memory.save_pattern_data("old_test", old_snapshot)
        # Очищаем старые данные
        deleted_count = pattern_memory.cleanup_old_patterns(days_to_keep=30)
        assert deleted_count > 0
        # Проверяем, что старые данные удалены
        stats = pattern_memory.get_memory_statistics()
        assert stats.total_snapshots == 0


class TestOutcomeType:
    """Тесты для OutcomeType"""

    def test_outcome_type_values(self: "TestOutcomeType") -> None:
        """Тест значений типов исходов"""
        assert OutcomeType.PROFITABLE.value == "profitable"
        assert OutcomeType.UNPROFITABLE.value == "unprofitable"
        assert OutcomeType.NEUTRAL.value == "neutral"
        assert OutcomeType.UNKNOWN.value == "unknown"

    def test_outcome_type_comparison(self: "TestOutcomeType") -> None:
        """Тест сравнения типов исходов"""
        assert OutcomeType.PROFITABLE != OutcomeType.UNPROFITABLE
        assert OutcomeType.PROFITABLE == OutcomeType.PROFITABLE


class TestPatternMatcher:
    """Тесты для PatternMatcher."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = PatternMemoryConfig()
        self.matcher = PatternMatcher(self.config)

    def test_calculate_similarity(self: "TestPatternMatcher") -> None:
        """Тест вычисления сходства."""
        vector1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        vector2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
        similarity = self.matcher.calculate_similarity(vector1, vector2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.8  # Должно быть высокое сходство

    def test_calculate_similarity_identical_vectors(self: "TestPatternMatcher") -> None:
        """Тест сходства идентичных векторов."""
        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        similarity = self.matcher.calculate_similarity(vector, vector)
        assert similarity == 1.0

    def test_calculate_similarity_opposite_vectors(self: "TestPatternMatcher") -> None:
        """Тест сходства противоположных векторов."""
        vector1 = np.array([1.0, 2.0, 3.0])
        vector2 = np.array([-1.0, -2.0, -3.0])
        similarity = self.matcher.calculate_similarity(vector1, vector2)
        assert similarity < 0.5  # Должно быть низкое сходство

    def test_find_similar_patterns(self: "TestPatternMatcher") -> None:
        """Тест поиска похожих паттернов."""
        current_features = MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )
        snapshots = [
            PatternSnapshot(
                pattern_id="test1",
                timestamp=Timestamp.from_datetime(datetime.now()),
                symbol="BTCUSDT",
                pattern_type=PatternType.WHALE_ABSORPTION,
                confidence=0.8,
                strength=0.7,
                direction="up",
                features=current_features,
                metadata={},
            )
        ]
        similar_patterns = self.matcher.find_similar_patterns(current_features, snapshots, similarity_threshold=0.5)
        assert len(similar_patterns) == 1
        assert similar_patterns[0][1] > 0.5  # Сходство должно быть выше порога

    def test_calculate_confidence_boost(self: "TestPatternMatcher") -> None:
        """Тест вычисления повышения уверенности."""
        snapshot = PatternSnapshot(
            pattern_id="test",
            timestamp=Timestamp.from_datetime(datetime.now()),
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=Mock(),
            metadata={},
        )
        boost = self.matcher.calculate_confidence_boost(0.9, snapshot)
        assert 0.0 <= boost <= 1.0
        assert boost > 0.5  # Должно быть значительное повышение

    def test_calculate_signal_strength(self: "TestPatternMatcher") -> None:
        """Тест вычисления силы сигнала."""
        snapshot = PatternSnapshot(
            pattern_id="test",
            timestamp=Timestamp.from_datetime(datetime.now()),
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=Mock(),
            metadata={},
        )
        strength = self.matcher.calculate_signal_strength(snapshot)
        assert -1.0 <= strength <= 1.0
        assert strength > 0.0  # Должен быть положительный сигнал


class TestPatternPredictor:
    """Тесты для PatternPredictor."""

    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = PatternMemoryConfig()
        self.predictor = PatternPredictor(self.config)

    def test_calculate_predicted_return(self: "TestPatternPredictor") -> None:
        """Тест вычисления прогнозируемой доходности."""
        outcomes = [
            PatternOutcome(
                pattern_id="test1",
                symbol="BTCUSDT",
                outcome_type=OutcomeType.PROFITABLE,
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=2.0,
                volume_change_percent=10.0,
                duration_minutes=30,
                max_profit_percent=3.0,
                max_loss_percent=-1.0,
                final_return_percent=2.0,
                volatility_during=0.05,
                volume_profile="increasing",
                market_regime="trending",
                metadata={},
            ),
            PatternOutcome(
                pattern_id="test2",
                symbol="BTCUSDT",
                outcome_type=OutcomeType.PROFITABLE,
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=1.0,
                volume_change_percent=5.0,
                duration_minutes=20,
                max_profit_percent=2.0,
                max_loss_percent=-0.5,
                final_return_percent=1.0,
                volatility_during=0.03,
                volume_profile="stable",
                market_regime="trending",
                metadata={},
            ),
        ]
        predicted_return = self.predictor.calculate_predicted_return(outcomes)
        assert predicted_return == 1.5  # Среднее значение

    def test_calculate_predicted_duration(self: "TestPatternPredictor") -> None:
        """Тест вычисления прогнозируемой длительности."""
        outcomes = [
            PatternOutcome(
                pattern_id="test1",
                symbol="BTCUSDT",
                outcome_type=OutcomeType.PROFITABLE,
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=2.0,
                volume_change_percent=10.0,
                duration_minutes=30,
                max_profit_percent=3.0,
                max_loss_percent=-1.0,
                final_return_percent=2.0,
                volatility_during=0.05,
                volume_profile="increasing",
                market_regime="trending",
                metadata={},
            ),
            PatternOutcome(
                pattern_id="test2",
                symbol="BTCUSDT",
                outcome_type=OutcomeType.PROFITABLE,
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=1.0,
                volume_change_percent=5.0,
                duration_minutes=20,
                max_profit_percent=2.0,
                max_loss_percent=-0.5,
                final_return_percent=1.0,
                volatility_during=0.03,
                volume_profile="stable",
                market_regime="trending",
                metadata={},
            ),
        ]
        predicted_duration = self.predictor.calculate_predicted_duration(outcomes)
        assert predicted_duration == 25  # Среднее значение

    def test_calculate_success_rate(self: "TestPatternPredictor") -> None:
        """Тест вычисления успешности."""
        outcomes = [
            PatternOutcome(
                pattern_id="test1",
                symbol="BTCUSDT",
                outcome_type=OutcomeType.PROFITABLE,
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=2.0,
                volume_change_percent=10.0,
                duration_minutes=30,
                max_profit_percent=3.0,
                max_loss_percent=-1.0,
                final_return_percent=2.0,
                volatility_during=0.05,
                volume_profile="increasing",
                market_regime="trending",
                metadata={},
            ),
            PatternOutcome(
                pattern_id="test2",
                symbol="BTCUSDT",
                outcome_type=OutcomeType.UNPROFITABLE,
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=-1.0,
                volume_change_percent=-5.0,
                duration_minutes=20,
                max_profit_percent=0.5,
                max_loss_percent=-2.0,
                final_return_percent=-1.0,
                volatility_during=0.03,
                volume_profile="decreasing",
                market_regime="volatile",
                metadata={},
            ),
        ]
        success_rate = self.predictor._calculate_success_rate(outcomes)
        assert success_rate == 0.5  # 50% успешность

    def test_generate_prediction(self: "TestPatternPredictor") -> None:
        """Тест генерации прогноза."""
        features = MarketFeatures(
            price=100.0,
            price_change_1m=0.01,
            price_change_5m=0.02,
            price_change_15m=0.03,
            volatility=0.05,
            volume=1000000.0,
            volume_change_1m=0.1,
            volume_change_5m=0.2,
            volume_sma_ratio=1.0,
            spread=0.001,
            spread_change=0.0001,
            bid_volume=500000.0,
            ask_volume=500000.0,
            order_book_imbalance=0.0,
            depth_absorption=0.5,
            entropy=0.7,
            gravity=0.3,
            latency=10.0,
            correlation=0.8,
            whale_signal=0.2,
            mm_signal=0.1,
            external_sync=True,
        )
        snapshot = PatternSnapshot(
            pattern_id="test1",
            timestamp=Timestamp.from_datetime(datetime.now()),
            symbol="BTCUSDT",
            pattern_type=PatternType.WHALE_ABSORPTION,
            confidence=0.8,
            strength=0.7,
            direction="up",
            features=features,
            metadata={},
        )
        outcome = PatternOutcome(
            pattern_id="test1",
            symbol="BTCUSDT",
            outcome_type=OutcomeType.PROFITABLE,
            timestamp=Timestamp.from_datetime(datetime.now()),
            price_change_percent=2.0,
            volume_change_percent=10.0,
            duration_minutes=30,
            max_profit_percent=3.0,
            max_loss_percent=-1.0,
            final_return_percent=2.0,
            volatility_during=0.05,
            volume_profile="increasing",
            market_regime="trending",
            metadata={},
        )
        similar_cases = [(snapshot, 0.9)]
        outcomes = [outcome]
        prediction = self.predictor.generate_prediction(similar_cases, outcomes, features, "BTCUSDT")
        assert prediction is not None
        assert prediction.symbol == "BTCUSDT"
        assert prediction.confidence > 0.0
        assert prediction.similar_cases_count == 1
        assert prediction.success_rate == 1.0
