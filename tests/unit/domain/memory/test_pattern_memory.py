"""
Unit тесты для pattern_memory.py.

Покрывает:
- PatternMatcher - сопоставление паттернов
- PatternPredictor - предсказание паттернов
- SQLitePatternMemoryRepository - репозиторий паттернов
- PatternMemory - память паттернов
- Вспомогательные функции
"""

import pytest
import asyncio
import sqlite3
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from collections import deque

from domain.memory.pattern_memory import (
    PatternMatcher,
    PatternPredictor,
    SQLitePatternMemoryRepository,
    PatternMemory,
    _calculate_cosine_similarity,
    _calculate_euclidean_similarity,
    _calculate_similarity_score,
)
from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternFeatures,
    PatternResult,
    PatternMemory as MMPatternMemory,
)
from domain.type_definitions.market_maker_types import (
    MarketMakerPatternType,
    PatternOutcome,
    BookPressure,
    VolumeDelta,
    PriceReaction,
    SpreadChange,
    OrderImbalance,
    LiquidityDepth,
    TimeDuration,
    VolumeConcentration,
    PriceVolatility,
    Confidence,
    Accuracy,
    AverageReturn,
    SuccessCount,
    TotalCount,
    SimilarityScore,
    SignalStrength,
)


class TestPatternMatcher:
    """Тесты для PatternMatcher."""

    @pytest.fixture
    def pattern_matcher(self):
        """Создает экземпляр PatternMatcher."""
        return PatternMatcher()

    @pytest.fixture
    def sample_pattern(self) -> MarketMakerPattern:
        """Создает тестовый паттерн."""
        return MarketMakerPattern(
            id="test_pattern_001",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.7),
                volume_delta=VolumeDelta(0.5),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.6),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.8),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.85),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

    @pytest.fixture
    def sample_pattern_memory(self) -> MMPatternMemory:
        """Создает тестовую память паттерна."""
        return MMPatternMemory(
            pattern_id="test_pattern_001",
            total_count=TotalCount(10),
            success_count=SuccessCount(7),
            accuracy=Accuracy(0.7),
            average_return=AverageReturn(0.05),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 7 + [PatternOutcome.FAILURE] * 3, maxlen=20),
        )

    def test_pattern_matcher_creation(self, pattern_matcher):
        """Тест создания PatternMatcher."""
        assert pattern_matcher is not None
        assert hasattr(pattern_matcher, "find_similar_patterns")

    def test_find_similar_patterns_empty(self, pattern_matcher, sample_pattern):
        """Тест поиска похожих паттернов в пустом списке."""
        patterns = []
        memory_map = {}

        result = pattern_matcher.find_similar_patterns(
            target_pattern=sample_pattern, patterns=patterns, memory_map=memory_map, similarity_threshold=0.8
        )

        assert result == []

    def test_find_similar_patterns_exact_match(self, pattern_matcher, sample_pattern):
        """Тест поиска точного совпадения паттерна."""
        patterns = [sample_pattern]
        memory_map = {
            sample_pattern.id: MMPatternMemory(
                pattern_id=sample_pattern.id,
                total_count=TotalCount(5),
                success_count=SuccessCount(4),
                accuracy=Accuracy(0.8),
                average_return=AverageReturn(0.03),
                last_seen=datetime.now(),
                behavior_history=deque([PatternOutcome.SUCCESS] * 4 + [PatternOutcome.FAILURE], maxlen=20),
            )
        }

        result = pattern_matcher.find_similar_patterns(
            target_pattern=sample_pattern, patterns=patterns, memory_map=memory_map, similarity_threshold=0.5
        )

        assert len(result) == 1
        assert result[0].pattern.id == sample_pattern.id
        assert result[0].similarity_score > 0.99  # Почти точное совпадение

    def test_find_similar_patterns_with_threshold(self, pattern_matcher):
        """Тест поиска паттернов с порогом схожести."""
        # Создаем два похожих паттерна
        pattern1 = MarketMakerPattern(
            id="pattern_1",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.7),
                volume_delta=VolumeDelta(0.5),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.6),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.8),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.85),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        pattern2 = MarketMakerPattern(
            id="pattern_2",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.2),  # Очень разные признаки
                volume_delta=VolumeDelta(0.1),
                price_reaction=PriceReaction(0.8),
                spread_change=SpreadChange(0.9),
                order_imbalance=OrderImbalance(0.1),
                liquidity_depth=LiquidityDepth(0.9),
                time_duration=TimeDuration(100),
                volume_concentration=VolumeConcentration(0.1),
                price_volatility=PriceVolatility(0.9),
            ),
            confidence=Confidence(0.75),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        patterns = [pattern1, pattern2]
        memory_map = {
            pattern1.id: MMPatternMemory(
                pattern_id=pattern1.id,
                total_count=TotalCount(5),
                success_count=SuccessCount(4),
                accuracy=Accuracy(0.8),
                average_return=AverageReturn(0.03),
                last_seen=datetime.now(),
                behavior_history=deque([PatternOutcome.SUCCESS] * 4 + [PatternOutcome.FAILURE], maxlen=20),
            ),
            pattern2.id: MMPatternMemory(
                pattern_id=pattern2.id,
                total_count=TotalCount(3),
                success_count=SuccessCount(1),
                accuracy=Accuracy(0.33),
                average_return=AverageReturn(-0.02),
                last_seen=datetime.now(),
                behavior_history=deque([PatternOutcome.SUCCESS] + [PatternOutcome.FAILURE] * 2, maxlen=20),
            ),
        }

        # Ищем паттерны похожие на pattern1
        result = pattern_matcher.find_similar_patterns(
            target_pattern=pattern1, patterns=patterns, memory_map=memory_map, similarity_threshold=0.7
        )

        # Должен найти только pattern1 (точное совпадение)
        assert len(result) == 1
        assert result[0].pattern.id == pattern1.id

    def test_find_similar_patterns_combined_confidence(self, pattern_matcher, sample_pattern):
        """Тест расчета комбинированной уверенности."""
        patterns = [sample_pattern]
        memory_map = {
            sample_pattern.id: MMPatternMemory(
                pattern_id=sample_pattern.id,
                total_count=TotalCount(10),
                success_count=SuccessCount(8),
                accuracy=Accuracy(0.8),
                average_return=AverageReturn(0.05),
                last_seen=datetime.now(),
                behavior_history=deque([PatternOutcome.SUCCESS] * 8 + [PatternOutcome.FAILURE] * 2, maxlen=20),
            )
        }

        result = pattern_matcher.find_similar_patterns(
            target_pattern=sample_pattern, patterns=patterns, memory_map=memory_map, similarity_threshold=0.5
        )

        assert len(result) == 1
        matched_pattern = result[0]

        # Проверяем, что комбинированная уверенность учитывает схожесть и историю
        assert matched_pattern.combined_confidence > 0
        assert matched_pattern.combined_confidence <= 1.0

    def test_find_similar_patterns_trading_signal(self, pattern_matcher, sample_pattern):
        """Тест генерации торгового сигнала."""
        patterns = [sample_pattern]
        memory_map = {
            sample_pattern.id: MMPatternMemory(
                pattern_id=sample_pattern.id,
                total_count=TotalCount(15),
                success_count=SuccessCount(12),
                accuracy=Accuracy(0.8),
                average_return=AverageReturn(0.06),
                last_seen=datetime.now(),
                behavior_history=deque([PatternOutcome.SUCCESS] * 12 + [PatternOutcome.FAILURE] * 3, maxlen=20),
            )
        }

        result = pattern_matcher.find_similar_patterns(
            target_pattern=sample_pattern, patterns=patterns, memory_map=memory_map, similarity_threshold=0.5
        )

        assert len(result) == 1
        matched_pattern = result[0]

        # Проверяем торговый сигнал
        assert matched_pattern.trading_signal is not None
        assert matched_pattern.trading_signal.signal_strength > 0
        assert matched_pattern.trading_signal.confidence > 0


class TestPatternPredictor:
    """Тесты для PatternPredictor."""

    @pytest.fixture
    def pattern_predictor(self):
        """Создает экземпляр PatternPredictor."""
        return PatternPredictor()

    @pytest.fixture
    def sample_matched_patterns(self):
        """Создает тестовые совпадения паттернов."""
        from domain.market_maker.mm_pattern import MatchedPattern
        from domain.market_maker.mm_pattern_memory import MatchedPattern as MM_MatchedPattern

        pattern1 = MarketMakerPattern(
            id="pattern_1",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.7),
                volume_delta=VolumeDelta(0.5),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.6),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.8),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.85),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        pattern2 = MarketMakerPattern(
            id="pattern_2",
            pattern_type=MarketMakerPatternType.STOP_HUNT,
            features=PatternFeatures(
                book_pressure=BookPressure(0.6),
                volume_delta=VolumeDelta(0.4),
                price_reaction=PriceReaction(0.4),
                spread_change=SpreadChange(0.3),
                order_imbalance=OrderImbalance(0.5),
                liquidity_depth=LiquidityDepth(0.3),
                time_duration=TimeDuration(250),
                volume_concentration=VolumeConcentration(0.7),
                price_volatility=PriceVolatility(0.2),
            ),
            confidence=Confidence(0.75),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        return [
            MatchedPattern(
                pattern=pattern1,
                similarity_score=SimilarityScore(0.95),
                combined_confidence=Confidence(0.82),
                trading_signal=Mock(signal_strength=SignalStrength(0.8), confidence=Confidence(0.82)),
            ),
            MatchedPattern(
                pattern=pattern2,
                similarity_score=SimilarityScore(0.85),
                combined_confidence=Confidence(0.78),
                trading_signal=Mock(signal_strength=SignalStrength(0.7), confidence=Confidence(0.78)),
            ),
        ]

    def test_pattern_predictor_creation(self, pattern_predictor):
        """Тест создания PatternPredictor."""
        assert pattern_predictor is not None
        assert hasattr(pattern_predictor, "generate_prediction")

    def test_generate_prediction_empty_list(self, pattern_predictor):
        """Тест генерации предсказания для пустого списка."""
        result = pattern_predictor.generate_prediction([])

        assert result is None

    def test_generate_prediction_single_pattern(self, pattern_predictor, sample_matched_patterns):
        """Тест генерации предсказания для одного паттерна."""
        single_pattern = [sample_matched_patterns[0]]

        result = pattern_predictor.generate_prediction(single_pattern)

        assert result is not None
        assert result.predicted_outcome in [PatternOutcome.SUCCESS, PatternOutcome.FAILURE]
        assert result.confidence > 0
        assert result.confidence <= 1.0
        assert result.signal_strength > 0
        assert result.signal_strength <= 1.0

    def test_generate_prediction_multiple_patterns(self, pattern_predictor, sample_matched_patterns):
        """Тест генерации предсказания для нескольких паттернов."""
        result = pattern_predictor.generate_prediction(sample_matched_patterns)

        assert result is not None
        assert result.predicted_outcome in [PatternOutcome.SUCCESS, PatternOutcome.FAILURE]
        assert result.confidence > 0
        assert result.confidence <= 1.0
        assert result.signal_strength > 0
        assert result.signal_strength <= 1.0

    def test_generate_prediction_confidence_weighting(self, pattern_predictor):
        """Тест взвешивания по уверенности."""
        # Создаем паттерны с разной уверенностью
        high_confidence_pattern = MarketMakerPattern(
            id="high_conf",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.8),
                volume_delta=VolumeDelta(0.6),
                price_reaction=PriceReaction(0.4),
                spread_change=SpreadChange(0.3),
                order_imbalance=OrderImbalance(0.7),
                liquidity_depth=LiquidityDepth(0.5),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.9),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.95),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        low_confidence_pattern = MarketMakerPattern(
            id="low_conf",
            pattern_type=MarketMakerPatternType.STOP_HUNT,
            features=PatternFeatures(
                book_pressure=BookPressure(0.3),
                volume_delta=VolumeDelta(0.2),
                price_reaction=PriceReaction(0.6),
                spread_change=SpreadChange(0.7),
                order_imbalance=OrderImbalance(0.2),
                liquidity_depth=LiquidityDepth(0.8),
                time_duration=TimeDuration(150),
                volume_concentration=VolumeConcentration(0.3),
                price_volatility=PriceVolatility(0.8),
            ),
            confidence=Confidence(0.45),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        patterns = [
            MatchedPattern(
                pattern=high_confidence_pattern,
                similarity_score=SimilarityScore(0.9),
                combined_confidence=Confidence(0.9),
                trading_signal=Mock(signal_strength=SignalStrength(0.9), confidence=Confidence(0.9)),
            ),
            MatchedPattern(
                pattern=low_confidence_pattern,
                similarity_score=SimilarityScore(0.6),
                combined_confidence=Confidence(0.4),
                trading_signal=Mock(signal_strength=SignalStrength(0.4), confidence=Confidence(0.4)),
            ),
        ]

        result = pattern_predictor.generate_prediction(patterns)

        # Высокоуверенный паттерн должен иметь большее влияние
        assert result is not None
        assert result.confidence > 0.6  # Должен быть ближе к высокой уверенности

    def test_generate_prediction_signal_strength_calculation(self, pattern_predictor, sample_matched_patterns):
        """Тест расчета силы сигнала."""
        result = pattern_predictor.generate_prediction(sample_matched_patterns)

        assert result is not None
        # Сила сигнала должна быть средневзвешенной от всех паттернов
        assert result.signal_strength > 0
        assert result.signal_strength <= 1.0

    def test_generate_prediction_outcome_determination(self, pattern_predictor):
        """Тест определения исхода предсказания."""
        # Создаем паттерны с явными исходами
        success_pattern = MarketMakerPattern(
            id="success_pattern",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.8),
                volume_delta=VolumeDelta(0.7),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.8),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.9),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.9),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        patterns = [
            MatchedPattern(
                pattern=success_pattern,
                similarity_score=SimilarityScore(0.95),
                combined_confidence=Confidence(0.9),
                trading_signal=Mock(signal_strength=SignalStrength(0.9), confidence=Confidence(0.9)),
            )
        ]

        result = pattern_predictor.generate_prediction(patterns)

        assert result is not None
        # При высокой уверенности и сильном сигнале должен быть SUCCESS
        if result.confidence > 0.8 and result.signal_strength > 0.8:
            assert result.predicted_outcome == PatternOutcome.SUCCESS


class TestSQLitePatternMemoryRepository:
    """Тесты для SQLitePatternMemoryRepository."""

    @pytest.fixture
    def temp_db_path(self):
        """Создает временный путь для базы данных."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            return tmp.name

    @pytest.fixture
    async def repository(self, temp_db_path):
        """Создает экземпляр репозитория с временной БД."""
        repo = SQLitePatternMemoryRepository(db_path=temp_db_path)
        await repo.initialize()
        yield repo
        # Очистка после тестов
        await repo.close()
        try:
            Path(temp_db_path).unlink()
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_pattern(self) -> MarketMakerPattern:
        """Создает тестовый паттерн."""
        return MarketMakerPattern(
            id="test_pattern_001",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.7),
                volume_delta=VolumeDelta(0.5),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.6),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.8),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.85),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

    @pytest.fixture
    def sample_pattern_memory(self) -> MMPatternMemory:
        """Создает тестовую память паттерна."""
        return MMPatternMemory(
            pattern_id="test_pattern_001",
            total_count=TotalCount(10),
            success_count=SuccessCount(7),
            accuracy=Accuracy(0.7),
            average_return=AverageReturn(0.05),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 7 + [PatternOutcome.FAILURE] * 3, maxlen=20),
        )

    @pytest.mark.asyncio
    async def test_repository_creation(self, temp_db_path):
        """Тест создания репозитория."""
        repo = SQLitePatternMemoryRepository(db_path=temp_db_path)
        assert repo is not None
        assert repo.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_initialize_database(self, repository):
        """Тест инициализации базы данных."""
        # Проверяем, что таблицы созданы
        async with repository._get_connection() as conn:
            cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = await cursor.fetchall()
            table_names = [table[0] for table in tables]

            assert "patterns" in table_names
            assert "pattern_memory" in table_names

    @pytest.mark.asyncio
    async def test_save_pattern(self, repository, sample_pattern):
        """Тест сохранения паттерна."""
        await repository.save_pattern(sample_pattern)

        # Проверяем, что паттерн сохранен
        saved_pattern = await repository.get_pattern(sample_pattern.id)
        assert saved_pattern is not None
        assert saved_pattern.id == sample_pattern.id
        assert saved_pattern.pattern_type == sample_pattern.pattern_type
        assert saved_pattern.symbol == sample_pattern.symbol

    @pytest.mark.asyncio
    async def test_get_pattern_not_found(self, repository):
        """Тест получения несуществующего паттерна."""
        result = await repository.get_pattern("non_existent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_pattern_memory(self, repository, sample_pattern_memory):
        """Тест сохранения памяти паттерна."""
        await repository.save_pattern_memory(sample_pattern_memory)

        # Проверяем, что память сохранена
        saved_memory = await repository.get_pattern_memory(sample_pattern_memory.pattern_id)
        assert saved_memory is not None
        assert saved_memory.pattern_id == sample_pattern_memory.pattern_id
        assert saved_memory.total_count == sample_pattern_memory.total_count
        assert saved_memory.success_count == sample_pattern_memory.success_count

    @pytest.mark.asyncio
    async def test_get_pattern_memory_not_found(self, repository):
        """Тест получения несуществующей памяти паттерна."""
        result = await repository.get_pattern_memory("non_existent_id")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all_patterns(self, repository, sample_pattern):
        """Тест получения всех паттернов."""
        await repository.save_pattern(sample_pattern)

        patterns = await repository.get_all_patterns()
        assert len(patterns) == 1
        assert patterns[0].id == sample_pattern.id

    @pytest.mark.asyncio
    async def test_get_all_pattern_memory(self, repository, sample_pattern_memory):
        """Тест получения всей памяти паттернов."""
        await repository.save_pattern_memory(sample_pattern_memory)

        memory_list = await repository.get_all_pattern_memory()
        assert len(memory_list) == 1
        assert memory_list[0].pattern_id == sample_pattern_memory.pattern_id

    @pytest.mark.asyncio
    async def test_update_pattern_memory(self, repository, sample_pattern_memory):
        """Тест обновления памяти паттерна."""
        # Сохраняем исходную память
        await repository.save_pattern_memory(sample_pattern_memory)

        # Обновляем память
        updated_memory = MMPatternMemory(
            pattern_id=sample_pattern_memory.pattern_id,
            total_count=TotalCount(15),
            success_count=SuccessCount(12),
            accuracy=Accuracy(0.8),
            average_return=AverageReturn(0.06),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 12 + [PatternOutcome.FAILURE] * 3, maxlen=20),
        )

        await repository.save_pattern_memory(updated_memory)

        # Проверяем обновление
        saved_memory = await repository.get_pattern_memory(sample_pattern_memory.pattern_id)
        assert saved_memory.total_count == updated_memory.total_count
        assert saved_memory.success_count == updated_memory.success_count
        assert saved_memory.accuracy == updated_memory.accuracy

    @pytest.mark.asyncio
    async def test_delete_pattern(self, repository, sample_pattern):
        """Тест удаления паттерна."""
        await repository.save_pattern(sample_pattern)

        # Проверяем, что паттерн сохранен
        saved_pattern = await repository.get_pattern(sample_pattern.id)
        assert saved_pattern is not None

        # Удаляем паттерн
        await repository.delete_pattern(sample_pattern.id)

        # Проверяем, что паттерн удален
        deleted_pattern = await repository.get_pattern(sample_pattern.id)
        assert deleted_pattern is None

    @pytest.mark.asyncio
    async def test_delete_pattern_memory(self, repository, sample_pattern_memory):
        """Тест удаления памяти паттерна."""
        await repository.save_pattern_memory(sample_pattern_memory)

        # Проверяем, что память сохранена
        saved_memory = await repository.get_pattern_memory(sample_pattern_memory.pattern_id)
        assert saved_memory is not None

        # Удаляем память
        await repository.delete_pattern_memory(sample_pattern_memory.pattern_id)

        # Проверяем, что память удалена
        deleted_memory = await repository.get_pattern_memory(sample_pattern_memory.pattern_id)
        assert deleted_memory is None

    @pytest.mark.asyncio
    async def test_get_patterns_by_symbol(self, repository):
        """Тест получения паттернов по символу."""
        pattern1 = MarketMakerPattern(
            id="btc_pattern_1",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.7),
                volume_delta=VolumeDelta(0.5),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.6),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.8),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.85),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        pattern2 = MarketMakerPattern(
            id="eth_pattern_1",
            pattern_type=MarketMakerPatternType.STOP_HUNT,
            features=PatternFeatures(
                book_pressure=BookPressure(0.6),
                volume_delta=VolumeDelta(0.4),
                price_reaction=PriceReaction(0.4),
                spread_change=SpreadChange(0.3),
                order_imbalance=OrderImbalance(0.5),
                liquidity_depth=LiquidityDepth(0.3),
                time_duration=TimeDuration(250),
                volume_concentration=VolumeConcentration(0.7),
                price_volatility=PriceVolatility(0.2),
            ),
            confidence=Confidence(0.75),
            timestamp=datetime.now(),
            symbol="ETH/USDT",
        )

        await repository.save_pattern(pattern1)
        await repository.save_pattern(pattern2)

        # Получаем паттерны BTC
        btc_patterns = await repository.get_patterns_by_symbol("BTC/USDT")
        assert len(btc_patterns) == 1
        assert btc_patterns[0].symbol == "BTC/USDT"

        # Получаем паттерны ETH
        eth_patterns = await repository.get_patterns_by_symbol("ETH/USDT")
        assert len(eth_patterns) == 1
        assert eth_patterns[0].symbol == "ETH/USDT"

    @pytest.mark.asyncio
    async def test_get_successful_patterns(self, repository):
        """Тест получения успешных паттернов."""
        # Создаем паттерны с разной успешностью
        successful_memory = MMPatternMemory(
            pattern_id="successful_pattern",
            total_count=TotalCount(10),
            success_count=SuccessCount(9),
            accuracy=Accuracy(0.9),
            average_return=AverageReturn(0.08),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 9 + [PatternOutcome.FAILURE], maxlen=20),
        )

        unsuccessful_memory = MMPatternMemory(
            pattern_id="unsuccessful_pattern",
            total_count=TotalCount(10),
            success_count=SuccessCount(3),
            accuracy=Accuracy(0.3),
            average_return=AverageReturn(-0.02),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 3 + [PatternOutcome.FAILURE] * 7, maxlen=20),
        )

        await repository.save_pattern_memory(successful_memory)
        await repository.save_pattern_memory(unsuccessful_memory)

        # Получаем успешные паттерны (accuracy > 0.7)
        successful_patterns = await repository.get_successful_patterns(min_accuracy=0.7)
        assert len(successful_patterns) == 1
        assert successful_patterns[0].pattern_id == "successful_pattern"

    @pytest.mark.asyncio
    async def test_close_connection(self, repository):
        """Тест закрытия соединения."""
        await repository.close()
        # Проверяем, что соединение закрыто
        assert repository._connection_pool is None


class TestPatternMemory:
    """Тесты для PatternMemory."""

    @pytest.fixture
    def temp_db_path(self):
        """Создает временный путь для базы данных."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            return tmp.name

    @pytest.fixture
    async def pattern_memory(self, temp_db_path):
        """Создает экземпляр PatternMemory с временной БД."""
        memory = PatternMemory(db_path=temp_db_path)
        await memory.initialize()
        yield memory
        # Очистка после тестов
        await memory.close()
        try:
            Path(temp_db_path).unlink()
        except FileNotFoundError:
            pass

    @pytest.fixture
    def sample_pattern(self) -> MarketMakerPattern:
        """Создает тестовый паттерн."""
        return MarketMakerPattern(
            id="test_pattern_001",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.7),
                volume_delta=VolumeDelta(0.5),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.6),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.8),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.85),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

    @pytest.mark.asyncio
    async def test_pattern_memory_creation(self, temp_db_path):
        """Тест создания PatternMemory."""
        memory = PatternMemory(db_path=temp_db_path)
        assert memory is not None
        assert memory.db_path == temp_db_path

    @pytest.mark.asyncio
    async def test_initialize(self, pattern_memory):
        """Тест инициализации PatternMemory."""
        # Проверяем, что компоненты инициализированы
        assert pattern_memory.matcher is not None
        assert pattern_memory.predictor is not None
        assert pattern_memory.repository is not None

    @pytest.mark.asyncio
    async def test_save_pattern(self, pattern_memory, sample_pattern):
        """Тест сохранения паттерна."""
        await pattern_memory.save_pattern(sample_pattern)

        # Проверяем, что паттерн сохранен
        saved_pattern = await pattern_memory.repository.get_pattern(sample_pattern.id)
        assert saved_pattern is not None
        assert saved_pattern.id == sample_pattern.id

    @pytest.mark.asyncio
    async def test_find_similar_patterns(self, pattern_memory, sample_pattern):
        """Тест поиска похожих паттернов."""
        # Сохраняем паттерн
        await pattern_memory.save_pattern(sample_pattern)

        # Создаем память паттерна
        pattern_memory_data = MMPatternMemory(
            pattern_id=sample_pattern.id,
            total_count=TotalCount(10),
            success_count=SuccessCount(7),
            accuracy=Accuracy(0.7),
            average_return=AverageReturn(0.05),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 7 + [PatternOutcome.FAILURE] * 3, maxlen=20),
        )
        await pattern_memory.repository.save_pattern_memory(pattern_memory_data)

        # Ищем похожие паттерны
        similar_patterns = await pattern_memory.find_similar_patterns(
            target_pattern=sample_pattern, similarity_threshold=0.5
        )

        assert len(similar_patterns) == 1
        assert similar_patterns[0].pattern.id == sample_pattern.id

    @pytest.mark.asyncio
    async def test_generate_prediction(self, pattern_memory, sample_pattern):
        """Тест генерации предсказания."""
        # Сохраняем паттерн и его память
        await pattern_memory.save_pattern(sample_pattern)
        pattern_memory_data = MMPatternMemory(
            pattern_id=sample_pattern.id,
            total_count=TotalCount(15),
            success_count=SuccessCount(12),
            accuracy=Accuracy(0.8),
            average_return=AverageReturn(0.06),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 12 + [PatternOutcome.FAILURE] * 3, maxlen=20),
        )
        await pattern_memory.repository.save_pattern_memory(pattern_memory_data)

        # Генерируем предсказание
        prediction = await pattern_memory.generate_prediction(target_pattern=sample_pattern, similarity_threshold=0.5)

        assert prediction is not None
        assert prediction.predicted_outcome in [PatternOutcome.SUCCESS, PatternOutcome.FAILURE]
        assert prediction.confidence > 0
        assert prediction.signal_strength > 0

    @pytest.mark.asyncio
    async def test_update_pattern_result(self, pattern_memory, sample_pattern):
        """Тест обновления результата паттерна."""
        # Сохраняем паттерн и его память
        await pattern_memory.save_pattern(sample_pattern)
        pattern_memory_data = MMPatternMemory(
            pattern_id=sample_pattern.id,
            total_count=TotalCount(10),
            success_count=SuccessCount(7),
            accuracy=Accuracy(0.7),
            average_return=AverageReturn(0.05),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 7 + [PatternOutcome.FAILURE] * 3, maxlen=20),
        )
        await pattern_memory.repository.save_pattern_memory(pattern_memory_data)

        # Обновляем результат
        result = PatternResult(
            outcome=PatternOutcome.SUCCESS, return_value=0.03, duration=timedelta(minutes=5), confidence_boost=0.1
        )

        await pattern_memory.update_pattern_result(sample_pattern.id, result)

        # Проверяем обновление
        updated_memory = await pattern_memory.repository.get_pattern_memory(sample_pattern.id)
        assert updated_memory.total_count == TotalCount(11)
        assert updated_memory.success_count == SuccessCount(8)

    @pytest.mark.asyncio
    async def test_get_pattern_statistics(self, pattern_memory, sample_pattern):
        """Тест получения статистики паттерна."""
        # Сохраняем паттерн и его память
        await pattern_memory.save_pattern(sample_pattern)
        pattern_memory_data = MMPatternMemory(
            pattern_id=sample_pattern.id,
            total_count=TotalCount(20),
            success_count=SuccessCount(16),
            accuracy=Accuracy(0.8),
            average_return=AverageReturn(0.06),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 16 + [PatternOutcome.FAILURE] * 4, maxlen=20),
        )
        await pattern_memory.repository.save_pattern_memory(pattern_memory_data)

        # Получаем статистику
        stats = await pattern_memory.get_pattern_statistics(sample_pattern.id)

        assert stats is not None
        assert stats.total_count == 20
        assert stats.success_count == 16
        assert stats.accuracy == 0.8
        assert stats.average_return == 0.06

    @pytest.mark.asyncio
    async def test_get_successful_patterns(self, pattern_memory):
        """Тест получения успешных паттернов."""
        # Создаем несколько паттернов с разной успешностью
        pattern1 = MarketMakerPattern(
            id="successful_pattern",
            pattern_type=MarketMakerPatternType.LIQUIDITY_GRAB,
            features=PatternFeatures(
                book_pressure=BookPressure(0.8),
                volume_delta=VolumeDelta(0.6),
                price_reaction=PriceReaction(0.3),
                spread_change=SpreadChange(0.2),
                order_imbalance=OrderImbalance(0.7),
                liquidity_depth=LiquidityDepth(0.4),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.9),
                price_volatility=PriceVolatility(0.1),
            ),
            confidence=Confidence(0.9),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        pattern2 = MarketMakerPattern(
            id="unsuccessful_pattern",
            pattern_type=MarketMakerPatternType.STOP_HUNT,
            features=PatternFeatures(
                book_pressure=BookPressure(0.3),
                volume_delta=VolumeDelta(0.2),
                price_reaction=PriceReaction(0.7),
                spread_change=SpreadChange(0.8),
                order_imbalance=OrderImbalance(0.2),
                liquidity_depth=LiquidityDepth(0.8),
                time_duration=TimeDuration(150),
                volume_concentration=VolumeConcentration(0.3),
                price_volatility=PriceVolatility(0.8),
            ),
            confidence=Confidence(0.4),
            timestamp=datetime.now(),
            symbol="BTC/USDT",
        )

        await pattern_memory.save_pattern(pattern1)
        await pattern_memory.save_pattern(pattern2)

        # Создаем память для паттернов
        memory1 = MMPatternMemory(
            pattern_id=pattern1.id,
            total_count=TotalCount(15),
            success_count=SuccessCount(13),
            accuracy=Accuracy(0.87),
            average_return=AverageReturn(0.08),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 13 + [PatternOutcome.FAILURE] * 2, maxlen=20),
        )

        memory2 = MMPatternMemory(
            pattern_id=pattern2.id,
            total_count=TotalCount(10),
            success_count=SuccessCount(3),
            accuracy=Accuracy(0.3),
            average_return=AverageReturn(-0.02),
            last_seen=datetime.now(),
            behavior_history=deque([PatternOutcome.SUCCESS] * 3 + [PatternOutcome.FAILURE] * 7, maxlen=20),
        )

        await pattern_memory.repository.save_pattern_memory(memory1)
        await pattern_memory.repository.save_pattern_memory(memory2)

        # Получаем успешные паттерны
        successful_patterns = await pattern_memory.get_successful_patterns(min_accuracy=0.7)
        assert len(successful_patterns) == 1
        assert successful_patterns[0].pattern_id == "successful_pattern"

    @pytest.mark.asyncio
    async def test_close(self, pattern_memory):
        """Тест закрытия PatternMemory."""
        await pattern_memory.close()
        # Проверяем, что репозиторий закрыт
        assert pattern_memory.repository._connection_pool is None


class TestHelperFunctions:
    """Тесты для вспомогательных функций."""

    def test_calculate_cosine_similarity(self):
        """Тест расчета косинусного сходства."""
        features1 = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.5),
            price_reaction=PriceReaction(0.3),
            spread_change=SpreadChange(0.2),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.4),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.1),
        )

        features2 = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.5),
            price_reaction=PriceReaction(0.3),
            spread_change=SpreadChange(0.2),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.4),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.1),
        )

        # Идентичные признаки должны иметь сходство 1.0
        similarity = _calculate_cosine_similarity(features1, features2)
        assert abs(similarity - 1.0) < 0.001

        # Разные признаки должны иметь меньшее сходство
        features3 = PatternFeatures(
            book_pressure=BookPressure(0.1),
            volume_delta=VolumeDelta(0.1),
            price_reaction=PriceReaction(0.9),
            spread_change=SpreadChange(0.9),
            order_imbalance=OrderImbalance(0.1),
            liquidity_depth=LiquidityDepth(0.9),
            time_duration=TimeDuration(100),
            volume_concentration=VolumeConcentration(0.1),
            price_volatility=PriceVolatility(0.9),
        )

        similarity = _calculate_cosine_similarity(features1, features3)
        assert similarity < 0.5

    def test_calculate_euclidean_similarity(self):
        """Тест расчета евклидова сходства."""
        features1 = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.5),
            price_reaction=PriceReaction(0.3),
            spread_change=SpreadChange(0.2),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.4),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.1),
        )

        features2 = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.5),
            price_reaction=PriceReaction(0.3),
            spread_change=SpreadChange(0.2),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.4),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.1),
        )

        # Идентичные признаки должны иметь сходство 1.0
        similarity = _calculate_euclidean_similarity(features1, features2)
        assert abs(similarity - 1.0) < 0.001

    def test_calculate_similarity_score(self):
        """Тест расчета общего сходства."""
        features1 = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.5),
            price_reaction=PriceReaction(0.3),
            spread_change=SpreadChange(0.2),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.4),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.1),
        )

        features2 = PatternFeatures(
            book_pressure=BookPressure(0.7),
            volume_delta=VolumeDelta(0.5),
            price_reaction=PriceReaction(0.3),
            spread_change=SpreadChange(0.2),
            order_imbalance=OrderImbalance(0.6),
            liquidity_depth=LiquidityDepth(0.4),
            time_duration=TimeDuration(300),
            volume_concentration=VolumeConcentration(0.8),
            price_volatility=PriceVolatility(0.1),
        )

        # Идентичные признаки должны иметь высокое сходство
        similarity = _calculate_similarity_score(features1, features2)
        assert similarity > 0.9
        assert similarity <= 1.0
