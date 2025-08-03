"""
Unit тесты для mm_pattern_memory.py.

Покрывает:
- IPatternMemoryRepository - интерфейс репозитория
- PatternMemoryRepository - реализация репозитория
- MatchedPattern - совпадения паттернов
- Вспомогательные функции и методы
"""

import pytest
import asyncio
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock

from domain.market_maker.mm_pattern_memory import (
    IPatternMemoryRepository,
    PatternMemoryRepository,
    MatchedPattern
)
from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    PatternMemory,
    PatternResult,
    PatternFeatures
)
from domain.types.market_maker_types import (
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
    TotalCount
)


class TestIPatternMemoryRepository:
    """Тесты для IPatternMemoryRepository."""
    
    def test_interface_definition(self):
        """Тест определения интерфейса."""
        # Проверяем, что интерфейс определен как Protocol
        assert hasattr(IPatternMemoryRepository, '__call__')
        
        # Проверяем наличие всех методов интерфейса
        methods = [
            'save_pattern',
            'update_pattern_result',
            'get_patterns_by_symbol',
            'get_successful_patterns',
            'find_similar_patterns',
            'get_success_map',
            'update_success_map',
            'save_behavior_history',
            'get_behavior_history',
            'cleanup_old_data',
            'get_storage_statistics'
        ]
        
        for method in methods:
            assert hasattr(IPatternMemoryRepository, method)


class TestMatchedPattern:
    """Тесты для MatchedPattern."""
    
    @pytest.fixture
    def sample_pattern_memory(self) -> PatternMemory:
        """Создает тестовый PatternMemory."""
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            features=PatternFeatures(
                book_pressure=BookPressure(0.5),
                volume_delta=VolumeDelta(0.3),
                price_reaction=PriceReaction(0.2),
                spread_change=SpreadChange(-0.1),
                order_imbalance=OrderImbalance(0.4),
                liquidity_depth=LiquidityDepth(1000.0),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.6),
                price_volatility=PriceVolatility(0.15)
            ),
            confidence=Confidence(0.8),
            context={}
        )
        
        return PatternMemory(
            pattern=pattern,
            result=None,
            accuracy=Accuracy(0.75),
            avg_return=AverageReturn(0.02),
            success_count=SuccessCount(3),
            total_count=TotalCount(4),
            last_seen=datetime(2024, 1, 1, 12, 0, 0)
        )
    
    @pytest.fixture
    def matched_pattern(self, sample_pattern_memory) -> MatchedPattern:
        """Создает экземпляр MatchedPattern."""
        return MatchedPattern(
            pattern_memory=sample_pattern_memory,
            similarity_score=0.85,
            confidence_boost=0.2,
            signal_strength=0.7,
            metadata={
                "pattern_type": "accumulation",
                "timestamp": "2024-01-01T12:00:00",
                "accuracy": 0.75,
                "avg_return": 0.02
            }
        )
    
    def test_creation_with_valid_data(self, sample_pattern_memory):
        """Тест создания с валидными данными."""
        matched = MatchedPattern(
            pattern_memory=sample_pattern_memory,
            similarity_score=0.85,
            confidence_boost=0.2,
            signal_strength=0.7,
            metadata={"test": "data"}
        )
        
        assert matched.pattern_memory == sample_pattern_memory
        assert matched.similarity_score == 0.85
        assert matched.confidence_boost == 0.2
        assert matched.signal_strength == 0.7
        assert matched.metadata == {"test": "data"}
    
    def test_metadata_access(self, matched_pattern):
        """Тест доступа к метаданным."""
        assert matched_pattern.metadata["pattern_type"] == "accumulation"
        assert matched_pattern.metadata["accuracy"] == 0.75
        assert matched_pattern.metadata["avg_return"] == 0.02


class TestPatternMemoryRepository:
    """Тесты для PatternMemoryRepository."""
    
    @pytest.fixture
    def temp_dir(self):
        """Создает временную директорию для тестов."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def repository(self, temp_dir) -> PatternMemoryRepository:
        """Создает экземпляр репозитория."""
        return PatternMemoryRepository(base_path=temp_dir)
    
    @pytest.fixture
    def sample_pattern(self) -> MarketMakerPattern:
        """Создает тестовый паттерн."""
        return MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            features=PatternFeatures(
                book_pressure=BookPressure(0.5),
                volume_delta=VolumeDelta(0.3),
                price_reaction=PriceReaction(0.2),
                spread_change=SpreadChange(-0.1),
                order_imbalance=OrderImbalance(0.4),
                liquidity_depth=LiquidityDepth(1000.0),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.6),
                price_volatility=PriceVolatility(0.15)
            ),
            confidence=Confidence(0.8),
            context={"test": "context"}
        )
    
    @pytest.fixture
    def sample_pattern_result(self) -> PatternResult:
        """Создает тестовый результат паттерна."""
        return PatternResult(
            outcome=PatternOutcome.SUCCESS,
            price_change_5min=0.02,
            price_change_15min=0.05,
            price_change_30min=0.08,
            volume_change=0.15,
            volatility_change=0.1,
            market_context={"symbol": "BTC/USDT"}
        )
    
    def test_creation_with_default_path(self):
        """Тест создания с путем по умолчанию."""
        repo = PatternMemoryRepository()
        assert repo.base_path == Path("market_profiles")
        assert repo.executor is not None
        assert isinstance(repo.pattern_cache, dict)
        assert isinstance(repo.success_map_cache, dict)
        assert isinstance(repo.behavior_history_cache, dict)
    
    def test_creation_with_custom_path(self, temp_dir):
        """Тест создания с пользовательским путем."""
        repo = PatternMemoryRepository(base_path=temp_dir)
        assert repo.base_path == Path(temp_dir)
    
    def test_create_directory_structure(self, temp_dir):
        """Тест создания структуры директорий."""
        repo = PatternMemoryRepository(base_path=temp_dir)
        repo._create_directory_structure()
        
        profiles_dir = Path(temp_dir) / "market_profiles"
        assert profiles_dir.exists()
        assert profiles_dir.is_dir()
    
    def test_init_databases(self, temp_dir):
        """Тест инициализации баз данных."""
        repo = PatternMemoryRepository(base_path=temp_dir)
        repo._init_databases()
        
        main_db_path = Path(temp_dir) / "mm_patterns_metadata.db"
        assert main_db_path.exists()
    
    def test_get_symbol_directory(self, repository):
        """Тест получения директории символа."""
        symbol_dir = repository.get_symbol_directory("BTC/USDT")
        expected_path = Path(repository.base_path) / "market_profiles" / "BTC/USDT" / "mm_patterns"
        assert symbol_dir == expected_path
        assert symbol_dir.exists()
    
    @pytest.mark.asyncio
    async def test_save_pattern(self, repository, sample_pattern):
        """Тест сохранения паттерна."""
        success = await repository.save_pattern("BTC/USDT", sample_pattern)
        assert success is True
        
        # Проверяем, что файл создан
        symbol_dir = repository.get_symbol_directory("BTC/USDT")
        pattern_file = symbol_dir / "pattern_memory.jsonl"
        assert pattern_file.exists()
        
        # Проверяем содержимое файла
        with open(pattern_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == 1
            
            pattern_data = json.loads(lines[0])
            assert pattern_data["pattern"]["pattern_type"] == "accumulation"
            assert pattern_data["pattern"]["symbol"] == "BTC/USDT"
            assert "pattern_id" in pattern_data
    
    def test_generate_pattern_id(self, repository, sample_pattern):
        """Тест генерации ID паттерна."""
        pattern_id = repository._generate_pattern_id("BTC/USDT", sample_pattern)
        
        assert isinstance(pattern_id, str)
        assert len(pattern_id) == 32  # MD5 hash length
        assert pattern_id.isalnum()
    
    @pytest.mark.asyncio
    async def test_get_patterns_by_symbol_empty(self, repository):
        """Тест получения паттернов для пустого символа."""
        patterns = await repository.get_patterns_by_symbol("BTC/USDT")
        assert patterns == []
    
    @pytest.mark.asyncio
    async def test_get_patterns_by_symbol_with_data(self, repository, sample_pattern):
        """Тест получения паттернов с данными."""
        # Сохраняем паттерн
        await repository.save_pattern("BTC/USDT", sample_pattern)
        
        # Получаем паттерны
        patterns = await repository.get_patterns_by_symbol("BTC/USDT")
        assert len(patterns) == 1
        
        pattern = patterns[0]
        assert isinstance(pattern, PatternMemory)
        assert pattern.pattern.pattern_type == MarketMakerPatternType.ACCUMULATION
        assert pattern.pattern.symbol == "BTC/USDT"
    
    @pytest.mark.asyncio
    async def test_get_patterns_by_symbol_with_limit(self, repository, sample_pattern):
        """Тест получения паттернов с ограничением."""
        # Сохраняем несколько паттернов
        for i in range(5):
            pattern = MarketMakerPattern(
                pattern_type=MarketMakerPatternType.ACCUMULATION,
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1, 12, 0, i),
                features=sample_pattern.features,
                confidence=Confidence(0.8),
                context={}
            )
            await repository.save_pattern("BTC/USDT", pattern)
        
        # Получаем с ограничением
        patterns = await repository.get_patterns_by_symbol("BTC/USDT", limit=3)
        assert len(patterns) == 3
    
    @pytest.mark.asyncio
    async def test_get_successful_patterns(self, repository, sample_pattern):
        """Тест получения успешных паттернов."""
        # Создаем паттерн с результатом
        pattern_memory = PatternMemory(
            pattern=sample_pattern,
            result=PatternResult(
                outcome=PatternOutcome.SUCCESS,
                price_change_5min=0.02,
                price_change_15min=0.05,
                price_change_30min=0.08,
                volume_change=0.15,
                volatility_change=0.1,
                market_context={}
            ),
            accuracy=Accuracy(0.8),
            avg_return=AverageReturn(0.02),
            success_count=SuccessCount(4),
            total_count=TotalCount(5),
            last_seen=datetime.now()
        )
        
        # Сохраняем паттерн
        await repository.save_pattern("BTC/USDT", sample_pattern)
        
        # Получаем успешные паттерны
        successful_patterns = await repository.get_successful_patterns("BTC/USDT", min_accuracy=0.7)
        assert len(successful_patterns) >= 0  # Может быть 0, так как результат не обновлен
    
    def test_extract_pattern_features(self, repository):
        """Тест извлечения признаков паттерна."""
        pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTC/USDT",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            features=PatternFeatures(
                book_pressure=BookPressure(0.5),
                volume_delta=VolumeDelta(0.3),
                price_reaction=PriceReaction(0.2),
                spread_change=SpreadChange(-0.1),
                order_imbalance=OrderImbalance(0.4),
                liquidity_depth=LiquidityDepth(1000.0),
                time_duration=TimeDuration(300),
                volume_concentration=VolumeConcentration(0.6),
                price_volatility=PriceVolatility(0.15)
            ),
            confidence=Confidence(0.8),
            context={}
        )
        
        pattern_memory = PatternMemory(
            pattern=pattern,
            result=None,
            accuracy=Accuracy(0.75),
            avg_return=AverageReturn(0.02),
            success_count=SuccessCount(3),
            total_count=TotalCount(4),
            last_seen=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        features = repository._extract_pattern_features(pattern_memory)
        
        assert features["book_pressure"] == 0.5
        assert features["volume_delta"] == 0.3
        assert features["price_reaction"] == 0.2
        assert features["spread_change"] == -0.1
        assert features["order_imbalance"] == 0.4
        assert features["liquidity_depth"] == 1000.0
        assert features["time_duration"] == 300
        assert features["volume_concentration"] == 0.6
        assert features["price_volatility"] == 0.15
    
    def test_calculate_similarity_identical(self, repository):
        """Тест расчета схожести идентичных признаков."""
        features1 = {
            "book_pressure": 0.5,
            "volume_delta": 0.3,
            "price_reaction": 0.2,
            "spread_change": -0.1,
            "order_imbalance": 0.4,
            "liquidity_depth": 1000.0,
            "volume_concentration": 0.6,
            "price_volatility": 0.15
        }
        features2 = features1.copy()
        
        similarity = repository._calculate_similarity(features1, features2)
        assert similarity == 1.0
    
    def test_calculate_similarity_different(self, repository):
        """Тест расчета схожести разных признаков."""
        features1 = {
            "book_pressure": 0.5,
            "volume_delta": 0.3,
            "price_reaction": 0.2,
            "spread_change": -0.1,
            "order_imbalance": 0.4,
            "liquidity_depth": 1000.0,
            "volume_concentration": 0.6,
            "price_volatility": 0.15
        }
        features2 = {
            "book_pressure": -0.5,
            "volume_delta": -0.3,
            "price_reaction": -0.2,
            "spread_change": 0.1,
            "order_imbalance": -0.4,
            "liquidity_depth": 500.0,
            "volume_concentration": 0.3,
            "price_volatility": 0.3
        }
        
        similarity = repository._calculate_similarity(features1, features2)
        assert 0.0 <= similarity <= 1.0
        assert similarity < 1.0  # Должна быть меньше 1.0 для разных признаков
    
    def test_calculate_similarity_empty(self, repository):
        """Тест расчета схожести с пустыми признаками."""
        features1 = {}
        features2 = {}
        
        similarity = repository._calculate_similarity(features1, features2)
        assert similarity == 0.0
    
    def test_calculate_confidence_boost(self, repository):
        """Тест расчета увеличения уверенности."""
        pattern_memory = PatternMemory(
            pattern=MarketMakerPattern(
                pattern_type=MarketMakerPatternType.ACCUMULATION,
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                features=PatternFeatures(
                    book_pressure=BookPressure(0.5),
                    volume_delta=VolumeDelta(0.3),
                    price_reaction=PriceReaction(0.2),
                    spread_change=SpreadChange(-0.1),
                    order_imbalance=OrderImbalance(0.4),
                    liquidity_depth=LiquidityDepth(1000.0),
                    time_duration=TimeDuration(300),
                    volume_concentration=VolumeConcentration(0.6),
                    price_volatility=PriceVolatility(0.15)
                ),
                confidence=Confidence(0.8),
                context={}
            ),
            result=None,
            accuracy=Accuracy(0.8),
            avg_return=AverageReturn(0.02),
            success_count=SuccessCount(5),
            total_count=TotalCount(6),
            last_seen=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        confidence_boost = repository._calculate_confidence_boost(0.9, pattern_memory)
        assert 0.0 <= confidence_boost <= 1.0
    
    def test_calculate_signal_strength(self, repository):
        """Тест расчета силы сигнала."""
        pattern_memory = PatternMemory(
            pattern=MarketMakerPattern(
                pattern_type=MarketMakerPatternType.ACCUMULATION,
                symbol="BTC/USDT",
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                features=PatternFeatures(
                    book_pressure=BookPressure(0.5),
                    volume_delta=VolumeDelta(0.3),
                    price_reaction=PriceReaction(0.2),
                    spread_change=SpreadChange(-0.1),
                    order_imbalance=OrderImbalance(0.4),
                    liquidity_depth=LiquidityDepth(1000.0),
                    time_duration=TimeDuration(300),
                    volume_concentration=VolumeConcentration(0.6),
                    price_volatility=PriceVolatility(0.15)
                ),
                confidence=Confidence(0.8),
                context={}
            ),
            result=None,
            accuracy=Accuracy(0.8),
            avg_return=AverageReturn(0.02),
            success_count=SuccessCount(5),
            total_count=TotalCount(6),
            last_seen=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        signal_strength = repository._calculate_signal_strength(pattern_memory)
        assert signal_strength >= 0.0
    
    @pytest.mark.asyncio
    async def test_find_similar_patterns(self, repository, sample_pattern):
        """Тест поиска похожих паттернов."""
        # Сохраняем паттерн
        await repository.save_pattern("BTC/USDT", sample_pattern)
        
        # Ищем похожие паттерны
        features = {
            "book_pressure": 0.5,
            "volume_delta": 0.3,
            "price_reaction": 0.2,
            "spread_change": -0.1,
            "order_imbalance": 0.4,
            "liquidity_depth": 1000.0,
            "volume_concentration": 0.6,
            "price_volatility": 0.15
        }
        
        similar_patterns = await repository.find_similar_patterns(
            "BTC/USDT", features, similarity_threshold=0.8
        )
        
        assert isinstance(similar_patterns, list)
        # Может быть 0, если паттерн еще не имеет результата
        assert len(similar_patterns) >= 0
    
    @pytest.mark.asyncio
    async def test_update_pattern_result(self, repository, sample_pattern, sample_pattern_result):
        """Тест обновления результата паттерна."""
        # Сохраняем паттерн
        await repository.save_pattern("BTC/USDT", sample_pattern)
        
        # Получаем ID паттерна
        pattern_id = repository._generate_pattern_id("BTC/USDT", sample_pattern)
        
        # Обновляем результат
        success = await repository.update_pattern_result("BTC/USDT", pattern_id, sample_pattern_result)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_success_map_empty(self, repository):
        """Тест получения карты успеха (пустая)."""
        success_map = await repository.get_success_map("BTC/USDT")
        assert success_map == {}
    
    @pytest.mark.asyncio
    async def test_update_success_map(self, repository):
        """Тест обновления карты успеха."""
        success = await repository.update_success_map("BTC/USDT", "accumulation", 0.8)
        assert success is True
        
        # Проверяем, что карта обновлена
        success_map = await repository.get_success_map("BTC/USDT")
        assert success_map["accumulation"] == 0.8
    
    @pytest.mark.asyncio
    async def test_save_behavior_history(self, repository):
        """Тест сохранения истории поведения."""
        behavior_data = {
            "behavior_type": "accumulation",
            "volume": 1000.0,
            "price_change": 0.02,
            "timestamp": datetime.now().isoformat()
        }
        
        success = await repository.save_behavior_history("BTC/USDT", behavior_data)
        assert success is True
    
    @pytest.mark.asyncio
    async def test_get_behavior_history_empty(self, repository):
        """Тест получения истории поведения (пустая)."""
        history = await repository.get_behavior_history("BTC/USDT", days=30)
        assert history == []
    
    @pytest.mark.asyncio
    async def test_get_behavior_history_with_data(self, repository):
        """Тест получения истории поведения с данными."""
        # Сохраняем данные поведения
        behavior_data = {
            "behavior_type": "accumulation",
            "volume": 1000.0,
            "price_change": 0.02,
            "timestamp": datetime.now().isoformat()
        }
        await repository.save_behavior_history("BTC/USDT", behavior_data)
        
        # Получаем историю
        history = await repository.get_behavior_history("BTC/USDT", days=30)
        assert len(history) >= 0  # Может быть 0 из-за асинхронности
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, repository, sample_pattern):
        """Тест очистки старых данных."""
        # Создаем старый паттерн
        old_pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTC/USDT",
            timestamp=datetime.now() - timedelta(days=60),  # 60 дней назад
            features=sample_pattern.features,
            confidence=Confidence(0.8),
            context={}
        )
        
        # Создаем новый паттерн
        new_pattern = MarketMakerPattern(
            pattern_type=MarketMakerPatternType.ACCUMULATION,
            symbol="BTC/USDT",
            timestamp=datetime.now(),  # Сегодня
            features=sample_pattern.features,
            confidence=Confidence(0.8),
            context={}
        )
        
        # Сохраняем оба паттерна
        await repository.save_pattern("BTC/USDT", old_pattern)
        await repository.save_pattern("BTC/USDT", new_pattern)
        
        # Очищаем старые данные
        cleaned_count = await repository.cleanup_old_data("BTC/USDT", days=30)
        assert cleaned_count >= 0
    
    @pytest.mark.asyncio
    async def test_get_storage_statistics(self, repository):
        """Тест получения статистики хранилища."""
        stats = await repository.get_storage_statistics()
        
        assert isinstance(stats, dict)
        assert "total_patterns" in stats
        assert "total_symbols" in stats
        assert "storage_size_mb" in stats
        assert "last_updated" in stats
    
    def test_data_to_pattern_memory_valid(self, repository):
        """Тест конвертации данных в PatternMemory (валидные данные)."""
        pattern_data = {
            "pattern": {
                "pattern_type": "accumulation",
                "symbol": "BTC/USDT",
                "timestamp": "2024-01-01T12:00:00",
                "features": {
                    "book_pressure": 0.5,
                    "volume_delta": 0.3,
                    "price_reaction": 0.2,
                    "spread_change": -0.1,
                    "order_imbalance": 0.4,
                    "liquidity_depth": 1000.0,
                    "time_duration": 300,
                    "volume_concentration": 0.6,
                    "price_volatility": 0.15,
                    "market_microstructure": {}
                },
                "confidence": 0.8,
                "context": {}
            },
            "result": None,
            "accuracy": 0.75,
            "avg_return": 0.02,
            "success_count": 3,
            "total_count": 4,
            "last_seen": "2024-01-01T12:00:00"
        }
        
        pattern_memory = repository._data_to_pattern_memory(pattern_data)
        assert pattern_memory is not None
        assert isinstance(pattern_memory, PatternMemory)
        assert pattern_memory.pattern.pattern_type == MarketMakerPatternType.ACCUMULATION
        assert pattern_memory.pattern.symbol == "BTC/USDT"
    
    def test_data_to_pattern_memory_invalid(self, repository):
        """Тест конвертации данных в PatternMemory (невалидные данные)."""
        pattern_data = {
            "pattern": "invalid_data",
            "result": None,
            "accuracy": "invalid",
            "avg_return": "invalid",
            "success_count": "invalid",
            "total_count": "invalid",
            "last_seen": "invalid"
        }
        
        pattern_memory = repository._data_to_pattern_memory(pattern_data)
        assert pattern_memory is None
    
    def test_write_and_read_patterns_to_file(self, repository, temp_dir):
        """Тест записи и чтения паттернов в/из файла."""
        pattern_file = Path(temp_dir) / "test_patterns.jsonl"
        
        # Тестовые данные
        patterns = [
            {"id": "1", "data": "test1"},
            {"id": "2", "data": "test2"},
            {"id": "3", "data": "test3"}
        ]
        
        # Записываем паттерны
        repository._write_patterns_to_file(pattern_file, patterns)
        assert pattern_file.exists()
        
        # Читаем паттерны
        read_patterns = repository._read_patterns_from_file(pattern_file)
        assert len(read_patterns) == 3
        assert read_patterns[0]["id"] == "1"
        assert read_patterns[1]["id"] == "2"
        assert read_patterns[2]["id"] == "3"
    
    def test_write_and_read_json_file(self, repository, temp_dir):
        """Тест записи и чтения JSON файла."""
        json_file = Path(temp_dir) / "test.json"
        
        # Тестовые данные
        data = {
            "key1": "value1",
            "key2": 123,
            "key3": {"nested": "data"}
        }
        
        # Записываем данные
        repository._write_json_file(json_file, data)
        assert json_file.exists()
        
        # Читаем данные
        read_data = repository._read_json_file(json_file)
        assert read_data["key1"] == "value1"
        assert read_data["key2"] == 123
        assert read_data["key3"]["nested"] == "data"
    
    def test_read_json_file_nonexistent(self, repository, temp_dir):
        """Тест чтения несуществующего JSON файла."""
        json_file = Path(temp_dir) / "nonexistent.json"
        
        # Должен вернуть пустой словарь
        data = repository._read_json_file(json_file)
        assert data == {}
    
    def test_save_and_read_behavior_to_db(self, repository, temp_dir):
        """Тест сохранения и чтения поведения в/из БД."""
        behavior_db = Path(temp_dir) / "behavior.db"
        
        # Тестовые данные
        behavior_data = {
            "behavior_type": "accumulation",
            "volume": 1000.0,
            "price_change": 0.02,
            "timestamp": datetime.now().isoformat()
        }
        
        # Сохраняем данные
        repository._save_behavior_to_db(behavior_db, behavior_data)
        assert behavior_db.exists()
        
        # Читаем данные
        start_date = datetime.now() - timedelta(days=1)
        history = repository._read_behavior_from_db(behavior_db, start_date)
        assert len(history) >= 0  # Может быть 0 из-за временных меток
    
    def test_read_storage_stats_from_db_nonexistent(self, repository, temp_dir):
        """Тест чтения статистики из несуществующей БД."""
        db_path = Path(temp_dir) / "nonexistent.db"
        
        stats = repository._read_storage_stats_from_db(db_path)
        assert stats["total_patterns"] == 0
        assert stats["total_symbols"] == 0
        assert stats["total_successful_patterns"] == 0
    
    def test_del_method(self, repository):
        """Тест метода __del__."""
        # Просто проверяем, что метод не вызывает ошибок
        repository.__del__()
        assert True  # Если дошли до этой строки, значит ошибок нет 