"""
Юнит-тесты для EvolutionCache.
"""
import time
from datetime import datetime, timedelta
from uuid import uuid4
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import EvolutionContext, StrategyCandidate
from infrastructure.evolution.cache import EvolutionCache
from infrastructure.evolution.exceptions import CacheError
from infrastructure.evolution.types import CacheKey

class TestEvolutionCache:
    """Тесты для EvolutionCache."""
    def test_init_default_config(self: "TestEvolutionCache") -> None:
        """Тест инициализации с конфигурацией по умолчанию."""
        cache = EvolutionCache()
        assert cache.max_size == 1000
        assert cache.ttl == 300
        assert cache.strategy == "lru"
        assert len(cache._cache) == 0
    def test_init_custom_config(self: "TestEvolutionCache") -> None:
        """Тест инициализации с пользовательской конфигурацией."""
        config = {
            "cache_size": 500,
            "cache_ttl": 600,
            "cache_strategy": "fifo"
        }
        cache = EvolutionCache(config)
        assert cache.max_size == 500
        assert cache.ttl == 600
        assert cache.strategy == "fifo"
    def test_init_invalid_config(self: "TestEvolutionCache") -> None:
        """Тест инициализации с некорректной конфигурацией."""
        config = {
            "cache_size": -1,  # Некорректный размер
            "cache_ttl": 0,    # Некорректный TTL
            "cache_strategy": "invalid"  # Некорректная стратегия
        }
        with pytest.raises(CacheError) as exc_info:
            EvolutionCache(config)
        assert "Некорректная конфигурация кэша" in str(exc_info.value)
    def test_set_get_candidate(self, cache: EvolutionCache, sample_candidate: StrategyCandidate) -> None:
        """Тест установки и получения кандидата стратегии."""
        cache.set_candidate(str(sample_candidate.id), sample_candidate)
        retrieved = cache.get_candidate(str(sample_candidate.id))
        assert retrieved is not None
        assert retrieved.id == sample_candidate.id
        assert retrieved.name == sample_candidate.name
        assert retrieved.strategy_type == sample_candidate.strategy_type
    def test_get_candidate_not_found(self, cache: EvolutionCache) -> None:
        """Тест получения несуществующего кандидата."""
        retrieved = cache.get_candidate(str(uuid4()))
        assert retrieved is None
    def test_get_candidate_expired(self, cache: EvolutionCache, sample_candidate: StrategyCandidate) -> None:
        """Тест получения истекшего кандидата."""
        cache.set_candidate(str(sample_candidate.id), sample_candidate)
        # Установить время истечения в прошлое
        cache._cache[str(sample_candidate.id)]["expires_at"] = datetime.now() - timedelta(seconds=1)
        retrieved = cache.get_candidate(str(sample_candidate.id))
        assert retrieved is None
        assert str(sample_candidate.id) not in cache._cache
    def test_set_get_evaluation(self, cache: EvolutionCache, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест установки и получения результата оценки."""
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        retrieved = cache.get_evaluation(sample_evaluation.id)
        assert retrieved is not None
        assert retrieved.id == sample_evaluation.id
        assert retrieved.strategy_id == sample_evaluation.strategy_id
        assert retrieved.total_trades == sample_evaluation.total_trades
    def test_get_evaluation_not_found(self, cache: EvolutionCache) -> None:
        """Тест получения несуществующего результата оценки."""
        retrieved = cache.get_evaluation(uuid4())
        assert retrieved is None
    def test_set_get_context(self, cache: EvolutionCache, sample_context: EvolutionContext) -> None:
        """Тест установки и получения контекста эволюции."""
        cache.set_context(sample_context.id, sample_context)
        retrieved = cache.get_context(sample_context.id)
        assert retrieved is not None
        assert retrieved.id == sample_context.id
        assert retrieved.name == sample_context.name
        assert retrieved.population_size == sample_context.population_size
    def test_get_context_not_found(self, cache: EvolutionCache) -> None:
        """Тест получения несуществующего контекста."""
        retrieved = cache.get_context(uuid4())
        assert retrieved is None
    def test_set_get_custom_data(self, cache: EvolutionCache) -> None:
        """Тест установки и получения пользовательских данных."""
        key = CacheKey("test_key")
        data = {"test": "data", "number": 42}
        cache.set(key, data)
        retrieved = cache.get(key)
        assert retrieved == data
    def test_get_custom_data_not_found(self, cache: EvolutionCache) -> None:
        """Тест получения несуществующих пользовательских данных."""
        retrieved = cache.get(CacheKey("non_existent_key"))
        assert retrieved is None
    def test_delete_candidate(self, cache: EvolutionCache, sample_candidate: StrategyCandidate) -> None:
        """Тест удаления кандидата стратегии."""
        cache.set_candidate(sample_candidate.id, sample_candidate)
        assert str(sample_candidate.id) in cache._cache
        cache.delete_candidate(sample_candidate.id)
        assert str(sample_candidate.id) not in cache._cache
    def test_delete_evaluation(self, cache: EvolutionCache, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест удаления результата оценки."""
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        assert str(sample_evaluation.id) in cache._cache
        cache.delete_evaluation(sample_evaluation.id)
        assert str(sample_evaluation.id) not in cache._cache
    def test_delete_context(self, cache: EvolutionCache, sample_context: EvolutionContext) -> None:
        """Тест удаления контекста эволюции."""
        cache.set_context(sample_context.id, sample_context)
        assert str(sample_context.id) in cache._cache
        cache.delete_context(sample_context.id)
        assert str(sample_context.id) not in cache._cache
    def test_delete_custom_data(self, cache: EvolutionCache) -> None:
        """Тест удаления пользовательских данных."""
        key = CacheKey("test_key")
        data = {"test": "data"}
        cache.set(key, data)
        assert key in cache._cache
        cache.delete(key)
        assert key not in cache._cache
    def test_clear(self, cache: EvolutionCache, sample_candidate: StrategyCandidate,
                  sample_evaluation: StrategyEvaluationResult, sample_context: EvolutionContext) -> None:
        """Тест полной очистки кэша."""
        # Добавить данные в кэш
        cache.set_candidate(sample_candidate.id, sample_candidate)
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        cache.set_context(sample_context.id, sample_context)
        cache.set(CacheKey("custom_key"), {"data": "value"})
        assert len(cache._cache) == 4
        cache.clear()
        assert len(cache._cache) == 0
    def test_get_stats(self, cache: EvolutionCache, sample_candidate: StrategyCandidate,
                      sample_evaluation: StrategyEvaluationResult, sample_context: EvolutionContext) -> None:
        """Тест получения статистики кэша."""
        # Добавить данные в кэш
        cache.set_candidate(sample_candidate.id, sample_candidate)
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        cache.set_context(sample_context.id, sample_context)
        cache.set(CacheKey("custom_key"), {"data": "value"})
        stats = cache.get_stats()
        assert stats["total_items"] == 4
        assert stats["candidates"] == 1
        assert stats["evaluations"] == 1
        assert stats["contexts"] == 1
        assert stats["custom_data"] == 1
        assert stats["hit_rate"] == 0.0
        assert stats["miss_rate"] == 0.0
    def test_lru_eviction(self: "TestEvolutionCache") -> None:
        """Тест вытеснения по LRU стратегии."""
        config = {
            "cache_size": 2,
            "cache_ttl": 300,
            "cache_strategy": "lru"
        }
        cache = EvolutionCache(config)
        # Добавить 3 элемента
        cache.set(CacheKey("key1"), "value1")
        cache.set(CacheKey("key2"), "value2")
        cache.set(CacheKey("key3"), "value3")
        # Первый элемент должен быть вытеснен
        assert CacheKey("key1") not in cache._cache
        assert CacheKey("key2") in cache._cache
        assert CacheKey("key3") in cache._cache
        # Обратиться к key2, чтобы обновить его позицию в LRU
        cache.get(CacheKey("key2"))
        # Добавить еще один элемент
        cache.set(CacheKey("key4"), "value4")
        # Теперь key3 должен быть вытеснен
        assert CacheKey("key2") in cache._cache
        assert CacheKey("key3") not in cache._cache
        assert CacheKey("key4") in cache._cache
    def test_fifo_eviction(self: "TestEvolutionCache") -> None:
        """Тест вытеснения по FIFO стратегии."""
        config = {
            "cache_size": 2,
            "cache_ttl": 300,
            "cache_strategy": "fifo"
        }
        cache = EvolutionCache(config)
        # Добавить 3 элемента
        cache.set(CacheKey("key1"), "value1")
        cache.set(CacheKey("key2"), "value2")
        cache.set(CacheKey("key3"), "value3")
        # Первый элемент должен быть вытеснен
        assert CacheKey("key1") not in cache._cache
        assert CacheKey("key2") in cache._cache
        assert CacheKey("key3") in cache._cache
        # Обратиться к key2 не должно изменить порядок вытеснения
        cache.get(CacheKey("key2"))
        # Добавить еще один элемент
        cache.set(CacheKey("key4"), "value4")
        # Теперь key2 должен быть вытеснен (FIFO)
        assert CacheKey("key2") not in cache._cache
        assert CacheKey("key3") in cache._cache
        assert CacheKey("key4") in cache._cache
    def test_ttl_expiration(self: "TestEvolutionCache") -> None:
        """Тест истечения по TTL."""
        config = {
            "cache_size": 100,
            "cache_ttl": 1,  # 1 секунда
            "cache_strategy": "lru"
        }
        cache = EvolutionCache(config)
        cache.set(CacheKey("key1"), "value1")
        # Данные должны быть доступны сразу
        assert cache.get(CacheKey("key1")) == "value1"
        # Подождать истечения TTL
        time.sleep(1.1)
        # Данные должны быть удалены
        assert cache.get(CacheKey("key1")) is None
    def test_cache_hit_miss_stats(self, cache: EvolutionCache) -> None:
        """Тест статистики попаданий и промахов."""
        cache.set(CacheKey("key1"), "value1")
        # Попадание
        cache.get(CacheKey("key1"))
        # Промах
        cache.get(CacheKey("key2"))
        stats = cache.get_stats()
        assert stats["hit_rate"] == 0.5
        assert stats["miss_rate"] == 0.5
    def test_cache_size_limit(self: "TestEvolutionCache") -> None:
        """Тест ограничения размера кэша."""
        config = {
            "cache_size": 1,
            "cache_ttl": 300,
            "cache_strategy": "lru"
        }
        cache = EvolutionCache(config)
        cache.set(CacheKey("key1"), "value1")
        cache.set(CacheKey("key2"), "value2")
        assert len(cache._cache) == 1
        assert CacheKey("key1") not in cache._cache
        assert CacheKey("key2") in cache._cache
    def test_cache_invalid_key(self, cache: EvolutionCache) -> None:
        """Тест обработки некорректных ключей."""
        with pytest.raises(CacheError) as exc_info:
            cache.set(CacheKey(""), "value")  # Пустая строка вместо None
        assert "Ключ кэша не может быть пустым" in str(exc_info.value)
        with pytest.raises(CacheError) as exc_info:
            cache.set(CacheKey(""), "value")
        assert "Ключ кэша не может быть пустым" in str(exc_info.value)
    def test_cache_invalid_value(self, cache: EvolutionCache) -> None:
        """Тест обработки некорректных значений."""
        with pytest.raises(CacheError) as exc_info:
            cache.set(CacheKey("key"), None)
        assert "Значение кэша не может быть пустым" in str(exc_info.value)
    def test_cache_cleanup_expired(self, cache: EvolutionCache) -> None:
        """Тест очистки истекших элементов."""
        cache.set(CacheKey("key1"), "value1")
        cache.set(CacheKey("key2"), "value2")
        # Установить время истечения в прошлое для key1
        cache._cache[CacheKey("key1")]["expires_at"] = datetime.now() - timedelta(seconds=1)
        # Очистить истекшие элементы
        cache._cleanup_expired()
        assert CacheKey("key1") not in cache._cache
        assert CacheKey("key2") in cache._cache
    def test_cache_serialization(self, cache: EvolutionCache, sample_candidate: StrategyCandidate) -> None:
        """Тест сериализации данных в кэше."""
        cache.set_candidate(sample_candidate.id, sample_candidate)
        # Получить данные из кэша
        cached_data = cache._cache[str(sample_candidate.id)]["data"]
        # Проверить, что данные сериализованы корректно
        assert isinstance(cached_data, dict)
        assert cached_data["id"] == str(sample_candidate.id)
        assert cached_data["name"] == sample_candidate.name
    def test_cache_deserialization(self, cache: EvolutionCache, sample_candidate: StrategyCandidate) -> None:
        """Тест десериализации данных из кэша."""
        cache.set_candidate(sample_candidate.id, sample_candidate)
        # Получить десериализованный объект
        retrieved = cache.get_candidate(sample_candidate.id)
        # Проверить, что объект десериализован корректно
        assert isinstance(retrieved, StrategyCandidate)
        assert retrieved.id == sample_candidate.id
        assert retrieved.name == sample_candidate.name
        assert retrieved.strategy_type == sample_candidate.strategy_type 
