"""
Основной модуль интеграции рыночной памяти.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from .types import (
    MarketMemory,
    MarketMemoryConfig,
    MemoryQuery,
    MemoryResult,
    MemoryType,
)

logger = logger


class MarketMemoryIntegration:
    """
    Интеграция рыночной памяти для агентов.
    """

    def __init__(self, config: Optional[MarketMemoryConfig] = None):
        """
        Инициализация интеграции рыночной памяти.
        :param config: конфигурация памяти
        """
        self.config = config or MarketMemoryConfig()

        # Хранилище памяти
        self.memories: Dict[str, MarketMemory] = {}
        self.memory_index: Dict[str, List[str]] = {}

        # Статистика
        self.stats: Dict[str, Any] = {
            "total_memories": 0,
            "queries": 0,
            "hits": 0,
            "misses": 0,
            "cleanups": 0,
        }

        # Асинхронные задачи
        self.cleanup_task: Optional[asyncio.Task] = None
        self.is_running = False

        logger.info("MarketMemoryIntegration initialized")

    async def start(self) -> None:
        """Запуск интеграции памяти."""
        try:
            if self.is_running:
                return

            self.is_running = True

            # Запуск очистки памяти
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())

            logger.info("MarketMemoryIntegration started")

        except Exception as e:
            logger.error(f"Error starting MarketMemoryIntegration: {e}")
            self.is_running = False

    async def stop(self) -> None:
        """Остановка интеграции памяти."""
        try:
            if not self.is_running:
                return

            self.is_running = False

            # Отмена задачи очистки
            if self.cleanup_task:
                self.cleanup_task.cancel()
                try:
                    await self.cleanup_task
                except asyncio.CancelledError:
                    pass

            logger.info("MarketMemoryIntegration stopped")

        except Exception as e:
            logger.error(f"Error stopping MarketMemoryIntegration: {e}")

    async def store_memory(self, memory: MarketMemory) -> bool:
        """Сохранение записи в память."""
        try:
            # Проверяем лимит памяти
            if len(self.memories) >= self.config.max_memories:
                await self._cleanup_old_memories()

            # Сохраняем память
            self.memories[memory.memory_id] = memory

            # Индексируем память
            if self.config.enable_indexing:
                self._index_memory(memory)

            self.stats["total_memories"] += 1

            if self.config.log_memory_operations:
                logger.info(f"Stored memory {memory.memory_id} for {memory.symbol}")

            return True

        except Exception as e:
            logger.error(f"Error storing memory {memory.memory_id}: {e}")
            return False

    async def query_memory(self, query: MemoryQuery) -> MemoryResult:
        """Запрос к памяти."""
        start_time = time.time()

        try:
            self.stats["queries"] += 1

            # Фильтруем памяти по запросу
            filtered_memories = self._filter_memories(query)

            # Сортируем по релевантности
            sorted_memories = self._sort_by_relevance(filtered_memories, query)

            # Ограничиваем результат
            limited_memories = sorted_memories[: query.limit]

            # Вычисляем релевантность
            relevance_score = self._calculate_relevance(limited_memories, query)

            query_time = time.time() - start_time

            result = MemoryResult(
                memories=limited_memories,
                total_count=len(filtered_memories),
                query_time=query_time,
                relevance_score=relevance_score,
            )

            self.stats["hits"] += 1

            return result

        except Exception as e:
            logger.error(f"Error querying memory: {e}")
            self.stats["misses"] += 1

            return MemoryResult(
                memories=[],
                total_count=0,
                query_time=time.time() - start_time,
                relevance_score=0.0,
            )

    async def get_pattern_memories(
        self, symbol: str, hours: int = 24
    ) -> List[MarketMemory]:
        """Получение паттернов из памяти."""
        try:
            query = MemoryQuery(
                symbol=symbol,
                memory_type=MemoryType.PATTERN,
                start_time=datetime.now() - timedelta(hours=hours),
                limit=100,
            )

            result = await self.query_memory(query)
            return result.memories

        except Exception as e:
            logger.error(f"Error getting pattern memories for {symbol}: {e}")
            return []

    async def get_event_memories(
        self, symbol: str, event_type: str, hours: int = 24
    ) -> List[MarketMemory]:
        """Получение событий из памяти."""
        try:
            query = MemoryQuery(
                symbol=symbol,
                memory_type=MemoryType.EVENT,
                start_time=datetime.now() - timedelta(hours=hours),
                tags=[event_type],
                limit=100,
            )

            result = await self.query_memory(query)
            return result.memories

        except Exception as e:
            logger.error(f"Error getting event memories for {symbol}: {e}")
            return []

    async def get_decision_memories(
        self, symbol: str, hours: int = 24
    ) -> List[MarketMemory]:
        """Получение решений из памяти."""
        try:
            query = MemoryQuery(
                symbol=symbol,
                memory_type=MemoryType.DECISION,
                start_time=datetime.now() - timedelta(hours=hours),
                limit=100,
            )

            result = await self.query_memory(query)
            return result.memories

        except Exception as e:
            logger.error(f"Error getting decision memories for {symbol}: {e}")
            return []

    async def get_outcome_memories(
        self, symbol: str, hours: int = 24
    ) -> List[MarketMemory]:
        """Получение результатов из памяти."""
        try:
            query = MemoryQuery(
                symbol=symbol,
                memory_type=MemoryType.OUTCOME,
                start_time=datetime.now() - timedelta(hours=hours),
                limit=100,
            )

            result = await self.query_memory(query)
            return result.memories

        except Exception as e:
            logger.error(f"Error getting outcome memories for {symbol}: {e}")
            return []

    def _filter_memories(self, query: MemoryQuery) -> List[MarketMemory]:
        """Фильтрация памяти по запросу."""
        try:
            filtered = []

            for memory in self.memories.values():
                # Фильтр по символу
                if memory.symbol != query.symbol:
                    continue

                # Фильтр по типу памяти
                if query.memory_type and memory.memory_type != query.memory_type:
                    continue

                # Фильтр по времени
                if query.start_time and memory.timestamp < query.start_time:
                    continue

                if query.end_time and memory.timestamp > query.end_time:
                    continue

                # Фильтр по тегам
                if query.tags and not any(tag in memory.tags for tag in query.tags):
                    continue

                # Фильтр по уверенности
                if memory.confidence < query.min_confidence:
                    continue

                filtered.append(memory)

            return filtered

        except Exception as e:
            logger.error(f"Error filtering memories: {e}")
            return []

    def _sort_by_relevance(
        self, memories: List[MarketMemory], query: MemoryQuery
    ) -> List[MarketMemory]:
        """Сортировка памяти по релевантности."""
        try:

            def relevance_score(memory: MarketMemory) -> float:
                score = memory.confidence

                # Бонус за приоритет
                score += memory.priority.value * 0.1

                # Бонус за свежесть
                age_hours = (datetime.now() - memory.timestamp).total_seconds() / 3600
                freshness_bonus = max(0, 1.0 - age_hours / 24.0)  # Максимум 24 часа
                score += freshness_bonus * 0.2

                # Бонус за совпадение тегов
                if query.tags:
                    tag_matches = sum(1 for tag in query.tags if tag in memory.tags)
                    score += tag_matches * 0.1

                return score

            return sorted(memories, key=relevance_score, reverse=True)

        except Exception as e:
            logger.error(f"Error sorting memories by relevance: {e}")
            return memories

    def _calculate_relevance(
        self, memories: List[MarketMemory], query: MemoryQuery
    ) -> float:
        """Вычисление релевантности результата."""
        try:
            if not memories:
                return 0.0

            # Средняя уверенность
            avg_confidence = sum(m.confidence for m in memories) / len(memories)

            # Средний приоритет
            avg_priority = sum(m.priority.value for m in memories) / len(memories)

            # Свежесть
            avg_age_hours = sum(
                (datetime.now() - m.timestamp).total_seconds() / 3600 for m in memories
            ) / len(memories)
            freshness = max(0, 1.0 - avg_age_hours / 24.0)

            # Общая релевантность
            relevance = avg_confidence * 0.5 + avg_priority * 0.2 + freshness * 0.3

            return min(1.0, relevance)

        except Exception as e:
            logger.error(f"Error calculating relevance: {e}")
            return 0.0

    def _index_memory(self, memory: MarketMemory) -> None:
        """Индексирование памяти."""
        try:
            # Индекс по символу
            if memory.symbol not in self.memory_index:
                self.memory_index[memory.symbol] = []
            self.memory_index[memory.symbol].append(memory.memory_id)

            # Индекс по типу
            type_key = f"type_{memory.memory_type.value}"
            if type_key not in self.memory_index:
                self.memory_index[type_key] = []
            self.memory_index[type_key].append(memory.memory_id)

            # Индекс по тегам
            for tag in memory.tags:
                tag_key = f"tag_{tag}"
                if tag_key not in self.memory_index:
                    self.memory_index[tag_key] = []
                self.memory_index[tag_key].append(memory.memory_id)

        except Exception as e:
            logger.error(f"Error indexing memory {memory.memory_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Цикл очистки памяти."""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cleanup_interval)
                await self._cleanup_old_memories()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    async def _cleanup_old_memories(self) -> None:
        """Очистка старых записей памяти."""
        try:
            current_time = datetime.now()
            memories_to_remove = []

            for memory_id, memory in self.memories.items():
                age_seconds = (current_time - memory.timestamp).total_seconds()
                if age_seconds > self.config.memory_ttl:
                    memories_to_remove.append(memory_id)

            # Удаляем старые записи
            for memory_id in memories_to_remove:
                del self.memories[memory_id]

            # Очищаем индексы
            if memories_to_remove:
                self._cleanup_indexes(memories_to_remove)

            self.stats["cleanups"] += 1

            if memories_to_remove:
                logger.info(f"Cleaned up {len(memories_to_remove)} old memories")

        except Exception as e:
            logger.error(f"Error cleaning up old memories: {e}")

    def _cleanup_indexes(self, removed_memory_ids: List[str]) -> None:
        """Очистка индексов от удаленных записей."""
        try:
            for index_key, memory_ids in self.memory_index.items():
                self.memory_index[index_key] = [
                    mid for mid in memory_ids if mid not in removed_memory_ids
                ]

        except Exception as e:
            logger.error(f"Error cleaning up indexes: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики памяти."""
        try:
            return {
                "total_memories": len(self.memories),
                "memory_types": self._get_memory_type_stats(),
                "symbols": list(set(m.symbol for m in self.memories.values())),
                "index_size": len(self.memory_index),
                "stats": self.stats.copy(),
                "config": {
                    "max_memories": self.config.max_memories,
                    "cleanup_interval": self.config.cleanup_interval,
                    "memory_ttl": self.config.memory_ttl,
                },
            }

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def _get_memory_type_stats(self) -> Dict[str, int]:
        """Получение статистики по типам памяти."""
        try:
            stats: Dict[str, int] = {}
            for memory in self.memories.values():
                memory_type = memory.memory_type.value
                stats[memory_type] = stats.get(memory_type, 0) + 1
            return stats

        except Exception as e:
            logger.error(f"Error getting memory type stats: {e}")
            return {}

    def clear_memory(self) -> None:
        """Очистка всей памяти."""
        try:
            self.memories.clear()
            self.memory_index.clear()
            self.stats["total_memories"] = 0

            logger.info("Memory cleared")

        except Exception as e:
            logger.error(f"Error clearing memory: {e}")
