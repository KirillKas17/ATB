"""
Базовый класс для репозиториев.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar
from uuid import UUID
from domain.exceptions.base_exceptions import RepositoryError

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Базовый класс для всех репозиториев."""

    def __init__(self) -> None:
        self._cache: Dict[str, T] = {}
        self._cache_enabled: bool = True
        self._cache_ttl: int = 300  # 5 минут

    @abstractmethod
    def create(self, entity: T) -> T:
        """Создание сущности."""
        pass

    @abstractmethod
    def get_by_id(self, entity_id: str | UUID) -> Optional[T]:
        """Получение сущности по ID."""
        pass

    @abstractmethod
    def update(self, entity: T) -> T:
        """Обновление сущности."""
        pass

    @abstractmethod
    def delete(self, entity_id: str | UUID) -> bool:
        """Удаление сущности."""
        pass

    @abstractmethod
    def get_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[T]:
        """Получение всех сущностей."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Подсчет количества сущностей."""
        pass

    def exists(self, entity_id: str | UUID) -> bool:
        """Проверка существования сущности."""
        return self.get_by_id(entity_id) is not None

    def get_or_create(self, entity: T, **kwargs: Any) -> T:
        """Получение существующей или создание новой сущности."""
        existing = self.find_by_criteria(**kwargs)
        if existing:
            return existing[0]
        return self.create(entity)

    def find_by_criteria(self, **kwargs: Any) -> List[T]:
        """Поиск по критериям."""
        entities = self.get_all()
        result = []
        for entity in entities:
            if self._matches_criteria(entity, kwargs):
                result.append(entity)
        return result

    def find_one_by_criteria(self, **kwargs: Any) -> Optional[T]:
        """Поиск одной сущности по критериям."""
        entities = self.find_by_criteria(**kwargs)
        return entities[0] if entities else None

    def update_by_criteria(
        self, criteria: Dict[str, Any], updates: Dict[str, Any]
    ) -> int:
        """Обновление сущностей по критериям."""
        entities = self.find_by_criteria(**criteria)
        updated_count = 0
        for entity in entities:
            try:
                self._apply_updates(entity, updates)
                self.update(entity)
                updated_count += 1
            except Exception as e:
                raise RepositoryError(f"Failed to update entity: {str(e)}")
        return updated_count

    def delete_by_criteria(self, **kwargs: Any) -> int:
        """Удаление сущностей по критериям."""
        entities = self.find_by_criteria(**kwargs)
        deleted_count = 0
        for entity in entities:
            try:
                entity_id = self._get_entity_id(entity)
                if self.delete(entity_id):
                    deleted_count += 1
            except Exception as e:
                raise RepositoryError(f"Failed to delete entity: {str(e)}")
        return deleted_count

    def bulk_create(self, entities: List[T]) -> List[T]:
        """Массовое создание сущностей."""
        created_entities = []
        for entity in entities:
            try:
                created_entity = self.create(entity)
                created_entities.append(created_entity)
            except Exception as e:
                raise RepositoryError(f"Failed to create entity in bulk: {str(e)}")
        return created_entities

    def bulk_update(self, entities: List[T]) -> List[T]:
        """Массовое обновление сущностей."""
        updated_entities = []
        for entity in entities:
            try:
                updated_entity = self.update(entity)
                updated_entities.append(updated_entity)
            except Exception as e:
                raise RepositoryError(f"Failed to update entity in bulk: {str(e)}")
        return updated_entities

    def bulk_delete(self, entity_ids: List[str | UUID]) -> int:
        """Массовое удаление сущностей."""
        deleted_count = 0
        for entity_id in entity_ids:
            try:
                if self.delete(entity_id):
                    deleted_count += 1
            except Exception as e:
                raise RepositoryError(f"Failed to delete entity in bulk: {str(e)}")
        return deleted_count

    def get_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        """Получение пагинированных данных."""
        offset = (page - 1) * page_size
        entities = self.get_all(limit=page_size, offset=offset)
        total_count = self.count()
        if sort_by:
            entities = self._sort_entities(entities, sort_by, sort_order)
        return {
            "entities": entities,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size,
            "has_next": page * page_size < total_count,
            "has_prev": page > 1,
        }

    def search(
        self, query: str, fields: List[str], limit: Optional[int] = None
    ) -> List[T]:
        """Поиск по текстовому запросу."""
        entities = self.get_all()
        results = []
        for entity in entities:
            if self._matches_search_query(entity, query, fields):
                results.append(entity)
                if limit and len(results) >= limit:
                    break
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики репозитория."""
        total_count = self.count()
        entities = self.get_all()
        return {
            "total_count": total_count,
            "cache_size": len(self._cache),
            "cache_enabled": self._cache_enabled,
            "cache_ttl": self._cache_ttl,
        }

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._cache.clear()

    def enable_cache(self) -> None:
        """Включение кэша."""
        self._cache_enabled = True

    def disable_cache(self) -> None:
        """Отключение кэша."""
        self._cache_enabled = False
        self.clear_cache()

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Установка времени жизни кэша."""
        self._cache_ttl = ttl_seconds

    def _get_cache_key(self, entity_id: str | UUID) -> str:
        """Получение ключа кэша."""
        return f"{self.__class__.__name__}:{entity_id}"

    def _get_from_cache(self, entity_id: str | UUID) -> Optional[T]:
        """Получение из кэша."""
        if not self._cache_enabled:
            return None
        cache_key = self._get_cache_key(entity_id)
        return self._cache.get(cache_key)

    def _set_cache(self, entity_id: str | UUID, entity: T) -> None:
        """Установка в кэш."""
        if not self._cache_enabled:
            return
        cache_key = self._get_cache_key(entity_id)
        self._cache[cache_key] = entity

    def _remove_from_cache(self, entity_id: str | UUID) -> None:
        """Удаление из кэша."""
        cache_key = self._get_cache_key(entity_id)
        self._cache.pop(cache_key, None)

    def _get_entity_id(self, entity: T) -> str | UUID:
        """Получение ID сущности."""
        if hasattr(entity, "id"):
            entity_id: Any = entity.id
            if isinstance(entity_id, (str, UUID)):
                return entity_id
            else:
                return str(entity_id)
        raise RepositoryError("Entity does not have 'id' attribute")

    def _matches_criteria(self, entity: T, criteria: Dict[str, Any]) -> bool:
        """Проверка соответствия критериям."""
        for key, value in criteria.items():
            if not hasattr(entity, key):
                return False
            entity_value = getattr(entity, key)
            if entity_value != value:
                return False
        return True

    def _matches_search_query(self, entity: T, query: str, fields: List[str]) -> bool:
        """Проверка соответствия поисковому запросу."""
        query_lower = query.lower()
        for field in fields:
            if not hasattr(entity, field):
                continue
            field_value = getattr(entity, field)
            if field_value and query_lower in str(field_value).lower():
                return True
        return False

    def _apply_updates(self, entity: T, updates: Dict[str, Any]) -> None:
        """Применение обновлений к сущности."""
        for key, value in updates.items():
            if hasattr(entity, key):
                setattr(entity, key, value)

    def _sort_entities(
        self, entities: List[T], sort_by: str, sort_order: str
    ) -> List[T]:
        """Сортировка сущностей."""
        reverse = sort_order.lower() == "desc"

        def get_sort_key(entity: T) -> Any:
            if hasattr(entity, sort_by):
                return getattr(entity, sort_by)
            return None

        return sorted(entities, key=get_sort_key, reverse=reverse)

    def __iter__(self) -> Iterator[T]:
        """Итератор по всем сущностям."""
        return iter(self.get_all())

    def __len__(self) -> int:
        """Количество сущностей."""
        return self.count()
