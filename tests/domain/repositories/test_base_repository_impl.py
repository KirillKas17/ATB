"""
Unit тесты для domain/repositories/base_repository_impl.py.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timezone

from domain.repositories.base_repository_impl import BaseRepositoryImpl
from domain.exceptions.base_exceptions import EntityNotFoundError, ValidationError


# Временная заглушка для BaseEntity
class BaseEntity:
    """Базовая сущность."""

    def __init__(self, id: str):
        self.id = id


class MockEntity(BaseEntity):
    """Тестовая сущность для тестирования репозитория."""

    def __init__(self, id: str, name: str, value: Decimal):
        super().__init__(id)
        self._name = name
        self._value = value

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> Decimal:
        return self._value

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self._name, "value": str(self._value)}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockEntity":
        return cls(id=data["id"], name=data["name"], value=Decimal(data["value"]))

    def __eq__(self, other):
        if not isinstance(other, MockEntity):
            return False
        return self.id == other.id and self._name == other._name and self._value == other._value

    def __hash__(self):
        return hash((self.id, self._name, self._value))


class TestBaseRepositoryImpl:
    """Тесты для BaseRepositoryImpl."""

    @pytest.fixture
    def repository(self):
        """Создание репозитория."""
        return BaseRepositoryImpl[MockEntity]()

    @pytest.fixture
    def sample_entities(self) -> List[MockEntity]:
        """Тестовые сущности."""
        return [
            MockEntity("1", "Entity 1", Decimal("100.00")),
            MockEntity("2", "Entity 2", Decimal("200.00")),
            MockEntity("3", "Entity 3", Decimal("300.00")),
        ]

    def test_add_entity(self, repository, sample_entities):
        """Тест добавления сущности."""
        entity = sample_entities[0]

        result = repository.add(entity)

        assert result == entity
        assert repository.get_by_id("1") == entity
        assert len(repository.get_all()) == 1

    def test_add_multiple_entities(self, repository, sample_entities):
        """Тест добавления нескольких сущностей."""
        for entity in sample_entities:
            repository.add(entity)

        all_entities = repository.get_all()
        assert len(all_entities) == 3

        for entity in sample_entities:
            assert entity in all_entities

    def test_add_duplicate_entity(self, repository, sample_entities):
        """Тест добавления дублирующейся сущности."""
        entity = sample_entities[0]
        repository.add(entity)

        # Попытка добавить сущность с тем же ID
        duplicate_entity = MockEntity("1", "Duplicate", Decimal("999.00"))

        with pytest.raises(ValidationError, match="Entity with id 1 already exists"):
            repository.add(duplicate_entity)

    def test_get_by_id_existing(self, repository, sample_entities):
        """Тест получения существующей сущности по ID."""
        entity = sample_entities[0]
        repository.add(entity)

        result = repository.get_by_id("1")

        assert result == entity
        assert result.id == "1"
        assert result.name == "Entity 1"
        assert result.value == Decimal("100.00")

    def test_get_by_id_not_found(self, repository):
        """Тест получения несуществующей сущности по ID."""
        with pytest.raises(EntityNotFoundError, match="Entity with id 999 not found"):
            repository.get_by_id("999")

    def test_get_all_empty(self, repository):
        """Тест получения всех сущностей из пустого репозитория."""
        result = repository.get_all()

        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_all_with_entities(self, repository, sample_entities):
        """Тест получения всех сущностей."""
        for entity in sample_entities:
            repository.add(entity)

        result = repository.get_all()

        assert len(result) == 3
        for entity in sample_entities:
            assert entity in result

    def test_update_existing_entity(self, repository, sample_entities):
        """Тест обновления существующей сущности."""
        entity = sample_entities[0]
        repository.add(entity)

        updated_entity = MockEntity("1", "Updated Entity", Decimal("150.00"))
        result = repository.update(updated_entity)

        assert result == updated_entity
        assert repository.get_by_id("1") == updated_entity
        assert repository.get_by_id("1").name == "Updated Entity"
        assert repository.get_by_id("1").value == Decimal("150.00")

    def test_update_not_found(self, repository, sample_entities):
        """Тест обновления несуществующей сущности."""
        entity = MockEntity("999", "Not Found", Decimal("999.00"))

        with pytest.raises(EntityNotFoundError, match="Entity with id 999 not found"):
            repository.update(entity)

    def test_delete_existing_entity(self, repository, sample_entities):
        """Тест удаления существующей сущности."""
        entity = sample_entities[0]
        repository.add(entity)

        result = repository.delete("1")

        assert result == entity
        with pytest.raises(EntityNotFoundError):
            repository.get_by_id("1")
        assert len(repository.get_all()) == 0

    def test_delete_not_found(self, repository):
        """Тест удаления несуществующей сущности."""
        with pytest.raises(EntityNotFoundError, match="Entity with id 999 not found"):
            repository.delete("999")

    def test_exists_true(self, repository, sample_entities):
        """Тест проверки существования сущности (существует)."""
        entity = sample_entities[0]
        repository.add(entity)

        assert repository.exists("1") is True

    def test_exists_false(self, repository):
        """Тест проверки существования сущности (не существует)."""
        assert repository.exists("999") is False

    def test_count_empty(self, repository):
        """Тест подсчета сущностей в пустом репозитории."""
        assert repository.count() == 0

    def test_count_with_entities(self, repository, sample_entities):
        """Тест подсчета сущностей."""
        for entity in sample_entities:
            repository.add(entity)

        assert repository.count() == 3

    def test_clear(self, repository, sample_entities):
        """Тест очистки репозитория."""
        for entity in sample_entities:
            repository.add(entity)

        repository.clear()

        assert repository.count() == 0
        assert len(repository.get_all()) == 0

    def test_find_by_criteria(self, repository, sample_entities):
        """Тест поиска по критериям."""
        for entity in sample_entities:
            repository.add(entity)

        # Поиск по имени
        results = repository.find_by(lambda e: e.name == "Entity 1")
        assert len(results) == 1
        assert results[0] == sample_entities[0]

        # Поиск по значению
        results = repository.find_by(lambda e: e.value > Decimal("150.00"))
        assert len(results) == 2
        assert sample_entities[1] in results
        assert sample_entities[2] in results

    def test_find_by_criteria_no_results(self, repository, sample_entities):
        """Тест поиска по критериям без результатов."""
        for entity in sample_entities:
            repository.add(entity)

        results = repository.find_by(lambda e: e.name == "Non-existent")
        assert len(results) == 0

    def test_find_one_by_criteria(self, repository, sample_entities):
        """Тест поиска одной сущности по критериям."""
        for entity in sample_entities:
            repository.add(entity)

        result = repository.find_one_by(lambda e: e.name == "Entity 1")
        assert result == sample_entities[0]

    def test_find_one_by_criteria_not_found(self, repository, sample_entities):
        """Тест поиска одной сущности по критериям (не найдена)."""
        for entity in sample_entities:
            repository.add(entity)

        result = repository.find_one_by(lambda e: e.name == "Non-existent")
        assert result is None

    def test_find_one_by_criteria_multiple_results(self, repository, sample_entities):
        """Тест поиска одной сущности при множественных результатах."""
        # Добавляем две сущности с одинаковым значением
        entity1 = MockEntity("1", "Entity 1", Decimal("100.00"))
        entity2 = MockEntity("2", "Entity 2", Decimal("100.00"))
        repository.add(entity1)
        repository.add(entity2)

        # Должен вернуть первую найденную
        result = repository.find_one_by(lambda e: e.value == Decimal("100.00"))
        assert result is not None
        assert result.value == Decimal("100.00")

    def test_get_page(self, repository, sample_entities):
        """Тест получения страницы сущностей."""
        for entity in sample_entities:
            repository.add(entity)

        # Первая страница с размером 2
        page = repository.get_page(page=1, size=2)
        assert len(page) == 2
        assert page[0] in sample_entities
        assert page[1] in sample_entities

        # Вторая страница
        page = repository.get_page(page=2, size=2)
        assert len(page) == 1
        assert page[0] in sample_entities

    def test_get_page_empty(self, repository):
        """Тест получения страницы из пустого репозитория."""
        page = repository.get_page(page=1, size=10)
        assert len(page) == 0

    def test_get_page_invalid_page(self, repository, sample_entities):
        """Тест получения несуществующей страницы."""
        for entity in sample_entities:
            repository.add(entity)

        page = repository.get_page(page=999, size=10)
        assert len(page) == 0

    def test_get_page_invalid_size(self, repository, sample_entities):
        """Тест получения страницы с невалидным размером."""
        for entity in sample_entities:
            repository.add(entity)

        # Отрицательный размер
        page = repository.get_page(page=1, size=-1)
        assert len(page) == 0

        # Нулевой размер
        page = repository.get_page(page=1, size=0)
        assert len(page) == 0

    def test_get_ids(self, repository, sample_entities):
        """Тест получения всех ID."""
        for entity in sample_entities:
            repository.add(entity)

        ids = repository.get_ids()

        assert len(ids) == 3
        assert "1" in ids
        assert "2" in ids
        assert "3" in ids

    def test_get_ids_empty(self, repository):
        """Тест получения ID из пустого репозитория."""
        ids = repository.get_ids()
        assert len(ids) == 0

    def test_bulk_add(self, repository, sample_entities):
        """Тест массового добавления сущностей."""
        repository.bulk_add(sample_entities)

        assert repository.count() == 3
        for entity in sample_entities:
            assert repository.exists(entity.id)

    def test_bulk_add_empty_list(self, repository):
        """Тест массового добавления пустого списка."""
        repository.bulk_add([])
        assert repository.count() == 0

    def test_bulk_add_duplicates(self, repository, sample_entities):
        """Тест массового добавления с дубликатами."""
        repository.add(sample_entities[0])

        with pytest.raises(ValidationError):
            repository.bulk_add(sample_entities)

    def test_bulk_update(self, repository, sample_entities):
        """Тест массового обновления сущностей."""
        for entity in sample_entities:
            repository.add(entity)

        updated_entities = [
            MockEntity("1", "Updated 1", Decimal("150.00")),
            MockEntity("2", "Updated 2", Decimal("250.00")),
            MockEntity("3", "Updated 3", Decimal("350.00")),
        ]

        repository.bulk_update(updated_entities)

        for entity in updated_entities:
            stored = repository.get_by_id(entity.id)
            assert stored.name == entity.name
            assert stored.value == entity.value

    def test_bulk_update_not_found(self, repository, sample_entities):
        """Тест массового обновления с несуществующими сущностями."""
        for entity in sample_entities:
            repository.add(entity)

        updated_entities = [
            MockEntity("1", "Updated 1", Decimal("150.00")),
            MockEntity("999", "Not Found", Decimal("999.00")),
        ]

        with pytest.raises(EntityNotFoundError):
            repository.bulk_update(updated_entities)

    def test_bulk_delete(self, repository, sample_entities):
        """Тест массового удаления сущностей."""
        for entity in sample_entities:
            repository.add(entity)

        ids_to_delete = ["1", "3"]
        deleted = repository.bulk_delete(ids_to_delete)

        assert len(deleted) == 2
        assert repository.count() == 1
        assert repository.exists("2")
        assert not repository.exists("1")
        assert not repository.exists("3")

    def test_bulk_delete_not_found(self, repository, sample_entities):
        """Тест массового удаления с несуществующими ID."""
        for entity in sample_entities:
            repository.add(entity)

        ids_to_delete = ["1", "999"]

        with pytest.raises(EntityNotFoundError):
            repository.bulk_delete(ids_to_delete)

    def test_get_by_ids(self, repository, sample_entities):
        """Тест получения сущностей по списку ID."""
        for entity in sample_entities:
            repository.add(entity)

        ids = ["1", "3"]
        entities = repository.get_by_ids(ids)

        assert len(entities) == 2
        assert any(e.id == "1" for e in entities)
        assert any(e.id == "3" for e in entities)

    def test_get_by_ids_not_found(self, repository, sample_entities):
        """Тест получения сущностей по несуществующим ID."""
        for entity in sample_entities:
            repository.add(entity)

        ids = ["1", "999"]

        with pytest.raises(EntityNotFoundError):
            repository.get_by_ids(ids)

    def test_get_by_ids_empty_list(self, repository):
        """Тест получения сущностей по пустому списку ID."""
        entities = repository.get_by_ids([])
        assert len(entities) == 0

    def test_save_new_entity(self, repository, sample_entities):
        """Тест сохранения новой сущности."""
        entity = sample_entities[0]

        result = repository.save(entity)

        assert result == entity
        assert repository.exists(entity.id)

    def test_save_existing_entity(self, repository, sample_entities):
        """Тест сохранения существующей сущности."""
        entity = sample_entities[0]
        repository.add(entity)

        updated_entity = MockEntity("1", "Updated", Decimal("999.00"))
        result = repository.save(updated_entity)

        assert result == updated_entity
        stored = repository.get_by_id("1")
        assert stored.name == "Updated"
        assert stored.value == Decimal("999.00")

    def test_repository_persistence(self, repository, sample_entities):
        """Тест персистентности репозитория."""
        for entity in sample_entities:
            repository.add(entity)

        # Проверяем, что данные сохранились
        assert repository.count() == 3

        # Получаем все сущности
        all_entities = repository.get_all()
        assert len(all_entities) == 3

        # Проверяем каждую сущность
        for entity in sample_entities:
            stored = repository.get_by_id(entity.id)
            assert stored == entity
            assert stored.name == entity.name
            assert stored.value == entity.value

    def test_repository_immutability(self, repository, sample_entities):
        """Тест неизменяемости данных в репозитории."""
        entity = sample_entities[0]
        repository.add(entity)

        # Получаем сущность
        stored = repository.get_by_id(entity.id)

        # Изменяем полученную сущность (не должно влиять на репозиторий)
        # Поскольку MockEntity неизменяема, создаем новую
        modified_entity = MockEntity(entity.id, "Modified", Decimal("999.00"))

        # Проверяем, что в репозитории ничего не изменилось
        original = repository.get_by_id(entity.id)
        assert original.name == "Entity 1"
        assert original.value == Decimal("100.00")

    def test_repository_thread_safety(self, repository, sample_entities):
        """Тест потокобезопасности репозитория."""
        import threading
        import time

        def add_entities():
            for i in range(10):
                entity = MockEntity(f"thread_{i}", f"Thread Entity {i}", Decimal(str(i)))
                repository.add(entity)
                time.sleep(0.001)

        # Создаем несколько потоков
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=add_entities)
            threads.append(thread)
            thread.start()

        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()

        # Проверяем, что все сущности добавлены
        assert repository.count() == 30  # 3 потока * 10 сущностей
