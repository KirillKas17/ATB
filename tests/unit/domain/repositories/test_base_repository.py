"""
Unit тесты для BaseRepository.

Покрывает:
- Базовые операции CRUD
- Валидацию данных
- Обработку ошибок
- Абстрактные методы репозитория
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from domain.repositories.base_repository import BaseRepository
from domain.exceptions import RepositoryError, EntityNotFoundError


class MockEntity:
    """Мок-сущность для тестирования."""

    def __init__(self, id: str, name: str, created_at: str = None):
        self.id = id
        self.name = name
        self.created_at = created_at or datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MockEntity":
        return cls(id=data["id"], name=data["name"], created_at=data.get("created_at"))


class TestBaseRepository:
    """Тесты для BaseRepository."""

    @pytest.fixture
    def base_repository(self):
        """Создание экземпляра BaseRepository."""
        return BaseRepository()

    @pytest.fixture
    def sample_entity_data(self) -> Dict[str, Any]:
        """Тестовые данные сущности."""
        return {"id": "test_001", "name": "Test Entity", "created_at": "2024-01-01T00:00:00Z"}

    @pytest.fixture
    def sample_entities_data(self) -> List[Dict[str, Any]]:
        """Тестовые данные нескольких сущностей."""
        return [
            {"id": "test_001", "name": "Test Entity 1", "created_at": "2024-01-01T00:00:00Z"},
            {"id": "test_002", "name": "Test Entity 2", "created_at": "2024-01-01T01:00:00Z"},
            {"id": "test_003", "name": "Test Entity 3", "created_at": "2024-01-01T02:00:00Z"},
        ]

    def test_base_repository_creation(self, base_repository):
        """Тест создания базового репозитория."""
        assert base_repository is not None
        assert isinstance(base_repository, BaseRepository)

    @pytest.mark.asyncio
    async def test_save_entity(self, base_repository, sample_entity_data):
        """Тест сохранения сущности."""
        entity = MockEntity.from_dict(sample_entity_data)

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.save(entity)

    @pytest.mark.asyncio
    async def test_find_by_id(self, base_repository):
        """Тест поиска по ID."""
        entity_id = "test_001"

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.find_by_id(entity_id)

    @pytest.mark.asyncio
    async def test_find_all(self, base_repository):
        """Тест поиска всех сущностей."""
        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.find_all()

    @pytest.mark.asyncio
    async def test_find_by_criteria(self, base_repository):
        """Тест поиска по критериям."""
        criteria = {"name": "Test"}

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.find_by_criteria(criteria)

    @pytest.mark.asyncio
    async def test_update_entity(self, base_repository, sample_entity_data):
        """Тест обновления сущности."""
        entity = MockEntity.from_dict(sample_entity_data)

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.update(entity)

    @pytest.mark.asyncio
    async def test_delete_entity(self, base_repository):
        """Тест удаления сущности."""
        entity_id = "test_001"

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.delete(entity_id)

    @pytest.mark.asyncio
    async def test_exists(self, base_repository):
        """Тест проверки существования сущности."""
        entity_id = "test_001"

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.exists(entity_id)

    @pytest.mark.asyncio
    async def test_count(self, base_repository):
        """Тест подсчета сущностей."""
        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.count()

    @pytest.mark.asyncio
    async def test_count_by_criteria(self, base_repository):
        """Тест подсчета по критериям."""
        criteria = {"name": "Test"}

        # Тестируем абстрактный метод
        with pytest.raises(NotImplementedError):
            await base_repository.count_by_criteria(criteria)

    def test_validate_entity(self, base_repository, sample_entity_data):
        """Тест валидации сущности."""
        entity = MockEntity.from_dict(sample_entity_data)

        # Тестируем базовую валидацию
        is_valid = base_repository.validate_entity(entity)
        assert is_valid is True

    def test_validate_entity_none(self, base_repository):
        """Тест валидации None сущности."""
        with pytest.raises(ValueError):
            base_repository.validate_entity(None)

    def test_validate_entity_invalid_type(self, base_repository):
        """Тест валидации сущности неверного типа."""
        invalid_entity = {"id": "test", "name": "test"}

        with pytest.raises(ValueError):
            base_repository.validate_entity(invalid_entity)

    def test_validate_entity_id(self, base_repository, sample_entity_data):
        """Тест валидации ID сущности."""
        entity = MockEntity.from_dict(sample_entity_data)

        # Тестируем валидацию ID
        is_valid = base_repository.validate_entity_id(entity.id)
        assert is_valid is True

    def test_validate_entity_id_empty(self, base_repository):
        """Тест валидации пустого ID."""
        with pytest.raises(ValueError):
            base_repository.validate_entity_id("")

    def test_validate_entity_id_none(self, base_repository):
        """Тест валидации None ID."""
        with pytest.raises(ValueError):
            base_repository.validate_entity_id(None)

    def test_validate_criteria(self, base_repository):
        """Тест валидации критериев поиска."""
        criteria = {"name": "Test", "status": "active"}

        # Тестируем валидацию критериев
        is_valid = base_repository.validate_criteria(criteria)
        assert is_valid is True

    def test_validate_criteria_none(self, base_repository):
        """Тест валидации None критериев."""
        with pytest.raises(ValueError):
            base_repository.validate_criteria(None)

    def test_validate_criteria_empty(self, base_repository):
        """Тест валидации пустых критериев."""
        with pytest.raises(ValueError):
            base_repository.validate_criteria({})

    def test_validate_criteria_invalid_type(self, base_repository):
        """Тест валидации критериев неверного типа."""
        with pytest.raises(ValueError):
            base_repository.validate_criteria("invalid_criteria")

    def test_validate_pagination_params(self, base_repository):
        """Тест валидации параметров пагинации."""
        page = 1
        size = 10

        # Тестируем валидацию параметров пагинации
        is_valid = base_repository.validate_pagination_params(page, size)
        assert is_valid is True

    def test_validate_pagination_params_invalid_page(self, base_repository):
        """Тест валидации неверной страницы."""
        with pytest.raises(ValueError):
            base_repository.validate_pagination_params(0, 10)

        with pytest.raises(ValueError):
            base_repository.validate_pagination_params(-1, 10)

    def test_validate_pagination_params_invalid_size(self, base_repository):
        """Тест валидации неверного размера."""
        with pytest.raises(ValueError):
            base_repository.validate_pagination_params(1, 0)

        with pytest.raises(ValueError):
            base_repository.validate_pagination_params(1, -1)

    def test_validate_sort_params(self, base_repository):
        """Тест валидации параметров сортировки."""
        sort_by = "name"
        sort_order = "asc"

        # Тестируем валидацию параметров сортировки
        is_valid = base_repository.validate_sort_params(sort_by, sort_order)
        assert is_valid is True

    def test_validate_sort_params_invalid_order(self, base_repository):
        """Тест валидации неверного порядка сортировки."""
        with pytest.raises(ValueError):
            base_repository.validate_sort_params("name", "invalid_order")

    def test_validate_sort_params_empty_field(self, base_repository):
        """Тест валидации пустого поля сортировки."""
        with pytest.raises(ValueError):
            base_repository.validate_sort_params("", "asc")

    def test_build_query_filters(self, base_repository):
        """Тест построения фильтров запроса."""
        criteria = {"name": "Test", "status": "active"}

        filters = base_repository.build_query_filters(criteria)

        assert isinstance(filters, dict)
        assert "name" in filters
        assert "status" in filters
        assert filters["name"] == "Test"
        assert filters["status"] == "active"

    def test_build_query_filters_empty(self, base_repository):
        """Тест построения пустых фильтров."""
        filters = base_repository.build_query_filters({})

        assert isinstance(filters, dict)
        assert len(filters) == 0

    def test_build_sort_query(self, base_repository):
        """Тест построения запроса сортировки."""
        sort_by = "name"
        sort_order = "desc"

        sort_query = base_repository.build_sort_query(sort_by, sort_order)

        assert isinstance(sort_query, dict)
        assert sort_by in sort_query
        assert sort_query[sort_by] == -1  # desc = -1

    def test_build_sort_query_asc(self, base_repository):
        """Тест построения запроса сортировки по возрастанию."""
        sort_by = "name"
        sort_order = "asc"

        sort_query = base_repository.build_sort_query(sort_by, sort_order)

        assert isinstance(sort_query, dict)
        assert sort_by in sort_query
        assert sort_query[sort_by] == 1  # asc = 1

    def test_build_pagination_query(self, base_repository):
        """Тест построения запроса пагинации."""
        page = 2
        size = 10

        pagination_query = base_repository.build_pagination_query(page, size)

        assert isinstance(pagination_query, dict)
        assert "skip" in pagination_query
        assert "limit" in pagination_query
        assert pagination_query["skip"] == 10  # (page - 1) * size
        assert pagination_query["limit"] == 10

    def test_handle_repository_error(self, base_repository):
        """Тест обработки ошибки репозитория."""
        error_message = "Database connection failed"

        with pytest.raises(RepositoryError):
            base_repository.handle_repository_error(error_message)

    def test_handle_entity_not_found(self, base_repository):
        """Тест обработки ошибки сущность не найдена."""
        entity_id = "test_001"

        with pytest.raises(EntityNotFoundError):
            base_repository.handle_entity_not_found(entity_id)

    def test_handle_validation_error(self, base_repository):
        """Тест обработки ошибки валидации."""
        field_name = "name"
        error_message = "Field is required"

        with pytest.raises(ValueError):
            base_repository.handle_validation_error(field_name, error_message)

    def test_log_operation(self, base_repository):
        """Тест логирования операции."""
        operation = "save"
        entity_id = "test_001"

        # Тестируем логирование (должно выполняться без ошибок)
        base_repository.log_operation(operation, entity_id)

    def test_log_error(self, base_repository):
        """Тест логирования ошибки."""
        operation = "save"
        error = Exception("Test error")

        # Тестируем логирование ошибки (должно выполняться без ошибок)
        base_repository.log_error(operation, error)

    def test_get_entity_type_name(self, base_repository):
        """Тест получения имени типа сущности."""
        entity_type_name = base_repository.get_entity_type_name()

        assert isinstance(entity_type_name, str)
        assert entity_type_name == "Entity"  # Базовое значение

    def test_get_repository_name(self, base_repository):
        """Тест получения имени репозитория."""
        repository_name = base_repository.get_repository_name()

        assert isinstance(repository_name, str)
        assert repository_name == "BaseRepository"  # Базовое значение
