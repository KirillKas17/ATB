"""
Юнит-тесты для StrategyStorage.
"""
import json
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from sqlmodel import Session, select
from domain.evolution.strategy_fitness import StrategyEvaluationResult
from domain.evolution.strategy_model import EvolutionContext, EvolutionStatus, StrategyCandidate
from infrastructure.evolution.exceptions import (
    ConnectionError, QueryError, StorageError, ValidationError
)
from infrastructure.evolution.models import (
    EvolutionContextModel, StrategyCandidateModel, StrategyEvaluationModel
)
from infrastructure.evolution.storage import StrategyStorage
class TestStrategyStorage:
    """Тесты для StrategyStorage."""
    def test_init_success(self, temp_db_path: str) -> None:
        """Тест успешной инициализации хранилища."""
        storage = StrategyStorage(temp_db_path)
        assert storage.db_path == temp_db_path
        assert storage.engine is not None
    def test_init_connection_error(self, monkeypatch) -> None:
        """Тест ошибки подключения при инициализации."""
        def mock_create_engine(*args, **kwargs) -> Any:
            raise Exception("Connection failed")
        monkeypatch.setattr("infrastructure.evolution.storage.create_engine", mock_create_engine)
        with pytest.raises(ConnectionError) as exc_info:
            StrategyStorage("invalid_path.db")
        assert "Не удалось подключиться к БД" in str(exc_info.value)
    def test_create_tables_error(self, temp_db_path: str, monkeypatch) -> None:
        """Тест ошибки создания таблиц."""
        def mock_create_all(*args, **kwargs) -> Any:
            raise Exception("Table creation failed")
        monkeypatch.setattr("sqlmodel.SQLModel.metadata.create_all", mock_create_all)
        with pytest.raises(QueryError) as exc_info:
            StrategyStorage(temp_db_path)
        assert "Не удалось создать таблицы" in str(exc_info.value)
    def test_save_strategy_candidate_success(self, storage: StrategyStorage, sample_candidate: StrategyCandidate) -> None:
        """Тест успешного сохранения кандидата стратегии."""
        storage.save_strategy_candidate(sample_candidate)
        # Проверить, что кандидат сохранен
        with Session(storage.engine) as session:
            model = session.exec(
                select(StrategyCandidateModel).where(
                    StrategyCandidateModel.id == str(sample_candidate.id)
                )
            ).first()
            assert model is not None
            assert model.name == sample_candidate.name
            assert model.strategy_type == sample_candidate.strategy_type.value
    def test_save_strategy_candidate_update(self, storage: StrategyStorage, sample_candidate: StrategyCandidate) -> None:
        """Тест обновления существующего кандидата стратегии."""
        # Сохранить первый раз
        storage.save_strategy_candidate(sample_candidate)
        # Изменить и сохранить второй раз
        sample_candidate.name = "Updated Strategy"
        sample_candidate.description = "Updated description"
        storage.save_strategy_candidate(sample_candidate)
        # Проверить обновление
        with Session(storage.engine) as session:
            model = session.exec(
                select(StrategyCandidateModel).where(
                    StrategyCandidateModel.id == str(sample_candidate.id)
                )
            ).first()
            if model is not None:
                assert model.name == "Updated Strategy"
                assert model.description == "Updated description"
    def test_save_strategy_candidate_validation_error(self, storage: StrategyStorage) -> None:
        """Тест ошибки валидации при сохранении кандидата."""
        invalid_candidate = StrategyCandidate(
            id=uuid4(),
            name="",  # Пустое имя - ошибка валидации
            position_size_pct=Decimal("1.5")  # > 1 - ошибка валидации
        )
        with pytest.raises(ValidationError) as exc_info:
            storage.save_strategy_candidate(invalid_candidate)
        assert "Имя кандидата стратегии обязательно" in str(exc_info.value)
    def test_get_strategy_candidate_success(self, storage: StrategyStorage, sample_candidate: StrategyCandidate) -> None:
        """Тест успешного получения кандидата стратегии."""
        storage.save_strategy_candidate(sample_candidate)
        retrieved = storage.get_strategy_candidate(sample_candidate.id)
        assert retrieved is not None
        assert retrieved.id == sample_candidate.id
        assert retrieved.name == sample_candidate.name
        assert retrieved.strategy_type == sample_candidate.strategy_type
    def test_get_strategy_candidate_not_found(self, storage: StrategyStorage) -> None:
        """Тест получения несуществующего кандидата стратегии."""
        retrieved = storage.get_strategy_candidate(uuid4())
        assert retrieved is None
    def test_get_strategy_candidates_all(self, storage: StrategyStorage, sample_candidate: StrategyCandidate) -> None:
        """Тест получения всех кандидатов стратегий."""
        storage.save_strategy_candidate(sample_candidate)
        candidates = storage.get_strategy_candidates()
        assert len(candidates) == 1
        assert candidates[0].id == sample_candidate.id
    def test_get_strategy_candidates_filtered(self, storage: StrategyStorage, sample_candidate: StrategyCandidate) -> None:
        """Тест получения кандидатов с фильтрами."""
        storage.save_strategy_candidate(sample_candidate)
        # Фильтр по статусу
        candidates = storage.get_strategy_candidates(status=EvolutionStatus.GENERATED)
        assert len(candidates) == 1
        # Фильтр по несуществующему статусу
        candidates = storage.get_strategy_candidates(status=EvolutionStatus.APPROVED)
        assert len(candidates) == 0
        # Фильтр по поколению
        candidates = storage.get_strategy_candidates(generation=1)
        assert len(candidates) == 1
        # Фильтр по несуществующему поколению
        candidates = storage.get_strategy_candidates(generation=999)
        assert len(candidates) == 0
    def test_get_strategy_candidates_limit(self, storage: StrategyStorage) -> None:
        """Тест ограничения количества результатов."""
        # Создать несколько кандидатов
        for i in range(5):
            candidate = StrategyCandidate(
                id=uuid4(),
                name=f"Strategy {i}",
                generation=i
            )
            storage.save_strategy_candidate(candidate)
        candidates = storage.get_strategy_candidates(limit=3)
        assert len(candidates) == 3
    def test_save_evaluation_result_success(self, storage: StrategyStorage, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест успешного сохранения результата оценки."""
        storage.save_evaluation_result(sample_evaluation)
        # Проверить, что результат сохранен
        with Session(storage.engine) as session:
            model = session.exec(
                select(StrategyEvaluationModel).where(
                    StrategyEvaluationModel.id == str(sample_evaluation.id)
                )
            ).first()
            assert model is not None
            assert model.strategy_id == str(sample_evaluation.strategy_id)
            assert model.total_trades == sample_evaluation.total_trades
    def test_save_evaluation_result_validation_error(self, storage: StrategyStorage) -> None:
        """Тест ошибки валидации при сохранении результата оценки."""
        invalid_evaluation = StrategyEvaluationResult(
            id=uuid4(),
            strategy_id=uuid4(),
            total_trades=-1  # Отрицательное значение - ошибка валидации
        )
        with pytest.raises(ValidationError) as exc_info:
            storage.save_evaluation_result(invalid_evaluation)
        assert "Общее количество сделок не может быть отрицательным" in str(exc_info.value)
    def test_get_evaluation_result_success(self, storage: StrategyStorage, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест успешного получения результата оценки."""
        storage.save_evaluation_result(sample_evaluation)
        retrieved = storage.get_evaluation_result(sample_evaluation.id)
        assert retrieved is not None
        assert retrieved.id == sample_evaluation.id
        assert retrieved.strategy_id == sample_evaluation.strategy_id
        assert retrieved.total_trades == sample_evaluation.total_trades
    def test_get_evaluation_result_not_found(self, storage: StrategyStorage) -> None:
        """Тест получения несуществующего результата оценки."""
        retrieved = storage.get_evaluation_result(uuid4())
        assert retrieved is None
    def test_get_evaluation_results_filtered(self, storage: StrategyStorage, sample_evaluation: StrategyEvaluationResult) -> None:
        """Тест получения результатов оценки с фильтрами."""
        storage.save_evaluation_result(sample_evaluation)
        # Фильтр по ID стратегии
        evaluations = storage.get_evaluation_results(strategy_id=sample_evaluation.strategy_id)
        assert len(evaluations) == 1
        # Фильтр по статусу одобрения
        evaluations = storage.get_evaluation_results(is_approved=True)
        assert len(evaluations) == 1
        evaluations = storage.get_evaluation_results(is_approved=False)
        assert len(evaluations) == 0
    def test_save_evolution_context_success(self, storage: StrategyStorage, sample_context: EvolutionContext) -> None:
        """Тест успешного сохранения контекста эволюции."""
        storage.save_evolution_context(sample_context)
        # Проверить, что контекст сохранен
        with Session(storage.engine) as session:
            model = session.exec(
                select(EvolutionContextModel).where(
                    EvolutionContextModel.id == str(sample_context.id)
                )
            ).first()
            assert model is not None
            assert model.name == sample_context.name
            assert model.population_size == sample_context.population_size
    def test_save_evolution_context_validation_error(self, storage: StrategyStorage) -> None:
        """Тест ошибки валидации при сохранении контекста эволюции."""
        invalid_context = EvolutionContext(
            id=uuid4(),
            name="",  # Пустое имя - ошибка валидации
            population_size=0  # Нулевой размер - ошибка валидации
        )
        with pytest.raises(ValidationError) as exc_info:
            storage.save_evolution_context(invalid_context)
        assert "Имя контекста эволюции обязательно" in str(exc_info.value)
    def test_get_evolution_context_success(self, storage: StrategyStorage, sample_context: EvolutionContext) -> None:
        """Тест успешного получения контекста эволюции."""
        storage.save_evolution_context(sample_context)
        retrieved = storage.get_evolution_context(sample_context.id)
        assert retrieved is not None
        assert retrieved.id == sample_context.id
        assert retrieved.name == sample_context.name
        assert retrieved.population_size == sample_context.population_size
    def test_get_evolution_context_not_found(self, storage: StrategyStorage) -> None:
        """Тест получения несуществующего контекста эволюции."""
        retrieved = storage.get_evolution_context(uuid4())
        assert retrieved is None
    def test_get_evolution_contexts(self, storage: StrategyStorage, sample_context: EvolutionContext) -> None:
        """Тест получения списка контекстов эволюции."""
        storage.save_evolution_context(sample_context)
        contexts = storage.get_evolution_contexts()
        assert len(contexts) == 1
        assert contexts[0].id == sample_context.id
    def test_get_evolution_contexts_limit(self, storage: StrategyStorage) -> None:
        """Тест ограничения количества контекстов эволюции."""
        # Создать несколько контекстов
        for i in range(5):
            context = EvolutionContext(
                id=uuid4(),
                name=f"Context {i}",
                population_size=50
            )
            storage.save_evolution_context(context)
        contexts = storage.get_evolution_contexts(limit=3)
        assert len(contexts) == 3
    def test_get_statistics(self, storage: StrategyStorage, sample_candidate: StrategyCandidate, 
                          sample_evaluation: StrategyEvaluationResult, sample_context: EvolutionContext) -> None:
        """Тест получения статистики хранилища."""
        # Добавить тестовые данные
        storage.save_strategy_candidate(sample_candidate)
        storage.save_evaluation_result(sample_evaluation)
        storage.save_evolution_context(sample_context)
        stats = storage.get_statistics()
        assert stats["total_candidates"] == 1
        assert stats["total_evaluations"] == 1
        assert stats["total_contexts"] == 1
        assert stats["approval_rate"] == 1.0
        assert "generated" in stats["candidates_by_status"]
        assert stats["candidates_by_status"]["generated"] == 1
    def test_cleanup_old_data(self, storage: StrategyStorage) -> None:
        """Тест очистки старых данных."""
        # Создать старые данные
        old_candidate = StrategyCandidate(
            id=uuid4(),
            name="Old Strategy",
            created_at=datetime.now() - timedelta(days=40)
        )
        storage.save_strategy_candidate(old_candidate)
        old_evaluation = StrategyEvaluationResult(
            id=uuid4(),
            strategy_id=uuid4(),
            evaluation_time=datetime.now() - timedelta(days=40)
        )
        storage.save_evaluation_result(old_evaluation)
        # Создать новые данные
        new_candidate = StrategyCandidate(
            id=uuid4(),
            name="New Strategy",
            created_at=datetime.now()
        )
        storage.save_strategy_candidate(new_candidate)
        # Очистить старые данные
        deleted_count = storage.cleanup_old_data(days_to_keep=30)
        assert deleted_count == 2
        # Проверить, что новые данные остались
        candidates = storage.get_strategy_candidates()
        assert len(candidates) == 1
        assert candidates[0].name == "New Strategy"
    def test_export_data(self, storage: StrategyStorage, sample_candidate: StrategyCandidate,
                        sample_evaluation: StrategyEvaluationResult, sample_context: EvolutionContext) -> None:
        """Тест экспорта данных."""
        # Добавить тестовые данные
        storage.save_strategy_candidate(sample_candidate)
        storage.save_evaluation_result(sample_evaluation)
        storage.save_evolution_context(sample_context)
        # Экспортировать данные
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_path = f.name
        try:
            storage.export_data(export_path)
            # Проверить экспортированные данные
            with open(export_path, "r", encoding="utf-8") as f:
                export_data = json.load(f)
            assert "candidates" in export_data
            assert "evaluations" in export_data
            assert "contexts" in export_data
            assert "export_time" in export_data
            assert len(export_data["candidates"]) == 1
            assert len(export_data["evaluations"]) == 1
            assert len(export_data["contexts"]) == 1
        finally:
            Path(export_path).unlink(missing_ok=True)
    def test_import_data(self, storage: StrategyStorage, sample_candidate: StrategyCandidate,
                        sample_context: EvolutionContext) -> None:
        """Тест импорта данных."""
        # Подготовить данные для импорта
        import_data = {
            "candidates": [sample_candidate.to_dict()],
            "contexts": [sample_context.to_dict()],
            "evaluations": []
        }
        # Создать временный файл
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(import_data, f)
            import_path = f.name
        try:
            # Импортировать данные
            imported_count = storage.import_data(import_path)
            assert imported_count == 2
            # Проверить, что данные импортированы
            candidates = storage.get_strategy_candidates()
            contexts = storage.get_evolution_contexts()
            assert len(candidates) == 1
            assert len(contexts) == 1
        finally:
            Path(import_path).unlink(missing_ok=True)
    def test_storage_error_handling(self, storage: StrategyStorage, monkeypatch) -> None:
        """Тест обработки ошибок хранилища."""
        def mock_session_exec(*args, **kwargs) -> Any:
            raise Exception("Database error")
        monkeypatch.setattr(Session, "exec", mock_session_exec)
        with pytest.raises(StorageError) as exc_info:
            storage.get_strategy_candidate(uuid4())
        assert "Не удалось получить кандидата стратегии" in str(exc_info.value) 
