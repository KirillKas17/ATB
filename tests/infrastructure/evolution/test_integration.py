"""
Интеграционные тесты для infrastructure/evolution модуля.
"""
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from uuid import uuid4
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.evolution.strategy_fitness import StrategyEvaluationResult, TradeResult
from domain.evolution.strategy_model import (
    EvolutionContext, EvolutionStatus, StrategyCandidate,
    EntryRule, ExitRule, FilterConfig, FilterType,
    IndicatorConfig, IndicatorType, SignalType, StrategyType
)
from infrastructure.evolution.backup import EvolutionBackup
from infrastructure.evolution.cache import EvolutionCache
from infrastructure.evolution.migration import EvolutionMigration
from infrastructure.evolution.storage import StrategyStorage
import json

class TestEvolutionIntegration:
    """Интеграционные тесты для модуля evolution."""
    def test_storage_cache_integration(self, temp_db_path: str, sample_candidate: StrategyCandidate,
                                     sample_evaluation: StrategyEvaluationResult, sample_context: EvolutionContext) -> None:
        """Тест интеграции хранилища и кэша."""
        # Инициализация компонентов
        storage = StrategyStorage(temp_db_path)
        cache = EvolutionCache({
            "cache_size": 100,
            "cache_ttl": 300,
            "cache_strategy": "lru"
        })
        # Сохранение в хранилище
        storage.save_strategy_candidate(sample_candidate)
        storage.save_evaluation_result(sample_evaluation)
        storage.save_evolution_context(sample_context)
        # Кэширование данных
        cache.set_candidate(sample_candidate.id, sample_candidate)
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        cache.set_context(sample_context.id, sample_context)
        # Проверка получения из кэша
        cached_candidate = cache.get_candidate(sample_candidate.id)
        cached_evaluation = cache.get_evaluation(sample_evaluation.id)
        cached_context = cache.get_context(sample_context.id)
        assert cached_candidate is not None
        assert cached_evaluation is not None
        assert cached_context is not None
        # Проверка получения из хранилища
        stored_candidate = storage.get_strategy_candidate(sample_candidate.id)
        stored_evaluation = storage.get_evaluation_result(sample_evaluation.id)
        stored_context = storage.get_evolution_context(sample_context.id)
        assert stored_candidate is not None
        assert stored_evaluation is not None
        assert stored_context is not None
        # Проверка согласованности данных
        assert cached_candidate.id == stored_candidate.id
        assert cached_evaluation.id == stored_evaluation.id
        assert cached_context.id == stored_context.id
    def test_storage_backup_integration(self, temp_db_path: str, temp_backup_dir: Path,
                                      sample_candidate: StrategyCandidate, sample_evaluation: StrategyEvaluationResult,
                                      sample_context: EvolutionContext) -> None:
        """Тест интеграции хранилища и системы бэкапов."""
        # Инициализация компонентов
        storage = StrategyStorage(temp_db_path)
        backup = EvolutionBackup(str(temp_backup_dir))
        # Сохранение данных в хранилище
        storage.save_strategy_candidate(sample_candidate)
        storage.save_evaluation_result(sample_evaluation)
        storage.save_evolution_context(sample_context)
        # Получение данных для бэкапа
        candidates = storage.get_strategy_candidates()
        evaluations = storage.get_evaluation_results()
        contexts = storage.get_evolution_contexts()
        # Создание бэкапа
        backup_data = {
            "candidates": [candidate.to_dict() for candidate in candidates],
            "evaluations": [evaluation.to_dict() for evaluation in evaluations],
            "contexts": [context.to_dict() for context in contexts],
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_items": len(candidates) + len(evaluations) + len(contexts)
            }
        }
        # Исправление: сохраняем путь, а не dict
        backup_path = backup.create_backup(json.dumps(backup_data))
        # Проверка создания бэкапа
        assert backup_path is not None
        # Восстановление из бэкапа
        restored_data = backup.restore_backup(str(backup_path))
        # Проверка восстановленных данных
        assert isinstance(restored_data, dict) and "candidates" in restored_data
        assert "evaluations" in restored_data
        assert "contexts" in restored_data
        assert len(restored_data["candidates"]) == 1
        assert len(restored_data["evaluations"]) == 1
        assert len(restored_data["contexts"]) == 1
    def test_cache_backup_integration(self, temp_backup_dir: Path, sample_candidate: StrategyCandidate,
                                    sample_evaluation: StrategyEvaluationResult, sample_context: EvolutionContext) -> None:
        """Тест интеграции кэша и системы бэкапов."""
        # Инициализация компонентов
        cache = EvolutionCache({
            "cache_size": 100,
            "cache_ttl": 300,
            "cache_strategy": "lru"
        })
        backup = EvolutionBackup(str(temp_backup_dir))
        # Кэширование данных
        cache.set_candidate(sample_candidate.id, sample_candidate)
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        cache.set_context(sample_context.id, sample_context)
        # Получение статистики кэша
        cache_stats = cache.get_stats()
        # Создание бэкапа кэша
        cache_backup_data = {
            "cache_stats": cache_stats,
            "cached_candidates": [sample_candidate.to_dict()],
            "cached_evaluations": [sample_evaluation.to_dict()],
            "cached_contexts": [sample_context.to_dict()],
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "cache_size": cache_stats["total_items"]
            }
        }
        backup_path = backup.create_backup(json.dumps(cache_backup_data))
        # Проверка создания бэкапа кэша
        assert backup_path is not None
        # Восстановление из бэкапа кэша
        restored_cache_data = backup.restore_backup(str(backup_path))
        # Проверка восстановленных данных кэша
        assert isinstance(restored_cache_data, dict)
        assert "cache_stats" in restored_cache_data
        assert "cached_candidates" in restored_cache_data
        assert "cached_evaluations" in restored_cache_data
        assert "cached_contexts" in restored_cache_data
    def test_migration_storage_integration(self, temp_db_path: str, temp_migration_dir: Path) -> None:
        """Тест интеграции миграций и хранилища."""
        # Инициализация компонентов
        storage = StrategyStorage(temp_db_path)
        migration = EvolutionMigration(str(temp_migration_dir))
        # Создание миграции для добавления новой таблицы
        migration_data = {
            "version": "1.1",
            "description": "Add strategy metadata table",
            "scripts": [
                """
                CREATE TABLE IF NOT EXISTS strategy_metadata (
                    id INTEGER PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (strategy_id) REFERENCES strategy_candidates(id)
                )
                """,
                """
                CREATE INDEX IF NOT EXISTS idx_strategy_metadata_strategy_id 
                ON strategy_metadata(strategy_id)
                """
            ],
            "rollback_scripts": [
                "DROP TABLE IF EXISTS strategy_metadata"
            ],
            "rollback_supported": True,
            "dependencies": []
        }
        # Создание файла миграции
        migration_path = migration.create_migration(migration_data)
        # Проверка создания миграции
        assert migration_path is not None
        # Выполнение миграции
        result = migration.execute_migration(migration_path)
        # Проверка успешного выполнения миграции
        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["version"] == "1.1"
        assert result["executed_scripts"] == 2
    def test_full_workflow_integration(self, temp_db_path: str, temp_backup_dir: Path, temp_migration_dir: Path,
                                     sample_candidate: StrategyCandidate, sample_evaluation: StrategyEvaluationResult,
                                     sample_context: EvolutionContext) -> None:
        """Тест полного рабочего процесса интеграции."""
        # Инициализация всех компонентов
        storage = StrategyStorage(temp_db_path)
        cache = EvolutionCache({
            "cache_size": 100,
            "cache_ttl": 300,
            "cache_strategy": "lru"
        })
        backup = EvolutionBackup(str(temp_backup_dir))
        migration = EvolutionMigration(str(temp_migration_dir))
        # Шаг 1: Выполнение миграции для подготовки БД
        migration_data = {
            "version": "1.0",
            "description": "Initial schema setup",
            "scripts": [
                "CREATE TABLE IF NOT EXISTS test_table (id INTEGER PRIMARY KEY, name TEXT)"
            ],
            "rollback_scripts": [
                "DROP TABLE IF EXISTS test_table"
            ],
            "rollback_supported": True,
            "dependencies": []
        }
        migration_path = migration.create_migration(migration_data)
        migration_result = migration.execute_migration(migration_path)
        assert isinstance(migration_result, dict)
        assert migration_result["success"] is True
        # Шаг 2: Сохранение данных в хранилище
        storage.save_strategy_candidate(sample_candidate)
        storage.save_evaluation_result(sample_evaluation)
        storage.save_evolution_context(sample_context)
        # Шаг 3: Кэширование данных
        cache.set_candidate(sample_candidate.id, sample_candidate)
        cache.set_evaluation(sample_evaluation.id, sample_evaluation)
        cache.set_context(sample_context.id, sample_context)
        # Шаг 4: Создание бэкапа
        backup_data = {
            "candidates": [sample_candidate.to_dict()],
            "evaluations": [sample_evaluation.to_dict()],
            "contexts": [sample_context.to_dict()],
            "cache_stats": cache.get_stats(),
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_items": 3
            }
        }
        backup_path = backup.create_backup(json.dumps(backup_data))
        assert backup_path is not None
        # Шаг 5: Проверка целостности данных
        # Из хранилища
        stored_candidate = storage.get_strategy_candidate(sample_candidate.id)
        stored_evaluation = storage.get_evaluation_result(sample_evaluation.id)
        stored_context = storage.get_evolution_context(sample_context.id)
        # Из кэша
        cached_candidate = cache.get_candidate(sample_candidate.id)
        cached_evaluation = cache.get_evaluation(sample_evaluation.id)
        cached_context = cache.get_context(sample_context.id)
        # Из бэкапа
        restored_data = backup.restore_backup(str(backup_path))
        # Проверка согласованности
        assert stored_candidate is not None
        assert cached_candidate is not None
        assert isinstance(restored_data, dict)
        assert len(restored_data["candidates"]) == 1
        assert stored_candidate.id == cached_candidate.id
        assert stored_candidate.name == cached_candidate.name
        assert stored_candidate.name == sample_candidate.name
    def test_error_handling_integration(self, temp_db_path: str, temp_backup_dir: Path) -> None:
        """Тест интеграции обработки ошибок."""
        # Инициализация компонентов
        storage = StrategyStorage(temp_db_path)
        cache = EvolutionCache()
        backup = EvolutionBackup(str(temp_backup_dir))
        # Тест обработки ошибок хранилища
        try:
            # Попытка получить несуществующий кандидат
            non_existent_candidate = storage.get_strategy_candidate(uuid4())
            assert non_existent_candidate is None
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка: {e}")
        # Тест обработки ошибок кэша
        try:
            # Попытка получить несуществующий элемент из кэша
            non_existent_cache_item = cache.get_candidate(uuid4())
            assert non_existent_cache_item is None
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка: {e}")
        # Тест обработки ошибок бэкапа
        try:
            # Попытка восстановить несуществующий бэкап
            non_existent_backup_path = temp_backup_dir / "non_existent.json"
            with pytest.raises(Exception):
                backup.restore_backup(str(non_existent_backup_path))
        except Exception as e:
            pytest.fail(f"Неожиданная ошибка: {e}")
    def test_performance_integration(self, temp_db_path: str, temp_backup_dir: Path) -> None:
        """Тест интеграции производительности."""
        # Инициализация компонентов
        storage = StrategyStorage(temp_db_path)
        cache = EvolutionCache({
            "cache_size": 1000,
            "cache_ttl": 300,
            "cache_strategy": "lru"
        })
        backup = EvolutionBackup(str(temp_backup_dir))
        # Создание большого количества тестовых данных
        candidates = []
        evaluations = []
        contexts = []
        for i in range(100):
            candidate = StrategyCandidate(
                id=uuid4(),
                name=f"Strategy {i}",
                description=f"Test strategy {i}",
                indicators=[],
                filters=[],
                entry_rules=[],
                exit_rules=[],
                generation=i
            )
            candidates.append(candidate)
            evaluation = StrategyEvaluationResult(
                id=uuid4(),
                strategy_id=candidate.id,
                total_trades=10,
                winning_trades=7,
                losing_trades=3,
                win_rate=Decimal("0.7"),
                accuracy=Decimal("0.8"),
                total_pnl=Decimal("1000"),
                net_pnl=Decimal("950"),
                profitability=Decimal("0.05"),
                is_approved=True
            )
            evaluations.append(evaluation)
            context = EvolutionContext(
                id=uuid4(),
                name=f"Context {i}",
                description=f"Test context {i}",
                population_size=50,
                generations=100
            )
            contexts.append(context)
        # Тест производительности сохранения
        import time
        start_time = time.time()
        for candidate in candidates:
            storage.save_strategy_candidate(candidate)
        storage_time = time.time() - start_time
        # Тест производительности кэширования
        start_time = time.time()
        for candidate in candidates:
            cache.set_candidate(candidate.id, candidate)
        cache_time = time.time() - start_time
        # Тест производительности бэкапа
        backup_data = {
            "candidates": [c.to_dict() for c in candidates],
            "evaluations": [e.to_dict() for e in evaluations],
            "contexts": [ctx.to_dict() for ctx in contexts],
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_items": len(candidates) + len(evaluations) + len(contexts)
            }
        }
        start_time = time.time()
        backup_path = backup.create_backup(json.dumps(backup_data))
        backup_time = time.time() - start_time
        # Проверка производительности
        assert storage_time < 10.0  # Сохранение должно быть быстрым
        assert cache_time < 1.0     # Кэширование должно быть очень быстрым
        assert backup_time < 5.0    # Бэкап должен быть разумно быстрым
        # Проверка целостности данных
        assert backup_path is not None  # Проверяю None вместо exists()
        assert len(storage.get_strategy_candidates()) == 100
        assert cache.get_stats()["total_items"] == 100
    def test_concurrent_access_integration(self, temp_db_path: str) -> None:
        """Тест интеграции при конкурентном доступе."""
        import threading
        import time
        storage = StrategyStorage(temp_db_path)
        cache = EvolutionCache()
        # Создание тестовых данных
        test_candidates = []
        for i in range(10):
            candidate = StrategyCandidate(
                id=uuid4(),
                name=f"Concurrent Strategy {i}",
                indicators=[],
                filters=[],
                entry_rules=[],
                exit_rules=[],
                generation=i
            )
            test_candidates.append(candidate)
        # Функция для конкурентного доступа
        def concurrent_operations(candidate_id, candidate_data) -> Any:
            # Сохранение в хранилище
            storage.save_strategy_candidate(candidate_data)
            # Кэширование
            cache.set_candidate(candidate_id, candidate_data)
            # Чтение из хранилища
            stored = storage.get_strategy_candidate(candidate_id)
            # Чтение из кэша
            cached = cache.get_candidate(candidate_id)
            # Проверка согласованности
            assert stored is not None
            assert cached is not None
            assert stored.id == cached.id
        # Запуск конкурентных операций
        threads = []
        for candidate in test_candidates:
            thread = threading.Thread(
                target=concurrent_operations,
                args=(candidate.id, candidate)
            )
            threads.append(thread)
            thread.start()
        # Ожидание завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверка финального состояния
        assert len(storage.get_strategy_candidates()) == 10
        assert cache.get_stats()["total_items"] == 10
    def test_data_consistency_integration(self, temp_db_path: str, temp_backup_dir: Path,
                                        sample_candidate: StrategyCandidate) -> None:
        """Тест интеграции согласованности данных."""
        storage = StrategyStorage(temp_db_path)
        cache = EvolutionCache()
        backup = EvolutionBackup(str(temp_backup_dir))
        # Сохранение данных
        storage.save_strategy_candidate(sample_candidate)
        cache.set_candidate(sample_candidate.id, sample_candidate)
        # Изменение данных
        sample_candidate.name = "Updated Strategy"
        sample_candidate.description = "Updated description"
        # Обновление в хранилище
        storage.save_strategy_candidate(sample_candidate)
        # Обновление в кэше
        cache.set_candidate(sample_candidate.id, sample_candidate)
        # Создание бэкапа
        backup_data = {
            "candidates": [sample_candidate.to_dict()],
            "metadata": {
                "version": "1.0",
                "created_at": datetime.now().isoformat(),
                "total_items": 1
            }
        }
        backup_path = backup.create_backup(json.dumps(backup_data))
        # Проверка согласованности данных
        stored = storage.get_strategy_candidate(sample_candidate.id)
        cached = cache.get_candidate(sample_candidate.id)
        restored_data = backup.restore_backup(str(backup_path))
        assert stored.name == "Updated Strategy"
        assert cached.name == "Updated Strategy"
        assert stored.name == cached.name
        assert stored.description == "Updated description"
        assert cached.description == "Updated description"
        assert stored.description == cached.description
        # Проверка данных в бэкапе
        assert isinstance(restored_data, dict)
        assert len(restored_data["candidates"]) == 1
        assert restored_data["candidates"][0]["name"] == "Updated Strategy"
        assert restored_data["candidates"][0]["description"] == "Updated description" 
