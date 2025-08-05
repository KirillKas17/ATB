"""
Тесты для модулей памяти Entity System.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from infrastructure.entity_system.memory.base import (
    SnapshotManager, MemoryManager, SnapshotConfig, MemoryConfig,
    MemoryPolicy, SnapshotFormat
)
from infrastructure.entity_system.memory.utils import (
    generate_snapshot_id, calculate_checksum,
    serialize_snapshot, deserialize_snapshot, validate_snapshot_data
)
from domain.type_definitions.entity_system_types import (
    EntityState, MemorySnapshot, AnalysisResult, Hypothesis, 
    Experiment, Improvement, MemoryError
)
class TestSnapshotManager:
    """Тесты для SnapshotManager."""
    @pytest.fixture
    def temp_dir(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Временная директория для тестов."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    @pytest.fixture
    def snapshot_config(self, temp_dir) -> Any:
        """Конфигурация для тестов."""
        # return SnapshotConfig(
        #     storage_path=temp_dir / "snapshots",
        #     max_snapshots=10,
        #     max_snapshot_age_days=1,
        #     compression_enabled=True,
        #     default_format=SnapshotFormat.COMPRESSED,
        #     auto_cleanup=True,
        #     backup_enabled=False,
        #     checksum_verification=True,
        #     encryption_enabled=False
        # )
        return SnapshotConfig(
            storage_path=temp_dir / "snapshots",
            max_snapshots=10,
            max_snapshot_age_days=1,
            compression_enabled=True,
            default_format=SnapshotFormat.COMPRESSED,
            auto_cleanup=True,
            backup_enabled=False,
            checksum_verification=True,
            encryption_enabled=False
        )
    @pytest.fixture
    def snapshot_manager(self, snapshot_config) -> Any:
        """Экземпляр SnapshotManager для тестов."""
        return SnapshotManager(snapshot_config)
    @pytest.fixture
    def sample_entity_state(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Образцовое состояние системы."""
        return EntityState(
            is_running=True,
            current_phase="analysis",
            ai_confidence=0.85,
            optimization_level="high",
            system_health=95.0,
            performance_score=0.92,
            efficiency_score=0.88,
            last_update=datetime.now()
        )
    @pytest.mark.asyncio
    async def test_create_snapshot(self, snapshot_manager, sample_entity_state) -> None:
        """Тест создания снимка."""
        snapshot = await snapshot_manager.create_snapshot(
            system_state=sample_entity_state,
            description="Test snapshot",
            tags=["test", "memory"]
        )
        assert snapshot["id"] is not None
        assert snapshot["timestamp"] is not None
        assert snapshot["system_state"] == sample_entity_state
        assert snapshot["analysis_results"] == []
        assert snapshot["active_hypotheses"] == []
        assert snapshot["active_experiments"] == []
        assert snapshot["applied_improvements"] == []
        assert snapshot["performance_metrics"] == {}
    @pytest.mark.asyncio
    async def test_load_snapshot(self, snapshot_manager, sample_entity_state) -> None:
        """Тест загрузки снимка."""
        # Создаем снимок
        created_snapshot = await snapshot_manager.create_snapshot(
            system_state=sample_entity_state,
            description="Test snapshot for loading"
        )
        # Загружаем снимок
        loaded_snapshot = await snapshot_manager.load_snapshot(created_snapshot["id"])
        assert loaded_snapshot["id"] == created_snapshot["id"]
        assert loaded_snapshot["system_state"] == sample_entity_state
    @pytest.mark.asyncio
    async def test_list_snapshots(self, snapshot_manager, sample_entity_state) -> None:
        """Тест получения списка снимков."""
        # Создаем несколько снимков
        await snapshot_manager.create_snapshot(
            system_state=sample_entity_state,
            description="Snapshot 1",
            tags=["test"]
        )
        await snapshot_manager.create_snapshot(
            system_state=sample_entity_state,
            description="Snapshot 2",
            tags=["test", "important"]
        )
        # Получаем все снимки
        all_snapshots = await snapshot_manager.list_snapshots()
        assert len(all_snapshots) == 2
        # Фильтруем по тегу
        test_snapshots = await snapshot_manager.list_snapshots(tags=["test"])
        assert len(test_snapshots) == 2
        important_snapshots = await snapshot_manager.list_snapshots(tags=["important"])
        assert len(important_snapshots) == 1
    @pytest.mark.asyncio
    async def test_delete_snapshot(self, snapshot_manager, sample_entity_state) -> None:
        """Тест удаления снимка."""
        # Создаем снимок
        snapshot = await snapshot_manager.create_snapshot(
            system_state=sample_entity_state,
            description="Snapshot to delete"
        )
        # Проверяем, что снимок существует
        snapshots = await snapshot_manager.list_snapshots()
        assert len(snapshots) == 1
        # Удаляем снимок
        result = await snapshot_manager.delete_snapshot(snapshot["id"])
        assert result is True
        # Проверяем, что снимок удален
        snapshots = await snapshot_manager.list_snapshots()
        assert len(snapshots) == 0
    @pytest.mark.asyncio
    async def test_cleanup_old_snapshots(self, snapshot_manager, sample_entity_state) -> None:
        """Тест очистки старых снимков."""
        # Создаем снимок
        await snapshot_manager.create_snapshot(
            system_state=sample_entity_state,
            description="Test snapshot"
        )
        # Проверяем, что снимок существует
        snapshots = await snapshot_manager.list_snapshots()
        assert len(snapshots) == 1
        # Очищаем старые снимки (должен удалить все, так как max_age_days=1)
        deleted_count = await snapshot_manager.cleanup_old_snapshots()
        assert deleted_count == 1
        # Проверяем, что снимок удален
        snapshots = await snapshot_manager.list_snapshots()
        assert len(snapshots) == 0
    def test_get_storage_stats(self, snapshot_manager) -> None:
        """Тест получения статистики хранилища."""
        stats = snapshot_manager.get_storage_stats()
        assert "total_snapshots" in stats
        assert "total_size_bytes" in stats
        assert "storage_path" in stats
        assert "backup_enabled" in stats
        assert stats["total_snapshots"] == 0
class TestMemoryManager:
    """Тесты для MemoryManager."""
    @pytest.fixture
    def memory_config(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Конфигурация для тестов."""
        return MemoryConfig(
            max_memory_percent=80.0,
            critical_memory_percent=90.0,
            gc_threshold_percent=70.0,
            cache_cleanup_threshold_percent=75.0,
            memory_policy=MemoryPolicy.BALANCED,
            enable_auto_cleanup=True,
            enable_memory_monitoring=True,
            monitoring_interval=1,
            max_cache_size_mb=10,
            cache_ttl_seconds=60,
            enable_compression=True,
            gc_frequency=300,
            memory_history_size=100
        )
    @pytest.fixture
    def memory_manager(self, memory_config) -> Any:
        """Экземпляр MemoryManager для тестов."""
        return MemoryManager(memory_config)
    @pytest.mark.asyncio
    async def test_cache_operations(self, memory_manager) -> None:
        """Тест операций с кэшем."""
        # Устанавливаем значение в кэш
        await memory_manager.cache_set("test_key", "test_value", ttl=60)
        # Получаем значение из кэша
        value = await memory_manager.cache_get("test_key")
        assert value == "test_value"
        # Получаем несуществующий ключ
        missing_value = await memory_manager.cache_get("missing_key")
        assert missing_value is None
        # Удаляем значение
        result = await memory_manager.cache_delete("test_key")
        assert result is True
        # Проверяем, что значение удалено
        deleted_value = await memory_manager.cache_get("test_key")
        assert deleted_value is None
    @pytest.mark.asyncio
    async def test_cache_clear(self, memory_manager) -> None:
        """Тест очистки кэша."""
        # Добавляем несколько значений
        await memory_manager.cache_set("key1", "value1")
        await memory_manager.cache_set("key2", "value2")
        # Проверяем, что значения добавлены
        assert await memory_manager.cache_get("key1") == "value1"
        assert await memory_manager.cache_get("key2") == "value2"
        # Очищаем кэш
        await memory_manager.cache_clear()
        # Проверяем, что все значения удалены
        assert await memory_manager.cache_get("key1") is None
        assert await memory_manager.cache_get("key2") is None
    @pytest.mark.asyncio
    async def test_memory_usage(self, memory_manager) -> None:
        """Тест получения информации об использовании памяти."""
        usage = await memory_manager.get_memory_usage()
        assert "process_memory_mb" in usage
        assert "process_memory_percent" in usage
        assert "system_memory_percent" in usage
        assert "cache_memory_mb" in usage
        assert "cache_size" in usage
        assert isinstance(usage["process_memory_mb"], float)
        assert isinstance(usage["cache_size"], int)
    @pytest.mark.asyncio
    async def test_garbage_collection(self, memory_manager) -> None:
        """Тест сборки мусора."""
        result = await memory_manager.force_garbage_collection()
        assert "collected_objects" in result
        assert "before_memory_mb" in result
        assert "after_memory_mb" in result
        assert "memory_freed_mb" in result
        assert "timestamp" in result
        assert isinstance(result["collected_objects"], int)
        assert isinstance(result["memory_freed_mb"], float)
    @pytest.mark.asyncio
    async def test_memory_optimization(self, memory_manager) -> None:
        """Тест оптимизации памяти."""
        result = await memory_manager.optimize_memory()
        assert "optimizations_applied" in result
        assert "memory_usage_before" in result
        assert "memory_usage_after" in result
        assert "timestamp" in result
        assert isinstance(result["optimizations_applied"], list)
    def test_memory_statistics(self, memory_manager) -> None:
        """Тест получения статистики памяти."""
        stats = memory_manager.get_memory_statistics()
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "cache_hit_rate" in stats
        assert "cache_size" in stats
        assert "cache_memory_mb" in stats
        assert "cleanup_count" in stats
        assert isinstance(stats["cache_hits"], int)
        assert isinstance(stats["cache_misses"], int)
        assert isinstance(stats["cache_hit_rate"], float)
        assert 0.0 <= stats["cache_hit_rate"] <= 1.0
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, memory_manager) -> None:
        """Тест жизненного цикла мониторинга."""
        # Запускаем мониторинг
        await memory_manager.start_monitoring()
        # Проверяем, что мониторинг запущен
        assert memory_manager._is_monitoring is True
        # Ждем немного для выполнения цикла мониторинга
        await asyncio.sleep(0.1)
        # Останавливаем мониторинг
        await memory_manager.stop_monitoring()
        # Проверяем, что мониторинг остановлен
        assert memory_manager._is_monitoring is False
class TestMemoryUtils:
    """Тесты для утилит памяти."""
    def test_generate_snapshot_id(self: "TestMemoryUtils") -> None:
        """Тест генерации ID снимка."""
        id1 = generate_snapshot_id()
        id2 = generate_snapshot_id()
        assert isinstance(id1, str)
        assert isinstance(id2, str)
        assert id1 != id2  # ID должны быть уникальными
        assert id1.startswith("snapshot_")
        assert id2.startswith("snapshot_")
    def test_calculate_checksum(self: "TestMemoryUtils") -> None:
        """Тест вычисления контрольной суммы."""
        data = b"test data for checksum"
        checksum = calculate_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hash length
        assert checksum.isalnum()
    def test_serialize_deserialize_snapshot(self: "TestMemoryUtils") -> None:
        """Тест сериализации и десериализации снимка."""
        snapshot: MemorySnapshot = {
            "id": "test_snapshot",
            "timestamp": datetime.now(),
            "system_state": EntityState(
                is_running=True,
                current_phase="test",
                ai_confidence=0.5,
                optimization_level="medium",
                system_health=80.0,
                performance_score=0.7,
                efficiency_score=0.6,
                last_update=datetime.now()
            ),
            "analysis_results": [],
            "active_hypotheses": [],
            "active_experiments": [],
            "applied_improvements": [],
            "performance_metrics": {}
        }
        # Тестируем JSON формат
        json_data = serialize_snapshot(snapshot, SnapshotFormat.JSON)  # type: ignore[arg-type]
        deserialized_json = deserialize_snapshot(json_data, SnapshotFormat.JSON.value)  # type: ignore[arg-type]
        assert deserialized_json["id"] == snapshot["id"]
        # Тестируем сжатый формат
        compressed_data = serialize_snapshot(snapshot, SnapshotFormat.COMPRESSED)  # type: ignore[arg-type]
        deserialized_compressed = deserialize_snapshot(compressed_data, SnapshotFormat.COMPRESSED.value)  # type: ignore[arg-type]
        assert deserialized_compressed["id"] == snapshot["id"]
    def test_validate_snapshot_data(self: "TestMemoryUtils") -> None:
        """Тест валидации данных снимка."""
        # Валидный снимок
        valid_snapshot: MemorySnapshot = {
            "id": "test",
            "timestamp": datetime.now(),
            "system_state": EntityState(
                is_running=True,
                current_phase="test",
                ai_confidence=0.5,
                optimization_level="medium",
                system_health=80.0,
                performance_score=0.7,
                efficiency_score=0.6,
                last_update=datetime.now()
            ),
            "analysis_results": [],
            "active_hypotheses": [],
            "active_experiments": [],
            "applied_improvements": [],
            "performance_metrics": {}
        }
        assert validate_snapshot_data(valid_snapshot) is True  # type: ignore[arg-type]
        # Невалидный снимок (отсутствует id)
        invalid_snapshot = {
            "timestamp": datetime.now(),
            "system_state": valid_snapshot["system_state"],
            "analysis_results": [],
            "active_hypotheses": [],
            "active_experiments": [],
            "applied_improvements": [],
            "performance_metrics": {}
        }
        assert validate_snapshot_data(invalid_snapshot) is False
class TestMemoryIntegration:
    """Интеграционные тесты для модулей памяти."""
    @pytest.fixture
    def temp_dir(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Временная директория для тестов."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    @pytest.mark.asyncio
    async def test_snapshot_and_memory_integration(self, temp_dir) -> None:
        """Тест интеграции SnapshotManager и MemoryManager."""
        # Создаем менеджеры
        snapshot_config = SnapshotConfig(
            storage_path=temp_dir / "snapshots",
            max_snapshots=5,
            auto_cleanup=True
        )
        memory_config = MemoryConfig(
            max_cache_size_mb=1,
            cache_ttl_seconds=30
        )
        snapshot_manager = SnapshotManager(snapshot_config)
        memory_manager = MemoryManager(memory_config)
        # Создаем состояние системы
        entity_state = EntityState(
            is_running=True,
            current_phase="test",
            ai_confidence=0.8,
            optimization_level="high",
            system_health=90.0,
            performance_score=0.85,
            efficiency_score=0.82,
            last_update=datetime.now()
        )
        # Создаем снимок
        snapshot = await snapshot_manager.create_snapshot(
            system_state=entity_state,
            description="Integration test snapshot"
        )
        # Кэшируем снимок в MemoryManager
        await memory_manager.cache_set(
            f"snapshot_{snapshot['id']}", 
            snapshot, 
            ttl=60
        )
        # Получаем снимок из кэша
        cached_snapshot = await memory_manager.cache_get(f"snapshot_{snapshot['id']}")
        assert cached_snapshot is not None
        assert cached_snapshot["id"] == snapshot["id"]
        # Загружаем снимок из хранилища
        loaded_snapshot = await snapshot_manager.load_snapshot(snapshot["id"])
        assert loaded_snapshot["id"] == snapshot["id"]
        # Проверяем статистику
        snapshot_stats = snapshot_manager.get_storage_stats()
        memory_stats = memory_manager.get_memory_statistics()
        assert snapshot_stats["total_snapshots"] == 1
        assert memory_stats["cache_size"] == 1 
