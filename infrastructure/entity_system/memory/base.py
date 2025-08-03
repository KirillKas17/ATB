"""
Базовые типы и интерфейсы для модуля memory.
"""

import asyncio
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

from domain.types.entity_system_types import (
    AnalysisResult,
    EntityState,
    Experiment,
    Hypothesis,
    Improvement,
    MemoryError,
    MemorySnapshot,
)


class MemoryPolicy(Enum):
    """Политики управления памятью."""

    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class SnapshotFormat(Enum):
    """Форматы сохранения снимков."""

    JSON = "json"
    PICKLE = "pickle"
    COMPRESSED = "compressed"


class SnapshotMetadata(TypedDict):
    """Метаданные снимка."""

    id: str
    timestamp: str  # ISO format string
    format: str
    size_bytes: int
    checksum: str
    description: str
    tags: List[str]
    version: str


class MemoryEvent(TypedDict):
    """Событие памяти."""

    timestamp: datetime
    event_type: str
    memory_usage: float
    threshold: float
    action_taken: str
    details: Dict[str, Any]


@dataclass
class SnapshotConfig:
    """Конфигурация менеджера снимков."""

    storage_path: Path = field(default_factory=lambda: Path("state/snapshots"))
    max_snapshots: int = 100
    max_snapshot_age_days: int = 30
    compression_enabled: bool = True
    default_format: SnapshotFormat = SnapshotFormat.COMPRESSED
    auto_cleanup: bool = True
    backup_enabled: bool = True
    backup_path: Path = field(default_factory=lambda: Path("state/snapshots/backup"))
    checksum_verification: bool = True
    encryption_enabled: bool = False
    encryption_key: Optional[str] = None


@dataclass
class MemoryConfig:
    """Конфигурация менеджера памяти."""

    max_memory_percent: float = 80.0
    critical_memory_percent: float = 90.0
    gc_threshold_percent: float = 70.0
    cache_cleanup_threshold_percent: float = 75.0
    memory_policy: MemoryPolicy = MemoryPolicy.BALANCED
    enable_auto_cleanup: bool = True
    enable_memory_monitoring: bool = True
    monitoring_interval: int = 30
    max_cache_size_mb: int = 512
    cache_ttl_seconds: int = 3600
    enable_compression: bool = True
    gc_frequency: int = 300
    memory_history_size: int = 1000


class BaseSnapshotManager(ABC):
    """Базовый класс для менеджера снимков."""

    @abstractmethod
    async def create_snapshot(
        self,
        system_state: EntityState,
        analysis_results: Optional[List[AnalysisResult]] = None,
        active_hypotheses: Optional[List[Hypothesis]] = None,
        active_experiments: Optional[List[Experiment]] = None,
        applied_improvements: Optional[List[Improvement]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> MemorySnapshot:
        """Создание нового снимка состояния системы."""
        pass

    @abstractmethod
    async def load_snapshot(self, snapshot_id: str) -> MemorySnapshot:
        """Загрузка снимка по ID."""
        pass

    @abstractmethod
    async def list_snapshots(
        self,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[SnapshotMetadata]:
        """Получение списка доступных снимков с фильтрацией."""
        pass

    @abstractmethod
    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Удаление снимка."""
        pass


class BaseMemoryManager(ABC):
    """Базовый класс для менеджера памяти."""

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Запуск мониторинга памяти."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Остановка мониторинга памяти."""
        pass

    @abstractmethod
    async def get_memory_usage(self) -> Dict[str, float]:
        """Получение информации об использовании памяти."""
        pass

    @abstractmethod
    async def cache_get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        pass

    @abstractmethod
    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Установка значения в кэш."""
        pass

    @abstractmethod
    async def cache_delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        pass

    @abstractmethod
    async def cache_clear(self) -> None:
        """Очистка всего кэша."""
        pass

    @abstractmethod
    async def force_garbage_collection(self) -> Dict[str, Any]:
        """Принудительная сборка мусора."""
        pass

    @abstractmethod
    async def optimize_memory(self) -> Dict[str, Any]:
        """Оптимизация памяти согласно политике."""
        pass


import asyncio
import gc
import gzip
import hashlib

# ============================================================================
# Реализации абстрактных классов
# ============================================================================
import json
import logging
import pickle
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TypedDict, Union

import psutil

logger = logging.getLogger(__name__)


class SnapshotManager(BaseSnapshotManager):
    """Промышленная реализация менеджера снимков."""

    def __init__(self, config: SnapshotConfig):
        self.config = config
        self.storage_path = config.storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.backup_path = config.backup_path
        self.backup_path.mkdir(parents=True, exist_ok=True)
        # Кэш метаданных
        self._metadata_cache: Dict[str, SnapshotMetadata] = {}
        self._load_metadata_cache()

    async def create_snapshot(
        self,
        system_state: EntityState,
        analysis_results: Optional[List[AnalysisResult]] = None,
        active_hypotheses: Optional[List[Hypothesis]] = None,
        active_experiments: Optional[List[Experiment]] = None,
        applied_improvements: Optional[List[Improvement]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> MemorySnapshot:
        """Создание нового снимка состояния системы."""
        try:
            # Генерируем уникальный ID
            snapshot_id = f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(system_state))}"
            # Создаем снимок
            snapshot = MemorySnapshot(
                id=snapshot_id,
                timestamp=datetime.now(),
                system_state=system_state,
                analysis_results=analysis_results or [],
                active_hypotheses=active_hypotheses or [],
                active_experiments=active_experiments or [],
                applied_improvements=applied_improvements or [],
                performance_metrics=performance_metrics or {},
            )
            # Сохраняем снимок
            await self._save_snapshot(snapshot, description, tags or [])
            # Очистка старых снимков
            if self.config.auto_cleanup:
                await self._cleanup_old_snapshots()
            logger.info(f"Created snapshot: {snapshot_id}")
            return snapshot
        except Exception as e:
            logger.error(f"Error creating snapshot: {e}")
            raise MemoryError(f"Failed to create snapshot: {e}")

    async def load_snapshot(self, snapshot_id: str) -> MemorySnapshot:
        """Загрузка снимка по ID."""
        try:
            snapshot_path = (
                self.storage_path / f"{snapshot_id}.{self.config.default_format.value}"
            )
            if not snapshot_path.exists():
                raise MemoryError(f"Snapshot {snapshot_id} not found")
            # Загружаем снимок
            snapshot = await self._load_snapshot_file(snapshot_path)
            # Проверяем контрольную сумму
            if self.config.checksum_verification:
                await self._verify_snapshot_checksum(snapshot_path, snapshot_id)
            logger.info(f"Loaded snapshot: {snapshot_id}")
            return snapshot
        except Exception as e:
            logger.error(f"Error loading snapshot {snapshot_id}: {e}")
            raise MemoryError(f"Failed to load snapshot {snapshot_id}: {e}")

    async def list_snapshots(
        self,
        tags: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
    ) -> List[SnapshotMetadata]:
        """Получение списка доступных снимков с фильтрацией."""
        try:
            snapshots = []
            for snapshot_file in self.storage_path.glob(
                f"*.{self.config.default_format.value}"
            ):
                snapshot_id = snapshot_file.stem
                if snapshot_id in self._metadata_cache:
                    metadata = self._metadata_cache[snapshot_id]
                    # Фильтрация по тегам
                    if tags and not any(
                        tag in metadata.get("tags", []) for tag in tags
                    ):
                        continue
                    # Фильтрация по дате
                    meta_dt = datetime.fromisoformat(metadata["timestamp"])
                    if date_from and meta_dt < date_from:
                        continue
                    if date_to and meta_dt > date_to:
                        continue
                    snapshots.append(metadata)
            # Сортируем по времени создания
            snapshots.sort(key=lambda x: datetime.fromisoformat(x["timestamp"]), reverse=True)
            return snapshots
        except Exception as e:
            logger.error(f"Error listing snapshots: {e}")
            return []

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Удаление снимка."""
        try:
            snapshot_path = (
                self.storage_path / f"{snapshot_id}.{self.config.default_format.value}"
            )
            if not snapshot_path.exists():
                return False
            # Удаляем файл
            snapshot_path.unlink()
            # Удаляем из кэша метаданных
            self._metadata_cache.pop(snapshot_id, None)
            # Удаляем резервную копию
            backup_path = (
                self.backup_path / f"{snapshot_id}.{self.config.default_format.value}"
            )
            if backup_path.exists():
                backup_path.unlink()
            logger.info(f"Deleted snapshot: {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting snapshot {snapshot_id}: {e}")
            return False

    async def _save_snapshot(
        self, snapshot: MemorySnapshot, description: str, tags: List[str]
    ) -> None:
        """Сохранение снимка в файл."""
        snapshot_path = (
            self.storage_path / f"{snapshot['id']}.{self.config.default_format.value}"
        )
        # Сериализуем данные
        if self.config.default_format == SnapshotFormat.JSON:
            data = self._serialize_to_json(snapshot)
        elif self.config.default_format == SnapshotFormat.PICKLE:
            data = self._serialize_to_pickle(snapshot)
        elif self.config.default_format == SnapshotFormat.COMPRESSED:
            data = self._serialize_to_compressed(snapshot)
        else:
            raise ValueError(f"Unsupported format: {self.config.default_format}")
        # Шифруем если нужно
        if self.config.encryption_enabled:
            data = self._encrypt_data(data)
        # Сохраняем файл
        with open(snapshot_path, "wb") as f:
            f.write(data)
        # Создаем резервную копию
        if self.config.backup_enabled:
            backup_path = (
                self.backup_path / f"{snapshot['id']}.{self.config.default_format.value}"
            )
            with open(backup_path, "wb") as f:
                f.write(data)
        # Сохраняем метаданные
        metadata = SnapshotMetadata(
            id=snapshot['id'],
            timestamp=snapshot['timestamp'].isoformat() if isinstance(snapshot['timestamp'], datetime) else str(snapshot['timestamp']),
            format=self.config.default_format.value,
            size_bytes=len(data),
            checksum=self._calculate_checksum(data),
            description=description,
            tags=tags,
            version="1.0",
        )
        self._metadata_cache[snapshot['id']] = metadata
        self._save_metadata_cache()

    async def _load_snapshot_file(self, snapshot_path: Path) -> MemorySnapshot:
        """Загрузка снимка из файла."""
        with open(snapshot_path, "rb") as f:
            data = f.read()
        # Расшифровываем если нужно
        if self.config.encryption_enabled:
            data = self._decrypt_data(data)
        # Десериализуем данные
        if snapshot_path.suffix == ".json":
            return self._deserialize_from_json(data)
        elif snapshot_path.suffix == ".pkl":
            return self._deserialize_from_pickle(data)
        elif snapshot_path.suffix == ".gz":
            return self._deserialize_from_compressed(data)
        else:
            raise ValueError(f"Unsupported file format: {snapshot_path.suffix}")

    def _serialize_to_json(self, snapshot: MemorySnapshot) -> bytes:
        """Сериализация в JSON."""
        # Преобразуем в словарь
        snapshot_dict = {
            "id": snapshot['id'],
            "timestamp": snapshot['timestamp'].isoformat(),
            "system_state": snapshot['system_state'],
            "analysis_results": snapshot['analysis_results'],
            "active_hypotheses": snapshot['active_hypotheses'],
            "active_experiments": snapshot['active_experiments'],
            "applied_improvements": snapshot['applied_improvements'],
            "performance_metrics": snapshot['performance_metrics'],
        }
        return json.dumps(snapshot_dict, default=str).encode("utf-8")

    def _serialize_to_pickle(self, snapshot: MemorySnapshot) -> bytes:
        """Сериализация в pickle."""
        return pickle.dumps(snapshot)

    def _serialize_to_compressed(self, snapshot: MemorySnapshot) -> bytes:
        """Сериализация в сжатый формат."""
        json_data = self._serialize_to_json(snapshot)
        return gzip.compress(json_data)

    def _deserialize_from_json(self, data: bytes) -> MemorySnapshot:
        """Десериализация из JSON."""
        snapshot_dict = json.loads(data.decode("utf-8"))
        return MemorySnapshot(
            id=snapshot_dict["id"],
            timestamp=datetime.fromisoformat(snapshot_dict["timestamp"]),
            system_state=snapshot_dict["system_state"],
            analysis_results=snapshot_dict["analysis_results"],
            active_hypotheses=snapshot_dict["active_hypotheses"],
            active_experiments=snapshot_dict["active_experiments"],
            applied_improvements=snapshot_dict["applied_improvements"],
            performance_metrics=snapshot_dict["performance_metrics"],
        )

    def _deserialize_from_pickle(self, data: bytes) -> MemorySnapshot:
        """Десериализация из pickle."""
        result: MemorySnapshot = pickle.loads(data)
        return result

    def _deserialize_from_compressed(self, data: bytes) -> MemorySnapshot:
        """Десериализация из сжатого формата."""
        json_data = gzip.decompress(data)
        return self._deserialize_from_json(json_data)

    def _calculate_checksum(self, data: bytes) -> str:
        """Вычисление контрольной суммы."""
        return hashlib.sha256(data).hexdigest()

    def _encrypt_data(self, data: bytes) -> bytes:
        """Шифрование данных."""
        # Простая реализация шифрования
        if not self.config.encryption_key:
            return data
        key = self.config.encryption_key.encode()
        encrypted = bytearray()
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key[i % len(key)])
        return bytes(encrypted)

    def _decrypt_data(self, data: bytes) -> bytes:
        """Расшифрование данных."""
        # Простая реализация расшифрования
        if not self.config.encryption_key:
            return data
        key = self.config.encryption_key.encode()
        decrypted = bytearray()
        for i, byte in enumerate(data):
            decrypted.append(byte ^ key[i % len(key)])
        return bytes(decrypted)

    async def _verify_snapshot_checksum(
        self, snapshot_path: Path, snapshot_id: str
    ) -> None:
        """Проверка контрольной суммы снимка."""
        with open(snapshot_path, "rb") as f:
            data = f.read()
        calculated_checksum = self._calculate_checksum(data)
        stored_checksum: Union[SnapshotMetadata, Dict[Any, Any]] = self._metadata_cache.get(snapshot_id, {})
        checksum: Optional[str] = None
        if isinstance(stored_checksum, dict):
            checksum = stored_checksum.get("checksum")
        if checksum and calculated_checksum != checksum:
            raise MemoryError(
                f"Checksum verification failed for snapshot {snapshot_id}"
            )

    async def _cleanup_old_snapshots(self) -> None:
        """Очистка старых снимков."""
        try:
            current_time = datetime.now()
            snapshots_to_delete = []
            for snapshot_id, metadata in self._metadata_cache.items():
                meta_dt = datetime.fromisoformat(metadata["timestamp"])
                age_days = (current_time - meta_dt).days
                if age_days > self.config.max_snapshot_age_days:
                    snapshots_to_delete.append(snapshot_id)
            # Удаляем старые снимки
            for snapshot_id in snapshots_to_delete:
                await self.delete_snapshot(snapshot_id)
            # Ограничиваем количество снимков
            if len(self._metadata_cache) > self.config.max_snapshots:
                sorted_snapshots = sorted(
                    self._metadata_cache.items(), key=lambda x: datetime.fromisoformat(x[1]["timestamp"])
                )
                snapshots_to_delete = [
                    snapshot_id
                    for snapshot_id, _ in sorted_snapshots[: -self.config.max_snapshots]
                ]
                for snapshot_id in snapshots_to_delete:
                    await self.delete_snapshot(snapshot_id)
        except Exception as e:
            logger.error(f"Error during snapshot cleanup: {e}")

    def _load_metadata_cache(self) -> None:
        """Загрузка кэша метаданных."""
        try:
            metadata_path = self.storage_path / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata_dict = json.load(f)
                for snapshot_id, metadata in metadata_dict.items():
                    # timestamp оставляем строкой, не преобразуем
                    self._metadata_cache[snapshot_id] = metadata
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")

    def _save_metadata_cache(self) -> None:
        """Сохранение кэша метаданных."""
        try:
            metadata_path = self.storage_path / "metadata.json"
            # Преобразуем datetime в строки для JSON
            metadata_dict = {}
            for snapshot_id, metadata in self._metadata_cache.items():
                metadata_copy = metadata.copy()
                # timestamp уже должен быть строкой согласно SnapshotMetadata
                # if isinstance(metadata["timestamp"], datetime):
                #     metadata_copy["timestamp"] = metadata["timestamp"].isoformat()
                metadata_dict[snapshot_id] = metadata_copy
            with open(metadata_path, "w") as f:
                json.dump(metadata_dict, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")


class MemoryManager(BaseMemoryManager):
    """Промышленная реализация менеджера памяти."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._memory_history: deque = deque(maxlen=config.memory_history_size)
        self._is_monitoring = False
        # Инициализация мониторинга
        if config.enable_memory_monitoring:
            asyncio.create_task(self.start_monitoring())

    async def start_monitoring(self) -> None:
        """Запуск мониторинга памяти."""
        if self._is_monitoring:
            return
        self._is_monitoring = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory monitoring started")

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга памяти."""
        if not self._is_monitoring:
            return
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory monitoring stopped")

    async def get_memory_usage(self) -> Dict[str, float]:
        """Получение информации об использовании памяти."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            # Системная память
            system_memory = psutil.virtual_memory()
            # Процесс память
            process_memory_percent = process.memory_percent()
            # Кэш память
            cache_memory_mb = len(self._cache) * 0.001  # Примерная оценка
            return {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "process_memory_percent": process_memory_percent,
                "system_memory_percent": system_memory.percent,
                "system_memory_available_mb": system_memory.available / 1024 / 1024,
                "cache_memory_mb": cache_memory_mb,
                "cache_size": len(self._cache),
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                "process_memory_mb": 0.0,
                "process_memory_percent": 0.0,
                "system_memory_percent": 0.0,
                "system_memory_available_mb": 0.0,
                "cache_memory_mb": 0.0,
                "cache_size": 0,
            }

    async def cache_get(self, key: str) -> Optional[Any]:
        """Получение значения из кэша."""
        if key not in self._cache:
            return None
        # Проверяем TTL
        if key in self._cache_timestamps:
            age = (datetime.now() - self._cache_timestamps[key]).total_seconds()
            if age > self.config.cache_ttl_seconds:
                await self.cache_delete(key)
                return None
        return self._cache[key]

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Установка значения в кэш."""
        # Проверяем размер кэша
        cache_size_mb = len(self._cache) * 0.001  # Примерная оценка
        if cache_size_mb > self.config.max_cache_size_mb:
            await self._cleanup_cache()
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()

    async def cache_delete(self, key: str) -> bool:
        """Удаление значения из кэша."""
        if key in self._cache:
            del self._cache[key]
            self._cache_timestamps.pop(key, None)
            return True
        return False

    async def cache_clear(self) -> None:
        """Очистка всего кэша."""
        self._cache.clear()
        self._cache_timestamps.clear()
        logger.info("Cache cleared")

    async def force_garbage_collection(self) -> Dict[str, Any]:
        """Принудительная сборка мусора."""
        try:
            # Получаем статистику до сборки
            before_stats = await self.get_memory_usage()
            # Выполняем сборку мусора
            collected = gc.collect()
            # Получаем статистику после сборки
            after_stats = await self.get_memory_usage()
            memory_freed = (
                before_stats["process_memory_mb"] - after_stats["process_memory_mb"]
            )
            logger.info(
                f"Garbage collection completed: {collected} objects collected, {memory_freed:.2f} MB freed"
            )
            return {
                "objects_collected": collected,
                "memory_freed_mb": memory_freed,
                "before_stats": before_stats,
                "after_stats": after_stats,
            }
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")
            return {"objects_collected": 0, "memory_freed_mb": 0.0, "error": str(e)}

    async def optimize_memory(self) -> Dict[str, Any]:
        """Оптимизация памяти согласно политике."""
        try:
            memory_usage = await self.get_memory_usage()
            optimizations = []
            # Проверяем использование памяти процесса
            if (
                memory_usage["process_memory_percent"]
                > self.config.critical_memory_percent
            ):
                # Критический уровень - агрессивная оптимизация
                optimizations.append("critical_memory_cleanup")
                await self._critical_memory_cleanup()
            elif (
                memory_usage["process_memory_percent"]
                > self.config.gc_threshold_percent
            ):
                # Высокий уровень - умеренная оптимизация
                optimizations.append("moderate_memory_cleanup")
                await self._moderate_memory_cleanup()
            elif (
                memory_usage["process_memory_percent"]
                > self.config.cache_cleanup_threshold_percent
            ):
                # Средний уровень - очистка кэша
                optimizations.append("cache_cleanup")
                await self._cleanup_cache()
            # Применяем политику памяти
            if self.config.memory_policy == MemoryPolicy.AGGRESSIVE:
                optimizations.append("aggressive_optimization")
                await self._aggressive_optimization()
            elif self.config.memory_policy == MemoryPolicy.CONSERVATIVE:
                optimizations.append("conservative_optimization")
                await self._conservative_optimization()
            # Записываем в историю
            self._memory_history.append(
                {
                    "timestamp": datetime.now(),
                    "memory_usage": memory_usage,
                    "optimizations": optimizations,
                }
            )
            return {
                "optimizations_applied": optimizations,
                "memory_usage": memory_usage,
                "optimization_success": True,
            }
        except Exception as e:
            logger.error(f"Error during memory optimization: {e}")
            return {
                "optimizations_applied": [],
                "error": str(e),
                "optimization_success": False,
            }

    async def _monitoring_loop(self) -> None:
        """Основной цикл мониторинга памяти."""
        while self._is_monitoring:
            try:
                # Проверяем использование памяти
                memory_usage = await self.get_memory_usage()
                # Записываем в историю
                self._memory_history.append(
                    {"timestamp": datetime.now(), "memory_usage": memory_usage}
                )
                # Проверяем необходимость оптимизации
                if (
                    memory_usage["process_memory_percent"]
                    > self.config.max_memory_percent
                ):
                    logger.warning(
                        f"High memory usage detected: {memory_usage['process_memory_percent']:.1f}%"
                    )
                    await self.optimize_memory()
                # Ждем следующей проверки
                await asyncio.sleep(self.config.monitoring_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory monitoring loop: {e}")
                await asyncio.sleep(self.config.monitoring_interval)

    async def _cleanup_cache(self) -> None:
        """Очистка кэша."""
        try:
            current_time = datetime.now()
            keys_to_delete = []
            for key, timestamp in self._cache_timestamps.items():
                age = (current_time - timestamp).total_seconds()
                if age > self.config.cache_ttl_seconds:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                await self.cache_delete(key)
            logger.info(f"Cache cleanup completed: {len(keys_to_delete)} items removed")
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    async def _critical_memory_cleanup(self) -> None:
        """Критическая очистка памяти."""
        # Очищаем весь кэш
        await self.cache_clear()
        # Принудительная сборка мусора
        await self.force_garbage_collection()
        logger.warning("Critical memory cleanup completed")

    async def _moderate_memory_cleanup(self) -> None:
        """Умеренная очистка памяти."""
        # Очищаем старые элементы кэша
        await self._cleanup_cache()
        # Сборка мусора
        await self.force_garbage_collection()
        logger.info("Moderate memory cleanup completed")

    async def _aggressive_optimization(self) -> None:
        """Агрессивная оптимизация памяти."""
        # Очищаем кэш
        await self.cache_clear()
        # Множественная сборка мусора
        for _ in range(3):
            await self.force_garbage_collection()
            await asyncio.sleep(0.1)
        logger.info("Aggressive memory optimization completed")

    async def _conservative_optimization(self) -> None:
        """Консервативная оптимизация памяти."""
        # Только очистка кэша
        await self._cleanup_cache()
        logger.info("Conservative memory optimization completed")
