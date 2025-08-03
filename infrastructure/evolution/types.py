"""
Типы для эволюционной инфраструктуры - Production Ready
"""

from dataclasses import dataclass
from enum import Enum
from typing import (
    Any,
    Dict,
    Final,
    List,
    Literal,
    NewType,
    Optional,
    Protocol,
    TypedDict,
    Union,
)
from uuid import UUID

# Forward declarations для типов
StrategyCandidate = Any  # Будет определен позже
EvolutionStatus = Any    # Будет определен позже
StrategyEvaluationResult = Any  # Будет определен позже
EvolutionContext = Any   # Будет определен позже

# NewType для типобезопасности
DatabasePath = NewType('DatabasePath', str)
BackupPath = NewType('BackupPath', str)
ExportPath = NewType('ExportPath', str)
ImportPath = NewType('ImportPath', str)
CacheKey = NewType('CacheKey', str)
MigrationVersion = NewType('MigrationVersion', str)
# Literal типы для констант
StorageType = Literal["sqlite", "postgresql", "mysql", "memory"]
BackupFormat = Literal["json", "sql", "pickle", "yaml"]
CacheStrategy = Literal["lru", "ttl", "fifo", "lfu"]
MigrationStatus = Literal["pending", "running", "completed", "failed", "rolled_back"]
# Final константы
DEFAULT_DB_PATH: Final = "evolution_strategies.db"
DEFAULT_BACKUP_PATH: Final = "backups/evolution_backup"
DEFAULT_CACHE_SIZE: Final = 1000
DEFAULT_CACHE_TTL: Final = 3600  # секунды
DEFAULT_BATCH_SIZE: Final = 100
DEFAULT_MIGRATION_TIMEOUT: Final = 300  # секунды
# TypedDict для структурированных данных
class StorageConfig(TypedDict, total=False):
    """Конфигурация хранилища."""
    db_path: DatabasePath
    storage_type: StorageType
    connection_pool_size: int
    max_connections: int
    timeout: int
    enable_foreign_keys: bool
    enable_wal_mode: bool
    enable_journal_mode: bool
class CacheConfig(TypedDict, total=False):
    """Конфигурация кэша."""
    cache_size: int
    cache_ttl: int
    cache_strategy: CacheStrategy
    enable_persistence: bool
    persistence_path: str
    enable_compression: bool
    compression_level: int
class BackupConfig(TypedDict, total=False):
    """Конфигурация резервного копирования."""
    backup_path: BackupPath
    backup_format: BackupFormat
    enable_compression: bool
    compression_level: int
    max_backups: int
    auto_backup: bool
    backup_interval: int  # секунды
class MigrationConfig(TypedDict, total=False):
    """Конфигурация миграций."""
    migration_path: str
    migration_timeout: int
    enable_rollback: bool
    backup_before_migration: bool
    validate_after_migration: bool
class StorageStatistics(TypedDict):
    """Статистика хранилища."""
    total_candidates: int
    total_evaluations: int
    total_contexts: int
    candidates_by_status: Dict[str, int]
    approval_rate: float
    storage_size_bytes: int
    last_backup_time: Optional[str]
    cache_hit_rate: float
    average_query_time: float
class BackupMetadata(TypedDict):
    """Метаданные резервной копии."""
    backup_id: str
    backup_path: str
    backup_time: str
    backup_format: BackupFormat
    backup_size_bytes: int
    compression_ratio: float
    checksum: str
    version: str
    description: str
class MigrationMetadata(TypedDict):
    """Метаданные миграции."""
    migration_id: str
    version: MigrationVersion
    description: str
    applied_at: str
    execution_time: float
    status: MigrationStatus
    rollback_supported: bool
    dependencies: List[str]
class CacheEntry(TypedDict):
    """Запись кэша."""
    key: CacheKey
    value: Any
    created_at: str
    expires_at: Optional[str]
    access_count: int
    last_accessed: str
    size_bytes: int
# Protocol интерфейсы
class EvolutionStorageProtocol(Protocol):
    """Протокол для хранилища эволюции."""
    def save_strategy_candidate(self, candidate: 'StrategyCandidate') -> None:
        """Сохранить кандидата стратегии."""
        ...
    def get_strategy_candidate(self, candidate_id: UUID) -> Optional['StrategyCandidate']:
        """Получить кандидата стратегии по ID."""
        ...
    def get_strategy_candidates(
        self,
        status: Optional['EvolutionStatus'] = None,
        generation: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List['StrategyCandidate']:
        """Получить список кандидатов стратегий."""
        ...
    def save_evaluation_result(self, evaluation: 'StrategyEvaluationResult') -> None:
        """Сохранить результат оценки."""
        ...
    def get_evaluation_result(
        self, evaluation_id: UUID
    ) -> Optional['StrategyEvaluationResult']:
        """Получить результат оценки по ID."""
        ...
    def get_evaluation_results(
        self,
        strategy_id: Optional[UUID] = None,
        is_approved: Optional[bool] = None,
        limit: Optional[int] = None,
    ) -> List['StrategyEvaluationResult']:
        """Получить список результатов оценки."""
        ...
    def save_evolution_context(self, context: 'EvolutionContext') -> None:
        """Сохранить контекст эволюции."""
        ...
    def get_evolution_context(self, context_id: UUID) -> Optional['EvolutionContext']:
        """Получить контекст эволюции по ID."""
        ...
    def get_evolution_contexts(
        self, limit: Optional[int] = None
    ) -> List['EvolutionContext']:
        """Получить список контекстов эволюции."""
        ...
    def get_statistics(self) -> StorageStatistics:
        """Получить статистику хранилища."""
        ...
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Очистить старые данные."""
        ...
    def export_data(self, export_path: ExportPath) -> None:
        """Экспортировать данные в файл."""
        ...
    def import_data(self, import_path: ImportPath) -> int:
        """Импортировать данные из файла."""
        ...
class EvolutionCacheProtocol(Protocol):
    """Протокол для кэша эволюции."""
    def get(self, key: CacheKey) -> Optional[Any]:
        """Получить значение из кэша."""
        ...
    def set(self, key: CacheKey, value: Any, ttl: Optional[int] = None) -> None:
        """Установить значение в кэш."""
        ...
    def delete(self, key: CacheKey) -> bool:
        """Удалить значение из кэша."""
        ...
    def clear(self) -> None:
        """Очистить кэш."""
        ...
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        ...
class EvolutionBackupProtocol(Protocol):
    """Протокол для резервного копирования эволюции."""
    def create_backup(self, backup_path: Optional[BackupPath] = None) -> BackupMetadata:
        """Создать резервную копию."""
        ...
    def restore_backup(self, backup_id: str) -> bool:
        """Восстановить резервную копию."""
        ...
    def list_backups(self) -> List[BackupMetadata]:
        """Получить список резервных копий."""
        ...
    def delete_backup(self, backup_id: str) -> bool:
        """Удалить резервную копию."""
        ...
class EvolutionMigrationProtocol(Protocol):
    """Протокол для миграций эволюции."""
    def apply_migration(self, migration_id: str) -> MigrationMetadata:
        """Применить миграцию."""
        ...
    def rollback_migration(self, migration_id: str) -> bool:
        """Откатить миграцию."""
        ...
    def list_migrations(self) -> List[MigrationMetadata]:
        """Получить список миграций."""
        ...
    def get_pending_migrations(self) -> List[str]:
        """Получить список ожидающих миграций."""
        ...
# Enum классы
class StorageErrorType(Enum):
    """Типы ошибок хранилища."""
    CONNECTION_ERROR = "connection_error"
    QUERY_ERROR = "query_error"
    VALIDATION_ERROR = "validation_error"
    SERIALIZATION_ERROR = "serialization_error"
    DESERIALIZATION_ERROR = "deserialization_error"
    CONSTRAINT_ERROR = "constraint_error"
    TIMEOUT_ERROR = "timeout_error"
    PERMISSION_ERROR = "permission_error"
class CacheErrorType(Enum):
    """Типы ошибок кэша."""
    KEY_NOT_FOUND = "key_not_found"
    EXPIRED = "expired"
    INVALID_KEY = "invalid_key"
    STORAGE_ERROR = "storage_error"
    SERIALIZATION_ERROR = "serialization_error"
class BackupErrorType(Enum):
    """Типы ошибок резервного копирования."""
    BACKUP_FAILED = "backup_failed"
    RESTORE_FAILED = "restore_failed"
    INVALID_FORMAT = "invalid_format"
    COMPRESSION_ERROR = "compression_error"
    STORAGE_ERROR = "storage_error"
class MigrationErrorType(Enum):
    """Типы ошибок миграций."""
    MIGRATION_FAILED = "migration_failed"
    ROLLBACK_FAILED = "rollback_failed"
    DEPENDENCY_ERROR = "dependency_error"
    VERSION_CONFLICT = "version_conflict"
    VALIDATION_ERROR = "validation_error"
# Dataclass конфигурации
@dataclass
class EvolutionInfrastructureConfig:
    """Конфигурация инфраструктуры эволюции."""
    # Основные настройки
    storage_config: StorageConfig
    cache_config: CacheConfig
    backup_config: BackupConfig
    migration_config: MigrationConfig
    # Дополнительные настройки
    enable_logging: bool = True
    enable_metrics: bool = True
    enable_health_checks: bool = True
    max_retry_attempts: int = 3
    retry_delay: float = 1.0
    def __post_init__(self) -> None:
        """Валидация конфигурации."""
        if self.max_retry_attempts < 0:
            raise ValueError("Max retry attempts cannot be negative")
        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")
# Импорты для типизации
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
