"""
Модуль управления памятью Entity System.
"""

from .base import (
    BaseMemoryManager,
    BaseSnapshotManager,
    MemoryConfig,
    MemoryEvent,
    MemoryPolicy,
    SnapshotConfig,
    SnapshotFormat,
    SnapshotMetadata,
)

# from .memory_manager import MemoryManager
from .utils import (
    calculate_checksum,
    cleanup_old_files,
    decrypt_data,
    deserialize_snapshot,
    encrypt_data,
    estimate_object_size,
    generate_snapshot_id,
    serialize_snapshot,
    validate_snapshot_data,
)

__all__ = [
    # Types
    "MemoryPolicy",
    "SnapshotFormat",
    "SnapshotMetadata",
    "MemoryEvent",
    "SnapshotConfig",
    "MemoryConfig",
    "BaseSnapshotManager",
    "BaseMemoryManager",
    # Managers
    "SnapshotManager",
    # "MemoryManager",
    # Utils
    "generate_snapshot_id",
    "calculate_checksum",
    "serialize_snapshot",
    "deserialize_snapshot",
    "encrypt_data",
    "decrypt_data",
    "validate_snapshot_data",
    "estimate_object_size",
    "cleanup_old_files",
]
