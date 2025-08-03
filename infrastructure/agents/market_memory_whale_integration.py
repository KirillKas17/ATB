"""
Интеграция памяти китов - основной файл.
Импортирует функциональность из модульной архитектуры.
"""

from .whale_memory.types import (
    WhaleActivity,
    WhaleActivityType,
    WhaleMemory,
    WhaleMemoryConfig,
    WhalePattern,
    WhaleQuery,
    WhaleSize,
)

__all__ = [
    "WhaleActivityType",
    "WhaleSize",
    "WhaleActivity",
    "WhalePattern",
    "WhaleMemory",
    "WhaleQuery",
    "WhaleMemoryConfig",
    "WhaleMemoryIntegration",
]
