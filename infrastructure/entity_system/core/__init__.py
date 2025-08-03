"""
Core-модули Entity System: оркестрация, планирование, ресурсы, координация, аналитика.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from .entity_analytics import EntityAnalytics
from .entity_controller import EntityController
from .entity_orchestrator import EntityOrchestrator
from .task_scheduler import TaskScheduler
from .resource_manager import ResourceManager
from .coordination_engine import CoordinationEngine

# Глобальные экземпляры
_entity_analytics: Optional[EntityAnalytics] = None

def get_entity_analytics() -> EntityAnalytics:
    """Получить глобальный экземпляр EntityAnalytics."""
    global _entity_analytics
    if _entity_analytics is None:
        _entity_analytics = EntityAnalytics()
    return _entity_analytics

async def force_entity_analysis() -> Dict[str, Any]:
    """Принудительно запустить анализ Entity System."""
    analytics = get_entity_analytics()
    if not analytics.is_running:
        await analytics.start()
    return {"status": "analysis_started", "timestamp": datetime.now().isoformat()}

def get_entity_status() -> Dict[str, Any]:
    """Получить статус Entity System."""
    analytics = get_entity_analytics()
    return analytics.get_status()

__all__ = [
    "EntityController",
    "EntityOrchestrator", 
    "TaskScheduler",
    "ResourceManager",
    "CoordinationEngine",
    "EntityAnalytics",
    "force_entity_analysis",
    "get_entity_status",
    "get_entity_analytics",
]
