"""
Entity System - Система управления сущностями и их взаимодействиями.
Модуль предоставляет:
- Управление жизненным циклом сущностей
- Координацию между компонентами системы
- Аналитику и мониторинг
- Эволюционную оптимизацию
- ИИ-улучшения
- Память и восприятие
- Эксперименты и A/B тестирование
- Применение улучшений с CI/CD
"""

# Импорты основных реализаций
from .entity_controller_impl import EntityControllerImpl
from .code_scanner_impl import CodeScannerImpl
from .code_analyzer_impl import CodeAnalyzerImpl
from .experiment_runner_impl import ExperimentRunnerImpl
from .improvement_applier_impl import ImprovementApplierImpl
from .memory_manager_impl import MemoryManagerImpl
from .ai_enhancement_impl import AIEnhancementImpl
from .evolution_engine_impl import EvolutionEngineImpl

# Основные реализации Entity System
# Дополнительные компоненты
__all__ = [
    # Основные реализации Entity System
    "EntityControllerImpl",
    "CodeScannerImpl",
    "CodeAnalyzerImpl",
    "ExperimentRunnerImpl",
    "ImprovementApplierImpl",
    "MemoryManagerImpl",
    "AIEnhancementImpl",
    "EvolutionEngineImpl",
    # Дополнительные компоненты
    "StrategyScanner",
    "SnapshotManager",
]
