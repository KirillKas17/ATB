"""
Интерфейсы для AI-улучшений кода.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from domain.types.entity_system_types import CodeStructure


class BaseAIEnhancement(ABC):
    """Базовый интерфейс для AI-улучшений кода."""

    @abstractmethod
    async def predict_code_quality(self, code_structure: CodeStructure) -> Dict[str, float]:
        """Предсказание качества кода."""
        pass

    @abstractmethod
    async def suggest_improvements(self, code_structure: CodeStructure) -> List[Dict[str, Any]]:
        """Предложение улучшений кода."""
        pass

    @abstractmethod
    async def optimize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Оптимизация параметров."""
        pass 