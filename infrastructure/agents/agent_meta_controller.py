"""
Мета-контроллер агент для координации других агентов.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class MetaControllerConfig:
    """Конфигурация мета-контроллера."""

    max_agents: int = 10
    coordination_interval: float = 1.0
    enable_learning: bool = True


class MetaControllerAgent(ABC):
    """Абстрактный мета-контроллер агент."""

    def __init__(self, config: Optional[MetaControllerConfig] = None) -> None:
        self.config = config or MetaControllerConfig()
        self.agents: Dict[str, Any] = {}
        self.coordination_active = False

    @abstractmethod
    async def coordinate_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Координация агентов."""

    @abstractmethod
    async def add_agent(self, agent_id: str, agent: Any) -> bool:
        """Добавление агента."""

    @abstractmethod
    async def remove_agent(self, agent_id: str) -> bool:
        """Удаление агента."""

    @abstractmethod
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Получение статуса агента."""


class DefaultMetaControllerAgent(MetaControllerAgent):
    """Реализация мета-контроллера по умолчанию."""

    async def coordinate_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Координация агентов."""
        try:
            results = {}
            for agent_id, agent in self.agents.items():
                if hasattr(agent, "process") and callable(agent.process):
                    results[agent_id] = await agent.process(context)
                else:
                    results[agent_id] = {"status": "no_process_method"}
            return results
        except Exception as e:
            return {"error": str(e)}

    async def add_agent(self, agent_id: str, agent: Any) -> bool:
        """Добавление агента."""
        if len(self.agents) >= self.config.max_agents:
            return False

        self.agents[agent_id] = agent
        return True

    async def remove_agent(self, agent_id: str) -> bool:
        """Удаление агента."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            return True
        return False

    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Получение статуса агента."""
        if agent_id not in self.agents:
            return None

        agent = self.agents[agent_id]
        return {
            "agent_id": agent_id,
            "type": type(agent).__name__,
            "active": hasattr(agent, "is_active") and agent.is_active,
        }
