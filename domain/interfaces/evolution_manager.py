"""
Протокол для менеджера эволюции стратегий.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

class EvolutionManagerProtocol(Protocol):
    """Протокол для управления эволюцией стратегий."""
    
    async def initialize(self) -> None:
        """Инициализация менеджера эволюции."""
        ...
    
    async def evolve_strategies(self, 
                              generation_count: int = 10,
                              population_size: int = 50) -> List[Dict[str, Any]]:
        """Эволюция стратегий."""
        ...
    
    async def evaluate_strategy(self, strategy_config: Dict[str, Any]) -> float:
        """Оценка производительности стратегии."""
        ...
    
    async def get_best_strategies(self, count: int = 5) -> List[Dict[str, Any]]:
        """Получение лучших стратегий."""
        ...
    
    async def archive_generation(self, generation_id: str, 
                               strategies: List[Dict[str, Any]]) -> None:
        """Архивирование поколения стратегий."""
        ...
    
    async def load_population(self, population_id: str) -> List[Dict[str, Any]]:
        """Загрузка популяции стратегий."""
        ...
    
    async def get_evolution_statistics(self) -> Dict[str, Any]:
        """Получение статистики эволюции."""
        ...
    
    async def start_evolution_cycle(self) -> None:
        """Запуск цикла эволюции."""
        ...
    
    async def stop_evolution_cycle(self) -> None:
        """Остановка цикла эволюции."""
        ...
    
    @property
    def is_running(self) -> bool:
        """Проверка, запущен ли процесс эволюции."""
        ...

class BaseEvolutionManager(ABC):
    """Базовый класс для менеджера эволюции."""
    
    def __init__(self):
        self._is_running = False
        self._current_generation = 0
        self._best_fitness = 0.0
    
    @abstractmethod
    async def initialize(self) -> None:
        """Инициализация менеджера."""
        pass
    
    @abstractmethod
    async def evolve_strategies(self, 
                              generation_count: int = 10,
                              population_size: int = 50) -> List[Dict[str, Any]]:
        """Эволюция стратегий."""
        pass
    
    @abstractmethod
    async def evaluate_strategy(self, strategy_config: Dict[str, Any]) -> float:
        """Оценка стратегии."""
        pass
    
    @property
    def is_running(self) -> bool:
        """Состояние процесса эволюции."""
        return self._is_running
    
    @property
    def current_generation(self) -> int:
        """Текущее поколение."""
        return self._current_generation
    
    @property
    def best_fitness(self) -> float:
        """Лучший показатель приспособленности."""
        return self._best_fitness