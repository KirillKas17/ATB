"""
Протокол для реестра торговых стратегий.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Type
from datetime import datetime
from enum import Enum

class StrategyStatus(Enum):
    """Статусы стратегий."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    ERROR = "error"

class StrategyType(Enum):
    """Типы стратегий."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    SCALPING = "scalping"
    SWING = "swing"
    NEWS = "news"
    SENTIMENT = "sentiment"

class StrategyRegistryProtocol(Protocol):
    """Протокол для реестра стратегий."""
    
    async def register_strategy(self, strategy_id: str, 
                              strategy_class: Type, 
                              config: Dict[str, Any]) -> bool:
        """Регистрация новой стратегии."""
        ...
    
    async def unregister_strategy(self, strategy_id: str) -> bool:
        """Отмена регистрации стратегии."""
        ...
    
    async def get_strategy(self, strategy_id: str) -> Optional[Any]:
        """Получение стратегии по ID."""
        ...
    
    async def get_all_strategies(self) -> Dict[str, Any]:
        """Получение всех зарегистрированных стратегий."""
        ...
    
    async def get_strategies_by_type(self, strategy_type: StrategyType) -> List[Any]:
        """Получение стратегий по типу."""
        ...
    
    async def get_active_strategies(self) -> List[Any]:
        """Получение активных стратегий."""
        ...
    
    async def update_strategy_status(self, strategy_id: str, 
                                   status: StrategyStatus) -> bool:
        """Обновление статуса стратегии."""
        ...
    
    async def update_strategy_config(self, strategy_id: str, 
                                   config: Dict[str, Any]) -> bool:
        """Обновление конфигурации стратегии."""
        ...
    
    async def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Получение показателей производительности стратегии."""
        ...
    
    async def backup_strategies(self) -> Dict[str, Any]:
        """Создание резервной копии стратегий."""
        ...
    
    async def restore_strategies(self, backup_data: Dict[str, Any]) -> bool:
        """Восстановление стратегий из резервной копии."""
        ...
    
    async def validate_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """Валидация стратегии."""
        ...

class BaseStrategyRegistry(ABC):
    """Базовый класс для реестра стратегий."""
    
    def __init__(self):
        self._strategies = {}
        self._strategy_configs = {}
        self._strategy_metadata = {}
        self._performance_cache = {}
    
    @abstractmethod
    async def _save_registry_state(self) -> None:
        """Сохранение состояния реестра."""
        pass
    
    @abstractmethod
    async def _load_registry_state(self) -> None:
        """Загрузка состояния реестра."""
        pass
    
    def _generate_strategy_id(self, strategy_name: str) -> str:
        """Генерация ID стратегии."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{strategy_name}_{timestamp}"
    
    def _validate_strategy_config(self, config: Dict[str, Any]) -> bool:
        """Валидация конфигурации стратегии."""
        required_fields = ['name', 'type', 'parameters']
        return all(field in config for field in required_fields)
    
    @property
    def strategy_count(self) -> int:
        """Количество зарегистрированных стратегий."""
        return len(self._strategies)
    
    @property
    def active_strategy_count(self) -> int:
        """Количество активных стратегий."""
        return sum(1 for meta in self._strategy_metadata.values() 
                  if meta.get('status') == StrategyStatus.ACTIVE)
    
    def get_strategy_ids(self) -> List[str]:
        """Получение списка ID всех стратегий."""
        return list(self._strategies.keys())
    
    def is_strategy_registered(self, strategy_id: str) -> bool:
        """Проверка регистрации стратегии."""
        return strategy_id in self._strategies