"""
Репозиторий для работы со стратегиями.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from domain.repositories.base_repository import BaseRepository
from domain.types.repository_types import EntityId, QueryOptions, QueryFilter


class StrategyRepository(BaseRepository[Strategy]):
    """Репозиторий для работы со стратегиями."""

    @abstractmethod
    async def save(self, strategy: Strategy) -> Strategy:
        """Сохранение стратегии."""

    @abstractmethod
    async def get_by_id(self, strategy_id: EntityId) -> Optional[Strategy]:
        """Получение стратегии по ID."""

    @abstractmethod
    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Strategy]:
        """Получение всех стратегий."""

    @abstractmethod
    async def delete(self, strategy_id: Union[UUID, str]) -> bool:
        """Удаление стратегии."""

    @abstractmethod
    async def get_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """Получение стратегий по типу."""

    @abstractmethod
    async def get_by_status(self, status: StrategyStatus) -> List[Strategy]:
        """Получение стратегий по статусу."""

    @abstractmethod
    async def get_by_trading_pair(self, trading_pair: str) -> List[Strategy]:
        """Получение стратегий по торговой паре."""

    @abstractmethod
    async def update_metrics(
        self, strategy_id: EntityId, metrics: Dict[str, Any]
    ) -> Strategy:
        """Обновление метрик стратегии."""

    @abstractmethod
    async def add_signal(self, strategy_id: EntityId, signal: Any) -> Strategy:
        """Добавление сигнала к стратегии."""

    @abstractmethod
    async def get_latest_signal(
        self, strategy_id: EntityId, trading_pair: str = ""
    ) -> Optional[Any]:
        """Получение последнего сигнала стратегии."""

    # Реализация методов базового протокола для совместимости
    async def find_by_id(self, id: EntityId) -> Optional[Strategy]:
        """Алиас для get_by_id для совместимости с базовым протоколом."""
        return await self.get_by_id(id)

    async def find_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Strategy]:
        """Получение всех стратегий с пагинацией."""
        strategies = await self.get_all()
        if limit:
            strategies = strategies[:limit]
        if offset:
            strategies = strategies[offset:]
        return strategies

    async def find_by_criteria(
        self, criteria: Dict[str, Any], limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Strategy]:
        """Поиск стратегий по критериям."""
        if "type" in criteria:
            strategies = await self.get_by_type(criteria["type"])
        elif "status" in criteria:
            strategies = await self.get_by_status(criteria["status"])
        elif "trading_pair" in criteria:
            strategies = await self.get_by_trading_pair(criteria["trading_pair"])
        else:
            strategies = await self.find_all(limit)
        if limit:
            strategies = strategies[:limit]
        if offset:
            strategies = strategies[offset:]
        return strategies

    async def update(self, entity: Strategy) -> Strategy:
        """Обновление стратегии."""
        return await self.save(entity)

    async def exists(self, id: Union[UUID, str]) -> bool:
        """Проверка существования стратегии."""
        strategy = await self.get_by_id(EntityId(id) if isinstance(id, UUID) else EntityId(UUID(id)))
        return strategy is not None

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """Подсчет количества стратегий."""
        if filters:
            # Преобразуем QueryFilter в критерии для совместимости
            criteria: Dict[str, Any] = {}
            for filter_obj in filters:
                criteria[filter_obj.field] = filter_obj.value
            strategies = await self.find_by_criteria(criteria)
        else:
            strategies = await self.get_all()
        return len(strategies)


class InMemoryStrategyRepository(StrategyRepository):
    def __init__(self) -> None:
        self._strategies: Dict[EntityId, Strategy] = {}
        self._signals: Dict[EntityId, List[Any]] = {}
        self._strategy_metrics: Dict[EntityId, Dict[str, Any]] = {}
        super().__init__()

    async def save(self, strategy: Strategy) -> Strategy:
        self._strategies[EntityId(strategy.id)] = strategy
        return strategy

    async def get_by_id(self, strategy_id: EntityId) -> Optional[Strategy]:
        return self._strategies.get(strategy_id)

    async def get_all(self, options: Optional[QueryOptions] = None) -> List[Strategy]:
        return list(self._strategies.values())

    async def delete(self, strategy_id: Union[UUID, str]) -> bool:
        entity_id = EntityId(strategy_id) if isinstance(strategy_id, UUID) else EntityId(UUID(strategy_id))
        return self._strategies.pop(entity_id, None) is not None

    async def get_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        return [
            s for s in self._strategies.values() if s.strategy_type == strategy_type
        ]

    async def get_by_status(self, status: StrategyStatus) -> List[Strategy]:
        return [s for s in self._strategies.values() if s.status == status]

    async def get_by_trading_pair(self, trading_pair: str) -> List[Strategy]:
        return [
            s
            for s in self._strategies.values()
            if getattr(s, "trading_pair", None) == trading_pair
        ]

    async def update_metrics(
        self, strategy_id: EntityId, metrics: Dict[str, Any]
    ) -> Strategy:
        if strategy_id not in self._strategies:
            raise ValueError("Strategy not found")
        self._strategy_metrics[strategy_id] = metrics
        return self._strategies[strategy_id]

    async def add_signal(self, strategy_id: EntityId, signal: Any) -> Strategy:
        if strategy_id not in self._strategies:
            raise ValueError("Strategy not found")
        self._signals.setdefault(strategy_id, []).append(signal)
        return self._strategies[strategy_id]

    async def get_latest_signal(
        self, strategy_id: EntityId, trading_pair: str = ""
    ) -> Optional[Any]:
        signals = self._signals.get(strategy_id, [])
        return signals[-1] if signals else None

    async def find_by_id(self, id: EntityId) -> Optional[Strategy]:
        return await self.get_by_id(id)

    async def find_all(
        self, limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Strategy]:
        strategies = list(self._strategies.values())
        if offset:
            strategies = strategies[offset:]
        if limit:
            strategies = strategies[:limit]
        return strategies

    async def find_by_criteria(
        self, criteria: Dict[str, Any], limit: Optional[int] = None, offset: Optional[int] = None
    ) -> List[Strategy]:
        result = list(self._strategies.values())
        for key, value in criteria.items():
            result = [s for s in result if getattr(s, key, None) == value]
        if offset:
            result = result[offset:]
        if limit:
            result = result[:limit]
        return result

    async def update(self, entity: Strategy) -> Strategy:
        return await self.save(entity)

    async def exists(self, id: Union[UUID, str]) -> bool:
        entity_id = EntityId(id) if isinstance(id, UUID) else EntityId(UUID(id))
        return entity_id in self._strategies

    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        if filters:
            # Преобразуем QueryFilter в критерии для совместимости
            criteria: Dict[str, Any] = {}
            for filter_obj in filters:
                criteria[filter_obj.field] = filter_obj.value
            return len(await self.find_by_criteria(criteria))
        return len(self._strategies)
