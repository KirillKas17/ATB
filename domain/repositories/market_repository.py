"""
Репозиторий для работы с рыночными данными.
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from domain.entities.market import MarketData, MarketState
from domain.protocols.repository_protocol import RepositoryProtocol
from domain.entities.market import Market, MarketId


class MarketRepository(RepositoryProtocol):
    """Репозиторий для работы с рыночными данными."""

    @abstractmethod
    async def save_market_data(self, market_data: MarketData) -> MarketData:
        """Сохранение рыночных данных."""

    @abstractmethod
    async def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[MarketData]:
        """Получение рыночных данных."""

    @abstractmethod
    async def save_market_data_batch(
        self, market_data_list: List[MarketData]
    ) -> List[MarketData]:
        """Сохранение пакета рыночных данных."""

    @abstractmethod
    async def save_market_state(self, market_state: MarketState) -> MarketState:
        """Сохранение состояния рынка."""

    @abstractmethod
    async def get_market_state(self, symbol: str) -> Optional[MarketState]:
        """Получение состояния рынка."""

    @abstractmethod
    async def get_available_symbols(self) -> List[str]:
        """Получение доступных символов."""

    @abstractmethod
    async def get_available_timeframes(self) -> List[str]:
        """Получение доступных таймфреймов."""

    @abstractmethod
    async def delete_old_data(
        self, symbol: str, timeframe: str, before_date: datetime
    ) -> int:
        """Удаление старых данных."""


class InMemoryMarketRepository(MarketRepository):
    def __init__(self) -> None:
        """Инициализация репозитория."""
        self._markets: Dict[MarketId, Market] = {}
        self._market_data: Dict[MarketId, List[MarketData]] = {}
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._metrics: Dict[str, Any] = {}

    async def save(self, entity: Market) -> Market:
        """Сохранить рынок."""
        self._markets[entity.id] = entity
        return entity

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Market]:
        """Получить рынок по ID."""
        # Преобразуем entity_id в MarketId для поиска в словаре
        if isinstance(entity_id, str):
            market_id = MarketId(UUID(entity_id))
        else:  # isinstance(entity_id, UUID)
            market_id = MarketId(entity_id)
        market = self._markets.get(market_id)
        return market if market is not None else None

    async def update(self, entity: Market) -> Market:
        """Обновить рынок."""
        if entity.id in self._markets:
            self._markets[entity.id] = entity
        return entity

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить рынок."""
        # Преобразуем entity_id в MarketId для поиска в словаре
        if isinstance(entity_id, str):
            market_id = MarketId(UUID(entity_id))
        else:  # isinstance(entity_id, UUID)
            market_id = MarketId(entity_id)
        if market_id in self._markets:
            del self._markets[market_id]
            return True
        return False
