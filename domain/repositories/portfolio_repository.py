from abc import abstractmethod
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from domain.entities.portfolio_fixed import Balance, Portfolio, Position
from domain.protocols.repository_protocol import RepositoryProtocol
from domain.types.repository_types import EntityId
from domain.types import PortfolioId
from domain.value_objects.currency import Currency


class PortfolioRepository(RepositoryProtocol):
    """Абстрактный репозиторий для портфеля"""

    @abstractmethod
    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Сохранить портфель"""

    @abstractmethod
    async def get_portfolio(self, portfolio_id: EntityId) -> Optional[Portfolio]:
        """Получить портфель по ID"""

    @abstractmethod
    async def get_portfolio_by_account(self, account_id: str) -> Optional[Portfolio]:
        """Получить портфель по ID аккаунта"""

    @abstractmethod
    async def update_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Обновить портфель"""

    @abstractmethod
    async def save_position(
        self, portfolio_id: EntityId, position: Position
    ) -> Position:
        """Сохранить позицию"""

    @abstractmethod
    async def get_position(
        self, portfolio_id: EntityId, trading_pair: str
    ) -> Optional[Position]:
        """Получить позицию"""

    @abstractmethod
    async def get_all_positions(self, portfolio_id: EntityId) -> List[Position]:
        """Получить все позиции портфеля"""

    @abstractmethod
    async def get_open_positions(self, portfolio_id: EntityId) -> List[Position]:
        """Получить открытые позиции"""

    @abstractmethod
    async def update_position(
        self, portfolio_id: EntityId, position: Position
    ) -> Position:
        """Обновить позицию"""

    @abstractmethod
    async def delete_position(self, portfolio_id: EntityId, trading_pair: str) -> bool:
        """Удалить позицию"""

    @abstractmethod
    async def save_balance(self, portfolio_id: EntityId, balance: Balance) -> Balance:
        """Сохранить баланс"""

    @abstractmethod
    async def get_balance(
        self, portfolio_id: EntityId, currency: Currency
    ) -> Optional[Balance]:
        """Получить баланс"""

    @abstractmethod
    async def get_all_balances(self, portfolio_id: EntityId) -> List[Balance]:
        """Получить все балансы портфеля"""

    @abstractmethod
    async def update_balance(self, portfolio_id: EntityId, balance: Balance) -> Balance:
        """Обновить баланс"""


class InMemoryPortfolioRepository(PortfolioRepository):
    """In-memory реализация репозитория портфеля"""

    def __init__(self) -> None:
        """Инициализация репозитория."""
        self._portfolios: Dict[PortfolioId, Portfolio] = {}
        self._cache: Dict[Union[UUID, str], Any] = {}
        self._metrics: Dict[str, Any] = {}
        # Добавляем недостающие атрибуты
        self.portfolios: Dict[EntityId, Portfolio] = {}
        self.positions: Dict[EntityId, Dict[str, Position]] = {}
        self.balances: Dict[EntityId, Dict[str, Balance]] = {}

    async def save(self, entity: Portfolio) -> Portfolio:
        """Сохранить портфель."""
        portfolio_id = PortfolioId(entity.id)
        self._portfolios[portfolio_id] = entity
        return entity

    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель по ID."""
        # Преобразуем entity_id в PortfolioId для поиска в словаре
        if isinstance(entity_id, str):
            portfolio_id = PortfolioId(UUID(entity_id))
        else:  # isinstance(entity_id, UUID)
            portfolio_id = PortfolioId(entity_id)
        return self._portfolios.get(portfolio_id)

    async def update(self, entity: Portfolio) -> Portfolio:
        """Обновить портфель."""
        portfolio_id = PortfolioId(entity.id)
        if portfolio_id in self._portfolios:
            self._portfolios[portfolio_id] = entity
        return entity

    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """Удалить портфель."""
        # Преобразуем entity_id в PortfolioId для поиска в словаре
        if isinstance(entity_id, str):
            portfolio_id = PortfolioId(UUID(entity_id))
        else:  # isinstance(entity_id, UUID)
            portfolio_id = PortfolioId(entity_id)
        if portfolio_id in self._portfolios:
            del self._portfolios[portfolio_id]
            return True
        return False

    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Сохранить портфель"""
        self.portfolios[EntityId(portfolio.id)] = portfolio
        self.positions[EntityId(portfolio.id)] = {}
        self.balances[EntityId(portfolio.id)] = {}
        return portfolio

    async def get_portfolio(self, portfolio_id: EntityId) -> Optional[Portfolio]:
        """Получить портфель по ID"""
        return self.portfolios.get(portfolio_id)

    async def get_portfolio_by_account(self, account_id: str) -> Optional[Portfolio]:
        """Получить портфель по ID аккаунта"""
        for portfolio in self.portfolios.values():
            if getattr(portfolio, "account_id", None) == account_id:
                return portfolio
        return None

    async def update_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """Обновить портфель"""
        if EntityId(portfolio.id) in self.portfolios:
            self.portfolios[EntityId(portfolio.id)] = portfolio
        return portfolio

    async def save_position(
        self, portfolio_id: EntityId, position: Position
    ) -> Position:
        """Сохранить позицию"""
        if portfolio_id not in self.positions:
            self.positions[portfolio_id] = {}
        self.positions[portfolio_id][position.trading_pair] = position
        return position

    async def get_position(
        self, portfolio_id: EntityId, trading_pair: str
    ) -> Optional[Position]:
        """Получить позицию"""
        return self.positions.get(portfolio_id, {}).get(trading_pair)

    async def get_all_positions(self, portfolio_id: EntityId) -> List[Position]:
        """Получить все позиции портфеля"""
        return list(self.positions.get(portfolio_id, {}).values())

    async def get_open_positions(self, portfolio_id: EntityId) -> List[Position]:
        """Получить открытые позиции"""
        positions = self.positions.get(portfolio_id, {})
        return [pos for pos in positions.values() if getattr(pos, "is_open", False)]

    async def update_position(
        self, portfolio_id: EntityId, position: Position
    ) -> Position:
        """Обновить позицию"""
        if portfolio_id in self.positions:
            self.positions[portfolio_id][position.trading_pair] = position
        return position

    async def delete_position(self, portfolio_id: EntityId, trading_pair: str) -> bool:
        """Удалить позицию"""
        if (
            portfolio_id in self.positions
            and trading_pair in self.positions[portfolio_id]
        ):
            del self.positions[portfolio_id][trading_pair]
            return True
        return False

    async def save_balance(self, portfolio_id: EntityId, balance: Balance) -> Balance:
        """Сохранить баланс"""
        if portfolio_id not in self.balances:
            self.balances[portfolio_id] = {}
        self.balances[portfolio_id][str(balance.currency)] = balance
        return balance

    async def get_balance(
        self, portfolio_id: EntityId, currency: Currency
    ) -> Optional[Balance]:
        """Получить баланс"""
        return self.balances.get(portfolio_id, {}).get(str(currency))

    async def get_all_balances(self, portfolio_id: EntityId) -> List[Balance]:
        """Получить все балансы портфеля"""
        return list(self.balances.get(portfolio_id, {}).values())

    async def update_balance(self, portfolio_id: EntityId, balance: Balance) -> Balance:
        """Обновить баланс"""
        if portfolio_id in self.balances:
            self.balances[portfolio_id][str(balance.currency)] = balance
        return balance
