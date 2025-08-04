"""
Интерфейс репозитория для позиций
Определяет контракт для работы с торговыми позициями
"""

from abc import abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID

from domain.entities.trading import Position, PositionSide
from domain.protocols.repository_protocol import QueryFilter, RepositoryProtocol
from domain.type_definitions import TradingPair


class PositionRepository(RepositoryProtocol):
    """
    Абстрактный репозиторий для работы с торговыми позициями

    Определяет контракт для CRUD операций с позициями
    """

    @abstractmethod
    async def save(self, position: Position) -> bool:
        """
        Сохранение позиции

        Args:
            position: Позиция для сохранения

        Returns:
            True если сохранено успешно

        Raises:
            RepositoryError: При ошибке сохранения
        """

    @abstractmethod
    async def get_by_id(self, position_id: Union[UUID, str]) -> Optional[Position]:
        """
        Получение позиции по ID

        Args:
            position_id: ID позиции

        Returns:
            Позиция или None если не найдена

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_by_trading_pair(
        self, trading_pair: TradingPair, open_only: bool = True
    ) -> List[Position]:
        """
        Получение позиций по торговой паре

        Args:
            trading_pair: Торговая пара
            open_only: Только открытые позиции

        Returns:
            Список позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_open_positions(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Position]:
        """
        Получение открытых позиций

        Args:
            trading_pair: Торговая пара (опционально)

        Returns:
            Список открытых позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_closed_positions(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """
        Получение закрытых позиций

        Args:
            trading_pair: Торговая пара (опционально)
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)

        Returns:
            Список закрытых позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_positions_by_side(
        self,
        side: PositionSide,
        trading_pair: Optional[TradingPair] = None,
        open_only: bool = True,
    ) -> List[Position]:
        """
        Получение позиций по стороне

        Args:
            side: Сторона позиций
            trading_pair: Торговая пара (опционально)
            open_only: Только открытые позиции

        Returns:
            Список позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_positions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        trading_pair: Optional[TradingPair] = None,
    ) -> List[Position]:
        """
        Получение позиций по диапазону дат

        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            trading_pair: Торговая пара (опционально)

        Returns:
            Список позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def update(self, position: Position) -> Position:
        """
        Обновление позиции

        Args:
            position: Позиция для обновления

        Returns:
            Обновленная позиция

        Raises:
            RepositoryError: При ошибке обновления
        """

    @abstractmethod
    async def delete(self, position_id: Union[UUID, str]) -> bool:
        """
        Удаление позиции

        Args:
            position_id: ID позиции

        Returns:
            True если удалена, False если не найдена

        Raises:
            RepositoryError: При ошибке удаления
        """

    @abstractmethod
    async def exists(self, position_id: Union[UUID, str]) -> bool:
        """
        Проверка существования позиции

        Args:
            position_id: ID позиции

        Returns:
            True если существует, False иначе

        Raises:
            RepositoryError: При ошибке проверки
        """

    @abstractmethod
    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """
        Подсчет количества позиций

        Args:
            filters: Фильтры для подсчета

        Returns:
            Количество позиций

        Raises:
            RepositoryError: При ошибке подсчета
        """

    @abstractmethod
    async def get_profitable_positions(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """
        Получение прибыльных позиций

        Args:
            trading_pair: Торговая пара (опционально)
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)

        Returns:
            Список прибыльных позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_losing_positions(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Position]:
        """
        Получение убыточных позиций

        Args:
            trading_pair: Торговая пара (опционально)
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)

        Returns:
            Список убыточных позиций

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_positions_with_stop_loss(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Position]:
        """
        Получение позиций со стоп-лоссом

        Args:
            trading_pair: Торговая пара (опционально)

        Returns:
            Список позиций со стоп-лоссом

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_positions_with_take_profit(
        self, trading_pair: Optional[TradingPair] = None
    ) -> List[Position]:
        """
        Получение позиций с тейк-профитом

        Args:
            trading_pair: Торговая пара (опционально)

        Returns:
            Список позиций с тейк-профитом

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_statistics(
        self,
        trading_pair: Optional[TradingPair] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict:
        """
        Получение статистики по позициям

        Args:
            trading_pair: Торговая пара (опционально)
            start_date: Начальная дата (опционально)
            end_date: Конечная дата (опционально)

        Returns:
            Словарь со статистикой

        Raises:
            RepositoryError: При ошибке получения статистики
        """

    @abstractmethod
    async def get_total_exposure(
        self, trading_pair: Optional[TradingPair] = None
    ) -> Dict:
        """
        Получение общего риска по позициям

        Args:
            trading_pair: Торговая пара (опционально)

        Returns:
            Словарь с данными о риске

        Raises:
            RepositoryError: При ошибке получения данных о риске
        """

    @abstractmethod
    async def cleanup_old_positions(self, before_date: datetime) -> int:
        """
        Очистка старых позиций

        Args:
            before_date: Дата, до которой удалять позиции

        Returns:
            Количество удаленных позиций

        Raises:
            RepositoryError: При ошибке очистки
        """

    @abstractmethod
    async def get_by_symbol(
        self, portfolio_id: UUID, symbol: str
    ) -> Optional[Position]:
        """
        Получение позиции по портфелю и символу
        Args:
            portfolio_id: ID портфеля
            symbol: Символ инструмента
        Returns:
            Позиция или None
        Raises:
            RepositoryError: При ошибке получения
        """
