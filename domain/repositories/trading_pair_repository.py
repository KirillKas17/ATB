"""
Интерфейс репозитория для торговых пар
Определяет контракт для работы с торговыми парами
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

from ..entities.trading_pair import TradingPair


class TradingPairRepository(ABC):
    """
    Абстрактный репозиторий для работы с торговыми парами

    Определяет контракт для CRUD операций с торговыми парами
    """

    @abstractmethod
    async def save(self, trading_pair: TradingPair) -> TradingPair:
        """
        Сохранение торговой пары

        Args:
            trading_pair: Торговая пара для сохранения

        Returns:
            Сохраненная торговая пара

        Raises:
            RepositoryError: При ошибке сохранения
        """

    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> Optional[TradingPair]:
        """
        Получение торговой пары по символу

        Args:
            symbol: Символ торговой пары

        Returns:
            Торговая пара или None если не найдена

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_all(self, active_only: bool = True) -> List[TradingPair]:
        """
        Получение всех торговых пар

        Args:
            active_only: Только активные пары

        Returns:
            Список торговых пар

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def get_by_currencies(
        self, base_currency: str, quote_currency: str
    ) -> List[TradingPair]:
        """
        Получение торговых пар по валютам

        Args:
            base_currency: Код базовой валюты
            quote_currency: Код котируемой валюты

        Returns:
            Список торговых пар

        Raises:
            RepositoryError: При ошибке получения
        """

    @abstractmethod
    async def update(self, trading_pair: TradingPair) -> TradingPair:
        """
        Обновление торговой пары

        Args:
            trading_pair: Торговая пара для обновления

        Returns:
            Обновленная торговая пара

        Raises:
            RepositoryError: При ошибке обновления
        """

    @abstractmethod
    async def delete(self, symbol: str) -> bool:
        """
        Удаление торговой пары

        Args:
            symbol: Символ торговой пары

        Returns:
            True если удалена, False если не найдена

        Raises:
            RepositoryError: При ошибке удаления
        """

    @abstractmethod
    async def exists(self, symbol: str) -> bool:
        """
        Проверка существования торговой пары

        Args:
            symbol: Символ торговой пары

        Returns:
            True если существует, False иначе

        Raises:
            RepositoryError: При ошибке проверки
        """

    @abstractmethod
    async def count(self, active_only: bool = True) -> int:
        """
        Подсчет количества торговых пар

        Args:
            active_only: Только активные пары

        Returns:
            Количество торговых пар

        Raises:
            RepositoryError: При ошибке подсчета
        """

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> List[TradingPair]:
        """
        Поиск торговых пар

        Args:
            query: Поисковый запрос
            limit: Максимальное количество результатов

        Returns:
            Список найденных торговых пар

        Raises:
            RepositoryError: При ошибке поиска
        """

    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики по торговым парам

        Returns:
            Словарь со статистикой

        Raises:
            RepositoryError: При ошибке получения статистики
        """
