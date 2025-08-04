"""
Промышленные протоколы для репозиториев домена.
Обеспечивают типобезопасное хранение и управление доменными сущностями.
"""

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, AbstractAsyncContextManager, _AsyncGeneratorContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from types import TracebackType
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
    AsyncContextManager,
)
from uuid import UUID

from domain.entities.market import MarketData
from domain.entities.ml import Model, Prediction
from domain.entities.order import Order
from domain.entities.portfolio import Portfolio
from domain.entities.position import Position
from domain.entities.risk import RiskProfile
from domain.entities.strategy import Strategy
from domain.entities.trading import Trade
from domain.entities.account import Account
from domain.value_objects.money import Money
from domain.exceptions.protocol_exceptions import (
    ConnectionError,
    EntityDeleteError,
    EntityNotFoundError,
    EntitySaveError,
    EntityUpdateError,
    TransactionError,
    ValidationError,
)
from domain.type_definitions import (
    MetadataDict,
    OrderId,
    PortfolioId,
    PositionId,
    StrategyId,
    Symbol,
    TimestampValue,
    TradeId,
    TradingPair,
)
from domain.type_definitions.protocol_types import (
    HealthCheckDict,
    ModelId,
    PerformanceMetricsDict,
    QueryFilterDict,
    RiskProfileId,
)
from domain.type_definitions.repository_types import (
    BulkOperationResult,
    QueryFilter,
    QueryOperator,
    QueryOptions,
    RepositoryResponse,
)


class RepositoryState(Enum):
    """Состояния репозитория."""

    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"








@dataclass(frozen=True)
class SortOrder:
    """Порядок сортировки."""

    field: str
    direction: str = "asc"  # "asc" или "desc"


@dataclass(frozen=True)
class Pagination:
    """Пагинация."""

    page: int = 1
    page_size: int = 100
    offset: Optional[int] = None
    limit: Optional[int] = None


T = TypeVar("T")


@runtime_checkable
class TransactionProtocol(Protocol):
    """Протокол транзакции."""

    async def __aenter__(self) -> "TransactionProtocol": ...
    async def __aexit__(
        self,
        exc_type: type[BaseException],
        exc_val: BaseException,
        exc_tb: Optional[TracebackType],
    ) -> None: ...
    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def is_active(self) -> bool: ...
@runtime_checkable
class ConnectionProtocol(Protocol):
    """Протокол соединения с БД."""

    async def get_transaction(self) -> TransactionProtocol: ...
    async def close(self) -> None: ...
    async def is_connected(self) -> bool: ...
    async def ping(self) -> bool: ...
    async def get_connection_info(self) -> Dict[str, Any]: ...
class AsyncRepositoryProtocol(Protocol[T]):
    """Асинхронный протокол репозитория."""

    async def save(self, entity: T) -> T: ...
    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[T]: ...
    async def get_all(self, options: Optional[QueryOptions] = None) -> List[T]: ...
    async def update(self, entity: T) -> T: ...
    async def delete(self, entity_id: Union[UUID, str]) -> bool: ...
    async def exists(self, entity_id: Union[UUID, str]) -> bool: ...
    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int: ...
class RepositoryProtocol(ABC, Generic[T]):
    """
    Базовый промышленный протокол для репозиториев.
    Обеспечивает CRUD операции с типизированными доменными сущностями:
    - Создание и сохранение с валидацией
    - Чтение и поиск с фильтрацией и пагинацией
    - Обновление с оптимистичной блокировкой
    - Удаление с мягким удалением
    - Транзакционность и кэширование
    - Пакетные операции и мониторинг
    """

    def __init__(self) -> None:
        """Инициализация репозитория."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache: Dict[Union[UUID, str], T] = {}
        self._cache_ttl: Dict[Union[UUID, str], datetime] = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300  # 5 минут
        self._state = RepositoryState.DISCONNECTED
        self._connection_pool_size = 10
        self._query_timeout = 30.0

    @property
    def state(self) -> RepositoryState:
        """Текущее состояние репозитория."""
        return self._state

    # ============================================================================
    # БАЗОВЫЕ CRUD ОПЕРАЦИИ
    # ============================================================================
    @abstractmethod
    async def save(self, entity: T) -> T:
        """
        Сохранение сущности.
        Args:
            entity: Сущность для сохранения
        Returns:
            T: Сохраненная сущность
        Raises:
            EntitySaveError: Ошибка сохранения
            ValidationError: Ошибка валидации
        """
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[T]:
        """
        Получение сущности по ID.
        Args:
            entity_id: ID сущности
        Returns:
            Optional[T]: Сущность или None
        Raises:
            EntityNotFoundError: Сущность не найдена
        """
        pass

    @abstractmethod
    async def get_all(self, options: Optional[QueryOptions] = None) -> List[T]:
        """
        Получение всех сущностей с фильтрацией и пагинацией.
        Args:
            options: Опции запроса
        Returns:
            List[T]: Список сущностей
        """
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Обновление сущности.
        Args:
            entity: Сущность для обновления
        Returns:
            T: Обновленная сущность
        Raises:
            EntityNotFoundError: Сущность не найдена
            EntityUpdateError: Ошибка обновления
        """
        pass

    @abstractmethod
    async def delete(self, entity_id: Union[UUID, str]) -> bool:
        """
        Удаление сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: Успешность удаления
        Raises:
            EntityNotFoundError: Сущность не найдена
            EntityDeleteError: Ошибка удаления
        """
        pass

    @abstractmethod
    async def soft_delete(self, entity_id: Union[UUID, str]) -> bool:
        """
        Мягкое удаление сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: Успешность удаления
        """
        pass

    @abstractmethod
    async def restore(self, entity_id: Union[UUID, str]) -> bool:
        """
        Восстановление мягко удаленной сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: Успешность восстановления
        """
        pass

    # ============================================================================
    # ПОИСК И ФИЛЬТРАЦИЯ
    # ============================================================================
    @abstractmethod
    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[T]:
        """
        Поиск сущностей по фильтрам.
        Args:
            filters: Список фильтров
            options: Опции запроса
        Returns:
            List[T]: Список найденных сущностей
        """
        pass

    @abstractmethod
    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[T]:
        """
        Поиск одной сущности по фильтрам.
        Args:
            filters: Список фильтров
        Returns:
            Optional[T]: Найденная сущность или None
        """
        pass

    @abstractmethod
    async def exists(self, entity_id: Union[UUID, str]) -> bool:
        """
        Проверка существования сущности.
        Args:
            entity_id: ID сущности
        Returns:
            bool: Существует ли сущность
        """
        pass

    @abstractmethod
    async def count(self, filters: Optional[List[QueryFilter]] = None) -> int:
        """
        Подсчет количества сущностей.
        Args:
            filters: Фильтры для подсчета
        Returns:
            int: Количество сущностей
        """
        pass

    @abstractmethod
    def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[T]:
        """
        Потоковое чтение сущностей.
        Args:
            options: Опции запроса
            batch_size: Размер батча
        Yields:
            T: Сущности
        """
        pass

    # ============================================================================
    # ТРАНЗАКЦИИ
    # ============================================================================
    @abstractmethod
    def transaction(self) -> _AsyncGeneratorContextManager[TransactionProtocol, None]:
        """
        Контекстный менеджер для транзакций.
        Yields:
            TransactionProtocol: Транзакция
        """
        pass

    @abstractmethod
    async def execute_in_transaction(self, operation: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        Выполнение операции в транзакции.
        Args:
            operation: Операция для выполнения
            *args: Аргументы операции
            **kwargs: Ключевые аргументы операции
        Returns:
            Any: Результат операции
        """
        pass

    # ============================================================================
    # ПАКЕТНЫЕ ОПЕРАЦИИ
    # ============================================================================
    @abstractmethod
    async def bulk_save(self, entities: List[T]) -> BulkOperationResult:
        """
        Пакетное сохранение сущностей.
        Args:
            entities: Список сущностей
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def bulk_update(self, entities: List[T]) -> BulkOperationResult:
        """
        Пакетное обновление сущностей.
        Args:
            entities: Список сущностей
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def bulk_delete(
        self, entity_ids: List[Union[UUID, str]]
    ) -> BulkOperationResult:
        """
        Пакетное удаление сущностей.
        Args:
            entity_ids: Список ID сущностей
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def bulk_upsert(
        self, entities: List[T], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """
        Пакетное обновление или вставка сущностей.
        Args:
            entities: Список сущностей
            conflict_fields: Поля для разрешения конфликтов
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    # ============================================================================
    # КЭШИРОВАНИЕ
    # ============================================================================
    @abstractmethod
    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[T]:
        """
        Получение сущности из кэша.
        Args:
            key: Ключ кэша
        Returns:
            Optional[T]: Сущность из кэша или None
        """
        pass

    @abstractmethod
    async def set_cache(
        self, key: Union[UUID, str], entity: T, ttl: Optional[int] = None
    ) -> None:
        """
        Сохранение сущности в кэш.
        Args:
            key: Ключ кэша
            entity: Сущность
            ttl: Время жизни кэша в секундах
        """
        pass

    @abstractmethod
    async def invalidate_cache(self, key: Union[UUID, str]) -> None:
        """
        Инвалидация кэша.
        Args:
            key: Ключ кэша
        """
        pass

    @abstractmethod
    async def clear_cache(self) -> None:
        """Очистка всего кэша."""
        pass

    # ============================================================================
    # МОНИТОРИНГ И МЕТРИКИ
    # ============================================================================
    @abstractmethod
    async def get_repository_stats(self) -> RepositoryResponse:
        """
        Получение статистики репозитория.
        Returns:
            RepositoryResponse: Статистика репозитория
        """
        pass

    @abstractmethod
    async def get_performance_metrics(self) -> PerformanceMetricsDict:
        """
        Получение метрик производительности.
        Returns:
            PerformanceMetricsDict: Метрики производительности
        """
        pass

    @abstractmethod
    async def get_cache_stats(self) -> RepositoryResponse:
        """
        Получение статистики кэша.
        Returns:
            RepositoryResponse: Статистика кэша
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheckDict:
        """
        Проверка здоровья репозитория.
        Returns:
            HealthCheckDict: Статус здоровья
        """
        pass

    # ============================================================================
    # СПЕЦИАЛИЗИРОВАННЫЕ РЕПОЗИТОРИИ
    # ============================================================================


class TradingRepositoryProtocol(RepositoryProtocol[Order]):
    """Репозиторий для торговых операций."""

    @abstractmethod
    async def save_order(self, order: Order) -> Order:
        """
        Сохранение ордера.
        Args:
            order: Ордер для сохранения
        Returns:
            Order: Сохраненный ордер
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: OrderId) -> Optional[Order]:
        """
        Получение ордера по ID.
        Args:
            order_id: ID ордера
        Returns:
            Optional[Order]: Ордер или None
        """
        pass

    @abstractmethod
    async def get_orders_by_status(self, status: str) -> List[Order]:
        """
        Получение ордеров по статусу.
        Args:
            status: Статус ордеров
        Returns:
            List[Order]: Список ордеров
        """
        pass

    @abstractmethod
    async def get_orders_by_symbol(
        self, symbol: Symbol, limit: Optional[int] = None
    ) -> List[Order]:
        """
        Получение ордеров по символу.
        Args:
            symbol: Торговый символ
            limit: Лимит записей
        Returns:
            List[Order]: Список ордеров
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[Symbol] = None) -> List[Order]:
        """
        Получение открытых ордеров.
        Args:
            symbol: Фильтр по символу
        Returns:
            List[Order]: Список открытых ордеров
        """
        pass

    @abstractmethod
    async def save_trade(self, trade: Trade) -> Trade:
        """
        Сохранение сделки.
        Args:
            trade: Сделка для сохранения
        Returns:
            Trade: Сохраненная сделка
        """
        pass

    @abstractmethod
    async def get_trades_by_order(self, order_id: OrderId) -> List[Trade]:
        """
        Получение сделок по ордеру.
        Args:
            order_id: ID ордера
        Returns:
            List[Trade]: Список сделок
        """
        pass

    @abstractmethod
    async def get_trades_by_symbol(
        self,
        symbol: Symbol,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Trade]:
        """
        Получение сделок по символу.
        Args:
            symbol: Торговый символ
            start_date: Начальная дата
            end_date: Конечная дата
            limit: Лимит записей
        Returns:
            List[Trade]: Список сделок
        """
        pass

    @abstractmethod
    async def get_trade(
        self, 
        symbol: Optional[Symbol] = None, 
        limit: int = 100,
        account_id: Optional[str] = None
    ) -> List[Trade]:
        """
        Получение сделок с фильтрацией.
        Args:
            symbol: Фильтр по символу
            limit: Лимит записей
            account_id: Фильтр по аккаунту
        Returns:
            List[Trade]: Список сделок
        """
        pass

    @abstractmethod
    async def get_balance(self, account_id: Optional[str] = None) -> Dict[str, Money]:
        """
        Получение баланса аккаунта.
        Args:
            account_id: ID аккаунта (если None, возвращает баланс основного аккаунта)
        Returns:
            Dict[str, Money]: Баланс по валютам
        """
        pass

    @abstractmethod
    async def save_account(self, account: Account) -> Account:
        """
        Сохранение аккаунта.
        Args:
            account: Аккаунт для сохранения
        Returns:
            Account: Сохраненный аккаунт
        """
        pass

    @abstractmethod
    async def get_account(self, account_id: str) -> Optional[Account]:
        """
        Получение аккаунта по ID.
        Args:
            account_id: ID аккаунта
        Returns:
            Optional[Account]: Аккаунт или None
        """
        pass

    @abstractmethod
    async def update_account_balance(
        self, 
        account_id: str, 
        currency: str, 
        amount: Money
    ) -> bool:
        """
        Обновление баланса аккаунта.
        Args:
            account_id: ID аккаунта
            currency: Валюта
            amount: Сумма для обновления
        Returns:
            bool: Успешность обновления
        """
        pass

    @abstractmethod
    async def get_trading_statistics(
        self,
        symbol: Optional[Symbol] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> RepositoryResponse:
        """
        Получение торговой статистики.
        Args:
            symbol: Фильтр по символу
            start_date: Начальная дата
            end_date: Конечная дата
        Returns:
            RepositoryResponse: Торговая статистика
        """
        pass


class PortfolioRepositoryProtocol(RepositoryProtocol[Portfolio]):
    """Протокол репозитория портфелей с поддержкой позиций."""
    
    @abstractmethod
    async def get_by_id(self, entity_id: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель по ID."""
        ...
    
    @abstractmethod
    async def update(self, entity: Portfolio) -> Portfolio:
        """Обновить портфель."""
        ...
    
    @abstractmethod
    async def find_by(
        self, filters: List[QueryFilter], options: Optional[QueryOptions] = None
    ) -> List[Portfolio]:
        """Поиск портфелей по фильтрам."""
        ...
    
    @abstractmethod
    async def find_one_by(self, filters: List[QueryFilter]) -> Optional[Portfolio]:
        """Поиск одного портфеля по фильтрам."""
        ...
    
    @abstractmethod
    async def get_from_cache(self, key: Union[UUID, str]) -> Optional[Portfolio]:
        """Получить портфель из кэша."""
        ...
    
    @abstractmethod
    async def bulk_save(self, entities: List[Portfolio]) -> BulkOperationResult:
        """Пакетное сохранение портфелей."""
        ...
    
    @abstractmethod
    async def bulk_update(self, entities: List[Portfolio]) -> BulkOperationResult:
        """Пакетное обновление портфелей."""
        ...
    
    @abstractmethod
    async def bulk_upsert(
        self, entities: List[Portfolio], conflict_fields: List[str]
    ) -> BulkOperationResult:
        """Пакетное обновление или вставка портфелей."""
        ...
    """Репозиторий для портфелей."""

    @abstractmethod
    async def save_portfolio(self, portfolio: Portfolio) -> Portfolio:
        """
        Сохранение портфеля.
        Args:
            portfolio: Портфель для сохранения
        Returns:
            Portfolio: Сохраненный портфель
        """
        pass

    @abstractmethod
    async def get_portfolio(
        self,
        portfolio_id: PortfolioId = PortfolioId(
            UUID("00000000-0000-0000-0000-000000000000")
        ),
    ) -> Optional[Portfolio]:
        """
        Получение портфеля по ID.
        Args:
            portfolio_id: ID портфеля
        Returns:
            Optional[Portfolio]: Портфель или None
        """
        pass

    @abstractmethod
    async def save_position(self, position: Position) -> Position:
        """
        Сохранение позиции.
        Args:
            position: Позиция для сохранения
        Returns:
            Position: Сохраненная позиция
        """
        pass

    @abstractmethod
    async def get_position(self, position_id: PositionId) -> Optional[Position]:
        """
        Получение позиции по ID.
        Args:
            position_id: ID позиции
        Returns:
            Optional[Position]: Позиция или None
        """
        pass

    @abstractmethod
    async def get_positions_by_symbol(self, symbol: Symbol) -> List[Position]:
        """
        Получение позиций по символу.
        Args:
            symbol: Торговый символ
        Returns:
            List[Position]: Список позиций
        """
        pass

    @abstractmethod
    async def get_open_positions(
        self, portfolio_id: Optional[PortfolioId] = None
    ) -> List[Position]:
        """
        Получение открытых позиций.
        Args:
            portfolio_id: ID портфеля
        Returns:
            List[Position]: Список открытых позиций
        """
        pass

    @abstractmethod
    async def update_position(self, position: Position) -> Position:
        """
        Обновление позиции.
        Args:
            position: Позиция для обновления
        Returns:
            Position: Обновленная позиция
        """
        pass

    @abstractmethod
    async def calculate_portfolio_value(
        self, portfolio_id: PortfolioId, current_prices: Dict[Symbol, Decimal]
    ) -> Dict[str, Decimal]:
        """
        Расчет стоимости портфеля.
        Args:
            portfolio_id: ID портфеля
            current_prices: Текущие цены
        Returns:
            Dict[str, Decimal]: Стоимость портфеля
        """
        pass


class StrategyRepositoryProtocol(RepositoryProtocol[Strategy]):
    """Репозиторий для стратегий."""

    @abstractmethod
    async def save_strategy(self, strategy: Strategy) -> Strategy:
        """
        Сохранение стратегии.
        Args:
            strategy: Стратегия для сохранения
        Returns:
            Strategy: Сохраненная стратегия
        """
        pass

    @abstractmethod
    async def get_strategy(self, strategy_id: StrategyId) -> Optional[Strategy]:
        """
        Получение стратегии по ID.
        Args:
            strategy_id: ID стратегии
        Returns:
            Optional[Strategy]: Стратегия или None
        """
        pass

    @abstractmethod
    async def get_strategies_by_type(self, strategy_type: str) -> List[Strategy]:
        """
        Получение стратегий по типу.
        Args:
            strategy_type: Тип стратегии
        Returns:
            List[Strategy]: Список стратегий
        """
        pass

    @abstractmethod
    async def get_active_strategies(self) -> List[Strategy]:
        """
        Получение активных стратегий.
        Returns:
            List[Strategy]: Список активных стратегий
        """
        pass

    @abstractmethod
    async def update_strategy_performance(
        self, strategy_id: StrategyId, performance_metrics: Dict[str, Any]
    ) -> bool:
        """
        Обновление производительности стратегии.
        Args:
            strategy_id: ID стратегии
            performance_metrics: Метрики производительности
        Returns:
            bool: Успешность обновления
        """
        pass

    @abstractmethod
    async def get_strategy_performance_history(
        self,
        strategy_id: StrategyId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Получение истории производительности стратегии.
        Args:
            strategy_id: ID стратегии
            start_date: Начальная дата
            end_date: Конечная дата
        Returns:
            List[Dict[str, Any]]: История производительности
        """
        pass


class MarketRepositoryProtocol(RepositoryProtocol[MarketData]):
    """Репозиторий для рыночных данных."""

    @abstractmethod
    async def save_market_data(self, market_data: MarketData) -> MarketData:
        """
        Сохранение рыночных данных.
        Args:
            market_data: Рыночные данные для сохранения
        Returns:
            MarketData: Сохраненные рыночные данные
        """
        pass

    @abstractmethod
    async def get_market_data(
        self,
        symbol: Symbol,
        timeframe: str,
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[MarketData]:
        """
        Получение рыночных данных.
        Args:
            symbol: Торговый символ
            timeframe: Временной интервал
            limit: Лимит записей
            start_date: Начальная дата
            end_date: Конечная дата
        Returns:
            List[MarketData]: Список рыночных данных
        """
        pass

    @abstractmethod
    async def get_latest_market_data(self, symbol: Symbol) -> Optional[MarketData]:
        """
        Получение последних рыночных данных.
        Args:
            symbol: Торговый символ
        Returns:
            Optional[MarketData]: Последние рыночные данные или None
        """
        pass

    @abstractmethod
    async def bulk_save_market_data(
        self, market_data_list: List[MarketData]
    ) -> BulkOperationResult:
        """
        Пакетное сохранение рыночных данных.
        Args:
            market_data_list: Список рыночных данных
        Returns:
            BulkOperationResult: Результат операции
        """
        pass

    @abstractmethod
    async def cleanup_old_data(self, symbol: Symbol, older_than: datetime) -> int:
        """
        Очистка старых данных.
        Args:
            symbol: Торговый символ
            older_than: Дата, старше которой удалять
        Returns:
            int: Количество удаленных записей
        """
        pass

    @abstractmethod
    def stream(
        self, options: Optional[QueryOptions] = None, batch_size: int = 100
    ) -> AsyncIterator[MarketData]:
        """
        Потоковое чтение рыночных данных.
        Args:
            options: Опции запроса
            batch_size: Размер батча
        Yields:
            MarketData: Рыночные данные
        """
        pass

    @abstractmethod
    def transaction(self) -> _AsyncGeneratorContextManager[TransactionProtocol, None]:
        """
        Контекстный менеджер для транзакций.
        Yields:
            TransactionProtocol: Транзакция
        """
        pass


class RiskRepositoryProtocol(RepositoryProtocol[RiskProfile]):
    """Репозиторий для управления рисками."""

    @abstractmethod
    async def save_risk_profile(self, risk_profile: RiskProfile) -> RiskProfile:
        """
        Сохранение профиля риска.
        Args:
            risk_profile: Профиль риска для сохранения
        Returns:
            RiskProfile: Сохраненный профиль риска
        """
        pass

    @abstractmethod
    async def get_risk_profile(
        self, profile_id: RiskProfileId
    ) -> Optional[RiskProfile]:
        """
        Получение профиля риска по ID.
        Args:
            profile_id: ID профиля риска
        Returns:
            Optional[RiskProfile]: Профиль риска или None
        """
        pass

    @abstractmethod
    async def get_default_risk_profile(self) -> Optional[RiskProfile]:
        """
        Получение профиля риска по умолчанию.
        Returns:
            Optional[RiskProfile]: Профиль риска по умолчанию или None
        """
        pass

    @abstractmethod
    async def update_risk_limits(
        self, profile_id: RiskProfileId, risk_limits: Dict[str, Any]
    ) -> bool:
        """
        Обновление лимитов риска.
        Args:
            profile_id: ID профиля риска
            risk_limits: Новые лимиты риска
        Returns:
            bool: Успешность обновления
        """
        pass

    @abstractmethod
    async def get_risk_metrics(
        self,
        portfolio_id: PortfolioId,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """
        Получение метрик риска.
        Args:
            portfolio_id: ID портфеля
            start_date: Начальная дата
            end_date: Конечная дата
        Returns:
            Dict[str, float]: Метрики риска
        """
        pass


class MLRepositoryProtocol(RepositoryProtocol[Model]):
    """Репозиторий для ML моделей."""

    @abstractmethod
    async def save_model(self, model: Model) -> Model:
        """
        Сохранение ML модели.
        Args:
            model: Модель для сохранения
        Returns:
            Model: Сохраненная модель
        """
        pass

    @abstractmethod
    async def get_model(self, model_id: ModelId) -> Optional[Model]:
        """
        Получение ML модели по ID.
        Args:
            model_id: ID модели
        Returns:
            Optional[Model]: Модель или None
        """
        pass

    @abstractmethod
    async def get_models_by_type(self, model_type: str) -> List[Model]:
        """
        Получение моделей по типу.
        Args:
            model_type: Тип модели
        Returns:
            List[Model]: Список моделей
        """
        pass

    @abstractmethod
    async def get_best_model(self, model_type: str) -> Optional[Model]:
        """
        Получение лучшей модели по типу.
        Args:
            model_type: Тип модели
        Returns:
            Optional[Model]: Лучшая модель или None
        """
        pass

    @abstractmethod
    async def save_prediction(self, prediction: Prediction) -> Prediction:
        """
        Сохранение предсказания.
        Args:
            prediction: Предсказание для сохранения
        Returns:
            Prediction: Сохраненное предсказание
        """
        pass

    @abstractmethod
    async def get_predictions_by_model(
        self, model_id: ModelId, limit: int = 100
    ) -> List[Prediction]:
        """
        Получение предсказаний по модели.
        Args:
            model_id: ID модели
            limit: Лимит записей
        Returns:
            List[Prediction]: Список предсказаний
        """
        pass

    @abstractmethod
    async def get_latest_prediction(self, model_id: ModelId) -> Optional[Prediction]:
        """
        Получение последнего предсказания модели.
        Args:
            model_id: ID модели
        Returns:
            Optional[Prediction]: Последнее предсказание или None
        """
        pass

    @abstractmethod
    async def update_model_metrics(
        self, model_id: ModelId, metrics: Dict[str, float]
    ) -> bool:
        """
        Обновление метрик модели.
        Args:
            model_id: ID модели
            metrics: Новые метрики
        Returns:
            bool: Успешность обновления
        """
        pass


class OrderRepositoryProtocol(RepositoryProtocol[Order]):
    """Репозиторий для ордеров."""

    @abstractmethod
    async def save_order(self, order: Order) -> bool:
        """Сохранение ордера."""
        pass

    @abstractmethod
    async def get_order(self, order_id: OrderId) -> Optional[Order]:
        """Получение ордера по ID."""
        pass

    @abstractmethod
    async def get_orders_by_symbol(self, symbol: Symbol) -> List[Order]:
        """Получение ордеров по символу."""
        pass

    @abstractmethod
    async def get_orders_by_status(self, status: str) -> List[Order]:
        """Получение ордеров по статусу."""
        pass


class PositionRepositoryProtocol(RepositoryProtocol[Position]):
    """Репозиторий для позиций."""

    @abstractmethod
    async def save_position(self, position: Position) -> bool:
        """Сохранение позиции."""
        pass

    @abstractmethod
    async def get_position(self, position_id: PositionId) -> Optional[Position]:
        """Получение позиции по ID."""
        pass

    @abstractmethod
    async def get_positions_by_symbol(self, symbol: Symbol) -> List[Position]:
        """Получение позиций по символу."""
        pass

    @abstractmethod
    async def get_positions_by_portfolio(
        self, portfolio_id: PortfolioId
    ) -> List[Position]:
        """Получение позиций по портфелю."""
        pass


class TradingPairRepositoryProtocol(RepositoryProtocol[Any]):
    """Репозиторий для торговых пар."""

    @abstractmethod
    async def save_trading_pair(self, pair: Any) -> bool:
        """Сохранение торговой пары."""
        pass

    @abstractmethod
    async def get_trading_pair(self, pair_id: Union[UUID, str]) -> Optional[Any]:
        """Получение торговой пары по ID."""
        pass

    @abstractmethod
    async def get_all_trading_pairs(self) -> List[Any]:
        """Получение всех торговых пар."""
        pass
