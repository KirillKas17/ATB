from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    ClassVar,
    Coroutine,
    Dict,
    ForwardRef,
    Generic,
    Hashable,
    Iterator,
    List,
    Literal,
    Mapping,
    NewType,
    NoReturn,
    Optional,
    Protocol,
    Sequence,
    Sized,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    cast,
    final,
    get_args,
    get_origin,
    overload,
    runtime_checkable,
)
from typing_extensions import Annotated, Concatenate, ParamSpec, Self
from uuid import UUID
from datetime import datetime, timezone
from enum import Enum
from dataclasses import dataclass, field

# Универсальный тип идентификатора сущности
EntityId = NewType('EntityId', UUID)
# Типы для кэширования
CacheKey = NewType('CacheKey', str)
CacheTTL = NewType('CacheTTL', int)
CacheValue = TypeVar('CacheValue')
CacheValue_co = TypeVar('CacheValue_co', covariant=True)
# Типы для транзакций
TransactionId = NewType('TransactionId', UUID)
TransactionStatus = Literal['pending', 'committed', 'rolled_back', 'failed']
# Типы для метрик
MetricName = NewType('MetricName', str)
MetricValue = Union[int, float, str, bool, datetime]
# Типы для состояния
StateValue = Dict[str, Any]
HealthStatus = Literal['healthy', 'degraded', 'unhealthy', 'unknown']
# Типы для конфигурации
ConfigDict = Dict[str, Any]
# Типы для запросов
QueryResult = List[Any]
QueryCount = NewType('QueryCount', int)
TotalPages = NewType('TotalPages', int)
TotalRecords = NewType('TotalRecords', int)
PageSize = NewType('PageSize', int)
# Типы для сортировки и фильтрации
SortField = NewType('SortField', str)
FilterField = NewType('FilterField', str)
FilterOperator = NewType('FilterOperator', str)
# TypeVar для протоколов с правильными ограничениями
T = TypeVar('T')
R = TypeVar('R')
T_contra = TypeVar('T_contra', contravariant=True)
R_co = TypeVar('R_co', covariant=True)

# Критерии поиска для репозиториев
class CriteriaDict(TypedDict, total=False):
    field: str
    value: Any
    operator: str  # '=', '!=', '>', '<', 'in', 'like', etc.

# Параметры пагинации
class PaginationParams(TypedDict, total=False):
    limit: int
    offset: int

# Параметры сортировки
class SortParams(TypedDict, total=False):
    sort_by: str
    sort_order: str  # 'asc' | 'desc'

# Тип для сложных поисковых запросов
class AdvancedQuery(TypedDict, total=False):
    criteria: List[CriteriaDict]
    sort: Optional[SortParams]
    pagination: Optional[PaginationParams]

# Расширенные типы для репозиториев
class RepositoryMetrics(TypedDict, total=False):
    """Метрики репозитория."""
    total_entities: int
    cache_hit_rate: float
    avg_query_time: float
    error_rate: float
    last_cleanup: str  # Изменено с datetime на str для совместимости
    uptime_seconds: float
    memory_usage_mb: float
    disk_usage_mb: float

class CacheMetrics(TypedDict, total=False):
    """Метрики кэша."""
    hits: int
    misses: int
    hit_rate: float
    size: int
    max_size: int
    evictions: int
    avg_ttl: float

class TransactionMetrics(TypedDict, total=False):
    """Метрики транзакций."""
    total_transactions: int
    committed_transactions: int
    rolled_back_transactions: int
    avg_transaction_time: float
    concurrent_transactions: int
    deadlocks: int

class PerformanceMetrics(TypedDict, total=False):
    """Метрики производительности."""
    repository: RepositoryMetrics
    cache: CacheMetrics
    transactions: TransactionMetrics
    custom_metrics: Dict[str, Any]

class HealthCheckResult(TypedDict, total=False):
    """Результат проверки здоровья."""
    status: HealthStatus
    timestamp: datetime
    response_time_ms: float
    error_count: int
    last_error: Optional[str]
    uptime_seconds: float
    memory_usage_mb: float
    disk_usage_mb: float
    connection_status: str
    cache_status: str

class RepositoryConfig(TypedDict, total=False):
    """Конфигурация репозитория."""
    connection_string: str
    pool_size: int
    timeout: float
    retry_attempts: int
    cache_enabled: bool
    cache_ttl: int
    cache_max_size: int
    enable_metrics: bool
    enable_logging: bool
    log_level: str
    cleanup_interval: int
    max_entities: Optional[int]

# Enum для операторов запросов
class QueryOperator(Enum):
    """Операторы для запросов."""
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    ILIKE = "ilike"
    BETWEEN = "between"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    REGEX = "regex"

# Enum для статусов репозитория
class RepositoryState(Enum):
    """Состояния репозитория."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    INITIALIZING = "initializing"
    SHUTTING_DOWN = "shutting_down"

# Enum для типов кэша
class CacheType(Enum):
    """Типы кэширования."""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"

# Enum для стратегий эвакуации кэша
class CacheEvictionStrategy(Enum):
    """Стратегии эвакуации кэша."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    RANDOM = "random"
    TTL = "ttl"  # Time To Live

# Dataclass для фильтра запроса
@dataclass(frozen=True)
class QueryFilter:
    """Фильтр для запросов."""
    field: str
    operator: QueryOperator
    value: Any
    case_sensitive: bool = True

# Dataclass для сортировки
@dataclass(frozen=True)
class SortOrder:
    """Порядок сортировки."""
    field: str
    direction: Literal['asc', 'desc'] = 'asc'
    nulls_first: bool = False

# Dataclass для пагинации
@dataclass(frozen=True)
class Pagination:
    """Пагинация."""
    page: int = 1
    page_size: int = 100
    offset: Optional[int] = None
    limit: Optional[int] = None

# Dataclass для опций запроса
@dataclass
class QueryOptions:
    """Опции запроса."""
    filters: List[QueryFilter] = field(default_factory=list)
    sort_orders: List[SortOrder] = field(default_factory=list)
    pagination: Optional[Pagination] = None
    include_deleted: bool = False
    cache_result: bool = True
    timeout: Optional[float] = None
    max_results: Optional[int] = None

# Dataclass для результата пакетной операции
@dataclass
class BulkOperationResult:
    """Результат пакетной операции."""
    success_count: int
    error_count: int
    errors: List[Dict[str, Any]] = field(default_factory=list)
    processed_ids: List[Union[UUID, str]] = field(default_factory=list)
    execution_time: float = 0.0
    batch_size: int = 0

# Dataclass для ответа репозитория
@dataclass
class RepositoryResponse:
    """Ответ от репозитория."""
    success: bool
    data: Any
    total_count: Optional[int] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    cache_hit: bool = False

# Protocol для кэшируемых объектов
@runtime_checkable
class CacheableProtocol(Protocol[CacheValue_co]):
    """Протокол для кэшируемых объектов."""
    def get_cache_key(self) -> CacheKey: ...
    def get_cache_value(self) -> CacheValue_co: ...
    def get_cache_ttl(self) -> CacheTTL: ...
    def is_cache_valid(self) -> bool: ...

# Protocol для транзакционных объектов
@runtime_checkable
class TransactionalProtocol(Protocol):
    """Протокол для транзакционных объектов."""
    async def begin_transaction(self) -> TransactionId: ...
    async def commit_transaction(self, transaction_id: TransactionId) -> bool: ...
    async def rollback_transaction(self, transaction_id: TransactionId) -> bool: ...
    async def get_transaction_status(self, transaction_id: TransactionId) -> TransactionStatus: ...

# Protocol для запрашиваемых объектов
@runtime_checkable
class QueryableProtocol(Protocol):
    """Протокол для запрашиваемых объектов."""
    async def query(self, filters: List[QueryFilter], options: Optional[QueryOptions] = None) -> QueryResult: ...
    async def count(self, filters: Optional[List[QueryFilter]] = None) -> QueryCount: ...
    async def exists(self, filters: List[QueryFilter]) -> bool: ...

# Protocol для пагинируемых объектов
@runtime_checkable
class PaginatableProtocol(Protocol):
    """Протокол для пагинируемых объектов."""
    async def get_page(self, pagination: Pagination) -> QueryResult: ...
    async def get_total_pages(self, page_size: PageSize) -> TotalPages: ...
    async def get_total_records(self) -> TotalRecords: ...

# Protocol для сортируемых объектов
@runtime_checkable
class SortableProtocol(Protocol):
    """Протокол для сортируемых объектов."""
    async def sort(self, sort_orders: List[SortOrder]) -> QueryResult: ...
    async def get_sortable_fields(self) -> List[SortField]: ...

# Protocol для фильтруемых объектов
@runtime_checkable
class FilterableProtocol(Protocol):
    """Протокол для фильтруемых объектов."""
    async def filter(self, filters: List[QueryFilter]) -> QueryResult: ...
    async def get_filterable_fields(self) -> List[FilterField]: ...
    async def get_filter_operators(self, field: FilterField) -> List[FilterOperator]: ...

@dataclass
class RepositoryResult:
    """Результат операции репозитория."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    cache_hit: bool = False
    total_count: Optional[int] = None

@dataclass
class RepositoryError:
    """Ошибка репозитория."""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    retryable: bool = False

# Типизированные протоколы для операций
@runtime_checkable
class TransactionOperation(Protocol):
    """Протокол для типизированных транзакционных операций."""
    async def __call__(self, *args: Any) -> Any: ...

@runtime_checkable
class DatabaseOperation(Protocol):
    """Протокол для операций с базой данных."""
    async def __call__(self, connection: Any, *args: Any) -> Any: ...

@runtime_checkable
class RepositoryOperation(Protocol):
    """Протокол для операций репозитория."""
    async def __call__(self, *args: Any) -> Any: ... 