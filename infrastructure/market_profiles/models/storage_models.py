"""
Модели данных для хранения паттернов маркет-мейкера.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Final, List, Optional

from domain.type_definitions.market_maker_types import (
    Accuracy,
    AverageReturn,
    Confidence,
    MarketMakerPatternType,
    PatternOutcome,
    SuccessCount,
    TotalCount,
)


class StorageStatus(Enum):
    """Статус хранилища."""

    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    READONLY = "readonly"


class DataIntegrityStatus(Enum):
    """Статус целостности данных."""

    VALID = "valid"
    CORRUPTED = "corrupted"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StorageStatistics:
    """
    Статистика хранилища паттернов.
    Attributes:
        total_patterns: Общее количество паттернов
        total_symbols: Общее количество символов
        total_successful_patterns: Общее количество успешных паттернов
        total_storage_size_bytes: Общий размер хранилища в байтах
        avg_pattern_size_bytes: Средний размер паттерна в байтах
        compression_ratio: Коэффициент сжатия
        cache_hit_ratio: Коэффициент попаданий в кэш
        avg_read_time_ms: Среднее время чтения в миллисекундах
        avg_write_time_ms: Среднее время записи в миллисекундах
        last_cleanup: Время последней очистки
        last_backup: Время последнего резервного копирования
        integrity_status: Статус целостности данных
        storage_status: Статус хранилища
        error_count: Количество ошибок
        warning_count: Количество предупреждений
    """

    total_patterns: int = 0
    total_symbols: int = 0
    total_successful_patterns: int = 0
    total_storage_size_bytes: int = 0
    avg_pattern_size_bytes: int = 0
    compression_ratio: float = 1.0
    cache_hit_ratio: float = 0.0
    avg_read_time_ms: float = 0.0
    avg_write_time_ms: float = 0.0
    last_cleanup: Optional[datetime] = None
    last_backup: Optional[datetime] = None
    integrity_status: DataIntegrityStatus = DataIntegrityStatus.UNKNOWN
    storage_status: StorageStatus = StorageStatus.ACTIVE
    error_count: int = 0
    warning_count: int = 0

    def __post_init__(self) -> None:
        """Валидация статистики после инициализации."""
        if self.total_patterns < 0:
            raise ValueError("total_patterns cannot be negative")
        if self.total_symbols < 0:
            raise ValueError("total_symbols cannot be negative")
        if self.total_successful_patterns < 0:
            raise ValueError("total_successful_patterns cannot be negative")
        if self.total_storage_size_bytes < 0:
            raise ValueError("total_storage_size_bytes cannot be negative")
        if self.avg_pattern_size_bytes < 0:
            raise ValueError("avg_pattern_size_bytes cannot be negative")
        if not (0.0 <= self.compression_ratio <= 10.0):
            raise ValueError("compression_ratio must be between 0.0 and 10.0")
        if not (0.0 <= self.cache_hit_ratio <= 1.0):
            raise ValueError("cache_hit_ratio must be between 0.0 and 1.0")
        if self.avg_read_time_ms < 0:
            raise ValueError("avg_read_time_ms cannot be negative")
        if self.avg_write_time_ms < 0:
            raise ValueError("avg_write_time_ms cannot be negative")
        if self.error_count < 0:
            raise ValueError("error_count cannot be negative")
        if self.warning_count < 0:
            raise ValueError("warning_count cannot be negative")

    @property
    def success_rate(self) -> float:
        """Коэффициент успешности паттернов."""
        if self.total_patterns == 0:
            return 0.0
        return self.total_successful_patterns / self.total_patterns

    @property
    def avg_patterns_per_symbol(self) -> float:
        """Среднее количество паттернов на символ."""
        if self.total_symbols == 0:
            return 0.0
        return self.total_patterns / self.total_symbols

    @property
    def storage_efficiency(self) -> float:
        """Эффективность хранения (сжатие + кэш)."""
        compression_efficiency = 1.0 / self.compression_ratio
        cache_efficiency = self.cache_hit_ratio
        return (compression_efficiency + cache_efficiency) / 2.0

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "total_patterns": self.total_patterns,
            "total_symbols": self.total_symbols,
            "total_successful_patterns": self.total_successful_patterns,
            "total_storage_size_bytes": self.total_storage_size_bytes,
            "avg_pattern_size_bytes": self.avg_pattern_size_bytes,
            "compression_ratio": self.compression_ratio,
            "cache_hit_ratio": self.cache_hit_ratio,
            "avg_read_time_ms": self.avg_read_time_ms,
            "avg_write_time_ms": self.avg_write_time_ms,
            "last_cleanup": (
                self.last_cleanup.isoformat() if self.last_cleanup else None
            ),
            "last_backup": self.last_backup.isoformat() if self.last_backup else None,
            "integrity_status": self.integrity_status.value,
            "storage_status": self.storage_status.value,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "success_rate": self.success_rate,
            "avg_patterns_per_symbol": self.avg_patterns_per_symbol,
            "storage_efficiency": self.storage_efficiency,
        }


@dataclass(frozen=True)
class PatternMetadata:
    """
    Метаданные паттерна.
    Attributes:
        symbol: Символ торговой пары
        pattern_type: Тип паттерна
        first_seen: Время первого появления
        last_seen: Время последнего появления
        total_count: Общее количество появлений
        success_count: Количество успешных появлений
        avg_accuracy: Средняя точность
        avg_return: Средняя доходность
        avg_confidence: Средняя уверенность
        avg_volume: Средний объем
        avg_spread: Средний спред
        avg_imbalance: Средний дисбаланс
        market_phases: Распределение по рыночным фазам
        volatility_regimes: Распределение по режимам волатильности
        liquidity_regimes: Распределение по режимам ликвидности
        time_distribution: Временное распределение
        volume_distribution: Распределение по объемам
        price_impact_distribution: Распределение по влиянию на цену
    """

    symbol: str
    pattern_type: MarketMakerPatternType
    first_seen: datetime
    last_seen: datetime
    total_count: TotalCount = TotalCount(0)
    success_count: SuccessCount = SuccessCount(0)
    avg_accuracy: Accuracy = Accuracy(0.0)
    avg_return: AverageReturn = AverageReturn(0.0)
    avg_confidence: Confidence = Confidence(0.0)
    avg_volume: float = 0.0
    avg_spread: float = 0.0
    avg_imbalance: float = 0.0
    market_phases: Dict[str, int] = field(default_factory=dict)
    volatility_regimes: Dict[str, int] = field(default_factory=dict)
    liquidity_regimes: Dict[str, int] = field(default_factory=dict)
    time_distribution: Dict[str, int] = field(default_factory=dict)
    volume_distribution: Dict[str, int] = field(default_factory=dict)
    price_impact_distribution: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация метаданных после инициализации."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.first_seen > self.last_seen:
            raise ValueError("first_seen cannot be later than last_seen")
        if self.total_count < 0:
            raise ValueError("total_count cannot be negative")
        if self.success_count < 0:
            raise ValueError("success_count cannot be negative")
        if self.success_count > self.total_count:
            raise ValueError("success_count cannot exceed total_count")
        if not (0.0 <= float(self.avg_accuracy) <= 1.0):
            raise ValueError("avg_accuracy must be between 0.0 and 1.0")
        if not (0.0 <= float(self.avg_confidence) <= 1.0):
            raise ValueError("avg_confidence must be between 0.0 and 1.0")
        if self.avg_volume < 0:
            raise ValueError("avg_volume cannot be negative")
        if self.avg_spread < 0:
            raise ValueError("avg_spread cannot be negative")

    @property
    def success_rate(self) -> float:
        """Коэффициент успешности."""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count

    @property
    def frequency_per_day(self) -> float:
        """Частота появления в день."""
        days = (self.last_seen - self.first_seen).days + 1
        if days == 0:
            return 0.0
        return self.total_count / days

    @property
    def is_reliable(self) -> bool:
        """Проверка надежности паттерна."""
        return (
            self.total_count >= 5
            and float(self.avg_accuracy) >= 0.6
            and self.success_rate >= 0.5
        )

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "symbol": self.symbol,
            "pattern_type": self.pattern_type.value,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "total_count": int(self.total_count),
            "success_count": int(self.success_count),
            "avg_accuracy": float(self.avg_accuracy),
            "avg_return": float(self.avg_return),
            "avg_confidence": float(self.avg_confidence),
            "avg_volume": self.avg_volume,
            "avg_spread": self.avg_spread,
            "avg_imbalance": self.avg_imbalance,
            "market_phases": dict(self.market_phases),
            "volatility_regimes": dict(self.volatility_regimes),
            "liquidity_regimes": dict(self.liquidity_regimes),
            "time_distribution": dict(self.time_distribution),
            "volume_distribution": dict(self.volume_distribution),
            "price_impact_distribution": dict(self.price_impact_distribution),
            "success_rate": self.success_rate,
            "frequency_per_day": self.frequency_per_day,
            "is_reliable": self.is_reliable,
        }


@dataclass(frozen=True)
class BehaviorRecord:
    """
    Запись поведения маркет-мейкера.
    Attributes:
        symbol: Символ торговой пары
        timestamp: Время записи
        pattern_type: Тип паттерна
        market_phase: Рыночная фаза
        volatility_regime: Режим волатильности
        liquidity_regime: Режим ликвидности
        volume_profile: Профиль объема
        price_action: Действие цены
        order_flow: Поток ордеров
        spread_behavior: Поведение спреда
        imbalance_behavior: Поведение дисбаланса
        pressure_behavior: Поведение давления
        reaction_time: Время реакции
        persistence: Устойчивость паттерна
        effectiveness: Эффективность
        risk_level: Уровень риска
        metadata: Дополнительные метаданные
    """

    symbol: str
    timestamp: datetime
    pattern_type: MarketMakerPatternType
    market_phase: str
    volatility_regime: str
    liquidity_regime: str
    volume_profile: Dict[str, float]
    price_action: Dict[str, float]
    order_flow: Dict[str, float]
    spread_behavior: Dict[str, float]
    imbalance_behavior: Dict[str, float]
    pressure_behavior: Dict[str, float]
    reaction_time: float
    persistence: float
    effectiveness: float
    risk_level: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Валидация записи поведения после инициализации."""
        if not self.symbol:
            raise ValueError("symbol cannot be empty")
        if self.reaction_time < 0:
            raise ValueError("reaction_time cannot be negative")
        if not (0.0 <= self.persistence <= 1.0):
            raise ValueError("persistence must be between 0.0 and 1.0")
        if not (0.0 <= self.effectiveness <= 1.0):
            raise ValueError("effectiveness must be between 0.0 and 1.0")
        if not self.risk_level:
            raise ValueError("risk_level cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "pattern_type": self.pattern_type.value,
            "market_phase": self.market_phase,
            "volatility_regime": self.volatility_regime,
            "liquidity_regime": self.liquidity_regime,
            "volume_profile": dict(self.volume_profile),
            "price_action": dict(self.price_action),
            "order_flow": dict(self.order_flow),
            "spread_behavior": dict(self.spread_behavior),
            "imbalance_behavior": dict(self.imbalance_behavior),
            "pressure_behavior": dict(self.pressure_behavior),
            "reaction_time": self.reaction_time,
            "persistence": self.persistence,
            "effectiveness": self.effectiveness,
            "risk_level": self.risk_level,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SuccessMapEntry:
    """
    Запись карты успешности паттернов.
    Attributes:
        pattern_type: Тип паттерна
        success_rate: Коэффициент успешности
        avg_return: Средняя доходность
        confidence: Уверенность в оценке
        sample_size: Размер выборки
        last_updated: Время последнего обновления
        market_conditions: Условия рынка
        time_periods: Временные периоды
        volume_ranges: Диапазоны объемов
        price_ranges: Диапазоны цен
        volatility_ranges: Диапазоны волатильности
    """

    pattern_type: MarketMakerPatternType
    success_rate: float
    avg_return: float
    confidence: Confidence
    sample_size: int
    last_updated: datetime
    market_conditions: Dict[str, float]
    time_periods: Dict[str, float]
    volume_ranges: Dict[str, float]
    price_ranges: Dict[str, float]
    volatility_ranges: Dict[str, float]

    def __post_init__(self) -> None:
        """Валидация записи карты успешности после инициализации."""
        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError("success_rate must be between 0.0 and 1.0")
        if not (0.0 <= float(self.confidence) <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")

    @property
    def is_reliable(self) -> bool:
        """Проверка надежности записи."""
        return self.sample_size >= 10 and float(self.confidence) >= 0.7

    @property
    def risk_adjusted_return(self) -> float:
        """Доходность с учетом риска."""
        if self.success_rate == 0:
            return 0.0
        return self.avg_return * self.success_rate

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь."""
        return {
            "pattern_type": self.pattern_type.value,
            "success_rate": self.success_rate,
            "avg_return": self.avg_return,
            "confidence": float(self.confidence),
            "sample_size": self.sample_size,
            "last_updated": self.last_updated.isoformat(),
            "market_conditions": dict(self.market_conditions),
            "time_periods": dict(self.time_periods),
            "volume_ranges": dict(self.volume_ranges),
            "price_ranges": dict(self.price_ranges),
            "volatility_ranges": dict(self.volatility_ranges),
            "is_reliable": self.is_reliable,
            "risk_adjusted_return": self.risk_adjusted_return,
        }
