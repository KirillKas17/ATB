"""
Доменная сущность исполнения стратегии.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, cast, TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from domain.entities.signal import Signal
    from domain.entities.strategy_performance import StrategyPerformance
    from domain.entities.trading import Trade

# Расширенные типы для metadata и config
ExtendedMetadataValue = Union[
    str,
    int,
    float,
    Decimal,
    bool,
    List[str],
    Dict[str, Union[str, int, float, Decimal, bool]],
]
ExtendedMetadataDict = Dict[str, ExtendedMetadataValue]
ExtendedConfigValue = Union[
    str,
    int,
    float,
    Decimal,
    bool,
    List[str],
    Dict[str, Union[str, int, float, Decimal, bool]],
]
ExtendedConfigDict = Dict[str, ExtendedConfigValue]


class ExecutionStatus(Enum):
    """Статусы исполнения стратегии."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ExecutionType(Enum):
    """Типы исполнения стратегии."""

    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    SIMULATION = "simulation"


@dataclass
class StrategyExecution:
    """Исполнение стратегии"""

    id: UUID = field(default_factory=uuid4)
    strategy_id: UUID = field(default_factory=uuid4)
    execution_type: ExecutionType = ExecutionType.PAPER
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    config: ExtendedConfigDict = field(default_factory=dict)
    signals: List["Signal"] = field(default_factory=list)
    trades: List["Trade"] = field(default_factory=list)
    performance: Optional["StrategyPerformance"] = field(
        default_factory=lambda: None  # Будет инициализировано в __post_init__
    )
    metadata: ExtendedMetadataDict = field(default_factory=dict)
    error_message: Optional[str] = None
    progress: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        """Пост-инициализация с валидацией"""
        if self.progress < Decimal("0") or self.progress > Decimal("1"):
            raise ValueError("Progress must be between 0 and 1")
        if self.start_time and self.end_time and self.start_time > self.end_time:
            raise ValueError("Start time cannot be after end time")
        # Инициализация performance если не задан
        if self.performance is None:
            from domain.entities.strategy_performance import StrategyPerformance
            self.performance = StrategyPerformance()

    @property
    def duration(self) -> Optional[Decimal]:
        """Длительность исполнения в секундах"""
        if self.start_time and self.end_time:
            duration_seconds = (self.end_time - self.start_time).total_seconds()
            return Decimal(str(duration_seconds))
        return None

    @property
    def is_active(self) -> bool:
        """Проверка активности исполнения"""
        return self.status in [ExecutionStatus.PENDING, ExecutionStatus.RUNNING]

    @property
    def is_completed(self) -> bool:
        """Проверка завершения исполнения"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        ]

    def add_signal(self, signal: "Signal") -> None:
        """Добавить сигнал к исполнению"""
        if TYPE_CHECKING:
            from domain.entities.signal import Signal
        signal.strategy_id = self.strategy_id
        self.signals.append(signal)

    def add_trade(self, trade: "Trade") -> None:
        """Добавить сделку к исполнению"""
        if TYPE_CHECKING:
            from domain.entities.trading import Trade
        self.trades.append(trade)

    def update_status(
        self, status: ExecutionStatus, error_message: Optional[str] = None
    ) -> None:
        """Обновить статус исполнения"""
        self.status = status
        if error_message:
            self.error_message = error_message
        if status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        ]:
            self.end_time = datetime.now()

    def update_progress(self, progress: Decimal) -> None:
        """Обновить прогресс исполнения"""
        if progress < Decimal("0") or progress > Decimal("1"):
            raise ValueError("Progress must be between 0 and 1")
        self.progress = progress

    def to_dict(
        self,
    ) -> Dict[
        str,
        Union[
            str,
            int,
            float,
            Decimal,
            bool,
            List[str],
            Dict[str, Union[str, int, float, Decimal, bool]],
            None,
        ],
    ]:
        """Преобразовать в словарь"""
        return {
            "id": str(self.id),
            "strategy_id": str(self.strategy_id),
            "execution_type": self.execution_type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "config": cast(
                Dict[str, Union[str, int, float, Decimal, bool]], self.config
            ),
            "signals": [str(signal.id) for signal in self.signals],
            "trades": [str(trade.id) for trade in self.trades],
            "performance": cast(
                Dict[str, Union[str, int, float, Decimal, bool]],
                self.performance.to_dict() if self.performance else {},
            ),
            "metadata": cast(
                Dict[str, Union[str, int, float, Decimal, bool]], self.metadata
            ),
            "error_message": self.error_message,
            "progress": str(self.progress),
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[
            str,
            Union[
                str,
                int,
                float,
                Decimal,
                bool,
                List[str],
                Dict[str, Union[str, int, float, Decimal, bool]],
                None,
            ],
        ],
    ) -> "StrategyExecution":
        """Создание из словаря"""
        # Безопасное извлечение и преобразование данных
        id_value = data.get("id", "")
        strategy_id_value = data.get("strategy_id", "")
        execution_type_value = data.get("execution_type", "paper")
        status_value = data.get("status", "pending")
        start_time_value = data.get("start_time", "")
        end_time_value = data.get("end_time")
        config_value = data.get("config", {})
        signals_value = data.get("signals", [])
        trades_value = data.get("trades", [])
        performance_value = data.get("performance", {})
        metadata_value = data.get("metadata", {})
        error_message = data.get("error_message")
        progress_value = data.get("progress", "0")
        # Преобразование UUID
        try:
            id_uuid = UUID(str(id_value)) if id_value else uuid4()
        except ValueError:
            id_uuid = uuid4()
        try:
            strategy_id = UUID(str(strategy_id_value)) if strategy_id_value else uuid4()
        except ValueError:
            strategy_id = uuid4()
        # Преобразование типа исполнения
        try:
            execution_type = ExecutionType(str(execution_type_value))
        except ValueError:
            execution_type = ExecutionType.PAPER
        # Преобразование статуса
        try:
            status = ExecutionStatus(str(status_value))
        except ValueError:
            status = ExecutionStatus.PENDING
        # Преобразование времени
        try:
            start_time = (
                datetime.fromisoformat(str(start_time_value))
                if start_time_value
                else datetime.now()
            )
        except ValueError:
            start_time = datetime.now()
        end_time = None
        if end_time_value is not None:
            try:
                end_time = datetime.fromisoformat(str(end_time_value))
            except ValueError:
                pass
        # Преобразование конфигурации
        if not isinstance(config_value, dict):
            config_value = {}
        # Преобразование сигналов
        signals: List["Signal"] = []
        # В to_dict() сохраняются только ID сигналов, поэтому восстанавливаем пустой список
        # Полные объекты сигналов должны восстанавливаться отдельно при необходимости
        
        # Преобразование сделок
        trades: List["Trade"] = []
        # В to_dict() сохраняются только ID сделок, поэтому восстанавливаем пустой список
        # Полные объекты сделок должны восстанавливаться отдельно при необходимости
        # Преобразование производительности
        try:
            from domain.entities.strategy_performance import StrategyPerformance

            if isinstance(performance_value, dict):
                performance = StrategyPerformance.from_dict(
                    cast(
                        Dict[
                            str,
                            Union[
                                str,
                                int,
                                float,
                                Decimal,
                                bool,
                                List[str],
                                Dict[str, Union[str, int, float, Decimal, bool]],
                            ],
                        ],
                        performance_value,
                    )
                )
            else:
                performance = StrategyPerformance()
        except (ValueError, TypeError):
            performance = StrategyPerformance()
        # Преобразование metadata
        if not isinstance(metadata_value, dict):
            metadata_value = {}
        # Преобразование прогресса
        try:
            progress = Decimal(str(progress_value))
        except (ValueError, TypeError):
            progress = Decimal("0")
        return cls(
            id=id_uuid,
            strategy_id=strategy_id,
            execution_type=execution_type,
            status=status,
            start_time=start_time,
            end_time=end_time,
            config=cast(ExtendedConfigDict, config_value),
            signals=signals,
            trades=trades,
            performance=performance,
            metadata=cast(ExtendedMetadataDict, metadata_value),
            error_message=cast(Optional[str], error_message),
            progress=progress,
        )

    def __str__(self) -> str:
        """Строковое представление исполнения"""
        return f"StrategyExecution({self.execution_type.value}, {self.status.value}, progress={self.progress})"

    def __repr__(self) -> str:
        """Представление для отладки"""
        return f"StrategyExecution(id={self.id}, strategy_id={self.strategy_id}, type={self.execution_type.value}, status={self.status.value})"



