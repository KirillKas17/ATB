from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict, Union

# =========================
# Метрики мониторинга
# =========================


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:
    name: str
    value: float
    type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class Alert:
    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    metric_name: Optional[str] = None
    threshold: Optional[float] = None
    current_value: Optional[float] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class TraceSpan:
    id: str
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children: List["TraceSpan"] = field(default_factory=list)


# =========================
# Логирование
# =========================


class LogLevel(Enum):
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class LogContext:
    request_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    timestamp: datetime
    level: LogLevel
    message: str
    context: LogContext
    exception: Optional[str] = None
    stack_trace: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None


# =========================
# Протоколы мониторинга
# =========================


class MetricCollectorProtocol(Protocol):
    def collect_metrics(self) -> Dict[str, float]: ...
    def get_metric(self, name: str) -> Optional[float]: ...
    def reset_metrics(self) -> None: ...


class AlertHandlerProtocol(Protocol):
    def handle_alert(self, alert: Alert) -> None: ...
    def get_alerts(self) -> List[Alert]: ...
    def resolve_alert(self, alert_id: str) -> bool: ...


class TraceProtocol(Protocol):
    def start_trace(self, name: str, tags: Optional[Dict[str, str]] = None) -> str: ...
    def end_trace(self, trace_id: str) -> None: ...
    def add_child_span(
        self, parent_id: str, name: str, tags: Optional[Dict[str, str]] = None
    ) -> str: ...


class LoggerProtocol(Protocol):
    def log(self, entry: LogEntry) -> None: ...
    def set_context(self, context: LogContext) -> None: ...
    def get_current_context(self) -> LogContext: ...
    def add_handler(self, handler: Callable[[LogEntry], None]) -> None: ...
