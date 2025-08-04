"""
Модуль трейсинга производительности.
Включает:
- Трейсинг операций
- Управление спанами
- Измерение производительности
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from domain.type_definitions.monitoring_types import TraceProtocol, TraceSpan


class PerformanceTracer(TraceProtocol):
    """
    Трейсер производительности.
    """

    def __init__(self) -> None:
        """Инициализация трейсера."""
        self.traces: Dict[str, TraceSpan] = {}
        self.active_spans: Dict[str, TraceSpan] = {}
        self._lock = threading.RLock()

    def start_trace(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """
        Начало трейсинга.
        Args:
            name: Имя трейса
            tags: Теги
        Returns:
            ID трейса
        """
        trace_id = f"trace_{int(time.time() * 1000000)}"
        span = TraceSpan(
            id=trace_id, name=name, start_time=datetime.now(), tags=tags or {}
        )
        with self._lock:
            self.traces[trace_id] = span
            self.active_spans[trace_id] = span
        logger.debug(f"Started trace: {name} (ID: {trace_id})")
        return trace_id

    def end_trace(self, trace_id: str) -> None:
        """
        Завершение трейсинга.
        Args:
            trace_id: ID трейса
        """
        with self._lock:
            if trace_id in self.active_spans:
                span = self.active_spans[trace_id]
                span.end_time = datetime.now()
                span.duration = (span.end_time - span.start_time).total_seconds()
                logger.debug(
                    f"Ended trace: {span.name} (ID: {trace_id}, Duration: {span.duration:.3f}s)"
                )
                del self.active_spans[trace_id]

    def add_child_span(
        self, parent_id: str, name: str, tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Добавление дочернего спана.
        Args:
            parent_id: ID родительского спана
            name: Имя спана
            tags: Теги
        Returns:
            ID дочернего спана
        """
        child_id = f"span_{int(time.time() * 1000000)}"
        child_span = TraceSpan(
            id=child_id,
            name=name,
            start_time=datetime.now(),
            parent_id=parent_id,
            tags=tags or {},
        )
        with self._lock:
            if parent_id in self.traces:
                self.traces[parent_id].children.append(child_span)
            self.traces[child_id] = child_span
            self.active_spans[child_id] = child_span
        logger.debug(f"Added child span: {name} (ID: {child_id}, Parent: {parent_id})")
        return child_id

    def get_traces(self, limit: int = 100) -> List[TraceSpan]:
        """
        Получение трейсов.
        Args:
            limit: Лимит записей
        Returns:
            Список трейсов
        """
        with self._lock:
            return list(self.traces.values())[-limit:]

    def get_active_traces(self) -> List[TraceSpan]:
        """
        Получение активных трейсов.
        Returns:
            Список активных трейсов
        """
        with self._lock:
            return list(self.active_spans.values())

    def clear_old_traces(self, max_age_hours: int = 24) -> int:
        """
        Очистка старых трейсов.
        Args:
            max_age_hours: Максимальный возраст в часах
        Returns:
            Количество удаленных трейсов
        """
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        removed_count = 0
        with self._lock:
            traces_to_remove = []
            for trace_id, span in self.traces.items():
                if span.start_time.timestamp() < cutoff_time:
                    traces_to_remove.append(trace_id)
            for trace_id in traces_to_remove:
                del self.traces[trace_id]
                removed_count += 1
        if removed_count > 0:
            logger.info(f"Cleared {removed_count} old traces")
        return removed_count

    def get_trace_statistics(self) -> Dict[str, Any]:
        """
        Получение статистики трейсов.
        Returns:
            Статистика трейсов
        """
        with self._lock:
            if not self.traces:
                return {
                    "total_traces": 0,
                    "active_traces": 0,
                    "avg_duration": 0.0,
                    "max_duration": 0.0,
                    "min_duration": 0.0,
                }
            durations = [
                span.duration
                for span in self.traces.values()
                if span.duration is not None
            ]
            return {
                "total_traces": len(self.traces),
                "active_traces": len(self.active_spans),
                "avg_duration": sum(durations) / len(durations) if durations else 0.0,
                "max_duration": max(durations) if durations else 0.0,
                "min_duration": min(durations) if durations else 0.0,
            }


# Глобальный экземпляр трейсера
_global_tracer: Optional[PerformanceTracer] = None


def get_tracer() -> PerformanceTracer:
    """Получение глобального трейсера."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = PerformanceTracer()
    return _global_tracer


def start_trace(name: str, tags: Optional[Dict[str, str]] = None) -> str:
    """Начало трейсинга."""
    return get_tracer().start_trace(name, tags)


def end_trace(trace_id: str) -> None:
    """Завершение трейсинга."""
    get_tracer().end_trace(trace_id)


def add_child_span(
    parent_id: str, name: str, tags: Optional[Dict[str, str]] = None
) -> str:
    """Добавление дочернего спана."""
    return get_tracer().add_child_span(parent_id, name, tags)


class PerformanceMetrics:
    """Метрики производительности."""
    
    def __init__(self) -> None:
        """Инициализация метрик."""
        self.cpu_usage: float = 0.0
        self.memory_usage: int = 0
        self.response_time: float = 0.0
        self.throughput: int = 0
        self.error_rate: float = 0.0
        self.timestamp: datetime = datetime.now()
    
    def update(self, **kwargs: Any) -> None:
        """Обновление метрик."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "error_rate": self.error_rate,
            "timestamp": self.timestamp.isoformat()
        }


class TraceContext:
    """Контекст трейсинга."""
    
    def __init__(self, trace_id: str, operation: str, **kwargs: Any) -> None:
        """Инициализация контекста."""
        self.trace_id: str = trace_id
        self.operation: str = operation
        self.start_time: datetime = datetime.now()
        self.end_time: Optional[datetime] = None
        self.status: str = "active"
        self.error: Optional[Exception] = None
        self.metadata: Dict[str, Any] = kwargs
    
    def end(self, status: str = "success", error: Optional[Exception] = None) -> None:
        """Завершение контекста."""
        self.end_time = datetime.now()
        self.status = status
        self.error = error
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "error": str(self.error) if self.error else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceContext":
        """Создание из словаря."""
        context = cls(data["trace_id"], data["operation"])
        context.start_time = datetime.fromisoformat(data["start_time"])
        if data["end_time"]:
            context.end_time = datetime.fromisoformat(data["end_time"])
        context.status = data["status"]
        context.metadata = data.get("metadata", {})
        return context
