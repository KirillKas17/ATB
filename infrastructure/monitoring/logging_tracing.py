from contextlib import contextmanager
from typing import Any, Dict, Optional

from domain.types.monitoring_types import LogContext, LogEntry, LogLevel
from infrastructure.monitoring.logging_system import StructuredLogger


# RequestTracer и функции логирования
class RequestTracer:
    def __init__(self, logger: StructuredLogger) -> None:
        self.logger = logger

    @contextmanager
    def trace_request(self, request_id: Optional[str] = None, **kwargs: Any) -> Any:
        context = self.logger.get_current_context()
        new_context = LogContext(
            request_id=request_id or context.request_id,
            user_id=context.user_id,
            session_id=context.session_id,
            component=context.component,
            operation=context.operation,
            extra={**context.extra, **kwargs.get("extra", {})},
        )
        self.logger.set_context(new_context)
        try:
            yield
        finally:
            self.logger.set_context(context)

    def add_span(self, name: str, **kwargs: Any) -> None:
        # Пример: добавить спан в трейсы
        self.logger.info(f"Span: {name}", **kwargs)


_tracer: Optional[RequestTracer] = None


def get_tracer() -> RequestTracer:
    global _tracer
    if _tracer is None:
        from infrastructure.monitoring.logging_system import get_logger

        _tracer = RequestTracer(get_logger())
    return _tracer


def log_trace(message: str, **kwargs: Any) -> None:
    from infrastructure.monitoring.logging_system import get_logger

    get_logger().trace(message, **kwargs)


def log_debug(message: str, **kwargs: Any) -> None:
    from infrastructure.monitoring.logging_system import get_logger

    get_logger().debug(message, **kwargs)


def log_info(message: str, **kwargs: Any) -> None:
    from infrastructure.monitoring.logging_system import get_logger

    get_logger().info(message, **kwargs)


def log_warning(message: str, **kwargs: Any) -> None:
    from infrastructure.monitoring.logging_system import get_logger

    get_logger().warning(message, **kwargs)


def log_error(message: str, **kwargs: Any) -> None:
    from infrastructure.monitoring.logging_system import get_logger

    get_logger().error(message, **kwargs)


def log_critical(message: str, **kwargs: Any) -> None:
    from infrastructure.monitoring.logging_system import get_logger

    get_logger().critical(message, **kwargs)
