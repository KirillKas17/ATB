"""
Context utilities for distributed tracing.
"""

import contextvars
import logging
from typing import Any, Dict, Optional

try:
    from opentelemetry import trace
    from opentelemetry.trace import Span

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Context utilities will be disabled.")

logger = logging.getLogger(__name__)

# Context variables for storing additional context
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "user_id", default=None
)
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "session_id", default=None
)
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_current_span() -> Optional["Span"]:
    """Get the current active span."""
    if not OPENTELEMETRY_AVAILABLE:
        return None

    try:
        return trace.get_current_span()
    except Exception as e:
        logger.error(f"Failed to get current span: {e}")
        return None


def set_span_attribute(key: str, value: Any) -> None:
    """Set attribute on current span."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(key, str(value))
    except Exception as e:
        logger.error(f"Failed to set span attribute {key}: {e}")


def set_span_attributes(attributes: Dict[str, Any]) -> None:
    """Set multiple attributes on current span."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, str(value))
    except Exception as e:
        logger.error(f"Failed to set span attributes: {e}")


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add event to current span."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(name, attributes or {})
    except Exception as e:
        logger.error(f"Failed to add span event {name}: {e}")


def record_exception(
    exception: Exception, attributes: Optional[Dict[str, Any]] = None
) -> None:
    """Record exception in current span."""
    if not OPENTELEMETRY_AVAILABLE:
        return

    try:
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.record_exception(exception, attributes or {})
    except Exception as e:
        logger.error(f"Failed to record exception: {e}")


# Request context management
def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
) -> None:
    """Set request context variables."""
    if request_id is not None:
        _request_id.set(request_id)
    if user_id is not None:
        _user_id.set(user_id)
    if session_id is not None:
        _session_id.set(session_id)
    if correlation_id is not None:
        _correlation_id.set(correlation_id)

    # Add to current span if available
    if OPENTELEMETRY_AVAILABLE:
        current_span = trace.get_current_span()
        if current_span is not None and hasattr(current_span, "is_recording") and current_span.is_recording():
            if request_id:
                current_span.set_attribute("request.id", request_id)
            if user_id:
                current_span.set_attribute("user.id", user_id)
            if session_id:
                current_span.set_attribute("session.id", session_id)
            if correlation_id:
                current_span.set_attribute("correlation.id", correlation_id)


def get_request_context() -> Dict[str, Optional[str]]:
    """Get current request context."""
    return {
        "request_id": _request_id.get(),
        "user_id": _user_id.get(),
        "session_id": _session_id.get(),
        "correlation_id": _correlation_id.get(),
    }


def clear_request_context() -> None:
    """Clear request context variables."""
    _request_id.set(None)
    _user_id.set(None)
    _session_id.set(None)
    _correlation_id.set(None)


# Context managers for automatic context propagation
class RequestContext:
    """Context manager for request context."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.session_id = session_id
        self.correlation_id = correlation_id
        self._old_context: Dict[str, str | None] = {}

    def __enter__(self) -> "RequestContext":
        # Store old context
        self._old_context = get_request_context()

        # Set new context
        set_request_context(
            request_id=self.request_id,
            user_id=self.user_id,
            session_id=self.session_id,
            correlation_id=self.correlation_id,
        )

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore old context
        set_request_context(**self._old_context)


class SpanContext:
    """Context manager for span context."""

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span = None

    def __enter__(self) -> Any:
        if OPENTELEMETRY_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            self.span = tracer.start_as_current_span(self.name)

            # Set attributes
            for key, value in self.attributes.items():
                if self.span is not None:
                    self.span.set_attribute(key, str(value))

            # Add request context attributes
            request_context = get_request_context()
            for key, value in request_context.items():
                if value is not None and self.span is not None:
                    self.span.set_attribute(f"context.{key}", value)

        return self.span

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.span is not None:
            if exc_type is not None:
                self.span.record_exception(exc_val)
            self.span.end()


# Utility functions for common tracing patterns
def trace_database_query(table: str, operation: str, query: Optional[str] = None) -> Any:
    """Trace database query execution."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SpanContext(
                f"db.{operation}",
                {"db.table": table, "db.operation": operation, "db.query": query},
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_cache_operation(operation: str, key: str) -> Any:
    """Trace cache operation."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SpanContext(
                f"cache.{operation}", {"cache.operation": operation, "cache.key": key}
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_external_api(method: str, url: str) -> Any:
    """Trace external API call."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SpanContext(
                f"http.{method}", {"http.method": method, "http.url": url}
            ):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def trace_business_operation(operation: str, **attributes: Any) -> Any:
    """Trace business operation."""

    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with SpanContext(f"business.{operation}", attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Async context managers
class AsyncSpanContext:
    """Async context manager for span context."""

    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        self.name = name
        self.attributes = attributes or {}
        self.span = None

    async def __aenter__(self) -> Any:
        if OPENTELEMETRY_AVAILABLE:
            tracer = trace.get_tracer(__name__)
            self.span = tracer.start_as_current_span(self.name)

            # Set attributes
            for key, value in self.attributes.items():
                if self.span is not None:
                    self.span.set_attribute(key, str(value))

            # Add request context attributes
            request_context = get_request_context()
            for key, value in request_context.items():
                if value is not None and self.span is not None:
                    self.span.set_attribute(f"context.{key}", value)

        return self.span

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.span is not None:
            if exc_type is not None:
                self.span.record_exception(exc_val)
            self.span.end()


# Async utility functions
def trace_async_database_query(table: str, operation: str, query: Optional[str] = None) -> Any:
    """Trace async database query execution."""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with AsyncSpanContext(
                f"db.{operation}",
                {"db.table": table, "db.operation": operation, "db.query": query},
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_cache_operation(operation: str, key: str) -> Any:
    """Trace async cache operation."""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with AsyncSpanContext(
                f"cache.{operation}", {"cache.operation": operation, "cache.key": key}
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_external_api(method: str, url: str) -> Any:
    """Trace async external API call."""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with AsyncSpanContext(
                f"http.{method}", {"http.method": method, "http.url": url}
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_async_business_operation(operation: str, **attributes: Any) -> Any:
    """Trace async business operation."""

    def decorator(func: Any) -> Any:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with AsyncSpanContext(f"business.{operation}", attributes):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
