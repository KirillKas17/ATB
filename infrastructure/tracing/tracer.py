"""
Distributed tracing implementation using OpenTelemetry.
"""

import functools
import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, Generator, TypeVar

F = TypeVar('F', bound=Callable[..., Any])

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        OTLPSpanExporter,
    )
    from opentelemetry.trace import Status, StatusCode

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Tracing will be disabled.")
logger = logging.getLogger(__name__)
# Global tracer provider
_tracer_provider: Optional[TracerProvider] = None
_tracer: Optional[trace.Tracer] = None


def initialize_tracing(
    service_name: str = "atb-trading-system",
    service_version: str = "1.0.0",
    environment: str = "development",
    jaeger_endpoint: Optional[str] = None,
    zipkin_endpoint: Optional[str] = None,
    otlp_endpoint: Optional[str] = None,
    console_export: bool = True,
) -> None:
    """Initialize OpenTelemetry tracing."""
    global _tracer_provider, _tracer
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, tracing disabled")
        return
    try:
        # Create resource
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
                "environment": environment,
            }
        )
        # Create tracer provider
        _tracer_provider = TracerProvider(resource=resource)
        # Add span processors
        if console_export:
            console_exporter = ConsoleSpanExporter()
            _tracer_provider.add_span_processor(BatchSpanProcessor(console_exporter))
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(":")[0],
                agent_port=int(jaeger_endpoint.split(":")[1]),
            )
            _tracer_provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        if zipkin_endpoint:
            zipkin_exporter = ZipkinExporter(endpoint=zipkin_endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(zipkin_exporter))
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        # Set global tracer provider
        trace.set_tracer_provider(_tracer_provider)
        # Create tracer
        _tracer = trace.get_tracer(service_name, service_version)
        logger.info(f"Tracing initialized for service: {service_name}")
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        _tracer_provider = None
        _tracer = None


def get_tracer() -> Optional[trace.Tracer]:
    """Get the current tracer instance."""
    global _tracer
    if _tracer is None:
        # Initialize with defaults if not already initialized
        initialize_tracing()
    return _tracer


def trace_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exceptions: bool = True,
) -> Callable[[F], F]:
    """Decorator to trace function execution."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)
            span_name = name or f"{func.__module__}.{func.__name__}"
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, str(value))
                    # Add function info
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.qualname", func.__qualname__)
                    # Execute function
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    # Record execution time
                    span.set_attribute("execution.time", execution_time)
                    return result
                except Exception as e:
                    if record_exceptions:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator


def trace_async_function(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    record_exceptions: bool = True,
) -> Callable[[F], F]:
    """Decorator to trace async function execution."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)
            span_name = name or f"{func.__module__}.{func.__name__}"
            with tracer.start_as_current_span(span_name) as span:
                try:
                    # Add function attributes
                    if attributes:
                        for key, value in attributes.items():
                            span.set_attribute(key, str(value))
                    # Add function info
                    span.set_attribute("function.module", func.__module__)
                    span.set_attribute("function.name", func.__name__)
                    span.set_attribute("function.qualname", func.__qualname__)
                    # Execute function
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    # Record execution time
                    span.set_attribute("execution.time", execution_time)
                    return result
                except Exception as e:
                    if record_exceptions:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exceptions: bool = True,
) -> Generator[Optional[trace.Span], None, None]:
    """Context manager for tracing spans."""
    tracer = get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        try:
            # Add attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            yield span
        except Exception as e:
            if record_exceptions:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def add_span_attribute(key: str, value: Any) -> None:
    """Add attribute to current span."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """Add event to current span."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(name, attributes or {})


def record_exception(
    exception: Exception, attributes: Optional[Dict[str, Any]] = None
) -> None:
    """Record exception in current span."""
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.record_exception(exception, attributes or {})


# Database tracing utilities
def trace_database_operation(operation: str, table: str, query: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to trace database operations."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(
                f"db.{operation}",
                {
                    "db.operation": operation,
                    "db.table": table,
                    "db.query": query,
                },
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# Cache tracing utilities
def trace_cache_operation(operation: str, key: str) -> Callable[[Callable], Callable]:
    """Decorator to trace cache operations."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(
                f"cache.{operation}",
                {
                    "cache.operation": operation,
                    "cache.key": key,
                },
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


# HTTP tracing utilities
def trace_http_request(method: str, url: str) -> Callable[[Callable], Callable]:
    """Decorator to trace HTTP requests."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(
                f"http.{method}",
                {
                    "http.method": method,
                    "http.url": url,
                },
            ):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def shutdown_tracing() -> None:
    """Shutdown tracing system."""
    global _tracer
    if _tracer is not None:
        # Force flush any pending spans
        _tracer = None
