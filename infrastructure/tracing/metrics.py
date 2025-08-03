"""
Metrics implementation using OpenTelemetry.
"""

import functools
import logging
import time
from typing import Any, Callable, Dict, Optional, Union

try:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.resources import Resource

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry metrics not available. Metrics will be disabled.")

logger = logging.getLogger(__name__)

# Global meter provider
_meter_provider: Optional[MeterProvider] = None
_meter: Optional[metrics.Meter] = None


def initialize_metrics(
    service_name: str = "atb-trading-system",
    service_version: str = "1.0.0",
    environment: str = "development",
    console_export: bool = True,
) -> None:
    """Initialize OpenTelemetry metrics."""
    global _meter_provider, _meter

    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, metrics disabled")
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

        # Create metric readers
        readers = []
        if console_export:
            console_exporter = ConsoleMetricExporter()
            readers.append(PeriodicExportingMetricReader(console_exporter))

        # Create meter provider
        _meter_provider = MeterProvider(resource=resource, metric_readers=readers)

        # Set global meter provider
        metrics.set_meter_provider(_meter_provider)

        # Create meter
        _meter = metrics.get_meter(service_name, service_version)

        logger.info(f"Metrics initialized for service: {service_name}")

    except Exception as e:
        logger.error(f"Failed to initialize metrics: {e}")
        _meter_provider = None
        _meter = None


def get_meter() -> Optional[metrics.Meter]:
    """Get the global meter instance."""
    global _meter

    if _meter is None and OPENTELEMETRY_AVAILABLE:
        # Initialize with defaults if not already initialized
        initialize_metrics()

    return _meter


def record_metric(
    name: str,
    value: Union[int, float],
    metric_type: str = "counter",
    attributes: Optional[Dict[str, Any]] = None,
    description: str = "",
) -> None:
    """Record a metric value."""
    meter = get_meter()
    if meter is None:
        return

    try:
        if metric_type == "counter":
            counter = meter.create_counter(name, description=description)
            counter.add(value, attributes or {})
        elif metric_type == "gauge":
            gauge = meter.create_up_down_counter(name, description=description)
            gauge.add(value, attributes or {})
        elif metric_type == "histogram":
            histogram = meter.create_histogram(name, description=description)
            histogram.record(value, attributes or {})
        else:
            logger.warning(f"Unknown metric type: {metric_type}")

    except Exception as e:
        logger.error(f"Failed to record metric {name}: {e}")


def metric_counter(
    name: str, description: str = "", attributes: Optional[Dict[str, Any]] = None
) -> Callable[[Callable], Callable]:
    """Decorator to count function calls."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = func(*args, **kwargs)
                record_metric(name, 1, "counter", attributes, description)
                return result
            except Exception as e:
                record_metric(
                    f"{name}_errors",
                    1,
                    "counter",
                    attributes,
                    f"Errors in {description}",
                )
                raise

        return wrapper

    return decorator


def metric_timer(
    name: str, description: str = "", attributes: Optional[Dict[str, Any]] = None
) -> Callable[[Callable], Callable]:
    """Decorator to measure function execution time."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                record_metric(
                    name, execution_time, "histogram", attributes, description
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                record_metric(
                    f"{name}_errors",
                    execution_time,
                    "histogram",
                    attributes,
                    f"Errors in {description}",
                )
                raise

        return wrapper

    return decorator


def metric_async_timer(
    name: str, description: str = "", attributes: Optional[Dict[str, Any]] = None
) -> Callable[[Callable], Callable]:
    """Decorator to measure async function execution time."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                record_metric(
                    name, execution_time, "histogram", attributes, description
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                record_metric(
                    f"{name}_errors",
                    execution_time,
                    "histogram",
                    attributes,
                    f"Errors in {description}",
                )
                raise

        return wrapper

    return decorator


# Database metrics
def record_database_metric(
    operation: str,
    table: str,
    execution_time: float,
    success: bool = True,
    error_type: Optional[str] = None,
) -> None:
    """Record database operation metrics."""
    attributes = {"operation": operation, "table": table, "success": success}

    if error_type:
        attributes["error_type"] = error_type

    record_metric(
        "database_operations", 1, "counter", attributes, "Database operations count"
    )
    record_metric(
        "database_execution_time",
        execution_time,
        "histogram",
        attributes,
        "Database execution time",
    )


# Cache metrics
def record_cache_metric(
    operation: str,
    key: str,
    execution_time: float,
    hit: bool = True,
    success: bool = True,
) -> None:
    """Record cache operation metrics."""
    attributes = {"operation": operation, "hit": hit, "success": success}

    record_metric(
        "cache_operations", 1, "counter", attributes, "Cache operations count"
    )
    record_metric(
        "cache_execution_time",
        execution_time,
        "histogram",
        attributes,
        "Cache execution time",
    )


# HTTP metrics
def record_http_metric(
    method: str, url: str, status_code: int, execution_time: float
) -> None:
    """Record HTTP request metrics."""
    attributes = {"method": method, "url": url, "status_code": status_code}

    record_metric("http_requests", 1, "counter", attributes, "HTTP requests count")
    record_metric(
        "http_execution_time",
        execution_time,
        "histogram",
        attributes,
        "HTTP execution time",
    )


# Business metrics
def record_trading_metric(
    metric_name: str,
    value: Union[int, float],
    trading_pair: Optional[str] = None,
    strategy: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Record trading-specific metrics."""
    metric_attributes = attributes or {}

    if trading_pair:
        metric_attributes["trading_pair"] = trading_pair
    if strategy:
        metric_attributes["strategy"] = strategy

    record_metric(
        f"trading_{metric_name}",
        value,
        "counter",
        metric_attributes,
        f"Trading {metric_name}",
    )


def record_order_metric(
    order_type: str,
    trading_pair: str,
    quantity: float,
    price: float,
    success: bool = True,
) -> None:
    """Record order metrics."""
    attributes = {
        "order_type": order_type,
        "trading_pair": trading_pair,
        "success": success,
    }

    record_metric("orders_placed", 1, "counter", attributes, "Orders placed count")
    record_metric("order_volume", quantity, "counter", attributes, "Order volume")
    record_metric("order_value", quantity * price, "counter", attributes, "Order value")


def record_position_metric(
    trading_pair: str, side: str, quantity: float, pnl: Optional[float] = None
) -> None:
    """Record position metrics."""
    attributes = {"trading_pair": trading_pair, "side": side}

    record_metric(
        "positions_opened", 1, "counter", attributes, "Positions opened count"
    )
    record_metric("position_volume", quantity, "counter", attributes, "Position volume")

    if pnl is not None:
        record_metric("position_pnl", pnl, "gauge", attributes, "Position P&L")


# System metrics
def record_system_metric(
    metric_name: str,
    value: Union[int, float],
    component: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> None:
    """Record system metrics."""
    metric_attributes = attributes or {}
    metric_attributes["component"] = component

    record_metric(
        f"system_{metric_name}",
        value,
        "gauge",
        metric_attributes,
        f"System {metric_name}",
    )


def record_memory_usage(component: str, memory_mb: float) -> None:
    """Record memory usage for a component."""
    record_system_metric("memory_usage", memory_mb, component, {"unit": "MB"})


def record_cpu_usage(component: str, cpu_percent: float) -> None:
    """Record CPU usage for a component."""
    record_system_metric("cpu_usage", cpu_percent, component, {"unit": "percent"})


def record_error_metric(
    error_type: str, component: str, error_message: Optional[str] = None
) -> None:
    """Record error metrics."""
    attributes = {"error_type": error_type, "component": component}

    if error_message:
        attributes["error_message"] = error_message

    record_metric("errors", 1, "counter", attributes, "Error count")


def shutdown_metrics() -> None:
    """Shutdown metrics system."""
    global _meter_provider
    if _meter_provider is not None:
        try:
            _meter_provider.force_flush()
            _meter_provider.shutdown()
            logger.info("Metrics shutdown completed")
        except Exception as e:
            logger.error(f"Error during metrics shutdown: {e}")
        finally:
            _meter_provider = None
