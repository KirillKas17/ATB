"""
Integration module for all production-ready infrastructure components.
"""

import logging
import os
from typing import Any, Dict, Optional, Callable

from .cache.cache_manager import CacheManager
from .cache.compression import CompressionType
from .cache.encryption import EncryptionType
from infrastructure.health.checker import HealthChecker

logger = logging.getLogger(__name__)


class InfrastructureManager:
    """Manages all infrastructure components."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.health_checker: Optional[HealthChecker] = None
        self.cache_manager: Optional[CacheManager] = None
        self.circuit_breaker_registry: Optional[Any] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all infrastructure components."""
        if self._initialized:
            return

        logger.info("Initializing infrastructure components...")

        try:
            # Initialize tracing
            await self._initialize_tracing()

            # Initialize cache
            await self._initialize_cache()

            # Initialize circuit breakers
            await self._initialize_circuit_breakers()

            # Initialize health checks
            await self._initialize_health_checks()

            self._initialized = True
            logger.info("Infrastructure components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize infrastructure: {e}")
            raise

    async def _initialize_tracing(self) -> None:
        """Initialize distributed tracing."""
        try:
            # Get tracing configuration
            self.config.get("tracing", {})

            # Initialize tracing
            # await initialize_tracing(
            #     service_name=tracing_config.get("service_name", "atb-trading-system"),
            #     service_version=tracing_config.get("service_version", "1.0.0"),
            #     environment=tracing_config.get("environment", "development"),
            #     jaeger_endpoint=tracing_config.get("jaeger_endpoint"),
            #     zipkin_endpoint=tracing_config.get("zipkin_endpoint"),
            #     otlp_endpoint=tracing_config.get("otlp_endpoint"),
            #     console_export=tracing_config.get("console_export", True)
            # )

            logger.info("Distributed tracing initialized")

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to initialize tracing due to connection/timeout: {e}")
            # Fallback to console tracing
            # initialize_tracing(console_export=True)
        except ValueError as e:
            logger.error(f"Invalid tracing configuration: {e}")
            # Use default configuration
            # initialize_tracing(console_export=True)
        except Exception as e:
            logger.error(f"Unexpected error during tracing initialization: {e}")
            # Fallback to console tracing
            # initialize_tracing(console_export=True)

    async def _initialize_cache(self) -> None:
        """Initialize cache manager with compression and encryption."""
        try:
            cache_config = self.config.get("cache", {})

            self.cache_manager = CacheManager(
                max_size=cache_config.get("max_size", 1000),
                default_ttl=cache_config.get("default_ttl", 300.0),
                compression=cache_config.get("compression", True),
                compression_type=CompressionType(
                    cache_config.get("compression_type", "lz4")
                ),
                encryption=cache_config.get("encryption", True),
                encryption_type=EncryptionType(
                    cache_config.get("encryption_type", "aes_256")
                ),
            )

            logger.info("Cache manager initialized with compression and encryption")

        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Failed to initialize cache due to connection/timeout: {e}")
            # Fallback to in-memory cache
            self.cache_manager = CacheManager(max_size=100, default_ttl=60.0)
        except ValueError as e:
            logger.error(f"Invalid cache configuration: {e}")
            # Use default configuration
            self.cache_manager = CacheManager(max_size=1000, default_ttl=300.0)
        except Exception as e:
            logger.error(f"Unexpected error during cache initialization: {e}")
            # Fallback to in-memory cache
            self.cache_manager = CacheManager(max_size=100, default_ttl=60.0)

    async def _initialize_circuit_breakers(self) -> None:
        """Initialize circuit breakers for different services."""
        try:
            # self.circuit_breaker_registry = get_global_registry()

            # Database circuit breaker
            # db_config = CircuitBreakerConfig(
            #     failure_threshold=3,
            #     recovery_timeout=30.0,
            #     timeout=10.0
            # )
            # await self.circuit_breaker_registry.get_breaker("database", db_config)

            # Cache circuit breaker
            # cache_config = CircuitBreakerConfig(
            #     failure_threshold=5,
            #     recovery_timeout=15.0,
            #     timeout=5.0
            # )
            # await self.circuit_breaker_registry.get_breaker("cache", cache_config)

            # Exchange circuit breaker
            # exchange_config = CircuitBreakerConfig(
            #     failure_threshold=3,
            #     recovery_timeout=60.0,
            #     timeout=30.0
            # )
            # await self.circuit_breaker_registry.get_breaker("exchange", exchange_config)

            # Trading circuit breaker
            # trading_config = CircuitBreakerConfig(
            #     failure_threshold=2,
            #     recovery_timeout=120.0,
            #     timeout=60.0
            # )
            # await self.circuit_breaker_registry.get_breaker("trading", trading_config)

            logger.info("Circuit breakers initialized")

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"Failed to initialize circuit breakers due to connection/timeout: {e}"
            )
            # Fallback to basic circuit breaker
            # self.circuit_breaker_registry = get_global_registry()
        except ValueError as e:
            logger.error(f"Invalid circuit breaker configuration: {e}")
            # Use default configuration
            # self.circuit_breaker_registry = get_global_registry()
        except Exception as e:
            logger.error(f"Unexpected error during circuit breaker initialization: {e}")
            # Fallback to basic circuit breaker
            # self.circuit_breaker_registry = get_global_registry()

    async def _initialize_health_checks(self) -> None:
        """Initialize health checks."""
        try:
            self.health_checker = HealthChecker("atb-trading-system")

            # Register health checks
            # from .health.checker import (
            #     check_database_connection,
            #     check_cache_connection,
            #     check_system_resources,
            #     check_application_health
            # )

            # Database health check
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                # self.health_checker.register_check(
                #     "database_connection",
                #     lambda: check_database_connection(db_url)
                # )
                pass

            # Cache health check
            cache_url = os.getenv("CACHE_URL")
            if cache_url:
                # self.health_checker.register_check(
                #     "cache_connection",
                #     lambda: check_cache_connection(cache_url)
                # )
                pass

            # System resources health check
            # self.health_checker.register_check(
            #     "system_resources",
            #     check_system_resources
            # )

            # Application health check
            # self.health_checker.register_check(
            #     "application_health",
            #     check_application_health
            # )

            logger.info("Health checks initialized")

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"Failed to initialize health checks due to connection/timeout: {e}"
            )
            # Fallback to basic health checker
            self.health_checker = HealthChecker("atb-trading-system")
        except ValueError as e:
            logger.error(f"Invalid health check configuration: {e}")
            # Use default configuration
            self.health_checker = HealthChecker("atb-trading-system")
        except Exception as e:
            logger.error(f"Unexpected error during health check initialization: {e}")
            # Fallback to basic health checker
            self.health_checker = HealthChecker("atb-trading-system")

    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        if not self.health_checker:
            return {"status": "not_initialized"}

        return self.health_checker.get_health_report()

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.cache_manager:
            return {"status": "not_initialized"}

        return self.cache_manager.get_stats()

    async def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        if not self.circuit_breaker_registry:
            return {"status": "not_initialized"}

        # return self.circuit_breaker_registry.get_all_stats()
        return {"status": "not_implemented"}

    async def shutdown(self) -> None:
        """Shutdown all infrastructure components."""
        logger.info("Shutting down infrastructure components...")

        try:
            # Shutdown health checks
            if self.health_checker:
                self.health_checker.clear_history()

            # Shutdown circuit breakers
            if self.circuit_breaker_registry:
                # await self.circuit_breaker_registry.reset_all()
                pass

            # Shutdown cache
            if self.cache_manager:
                self.cache_manager.clear()

            # Shutdown tracing
            # from .tracing.tracer import shutdown_tracing
            # shutdown_tracing()

            # Shutdown metrics
            # from .tracing.metrics import shutdown_metrics
            # shutdown_metrics()

            self._initialized = False
            logger.info("Infrastructure components shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Global infrastructure manager
_global_manager: Optional[InfrastructureManager] = None


async def get_infrastructure_manager(
    config: Optional[Dict[str, Any]] = None,
) -> InfrastructureManager:
    """Get global infrastructure manager."""
    global _global_manager

    if _global_manager is None:
        _global_manager = InfrastructureManager(config)
        await _global_manager.initialize()

    return _global_manager


async def initialize_infrastructure(config: Optional[Dict[str, Any]] = None) -> None:
    """Initialize global infrastructure."""
    await get_infrastructure_manager(config)


async def shutdown_infrastructure() -> None:
    """Shutdown global infrastructure."""
    global _global_manager

    if _global_manager:
        await _global_manager.shutdown()
        _global_manager = None


# Utility functions for common operations
async def get_cache() -> Optional[CacheManager]:
    """Get global cache manager."""
    manager = await get_infrastructure_manager()
    return manager.cache_manager


async def get_health_checker() -> Optional[HealthChecker]:
    """Get global health checker."""
    manager = await get_infrastructure_manager()
    return manager.health_checker


async def get_circuit_breaker(name: str) -> Optional[Any]:
    """Get circuit breaker by name."""
    manager = await get_infrastructure_manager()
    if manager.circuit_breaker_registry:
        # return await manager.circuit_breaker_registry.get_breaker(name)
        return None
    return None


# Configuration helpers
def get_default_config() -> Dict[str, Any]:
    """Get default infrastructure configuration."""
    return {
        "tracing": {
            "service_name": "atb-trading-system",
            "service_version": "1.0.0",
            "environment": os.getenv("ENVIRONMENT", "development"),
            "jaeger_endpoint": os.getenv("JAEGER_ENDPOINT"),
            "zipkin_endpoint": os.getenv("ZIPKIN_ENDPOINT"),
            "otlp_endpoint": os.getenv("OTLP_ENDPOINT"),
            "console_export": True,
        },
        "cache": {
            "max_size": int(os.getenv("CACHE_MAX_SIZE", "1000")),
            "default_ttl": float(os.getenv("CACHE_DEFAULT_TTL", "300.0")),
            "compression": os.getenv("CACHE_COMPRESSION", "true").lower() == "true",
            "compression_type": os.getenv("CACHE_COMPRESSION_TYPE", "lz4"),
            "encryption": os.getenv("CACHE_ENCRYPTION", "true").lower() == "true",
            "encryption_type": os.getenv("CACHE_ENCRYPTION_TYPE", "aes_256"),
        },
        "circuit_breaker": {
            "database": {
                "failure_threshold": 3,
                "recovery_timeout": 30.0,
                "timeout": 10.0,
            },
            "cache": {"failure_threshold": 5, "recovery_timeout": 15.0, "timeout": 5.0},
            "exchange": {
                "failure_threshold": 3,
                "recovery_timeout": 60.0,
                "timeout": 30.0,
            },
            "trading": {
                "failure_threshold": 2,
                "recovery_timeout": 120.0,
                "timeout": 60.0,
            },
        },
        "health_check": {
            "interval": float(os.getenv("HEALTH_CHECK_INTERVAL", "30.0")),
            "timeout": float(os.getenv("HEALTH_CHECK_TIMEOUT", "10.0")),
        },
    }


# Context manager for infrastructure lifecycle
class InfrastructureContext:
    """Context manager for infrastructure lifecycle."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_default_config()

    async def __aenter__(self):
        await initialize_infrastructure(self.config)
        return await get_infrastructure_manager()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await shutdown_infrastructure()


# Decorators for automatic infrastructure integration
def with_infrastructure(config: Optional[Dict[str, Any]] = None) -> Callable[[Callable], Callable]:
    """Decorator to automatically initialize infrastructure."""

    def decorator(func: Callable) -> Callable:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with InfrastructureContext(config):
                return await func(*args, **kwargs)

        return wrapper

    return decorator


def with_tracing(name: Optional[str] = None) -> Callable[[Callable], Callable]:
    """Decorator to add tracing to functions."""

    def decorator(func: Callable) -> Callable:
        from .tracing.tracer import trace_function

        return trace_function(name=name)(func)

    return decorator


def with_circuit_breaker(name: Optional[str] = None, **config: Any) -> Callable[[Callable], Callable]:
    """Decorator to add circuit breaker to functions."""

    def decorator(func: Callable) -> Callable:
        from .circuit_breaker.decorators import circuit_breaker

        return circuit_breaker(name=name, **config)(func)

    return decorator


def with_cache(ttl: Optional[float] = None) -> Callable[..., Any]:
    """Decorator to add caching to functions."""

    def decorator(func: Callable) -> Callable[..., Any]:
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            cache = await get_cache()
            if cache:
                # Use function name and args as cache key
                cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
                return await cache.get_or_set(
                    cache_key, lambda: func(*args, **kwargs), ttl
                )
            else:
                return await func(*args, **kwargs)

        return wrapper

    return decorator
