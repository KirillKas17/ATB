"""
Health checker implementation.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
try:
    import redis
except ImportError:
    redis = None

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check result."""

    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    response_time: float = 0.0
    error: Optional[str] = None


class HealthChecker:
    """Main health checker class."""

    def __init__(self, name: str = "atb-trading-system") -> None:
        self.name = name
        self.checks: Dict[str, Callable] = {}
        self.check_results: Dict[str, HealthCheck] = {}
        self.last_check_time = datetime.now()
        self.check_interval = 30.0  # seconds
        self._lock = asyncio.Lock()
        logger.info(f"Health checker '{name}' initialized")

    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check function."""
        self.checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def unregister_check(self, name: str) -> bool:
        """Unregister a health check."""
        if name in self.checks:
            del self.checks[name]
            if name in self.check_results:
                del self.check_results[name]
            logger.info(f"Unregistered health check: {name}")
            return True
        return False

    async def run_check(self, name: str) -> HealthCheck:
        """Run a specific health check."""
        if name not in self.checks:
            return HealthCheck(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check '{name}' not found",
                error="Check not registered",
            )
        check_func = self.checks[name]
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, check_func)
            response_time = time.time() - start_time
            if isinstance(result, HealthCheck):
                result.response_time = response_time
                return result
            elif isinstance(result, dict):
                return HealthCheck(
                    name=name,
                    status=HealthStatus(result.get("status", "unknown")),
                    message=result.get("message", ""),
                    details=result.get("details", {}),
                    response_time=response_time,
                )
            else:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message="Check completed successfully",
                    response_time=response_time,
                )
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                response_time=response_time,
                error=str(e),
            )

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks."""
        async with self._lock:
            results = {}
            # Run checks concurrently
            tasks = []
            for name in self.checks:
                task = asyncio.create_task(self.run_check(name))
                tasks.append((name, task))
            # Wait for all checks to complete
            for name, task in tasks:
                try:
                    result = await task
                    results[name] = result
                    self.check_results[name] = result
                except Exception as e:
                    logger.error(f"Health check task '{name}' failed: {e}")
                    results[name] = HealthCheck(
                        name=name,
                        status=HealthStatus.UNHEALTHY,
                        message=f"Task failed: {str(e)}",
                        error=str(e),
                    )
            self.last_check_time = datetime.now()
            return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.check_results:
            return HealthStatus.UNKNOWN
        statuses = [check.status for check in self.check_results.values()]
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall_status = self.get_overall_status()
        report: Dict[str, Any] = {
            "name": self.name,
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "last_check": self.last_check_time.isoformat(),
            "checks": {},
            "summary": {
                "total_checks": len(self.check_results),
                "healthy_checks": len(
                    [
                        c
                        for c in self.check_results.values()
                        if c.status == HealthStatus.HEALTHY
                    ]
                ),
                "degraded_checks": len(
                    [
                        c
                        for c in self.check_results.values()
                        if c.status == HealthStatus.DEGRADED
                    ]
                ),
                "unhealthy_checks": len(
                    [
                        c
                        for c in self.check_results.values()
                        if c.status == HealthStatus.UNHEALTHY
                    ]
                ),
                "unknown_checks": len(
                    [
                        c
                        for c in self.check_results.values()
                        if c.status == HealthStatus.UNKNOWN
                    ]
                ),
            },
        }
        checks_dict: Dict[str, Any] = report["checks"]
        for name, check in self.check_results.items():
            checks_dict[name] = {
                "status": check.status.value,
                "message": check.message,
                "details": check.details,
                "timestamp": check.timestamp.isoformat(),
                "response_time": check.response_time,
                "error": check.error,
            }
        return report

    def get_ready_status(self) -> Dict[str, Any]:
        """Get readiness status for Kubernetes."""
        overall_status = self.get_overall_status()
        return {
            "status": (
                "ready"
                if overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
                else "not_ready"
            ),
            "timestamp": datetime.now().isoformat(),
            "details": {
                "overall_status": overall_status.value,
                "total_checks": len(self.check_results),
                "failed_checks": len(
                    [
                        c
                        for c in self.check_results.values()
                        if c.status == HealthStatus.UNHEALTHY
                    ]
                ),
            },
        }

    def get_live_status(self) -> Dict[str, Any]:
        """Get liveness status for Kubernetes."""
        overall_status = self.get_overall_status()
        return {
            "status": (
                "alive" if overall_status != HealthStatus.UNHEALTHY else "not_alive"
            ),
            "timestamp": datetime.now().isoformat(),
            "details": {
                "overall_status": overall_status.value,
                "last_check": self.last_check_time.isoformat(),
            },
        }

    async def start_monitoring(self, interval: float = 30.0) -> None:
        """Start continuous health monitoring."""
        self.check_interval = interval
        logger.info(f"Starting health monitoring with {interval}s interval")
        while True:
            try:
                await self.run_all_checks()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(interval)

    def get_check_history(self, name: str, limit: int = 10) -> List[HealthCheck]:
        """Get history of health check results."""
        # This would typically be stored in a database
        # For now, return current result
        if name in self.check_results:
            return [self.check_results[name]]
        return []

    def clear_history(self) -> None:
        """Clear health check history."""
        self.check_results.clear()
        logger.info("Health check history cleared")


# Utility functions for common health checks
async def check_database_connection(database_url: str) -> HealthCheck:
    """Check database connection health."""
    try:
        try:
            import asyncpg
        except ImportError:
            return HealthCheck(
                name="database_connection",
                status=HealthStatus.UNKNOWN,
                message="asyncpg not installed",
                error="asyncpg module not available",
            )

        start_time = time.time()
        conn = await asyncpg.connect(database_url)
        # Test query
        await conn.execute("SELECT 1")
        await conn.close()
        response_time = time.time() - start_time
        return HealthCheck(
            name="database_connection",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            details={"response_time": response_time},
            response_time=response_time,
        )
    except Exception as e:
        return HealthCheck(
            name="database_connection",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            error=str(e),
        )


async def check_cache_connection(cache_url: str) -> HealthCheck:
    """Check cache connection health."""
    try:
        if redis is None:
            return HealthCheck(
                name="cache_connection",
                status=HealthStatus.UNKNOWN,
                message="redis not installed",
                error="redis module not available",
            )
        
        start_time = time.time()
        r = redis.from_url(cache_url)
        # Test ping
        await r.ping()
        await r.close()
        response_time = time.time() - start_time
        return HealthCheck(
            name="cache_connection",
            status=HealthStatus.HEALTHY,
            message="Cache connection successful",
            details={"response_time": response_time},
            response_time=response_time,
        )
    except Exception as e:
        return HealthCheck(
            name="cache_connection",
            status=HealthStatus.UNHEALTHY,
            message=f"Cache connection failed: {str(e)}",
            error=str(e),
        )


async def check_exchange_connection(exchange_name: str, api_key: str) -> HealthCheck:
    """Check exchange connection health."""
    try:
        # This would be implemented based on your exchange client
        # For now, return a mock check
        await asyncio.sleep(0.1)  # Simulate API call
        return HealthCheck(
            name=f"exchange_{exchange_name}",
            status=HealthStatus.HEALTHY,
            message=f"Exchange {exchange_name} connection successful",
            details={"exchange": exchange_name},
        )
    except Exception as e:
        return HealthCheck(
            name=f"exchange_{exchange_name}",
            status=HealthStatus.UNHEALTHY,
            message=f"Exchange {exchange_name} connection failed: {str(e)}",
            error=str(e),
        )


async def check_system_resources() -> HealthCheck:
    """Check system resource usage."""
    try:
        try:
            import psutil
        except ImportError:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
                error="psutil module not available",
            )

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
        }
        # Determine status based on thresholds
        if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
            status = HealthStatus.UNHEALTHY
            message = "System resources critically low"
        elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
            status = HealthStatus.DEGRADED
            message = "System resources elevated"
        else:
            status = HealthStatus.HEALTHY
            message = "System resources normal"
        return HealthCheck(
            name="system_resources", status=status, message=message, details=details
        )
    except Exception as e:
        return HealthCheck(
            name="system_resources",
            status=HealthStatus.UNKNOWN,
            message=f"System resource check failed: {str(e)}",
            error=str(e),
        )


async def check_application_health() -> HealthCheck:
    """Check application-specific health."""
    try:
        # Add your application-specific health checks here
        # For example, check if critical services are running
        return HealthCheck(
            name="application_health",
            status=HealthStatus.HEALTHY,
            message="Application is healthy",
            details={"version": "1.0.0", "uptime": "running"},
        )
    except Exception as e:
        return HealthCheck(
            name="application_health",
            status=HealthStatus.UNHEALTHY,
            message=f"Application health check failed: {str(e)}",
            error=str(e),
        )
