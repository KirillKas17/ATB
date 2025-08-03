"""
Health monitoring classes for system components.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Status of health check."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    status: HealthStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None


class BaseHealthMonitor(ABC):
    """Base class for health monitors."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def check_health(self) -> HealthCheckResult:
        """Perform health check."""
        pass
    
    async def is_healthy(self) -> bool:
        """Check if component is healthy."""
        result = await self.check_health()
        return result.status == HealthStatus.HEALTHY


class CacheHealthMonitor(BaseHealthMonitor):
    """Monitor for cache health."""
    
    def __init__(self, cache_client: Optional[Any] = None) -> None:
        super().__init__("cache")
        self.cache_client = cache_client
    
    async def check_health(self) -> HealthCheckResult:
        """Check cache health."""
        try:
            if not self.cache_client:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    message="Cache client not configured"
                )
            
            # Simple ping test
            await self.cache_client.ping()
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Cache is healthy"
            )
        except Exception as e:
            self.logger.error(f"Cache health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Cache health check failed: {str(e)}"
            )


class DatabaseHealthMonitor(BaseHealthMonitor):
    """Monitor for database health."""
    
    def __init__(self, db_client: Optional[Any] = None) -> None:
        super().__init__("database")
        self.db_client = db_client
    
    async def check_health(self) -> HealthCheckResult:
        """Check database health."""
        try:
            if not self.db_client:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    message="Database client not configured"
                )
            
            # Simple connection test
            await self.db_client.execute("SELECT 1")
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Database is healthy"
            )
        except Exception as e:
            self.logger.error(f"Database health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Database health check failed: {str(e)}"
            )


class ExchangeHealthMonitor(BaseHealthMonitor):
    """Monitor for exchange connectivity."""
    
    def __init__(self, exchange_client: Optional[Any] = None) -> None:
        super().__init__("exchange")
        self.exchange_client = exchange_client
    
    async def check_health(self) -> HealthCheckResult:
        """Check exchange health."""
        try:
            if not self.exchange_client:
                return HealthCheckResult(
                    status=HealthStatus.UNKNOWN,
                    message="Exchange client not configured"
                )
            
            # Test exchange connectivity
            await self.exchange_client.get_server_time()
            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                message="Exchange is healthy"
            )
        except Exception as e:
            self.logger.error(f"Exchange health check failed: {e}")
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                message=f"Exchange health check failed: {str(e)}"
            )


class SystemHealthMonitor(BaseHealthMonitor):
    """Monitor for overall system health."""
    
    def __init__(self, monitors: Optional[list[BaseHealthMonitor]] = None):
        super().__init__("system")
        self.monitors = monitors or []
    
    async def check_health(self) -> HealthCheckResult:
        """Check overall system health."""
        if not self.monitors:
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                message="No monitors configured"
            )
        
        results = []
        for monitor in self.monitors:
            try:
                result = await monitor.check_health()
                results.append(result)
            except Exception as e:
                self.logger.error(f"Monitor {monitor.name} failed: {e}")
                results.append(HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Monitor {monitor.name} failed: {str(e)}"
                ))
        
        # Determine overall status
        if all(r.status == HealthStatus.HEALTHY for r in results):
            status = HealthStatus.HEALTHY
            message = "All systems healthy"
        elif any(r.status == HealthStatus.UNHEALTHY for r in results):
            status = HealthStatus.UNHEALTHY
            message = "Some systems unhealthy"
        else:
            status = HealthStatus.DEGRADED
            message = "Some systems degraded"
        
        return HealthCheckResult(
            status=status,
            message=message,
            details={"monitors": [r.__dict__ for r in results]}
        ) 