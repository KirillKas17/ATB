"""
Health check endpoints for web frameworks.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from .checker import HealthChecker, HealthStatus


class HealthCheckEndpoint:
    """Health check endpoint for web frameworks."""

    def __init__(self, health_checker: Optional[HealthChecker] = None) -> None:
        """
        Initialize health check endpoint.

        Args:
            health_checker: Health checker instance
        """
        self.health_checker = health_checker or HealthChecker()

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status for HTTP endpoint.

        Returns:
            Health status dictionary
        """
        try:
            # Run health checks
            await self.health_checker.run_all_checks()

            # Get overall status
            overall_status = self.health_checker.get_overall_status()

            # Get health report
            health_report = self.health_checker.get_health_report()

            return {
                "status": overall_status.value,
                "timestamp": datetime.now().isoformat(),
                "checks": health_report["checks"],
                "summary": health_report["summary"],
            }

        except Exception as e:
            return {
                "status": "error",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def get_ready_status(self) -> Dict[str, Any]:
        """
        Get readiness status for Kubernetes.

        Returns:
            Readiness status dictionary
        """
        try:
            return self.health_checker.get_ready_status()
        except Exception as e:
            return {
                "status": "not_ready",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    async def get_live_status(self) -> Dict[str, Any]:
        """
        Get liveness status for Kubernetes.

        Returns:
            Liveness status dictionary
        """
        try:
            return self.health_checker.get_live_status()
        except Exception as e:
            return {
                "status": "not_alive",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }
