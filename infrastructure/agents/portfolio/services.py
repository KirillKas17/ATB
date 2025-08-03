"""
!5@28AK ?>@BD5;O.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

# from shared.logging import setup_logger
# logger = setup_logger(__name__)
# Используем только loguru.logger
from domain.entities.portfolio import Portfolio


class PortfolioCacheService:
    """!5@28A :MH8@>20=8O 40==KE ?>@BD5;O."""

    def __init__(self, cache_ttl: int = 300):
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Any]:
        """>;CG5=85 40==KE 87 :MH0."""
        try:
            if key not in self.cache:
                return None
            timestamp = self.cache_timestamps.get(key)
            if not timestamp:
                return None
            if datetime.now() - timestamp > timedelta(seconds=self.cache_ttl):
                del self.cache[key]
                del self.cache_timestamps[key]
                return None
            return self.cache[key]
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    def set(self, key: str, value: Any) -> None:
        """!>E@0=5=85 40==KE 2 :MH."""
        try:
            self.cache[key] = value
            self.cache_timestamps[key] = datetime.now()
        except Exception as e:
            logger.error(f"Error setting cache: {e}")

    def clear(self) -> None:
        """G8A:0 :MH0."""
        self.cache.clear()
        self.cache_timestamps.clear()

    def cleanup_expired(self) -> None:
        """G8A:0 CA0@52H8E 70?8A59."""
        try:
            current_time = datetime.now()
            expired_keys = [
                key
                for key, timestamp in self.cache_timestamps.items()
                if current_time - timestamp > timedelta(seconds=self.cache_ttl)
            ]
            for key in expired_keys:
                del self.cache[key]
                del self.cache_timestamps[key]
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")


class PortfolioMetricsService:
    """!5@28A @0AG5B0 <5B@8: ?>@BD5;O."""

    def __init__(self) -> None:
        self.metrics_history: List[Dict[str, Any]] = []

    def calculate_sharpe_ratio(
        self, returns: List[float], risk_free_rate: float = 0.02
    ) -> float:
        """0AG5B :>MDD8F85=B0 (@?0."""
        try:
            if not returns:
                return 0.0
            avg_return = sum(returns) / len(returns)
            std_return = (
                sum((r - avg_return) ** 2 for r in returns) / len(returns)
            ) ** 0.5
            if std_return == 0:
                return 0.0
            sharpe_ratio = (avg_return - risk_free_rate) / std_return
            return float(sharpe_ratio)
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0

    def calculate_max_drawdown(self, values: List[float]) -> float:
        """0AG5B <0:A8<0;L=>9 ?@>A04:8."""
        try:
            if not values:
                return 0.0
            peak = values[0]
            max_dd = 0.0
            for value in values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                max_dd = max(max_dd, dd)
            return max_dd
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0

    def calculate_volatility(self, returns: List[float]) -> float:
        """0AG5B 2>;0B8;L=>A8."""
        try:
            if not returns:
                return 0.0
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            volatility = variance**0.5
            return float(volatility)
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0

    def calculate_correlation(
        self, returns1: List[float], returns2: List[float]
    ) -> float:
        """0AG5B :>@@5;OF88 <564C 4C<O 0:B802<8."""
        try:
            if len(returns1) != len(returns2) or not returns1:
                return 0.0
            avg1 = sum(returns1) / len(returns1)
            avg2 = sum(returns2) / len(returns2)
            numerator = sum(
                (r1 - avg1) * (r2 - avg2) for r1, r2 in zip(returns1, returns2)
            )
            denominator1 = sum((r1 - avg1) ** 2 for r1 in returns1) ** 0.5
            denominator2 = sum((r2 - avg2) ** 2 for r2 in returns2) ** 0.5
            if denominator1 == 0 or denominator2 == 0:
                return 0.0
            correlation = numerator / (denominator1 * denominator2)
            return float(correlation)
        except Exception as e:
            logger.error(f"Error calculating correlation: {e}")
            return 0.0

    def calculate_portfolio_metrics(
        self, portfolio_state: Portfolio
    ) -> Dict[str, float]:
        """0AG5B 2A5E <5B@8: ?>@BD5;O."""
        try:
            metrics: Dict[str, float] = {
                "total_value": float(portfolio_state.total_equity.amount),
                "total_pnl": float(portfolio_state.total_pnl.amount),
                "total_pnl_percent": 0.0,  # Нужно вычислить
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }
            # 0AG5B 4>?>;=8B5;L=KE <5B@8: =0 >A=>25 8AB>@88
            # Используем историю метрик из сервиса
            if self.metrics_history:
                returns = []
                values = []
                for record in self.metrics_history:
                    if "return" in record:
                        returns.append(record["return"])
                    if "value" in record:
                        values.append(record["value"])
                if returns:
                    metrics["volatility"] = self.calculate_volatility(returns)
                    metrics["sharpe_ratio"] = self.calculate_sharpe_ratio(returns)
                if values:
                    metrics["max_drawdown"] = self.calculate_max_drawdown(values)
            return metrics
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {
                "total_value": 0.0,
                "total_pnl": 0.0,
                "total_pnl_percent": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }

    def add_metrics_record(self, metrics: Dict[str, Any]) -> None:
        """>10B;5=85 70?8A8 <5B@8: 2 8AB>@8N."""
        try:
            record = {"timestamp": datetime.now().isoformat(), **metrics}
            self.metrics_history.append(record)
            # 3@0=8G82A5< @07<5@ 8AB>@88
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
        except Exception as e:
            logger.error(f"Error adding metrics record: {e}")

    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """>;CG5=85 8AB>@88 <5B@8:."""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                record
                for record in self.metrics_history
                if datetime.fromisoformat(record["timestamp"]) > cutoff_time
            ]
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []

    def get_portfolio_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """>;CG5=85 :8<@0@ ?>@BD5;O."""
        try:
            metrics = self.calculate_portfolio_metrics(portfolio)
            return {
                "portfolio_id": str(portfolio.id),
                "name": portfolio.name,
                "status": portfolio.status.value,
                "total_equity": float(portfolio.total_equity.amount),
                "free_margin": float(portfolio.free_margin.amount),
                "used_margin": float(portfolio.used_margin.amount),
                "margin_ratio": float(portfolio.get_margin_ratio()),
                "risk_profile": portfolio.risk_profile.value,
                "max_leverage": float(portfolio.max_leverage),
                "metrics": metrics,
                "last_updated": str(portfolio.updated_at),
            }
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}

    def calculate_portfolio_performance(
        self, portfolio: Portfolio, period_days: int = 30
    ) -> Dict[str, float]:
        """0AG5B ?@>A0@@0A:0 ?>@BD5;O 0A ?5@8>1."""
        try:
            # Используем историю метрик для расчета производительности
            if not self.metrics_history:
                return {
                    "total_return": 0.0,
                    "daily_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                }

            cutoff_time = datetime.now() - timedelta(days=period_days)
            recent_metrics = [
                record
                for record in self.metrics_history
                if datetime.fromisoformat(record["timestamp"]) > cutoff_time
            ]

            if not recent_metrics:
                return {
                    "total_return": 0.0,
                    "daily_return": 0.0,
                    "volatility": 0.0,
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 0.0,
                }

            returns = [record.get("return", 0.0) for record in recent_metrics]
            values = [record.get("value", 0.0) for record in recent_metrics]

            total_return = sum(returns) if returns else 0.0
            daily_return = total_return / len(returns) if returns else 0.0
            volatility = self.calculate_volatility(returns)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(values)

            return {
                "total_return": total_return,
                "daily_return": daily_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio performance: {e}")
            return {
                "total_return": 0.0,
                "daily_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
            }
