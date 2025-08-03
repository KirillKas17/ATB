# -*- coding: utf-8 -*-
"""
Протоколы для сервисов.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol

from domain.entities.market import MarketData, OrderBook


class MarketDataServiceProtocol(Protocol):
    """Protocol for market data service."""

    async def get_market_data(self, symbol: str, timeframe: str = "1m") -> MarketData:
        """Get market data for symbol."""
        ...

    async def get_orderbook(self, symbol: str) -> OrderBook:
        """Get orderbook for symbol."""
        ...

    async def get_historical_data(
        self, symbol: str, start_time: str, end_time: str
    ) -> List[MarketData]:
        """Get historical market data."""
        ...


class TradingServiceProtocol(Protocol):
    """Protocol for trading service."""

    async def place_order(
        self, symbol: str, side: str, quantity: Decimal, price: Optional[Decimal] = None
    ) -> Dict[str, Any]:
        """Place a new order."""
        ...

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        ...

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status."""
        ...


class RiskServiceProtocol(Protocol):
    """Protocol for risk management service."""

    async def calculate_position_risk(
        self, symbol: str, quantity: Decimal, price: Decimal
    ) -> Dict[str, Any]:
        """Calculate position risk."""
        ...

    async def check_risk_limits(self, order_data: Dict[str, Any]) -> bool:
        """Check if order meets risk limits."""
        ...

    async def get_portfolio_risk(self) -> Dict[str, Any]:
        """Get portfolio risk metrics."""
        ...


class StrategyServiceProtocol(Protocol):
    """Protocol for strategy service."""

    async def execute_strategy(
        self, strategy_name: str, market_data: MarketData
    ) -> Dict[str, Any]:
        """Execute a trading strategy."""
        ...

    async def backtest_strategy(
        self, strategy_name: str, historical_data: List[MarketData]
    ) -> Dict[str, Any]:
        """Backtest a trading strategy."""
        ...

    async def optimize_strategy(
        self, strategy_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        ...


class MLServiceProtocol(Protocol):
    """Protocol for machine learning service."""

    async def predict_price(
        self, symbol: str, market_data: MarketData
    ) -> Dict[str, Any]:
        """Predict price movement."""
        ...

    async def predict_volatility(
        self, symbol: str, market_data: MarketData
    ) -> Dict[str, Any]:
        """Predict volatility."""
        ...

    async def classify_market_regime(self, market_data: MarketData) -> str:
        """Classify market regime."""
        ...


class CacheServiceProtocol(Protocol):
    """Protocol for cache service."""

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        ...


class NotificationServiceProtocol(Protocol):
    """Protocol for notification service."""

    async def send_notification(self, message: str, level: str = "info") -> bool:
        """Send notification."""
        ...

    async def send_alert(self, alert_type: str, data: Dict[str, Any]) -> bool:
        """Send alert."""
        ...


class LoggingServiceProtocol(Protocol):
    """Protocol for logging service."""

    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        ...

    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        ...

    def log_error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        ...

    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        ...


class MetricsServiceProtocol(Protocol):
    """Protocol for metrics service."""

    def record_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric."""
        ...

    def increment_counter(
        self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Increment a counter."""
        ...

    def record_histogram(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a histogram value."""
        ...


class ConfigurationServiceProtocol(Protocol):
    """Protocol for configuration service."""

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        ...

    def set_config(self, key: str, value: Any) -> None:
        """Set configuration value."""
        ...

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get configuration section."""
        ...


class DatabaseServiceProtocol(Protocol):
    """Protocol for database service."""

    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a database query."""
        ...

    async def execute_transaction(
        self, queries: List[str], params: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """Execute a database transaction."""
        ...

    async def close(self) -> None:
        """Close database connection."""
        ...
