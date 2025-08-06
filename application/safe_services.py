"""
Minimal safe service implementations for startup testing
"""

from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
from datetime import datetime

from safe_import_wrapper import SafeImportMock


class SafeTradingService:
    """Minimal trading service that can be instantiated for testing"""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.active = True
        self.orders: List[Any] = []
        self.positions: List[Any] = []
        
    def start_trading_session(self) -> Dict[str, Any]:
        return {"status": "started", "timestamp": datetime.now()}
    
    def end_trading_session(self) -> Dict[str, Any]:
        return {"status": "ended", "timestamp": datetime.now()}
    
    def create_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"order_id": "test_order", "status": "created"}
    
    def execute_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "executed", "timestamp": datetime.now()}
    
    def cancel_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "cancelled", "timestamp": datetime.now()}
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        return []
    
    def get_order_history(self) -> List[Dict[str, Any]]:
        return []
    
    def create_position(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"position_id": "test_position", "status": "created"}
    
    def close_position(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "closed", "timestamp": datetime.now()}
    
    def get_active_positions(self) -> List[Dict]:
        return []
    
    def get_position_history(self) -> List[Dict]:
        return []
    
    def calculate_pnl(self, *args: Any, **kwargs: Any) -> Decimal:
        return Decimal("0.0")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        return {
            "total_value": Decimal("10000.0"),
            "available_balance": Decimal("10000.0"),
            "positions_count": 0,
            "orders_count": 0
        }
    
    def get_trading_statistics(self) -> Dict[str, Any]:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        return {
            "var": Decimal("0.0"),
            "expected_shortfall": Decimal("0.0"),
            "max_drawdown": Decimal("0.0"),
            "sharpe_ratio": Decimal("0.0")
        }
    
    # Fallback methods for any missing abstract methods
    def __getattr__(self, name: str) -> Any:
        def mock_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"method": name, "status": "mocked", "result": None}
        return mock_method


class SafeRiskService:
    """Minimal risk service that can be instantiated for testing"""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.active = True
        self.risk_limits = {
            "max_position_size": Decimal("1000.0"),
            "max_daily_loss": Decimal("100.0"),
            "max_total_exposure": Decimal("5000.0")
        }
    
    def calculate_position_risk(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "risk_score": 0.1,
            "position_var": Decimal("10.0"),
            "margin_requirement": Decimal("100.0")
        }
    
    def validate_order(self, *args: Any, **kwargs: Any) -> Dict[str, bool]:
        return {
            "is_valid": True,
            "within_limits": True,
            "risk_acceptable": True
        }
    
    def get_portfolio_risk(self) -> Dict[str, Any]:
        return {
            "total_var": Decimal("50.0"),
            "concentration_risk": 0.1,
            "correlation_risk": 0.05,
            "leverage_ratio": 1.0
        }
    
    def get_risk_alerts(self) -> List[Dict]:
        return []
    
    # Fallback methods for any missing abstract methods
    def __getattr__(self, name: str) -> Any:
        def mock_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"method": name, "status": "mocked", "result": None}
        return mock_method


class SafeMarketService:
    """Minimal market service for testing"""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.active = True
        
    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "price": Decimal("100.0"),
            "volume": 1000,
            "timestamp": datetime.now()
        }
    
    def get_order_book(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": symbol,
            "bids": [],
            "asks": [],
            "timestamp": datetime.now()
        }
    
    def __getattr__(self, name: str) -> Any:
        def mock_method(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"method": name, "status": "mocked", "result": None}
        return mock_method