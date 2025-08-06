"""
Bybit Client mock.
"""

from typing import Dict, Any, Optional, List
import asyncio


class BybitClient:
    """Mock Bybit Client."""
    
    def __init__(self, api_key: str = "", secret: str = ""):
        self.api_key = api_key
        self.secret = secret
    
    async def get_balance(self) -> Dict[str, Any]:
        """Mock balance."""
        return {"BTC": 1.5, "USDT": 50000.0}
    
    async def place_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock place order."""
        return {"orderId": "mock_123", "status": "NEW"}
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Mock ticker."""
        return {"symbol": symbol, "price": "45000.0", "volume": "1000.0"}
