# -*- coding: utf-8 -*-
"""Exchange WebSocket streams package."""
try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

__all__ = [
    "BingXWebSocketClient",
    "BitgetWebSocketClient",
    "BybitWebSocketClient",
    "MarketStreamAggregator",
]
