from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from .market_types import MarketMetadataDict, MarketRegime, Timeframe


@runtime_checkable
class MarketProtocol(Protocol):
    id: str
    symbol: str
    name: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    metadata: MarketMetadataDict

    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketProtocol": ...


@runtime_checkable
class MarketDataProtocol(Protocol):
    id: str
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float]
    trades_count: Optional[int]
    taker_buy_volume: Optional[float]
    taker_buy_quote_volume: Optional[float]
    metadata: MarketMetadataDict

    @property
    def open_price(self) -> float: ...
    @property
    def high_price(self) -> float: ...
    @property
    def low_price(self) -> float: ...
    @property
    def close_price(self) -> float: ...
    def get_price_range(self) -> float: ...
    def get_body_size(self) -> float: ...
    def get_upper_shadow(self) -> float: ...
    def get_lower_shadow(self) -> float: ...
    def is_bullish(self) -> bool: ...
    def is_bearish(self) -> bool: ...
    def is_doji(self) -> bool: ...
    def get_volume_price_trend(self) -> Optional[float]: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketDataProtocol": ...
    @classmethod
    def from_dataframe(
        cls, df: Any, symbol: str, timeframe: Timeframe
    ) -> List["MarketDataProtocol"]: ...


@runtime_checkable
class MarketStateProtocol(Protocol):
    id: str
    symbol: str
    timestamp: datetime
    regime: MarketRegime
    volatility: float
    trend_strength: float
    volume_trend: float
    price_momentum: float
    support_level: Optional[float]
    resistance_level: Optional[float]
    pivot_point: Optional[float]
    rsi: Optional[float]
    macd: Optional[float]
    bollinger_upper: Optional[float]
    bollinger_lower: Optional[float]
    bollinger_middle: Optional[float]
    atr: Optional[float]
    metadata: MarketMetadataDict

    def is_trending(self) -> bool: ...
    def is_sideways(self) -> bool: ...
    def is_volatile(self) -> bool: ...
    def is_breakout(self) -> bool: ...
    def get_trend_direction(self) -> Optional[str]: ...
    def get_price_position(self, current_price: float) -> Optional[str]: ...
    def is_overbought(self) -> bool: ...
    def is_oversold(self) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketStateProtocol": ...
