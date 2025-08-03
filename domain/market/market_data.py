from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from .market_protocols import MarketDataProtocol
from .market_types import MarketMetadataDict, Timeframe


@dataclass(frozen=True)
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OHLCV":
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
        )


def _default_metadata() -> MarketMetadataDict:
    return {"source": "", "exchange": "", "extra": {}}


@dataclass
class MarketData(MarketDataProtocol):
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    timeframe: Timeframe = Timeframe.MINUTE_1
    timestamp: datetime = field(default_factory=datetime.now)
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    quote_volume: Optional[float] = None
    trades_count: Optional[int] = None
    taker_buy_volume: Optional[float] = None
    taker_buy_quote_volume: Optional[float] = None
    metadata: MarketMetadataDict = field(default_factory=_default_metadata)

    @property
    def open_price(self) -> float:
        return self.open

    @property
    def high_price(self) -> float:
        return self.high

    @property
    def low_price(self) -> float:
        return self.low

    @property
    def close_price(self) -> float:
        return self.close

    def get_price_range(self) -> float:
        return self.high - self.low

    def get_body_size(self) -> float:
        return abs(self.close - self.open)

    def get_upper_shadow(self) -> float:
        return self.high - max(self.open, self.close)

    def get_lower_shadow(self) -> float:
        return min(self.open, self.close) - self.low

    def is_bullish(self) -> bool:
        return self.close > self.open

    def is_bearish(self) -> bool:
        return self.close < self.open

    def is_doji(self) -> bool:
        return abs(self.close - self.open) < 0.1 * (self.high - self.low)

    def get_volume_price_trend(self) -> Optional[float]:
        if self.volume == 0:
            return None
        return (self.close - self.open) / self.volume

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "trades_count": self.trades_count,
            "taker_buy_volume": self.taker_buy_volume,
            "taker_buy_quote_volume": self.taker_buy_quote_volume,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketData":
        return cls(
            id=data.get("id", str(uuid4())),
            symbol=data.get("symbol", ""),
            timeframe=Timeframe(data["timeframe"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"]),
            quote_volume=(
                float(data["quote_volume"])
                if data.get("quote_volume") is not None
                else None
            ),
            trades_count=(
                int(data["trades_count"])
                if data.get("trades_count") is not None
                else None
            ),
            taker_buy_volume=(
                float(data["taker_buy_volume"])
                if data.get("taker_buy_volume") is not None
                else None
            ),
            taker_buy_quote_volume=(
                float(data["taker_buy_quote_volume"])
                if data.get("taker_buy_quote_volume") is not None
                else None
            ),
            metadata=data.get("metadata", {"source": "", "exchange": "", "extra": {}}),
        )

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, symbol: str, timeframe: Timeframe
    ) -> List[MarketDataProtocol]:
        result: List[MarketDataProtocol] = []
        for _, row in df.iterrows():
            result.append(
                cls(
                    id=str(uuid4()),
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=(
                        row["timestamp"]
                        if isinstance(row["timestamp"], datetime)
                        else datetime.fromisoformat(str(row["timestamp"]))
                    ),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    quote_volume=(
                        float(row["quote_volume"])
                        if "quote_volume" in row and row["quote_volume"] is not None
                        else None
                    ),
                    trades_count=(
                        int(row["trades_count"])
                        if "trades_count" in row and row["trades_count"] is not None
                        else None
                    ),
                    taker_buy_volume=(
                        float(row["taker_buy_volume"])
                        if "taker_buy_volume" in row
                        and row["taker_buy_volume"] is not None
                        else None
                    ),
                    taker_buy_quote_volume=(
                        float(row["taker_buy_quote_volume"])
                        if "taker_buy_quote_volume" in row
                        and row["taker_buy_quote_volume"] is not None
                        else None
                    ),
                    metadata={"source": "", "exchange": "", "extra": {}},
                )
            )
        return result
