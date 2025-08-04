from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, cast
from uuid import uuid4

from .market_protocols import MarketProtocol
from .market_types import MarketMetadataDict


@dataclass
class Market(MarketProtocol):
    id: str = field(default_factory=lambda: str(uuid4()))
    symbol: str = ""
    name: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: MarketMetadataDict = field(default_factory=lambda: cast(MarketMetadataDict, {"source": "", "exchange": "", "extra": {}}))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "name": self.name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Market":
        return cls(
            id=data.get("id", str(uuid4())),
            symbol=data.get("symbol", ""),
            name=data.get("name", ""),
            is_active=data.get("is_active", True),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now()
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if "updated_at" in data
                else datetime.now()
            ),
            metadata=data.get("metadata", {"source": "", "exchange": "", "extra": {}}),
        )
