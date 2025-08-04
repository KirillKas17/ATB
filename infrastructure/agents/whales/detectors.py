from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from shared.numpy_utils import np
import pandas as pd
import torch
import torch.nn as nn

from exchange.market_data import MarketData

from .types import WhaleActivity

logger = None  # Временно убираем setup_logger


class WhaleMLModel(nn.Module):
    """ML модель для детекции активности китов"""

    def __init__(self, input_dim: int = 15, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IDataProvider(ABC):
    @abstractmethod
    async def get_market_data(self, pair: str, interval: str = "1m") -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_order_book(self, pair: str) -> Dict[str, Any]:
        pass


class DefaultDataProvider(IDataProvider):
    def __init__(self, symbol: str = "BTC/USDT", interval: str = "1h") -> None:
        self.market_data = MarketData(symbol=symbol, interval=interval)
        self.whale_data: Dict[str, Any] = {}
        self.last_update = None

    async def get_market_data(self, pair: str, interval: str = "1m") -> pd.DataFrame:
        return self.market_data.df

    async def get_order_book(self, pair: str) -> Dict[str, Any]:
        try:
            if hasattr(self.market_data, "order_book") and self.market_data.order_book:
                return self.market_data.order_book
            return {
                "bids": [
                    {"price": 50000.0, "size": 1.5},
                    {"price": 49999.0, "size": 2.1},
                    {"price": 49998.0, "size": 0.8},
                ],
                "asks": [
                    {"price": 50001.0, "size": 1.2},
                    {"price": 50002.0, "size": 1.8},
                    {"price": 50003.0, "size": 0.9},
                ],
                "timestamp": datetime.now().isoformat(),
                "pair": pair,
            }
        except Exception as e:
            if logger:
                logger.error(f"Error getting order book for {pair}: {e}")
            return {"bids": [], "asks": [], "error": str(e)}


class WhaleActivityCache:
    def __init__(self) -> None:
        self.activities: Dict[str, List[WhaleActivity]] = {}

    def add(self, pair: str, activity: WhaleActivity) -> None:
        self.activities.setdefault(pair, []).append(activity)

    def get_recent(self, pair: str, lookback: int) -> List[WhaleActivity]:
        return self.activities.get(pair, [])[-lookback:]

    def clear(self) -> None:
        self.activities.clear()


class WhaleSignalAnalyzer:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def analyze_order_book(
        self, order_book: Dict[str, Any], config: Dict[str, Any]
    ) -> Optional[WhaleActivity]:
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            all_orders = bids + asks
            if not all_orders:
                return None
            prices = np.array([o["price"] for o in all_orders]).reshape(-1, 1)
            sizes = np.array([o["size"] for o in all_orders]).reshape(-1, 1)
            X = np.hstack([prices, sizes])
            from sklearn.cluster import DBSCAN
            from sklearn.preprocessing import StandardScaler

            X_scaled = StandardScaler().fit_transform(X)
            db = DBSCAN(eps=0.5, min_samples=2).fit(X_scaled)
            clusters = set(db.labels_)
            large_clusters = [
                label
                for label in clusters
                if label != -1 and np.sum(db.labels_ == label) > 1
            ]
            spoofing_detected = False
            iceberg_detected = False
            for label in large_clusters:
                cluster_orders = [
                    all_orders[i]
                    for i in range(len(all_orders))
                    if db.labels_[i] == label
                ]
                total_size = sum(o["size"] for o in cluster_orders)
                if total_size > config.get("min_whale_size", 100000):
                    spoofing_detected = True
                if any(
                    o["size"] > config.get("min_whale_size", 100000)
                    for o in cluster_orders
                ):
                    iceberg_detected = True
            if spoofing_detected or iceberg_detected:
                return WhaleActivity(
                    timestamp=pd.Timestamp(datetime.now()),
                    volume=float(np.sum(sizes)),
                    price=float(np.mean(prices)),
                    direction=(
                        "buy"
                        if np.sum([o["size"] for o in bids])
                        > np.sum([o["size"] for o in asks])
                        else "sell"
                    ),
                    confidence=0.9 if spoofing_detected else 0.7,
                    impact_score=1.0,
                    details={
                        "spoofing": spoofing_detected,
                        "iceberg": iceberg_detected,
                    },
                    activity_type="order_book",
                )
            return None
        except Exception as e:
            if logger:
                logger.error(f"Error analyzing order book: {e}")
            return None
