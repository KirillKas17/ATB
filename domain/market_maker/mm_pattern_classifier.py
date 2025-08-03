# -*- coding: utf-8 -*-
"""
Классификатор паттернов маркет-мейкера промышленного уровня.
"""
import logging
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime, timedelta
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

from domain.market_maker.mm_pattern import (
    MarketMakerPattern,
    MarketMakerPatternType,
    PatternFeatures,
)
from domain.types.market_maker_types import (
    BookPressure,
    Confidence,
    LiquidityDepth,
    MarketMakerPatternType,
    MarketMicrostructure,
    OrderBookLevel,
    OrderBookProtocol,
    OrderImbalance,
    PatternClassifierConfig,
    PatternConfidence,
    PatternContext,
    PatternFeaturesProtocol,
    PriceReaction,
    PriceVolatility,
    SpreadChange,
    Symbol,
    TimeDuration,
    TradeData,
    TradeSnapshotProtocol,
    VolumeConcentration,
    VolumeDelta,
)

logger = logging.getLogger(__name__)


class OrderBookSnapshot(OrderBookProtocol):
    def __init__(
        self,
        timestamp: datetime,
        symbol: Symbol,
        bids: List[OrderBookLevel],
        asks: List[OrderBookLevel],
        last_price: float,
        volume_24h: float,
        price_change_24h: float,
    ):
        self.timestamp = timestamp
        self.symbol = symbol
        self.bids = bids
        self.asks = asks
        self.last_price = last_price
        self.volume_24h = volume_24h
        self.price_change_24h = price_change_24h

    def get_bid_volume(self, levels: int = 5) -> float:
        return sum(bid["size"] for bid in self.bids[:levels])

    def get_ask_volume(self, levels: int = 5) -> float:
        return sum(ask["size"] for ask in self.asks[:levels])

    def get_mid_price(self) -> float:
        if not self.bids or not self.asks:
            return self.last_price
        return (self.bids[0]["price"] + self.asks[0]["price"]) / 2

    def get_spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return self.asks[0]["price"] - self.bids[0]["price"]

    def get_spread_percentage(self) -> float:
        mid_price = self.get_mid_price()
        if mid_price == 0:
            return 0.0
        return (self.get_spread() / mid_price) * 100

    def get_order_imbalance(self) -> OrderImbalance:
        bid_volume = self.get_bid_volume()
        ask_volume = self.get_ask_volume()
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return OrderImbalance(0.0)
        imbalance = (bid_volume - ask_volume) / total_volume
        return OrderImbalance(max(-1.0, min(1.0, imbalance)))

    def get_liquidity_depth(self) -> LiquidityDepth:
        total_volume = self.get_bid_volume(10) + self.get_ask_volume(10)
        return LiquidityDepth(total_volume)


class TradeSnapshot(TradeSnapshotProtocol):
    def __init__(self, timestamp: datetime, symbol: Symbol, trades: List[TradeData]):
        self.timestamp = timestamp
        self.symbol = symbol
        self.trades = trades

    def get_total_volume(self) -> float:
        return sum(trade["size"] for trade in self.trades)

    def get_buy_volume(self) -> float:
        return sum(trade["size"] for trade in self.trades if trade["side"] == "buy")

    def get_sell_volume(self) -> float:
        return sum(trade["size"] for trade in self.trades if trade["side"] == "sell")

    def get_volume_delta(self, window: int = 10) -> VolumeDelta:
        if len(self.trades) < window * 2:
            return VolumeDelta(0.0)
        recent_volume = sum(trade["size"] for trade in self.trades[-window:])
        older_volume = sum(
            trade["size"] for trade in self.trades[-window * 2 : -window]
        )
        if older_volume == 0:
            return VolumeDelta(0.0)
        delta = (recent_volume - older_volume) / older_volume
        return VolumeDelta(delta)

    def get_price_reaction(self) -> PriceReaction:
        if len(self.trades) < 2:
            return PriceReaction(0.0)
        first_price = self.trades[0]["price"]
        last_price = self.trades[-1]["price"]
        reaction = (last_price - first_price) / first_price
        return PriceReaction(reaction)

    def get_volume_concentration(self) -> VolumeConcentration:
        if not self.trades:
            return VolumeConcentration(0.0)
        volumes = [trade["size"] for trade in self.trades]
        total_volume = sum(volumes)
        if total_volume == 0:
            return VolumeConcentration(0.0)
        mean_volume = total_volume / len(volumes)
        variance = sum((v - mean_volume) ** 2 for v in volumes) / len(volumes)
        concentration = (variance**0.5) / mean_volume
        return VolumeConcentration(concentration)

    def get_price_volatility(self) -> PriceVolatility:
        if len(self.trades) < 2:
            return PriceVolatility(0.0)
        prices = [trade["price"] for trade in self.trades]
        returns = [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]
        if not returns:
            return PriceVolatility(0.0)
        volatility = (
            sum((r - mean(returns)) ** 2 for r in returns) / len(returns)
        ) ** 0.5
        return PriceVolatility(volatility)


class IPatternClassifier(ABC):
    @abstractmethod
    def classify_pattern(
        self, symbol: Symbol, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> Optional[MarketMakerPattern]:
        raise NotImplementedError("classify_pattern must be implemented in subclasses")

    @abstractmethod
    def extract_features(
        self, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> PatternFeatures:
        raise NotImplementedError("extract_features must be implemented in subclasses")


class MarketMakerPatternClassifier(IPatternClassifier):
    def __init__(self, config: Optional[PatternClassifierConfig] = None):
        self.config = config or PatternClassifierConfig()
        self.order_book_history: Dict[str, deque] = {}
        self.trade_history: Dict[str, deque] = {}
        self.max_history_size = self.config.max_history_size

    def classify_pattern(
        self, symbol: Symbol, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> Optional[MarketMakerPattern]:
        self._update_history(symbol, order_book, trades)
        features = self.extract_features(order_book, trades)
        pattern_type, confidence = self._determine_pattern_type(features)
        if confidence < self.config.min_confidence:
            return None
        return MarketMakerPattern(
            pattern_type=pattern_type,
            symbol=symbol,
            timestamp=order_book.timestamp,
            features=features,
            confidence=Confidence(confidence),
            context=self._build_context(symbol, order_book, trades),
        )

    def extract_features(
        self, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> PatternFeatures:
        order_imbalance = order_book.get_order_imbalance()
        return PatternFeatures(
            book_pressure=BookPressure(float(order_imbalance)),  # Преобразуем в float
            volume_delta=trades.get_volume_delta(),
            price_reaction=trades.get_price_reaction(),
            spread_change=SpreadChange(order_book.get_spread_percentage()),
            order_imbalance=order_imbalance,
            liquidity_depth=order_book.get_liquidity_depth(),
            time_duration=TimeDuration(0),
            volume_concentration=trades.get_volume_concentration(),
            price_volatility=trades.get_price_volatility(),
            market_microstructure={},
        )

    def _update_history(
        self, symbol: str, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> None:
        if symbol not in self.order_book_history:
            self.order_book_history[symbol] = deque(maxlen=self.max_history_size)
        self.order_book_history[symbol].append(order_book)
        if symbol not in self.trade_history:
            self.trade_history[symbol] = deque(maxlen=self.max_history_size)
        self.trade_history[symbol].append(trades)

    def _determine_pattern_type(
        self, features: PatternFeatures
    ) -> Tuple[MarketMakerPatternType, float]:
        if features.book_pressure > 0.3 and features.volume_delta > 0.2:
            return MarketMakerPatternType.ACCUMULATION, 0.8
        if features.book_pressure < -0.3 and features.volume_delta > 0.2:
            return MarketMakerPatternType.EXIT, 0.8
        return MarketMakerPatternType.ABSORPTION, 0.5

    def _build_context(
        self, symbol: Symbol, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> PatternContext:
        return {
            "symbol": symbol,
            "timestamp": order_book.timestamp.isoformat(),
            "last_price": order_book.last_price,
            "volume_24h": order_book.volume_24h,
            "price_change_24h": order_book.price_change_24h,
        }
