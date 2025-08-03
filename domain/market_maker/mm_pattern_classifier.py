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
from decimal import Decimal

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
    """Классификатор паттернов маркет-мейкинга."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def classify_pattern(
        self, symbol: Symbol, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> Optional[MarketMakerPattern]:
        """Классификация паттерна маркет-мейкинга."""
        try:
            features = self.extract_features(order_book, trades)
            
            # Анализ спреда
            spread_ratio = features.spread_ratio
            if spread_ratio > 0.02:  # Широкий спред
                if features.volume_imbalance > 0.6:
                    return MarketMakerPattern.WIDE_SPREAD_IMBALANCE
                return MarketMakerPattern.WIDE_SPREAD
            
            # Анализ ликвидности
            if features.liquidity_ratio < 0.3:  # Низкая ликвидность
                return MarketMakerPattern.LOW_LIQUIDITY
            
            # Анализ импульса
            if abs(features.price_momentum) > 0.05:  # Сильный импульс
                return MarketMakerPattern.MOMENTUM_PLAY
            
            # Анализ манипуляций
            if features.manipulation_score > 0.7:
                return MarketMakerPattern.MANIPULATION
            
            # Анализ арбитража
            if features.arbitrage_opportunity > 0.01:
                return MarketMakerPattern.ARBITRAGE
            
            # Нормальный маркет-мейкинг
            if 0.001 < spread_ratio < 0.01 and 0.4 < features.volume_imbalance < 0.6:
                return MarketMakerPattern.NORMAL_MARKET_MAKING
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error classifying pattern for {symbol}: {e}")
            return None
    
    def extract_features(
        self, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> PatternFeatures:
        """Извлечение признаков для классификации паттерна."""
        try:
            # Расчет спреда
            best_bid = order_book.bids[0].price if order_book.bids else Decimal('0')
            best_ask = order_book.asks[0].price if order_book.asks else Decimal('0')
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else Decimal('0')
            spread = best_ask - best_bid if best_bid and best_ask else Decimal('0')
            spread_ratio = float(spread / mid_price) if mid_price else 0.0
            
            # Расчет дисбаланса объемов
            total_bid_volume = sum(level.volume for level in order_book.bids[:5])
            total_ask_volume = sum(level.volume for level in order_book.asks[:5])
            total_volume = total_bid_volume + total_ask_volume
            volume_imbalance = float(total_bid_volume / total_volume) if total_volume else 0.5
            
            # Расчет ликвидности (объем в топ-5 уровнях относительно среднего объема)
            avg_trade_volume = trades.volume / max(len(trades.prices), 1) if trades.volume else Decimal('0')
            liquidity_ratio = float(total_volume / (avg_trade_volume * 10)) if avg_trade_volume else 0.0
            
            # Расчет momentum (изменение цены относительно предыдущих сделок)
            if len(trades.prices) >= 2:
                price_change = trades.prices[-1] - trades.prices[0]
                price_momentum = float(price_change / trades.prices[0]) if trades.prices[0] else 0.0
            else:
                price_momentum = 0.0
            
            # Простой скор манипуляций (на основе резких изменений)
            manipulation_score = min(abs(price_momentum) * 2 + max(0, spread_ratio - 0.01) * 5, 1.0)
            
            # Возможность арбитража (упрощенная оценка)
            arbitrage_opportunity = max(0, spread_ratio - 0.005)
            
            return PatternFeatures(
                spread_ratio=spread_ratio,
                volume_imbalance=volume_imbalance,
                liquidity_ratio=liquidity_ratio,
                price_momentum=price_momentum,
                manipulation_score=manipulation_score,
                arbitrage_opportunity=arbitrage_opportunity
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            # Возвращаем нейтральные значения при ошибке
            return PatternFeatures(
                spread_ratio=0.01,
                volume_imbalance=0.5,
                liquidity_ratio=0.5,
                price_momentum=0.0,
                manipulation_score=0.0,
                arbitrage_opportunity=0.0
            )
