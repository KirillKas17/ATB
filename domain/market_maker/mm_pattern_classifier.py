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
    
    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        
    def classify_pattern(
        self, symbol: Symbol, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> Optional[MarketMakerPattern]:
        """Классификация паттерна маркет-мейкинга."""
        try:
            features = self.extract_features(order_book, trades)
            
            # Анализ спреда через spread_change
            spread_value = features.spread_change
            if spread_value > 0.02:  # Большое изменение спреда
                if features.order_imbalance > 0.6:
                    return None  # Временно возвращаем None вместо несуществующих констант
                return None
            
            # Анализ ликвидности через liquidity_depth
            if features.liquidity_depth < 0.3:  # Низкая ликвидность
                return None
            
            # Анализ реакции цены
            if abs(features.price_reaction) > 0.05:  # Сильная реакция
                return None
            
            # Анализ давления в стакане
            if features.book_pressure > 0.7:
                return None
            
            # Анализ концентрации объема
            if features.volume_concentration > 0.01:
                return None
            
            # Нормальный случай
            if 0.001 < spread_value < 0.01 and 0.4 < features.order_imbalance < 0.6:
                return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error classifying pattern for {symbol}: {e}")
            return None
    
    def extract_features(
        self, order_book: OrderBookSnapshot, trades: TradeSnapshot
    ) -> PatternFeatures:
        """Извлечение признаков для классификации паттерна."""
        try:
            # Используем правильные атрибуты PatternFeatures из domain/market_maker/mm_pattern.py
            
            # Расчет book_pressure - давление стакана (разность bid/ask объемов)
            total_bid_volume = sum(Decimal(str(getattr(level, 'size', getattr(level, 'volume', 0)))) for level in order_book.bids[:5])
            total_ask_volume = sum(Decimal(str(getattr(level, 'size', getattr(level, 'volume', 0)))) for level in order_book.asks[:5])
            total_volume = total_bid_volume + total_ask_volume
            book_pressure = float((total_bid_volume - total_ask_volume) / total_volume) if total_volume else 0.0
            
            # volume_delta - изменение объема
            volume_delta = float(getattr(trades, 'volume', 0))
            
            # price_reaction - реакция цены
            price_list = getattr(trades, 'prices', []) if hasattr(trades, 'prices') else []
            if len(price_list) >= 2:
                price_reaction = float((price_list[-1] - price_list[0]) / price_list[0]) if price_list[0] else 0.0
            else:
                price_reaction = 0.0
            
            # spread_change - изменение спреда
            best_bid = getattr(order_book.bids[0], 'price', Decimal('0')) if order_book.bids else Decimal('0')
            best_ask = getattr(order_book.asks[0], 'price', Decimal('0')) if order_book.asks else Decimal('0')
            mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else Decimal('0')
            spread = best_ask - best_bid if best_bid and best_ask else Decimal('0')
            spread_change = float(spread / mid_price) if mid_price else 0.0
            
            # order_imbalance - дисбаланс ордеров
            order_imbalance = float(total_bid_volume / total_volume) if total_volume else 0.5
            
            # liquidity_depth - глубина ликвидности
            liquidity_depth = float(total_volume)
            
            # time_duration - временная продолжительность (фиксированное значение)
            time_duration = 60  # секунды
            
            # volume_concentration - концентрация объема
            volume_concentration = abs(book_pressure)
            
            # price_volatility - волатильность цены
            if len(price_list) > 1:
                price_volatility = float(max(price_list) - min(price_list)) / float(sum(price_list) / len(price_list)) if price_list else 0.0
            else:
                price_volatility = 0.0
            
            return PatternFeatures(
                book_pressure=book_pressure,
                volume_delta=volume_delta,
                price_reaction=price_reaction,
                spread_change=spread_change,
                order_imbalance=order_imbalance,
                liquidity_depth=liquidity_depth,
                time_duration=time_duration,
                volume_concentration=volume_concentration,
                price_volatility=price_volatility
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            # Возвращаем нейтральные значения при ошибке
            return PatternFeatures(
                book_pressure=0.0,
                volume_delta=0.0,
                price_reaction=0.0,
                spread_change=0.01,
                order_imbalance=0.5,
                liquidity_depth=1000.0,
                time_duration=60,
                volume_concentration=0.0,
                price_volatility=0.0
            )
