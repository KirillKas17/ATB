import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from exchange.market_data import MarketData
from utils.indicators import (
    calculate_atr,
    calculate_fractals,
    calculate_imbalance,
    calculate_liquidity_zones,
    calculate_market_structure,
    calculate_volume_profile,
)
from utils.logger import setup_logger

logger = setup_logger(__name__)


class SignalType(Enum):
    SPREAD = "spread"
    LIQUIDITY = "liquidity"
    FAKEOUT = "fakeout"


class LiquidityZoneType(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"


@dataclass
class MarketMakerSignal:
    """Класс для хранения сигналов маркет-мейкера"""

    timestamp: pd.Timestamp
    pair: str
    signal_type: SignalType
    confidence: float
    details: Dict[str, Any]
    priority: int = 0


class IDataProvider(ABC):
    @abstractmethod
    async def get_market_data(self, pair: str) -> pd.DataFrame:
        pass

    @abstractmethod
    async def get_order_book(self, pair: str) -> Dict[str, Any]:
        pass


class DefaultDataProvider(IDataProvider):
    def __init__(self):
        self.market_data = None  # Инициализируем как None

    async def get_market_data(self, pair: str, interval: str = "1m") -> pd.DataFrame:
        if self.market_data is None or self.market_data.symbol != pair or self.market_data.interval != interval:
            self.market_data = MarketData(symbol=pair, interval=interval)
            await self.market_data.start()
        return self.market_data.df

    async def get_order_book(self, pair: str, depth: int = 20) -> Dict[str, Any]:
        if self.market_data is None or self.market_data.symbol != pair:
            self.market_data = MarketData(symbol=pair, interval="1m")
            await self.market_data.start()
        return {
            "bids": self.market_data.df["close"].tail(depth).tolist(),
            "asks": self.market_data.df["close"].tail(depth).tolist(),
            "bids_volume": self.market_data.df["volume"].tail(depth).tolist(),
            "asks_volume": self.market_data.df["volume"].tail(depth).tolist()
        }


class CacheService:
    def __init__(self):
        self.volume_profiles: Dict[str, pd.DataFrame] = {}
        self.fractal_levels: Dict[str, Dict[str, List[float]]] = {}
        self.liquidity_zones: Dict[str, List[Dict]] = {}

    def clear(self):
        self.volume_profiles.clear()
        self.fractal_levels.clear()
        self.liquidity_zones.clear()


class MarketMakerCalculationStrategy(ABC):
    @abstractmethod
    def calculate(self, *args, **kwargs) -> Any:
        pass


class SpreadCalculationStrategy(MarketMakerCalculationStrategy):
    def calculate(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        try:
            best_bid = order_book["bids"][0]["price"]
            best_ask = order_book["asks"][0]["price"]
            spread = (best_ask - best_bid) / best_bid
            bid_volume = sum(order["size"] for order in order_book["bids"])
            ask_volume = sum(order["size"] for order in order_book["asks"])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            return {
                "spread": spread,
                "imbalance": imbalance,
                "confidence": min(abs(imbalance), 1.0),
            }
        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return {"spread": 0.0, "imbalance": 0.0, "confidence": 0.0}


class MarketMakerModelAgent:
    """
    Агент моделирования маркет-мейкера: анализ спреда, ликвидности, фейкаутов, зон.
    TODO: Вынести работу с данными, расчёты и кэширование в отдельные классы/модули (SRP).
    """

    config: Dict[str, Any]
    data_provider: IDataProvider
    signals: Dict[str, List[MarketMakerSignal]]
    volume_profiles: Dict[str, pd.DataFrame]
    fractal_levels: Dict[str, Dict[str, List[float]]]
    liquidity_zones: Dict[str, List[Dict[str, Any]]]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Инициализация агента моделирования маркет-мейкера.
        :param config: словарь параметров
        """
        self.config = config or {
            "spread_threshold": 0.001,
            "volume_threshold": 100000,
            "fakeout_threshold": 0.02,
            "liquidity_zone_size": 0.005,
            "lookback_period": 100,
            "confidence_threshold": 0.7,
        }
        self.data_provider = DefaultDataProvider()
        self.signals: Dict[str, List[MarketMakerSignal]] = {}
        self.volume_profiles: Dict[str, pd.DataFrame] = {}
        self.fractal_levels: Dict[str, Dict[str, List[float]]] = {}
        self.liquidity_zones: Dict[str, List[Dict[str, Any]]] = {}

    async def predict_next_move(self, pair: str) -> Dict[str, float]:
        """
        Предсказание следующего движения цены.
        :param pair: тикер пары
        :return: словарь с прогнозом
        """
        try:
            market_data = await self.data_provider.get_market_data(pair)
            order_book = await self.data_provider.get_order_book(pair)
            spread = self._analyze_spread(order_book)
            liquidity = self._analyze_liquidity(market_data, order_book)
            fakeouts = await self.identify_fakeouts(pair)
            prediction = self._calculate_prediction(spread, liquidity, fakeouts)
            return prediction
        except Exception as e:
            logger.error(f"Error predicting next move for {pair}: {str(e)}")
            return {"direction": 0.0, "confidence": 0.0, "expected_move": 0.0}

    async def liquidity_sweep_zone(self, pair: str) -> List[Dict[str, Any]]:
        """Определение зон сбора ликвидности"""
        try:
            await self._update_volume_profile(pair)
            await self._update_fractals(pair)
            await self._update_liquidity_zones(pair)
            significant_zones = [
                zone
                for zone in self.liquidity_zones.get(pair, [])
                if zone.get("volume", 0) > self.config["volume_threshold"]
            ]
            return significant_zones
        except Exception as e:
            logger.error(f"Error finding liquidity sweep zones for {pair}: {str(e)}")
            return []

    async def identify_fakeouts(self, pair: str) -> List[Dict[str, Any]]:
        """
        Расширенное определение фейкаутов и манипуляций.
        :param pair: тикер пары
        :return: список обнаруженных фейкаутов
        """
        try:
            market_data = await self.data_provider.get_market_data(pair)
            order_book = await self.data_provider.get_order_book(pair)
            fakeouts: List[Dict[str, Any]] = []
            df = pd.DataFrame(market_data)
            df["price_change"] = df["close"].pct_change()
            df["volume_change"] = df["volume"].pct_change()
            df["atr"] = calculate_atr(df["high"], df["low"], df["close"], 14)
            df["volatility"] = df["close"].pct_change().rolling(20).std()
            volume_profile = calculate_volume_profile(df)
            liquidity_zones = calculate_liquidity_zones(df)
            
            for i in range(1, len(df)):
                price_spike = abs(df["price_change"].iloc[i]) > df["volatility"].iloc[i] * 2
                volume_spike = df["volume_change"].iloc[i] > 2.0
                price_return = abs(df["close"].iloc[i] - df["close"].iloc[i - 1]) < df["atr"].iloc[i] * 0.5
                liquidity_drop = (
                    order_book["bids_volume"][i] + order_book["asks_volume"][i]
                ) < (
                    order_book["bids_volume"][i - 1] + order_book["asks_volume"][i - 1]
                ) * 0.7
                
                if price_spike and volume_spike and price_return and liquidity_drop:
                    fakeout = {
                        "timestamp": df.index[i],
                        "type": "fakeout",
                        "price_level": float(df["close"].iloc[i]),
                        "volume": float(df["volume"].iloc[i]),
                        "confidence": 0.8,
                        "details": {
                            "price_spike": bool(price_spike),
                            "volume_spike": bool(volume_spike),
                            "price_return": bool(price_return),
                            "liquidity_drop": bool(liquidity_drop)
                        }
                    }
                    fakeouts.append(fakeout)
            return fakeouts
        except Exception as e:
            logger.error(f"Error identifying fakeouts for {pair}: {str(e)}")
            return []

    def _analyze_spread(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Анализ спреда в ордербуке"""
        try:
            best_bid = float(order_book["bids"][0])
            best_ask = float(order_book["asks"][0])
            spread = (best_ask - best_bid) / best_bid

            bid_volume = sum(float(order) for order in order_book["bids_volume"])
            ask_volume = sum(float(order) for order in order_book["asks_volume"])
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

            return {
                "spread": float(spread),
                "imbalance": float(imbalance),
                "confidence": float(min(abs(imbalance), 1.0))
            }
        except Exception as e:
            logger.error(f"Error analyzing spread: {str(e)}")
            return {"spread": 0.0, "imbalance": 0.0, "confidence": 0.0}

    def _analyze_liquidity(self, market_data: pd.DataFrame, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Анализ ликвидности"""
        try:
            df = pd.DataFrame(market_data)
            volume = float(df["volume"].iloc[-1])
            price = float(df["close"].iloc[-1])
            
            bid_volume = sum(float(order) for order in order_book["bids_volume"])
            ask_volume = sum(float(order) for order in order_book["asks_volume"])
            
            liquidity_ratio = min(bid_volume, ask_volume) / max(bid_volume, ask_volume)
            volume_ratio = volume / (bid_volume + ask_volume)
            
            return {
                "liquidity_ratio": float(liquidity_ratio),
                "volume_ratio": float(volume_ratio),
                "confidence": float(min(liquidity_ratio * volume_ratio, 1.0))
            }
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {str(e)}")
            return {"liquidity_ratio": 0.0, "volume_ratio": 0.0, "confidence": 0.0}

    def _calculate_prediction(
        self, 
        spread: Dict[str, float], 
        liquidity: Dict[str, float], 
        fakeouts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Расчет прогноза"""
        try:
            # Базовые факторы
            spread_factor = spread["confidence"] * (1 if spread["imbalance"] > 0 else -1)
            liquidity_factor = liquidity["confidence"] * (1 if liquidity["liquidity_ratio"] > 0.5 else -1)
            
            # Фактор фейкаутов
            fakeout_factor = 0.0
            if fakeouts:
                latest_fakeout = fakeouts[-1]
                fakeout_factor = latest_fakeout["confidence"] * (1 if latest_fakeout["type"] == "bullish_fakeout" else -1)
            
            # Итоговый прогноз
            direction = (spread_factor + liquidity_factor + fakeout_factor) / 3
            confidence = abs(direction)
            
            return {
                "direction": float(direction),
                "confidence": float(confidence),
                "expected_move": float(direction * spread["spread"] * 100)
            }
        except Exception as e:
            logger.error(f"Error calculating prediction: {str(e)}")
            return {"direction": 0.0, "confidence": 0.0, "expected_move": 0.0}

    async def _update_volume_profile(self, pair: str):
        """Обновление профиля объема"""
        try:
            market_data = await self.data_provider.get_market_data(pair)
            self.volume_profiles[pair] = pd.DataFrame(calculate_volume_profile(market_data))
        except Exception as e:
            logger.error(f"Error updating volume profile for {pair}: {str(e)}")

    async def _update_fractals(self, pair: str):
        """Обновление фрактальных уровней"""
        try:
            market_data = await self.data_provider.get_market_data(pair)
            if market_data.empty:
                return
            fractals = calculate_fractals(market_data["high"], market_data["low"])
            self.fractal_levels[pair] = fractals
        except Exception as e:
            logger.error(f"Error updating fractals for {pair}: {str(e)}")

    async def _update_liquidity_zones(self, pair: str):
        """Обновление зон ликвидности"""
        try:
            market_data = await self.data_provider.get_market_data(pair)
            order_book = await self.data_provider.get_order_book(pair)
            zones = calculate_liquidity_zones(market_data)
            self.liquidity_zones[pair] = [
                {"type": k, "levels": v} for k, v in zones.items()
            ]
        except Exception as e:
            logger.error(f"Error updating liquidity zones for {pair}: {str(e)}")

    def _check_fakeout_patterns(self, df: pd.DataFrame) -> bool:
        """Проверка паттернов фейкаута"""
        if len(df) < 5:
            return False
        
        try:
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            price_change = abs(last_candle["close"] - prev_candle["close"]) / prev_candle["close"]
            volume_change = last_candle["volume"] / prev_candle["volume"]
            
            return bool(price_change > 0.02 and volume_change > 2.0)
        except Exception as e:
            logger.error(f"Error checking fakeout patterns: {str(e)}")
            return False

    def _check_level_break(self, price: float, levels: Dict[str, List[float]]) -> bool:
        """Проверка пробоя уровня"""
        try:
            for level_type, level_list in levels.items():
                for level in level_list:
                    if abs(price - level) / level < 0.001:  # 0.1% от уровня
                        return True
            return False
        except Exception as e:
            logger.error(f"Error checking level break: {str(e)}")
            return False

    def _determine_fakeout_type(self, df: pd.DataFrame, volume_profile: Dict[str, Any], liquidity_zones: Dict[str, Any]) -> str:
        """Определение типа фейкаута"""
        try:
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            if last_candle["close"] > prev_candle["close"]:
                return "bullish_fakeout"
            else:
                return "bearish_fakeout"
        except Exception as e:
            logger.error(f"Error determining fakeout type: {str(e)}")
            return "unknown"

    def _calculate_fakeout_confidence(self, df: pd.DataFrame, volume_profile: Dict[str, Any], liquidity_zones: Dict[str, Any]) -> float:
        """Расчет уверенности в фейкауте"""
        try:
            last_candle = df.iloc[-1]
            prev_candle = df.iloc[-2]
            
            price_change = abs(last_candle["close"] - prev_candle["close"]) / prev_candle["close"]
            volume_change = last_candle["volume"] / prev_candle["volume"]
            
            confidence = min(1.0, (price_change * 10 + volume_change) / 2)
            return float(confidence)
        except Exception as e:
            logger.error(f"Error calculating fakeout confidence: {str(e)}")
            return 0.0

    def _calculate_level_strength(
        self, levels: Dict[str, List[float]], order_book: pd.DataFrame, imbalance: float
    ) -> Dict[str, float]:
        """Расчет силы уровней."""
        strength = {}
        for level_type, level_list in levels.items():
            for level in level_list:
                key = f"{level_type}_{level}"
                strength[key] = float(imbalance)  # Преобразуем в float
        return strength

    def _merge_liquidity_zones(self, levels: Dict[str, List[float]], strength: Dict[str, float]) -> List[Dict[str, Any]]:
        """Объединение зон ликвидности."""
        merged_zones = []
        for level_type, level_list in levels.items():
            for level in level_list:
                key = f"{level_type}_{level}"
                zone = {
                    "type": level_type,
                    "price": float(level),
                    "strength": float(strength.get(key, 0.0))
                }
                merged_zones.append(zone)
        return merged_zones
