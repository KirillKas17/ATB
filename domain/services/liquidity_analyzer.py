import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from shared.numpy_utils import np
import pandas as pd
from pandas import Interval

from domain.types.service_types import (
    LiquidityAnalysisResult, LiquiditySweep, LiquidityZone, LiquidityScore, LiquidityZoneType,
    TimestampValue, PriceValue, SweepType, ConfidenceLevel, VolumeValue
)
from decimal import Decimal
from datetime import datetime


class LiquidityZoneType:
    SUPPORT = "support"
    RESISTANCE = "resistance"
    NEUTRAL = "neutral"


class ILiquidityAnalyzer(ABC):
    @abstractmethod
    async def analyze_liquidity(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> LiquidityAnalysisResult:
        pass

    @abstractmethod
    async def identify_liquidity_zones(
        self, market_data: pd.DataFrame
    ) -> List[LiquidityZone]:
        pass

    @abstractmethod
    async def detect_liquidity_sweeps(
        self, market_data: pd.DataFrame
    ) -> List[LiquiditySweep]:
        pass


class LiquidityAnalyzer(ILiquidityAnalyzer):
    """Сервис для анализа ликвидности и зон поддержки/сопротивления"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {
            "liquidity_zone_size": 0.005,
            "volume_threshold": 100000,
            "sweep_threshold": 0.02,
            "lookback_period": 100,
        }
        self.logger = logging.getLogger(__name__)
        # Кэши для оптимизации производительности
        self._volume_levels_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._price_levels_cache: Dict[str, List[Dict[str, Any]]] = {}

    async def analyze_liquidity(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> LiquidityAnalysisResult:
        """Анализирует ликвидность на основе рыночных данных и ордербука"""
        try:
            if len(market_data) == 0:
                return LiquidityAnalysisResult(
                    liquidity_score=LiquidityScore(Decimal("0.0")),
                    confidence=ConfidenceLevel(Decimal("0.0")),
                    volume_score=0.0,
                    order_book_score=0.0,
                    volatility_score=0.0,
                    zones=[],
                    sweeps=[]
                )
            # Анализ объема
            volume_analysis = self._analyze_volume_profile(market_data)
            # Анализ дисбаланса в ордербуке
            order_book_analysis = await self._analyze_order_book_liquidity(order_book)
            # Анализ волатильности
            volatility_analysis = self._analyze_volatility(market_data)
            # Комбинированный анализ
            liquidity_score = (
                volume_analysis["volume_score"] * 0.4
                + order_book_analysis["order_book_score"] * 0.4
                + volatility_analysis["volatility_score"] * 0.2
            )
            confidence = min(liquidity_score, 1.0)
            return LiquidityAnalysisResult(
                liquidity_score=LiquidityScore(Decimal(str(liquidity_score))),
                confidence=ConfidenceLevel(Decimal(str(confidence))),
                volume_score=volume_analysis["volume_score"],
                order_book_score=order_book_analysis["order_book_score"],
                volatility_score=volatility_analysis["volatility_score"],
                zones=[],  # TODO: реализовать создание зон ликвидности
                sweeps=[]  # TODO: реализовать обнаружение ликвидных всплесков
            )
        except Exception as e:
            self.logger.error(f"Error analyzing liquidity: {str(e)}")
            return LiquidityAnalysisResult(
                liquidity_score=LiquidityScore(Decimal("0.0")),
                confidence=ConfidenceLevel(Decimal("0.0")),
                volume_score=0.0,
                order_book_score=0.0,
                volatility_score=0.0,
                zones=[],
                sweeps=[]
            )

    async def identify_liquidity_zones(
        self, market_data: pd.DataFrame
    ) -> List[LiquidityZone]:
        """Идентифицирует зоны ликвидности (поддержка/сопротивление)"""
        try:
            if len(market_data) == 0:
                return []
            zones = []
            # Анализ уровней на основе объема
            volume_levels = self._find_volume_levels(market_data)
            # Анализ уровней на основе цен
            price_levels = self._find_price_levels(market_data)
            # Объединение уровней
            all_levels = volume_levels + price_levels
            for level in all_levels:
                zone_type = self._classify_zone_type(level, market_data)
                strength = self._calculate_zone_strength(level, market_data)
                zones.append(
                    LiquidityZone(
                        price=PriceValue(Decimal(str(level["price"]))),
                        zone_type=zone_type,
                        strength=strength,
                        volume=VolumeValue(Decimal(str(level.get("volume", 0)))),
                        touches=level.get("touches", 0),
                        timestamp=TimestampValue(datetime.now()),
                        confidence=ConfidenceLevel(Decimal(str(strength)))
                    )
                )
            return zones
        except Exception as e:
            self.logger.error(f"Error identifying liquidity zones: {str(e)}")
            return []

    async def detect_liquidity_sweeps(
        self, market_data: pd.DataFrame
    ) -> List[LiquiditySweep]:
        """Обнаруживает sweep'ы ликвидности"""
        try:
            if len(market_data) == 0 or len(market_data) < 20:
                return []
            sweeps = []
            # Анализ последних свечей на предмет sweep'ов
            for i in range(max(0, len(market_data) - 20), len(market_data)):
                if callable(market_data):
                    market_data = market_data()
                if len(market_data) > i:
                    candle = market_data.iloc[i]
                else:
                    continue
                # Проверка на sweep выше
                if self._is_sweep_high(candle, market_data, i):
                    sweeps.append(
                        LiquiditySweep(
                            timestamp=TimestampValue(candle.name),
                            price=PriceValue(Decimal(str(float(candle["high"]) if hasattr(candle, "__getitem__") else 0.0))),
                            sweep_type=SweepType("sweep_high"),
                            confidence=ConfidenceLevel(Decimal(str(self._calculate_sweep_confidence(candle, "high")))),
                            volume=VolumeValue(Decimal("0.0"))
                        )
                    )
                # Проверка на sweep ниже
                if self._is_sweep_low(candle, market_data, i):
                    sweeps.append(
                        LiquiditySweep(
                            timestamp=TimestampValue(candle.name),
                            price=PriceValue(Decimal(str(float(candle["low"]) if hasattr(candle, "__getitem__") else 0.0))),
                            sweep_type=SweepType("sweep_low"),
                            confidence=ConfidenceLevel(Decimal(str(self._calculate_sweep_confidence(candle, "low")))),
                            volume=VolumeValue(Decimal("0.0"))
                        )
                    )
            return sweeps
        except Exception as e:
            self.logger.error(f"Error detecting liquidity sweeps: {str(e)}")
            return []

    def _analyze_volume_profile(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Анализирует профиль объема"""
        try:
            if "volume" not in market_data.columns:
                return {"volume_score": 0.0}
            volume_series = market_data["volume"]
            recent_volume = float(np.mean(volume_series.tail(20).to_numpy()))
            avg_volume = float(np.mean(volume_series.to_numpy()))
            volume_score = min(recent_volume / avg_volume if avg_volume > 0 else 0, 2.0)
            return {"volume_score": volume_score}
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {str(e)}")
            return {"volume_score": 0.0}

    async def _analyze_order_book_liquidity(
        self, order_book: Dict[str, Any]
    ) -> Dict[str, float]:
        """Анализирует ликвидность в ордербуке"""
        try:
            if not order_book or "bids" not in order_book or "asks" not in order_book:
                return {"order_book_score": 0.0}
            bid_depth = sum(order["size"] for order in order_book["bids"][:10])
            ask_depth = sum(order["size"] for order in order_book["asks"][:10])
            total_depth = bid_depth + ask_depth
            if total_depth == 0:
                return {"order_book_score": 0.0}
            # Нормализация глубины
            volume_threshold = self.config.get("volume_threshold", 100000)
            depth_score = min(total_depth / volume_threshold, 1.0)
            return {"order_book_score": depth_score}
        except Exception as e:
            self.logger.error(f"Error analyzing order book liquidity: {str(e)}")
            return {"order_book_score": 0.0}

    def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Анализирует волатильность"""
        try:
            if "close" not in market_data.columns:
                return {"volatility_score": 0.0}
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.std()
            # Нормализация волатильности
            volatility_score = min(volatility * 100, 1.0)
            return {"volatility_score": volatility_score}
        except Exception as e:
            self.logger.error(f"Error analyzing volatility: {str(e)}")
            return {"volatility_score": 0.0}

    def _find_volume_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Находит уровни на основе объема (оптимизировано)"""
        levels: List[Dict[str, Any]] = []
        if "volume" not in market_data.columns or "close" not in market_data.columns:
            return levels
        
        # Кэширование для повторных вычислений
        cache_key = f"volume_levels_{hash(tuple(market_data['close'].tail(10)))}"
        if hasattr(self, '_volume_levels_cache') and cache_key in self._volume_levels_cache:
            return self._volume_levels_cache[cache_key]
        
        if not hasattr(self, '_volume_levels_cache'):
            self._volume_levels_cache = {}
        
        # Оптимизированная группировка по ценовым уровням
        price_bins = pd.cut(market_data["close"], bins=50)
        volume_by_price = market_data.groupby(price_bins, observed=True)["volume"].sum()
        # Векторизованное нахождение пиков объема
        threshold = volume_by_price.quantile(0.8)
        high_volume_levels = volume_by_price[volume_by_price > threshold]
        for level, volume in high_volume_levels.items():
            try:
                if hasattr(level, 'mid') and hasattr(level, 'left') and hasattr(level, 'right'):
                    # Для pd.Interval используем mid
                    levels.append(
                        {
                            "price": float(level.mid),
                            "volume": float(volume),
                            "touches": len(
                                market_data[
                                    (market_data["close"] >= float(level.left))
                                    & (market_data["close"] <= float(level.right))
                                ]
                            ),
                        }
                    )
                else:
                    # Fallback для других типов
                    price_value = (
                        float(level) if isinstance(level, (int, float, str)) else 0.0
                    )
                    levels.append(
                        {"price": price_value, "volume": float(volume), "touches": 0}
                    )
            except (ValueError, TypeError, AttributeError):
                # Если не удается привести к float или получить атрибуты, пропускаем
                continue
        
        # Кэшируем результат с ограничением размера кэша
        if len(self._volume_levels_cache) > 50:
            oldest_keys = list(self._volume_levels_cache.keys())[:25]
            for key in oldest_keys:
                del self._volume_levels_cache[key]
        
        self._volume_levels_cache[cache_key] = levels
        return levels

    def _find_price_levels(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Находит уровни на основе цен (оптимизировано)"""
        levels: List[Dict[str, Any]] = []
        if "close" not in market_data.columns:
            return levels
        
        # Кэширование для оптимизации
        cache_key = f"price_levels_{hash(tuple(market_data['high'].tail(10)))}"
        if hasattr(self, '_price_levels_cache') and cache_key in self._price_levels_cache:
            return self._price_levels_cache[cache_key]
        
        if not hasattr(self, '_price_levels_cache'):
            self._price_levels_cache = {}
        
        # Векторизованное нахождение локальных экстремумов
        rolling_high = market_data["high"].rolling(window=5, center=True)
        rolling_low = market_data["low"].rolling(window=5, center=True)
        highs = rolling_high.max()
        lows = rolling_low.min()
        # Группируем близкие уровни
        unique_highs = highs.dropna().unique()
        unique_lows = lows.dropna().unique()
        for high in unique_highs:
            levels.append(
                {
                    "price": float(high),
                    "volume": 0,
                    "touches": len(market_data[market_data["high"] >= high * 0.99]),
                }
            )
        for low in unique_lows:
            levels.append(
                {
                    "price": float(low),
                    "volume": 0,
                    "touches": len(market_data[market_data["low"] <= low * 1.01]),
                }
            )
        
        # Кэшируем результат с ограничением размера кэша
        if len(self._price_levels_cache) > 50:
            oldest_keys = list(self._price_levels_cache.keys())[:25]
            for key in oldest_keys:
                del self._price_levels_cache[key]
        
        self._price_levels_cache[cache_key] = levels
        return levels

    def _classify_zone_type(
        self, level: Dict[str, Any], market_data: pd.DataFrame
    ) -> LiquidityZoneType:
        """Классифицирует тип зоны"""
        current_price = market_data["close"].iloc[-1]
        level_price = level["price"]
        if level_price > current_price * 1.02:
            return LiquidityZoneType.RESISTANCE
        elif level_price < current_price * 0.98:
            return LiquidityZoneType.SUPPORT
        else:
            return LiquidityZoneType.NEUTRAL

    def _calculate_zone_strength(
        self, level: Dict[str, Any], market_data: pd.DataFrame
    ) -> float:
        """Вычисляет силу зоны"""
        volume_threshold = self.config.get("volume_threshold", 100000)
        volume_factor = min(
            level.get("volume", 0) / volume_threshold, 1.0
        )
        touches_factor = min(level.get("touches", 0) / 10, 1.0)
        return float(volume_factor * 0.6 + touches_factor * 0.4)

    def _is_sweep_high(
        self, candle: pd.Series, market_data: pd.DataFrame, index: int
    ) -> bool:
        """Проверяет, является ли свеча sweep'ом выше"""
        if index < 5:
            return False
        # Проверяем, что high не является callable перед индексированием
        if callable(market_data):
            market_data = market_data()
        # Проверяем, что previous_highs не является callable перед вызовом max
        if len(market_data) > index:
            previous_highs = market_data.iloc[max(0, index - 5) : index]["high"]
            if callable(previous_highs):
                previous_highs = previous_highs()
            return float(candle["high"]) > previous_highs.max() if hasattr(candle, "__getitem__") else False
        else:
            return False

    def _is_sweep_low(
        self, candle: pd.Series, market_data: pd.DataFrame, index: int
    ) -> bool:
        """Проверяет, является ли свеча sweep'ом ниже"""
        if index < 5:
            return False
        # Проверяем, что low не является callable перед индексированием
        if callable(market_data):
            market_data = market_data()
        # Проверяем, что previous_lows не является callable перед вызовом min
        if len(market_data) > index:
            previous_lows = market_data.iloc[max(0, index - 5) : index]["low"]
            if callable(previous_lows):
                previous_lows = previous_lows()
            return float(candle["low"]) < previous_lows.min() if hasattr(candle, "__getitem__") else False
        else:
            return False

    def _calculate_sweep_confidence(self, candle: pd.Series, sweep_type: str) -> float:
        """Вычисляет уверенность в sweep'е"""
        if not hasattr(candle, "__getitem__"):
            return 0.0
        if sweep_type == "high":
            body_size = abs(float(candle["close"]) - float(candle["open"]))
            wick_size = float(candle["high"]) - max(float(candle["open"]), float(candle["close"]))
        else:
            body_size = abs(float(candle["close"]) - float(candle["open"]))
            wick_size = min(float(candle["open"]), float(candle["close"])) - float(candle["low"])
        if body_size == 0:
            return 0.0
        ratio = wick_size / body_size
        return min(ratio, 1.0)


class AnalysisError(Exception):
    """Ошибка анализа ликвидности."""

    pass
