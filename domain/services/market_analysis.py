"""
Централизованный сервис анализа рынка для домена.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd

from domain.entities.market import MarketData
from domain.exceptions import MarketAnalysisError


@dataclass
class MarketSummary:
    """Сводка рынка."""

    symbol: str
    timeframe: str
    current_price: float
    price_change_24h: float
    price_change_percent: float
    volume_24h: float
    high_24h: float
    low_24h: float
    volatility: float
    trend_direction: str
    support_levels: List[float]
    resistance_levels: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VolumeProfile:
    """Профиль объёма."""

    symbol: str
    timeframe: str
    poc_price: float  # Point of Control
    total_volume: float
    volume_profile: Dict[float, float]
    price_range: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketRegime:
    """Рыночный режим."""

    symbol: str
    timeframe: str
    regime: str  # "volatile", "trending", "ranging", "quiet"
    volatility: float
    trend_strength: float
    price_trend: float
    volume_trend: float
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class MarketAnalysisProtocol(Protocol):
    """Протокол для анализа рынка."""

    def calculate_market_summary(
        self, market_data: List[MarketData], symbol: str, timeframe: str
    ) -> MarketSummary:
        """Рассчитать сводку рынка."""
        ...

    def calculate_volume_profile(
        self, market_data: List[MarketData], symbol: str, timeframe: str
    ) -> VolumeProfile:
        """Рассчитать профиль объёма."""
        ...

    def calculate_market_regime(
        self, market_data: List[MarketData], symbol: str, timeframe: str
    ) -> MarketRegime:
        """Рассчитать рыночный режим."""
        ...

    def find_support_levels(self, prices: List[float], window: int = 20) -> List[float]:
        """Найти уровни поддержки."""
        ...

    def find_resistance_levels(
        self, prices: List[float], window: int = 20
    ) -> List[float]:
        """Найти уровни сопротивления."""
        ...

    def determine_market_regime(self, prices: List[float], volumes: List[float]) -> str:
        """Определить рыночный режим."""
        ...


class MarketAnalysisService(MarketAnalysisProtocol):
    """Основной сервис анализа рынка."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def calculate_market_summary(
        self, market_data: List[MarketData], symbol: str, timeframe: str
    ) -> MarketSummary:
        """Рассчитать сводку рынка."""
        if not market_data:
            raise MarketAnalysisError("No market data provided")
        # Извлекаем данные
        prices = []
        volumes = []
        highs = []
        lows = []
        for data in market_data:
            try:
                close_price = self._extract_numeric_value(data.close_price)
                high_price = self._extract_numeric_value(data.high_price)
                low_price = self._extract_numeric_value(data.low_price)
                volume = self._extract_numeric_value(data.volume)
                prices.append(close_price)
                highs.append(high_price)
                lows.append(low_price)
                volumes.append(volume)
            except Exception:
                continue
        if not prices:
            raise MarketAnalysisError("Failed to extract price data")
        # Рассчитываем метрики
        current_price = prices[-1]
        price_change_24h = current_price - prices[0] if len(prices) > 1 else 0
        price_change_percent = (
            (price_change_24h / prices[0]) * 100 if prices[0] > 0 else 0
        )
        volume_24h = sum(volumes[-24:]) if len(volumes) >= 24 else sum(volumes)
        high_24h = max(highs[-24:]) if len(highs) >= 24 else max(highs)
        low_24h = min(lows[-24:]) if len(lows) >= 24 else min(lows)
        # Рассчитываем волатильность
        returns = np.diff(np.log(prices)) if len(prices) > 1 else [0]
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        # Определяем направление тренда
        if price_change_percent > 1:
            trend_direction = "up"
        elif price_change_percent < -1:
            trend_direction = "down"
        else:
            trend_direction = "sideways"
        # Находим уровни поддержки и сопротивления
        support_levels = self.find_support_levels(prices)
        resistance_levels = self.find_resistance_levels(prices)
        return MarketSummary(
            symbol=symbol,
            timeframe=timeframe,
            current_price=current_price,
            price_change_24h=price_change_24h,
            price_change_percent=price_change_percent,
            volume_24h=volume_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            volatility=volatility,
            trend_direction=trend_direction,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
        )

    def calculate_volume_profile(
        self, market_data: List[MarketData], symbol: str, timeframe: str
    ) -> VolumeProfile:
        """Рассчитать профиль объёма."""
        if not market_data:
            raise MarketAnalysisError("No market data provided")
        # Собираем данные
        price_volume_data = []
        for data in market_data:
            try:
                price = self._extract_numeric_value(data.close_price)
                volume = self._extract_numeric_value(data.volume)
                price_volume_data.append((price, volume))
            except Exception:
                continue
        if not price_volume_data:
            raise MarketAnalysisError("Failed to extract price-volume data")
        # Создаем профиль объема
        prices, volumes = zip(*price_volume_data)
        # Группируем по ценовым уровням
        price_bins = np.linspace(min(prices), max(prices), 50)
        volume_profile: Dict[float, float] = {}
        for price, volume in price_volume_data:
            bin_index = np.digitize(price, price_bins) - 1
            if 0 <= bin_index < len(price_bins):
                bin_price = price_bins[bin_index]
                volume_profile[bin_price] = volume_profile.get(bin_price, 0) + volume
        # Находим Point of Control (уровень с максимальным объемом)
        poc_price = max(volume_profile, key=lambda k: volume_profile[k]) if volume_profile else 0
        total_volume = sum(volume_profile.values())
        return VolumeProfile(
            symbol=symbol,
            timeframe=timeframe,
            poc_price=poc_price,
            total_volume=total_volume,
            volume_profile=volume_profile,
            price_range={
                "min": min(prices) if prices else 0,
                "max": max(prices) if prices else 0,
            },
        )

    def calculate_market_regime(
        self, market_data: List[MarketData], symbol: str, timeframe: str
    ) -> MarketRegime:
        """Рассчитать рыночный режим."""
        if not market_data:
            raise MarketAnalysisError("No market data provided")
        # Извлекаем данные
        prices = []
        volumes = []
        for data in market_data:
            try:
                price = self._extract_numeric_value(data.close_price)
                volume = self._extract_numeric_value(data.volume)
                prices.append(price)
                volumes.append(volume)
            except Exception:
                continue
        if len(prices) < 20:
            raise MarketAnalysisError("Insufficient data for market regime analysis")
        # Рассчитываем метрики
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        volume_trend = (
            np.polyfit(range(len(volumes)), volumes, 1)[0] if len(volumes) > 1 else 0
        )
        # Определяем режим
        regime = self.determine_market_regime(prices, volumes)
        # Рассчитываем силу тренда
        trend_strength = abs(price_trend) / volatility if volatility > 0 else 0
        confidence = min(trend_strength * 100, 100)
        return MarketRegime(
            symbol=symbol,
            timeframe=timeframe,
            regime=regime,
            volatility=volatility,
            trend_strength=trend_strength,
            price_trend=price_trend,
            volume_trend=volume_trend,
            confidence=confidence,
        )

    def find_support_levels(self, prices: List[float], window: int = 20) -> List[float]:
        """Найти уровни поддержки."""
        if len(prices) < window:
            return []
        support_levels = []
        for i in range(window, len(prices) - window):
            if all(prices[i] <= prices[j] for j in range(i - window, i + window + 1)):
                support_levels.append(prices[i])
        return sorted(list(set(support_levels)))

    def find_resistance_levels(
        self, prices: List[float], window: int = 20
    ) -> List[float]:
        """Найти уровни сопротивления."""
        if len(prices) < window:
            return []
        resistance_levels = []
        for i in range(window, len(prices) - window):
            if all(prices[i] >= prices[j] for j in range(i - window, i + window + 1)):
                resistance_levels.append(prices[i])
        return sorted(list(set(resistance_levels)))

    def determine_market_regime(self, prices: List[float], volumes: List[float]) -> str:
        """Определить рыночный режим."""
        if len(prices) < 20:
            return "unknown"
        # Рассчитываем метрики
        returns = np.diff(np.log(prices))
        volatility = np.std(returns) * np.sqrt(252)
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        # Определяем режим
        if volatility > 0.3:  # Высокая волатильность
            return "volatile"
        elif abs(price_trend) > 0.001:  # Сильный тренд
            return "trending"
        elif abs(price_trend) < 0.0001:  # Боковое движение
            return "ranging"
        else:
            return "quiet"

    def _extract_numeric_value(self, value_obj: Any) -> float:
        """Безопасное извлечение числового значения из value object."""
        try:
            if hasattr(value_obj, "amount"):
                return float(value_obj.amount)
            elif hasattr(value_obj, "value"):
                return float(value_obj.value)
            elif hasattr(value_obj, "__float__"):
                return float(value_obj)
            elif isinstance(value_obj, (int, float, Decimal)):
                return float(value_obj)
            else:
                return 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0


def create_market_analysis_service() -> MarketAnalysisProtocol:
    """Фабрика для создания сервиса анализа рынка."""
    return MarketAnalysisService()
