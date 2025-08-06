"""
Интегратор аналитических данных для маркет-мейкера.
"""

import pandas as pd
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from .types import (
    AnalyticalData,
    IAnalyticalIntegrator,
    IntegrationConfig,
    MarketMakerContext,
)


class AnalyticalIntegrator(IAnalyticalIntegrator):
    """Интегратор аналитических данных для маркет-мейкера."""

    def __init__(self, config: Optional[IntegrationConfig] = None) -> None:
        self.config = config or IntegrationConfig()
        self.context_cache: Dict[str, MarketMakerContext] = {}
        self.recommendations_cache: Dict[str, Dict[str, Any]] = {}
        self.analytical_cache: Dict[str, AnalyticalData] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        logger.info("AnalyticalIntegrator initialized")

    async def integrate_data(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> AnalyticalData:
        """Интегрирует данные для анализа маркет-мейкера."""
        try:
            # Анализ объема
            volume_analysis = self._analyze_volume(market_data)
            # Анализ волатильности
            volatility_analysis = self._analyze_volatility(market_data)
            # Анализ спреда
            spread_analysis = self._analyze_spread(order_book)
            # Анализ ликвидности
            liquidity_analysis = self._analyze_liquidity(order_book)
            # Рассчитываем общую уверенность
            confidence = self._calculate_confidence(
                volume_analysis,
                volatility_analysis,
                spread_analysis,
                liquidity_analysis,
            )
            return AnalyticalData(
                volume_metrics=volume_analysis,
                volatility_metrics=volatility_analysis,
                spread_metrics=spread_analysis,
                liquidity_metrics=liquidity_analysis,
                timestamp=datetime.now(),
                confidence=confidence,
            )
        except Exception as e:
            logger.error(f"Error integrating data: {e}")
            return AnalyticalData(
                volume_metrics={},
                volatility_metrics={},
                spread_metrics={},
                liquidity_metrics={},
                timestamp=datetime.now(),
                confidence=0.0,
            )

    async def get_market_maker_context(self, symbol: str) -> MarketMakerContext:
        """Получает контекст для маркет-мейкера."""
        try:
            # Проверяем кэш
            cache_key = f"context_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.context_cache[cache_key]
            # Получаем аналитические данные
            analytical_data = await self._get_analytical_data(symbol)
            # Определяем режим рынка
            market_regime = self._determine_market_regime(analytical_data)
            # Рассчитываем скоры
            liquidity_score = self._calculate_liquidity_score(analytical_data)
            volatility_score = self._calculate_volatility_score(analytical_data)
            spread_score = self._calculate_spread_score(analytical_data)
            context = MarketMakerContext(
                symbol=symbol,
                market_regime=market_regime,
                liquidity_score=liquidity_score,
                volatility_score=volatility_score,
                spread_score=spread_score,
                timestamp=datetime.now(),
            )
            # Кэшируем результат
            self._cache_context(cache_key, context)
            return context
        except Exception as e:
            logger.error(f"Error getting market maker context for {symbol}: {e}")
            return MarketMakerContext(
                symbol=symbol,
                market_regime="unknown",
                liquidity_score=0.0,
                volatility_score=0.0,
                spread_score=0.0,
                timestamp=datetime.now(),
            )

    def get_trading_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Получает торговые рекомендации."""
        try:
            # Проверяем кэш
            cache_key = f"recommendations_{symbol}"
            if self._is_cache_valid(cache_key):
                return self.recommendations_cache[cache_key]
            # Получаем аналитические данные
            analytical_data = self._get_cached_analytical_data(symbol)
            recommendations = {
                "action": self._determine_trading_action(analytical_data),
                "confidence": analytical_data.confidence,
                "price_adjustment": self._calculate_price_adjustment(analytical_data),
                "size_adjustment": self._calculate_size_adjustment(analytical_data),
                "aggressiveness_adjustment": self._calculate_aggressiveness_adjustment(
                    analytical_data
                ),
                "risk_level": self._calculate_risk_level(analytical_data),
                "market_conditions": self._analyze_market_conditions(analytical_data),
            }
            # Кэшируем результат
            self._cache_recommendations(cache_key, recommendations)
            return recommendations
        except Exception as e:
            logger.error(f"Error getting trading recommendations for {symbol}: {e}")
            return {
                "action": "hold",
                "confidence": 0.0,
                "price_adjustment": 0.0,
                "size_adjustment": 1.0,
                "aggressiveness_adjustment": 1.0,
                "risk_level": "high",
                "market_conditions": "unknown",
            }

    def should_proceed_with_trade(
        self, symbol: str, trade_aggression: float = 1.0
    ) -> bool:
        """Определяет, следует ли продолжать торговлю."""
        try:
            # Получаем аналитические данные
            analytical_data = self._get_cached_analytical_data(symbol)
            # Проверяем риск
            risk_level = self._calculate_risk_level(analytical_data)
            if risk_level == "extreme":
                return False
            # Проверяем ликвидность
            liquidity_score = analytical_data.liquidity_metrics.get(
                "total_liquidity", 0.0
            )
            if liquidity_score < 0.1:  # Минимальная ликвидность
                return False
            # Проверяем спред
            spread = analytical_data.spread_metrics.get("spread", 1.0)
            if spread > 0.01:  # Спред больше 1%
                return False
            # Корректируем на основе агрессивности
            if trade_aggression < 0.5 and risk_level == "high":
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking trade proceed for {symbol}: {e}")
            return True

    def get_adjusted_aggressiveness(
        self, symbol: str, base_aggressiveness: float
    ) -> float:
        """Получает скорректированную агрессивность."""
        try:
            analytical_data = self._get_cached_analytical_data(symbol)
            # Базовые факторы
            volatility_factor = 1.0 - analytical_data.volatility_metrics.get(
                "current_volatility", 0.0
            )
            liquidity_factor = analytical_data.liquidity_metrics.get(
                "total_liquidity", 1.0
            )
            spread_factor = (
                1.0 - analytical_data.spread_metrics.get("spread", 0.0) * 100
            )
            # Рассчитываем корректировку
            adjustment = (volatility_factor + liquidity_factor + spread_factor) / 3
            adjusted_aggressiveness = base_aggressiveness * adjustment
            # Ограничиваем значения
            return max(0.1, min(2.0, adjusted_aggressiveness))
        except Exception as e:
            logger.error(f"Error adjusting aggressiveness for {symbol}: {e}")
            return base_aggressiveness

    def get_adjusted_position_size(self, symbol: str, base_size: float) -> float:
        """Получает скорректированный размер позиции."""
        try:
            analytical_data = self._get_cached_analytical_data(symbol)
            # Факторы корректировки
            liquidity_factor = analytical_data.liquidity_metrics.get(
                "total_liquidity", 1.0
            )
            volatility_factor = 1.0 - analytical_data.volatility_metrics.get(
                "current_volatility", 0.0
            )
            confidence_factor = analytical_data.confidence
            # Рассчитываем корректировку
            adjustment = (liquidity_factor + volatility_factor + confidence_factor) / 3
            adjusted_size = base_size * adjustment
            # Ограничиваем значения
            return max(base_size * 0.1, min(base_size * 2.0, adjusted_size))
        except Exception as e:
            logger.error(f"Error adjusting position size for {symbol}: {e}")
            return base_size

    def get_adjusted_confidence(self, symbol: str, base_confidence: float) -> float:
        """Получает скорректированную уверенность."""
        try:
            analytical_data = self._get_cached_analytical_data(symbol)
            # Факторы корректировки
            data_confidence = analytical_data.confidence
            volatility_factor = 1.0 - analytical_data.volatility_metrics.get(
                "current_volatility", 0.0
            )
            liquidity_factor = analytical_data.liquidity_metrics.get(
                "total_liquidity", 1.0
            )
            # Рассчитываем корректировку
            adjustment = (data_confidence + volatility_factor + liquidity_factor) / 3
            adjusted_confidence = base_confidence * adjustment
            # Ограничиваем значения
            return max(0.0, min(1.0, adjusted_confidence))
        except Exception as e:
            logger.error(f"Error adjusting confidence for {symbol}: {e}")
            return base_confidence

    def get_price_offset(self, symbol: str, base_price: float, side: str) -> float:
        """Получает смещение цены."""
        try:
            analytical_data = self._get_cached_analytical_data(symbol)
            # Базовое смещение
            base_offset = base_price * 0.001  # 0.1%
            # Корректировка на основе спреда
            spread = analytical_data.spread_metrics.get("spread", 0.0)
            spread_adjustment = spread * base_price
            # Корректировка на основе волатильности
            volatility = analytical_data.volatility_metrics.get("current_volatility", 0.0)
            volatility_adjustment = volatility * base_price * 0.1
            # Общее смещение
            total_offset = base_offset + spread_adjustment + volatility_adjustment
            # Направление смещения
            if side.lower() == "buy":
                return total_offset
            else:
                return -total_offset
        except Exception as e:
            logger.error(f"Error calculating price offset for {symbol}: {e}")
            return 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики интегратора."""
        try:
            return {
                "cache_size": len(self.context_cache) + len(self.recommendations_cache) + len(self.analytical_cache),
                "cache_keys": list(self.context_cache.keys()) + list(self.recommendations_cache.keys()) + list(self.analytical_cache.keys()),
                "last_cache_update": max(self.cache_timestamps.values()).isoformat() if self.cache_timestamps else None,
                "config": {
                    "cache_ttl": self.config.cache_ttl,
                    "max_cache_size": self.config.max_cache_size,
                }
            }
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {"error": str(e)}

    def _analyze_volume(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Анализ объема."""
        try:
            if market_data.empty:
                return {}
            if "volume" not in market_data.columns:
                return {}
            volume = market_data["volume"]
            if volume.empty:
                return {}
            return {
                "current_volume": float(volume.iloc[-1]),
                "avg_volume": float(volume.mean()),
                "volume_ratio": (
                    float(volume.iloc[-1] / volume.mean()) if volume.mean() > 0 else 0.0
                ),
                "volume_trend": float(volume.pct_change().mean()),
                "volume_volatility": (
                    float(volume.std() / volume.mean()) if volume.mean() > 0 else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {}

    def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Анализ волатильности."""
        try:
            if market_data.empty:
                return {}
            if "close" not in market_data.columns:
                return {}
            close = market_data["close"]
            if close.empty:
                return {}
            returns = close.pct_change().dropna()
            if returns.empty:
                return {}
            return {
                "current_volatility": float(returns.std()),
                "avg_volatility": float(returns.rolling(20).std().mean()),
                "volatility_ratio": (
                    float(returns.std() / returns.rolling(20).std().mean())
                    if returns.rolling(20).std().mean() > 0
                    else 0.0
                ),
                "volatility_trend": float(
                    returns.rolling(10).std().pct_change().mean()
                ),
                "max_volatility": float(returns.rolling(20).std().max()),
            }
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}

    def _analyze_spread(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Анализ спреда."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            if not bids or not asks:
                return {}
            best_bid = float(bids[0].get("price", 0))
            best_ask = float(asks[0].get("price", 0))
            if best_bid <= 0 or best_ask <= 0:
                return {}
            spread = (best_ask - best_bid) / best_bid
            return {
                "spread": spread,
                "spread_bps": spread * 10000,
                "bid_volume": float(bids[0].get("size", 0)),
                "ask_volume": float(asks[0].get("size", 0)),
                "spread_imbalance": (
                    (float(asks[0].get("size", 0)) - float(bids[0].get("size", 0)))
                    / (float(asks[0].get("size", 0)) + float(bids[0].get("size", 0)))
                    if (float(asks[0].get("size", 0)) + float(bids[0].get("size", 0)))
                    > 0
                    else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing spread: {e}")
            return {}

    def _analyze_liquidity(self, order_book: Dict[str, Any]) -> Dict[str, float]:
        """Анализ ликвидности."""
        try:
            bids = order_book.get("bids", [])
            asks = order_book.get("asks", [])
            total_bid_volume = sum(float(bid.get("size", 0)) for bid in bids[:5])
            total_ask_volume = sum(float(ask.get("size", 0)) for ask in asks[:5])
            return {
                "total_liquidity": total_bid_volume + total_ask_volume,
                "bid_liquidity": total_bid_volume,
                "ask_liquidity": total_ask_volume,
                "liquidity_imbalance": (
                    (total_bid_volume - total_ask_volume)
                    / (total_bid_volume + total_ask_volume)
                    if (total_bid_volume + total_ask_volume) > 0
                    else 0.0
                ),
                "liquidity_depth": len(bids) + len(asks),
                "avg_order_size": (
                    (total_bid_volume + total_ask_volume) / (len(bids) + len(asks))
                    if (len(bids) + len(asks)) > 0
                    else 0.0
                ),
            }
        except Exception as e:
            logger.error(f"Error analyzing liquidity: {e}")
            return {}

    def _calculate_confidence(
        self,
        volume_metrics: Dict[str, float],
        volatility_metrics: Dict[str, float],
        spread_metrics: Dict[str, float],
        liquidity_metrics: Dict[str, float],
    ) -> float:
        """Рассчитывает общую уверенность."""
        try:
            factors = []
            # Фактор объема
            if volume_metrics:
                volume_ratio = volume_metrics.get("volume_ratio", 1.0)
                factors.append(min(1.0, volume_ratio))
            # Фактор волатильности
            if volatility_metrics:
                volatility_ratio = volatility_metrics.get("volatility_ratio", 1.0)
                factors.append(max(0.0, 1.0 - abs(volatility_ratio - 1.0)))
            # Фактор спреда
            if spread_metrics:
                spread = spread_metrics.get("spread", 0.001)
                factors.append(max(0.0, 1.0 - spread * 1000))
            # Фактор ликвидности
            if liquidity_metrics:
                total_liquidity = liquidity_metrics.get("total_liquidity", 0.0)
                factors.append(min(1.0, total_liquidity / 1000))  # Нормализация
            return sum(factors) / len(factors) if factors else 0.0
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Проверяет валидность кэша."""
        if cache_key not in self.cache_timestamps:
            return False
        # Проверяем наличие в любом из кэшей
        exists_in_cache = (cache_key in self.context_cache or 
                          cache_key in self.recommendations_cache or 
                          cache_key in self.analytical_cache)
        if not exists_in_cache:
            return False
        elapsed = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return elapsed < self.config.cache_ttl

    def _cache_context(self, cache_key: str, context: MarketMakerContext) -> None:
        """Кэширует контекст маркет-мейкера."""
        self.context_cache[cache_key] = context
        self.cache_timestamps[cache_key] = datetime.now()
        self._cleanup_cache()

    def _cache_recommendations(self, cache_key: str, recommendations: Dict[str, Any]) -> None:
        """Кэширует рекомендации."""
        self.recommendations_cache[cache_key] = recommendations
        self.cache_timestamps[cache_key] = datetime.now()
        self._cleanup_cache()

    def _cleanup_cache(self) -> None:
        """Очищает старые записи из кэша."""
        total_cache_size = len(self.context_cache) + len(self.recommendations_cache) + len(self.analytical_cache)
        if total_cache_size > self.config.max_cache_size:
            oldest_key = min(
                self.cache_timestamps.keys(), key=lambda k: self.cache_timestamps[k]
            )
            # Удаляем из соответствующего кэша
            if oldest_key in self.context_cache:
                del self.context_cache[oldest_key]
            elif oldest_key in self.recommendations_cache:
                del self.recommendations_cache[oldest_key]
            elif oldest_key in self.analytical_cache:
                del self.analytical_cache[oldest_key]
            del self.cache_timestamps[oldest_key]

    def _get_cached_analytical_data(self, symbol: str) -> AnalyticalData:
        """Получает кэшированные аналитические данные."""
        cache_key = f"analytical_data_{symbol}"
        if self._is_cache_valid(cache_key):
            return self.analytical_cache[cache_key]
        # Возвращаем пустые данные если нет в кэше
        return AnalyticalData()

    async def _get_analytical_data(self, symbol: str) -> AnalyticalData:
        """Получает аналитические данные для символа."""
        # Здесь должна быть логика получения реальных данных
        # Пока возвращаем пустые данные
        return AnalyticalData()

    def _determine_market_regime(self, analytical_data: AnalyticalData) -> str:
        """Определяет режим рынка."""
        volatility = analytical_data.volatility_metrics.get("current_volatility", 0.0)
        liquidity = analytical_data.liquidity_metrics.get("total_liquidity", 0.0)
        if volatility > 0.05:  # Высокая волатильность
            return "volatile"
        elif liquidity < 100:  # Низкая ликвидность
            return "illiquid"
        else:
            return "normal"

    def _calculate_liquidity_score(self, analytical_data: AnalyticalData) -> float:
        """Рассчитывает скор ликвидности."""
        total_liquidity = analytical_data.liquidity_metrics.get("total_liquidity", 0.0)
        return min(1.0, total_liquidity / 1000)

    def _calculate_volatility_score(self, analytical_data: AnalyticalData) -> float:
        """Рассчитывает скор волатильности."""
        volatility = analytical_data.volatility_metrics.get("current_volatility", 0.0)
        return max(0.0, 1.0 - volatility * 10)

    def _calculate_spread_score(self, analytical_data: AnalyticalData) -> float:
        """Рассчитывает скор спреда."""
        spread = analytical_data.spread_metrics.get("spread", 0.001)
        return max(0.0, 1.0 - spread * 1000)

    def _determine_trading_action(self, analytical_data: AnalyticalData) -> str:
        """Определяет торговое действие."""
        # Простая логика на основе аналитических данных
        confidence = analytical_data.confidence
        if confidence > 0.7:
            return "buy"
        elif confidence < 0.3:
            return "sell"
        else:
            return "hold"

    def _calculate_price_adjustment(self, analytical_data: AnalyticalData) -> float:
        """Рассчитывает корректировку цены."""
        spread = analytical_data.spread_metrics.get("spread", 0.001)
        volatility = analytical_data.volatility_metrics.get("current_volatility", 0.0)
        return (spread + volatility) * 0.5

    def _calculate_size_adjustment(self, analytical_data: AnalyticalData) -> float:
        """Рассчитывает корректировку размера."""
        liquidity = analytical_data.liquidity_metrics.get("total_liquidity", 1.0)
        confidence = analytical_data.confidence
        return min(2.0, (liquidity / 1000 + confidence) / 2)

    def _calculate_aggressiveness_adjustment(
        self, analytical_data: AnalyticalData
    ) -> float:
        """Рассчитывает корректировку агрессивности."""
        volatility = analytical_data.volatility_metrics.get("current_volatility", 0.0)
        return max(0.1, 1.0 - volatility * 5)

    def _calculate_risk_level(self, analytical_data: AnalyticalData) -> str:
        """Рассчитывает уровень риска."""
        volatility = analytical_data.volatility_metrics.get("current_volatility", 0.0)
        liquidity = analytical_data.liquidity_metrics.get("total_liquidity", 0.0)
        if volatility > 0.1 or liquidity < 50:
            return "extreme"
        elif volatility > 0.05 or liquidity < 200:
            return "high"
        elif volatility > 0.02 or liquidity < 500:
            return "medium"
        else:
            return "low"

    def _analyze_market_conditions(
        self, analytical_data: AnalyticalData
    ) -> Dict[str, Any]:
        """Анализирует рыночные условия."""
        return {
            "trend": self._determine_trend(analytical_data),
            "momentum": self._calculate_momentum(analytical_data),
            "support_resistance": self._identify_support_resistance(analytical_data),
        }

    def _determine_trend(self, analytical_data: AnalyticalData) -> str:
        """Определяет тренд."""
        volume_trend = analytical_data.volume_metrics.get("volume_trend", 0.0)
        if volume_trend > 0.1:
            return "bullish"
        elif volume_trend < -0.1:
            return "bearish"
        else:
            return "sideways"

    def _calculate_momentum(self, analytical_data: AnalyticalData) -> float:
        """Рассчитывает момент."""
        volume_ratio = analytical_data.volume_metrics.get("volume_ratio", 1.0)
        volatility_ratio = analytical_data.volatility_metrics.get(
            "volatility_ratio", 1.0
        )
        return (volume_ratio + volatility_ratio) / 2

    def _identify_support_resistance(
        self, analytical_data: AnalyticalData
    ) -> Dict[str, float]:
        """Идентифицирует уровни поддержки и сопротивления."""
        # Упрощенная логика
        return {"support": 0.95, "resistance": 1.05}
