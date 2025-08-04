"""
Промышленный сервис анализа рыночных условий.
Обеспечивает комплексный анализ рыночной среды для принятия решений о миграции агентов.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.types import Symbol
from domain.types.base_types import TimestampValue
from domain.value_objects.price import Price
from domain.entities.market import MarketData
from domain.repositories.market_repository import MarketRepository
from infrastructure.core.market_state import MarketState
from infrastructure.core.market_regime import MarketRegimeDetector
from infrastructure.services.technical_analysis_service import TechnicalAnalysisService


class MarketConditionType(Enum):
    """Типы рыночных условий."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_VOLATILE = "sideways_volatile"
    SIDEWAYS_STABLE = "sideways_stable"
    BREAKOUT_UP = "breakout_up"
    BREAKOUT_DOWN = "breakout_down"
    CONSOLIDATION = "consolidation"
    DISTRIBUTION = "distribution"
    ACCUMULATION = "accumulation"
    EXHAUSTION = "exhaustion"
    NO_STRUCTURE = "no_structure"


@dataclass
class MarketConditionsConfig:
    """Конфигурация анализа рыночных условий."""
    # Временные окна
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 200
    volatility_window: int = 30
    trend_window: int = 100
    
    # Пороги для классификации
    volatility_threshold_high: float = 0.03
    volatility_threshold_low: float = 0.01
    trend_strength_threshold: float = 0.6
    volume_threshold_high: float = 1.5
    volume_threshold_low: float = 0.7
    
    # Веса для расчета скора
    volatility_weight: float = 0.25
    trend_weight: float = 0.25
    volume_weight: float = 0.20
    momentum_weight: float = 0.15
    regime_weight: float = 0.15
    
    # Настройки кэширования
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000


@dataclass
class MarketConditionScore:
    """Скор рыночных условий."""
    overall_score: float
    volatility_score: float
    trend_score: float
    volume_score: float
    momentum_score: float
    regime_score: float
    condition_type: MarketConditionType
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MarketConditionsAnalyzer:
    """
    Промышленный анализатор рыночных условий.
    
    Обеспечивает комплексный анализ рыночной среды включая:
    - Волатильность и её динамику
    - Силу и направление трендов
    - Профиль объема и его изменения
    - Моментум и импульс
    - Режим рынка и его стабильность
    - Корреляции между активами
    - Микроструктурные паттерны
    """
    
    def __init__(
        self,
        market_repository: MarketRepository,
        technical_analysis_service: TechnicalAnalysisService,
        config: Optional[MarketConditionsConfig] = None
    ):
        self.market_repository = market_repository
        self.technical_analysis_service = technical_analysis_service
        self.config = config or MarketConditionsConfig()
        
        # Кэш для результатов анализа
        self._score_cache: Dict[str, Tuple[MarketConditionScore, datetime]] = {}
        self._market_data_cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        
        # Компоненты анализа
        self.regime_detector = MarketRegimeDetector()
        
        # Статистика использования
        self._analysis_stats = {
            "total_analyses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_processing_time": 0.0
        }
        
        logger.info("MarketConditionsAnalyzer initialized")

    async def calculate_market_score(
        self,
        symbol: Symbol,
        timeframe: str = "1h",
        lookback_periods: int = 100
    ) -> MarketConditionScore:
        """
        Расчет комплексного скора рыночных условий.
        
        Args:
            symbol: Торговая пара
            timeframe: Временной интервал
            lookback_periods: Количество периодов для анализа
            
        Returns:
            MarketConditionScore: Комплексный скор рыночных условий
        """
        start_time = datetime.now()
        
        try:
            # Проверяем кэш
            cache_key = f"{symbol}_{timeframe}_{lookback_periods}"
            cached_result = self._get_cached_score(cache_key)
            if cached_result:
                self._analysis_stats["cache_hits"] += 1
                return cached_result
            
            self._analysis_stats["cache_misses"] += 1
            
            # Получаем рыночные данные
            market_data = await self._get_market_data(symbol, timeframe, lookback_periods)
            if market_data.empty:
                logger.warning(f"No market data available for {symbol}")
                return self._create_default_score()
            
            # Выполняем комплексный анализ
            volatility_analysis = await self._analyze_volatility(market_data)
            trend_analysis = await self._analyze_trend(market_data)
            volume_analysis = await self._analyze_volume(market_data)
            momentum_analysis = await self._analyze_momentum(market_data)
            regime_analysis = await self._analyze_market_regime(market_data)
            
            # Рассчитываем общий скор
            overall_score = self._calculate_overall_score(
                volatility_analysis,
                trend_analysis,
                volume_analysis,
                momentum_analysis,
                regime_analysis
            )
            
            # Определяем тип рыночных условий
            condition_type = self._determine_condition_type(
                volatility_analysis,
                trend_analysis,
                volume_analysis,
                regime_analysis
            )
            
            # Рассчитываем уверенность
            confidence = self._calculate_confidence(
                volatility_analysis,
                trend_analysis,
                volume_analysis,
                momentum_analysis,
                regime_analysis
            )
            
            # Создаем результат
            score = MarketConditionScore(
                overall_score=overall_score,
                volatility_score=volatility_analysis["score"],
                trend_score=trend_analysis["score"],
                volume_score=volume_analysis["score"],
                momentum_score=momentum_analysis["score"],
                regime_score=regime_analysis["score"],
                condition_type=condition_type,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "lookback_periods": lookback_periods,
                    "volatility_analysis": volatility_analysis,
                    "trend_analysis": trend_analysis,
                    "volume_analysis": volume_analysis,
                    "momentum_analysis": momentum_analysis,
                    "regime_analysis": regime_analysis
                }
            )
            
            # Кэшируем результат
            self._cache_score(cache_key, score)
            
            # Обновляем статистику
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_processing_stats(processing_time)
            
            logger.debug(f"Market score calculated for {symbol}: {overall_score:.3f}")
            return score
            
        except Exception as e:
            logger.error(f"Error calculating market score for {symbol}: {e}")
            return self._create_default_score()

    async def _get_market_data(
        self,
        symbol: Symbol,
        timeframe: str,
        lookback_periods: int
    ) -> pd.DataFrame:
        """Получение рыночных данных с кэшированием."""
        cache_key = f"data_{symbol}_{timeframe}_{lookback_periods}"
        
        # Проверяем кэш данных
        cached_data = self._get_cached_market_data(cache_key)
        if cached_data is not None:
            return cached_data
        
        try:
            # Получаем данные из репозитория
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=lookback_periods)
            
            market_data_list = await self.market_repository.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=lookback_periods
            )
            
            if not market_data_list:
                return pd.DataFrame()
            
            # Преобразуем в DataFrame
            data_dict = {
                'timestamp': [],
                'open': [],
                'high': [],
                'low': [],
                'close': [],
                'volume': []
            }
            
            for data in market_data_list:
                data_dict['timestamp'].append(data.timestamp)
                data_dict['open'].append(float(data.open_price.amount))
                data_dict['high'].append(float(data.high_price.amount))
                data_dict['low'].append(float(data.low_price.amount))
                data_dict['close'].append(float(data.close_price.amount))
                data_dict['volume'].append(float(data.volume.amount))
            
            df = pd.DataFrame(data_dict)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Кэшируем данные
            self._cache_market_data(cache_key, df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return pd.DataFrame()

    async def _analyze_volatility(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ волатильности."""
        try:
            if market_data.empty or len(market_data) < self.config.volatility_window:
                return {"score": 0.5, "volatility": 0.0, "regime": "unknown"}
            
            # Рассчитываем доходности
            returns = market_data['close'].pct_change().dropna()
            
            # Текущая волатильность
            current_volatility = returns.rolling(window=self.config.volatility_window).std().iloc[-1]
            
            # Историческая волатильность
            historical_volatility = returns.std()
            
            # Волатильность волатильности
            vol_of_vol = returns.rolling(window=10).std().std()
            
            # Нормализованная волатильность
            normalized_vol = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
            
            # Определяем режим волатильности
            if current_volatility > self.config.volatility_threshold_high:
                vol_regime = "high"
                vol_score = 0.3  # Низкий скор для высокой волатильности
            elif current_volatility < self.config.volatility_threshold_low:
                vol_regime = "low"
                vol_score = 0.8  # Высокий скор для низкой волатильности
            else:
                vol_regime = "normal"
                vol_score = 0.6  # Средний скор для нормальной волатильности
            
            # Корректируем скор на основе стабильности
            stability_factor = 1.0 - min(vol_of_vol, 1.0)
            final_score = vol_score * stability_factor
            
            return {
                "score": float(final_score),
                "volatility": float(current_volatility),
                "historical_volatility": float(historical_volatility),
                "vol_of_vol": float(vol_of_vol),
                "normalized_vol": float(normalized_vol),
                "regime": vol_regime,
                "stability_factor": float(stability_factor)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {"score": 0.5, "volatility": 0.0, "regime": "unknown"}

    async def _analyze_trend(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ тренда."""
        try:
            if market_data.empty or len(market_data) < self.config.trend_window:
                return {"score": 0.5, "direction": "neutral", "strength": 0.0}
            
            close_prices = market_data['close']
            
            # Линейная регрессия для определения тренда
            x = np.arange(len(close_prices))
            y = close_prices.values
            
            if len(y) < 2:
                return {"score": 0.5, "direction": "neutral", "strength": 0.0}
            
            slope, intercept = np.polyfit(x, y, 1)
            
            # Рассчитываем силу тренда
            trend_strength = abs(slope) / close_prices.mean() if close_prices.mean() > 0 else 0.0
            
            # Определяем направление
            if slope > 0:
                direction = "up"
                direction_score = 0.7  # Благоприятный для роста
            elif slope < 0:
                direction = "down"
                direction_score = 0.3  # Неблагоприятный для роста
            else:
                direction = "neutral"
                direction_score = 0.5
            
            # Рассчитываем R-squared для уверенности в тренде
            y_pred = slope * x + intercept
            r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
            
            # Финальный скор тренда
            trend_score = direction_score * (0.5 + 0.5 * r_squared)
            
            return {
                "score": float(trend_score),
                "direction": direction,
                "strength": float(trend_strength),
                "slope": float(slope),
                "r_squared": float(r_squared),
                "direction_score": float(direction_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {"score": 0.5, "direction": "neutral", "strength": 0.0}

    async def _analyze_volume(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ объема."""
        try:
            if market_data.empty or len(market_data) < self.config.medium_window:
                return {"score": 0.5, "profile": "normal", "trend": "neutral"}
            
            volume = market_data['volume']
            
            # Средний объем
            avg_volume = volume.mean()
            current_volume = volume.iloc[-1]
            
            # Тренд объема
            volume_trend = volume.rolling(window=self.config.medium_window).mean()
            current_volume_trend = volume_trend.iloc[-1]
            
            # Относительный объем
            relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Определяем профиль объема
            if relative_volume > self.config.volume_threshold_high:
                volume_profile = "high"
                volume_score = 0.8  # Высокий объем благоприятен
            elif relative_volume < self.config.volume_threshold_low:
                volume_profile = "low"
                volume_score = 0.3  # Низкий объем неблагоприятен
            else:
                volume_profile = "normal"
                volume_score = 0.6
            
            # Анализ тренда объема
            if current_volume_trend > avg_volume * 1.1:
                volume_trend_direction = "increasing"
                trend_bonus = 0.1
            elif current_volume_trend < avg_volume * 0.9:
                volume_trend_direction = "decreasing"
                trend_bonus = -0.1
            else:
                volume_trend_direction = "stable"
                trend_bonus = 0.0
            
            final_score = max(0.0, min(1.0, volume_score + trend_bonus))
            
            return {
                "score": float(final_score),
                "profile": volume_profile,
                "trend": volume_trend_direction,
                "relative_volume": float(relative_volume),
                "current_volume": float(current_volume),
                "avg_volume": float(avg_volume),
                "volume_trend": float(current_volume_trend)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {"score": 0.5, "profile": "normal", "trend": "neutral"}

    async def _analyze_momentum(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ моментума."""
        try:
            if market_data.empty or len(market_data) < self.config.short_window:
                return {"score": 0.5, "momentum": 0.0, "direction": "neutral"}
            
            close_prices = market_data['close']
            
            # Рассчитываем моментум
            momentum_short = close_prices.iloc[-1] / close_prices.iloc[-self.config.short_window] - 1
            momentum_medium = close_prices.iloc[-1] / close_prices.iloc[-self.config.medium_window] - 1
            
            # RSI для дополнительного анализа моментума
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # Определяем направление моментума
            if momentum_short > 0 and momentum_medium > 0:
                momentum_direction = "positive"
                momentum_score = 0.8
            elif momentum_short < 0 and momentum_medium < 0:
                momentum_direction = "negative"
                momentum_score = 0.2
            else:
                momentum_direction = "mixed"
                momentum_score = 0.5
            
            # Корректируем на основе RSI
            if current_rsi > 70:
                rsi_adjustment = -0.1  # Перекупленность
            elif current_rsi < 30:
                rsi_adjustment = 0.1   # Перепроданность
            else:
                rsi_adjustment = 0.0
            
            final_score = max(0.0, min(1.0, momentum_score + rsi_adjustment))
            
            return {
                "score": float(final_score),
                "momentum_short": float(momentum_short),
                "momentum_medium": float(momentum_medium),
                "rsi": float(current_rsi),
                "direction": momentum_direction,
                "rsi_adjustment": float(rsi_adjustment)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing momentum: {e}")
            return {"score": 0.5, "momentum": 0.0, "direction": "neutral"}

    async def _analyze_market_regime(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Анализ режима рынка."""
        try:
            if market_data.empty or len(market_data) < self.config.long_window:
                return {"score": 0.5, "regime": "unknown", "stability": 0.0}
            
            # Используем технический анализ для определения режима
            technical_analysis = self.technical_analysis_service.perform_complete_analysis(market_data)
            
            # Анализируем структуру рынка
            market_structure = self.technical_analysis_service.analyze_market_structure(market_data)
            
            # Определяем режим на основе структуры
            if market_structure.get("trend_strength", 0) > self.config.trend_strength_threshold:
                if market_structure.get("trend_direction", "neutral") == "up":
                    regime = "bull_trending"
                    regime_score = 0.8
                else:
                    regime = "bear_trending"
                    regime_score = 0.3
            elif market_structure.get("volatility", 0) > self.config.volatility_threshold_high:
                regime = "sideways_volatile"
                regime_score = 0.4
            else:
                regime = "sideways_stable"
                regime_score = 0.6
            
            # Анализируем стабильность режима
            stability = self._calculate_regime_stability(market_data)
            
            # Корректируем скор на основе стабильности
            final_score = regime_score * (0.5 + 0.5 * stability)
            
            return {
                "score": float(final_score),
                "regime": regime,
                "stability": float(stability),
                "market_structure": market_structure,
                "technical_analysis": technical_analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing market regime: {e}")
            return {"score": 0.5, "regime": "unknown", "stability": 0.0}

    def _calculate_regime_stability(self, market_data: pd.DataFrame) -> float:
        """Расчет стабильности режима рынка."""
        try:
            if len(market_data) < 50:
                return 0.5
            
            # Анализируем консистентность тренда
            close_prices = market_data['close']
            
            # Разбиваем на сегменты и анализируем тренд в каждом
            segment_size = len(close_prices) // 5
            segment_trends = []
            
            for i in range(5):
                start_idx = i * segment_size
                end_idx = start_idx + segment_size
                segment = close_prices.iloc[start_idx:end_idx]
                
                if len(segment) < 2:
                    continue
                
                x = np.arange(len(segment))
                y = segment.values
                slope, _ = np.polyfit(x, y, 1)
                segment_trends.append(slope)
            
            if not segment_trends:
                return 0.5
            
            # Рассчитываем консистентность трендов
            positive_trends = sum(1 for trend in segment_trends if trend > 0)
            negative_trends = sum(1 for trend in segment_trends if trend < 0)
            
            # Стабильность = доля доминирующего направления
            max_direction = max(positive_trends, negative_trends)
            stability = max_direction / len(segment_trends)
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Error calculating regime stability: {e}")
            return 0.5

    def _calculate_overall_score(
        self,
        volatility_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        momentum_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any]
    ) -> float:
        """Расчет общего скора рыночных условий."""
        try:
            # Взвешенная сумма всех компонентов
            overall_score = (
                self.config.volatility_weight * volatility_analysis["score"] +
                self.config.trend_weight * trend_analysis["score"] +
                self.config.volume_weight * volume_analysis["score"] +
                self.config.momentum_weight * momentum_analysis["score"] +
                self.config.regime_weight * regime_analysis["score"]
            )
            
            return float(max(0.0, min(1.0, overall_score)))
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.5

    def _determine_condition_type(
        self,
        volatility_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any]
    ) -> MarketConditionType:
        """Определение типа рыночных условий."""
        try:
            trend_direction = trend_analysis.get("direction", "neutral")
            trend_strength = trend_analysis.get("strength", 0.0)
            volatility_regime = volatility_analysis.get("regime", "normal")
            volume_profile = volume_analysis.get("profile", "normal")
            market_regime = regime_analysis.get("regime", "unknown")
            
            # Определяем тип на основе комбинации факторов
            if trend_direction == "up" and trend_strength > self.config.trend_strength_threshold:
                if volume_profile == "high":
                    return MarketConditionType.BULL_TRENDING
                else:
                    return MarketConditionType.BREAKOUT_UP
            elif trend_direction == "down" and trend_strength > self.config.trend_strength_threshold:
                if volume_profile == "high":
                    return MarketConditionType.BEAR_TRENDING
                else:
                    return MarketConditionType.BREAKOUT_DOWN
            elif volatility_regime == "high":
                return MarketConditionType.SIDEWAYS_VOLATILE
            elif market_regime == "sideways_stable":
                return MarketConditionType.SIDEWAYS_STABLE
            elif volume_profile == "low" and trend_strength < 0.3:
                return MarketConditionType.CONSOLIDATION
            else:
                return MarketConditionType.NO_STRUCTURE
                
        except Exception as e:
            logger.error(f"Error determining condition type: {e}")
            return MarketConditionType.NO_STRUCTURE

    def _calculate_confidence(
        self,
        volatility_analysis: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        momentum_analysis: Dict[str, Any],
        regime_analysis: Dict[str, Any]
    ) -> float:
        """Расчет уверенности в анализе."""
        try:
            # Факторы уверенности
            trend_confidence = trend_analysis.get("r_squared", 0.0)
            regime_stability = regime_analysis.get("stability", 0.5)
            volatility_stability = volatility_analysis.get("stability_factor", 0.5)
            
            # Средняя уверенность
            confidence = (trend_confidence + regime_stability + volatility_stability) / 3
            
            return float(max(0.0, min(1.0, confidence)))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _get_cached_score(self, cache_key: str) -> Optional[MarketConditionScore]:
        """Получение скора из кэша."""
        if cache_key in self._score_cache:
            score, timestamp = self._score_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.config.cache_ttl_seconds:
                return score
            else:
                del self._score_cache[cache_key]
        return None

    def _cache_score(self, cache_key: str, score: MarketConditionScore) -> None:
        """Кэширование скора."""
        # Очищаем старые записи если кэш переполнен
        if len(self._score_cache) >= self.config.max_cache_size:
            oldest_key = min(self._score_cache.keys(), 
                           key=lambda k: self._score_cache[k][1])
            del self._score_cache[oldest_key]
        
        self._score_cache[cache_key] = (score, datetime.now())

    def _get_cached_market_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Получение рыночных данных из кэша."""
        if cache_key in self._market_data_cache:
            data, timestamp = self._market_data_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.config.cache_ttl_seconds:
                return data
            else:
                del self._market_data_cache[cache_key]
        return None

    def _cache_market_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Кэширование рыночных данных."""
        # Очищаем старые записи если кэш переполнен
        if len(self._market_data_cache) >= self.config.max_cache_size:
            oldest_key = min(self._market_data_cache.keys(), 
                           key=lambda k: self._market_data_cache[k][1])
            del self._market_data_cache[oldest_key]
        
        self._market_data_cache[cache_key] = (data, datetime.now())

    def _create_default_score(self) -> MarketConditionScore:
        """Создание скора по умолчанию."""
        return MarketConditionScore(
            overall_score=0.5,
            volatility_score=0.5,
            trend_score=0.5,
            volume_score=0.5,
            momentum_score=0.5,
            regime_score=0.5,
            condition_type=MarketConditionType.NO_STRUCTURE,
            confidence=0.0,
            timestamp=datetime.now()
        )

    def _update_processing_stats(self, processing_time: float) -> None:
        """Обновление статистики обработки."""
        self._analysis_stats["total_analyses"] += 1
        
        # Обновляем среднее время обработки
        total_analyses = self._analysis_stats["total_analyses"]
        current_avg = self._analysis_stats["avg_processing_time"]
        self._analysis_stats["avg_processing_time"] = (
            (current_avg * (total_analyses - 1) + processing_time) / total_analyses
        )

    def get_analysis_stats(self) -> Dict[str, Any]:
        """Получение статистики анализа."""
        return self._analysis_stats.copy()

    def clear_cache(self) -> None:
        """Очистка кэша."""
        self._score_cache.clear()
        self._market_data_cache.clear()
        logger.info("Market conditions analyzer cache cleared")

    async def shutdown(self) -> None:
        """Завершение работы анализатора."""
        self.clear_cache()
        logger.info("Market conditions analyzer shutdown completed") 