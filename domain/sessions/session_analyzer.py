# -*- coding: utf-8 -*-
"""Анализатор торговых сессий."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.types.session_types import (
    ConfidenceScore,
    MarketConditions,
    MarketRegime,
    PriceDirection,
    SessionAnalysisResult,
    SessionIntensity,
    SessionMetrics,
    SessionPhase,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .interfaces import BaseSessionAnalyzer, SessionRegistry
from .session_marker import MarketSessionContext, SessionMarker


@dataclass
class SessionAnalysisContext:
    """Контекст анализа сессии."""

    symbol: str
    timestamp: Timestamp
    session_context: MarketSessionContext
    market_data: pd.DataFrame
    session_profile: Optional[SessionProfile] = None
    # Дополнительные параметры анализа
    lookback_periods: int = 20
    confidence_threshold: float = 0.7
    volatility_threshold: float = 1.5

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        """Преобразование в словарь."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.to_iso(),
            "lookback_periods": self.lookback_periods,
            "confidence_threshold": self.confidence_threshold,
            "volatility_threshold": self.volatility_threshold,
        }


@dataclass
class SessionAnalysisMetrics:
    """Метрики анализа сессии."""

    # Базовые метрики
    volume_ratio: float = 0.0
    volatility_ratio: float = 0.0
    price_momentum: float = 0.0
    trend_strength: float = 0.0
    # Специфичные метрики сессии
    session_volume_multiplier: float = 1.0
    session_volatility_multiplier: float = 1.0
    session_momentum_multiplier: float = 1.0
    # Дополнительные метрики
    spread_impact: float = 1.0
    liquidity_impact: float = 1.0
    correlation_strength: float = 1.0

    def to_session_metrics(self) -> SessionMetrics:
        """Преобразование в SessionMetrics."""
        return SessionMetrics(
            volume_change_percent=(self.volume_ratio - 1.0) * 100,
            volatility_change_percent=(self.volatility_ratio - 1.0) * 100,
            price_direction_bias=self.price_momentum,
            momentum_strength=self.trend_strength,
            false_breakout_probability=self._calculate_false_breakout_probability(),
            reversal_probability=self._calculate_reversal_probability(),
            trend_continuation_probability=self._calculate_continuation_probability(),
            influence_duration_minutes=60,  # Базовое значение
            peak_influence_time_minutes=30,  # Базовое значение
            spread_impact=self.spread_impact,
            liquidity_impact=self.liquidity_impact,
            correlation_with_other_sessions=self.correlation_strength,
        )

    def _calculate_false_breakout_probability(self) -> float:
        """Расчет вероятности ложного пробоя."""
        # Высокая волатильность + слабый тренд = высокая вероятность ложного пробоя
        if self.volatility_ratio > 1.5 and self.trend_strength < 0.3:
            return 0.6
        elif self.volatility_ratio > 1.2 and self.trend_strength < 0.5:
            return 0.4
        else:
            return 0.2

    def _calculate_reversal_probability(self) -> float:
        """Расчет вероятности разворота."""
        # Сильный тренд + высокая волатильность = вероятность разворота
        if self.trend_strength > 0.7 and self.volatility_ratio > 1.3:
            return 0.4
        elif self.trend_strength > 0.5 and self.volatility_ratio > 1.1:
            return 0.3
        else:
            return 0.2

    def _calculate_continuation_probability(self) -> float:
        """Расчет вероятности продолжения тренда."""
        # Сильный тренд + стабильная волатильность = продолжение тренда
        if self.trend_strength > 0.6 and 0.8 < self.volatility_ratio < 1.2:
            return 0.7
        elif self.trend_strength > 0.4 and 0.9 < self.volatility_ratio < 1.1:
            return 0.6
        else:
            return 0.5


class SessionAnalyzer(BaseSessionAnalyzer):
    """Анализатор торговых сессий."""

    def __init__(
        self, registry: SessionRegistry, session_marker: SessionMarker
    ) -> None:
        super().__init__(registry)
        self.session_marker = session_marker

    def analyze_session(
        self, symbol: str, market_data: pd.DataFrame, timestamp: Timestamp
    ) -> Optional[SessionAnalysisResult]:
        """Анализ торговой сессии."""
        try:
            # Получаем контекст сессии
            session_context = self.session_marker.get_session_context(timestamp)
            if not session_context.primary_session:
                logger.warning(f"No active session found for {symbol} at {timestamp}")
                return None
            # Получаем профиль сессии
            session_profile = self.registry.get_profile(
                session_context.primary_session.session_type
            )
            if not session_profile:
                logger.error(
                    f"Session profile not found for {session_context.primary_session.session_type}"
                )
                return None
            # Создаем контекст анализа
            analysis_context = SessionAnalysisContext(
                symbol=symbol,
                timestamp=timestamp,
                session_context=session_context,
                market_data=market_data,
                session_profile=session_profile,
            )
            # Выполняем анализ
            analysis_metrics = self._analyze_session_metrics(analysis_context)
            # Генерируем прогнозы
            predictions = self._generate_predictions(analysis_context, analysis_metrics)
            # Определяем рыночные условия
            market_conditions = self._determine_market_conditions(analysis_metrics)
            # Рассчитываем уверенность
            confidence = self._calculate_confidence(analysis_context, analysis_metrics)
            # Извлекаем факторы риска
            risk_factors = self._extract_risk_factors(analysis_metrics)
            # Создаем результат анализа
            result = SessionAnalysisResult(
                session_type=session_context.primary_session.session_type,
                session_phase=session_context.primary_session.phase,
                timestamp=timestamp,
                confidence=ConfidenceScore(confidence),
                metrics=analysis_metrics.to_session_metrics(),
                market_conditions=market_conditions,
                predictions=predictions,
                risk_factors=risk_factors,
            )
            logger.info(
                f"Session analysis completed for {symbol} - "
                f"Session: {session_context.primary_session.session_type.value}, "
                f"Phase: {session_context.primary_session.phase.value}, "
                f"Confidence: {confidence:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Error analyzing session for {symbol}: {e}")
            return None

    def get_session_context(
        self, timestamp: Timestamp
    ) -> Dict[str, object]:
        """Получение контекста сессии."""
        context = self.session_marker.get_session_context(timestamp)
        return context.to_dict()

    def _analyze_session_metrics(
        self, context: SessionAnalysisContext
    ) -> SessionAnalysisMetrics:
        """Анализ метрик сессии."""
        if context.session_profile is None:
            raise ValueError("Session profile is required for analysis")
        # Анализируем объем
        volume_ratio = self._analyze_volume_ratio(context)
        # Анализируем волатильность
        volatility_ratio = self._analyze_volatility_ratio(context)
        # Анализируем импульс цены
        price_momentum = self._analyze_price_momentum(context)
        # Анализируем силу тренда
        trend_strength = self._analyze_trend_strength(context)
        # Корректируем на основе профиля сессии
        session_volume_multiplier = context.session_profile.typical_volume_multiplier
        session_volatility_multiplier = (
            context.session_profile.typical_volatility_multiplier
        )
        session_momentum_multiplier = context.session_profile.technical_signal_strength
        # Рассчитываем дополнительные метрики
        spread_impact = context.session_profile.typical_spread_multiplier
        liquidity_impact = 1.0 / spread_impact  # Обратная зависимость
        correlation_strength = (
            1.0 - context.session_profile.correlation_breakdown_probability
        )
        return SessionAnalysisMetrics(
            volume_ratio=volume_ratio * session_volume_multiplier,
            volatility_ratio=volatility_ratio * session_volatility_multiplier,
            price_momentum=price_momentum * session_momentum_multiplier,
            trend_strength=trend_strength,
            session_volume_multiplier=session_volume_multiplier,
            session_volatility_multiplier=session_volatility_multiplier,
            session_momentum_multiplier=session_momentum_multiplier,
            spread_impact=spread_impact,
            liquidity_impact=liquidity_impact,
            correlation_strength=correlation_strength,
        )

    def _analyze_volume_ratio(self, context: SessionAnalysisContext) -> float:
        """Анализ соотношения объема."""
        if (
            "volume" not in context.market_data.columns
            or len(context.market_data) < context.lookback_periods
        ):
            return 1.0
        volume_series: pd.Series = context.market_data["volume"]
        if len(volume_series) < context.lookback_periods:
            return 1.0
        # Рассчитываем текущий объем vs исторический
        recent_volume = volume_series.tail(5).mean()
        historical_volume = (
            volume_series.head(-5).tail(context.lookback_periods - 5).mean()
        )
        if historical_volume == 0:
            return 1.0
        return float(recent_volume / historical_volume)

    def _analyze_volatility_ratio(self, context: SessionAnalysisContext) -> float:
        """Анализ соотношения волатильности."""
        if (
            "close" not in context.market_data.columns
            or len(context.market_data) < context.lookback_periods
        ):
            return 1.0
        close_series: pd.Series = context.market_data["close"]
        if len(close_series) < context.lookback_periods:
            return 1.0
        # Рассчитываем волатильность
        returns = close_series.pct_change().dropna()
        if len(returns) < 10:
            return 1.0
        recent_volatility = returns.tail(5).std()
        historical_volatility = (
            returns.head(-5).tail(context.lookback_periods - 5).std()
        )
        if historical_volatility == 0:
            return 1.0
        return float(recent_volatility / historical_volatility)

    def _analyze_price_momentum(self, context: SessionAnalysisContext) -> float:
        """Анализ импульса цены на основе RSI."""
        if "close" not in context.market_data.columns or len(context.market_data) < 14:
            return 0.0
        close_series: pd.Series = context.market_data["close"]
        if len(close_series) < 14:
            return 0.0
        # Используем RSI для определения импульса
        delta = close_series.diff()
        gain = (delta.where(delta.gt(0), 0)).rolling(window=14).mean()
        loss = (delta.where(delta.lt(0), 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) == 0:
            return 0.0
        
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return 0.0
        
        current_rsi_float = float(current_rsi)
        if current_rsi_float > 70:
            momentum = (current_rsi_float - 70) / 30  # Сильный бычий импульс
            return float(np.clip(momentum, -1.0, 1.0))
        elif current_rsi_float < 30:
            momentum = -(30 - current_rsi_float) / 30  # Сильный медвежий импульс
            return float(np.clip(momentum, -1.0, 1.0))
        else:
            momentum = (current_rsi_float - 50) / 50  # Нейтральный импульс
            return float(np.clip(momentum, -1.0, 1.0))

    def _analyze_trend_strength(self, context: SessionAnalysisContext) -> float:
        """Анализ силы тренда."""
        if "close" not in context.market_data.columns or (hasattr(context.market_data.index, '__len__') and len(context.market_data.index) < 20):
            return 0.0
        close_series: pd.Series = context.market_data["close"]
        if hasattr(close_series, '__len__') and len(close_series) < 20:
            return 0.0
        # Используем линейную регрессию для определения силы тренда
        x = np.arange(len(close_series))
        y = close_series.values
        try:
            # Преобразуем y в numpy array для совместимости с polyfit
            y_array = np.asarray(y, dtype=float)
            slope = np.polyfit(x, y_array, 1)[0]
        except (ValueError, np.linalg.LinAlgError):
            return 0.0
        # Нормализуем наклон
        price_range = float(close_series.max() - close_series.min())
        if price_range == 0:
            return 0.0
        normalized_slope = abs(slope) / price_range
        return float(np.clip(normalized_slope * 10, 0.0, 1.0))

    def _generate_predictions(
        self, context: SessionAnalysisContext, metrics: SessionAnalysisMetrics
    ) -> Dict[str, float]:
        """Генерация прогнозов."""
        predictions: Dict[str, float] = {}
        # Прогноз волатильности
        base_volatility = 1.0
        predicted_volatility = base_volatility * metrics.volatility_ratio
        predictions["volatility"] = predicted_volatility
        # Прогноз объема
        base_volume = 1.0
        predicted_volume = base_volume * metrics.volume_ratio
        predictions["volume"] = predicted_volume
        # Прогноз направления
        predictions["direction"] = metrics.price_momentum
        # Прогноз импульса
        predictions["momentum"] = metrics.trend_strength
        return predictions

    def _determine_market_conditions(
        self, metrics: SessionAnalysisMetrics
    ) -> MarketConditions:
        """Определение рыночных условий."""
        # Определяем режим рынка
        if metrics.trend_strength > 0.6:
            market_regime = MarketRegime.TRENDING_BULL
        elif metrics.volatility_ratio > 1.3:
            market_regime = MarketRegime.VOLATILE
        else:
            market_regime = MarketRegime.RANGING
        # Определяем интенсивность сессии
        if metrics.volume_ratio > 1.5:
            session_intensity = SessionIntensity.HIGH
        elif metrics.volume_ratio < 0.7:
            session_intensity = SessionIntensity.LOW
        else:
            session_intensity = SessionIntensity.NORMAL
        return MarketConditions(
            volatility=metrics.volatility_ratio,
            volume=metrics.volume_ratio,
            spread=metrics.spread_impact,
            liquidity=metrics.liquidity_impact,
            momentum=metrics.trend_strength,
            trend_strength=abs(metrics.price_momentum),
            market_regime=market_regime,
            session_intensity=session_intensity,
        )

    def _calculate_confidence(
        self, context: SessionAnalysisContext, metrics: SessionAnalysisMetrics
    ) -> float:
        """Расчет уверенности в анализе."""
        confidence_factors: List[float] = []
        # Фактор качества данных
        if len(context.market_data.index) >= context.lookback_periods:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        # Фактор стабильности метрик
        if 0.8 < metrics.volatility_ratio < 1.2:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        # Фактор силы сигнала
        confidence_factors.append(metrics.trend_strength)
        # Фактор корреляции
        confidence_factors.append(metrics.correlation_strength)
        # Среднее значение всех факторов
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        return 0.5

    def _extract_risk_factors(self, metrics: SessionAnalysisMetrics) -> List[str]:
        """Извлечение факторов риска."""
        risk_factors: List[str] = []
        if metrics.volatility_ratio > 1.5:  # Используем фиксированное значение
            risk_factors.append("high_volatility_risk")
        if metrics.volume_ratio < 0.5:
            risk_factors.append("low_liquidity_risk")
        if metrics.correlation_strength < 0.7:
            risk_factors.append("correlation_breakdown_risk")
        if abs(metrics.price_momentum) > 0.8:
            risk_factors.append("extreme_momentum_risk")
        return risk_factors
