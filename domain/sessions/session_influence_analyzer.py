# -*- coding: utf-8 -*-
"""Анализатор влияния торговых сессий на рыночное поведение."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from shared.numpy_utils import np
import pandas as pd
from loguru import logger
from pandas import Series

from domain.type_definitions.session_types import (
    ConfidenceScore,
    InfluenceType,
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
class SessionInfluenceMetrics:
    """Метрики влияния сессии."""

    # Базовые метрики
    volume_change_percent: float = 0.0
    volatility_change_percent: float = 0.0
    price_direction_bias: float = 0.0  # -1.0 to 1.0
    momentum_strength: float = 0.0  # 0.0 to 1.0
    # Специфичные метрики
    false_breakout_probability: float = 0.0
    reversal_probability: float = 0.0
    trend_continuation_probability: float = 0.0
    # Временные характеристики
    influence_duration_minutes: int = 0
    peak_influence_time_minutes: int = 0
    # Дополнительные метрики
    spread_impact: float = 0.0
    liquidity_impact: float = 0.0
    correlation_with_other_sessions: float = 0.0

    def to_dict(self) -> Dict[str, Union[str, float, int]]:
        """Преобразование в словарь."""
        return {
            "volume_change_percent": self.volume_change_percent,
            "volatility_change_percent": self.volatility_change_percent,
            "price_direction_bias": self.price_direction_bias,
            "momentum_strength": self.momentum_strength,
            "false_breakout_probability": self.false_breakout_probability,
            "reversal_probability": self.reversal_probability,
            "trend_continuation_probability": self.trend_continuation_probability,
            "influence_duration_minutes": self.influence_duration_minutes,
            "peak_influence_time_minutes": self.peak_influence_time_minutes,
            "spread_impact": self.spread_impact,
            "liquidity_impact": self.liquidity_impact,
            "correlation_with_other_sessions": self.correlation_with_other_sessions,
        }

    def to_session_metrics(self) -> SessionMetrics:
        """Преобразование в SessionMetrics."""
        return SessionMetrics(
            volume_change_percent=self.volume_change_percent,
            volatility_change_percent=self.volatility_change_percent,
            price_direction_bias=self.price_direction_bias,
            momentum_strength=self.momentum_strength,
            false_breakout_probability=self.false_breakout_probability,
            reversal_probability=self.reversal_probability,
            trend_continuation_probability=self.trend_continuation_probability,
            influence_duration_minutes=self.influence_duration_minutes,
            peak_influence_time_minutes=self.peak_influence_time_minutes,
            spread_impact=self.spread_impact,
            liquidity_impact=self.liquidity_impact,
            correlation_with_other_sessions=self.correlation_with_other_sessions,
        )


@dataclass
class SessionInfluenceResult:
    """Результат анализа влияния сессии."""

    symbol: str
    session_type: SessionType
    session_phase: SessionPhase
    timestamp: Timestamp
    # Основные метрики влияния
    influence_metrics: SessionInfluenceMetrics
    # Прогнозные характеристики
    predicted_volatility: float = 0.0
    predicted_volume: float = 0.0
    predicted_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    confidence: float = 0.0
    # Дополнительная информация
    market_context: Dict[str, Union[str, float, int]] = field(default_factory=dict)
    historical_patterns: List[Dict[str, Union[str, float, int]]] = field(
        default_factory=list
    )

    @property
    def score(self) -> float:
        """Общий счет влияния сессии."""
        # Комбинируем confidence и predicted_volatility для общего счета
        return (self.confidence + self.predicted_volatility) / 2.0

    def to_dict(self) -> Dict[str, Union[str, float, int, Dict[str, Union[str, float, int]], List[Dict[str, Union[str, float, int]]]]]:
        """Преобразование в словарь."""
        return {
            "symbol": self.symbol,
            "session_type": self.session_type.value,
            "session_phase": self.session_phase.value,
            "timestamp": self.timestamp.to_iso(),
            "influence_metrics": self.influence_metrics.to_dict(),
            "predicted_volatility": self.predicted_volatility,
            "predicted_volume": self.predicted_volume,
            "predicted_direction": self.predicted_direction,
            "confidence": self.confidence,
            "market_context": self.market_context,
            "historical_patterns": self.historical_patterns,
        }

    def to_session_analysis_result(self) -> SessionAnalysisResult:
        """Преобразование в SessionAnalysisResult."""
        # Создаем базовые рыночные условия
        market_conditions = MarketConditions(
            volatility=self.predicted_volatility,
            volume=self.predicted_volume,
            spread=1.0,  # Базовое значение
            liquidity=1.0,  # Базовое значение
            momentum=self.influence_metrics.momentum_strength,
            trend_strength=abs(self.influence_metrics.price_direction_bias),
            market_regime=MarketRegime.RANGING,  # Базовое значение
            session_intensity=SessionIntensity.NORMAL,  # Базовое значение
        )
        # Определяем направление
        direction = PriceDirection.NEUTRAL
        if self.predicted_direction == "bullish":
            direction = PriceDirection.BULLISH
        elif self.predicted_direction == "bearish":
            direction = PriceDirection.BEARISH
        return SessionAnalysisResult(
            session_type=self.session_type,
            session_phase=self.session_phase,
            timestamp=self.timestamp,
            confidence=ConfidenceScore(self.confidence),
            metrics=self.influence_metrics.to_session_metrics(),
            market_conditions=market_conditions,
            predictions={
                "volatility": self.predicted_volatility,
                "volume": self.predicted_volume,
                "direction": float(direction.value == "bullish")
                - float(direction.value == "bearish"),
                "momentum": self.influence_metrics.momentum_strength,
            },
            risk_factors=self._extract_risk_factors(),
        )

    def _extract_risk_factors(self) -> List[str]:
        """Извлечение факторов риска."""
        risk_factors: List[str] = []
        if self.influence_metrics.false_breakout_probability > 0.4:
            risk_factors.append("high_false_breakout_risk")
        if self.influence_metrics.reversal_probability > 0.3:
            risk_factors.append("high_reversal_risk")
        if self.influence_metrics.correlation_with_other_sessions < 0.7:
            risk_factors.append("correlation_breakdown_risk")
        if self.predicted_volatility > 1.5:
            risk_factors.append("high_volatility_risk")
        return risk_factors


class SessionInfluenceAnalyzer(BaseSessionAnalyzer):
    """Анализатор влияния торговых сессий на рыночное поведение."""

    def __init__(
        self, registry: SessionRegistry, session_marker: SessionMarker
    ) -> None:
        super().__init__(registry)
        self.session_marker = session_marker

    def analyze_session(
        self, symbol: str, market_data: pd.DataFrame, timestamp: Timestamp
    ) -> Optional[SessionAnalysisResult]:
        """Анализ сессии."""
        try:
            # Получаем контекст сессии
            session_context = self.session_marker.get_session_context(timestamp)
            # Анализируем влияние
            influence_result = self.analyze_session_influence(
                symbol, market_data, session_context, timestamp
            )
            if influence_result:
                return influence_result.to_session_analysis_result()
            return None
        except Exception as e:
            logger.error(f"Error analyzing session for {symbol}: {e}")
            return None

    def get_session_context(self, timestamp: Timestamp) -> Dict[str, object]:
        """Получение контекста сессии."""
        context = self.session_marker.get_session_context(timestamp)
        return context.to_dict()

    def analyze_session_influence(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_context: Optional[MarketSessionContext] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[SessionInfluenceResult]:
        """
        Анализ влияния торговой сессии на рыночное поведение.
        Args:
            symbol: Торговая пара
            market_data: Рыночные данные
            session_context: Контекст сессий
            timestamp: Время анализа
        Returns:
            Результат анализа влияния
        """
        try:
            if timestamp is None:
                timestamp = Timestamp.now()
            if session_context is None:
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
            # Анализируем влияние
            influence_metrics = self._calculate_influence_metrics(
                symbol, market_data, session_profile, session_context
            )
            # Генерируем прогнозы
            predicted_volatility, predicted_volume, predicted_direction, confidence = (
                self._generate_predictions(
                    symbol, session_profile, session_context, influence_metrics
                )
            )
            # Анализируем исторические паттерны
            historical_patterns = self._analyze_historical_patterns(
                symbol, session_profile, session_context
            )
            # Создаем результат
            result = SessionInfluenceResult(
                symbol=symbol,
                session_type=session_context.primary_session.session_type,
                session_phase=session_context.primary_session.phase,
                timestamp=timestamp,
                influence_metrics=influence_metrics,
                predicted_volatility=predicted_volatility,
                predicted_volume=predicted_volume,
                predicted_direction=predicted_direction,
                confidence=confidence,
                historical_patterns=historical_patterns,
            )
            logger.info(
                f"Session influence analysis completed for {symbol} - "
                f"Session: {session_context.primary_session.session_type.value}, "
                f"Phase: {session_context.primary_session.phase.value}, "
                f"Confidence: {confidence:.2f}"
            )
            return result
        except Exception as e:
            logger.error(f"Error analyzing session influence for {symbol}: {e}")
            return None

    def _calculate_influence_metrics(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_profile: SessionProfile,
        session_context: MarketSessionContext,
    ) -> SessionInfluenceMetrics:
        """Расчет метрик влияния сессии."""
        # Базовые расчеты
        volume_change_percent = self._calculate_volume_change(
            market_data, session_profile
        )
        volatility_change_percent = self._calculate_volatility_change(
            market_data, session_profile
        )
        price_direction_bias = self._calculate_direction_bias(
            market_data, session_profile
        )
        momentum_strength = self._calculate_momentum_strength(
            market_data, session_profile
        )
        # Корректировки на основе профиля сессии
        adjusted_price_direction_bias = (
            price_direction_bias + session_profile.typical_direction_bias
        )
        adjusted_momentum_strength = (
            momentum_strength * session_profile.technical_signal_strength
        )
        # Специфичные метрики
        false_breakout_probability = session_profile.false_breakout_probability * (
            1 + session_profile.manipulation_susceptibility
        )
        reversal_probability = session_profile.reversal_probability
        trend_continuation_probability = session_profile.continuation_probability
        # Временные характеристики
        influence_duration_minutes = (
            session_profile.behavior.typical_volatility_spike_minutes
        )
        peak_influence_time_minutes = influence_duration_minutes // 2
        # Дополнительные метрики
        spread_impact = session_profile.typical_spread_multiplier
        liquidity_impact = (
            1.0 / session_profile.typical_spread_multiplier
        )  # Обратная зависимость
        correlation_with_other_sessions = (
            1.0 - session_profile.correlation_breakdown_probability
        )
        return SessionInfluenceMetrics(
            volume_change_percent=volume_change_percent,
            volatility_change_percent=volatility_change_percent,
            price_direction_bias=adjusted_price_direction_bias,
            momentum_strength=adjusted_momentum_strength,
            false_breakout_probability=false_breakout_probability,
            reversal_probability=reversal_probability,
            trend_continuation_probability=trend_continuation_probability,
            influence_duration_minutes=influence_duration_minutes,
            peak_influence_time_minutes=peak_influence_time_minutes,
            spread_impact=spread_impact,
            liquidity_impact=liquidity_impact,
            correlation_with_other_sessions=correlation_with_other_sessions,
        )

    def _generate_predictions(
        self,
        symbol: str,
        session_profile: SessionProfile,
        session_context: MarketSessionContext,
        influence_metrics: SessionInfluenceMetrics,
    ) -> Tuple[float, float, str, float]:
        """Генерация прогнозов на основе анализа сессии."""
        # Базовые прогнозы
        base_volatility = 1.0
        base_volume = 1.0
        # Корректируем на основе профиля сессии
        predicted_volatility = (
            base_volatility * session_profile.typical_volatility_multiplier
        )
        predicted_volume = base_volume * session_profile.typical_volume_multiplier
        # Корректировка по фазе
        if session_context.primary_session is not None:
            phase_adjustments = self._get_phase_adjustments(
                session_context.primary_session.phase
            )
            predicted_volatility *= phase_adjustments["volatility"]
            predicted_volume *= phase_adjustments["volume"]
        # Определяем направление
        direction_bias = influence_metrics.price_direction_bias
        if direction_bias > 0.1:
            predicted_direction = "bullish"
        elif direction_bias < -0.1:
            predicted_direction = "bearish"
        else:
            predicted_direction = "neutral"
        # Расчет уверенности
        confidence = self._calculate_prediction_confidence(
            session_context, influence_metrics, session_profile
        )
        return predicted_volatility, predicted_volume, predicted_direction, confidence

    def _calculate_prediction_confidence(
        self,
        session_context: MarketSessionContext,
        influence_metrics: SessionInfluenceMetrics,
        session_profile: SessionProfile,
    ) -> float:
        """Расчет уверенности в прогнозе."""
        confidence_factors: List[float] = []
        # Фактор технической силы сигналов
        confidence_factors.append(session_profile.technical_signal_strength)
        # Фактор силы импульса
        confidence_factors.append(influence_metrics.momentum_strength)
        # Фактор стабильности корреляций
        confidence_factors.append(influence_metrics.correlation_with_other_sessions)
        # Фактор ликвидности
        confidence_factors.append(min(influence_metrics.liquidity_impact, 1.0))
        # Фактор количества активных сессий
        active_sessions_count = len(session_context.active_sessions)
        if active_sessions_count > 0:
            session_factor = min(active_sessions_count / 3.0, 1.0)  # Нормализуем до 1.0
            confidence_factors.append(session_factor)
        # Среднее значение всех факторов
        if confidence_factors:
            return sum(confidence_factors) / len(confidence_factors)
        return 0.5  # Базовое значение

    def _analyze_historical_patterns(
        self,
        symbol: str,
        session_profile: SessionProfile,
        session_context: MarketSessionContext,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Анализ исторических паттернов для сессии."""
        patterns: List[Dict[str, Union[str, float, int]]] = []
        # Паттерны из профиля сессии
        for pattern in session_profile.behavior.common_patterns:
            patterns.append(
                {
                    "type": "session_pattern",
                    "name": pattern,
                    "probability": 0.7,  # Базовая вероятность
                    "description": f"Типичный паттерн для {session_profile.session_type.value} сессии",
                }
            )
        # Паттерны на основе фазы
        if session_context.primary_session is not None:
            phase_patterns = self._get_phase_patterns(
                session_context.primary_session.phase
            )
            patterns.extend(phase_patterns)
        # Паттерны на основе рыночных условий
        if session_context.market_conditions:
            market_patterns = self._get_market_patterns(
                session_context.market_conditions
            )
            patterns.extend(market_patterns)
        return patterns

    def _calculate_volume_change(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет изменения объема."""
        if "volume" not in market_data.columns or len(market_data) < 2:
            return 0.0
        volume_series: pd.Series = market_data["volume"]
        if len(volume_series) < 2:
            return 0.0
        # Рассчитываем текущий объем vs исторический
        recent_volume = volume_series.iloc[-10:].mean()
        historical_volume = volume_series.iloc[:-10].mean()
        if historical_volume == 0:
            return 0.0
        volume_change = (recent_volume - historical_volume) / historical_volume
        return float(volume_change * 100)  # В процентах

    def _calculate_volatility_change(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет изменения волатильности."""
        if "close" not in market_data.columns or len(market_data) < 20:
            return 0.0
        close_series: pd.Series = market_data["close"]
        if len(close_series) < 20:
            return 0.0
        # Рассчитываем волатильность
        returns = close_series.pct_change().dropna()
        if len(returns) < 10:
            return 0.0
        recent_volatility = returns.iloc[-10:].std()
        historical_volatility = returns.iloc[:-10].std()
        if historical_volatility == 0:
            return 0.0
        volatility_change = (
            recent_volatility - historical_volatility
        ) / historical_volatility
        return float(volatility_change * 100)  # В процентах

    def _calculate_direction_bias(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет смещения направления."""
        if "close" not in market_data.columns or len(market_data) < 20:
            return 0.0
        close_series: pd.Series = market_data["close"]
        if len(close_series) < 20:
            return 0.0
        # Рассчитываем направление тренда
        recent_prices = close_series.iloc[-20:]
        if len(recent_prices) < 2:
            return 0.0
        # Линейная регрессия для определения тренда
        x = np.arange(len(recent_prices))
        y = np.array(recent_prices.values)
        try:
            slope = np.polyfit(x, y, 1)[0]
        except (ValueError, np.linalg.LinAlgError):
            return session_profile.typical_direction_bias
        # Нормализуем наклон
        price_range = float(recent_prices.max() - recent_prices.min())
        if price_range == 0:
            return session_profile.typical_direction_bias
        normalized_slope = slope / price_range
        # Ограничиваем значения от -1 до 1
        return float(np.clip(normalized_slope * 10, -1.0, 1.0))

    def _calculate_momentum_strength(
        self, market_data: pd.DataFrame, session_profile: SessionProfile
    ) -> float:
        """Расчет силы импульса на основе RSI."""
        if "close" not in market_data.columns or len(market_data) < 14:
            return 0.0
        close_series: Series = market_data["close"]
        # Рассчитываем RSI
        delta = close_series.diff()
        gain = (delta.where(delta.gt(0.0), 0.0)).rolling(window=14).mean()
        loss = (delta.where(delta.lt(0.0), 0.0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) == 0:
            return 0.0
        
        # Нормализуем RSI к диапазону 0-1
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return 0.0
        
        current_rsi_float = float(current_rsi)
        if current_rsi_float > 70:
            momentum = (current_rsi_float - 70) / 30  # Сильный бычий импульс
            return float(momentum)
        elif current_rsi_float < 30:
            momentum = (30 - current_rsi_float) / 30  # Сильный медвежий импульс
            return float(momentum)
        else:
            return 0.5  # Нейтральный импульс

    def _get_phase_adjustments(self, phase: SessionPhase) -> Dict[str, float]:
        """Получение корректировок по фазе сессии."""
        adjustments = {
            "volatility": 1.0,
            "volume": 1.0,
            "direction": 1.0,
            "momentum": 1.0,
        }
        if phase == SessionPhase.OPENING:
            adjustments["volatility"] = 1.2
            adjustments["volume"] = 1.3
        elif phase == SessionPhase.MID_SESSION:
            adjustments["volatility"] = 1.0
            adjustments["volume"] = 1.0
        elif phase == SessionPhase.CLOSING:
            adjustments["volatility"] = 1.1
            adjustments["volume"] = 1.2
        return adjustments

    def _get_phase_patterns(
        self, phase: SessionPhase
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Получение паттернов для фазы сессии."""
        patterns: List[Dict[str, Union[str, float, int]]] = []
        if phase == SessionPhase.OPENING:
            patterns.append(
                {
                    "type": "phase_pattern",
                    "name": "opening_gap",
                    "probability": 0.6,
                    "description": "Паттерн открытия с гэпом",
                }
            )
        elif phase == SessionPhase.CLOSING:
            patterns.append(
                {
                    "type": "phase_pattern",
                    "name": "end_of_day_reversal",
                    "probability": 0.4,
                    "description": "Разворот в конце дня",
                }
            )
        return patterns

    def _get_market_patterns(
        self, market_conditions: MarketConditions
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Получение паттернов на основе рыночных условий."""
        patterns: List[Dict[str, Union[str, float, int]]] = []
        
        # Безопасное извлечение значений
        volatility = getattr(market_conditions, 'volatility', 1.0)
        volume = getattr(market_conditions, 'volume', 1.0)
        
        # Проверяем, что значения являются числами
        if isinstance(volatility, pd.Series):
            volatility_float: float = volatility.iloc[0] if len(volatility) > 0 else 1.0
        else:
            volatility_float = float(volatility)
        if isinstance(volume, pd.Series):
            volume_float: float = volume.iloc[0] if len(volume) > 0 else 1.0
        else:
            volume_float = float(volume)
        
        # Приводим к float для корректного сравнения
        volatility_final = float(volatility_float)
        volume_final = float(volume_float)
        
        if volatility_final > 1.5:
            patterns.append(
                {
                    "type": "market_pattern",
                    "name": "high_volatility_breakout",
                    "probability": 0.7,
                    "description": "Пробой при высокой волатильности",
                }
            )
        if volume_final > 1.3:
            patterns.append(
                {
                    "type": "market_pattern",
                    "name": "volume_spike",
                    "probability": 0.8,
                    "description": "Скачок объема",
                }
            )
        return patterns
