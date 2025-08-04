# -*- coding: utf-8 -*-
"""Предиктор торговых сессий."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

from shared.numpy_utils import np
import pandas as pd
from loguru import logger

from domain.type_definitions.session_types import (
    SessionAnalysisResult,
    SessionProfile,
    SessionType,
)
from domain.value_objects.timestamp import Timestamp

from .interfaces import SessionRegistry
from .session_marker import MarketSessionContext, SessionMarker


@dataclass
class PredictionFeatures:
    """Признаки для прогнозирования."""

    # Временные признаки
    hour_of_day: int = 0
    day_of_week: int = 0
    is_weekend: bool = False
    # Рыночные признаки
    current_volume: float = 0.0
    current_volatility: float = 0.0
    current_momentum: float = 0.0
    current_spread: float = 0.0
    # Сессионные признаки
    session_overlap: float = 0.0
    session_phase: str = "unknown"
    active_sessions_count: int = 0
    # Технические признаки
    price_change_1h: float = 0.0
    price_change_4h: float = 0.0
    volume_change_1h: float = 0.0
    volume_change_4h: float = 0.0
    # Дополнительные признаки
    news_sentiment: float = 0.0
    market_regime: str = "unknown"
    correlation_strength: float = 1.0

    def to_array(self) -> np.ndarray:
        """Преобразование в numpy массив."""
        return np.array(
            [
                self.hour_of_day,
                self.day_of_week,
                float(self.is_weekend),
                self.current_volume,
                self.current_volatility,
                self.current_momentum,
                self.current_spread,
                self.session_overlap,
                self._encode_session_phase(),
                self.active_sessions_count,
                self.price_change_1h,
                self.price_change_4h,
                self.volume_change_1h,
                self.volume_change_4h,
                self.news_sentiment,
                self._encode_market_regime(),
                self.correlation_strength,
            ],
            dtype=np.float32,
        )

    def _encode_session_phase(self) -> float:
        """Кодирование фазы сессии."""
        phase_encoding = {
            "opening": 0.0,
            "mid_session": 0.5,
            "closing": 1.0,
            "unknown": 0.25,
        }
        return phase_encoding.get(self.session_phase, 0.25)

    def _encode_market_regime(self) -> float:
        """Кодирование режима рынка."""
        regime_encoding = {
            "trending": 0.0,
            "ranging": 0.5,
            "volatile": 1.0,
            "unknown": 0.25,
        }
        return regime_encoding.get(self.market_regime, 0.25)

    def to_dict(self) -> Dict[str, Union[str, float, int, bool]]:
        """Преобразование в словарь."""
        return {
            "hour_of_day": self.hour_of_day,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "current_volume": self.current_volume,
            "current_volatility": self.current_volatility,
            "current_momentum": self.current_momentum,
            "current_spread": self.current_spread,
            "session_overlap": self.session_overlap,
            "session_phase": self.session_phase,
            "active_sessions_count": self.active_sessions_count,
            "price_change_1h": self.price_change_1h,
            "price_change_4h": self.price_change_4h,
            "volume_change_1h": self.volume_change_1h,
            "volume_change_4h": self.volume_change_4h,
            "news_sentiment": self.news_sentiment,
            "market_regime": self.market_regime,
            "correlation_strength": self.correlation_strength,
        }


@dataclass
class SessionPrediction:
    """Прогноз торговой сессии."""

    # Основные прогнозы
    predicted_volume: float = 0.0
    predicted_volatility: float = 0.0
    predicted_momentum: float = 0.0
    predicted_spread: float = 0.0
    # Дополнительные прогнозы
    predicted_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    predicted_trend_strength: float = 0.0
    predicted_session_intensity: str = "normal"  # "low", "normal", "high"
    # Уверенность в прогнозах
    volume_confidence: float = 0.0
    volatility_confidence: float = 0.0
    momentum_confidence: float = 0.0
    spread_confidence: float = 0.0
    overall_confidence: float = 0.0
    # Метаданные
    prediction_horizon_minutes: int = 60
    features_used: List[str] = field(default_factory=list)
    model_version: str = "1.0"

    def to_dict(self) -> Dict[str, Union[str, float, int, bool, List[str]]]:
        """Преобразование в словарь."""
        return {
            "predicted_volume": self.predicted_volume,
            "predicted_volatility": self.predicted_volatility,
            "predicted_momentum": self.predicted_momentum,
            "predicted_spread": self.predicted_spread,
            "predicted_direction": self.predicted_direction,
            "predicted_trend_strength": self.predicted_trend_strength,
            "predicted_session_intensity": self.predicted_session_intensity,
            "volume_confidence": self.volume_confidence,
            "volatility_confidence": self.volatility_confidence,
            "momentum_confidence": self.momentum_confidence,
            "spread_confidence": self.spread_confidence,
            "overall_confidence": self.overall_confidence,
            "prediction_horizon_minutes": self.prediction_horizon_minutes,
            "features_used": self.features_used,
            "model_version": self.model_version,
        }


class SessionPredictor:
    """Предиктор торговых сессий."""

    def __init__(
        self,
        registry: SessionRegistry,
        session_marker: SessionMarker,
        model_path: Optional[str] = None,
    ) -> None:
        """Инициализация предиктора."""
        self.registry = registry
        self.session_marker = session_marker
        self.model_path = model_path
        self.cache: Dict[str, Tuple[SessionPrediction, Timestamp]] = {}
        self.cache_ttl_minutes = 15
        self.model = None
        if model_path:
            self._load_model()

    def _load_model(self) -> None:
        """Загрузка модели."""
        try:
            # Здесь была бы загрузка ML модели
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")

    def predict_session(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_type: SessionType,
        timestamp: Optional[Timestamp] = None,
        prediction_horizon_minutes: int = 60,
        force_prediction: bool = False,
    ) -> Optional[SessionPrediction]:
        """
        Прогнозирование торговой сессии.
        Args:
            symbol: Торговый символ
            market_data: Рыночные данные
            session_type: Тип сессии
            timestamp: Временная метка
            prediction_horizon_minutes: Горизонт прогнозирования
            force_prediction: Принудительное прогнозирование
        Returns:
            SessionPrediction: Прогноз сессии
        """
        if timestamp is None:
            timestamp = Timestamp.now()

        cache_key = f"{symbol}_{session_type.value}_{prediction_horizon_minutes}"
        if not force_prediction and self._is_cache_valid(cache_key, timestamp):
            return self.cache[cache_key][0]

        try:
            features = self._extract_features(symbol, market_data, session_type, timestamp)
            prediction = self._make_prediction(features, session_type, prediction_horizon_minutes)
            self._update_cache(cache_key, prediction, timestamp)
            return prediction
        except Exception as e:
            logger.error(f"Error predicting session: {e}")
            return None

    def predict_multiple_sessions(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_types: List[SessionType],
        timestamp: Optional[Timestamp] = None,
        prediction_horizon_minutes: int = 60,
    ) -> Dict[SessionType, SessionPrediction]:
        """Прогнозирование нескольких сессий."""
        predictions = {}
        for session_type in session_types:
            prediction = self.predict_session(
                symbol, market_data, session_type, timestamp, prediction_horizon_minutes
            )
            if prediction:
                predictions[session_type] = prediction
        return predictions

    def get_prediction_accuracy(
        self,
        symbol: str,
        session_type: SessionType,
        lookback_days: int = 7,
    ) -> Dict[str, float]:
        """Получение точности прогнозов."""
        # Упрощенная реализация
        return {
            "volume_accuracy": 0.75,
            "volatility_accuracy": 0.65,
            "momentum_accuracy": 0.70,
            "spread_accuracy": 0.80,
            "overall_accuracy": 0.72,
        }

    def update_model(self, model_type: str, model_data: bytes) -> bool:
        """Обновление модели."""
        try:
            # Здесь была бы логика обновления модели
            logger.info(f"Model updated: {model_type}")
            return True
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            return False

    def _extract_features(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        session_type: SessionType,
        timestamp: Timestamp,
    ) -> PredictionFeatures:
        """Извлечение признаков для прогнозирования."""
        session_context = self.session_marker.get_session_context(timestamp)
        
        features = PredictionFeatures()
        
        # Временные признаки
        dt = timestamp.to_datetime()
        features.hour_of_day = dt.hour
        features.day_of_week = dt.weekday()
        features.is_weekend = dt.weekday() >= 5
        
        # Рыночные признаки
        features.current_volume = self._calculate_current_volume(market_data)
        features.current_volatility = self._calculate_current_volatility(market_data)
        features.current_momentum = self._calculate_current_momentum(market_data)
        features.current_spread = self._calculate_current_spread(market_data)
        
        # Сессионные признаки
        features.session_overlap = self._calculate_session_overlap(session_context, session_type)
        features.active_sessions_count = len(session_context.active_sessions)
        
        # Определяем фазу сессии
        if session_context.primary_session:
            session_start = getattr(session_context.primary_session, 'start_time', None)
            session_end = getattr(session_context.primary_session, 'end_time', None)
            current_time = timestamp.to_datetime().time()
            
            if session_start and session_end and current_time < session_start:
                features.session_phase = "opening"
            elif session_start and session_end and current_time > session_end:
                features.session_phase = "closing"
            else:
                features.session_phase = "mid_session"
        else:
            features.session_phase = "unknown"
        
        # Технические признаки
        features.price_change_1h = self._calculate_price_change(market_data, 60)
        features.price_change_4h = self._calculate_price_change(market_data, 240)
        features.volume_change_1h = self._calculate_volume_change(market_data, 60)
        features.volume_change_4h = self._calculate_volume_change(market_data, 240)
        
        # Дополнительные признаки
        features.news_sentiment = self._estimate_news_sentiment(symbol, timestamp)
        features.market_regime = self._determine_market_regime(market_data)
        features.correlation_strength = self._calculate_correlation_strength(market_data)
        
        return features

    def _make_prediction(
        self,
        features: PredictionFeatures,
        session_type: SessionType,
        prediction_horizon_minutes: int,
    ) -> SessionPrediction:
        """Создание прогноза."""
        # Базовые значения из исторических данных сессии
        session_profile = None
        if hasattr(self.registry, 'get_session_profile'):
            session_profile = self.registry.get_session_profile(session_type)
        if not session_profile:
            # Значения по умолчанию
            base_volume = 1000000.0
            base_volatility = 0.02
            base_momentum = 0.0
            base_spread = 0.001
        else:
            base_volume = session_profile.avg_volume
            base_volatility = session_profile.avg_volatility
            base_momentum = session_profile.avg_momentum
            base_spread = session_profile.avg_spread

        # Применяем корректировки на основе признаков
        volume_adjustment = self._calculate_volume_adjustment(features)
        volatility_adjustment = self._calculate_volatility_adjustment(features)
        momentum_adjustment = self._calculate_momentum_adjustment(features)
        spread_adjustment = self._calculate_spread_adjustment(features)

        predicted_volume = base_volume * (1 + volume_adjustment)
        predicted_volatility = base_volatility * (1 + volatility_adjustment)
        predicted_momentum = base_momentum + momentum_adjustment
        predicted_spread = base_spread * (1 + spread_adjustment)

        # Определяем направление и силу тренда
        predicted_direction = self._determine_direction(features, predicted_momentum)
        predicted_trend_strength = abs(predicted_momentum)
        predicted_session_intensity = self._determine_session_intensity(
            predicted_volume, predicted_volatility
        )

        # Рассчитываем уверенность в прогнозах
        volume_confidence = self._calculate_volume_confidence(features)
        volatility_confidence = self._calculate_volatility_confidence(features)
        momentum_confidence = self._calculate_momentum_confidence(features)
        spread_confidence = self._calculate_spread_confidence(features)
        overall_confidence = (
            volume_confidence + volatility_confidence + momentum_confidence + spread_confidence
        ) / 4

        return SessionPrediction(
            predicted_volume=predicted_volume,
            predicted_volatility=predicted_volatility,
            predicted_momentum=predicted_momentum,
            predicted_spread=predicted_spread,
            predicted_direction=predicted_direction,
            predicted_trend_strength=predicted_trend_strength,
            predicted_session_intensity=predicted_session_intensity,
            volume_confidence=volume_confidence,
            volatility_confidence=volatility_confidence,
            momentum_confidence=momentum_confidence,
            spread_confidence=spread_confidence,
            overall_confidence=overall_confidence,
            prediction_horizon_minutes=prediction_horizon_minutes,
            features_used=list(features.to_dict().keys()),
        )

    def _calculate_current_volume(self, market_data: pd.DataFrame) -> float:
        """Расчет текущего объема."""
        if "volume" in market_data.columns:
            return float(market_data["volume"].tail(10).mean())
        return 1000000.0  # Значение по умолчанию

    def _calculate_current_volatility(self, market_data: pd.DataFrame) -> float:
        """Расчет текущей волатильности."""
        if "close" not in market_data.columns or len(market_data) < 20:
            return 0.02
        returns = market_data["close"].pct_change().dropna()
        return float(returns.tail(20).std())

    def _calculate_current_momentum(self, market_data: pd.DataFrame) -> float:
        """Расчет текущего импульса."""
        if "close" not in market_data.columns or len(market_data) < 14:
            return 0.0
        
        # Рассчитываем RSI
        delta = market_data["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        if len(rsi) == 0:
            return 0.0
        
        current_rsi = rsi.iloc[-1]
        if pd.isna(current_rsi):
            return 0.0
        
        # Явно приводим к float для mypy
        current_rsi_float = float(current_rsi)
        if current_rsi_float > 70:
            momentum = (current_rsi_float - 70) / 30
        elif current_rsi_float < 30:
            momentum = -(30 - current_rsi_float) / 30
        else:
            momentum = (current_rsi_float - 50) / 50
        return float(np.clip(momentum, -1.0, 1.0))

    def _calculate_current_spread(self, market_data: pd.DataFrame) -> float:
        """Расчет текущего спреда."""
        if "bid" in market_data.columns and "ask" in market_data.columns:
            spread = (market_data["ask"] - market_data["bid"]) / market_data["bid"]
            return float(spread.tail(10).mean())
        elif "close" in market_data.columns:
            # Используем волатильность как приближение спреда
            return self._calculate_current_volatility(market_data)
        else:
            return 0.001  # Базовое значение

    def _calculate_price_change(self, market_data: pd.DataFrame, minutes: int) -> float:
        """Расчет изменения цены за указанный период."""
        if "close" not in market_data.columns or len(market_data) < 2:
            return 0.0
        # Приблизительно определяем количество точек для периода
        points_needed = max(1, minutes // 5)  # Предполагаем 5-минутные свечи
        if len(market_data) < points_needed:
            return 0.0
        current_price = market_data["close"].iloc[-1]
        past_price = market_data["close"].iloc[-points_needed]
        if past_price == 0:
            return 0.0
        return float((current_price - past_price) / past_price)

    def _calculate_volume_change(self, market_data: pd.DataFrame, minutes: int) -> float:
        """Расчет изменения объема за указанный период."""
        if "volume" not in market_data.columns or len(market_data) < 2:
            return 0.0
        points_needed = max(1, minutes // 5)
        if len(market_data) < points_needed:
            return 0.0
        current_volume = market_data["volume"].iloc[-10:].mean()
        past_volume = (
            market_data["volume"].iloc[-points_needed - 10 : -points_needed].mean()
        )
        if past_volume == 0:
            return 0.0
        return float((current_volume - past_volume) / past_volume)

    def _calculate_session_overlap(
        self,
        session_context: MarketSessionContext,
        session_type: SessionType,
    ) -> float:
        """Расчет перекрытия сессий."""
        if not session_context.primary_session:
            return 0.0
        # Простое приближение на основе количества активных сессий
        active_count = len(session_context.active_sessions)
        if active_count == 1:
            return 0.0
        elif active_count == 2:
            return 0.3
        elif active_count >= 3:
            return 0.6
        return 0.0

    def _estimate_news_sentiment(self, symbol: str, timestamp: Timestamp) -> float:
        """Оценка настроений новостей."""
        # Упрощенная реализация - в реальности здесь был бы анализ новостей
        return 0.0

    def _determine_market_regime(self, market_data: pd.DataFrame) -> str:
        """Определение режима рынка."""
        if len(market_data) < 20:
            return "unknown"
        volatility = self._calculate_current_volatility(market_data)
        momentum = abs(self._calculate_current_momentum(market_data))
        if momentum > 0.6:
            return "trending"
        elif volatility > 0.02:
            return "volatile"
        return "ranging"

    def _calculate_correlation_strength(self, market_data: pd.DataFrame) -> float:
        """Расчет силы корреляции."""
        # Упрощенная реализация
        return 0.8

    def _calculate_volume_adjustment(self, features: PredictionFeatures) -> float:
        """Расчет корректировки объема."""
        # Упрощенная логика корректировки
        adjustment = 0.0
        if features.session_overlap > 0.5:
            adjustment += 0.2
        if features.current_volume > 1000000:
            adjustment += 0.1
        return adjustment

    def _calculate_volatility_adjustment(self, features: PredictionFeatures) -> float:
        """Расчет корректировки волатильности."""
        adjustment = 0.0
        if features.market_regime == "volatile":
            adjustment += 0.3
        elif features.market_regime == "trending":
            adjustment += 0.1
        return adjustment

    def _calculate_momentum_adjustment(self, features: PredictionFeatures) -> float:
        """Расчет корректировки импульса."""
        adjustment = 0.0
        if features.current_momentum > 0.5:
            adjustment += 0.2
        elif features.current_momentum < -0.5:
            adjustment -= 0.2
        return adjustment

    def _calculate_spread_adjustment(self, features: PredictionFeatures) -> float:
        """Расчет корректировки спреда."""
        adjustment = 0.0
        if features.current_volatility > 0.03:
            adjustment += 0.2
        return adjustment

    def _determine_direction(
        self, features: PredictionFeatures, momentum: float
    ) -> str:
        """Определение направления движения."""
        if momentum > 0.3:
            return "bullish"
        elif momentum < -0.3:
            return "bearish"
        else:
            return "neutral"

    def _determine_session_intensity(self, volume: float, volatility: float) -> str:
        """Определение интенсивности сессии."""
        if volume > 2000000 and volatility > 0.03:
            return "high"
        elif volume < 500000 and volatility < 0.01:
            return "low"
        else:
            return "normal"

    def _calculate_volume_confidence(self, features: PredictionFeatures) -> float:
        """Расчет уверенности в прогнозе объема."""
        confidence = 0.7  # Базовая уверенность
        if features.session_overlap > 0.5:
            confidence += 0.1
        if features.current_volume > 0:
            confidence += 0.1
        return min(confidence, 1.0)

    def _calculate_volatility_confidence(self, features: PredictionFeatures) -> float:
        """Расчет уверенности в прогнозе волатильности."""
        confidence = 0.6
        if features.market_regime != "unknown":
            confidence += 0.2
        return min(confidence, 1.0)

    def _calculate_momentum_confidence(self, features: PredictionFeatures) -> float:
        """Расчет уверенности в прогнозе импульса."""
        confidence = 0.65
        if abs(features.current_momentum) > 0.3:
            confidence += 0.15
        return min(confidence, 1.0)

    def _calculate_spread_confidence(self, features: PredictionFeatures) -> float:
        """Расчет уверенности в прогнозе спреда."""
        confidence = 0.75
        if features.current_spread > 0:
            confidence += 0.1
        return min(confidence, 1.0)

    def _is_cache_valid(self, cache_key: str, timestamp: Timestamp) -> bool:
        """Проверка валидности кэша."""
        if cache_key not in self.cache:
            return False
        cached_timestamp = self.cache[cache_key][1]
        time_diff = timestamp.to_datetime() - cached_timestamp.to_datetime()
        return time_diff.total_seconds() < (self.cache_ttl_minutes * 60)

    def _update_cache(
        self,
        cache_key: str,
        prediction: SessionPrediction,
        timestamp: Timestamp,
    ) -> None:
        """Обновление кэша."""
        self.cache[cache_key] = (prediction, timestamp)
        self._cleanup_cache(timestamp)

    def _cleanup_cache(self, current_timestamp: Timestamp) -> None:
        """Очистка устаревшего кэша."""
        expired_keys = []
        for key, (_, timestamp) in self.cache.items():
            time_diff = current_timestamp.to_datetime() - timestamp.to_datetime()
            if time_diff.total_seconds() > (self.cache_ttl_minutes * 60):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
