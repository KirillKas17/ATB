# -*- coding: utf-8 -*-
"""Комбинированный предиктор с интеграцией сигналов торговых сессий."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger

from application.prediction.pattern_predictor import (
    EnhancedPredictionResult,
    PatternPredictor,
)
from application.signal.session_signal_engine import (
    SessionInfluenceSignal,
    SessionSignalEngine,
)
from domain.sessions.session_marker import SessionMarker
from domain.value_objects.timestamp import Timestamp


@dataclass
class CombinedPredictionResult:
    """Результат комбинированного прогнозирования."""

    # Базовый прогноз паттерна
    pattern_prediction: Optional[EnhancedPredictionResult] = None

    # Сигналы сессий
    session_signals: Dict[str, SessionInfluenceSignal] = field(default_factory=dict)
    aggregated_session_signal: Optional[SessionInfluenceSignal] = None

    # Комбинированный результат
    final_direction: str = "neutral"  # "bullish", "bearish", "neutral"
    final_confidence: float = 0.0
    final_return_percent: float = 0.0
    final_duration_minutes: int = 0

    # Модификаторы
    session_confidence_boost: float = 1.0
    session_aggressiveness_modifier: float = 1.0
    session_position_multiplier: float = 1.0

    # Метаданные
    prediction_timestamp: Timestamp = field(default_factory=Timestamp.now)
    alignment_score: float = 0.0  # Совпадение направлений паттерна и сессий

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "pattern_prediction": (
                self.pattern_prediction.to_dict() if self.pattern_prediction else None
            ),
            "session_signals": {
                k: v.to_dict() for k, v in self.session_signals.items()
            },
            "aggregated_session_signal": (
                self.aggregated_session_signal.to_dict()
                if self.aggregated_session_signal
                else None
            ),
            "final_direction": self.final_direction,
            "final_confidence": self.final_confidence,
            "final_return_percent": self.final_return_percent,
            "final_duration_minutes": self.final_duration_minutes,
            "session_confidence_boost": self.session_confidence_boost,
            "session_aggressiveness_modifier": self.session_aggressiveness_modifier,
            "session_position_multiplier": self.session_position_multiplier,
            "prediction_timestamp": self.prediction_timestamp.to_iso(),
            "alignment_score": self.alignment_score,
        }


class CombinedPredictor:
    """Комбинированный предиктор с интеграцией сигналов торговых сессий."""

    def __init__(
        self,
        pattern_predictor: Optional[PatternPredictor] = None,
        session_signal_engine: Optional[SessionSignalEngine] = None,
        session_marker: Optional[SessionMarker] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.pattern_predictor = pattern_predictor
        self.session_signal_engine = session_signal_engine
        self.session_marker = session_marker or SessionMarker()

        self.config = config or {
            "session_weight": 0.3,  # Вес сигналов сессий в финальном прогнозе
            "pattern_weight": 0.7,  # Вес прогноза паттернов
            "min_session_confidence": 0.6,  # Минимальная уверенность сигнала сессии
            "alignment_boost_threshold": 0.8,  # Порог для усиления при совпадении направлений
            "enable_session_modifiers": True,
            "enable_alignment_analysis": True,
        }

        # Статистика
        self.stats = {
            "total_predictions": 0,
            "session_aligned_predictions": 0,
            "high_confidence_predictions": 0,
            "session_boosted_predictions": 0,
        }

        logger.info("CombinedPredictor initialized")

    async def predict(
        self,
        symbol: str,
        market_data: Optional[Dict[str, Any]] = None,
        pattern_detection: Optional[Any] = None,
        timestamp: Optional[Timestamp] = None,
    ) -> Optional[CombinedPredictionResult]:
        """
        Комбинированное прогнозирование с учетом сигналов сессий.

        Args:
            symbol: Торговая пара
            market_data: Рыночные данные
            pattern_detection: Обнаруженный паттерн
            timestamp: Время прогнозирования

        Returns:
            Комбинированный результат прогнозирования
        """
        try:
            if timestamp is None:
                timestamp = Timestamp.now()

            # Получаем прогноз паттерна
            pattern_prediction = None
            if self.pattern_predictor and pattern_detection:
                pattern_prediction = self.pattern_predictor.predict_pattern_outcome(
                    pattern_detection, market_data
                )

            # Получаем сигналы сессий
            session_signals = {}
            aggregated_session_signal = None

            if self.session_signal_engine:
                # Генерируем сигналы для всех активных сессий
                current_signals = await self.session_signal_engine.get_current_signals(
                    symbol
                )
                for signal in current_signals:
                    if signal.confidence >= self.config["min_session_confidence"]:
                        session_signals[signal.session_type] = signal

                # Получаем агрегированный сигнал
                aggregated_session_signal = (
                    await self.session_signal_engine.get_aggregated_signal(symbol)
                )

            # Комбинируем результаты
            result = self._combine_predictions(
                symbol,
                pattern_prediction,
                session_signals,
                aggregated_session_signal,
                timestamp,
            )

            if result:
                # Обновляем статистику
                self._update_stats(result)

                logger.info(
                    f"Combined prediction for {symbol}: "
                    f"direction={result.final_direction}, "
                    f"confidence={result.final_confidence:.3f}, "
                    f"alignment={result.alignment_score:.3f}"
                )

            return result

        except Exception as e:
            logger.error(f"Error in combined prediction for {symbol}: {e}")
            return None

    def _combine_predictions(
        self,
        symbol: str,
        pattern_prediction: Optional[EnhancedPredictionResult],
        session_signals: Dict[str, SessionInfluenceSignal],
        aggregated_session_signal: Optional[SessionInfluenceSignal],
        timestamp: Timestamp,
    ) -> CombinedPredictionResult:
        """Комбинирование прогнозов паттернов и сигналов сессий."""

        # Инициализируем результат
        result = CombinedPredictionResult(
            pattern_prediction=pattern_prediction,
            session_signals=session_signals,
            aggregated_session_signal=aggregated_session_signal,
            prediction_timestamp=timestamp,
        )

        # Базовые значения из прогноза паттерна
        pattern_confidence = 0.0
        pattern_direction = "neutral"
        pattern_return = 0.0
        pattern_duration = 0

        if pattern_prediction:
            pattern_confidence = pattern_prediction.prediction.confidence
            pattern_direction = pattern_prediction.prediction.predicted_direction
            pattern_return = pattern_prediction.prediction.predicted_return_percent
            pattern_duration = pattern_prediction.prediction.predicted_duration_minutes

        # Значения из сигналов сессий
        session_confidence = 0.0
        session_direction = "neutral"
        session_score = 0.0

        if aggregated_session_signal:
            session_confidence = aggregated_session_signal.confidence
            session_direction = aggregated_session_signal.tendency
            session_score = aggregated_session_signal.score

        # Вычисляем финальные значения с весами
        pattern_weight = self.config["pattern_weight"]
        session_weight = self.config["session_weight"]

        # Комбинированная уверенность
        result.final_confidence = (
            pattern_confidence * pattern_weight + session_confidence * session_weight
        )

        # Определяем финальное направление
        if pattern_confidence > 0.7 and session_confidence > 0.6:
            # Оба сигнала достаточно уверены
            if pattern_direction == session_direction:
                result.final_direction = pattern_direction
                result.alignment_score = 1.0
                # Усиливаем уверенность при совпадении
                if result.final_confidence > self.config["alignment_boost_threshold"]:
                    result.final_confidence = min(1.0, result.final_confidence * 1.2)
                    result.session_confidence_boost = 1.2
            else:
                # Направления не совпадают - используем более уверенный
                if pattern_confidence > session_confidence:
                    result.final_direction = pattern_direction
                    result.alignment_score = 0.0
                else:
                    result.final_direction = session_direction
                    result.alignment_score = 0.0
        elif pattern_confidence > 0.7:
            # Только прогноз паттерна уверен
            result.final_direction = pattern_direction
            result.alignment_score = 0.5
        elif session_confidence > 0.6:
            # Только сигнал сессии уверен
            result.final_direction = session_direction
            result.alignment_score = 0.5
        else:
            # Ни один сигнал не уверен
            result.final_direction = "neutral"
            result.alignment_score = 0.0

        # Комбинированный возврат и длительность
        if pattern_prediction:
            result.final_return_percent = pattern_return
            result.final_duration_minutes = pattern_duration
        else:
            # Оцениваем на основе сигналов сессий
            result.final_return_percent = (
                session_score * 2.0
            )  # 2% при максимальном скоре
            result.final_duration_minutes = 30  # Базовая длительность

        # Применяем модификаторы сессий
        if self.config["enable_session_modifiers"] and aggregated_session_signal:
            result = self._apply_session_modifiers(result, aggregated_session_signal)

        return result

    def _apply_session_modifiers(
        self, result: CombinedPredictionResult, session_signal: SessionInfluenceSignal
    ) -> CombinedPredictionResult:
        """Применение модификаторов на основе сигналов сессий."""

        # Модификатор уверенности
        if session_signal.confidence >= 0.8:
            result.session_confidence_boost = 1.3
        elif session_signal.confidence >= 0.6:
            result.session_confidence_boost = 1.1

        # Модификатор агрессивности
        if session_signal.tendency == "bullish" and session_signal.score > 0.5:
            result.session_aggressiveness_modifier = 1.2
        elif session_signal.tendency == "bearish" and session_signal.score < -0.5:
            result.session_aggressiveness_modifier = 0.8
        else:
            result.session_aggressiveness_modifier = 1.0

        # Модификатор размера позиции
        if session_signal.volume_impact > 0.2:
            result.session_position_multiplier = 1.3
        elif session_signal.volume_impact < -0.2:
            result.session_position_multiplier = 0.7
        else:
            result.session_position_multiplier = 1.0

        return result

    def _update_stats(self, result: CombinedPredictionResult) -> None:
        """Обновление статистики."""
        self.stats["total_predictions"] = int(self.stats.get("total_predictions", 0)) + 1

        if result.alignment_score > 0.7:
            self.stats["session_aligned_predictions"] = int(self.stats.get("session_aligned_predictions", 0)) + 1

        if result.final_confidence >= 0.8:
            self.stats["high_confidence_predictions"] = int(self.stats.get("high_confidence_predictions", 0)) + 1

        if result.session_confidence_boost > 1.0:
            self.stats["session_boosted_predictions"] = int(self.stats.get("session_boosted_predictions", 0)) + 1

    def get_prediction_with_session_context(
        self,
        symbol: str,
        pattern_prediction: Optional[EnhancedPredictionResult] = None,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[CombinedPredictionResult]:
        """
        Получение прогноза с учетом контекста сессий.

        Args:
            symbol: Торговая пара
            pattern_prediction: Прогноз паттерна
            market_context: Рыночный контекст

        Returns:
            Комбинированный результат
        """
        try:
            # Получаем контекст сессий
            session_context = self.session_marker.get_session_context()

            # Создаем базовый результат
            result = CombinedPredictionResult(
                pattern_prediction=pattern_prediction,
                prediction_timestamp=Timestamp.now(),
            )

            if pattern_prediction:
                # Применяем базовые значения из прогноза паттерна
                result.final_direction = (
                    pattern_prediction.prediction.predicted_direction
                )
                result.final_confidence = pattern_prediction.prediction.confidence
                result.final_return_percent = (
                    pattern_prediction.prediction.predicted_return_percent
                )
                result.final_duration_minutes = (
                    pattern_prediction.prediction.predicted_duration_minutes
                )

                # Корректируем на основе контекста сессий
                if session_context.primary_session:
                    session_profile = self.session_marker.registry.get_profile(
                        session_context.primary_session.session_type
                    )

                    if session_profile:
                        # Применяем поведенческие характеристики сессии
                        behavior = session_profile.behavior

                        # Корректируем уверенность
                        if session_context.primary_session.phase.value == "opening":
                            result.final_confidence *= (
                                0.9  # Снижаем уверенность на открытии
                            )
                        elif session_context.primary_session.phase.value == "closing":
                            result.final_confidence *= (
                                1.1  # Повышаем уверенность на закрытии
                            )

                        # Корректируем направление на основе типичного смещения сессии
                        if behavior.typical_direction_bias > 0.1:
                            if result.final_direction == "bullish":
                                result.final_confidence *= 1.1
                            elif result.final_direction == "bearish":
                                result.final_confidence *= 0.9
                        elif behavior.typical_direction_bias < -0.1:
                            if result.final_direction == "bearish":
                                result.final_confidence *= 1.1
                            elif result.final_direction == "bullish":
                                result.final_confidence *= 0.9

                        # Применяем модификаторы
                        result.session_confidence_boost = (
                            1.0 + (behavior.avg_volatility_multiplier - 1.0) * 0.2
                        )
                        result.session_aggressiveness_modifier = (
                            behavior.avg_volume_multiplier
                        )
                        result.session_position_multiplier = (
                            1.0 + (behavior.avg_volume_multiplier - 1.0) * 0.3
                        )

            return result

        except Exception as e:
            logger.error(
                f"Error getting prediction with session context for {symbol}: {e}"
            )
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики предиктора."""
        return {
            "predictor_stats": self.stats,
            "config": self.config,
            "total_predictions": self.stats["total_predictions"],
            "alignment_rate": (
                self.stats["session_aligned_predictions"]
                / self.stats["total_predictions"]
                if self.stats["total_predictions"] > 0
                else 0
            ),
            "high_confidence_rate": (
                self.stats["high_confidence_predictions"]
                / self.stats["total_predictions"]
                if self.stats["total_predictions"] > 0
                else 0
            ),
            "session_boost_rate": (
                self.stats["session_boosted_predictions"]
                / self.stats["total_predictions"]
                if self.stats["total_predictions"] > 0
                else 0
            ),
        }
