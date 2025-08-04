"""
Предиктор паттернов для торговой системы.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence

from domain.types.intelligence_types import PatternDetection
from domain.types.intelligence_types import PatternType
from domain.memory.pattern_memory import PatternMemory
from domain.value_objects.timestamp import Timestamp

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Уровни уверенности прогноза."""

    LOW = "low"  # 0.0 - 0.3
    MEDIUM = "medium"  # 0.3 - 0.7
    HIGH = "high"  # 0.7 - 1.0


@dataclass
class MarketFeatures:
    """Рыночные характеристики."""

    volatility: float
    volume: float
    price_change: float
    external_sync: bool
    market_regime: str


@dataclass
class PredictionRequest:
    """Запрос на прогнозирование."""

    symbol: str
    pattern_type: PatternType
    current_features: MarketFeatures
    confidence_threshold: float = 0.7
    min_similar_cases: int = 3
    max_lookback_days: int = 30

    # Дополнительные параметры
    market_regime: Optional[str] = None
    time_of_day: Optional[str] = None
    volatility_regime: Optional[str] = None


@dataclass
class PredictionResult:
    """Результат прогнозирования."""

    predicted_direction: str
    predicted_return_percent: float
    predicted_duration_minutes: int
    confidence: float
    similar_cases_count: int
    success_rate: float
    avg_return: float

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "predicted_direction": self.predicted_direction,
            "predicted_return_percent": self.predicted_return_percent,
            "predicted_duration_minutes": self.predicted_duration_minutes,
            "confidence": self.confidence,
            "similar_cases_count": self.similar_cases_count,
            "success_rate": self.success_rate,
            "avg_return": self.avg_return,
        }


@dataclass
class EnhancedPredictionResult:
    """Расширенный результат прогнозирования."""

    # Базовый прогноз
    prediction: PredictionResult

    # Дополнительная аналитика
    market_context: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    trading_recommendations: Dict[str, Any]

    # Метаданные
    prediction_timestamp: Timestamp
    data_quality_score: float
    model_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Преобразование в словарь."""
        return {
            "prediction": self.prediction.to_dict(),
            "market_context": self.market_context,
            "risk_assessment": self.risk_assessment,
            "trading_recommendations": self.trading_recommendations,
            "prediction_timestamp": self.prediction_timestamp.to_iso(),
            "data_quality_score": self.data_quality_score,
            "model_version": self.model_version,
        }


class PatternPredictor:
    """Предиктор паттернов."""

    def __init__(
        self, pattern_memory: PatternMemory, config: Optional[Dict[str, Any]] = None
    ):
        self.pattern_memory = pattern_memory
        self.config = config or {}

        # Кэш прогнозов
        self.prediction_cache: Dict[str, tuple[EnhancedPredictionResult, float]] = {}

        # Статистика
        self.prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "high_confidence_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # Настройки по умолчанию
        self.default_config = {
            "confidence_threshold": 0.7,
            "min_similar_cases": 3,
            "max_lookback_days": 30,
            "prediction_cache_ttl_seconds": 300,  # 5 минут
            "risk_assessment_enabled": True,
            "trading_recommendations_enabled": True,
        }

        # Обновляем конфигурацию
        self.config = {**self.default_config, **self.config}

    def predict_pattern_outcome(
        self,
        pattern_detection: PatternDetection,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnhancedPredictionResult]:
        """
        Прогнозирование исхода паттерна.

        Args:
            pattern_detection: Обнаруженный паттерн
            market_context: Контекст рынка

        Returns:
            Расширенный результат прогнозирования
        """
        try:
            # Генерируем ключ кэша
            cache_key = self._generate_cache_key(pattern_detection, market_context)

            # Проверяем кэш
            cached_result = self._get_cached_prediction(cache_key)
            if cached_result:
                self.prediction_stats["cache_hits"] = int(self.prediction_stats.get("cache_hits", 0)) + 1
                return cached_result

            self.prediction_stats["cache_misses"] = int(self.prediction_stats.get("cache_misses", 0)) + 1

            # Извлекаем характеристики из обнаружения
            features = self._extract_features_from_detection(pattern_detection)

            # Создаем запрос
            request = PredictionRequest(
                symbol=pattern_detection.symbol,
                pattern_type=pattern_detection.pattern_type,  # type: ignore
                current_features=features,
                confidence_threshold=self.config["confidence_threshold"],
                min_similar_cases=self.config["min_similar_cases"],
                max_lookback_days=self.config["max_lookback_days"],
            )

            # Выполняем прогнозирование
            result = self._execute_prediction(request, market_context)

            if result:
                # Кэшируем результат
                self._cache_prediction(cache_key, result)

                # Обновляем статистику
                self.prediction_stats["total_predictions"] = int(self.prediction_stats.get("total_predictions", 0)) + 1
                if result.prediction.confidence >= 0.8:
                    self.prediction_stats["high_confidence_predictions"] = int(self.prediction_stats.get("high_confidence_predictions", 0)) + 1

            return result

        except Exception as e:
            logger.error(f"Error predicting pattern outcome: {e}")
            return None

    def predict_with_custom_features(
        self,
        symbol: str,
        pattern_type: PatternType,
        features: MarketFeatures,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnhancedPredictionResult]:
        """
        Прогнозирование с пользовательскими характеристиками.

        Args:
            symbol: Символ
            pattern_type: Тип паттерна
            features: Рыночные характеристики
            market_context: Контекст рынка

        Returns:
            Расширенный результат прогнозирования
        """
        try:
            request = PredictionRequest(
                symbol=symbol,
                pattern_type=pattern_type,
                current_features=features,
                confidence_threshold=self.config["confidence_threshold"],
                min_similar_cases=self.config["min_similar_cases"],
                max_lookback_days=self.config["max_lookback_days"],
            )

            result = self._execute_prediction(request, market_context)

            if result:
                self.prediction_stats["total_predictions"] += 1
                if result.prediction.confidence >= 0.8:
                    self.prediction_stats["high_confidence_predictions"] += 1

            return result

        except Exception as e:
            logger.error(f"Error predicting with custom features: {e}")
            return None

    def _execute_prediction(
        self,
        request: PredictionRequest,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[EnhancedPredictionResult]:
        """Выполнение прогнозирования."""
        try:
            # Получаем похожие случаи из памяти
            similar_cases = self.pattern_memory.find_similar_patterns(
                symbol=request.symbol,
                pattern_type=request.pattern_type,
                features=request.current_features,
                max_lookback_days=request.max_lookback_days,
            )

            if len(similar_cases) < request.min_similar_cases:
                logger.warning(
                    f"Insufficient similar cases: {len(similar_cases)} < {request.min_similar_cases}"
                )
                return None

            # Анализируем контекст рынка
            market_context = self._analyze_market_context(request, market_context)

            # Создаем базовый прогноз
            prediction = self._create_prediction(similar_cases, request)

            # Проверяем порог уверенности
            if prediction.confidence < request.confidence_threshold:
                logger.info(
                    f"Prediction confidence {prediction.confidence:.3f} below threshold {request.confidence_threshold}"
                )
                return None

            # Оценка риска
            risk_assessment = {}
            if self.config["risk_assessment_enabled"]:
                risk_assessment = self._assess_risk(prediction, request)

            # Торговые рекомендации
            trading_recommendations = {}
            if self.config["trading_recommendations_enabled"]:
                trading_recommendations = self._generate_trading_recommendations(
                    prediction, request, market_context
                )

            # Оценка качества данных
            data_quality_score = self._calculate_data_quality_score(prediction)

            # Создаем расширенный результат
            result = EnhancedPredictionResult(
                prediction=prediction,
                market_context=market_context,
                risk_assessment=risk_assessment,
                trading_recommendations=trading_recommendations,
                prediction_timestamp=Timestamp(datetime.now()),
                data_quality_score=data_quality_score,
            )

            return result

        except Exception as e:
            logger.error(f"Error executing prediction: {e}")
            return None

    def _extract_features_from_detection(
        self, pattern_detection: PatternDetection
    ) -> MarketFeatures:
        """Извлечение характеристик из обнаружения паттерна."""
        try:
            # Извлекаем базовые характеристики
            volatility = getattr(pattern_detection, "volatility", 0.02)
            volume = getattr(pattern_detection, "volume", 1.0)
            price_change = getattr(pattern_detection, "price_change", 0.0)
            external_sync = getattr(pattern_detection, "external_sync", False)
            market_regime = getattr(pattern_detection, "market_regime", "normal")

            return MarketFeatures(
                volatility=float(volatility),
                volume=float(volume),
                price_change=float(price_change),
                external_sync=bool(external_sync),
                market_regime=str(market_regime),
            )

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return MarketFeatures(
                volatility=0.02,
                volume=1.0,
                price_change=0.0,
                external_sync=False,
                market_regime="normal",
            )

    def _create_prediction(
        self, similar_cases: List[Any], request: PredictionRequest
    ) -> PredictionResult:
        """Создание прогноза на основе похожих случаев."""
        try:
            # Анализируем результаты похожих случаев
            directions = []
            returns = []
            durations = []
            successful_cases = 0

            for case in similar_cases:
                if hasattr(case, "outcome"):
                    outcome = case.outcome
                    if hasattr(outcome, "direction"):
                        directions.append(outcome.direction)
                    if hasattr(outcome, "return_percent"):
                        returns.append(outcome.return_percent)
                    if hasattr(outcome, "duration_minutes"):
                        durations.append(outcome.duration_minutes)
                    if hasattr(outcome, "is_successful") and outcome.is_successful:
                        successful_cases += 1

            if not directions:
                # Возвращаем нейтральный прогноз
                return PredictionResult(
                    predicted_direction="neutral",
                    predicted_return_percent=0.0,
                    predicted_duration_minutes=30,
                    confidence=0.0,
                    similar_cases_count=len(similar_cases),
                    success_rate=0.0,
                    avg_return=0.0,
                )

            # Определяем направление (большинство голосов)
            direction_counts: Dict[str, int] = {}
            for direction in directions:
                direction_counts[direction] = direction_counts.get(direction, 0) + 1

            predicted_direction = max(direction_counts, key=lambda k: direction_counts[k])

            # Рассчитываем средние значения
            avg_return = sum(returns) / len(returns) if returns else 0.0
            avg_duration = sum(durations) / len(durations) if durations else 30

            # Уверенность на основе согласованности
            direction_consistency = direction_counts[predicted_direction] / len(directions)
            success_rate = successful_cases / len(similar_cases) if similar_cases else 0.0

            # Общая уверенность
            confidence = (direction_consistency + success_rate) / 2

            return PredictionResult(
                predicted_direction=predicted_direction,
                predicted_return_percent=float(avg_return),
                predicted_duration_minutes=int(avg_duration),
                confidence=float(confidence),
                similar_cases_count=len(similar_cases),
                success_rate=float(success_rate),
                avg_return=float(avg_return),
            )

        except Exception as e:
            logger.error(f"Error creating prediction: {e}")
            return PredictionResult(
                predicted_direction="neutral",
                predicted_return_percent=0.0,
                predicted_duration_minutes=30,
                confidence=0.0,
                similar_cases_count=len(similar_cases),
                success_rate=0.0,
                avg_return=0.0,
            )

    def _analyze_market_context(
        self,
        request: PredictionRequest,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Анализ контекста рынка."""
        try:
            context = market_context or {}

            # Добавляем характеристики из запроса
            context.update({
                "symbol": request.symbol,
                "pattern_type": request.pattern_type.value,
                "volatility": request.current_features.volatility,
                "market_regime": request.current_features.market_regime,
                "external_sync": request.current_features.external_sync,
            })

            # Определяем режим рынка
            if request.current_features.volatility > 0.03:
                context["market_regime"] = "volatile"
            elif request.current_features.volatility < 0.01:
                context["market_regime"] = "stable"

            # Условия ликвидности
            if request.current_features.volume < 0.5:
                context["liquidity_conditions"] = "poor"
            else:
                context["liquidity_conditions"] = "good"

            return context

        except Exception as e:
            logger.error(f"Error analyzing market context: {e}")
            return {"error": str(e)}

    def _assess_risk(
        self, prediction: PredictionResult, request: PredictionRequest
    ) -> Dict[str, Any]:
        """Оценка риска прогноза."""
        try:
            risk_assessment = {
                "overall_risk_level": "medium",
                "confidence_risk": "medium",
                "data_quality_risk": "medium",
                "market_risk": "medium",
                "specific_risks": [],
            }

            # Риск уверенности
            if prediction.confidence < 0.5:
                risk_assessment["confidence_risk"] = "high"
                risk_assessment["specific_risks"].append("low_confidence")  # type: ignore
            elif prediction.confidence > 0.8:
                risk_assessment["confidence_risk"] = "low"

            # Риск качества данных
            if prediction.similar_cases_count < 5:
                risk_assessment["data_quality_risk"] = "high"
                risk_assessment["specific_risks"].append("insufficient_data")  # type: ignore

            if prediction.success_rate < 0.6:
                risk_assessment["data_quality_risk"] = "high"
                risk_assessment["specific_risks"].append("low_success_rate")  # type: ignore

            # Рыночный риск
            features = request.current_features
            if features.volatility > 0.03:
                risk_assessment["market_risk"] = "high"
                risk_assessment["specific_risks"].append("high_volatility")  # type: ignore

            if features.external_sync:
                risk_assessment["market_risk"] = "high"
                risk_assessment["specific_risks"].append("external_sync")  # type: ignore

            # Общий уровень риска
            risk_factors = [
                risk_assessment["confidence_risk"],
                risk_assessment["data_quality_risk"],
                risk_assessment["market_risk"],
            ]

            high_risk_count = risk_factors.count("high")
            if high_risk_count >= 2:
                risk_assessment["overall_risk_level"] = "high"
            elif high_risk_count == 0:
                risk_assessment["overall_risk_level"] = "low"

            return risk_assessment

        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return {"overall_risk_level": "unknown", "error": str(e)}

    def _generate_trading_recommendations(
        self,
        prediction: PredictionResult,
        request: PredictionRequest,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Генерация торговых рекомендаций."""
        try:
            recommendations = {
                "should_trade": False,
                "recommended_action": "hold",
                "position_size": "normal",
                "entry_timing": "immediate",
                "stop_loss_percent": 0.0,
                "take_profit_percent": 0.0,
                "confidence_boost": 1.0,
                "risk_adjustment": 1.0,
                "reasoning": [],
            }

            # Базовые условия для торговли
            if (
                prediction.confidence >= 0.7
                and prediction.similar_cases_count >= 5
                and prediction.success_rate >= 0.6
            ):

                recommendations["should_trade"] = True
                recommendations["recommended_action"] = prediction.predicted_direction

                # Размер позиции
                if prediction.confidence > 0.9 and prediction.success_rate > 0.8:
                    recommendations["position_size"] = "large"
                elif prediction.confidence < 0.8:
                    recommendations["position_size"] = "small"

                # Время входа
                if prediction.predicted_duration_minutes < 10:
                    recommendations["entry_timing"] = "immediate"
                else:
                    recommendations["entry_timing"] = "gradual"

                # Стоп-лосс и тейк-профит
                recommendations["stop_loss_percent"] = abs(prediction.avg_return) * 0.5
                recommendations["take_profit_percent"] = (
                    abs(prediction.avg_return) * 1.5
                )

                # Корректировки уверенности и риска
                recommendations["confidence_boost"] = prediction.confidence
                recommendations["risk_adjustment"] = 1.0 / prediction.confidence

                # Обоснование
                recommendations["reasoning"].append(  # type: ignore
                    f"High confidence: {prediction.confidence:.3f}"
                )
                recommendations["reasoning"].append(  # type: ignore
                    f"Success rate: {prediction.success_rate:.3f}"
                )
                recommendations["reasoning"].append(  # type: ignore
                    f"Similar cases: {prediction.similar_cases_count}"
                )

            else:
                recommendations["reasoning"].append(  # type: ignore
                    "Insufficient confidence or data quality"
                )

            # Дополнительные корректировки на основе контекста
            if market_context:
                if market_context.get("market_regime") == "volatile":
                    recommendations["position_size"] = "small"
                    recommendations["reasoning"].append(  # type: ignore
                        "High volatility - reducing position size"
                    )

                if market_context.get("liquidity_conditions") == "poor":
                    recommendations["entry_timing"] = "gradual"
                    recommendations["reasoning"].append(  # type: ignore
                        "Poor liquidity - gradual entry recommended"
                    )

            return recommendations

        except Exception as e:
            logger.error(f"Error generating trading recommendations: {e}")
            return {"should_trade": False, "error": str(e)}

    def _calculate_data_quality_score(self, prediction: PredictionResult) -> float:
        """Расчет оценки качества данных."""
        try:
            # Факторы качества
            confidence_factor = prediction.confidence
            cases_factor = min(1.0, prediction.similar_cases_count / 10.0)
            success_factor = prediction.success_rate

            # Взвешенная оценка
            quality_score = (
                0.4 * confidence_factor + 0.3 * cases_factor + 0.3 * success_factor
            )

            return float(quality_score)

        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            return 0.0

    def _generate_cache_key(
        self,
        pattern_detection: PatternDetection,
        market_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Генерация ключа кэша."""
        try:
            # Базовый ключ
            base_key = (
                f"{pattern_detection.symbol}_{pattern_detection.pattern_type.value}"
            )

            # Добавляем контекст если есть
            if market_context:
                context_hash = hash(str(sorted(market_context.items())))
                base_key += f"_ctx_{context_hash}"

            return base_key

        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return f"{pattern_detection.symbol}_{pattern_detection.pattern_type.value}"

    def _get_cached_prediction(
        self, cache_key: str
    ) -> Optional[EnhancedPredictionResult]:
        """Получение прогноза из кэша."""
        try:
            if cache_key in self.prediction_cache:
                result, timestamp = self.prediction_cache[cache_key]

                # Проверяем TTL
                if (datetime.now().timestamp() - timestamp) < self.config[
                    "prediction_cache_ttl_seconds"
                ]:
                    return result
                else:
                    # Удаляем устаревший кэш
                    del self.prediction_cache[cache_key]

            return None

        except Exception as e:
            logger.error(f"Error getting cached prediction: {e}")
            return None

    def _cache_prediction(
        self, cache_key: str, result: EnhancedPredictionResult
    ) -> None:
        """Кэширование прогноза."""
        try:
            # Ограничиваем размер кэша
            if len(self.prediction_cache) > 100:
                # Удаляем самый старый элемент
                oldest_key = min(
                    self.prediction_cache.keys(),
                    key=lambda k: self.prediction_cache[k][1],
                )
                del self.prediction_cache[oldest_key]

            self.prediction_cache[cache_key] = (result, datetime.now().timestamp())

        except Exception as e:
            logger.error(f"Error caching prediction: {e}")

    def clear_cache(self) -> None:
        """Очистка кэша прогнозов."""
        try:
            self.prediction_cache.clear()
            logger.info("Prediction cache cleared")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики предиктора."""
        total_predictions = self.prediction_stats.get("total_predictions", 0)
        successful_predictions = self.prediction_stats.get("successful_predictions", 0)
        high_confidence_predictions = self.prediction_stats.get("high_confidence_predictions", 0)
        cache_hits = self.prediction_stats.get("cache_hits", 0)
        cache_misses = self.prediction_stats.get("cache_misses", 0)

        # Расчет метрик
        success_rate = (
            successful_predictions / total_predictions if total_predictions > 0 else 0.0
        )
        high_confidence_rate = (
            high_confidence_predictions / total_predictions if total_predictions > 0 else 0.0
        )
        cache_hit_rate = (
            cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0.0
        )

        return {
            "total_predictions": total_predictions,
            "successful_predictions": successful_predictions,
            "high_confidence_predictions": high_confidence_predictions,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "success_rate": success_rate,
            "high_confidence_rate": high_confidence_rate,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.prediction_cache),
            "config": self.config,
        }

    def validate_prediction_accuracy(
        self, prediction: EnhancedPredictionResult, actual_outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Валидация точности прогноза.

        Args:
            prediction: Прогноз
            actual_outcome: Фактический исход

        Returns:
            Метрики точности
        """
        try:
            # Извлекаем фактические данные
            actual_direction = actual_outcome.get("direction", "neutral")
            actual_return = actual_outcome.get("return_percent", 0.0)
            actual_duration = actual_outcome.get("duration_minutes", 0)

            # Сравниваем направления
            direction_correct = (
                prediction.prediction.predicted_direction == actual_direction
            )

            # Ошибка прогноза возврата
            return_error = abs(
                prediction.prediction.predicted_return_percent - actual_return
            )

            # Ошибка прогноза длительности
            duration_error = abs(
                prediction.prediction.predicted_duration_minutes - actual_duration
            )

            # Определяем успешность
            is_successful = (
                direction_correct
                and return_error < 1.0  # Ошибка менее 1%
                and duration_error < 5  # Ошибка менее 5 минут
            )

            if is_successful:
                self.prediction_stats["successful_predictions"] += 1

            accuracy_metrics = {
                "direction_correct": direction_correct,
                "return_error_percent": return_error,
                "duration_error_minutes": duration_error,
                "is_successful": is_successful,
                "prediction_confidence": prediction.prediction.confidence,
                "actual_outcome": actual_outcome,
            }

            return accuracy_metrics

        except Exception as e:
            logger.error(f"Error validating prediction accuracy: {e}")
            return {"error": str(e)}
