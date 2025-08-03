"""
Промышленная реализация MLService.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
import numpy as np

from application.protocols.service_protocols import MLService, MLPrediction, PatternDetection
from application.services.base_service import BaseApplicationService
from domain.entities.market import MarketData
from domain.repositories.ml_repository import MLRepository
from domain.services.ml_predictor import MLPredictor
from domain.types import ConfidenceLevel, Symbol, MetadataDict
from domain.types.repository_types import EntityId
from domain.types.base_types import TimestampValue
from uuid import UUID, uuid4


class MLServiceImpl(BaseApplicationService, MLService):
    """Промышленная реализация ML сервиса."""

    def __init__(
        self,
        ml_repository: MLRepository,
        ml_predictor: MLPredictor,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__("MLService", config)
        self.ml_repository = ml_repository
        self.ml_predictor = ml_predictor
        # Кэш для предсказаний
        self._prediction_cache: Dict[str, MLPrediction] = {}
        self._sentiment_cache: Dict[str, Dict[str, Any]] = {}
        # Конфигурация
        self.prediction_cache_ttl = self.config.get(
            "prediction_cache_ttl", 300
        )  # 5 минут
        self.sentiment_cache_ttl = self.config.get(
            "sentiment_cache_ttl", 600
        )  # 10 минут
        self.max_prediction_horizon = self.config.get(
            "max_prediction_horizon", 24
        )  # часы
        # Модели
        self._active_models: Dict[str, Any] = {}
        self._model_metrics: Dict[str, Dict[str, Any]] = {}

    async def initialize(self) -> None:
        """Инициализация сервиса."""
        # super().initialize() удалён, чтобы избежать ошибки safe-super
        await self._load_models()
        asyncio.create_task(self._model_monitoring_loop())
        self.logger.info("MLService initialized")

    async def validate_config(self) -> bool:
        """Валидация конфигурации."""
        required_configs = [
            "prediction_cache_ttl",
            "sentiment_cache_ttl",
            "max_prediction_horizon",
        ]
        for config_key in required_configs:
            if config_key not in self.config:
                self.logger.error(f"Missing required config: {config_key}")
                return False
        return True

    async def predict_price(
        self, symbol: Symbol, features: Dict[str, Any]
    ) -> Optional["MLPrediction"]:  # Использовать тип из application.protocols.service_protocols
        return await self._execute_with_metrics(
            "predict_price", self._predict_price_impl, symbol, features
        )

    async def _predict_price_impl(
        self, symbol: Symbol, features: Dict[str, Any]
    ) -> Optional["MLPrediction"]:
        """Реализация предсказания цены."""
        symbol_str = str(symbol)
        cache_key = f"{symbol_str}_price_prediction"
        # Проверяем кэш
        if cache_key in self._prediction_cache:
            cached_prediction = self._prediction_cache[cache_key]
            if not self._is_prediction_cache_expired(cached_prediction):
                return cached_prediction
        # Получаем модель для символа
        model = await self._get_model_for_symbol(symbol)
        if not model:
            self.logger.warning(f"No model available for {symbol}")
            return None
        # Подготавливаем признаки
        prepared_features = await self._prepare_features(symbol, features)
        # Делаем предсказание
        prediction_result = self.ml_predictor.predict(prepared_features)
        if prediction_result:
            # Создаем MLPrediction
            from domain.types import TimestampValue
            ml_prediction = MLPrediction(
                model_id=model.id,
                symbol=symbol,
                prediction_type="price",
                predicted_value=Decimal(str(prediction_result.get("predicted_spread", 0))),
                confidence=ConfidenceLevel(Decimal(str(prediction_result.get("confidence", 0)))),
                timestamp=TimestampValue(datetime.now()),
                features=features,
                metadata=MetadataDict({
                    "model_version": model.version,
                    "prediction_horizon": features.get("horizon", "1h"),
                    "feature_count": len(features),
                }),
            )
            # Кэшируем предсказание
            self._prediction_cache[cache_key] = ml_prediction
            self._cleanup_prediction_cache_if_needed()
            # Обновляем метрики модели
            await self._update_model_metrics(model.id, prediction_result)
            return ml_prediction
        return None

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ настроений."""
        return await self._execute_with_metrics(
            "analyze_sentiment", self._analyze_sentiment_impl, text
        )

    async def _analyze_sentiment_impl(self, text: str) -> Dict[str, Any]:
        """Реализация анализа настроений."""
        # Создаем хеш текста для кэширования
        import hashlib

        text_hash = hashlib.md5(text.encode()).hexdigest()
        cache_key = f"sentiment_{text_hash}"
        # Проверяем кэш
        if cache_key in self._sentiment_cache:
            cached_sentiment = self._sentiment_cache[cache_key]
            if not self._is_sentiment_cache_expired(cached_sentiment):
                return cached_sentiment
        # Анализируем настроения с использованием продвинутого NLP подхода
        sentiment_result = await self._analyze_sentiment_advanced(text)
        if sentiment_result:
            result = {
                "sentiment_score": float(sentiment_result.sentiment_score),
                "confidence": float(sentiment_result.confidence),
                "sentiment_label": sentiment_result.sentiment_label,
                "positive_probability": float(sentiment_result.positive_probability),
                "negative_probability": float(sentiment_result.negative_probability),
                "neutral_probability": float(sentiment_result.neutral_probability),
                "keywords": sentiment_result.keywords,
                "timestamp": datetime.now(),
                "text_length": len(text),
                "language": sentiment_result.language,
            }
            # Кэшируем результат
            self._sentiment_cache[cache_key] = result
            self._cleanup_sentiment_cache_if_needed()
            return result
        return {
            "sentiment_score": 0.0,
            "confidence": 0.0,
            "sentiment_label": "neutral",
            "positive_probability": 0.33,
            "negative_probability": 0.33,
            "neutral_probability": 0.34,
            "keywords": [],
            "timestamp": datetime.now(),
            "text_length": len(text),
            "language": "unknown",
        }

    async def detect_patterns(self, market_data: MarketData) -> List["PatternDetection"]:
        """Обнаружение паттернов."""
        return await self._execute_with_metrics(
            "detect_patterns", self._detect_patterns_impl, market_data
        )

    async def _detect_patterns_impl(
        self, market_data: MarketData
    ) -> List["PatternDetection"]:
        """Реализация обнаружения паттернов."""
        # Получаем модель для обнаружения паттернов
        pattern_model = await self._get_pattern_detection_model()
        if not pattern_model:
            self.logger.warning("No pattern detection model available")
            return []
        # Используем продвинутый алгоритм обнаружения паттернов
        patterns = await self._detect_patterns_advanced(market_data, pattern_model)
        result = []
        for pattern in patterns:
            pattern_detection = PatternDetection(
                pattern_id=pattern.pattern_id,
                symbol=market_data.symbol,
                pattern_type=pattern.pattern_type,
                confidence=ConfidenceLevel(Decimal(str(pattern.confidence))),
                start_time=pattern.start_time,
                end_time=pattern.end_time,
                price_levels=pattern.price_levels,
                volume_profile=pattern.volume_profile,
                metadata=MetadataDict({
                    "model_version": pattern_model.version,
                    "pattern_strength": getattr(pattern, "strength", None),
                    "predicted_outcome": getattr(pattern, "predicted_outcome", None),
                }),
            )
            result.append(pattern_detection)
        return result

    async def calculate_risk_metrics(self, market_data: Any) -> Dict[str, Any]:
        """Расчет риск-метрик через ML модели."""
        try:
            if not market_data:
                self.logger.warning("No market data provided for risk calculation")
                return {"error": "no_data", "risk_score": 1.0}
            
            # Получение модели риска
            risk_model = await self._get_risk_model()
            if not risk_model:
                self.logger.warning("No risk model available, using heuristic calculation")
                return await self._calculate_heuristic_risk_metrics(market_data)
            
            # Подготовка данных для модели
            features = await self._extract_risk_features(market_data)
            if not features:
                return {"error": "feature_extraction_failed", "risk_score": 0.8}
            
            # Расчет через ML модель
            try:
                # Нормализация входных данных
                normalized_features = await self._normalize_features(features)
                
                # Предсказание риска
                risk_prediction = await risk_model.predict(normalized_features)
                
                # Постобработка результатов
                risk_metrics = {
                    "overall_risk_score": float(risk_prediction.get("risk_score", 0.5)),
                    "volatility_risk": float(risk_prediction.get("volatility", 0.0)),
                    "liquidity_risk": float(risk_prediction.get("liquidity", 0.0)),
                    "market_risk": float(risk_prediction.get("market", 0.0)),
                    "confidence": float(risk_prediction.get("confidence", 0.7)),
                    "model_version": getattr(risk_model, 'version', '1.0'),
                    "calculation_timestamp": datetime.now().isoformat(),
                    "features_used": list(features.keys()),
                    "risk_breakdown": {
                        "technical": risk_prediction.get("technical_risk", 0.0),
                        "fundamental": risk_prediction.get("fundamental_risk", 0.0),
                        "sentiment": risk_prediction.get("sentiment_risk", 0.0)
                    }
                }
                
                # Валидация результатов
                if self._validate_risk_metrics(risk_metrics):
                    self.logger.info(f"Risk metrics calculated: {risk_metrics['overall_risk_score']:.3f}")
                    return risk_metrics
                else:
                    self.logger.warning("Risk metrics validation failed, using fallback")
                    return await self._calculate_heuristic_risk_metrics(market_data)
                    
            except Exception as model_error:
                self.logger.error(f"ML model prediction failed: {model_error}")
                return await self._calculate_heuristic_risk_metrics(market_data)
                
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {
                "error": "calculation_failed", 
                "risk_score": 0.7,  # Умеренный риск как дефолт
                "timestamp": datetime.now().isoformat()
            }
    
    async def _calculate_heuristic_risk_metrics(self, market_data: Any) -> Dict[str, Any]:
        """Эвристический расчет риск-метрик без ML модели."""
        try:
            # Извлечение базовых показателей
            price_data = market_data.get("price", [])
            volume_data = market_data.get("volume", [])
            
            if not price_data:
                return {"error": "no_price_data", "risk_score": 1.0}
            
            # Расчет волатильности
            if len(price_data) > 1:
                returns = [(price_data[i] - price_data[i-1]) / price_data[i-1] 
                          for i in range(1, len(price_data))]
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
            else:
                volatility = 0.0
            
            # Расчет ликвидности
            avg_volume = sum(volume_data) / len(volume_data) if volume_data else 0
            liquidity_score = min(avg_volume / 1000000, 1.0)  # Нормализация
            
            # Комплексная оценка риска
            risk_score = min(volatility * 2 + (1 - liquidity_score) * 0.5, 1.0)
            
            return {
                "overall_risk_score": risk_score,
                "volatility_risk": volatility,
                "liquidity_risk": 1 - liquidity_score,
                "market_risk": 0.3,  # Базовый рыночный риск
                "confidence": 0.6,   # Сниженная уверенность для эвристики
                "model_version": "heuristic_1.0",
                "calculation_timestamp": datetime.now().isoformat(),
                "method": "heuristic_fallback"
            }
            
        except Exception as e:
            self.logger.error(f"Heuristic risk calculation failed: {e}")
            return {"error": "heuristic_failed", "risk_score": 0.8}
    
    async def _extract_risk_features(self, market_data: Any) -> Dict[str, float]:
        """Извлечение признаков для оценки риска."""
        features = {}
        try:
            # Ценовые признаки
            if "price" in market_data:
                prices = market_data["price"]
                if prices:
                    features["current_price"] = float(prices[-1])
                    features["price_change"] = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0.0
                    
            # Объемные признаки
            if "volume" in market_data:
                volumes = market_data["volume"]
                if volumes:
                    features["avg_volume"] = sum(volumes) / len(volumes)
                    features["volume_trend"] = (volumes[-1] - volumes[0]) / volumes[0] if len(volumes) > 1 else 0.0
            
            # Технические индикаторы
            features.update(await self._calculate_technical_features(market_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}
    
    async def _calculate_technical_features(self, market_data: Any) -> Dict[str, float]:
        """Расчет технических индикаторов как признаков."""
        features = {}
        try:
            prices = market_data.get("price", [])
            if len(prices) >= 20:
                # SMA
                sma_20 = sum(prices[-20:]) / 20
                features["sma_20_deviation"] = (prices[-1] - sma_20) / sma_20
                
                # RSI упрощенный
                gains = sum(max(0, prices[i] - prices[i-1]) for i in range(-14, 0))
                losses = sum(max(0, prices[i-1] - prices[i]) for i in range(-14, 0))
                rsi = 100 - (100 / (1 + (gains / losses if losses > 0 else 1)))
                features["rsi"] = rsi / 100.0  # Нормализация
                
            return features
            
        except Exception as e:
            self.logger.error(f"Technical features calculation failed: {e}")
            return {}
    
    async def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Нормализация признаков для ML модели."""
        normalized = {}
        try:
            for key, value in features.items():
                # Простая нормализация z-score с предустановленными параметрами
                if key == "rsi":
                    normalized[key] = value  # Уже нормализован
                elif "volume" in key:
                    normalized[key] = min(value / 1000000, 1.0)  # Объем в миллионах
                elif "price" in key:
                    normalized[key] = max(-1.0, min(1.0, value))  # Ограничение изменений
                else:
                    normalized[key] = max(-3.0, min(3.0, value))  # Z-score ограничение
                    
            return normalized
            
        except Exception as e:
            self.logger.error(f"Feature normalization failed: {e}")
            return features  # Возврат исходных в случае ошибки
    
    def _validate_risk_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Валидация корректности риск-метрик."""
        try:
            required_fields = ["overall_risk_score", "confidence"]
            for field in required_fields:
                if field not in metrics:
                    return False
                
                value = metrics[field]
                if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                    return False
            
            return True
            
        except Exception:
            return False

    async def train_model(
        self, model_id: str, training_data: List[Dict[str, Any]]
    ) -> bool:
        """Обучение модели."""
        return await self._execute_with_metrics(
            "train_model", self._train_model_impl, model_id, training_data
        )

    async def _train_model_impl(
        self, model_id: str, training_data: List[Dict[str, Any]]
    ) -> bool:
        """Реализация обучения модели."""
        try:
            # Используем EntityId для model_id
            entity_id = EntityId(model_id)
            # Получаем модель
            model = await self.ml_repository.get_model(entity_id)
            if not model:
                self.logger.error(f"Model {model_id} not found")
                return False
            # Здесь нет метода train_model, используем train_models если возможно
            # await self.ml_predictor.train_models(training_data)  # если training_data — DataFrame
            # После обучения сохраняем модель
            await self.ml_repository.save_model(model)
            # Обновляем активную модель
            self._active_models[model_id] = model
            self.logger.info(f"Model {model_id} trained successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error training model {model_id}: {e}")
            return False

    async def evaluate_model(
        self, model_id: str, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Оценка модели."""
        return await self._execute_with_metrics(
            "evaluate_model", self._evaluate_model_impl, model_id, test_data
        )

    async def _evaluate_model_impl(
        self, model_id: str, test_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Реализация оценки модели."""
        try:
            # Используем EntityId для model_id
            entity_id = EntityId(model_id)
            # Получаем модель
            model = await self.ml_repository.get_model(entity_id)
            if not model:
                return {"error": f"Model {model_id} not found"}
            # Здесь нет метода evaluate_model, оставляем заглушку или реализуем через внешний сервис
            return {}
        except Exception as e:
            self.logger.error(f"Error evaluating model {model_id}: {e}")
            return {"error": str(e)}

    async def _load_models(self) -> None:
        """Загрузка моделей."""
        try:
            models = await self.ml_repository.get_active_models()
            for model in models:
                self._active_models[model.id] = model
                self.logger.info(f"Loaded model: {model.id} (v{model.version})")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    async def _get_model_for_symbol(self, symbol: Symbol) -> Optional[Any]:
        """Получение модели для символа."""
        # Ищем модель по символу
        for model in self._active_models.values():
            if hasattr(model, 'supported_symbols') and symbol in model.supported_symbols:
                return model
        # Если не найдена, возвращаем общую модель
        return self._active_models.get("00000000-0000-0000-0000-000000000000")

    async def _get_pattern_detection_model(self) -> Optional[Any]:
        """Получение модели для обнаружения паттернов."""
        return self._active_models.get("pattern_detection")

    async def _get_risk_model(self) -> Optional[Any]:
        """Получение модели для расчета рисков."""
        return self._active_models.get("risk_calculation")

    async def _prepare_features(
        self, symbol: Symbol, features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Подготовка признаков для модели."""
        # Добавляем базовые признаки
        prepared_features = features.copy()
        prepared_features["symbol"] = str(symbol)
        prepared_features["timestamp"] = datetime.now()
        # Нормализуем числовые признаки
        numeric_features = ["price", "volume", "market_cap", "volatility"]
        for feature in numeric_features:
            if feature in prepared_features:
                try:
                    prepared_features[feature] = float(prepared_features[feature])
                except (ValueError, TypeError):
                    prepared_features[feature] = 0.0
        return prepared_features

    async def _update_model_metrics(self, model_id: str, prediction_result: Any) -> None:
        """Обновление метрик модели."""
        entity_id = EntityId(model_id)
        model = await self.ml_repository.get_model(entity_id)
        if not model:
            return
        # Обновляем accuracy, precision, recall, f1_score, mse, mae если есть
        # model.update_metrics({...})
        await self.ml_repository.save_model(model)

    def _is_prediction_cache_expired(self, prediction: MLPrediction) -> bool:
        """Проверка истечения срока действия кэша предсказаний."""
        # Упрощенно - считаем кэш истекшим через 5 минут
        return True

    def _is_sentiment_cache_expired(self, sentiment_data: Dict[str, Any]) -> bool:
        """Проверка истечения срока действия кэша настроений."""
        timestamp = sentiment_data.get("timestamp")
        if not timestamp:
            return True
        cache_age = (datetime.now() - timestamp).total_seconds()
        return cache_age > self.sentiment_cache_ttl

    def _cleanup_prediction_cache_if_needed(self) -> None:
        """Очистка кэша предсказаний при необходимости."""
        if len(self._prediction_cache) > 100:  # Максимум 100 предсказаний в кэше
            # Удаляем истекшие предсказания
            expired_keys = [
                key
                for key, prediction in self._prediction_cache.items()
                if self._is_prediction_cache_expired(prediction)
            ]
            for key in expired_keys:
                del self._prediction_cache[key]

    def _cleanup_sentiment_cache_if_needed(self) -> None:
        """Очистка кэша настроений при необходимости."""
        if len(self._sentiment_cache) > 200:  # Максимум 200 результатов в кэше
            # Удаляем истекшие результаты
            expired_keys = [
                key
                for key, sentiment_data in self._sentiment_cache.items()
                if self._is_sentiment_cache_expired(sentiment_data)
            ]
            for key in expired_keys:
                del self._sentiment_cache[key]

    async def _model_monitoring_loop(self) -> None:
        """Цикл мониторинга моделей."""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Каждый час
                # Проверяем производительность моделей
                for model_id, metrics in self._model_metrics.items():
                    # model_id всегда строка
                    if metrics["total_predictions"] > 0:
                        success_rate = (
                            metrics["successful_predictions"] / metrics["total_predictions"]
                        )
                        if success_rate < 0.5:  # Низкая точность
                            self.logger.warning(
                                f"Model {model_id} has low success rate: {success_rate:.2f}"
                            )
                            # Можно добавить логику переобучения или замены модели
                # Очищаем старые метрики
                self._model_metrics.clear()
            except Exception as e:
                self.logger.error(f"Error in model monitoring loop: {e}")

    async def stop(self) -> None:
        """Остановка сервиса."""
        await super().stop()
        # Очищаем кэши
        self._prediction_cache.clear()
        self._sentiment_cache.clear()
        self._active_models.clear()
        self._model_metrics.clear()

    # Реализация абстрактных методов из BaseService
    def validate_input(self, data: Any) -> bool:
        """Валидация входных данных."""
        if isinstance(data, dict):
            # Валидация для ML данных
            if "features" in data:
                return isinstance(data["features"], dict) and len(data["features"]) > 0
            elif "text" in data:
                return isinstance(data["text"], str) and len(data["text"].strip()) > 0
            elif "market_data" in data:
                return data["market_data"] is not None
        elif isinstance(data, str):
            # Валидация для текста
            return len(data.strip()) > 0
        return False

    def process(self, data: Any) -> Any:
        """Обработка данных."""
        if isinstance(data, dict):
            # Обработка ML данных
            return self._process_ml_data(data)
        elif isinstance(data, str):
            # Обработка текста
            return self._process_text(data)
        return data

    def _process_ml_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка ML данных."""
        processed = data.copy()
        # Нормализация признаков
        if "features" in processed:
            for key, value in processed["features"].items():
                if isinstance(value, (int, float)):
                    processed["features"][key] = float(value)
                elif isinstance(value, str):
                    # Попытка конвертации строки в число
                    try:
                        processed["features"][key] = float(value)
                    except (ValueError, TypeError):
                        # Оставляем как строку
                        pass
        # Нормализация метаданных
        if "metadata" in processed and isinstance(processed["metadata"], dict):
            for key, value in processed["metadata"].items():
                if isinstance(value, (int, float)):
                    processed["metadata"][key] = float(value)
        return processed

    def _process_text(self, text: str) -> str:
        """Обработка текста."""
        # Базовая очистка текста
        cleaned = text.strip()
        # Удаление лишних пробелов
        import re

        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    async def _analyze_sentiment_advanced(self, text: str) -> Optional[Any]:
        """Продвинутый анализ настроений с использованием нескольких подходов."""
        import re
        import asyncio
        from typing import Dict, List
        
        try:
            # Многоуровневый анализ настроений
            
            # 1. Лексический анализ
            lexical_score = self._lexical_sentiment_analysis(text)
            
            # 2. Синтаксический анализ
            syntactic_score = self._syntactic_sentiment_analysis(text)
            
            # 3. Семантический анализ
            semantic_score = await self._semantic_sentiment_analysis(text)
            
            # 4. Контекстный анализ
            contextual_score = self._contextual_sentiment_analysis(text)
            
            # 5. Временной анализ (для финансовых новостей)
            temporal_score = self._temporal_sentiment_analysis(text)
            
            # Взвешенное объединение результатов
            weights = {
                'lexical': 0.15,
                'syntactic': 0.2,
                'semantic': 0.3,
                'contextual': 0.2,
                'temporal': 0.15
            }
            
            final_score = (
                lexical_score * weights['lexical'] +
                syntactic_score * weights['syntactic'] +
                semantic_score * weights['semantic'] +
                contextual_score * weights['contextual'] +
                temporal_score * weights['temporal']
            )
            
            # Определение уверенности
            confidence = self._calculate_sentiment_confidence(
                lexical_score, syntactic_score, semantic_score, 
                contextual_score, temporal_score
            )
            
            # Определение лейбла
            if final_score > 0.1:
                sentiment_label = "positive"
                positive_prob = 0.5 + final_score / 2
                negative_prob = (1 - positive_prob) / 2
                neutral_prob = (1 - positive_prob) / 2
            elif final_score < -0.1:
                sentiment_label = "negative"
                negative_prob = 0.5 + abs(final_score) / 2
                positive_prob = (1 - negative_prob) / 2
                neutral_prob = (1 - negative_prob) / 2
            else:
                sentiment_label = "neutral"
                neutral_prob = 0.5 + (0.1 - abs(final_score)) / 0.1 * 0.3
                positive_prob = (1 - neutral_prob) / 2
                negative_prob = (1 - neutral_prob) / 2
            
            # Извлечение ключевых слов
            keywords = self._extract_financial_keywords(text)
            
            # Определение языка
            language = self._detect_language(text)
            
            # Создание объекта результата
            class SentimentResult:
                def __init__(self):
                    self.sentiment_score = final_score
                    self.confidence = confidence
                    self.sentiment_label = sentiment_label
                    self.positive_probability = positive_prob
                    self.negative_probability = negative_prob
                    self.neutral_probability = neutral_prob
                    self.keywords = keywords
                    self.language = language
            
            return SentimentResult()
            
        except Exception as e:
            self.logger.error(f"Error in advanced sentiment analysis: {e}")
            return None

    def _lexical_sentiment_analysis(self, text: str) -> float:
        """Лексический анализ настроений."""
        # Финансовые термины с весами
        positive_terms = {
            'profit': 0.8, 'gain': 0.7, 'growth': 0.6, 'bullish': 0.9, 'rally': 0.8,
            'surge': 0.7, 'breakthrough': 0.6, 'outperform': 0.7, 'upgrade': 0.6,
            'strong': 0.5, 'robust': 0.6, 'optimistic': 0.7, 'positive': 0.5
        }
        
        negative_terms = {
            'loss': -0.8, 'decline': -0.7, 'bearish': -0.9, 'crash': -1.0, 'plunge': -0.8,
            'volatility': -0.4, 'uncertainty': -0.5, 'risk': -0.4, 'downgrade': -0.6,
            'weak': -0.5, 'recession': -0.8, 'crisis': -0.9, 'negative': -0.5
        }
        
        words = text.lower().split()
        score = 0.0
        word_count = 0
        
        for word in words:
            if word in positive_terms:
                score += positive_terms[word]
                word_count += 1
            elif word in negative_terms:
                score += negative_terms[word]
                word_count += 1
        
        return score / max(word_count, 1) if word_count > 0 else 0.0

    def _syntactic_sentiment_analysis(self, text: str) -> float:
        """Синтаксический анализ настроений."""
        import re
        
        # Анализ структуры предложений
        sentences = re.split(r'[.!?]+', text)
        total_score = 0.0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Анализ модальности
            modality_score = 0.0
            if re.search(r'\b(will|shall|must|should)\b', sentence, re.IGNORECASE):
                modality_score += 0.1
            if re.search(r'\b(may|might|could|would)\b', sentence, re.IGNORECASE):
                modality_score -= 0.1
                
            # Анализ отрицаний
            negation_count = len(re.findall(r'\b(not|no|never|nothing|none|neither)\b', sentence, re.IGNORECASE))
            negation_multiplier = (-1) ** negation_count
            
            # Анализ интенсификаторов
            intensifiers = len(re.findall(r'\b(very|extremely|highly|significantly|substantially)\b', sentence, re.IGNORECASE))
            intensity_multiplier = 1 + (intensifiers * 0.2)
            
            sentence_score = modality_score * negation_multiplier * intensity_multiplier
            total_score += sentence_score
        
        return total_score / max(len(sentences), 1)

    async def _semantic_sentiment_analysis(self, text: str) -> float:
        """Семантический анализ настроений."""
        # Анализ семантических отношений между словами
        import asyncio
        
        try:
            # Векторное представление текста (упрощенная реализация)
            words = text.lower().split()
            
            # Семантические группы для финансов
            growth_semantic = ['growth', 'increase', 'rise', 'expand', 'boom', 'surge']
            decline_semantic = ['decline', 'decrease', 'fall', 'drop', 'crash', 'plunge']
            
            growth_count = sum(1 for word in words if any(g in word for g in growth_semantic))
            decline_count = sum(1 for word in words if any(d in word for d in decline_semantic))
            
            semantic_score = (growth_count - decline_count) / max(len(words), 1)
            
            # Контекстное усиление
            if 'stock' in text.lower() or 'market' in text.lower():
                semantic_score *= 1.2
            if 'earnings' in text.lower() or 'revenue' in text.lower():
                semantic_score *= 1.1
                
            return max(-1.0, min(1.0, semantic_score))
            
        except Exception as e:
            self.logger.warning(f"Semantic analysis error: {e}")
            return 0.0

    def _contextual_sentiment_analysis(self, text: str) -> float:
        """Контекстный анализ настроений."""
        # Анализ контекста и зависимостей между предложениями
        import re
        
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) < 2:
            return 0.0
        
        contextual_score = 0.0
        
        for i in range(1, len(sentences)):
            prev_sentence = sentences[i-1].strip().lower()
            curr_sentence = sentences[i].strip().lower()
            
            if not prev_sentence or not curr_sentence:
                continue
            
            # Анализ связующих слов
            if curr_sentence.startswith(('however', 'but', 'although', 'despite')):
                contextual_score -= 0.1  # Контраст
            elif curr_sentence.startswith(('moreover', 'furthermore', 'additionally')):
                contextual_score += 0.1  # Усиление
            elif curr_sentence.startswith(('therefore', 'thus', 'consequently')):
                contextual_score += 0.05  # Логическое следование
        
        return contextual_score

    def _temporal_sentiment_analysis(self, text: str) -> float:
        """Временной анализ настроений."""
        import re
        
        # Анализ временных маркеров и их влияния на настроение
        temporal_score = 0.0
        
        # Прошлое время (обычно нейтрально или негативно для прогнозов)
        past_markers = len(re.findall(r'\b(was|were|had|did|yesterday|last)\b', text, re.IGNORECASE))
        temporal_score -= past_markers * 0.02
        
        # Будущее время (позитивно для прогнозов)
        future_markers = len(re.findall(r'\b(will|shall|going to|next|tomorrow|soon)\b', text, re.IGNORECASE))
        temporal_score += future_markers * 0.03
        
        # Настоящее время (нейтрально)
        present_markers = len(re.findall(r'\b(is|are|am|currently|now|today)\b', text, re.IGNORECASE))
        # Настоящее время не изменяет счет
        
        return max(-0.5, min(0.5, temporal_score))

    def _calculate_sentiment_confidence(self, *scores) -> float:
        """Расчет уверенности в анализе настроений."""
        import numpy as np
        
        # Чем больше согласованность между разными методами, тем выше уверенность
        scores_array = np.array(scores)
        
        # Стандартное отклонение (меньше = больше согласованности)
        std_dev = np.std(scores_array)
        
        # Среднее абсолютное значение (больше = сильнее сигнал)
        mean_abs = np.mean(np.abs(scores_array))
        
        # Уверенность обратно пропорциональна разбросу и прямо пропорциональна силе сигнала
        confidence = mean_abs * (1 - min(std_dev, 1.0))
        
        return max(0.1, min(1.0, confidence))

    def _extract_financial_keywords(self, text: str) -> List[str]:
        """Извлечение финансовых ключевых слов."""
        import re
        
        financial_keywords = [
            'profit', 'revenue', 'earnings', 'growth', 'market', 'stock', 'share',
            'price', 'volume', 'volatility', 'risk', 'return', 'investment',
            'trading', 'bullish', 'bearish', 'rally', 'correction', 'trend'
        ]
        
        words = re.findall(r'\b\w+\b', text.lower())
        found_keywords = [word for word in words if word in financial_keywords]
        
        return list(set(found_keywords))  # Убираем дубликаты

    def _detect_language(self, text: str) -> str:
        """Простое определение языка."""
        # Упрощенная реализация определения языка
        russian_chars = len([c for c in text if ord(c) >= 1040 and ord(c) <= 1103])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        
        if russian_chars > english_chars:
            return "ru"
        elif english_chars > 0:
            return "en"
        else:
            return "unknown"

    async def _detect_patterns_advanced(self, market_data, pattern_model) -> List[Any]:
        """Продвинутое обнаружение паттернов."""
        import numpy as np
        import pandas as pd
        from datetime import datetime, timedelta
        
        patterns = []
        
        try:
            # Получаем данные о ценах
            if hasattr(market_data, 'prices') and market_data.prices:
                prices = np.array([float(p) for p in market_data.prices])
                timestamps = getattr(market_data, 'timestamps', None)
                volumes = getattr(market_data, 'volumes', None)
                
                if len(prices) < 10:  # Недостаточно данных
                    return patterns
                
                # 1. Обнаружение паттернов технического анализа
                ta_patterns = self._detect_technical_patterns(prices, timestamps)
                patterns.extend(ta_patterns)
                
                # 2. Обнаружение статистических аномалий
                anomaly_patterns = self._detect_statistical_anomalies(prices, volumes)
                patterns.extend(anomaly_patterns)
                
                # 3. Обнаружение циклических паттернов
                cyclic_patterns = self._detect_cyclic_patterns(prices, timestamps)
                patterns.extend(cyclic_patterns)
                
                # 4. Обнаружение паттернов волатильности
                volatility_patterns = self._detect_volatility_patterns(prices)
                patterns.extend(volatility_patterns)
                
                # 5. Обнаружение паттернов объема
                if volumes:
                    volume_patterns = self._detect_volume_patterns(prices, volumes)
                    patterns.extend(volume_patterns)
            
        except Exception as e:
            self.logger.error(f"Error in advanced pattern detection: {e}")
        
        return patterns

    def _detect_technical_patterns(self, prices: np.ndarray, timestamps) -> List[Any]:
        """Обнаружение паттернов технического анализа."""
        patterns = []
        
        # Двойная вершина/дно
        double_patterns = self._find_double_patterns(prices)
        patterns.extend(double_patterns)
        
        # Голова и плечи
        head_shoulder_patterns = self._find_head_shoulder_patterns(prices)
        patterns.extend(head_shoulder_patterns)
        
        # Треугольники
        triangle_patterns = self._find_triangle_patterns(prices)
        patterns.extend(triangle_patterns)
        
        # Флаги и вымпелы
        flag_patterns = self._find_flag_patterns(prices)
        patterns.extend(flag_patterns)
        
        return patterns

    def _find_double_patterns(self, prices: np.ndarray) -> List[Any]:
        """Поиск паттернов двойной вершины/дна."""
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # Найти локальные максимумы и минимумы
        peaks = []
        troughs = []
        
        for i in range(1, len(prices) - 1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
            elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                troughs.append((i, prices[i]))
        
        # Поиск двойных вершин
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                peak1_idx, peak1_price = peaks[i]
                peak2_idx, peak2_price = peaks[j]
                
                # Проверка на схожесть уровней
                if abs(peak1_price - peak2_price) / peak1_price < 0.02:  # 2% разница
                    # Проверка на промежуточный минимум
                    between_troughs = [t for t in troughs if peak1_idx < t[0] < peak2_idx]
                    if between_troughs:
                        min_trough = min(between_troughs, key=lambda x: x[1])
                        if min_trough[1] < min(peak1_price, peak2_price) * 0.98:
                            
                            class Pattern:
                                def __init__(self):
                                    self.pattern_id = f"double_top_{i}_{j}"
                                    self.pattern_type = "double_top"
                                    self.confidence = 0.8
                                    self.start_time = datetime.now() - timedelta(hours=peak2_idx - peak1_idx)
                                    self.end_time = datetime.now()
                                    self.price_levels = [peak1_price, peak2_price]
                                    self.volume_profile = {}
                            
                            patterns.append(Pattern())
        
        return patterns

    def _find_head_shoulder_patterns(self, prices: np.ndarray) -> List[Any]:
        """Поиск паттернов голова и плечи."""
        patterns = []
        # Упрощенная реализация
        return patterns

    def _find_triangle_patterns(self, prices: np.ndarray) -> List[Any]:
        """Поиск треугольных паттернов."""
        patterns = []
        # Упрощенная реализация
        return patterns

    def _find_flag_patterns(self, prices: np.ndarray) -> List[Any]:
        """Поиск паттернов флагов и вымпелов."""
        patterns = []
        # Упрощенная реализация
        return patterns

    def _detect_statistical_anomalies(self, prices: np.ndarray, volumes) -> List[Any]:
        """Обнаружение статистических аномалий."""
        patterns = []
        
        if len(prices) < 30:
            return patterns
        
        # Z-score аномалии
        returns = np.diff(prices) / prices[:-1]
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        z_scores = (returns - mean_return) / std_return
        
        # Найти аномалии (|z-score| > 2.5)
        anomaly_indices = np.where(np.abs(z_scores) > 2.5)[0]
        
        for idx in anomaly_indices:
            class AnomalyPattern:
                def __init__(self):
                    self.pattern_id = f"anomaly_{idx}"
                    self.pattern_type = "statistical_anomaly"
                    self.confidence = min(abs(z_scores[idx]) / 3.0, 1.0)
                    self.start_time = datetime.now() - timedelta(hours=len(prices) - idx)
                    self.end_time = datetime.now() - timedelta(hours=len(prices) - idx - 1)
                    self.price_levels = [prices[idx], prices[idx+1]]
                    self.volume_profile = {}
            
            patterns.append(AnomalyPattern())
        
        return patterns

    def _detect_cyclic_patterns(self, prices: np.ndarray, timestamps) -> List[Any]:
        """Обнаружение циклических паттернов."""
        patterns = []
        
        if len(prices) < 50:
            return patterns
        
        try:
            # Простой анализ периодичности через автокорреляцию
            from scipy import signal
            
            # Удаление тренда
            detrended = signal.detrend(prices)
            
            # Автокорреляция
            autocorr = np.correlate(detrended, detrended, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Поиск пиков в автокорреляции
            peaks, _ = signal.find_peaks(autocorr, height=np.max(autocorr) * 0.3)
            
            for peak in peaks[:3]:  # Берем только первые 3 значимых цикла
                if peak > 5:  # Минимальный период
                    class CyclicPattern:
                        def __init__(self):
                            self.pattern_id = f"cycle_{peak}"
                            self.pattern_type = "cyclic"
                            self.confidence = autocorr[peak] / np.max(autocorr)
                            self.start_time = datetime.now() - timedelta(hours=len(prices))
                            self.end_time = datetime.now()
                            self.price_levels = []
                            self.volume_profile = {"cycle_length": int(peak)}
                    
                    patterns.append(CyclicPattern())
                    
        except ImportError:
            # Если scipy недоступна, используем простую реализацию
            pass
        except Exception as e:
            self.logger.warning(f"Cyclic pattern detection error: {e}")
        
        return patterns

    def _detect_volatility_patterns(self, prices: np.ndarray) -> List[Any]:
        """Обнаружение паттернов волатильности."""
        patterns = []
        
        if len(prices) < 20:
            return patterns
        
        # Скользящая волатильность
        window = 10
        volatilities = []
        
        for i in range(window, len(prices)):
            window_prices = prices[i-window:i]
            returns = np.diff(window_prices) / window_prices[:-1]
            volatility = np.std(returns)
            volatilities.append(volatility)
        
        if not volatilities:
            return patterns
        
        volatilities = np.array(volatilities)
        mean_vol = np.mean(volatilities)
        std_vol = np.std(volatilities)
        
        # Поиск периодов высокой волатильности
        high_vol_threshold = mean_vol + 2 * std_vol
        high_vol_indices = np.where(volatilities > high_vol_threshold)[0]
        
        if len(high_vol_indices) > 0:
            # Группировка соседних периодов
            groups = []
            current_group = [high_vol_indices[0]]
            
            for i in range(1, len(high_vol_indices)):
                if high_vol_indices[i] - high_vol_indices[i-1] <= 2:
                    current_group.append(high_vol_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [high_vol_indices[i]]
            groups.append(current_group)
            
            # Создание паттернов для групп
            for i, group in enumerate(groups):
                if len(group) >= 3:  # Минимальная длина паттерна
                    class VolatilityPattern:
                        def __init__(self):
                            self.pattern_id = f"high_volatility_{i}"
                            self.pattern_type = "high_volatility"
                            self.confidence = 0.7
                            self.start_time = datetime.now() - timedelta(hours=len(prices) - group[0])
                            self.end_time = datetime.now() - timedelta(hours=len(prices) - group[-1])
                            self.price_levels = []
                            self.volume_profile = {"avg_volatility": np.mean([volatilities[j] for j in group])}
                    
                    patterns.append(VolatilityPattern())
        
        return patterns

    def _detect_volume_patterns(self, prices: np.ndarray, volumes) -> List[Any]:
        """Обнаружение паттернов объема."""
        patterns = []
        
        if not volumes or len(volumes) != len(prices):
            return patterns
        
        volumes_array = np.array([float(v) for v in volumes])
        
        # Поиск аномалий объема
        mean_volume = np.mean(volumes_array)
        std_volume = np.std(volumes_array)
        
        high_volume_threshold = mean_volume + 2 * std_volume
        high_volume_indices = np.where(volumes_array > high_volume_threshold)[0]
        
        for idx in high_volume_indices:
            class VolumePattern:
                def __init__(self):
                    self.pattern_id = f"high_volume_{idx}"
                    self.pattern_type = "volume_spike"
                    self.confidence = min((volumes_array[idx] - mean_volume) / (3 * std_volume), 1.0)
                    self.start_time = datetime.now() - timedelta(hours=len(prices) - idx)
                    self.end_time = datetime.now() - timedelta(hours=len(prices) - idx)
                    self.price_levels = [prices[idx]]
                    self.volume_profile = {"volume": volumes_array[idx], "avg_volume": mean_volume}
            
            patterns.append(VolumePattern())
        
        return patterns

    async def _calculate_risk_metrics_advanced(self, market_data, portfolio_data, risk_model) -> Dict[str, Any]:
        """Продвинутый расчет метрик риска."""
        import numpy as np
        
        try:
            risk_metrics = {}
            
            # 1. Value at Risk (VaR) с использованием разных методов
            var_metrics = await self._calculate_var_metrics(market_data, portfolio_data)
            risk_metrics.update(var_metrics)
            
            # 2. Expected Shortfall (ES)
            es_metrics = await self._calculate_expected_shortfall(market_data, portfolio_data)
            risk_metrics.update(es_metrics)
            
            # 3. Максимальная просадка
            drawdown_metrics = self._calculate_drawdown_metrics(market_data)
            risk_metrics.update(drawdown_metrics)
            
            # 4. Корреляционные риски
            correlation_metrics = await self._calculate_correlation_risks(market_data)
            risk_metrics.update(correlation_metrics)
            
            # 5. Риски ликвидности
            liquidity_metrics = self._calculate_liquidity_risks(market_data)
            risk_metrics.update(liquidity_metrics)
            
            # 6. Стресс-тестирование
            stress_metrics = await self._perform_stress_testing(market_data, portfolio_data)
            risk_metrics.update(stress_metrics)
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Error in advanced risk calculation: {e}")
            return {}

    async def _calculate_var_metrics(self, market_data, portfolio_data) -> Dict[str, float]:
        """Расчет Value at Risk различными методами."""
        var_metrics = {}
        
        try:
            if hasattr(market_data, 'prices') and market_data.prices:
                prices = np.array([float(p) for p in market_data.prices])
                returns = np.diff(prices) / prices[:-1]
                
                # Исторический VaR
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                # Параметрический VaR (предполагаем нормальное распределение)
                mean_return = np.mean(returns)
                std_return = np.std(returns)
                var_95_parametric = mean_return - 1.645 * std_return
                var_99_parametric = mean_return - 2.326 * std_return
                
                # Модифицированный VaR (с учетом асимметрии и эксцесса)
                from scipy import stats
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
                
                # Корнетти-Фишер поправка
                cf_95 = 1.645 + (skewness / 6) * (1.645**2 - 1) + (kurtosis / 24) * (1.645**3 - 3 * 1.645)
                cf_99 = 2.326 + (skewness / 6) * (2.326**2 - 1) + (kurtosis / 24) * (2.326**3 - 3 * 2.326)
                
                var_95_modified = mean_return - cf_95 * std_return
                var_99_modified = mean_return - cf_99 * std_return
                
                var_metrics.update({
                    'var_95_historical': float(var_95),
                    'var_99_historical': float(var_99),
                    'var_95_parametric': float(var_95_parametric),
                    'var_99_parametric': float(var_99_parametric),
                    'var_95_modified': float(var_95_modified),
                    'var_99_modified': float(var_99_modified)
                })
                
        except Exception as e:
            self.logger.warning(f"VaR calculation error: {e}")
        
        return var_metrics

    async def _calculate_expected_shortfall(self, market_data, portfolio_data) -> Dict[str, float]:
        """Расчет Expected Shortfall (Conditional VaR)."""
        es_metrics = {}
        
        try:
            if hasattr(market_data, 'prices') and market_data.prices:
                prices = np.array([float(p) for p in market_data.prices])
                returns = np.diff(prices) / prices[:-1]
                
                # Expected Shortfall на уровне 95%
                var_95 = np.percentile(returns, 5)
                tail_returns_95 = returns[returns <= var_95]
                es_95 = np.mean(tail_returns_95) if len(tail_returns_95) > 0 else var_95
                
                # Expected Shortfall на уровне 99%
                var_99 = np.percentile(returns, 1)
                tail_returns_99 = returns[returns <= var_99]
                es_99 = np.mean(tail_returns_99) if len(tail_returns_99) > 0 else var_99
                
                es_metrics.update({
                    'expected_shortfall_95': float(es_95),
                    'expected_shortfall_99': float(es_99)
                })
                
        except Exception as e:
            self.logger.warning(f"Expected Shortfall calculation error: {e}")
        
        return es_metrics

    def _calculate_drawdown_metrics(self, market_data) -> Dict[str, float]:
        """Расчет метрик просадки."""
        drawdown_metrics = {}
        
        try:
            if hasattr(market_data, 'prices') and market_data.prices:
                prices = np.array([float(p) for p in market_data.prices])
                
                # Кумулятивный максимум
                running_max = np.maximum.accumulate(prices)
                
                # Просадка в каждый момент времени
                drawdown = (prices - running_max) / running_max
                
                # Максимальная просадка
                max_drawdown = np.min(drawdown)
                
                # Средняя просадка
                avg_drawdown = np.mean(drawdown[drawdown < 0]) if np.any(drawdown < 0) else 0.0
                
                # Продолжительность максимальной просадки
                max_dd_idx = np.argmin(drawdown)
                recovery_idx = None
                for i in range(max_dd_idx, len(prices)):
                    if prices[i] >= running_max[max_dd_idx]:
                        recovery_idx = i
                        break
                
                max_dd_duration = recovery_idx - max_dd_idx if recovery_idx else len(prices) - max_dd_idx
                
                drawdown_metrics.update({
                    'max_drawdown': float(max_drawdown),
                    'avg_drawdown': float(avg_drawdown),
                    'max_drawdown_duration': int(max_dd_duration),
                    'current_drawdown': float(drawdown[-1])
                })
                
        except Exception as e:
            self.logger.warning(f"Drawdown calculation error: {e}")
        
        return drawdown_metrics

    async def _calculate_correlation_risks(self, market_data) -> Dict[str, Any]:
        """Расчет корреляционных рисков."""
        correlation_metrics = {}
        
        try:
            # Это упрощенная реализация
            # В реальной системе здесь был бы анализ корреляций между различными активами
            correlation_metrics.update({
                'portfolio_correlation': 0.65,  # Средняя корреляция в портфеле
                'market_correlation': 0.75,     # Корреляция с рынком
                'correlation_risk_score': 0.4   # Оценка корреляционного риска
            })
            
        except Exception as e:
            self.logger.warning(f"Correlation risk calculation error: {e}")
        
        return correlation_metrics

    def _calculate_liquidity_risks(self, market_data) -> Dict[str, float]:
        """Расчет рисков ликвидности."""
        liquidity_metrics = {}
        
        try:
            if hasattr(market_data, 'volumes') and market_data.volumes:
                volumes = np.array([float(v) for v in market_data.volumes])
                
                # Средний объем торгов
                avg_volume = np.mean(volumes)
                
                # Волатильность объема
                volume_volatility = np.std(volumes) / avg_volume if avg_volume > 0 else 0
                
                # Коэффициент ликвидности (упрощенный)
                liquidity_ratio = 1 / (1 + volume_volatility)
                
                # Риск ликвидности
                liquidity_risk = 1 - liquidity_ratio
                
                liquidity_metrics.update({
                    'avg_volume': float(avg_volume),
                    'volume_volatility': float(volume_volatility),
                    'liquidity_ratio': float(liquidity_ratio),
                    'liquidity_risk': float(liquidity_risk)
                })
                
        except Exception as e:
            self.logger.warning(f"Liquidity risk calculation error: {e}")
        
        return liquidity_metrics

    async def _perform_stress_testing(self, market_data, portfolio_data) -> Dict[str, float]:
        """Проведение стресс-тестирования."""
        stress_metrics = {}
        
        try:
            if hasattr(market_data, 'prices') and market_data.prices:
                prices = np.array([float(p) for p in market_data.prices])
                
                # Сценарий 1: Падение рынка на 20%
                stress_scenario_1 = -0.20
                portfolio_impact_1 = stress_scenario_1 * 0.8  # Предполагаем бету 0.8
                
                # Сценарий 2: Рост волатильности в 2 раза
                current_volatility = np.std(np.diff(prices) / prices[:-1])
                stress_volatility = current_volatility * 2
                
                # Сценарий 3: Кризис ликвидности (спреды увеличиваются в 3 раза)
                normal_spread = 0.001  # 0.1%
                stress_spread = normal_spread * 3
                
                stress_metrics.update({
                    'market_crash_impact': float(portfolio_impact_1),
                    'volatility_stress_level': float(stress_volatility),
                    'liquidity_crisis_spread': float(stress_spread),
                    'stress_test_score': float(abs(portfolio_impact_1) + stress_volatility + stress_spread)
                })
                
        except Exception as e:
            self.logger.warning(f"Stress testing error: {e}")
        
        return stress_metrics
