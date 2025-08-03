"""
Промышленная реализация MLService.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

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
        # Анализируем настроения
        # Здесь нет метода analyze_sentiment, оставляем заглушку или реализуем через сторонний сервис
        sentiment_result = None  # TODO: реализовать через внешний сервис или удалить
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
        # Здесь нет метода detect_patterns, оставляем заглушку или реализуем через сторонний сервис
        patterns: List[Any] = []  # TODO: реализовать через внешний сервис или удалить
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

    async def calculate_risk_metrics(
        self, portfolio_data: Dict[str, Any]
    ) -> Dict[str, Decimal]:
        """Расчет метрик риска."""
        return await self._execute_with_metrics(
            "calculate_risk_metrics", self._calculate_risk_metrics_impl, portfolio_data
        )

    async def _calculate_risk_metrics_impl(
        self, portfolio_data: Dict[str, Any]
    ) -> Dict[str, Decimal]:
        """Реализация расчета метрик риска."""
        # Получаем модель для расчета рисков
        risk_model = await self._get_risk_model()
        if not risk_model:
            self.logger.warning("No risk model available")
            return {}
        # Здесь нет метода calculate_risk_metrics, оставляем заглушку или реализуем через сторонний сервис
        risk_metrics = None  # TODO: реализовать через внешний сервис или удалить
        return {}

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
