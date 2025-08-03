"""
Агент анализа поведения китов.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from domain.types.agent_types import ProcessingResult, AgentType, AgentConfig
from infrastructure.agents.base_agent import AgentStatus, BaseAgent

from .detectors import (
    DefaultDataProvider,
    IDataProvider,
    WhaleActivityCache,
    WhaleMLModel,
    WhaleSignalAnalyzer,
)
from .types import WhaleActivity, WhaleAnalysis


class WhalesAgent(BaseAgent):
    """Агент анализа поведения китов."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # [2] создаю AgentConfig для базового агента
        agent_config = {
            "name": "WhalesAgent",
            "type": "whale_analyzer",
            "metadata": {
                "min_whale_size": 100000,
                "confidence_threshold": 0.7,
                "impact_threshold": 0.5,
                "analysis_window": 3600,  # 1 час
                "enable_ml_model": True,
                "enable_signal_analysis": True,
                "enable_caching": True,
            }
        }
        super().__init__("WhalesAgent", "whale_analyzer", agent_config)  # config должен быть dict[str, Any]
        whale_config = config or {
            "min_whale_size": 100000,
            "confidence_threshold": 0.7,
            "impact_threshold": 0.5,
            "analysis_window": 3600,  # 1 час
            "enable_ml_model": True,
            "enable_signal_analysis": True,
            "enable_caching": True,
        }
        # Компоненты агента
        self.ml_model = WhaleMLModel() if whale_config["enable_ml_model"] else None
        self.signal_analyzer = (
            WhaleSignalAnalyzer(whale_config)
            if whale_config["enable_signal_analysis"]
            else None
        )
        self.activity_cache = (
            WhaleActivityCache() if whale_config["enable_caching"] else None
        )
        self.data_provider = DefaultDataProvider()
        # Состояние агента
        self.whale_activities: List[WhaleActivity] = []
        self.current_analysis: Optional[WhaleAnalysis] = None
        self.stats: Dict[str, int] = {
            "total_activities": 0,
            "whale_detections": 0,
            "high_impact_activities": 0,
            "analysis_count": 0,
        }

    async def initialize(self) -> bool:
        """Инициализация агента."""
        try:
            if not self.validate_config():
                return False
            # Инициализация ML модели
            if self.ml_model:
                # Здесь можно загрузить предобученную модель
                pass
            self._update_state(AgentStatus.HEALTHY)
            self.update_confidence(0.8)
            logger.info("WhalesAgent initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize WhalesAgent: {e}")
            self.record_error(f"Initialization failed: {e}")
            return False

    async def process(self, data: Any) -> ProcessingResult:
        """Обработка данных."""
        start_time = datetime.now()
        try:
            if isinstance(data, dict):
                symbol = data.get("symbol", "")
                if not symbol:
                    raise ValueError("Symbol is required for whale analysis")
                # Получение рыночных данных
                market_data = await self.data_provider.get_market_data(symbol)
                order_book = await self.data_provider.get_order_book(symbol)
                # Анализ активности китов
                activities = await self.detect_whale_activities(
                    symbol, market_data, order_book
                )
                # Анализ поведения китов
                analysis = await self.analyze_whale_behavior(symbol, activities)
                # Обновление кэша
                if self.activity_cache:
                    for activity in activities:
                        self.activity_cache.add(symbol, activity)
                result_data = {
                    "whale_activities": [a.__dict__ for a in activities],
                    "whale_analysis": analysis.__dict__ if analysis else {},
                    "detection_count": len(activities),
                    "high_impact_count": len(
                        [
                            a
                            for a in activities
                            if a.impact_score > float(self.config.get("metadata", {}).get("impact_threshold", 0.5))
                        ]
                    ),
                    "timestamp": datetime.now().isoformat(),
                }
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self.record_success(processing_time)
                return ProcessingResult(
                    success=True,
                    data=result_data,
                    confidence=0.7,
                    risk_score=0.3,
                    processing_time_ms=processing_time,  # [3] исправляю на processing_time_ms
                    timestamp=datetime.now(),  # [3] добавляю timestamp
                    metadata={},  # [3] добавляю metadata
                    errors=[],  # [3] добавляю errors
                    warnings=[]  # [3] добавляю warnings
                )
            else:
                raise ValueError("Invalid data format")
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.record_error(f"Processing failed: {e}", processing_time)
            return ProcessingResult(
                success=False,
                data={"error": str(e)},
                confidence=0.0,
                risk_score=1.0,
                processing_time_ms=processing_time,  # [3] исправляю на processing_time_ms
                timestamp=datetime.now(),  # [3] добавляю timestamp
                metadata={},  # [3] добавляю metadata
                errors=[str(e)],  # [3] добавляю errors
                warnings=[]  # [3] добавляю warnings
            )

    async def cleanup(self) -> None:
        """Очистка ресурсов."""
        try:
            if self.activity_cache:
                self.activity_cache.clear()
            self.whale_activities.clear()
            self.current_analysis = None
            logger.info("WhalesAgent cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def validate_config(self) -> bool:
        """Валидация конфигурации."""
        try:
            # [4] правильный доступ к TypedDict через metadata
            metadata = self.config.get("metadata", {})
            required_keys = [
                "min_whale_size",
                "confidence_threshold",
                "impact_threshold",
                "analysis_window",
            ]
            for key in required_keys:
                if key not in metadata:
                    logger.error(f"Missing required config key: {key}")
                    return False
                value = metadata[key]
                if not isinstance(value, (int, float)) or float(value) <= 0:
                    logger.error(f"Invalid config value for {key}: {value}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Config validation error: {e}")
            return False

    async def detect_whale_activities(
        self, symbol: str, market_data: Any, order_book: Dict[str, Any]
    ) -> List[WhaleActivity]:
        """Обнаружение активности китов."""
        try:
            activities = []
            # Анализ стакана ордеров
            if self.signal_analyzer and order_book:
                order_book_activity = self.signal_analyzer.analyze_order_book(
                    order_book, self.config.get("metadata", {})
                )
                if order_book_activity:
                    activities.append(order_book_activity)
            # Анализ объемов
            volume_activity = await self._analyze_volume_activity(symbol, market_data)
            if volume_activity:
                activities.append(volume_activity)
            # Анализ импульсов
            impulse_activity = await self._analyze_impulse_activity(symbol, market_data)
            if impulse_activity:
                activities.append(impulse_activity)
            # Анализ доминирования
            dominance_activity = await self._analyze_dominance_activity(
                symbol, market_data
            )
            if dominance_activity:
                activities.append(dominance_activity)
            # Фильтрация по порогам
            filtered_activities = [
                activity
                for activity in activities
                if activity.confidence >= self.config.get("confidence_threshold", 0.7)
            ]
            self.stats["total_activities"] += len(filtered_activities)
            self.stats["whale_detections"] += len(filtered_activities)
            return filtered_activities
        except Exception as e:
            logger.error(f"Error detecting whale activities: {e}")
            return []

    async def analyze_whale_behavior(
        self, symbol: str, activities: List[WhaleActivity]
    ) -> Optional[WhaleAnalysis]:
        """Анализ поведения китов."""
        try:
            if not activities:
                return None
            # Расчет общих метрик
            total_volume = sum(activity.volume for activity in activities)
            buy_volume = sum(
                activity.volume
                for activity in activities
                if activity.direction == "buy"
            )
            sell_volume = sum(
                activity.volume
                for activity in activities
                if activity.direction == "sell"
            )
            net_direction = "buy" if buy_volume > sell_volume else "sell"
            # Расчет среднего скора воздействия
            impact_score = sum(activity.impact_score for activity in activities) / len(
                activities
            )
            # Расчет средней уверенности
            confidence = sum(activity.confidence for activity in activities) / len(
                activities
            )
            # Подсчет крупных транзакций
            min_whale_size = self.config.get("metadata", {}).get("min_whale_size", 100000)
            if isinstance(min_whale_size, (int, float)):
                large_transactions = len(
                    [
                        activity
                        for activity in activities
                        if float(activity.volume) >= float(min_whale_size)
                    ]
                )
            else:
                large_transactions = 0
            analysis = WhaleAnalysis(
                whale_activities=activities,
                total_volume=total_volume,
                net_direction=net_direction,
                impact_score=impact_score,
                confidence=confidence,
                whale_count=len(
                    set(
                        activity.details.get("whale_id", "")
                        for activity in activities
                        if activity.details
                    )
                ),
                large_transactions=large_transactions,
            )
            self.current_analysis = analysis
            self.stats["analysis_count"] += 1
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing whale behavior: {e}")
            return None

    async def _analyze_volume_activity(
        self, symbol: str, market_data: Any
    ) -> Optional[WhaleActivity]:
        """Анализ активности по объемам."""
        try:
            # Здесь должна быть логика анализа объемов
            # Пока возвращаем заглушку
            return WhaleActivity(
                timestamp=pd.Timestamp(datetime.now()),
                volume=1000000.0,
                price=50000.0,
                direction="buy",
                confidence=0.8,
                impact_score=0.7,
                details={"analysis_type": "volume"},
            )
        except Exception as e:
            logger.error(f"Error analyzing volume activity: {e}")
            return None

    async def _analyze_impulse_activity(
        self, symbol: str, market_data: Any
    ) -> Optional[WhaleActivity]:
        """Анализ импульсной активности."""
        try:
            # Здесь должна быть логика анализа импульсов
            # Пока возвращаем заглушку
            return WhaleActivity(
                timestamp=pd.Timestamp(datetime.now()),
                volume=800000.0,
                price=51000.0,
                direction="sell",
                confidence=0.6,
                impact_score=0.5,
                details={"analysis_type": "impulse"},
            )
        except Exception as e:
            logger.error(f"Error analyzing impulse activity: {e}")
            return None

    async def _analyze_dominance_activity(
        self, symbol: str, market_data: Any
    ) -> Optional[WhaleActivity]:
        """Анализ доминирующей активности."""
        try:
            # Здесь должна быть логика анализа доминирования
            # Пока возвращаем заглушку
            return WhaleActivity(
                timestamp=pd.Timestamp(datetime.now()),
                volume=1200000.0,
                price=52000.0,
                direction="buy",
                confidence=0.9,
                impact_score=0.8,
                details={"analysis_type": "dominance"},
            )
        except Exception as e:
            logger.error(f"Error analyzing dominance activity: {e}")
            return None

    def get_whale_summary(self) -> Dict[str, Any]:
        """Получение сводки агента китов."""
        return {
            "total_activities": self.stats["total_activities"],
            "whale_detections": self.stats["whale_detections"],
            "high_impact_activities": self.stats["high_impact_activities"],
            "analysis_count": self.stats["analysis_count"],
            "current_analysis": (
                self.current_analysis.__dict__ if self.current_analysis else {}
            ),
            "config": self.config,
        }

    def reset_whale_agent(self) -> None:
        """Сброс состояния агента китов."""
        self.whale_activities.clear()
        self.current_analysis = None
        if self.activity_cache:
            self.activity_cache.clear()
        logger.info("WhalesAgent state reset")
