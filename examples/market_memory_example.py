# -*- coding: utf-8 -*-
"""Пример использования системы рыночной памяти."""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from application.prediction.pattern_predictor import PatternPredictor
from domain.intelligence.market_pattern_recognizer import (
    MarketPatternRecognizer, PatternDetection)
from domain.memory.pattern_memory import (MarketFeatures, OutcomeType,
                                          PatternMemory, PatternOutcome,
                                          PatternSnapshot)
from domain.value_objects.timestamp import Timestamp
from infrastructure.agents.agent_context_refactored import AgentContextManager
from infrastructure.agents.market_memory.types import MarketMemoryConfig
from infrastructure.agents.market_memory.integration import MarketMemoryIntegration


class MarketMemoryExample:
    """Пример использования системы рыночной памяти."""

    def __init__(self):
        self.config = MarketMemoryConfig(
            max_memories=10000,
            cleanup_interval=3600,
            memory_ttl=86400 * 7,
            enable_compression=True,
            enable_indexing=True,
            log_memory_operations=False,
        )

        # Инициализация компонентов
        self.pattern_memory = PatternMemory("data/example_pattern_memory.db")
        self.pattern_recognizer = MarketPatternRecognizer()
        self.pattern_predictor = PatternPredictor(self.pattern_memory)
        self.context_manager = AgentContextManager()

        # Интегратор
        self.integration = MarketMemoryIntegration(config=self.config)

        logger.info("MarketMemoryExample initialized")

    async def run_complete_example(self):
        """Запуск полного примера работы системы."""
        try:
            logger.info("Starting Market Memory System example...")

            # 1. Создаем тестовые данные
            await self._create_test_data()

            # 2. Демонстрируем обнаружение паттернов
            await self._demonstrate_pattern_detection()

            # 3. Демонстрируем прогнозирование
            await self._demonstrate_prediction()

            # 4. Демонстрируем интеграцию с AgentContext
            await self._demonstrate_agent_integration()

            # 5. Показываем статистику
            self._show_statistics()

            logger.info("Market Memory System example completed successfully!")

        except Exception as e:
            logger.error(f"Error in complete example: {e}")

    async def _create_test_data(self):
        """Создание тестовых данных для демонстрации."""
        logger.info("Creating test data...")

        try:
            # Создаем несколько исторических паттернов
            test_patterns = [
                {
                    "symbol": "BTC/USDT",
                    "pattern_type": "whale_absorption",
                    "features": {
                        "price": 50000.0,
                        "volume": 1000000.0,
                        "volume_sma_ratio": 2.5,
                        "order_book_imbalance": 0.3,
                        "spread": 0.001,
                        "volatility": 0.02,
                    },
                    "outcome": {
                        "price_change_percent": 1.2,
                        "duration_minutes": 15,
                        "outcome_type": "profitable",
                    },
                },
                {
                    "symbol": "BTC/USDT",
                    "pattern_type": "mm_spoofing",
                    "features": {
                        "price": 51000.0,
                        "volume": 800000.0,
                        "volume_sma_ratio": 1.8,
                        "order_book_imbalance": -0.4,
                        "spread": 0.002,
                        "volatility": 0.03,
                    },
                    "outcome": {
                        "price_change_percent": -0.8,
                        "duration_minutes": 10,
                        "outcome_type": "profitable",
                    },
                },
                {
                    "symbol": "ETH/USDT",
                    "pattern_type": "whale_absorption",
                    "features": {
                        "price": 3000.0,
                        "volume": 500000.0,
                        "volume_sma_ratio": 2.2,
                        "order_book_imbalance": 0.25,
                        "spread": 0.0015,
                        "volatility": 0.025,
                    },
                    "outcome": {
                        "price_change_percent": 0.9,
                        "duration_minutes": 12,
                        "outcome_type": "profitable",
                    },
                },
            ]

            # Сохраняем тестовые паттерны
            for i, pattern_data in enumerate(test_patterns):
                await self._save_test_pattern(f"test_pattern_{i}", pattern_data)

            logger.info(f"Created {len(test_patterns)} test patterns")

        except Exception as e:
            logger.error(f"Error creating test data: {e}")

    async def _save_test_pattern(self, pattern_id: str, pattern_data: Dict[str, Any]):
        """Сохранение тестового паттерна."""
        try:
            # Создаем снимок паттерна
            features = MarketFeatures(
                price=pattern_data["features"]["price"],
                price_change_1m=0.0,
                price_change_5m=0.0,
                price_change_15m=0.0,
                volatility=pattern_data["features"]["volatility"],
                volume=pattern_data["features"]["volume"],
                volume_change_1m=0.0,
                volume_change_5m=0.0,
                volume_sma_ratio=pattern_data["features"]["volume_sma_ratio"],
                spread=pattern_data["features"]["spread"],
                spread_change=0.0,
                bid_volume=pattern_data["features"]["volume"] * 0.6,
                ask_volume=pattern_data["features"]["volume"] * 0.4,
                order_book_imbalance=pattern_data["features"]["order_book_imbalance"],
                depth_absorption=0.1,
                entropy=0.5,
                gravity=0.0,
                latency=0.0,
                correlation=0.0,
                whale_signal=0.8,
                mm_signal=0.0,
                external_sync=False,
            )

            snapshot = PatternSnapshot(
                pattern_id=pattern_id,
                timestamp=Timestamp.from_datetime(datetime.now()),
                symbol=pattern_data["symbol"],
                pattern_type=pattern_data["pattern_type"],
                confidence=0.85,
                strength=0.8,
                direction=(
                    "up"
                    if pattern_data["outcome"]["price_change_percent"] > 0
                    else "down"
                ),
                features=features,
                metadata={"test_data": True},
            )

            # Сохраняем снимок
            self.pattern_memory.save_snapshot(pattern_id, snapshot)

            # Создаем исход
            outcome = PatternOutcome(
                pattern_id=pattern_id,
                symbol=pattern_data["symbol"],
                outcome_type=OutcomeType(pattern_data["outcome"]["outcome_type"]),
                timestamp=Timestamp.from_datetime(datetime.now()),
                price_change_percent=pattern_data["outcome"]["price_change_percent"],
                volume_change_percent=10.0,
                duration_minutes=pattern_data["outcome"]["duration_minutes"],
                max_profit_percent=pattern_data["outcome"]["price_change_percent"]
                if pattern_data["outcome"]["price_change_percent"] > 0
                else 0.0,
                max_loss_percent=abs(pattern_data["outcome"]["price_change_percent"])
                if pattern_data["outcome"]["price_change_percent"] < 0
                else 0.0,
                final_return_percent=pattern_data["outcome"]["price_change_percent"],
                volatility_during=0.02,
                volume_profile="normal",
                market_regime="trending",
                metadata={"test_data": True},
            )

            # Сохраняем исход
            self.pattern_memory.save_outcome(pattern_id, outcome)

            logger.debug(f"Saved test pattern: {pattern_id}")

        except Exception as e:
            logger.error(f"Error saving test pattern {pattern_id}: {e}")

    async def _demonstrate_pattern_detection(self):
        """Демонстрация обнаружения паттернов."""
        logger.info("Demonstrating pattern detection...")

        try:
            # Создаем тестовые рыночные данные
            market_data = self._create_test_market_data()
            order_book = self._create_test_order_book()

            # Обнаруживаем паттерны
            patterns = await self._detect_patterns_in_data(market_data, order_book)

            logger.info(f"Detected {len(patterns)} patterns:")
            for pattern in patterns:
                logger.info(
                    f"  - {pattern.pattern_type}: confidence={pattern.confidence:.3f}, "
                    f"strength={pattern.strength:.3f}"
                )

            # Сохраняем обнаруженные паттерны в память
            for pattern in patterns:
                features = MarketFeatures(
                    price=market_data["close"].iloc[-1],
                    price_change_1m=0.0,
                    price_change_5m=0.0,
                    price_change_15m=0.0,
                    volatility=market_data["close"].pct_change().std(),
                    volume=market_data["volume"].iloc[-1],
                    volume_change_1m=0.0,
                    volume_change_5m=0.0,
                    volume_sma_ratio=1.5,
                    spread=0.001,
                    spread_change=0.0,
                    bid_volume=order_book["bids"][0][1] if order_book["bids"] else 0.0,
                    ask_volume=order_book["asks"][0][1] if order_book["asks"] else 0.0,
                    order_book_imbalance=0.1,
                    depth_absorption=0.05,
                    entropy=0.6,
                    gravity=0.0,
                    latency=0.0,
                    correlation=0.0,
                    whale_signal=0.7,
                    mm_signal=0.3,
                    external_sync=False,
                )

                snapshot = PatternSnapshot(
                    pattern_id=f"detected_{pattern.pattern_type}_{datetime.now().timestamp()}",
                    timestamp=Timestamp.from_datetime(datetime.now()),
                    symbol="BTC/USDT",
                    pattern_type=pattern.pattern_type,
                    confidence=pattern.confidence,
                    strength=pattern.strength,
                    direction=pattern.direction,
                    features=features,
                    metadata={"detected": True},
                )

                self.pattern_memory.save_snapshot(snapshot.pattern_id, snapshot)

        except Exception as e:
            logger.error(f"Error in pattern detection demonstration: {e}")

    async def _demonstrate_prediction(self):
        """Демонстрация прогнозирования."""
        logger.info("Demonstrating pattern prediction...")

        try:
            # Создаем тестовые рыночные данные
            market_data = self._create_test_market_data()
            order_book = self._create_test_order_book()

            # Получаем прогноз
            prediction = await self.pattern_predictor.predict_pattern_outcome(
                market_data, order_book
            )

            if prediction:
                logger.info("Pattern prediction result:")
                logger.info(f"  - Pattern type: {prediction.pattern_type}")
                logger.info(f"  - Confidence: {prediction.confidence:.3f}")
                logger.info(f"  - Expected direction: {prediction.expected_direction}")
                logger.info(f"  - Expected return: {prediction.expected_return:.3f}")
                logger.info(f"  - Risk level: {prediction.risk_level}")
            else:
                logger.info("No pattern prediction available")

        except Exception as e:
            logger.error(f"Error in prediction demonstration: {e}")

    async def _demonstrate_agent_integration(self):
        """Демонстрация интеграции с AgentContext."""
        logger.info("Demonstrating AgentContext integration...")

        try:
            # Создаем контекст агента
            context = self.context_manager.create_context("BTC/USDT")

            # Создаем тестовые данные
            market_data = self._create_test_market_data()
            order_book = self._create_test_order_book()

            # Обнаруживаем паттерны и сохраняем в контекст
            patterns = await self._detect_patterns_in_data(market_data, order_book)
            if patterns:
                context.market_pattern_result = patterns[0]
                logger.info(f"Saved pattern to context: {patterns[0].pattern_type}")

            # Получаем прогноз и сохраняем в контекст
            prediction = await self.pattern_predictor.predict_pattern_outcome(
                market_data, order_book
            )
            if prediction:
                context.pattern_prediction.expected_direction = prediction.expected_direction
                context.pattern_prediction.expected_return = prediction.expected_return
                context.pattern_prediction.confidence = prediction.confidence
                logger.info("Saved prediction to context")

            # Показываем состояние контекста
            logger.info("AgentContext state:")
            logger.info(f"  - Symbol: {context.symbol}")
            logger.info(f"  - Pattern detected: {context.market_pattern_result is not None}")
            logger.info(f"  - Prediction available: {context.pattern_prediction.expected_direction is not None}")

        except Exception as e:
            logger.error(f"Error in agent integration demonstration: {e}")

    def _create_test_market_data(self) -> pd.DataFrame:
        """Создание тестовых рыночных данных."""
        try:
            # Создаем временной ряд
            dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")
            
            # Создаем OHLCV данные
            np.random.seed(42)  # Для воспроизводимости
            base_price = 50000.0
            returns = np.random.normal(0, 0.001, 100)  # 0.1% волатильность
            prices = [base_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Создаем DataFrame
            data = {
                "timestamp": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(100000, 1000000, 100),
            }
            
            df = pd.DataFrame(data)
            df.set_index("timestamp", inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating test market data: {e}")
            return pd.DataFrame()

    def _create_test_order_book(self) -> Dict[str, Any]:
        """Создание тестового ордербука."""
        try:
            base_price = 50000.0
            spread = 10.0
            
            # Создаем bids
            bids = []
            for i in range(10):
                price = base_price - i * 10
                volume = np.random.uniform(0.1, 2.0)
                bids.append([price, volume])
            
            # Создаем asks
            asks = []
            for i in range(10):
                price = base_price + spread + i * 10
                volume = np.random.uniform(0.1, 2.0)
                asks.append([price, volume])
            
            return {
                "bids": bids,
                "asks": asks,
                "timestamp": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error creating test order book: {e}")
            return {"bids": [], "asks": [], "timestamp": datetime.now().isoformat()}

    async def _detect_patterns_in_data(
        self, market_data: pd.DataFrame, order_book: Dict[str, Any]
    ) -> List[PatternDetection]:
        """Обнаружение паттернов в данных."""
        try:
            if market_data.empty:
                return []
            
            # Обнаруживаем паттерны
            patterns = await self.pattern_recognizer.detect_patterns(
                market_data, order_book
            )
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error detecting patterns: {e}")
            return []

    def _show_statistics(self):
        """Показ статистики системы."""
        logger.info("System statistics:")
        
        try:
            # Статистика памяти паттернов
            pattern_stats = self.pattern_memory.get_statistics()
            logger.info(f"Pattern memory: {pattern_stats}")
            
            # Статистика интеграции
            integration_stats = self.integration.get_statistics()
            logger.info(f"Memory integration: {integration_stats}")
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")


async def main():
    """Основная функция примера."""
    logger.info("Starting Market Memory Example")
    
    try:
        example = MarketMemoryExample()
        await example.run_complete_example()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    asyncio.run(main())
