#!/usr/bin/env python3
"""
Пример использования улучшенной ИИ-системы торговли ATB
"""

import asyncio
import sys
from pathlib import Path

from shared.numpy_utils import np
import pandas as pd

# Добавление корневой директории в путь
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from infrastructure.ml_services.decision_reasoner import DecisionReasoner
from ml.live_adaptation import LiveAdaptation
from ml.meta_learning import MetaLearner, StrategyEvolution
from ml.transformer_predictor import TransformerPredictor
from utils.event_bus import EventBus


class AITradingExample:
    """Пример использования ИИ-системы торговли"""

    def __init__(self):
        self.event_bus = EventBus()

        # Инициализация ИИ-компонентов
        self.decision_reasoner = DecisionReasoner()
        self.transformer_predictor = TransformerPredictor()
        self.meta_learner = MetaLearner()
        self.live_adaptation = LiveAdaptation()

        # Генерация тестовых данных
        self.test_data = self._generate_test_data()

    def _generate_test_data(self) -> pd.DataFrame:
        """Генерация тестовых рыночных данных"""
        np.random.seed(42)

        # Создание временного ряда
        # Исправляем использование pd.date_range на pd.DatetimeIndex
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="1H")
        n_periods = len(dates)  # Исправляем: используем len() для DatetimeIndex

        # Генерация цен
        returns = np.random.normal(0, 0.02, n_periods)
        prices = 100 * np.exp(np.cumsum(returns))

        # Создание DataFrame
        data = pd.DataFrame(
            {
                "timestamp": dates,
                "open": prices * (1 + np.random.normal(0, 0.005, n_periods)),
                "high": prices * (1 + abs(np.random.normal(0, 0.01, n_periods))),
                "low": prices * (1 - abs(np.random.normal(0, 0.01, n_periods))),
                "close": prices,
                "volume": np.random.lognormal(10, 1, n_periods),
            }
        )

        return data

    async def run_decision_reasoner_example(self):
        """Пример работы DecisionReasoner 2.0"""
        logger.info("=== DecisionReasoner 2.0 Example ===")

        # Подготовка данных
        market_data = self.test_data.tail(100)

        # Симуляция сигналов от стратегий
        strategy_signals = [
            {
                "strategy_name": "momentum",
                "weight": 1.0,
                "confidence": 0.8,
                "action": "buy",
                "strategy_type": "momentum",
            },
            {
                "strategy_name": "mean_reversion",
                "weight": 0.7,
                "confidence": 0.6,
                "action": "sell",
                "strategy_type": "mean_reversion",
            },
        ]

        # Симуляция ML предсказаний
        ml_predictions = [
            {
                "model_name": "transformer",
                "weight": 1.0,
                "confidence": 0.75,
                "prediction": "buy",
            }
        ]

        # Симуляция технических сигналов
        technical_signals = [
            {"indicator": "rsi", "weight": 0.5, "confidence": 0.7, "signal": "buy"}
        ]

        # Контекст риска
        risk_context = {"symbol": "BTC/USDT", "position_size": 0.1}

        # Принятие решения
        decision = self.decision_reasoner.make_enhanced_decision(
            market_data,
            strategy_signals,
            ml_predictions,
            technical_signals,
            risk_context,
        )

        logger.info(f"Decision: {decision.action} {decision.direction}")
        logger.info(f"Confidence: {decision.confidence:.3f}")
        logger.info(f"Volume: {decision.volume:.3f}")
        logger.info(f"Stop Loss: {decision.stop_loss:.2f}")
        logger.info(f"Take Profit: {decision.take_profit:.2f}")

        # Получение метрик
        metrics = self.decision_reasoner.get_enhanced_metrics()
        logger.info(f"Decision Metrics: {metrics}")

    async def run_transformer_example(self):
        """Пример работы TransformerPredictor"""
        logger.info("=== TransformerPredictor Example ===")

        # Подготовка данных
        data = self.test_data.tail(200)

        # Обновление модели
        self.transformer_predictor.update(data, "test_model")

        # Предсказание
        predictions, confidence = self.transformer_predictor.predict(data, "test_model")

        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Confidence: {confidence:.3f}")

        # Получение метрик
        metrics = self.transformer_predictor.get_metrics("test_model")
        logger.info(f"Transformer Metrics: {metrics}")

    async def run_meta_learning_example(self):
        """Пример работы MetaLearning"""
        logger.info("=== MetaLearning Example ===")

        # Подготовка данных
        data = self.test_data.tail(300)

        # Симуляция модели
        class MockModel:
            def predict(self, X):
                return np.random.choice([0, 1], size=len(X))

        mock_model = MockModel()

        # Обновление мета-обучения
        self.meta_learner.update(data, "test_model", mock_model)

        # Предсказание
        predictions, confidence = self.meta_learner.predict(data, "test_model")

        logger.info(f"Meta Learning Predictions: {predictions}")
        logger.info(f"Confidence: {confidence:.3f}")

        # Получение метрик
        metrics = self.meta_learner.get_metrics("test_model")
        logger.info(f"Meta Learning Metrics: {metrics}")

    async def run_live_adaptation_example(self):
        """Пример работы LiveAdaptation"""
        logger.info("=== LiveAdaptation Example ===")

        # Подготовка данных
        data = self.test_data.tail(100)

        # Обновление системы адаптации
        await self.live_adaptation.update(data)

        # Предсказание
        prediction, confidence = await self.live_adaptation.predict(data)

        logger.info(f"Live Adaptation Prediction: {prediction:.3f}")
        logger.info(f"Confidence: {confidence:.3f}")

        # Получение метрик
        metrics = await self.live_adaptation.get_metrics()
        logger.info(f"Live Adaptation Metrics: {metrics}")

    async def run_strategy_evolution_example(self):
        """Пример эволюции стратегий"""
        logger.info("=== Strategy Evolution Example ===")

        # Создание эволюционного оптимизатора
        strategy_evolution = StrategyEvolution()

        # Подготовка данных
        data = self.test_data.tail(500)

        # Симуляция истории производительности
        performance_history = [0.6, 0.65, 0.7, 0.68, 0.72, 0.75, 0.73, 0.77, 0.8, 0.78]

        # Эволюция стратегий
        strategy_evolution.evolve_strategies(data, performance_history)

        # Получение лучших стратегий
        best_strategies = strategy_evolution.get_best_strategies(3)

        logger.info(f"Best strategies found: {len(best_strategies)}")
        for i, strategy in enumerate(best_strategies):
            logger.info(f"Strategy {i+1}: {strategy.name}")
            logger.info(f"  Type: {strategy.strategy_type}")
            logger.info(f"  Parameters: {strategy.parameters}")
            logger.info(
                f"  Performance: {np.mean(strategy.performance_history[-10:]) if strategy.performance_history else 0:.3f}"
            )

    async def run_integrated_example(self):
        """Интегрированный пример работы всей системы"""
        logger.info("=== Integrated AI Trading System Example ===")

        # Симуляция торгового цикла
        for i in range(10):
            logger.info(f"\n--- Trading Cycle {i+1} ---")

            # Получение новых данных
            start_idx = 100 + i * 10
            end_idx = start_idx + 100
            market_data = self.test_data.iloc[start_idx:end_idx]

            # 1. Определение рыночного режима
            regime_detector = MarketRegimeDetector()
            market_regime = regime_detector.detect_regime(market_data)
            logger.info(f"Market Regime: {market_regime}")

            # 2. Обновление ML моделей
            await self.live_adaptation.update(market_data)
            self.transformer_predictor.update(market_data, f"model_{i}")

            # 3. Генерация сигналов
            strategy_signals = [
                {
                    "strategy_name": f"strategy_{i}",
                    "weight": 1.0,
                    "confidence": 0.7 + 0.1 * np.random.random(),
                    "action": np.random.choice(["buy", "sell", "hold"]),
                    "strategy_type": "momentum",
                }
            ]

            ml_predictions = [
                {
                    "model_name": f"transformer_{i}",
                    "weight": 1.0,
                    "confidence": 0.6 + 0.2 * np.random.random(),
                    "prediction": np.random.choice(["buy", "sell", "hold"]),
                }
            ]

            technical_signals = [
                {
                    "indicator": "rsi",
                    "weight": 0.5,
                    "confidence": 0.5 + 0.3 * np.random.random(),
                    "signal": np.random.choice(["buy", "sell", "hold"]),
                }
            ]

            # 4. Принятие решения
            risk_context = {"symbol": "BTC/USDT", "position_size": 0.1}

            decision = self.decision_reasoner.make_enhanced_decision(
                market_data,
                strategy_signals,
                ml_predictions,
                technical_signals,
                risk_context,
            )

            logger.info(f"Decision: {decision.action} {decision.direction}")
            logger.info(f"Confidence: {decision.confidence:.3f}")

            # 5. Обновление метрик
            if i % 3 == 0:
                metrics = self.decision_reasoner.get_enhanced_metrics()
                logger.info(f"System Metrics: {metrics}")

            # Пауза между циклами
            await asyncio.sleep(0.1)

    async def run_all_examples(self):
        """Запуск всех примеров"""
        logger.info("Starting AI Trading System Examples...")

        try:
            # Запуск отдельных примеров
            await self.run_decision_reasoner_example()
            await asyncio.sleep(1)

            await self.run_transformer_example()
            await asyncio.sleep(1)

            await self.run_meta_learning_example()
            await asyncio.sleep(1)

            await self.run_live_adaptation_example()
            await asyncio.sleep(1)

            await self.run_strategy_evolution_example()
            await asyncio.sleep(1)

            # Интегрированный пример
            await self.run_integrated_example()

            logger.info("All examples completed successfully!")

        except Exception as e:
            logger.error(f"Error running examples: {e}")
            raise


async def main():
    """Основная функция"""
    # Создание и запуск примера
    example = AITradingExample()
    await example.run_all_examples()


if __name__ == "__main__":
    # Настройка логирования
    logger.add(
        "logs/ai_example.log",
        rotation="1 day",
        retention="7 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    # Запуск
    asyncio.run(main())
