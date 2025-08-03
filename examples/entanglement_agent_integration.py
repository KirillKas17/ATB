# -*- coding: utf-8 -*-
"""
Пример интеграции системы Quantum Order Entanglement Detection с агентской системой.

Этот пример демонстрирует:
1. Интеграцию с MetaControllerAgent
2. Применение модификаторов запутанности к торговым сигналам
3. Мониторинг и статистику запутанности
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger

from infrastructure.agents.agent_meta_controller import MetaControllerAgent
from infrastructure.agents.entanglement_integration import (
    EntanglementIntegration, apply_entanglement_to_signal,
    get_entanglement_statistics)


class EntanglementAwareMetaController(MetaControllerAgent):
    """Мета-контроллер с поддержкой обнаружения запутанности."""

    def __init__(self, config=None):
        super().__init__(config)
        self.entanglement_integration = EntanglementIntegration(
            enabled=True, impact_threshold=0.3
        )

    async def initialize(self) -> None:
        """Инициализация с запуском мониторинга запутанности."""
        # Запускаем мониторинг запутанности
        await self.entanglement_integration.start()
        logger.info("Entanglement monitoring started in MetaController")

    async def evaluate_strategies(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Оценка стратегий с учетом запутанности."""
        # Базовая реализация - возвращаем None
        return None

    async def coordinate_agents(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Координация агентов."""
        return await super().coordinate_agents(context)  # type: ignore

    async def add_agent(self, agent_id: str, agent: Any) -> bool:
        """Добавление агента."""
        return await super().add_agent(agent_id, agent)  # type: ignore

    async def remove_agent(self, agent_id: str) -> bool:
        """Удаление агента."""
        return await super().remove_agent(agent_id)  # type: ignore

    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Получение статуса агента."""
        return await super().get_agent_status(agent_id)  # type: ignore

    def get_entanglement_insights(self, symbol: str) -> Dict[str, Any]:
        """Получение инсайтов по запутанности для символа."""
        status = self.entanglement_integration.get_entanglement_status(symbol)

        if not status:
            return {"has_entanglement": False}

        insights = {
            "has_entanglement": status["is_entangled"],
            "exchange_pair": status["exchange_pair"],
            "lag_ms": status["lag_ms"],
            "correlation_score": status["correlation_score"],
            "impact_on_trading": status["impact_on_decision"],
            "recommendation": self._get_entanglement_recommendation(status),
        }

        return insights

    def _get_entanglement_recommendation(self, status: Dict[str, Any]) -> str:
        """Получение рекомендации на основе запутанности."""
        if not status["is_entangled"]:
            return "normal_trading"

        impact = status["impact_on_decision"]
        correlation = status["correlation_score"]
        lag = status["lag_ms"]

        if impact > 0.5:
            return "reduce_position_size"
        elif correlation > 0.98 and lag < 2.0:
            return "avoid_trading"
        elif correlation > 0.95:
            return "increase_caution"
        else:
            return "normal_trading"


async def main():
    """Основная функция примера."""
    logger.info("=== Entanglement Agent Integration Example ===")

    # Создаем мета-контроллер с поддержкой запутанности
    controller = EntanglementAwareMetaController(
        {
            "min_win_rate": 0.55,
            "min_profit_factor": 1.5,
            "min_sharpe": 1.0,
            "min_trades": 100,
            "retrain_interval": 24,
            "max_drawdown": 0.15,
            "confidence_threshold": 0.7,
        }
    )

    # Инициализируем контроллер
    await controller.initialize()

    # Симулируем торговые пары
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]

    try:
        # Мониторим в течение 60 секунд
        logger.info("Starting 60-second monitoring session...")

        for i in range(60):
            await asyncio.sleep(1)

            # Каждые 10 секунд выводим статистику
            if (i + 1) % 10 == 0:
                logger.info(f"=== Status at {i+1}s ===")

                # Статистика запутанности
                stats = get_entanglement_statistics()
                logger.info(f"Entanglement Statistics: {stats}")

                # Анализ каждой пары
                for symbol in symbols:
                    insights = controller.get_entanglement_insights(symbol)
                    logger.info(f"{symbol} insights: {insights}")

                    # Симулируем торговый сигнал
                    fake_signal = {
                        "action": "buy",
                        "confidence": 0.8,
                        "position_size": 0.1,
                        "stop_loss": 45000,
                        "take_profit": 55000,
                        "source": "strategy",
                        "timestamp": datetime.now(),
                        "explanation": "Technical analysis signal",
                    }

                    # Применяем модификатор запутанности
                    modified_signal = apply_entanglement_to_signal(fake_signal, symbol)

                    if modified_signal.get("entanglement_warning"):
                        logger.warning(
                            f"Signal modified for {symbol}: "
                            f"confidence {fake_signal['confidence']:.3f} → "
                            f"{modified_signal['confidence']:.3f}"
                        )

        # Финальная статистика
        logger.info("=== Final Statistics ===")
        final_stats = get_entanglement_statistics()
        logger.info(f"Final entanglement statistics: {final_stats}")

        # Анализ всех контекстов
        all_contexts = (
            controller.entanglement_integration.get_all_entanglement_contexts()
        )
        logger.info(f"All entanglement contexts: {all_contexts}")

    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")

    finally:
        # Останавливаем мониторинг
        await controller.entanglement_integration.stop_monitoring()
        logger.info("Entanglement monitoring stopped")

    logger.info("=== Example completed ===")


def test_entanglement_modification():
    """Тест модификации сигналов запутанностью."""
    logger.info("=== Testing Entanglement Signal Modification ===")

    # Создаем тестовый сигнал
    test_signal = {
        "action": "buy",
        "confidence": 0.85,
        "position_size": 0.2,
        "stop_loss": 45000,
        "take_profit": 55000,
        "source": "strategy",
        "timestamp": datetime.now(),
        "explanation": "Test signal",
    }

    # Тестируем без запутанности
    modified_signal = apply_entanglement_to_signal(test_signal, "BTCUSDT")
    assert (
        modified_signal == test_signal
    ), "Signal should not be modified without entanglement"

    logger.info("Signal modification test passed")


def test_entanglement_integration():
    """Тест интеграции запутанности."""
    logger.info("=== Testing Entanglement Integration ===")

    # Создаем интеграцию
    integration = EntanglementIntegration(enabled=True)

    # Проверяем статистику
    stats = integration.get_entanglement_statistics()
    assert stats["enabled"] == True, "Integration should be enabled"

    # Проверяем статус для несуществующего символа
    status = integration.get_entanglement_status("NONEXISTENT")
    assert status is None, "Status should be None for non-existent symbol"

    logger.info("Entanglement integration test passed")


if __name__ == "__main__":
    # Запускаем тесты
    test_entanglement_modification()
    test_entanglement_integration()

    # Запускаем основной пример
    asyncio.run(main())
