"""
Модуль для модификаторов торговой системы.
"""

from loguru import logger


class Modifiers:
    """Класс для применения модификаторов к торговым решениям."""

    def __init__(self, orchestrator):
        """Инициализация модификаторов."""
        self.orchestrator = orchestrator

    async def apply_analytical_integration_modifier(self, symbol: str) -> None:
        """Применение модификатора аналитической интеграции."""
        try:
            if self.orchestrator.analytical_integration:
                await self.orchestrator.analytical_integration.apply_modifier(symbol)
        except Exception as e:
            logger.error(f"Error applying analytical integration modifier: {e}")

    async def apply_entanglement_integration_modifier(self, symbol: str) -> None:
        """Применение модификатора интеграции запутанности."""
        try:
            if self.orchestrator.entanglement_integration:
                await self.orchestrator.entanglement_integration.apply_modifier(symbol)
        except Exception as e:
            logger.error(f"Error applying entanglement integration modifier: {e}")

    async def apply_agent_order_executor_modifier(self, symbol: str) -> None:
        """Применение модификатора агента-исполнителя ордеров."""
        try:
            if self.orchestrator.agent_order_executor:
                await self.orchestrator.agent_order_executor.apply_modifier(symbol)
        except Exception as e:
            logger.error(f"Error applying agent order executor modifier: {e}")

    async def apply_agent_market_regime_modifier(self, symbol: str) -> None:
        """Применение модификатора агента рыночного режима."""
        try:
            if self.orchestrator.agent_market_regime:
                await self.orchestrator.agent_market_regime.apply_modifier(symbol)
        except Exception as e:
            logger.error(f"Error applying agent market regime modifier: {e}")

    async def apply_agent_market_maker_model_modifier(self, symbol: str) -> None:
        """Применение модификатора модели агента маркет-мейкера."""
        try:
            if self.orchestrator.agent_market_maker_model:
                await self.orchestrator.agent_market_maker_model.apply_modifier(symbol)
        except Exception as e:
            logger.error(f"Error applying agent market maker model modifier: {e}")

    async def apply_sandbox_trainer_modifier(self, symbol: str) -> None:
        """Применение модификатора песочницы для обучения."""
        try:
            if self.orchestrator.sandbox_trainer:
                await self.orchestrator.sandbox_trainer.apply_modifier(symbol)
        except Exception as e:
            logger.error(f"Error applying sandbox trainer modifier: {e}")
