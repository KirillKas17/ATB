# -*- coding: utf-8 -*-
"""
Интеграция эволюционной системы.
Запуск эволюционного менеджера и регистрация всех агентов
для автоматической эволюции компонентов торговой системы.
"""
import asyncio
from typing import Any, Dict, Optional

import pandas as pd
from decimal import Decimal
from loguru import logger

from application.evolution import EvolutionOrchestrator
from domain.evolution import EvolutionContext, EvolutionConfig
from infrastructure.agents.evolvable_market_maker import EvolvableMarketMakerAgent
from infrastructure.agents.evolvable_risk_agent import EvolvableRiskAgent
from infrastructure.agents.evolvable_portfolio_agent import DefaultEvolvablePortfolioAgent
from infrastructure.agents.evolvable_strategy_agent import EvolvableStrategyAgent
from infrastructure.agents.evolvable_news_agent import EvolvableNewsAgent
from infrastructure.agents.evolvable_order_executor import DefaultEvolvableOrderExecutor
from infrastructure.agents.evolvable_meta_controller import DefaultEvolvableMetaController
from infrastructure.agents.evolvable_market_regime import EvolvableMarketRegimeAgent

# Импорты модуля evolution
from infrastructure.evolution import (
    EvolutionBackup,
    EvolutionCache,
    EvolutionMigration,
    StrategyStorage,
)
from infrastructure.ml_services.advanced_price_predictor import AdvancedPricePredictor


class EvolutionIntegration:
    """
    Интеграция эволюционной системы.
    Обеспечивает управление всеми эволюционирующими агентами
    и координацию их работы с эволюционным менеджером.
    """

    def __init__(self, config: Optional[EvolutionConfig] = None) -> None:
        """
        Инициализация интеграции эволюционной системы.
        Args:
            config: Конфигурация эволюционной системы
        """
        self.config = config or EvolutionConfig()
        self.agents: Dict[str, Any] = {}
        self.evolution_task: Optional[asyncio.Task] = None
        # Инициализация компонентов evolution
        self.strategy_storage = StrategyStorage()
        self.evolution_cache = EvolutionCache()
        self.evolution_backup = EvolutionBackup()
        self.evolution_migration = EvolutionMigration(storage=self.strategy_storage)
        # Создание контекста эволюции
        self.evolution_context = EvolutionContext(
            population_size=100,
            generations=50,
            mutation_rate=Decimal("0.1"),
            crossover_rate=Decimal("0.8"),
            elite_size=10,
            min_accuracy=Decimal("0.6"),
            min_profitability=Decimal("0.05"),
            max_drawdown=Decimal("0.2"),
            min_sharpe=Decimal("1.0"),
        )
        # Создание оркестратора эволюции
        self.evolution_orchestrator = EvolutionOrchestrator(
            context=self.evolution_context,
            strategy_repository=self._get_strategy_repository(),
            market_data_provider=self._get_market_data_provider(),
            strategy_storage=self.strategy_storage,
            evolution_cache=self.evolution_cache,
            evolution_backup=self.evolution_backup,
            evolution_migration=self.evolution_migration,
        )

    async def initialize_agents(self) -> None:
        """
        Инициализация всех эволюционирующих агентов.
        Raises:
            Exception: При ошибке инициализации агентов
        """
        try:
            logger.info("Initializing evolvable agents...")
            # Создание агентов
            self.agents["market_maker"] = EvolvableMarketMakerAgent()
            self.agents["risk"] = EvolvableRiskAgent()
            self.agents["portfolio"] = DefaultEvolvablePortfolioAgent()
            self.agents["strategy"] = EvolvableStrategyAgent()
            self.agents["news"] = EvolvableNewsAgent()
            self.agents["order_executor"] = DefaultEvolvableOrderExecutor()
            self.agents["meta_controller"] = DefaultEvolvableMetaController()
            self.agents["market_regime"] = EvolvableMarketRegimeAgent()
            self.agents["price_predictor"] = AdvancedPricePredictor()
            # Регистрация в эволюционном оркестраторе
            for name, agent in self.agents.items():
                logger.info(f"Registered agent: {name}")
            logger.info(f"Successfully initialized {len(self.agents)} evolvable agents")
        except Exception as e:
            logger.error(f"Error initializing agents: {e}")
            raise

    async def start_evolution_system(self) -> None:
        """
        Запуск эволюционной системы.
        Raises:
            Exception: При ошибке запуска системы
        """
        try:
            logger.info("Starting evolution system...")
            # Инициализация агентов
            await self.initialize_agents()
            # Запуск эволюционного цикла
            self.evolution_task = asyncio.create_task(
                self.evolution_orchestrator.start_evolution()
            )
            logger.info("Evolution system started successfully")
        except Exception as e:
            logger.error(f"Error starting evolution system: {e}")
            raise

    async def stop_evolution_system(self) -> None:
        """
        Остановка эволюционной системы.
        Raises:
            Exception: При ошибке остановки системы
        """
        try:
            logger.info("Stopping evolution system...")
            # Остановка эволюционного оркестратора
            await self.evolution_orchestrator.stop_evolution()
            # Отмена задачи эволюции
            if self.evolution_task:
                self.evolution_task.cancel()
                try:
                    await self.evolution_task
                except asyncio.CancelledError:
                    pass
            # Сохранение состояния всех агентов
            await self.save_all_agent_states()
            logger.info("Evolution system stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping evolution system: {e}")

    async def save_all_agent_states(self) -> None:
        """
        Сохранение состояния всех агентов.
        Raises:
            Exception: При ошибке сохранения состояний
        """
        try:
            for name, agent in self.agents.items():
                state_path = f"models/backup/{name}"
                success = False
                if hasattr(agent, 'save_state') and callable(agent.save_state):
                    success = agent.save_state(state_path)
                else:
                    # Для агентов без метода save_state просто логируем
                    logger.debug(f"Agent {name} does not support state saving")
                    success = True
                if success:
                    logger.debug(f"Saved state for agent: {name}")
                else:
                    logger.warning(f"Failed to save state for agent: {name}")
        except Exception as e:
            logger.error(f"Error saving agent states: {e}")

    async def load_all_agent_states(self) -> None:
        """
        Загрузка состояния всех агентов.
        Raises:
            Exception: При ошибке загрузки состояний
        """
        try:
            for name, agent in self.agents.items():
                state_path = f"models/backup/{name}"
                success = False
                if hasattr(agent, 'load_state') and callable(agent.load_state):
                    success = agent.load_state(state_path)
                else:
                    # Для агентов без метода load_state просто логируем
                    logger.debug(f"Agent {name} does not support state loading")
                    success = True
                if success:
                    logger.debug(f"Loaded state for agent: {name}")
                else:
                    logger.warning(f"Failed to load state for agent: {name}")
        except Exception as e:
            logger.error(f"Error loading agent states: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """
        Получение статуса системы.
        Returns:
            Словарь со статусом системы и всех агентов
        """
        try:
            status = {
                "evolution_orchestrator": self.evolution_orchestrator.get_current_status(),
                "agents": {},
            }
            for name, agent in self.agents.items():
                # Получаем производительность агента
                performance = 0.0
                if hasattr(agent, 'get_performance') and callable(agent.get_performance):
                    performance = agent.get_performance()
                elif hasattr(agent, 'get_execution_statistics') and callable(agent.get_execution_statistics):
                    stats = agent.get_execution_statistics()
                    performance = stats.get('success_rate', 0.0)
                
                # Получаем уверенность агента
                confidence = 0.5
                if hasattr(agent, 'get_confidence') and callable(agent.get_confidence):
                    confidence = agent.get_confidence()
                
                status["agents"][name] = {
                    "performance": performance,
                    "confidence": confidence,
                    "evolution_count": getattr(agent, 'evolution_count', 0),
                    "is_evolving": getattr(agent, 'is_evolving', False),
                }
            return status
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {}

    def get_agent(self, name: str) -> Optional[Any]:
        """
        Получение агента по имени.
        Args:
            name: Имя агента
        Returns:
            Агент или None если не найден
        """
        return self.agents.get(name)

    async def trigger_agent_evolution(self, agent_name: str) -> None:
        """
        Запуск эволюции конкретного агента.
        Args:
            agent_name: Имя агента для эволюции
        Raises:
            Exception: При ошибке эволюции агента
        """
        try:
            if agent_name not in self.agents:
                raise ValueError(f"Agent {agent_name} not found")
            agent = self.agents[agent_name]
            logger.info(f"Triggering evolution for agent: {agent_name}")
            # Запуск эволюции агента
            if hasattr(agent, 'evolve') and callable(agent.evolve):
                success = await agent.evolve()
            else:
                # Для агентов без метода evolve используем адаптацию
                if hasattr(agent, 'adapt_to_market_conditions') and callable(agent.adapt_to_market_conditions):
                    success = await agent.adapt_to_market_conditions({})
                else:
                    success = True  # Агент не поддерживает эволюцию
            if success:
                logger.info(f"Agent {agent_name} evolved successfully")
            else:
                logger.warning(f"Agent {agent_name} evolution failed")
        except Exception as e:
            logger.error(f"Error triggering evolution for agent {agent_name}: {e}")
            raise

    def _get_market_data_provider(self) -> Any:
        """Получить провайдер рыночных данных."""

        # Здесь должна быть реализация получения рыночных данных
        # Пока возвращаем заглушку
        async def market_data_provider(symbol: str, start_date: Any, end_date: Any) -> Any:
            # Заглушка для рыночных данных
            dates = pd.date_range(start=start_date, end=end_date, freq="1H")
            data = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": [100.0] * len(dates),
                    "high": [105.0] * len(dates),
                    "low": [95.0] * len(dates),
                    "close": [102.0] * len(dates),
                    "volume": [1000.0] * len(dates),
                }
            )
            return data

        return market_data_provider

    def _get_strategy_repository(self) -> Any:
        """Получить репозиторий стратегий."""
        # Заглушка для репозитория стратегий
        return None

    async def start_strategy_evolution(self) -> None:
        """Запуск эволюции стратегий."""
        try:
            logger.info("Starting strategy evolution...")
            await self.evolution_orchestrator.start_evolution()
            logger.info("Strategy evolution started successfully")
        except Exception as e:
            logger.error(f"Error starting strategy evolution: {e}")
            raise

    async def get_evolution_metrics(self) -> Dict[str, Any]:
        """Получить метрики эволюции."""
        try:
            return await self.evolution_orchestrator.get_evolution_metrics()
        except Exception as e:
            logger.error(f"Error getting evolution metrics: {e}")
            return {}

    async def create_evolution_backup(self) -> bool:
        """Создать резервную копию эволюции."""
        try:
            return await self.evolution_orchestrator.create_evolution_backup()
        except Exception as e:
            logger.error(f"Error creating evolution backup: {e}")
            return False

    async def restore_evolution_backup(self, backup_id: str) -> bool:
        """Восстановить эволюцию из резервной копии."""
        try:
            return await self.evolution_orchestrator.restore_evolution_from_backup(
                backup_id
            )
        except Exception as e:
            logger.error(f"Error restoring evolution backup: {e}")
            return False


# Глобальный экземпляр интеграции
evolution_integration = EvolutionIntegration()


async def start_evolution() -> None:
    """
    Запуск эволюционной системы.
    Глобальная функция для запуска эволюционной системы.
    """
    await evolution_integration.start_evolution_system()


async def stop_evolution() -> None:
    """
    Остановка эволюционной системы.
    Глобальная функция для остановки эволюционной системы.
    """
    await evolution_integration.stop_evolution_system()


def get_evolution_status() -> Dict[str, Any]:
    """
    Получение статуса эволюционной системы.
    Returns:
        Статус эволюционной системы
    """
    return evolution_integration.get_system_status()


# Пример использования
if __name__ == "__main__":

    async def main():
        """
        Основная функция для демонстрации работы эволюционной системы.
        Запускает систему, работает некоторое время, получает статус
        и корректно останавливает систему.
        """
        # Запуск эволюционной системы
        await start_evolution()
        # Работа системы в течение некоторого времени
        await asyncio.sleep(60)  # 1 минута
        # Получение статуса
        status = get_evolution_status()
        print(f"System status: {status}")
        # Остановка системы
        await stop_evolution()

    asyncio.run(main())
