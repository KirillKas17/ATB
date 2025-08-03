"""
Автоматический менеджер миграции агентов.
Управляет переходом от классических к эволюционным агентам на основе производительности.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from infrastructure.core.evolution_integration import evolution_integration
from infrastructure.services.market_conditions_analyzer import MarketConditionsAnalyzer


@dataclass
class MigrationConfig:
    # Пороги для миграции
    performance_threshold: float = (
        0.7  # Минимальная производительность для эволюционных агентов
    )
    confidence_threshold: float = 0.6  # Минимальная уверенность
    stability_period: int = 3600  # Период стабильности в секундах (1 час)
    migration_cooldown: int = 1800  # Кулдаун между миграциями (30 минут)
    # Веса для принятия решений
    performance_weight: float = 0.4
    confidence_weight: float = 0.3
    market_volatility_weight: float = 0.2
    time_weight: float = 0.1
    # Автоматические настройки
    auto_migrate: bool = True
    fallback_on_failure: bool = True
    gradual_migration: bool = True


class AutoMigrationManager:
    """
    Автоматический менеджер миграции
    Принимает решения о миграции на основе производительности, уверенности и рыночных условий
    """

    def __init__(self, config: Optional[MigrationConfig] = None) -> None:
        self.config = config or MigrationConfig()
        self.migration_history: List[Dict[str, Any]] = []
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.last_migration_time: Optional[datetime] = None
        self.migration_task: Optional[asyncio.Task[Any]] = None
        self.is_running = False
        
        # Инициализация анализатора рыночных условий
        # Примечание: зависимости будут внедрены через DI контейнер
        self.market_conditions_analyzer: Optional[MarketConditionsAnalyzer] = None

    async def start(self) -> None:
        """Запуск автоматического менеджера миграции"""
        try:
            logger.info("Starting Auto Migration Manager...")
            self.is_running = True
            
            # Инициализация анализатора рыночных условий
            await self._initialize_market_conditions_analyzer()
            
            # Запуск цикла миграции
            self.migration_task = asyncio.create_task(self._migration_loop())
            logger.info("Auto Migration Manager started successfully")
        except Exception as e:
            logger.error(f"Error starting Auto Migration Manager: {e}")
            raise

    async def _initialize_market_conditions_analyzer(self) -> None:
        """Инициализация анализатора рыночных условий."""
        try:
            # В реальной реализации здесь будет внедрение через DI контейнер
            # Пока создаем заглушку для демонстрации
            logger.info("Initializing market conditions analyzer...")
            
            # Примечание: в продакшене это будет внедрено через DI контейнер
            # self.market_conditions_analyzer = container.get(MarketConditionsAnalyzer)
            
            logger.info("Market conditions analyzer initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing market conditions analyzer: {e}")
            # Не прерываем работу менеджера, используем fallback

    async def stop(self) -> None:
        """Остановка автоматического менеджера миграции"""
        try:
            logger.info("Stopping Auto Migration Manager...")
            self.is_running = False
            if self.migration_task:
                self.migration_task.cancel()
                try:
                    await self.migration_task
                except asyncio.CancelledError:
                    pass
            logger.info("Auto Migration Manager stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping Auto Migration Manager: {e}")

    async def _migration_loop(self) -> None:
        """Основной цикл миграции"""
        while self.is_running:
            try:
                # Анализ текущего состояния
                await self._analyze_current_state()
                # Принятие решений о миграции
                await self._make_migration_decisions()
                # Обновление истории
                await self._update_migration_history()
                # Пауза между циклами
                await asyncio.sleep(300)  # 5 минут
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in migration loop: {e}")
                await asyncio.sleep(60)

    async def _analyze_current_state(self) -> None:
        """Анализ текущего состояния системы"""
        try:
            # Получение статуса эволюционной системы
            evolution_status = evolution_integration.get_system_status()
            # Анализ каждого агента
            for agent_name, agent_data in evolution_status.get("agents", {}).items():
                performance = agent_data.get("performance", 0.0)
                confidence = agent_data.get("confidence", 0.0)
                evolution_count = agent_data.get("evolution_count", 0)
                # Сохранение состояния агента
                self.agent_states[agent_name] = {
                    "performance": performance,
                    "confidence": confidence,
                    "evolution_count": evolution_count,
                    "timestamp": datetime.now(),
                    "is_evolving": agent_data.get("is_evolving", False),
                }
                logger.debug(
                    f"Agent {agent_name}: performance={performance:.3f}, "
                    f"confidence={confidence:.3f}"
                )
        except Exception as e:
            logger.error(f"Error analyzing current state: {e}")

    async def _make_migration_decisions(self) -> None:
        """Принятие решений о миграции"""
        try:
            # Проверка кулдауна
            if (
                self.last_migration_time
                and datetime.now() - self.last_migration_time
                < timedelta(seconds=self.config.migration_cooldown)
            ):
                return
            # Анализ каждого агента
            for agent_name, agent_state in self.agent_states.items():
                migration_decision = await self._evaluate_agent_migration(
                    agent_name, agent_state
                )
                if migration_decision["should_migrate"]:
                    await self._execute_migration(agent_name, migration_decision)
        except Exception as e:
            logger.error(f"Error making migration decisions: {e}")

    async def _evaluate_agent_migration(
        self, agent_name: str, agent_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Оценка необходимости миграции для конкретного агента
        Args:
            agent_name: Имя агента для оценки
            agent_state: Состояние агента
        Returns:
            Dict[str, Any]: Решение о миграции
        """
        try:
            performance = agent_state.get("performance", 0.0)
            confidence = agent_state.get("confidence", 0.0)
            evolution_count = agent_state.get("evolution_count", 0)
            # Базовые критерии
            performance_ok = performance >= self.config.performance_threshold
            confidence_ok = confidence >= self.config.confidence_threshold
            evolution_ready = evolution_count >= 3  # Минимум 3 эволюции
            # Анализ стабильности
            stability_score = self._calculate_stability_score(agent_name)
            # Анализ рыночных условий
            market_score = await self._calculate_market_score()
            # Временной фактор
            time_score = self._calculate_time_score(agent_state.get("timestamp"))
            # Общий скор миграции
            migration_score = (
                self.config.performance_weight * (1.0 if performance_ok else 0.0)
                + self.config.confidence_weight * (1.0 if confidence_ok else 0.0)
                + self.config.market_volatility_weight * market_score
                + self.config.time_weight * time_score
            )
            # Принятие решения
            should_migrate = (
                performance_ok
                and confidence_ok
                and evolution_ready
                and stability_score > 0.7
                and migration_score > 0.6
            )
            return {
                "should_migrate": should_migrate,
                "migration_score": migration_score,
                "performance_ok": performance_ok,
                "confidence_ok": confidence_ok,
                "evolution_ready": evolution_ready,
                "stability_score": stability_score,
                "market_score": market_score,
                "time_score": time_score,
                "reason": self._get_migration_reason(
                    should_migrate, performance_ok, confidence_ok, evolution_ready
                ),
            }
        except Exception as e:
            logger.error(f"Error evaluating migration for {agent_name}: {e}")
            return {"should_migrate": False, "reason": f"Error: {e}"}

    def _calculate_stability_score(self, agent_name: str) -> float:
        """
        Расчет скора стабильности агента
        Args:
            agent_name: Имя агента
        Returns:
            float: Скор стабильности от 0 до 1
        """
        try:
            # Анализ истории производительности
            history = [
                state
                for state in self.migration_history
                if state.get("agent_name") == agent_name
            ]
            if len(history) < 5:
                return 0.5  # Недостаточно данных
            recent_performances = [h.get("performance", 0.0) for h in history[-5:]]
            # Стабильность = 1 - стандартное отклонение
            stability = 1.0 - np.std(recent_performances)
            return max(0.0, min(1.0, stability))
        except Exception as e:
            logger.error(f"Error calculating stability score: {e}")
            return 0.5

    async def _calculate_market_score(self) -> float:
        """
        Расчет скора рыночных условий
        Returns:
            float: Скор рыночных условий от 0 до 1
        """
        try:
            # Используем промышленный анализатор рыночных условий
            if self.market_conditions_analyzer is None:
                logger.warning("Market conditions analyzer not initialized, using fallback")
                return 0.7
            
            # Анализируем основные торговые пары
            symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
            total_score = 0.0
            valid_symbols = 0
            
            for symbol in symbols:
                try:
                    market_score = await self.market_conditions_analyzer.calculate_market_score(
                        symbol=symbol,
                        timeframe="1h",
                        lookback_periods=100
                    )
                    total_score += market_score.overall_score
                    valid_symbols += 1
                    
                    logger.debug(f"Market score for {symbol}: {market_score.overall_score:.3f} "
                               f"({market_score.condition_type.value})")
                    
                except Exception as e:
                    logger.warning(f"Error analyzing market conditions for {symbol}: {e}")
                    continue
            
            # Возвращаем средний скор по всем символам
            if valid_symbols > 0:
                average_score = total_score / valid_symbols
                logger.debug(f"Average market score: {average_score:.3f}")
                return average_score
            else:
                logger.warning("No valid market scores calculated, using fallback")
                return 0.7
                
        except Exception as e:
            logger.error(f"Error calculating market score: {e}")
            return 0.5

    def _calculate_time_score(self, last_update: Optional[datetime]) -> float:
        """
        Расчет временного скора
        Args:
            last_update: Время последнего обновления
        Returns:
            float: Временной скор от 0 до 1
        """
        try:
            if not last_update:
                return 0.0
            time_diff = datetime.now() - last_update
            hours_diff = time_diff.total_seconds() / 3600
            # Скор растет со временем (больше времени = больше данных)
            time_score = min(1.0, hours_diff / 24)  # Нормализация к 24 часам
            return time_score
        except Exception as e:
            logger.error(f"Error calculating time score: {e}")
            return 0.0

    def _get_migration_reason(
        self,
        should_migrate: bool,
        performance_ok: bool,
        confidence_ok: bool,
        evolution_ready: bool,
    ) -> str:
        """
        Получение причины решения о миграции
        Args:
            should_migrate: Следует ли мигрировать
            performance_ok: Достаточна ли производительность
            confidence_ok: Достаточна ли уверенность
            evolution_ready: Готов ли к эволюции
        Returns:
            str: Причина решения
        """
        if should_migrate:
            return "All criteria met - ready for migration"
        else:
            reasons = []
            if not performance_ok:
                reasons.append("performance below threshold")
            if not confidence_ok:
                reasons.append("confidence below threshold")
            if not evolution_ready:
                reasons.append("insufficient evolution count")
            return f"Migration blocked: {', '.join(reasons)}"

    async def _execute_migration(self, agent_name: str, decision: Dict[str, Any]) -> None:
        """
        Выполнение миграции
        Args:
            agent_name: Имя агента для миграции
            decision: Решение о миграции
        """
        try:
            logger.info(f"Executing migration for {agent_name}: {decision['reason']}")
            # Принудительная эволюция агента
            await evolution_integration.trigger_agent_evolution(agent_name)
            if success:
                # Обновление времени последней миграции
                self.last_migration_time = datetime.now()
                # Сохранение в историю
                migration_record = {
                    "agent_name": agent_name,
                    "timestamp": datetime.now(),
                    "migration_score": decision["migration_score"],
                    "reason": decision["reason"],
                    "success": True,
                }
                self.migration_history.append(migration_record)
                logger.info(f"Migration completed successfully for {agent_name}")
            else:
                logger.warning(f"Migration failed for {agent_name}")
        except Exception as e:
            logger.error(f"Error executing migration for {agent_name}: {e}")

    async def _handle_migration_failure(self, agent_name: str) -> None:
        """
        Обработка неудачной миграции
        Args:
            agent_name: Имя агента с неудачной миграцией
        """
        try:
            logger.warning(f"Handling migration failure for {agent_name}")
            # Здесь можно добавить логику fallback к классическим агентам
            # или другие методы восстановления
        except Exception as e:
            logger.error(f"Error handling migration failure: {e}")

    async def _update_migration_history(self) -> None:
        """Обновление истории миграций"""
        try:
            # Ограничение размера истории
            if len(self.migration_history) > 1000:
                self.migration_history = self.migration_history[-1000:]
        except Exception as e:
            logger.error(f"Error updating migration history: {e}")

    def get_migration_status(self) -> Dict[str, Any]:
        """
        Получение статуса миграции
        Returns:
            Dict[str, Any]: Статус миграции
        """
        try:
            return {
                "is_running": self.is_running,
                "last_migration_time": (
                    self.last_migration_time.isoformat()
                    if self.last_migration_time
                    else None
                ),
                "agent_states": self.agent_states,
                "recent_migrations": self.migration_history[
                    -10:
                ],  # Последние 10 миграций
                "total_migrations": len(self.migration_history),
            }
        except Exception as e:
            logger.error(f"Error getting migration status: {e}")
            return {}

    async def force_migration(self, agent_name: str) -> None:
        """
        Принудительная миграция агента
        Args:
            agent_name: Имя агента для принудительной миграции
        """
        try:
            logger.info(f"Forcing migration for {agent_name}")
            decision = {
                "should_migrate": True,
                "migration_score": 1.0,
                "reason": "Forced migration",
            }
            await self._execute_migration(agent_name, decision)
        except Exception as e:
            logger.error(f"Error forcing migration for {agent_name}: {e}")

    def set_market_conditions_analyzer(self, analyzer: MarketConditionsAnalyzer) -> None:
        """
        Установка анализатора рыночных условий.
        Args:
            analyzer: Анализатор рыночных условий
        """
        self.market_conditions_analyzer = analyzer
        logger.info("Market conditions analyzer set successfully")

    def get_agent_recommendations(self) -> Dict[str, str]:
        """
        Получение рекомендаций по агентам
        Returns:
            Dict[str, str]: Рекомендации по агентам
        """
        try:
            recommendations = {}
            for agent_name, agent_state in self.agent_states.items():
                performance = agent_state.get("performance", 0.0)
                confidence = agent_state.get("confidence", 0.0)
                if performance < 0.5:
                    recommendations[agent_name] = "Low performance - needs optimization"
                elif confidence < 0.5:
                    recommendations[agent_name] = "Low confidence - needs training"
                else:
                    recommendations[agent_name] = "Performing well"
            return recommendations
        except Exception as e:
            logger.error(f"Error getting agent recommendations: {e}")
            return {}


# Глобальный экземпляр менеджера миграции
auto_migration_manager = AutoMigrationManager()


async def start_auto_migration() -> None:
    """Запуск автоматической миграции"""
    await auto_migration_manager.start()


async def stop_auto_migration() -> None:
    """Остановка автоматической миграции"""
    await auto_migration_manager.stop()


def get_migration_status() -> Dict[str, Any]:
    """
    Получение статуса миграции
    Returns:
        Dict[str, Any]: Статус миграции
    """
    return auto_migration_manager.get_migration_status()


def get_agent_recommendations() -> Dict[str, str]:
    """
    Получение рекомендаций по агентам
    Returns:
        Dict[str, str]: Рекомендации по агентам
    """
    return auto_migration_manager.get_agent_recommendations()


# Интеграция с основной системой
async def integrate_with_main_system() -> None:
    """Интеграция с основной системой"""
    try:
        # Запуск автоматической миграции
        await start_auto_migration()
        # Интеграция с эволюционной системой
        await evolution_integration.start_evolution_system()
        logger.info("Auto migration integrated with main system")
    except Exception as e:
        logger.error(f"Error integrating auto migration: {e}")
        raise
