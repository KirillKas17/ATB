"""
Менеджер эволюции стратегий
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict, Union

from loguru import logger

from domain.types.strategy_types import EvolutionConfig, EvolutionMetrics


class EvolutionRecord(TypedDict):
    strategy_id: str
    type: str
    params: Dict[str, Any]
    start_time: datetime
    status: str
    end_time: Optional[datetime]


class ActiveEvolution(TypedDict, total=False):
    strategy_id: str
    type: str
    params: Dict[str, Any]
    start_time: datetime
    status: str
    end_time: Optional[datetime]


class EvolutionManager:
    """Менеджер эволюции стратегий"""

    def __init__(self, config: Optional[EvolutionConfig] = None):
        self.config = config or EvolutionConfig()
        self.evolution_history: List[EvolutionRecord] = []
        self.active_evolutions: Dict[str, ActiveEvolution] = {}

    async def start_evolution(
        self, strategy_id: str, evolution_type: str, params: Dict[str, Any]
    ) -> bool:
        """Запуск эволюции стратегии"""
        try:
            evolution_id = (
                f"{strategy_id}_{evolution_type}_{datetime.now().timestamp()}"
            )

            self.active_evolutions[evolution_id] = {
                "strategy_id": strategy_id,
                "type": evolution_type,
                "params": params,
                "start_time": datetime.now(),
                "status": "running",
            }

            # Запуск эволюции в отдельной задаче
            asyncio.create_task(self._run_evolution(evolution_id))

            logger.info(f"Started evolution {evolution_id} for strategy {strategy_id}")
            return True

        except Exception as e:
            logger.error(f"Error starting evolution: {str(e)}")
            return False

    async def _run_evolution(self, evolution_id: str) -> None:
        """Выполнение эволюции"""
        try:
            evolution = self.active_evolutions[evolution_id]

            # Симуляция эволюционного процесса
            await asyncio.sleep(5)  # Placeholder

            # Завершение эволюции
            evolution["status"] = "completed"
            evolution["end_time"] = datetime.now()

            # Сохранение в историю с правильной типизацией
            evolution_record: EvolutionRecord = {
                "strategy_id": evolution["strategy_id"],
                "type": evolution["type"],
                "params": evolution["params"],
                "start_time": evolution["start_time"],
                "status": evolution["status"],
                "end_time": evolution.get("end_time"),
            }
            self.evolution_history.append(evolution_record)

            logger.info(f"Evolution {evolution_id} completed")

        except Exception as e:
            logger.error(f"Error running evolution {evolution_id}: {str(e)}")
            if evolution_id in self.active_evolutions:
                self.active_evolutions[evolution_id]["status"] = "failed"

    def get_evolution_status(self, evolution_id: str) -> Optional[ActiveEvolution]:
        """Получение статуса эволюции"""
        return self.active_evolutions.get(evolution_id)

    def get_evolution_history(
        self, strategy_id: Optional[str] = None
    ) -> List[EvolutionRecord]:
        """Получение истории эволюций"""
        if strategy_id:
            return [e for e in self.evolution_history if e["strategy_id"] == strategy_id]
        return self.evolution_history

    def get_evolution_metrics(self) -> EvolutionMetrics:
        """Получение метрик эволюции"""
        try:
            total_evolutions = len(self.evolution_history)
            successful_evolutions = len(
                [e for e in self.evolution_history if e["status"] == "completed"]
            )
            failed_evolutions = len(
                [e for e in self.evolution_history if e["status"] == "failed"]
            )

            success_rate = (
                successful_evolutions / total_evolutions
                if total_evolutions > 0
                else 0.0
            )

            return EvolutionMetrics(
                generation=total_evolutions,
                best_fitness=success_rate,
                avg_fitness=success_rate,
                diversity=1.0 - success_rate,
                convergence_rate=success_rate,
                improvement_rate=success_rate,
                adaptation_success=success_rate,
            )

        except Exception as e:
            logger.error(f"Error getting evolution metrics: {str(e)}")
            return EvolutionMetrics(
                generation=0,
                best_fitness=0.0,
                avg_fitness=0.0,
                diversity=0.0,
                convergence_rate=0.0,
                improvement_rate=0.0,
                adaptation_success=0.0,
            )
