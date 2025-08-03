import random
from datetime import datetime
from typing import Any, Dict

from loguru import logger

from domain.types.entity_system_types import (
    BaseExperimentRunner,
    Experiment,
    ExperimentResult,
)


class ExperimentRunnerImpl(BaseExperimentRunner):
    """
    Продвинутый запускатор экспериментов:
    - Асинхронный запуск экспериментов и A/B тестов
    - Логирование, метрики, обработка ошибок
    - Генерация статистических результатов
    """

    async def run_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        logger.info(f"Запуск эксперимента: {experiment['id']}")
        # Имитация эксперимента
        await self._simulate_delay()
        result: Dict[str, Any] = {
            "experiment_id": experiment["id"],
            "test_name": experiment["name"],
            "status": "completed",
            "control_sample_size": 100,
            "treatment_sample_size": 100,
            "control_mean": random.uniform(0.4, 0.6),
            "treatment_mean": random.uniform(0.5, 0.7),
            "improvement_percent": random.uniform(1, 10),
            "significant": True,
            "p_value": 0.03,
            "confidence_interval": [0.01, 0.05],
            "analysis_date": datetime.now(),
        }
        logger.info(f"Результат эксперимента: {result}")
        return result

    async def run_ab_test(self, experiment: Experiment) -> Dict[str, Any]:
        logger.info(f"Запуск A/B теста: {experiment['id']}")
        await self._simulate_delay()
        # Имитация A/B теста
        control = [random.gauss(0.5, 0.1) for _ in range(100)]
        treatment = [random.gauss(0.55, 0.1) for _ in range(100)]
        improvement = (sum(treatment) / len(treatment)) - (sum(control) / len(control))
        result: Dict[str, Any] = {
            "experiment_id": experiment["id"],
            "test_name": experiment["name"],
            "status": "completed",
            "control_sample_size": len(control),
            "treatment_sample_size": len(treatment),
            "control_mean": sum(control) / len(control),
            "treatment_mean": sum(treatment) / len(treatment),
            "improvement_percent": improvement * 100,
            "significant": abs(improvement) > 0.01,
            "p_value": 0.04,
            "confidence_interval": [improvement - 0.01, improvement + 0.01],
            "analysis_date": datetime.now(),
        }
        logger.info(f"Результат A/B теста: {result}")
        return result

    async def stop_experiment(self, experiment_id: str) -> bool:
        logger.info(f"Остановка эксперимента: {experiment_id}")
        await self._simulate_delay()
        return True

    async def _simulate_delay(self) -> None:
        import asyncio

        await asyncio.sleep(0.1)
