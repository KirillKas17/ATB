import random
from typing import Any, Callable, Dict, List

from loguru import logger

from domain.type_definitions.entity_system_types import BaseEvolutionEngine


class EvolutionEngineImpl(BaseEvolutionEngine):
    """
    Продвинутый эволюционный движок:
    - Эволюция популяций с селекцией, кроссовером и мутацией
    - Адаптация сущностей к среде
    - Обучение на данных
    """

    async def evolve(
        self, population: List[Dict[str, Any]], fitness_function: Callable[[Dict[str, Any]], float]
    ) -> List[Dict[str, Any]]:
        logger.info(f"Эволюция популяции: {len(population)} особей")
        # Селекция
        scored = [(entity, fitness_function(entity)) for entity in population]
        scored.sort(key=lambda x: x[1], reverse=True)
        survivors = [e for e, _ in scored[: len(population) // 2]]
        # Кроссовер и мутация
        next_gen: List[Dict[str, Any]] = []
        while len(next_gen) < len(population):
            p1, p2 = random.sample(survivors, 2)
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            next_gen.append(child)
        logger.info("Эволюция завершена")
        return next_gen

    async def adapt(
        self, entity: Dict[str, Any], environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.info(f"Адаптация сущности к среде: {environment}")
        adapted = dict(entity)
        for k, v in environment.items():
            if k in adapted and isinstance(adapted[k], (int, float)):
                adapted[k] += 0.1 * (random.random() - 0.5)
        return adapted

    async def learn(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        logger.info(f"Обучение на {len(data)} примерах")
        # Имитация обучения: усреднение параметров
        if not data:
            return {}
        keys = data[0].keys()
        learned = {
            k: sum(d[k] for d in data if isinstance(d[k], (int, float))) / len(data)
            for k in keys
        }
        return learned

    def _crossover(self, p1: Dict[str, Any], p2: Dict[str, Any]) -> Dict[str, Any]:
        child = dict(p1)
        for k in p2:
            if random.random() > 0.5:
                child[k] = p2[k]
        return child

    def _mutate(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        mutated = dict(entity)
        for k in mutated:
            if isinstance(mutated[k], (int, float)) and random.random() < 0.1:
                mutated[k] += random.uniform(-0.1, 0.1)
        return mutated
