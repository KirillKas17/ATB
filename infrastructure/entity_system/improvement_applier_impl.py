import random

from loguru import logger

from domain.types.entity_system_types import BaseImprovementApplier, Improvement


class ImprovementApplierImpl(BaseImprovementApplier):
    """
    Продвинутый применятель улучшений:
    - Применяет улучшения с логированием и проверками
    - Поддерживает откат и валидацию
    """

    async def apply_improvement(self, improvement: Improvement) -> bool:
        logger.info(
            f"Применение улучшения: {improvement['id']} - {improvement['name']}"
        )
        # Имитация применения
        if random.random() > 0.05:
            logger.info(f"Улучшение успешно применено: {improvement['id']}")
            return True
        else:
            logger.error(f"Ошибка применения улучшения: {improvement['id']}")
            return False

    async def rollback_improvement(self, improvement_id: str) -> bool:
        logger.info(f"Откат улучшения: {improvement_id}")
        # Имитация отката
        if random.random() > 0.1:
            logger.info(f"Улучшение успешно откатено: {improvement_id}")
            return True
        else:
            logger.error(f"Ошибка отката улучшения: {improvement_id}")
            return False

    async def validate_improvement(self, improvement: Improvement) -> bool:
        logger.info(f"Валидация улучшения: {improvement['id']}")
        # Пример: валидация по категории и наличию rollback-плана
        valid = improvement["category"] in [
            "performance",
            "maintainability",
            "architecture",
        ] and bool(improvement.get("rollback_plan"))
        if valid:
            logger.info(f"Улучшение валидно: {improvement['id']}")
        else:
            logger.warning(f"Улучшение не прошло валидацию: {improvement['id']}")
        return valid
