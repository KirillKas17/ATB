"""Оркестратор управления сущностями."""

from typing import Any, Dict

from loguru import logger

from .entity_controller import EntityController


class EntityOrchestrator:
    def __init__(self) -> None:
        self.controllers: Dict[str, EntityController] = {}
        self.is_orchestrating: bool = False
        self.status: str = "idle"

    async def register_controller(
        self, name: str, controller: EntityController
    ) -> None:
        self.controllers[name] = controller
        logger.info(f"Контроллер {name} зарегистрирован в оркестраторе")

    async def start_orchestration(self) -> None:
        if self.is_orchestrating:
            logger.warning("Оркестрация уже запущена")
            return
        self.is_orchestrating = True
        self.status = "running"
        logger.info("Оркестрация запущена")
        for name, controller in self.controllers.items():
            await controller.start()

    async def stop_orchestration(self) -> None:
        if not self.is_orchestrating:
            logger.warning("Оркестрация уже остановлена")
            return
        self.is_orchestrating = False
        self.status = "stopped"
        logger.info("Оркестрация остановлена")
        for name, controller in self.controllers.items():
            await controller.stop()

    def get_orchestration_status(self) -> Dict[str, Any]:
        return {"status": self.status, "controllers": list(self.controllers.keys())}
