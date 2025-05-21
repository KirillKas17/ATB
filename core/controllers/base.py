from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from ..models import SystemState


class BaseController:
    """Базовый класс контроллера"""

    def __init__(self):
        self.state = SystemState()
        self.config: Dict[str, Any] = {}
        self.trading_pairs: Dict[str, Any] = {}
        self.active_orders: Dict[str, Any] = {}
        self.positions: Dict[str, Any] = {}
        self.decision_history: list = []

    async def start(self) -> None:
        """Запуск контроллера"""
        try:
            self.state.is_running = True
            self.state.last_update = datetime.now()
            logger.info("Controller started")
        except Exception as e:
            logger.error(f"Error starting controller: {e}")
            raise

    async def stop(self) -> None:
        """Остановка контроллера"""
        try:
            self.state.is_running = False
            self.state.last_update = datetime.now()
            logger.info("Controller stopped")
        except Exception as e:
            logger.error(f"Error stopping controller: {e}")
            raise

    async def update_state(self) -> None:
        """Обновление состояния"""
        try:
            self.state.last_update = datetime.now()
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            raise
