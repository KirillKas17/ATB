"""
Система наблюдателей для новостного агента.
"""

from abc import ABC, abstractmethod
from typing import Any, List

from shared.logging import setup_logger

logger = setup_logger(__name__)


class NewsAgentObserver(ABC):
    """Абстрактный класс наблюдателя новостного агента"""

    @abstractmethod
    def notify(self, event: str, data: Any) -> None:
        """Уведомление о событии"""


class PrintNewsObserver(NewsAgentObserver):
    """Наблюдатель для вывода новостей в консоль"""

    def notify(self, event: str, data: Any) -> None:
        """Вывод события в консоль"""
        logger.info(f"News Event [{event}]: {data}")


class NewsObserverManager:
    """Менеджер наблюдателей новостного агента"""

    def __init__(self) -> None:
        self.observers: List[NewsAgentObserver] = []

    def add_observer(self, observer: NewsAgentObserver) -> None:
        """Добавление наблюдателя"""
        if observer not in self.observers:
            self.observers.append(observer)
            logger.debug(f"Added news observer: {type(observer).__name__}")

    def remove_observer(self, observer: NewsAgentObserver) -> None:
        """Удаление наблюдателя"""
        if observer in self.observers:
            self.observers.remove(observer)
            logger.debug(f"Removed news observer: {type(observer).__name__}")

    def notify_observers(self, event: str, data: Any) -> None:
        """Уведомление всех наблюдателей"""
        for observer in self.observers:
            try:
                observer.notify(event, data)
            except Exception as e:
                logger.error(f"Error notifying observer {type(observer).__name__}: {e}")

    def clear_observers(self) -> None:
        """Очистка всех наблюдателей"""
        self.observers.clear()
        logger.debug("Cleared all news observers")
