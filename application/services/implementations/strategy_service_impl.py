"""
Реализация сервиса стратегий для торговой системы.
Обеспечивает управление жизненным циклом торговых стратегий.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from domain.entities.strategy import AbstractStrategy, StrategyState, StrategyStatus
from domain.interfaces.strategy_service import StrategyService
from domain.repositories.strategy_repository import StrategyRepository
from domain.value_objects.strategy_id import StrategyId
from domain.exceptions import StrategyExecutionError, ValidationError
from shared.logging import get_logger


class StrategyServiceImpl(StrategyService):
    """Реализация сервиса управления торговыми стратегиями."""
    
    def __init__(
        self,
        strategy_repository: StrategyRepository,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Инициализация сервиса стратегий.
        
        Args:
            strategy_repository: Репозиторий для работы со стратегиями
            config: Конфигурация сервиса
        """
        self._repository = strategy_repository
        self._config = config or {}
        self._logger = get_logger(__name__)
        self._active_strategies: Dict[StrategyId, AbstractStrategy] = {}
        
    async def create_strategy(
        self,
        strategy_data: Dict[str, Any]
    ) -> AbstractStrategy:
        """
        Создание новой торговой стратегии.
        
        Args:
            strategy_data: Данные для создания стратегии
            
        Returns:
            Созданная стратегия
            
        Raises:
            ValidationError: При некорректных данных стратегии
        """
        try:
            # Валидация обязательных полей
            required_fields = ["name", "type", "parameters"]
            for field in required_fields:
                if field not in strategy_data:
                    raise ValidationError(f"Отсутствует обязательное поле: {field}")
            
            # Создание ID стратегии
            strategy_id = StrategyId(uuid4())
            
            # Создание стратегии на основе типа
            strategy = await self._create_strategy_instance(
                strategy_id=strategy_id,
                strategy_data=strategy_data
            )
            
            # Сохранение в репозитории
            await self._repository.save(strategy)
            
            self._logger.info(f"Создана стратегия {strategy.name} с ID {strategy_id}")
            return strategy
            
        except Exception as e:
            self._logger.error(f"Ошибка создания стратегии: {e}")
            raise StrategyExecutionError(f"Не удалось создать стратегию: {e}")
    
    async def get_strategy(self, strategy_id: StrategyId) -> Optional[AbstractStrategy]:
        """
        Получение стратегии по ID.
        
        Args:
            strategy_id: Идентификатор стратегии
            
        Returns:
            Стратегия или None если не найдена
        """
        try:
            # Сначала проверяем активные стратегии
            if strategy_id in self._active_strategies:
                return self._active_strategies[strategy_id]
                
            # Затем ищем в репозитории
            strategy = await self._repository.get_by_id(strategy_id)
            if strategy:
                self._logger.debug(f"Загружена стратегия {strategy_id} из репозитория")
                
            return strategy
            
        except Exception as e:
            self._logger.error(f"Ошибка получения стратегии {strategy_id}: {e}")
            return None
    
    async def update_strategy(
        self,
        strategy_id: StrategyId,
        updates: Dict[str, Any]
    ) -> bool:
        """
        Обновление параметров стратегии.
        
        Args:
            strategy_id: Идентификатор стратегии
            updates: Данные для обновления
            
        Returns:
            True если обновление успешно
        """
        try:
            strategy = await self.get_strategy(strategy_id)
            if not strategy:
                self._logger.warning(f"Стратегия {strategy_id} не найдена для обновления")
                return False
            
            # Обновление параметров
            if "parameters" in updates:
                strategy.update_parameters(updates["parameters"])
                
            if "status" in updates:
                strategy.status = StrategyStatus(updates["status"])
                
            strategy.updated_at = datetime.now()
            
            # Сохранение изменений
            await self._repository.save(strategy)
            
            self._logger.info(f"Обновлена стратегия {strategy_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Ошибка обновления стратегии {strategy_id}: {e}")
            return False
    
    async def delete_strategy(self, strategy_id: StrategyId) -> bool:
        """
        Удаление стратегии.
        
        Args:
            strategy_id: Идентификатор стратегии
            
        Returns:
            True если удаление успешно
        """
        try:
            # Остановка стратегии если она активна
            if strategy_id in self._active_strategies:
                await self.stop_strategy(strategy_id)
                
            # Удаление из репозитория
            success = await self._repository.delete(strategy_id)
            
            if success:
                self._logger.info(f"Удалена стратегия {strategy_id}")
            else:
                self._logger.warning(f"Стратегия {strategy_id} не найдена для удаления")
                
            return success
            
        except Exception as e:
            self._logger.error(f"Ошибка удаления стратегии {strategy_id}: {e}")
            return False
    
    async def start_strategy(self, strategy_id: StrategyId) -> bool:
        """
        Запуск стратегии.
        
        Args:
            strategy_id: Идентификатор стратегии
            
        Returns:
            True если запуск успешен
        """
        try:
            strategy = await self.get_strategy(strategy_id)
            if not strategy:
                return False
                
            # Проверка состояния стратегии
            if strategy.status == StrategyStatus.RUNNING:
                self._logger.warning(f"Стратегия {strategy_id} уже запущена")
                return True
                
            # Запуск стратегии
            strategy.status = StrategyStatus.RUNNING
            strategy.state.start_time = datetime.now()
            
            # Добавление в активные стратегии
            self._active_strategies[strategy_id] = strategy
            
            # Сохранение состояния
            await self._repository.save(strategy)
            
            self._logger.info(f"Запущена стратегия {strategy_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Ошибка запуска стратегии {strategy_id}: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: StrategyId) -> bool:
        """
        Остановка стратегии.
        
        Args:
            strategy_id: Идентификатор стратегии
            
        Returns:
            True если остановка успешна
        """
        try:
            strategy = await self.get_strategy(strategy_id)
            if not strategy:
                return False
                
            # Остановка стратегии
            strategy.status = StrategyStatus.STOPPED
            strategy.state.end_time = datetime.now()
            
            # Удаление из активных стратегий
            if strategy_id in self._active_strategies:
                del self._active_strategies[strategy_id]
            
            # Сохранение состояния
            await self._repository.save(strategy)
            
            self._logger.info(f"Остановлена стратегия {strategy_id}")
            return True
            
        except Exception as e:
            self._logger.error(f"Ошибка остановки стратегии {strategy_id}: {e}")
            return False
    
    async def get_all_strategies(self) -> List[AbstractStrategy]:
        """
        Получение всех стратегий.
        
        Returns:
            Список всех стратегий
        """
        try:
            strategies = await self._repository.get_all()
            self._logger.debug(f"Загружено {len(strategies)} стратегий")
            return strategies
            
        except Exception as e:
            self._logger.error(f"Ошибка получения всех стратегий: {e}")
            return []
    
    async def get_active_strategies(self) -> List[AbstractStrategy]:
        """
        Получение активных стратегий.
        
        Returns:
            Список активных стратегий
        """
        return list(self._active_strategies.values())
    
    async def _create_strategy_instance(
        self,
        strategy_id: StrategyId,
        strategy_data: Dict[str, Any]
    ) -> AbstractStrategy:
        """
        Создание экземпляра стратегии на основе типа.
        
        Args:
            strategy_id: Идентификатор стратегии
            strategy_data: Данные стратегии
            
        Returns:
            Созданный экземпляр стратегии
        """
        # Базовая реализация - создание абстрактной стратегии
        # В реальной системе здесь должна быть фабрика стратегий
        from domain.strategies.base_strategies import BaseStrategy
        
        strategy = BaseStrategy(
            strategy_id=strategy_id,
            name=strategy_data["name"],
            parameters=strategy_data.get("parameters", {}),
            state=StrategyState()
        )
        
        return strategy