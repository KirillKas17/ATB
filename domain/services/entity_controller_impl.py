from datetime import datetime
from typing import Optional
from domain.types.entity_system_types import (
    BaseEntityController,
    EntityState,
    OperationMode,
    OptimizationLevel,
)


class EntityControllerImpl(BaseEntityController):
    """Реализация контроллера сущностей системы торговли."""
    
    def __init__(self) -> None:
        self._is_running: bool = False
        self._operation_mode: OperationMode = OperationMode.STANDARD
        self._optimization_level: OptimizationLevel = OptimizationLevel.LOW
        self._current_phase: str = "stopped"
        self._start_time: Optional[datetime] = None
        
    async def start(self) -> None:
        """Запуск контроллера сущностей."""
        if self._is_running:
            return
            
        self._is_running = True
        self._current_phase = "starting"
        self._start_time = datetime.now()
        
        # Инициализация системы
        self._current_phase = "running"

    async def stop(self) -> None:
        """Остановка контроллера сущностей."""
        if not self._is_running:
            return
            
        self._current_phase = "stopping"
        self._is_running = False
        self._start_time = None
        self._current_phase = "stopped"

    async def get_status(self) -> EntityState:
        """Получение текущего состояния контроллера."""
        uptime_hours = 0.0
        if self._start_time and self._is_running:
            uptime_hours = (datetime.now() - self._start_time).total_seconds() / 3600
            
        # Расчёт метрик на основе состояния
        system_health = 1.0 if self._is_running else 0.5
        performance_score = min(1.0, uptime_hours / 24)  # Улучшается со временем работы
        efficiency_score = 0.8 if self._optimization_level == OptimizationLevel.HIGH else 0.6
        
        return EntityState(
            is_running=self._is_running,
            current_phase=self._current_phase,
            ai_confidence=system_health,
            optimization_level=self._optimization_level.value.lower(),
            system_health=system_health,
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            last_update=datetime.now(),
        )

    def set_operation_mode(self, mode: OperationMode) -> None:
        """Установка режима работы контроллера."""
        self._operation_mode = mode
        if self._is_running:
            # Адаптация к новому режиму
            if mode == OperationMode.AGGRESSIVE:
                self._current_phase = "aggressive_mode"
            elif mode == OperationMode.CONSERVATIVE:
                self._current_phase = "conservative_mode"
            else:
                self._current_phase = "standard_mode"

    def set_optimization_level(self, level: OptimizationLevel) -> None:
        """Установка уровня оптимизации контроллера."""
        self._optimization_level = level
