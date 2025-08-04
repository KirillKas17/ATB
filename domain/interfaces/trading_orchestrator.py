"""
Протокол для торгового оркестратора.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime
from decimal import Decimal
from enum import Enum
from domain.types.trading_types import TradingConfig

class OrchestratorMode(Enum):
    """Режимы работы оркестратора."""
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    FULL_AUTO = "full_auto"
    SIMULATION = "simulation"
    BACKTEST = "backtest"

class TradingPhase(Enum):
    """Фазы торговли."""
    INITIALIZATION = "initialization"
    MARKET_ANALYSIS = "market_analysis"
    SIGNAL_GENERATION = "signal_generation"
    RISK_ASSESSMENT = "risk_assessment"
    ORDER_EXECUTION = "order_execution"
    POSITION_MONITORING = "position_monitoring"
    PORTFOLIO_REBALANCING = "portfolio_rebalancing"
    REPORTING = "reporting"

class TradingOrchestratorProtocol(Protocol):
    """Протокол для торгового оркестратора."""
    
    async def start(self) -> None:
        """Запуск торгового оркестратора."""
        ...
    
    async def stop(self) -> None:
        """Остановка торгового оркестратора."""
        ...
    
    async def pause(self) -> None:
        """Приостановка торговли."""
        ...
    
    async def resume(self) -> None:
        """Возобновление торговли."""
        ...
    
    async def set_mode(self, mode: OrchestratorMode) -> None:
        """Установка режима работы."""
        ...
    
    async def execute_trading_cycle(self) -> Dict[str, Any]:
        """Выполнение торгового цикла."""
        ...
    
    async def analyze_market_conditions(self) -> Dict[str, Any]:
        """Анализ рыночных условий."""
        ...
    
    async def generate_trading_signals(self) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов."""
        ...
    
    async def assess_risks(self) -> Dict[str, Any]:
        """Оценка рисков."""
        ...
    
    async def execute_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Исполнение ордеров."""
        ...
    
    async def monitor_positions(self) -> Dict[str, Any]:
        """Мониторинг позиций."""
        ...
    
    async def rebalance_portfolio(self) -> Dict[str, Any]:
        """Ребалансировка портфеля."""
        ...
    
    async def emergency_stop(self) -> None:
        """Экстренная остановка."""
        ...
    
    async def get_status(self) -> Dict[str, Any]:
        """Получение статуса оркестратора."""
        ...
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Получение метрик производительности."""
        ...
    
    async def generate_report(self) -> Dict[str, Any]:
        """Генерация отчета."""
        ...

class BaseTradingOrchestrator(ABC):
    """Базовый класс для торгового оркестратора."""
    
    def __init__(self, *, config: Optional[TradingConfig] = None) -> None:
        self._mode = OrchestratorMode.MANUAL
        self._current_phase = TradingPhase.INITIALIZATION
        self._is_running = False
        self._is_paused = False
        self._emergency_stop_active = False
        self._cycle_count = 0
        self._last_cycle_time: Optional[datetime] = None
        self._strategies: Dict[str, Any] = {}
        self._performance_metrics: Dict[str, Any] = {}
        self._last_update: Optional[datetime] = None
        self._error_count = 0
    
    @abstractmethod
    async def _initialize_components(self) -> None:
        """Инициализация компонентов."""
        pass
    
    @abstractmethod
    async def _cleanup_resources(self) -> None:
        """Очистка ресурсов."""
        pass
    
    @property
    def mode(self) -> OrchestratorMode:
        """Текущий режим работы."""
        return self._mode
    
    @property
    def current_phase(self) -> TradingPhase:
        """Текущая фаза торговли."""
        return self._current_phase
    
    @property
    def is_running(self) -> bool:
        """Статус работы оркестратора."""
        return self._is_running
    
    @property
    def is_paused(self) -> bool:
        """Статус паузы."""
        return self._is_paused
    
    @property
    def emergency_stop_active(self) -> bool:
        """Статус экстренной остановки."""
        return self._emergency_stop_active
    
    @property
    def cycle_count(self) -> int:
        """Количество выполненных циклов."""
        return self._cycle_count
    
    def _increment_cycle_count(self) -> None:
        """Увеличение счетчика циклов."""
        self._cycle_count += 1
        self._last_cycle_time = datetime.now()
    
    def _set_phase(self, phase: TradingPhase) -> None:
        """Установка текущей фазы."""
        self._current_phase = phase
    
    def _update_performance_metric(self, metric_name: str, value: Any) -> None:
        """Обновление метрики производительности."""
        self._performance_metrics[metric_name] = {
            'value': value,
            'timestamp': datetime.now()
        }
        # Fix the datetime assignment issue
        if self._last_update is None:
            self._last_update = datetime.now()
        
        # Fix the dict assignment to float
        # Note: _get_performance_data method should be implemented in subclasses
        # self._performance_metrics['overall_score'] = self._calculate_overall_score()