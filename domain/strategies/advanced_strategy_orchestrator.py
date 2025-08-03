"""
Advanced Strategy Orchestrator - Sophisticated Pattern Implementation

This module implements a multi-pattern architecture combining:
- Strategy Pattern for algorithmic selection
- Chain of Responsibility for signal processing
- Observer Pattern for event handling
- Factory Pattern for strategy instantiation
- Command Pattern for operation execution
- State Pattern for strategy lifecycle management
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Protocol, Generic, TypeVar
import asyncio
import logging
from collections import defaultdict
import uuid

# Domain imports
from domain.value_objects.signal import Signal
from domain.value_objects.money import Money
from domain.entities.strategy_parameters import StrategyParameters
from domain.protocols.strategy_protocol import StrategyInterface

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Состояния стратегии в жизненном цикле."""
    CREATED = auto()
    INITIALIZED = auto()
    ACTIVE = auto()
    PAUSED = auto()
    SUSPENDED = auto()
    TERMINATED = auto()
    ERROR = auto()


class SignalPriority(Enum):
    """Приоритеты сигналов для обработки."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass(frozen=True)
class StrategyEvent:
    """Событие стратегии для паттерна Observer."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strategy_id: str = ""
    event_type: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    priority: SignalPriority = SignalPriority.MEDIUM


class EventObserver(Protocol):
    """Протокол для наблюдателей событий."""
    
    async def handle_event(self, event: StrategyEvent) -> None:
        """Обработка события."""
        ...


class SignalProcessor(ABC):
    """Абстрактный процессор сигналов для Chain of Responsibility."""
    
    def __init__(self, next_processor: Optional['SignalProcessor'] = None):
        self._next_processor = next_processor
    
    async def process(self, signal: Signal, context: Dict[str, Any]) -> Signal:
        """Обработка сигнала с возможностью передачи дальше по цепочке."""
        processed_signal = await self._process_signal(signal, context)
        
        if self._next_processor and processed_signal:
            return await self._next_processor.process(processed_signal, context)
        
        return processed_signal
    
    @abstractmethod
    async def _process_signal(self, signal: Signal, context: Dict[str, Any]) -> Signal:
        """Конкретная реализация обработки сигнала."""
        pass


class RiskFilterProcessor(SignalProcessor):
    """Фильтр риска в цепочке обработки сигналов."""
    
    def __init__(self, max_risk_level: Decimal, next_processor: Optional[SignalProcessor] = None):
        super().__init__(next_processor)
        self.max_risk_level = max_risk_level
    
    async def _process_signal(self, signal: Signal, context: Dict[str, Any]) -> Signal:
        """Фильтрация сигналов по уровню риска."""
        risk_level = context.get('risk_level', Decimal('0'))
        
        if risk_level > self.max_risk_level:
            logger.warning(f"Signal filtered due to high risk: {risk_level}")
            return None
        
        return signal


class VolumeValidationProcessor(SignalProcessor):
    """Валидация объемов торговли."""
    
    def __init__(self, min_volume: Decimal, next_processor: Optional[SignalProcessor] = None):
        super().__init__(next_processor)
        self.min_volume = min_volume
    
    async def _process_signal(self, signal: Signal, context: Dict[str, Any]) -> Signal:
        """Валидация минимального объема."""
        volume = context.get('volume', Decimal('0'))
        
        if volume < self.min_volume:
            logger.info(f"Signal volume {volume} below minimum {self.min_volume}")
            return None
        
        return signal


class TimingOptimizationProcessor(SignalProcessor):
    """Оптимизация времени входа в позицию."""
    
    async def _process_signal(self, signal: Signal, context: Dict[str, Any]) -> Signal:
        """Оптимизация timing для сигнала."""
        current_hour = datetime.now().hour
        
        # Избегаем торговли в низколиквидные часы
        if 22 <= current_hour or current_hour <= 6:
            # Снижаем силу сигнала в нерабочие часы
            optimized_signal = Signal(
                signal_type=signal.signal_type,
                strength=signal.strength * Decimal('0.7'),
                confidence=signal.confidence * Decimal('0.8'),
                trading_pair=signal.trading_pair,
                metadata={**signal.metadata, 'timing_adjusted': True}
            )
            return optimized_signal
        
        return signal


T = TypeVar('T')


class StrategyCommand(ABC, Generic[T]):
    """Абстрактная команда для паттерна Command."""
    
    @abstractmethod
    async def execute(self) -> T:
        """Выполнение команды."""
        pass
    
    @abstractmethod
    async def undo(self) -> None:
        """Отмена команды."""
        pass


class ActivateStrategyCommand(StrategyCommand[bool]):
    """Команда активации стратегии."""
    
    def __init__(self, strategy: StrategyInterface, orchestrator: 'AdvancedStrategyOrchestrator'):
        self.strategy = strategy
        self.orchestrator = orchestrator
        self._executed = False
    
    async def execute(self) -> bool:
        """Активация стратегии."""
        try:
            await self.strategy.start()
            self.orchestrator._active_strategies[self.strategy.get_id()] = self.strategy
            self._executed = True
            
            # Отправляем событие об активации
            event = StrategyEvent(
                strategy_id=self.strategy.get_id(),
                event_type="strategy_activated",
                data={"strategy_name": self.strategy.get_name()}
            )
            await self.orchestrator._notify_observers(event)
            
            return True
        except Exception as e:
            logger.error(f"Failed to activate strategy {self.strategy.get_id()}: {e}")
            return False
    
    async def undo(self) -> None:
        """Отмена активации стратегии."""
        if self._executed:
            await self.strategy.stop()
            self.orchestrator._active_strategies.pop(self.strategy.get_id(), None)


class PauseStrategyCommand(StrategyCommand[bool]):
    """Команда приостановки стратегии."""
    
    def __init__(self, strategy_id: str, orchestrator: 'AdvancedStrategyOrchestrator'):
        self.strategy_id = strategy_id
        self.orchestrator = orchestrator
        self._previous_state = None
    
    async def execute(self) -> bool:
        """Приостановка стратегии."""
        strategy = self.orchestrator._active_strategies.get(self.strategy_id)
        if not strategy:
            return False
        
        try:
            self._previous_state = StrategyState.ACTIVE
            await strategy.pause()
            self.orchestrator._strategy_states[self.strategy_id] = StrategyState.PAUSED
            
            event = StrategyEvent(
                strategy_id=self.strategy_id,
                event_type="strategy_paused",
                data={"reason": "manual_pause"}
            )
            await self.orchestrator._notify_observers(event)
            
            return True
        except Exception as e:
            logger.error(f"Failed to pause strategy {self.strategy_id}: {e}")
            return False
    
    async def undo(self) -> None:
        """Отмена приостановки."""
        if self._previous_state == StrategyState.ACTIVE:
            strategy = self.orchestrator._active_strategies.get(self.strategy_id)
            if strategy:
                await strategy.resume()
                self.orchestrator._strategy_states[self.strategy_id] = StrategyState.ACTIVE


class StrategySelector(ABC):
    """Абстрактный селектор стратегий для паттерна Strategy."""
    
    @abstractmethod
    async def select_strategy(self, market_conditions: Dict[str, Any]) -> Optional[str]:
        """Выбор оптимальной стратегии на основе рыночных условий."""
        pass


class AdaptiveStrategySelector(StrategySelector):
    """Адаптивный селектор стратегий на основе ML."""
    
    def __init__(self):
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.market_regime_weights = {
            'trending': {'trend_following': 0.8, 'mean_reversion': 0.2, 'breakout': 0.6},
            'sideways': {'trend_following': 0.2, 'mean_reversion': 0.8, 'breakout': 0.3},
            'volatile': {'trend_following': 0.4, 'mean_reversion': 0.3, 'breakout': 0.9},
        }
    
    async def select_strategy(self, market_conditions: Dict[str, Any]) -> Optional[str]:
        """Выбор стратегии на основе адаптивного алгоритма."""
        regime = market_conditions.get('regime', 'sideways')
        volatility = market_conditions.get('volatility', 0.0)
        trend_strength = market_conditions.get('trend_strength', 0.0)
        
        # Получаем веса для текущего режима
        weights = self.market_regime_weights.get(regime, {})
        
        # Корректируем веса на основе исторической производительности
        adjusted_weights = {}
        for strategy_name, base_weight in weights.items():
            historical_performance = self._get_avg_performance(strategy_name)
            performance_multiplier = 1.0 + (historical_performance - 0.5) * 0.5
            adjusted_weights[strategy_name] = base_weight * performance_multiplier
        
        # Добавляем факторы рынка
        if volatility > 0.8:
            adjusted_weights['breakout'] = adjusted_weights.get('breakout', 0.5) * 1.3
        
        if abs(trend_strength) > 0.7:
            adjusted_weights['trend_following'] = adjusted_weights.get('trend_following', 0.5) * 1.2
        
        # Выбираем стратегию с максимальным весом
        if adjusted_weights:
            selected_strategy = max(adjusted_weights.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected strategy: {selected_strategy} for regime: {regime}")
            return selected_strategy
        
        return None
    
    def _get_avg_performance(self, strategy_name: str) -> float:
        """Получение средней исторической производительности."""
        performances = self.performance_history.get(strategy_name, [0.5])
        return sum(performances) / len(performances)
    
    def update_performance(self, strategy_name: str, performance: float) -> None:
        """Обновление исторических данных производительности."""
        self.performance_history[strategy_name].append(performance)
        # Ограничиваем историю последними 100 записями
        if len(self.performance_history[strategy_name]) > 100:
            self.performance_history[strategy_name] = self.performance_history[strategy_name][-100:]


class PerformanceObserver:
    """Наблюдатель за производительностью стратегий."""
    
    def __init__(self):
        self.performance_metrics: Dict[str, Dict[str, Any]] = {}
        self.alert_thresholds = {
            'max_drawdown': -0.15,  # -15%
            'min_win_rate': 0.4,    # 40%
            'min_profit_factor': 1.1
        }
    
    async def handle_event(self, event: StrategyEvent) -> None:
        """Обработка событий производительности."""
        if event.event_type == "performance_update":
            await self._process_performance_update(event)
        elif event.event_type == "strategy_completed_trade":
            await self._process_trade_completion(event)
    
    async def _process_performance_update(self, event: StrategyEvent) -> None:
        """Обработка обновления производительности."""
        strategy_id = event.strategy_id
        metrics = event.data.get('metrics', {})
        
        self.performance_metrics[strategy_id] = metrics
        
        # Проверяем пороговые значения
        for metric, threshold in self.alert_thresholds.items():
            current_value = metrics.get(metric, 0)
            if (metric.startswith('min_') and current_value < threshold) or \
               (metric.startswith('max_') and current_value < threshold):
                await self._send_alert(strategy_id, metric, current_value, threshold)
    
    async def _process_trade_completion(self, event: StrategyEvent) -> None:
        """Обработка завершения сделки."""
        strategy_id = event.strategy_id
        trade_result = event.data.get('trade_result', {})
        
        # Анализируем результат сделки и обновляем метрики
        profit_loss = trade_result.get('profit_loss', 0)
        logger.info(f"Strategy {strategy_id} completed trade with P&L: {profit_loss}")
    
    async def _send_alert(self, strategy_id: str, metric: str, current: float, threshold: float) -> None:
        """Отправка предупреждения о производительности."""
        logger.warning(f"Performance alert for strategy {strategy_id}: "
                      f"{metric} = {current} crossed threshold {threshold}")


class AdvancedStrategyOrchestrator:
    """
    Продвинутый оркестратор стратегий, реализующий множественные паттерны проектирования.
    """
    
    def __init__(self):
        self._active_strategies: Dict[str, StrategyInterface] = {}
        self._strategy_states: Dict[str, StrategyState] = {}
        self._observers: List[EventObserver] = []
        self._command_history: List[StrategyCommand] = []
        self._signal_processing_chain: Optional[SignalProcessor] = None
        self._strategy_selector: Optional[StrategySelector] = None
        self._performance_observer = PerformanceObserver()
        
        # Инициализируем компоненты
        self._initialize_signal_processing_chain()
        self._initialize_strategy_selector()
        self.add_observer(self._performance_observer)
    
    def _initialize_signal_processing_chain(self) -> None:
        """Инициализация цепочки обработки сигналов."""
        # Создаем цепочку: Risk Filter -> Volume Validation -> Timing Optimization
        timing_processor = TimingOptimizationProcessor()
        volume_processor = VolumeValidationProcessor(
            min_volume=Decimal('100'), 
            next_processor=timing_processor
        )
        risk_processor = RiskFilterProcessor(
            max_risk_level=Decimal('0.8'), 
            next_processor=volume_processor
        )
        
        self._signal_processing_chain = risk_processor
    
    def _initialize_strategy_selector(self) -> None:
        """Инициализация селектора стратегий."""
        self._strategy_selector = AdaptiveStrategySelector()
    
    def add_observer(self, observer: EventObserver) -> None:
        """Добавление наблюдателя событий."""
        self._observers.append(observer)
    
    def remove_observer(self, observer: EventObserver) -> None:
        """Удаление наблюдателя событий."""
        if observer in self._observers:
            self._observers.remove(observer)
    
    async def _notify_observers(self, event: StrategyEvent) -> None:
        """Уведомление всех наблюдателей о событии."""
        for observer in self._observers:
            try:
                await observer.handle_event(event)
            except Exception as e:
                logger.error(f"Error notifying observer: {e}")
    
    async def execute_command(self, command: StrategyCommand) -> Any:
        """Выполнение команды с сохранением в истории."""
        result = await command.execute()
        self._command_history.append(command)
        
        # Ограничиваем историю команд
        if len(self._command_history) > 1000:
            self._command_history = self._command_history[-1000:]
        
        return result
    
    async def undo_last_command(self) -> bool:
        """Отмена последней команды."""
        if self._command_history:
            last_command = self._command_history.pop()
            try:
                await last_command.undo()
                return True
            except Exception as e:
                logger.error(f"Failed to undo command: {e}")
                # Возвращаем команду в историю при ошибке отмены
                self._command_history.append(last_command)
        
        return False
    
    async def process_signal(self, signal: Signal, context: Optional[Dict[str, Any]] = None) -> Optional[Signal]:
        """Обработка торгового сигнала через цепочку процессоров."""
        if not self._signal_processing_chain:
            return signal
        
        processing_context = context or {}
        processed_signal = await self._signal_processing_chain.process(signal, processing_context)
        
        if processed_signal:
            # Отправляем событие об обработанном сигнале
            event = StrategyEvent(
                event_type="signal_processed",
                data={
                    'signal_type': str(processed_signal.signal_type),
                    'strength': float(processed_signal.strength),
                    'confidence': float(processed_signal.confidence)
                }
            )
            await self._notify_observers(event)
        
        return processed_signal
    
    async def select_optimal_strategy(self, market_conditions: Dict[str, Any]) -> Optional[str]:
        """Выбор оптимальной стратегии для текущих рыночных условий."""
        if not self._strategy_selector:
            return None
        
        selected_strategy = await self._strategy_selector.select_strategy(market_conditions)
        
        if selected_strategy:
            event = StrategyEvent(
                event_type="strategy_selected",
                data={
                    'selected_strategy': selected_strategy,
                    'market_conditions': market_conditions
                }
            )
            await self._notify_observers(event)
        
        return selected_strategy
    
    async def activate_strategy(self, strategy: StrategyInterface) -> bool:
        """Активация стратегии через Command Pattern."""
        command = ActivateStrategyCommand(strategy, self)
        return await self.execute_command(command)
    
    async def pause_strategy(self, strategy_id: str) -> bool:
        """Приостановка стратегии через Command Pattern."""
        command = PauseStrategyCommand(strategy_id, self)
        return await self.execute_command(command)
    
    def get_active_strategies(self) -> Dict[str, StrategyInterface]:
        """Получение активных стратегий."""
        return self._active_strategies.copy()
    
    def get_strategy_state(self, strategy_id: str) -> Optional[StrategyState]:
        """Получение состояния стратегии."""
        return self._strategy_states.get(strategy_id)
    
    async def shutdown(self) -> None:
        """Корректное завершение работы оркестратора."""
        # Останавливаем все активные стратегии
        for strategy_id, strategy in self._active_strategies.items():
            try:
                await strategy.stop()
                self._strategy_states[strategy_id] = StrategyState.TERMINATED
            except Exception as e:
                logger.error(f"Error stopping strategy {strategy_id}: {e}")
        
        self._active_strategies.clear()
        
        # Отправляем событие о завершении работы
        event = StrategyEvent(
            event_type="orchestrator_shutdown",
            data={'shutdown_time': datetime.now(timezone.utc).isoformat()}
        )
        await self._notify_observers(event)


# Фабрика для создания оркестратора с различными конфигурациями
class OrchestratorFactory:
    """Фабрика для создания оркестраторов с различными конфигурациями."""
    
    @staticmethod
    def create_conservative_orchestrator() -> AdvancedStrategyOrchestrator:
        """Создание консервативного оркестратора с низким риском."""
        orchestrator = AdvancedStrategyOrchestrator()
        
        # Более строгие фильтры риска
        timing_processor = TimingOptimizationProcessor()
        volume_processor = VolumeValidationProcessor(
            min_volume=Decimal('500'), 
            next_processor=timing_processor
        )
        risk_processor = RiskFilterProcessor(
            max_risk_level=Decimal('0.5'), 
            next_processor=volume_processor
        )
        
        orchestrator._signal_processing_chain = risk_processor
        return orchestrator
    
    @staticmethod
    def create_aggressive_orchestrator() -> AdvancedStrategyOrchestrator:
        """Создание агрессивного оркестратора с высоким риском."""
        orchestrator = AdvancedStrategyOrchestrator()
        
        # Более либеральные фильтры
        timing_processor = TimingOptimizationProcessor()
        volume_processor = VolumeValidationProcessor(
            min_volume=Decimal('50'), 
            next_processor=timing_processor
        )
        risk_processor = RiskFilterProcessor(
            max_risk_level=Decimal('0.95'), 
            next_processor=volume_processor
        )
        
        orchestrator._signal_processing_chain = risk_processor
        return orchestrator


# Экспорт основных классов для использования в других модулях
__all__ = [
    'AdvancedStrategyOrchestrator',
    'OrchestratorFactory',
    'StrategyEvent',
    'StrategyState',
    'SignalPriority',
    'EventObserver',
    'StrategySelector',
    'AdaptiveStrategySelector',
    'SignalProcessor',
    'RiskFilterProcessor',
    'VolumeValidationProcessor',
    'TimingOptimizationProcessor',
    'StrategyCommand',
    'ActivateStrategyCommand',
    'PauseStrategyCommand',
    'PerformanceObserver'
]