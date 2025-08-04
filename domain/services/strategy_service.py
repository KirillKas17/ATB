"""
Продвинутый сервис управления торговыми стратегиями.
Включает оптимизацию параметров, адаптацию к рыночным условиям и управление жизненным циклом стратегий.
"""
import asyncio
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import uuid
import json
import pickle
from collections import defaultdict, deque
import threading
import weakref
import math

# Optimization imports
try:
    from scipy.optimize import differential_evolution, minimize
    from sklearn.model_selection import ParameterGrid
    HAS_OPTIMIZATION = True
except ImportError:
    HAS_OPTIMIZATION = False

logger = logging.getLogger(__name__)

class StrategyState(Enum):
    """Состояния стратегии."""
    CREATED = "created"
    INITIALIZED = "initialized"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    OPTIMIZING = "optimizing"
    ADAPTING = "adapting"

class StrategyType(Enum):
    """Типы стратегий."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    PAIRS_TRADING = "pairs_trading"
    VOLATILITY = "volatility"
    ML_BASED = "ml_based"
    HYBRID = "hybrid"

class OptimizationMethod(Enum):
    """Методы оптимизации параметров."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    BAYESIAN = "bayesian"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"

@dataclass
class StrategyParameter:
    """Параметр стратегии."""
    name: str
    value: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    parameter_type: str = "float"
    description: str = ""
    is_optimizable: bool = True
    sensitivity: float = 1.0  # Чувствительность к изменениям
    
    def validate(self) -> bool:
        """Валидация параметра."""
        if self.min_value is not None and self.value < self.min_value:
            return False
        if self.max_value is not None and self.value > self.max_value:
            return False
        return True
    
    def normalize(self) -> float:
        """Нормализация значения в диапазон [0, 1]."""
        if self.min_value is None or self.max_value is None:
            return 0.5
        return (self.value - self.min_value) / (self.max_value - self.min_value)

@dataclass
class StrategyMetrics:
    """Метрики производительности стратегии."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    max_consecutive_losses: int = 0
    avg_trade_duration: float = 0.0
    profit_factor: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """Коэффициент выигрышных сделок."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades
    
    @property
    def average_trade_pnl(self) -> float:
        """Средний PnL на сделку."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    def update_from_trade(self, pnl: float, duration: float) -> None:
        """Обновление метрик после сделки."""
        self.total_trades += 1
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Обновление средней продолжительности
        if self.total_trades == 1:
            self.avg_trade_duration = duration
        else:
            self.avg_trade_duration = (
                (self.avg_trade_duration * (self.total_trades - 1) + duration) / self.total_trades
            )

@dataclass
class OptimizationResult:
    """Результат оптимизации стратегии."""
    best_parameters: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    execution_time: float = 0.0
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    
    def get_parameter_sensitivity(self) -> Dict[str, float]:
        """Анализ чувствительности параметров."""
        if len(self.optimization_history) < 10:
            return {}
        
        sensitivity = {}
        history_df = pd.DataFrame(self.optimization_history)
        
        for param in self.best_parameters.keys():
            if param in history_df.columns:
                correlation = abs(history_df[param].corr(history_df['score']))
                sensitivity[param] = correlation if not np.isnan(correlation) else 0.0
        
        return sensitivity

class AdaptationSignal:
    """Сигнал для адаптации стратегии."""
    
    def __init__(self, signal_type: str, strength: float, description: str = "", metadata: Optional[Dict] = None):
        self.signal_type = signal_type
        self.strength = max(0.0, min(1.0, strength))  # Нормализация в [0, 1]
        self.description = description
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.id = str(uuid.uuid4())
    
    def is_strong(self, threshold: float = 0.7) -> bool:
        """Проверка силы сигнала."""
        return self.strength >= threshold

class StrategyBase(ABC):
    """Базовый класс для всех стратегий."""
    
    def __init__(self, strategy_id: Optional[str] = None, strategy_type: Optional[StrategyType] = None):
        self.strategy_id = strategy_id or str(uuid.uuid4())
        self.strategy_type = strategy_type or StrategyType.HYBRID
        self.state = StrategyState.CREATED
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        # Параметры и конфигурация
        self.parameters: Dict[str, StrategyParameter] = {}
        self.metrics = StrategyMetrics()
        
        # Система адаптации
        self.adaptation_enabled = True
        self.adaptation_signals: deque = deque(maxlen=100)
        self.adaptation_threshold = 0.5
        
        # Оптимизация
        self.optimization_history: List[OptimizationResult] = []
        self.last_optimization = None
        
        # Управление состоянием
        self._lock = asyncio.Lock()
        self._observers: Set[Callable] = set()
    
    @abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Генерация торговых сигналов."""
        pass
    
    @abstractmethod
    async def calculate_position_size(self, signal: Dict[str, Any], account_info: Dict[str, Any]) -> float:
        """Расчёт размера позиции."""
        pass
    
    @abstractmethod
    async def should_exit(self, position: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Проверка условий выхода из позиции."""
        pass
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Инициализация стратегии."""
        try:
            await self._load_parameters(config)
            await self._validate_parameters()
            self.state = StrategyState.INITIALIZED
            await self._notify_observers("initialized")
            logger.info(f"Strategy {self.strategy_id} initialized successfully")
        except Exception as e:
            self.state = StrategyState.ERROR
            logger.error(f"Failed to initialize strategy {self.strategy_id}: {e}")
            raise
    
    async def start(self) -> None:
        """Запуск стратегии."""
        if self.state not in [StrategyState.INITIALIZED, StrategyState.PAUSED]:
            raise ValueError(f"Cannot start strategy in state {self.state}")
        
        self.state = StrategyState.ACTIVE
        await self._notify_observers("started")
        logger.info(f"Strategy {self.strategy_id} started")
    
    async def pause(self) -> None:
        """Приостановка стратегии."""
        if self.state != StrategyState.ACTIVE:
            raise ValueError(f"Cannot pause strategy in state {self.state}")
        
        self.state = StrategyState.PAUSED
        await self._notify_observers("paused")
        logger.info(f"Strategy {self.strategy_id} paused")
    
    async def stop(self) -> None:
        """Остановка стратегии."""
        self.state = StrategyState.STOPPED
        await self._notify_observers("stopped")
        logger.info(f"Strategy {self.strategy_id} stopped")
    
    async def add_adaptation_signal(self, signal: AdaptationSignal) -> None:
        """Добавление сигнала адаптации."""
        if not self.adaptation_enabled:
            return
        
        async with self._lock:
            self.adaptation_signals.append(signal)
            
            # Проверка необходимости адаптации
            if signal.is_strong() and len(self.adaptation_signals) >= 5:
                await self._trigger_adaptation()
    
    async def optimize_parameters(
        self,
        market_data: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.DIFFERENTIAL_EVOLUTION,
        target_metric: str = "sharpe_ratio",
        max_iterations: int = 100
    ) -> OptimizationResult:
        """Оптимизация параметров стратегии."""
        if not HAS_OPTIMIZATION:
            logger.warning("Optimization libraries not available")
            return OptimizationResult(
                best_parameters=self.get_parameter_values(),
                best_score=0.0
            )
        
        self.state = StrategyState.OPTIMIZING
        start_time = datetime.now()
        
        try:
            # Получение оптимизируемых параметров
            optimizable_params = {
                name: param for name, param in self.parameters.items()
                if param.is_optimizable and param.min_value is not None and param.max_value is not None
            }
            
            if not optimizable_params:
                logger.warning("No optimizable parameters found")
                return OptimizationResult(
                    best_parameters=self.get_parameter_values(),
                    best_score=0.0
                )
            
            # Определение границ параметров
            bounds = [(param.min_value, param.max_value) for param in optimizable_params.values()]
            param_names = list(optimizable_params.keys())
            
            # Функция оптимизации
            optimization_history = []
            
            def objective_function(params):
                # Установка параметров
                param_dict = dict(zip(param_names, params))
                
                # Запуск бэктеста (упрощённая версия)
                score = self._evaluate_strategy(market_data, param_dict, target_metric)
                
                # Сохранение в историю
                optimization_history.append({
                    **param_dict,
                    'score': score,
                    'timestamp': datetime.now()
                })
                
                return -score  # Минимизация (поэтому отрицательный)
            
            # Выбор метода оптимизации
            if method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=max_iterations,
                    seed=42,
                    atol=1e-6,
                    tol=1e-6
                )
                best_params = dict(zip(param_names, result.x))
                best_score = -result.fun
                
            elif method == OptimizationMethod.GRID_SEARCH:
                # Сетка параметров
                param_grid = {}
                for name, param in optimizable_params.items():
                    param_grid[name] = np.linspace(param.min_value, param.max_value, 10)
                
                grid = ParameterGrid(param_grid)
                best_score = -np.inf
                best_params = {}
                
                for params in grid:
                    score = -objective_function(list(params.values()))
                    if score > best_score:
                        best_score = score
                        best_params = params
            
            else:
                # Fallback к случайному поиску
                best_score = -np.inf
                best_params = {}
                
                for _ in range(max_iterations):
                    params = []
                    for param in optimizable_params.values():
                        params.append(np.random.uniform(param.min_value, param.max_value))
                    
                    score = -objective_function(params)
                    if score > best_score:
                        best_score = score
                        best_params = dict(zip(param_names, params))
            
            # Применение лучших параметров
            await self._update_parameters(best_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            optimization_result = OptimizationResult(
                best_parameters=best_params,
                best_score=best_score,
                optimization_history=optimization_history,
                iterations=len(optimization_history),
                execution_time=execution_time
            )
            
            self.optimization_history.append(optimization_result)
            self.last_optimization = datetime.now()
            
            self.state = StrategyState.ACTIVE
            await self._notify_observers("optimized")
            
            logger.info(f"Strategy {self.strategy_id} optimized: score={best_score:.4f}, "
                       f"iterations={len(optimization_history)}, time={execution_time:.2f}s")
            
            return optimization_result
            
        except Exception as e:
            self.state = StrategyState.ERROR
            logger.error(f"Optimization failed for strategy {self.strategy_id}: {e}")
            raise
    
    def add_observer(self, observer: Callable) -> None:
        """Добавление наблюдателя за изменениями стратегии."""
        self._observers.add(observer)
    
    def remove_observer(self, observer: Callable) -> None:
        """Удаление наблюдателя."""
        self._observers.discard(observer)
    
    def get_parameter_values(self) -> Dict[str, Any]:
        """Получение значений всех параметров."""
        return {name: param.value for name, param in self.parameters.items()}
    
    def get_state_info(self) -> Dict[str, Any]:
        """Получение информации о состоянии стратегии."""
        return {
            'strategy_id': self.strategy_id,
            'strategy_type': self.strategy_type.value,
            'state': self.state.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'metrics': {
                'total_trades': self.metrics.total_trades,
                'win_rate': self.metrics.win_rate,
                'total_pnl': self.metrics.total_pnl,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'max_drawdown': self.metrics.max_drawdown
            },
            'adaptation_signals_count': len(self.adaptation_signals),
            'optimization_count': len(self.optimization_history),
            'last_optimization': self.last_optimization.isoformat() if self.last_optimization else None
        }
    
    async def _load_parameters(self, config: Dict[str, Any]) -> None:
        """Загрузка параметров из конфигурации."""
        for name, value in config.get('parameters', {}).items():
            if isinstance(value, dict):
                param = StrategyParameter(**value)
            else:
                param = StrategyParameter(name=name, value=value)
            
            self.parameters[name] = param
    
    async def _validate_parameters(self) -> None:
        """Валидация параметров."""
        for name, param in self.parameters.items():
            if not param.validate():
                raise ValueError(f"Invalid parameter {name}: {param.value}")
    
    async def _update_parameters(self, new_params: Dict[str, Any]) -> None:
        """Обновление параметров стратегии."""
        async with self._lock:
            for name, value in new_params.items():
                if name in self.parameters:
                    self.parameters[name].value = value
            
            self.last_updated = datetime.now()
            await self._notify_observers("parameters_updated")
    
    async def _trigger_adaptation(self) -> None:
        """Запуск процесса адаптации."""
        if self.state != StrategyState.ACTIVE:
            return
        
        self.state = StrategyState.ADAPTING
        
        try:
            # Анализ сигналов адаптации
            recent_signals = list(self.adaptation_signals)[-10:]
            strong_signals = [s for s in recent_signals if s.is_strong()]
            
            if len(strong_signals) >= 3:
                # Применение адаптивных изменений
                await self._apply_adaptation(strong_signals)
            
            self.state = StrategyState.ACTIVE
            await self._notify_observers("adapted")
            
        except Exception as e:
            self.state = StrategyState.ERROR
            logger.error(f"Adaptation failed for strategy {self.strategy_id}: {e}")
    
    async def _apply_adaptation(self, signals: List[AdaptationSignal]) -> None:
        """Применение адаптивных изменений."""
        # Группировка сигналов по типу
        signal_groups = defaultdict(list)
        for signal in signals:
            signal_groups[signal.signal_type].append(signal)
        
        # Применение изменений на основе доминирующих сигналов
        for signal_type, group_signals in signal_groups.items():
            avg_strength = np.mean([s.strength for s in group_signals])
            
            if signal_type == "volatility_change" and avg_strength > 0.7:
                # Адаптация к изменению волатильности
                await self._adapt_to_volatility(avg_strength)
            
            elif signal_type == "trend_change" and avg_strength > 0.6:
                # Адаптация к смене тренда
                await self._adapt_to_trend_change(avg_strength)
            
            elif signal_type == "liquidity_change" and avg_strength > 0.5:
                # Адаптация к изменению ликвидности
                await self._adapt_to_liquidity_change(avg_strength)
    
    async def _adapt_to_volatility(self, strength: float) -> None:
        """Адаптация к изменению волатильности."""
        # Базовая реализация - корректировка risk management параметров
        if 'risk_multiplier' in self.parameters:
            current_value = self.parameters['risk_multiplier'].value
            adjustment = 1.0 - (strength - 0.5) * 0.2  # Уменьшаем риск при высокой волатильности
            new_value = max(0.1, min(2.0, current_value * adjustment))
            await self._update_parameters({'risk_multiplier': new_value})
    
    async def _adapt_to_trend_change(self, strength: float) -> None:
        """Адаптация к смене тренда."""
        # Базовая реализация - корректировка trend following параметров
        if 'trend_sensitivity' in self.parameters:
            current_value = self.parameters['trend_sensitivity'].value
            adjustment = 1.0 + (strength - 0.5) * 0.3  # Увеличиваем чувствительность
            new_value = max(0.1, min(3.0, current_value * adjustment))
            await self._update_parameters({'trend_sensitivity': new_value})
    
    async def _adapt_to_liquidity_change(self, strength: float) -> None:
        """Адаптация к изменению ликвидности."""
        # Базовая реализация - корректировка размера позиций
        if 'position_size_multiplier' in self.parameters:
            current_value = self.parameters['position_size_multiplier'].value
            adjustment = 1.0 - (strength - 0.5) * 0.4  # Уменьшаем размер при низкой ликвидности
            new_value = max(0.1, min(2.0, current_value * adjustment))
            await self._update_parameters({'position_size_multiplier': new_value})
    
    async def _notify_observers(self, event_type: str) -> None:
        """Уведомление наблюдателей."""
        for observer in self._observers.copy():
            try:
                if asyncio.iscoroutinefunction(observer):
                    await observer(self, event_type)
                else:
                    observer(self, event_type)
            except Exception as e:
                logger.error(f"Error in observer notification: {e}")
    
    def _evaluate_strategy(self, market_data: pd.DataFrame, params: Dict[str, Any], metric: str) -> float:
        """Оценка стратегии с заданными параметрами."""
        # Упрощённая оценка - в реальности здесь был бы полный бэктест
        try:
            # Симуляция применения параметров
            returns = np.random.normal(0.001, 0.02, len(market_data))  # Случайные доходности
            
            # Расчёт метрик
            if metric == "sharpe_ratio":
                return np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0
            elif metric == "total_return":
                return np.sum(returns)
            elif metric == "max_drawdown":
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                return -np.min(drawdown)  # Отрицательное значение для минимизации
            else:
                return np.mean(returns) * 252  # Годовая доходность
                
        except Exception as e:
            logger.error(f"Error evaluating strategy: {e}")
            return -1.0  # Плохой результат при ошибке

class StrategyService:
    """Продвинутый сервис управления стратегиями."""
    
    def __init__(self):
        self._strategies: Dict[str, StrategyBase] = {}
        self._strategy_lock = asyncio.Lock()
        self._performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # ИСПРАВЛЕНО: Добавлена защита для глобальных метрик
        self._metrics_lock = asyncio.Lock()
        self._global_metrics = {
            'total_strategies': 0,
            'active_strategies': 0,
            'total_trades': 0,
            'total_pnl': 0.0
        }
        
        # Мониторинг и адаптация
        self._monitoring_task: Optional[asyncio.Task] = None
        self._adaptation_engine = AdaptationEngine()
        
        # Автоматическая оптимизация
        self._auto_optimization_enabled = False
        self._optimization_interval = timedelta(hours=24)
        
        # Мониторинг будет запущен явно через start_monitoring_async
        self._monitoring_task: Optional[asyncio.Task] = None
    
    async def register_strategy(self, strategy: StrategyBase, config: Dict[str, Any]) -> str:
        """Регистрация новой стратегии."""
        async with self._strategy_lock:
            try:
                await strategy.initialize(config)
                self._strategies[strategy.strategy_id] = strategy
                
                # Добавление наблюдателя за изменениями
                strategy.add_observer(self._strategy_event_handler)
                
                # ИСПРАВЛЕНО: Защищенное обновление метрик
                async with self._metrics_lock:
                    self._global_metrics['total_strategies'] += 1
                
                logger.info(f"Registered strategy {strategy.strategy_id} of type {strategy.strategy_type.value}")
                return strategy.strategy_id
                
            except Exception as e:
                logger.error(f"Failed to register strategy: {e}")
                raise
    
    async def start_strategy(self, strategy_id: str) -> None:
        """Запуск стратегии."""
        strategy = await self._get_strategy(strategy_id)
        await strategy.start()
        # ИСПРАВЛЕНО: Защищенное обновление метрик
        async with self._metrics_lock:
            self._global_metrics['active_strategies'] += 1
    
    async def pause_strategy(self, strategy_id: str) -> None:
        """Приостановка стратегии."""
        strategy = await self._get_strategy(strategy_id)
        await strategy.pause()
        # ИСПРАВЛЕНО: Защищенное обновление метрик
        async with self._metrics_lock:
            self._global_metrics['active_strategies'] -= 1
    
    async def stop_strategy(self, strategy_id: str) -> None:
        """Остановка стратегии."""
        strategy = await self._get_strategy(strategy_id)
        await strategy.stop()
        if strategy.state == StrategyState.ACTIVE:
            self._global_metrics['active_strategies'] -= 1
    
    async def remove_strategy(self, strategy_id: str) -> bool:
        """Удаление стратегии."""
        async with self._strategy_lock:
            if strategy_id not in self._strategies:
                return False
            
            strategy = self._strategies[strategy_id]
            
            # Остановка стратегии если активна
            if strategy.state == StrategyState.ACTIVE:
                await strategy.stop()
                self._global_metrics['active_strategies'] -= 1
            
            # Удаление из реестра
            del self._strategies[strategy_id]
            self._global_metrics['total_strategies'] -= 1
            
            logger.info(f"Removed strategy {strategy_id}")
            return True
    
    async def optimize_strategy(
        self,
        strategy_id: str,
        market_data: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.DIFFERENTIAL_EVOLUTION,
        target_metric: str = "sharpe_ratio"
    ) -> OptimizationResult:
        """Оптимизация стратегии."""
        strategy = await self._get_strategy(strategy_id)
        return await strategy.optimize_parameters(market_data, method, target_metric)
    
    async def optimize_all_strategies(self, market_data: pd.DataFrame) -> Dict[str, OptimizationResult]:
        """Оптимизация всех активных стратегий."""
        results = {}
        
        async with self._strategy_lock:
            active_strategies = [
                (sid, strategy) for sid, strategy in self._strategies.items()
                if strategy.state == StrategyState.ACTIVE
            ]
        
        for strategy_id, strategy in active_strategies:
            try:
                result = await strategy.optimize_parameters(market_data)
                results[strategy_id] = result
                logger.info(f"Optimized strategy {strategy_id}: score={result.best_score:.4f}")
            except Exception as e:
                logger.error(f"Failed to optimize strategy {strategy_id}: {e}")
        
        return results
    
    async def get_strategy_performance(self, strategy_id: str) -> Dict[str, Any]:
        """Получение данных производительности стратегии."""
        strategy = await self._get_strategy(strategy_id)
        
        performance_data = {
            'strategy_info': strategy.get_state_info(),
            'current_metrics': strategy.metrics.__dict__,
            'parameters': strategy.get_parameter_values(),
            'recent_adaptations': [
                {
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'timestamp': signal.timestamp.isoformat()
                }
                for signal in list(strategy.adaptation_signals)[-10:]
            ],
            'optimization_history': [
                {
                    'best_score': opt.best_score,
                    'iterations': opt.iterations,
                    'execution_time': opt.execution_time
                }
                for opt in strategy.optimization_history[-5:]
            ]
        }
        
        return performance_data
    
    async def get_global_metrics(self) -> Dict[str, Any]:
        """Получение глобальных метрик всех стратегий."""
        async with self._strategy_lock:
            strategies_by_state = defaultdict(int)
            strategies_by_type = defaultdict(int)
            
            for strategy in self._strategies.values():
                strategies_by_state[strategy.state.value] += 1
                strategies_by_type[strategy.strategy_type.value] += 1
        
        return {
            **self._global_metrics,
            'strategies_by_state': dict(strategies_by_state),
            'strategies_by_type': dict(strategies_by_type),
            'monitoring_active': self._monitoring_task is not None and not self._monitoring_task.done(),
            'auto_optimization_enabled': self._auto_optimization_enabled
        }
    
    async def send_adaptation_signal(
        self,
        strategy_id: str,
        signal_type: str,
        strength: float,
        description: str = "",
        metadata: Optional[Dict] = None
    ) -> None:
        """Отправка сигнала адаптации стратегии."""
        strategy = await self._get_strategy(strategy_id)
        signal = AdaptationSignal(signal_type, strength, description, metadata)
        await strategy.add_adaptation_signal(signal)
    
    async def enable_auto_optimization(self, interval_hours: int = 24) -> None:
        """Включение автоматической оптимизации."""
        self._auto_optimization_enabled = True
        self._optimization_interval = timedelta(hours=interval_hours)
        logger.info(f"Auto-optimization enabled with {interval_hours}h interval")
    
    async def disable_auto_optimization(self) -> None:
        """Отключение автоматической оптимизации."""
        self._auto_optimization_enabled = False
        logger.info("Auto-optimization disabled")
    
    async def _get_strategy(self, strategy_id: str) -> StrategyBase:
        """Получение стратегии по ID."""
        async with self._strategy_lock:
            if strategy_id not in self._strategies:
                raise ValueError(f"Strategy {strategy_id} not found")
            return self._strategies[strategy_id]
    
    async def _strategy_event_handler(self, strategy: StrategyBase, event_type: str) -> None:
        """Обработчик событий стратегии."""
        try:
            # Обновление глобальных метрик
            if event_type == "trade_completed":
                self._global_metrics['total_trades'] += 1
                self._global_metrics['total_pnl'] += getattr(strategy, 'last_trade_pnl', 0.0)
            
            # Сохранение в историю производительности
            performance_point = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'metrics': strategy.metrics.__dict__.copy(),
                'state': strategy.state.value
            }
            
            self._performance_history[strategy.strategy_id].append(performance_point)
            
            # Ограничение размера истории
            if len(self._performance_history[strategy.strategy_id]) > 1000:
                self._performance_history[strategy.strategy_id] = (
                    self._performance_history[strategy.strategy_id][-500:]
                )
            
            logger.debug(f"Strategy {strategy.strategy_id} event: {event_type}")
            
        except Exception as e:
            logger.error(f"Error handling strategy event: {e}")
    
    async def start_monitoring_async(self) -> None:
        """Запуск фонового мониторинга (асинхронно)."""
        if self._monitoring_task is not None:
            return  # Мониторинг уже запущен
            
        async def monitoring_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Каждые 5 минут
                    await self._monitor_strategies()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in strategy monitoring: {e}")
        
        self._monitoring_task = asyncio.create_task(monitoring_loop())
    
    async def stop_monitoring_async(self) -> None:
        """Остановка фонового мониторинга."""
        if self._monitoring_task is not None:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            finally:
                self._monitoring_task = None
    
    async def _monitor_strategies(self) -> None:
        """Мониторинг состояния стратегий."""
        async with self._strategy_lock:
            strategies_to_check = list(self._strategies.values())
        
        for strategy in strategies_to_check:
            try:
                # Проверка необходимости автооптимизации
                if (self._auto_optimization_enabled and 
                    strategy.state == StrategyState.ACTIVE and
                    strategy.last_optimization and
                    datetime.now() - strategy.last_optimization > self._optimization_interval):
                    
                    # Запуск автооптимизации (упрощённо)
                    logger.info(f"Triggering auto-optimization for strategy {strategy.strategy_id}")
                    # В реальности здесь был бы запуск с актуальными рыночными данными
                
                # Мониторинг производительности
                if strategy.metrics.total_trades > 0:
                    # Проверка на деградацию производительности
                    if strategy.metrics.win_rate < 0.3 and strategy.metrics.total_trades > 20:
                        await strategy.add_adaptation_signal(
                            AdaptationSignal(
                                "performance_degradation",
                                0.8,
                                f"Win rate dropped to {strategy.metrics.win_rate:.2%}"
                            )
                        )
                
            except Exception as e:
                logger.error(f"Error monitoring strategy {strategy.strategy_id}: {e}")

class AdaptationEngine:
    """Двигатель адаптации стратегий."""
    
    def __init__(self):
        self.market_regime_detector = MarketRegimeDetector()
        self.adaptation_rules: List[Callable] = []
        self._adaptation_history: List[Dict[str, Any]] = []
    
    def add_adaptation_rule(self, rule: Callable) -> None:
        """Добавление правила адаптации."""
        self.adaptation_rules.append(rule)
    
    async def analyze_market_conditions(self, market_data: pd.DataFrame) -> List[AdaptationSignal]:
        """Анализ рыночных условий и генерация сигналов адаптации."""
        signals = []
        
        try:
            # Определение рыночного режима
            regime = await self.market_regime_detector.detect_regime(market_data)
            
            # Анализ волатильности
            volatility_signal = self._analyze_volatility(market_data)
            if volatility_signal:
                signals.append(volatility_signal)
            
            # Анализ тренда
            trend_signal = self._analyze_trend(market_data)
            if trend_signal:
                signals.append(trend_signal)
            
            # Анализ ликвидности
            liquidity_signal = self._analyze_liquidity(market_data)
            if liquidity_signal:
                signals.append(liquidity_signal)
            
        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
        
        return signals
    
    def _analyze_volatility(self, market_data: pd.DataFrame) -> Optional[AdaptationSignal]:
        """Анализ изменения волатильности."""
        if 'close' not in market_data.columns or len(market_data) < 40:
            return None
        
        returns = market_data['close'].pct_change().dropna()
        current_vol = returns.tail(20).std()
        historical_vol = returns.head(-20).std()
        
        if historical_vol > 0:
            vol_change = abs(current_vol - historical_vol) / historical_vol
            if vol_change > 0.5:  # Изменение волатильности более 50%
                strength = min(1.0, vol_change)
                return AdaptationSignal(
                    "volatility_change",
                    strength,
                    f"Volatility changed by {vol_change:.1%}",
                    {"current_vol": current_vol, "historical_vol": historical_vol}
                )
        
        return None
    
    def _analyze_trend(self, market_data: pd.DataFrame) -> Optional[AdaptationSignal]:
        """Анализ изменения тренда."""
        if 'close' not in market_data.columns or len(market_data) < 50:
            return None
        
        prices = market_data['close']
        
        # Простой анализ тренда через скользящие средние
        ma_short = prices.tail(20).mean()
        ma_long = prices.tail(50).mean()
        prev_ma_short = prices.tail(40).head(20).mean()
        prev_ma_long = prices.tail(70).head(50).mean()
        
        # Определение смены тренда
        current_trend = 1 if ma_short > ma_long else -1
        prev_trend = 1 if prev_ma_short > prev_ma_long else -1
        
        if current_trend != prev_trend:
            strength = 0.7  # Средняя сила сигнала при смене тренда
            return AdaptationSignal(
                "trend_change",
                strength,
                f"Trend changed from {'bullish' if prev_trend > 0 else 'bearish'} to {'bullish' if current_trend > 0 else 'bearish'}",
                {"current_trend": current_trend, "previous_trend": prev_trend}
            )
        
        return None
    
    def _analyze_liquidity(self, market_data: pd.DataFrame) -> Optional[AdaptationSignal]:
        """Анализ изменения ликвидности."""
        if 'volume' not in market_data.columns or len(market_data) < 30:
            return None
        
        current_volume = market_data['volume'].tail(10).mean()
        historical_volume = market_data['volume'].head(-10).mean()
        
        if historical_volume > 0:
            volume_change = abs(current_volume - historical_volume) / historical_volume
            if volume_change > 0.3:  # Изменение объёма более 30%
                strength = min(1.0, volume_change)
                direction = "increased" if current_volume > historical_volume else "decreased"
                return AdaptationSignal(
                    "liquidity_change",
                    strength,
                    f"Liquidity {direction} by {volume_change:.1%}",
                    {"current_volume": current_volume, "historical_volume": historical_volume}
                )
        
        return None

class MarketRegimeDetector:
    """Детектор рыночных режимов."""
    
    async def detect_regime(self, market_data: pd.DataFrame) -> str:
        """Определение текущего рыночного режима."""
        if 'close' not in market_data.columns or len(market_data) < 100:
            return "unknown"
        
        try:
            prices = market_data['close']
            returns = prices.pct_change().dropna()
            
            # Анализ волатильности
            volatility = returns.std()
            
            # Анализ тренда
            ma_20 = prices.tail(20).mean()
            ma_50 = prices.tail(50).mean()
            
            # Определение режима
            if volatility > 0.03:  # Высокая волатильность
                return "high_volatility"
            elif ma_20 > ma_50 * 1.02:  # Восходящий тренд
                return "bullish_trend"
            elif ma_20 < ma_50 * 0.98:  # Нисходящий тренд
                return "bearish_trend"
            else:
                return "sideways"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"

# Глобальный экземпляр сервиса стратегий
strategy_service = StrategyService()
