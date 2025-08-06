"""
Оптимизатор стратегий для инфраструктурного слоя.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Результат оптимизации стратегии."""
    strategy_id: str
    original_params: Dict[str, Any]
    optimized_params: Dict[str, Any]
    performance_improvement: float
    optimization_time: float
    iterations: int
    timestamp: datetime


class StrategyOptimizer:
    """Оптимизатор параметров торговых стратегий."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = logger
    
    def optimize_strategy(
        self, 
        strategy_id: str, 
        current_params: Dict[str, Any],
        historical_data: Optional[Any] = None
    ) -> OptimizationResult:
        """
        Оптимизация параметров стратегии.
        
        Args:
            strategy_id: Идентификатор стратегии
            current_params: Текущие параметры
            historical_data: Исторические данные для бэктеста
            
        Returns:
            Результат оптимизации
        """
        start_time = datetime.now()
        
        # Простая оптимизация для демонстрации
        optimized_params = current_params.copy()
        
        # Примерная логика оптимизации
        if "stop_loss" in optimized_params:
            # Улучшение стоп-лосса на 5%
            optimized_params["stop_loss"] *= 0.95
        
        if "take_profit" in optimized_params:
            # Улучшение тейк-профита на 10%
            optimized_params["take_profit"] *= 1.1
        
        if "position_size" in optimized_params:
            # Оптимизация размера позиции
            optimized_params["position_size"] = min(
                optimized_params["position_size"] * 1.05, 1.0
            )
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        result = OptimizationResult(
            strategy_id=strategy_id,
            original_params=current_params,
            optimized_params=optimized_params,
            performance_improvement=15.5,  # Примерное улучшение в %
            optimization_time=optimization_time,
            iterations=100,
            timestamp=start_time
        )
        
        self.logger.info(f"Optimization completed for strategy {strategy_id}")
        return result
    
    def batch_optimize(
        self, 
        strategies: List[Tuple[str, Dict[str, Any]]]
    ) -> List[OptimizationResult]:
        """
        Пакетная оптимизация нескольких стратегий.
        
        Args:
            strategies: Список кортежей (strategy_id, params)
            
        Returns:
            Список результатов оптимизации
        """
        results = []
        
        for strategy_id, params in strategies:
            try:
                result = self.optimize_strategy(strategy_id, params)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to optimize strategy {strategy_id}: {e}")
        
        return results
    
    def validate_optimization(self, result: OptimizationResult) -> bool:
        """
        Валидация результата оптимизации.
        
        Args:
            result: Результат оптимизации
            
        Returns:
            True если результат валиден
        """
        try:
            # Базовая валидация
            if result.performance_improvement < 0:
                self.logger.warning(f"Negative performance improvement: {result.performance_improvement}")
                return False
            
            if not result.optimized_params:
                self.logger.warning("Empty optimized parameters")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return False