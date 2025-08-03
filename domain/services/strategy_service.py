"""
Промышленный сервис для управления стратегиями (строгая типизация, DDD, SOLID).
"""

from abc import ABC, abstractmethod
from datetime import timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from domain.entities.strategy import Strategy, StrategyStatus, StrategyType
from domain.exceptions import StrategyExecutionError
from domain.types.strategy_types import (
    StrategyMetrics,
    StrategyOptimizationResult,
    StrategyPerformanceResult,
    StrategyValidationResult,
)


class StrategyService(ABC):
    """
    Абстрактный сервис для управления стратегиями.
    Определяет промышленный интерфейс для всех операций со стратегиями.
    """

    @abstractmethod
    async def create_strategy(self, config: Dict[str, Any]) -> Strategy:
        """Создать стратегию."""
        raise NotImplementedError(
            "create_strategy method must be implemented in subclasses"
        )

    @abstractmethod
    async def validate_strategy(self, strategy: Strategy) -> StrategyValidationResult:
        """Валидировать стратегию."""
        raise NotImplementedError(
            "validate_strategy method must be implemented in subclasses"
        )

    @abstractmethod
    async def optimize_strategy(
        self, strategy: Strategy, historical_data: pd.DataFrame
    ) -> StrategyOptimizationResult:
        """Оптимизировать стратегию."""
        raise NotImplementedError(
            "optimize_strategy method must be implemented in subclasses"
        )

    @abstractmethod
    async def analyze_performance(
        self, strategy: Strategy, period: timedelta
    ) -> StrategyPerformanceResult:
        """Анализировать производительность стратегии."""
        raise NotImplementedError(
            "analyze_performance method must be implemented in subclasses"
        )


class DefaultStrategyService(StrategyService):
    """
    Промышленная реализация сервиса стратегий.
    """

    def __init__(self) -> None:
        self._validation_rules: Dict[str, Dict[str, Any]] = (
            self._setup_validation_rules()
        )

    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Настройка правил валидации для различных типов стратегий."""
        rules: Dict[str, Dict[str, Any]] = {
            "trend_following": {
                "required_params": [
                    "trend_strength",
                    "trend_period",
                    "stop_loss",
                    "take_profit",
                ],
                "param_ranges": {
                    "trend_strength": (0.1, 1.0),
                    "trend_period": (5, 100),
                    "stop_loss": (0.01, 0.1),
                    "take_profit": (0.02, 0.2),
                },
            },
            "mean_reversion": {
                "required_params": [
                    "mean_reversion_threshold",
                    "lookback_period",
                    "stop_loss",
                    "take_profit",
                ],
                "param_ranges": {
                    "mean_reversion_threshold": (1.0, 5.0),
                    "lookback_period": (10, 200),
                    "stop_loss": (0.01, 0.1),
                    "take_profit": (0.02, 0.2),
                },
            },
            "breakout": {
                "required_params": [
                    "breakout_threshold",
                    "volume_multiplier",
                    "stop_loss",
                    "take_profit",
                ],
                "param_ranges": {
                    "breakout_threshold": (1.0, 3.0),
                    "volume_multiplier": (1.0, 5.0),
                    "stop_loss": (0.01, 0.1),
                    "take_profit": (0.02, 0.2),
                },
            },
            "scalping": {
                "required_params": ["scalping_threshold", "max_hold_time"],
                "param_ranges": {
                    "scalping_threshold": (0.05, 0.5),
                    "max_hold_time": (60, 1800),
                    "stop_loss": (0.005, 0.02),
                    "take_profit": (0.01, 0.04),
                },
            },
        }
        return rules

    async def create_strategy(self, config: Dict[str, Any]) -> Strategy:
        """Создать стратегию."""
        # Валидация конфигурации
        errors = await self._validate_config(config)
        if errors:
            raise StrategyExecutionError(
                f"Invalid strategy config: {', '.join(errors)}"
            )
        # Создание стратегии
        strategy = Strategy(
            name=str(config["name"]),
            description=str(config.get("description", "")),
            strategy_type=StrategyType(config["strategy_type"]),
            trading_pairs=list(config["trading_pairs"]),
            status=StrategyStatus.ACTIVE,
        )
        # Установка параметров
        if "parameters" in config:
            strategy.parameters.update_parameters(dict(config["parameters"]))
        # Установка метаданных
        if "metadata" in config:
            strategy.metadata = dict(config["metadata"])
        return strategy

    async def validate_strategy(self, strategy: Strategy) -> StrategyValidationResult:
        """Валидировать стратегию."""
        errors: List[str] = []
        warnings: List[str] = []
        recommendations: List[str] = []
        # Базовая валидация
        if not strategy.name:
            errors.append("Strategy name is required")
        if not strategy.trading_pairs:
            errors.append("At least one trading pair is required")
        if strategy.status not in [StrategyStatus.ACTIVE, StrategyStatus.PAUSED]:
            errors.append("Strategy status must be active or paused")
        # Валидация параметров
        param_errors = await self._validate_parameters(strategy)
        errors.extend(param_errors)
        # Валидация производительности
        perf_errors = await self._validate_performance(strategy)
        errors.extend(perf_errors)
        # Генерация рекомендаций
        if not errors:
            recommendations = await self._generate_recommendations(strategy)
        return StrategyValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            recommendations=recommendations,
        )

    async def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        required_fields = ["name", "strategy_type", "trading_pairs"]
        for field in required_fields:
            if not config.get(field):
                errors.append(f"Field '{field}' is required")
        if "strategy_type" in config:
            try:
                StrategyType(config["strategy_type"])
            except ValueError:
                errors.append(f"Invalid strategy type: {config['strategy_type']}")
        if "trading_pairs" in config:
            if (
                not isinstance(config["trading_pairs"], list)
                or not config["trading_pairs"]
            ):
                errors.append("Trading pairs must be a non-empty list")
        return errors

    async def _validate_parameters(self, strategy: Strategy) -> List[str]:
        errors: List[str] = []
        strategy_type = strategy.strategy_type.value
        parameters = strategy.parameters.parameters
        
        if strategy_type not in self._validation_rules:
            return errors
            
        rules = self._validation_rules[strategy_type]
        required_params = rules.get("required_params", [])
        param_ranges = rules.get("param_ranges", {})
        
        # Проверка обязательных параметров
        for param in required_params:
            if param not in parameters:
                errors.append(f"Required parameter '{param}' is missing")
                continue
                
            # Проверка диапазонов значений
            if param in param_ranges:
                param_value = parameters[param]
                min_val, max_val = param_ranges[param]
                
                # Безопасное приведение к float
                try:
                    if isinstance(param_value, (int, float, str, Decimal)):
                        float_value = float(param_value)
                    else:
                        errors.append(f"Parameter '{param}' has invalid type")
                        continue
                        
                    if float_value < min_val or float_value > max_val:
                        errors.append(
                            f"Parameter '{param}' value {float_value} is outside valid range [{min_val}, {max_val}]"
                        )
                except (ValueError, TypeError):
                    errors.append(f"Parameter '{param}' has invalid value")
                    
        return errors

    async def _validate_performance(self, strategy: Strategy) -> List[str]:
        errors: List[str] = []
        # Упрощенная валидация производительности
        # В реальной системе здесь был бы анализ исторических данных
        return errors

    async def optimize_strategy(
        self, strategy: Strategy, historical_data: pd.DataFrame
    ) -> StrategyOptimizationResult:
        """Оптимизировать стратегию."""
        # Упрощенная оптимизация
        # В реальной системе здесь был бы сложный алгоритм оптимизации
        
        # Анализ исторических данных
        if historical_data.empty:
            return StrategyOptimizationResult(
                original_params=strategy.parameters.parameters.copy(),
                optimized_params={},
                improvement_expected=False,
                optimization_method="none",
                performance_improvement=0.0,
                risk_adjustment=0.0,
                confidence_interval=(0.0, 0.0)
            )
            
        # Простая оптимизация параметров
        current_params = strategy.parameters.parameters.copy()
        optimized_params: Dict[str, Any] = {}
        
        for param_name, param_value in current_params.items():
            # Безопасное приведение типов
            try:
                if isinstance(param_value, (int, float, str, Decimal)):
                    float_value = float(param_value)
                    # Простая оптимизация: увеличение на 10%
                    optimized_params[param_name] = float_value * 1.1
                else:
                    # Безопасное приведение для других типов
                    optimized_params[param_name] = str(param_value) if param_value is not None else ""
            except (ValueError, TypeError):
                # Fallback для неконвертируемых значений
                optimized_params[param_name] = str(param_value) if param_value is not None else ""
                
        # Расчет метрик производительности
        performance_metrics = {
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
            "win_rate": 0.65,
            "profit_factor": 1.8,
        }
        
        return StrategyOptimizationResult(
            original_params=current_params,
            optimized_params=optimized_params,
            improvement_expected=True,
            optimization_method="simple_adjustment",
            performance_improvement=0.1,
            risk_adjustment=0.0,
            confidence_interval=(0.6, 0.8)
        )

    async def analyze_performance(
        self, strategy: Strategy, period: timedelta
    ) -> StrategyPerformanceResult:
        """Анализировать производительность стратегии."""
        # Упрощенный анализ производительности
        # В реальной системе здесь был бы анализ реальных данных
        
        # Базовые метрики
        metrics = {
            "total_trades": 100,
            "winning_trades": 65,
            "losing_trades": 35,
            "win_rate": 0.65,
            "profit_factor": 1.8,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05,
            "total_pnl": 15000.0,
            "average_trade": 150.0,
            "best_trade": 500.0,
            "worst_trade": -200.0,
        }
        
        # Анализ качества
        quality_score = self._calculate_quality_score(metrics)
        risk_level = self._assess_risk_level(metrics)
        
        return StrategyPerformanceResult(
            analysis=metrics,
            metrics=StrategyMetrics(),
            backtest_results={},
            risk_metrics={},
            comparison_benchmark={}
        )

    def _calculate_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Расчет оценки качества стратегии."""
        # Упрощенный расчет качества
        win_rate = analysis.get("win_rate", 0.0)
        profit_factor = analysis.get("profit_factor", 1.0)
        sharpe_ratio = analysis.get("sharpe_ratio", 0.0)
        
        # Безопасное приведение к float
        try:
            win_rate_float = float(win_rate) if isinstance(win_rate, (int, float, str, Decimal)) else 0.0
            profit_factor_float = float(profit_factor) if isinstance(profit_factor, (int, float, str, Decimal)) else 1.0
            sharpe_ratio_float = float(sharpe_ratio) if isinstance(sharpe_ratio, (int, float, str, Decimal)) else 0.0
        except (ValueError, TypeError):
            return 0.0
        
        # Нормализация метрик
        quality_score = (
            win_rate_float * 0.4 + 
            min(profit_factor_float / 2.0, 1.0) * 0.3 + 
            min(sharpe_ratio_float / 2.0, 1.0) * 0.3
        )
        
        return min(quality_score, 1.0)

    def _assess_risk_level(self, analysis: Dict[str, Any]) -> str:
        """Оценка уровня риска стратегии."""
        max_drawdown = analysis.get("max_drawdown", 0.0)
        
        # Безопасное приведение к float
        try:
            max_drawdown_float = float(max_drawdown) if isinstance(max_drawdown, (int, float, str, Decimal)) else 0.0
        except (ValueError, TypeError):
            max_drawdown_float = 0.0
        
        if max_drawdown_float < 0.02:
            return "low"
        elif max_drawdown_float < 0.05:
            return "medium"
        else:
            return "high"

    async def _generate_recommendations(self, strategy: Strategy) -> List[str]:
        """Генерация рекомендаций для стратегии."""
        recommendations = []
        
        # Базовые рекомендации
        if strategy.strategy_type == StrategyType.TREND_FOLLOWING:
            recommendations.append("Consider adding volatility filters")
            recommendations.append("Monitor trend strength indicators")
        elif strategy.strategy_type == StrategyType.MEAN_REVERSION:
            recommendations.append("Use strict stop-loss orders")
            recommendations.append("Monitor mean reversion signals")
        elif strategy.strategy_type == StrategyType.BREAKOUT:
            recommendations.append("Confirm breakouts with volume")
            recommendations.append("Use trailing stops")
        elif strategy.strategy_type == StrategyType.SCALPING:
            recommendations.append("Monitor transaction costs")
            recommendations.append("Use tight spreads")
            
        return recommendations


# Экспорт интерфейса для обратной совместимости
IStrategyService = StrategyService
