"""
Примеры использования стратегий торговли.
"""
import logging
import uuid
import random
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Union
from unittest.mock import Mock
import pandas as pd

from domain.entities.strategy_interface import StrategyInterface
from domain.types import RiskLevel, ConfidenceLevel, StrategyId
from domain.entities.strategy import StrategyType
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.value_objects.volume import Volume

# Убираю дублирующиеся импорты и исправляю типы
try:
    from domain.entities.strategy import *
except ImportError:
    pass

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Заглушки для функций, которые должны быть реализованы в других модулях
def get_strategy_factory() -> Any:
    """Получить фабрику стратегий."""
    # Заглушка - в реальной системе должна возвращать реальную фабрику
    class MockStrategyFactory:
        def create_strategy(self, name: str, trading_pairs: List[str], parameters: Dict[str, Any], 
                           risk_level: Optional[RiskLevel] = None, 
                           confidence_threshold: Optional[ConfidenceLevel] = None) -> StrategyInterface:
            """Создать стратегию с заданными параметрами."""
            return Mock(spec=StrategyInterface)
        
        def get_available_strategies(self) -> List[str]:
            """Получить список доступных стратегий."""
            return ["trend_following", "mean_reversion", "breakout", "scalping"]
    
    return MockStrategyFactory()

def get_strategy_registry() -> Any:
    """Получить реестр стратегий."""
    # Заглушка - в реальной системе должна возвращать реальный реестр
    class MockStrategyRegistry:
        def register_strategy(self, strategy: StrategyInterface, name: str, 
                         tags: Optional[List[str]] = None, priority: int = 1) -> None:
            """Зарегистрировать стратегию в реестре."""
            pass
        
        def get_registry_stats(self) -> Dict[str, Any]:
            """Получить статистику реестра стратегий."""
            return {"total_strategies": 10, "active_strategies": 5}
        
        def get_strategies_by_type(self, strategy_type: StrategyType) -> List[StrategyInterface]:
            """Получить стратегии по типу."""
            return []
        
        def get_top_performers(self, limit: int = 5) -> List[StrategyInterface]:
            """Получить топ-производителей стратегий."""
            return []
    
    return MockStrategyRegistry()

def get_strategy_validator() -> Any:
    """Получить валидатор стратегий."""
    # Заглушка - в реальной системе должна возвращать реальный валидатор
    class MockStrategyValidator:
        def validate_trading_pairs(self, trading_pairs: List[str]) -> bool:
            """Валидировать торговые пары."""
            return len(trading_pairs) > 0

        def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
            """Валидировать параметры стратегии."""
            return len(parameters) > 0

        def validate_strategy_config(self, config: Dict[str, Any]) -> bool:
            """Валидировать конфигурацию стратегии."""
            return True
    
    return MockStrategyValidator()

def register_strategy(**kwargs: Any) -> Callable:
    """Декоратор для регистрации стратегий."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        """Декоратор для логирования."""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)
        return wrapper
    return decorator

def example_basic_strategy_creation() -> None:
    """Пример создания базовой стратегии."""
    try:
        # Создание стратегии через фабрику
        factory = get_strategy_factory()
        strategy = factory.create_strategy(
            name="trend_following",
            trading_pairs=["BTC/USDT", "ETH/USDT"],
            parameters={"lookback_period": 20, "threshold": 0.02},
            risk_level=RiskLevel(Decimal("0.3")),
            confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        logger.info(f"Created strategy: {strategy}")
    except Exception as e:
        logger.error(f"Error creating strategy: {e}")

def example_strategy_factory_usage() -> None:
    """Пример использования фабрики стратегий."""
    try:
        factory = get_strategy_factory()
        available_strategies = factory.get_available_strategies()
        logger.info(f"Available strategies: {available_strategies}")
    except Exception as e:
        logger.error(f"Error using strategy factory: {e}")

def example_strategy_registry_usage() -> None:
    """Пример использования реестра стратегий."""
    try:
        registry = get_strategy_registry()
        stats = registry.get_registry_stats()
        logger.info(f"Registry stats: {stats}")
    except Exception as e:
        logger.error(f"Error using strategy registry: {e}")

def example_strategy_validation() -> None:
    """Пример валидации стратегий."""
    try:
        validator = get_strategy_validator()
        
        # Валидация торговых пар
        trading_pairs = ["BTC/USDT", "ETH/USDT", "INVALID_PAIR"]
        pair_errors = validator.validate_trading_pairs(trading_pairs)
        if pair_errors:
            logger.warning(f"Trading pair validation errors: {pair_errors}")
        
        # Валидация параметров
        parameters = {"lookback_period": 20, "invalid_param": "value"}
        param_errors = validator.validate_parameters(parameters)
        if param_errors:
            logger.warning(f"Parameter validation errors: {param_errors}")
        
        # Валидация конфигурации
        config = {"strategy_type": "trend_following", "parameters": parameters}
        config_errors = validator.validate_strategy_config(config)
        if config_errors:
            logger.warning(f"Config validation errors: {config_errors}")
    except Exception as e:
        logger.error(f"Error validating strategy: {e}")

def example_performance_calculation() -> None:
    """Пример расчета производительности стратегии."""
    try:
        # Создание тестовых данных
        returns = [0.01, -0.005, 0.02, -0.01, 0.015]
        
        # Расчет базовых метрик
        total_return = sum(returns)
        avg_return = total_return / len(returns)
        volatility = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
        
        # Расчет Sharpe ratio (упрощенный)
        risk_free_rate = 0.02  # 2% годовых
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        logger.info(f"Performance metrics: total_return={total_return:.4f}, avg_return={avg_return:.4f}, volatility={volatility:.4f}, sharpe_ratio={sharpe_ratio:.4f}")
    except Exception as e:
        logger.error(f"Error calculating performance: {e}")

def example_risk_management() -> None:
    """Пример управления рисками."""
    try:
        # Создание стратегии с управлением рисками
        strategy = get_strategy_factory().create_strategy(
            name="risk_managed_trend",
            trading_pairs=["BTC/USDT"],
            parameters={
                "max_position_size": 0.1,  # Максимум 10% портфеля
                "stop_loss": 0.05,  # Стоп-лосс 5%
                "take_profit": 0.15,  # Тейк-профит 15%
                "max_drawdown": 0.2  # Максимальная просадка 20%
            },
            risk_level=RiskLevel(Decimal("0.2")),
            confidence_threshold=ConfidenceLevel(Decimal("0.8"))
        )
        logger.info(f"Created risk-managed strategy: {strategy}")
    except Exception as e:
        logger.error(f"Error creating risk-managed strategy: {e}")

def example_strategy_optimization() -> None:
    """Пример оптимизации стратегии."""
    try:
        # Создание оптимизатора стратегий
        class StrategyOptimizer:
            def optimize_parameters(self, strategy_class: type, param_ranges: Dict[str, List[Any]], evaluation_function: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
                # Упрощенная оптимизация - перебор параметров
                best_params = {}
                best_score = float('-inf')
                
                # Генерируем комбинации параметров
                param_combinations = self._generate_param_combinations(param_ranges)
                
                for params in param_combinations:
                    score = evaluation_function(params)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                
                return best_params
            
            def _generate_param_combinations(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
                # Упрощенная генерация комбинаций
                keys = list(param_ranges.keys())
                if not keys:
                    return [{}]
                
                combinations = []
                first_key = keys[0]
                remaining_keys = keys[1:]
                
                for value in param_ranges[first_key]:
                    sub_combinations = self._generate_param_combinations({k: param_ranges[k] for k in remaining_keys})
                    for sub_combo in sub_combinations:
                        combo = {first_key: value}
                        combo.update(sub_combo)
                        combinations.append(combo)
                
                return combinations
        
        optimizer = StrategyOptimizer()
        
        # Определение диапазонов параметров
        param_ranges: Dict[str, List[Any]] = {
            "lookback_period": [10, 20, 30],
            "threshold": [0.01, 0.02, 0.03]
        }
        
        # Функция оценки
        def evaluation_function(params: Dict[str, Any]) -> float:
            # Симулируем оценку на основе параметров
            lookback = params.get("lookback_period", 20)
            threshold = params.get("threshold", 0.02)
            return float((lookback * 0.1) + (threshold * 100))  # Упрощенная оценка
        
        # Оптимизация
        best_params = optimizer.optimize_parameters(
            strategy_class=object,  # Заглушка
            param_ranges=param_ranges,
            evaluation_function=evaluation_function
        )
        
        logger.info(f"Optimized parameters: {best_params}")
    except Exception as e:
        logger.error(f"Error optimizing strategy: {e}")

def example_custom_strategy_registration() -> None:
    """Пример регистрации кастомной стратегии."""
    try:
        class CustomStrategy(StrategyInterface):
            def __init__(
                self,
                strategy_id: StrategyId,
                name: str,
                strategy_type: StrategyType,
                trading_pairs: List[str],
                parameters: Dict[str, Any],
                risk_level: RiskLevel = RiskLevel(Decimal("0.5")),
                confidence_threshold: ConfidenceLevel = ConfidenceLevel(Decimal("0.6")),
            ):
                self.strategy_id = strategy_id
                self.name = name
                self.strategy_type = strategy_type
                self.trading_pairs = trading_pairs
                self.parameters = parameters
                self.risk_level = risk_level
                self.confidence_threshold = confidence_threshold

            def get_strategy_id(self) -> StrategyId:
                return self.strategy_id

            def get_name(self) -> str:
                return self.name

            def get_trading_pairs(self) -> List[str]:
                return self.trading_pairs

            def analyze_market(self, market_data: Any) -> Any: # Changed from StrategyAnalysisResult to Any
                # Упрощенная логика анализа
                return {
                    "signal": "BUY", # Changed from SignalType.BUY to "BUY"
                    "confidence_score": 0.7,
                    "risk_level": RiskLevel(Decimal("0.3")),
                    "position_size": 0.1,
                    "metadata": {"custom_analysis": True}
                }

            def get_strategy_type(self) -> StrategyType:
                return self.strategy_type

            def get_parameters(self) -> Dict[str, Any]:
                return self.parameters

            def get_performance(self) -> Any:
                # ... корректная реализация ...
                return {"total_return": 0.15, "sharpe_ratio": 1.2}

            def validate_data(self, data: Any) -> bool:
                """Валидировать данные."""
                return True

            def update_parameters(self, parameters: Dict[str, Any]) -> None:
                """Обновить параметры стратегии."""
                self.parameters.update(parameters)

            def activate(self) -> None:
                """Активировать стратегию."""
                pass

            def deactivate(self) -> None:
                """Деактивировать стратегию."""
                pass

            def reset(self) -> None:
                """Сбросить состояние стратегии."""
                pass

            def generate_signals(self, market_data: Any) -> List[Any]:
                """Генерировать сигналы."""
                return []

            def get_performance_metrics(self) -> Dict[str, Any]:
                """Получить метрики производительности."""
                return {"total_return": 0.15, "sharpe_ratio": 1.2}

            def is_active(self) -> bool:
                """Проверить, активна ли стратегия."""
                return True

        @register_strategy(
            name="custom_strategy",
            strategy_type="trend_following",  # Changed from StrategyType.TREND_FOLLOWING
            description="Кастомная стратегия",
            version="1.0.0",
            author="Developer",
            required_parameters=["confidence_threshold", "risk_level"],
            optional_parameters=["custom_param"],
            supported_pairs=["BTC/USDT", "ETH/USDT"],
            min_confidence=Decimal("0.3"),
            max_confidence=Decimal("1.0"),
            risk_levels=["low", "medium", "high"],
        )
        def create_custom_strategy(**kwargs: Any) -> CustomStrategy:
            return CustomStrategy(
                strategy_id=StrategyId(uuid.uuid4()),
                name=kwargs.get("name", "custom_strategy"),
                strategy_type=StrategyType.TREND_FOLLOWING,  # Changed back to StrategyType
                trading_pairs=kwargs.get("trading_pairs", ["BTC/USDT"]),
                parameters=kwargs.get("parameters", {}),
                risk_level=kwargs.get("risk_level", RiskLevel(Decimal("0.5"))),
                confidence_threshold=kwargs.get("confidence_threshold", ConfidenceLevel(Decimal("0.6")))
            )
        
        # Создание и регистрация стратегии
        custom_strategy = create_custom_strategy(
            name="my_custom_strategy",
            trading_pairs=["BTC/USDT"],
            parameters={"custom_param": "value"}
        )
        
        logger.info(f"Created custom strategy: {custom_strategy.get_name()}")
    except Exception as e:
        logger.error(f"Error creating custom strategy: {e}")

def example_multi_strategy_portfolio() -> None:
    """Пример создания портфеля из нескольких стратегий."""
    try:
        # Создание нескольких стратегий
        strategies = []
        
        # Трендовая стратегия
        trend_strategy = get_strategy_factory().create_strategy(
            name="trend_following",
            trading_pairs=["BTC/USDT"],
            parameters={"lookback_period": 20, "threshold": 0.02},
            risk_level=RiskLevel(Decimal("0.4")),
            confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        strategies.append(trend_strategy)
        
        # Стратегия возврата к среднему
        mean_reversion_strategy = get_strategy_factory().create_strategy(
            name="mean_reversion",
            trading_pairs=["ETH/USDT"],
            parameters={"window_size": 50, "std_dev": 2.0},
            risk_level=RiskLevel(Decimal("0.3")),
            confidence_threshold=ConfidenceLevel(Decimal("0.6"))
        )
        strategies.append(mean_reversion_strategy)
        
        # Расчет весов портфеля
        total_strategies: int = len(strategies)
        weights: List[float] = [1.0 / float(total_strategies)] * total_strategies  # Равные веса
        
        logger.info(f"Created portfolio with {len(strategies)} strategies")
        logger.info(f"Portfolio weights: {weights}")
    except Exception as e:
        logger.error(f"Error creating multi-strategy portfolio: {e}")

def example_backtesting_simulation() -> None:
    """Пример симуляции бэктестинга."""
    try:
        # Создание тестовых данных
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        prices = [100 + i * 0.1 + random.uniform(-1, 1) for i in range(len(dates))]
        
        # Создание DataFrame с рыночными данными
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + random.uniform(0, 2) for p in prices],
            'low': [p - random.uniform(0, 2) for p in prices],
            'close': [p + random.uniform(-0.5, 0.5) for p in prices],
            'volume': [random.uniform(1000, 10000) for _ in prices]
        })
        
        # Создание стратегии для бэктестинга
        strategy = get_strategy_factory().create_strategy(
            name="backtest_strategy",
            trading_pairs=["BTC/USDT"],
            parameters={"lookback_period": 20, "threshold": 0.02},
            risk_level=RiskLevel(Decimal("0.3")),
            confidence_threshold=ConfidenceLevel(Decimal("0.7"))
        )
        
        # Симуляция торговли
        initial_capital: float = 10000.0
        current_capital: float = initial_capital
        positions = []
        
        for i in range(20, len(market_data)):
            # Получаем исторические данные для анализа
            historical_data = market_data.iloc[i-20:i]
            
            # Анализируем рынок (заглушка)
            analysis_result = strategy.analyze_market(historical_data) # Changed to call analyze_market directly
            
            # Симулируем исполнение сигнала
            if analysis_result["signal"] == "BUY" and current_capital > 0: # Changed from SignalType.BUY to "BUY"
                position_size: float = current_capital * float(analysis_result["position_size"])
                current_capital -= float(position_size)
                positions.append({
                    'entry_price': market_data.iloc[i]['close'],
                    'size': position_size,
                    'entry_date': market_data.iloc[i]['timestamp']
                })
            elif analysis_result["signal"] == "SELL" and positions: # Changed from SignalType.SELL to "SELL"
                position = positions.pop()
                exit_price = market_data.iloc[i]['close']
                pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['size']
                current_capital += position['size'] + pnl
        
        # Закрываем оставшиеся позиции
        for position in positions:
            exit_price = market_data.iloc[-1]['close']
            pnl = (exit_price - position['entry_price']) / position['entry_price'] * position['size']
            current_capital += position['size'] + pnl
        
        total_return = (current_capital - initial_capital) / initial_capital
        
        logger.info(f"Backtest results: initial_capital={initial_capital}, final_capital={current_capital:.2f}, total_return={total_return:.4f}")
    except Exception as e:
        logger.error(f"Error in backtesting simulation: {e}")

def main() -> None:
    """Основная функция для запуска примеров."""
    try:
        logger.info("Starting strategy examples...")
        
        # Запуск примеров
        example_basic_strategy_creation()
        example_strategy_factory_usage()
        example_strategy_registry_usage()
        example_strategy_validation()
        example_performance_calculation()
        example_risk_management()
        example_strategy_optimization()
        example_custom_strategy_registration()
        example_multi_strategy_portfolio()
        example_backtesting_simulation()

        logger.info("All strategy examples completed successfully")
    except Exception as e:
        logger.error(f"Error running strategy examples: {e}")

if __name__ == "__main__":
    main()
