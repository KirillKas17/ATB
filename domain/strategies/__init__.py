"""
Модуль стратегий - промышленная реализация торговых стратегий.
Этот модуль предоставляет полный набор инструментов для создания, управления
и выполнения торговых стратегий в соответствии с принципами DDD и SOLID.
"""

# Базовые стратегии
from domain.strategies.base_strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy,
    ScalpingStrategy,
    ArbitrageStrategy,
)

# Новые продвинутые модули
from domain.strategies.quantum_arbitrage_strategy import QuantumArbitrageStrategy
# from domain.intelligence.pattern_analyzer import QuantumPatternAnalyzer  # Временно отключен
from domain.prediction.neural_market_predictor import NeuralMarketPredictor

# Фабрика и реестр
from .strategy_factory import StrategyFactory, get_strategy_factory, register_strategy
from .strategy_registry import StrategyRegistry, get_strategy_registry

# Валидаторы
from .validators import StrategyValidator, get_strategy_validator

# Отдельные файлы стратегий (для обратной совместимости)
# Фабрика и реестр
# Валидаторы и исключения
from .exceptions import (
    StrategyConfigurationError,
    StrategyCreationError,
    StrategyDuplicateError,
    StrategyError,
    StrategyExecutionError,
    StrategyFactoryError,
    StrategyMarketError,
    StrategyNotFoundError,
    StrategyPerformanceError,
    StrategyRegistryError,
    StrategyResourceError,
    StrategyRiskError,
    StrategySignalError,
    StrategyStateError,
    StrategyTimeoutError,
    StrategyValidationError,
    create_strategy_error,
    get_strategy_error_details,
    is_strategy_error,
)

# Основные интерфейсы и типы
from domain.type_definitions.strategy_types import (
    ArbitrageParams,
    BreakoutParams,
    MarketRegime,
    MeanReversionParams,
    RiskProfile,
    ScalpingParams,
    StrategyCategory,
    StrategyConfig,
    StrategyMetrics,
    StrategyParameters,
    Timeframe,
    TimeHorizon,
    TrendFollowingParams,
)

# Утилиты и вспомогательные функции
from .utils import (
    StrategyOptimizer,
    StrategyPerformanceCalculator,
    StrategyRiskManager,
    calculate_avg_trade,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_win_rate,
    normalize_parameters,
    validate_strategy_parameters,
    validate_trading_pair,
)

# Версия модуля
__version__ = "2.0.0"
# Основные экспорты
__all__ = [
    # Интерфейсы и типы
    "StrategyInterface",
    "StrategyCategory",
    "RiskProfile",
    "TimeHorizon",
    "MarketCondition",
    "Timeframe",
    "MarketRegime",
    "StrategyParameters",
    "TrendFollowingParams",
    "MeanReversionParams",
    "BreakoutParams",
    "ScalpingParams",
    "ArbitrageParams",
    "StrategyConfig",
    "StrategyMetrics",
    # Базовые стратегии
    "TrendFollowingStrategy",
    "MeanReversionStrategy",
    "BreakoutStrategy",
    "ScalpingStrategy",
    "ArbitrageStrategy",
    # Новые продвинутые модули
    "QuantumArbitrageStrategy",
    # "QuantumPatternAnalyzer",  # Временно отключен
    "NeuralMarketPredictor",
    # Реализации стратегий
    "TrendFollowingStrategyImpl",
    "MeanReversionStrategyImpl",
    "BreakoutStrategyImpl",
    "ScalpingStrategyImpl",
    "ArbitrageStrategyImpl",
    # Фабрика и реестр
    "StrategyFactory",
    "get_strategy_factory",
    "register_strategy",
    "StrategyRegistry",
    "get_strategy_registry",
    # Валидаторы и исключения
    "StrategyValidator",
    "StrategyError",
    "StrategyCreationError",
    "StrategyValidationError",
    "StrategyExecutionError",
    "StrategyConfigurationError",
    "StrategyStateError",
    "StrategyNotFoundError",
    "StrategyDuplicateError",
    "StrategyPerformanceError",
    "StrategyRiskError",
    "StrategyMarketError",
    "StrategySignalError",
    "StrategyFactoryError",
    "StrategyRegistryError",
    "StrategyTimeoutError",
    "StrategyResourceError",
    "create_strategy_error",
    "is_strategy_error",
    "get_strategy_error_details",
    # Утилиты
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_win_rate",
    "calculate_profit_factor",
    "calculate_avg_trade",
    "validate_trading_pair",
    "validate_strategy_parameters",
    "normalize_parameters",
    "StrategyPerformanceCalculator",
    "StrategyRiskManager",
    "StrategyOptimizer",
    # Версия
    "__version__",
]
# Дополнительные экспорты для обратной совместимости
__all__.extend(
    [
        # Алиасы для обратной совместимости
        "TrendFollowingParameters",
        "MeanReversionParameters",
        "BreakoutParameters",
        "ScalpingParameters",
        "ArbitrageParameters",
        "StrategyConfiguration",
    ]
)


# Инициализация глобальных экземпляров
def _initialize_globals() -> bool:
    """Инициализировать глобальные экземпляры."""
    try:
        # Инициализируем фабрику
        factory = get_strategy_factory()
        # Инициализируем реестр
        registry = get_strategy_registry()
        return True
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to initialize strategy module globals: {e}")
        return False


# Автоматическая инициализация при импорте модуля
_initialize_globals()
