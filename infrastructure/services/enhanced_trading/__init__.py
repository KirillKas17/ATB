"""
Модуль enhanced trading.

Содержит декомпозированные компоненты для enhanced trading:
создание ордеров, исполнение стратегий, анализ настроений.
"""

# Создание ордеров
from .order_creation import (
    calculate_optimal_order_size,
    calculate_order_timing,
    calculate_slippage_estimate,
    create_bracket_order,
    create_iceberg_order,
    create_limit_order,
    create_market_order,
    create_smart_order,
    create_stop_limit_order,
    create_stop_order,
    create_twap_order,
    create_vwap_order,
    optimize_order_execution,
    validate_order_parameters,
)

# Анализ настроений
from .sentiment_adjustment import (
    adjust_trading_parameters,
    analyze_market_sentiment,
    analyze_news_sentiment,
    analyze_social_sentiment,
    calculate_sentiment_score,
    combine_sentiment_sources,
    detect_sentiment_shifts,
    generate_sentiment_alerts,
    validate_sentiment_data,
)

# Исполнение стратегий
from .strategy_execution import (
    apply_risk_management,
    calculate_position_size,
    calculate_risk_metrics,
    create_execution_plan,
    execute_algorithm,
    monitor_execution,
    optimize_strategy_parameters,
    validate_strategy_parameters,
)

# Утилиты
from .utils import (
    EnhancedTradingCache,
    clean_cache,
    convert_to_decimal,
    convert_to_money,
    create_empty_execution_plan,
    create_empty_order,
    create_empty_sentiment_analysis,
    normalize_data,
    validate_market_data,
    validate_order_data,
    validate_strategy_data,
    validate_trading_parameters,
    calculate_performance_metrics,
)

__all__ = [
    # Order creation
    "create_market_order",
    "create_limit_order",
    "create_stop_order",
    "create_stop_limit_order",
    "create_twap_order",
    "create_vwap_order",
    "create_iceberg_order",
    "create_bracket_order",
    "calculate_optimal_order_size",
    "calculate_order_timing",
    "validate_order_parameters",
    "calculate_slippage_estimate",
    "optimize_order_execution",
    "create_smart_order",
    # Strategy execution
    "calculate_position_size",
    "calculate_risk_metrics",
    "apply_risk_management",
    "execute_algorithm",
    "monitor_execution",
    "optimize_strategy_parameters",
    "validate_strategy_parameters",
    "create_execution_plan",
    # Sentiment adjustment
    "analyze_market_sentiment",
    "analyze_news_sentiment",
    "analyze_social_sentiment",
    "calculate_sentiment_score",
    "adjust_trading_parameters",
    "detect_sentiment_shifts",
    "validate_sentiment_data",
    "combine_sentiment_sources",
    "generate_sentiment_alerts",
    # Utils
    "validate_order_data",
    "validate_market_data",
    "validate_strategy_data",
    "create_empty_order",
    "create_empty_execution_plan",
    "create_empty_sentiment_analysis",
    "convert_to_decimal",
    "convert_to_money",
    "clean_cache",
    "validate_trading_parameters",
    "normalize_data",
    "calculate_performance_metrics",
    "EnhancedTradingCache",
]
