"""
Константы стратегий - промышленные конфигурации и настройки.
"""

from decimal import Decimal

from domain.strategies.strategy_types import (
    ArbitrageParams,
    BreakoutParams,
    MarketCondition,
    MarketRegime,
    MeanReversionParams,
    RiskProfile,
    ScalpingParams,
    StrategyCategory,
    StrategyParameters,
    Timeframe,
    TimeHorizon,
    TrendFollowingParams,
)

# Поддерживаемые торговые пары
SUPPORTED_TRADING_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "ADA/USDT",
    "DOT/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "XRP/USDT",
    "EOS/USDT",
    "TRX/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "AVAX/USDT",
    "MATIC/USDT",
    "UNI/USDT",
    "ATOM/USDT",
    "FTM/USDT",
    "NEAR/USDT",
    "ALGO/USDT",
    "VET/USDT",
]
# Уровни риска
RISK_LEVELS = {
    "low": {
        "max_position_size": Decimal("0.05"),
        "stop_loss": Decimal("0.01"),
        "take_profit": Decimal("0.02"),
        "max_daily_loss": Decimal("0.02"),
        "max_drawdown": Decimal("0.1"),
    },
    "medium": {
        "max_position_size": Decimal("0.1"),
        "stop_loss": Decimal("0.02"),
        "take_profit": Decimal("0.04"),
        "max_daily_loss": Decimal("0.05"),
        "max_drawdown": Decimal("0.2"),
    },
    "high": {
        "max_position_size": Decimal("0.2"),
        "stop_loss": Decimal("0.03"),
        "take_profit": Decimal("0.06"),
        "max_daily_loss": Decimal("0.1"),
        "max_drawdown": Decimal("0.3"),
    },
}
# Пороги уверенности
CONFIDENCE_THRESHOLDS = {
    "very_low": Decimal("0.3"),
    "low": Decimal("0.5"),
    "medium": Decimal("0.7"),
    "high": Decimal("0.8"),
    "very_high": Decimal("0.9"),
}
# Метрики производительности
PERFORMANCE_METRICS = [
    "total_trades",
    "winning_trades",
    "losing_trades",
    "win_rate",
    "profit_factor",
    "total_pnl",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "avg_trade",
    "largest_win",
    "largest_loss",
    "consecutive_wins",
    "consecutive_losses",
    "avg_execution_time",
    "success_rate",
]
# Категории стратегий
STRATEGY_CATEGORIES = {
    StrategyCategory.TREND_FOLLOWING: {
        "name": "Trend Following",
        "description": "Стратегии следования за трендом",
        "risk_profile": RiskProfile.MODERATE,
        "time_horizon": TimeHorizon.MEDIUM,
        "suitable_market_conditions": [
            MarketCondition.TRENDING,
            MarketCondition.BULL_MARKET,
            MarketCondition.BEAR_MARKET,
        ],
    },
    StrategyCategory.MEAN_REVERSION: {
        "name": "Mean Reversion",
        "description": "Стратегии возврата к среднему",
        "risk_profile": RiskProfile.CONSERVATIVE,
        "time_horizon": TimeHorizon.SHORT,
        "suitable_market_conditions": [
            MarketCondition.SIDEWAYS,
            MarketCondition.RANGING,
        ],
    },
    StrategyCategory.BREAKOUT: {
        "name": "Breakout",
        "description": "Стратегии пробоя",
        "risk_profile": RiskProfile.AGGRESSIVE,
        "time_horizon": TimeHorizon.SHORT,
        "suitable_market_conditions": [
            MarketCondition.BREAKOUT,
            MarketCondition.VOLATILE,
        ],
    },
    StrategyCategory.SCALPING: {
        "name": "Scalping",
        "description": "Скальпинг стратегии",
        "risk_profile": RiskProfile.AGGRESSIVE,
        "time_horizon": TimeHorizon.ULTRA_SHORT,
        "suitable_market_conditions": [
            MarketCondition.VOLATILE,
            MarketCondition.TRENDING,
        ],
    },
    StrategyCategory.ARBITRAGE: {
        "name": "Arbitrage",
        "description": "Арбитражные стратегии",
        "risk_profile": RiskProfile.CONSERVATIVE,
        "time_horizon": TimeHorizon.ULTRA_SHORT,
        "suitable_market_conditions": [
            MarketCondition.VOLATILE,
            MarketCondition.SIDEWAYS,
        ],
    },
}
# Конфигурации по умолчанию для стратегий
DEFAULT_STRATEGY_CONFIG = {
    "general": {
        "enable_logging": True,
        "enable_metrics": True,
        "enable_alerts": True,
        "max_signals": 5,
        "signal_cooldown": 300,
        "execution_timeout": 30,
        "max_slippage": Decimal("0.001"),
        "version": "1.0.0",
        "author": "System",
        "description": "Default strategy configuration",
    },
    "risk_management": {
        "max_concurrent_positions": 5,
        "max_daily_trades": 50,
        "max_daily_loss": Decimal("0.05"),
        "max_drawdown": Decimal("0.2"),
        "position_sizing_method": "fixed",
        "risk_per_trade": Decimal("0.02"),
    },
    "monitoring": {
        "performance_check_interval": 60,
        "risk_check_interval": 30,
        "alert_thresholds": {
            "drawdown": Decimal("0.15"),
            "daily_loss": Decimal("0.03"),
            "consecutive_losses": 5,
        },
    },
}
# Конфигурации по умолчанию для конкретных стратегий
DEFAULT_TREND_FOLLOWING_CONFIG = TrendFollowingParams(
    short_period=10,
    long_period=20,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    trend_strength_threshold=Decimal("0.7"),
    trend_confirmation_period=3,
    volume_confirmation=True,
    volume_threshold=Decimal("1.5"),
)
DEFAULT_MEAN_REVERSION_CONFIG = MeanReversionParams(
    lookback_period=50,
    deviation_threshold=Decimal("2.0"),
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    bb_period=20,
    bb_std_dev=Decimal("2.0"),
    min_reversion_probability=Decimal("0.6"),
    max_holding_period=24,
)
DEFAULT_BREAKOUT_CONFIG = BreakoutParams(
    breakout_threshold=Decimal("1.5"),
    volume_multiplier=Decimal("2.0"),
    volume_confirmation_period=3,
    support_resistance_period=20,
    level_tolerance=Decimal("0.001"),
    confirmation_period=2,
    false_breakout_filter=True,
    min_volatility=Decimal("0.01"),
    max_volatility=Decimal("0.1"),
)
DEFAULT_SCALPING_CONFIG = ScalpingParams(
    profit_threshold=Decimal("0.001"),
    stop_loss=Decimal("0.0005"),
    max_hold_time=300,
    min_hold_time=10,
    min_volatility=Decimal("0.0001"),
    max_volatility=Decimal("0.01"),
    min_volume=Decimal("1000"),
    volume_spike_threshold=Decimal("2.0"),
    execution_timeout=5,
    max_slippage=Decimal("0.0001"),
)
DEFAULT_ARBITRAGE_CONFIG = ArbitrageParams(
    min_spread=Decimal("0.001"),
    max_slippage=Decimal("0.0005"),
    execution_timeout=10,
    max_hold_time=60,
    min_liquidity=Decimal("10000"),
    max_order_size=Decimal("1000"),
    max_total_fees=Decimal("0.002"),
    include_fees_in_calculation=True,
    exchange_monitoring_interval=1,
    price_update_frequency=1,
)
# Временные фреймы и их настройки
TIMEFRAME_CONFIGS = {
    Timeframe.TICK: {
        "description": "Tick data",
        "min_interval": 1,
        "max_bars": 10000,
        "suitable_strategies": [StrategyCategory.SCALPING, StrategyCategory.ARBITRAGE],
    },
    Timeframe.SECOND_1: {
        "description": "1 second",
        "min_interval": 1,
        "max_bars": 86400,
        "suitable_strategies": [StrategyCategory.SCALPING, StrategyCategory.ARBITRAGE],
    },
    Timeframe.SECOND_5: {
        "description": "5 seconds",
        "min_interval": 5,
        "max_bars": 17280,
        "suitable_strategies": [StrategyCategory.SCALPING],
    },
    Timeframe.SECOND_15: {
        "description": "15 seconds",
        "min_interval": 15,
        "max_bars": 5760,
        "suitable_strategies": [StrategyCategory.SCALPING],
    },
    Timeframe.SECOND_30: {
        "description": "30 seconds",
        "min_interval": 30,
        "max_bars": 2880,
        "suitable_strategies": [StrategyCategory.SCALPING],
    },
    Timeframe.MINUTE_1: {
        "description": "1 minute",
        "min_interval": 60,
        "max_bars": 1440,
        "suitable_strategies": [
            StrategyCategory.SCALPING,
            StrategyCategory.MEAN_REVERSION,
        ],
    },
    Timeframe.MINUTE_5: {
        "description": "5 minutes",
        "min_interval": 300,
        "max_bars": 288,
        "suitable_strategies": [
            StrategyCategory.MEAN_REVERSION,
            StrategyCategory.BREAKOUT,
        ],
    },
    Timeframe.MINUTE_15: {
        "description": "15 minutes",
        "min_interval": 900,
        "max_bars": 96,
        "suitable_strategies": [
            StrategyCategory.MEAN_REVERSION,
            StrategyCategory.BREAKOUT,
        ],
    },
    Timeframe.MINUTE_30: {
        "description": "30 minutes",
        "min_interval": 1800,
        "max_bars": 48,
        "suitable_strategies": [
            StrategyCategory.BREAKOUT,
            StrategyCategory.TREND_FOLLOWING,
        ],
    },
    Timeframe.HOUR_1: {
        "description": "1 hour",
        "min_interval": 3600,
        "max_bars": 24,
        "suitable_strategies": [
            StrategyCategory.TREND_FOLLOWING,
            StrategyCategory.BREAKOUT,
        ],
    },
    Timeframe.HOUR_4: {
        "description": "4 hours",
        "min_interval": 14400,
        "max_bars": 6,
        "suitable_strategies": [StrategyCategory.TREND_FOLLOWING],
    },
    Timeframe.HOUR_6: {
        "description": "6 hours",
        "min_interval": 21600,
        "max_bars": 4,
        "suitable_strategies": [StrategyCategory.TREND_FOLLOWING],
    },
    Timeframe.HOUR_12: {
        "description": "12 hours",
        "min_interval": 43200,
        "max_bars": 2,
        "suitable_strategies": [StrategyCategory.TREND_FOLLOWING],
    },
    Timeframe.DAY_1: {
        "description": "1 day",
        "min_interval": 86400,
        "max_bars": 365,
        "suitable_strategies": [StrategyCategory.TREND_FOLLOWING],
    },
}
# Ограничения и лимиты
STRATEGY_LIMITS = {
    "max_strategies_per_account": 10,
    "max_concurrent_strategies": 5,
    "max_trades_per_strategy": 1000,
    "max_positions_per_strategy": 10,
    "min_account_balance": Decimal("100"),
    "max_strategy_memory_mb": 512,
    "max_strategy_cpu_percent": 50,
    "max_strategy_disk_mb": 100,
}
# Настройки логирования
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "strategies.log",
    "max_file_size_mb": 100,
    "backup_count": 5,
    "enable_console": True,
    "enable_file": True,
}
# Настройки мониторинга
MONITORING_CONFIG = {
    "enable_performance_monitoring": True,
    "enable_risk_monitoring": True,
    "enable_system_monitoring": True,
    "performance_check_interval": 60,
    "risk_check_interval": 30,
    "system_check_interval": 300,
    "alert_channels": ["log", "email", "webhook"],
    "metrics_retention_days": 30,
}
# Настройки оптимизации
OPTIMIZATION_CONFIG = {
    "enable_auto_optimization": False,
    "optimization_interval_hours": 24,
    "optimization_method": "grid_search",
    "max_optimization_iterations": 1000,
    "optimization_metrics": ["sharpe_ratio", "profit_factor", "max_drawdown"],
    "cross_validation_folds": 5,
    "min_optimization_data_points": 100,
}
# Настройки бэктестинга
BACKTEST_CONFIG = {
    "default_commission": Decimal("0.001"),
    "default_slippage": Decimal("0.0005"),
    "default_initial_balance": Decimal("10000"),
    "default_currency": "USDT",
    "enable_realistic_execution": True,
    "enable_commission_calculation": True,
    "enable_slippage_simulation": True,
    "min_data_points": 1000,
}
# Настройки уведомлений
ALERT_CONFIG = {
    "enable_alerts": True,
    "alert_levels": ["info", "warning", "error", "critical"],
    "alert_channels": {
        "info": ["log"],
        "warning": ["log", "email"],
        "error": ["log", "email", "webhook"],
        "critical": ["log", "email", "webhook", "sms"],
    },
    "alert_thresholds": {
        "drawdown": Decimal("0.15"),
        "daily_loss": Decimal("0.03"),
        "consecutive_losses": 5,
        "execution_time": 60,
        "error_rate": Decimal("0.1"),
    },
}
# Настройки безопасности
SECURITY_CONFIG = {
    "enable_parameter_validation": True,
    "enable_risk_checks": True,
    "enable_execution_limits": True,
    "max_api_calls_per_minute": 100,
    "max_order_size_usd": Decimal("10000"),
    "max_daily_volume_usd": Decimal("100000"),
    "enable_fraud_detection": True,
    "suspicious_activity_threshold": 5,
}
# Настройки кэширования
CACHE_CONFIG = {
    "enable_strategy_cache": True,
    "enable_data_cache": True,
    "strategy_cache_ttl": 3600,
    "data_cache_ttl": 300,
    "max_cache_size_mb": 1000,
    "cache_cleanup_interval": 3600,
}
# Настройки производительности
PERFORMANCE_CONFIG = {
    "enable_profiling": False,
    "enable_metrics_collection": True,
    "metrics_aggregation_interval": 60,
    "performance_thresholds": {
        "max_execution_time": 30,
        "max_memory_usage_mb": 512,
        "max_cpu_usage_percent": 80,
        "min_success_rate": Decimal("0.5"),
    },
}
