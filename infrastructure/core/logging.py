"""
Логирование для инфраструктурного слоя.
Использует объединенную систему логирования из shared/logging.py
"""

from shared.logging import log_portfolio_update_infrastructure as log_portfolio_update
from shared.logging import log_strategy_signal_infrastructure as log_strategy_signal
from shared.logging import log_system_health_infrastructure as log_system_health

# Экспорт функций для совместимости
__all__ = [
    "setup_logger",
    "get_logger",
    "log_trade",
    "log_error",
    "log_performance",
    "log_market_data",
    "log_strategy_signal",
    "log_portfolio_update",
    "log_risk_alert",
    "log_system_health",
]
