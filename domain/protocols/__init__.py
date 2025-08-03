"""
Протоколы домена - промышленный уровень.
Этот пакет содержит протоколы для взаимодействия с внешними системами:
- Биржевые протоколы (ExchangeProtocol)
- ML протоколы (MLProtocol)
- Протоколы стратегий (StrategyProtocol)
- Протоколы репозиториев (RepositoryProtocol)
- Утилиты и декораторы
- Системы мониторинга и безопасности
"""

from .strategy_protocol import StrategyProtocol
from .market_analysis_protocol import MarketRegime, StrategyState
from .signal_generation_protocol import SignalGenerationProtocol
from .strategy_execution_protocol import StrategyExecutionProtocol
from .risk_management_protocol import RiskManagementProtocol
from .performance_analytics_protocol import PerformanceAnalyticsProtocol
from .strategy_optimization_protocol import StrategyOptimizationProtocol
from .lifecycle_management_protocol import LifecycleManagementProtocol
from .error_handling_protocol import ErrorHandlingProtocol
from .strategy_utilities_protocol import StrategyUtilitiesProtocol

from .decorators import (
    cache,
    circuit_breaker,
    log_operation,
    metrics,
    rate_limit,
    retry,
    timeout,
    validate_input,
)
from .monitoring import (
    AlertManager,
    HealthChecker,
    MetricsCollector,
    PerformanceMonitor,
    ProtocolMonitor,
    alert_on_error,
    monitor_protocol,
)
from .performance import (
    BenchmarkRunner,
    PerformanceOptimizer,
    PerformanceProfiler,
    benchmark_performance,
    get_performance_report,
    optimize_performance,
    optimize_slow_functions,
    profile_performance,
)
from .security import (
    AuditManager,
    AuthenticationManager,
    AuthorizationManager,
    CryptoManager,
    SecurityManager,
    audit_security_events,
    encrypt_sensitive_fields,
    require_authentication,
)

__all__ = [
    # Основные протоколы
    "ExchangeProtocol",
    "MLProtocol",
    "StrategyProtocol",
    "RepositoryProtocol",
    # Классы стратегий
    "StrategyState",
    "MarketRegime",
    # Протоколы анализа
    "MarketAnalysisProtocol",
    "SignalGenerationProtocol",
    "StrategyExecutionProtocol",
    "RiskManagementProtocol",
    "PerformanceAnalyticsProtocol",
    "StrategyOptimizationProtocol",
    "LifecycleManagementProtocol",
    "ErrorHandlingProtocol",
    "StrategyUtilitiesProtocol",
    # Декораторы
    "retry",
    "timeout",
    "validate_input",
    "cache",
    "metrics",
    "circuit_breaker",
    "rate_limit",
    "log_operation",
    # Мониторинг
    "ProtocolMonitor",
    "MetricsCollector",
    "AlertManager",
    "HealthChecker",
    "PerformanceMonitor",
    "monitor_protocol",
    "alert_on_error",
    # Производительность
    "PerformanceProfiler",
    "BenchmarkRunner",
    "PerformanceOptimizer",
    "profile_performance",
    "benchmark_performance",
    "optimize_performance",
    "get_performance_report",
    "optimize_slow_functions",
    # Безопасность
    "SecurityManager",
    "CryptoManager",
    "AuthenticationManager",
    "AuthorizationManager",
    "AuditManager",
    "require_authentication",
    "encrypt_sensitive_fields",
    "audit_security_events",
]
