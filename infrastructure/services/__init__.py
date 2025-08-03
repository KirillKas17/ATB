"""
Сервисы инфраструктуры - промышленные реализации бизнес-логики.

Этот модуль предоставляет сервисы для обработки торговых операций,
анализа рисков, технического анализа и других бизнес-процессов.
"""

from typing import Any, Dict, Type

from .enhanced_trading_service import EnhancedTradingService
from .enhanced_trading_service_refactored import (
    AdvancedOrderCreator,
    BaseOrderCreator,
    BaseSentimentAnalyzer,
    BaseStrategyExecutor,
    EnhancedTradingService,
    ExecutionParameters,
    OrderCreator,
    OrderParameters,
    PerformanceMetrics,
    SentimentAnalysis,
    SentimentAnalyzer,
    StrategyExecutor,
)
from .risk_analysis_service import RiskAnalysisService
from .technical_analysis_service import TechnicalAnalysisService

# Реестр сервисов для DI
SERVICE_REGISTRY: Dict[str, Type[Any]] = {
    "enhanced_trading": EnhancedTradingService,
    "risk_analysis": RiskAnalysisService,
    "technical_analysis": TechnicalAnalysisService,
}

__all__ = [
    # Основные сервисы
    "EnhancedTradingService",
    "RiskAnalysisService",
    "TechnicalAnalysisService",
    # Протоколы и компоненты refactored
    "OrderCreator",
    "StrategyExecutor",
    "SentimentAnalyzer",
    "ExecutionParameters",
    "OrderParameters",
    "SentimentAnalysis",
    "PerformanceMetrics",
    "AdvancedOrderCreator",
    "BaseOrderCreator",
    "BaseStrategyExecutor",
    "BaseSentimentAnalyzer",
    # Реестр
    "SERVICE_REGISTRY",
]
