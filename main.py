#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syntra - Main Entry Point
Domain-Driven Design Architecture with News Sentiment Integration
"""

import asyncio
import logging
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any, List, Union
import pandas as pd
import signal
import threading
from pathlib import Path

# Импорт типизированных конфигураций
from shared.models.config import ApplicationConfig, TradingConfig, RiskConfig, create_default_config, validate_config

# Импорт DI контейнера
from application.di_container_refactored import get_service_locator

# Импорты для торговой оркестрации
from application.use_cases.manage_orders import DefaultOrderManagementUseCase
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
from application.use_cases.manage_risk import DefaultRiskManagementUseCase
from application.use_cases.manage_trading_pairs import DefaultTradingPairManagementUseCase
from application.use_cases.manage_trading_orchestrator import DefaultTradingOrchestratorUseCase

# Импорты сервисов
from application.services.market_service import MarketService
from application.services.trading_service import TradingService
from application.services.risk_service import RiskService

# Импорты новых компонентов domain/strategies
from domain.strategies import (
    StrategyFactory, get_strategy_factory,
    StrategyRegistry, get_strategy_registry,
    StrategyValidator, get_strategy_validator
)

# Импорты стратегий из infrastructure/strategies
from infrastructure.strategies.trend_strategies import TrendStrategy
from infrastructure.strategies.sideways_strategies import SidewaysStrategy
from infrastructure.strategies.adaptive.adaptive_strategy_generator import AdaptiveStrategyGenerator
from infrastructure.strategies.evolution.evolvable_base_strategy import EvolvableBaseStrategy
from infrastructure.strategies.manipulation_strategies import ManipulationStrategy
from infrastructure.strategies.volatility_strategy import VolatilityStrategy
from infrastructure.strategies.pairs_trading_strategy import PairsTradingStrategy

# Импорты для интеграции с эволюционными агентами
from infrastructure.core.evolution_integration import EvolutionIntegration
from infrastructure.core.integration_manager import IntegrationManager

# Импорты модуля симуляции
from infrastructure.simulation.simulator import MarketSimulator
from unittest.mock import Mock
from infrastructure.simulation.types import (
    SimulationConfig,
    MarketSimulationConfig,
    BacktestConfig,
    SimulationMoney,
    Symbol
)

# Добавляю импорты всех компонентов infrastructure/core
from infrastructure.core.health_checker import HealthChecker
from infrastructure.core.metrics import MetricsCollector
from infrastructure.core.system_monitor import SystemMonitor
from infrastructure.core.autonomous_controller import AutonomousController
from infrastructure.core.circuit_breaker import CircuitBreaker
from infrastructure.core.risk_manager import RiskManager
from infrastructure.core.portfolio_manager import PortfolioManager
from infrastructure.core.position_manager import PositionManager
from infrastructure.core.signal_processor import SignalProcessor
from infrastructure.core.efficiency_validator import EfficiencyValidator
from infrastructure.core.correlation_chain import CorrelationChain
from infrastructure.core.feature_engineering import FeatureEngineer
from infrastructure.core.order_utils import OrderUtils
from infrastructure.core.evolvable_components import EvolvableComponentFactory
from infrastructure.core.optimizer import StrategyOptimizer
from infrastructure.core.database import Database
from infrastructure.core.optimized_database import OptimizedDatabase
from infrastructure.core.config_manager import ConfigManager
from infrastructure.core.market_state import MarketState
from infrastructure.core.data_pipeline import DataPipeline
from infrastructure.core.exchange import Exchange
from infrastructure.core.ml_integration import MLIntegration
from infrastructure.core.evolution_manager import EvolutionManager
from infrastructure.core.auto_migration_manager import AutoMigrationManager

# Импорты новых модулей мониторинга
from infrastructure.monitoring import (
    PerformanceMonitor,
    AlertManager,
    PerformanceTracer,
    MonitoringDashboard as InfraMonitoringDashboard,
    get_monitor,
    get_alert_manager,
    get_tracer,
    get_dashboard,
    start_monitoring,
    stop_monitoring,
    record_metric,
    create_alert
)

# Настройка логирования
logger = logging.getLogger(__name__)

# Импорт основных компонентов
try:
    from shared.performance_monitor import performance_monitor
    from shared.metrics_analyzer import MetricsAnalyzer
    from shared.config_validator import config_validator
    from shared.monitoring_dashboard import MonitoringDashboard
    from shared.exception_handler import SafeExceptionHandler as ExceptionHandler
    from scripts.deployment import DeploymentOrchestrator
except ImportError as e:
    logger.error(f"Failed to import components: {e}")
    sys.exit(1)

# Импорт основных модулей ATB
try:
    from application.di_container_refactored import Container, get_service_locator
    from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
    from domain.strategies import get_strategy_registry
    from infrastructure.agents.agent_context_refactored import AgentContext
except ImportError as e:
    logger.error(f"Failed to import ATB modules: {e}")
    sys.exit(1)

from application.orchestration.trading_orchestrator import TradingOrchestrator

async def main() -> None:
    """Main entry point for the trading system."""
    config = create_default_config()
    service_locator = get_service_locator()
    orchestrator = service_locator.get_use_case(DefaultTradingOrchestratorUseCase)

    def shutdown_handler(signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        asyncio.create_task(orchestrator.stop())
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    await orchestrator.start()

if __name__ == "__main__":
    asyncio.run(main())
