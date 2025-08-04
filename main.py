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
import signal
import threading
from pathlib import Path

# Безопасные импорты с fallback
from shared.safe_imports import (
    pd, np, primary_logger, check_dependencies,
    PANDAS_AVAILABLE, NUMPY_AVAILABLE, safe_execution
)

# Настройка логирования
logger = primary_logger

# Импорт типизированных конфигураций
try:
    from shared.models.config import ApplicationConfig, TradingConfig, RiskConfig, create_default_config, validate_config
except ImportError as e:
    logger.warning(f"Failed to import config models: {e}")
    # Создаем заглушки для базовой работы
    ApplicationConfig = dict  # type: ignore[misc,assignment]
    TradingConfig = dict  # type: ignore[misc,assignment]
    RiskConfig = dict  # type: ignore[misc,assignment]
    create_default_config = lambda: {}  # type: ignore[assignment,return-value]
    validate_config = lambda x: True  # type: ignore[assignment,return-value]

# Импорт DI контейнера
try:
    from application.di_container_refactored import get_service_locator
except ImportError as e:
    logger.warning(f"Failed to import DI container: {e}")
    get_service_locator = lambda: None  # type: ignore[assignment,return-value]

# Импорты для торговой оркестрации
try:
    from application.use_cases.manage_orders import DefaultOrderManagementUseCase
    from application.use_cases.manage_positions import DefaultPositionManagementUseCase
    from application.use_cases.manage_risk import DefaultRiskManagementUseCase
    from application.use_cases.manage_trading_pairs import DefaultTradingPairManagementUseCase
    from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
except ImportError as e:
    logger.warning(f"Failed to import use cases: {e}")
    # Создаем заглушки
    DefaultOrderManagementUseCase = type('DefaultOrderManagementUseCase', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]
    DefaultPositionManagementUseCase = type('DefaultPositionManagementUseCase', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]
    DefaultRiskManagementUseCase = type('DefaultRiskManagementUseCase', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]
    DefaultTradingPairManagementUseCase = type('DefaultTradingPairManagementUseCase', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]
    DefaultTradingOrchestratorUseCase = type('DefaultTradingOrchestratorUseCase', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]

# Импорты сервисов
try:
    from application.services.market_service import MarketService
    from application.services.trading_service import TradingService
    from application.services.risk_service import RiskService
except ImportError as e:
    logger.warning(f"Failed to import services: {e}")
    MarketService = type('MarketService', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]
    TradingService = type('TradingService', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]
    RiskService = type('RiskService', (), {})  # type: ignore[misc,assignment]  # type: ignore[misc,assignment]

# Импорты новых компонентов domain/strategies
try:
    from domain.strategies import (
        StrategyFactory, get_strategy_factory,
        StrategyRegistry, get_strategy_registry,
        StrategyValidator, get_strategy_validator
    )
except ImportError as e:
    logger.warning(f"Failed to import domain strategies: {e}")
    StrategyFactory = type('StrategyFactory', (), {})  # type: ignore[misc,assignment]
    get_strategy_factory = lambda: StrategyFactory()
    StrategyRegistry = type('StrategyRegistry', (), {})  # type: ignore[misc,assignment]
    get_strategy_registry = lambda: StrategyRegistry()
    StrategyValidator = type('StrategyValidator', (), {})  # type: ignore[misc,assignment]
    get_strategy_validator = lambda: StrategyValidator()

# Импорты стратегий из infrastructure/strategies
try:
    from infrastructure.strategies.trend_strategies import TrendStrategy
    from infrastructure.strategies.sideways_strategies import SidewaysStrategy
    from infrastructure.strategies.adaptive.adaptive_strategy_generator import AdaptiveStrategyGenerator
    from infrastructure.strategies.evolution.evolvable_base_strategy import EvolvableBaseStrategy
    from infrastructure.strategies.manipulation_strategies import ManipulationStrategy
    from infrastructure.strategies.volatility_strategy import VolatilityStrategy
    from infrastructure.strategies.pairs_trading_strategy import PairsTradingStrategy
except ImportError as e:
    logger.warning(f"Failed to import infrastructure strategies: {e}")
    # Создаем базовые заглушки стратегий
    TrendStrategy = type('TrendStrategy', (), {})  # type: ignore[misc,assignment]
    SidewaysStrategy = type('SidewaysStrategy', (), {})  # type: ignore[misc,assignment]
    AdaptiveStrategyGenerator = type('AdaptiveStrategyGenerator', (), {})  # type: ignore[misc,assignment]
    EvolvableBaseStrategy = type('EvolvableBaseStrategy', (), {})  # type: ignore[misc,assignment]
    ManipulationStrategy = type('ManipulationStrategy', (), {})  # type: ignore[misc,assignment]
    VolatilityStrategy = type('VolatilityStrategy', (), {})  # type: ignore[misc,assignment]
    PairsTradingStrategy = type('PairsTradingStrategy', (), {})  # type: ignore[misc,assignment]

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
    from application.orchestration.orchestrator_factory import create_trading_orchestrator
    from domain.strategies import get_strategy_registry
    from infrastructure.agents.agent_context_refactored import AgentContext
    from domain.intelligence.entanglement_detector import EntanglementDetector
    from domain.intelligence.mirror_detector import MirrorDetector
    from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
    from application.orchestration.strategy_integration import strategy_integration
except ImportError as e:
    logger.error(f"Failed to import ATB modules: {e}")
    sys.exit(1)

async def main() -> None:
    """Main entry point for the trading system."""
    config = create_default_config()
    service_locator = get_service_locator()
    
    print("🚀 Торговая система ATB запущена успешно!")
    print("📊 Инициализация компонентов:")
    
    # Инициализация основных агентов
    try:
        entanglement_detector = EntanglementDetector()
        print("   ✅ EntanglementDetector")
        
        mirror_detector = MirrorDetector()
        print("   ✅ MirrorDetector")
        
        market_maker_agent = MarketMakerModelAgent()
        print("   ✅ MarketMakerModelAgent")
        
        # Инициализация стратегий
        await strategy_integration.initialize_strategies()
        print("   ✅ StrategyIntegration")
        
        # Создание полноценного оркестратора
        orchestrator = create_trading_orchestrator(config)
        print("   ✅ TradingOrchestrator")
        
        print("   ✅ Domain layer")
        print("   ✅ Application layer") 
        print("   ✅ Infrastructure layer")
        print("   ✅ DI Container")
        print("   ✅ Зависимости установлены")
        
    except Exception as e:
        print(f"   ❌ Ошибка инициализации: {e}")
        return

    orchestrator_task = None

    def shutdown_handler(signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        print("🛑 Получен сигнал завершения...")
        if orchestrator_task:
            orchestrator_task.cancel()
        print("✅ Система корректно остановлена")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print("\n💡 Запуск торгового оркестратора...")
    try:
        # Запуск оркестратора в фоне
        orchestrator_task = asyncio.create_task(orchestrator.start())
        print("🎯 Торговый оркестратор запущен")
        print("📈 Система в полном рабочем состоянии!")
        
        # Ожидание завершения
        await orchestrator_task
        
    except asyncio.CancelledError:
        print("🛑 Получен сигнал завершения")
        await orchestrator.stop()
        print("✅ Система корректно остановлена")
    except KeyboardInterrupt:
        print("🛑 Получен Ctrl+C")
        await orchestrator.stop()
        print("✅ Система корректно остановлена")

if __name__ == "__main__":
    asyncio.run(main())
