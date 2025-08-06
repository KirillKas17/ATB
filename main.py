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

# Проверка критических зависимостей
def check_critical_dependencies() -> bool:
    """Проверка наличия критических зависимостей."""
    critical_modules = [
        'shared.models.config',
        'application.di_container_refactored',
        'domain.strategies',
        'infrastructure.core.evolution_integration'
    ]
    
    missing_modules = []
    for module in critical_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Критические модули отсутствуют: {missing_modules}")
        logger.error("Установите зависимости: pip install -r requirements_enhanced.txt")
        return False
    
    return True

# Импорт типизированных конфигураций
from shared.models.config import ApplicationConfig, TradingConfig, RiskConfig, create_default_config, validate_config

# Импорт DI контейнера
from application.di_container_refactored import get_service_locator

# Импорты для торговой оркестрации
from application.use_cases.manage_orders import DefaultOrderManagementUseCase
from application.use_cases.manage_positions import DefaultPositionManagementUseCase
from application.use_cases.manage_risk import DefaultRiskManagementUseCase
from application.use_cases.manage_trading_pairs import DefaultTradingPairManagementUseCase
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase

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

# Импорт основных компонентов
from shared.performance_monitor import performance_monitor
from shared.metrics_analyzer import MetricsAnalyzer
from shared.config_validator import config_validator
from shared.monitoring_dashboard import MonitoringDashboard
from shared.exception_handler import SafeExceptionHandler as ExceptionHandler
from scripts.deployment import DeploymentOrchestrator

# Импорт основных модулей ATB
from application.di_container_refactored import Container, get_service_locator
from application.orchestration.orchestrator_factory import create_trading_orchestrator
from domain.strategies import get_strategy_registry
from infrastructure.agents.agent_context_refactored import AgentContext
from domain.intelligence.entanglement_detector import EntanglementDetector
from domain.intelligence.mirror_detector import MirrorDetector
from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
from application.orchestration.strategy_integration import strategy_integration

async def main() -> None:
    """Main entry point for the trading system."""
    
    # Проверка критических зависимостей
    if not check_critical_dependencies():
        logger.error("Критические зависимости отсутствуют. Завершение работы.")
        sys.exit(1)
    
    print("🚀 Торговая система ATB запущена успешно!")
    print("📊 Инициализация компонентов:")
    
    try:
        # Создание конфигурации
        config = create_default_config()
        service_locator = get_service_locator()
        
        # Инициализация основных агентов
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
        logger.error(f"Ошибка инициализации: {e}")
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
