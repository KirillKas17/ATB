"""
Системная интеграция всех компонентов infrastructure/core.
Обеспечивает подключение всех модулей к основному циклу системы,
инициализацию, мониторинг и координацию взаимодействий.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from infrastructure.messaging.event_bus import (
    Event,
    EventPriority,
    EventType,
    EventName,
)
from infrastructure.messaging.event_bus import EventBus

# Импорты новых модулей мониторинга
from infrastructure.monitoring import (
    AlertManager,
    MonitoringDashboard,
    PerformanceMonitor,
    PerformanceTracer,
    get_alert_manager,
    get_dashboard,
    get_monitor,
    get_tracer,
    start_monitoring,
    stop_monitoring,
)

from .auto_migration_manager import AutoMigrationManager
from .autonomous_controller import AutonomousController
from .circuit_breaker import CircuitBreaker
from .config_manager import ConfigManager
from .correlation_chain import CorrelationChain
from .data_pipeline import DataPipeline
from .database import Database
from .efficiency_validator import EfficiencyValidator
from .evolution_integration import EvolutionIntegration
from .evolution_manager import EvolutionManager
from .evolvable_components import EvolvableComponentFactory
from .exchange import Exchange
from .feature_engineering import FeatureEngineer
from .fibonacci_tools import (
    FibonacciLevels,
)

# Импорты всех компонентов infrastructure/core
from .health_checker import HealthChecker
from .logger import Logger
from .market_regime import (
    MarketRegimeDetector,
)
from .market_state import MarketState
from .math_utils import (
    calculate_fibonacci_levels,
)
from .metrics import MetricsCollector
from .ml_integration import MLIntegration
from .optimized_database import OptimizedDatabase
from .optimizer import StrategyOptimizer
from .order_utils import OrderUtils
from .portfolio_manager import PortfolioManager
from .position_manager import PositionManager
from .risk_manager import RiskManager
from .signal_processor import SignalProcessor
# Удаляем импорты технического анализа, так как их нет в .technical
# from .technical import (
#     MomentumAnalysis,
#     SupportResistance,
#     TechnicalAnalysis,
#     TechnicalIndicatorsService,
#     TechnicalPatterns,
#     TrendAnalysis,
# )


@dataclass
class ComponentStatus:
    """Статус компонента."""

    name: str
    is_initialized: bool
    is_running: bool
    last_heartbeat: datetime
    error_count: int = 0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemIntegrationConfig:
    """Конфигурация системной интеграции."""

    # Компоненты для инициализации
    enable_health_checker: bool = True
    enable_metrics: bool = True
    enable_system_monitor: bool = True
    enable_autonomous_controller: bool = True
    enable_circuit_breaker: bool = True
    enable_risk_manager: bool = True
    enable_portfolio_manager: bool = True
    enable_position_manager: bool = True
    enable_signal_processor: bool = True
    enable_efficiency_validator: bool = True
    enable_correlation_chain: bool = True
    enable_feature_engineering: bool = True
    enable_order_utils: bool = True
    enable_evolvable_components: bool = True
    enable_optimizer: bool = True
    enable_database: bool = True
    enable_optimized_database: bool = True
    enable_config_manager: bool = True
    enable_technical_analysis: bool = True
    enable_visualization: bool = True
    enable_data_pipeline: bool = True
    enable_exchange_manager: bool = True
    enable_ml_integration: bool = True
    enable_evolution_manager: bool = True
    enable_evolution_integration: bool = True
    enable_auto_migration: bool = True
    enable_market_state: bool = True
    # Настройки мониторинга
    heartbeat_interval: int = 30  # секунды
    health_check_interval: int = 60  # секунды
    performance_monitoring: bool = True
    error_threshold: int = 5
    # Настройки восстановления
    auto_restart_failed_components: bool = True
    max_restart_attempts: int = 3
    restart_delay: int = 30  # секунды


class SystemIntegration:
    """
    Системная интеграция всех компонентов infrastructure/core.
    Обеспечивает:
    - Инициализацию всех компонентов
    - Мониторинг состояния
    - Координацию взаимодействий
    - Автоматическое восстановление
    - Интеграцию с основным циклом системы
    """

    def __init__(self, config: Optional[SystemIntegrationConfig] = None) -> None:
        """Инициализация системной интеграции."""
        self.config = config or SystemIntegrationConfig()
        self.event_bus = EventBus()
        # Компоненты системы
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        # Состояние системы
        self.is_initialized = False
        self.is_running = False
        self.start_time: Optional[datetime] = None
        # Мониторинг
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.performance_task: Optional[asyncio.Task] = None
        # Статистика
        self.stats = {
            "total_components": 0,
            "active_components": 0,
            "failed_components": 0,
            "restart_count": 0,
            "total_errors": 0,
        }
        logger.info("System Integration initialized")

    async def initialize(self) -> None:
        """Инициализация всех компонентов."""
        try:
            logger.info("Starting system integration initialization")
            # 1. Инициализация базовых компонентов
            await self._initialize_basic_components()
            # 2. Инициализация торговых компонентов
            await self._initialize_trading_components()
            # 3. Инициализация ML компонентов
            await self._initialize_ml_components()
            # 4. Инициализация эволюционных компонентов
            await self._initialize_evolution_components()
            # 5. Инициализация аналитических компонентов
            await self._initialize_analytical_components()
            # 6. Настройка взаимодействий
            await self._setup_component_interactions()
            # 7. Запуск мониторинга
            await self._start_monitoring()
            self.is_initialized = True
            self.start_time = datetime.now()
            logger.info(
                f"System integration initialized with {len(self.components)} components"
            )
        except Exception as e:
            logger.error(f"System integration initialization failed: {e}")
            await self._handle_initialization_error(e)
            raise

    async def _initialize_basic_components(self) -> None:
        """Инициализация базовых компонентов."""
        try:
            # Инициализация EventBus - убираем вызов initialize(), так как его нет
            await self._register_component("event_bus")

            # Инициализация CircuitBreaker - используем правильные аргументы
            if self.config.enable_circuit_breaker:
                from infrastructure.circuit_breaker.breaker import CircuitBreaker, CircuitBreakerConfig
                circuit_config = CircuitBreakerConfig()
                self.components["circuit_breaker"] = CircuitBreaker("system_circuit_breaker", circuit_config)
                await self._register_component("circuit_breaker")

            # Инициализация Database
            if self.config.enable_database:
                self.components["database"] = Database("sqlite:///test.db")
                await self._register_component("database")

            # Инициализация OptimizedDatabase
            if self.config.enable_optimized_database:
                self.components["optimized_database"] = OptimizedDatabase("sqlite:///test_optimized.db")
                await self._register_component("optimized_database")

            # Инициализация Exchange
            if self.config.enable_exchange_manager:
                self.components["exchange"] = Exchange({"api_key": "test", "api_secret": "test"})
                await self._register_component("exchange")

            # Инициализация PositionManager
            if self.config.enable_position_manager:
                self.components["position_manager"] = PositionManager({"max_positions": 10})
                await self._register_component("position_manager")

            logger.info("Basic components initialized")
        except Exception as e:
            logger.error(f"Basic components initialization failed: {e}")
            raise

    async def _initialize_trading_components(self) -> None:
        """Инициализация торговых компонентов."""
        try:
            # Инициализация RiskManager - добавляем недостающие параметры
            if self.config.enable_risk_manager:
                self.components["risk_manager"] = RiskManager(self.event_bus, {})
                await self._register_component("risk_manager")

            # Инициализация PortfolioManager - добавляем недостающие параметры
            if self.config.enable_portfolio_manager:
                self.components["portfolio_manager"] = PortfolioManager(self.event_bus, {})
                await self._register_component("portfolio_manager")

            # Инициализация SignalProcessor
            if self.config.enable_signal_processor:
                self.components["signal_processor"] = SignalProcessor()
                await self._register_component("signal_processor")

            # Инициализация OrderUtils
            if self.config.enable_order_utils:
                self.components["order_utils"] = OrderUtils()
                await self._register_component("order_utils")

            logger.info("Trading components initialized")
        except Exception as e:
            logger.error(f"Trading components initialization failed: {e}")
            raise

    async def _initialize_ml_components(self) -> None:
        """Инициализация ML компонентов."""
        try:
            # Инициализация MLIntegration
            if self.config.enable_ml_integration:
                self.components["ml_integration"] = MLIntegration()
                await self._register_component("ml_integration")

            logger.info("ML components initialized")
        except Exception as e:
            logger.error(f"ML components initialization failed: {e}")
            raise

    async def _initialize_evolution_components(self) -> None:
        """Инициализация эволюционных компонентов."""
        try:
            # Инициализация EvolutionManager
            if self.config.enable_evolution_manager:
                self.components["evolution_manager"] = EvolutionManager()
                await self._register_component("evolution_manager")

            # Инициализация EvolutionIntegration
            if self.config.enable_evolution_integration:
                self.components["evolution_integration"] = EvolutionIntegration()
                await self._register_component("evolution_integration")

            # Инициализация EvolvableComponentFactory
            if self.config.enable_evolvable_components:
                self.components["evolvable_factory"] = EvolvableComponentFactory()
                await self._register_component("evolvable_factory")

            logger.info("Evolution components initialized")
        except Exception as e:
            logger.error(f"Evolution components initialization failed: {e}")
            raise

    async def _initialize_analytical_components(self) -> None:
        """Инициализация аналитических компонентов."""
        try:
            # Инициализация MarketState
            if self.config.enable_market_state:
                self.components["market_state"] = MarketState(
                    timestamp=datetime.now(),
                    price=50000.0,
                    volume=1000.0,
                    volatility=0.02,
                    trend="bullish",
                    indicators={},
                    market_regime="trending",
                    liquidity=0.8,
                    momentum=0.6,
                    sentiment=0.7,
                    support_levels=[45000.0, 48000.0],
                    resistance_levels=[52000.0, 55000.0],
                    market_depth={},
                    correlation_matrix={},
                    market_impact=0.01,
                    volume_profile={}
                )
                await self._register_component("market_state")

            # Инициализация MarketRegimeDetector
            if self.config.enable_technical_analysis:
                self.components["market_regime_detector"] = MarketRegimeDetector()
                await self._register_component("market_regime_detector")

            logger.info("Analytical components initialized")
        except Exception as e:
            logger.error(f"Analytical components initialization failed: {e}")
            raise

    async def _register_component(self, name: str) -> None:
        """Регистрация компонента в системе."""
        try:
            component = self.components.get(name)
            if component:
                # Регистрируем статус компонента
                self.component_status[name] = ComponentStatus(
                    name=name,
                    is_initialized=True,
                    is_running=False,
                    last_heartbeat=datetime.now(),
                )
                self.stats["total_components"] += 1
                logger.debug(f"Component {name} registered")
        except Exception as e:
            logger.error(f"Failed to register component {name}: {e}")

    async def _setup_component_interactions(self) -> None:
        """Настройка взаимодействий между компонентами."""
        try:
            # Настройка обработчиков событий
            await self._setup_event_handlers()
            # Настройка зависимостей компонентов
            await self._setup_component_dependencies()
            logger.info("Component interactions configured")
        except Exception as e:
            logger.error(f"Component interactions setup failed: {e}")
            raise

    async def _setup_event_handlers(self) -> None:
        """Настройка обработчиков событий."""
        try:
            # Подписываемся на события системы - убираем await и используем EventName
            self.event_bus.subscribe(EventName("health_issues"), self._on_health_issues)
            self.event_bus.subscribe(EventName("risk_limit_exceeded"), self._on_risk_limit_exceeded)
            self.event_bus.subscribe(EventName("trade_executed"), self._on_trade_executed)
            self.event_bus.subscribe(EventName("agent_evolved"), self._on_agent_evolved)
            logger.info("Event handlers configured")
        except Exception as e:
            logger.error(f"Event handlers setup failed: {e}")
            raise

    async def _setup_component_dependencies(self) -> None:
        """Настройка зависимостей между компонентами."""
        try:
            # Настройка зависимостей RiskManager
            if "risk_manager" in self.components and "portfolio_manager" in self.components:
                risk_manager = self.components["risk_manager"]
                portfolio_manager = self.components["portfolio_manager"]
                # Устанавливаем зависимости
                if hasattr(risk_manager, "set_portfolio_manager"):
                    risk_manager.set_portfolio_manager(portfolio_manager)

            # Настройка зависимостей PortfolioManager
            if "portfolio_manager" in self.components and "position_manager" in self.components:
                portfolio_manager = self.components["portfolio_manager"]
                position_manager = self.components["position_manager"]
                # Устанавливаем зависимости
                if hasattr(portfolio_manager, "set_position_manager"):
                    portfolio_manager.set_position_manager(position_manager)

            # Настройка зависимостей SignalProcessor
            if "signal_processor" in self.components and "risk_manager" in self.components:
                signal_processor = self.components["signal_processor"]
                risk_manager = self.components["risk_manager"]
                # Устанавливаем зависимости
                if hasattr(signal_processor, "set_risk_manager"):
                    signal_processor.set_risk_manager(risk_manager)

            logger.info("Component dependencies configured")
        except Exception as e:
            logger.error(f"Component dependencies setup failed: {e}")
            raise

    async def start(self) -> None:
        """Запуск системной интеграции."""
        if not self.is_initialized:
            await self.initialize()
        try:
            logger.info("Starting system integration")
            self.is_running = True
            # Запуск всех компонентов
            await self._start_all_components()
            # Запуск мониторинга
            await self._start_monitoring()
            logger.info("System integration started successfully")
        except Exception as e:
            logger.error(f"Failed to start system integration: {e}")
            await self.stop()
            raise

    async def _start_all_components(self) -> None:
        """Запуск всех компонентов."""
        logger.info("Starting all components")
        for name, component in self.components.items():
            try:
                # Запуск компонента если есть метод start
                if hasattr(component, "start") and callable(
                    getattr(component, "start")
                ):
                    await component.start()
                # Обновление статуса
                if name in self.component_status:
                    self.component_status[name].is_running = True
                    self.component_status[name].last_heartbeat = datetime.now()
                self.stats["active_components"] += 1
                logger.debug(f"Component {name} started")
            except Exception as e:
                logger.error(f"Failed to start component {name}: {e}")
                self.component_status[name].error_count += 1
                self.stats["total_errors"] += 1

    async def _start_monitoring(self) -> None:
        """Запуск мониторинга."""
        logger.info("Starting monitoring")
        # Запуск новых модулей мониторинга
        if self.config.enable_system_monitor:
            start_monitoring()  # Запуск PerformanceMonitor
            logger.info("New monitoring modules started")
        # Запуск heartbeat
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        # Запуск проверки здоровья
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        # Запуск мониторинга производительности
        if self.config.performance_monitoring:
            self.performance_task = asyncio.create_task(
                self._performance_monitoring_loop()
            )

    async def _heartbeat_loop(self) -> None:
        """Цикл heartbeat."""
        while self.is_running:
            try:
                for name, status in self.component_status.items():
                    # Обновление heartbeat
                    status.last_heartbeat = datetime.now()
                    # Проверка состояния компонента
                    if name in self.components:
                        component = self.components[name]
                        # Проверка метода get_status если есть
                        if hasattr(component, "get_status") and callable(
                            getattr(component, "get_status")
                        ):
                            try:
                                component_status = await component.get_status()
                                status.performance_metrics = component_status
                            except Exception as e:
                                logger.warning(f"Failed to get status for {name}: {e}")
                                status.error_count += 1
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(60)

    async def _health_check_loop(self) -> None:
        """Цикл проверки здоровья."""
        while self.is_running:
            try:
                # Проверка состояния всех компонентов
                await self._check_all_components_health()
                # Автоматическое восстановление
                if self.config.auto_restart_failed_components:
                    await self._restart_failed_components()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)

    async def _performance_monitoring_loop(self) -> None:
        """Цикл мониторинга производительности."""
        while self.is_running:
            try:
                # Сбор метрик производительности
                await self._collect_performance_metrics()
                # Анализ производительности
                await self._analyze_performance()
                await asyncio.sleep(60)  # Каждую минуту
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(60)

    async def _check_all_components_health(self) -> None:
        """Проверка здоровья всех компонентов."""
        for name, status in self.component_status.items():
            try:
                if name in self.components:
                    component = self.components[name]
                    # Проверка метода health_check если есть
                    if hasattr(component, "health_check") and callable(
                        getattr(component, "health_check")
                    ):
                        health_status = await component.health_check()
                        if not health_status.get("is_healthy", True):
                            status.error_count += 1
                            logger.warning(f"Component {name} health check failed")
                    # Проверка по количеству ошибок
                    if status.error_count > self.config.error_threshold:
                        logger.error(f"Component {name} exceeded error threshold")
                        status.is_running = False
            except Exception as e:
                logger.error(f"Error checking health for {name}: {e}")
                status.error_count += 1

    async def _restart_failed_components(self) -> None:
        """Перезапуск неудачных компонентов."""
        for name, status in self.component_status.items():
            if not status.is_running and status.error_count > 0:
                try:
                    logger.info(f"Attempting to restart component {name}")
                    # Остановка компонента
                    if name in self.components:
                        component = self.components[name]
                        if hasattr(component, "stop") and callable(
                            getattr(component, "stop")
                        ):
                            await component.stop()
                    # Перезапуск компонента
                    await self._restart_component(name)
                    self.stats["restart_count"] += 1
                except Exception as e:
                    logger.error(f"Failed to restart component {name}: {e}")

    async def _restart_component(self, name: str) -> None:
        """Перезапуск конкретного компонента."""
        try:
            # Создание нового экземпляра компонента
            component_class = type(self.components[name])
            new_component = component_class()
            # Инициализация
            if hasattr(new_component, "initialize") and callable(
                getattr(new_component, "initialize")
            ):
                await new_component.initialize()
            # Запуск
            if hasattr(new_component, "start") and callable(
                getattr(new_component, "start")
            ):
                await new_component.start()
            # Обновление
            self.components[name] = new_component
            self.component_status[name].is_running = True
            self.component_status[name].error_count = 0
            self.component_status[name].last_heartbeat = datetime.now()
            logger.info(f"Component {name} restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart component {name}: {e}")
            self.component_status[name].error_count += 1

    async def _collect_performance_metrics(self) -> None:
        """Сбор метрик производительности."""
        try:
            # Сбор системных метрик
            system_metrics = await self._get_system_metrics()
            # Отправка метрик
            if "metrics" in self.components:
                await self.components["metrics"].update_all_metrics()
            # Кэширование метрик
            if "cache" in self.components:
                await self.components["cache"].set(
                    "performance_metrics", system_metrics, ttl=300
                )
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    async def _analyze_performance(self) -> None:
        """Анализ производительности."""
        try:
            # Анализ производительности компонентов
            for name, status in self.component_status.items():
                if name in self.components:
                    component = self.components[name]
                    # Анализ метрик производительности
                    if hasattr(component, "get_performance") and callable(
                        getattr(component, "get_performance")
                    ):
                        performance = component.get_performance()
                        status.performance_metrics["performance"] = performance
                        # Логирование низкой производительности
                        if performance < 0.5:
                            logger.warning(
                                f"Low performance detected for {name}: {performance}"
                            )
        except Exception as e:
            logger.error(f"Error analyzing performance: {e}")

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Получение системных метрик."""
        return {
            "timestamp": datetime.now().isoformat(),
            "uptime": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            ),
            "total_components": self.stats["total_components"],
            "active_components": self.stats["active_components"],
            "failed_components": self.stats["failed_components"],
            "restart_count": self.stats["restart_count"],
            "total_errors": self.stats["total_errors"],
            "system_health": self._calculate_system_health(),
        }

    def _calculate_system_health(self) -> float:
        """Расчет здоровья системы."""
        if self.stats["total_components"] == 0:
            return 0.0
        active_ratio = self.stats["active_components"] / self.stats["total_components"]
        error_ratio = min(
            self.stats["total_errors"] / max(self.stats["total_components"], 1), 1.0
        )
        health = active_ratio * (1 - error_ratio)
        return max(0.0, min(1.0, health))

    async def stop(self) -> None:
        """Остановка системной интеграции."""
        logger.info("Stopping system integration")
        self.is_running = False
        # Остановка новых модулей мониторинга
        if self.config.enable_system_monitor:
            stop_monitoring()  # Остановка PerformanceMonitor
            logger.info("New monitoring modules stopped")
        # Остановка мониторинга
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.health_check_task:
            self.health_check_task.cancel()
        if self.performance_task:
            self.performance_task.cancel()
        # Остановка всех компонентов
        await self._stop_all_components()
        logger.info("System integration stopped")

    async def _stop_all_components(self) -> None:
        """Остановка всех компонентов."""
        logger.info("Stopping all components")
        for name, component in self.components.items():
            try:
                if hasattr(component, "stop") and callable(getattr(component, "stop")):
                    await component.stop()
                if name in self.component_status:
                    self.component_status[name].is_running = False
                logger.debug(f"Component {name} stopped")
            except Exception as e:
                logger.error(f"Error stopping component {name}: {e}")

    async def get_component(self, name: str) -> Optional[Any]:
        """Получение компонента по имени."""
        return self.components.get(name)

    async def get_system_status(self) -> Dict[str, Any]:
        """Получение статуса системы."""
        return {
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": (
                (datetime.now() - self.start_time).total_seconds()
                if self.start_time
                else 0
            ),
            "stats": self.stats,
            "components": {
                name: {
                    "is_initialized": status.is_initialized,
                    "is_running": status.is_running,
                    "last_heartbeat": status.last_heartbeat.isoformat(),
                    "error_count": status.error_count,
                    "performance_metrics": status.performance_metrics,
                }
                for name, status in self.component_status.items()
            },
        }

    # Event handlers
    async def _on_health_issues(self, event: Event) -> None:
        """Обработчик событий здоровья системы."""
        logger.warning(f"Health issues detected: {event.data}")
        # Уведомление компонентов о проблемах здоровья
        if "system_monitor" in self.components:
            await self.components["system_monitor"].handle_health_alert(event.data)

    async def _on_risk_limit_exceeded(self, event: Event) -> None:
        """Обработчик превышения лимитов риска."""
        logger.warning(f"Risk limit exceeded: {event.data}")
        # Принятие мер по снижению риска
        if "risk_manager" in self.components:
            await self.components["risk_manager"].handle_risk_limit_exceeded(event.data)

    async def _on_trade_executed(self, event: Event) -> None:
        """Обработчик исполнения сделки."""
        logger.info(f"Trade executed: {event.data}")
        # Обновление портфеля
        if "portfolio_manager" in self.components:
            await self.components["portfolio_manager"].handle_trade_executed(event.data)

    async def _on_agent_evolved(self, event: Event) -> None:
        """Обработчик эволюции агента."""
        logger.info(f"Agent evolved: {event.data}")
        # Обновление эволюционной системы
        if "evolution_manager" in self.components:
            await self.components["evolution_manager"].handle_agent_evolution(
                event.data
            )

    async def _handle_initialization_error(self, error: Exception) -> None:
        """Обработка ошибки инициализации."""
        logger.error(f"Initialization error: {error}")
        # Попытка восстановления
        try:
            await self._cleanup_failed_initialization()
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

    async def _cleanup_failed_initialization(self) -> None:
        """Очистка после неудачной инициализации."""
        # Остановка всех инициализированных компонентов
        for name, component in self.components.items():
            try:
                if hasattr(component, "stop") and callable(getattr(component, "stop")):
                    await component.stop()
            except Exception as e:
                logger.error(f"Error stopping component {name} during cleanup: {e}")
        # Очистка состояния
        self.components.clear()
        self.component_status.clear()
        self.is_initialized = False


# Глобальный экземпляр системной интеграции
_system_integration: Optional[SystemIntegration] = None


def get_system_integration(
    config: Optional[SystemIntegrationConfig] = None,
) -> SystemIntegration:
    """Получение глобального экземпляра системной интеграции."""
    global _system_integration
    if _system_integration is None:
        _system_integration = SystemIntegration(config)
    return _system_integration


async def initialize_system_integration(
    config: Optional[SystemIntegrationConfig] = None,
) -> SystemIntegration:
    """Инициализация системной интеграции."""
    integration = get_system_integration(config)
    await integration.initialize()
    return integration


async def start_system_integration() -> SystemIntegration:
    """Запуск системной интеграции."""
    integration = get_system_integration()
    await integration.start()
    return integration


async def stop_system_integration() -> None:
    """Остановка системной интеграции."""
    integration = get_system_integration()
    await integration.stop()
