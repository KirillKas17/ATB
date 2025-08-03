#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Syntra - Интегрированная торговая система
Главный цикл с интеграцией всех модулей
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Добавляем путь для безопасных импортов
sys.path.insert(0, str(Path(__file__).parent))

from safe_import_wrapper import safe_import
from application.di_container_safe import get_safe_service_locator
from application.safe_services import SafeTradingService, SafeRiskService, SafeMarketService

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedTradingSystem:
    """Интегрированная торговая система с полным циклом работы"""
    
    def __init__(self):
        self.running = False
        self.service_locator = None
        self.services = {}
        self.strategies = []
        self.monitored_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"]
        self.market_data_cache = {}
        self.signals_cache = {}
        self.performance_metrics = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0
        }
    
    async def initialize(self):
        """Инициализация всех компонентов системы"""
        logger.info("🚀 Инициализация интегрированной торговой системы...")
        
        try:
            # Загрузка переменных окружения
            try:
                from dotenv import load_dotenv
                load_dotenv()
                logger.info("✅ Переменные окружения загружены")
            except ImportError:
                logger.warning("⚠️ python-dotenv не доступен")
            
            # Инициализация service locator
            self.service_locator = get_safe_service_locator()
            logger.info("✅ Service locator инициализирован")
            
            # Инициализация основных сервисов
            await self._initialize_core_services()
            
            # Инициализация стратегий
            await self._initialize_strategies()
            
            # Инициализация мониторинга
            await self._initialize_monitoring()
            
            # Инициализация риск-менеджмента
            await self._initialize_risk_management()
            
            # Инициализация эволюционных систем
            await self._initialize_evolution_systems()
            
            # Инициализация управления сессиями
            await self._initialize_session_management()
            
            # Инициализация симуляции и backtesting
            await self._initialize_simulation_systems()
            
            logger.info("🎉 Система успешно инициализирована!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации системы: {e}")
            logger.exception("Детали ошибки:")
            return False
    
    async def _initialize_core_services(self):
        """Инициализация основных сервисов"""
        logger.info("🔧 Инициализация основных сервисов...")
        
        # Торговый сервис
        self.services["trading"] = self.service_locator.trading_service()
        logger.info(f"✅ Торговый сервис: {type(self.services['trading']).__name__}")
        
        # Сервис рисков
        self.services["risk"] = self.service_locator.risk_service()
        logger.info(f"✅ Сервис рисков: {type(self.services['risk']).__name__}")
        
        # Рыночный сервис
        self.services["market"] = self.service_locator.market_service()
        logger.info(f"✅ Рыночный сервис: {type(self.services['market']).__name__}")
        
        # Agent Context - ядро агентной архитектуры
        try:
            from infrastructure.agents.agent_context_refactored import AgentContext
            self.services["agent_context"] = AgentContext()
            logger.info("✅ Agent Context инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Agent Context недоступен: {e}")
            self.services["agent_context"] = safe_import("safe_import_wrapper", "SafeImportMock")("AgentContext")
        
        # Репозитории данных
        await self._initialize_repositories()
        
        # Дополнительные сервисы (с безопасными импортами)
        try:
            from application.services.service_factory import ServiceFactory
            factory = ServiceFactory()
            
            self.services["ml_predictor"] = factory._get_ml_predictor()
            self.services["signal_service"] = factory._get_signal_service()
            self.services["portfolio_optimizer"] = factory._get_portfolio_optimizer()
            
            logger.info("✅ Дополнительные сервисы инициализированы")
        except Exception as e:
            logger.warning(f"⚠️ Некоторые сервисы недоступны: {e}")
    
    async def _initialize_repositories(self):
        """Инициализация репозиториев данных"""
        logger.info("🗄️ Инициализация репозиториев...")
        
        # Market Repository
        try:
            from infrastructure.repositories.market_repository import MarketRepositoryImpl
            self.services["market_repository"] = MarketRepositoryImpl()
            logger.info("✅ Market Repository инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Market Repository недоступен: {e}")
            self.services["market_repository"] = safe_import("safe_import_wrapper", "SafeImportMock")("MarketRepository")
        
        # Trading Repository
        try:
            from infrastructure.repositories.trading_repository import TradingRepositoryImpl
            self.services["trading_repository"] = TradingRepositoryImpl()
            logger.info("✅ Trading Repository инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Trading Repository недоступен: {e}")
            self.services["trading_repository"] = safe_import("safe_import_wrapper", "SafeImportMock")("TradingRepository")
        
        # Portfolio Repository
        try:
            from infrastructure.repositories.portfolio_repository import PortfolioRepositoryImpl
            self.services["portfolio_repository"] = PortfolioRepositoryImpl()
            logger.info("✅ Portfolio Repository инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Portfolio Repository недоступен: {e}")
            self.services["portfolio_repository"] = safe_import("safe_import_wrapper", "SafeImportMock")("PortfolioRepository")
        
        # ML Repository
        try:
            from infrastructure.repositories.ml_repository import MLRepositoryImpl
            self.services["ml_repository"] = MLRepositoryImpl()
            logger.info("✅ ML Repository инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ ML Repository недоступен: {e}")
            self.services["ml_repository"] = safe_import("safe_import_wrapper", "SafeImportMock")("MLRepository")
    
    async def _initialize_strategies(self):
        """Инициализация торговых стратегий"""
        logger.info("📈 Инициализация торговых стратегий...")
        
        # Безопасная загрузка стратегий
        strategy_classes = [
            ("infrastructure.strategies.trend_strategy", "TrendStrategy"),
            ("infrastructure.strategies.adaptive.adaptive_strategy_generator", "AdaptiveStrategyGenerator"),
            ("infrastructure.strategies.mean_reversion_strategy", "MeanReversionStrategy")
        ]
        
        for module_name, class_name in strategy_classes:
            try:
                strategy_class = safe_import(module_name, class_name)
                if hasattr(strategy_class, '__call__'):
                    strategy = strategy_class()
                    self.strategies.append({
                        "name": class_name,
                        "instance": strategy,
                        "enabled": True,
                        "performance": {"trades": 0, "wins": 0, "pnl": 0.0}
                    })
                    logger.info(f"✅ Стратегия загружена: {class_name}")
                else:
                    logger.warning(f"⚠️ Стратегия {class_name} недоступна (mock)")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить стратегию {class_name}: {e}")
        
        if not self.strategies:
            # Создаём базовую стратегию
            self.strategies.append({
                "name": "BasicStrategy",
                "instance": safe_import("safe_import_wrapper", "SafeImportMock")("BasicStrategy"),
                "enabled": True,
                "performance": {"trades": 0, "wins": 0, "pnl": 0.0}
            })
            logger.info("✅ Базовая стратегия создана")
    
    async def _initialize_monitoring(self):
        """Инициализация системы мониторинга"""
        logger.info("📊 Инициализация системы мониторинга...")
        
        # Event Bus - ядро event-driven архитектуры
        try:
            from infrastructure.messaging.event_bus import EventBus
            self.services["event_bus"] = EventBus()
            logger.info("✅ Event Bus инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Event Bus недоступен: {e}")
            self.services["event_bus"] = safe_import("safe_import_wrapper", "SafeImportMock")("EventBus")
        
        # Message Queue для асинхронных сообщений
        try:
            from infrastructure.messaging.message_queue import MessageQueue
            self.services["message_queue"] = MessageQueue()
            logger.info("✅ Message Queue инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Message Queue недоступен: {e}")
            self.services["message_queue"] = safe_import("safe_import_wrapper", "SafeImportMock")("MessageQueue")
        
        # Health Monitoring
        try:
            from infrastructure.health.checker import HealthChecker
            self.services["health_checker"] = HealthChecker()
            logger.info("✅ Health Checker инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Health Checker недоступен: {e}")
            self.services["health_checker"] = safe_import("safe_import_wrapper", "SafeImportMock")("HealthChecker")
        
        # Мониторинг производительности
        self.services["performance_monitor"] = safe_import(
            "shared.performance_monitor", "PerformanceMonitor"
        )()
        
        # Мониторинг системы
        self.services["system_monitor"] = safe_import(
            "infrastructure.monitoring.system_monitor", "SystemMonitor"
        )()
        
        # Дашборд мониторинга
        self.services["monitoring_dashboard"] = safe_import(
            "shared.monitoring_dashboard", "MonitoringDashboard"
        )()
        
        logger.info("✅ Система мониторинга инициализирована")
    
    async def _initialize_risk_management(self):
        """Инициализация риск-менеджмента"""
        logger.info("🛡️ Инициализация риск-менеджмента...")
        
        # Circuit Breaker для защиты от сбоев
        try:
            from infrastructure.circuit_breaker.breaker import CircuitBreaker
            self.services["circuit_breaker"] = CircuitBreaker()
            logger.info("✅ Circuit Breaker инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Circuit Breaker недоступен: {e}")
            self.services["circuit_breaker"] = safe_import("safe_import_wrapper", "SafeImportMock")("CircuitBreaker")
        
        # Fallback механизмы
        try:
            from infrastructure.circuit_breaker.fallback import FallbackHandler
            self.services["fallback_handler"] = FallbackHandler()
            logger.info("✅ Fallback Handler инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Fallback Handler недоступен: {e}")
            self.services["fallback_handler"] = safe_import("safe_import_wrapper", "SafeImportMock")("FallbackHandler")
        
        # Анализ корреляций
        try:
            EntanglementMonitor = safe_import(
                "application.analysis.entanglement_monitor", "EntanglementMonitor"
            )
            # Создаём с пустыми коннекторами для тестирования
            self.services["correlation_analyzer"] = EntanglementMonitor([])
        except Exception as e:
            logger.warning(f"⚠️ EntanglementMonitor недоступен: {e}")
            self.services["correlation_analyzer"] = safe_import("safe_import_wrapper", "SafeImportMock")("CorrelationAnalyzer")
        
        # Внешние сервисы
        await self._initialize_external_services()
        
        logger.info("✅ Риск-менеджмент инициализирован")
    
    async def _initialize_external_services(self):
        """Инициализация внешних сервисов"""
        logger.info("🌐 Инициализация внешних сервисов...")
        
        # Exchange Services для подключения к биржам
        try:
            from infrastructure.external_services.exchanges.base_exchange_service import BaseExchangeService
            self.services["exchange_service"] = BaseExchangeService()
            logger.info("✅ Exchange Service инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Exchange Service недоступен: {e}")
            self.services["exchange_service"] = safe_import("safe_import_wrapper", "SafeImportMock")("ExchangeService")
        
        # Technical Analysis Service
        try:
            from infrastructure.external_services.technical_analysis_adapter import TechnicalAnalysisAdapter
            self.services["technical_analysis"] = TechnicalAnalysisAdapter()
            logger.info("✅ Technical Analysis Service инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Technical Analysis Service недоступен: {e}")
            self.services["technical_analysis"] = safe_import("safe_import_wrapper", "SafeImportMock")("TechnicalAnalysis")
        
        # Risk Analysis Adapter
        try:
            from infrastructure.external_services.risk_analysis_adapter import RiskAnalysisServiceAdapter
            self.services["risk_analysis_adapter"] = RiskAnalysisServiceAdapter()
            logger.info("✅ Risk Analysis Adapter инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Risk Analysis Adapter недоступен: {e}")
            self.services["risk_analysis_adapter"] = safe_import("safe_import_wrapper", "SafeImportMock")("RiskAnalysisAdapter")
    
    async def _initialize_evolution_systems(self):
        """Инициализация эволюционных систем"""
        logger.info("🧬 Инициализация эволюционных систем...")
        
        # Strategy Generator для создания адаптивных стратегий
        try:
            from domain.evolution.strategy_generator import StrategyGenerator
            self.services["strategy_generator"] = StrategyGenerator()
            logger.info("✅ Strategy Generator инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Strategy Generator недоступен: {e}")
            self.services["strategy_generator"] = safe_import("safe_import_wrapper", "SafeImportMock")("StrategyGenerator")
        
        # Strategy Optimizer для оптимизации стратегий
        try:
            from domain.evolution.strategy_optimizer import StrategyOptimizer
            self.services["strategy_optimizer"] = StrategyOptimizer()
            logger.info("✅ Strategy Optimizer инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Strategy Optimizer недоступен: {e}")
            self.services["strategy_optimizer"] = safe_import("safe_import_wrapper", "SafeImportMock")("StrategyOptimizer")
        
        # Evolution Storage для хранения эволюционных данных
        try:
            from infrastructure.evolution.strategy_storage import StrategyStorage
            self.services["evolution_storage"] = StrategyStorage()
            logger.info("✅ Evolution Storage инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Evolution Storage недоступен: {e}")
            self.services["evolution_storage"] = safe_import("safe_import_wrapper", "SafeImportMock")("EvolutionStorage")
    
    async def _initialize_session_management(self):
        """Инициализация управления сессиями"""
        logger.info("🔄 Инициализация управления сессиями...")
        
        # Session Manager для управления торговыми сессиями
        try:
            from domain.sessions.session_manager import SessionManager
            self.services["session_manager"] = SessionManager()
            logger.info("✅ Session Manager инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Session Manager недоступен: {e}")
            self.services["session_manager"] = safe_import("safe_import_wrapper", "SafeImportMock")("SessionManager")
        
        # Session Predictor для предсказания сессий
        try:
            from domain.sessions.session_predictor import SessionPredictor
            self.services["session_predictor"] = SessionPredictor()
            logger.info("✅ Session Predictor инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Session Predictor недоступен: {e}")
            self.services["session_predictor"] = safe_import("safe_import_wrapper", "SafeImportMock")("SessionPredictor")
        
        # Session Analyzer для анализа сессий
        try:
            from domain.sessions.session_analyzer import SessionAnalyzer
            self.services["session_analyzer"] = SessionAnalyzer()
            logger.info("✅ Session Analyzer инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Session Analyzer недоступен: {e}")
            self.services["session_analyzer"] = safe_import("safe_import_wrapper", "SafeImportMock")("SessionAnalyzer")
    
    async def _initialize_simulation_systems(self):
        """Инициализация систем симуляции"""
        logger.info("🔧 Инициализация систем симуляции...")
        
        # Market Simulator для симуляции рынка
        try:
            from infrastructure.simulation.market_simulator import MarketSimulator
            self.services["market_simulator"] = MarketSimulator()
            logger.info("✅ Market Simulator инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Market Simulator недоступен: {e}")
            self.services["market_simulator"] = safe_import("safe_import_wrapper", "SafeImportMock")("MarketSimulator")
        
        # Backtester для тестирования стратегий
        try:
            from infrastructure.simulation.backtester import Backtester
            self.services["backtester"] = Backtester()
            logger.info("✅ Backtester инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Backtester недоступен: {e}")
            self.services["backtester"] = safe_import("safe_import_wrapper", "SafeImportMock")("Backtester")
        
        # Backtest Explainer для объяснения результатов
        try:
            from infrastructure.simulation.backtest_explainer import BacktestExplainer
            self.services["backtest_explainer"] = BacktestExplainer()
            logger.info("✅ Backtest Explainer инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Backtest Explainer недоступен: {e}")
            self.services["backtest_explainer"] = safe_import("safe_import_wrapper", "SafeImportMock")("BacktestExplainer")
    
    async def start_trading(self):
        """Запуск основного торгового цикла"""
        logger.info("🎯 Запуск торгового цикла...")
        
        self.running = True
        
        # Настройка обработчиков сигналов для graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Основной цикл
            while self.running:
                await self._main_trading_cycle()
                await asyncio.sleep(5)  # Пауза между циклами
                
        except Exception as e:
            logger.error(f"❌ Ошибка в торговом цикле: {e}")
            logger.exception("Детали ошибки:")
        finally:
            await self.shutdown()
    
    async def _main_trading_cycle(self):
        """Основной торговый цикл"""
        cycle_start = datetime.now()
        
        try:
            # 0. Проверка здоровья системы
            await self._check_system_health()
            
            # 1. Сбор рыночных данных
            await self._collect_market_data()
            
            # 2. Анализ сессий и контекста
            await self._analyze_sessions_and_context()
            
            # 3. Анализ рынка и генерация сигналов
            await self._analyze_market_and_generate_signals()
            
            # 4. Эволюция стратегий
            await self._evolve_strategies()
            
            # 5. Оценка рисков с Circuit Breaker
            await self._assess_risks_with_protection()
            
            # 6. Выполнение торговых операций через агентов
            await self._execute_trades_with_agents()
            
            # 7. Мониторинг позиций
            await self._monitor_positions()
            
            # 8. Event Bus обработка
            await self._process_events()
            
            # 9. Обновление метрик производительности
            await self._update_performance_metrics()
            
            # 10. Логирование состояния
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            active_services = len([s for s in self.services.values() if hasattr(s, '__call__')])
            logger.info(f"💫 Цикл завершен за {cycle_duration:.2f}с | "
                       f"Символов: {len(self.monitored_symbols)} | "
                       f"Сигналов: {len(self.signals_cache)} | "
                       f"Стратегий: {len([s for s in self.strategies if s['enabled']])} | "
                       f"Сервисов: {active_services}")
        
        except Exception as e:
            logger.error(f"❌ Ошибка в торговом цикле: {e}")
            # Отправляем событие об ошибке через Event Bus
            await self._send_error_event(e)
    
    async def _collect_market_data(self):
        """Сбор рыночных данных"""
        market_service = self.services.get("market")
        if not market_service:
            return
        
        for symbol in self.monitored_symbols:
            try:
                # Получаем рыночные данные
                market_data = await market_service.get_market_data(symbol)
                if market_data:
                    self.market_data_cache[symbol] = market_data
                
                # Получаем технические индикаторы
                if hasattr(market_service, 'get_technical_indicators'):
                    tech_indicators = await market_service.get_technical_indicators(symbol)
                    if tech_indicators:
                        self.market_data_cache[f"{symbol}_indicators"] = tech_indicators
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка получения данных для {symbol}: {e}")
    
    async def _analyze_market_and_generate_signals(self):
        """Анализ рынка и генерация торговых сигналов с продвинутыми методами"""
        
        # 1. Инициализируем улучшенный сервис прогнозирования
        enhanced_prediction_service = None
        try:
            from application.services.enhanced_prediction_service import EnhancedPredictionService
            enhanced_prediction_service = EnhancedPredictionService({
                "advanced_engine": {
                    "min_fvg_size": 0.001,
                    "snr_window": 50,
                    "orderflow_window": 20
                }
            })
            logger.debug("✅ Enhanced Prediction Service инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ Enhanced Prediction Service недоступен: {e}")
        
        # 2. Генерируем продвинутые прогнозы для каждого символа
        for symbol in self.monitored_symbols:
            try:
                market_data = self.market_data_cache.get(symbol)
                if not market_data:
                    continue
                
                # Продвинутый анализ с FVG, SNR, OrderFlow
                if enhanced_prediction_service:
                    enhanced_prediction = await enhanced_prediction_service.generate_enhanced_prediction(
                        symbol=symbol,
                        market_service=self.services.get("market"),
                        timeframe="4H"
                    )
                    
                    if enhanced_prediction and enhanced_prediction.confidence > 0.3:
                        signal_key = f"{symbol}_enhanced"
                        self.signals_cache[signal_key] = {
                            "symbol": symbol,
                            "strategy": "enhanced_prediction",
                            "signal": {
                                "action": enhanced_prediction.direction,
                                "confidence": enhanced_prediction.confidence,
                                "target_price": enhanced_prediction.target_price,
                                "stop_loss": enhanced_prediction.stop_loss,
                                "risk_reward_ratio": enhanced_prediction.risk_reward_ratio,
                                "market_structure": enhanced_prediction.market_structure,
                                "volatility_regime": enhanced_prediction.volatility_regime,
                                "snr_ratio": enhanced_prediction.snr_metrics.snr_ratio,
                                "clarity_score": enhanced_prediction.snr_metrics.clarity_score,
                                "fvg_count": len(enhanced_prediction.fvg_signals),
                                "orderflow_count": len(enhanced_prediction.orderflow_signals),
                                "liquidity_levels": len(enhanced_prediction.liquidity_levels),
                                "prediction_type": "advanced"
                            },
                            "timestamp": datetime.now(),
                            "market_data": market_data,
                            "enhanced_data": {
                                "fvg_signals": enhanced_prediction.fvg_signals,
                                "orderflow_signals": enhanced_prediction.orderflow_signals,
                                "liquidity_levels": enhanced_prediction.liquidity_levels,
                                "snr_metrics": enhanced_prediction.snr_metrics
                            }
                        }
                        
                        logger.debug(f"📊 Enhanced сигнал для {symbol}: {enhanced_prediction.direction} "
                                   f"(уверенность: {enhanced_prediction.confidence:.3f}, "
                                   f"SNR: {enhanced_prediction.snr_metrics.snr_ratio:.2f})")
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка продвинутого анализа для {symbol}: {e}")
        
        # 3. Традиционные стратегии (в дополнение к продвинутому анализу)
        for strategy in self.strategies:
            if not strategy["enabled"]:
                continue
                
            strategy_name = strategy["name"]
            strategy_instance = strategy["instance"]
            
            try:
                for symbol in self.monitored_symbols:
                    market_data = self.market_data_cache.get(symbol)
                    if not market_data:
                        continue
                    
                    # Генерируем сигнал через традиционную стратегию
                    signal = await self._generate_strategy_signal(
                        strategy_instance, symbol, market_data
                    )
                    
                    if signal:
                        signal_key = f"{symbol}_{strategy_name}"
                        self.signals_cache[signal_key] = {
                            "symbol": symbol,
                            "strategy": strategy_name,
                            "signal": signal,
                            "timestamp": datetime.now(),
                            "market_data": market_data
                        }
                        
            except Exception as e:
                logger.warning(f"⚠️ Ошибка генерации сигналов для {strategy_name}: {e}")
        
        # 4. Логируем статистику сигналов
        enhanced_signals = len([s for s in self.signals_cache.values() 
                              if s.get("signal", {}).get("prediction_type") == "advanced"])
        traditional_signals = len(self.signals_cache) - enhanced_signals
        
        if enhanced_signals > 0:
            avg_confidence = sum(s["signal"]["confidence"] for s in self.signals_cache.values() 
                               if s.get("signal", {}).get("prediction_type") == "advanced") / enhanced_signals
            avg_snr = sum(s["signal"]["snr_ratio"] for s in self.signals_cache.values() 
                         if s.get("signal", {}).get("prediction_type") == "advanced") / enhanced_signals
            
            logger.info(f"🎯 Сгенерировано сигналов: Enhanced={enhanced_signals}, Traditional={traditional_signals}")
            logger.info(f"📈 Enhanced качество: avg_confidence={avg_confidence:.3f}, avg_SNR={avg_snr:.2f}")
    
    async def _generate_strategy_signal(self, strategy, symbol, market_data):
        """Генерация сигнала от стратегии"""
        try:
            # Если стратегия имеет метод генерации сигналов
            if hasattr(strategy, 'generate_signal'):
                return await strategy.generate_signal(symbol, market_data)
            elif hasattr(strategy, 'analyze'):
                return await strategy.analyze(symbol, market_data)
            else:
                # Базовая логика для mock стратегий
                price = float(market_data.get("price", 0))
                volume = float(market_data.get("volume", 0))
                
                # Простая логика: покупка при низком RSI, продажа при высоком
                indicators = self.market_data_cache.get(f"{symbol}_indicators")
                if indicators and hasattr(indicators, 'data'):
                    rsi = indicators.data.get("rsi", 50)
                    if rsi and float(rsi) < 30:
                        return {"action": "buy", "confidence": 0.7, "reason": "oversold"}
                    elif rsi and float(rsi) > 70:
                        return {"action": "sell", "confidence": 0.7, "reason": "overbought"}
                
                return None
        except Exception as e:
            logger.warning(f"⚠️ Ошибка генерации сигнала: {e}")
            return None
    
    async def _assess_risks(self):
        """Оценка рисков"""
        risk_service = self.services.get("risk")
        if not risk_service:
            return
        
        try:
            # Оценка рисков портфеля
            portfolio_risk = await risk_service.get_portfolio_risk()
            
            # Проверка лимитов для каждого сигнала
            for signal_key, signal_data in self.signals_cache.items():
                symbol = signal_data["symbol"]
                signal = signal_data["signal"]
                
                if signal and signal.get("action") in ["buy", "sell"]:
                    # Валидация ордера через риск-сервис
                    risk_validation = await risk_service.validate_order(
                        symbol=symbol,
                        action=signal["action"],
                        amount=1000  # базовая сумма
                    )
                    
                    signal_data["risk_validated"] = risk_validation.get("is_valid", False)
                    signal_data["risk_score"] = risk_validation.get("risk_score", 1.0)
                    
        except Exception as e:
            logger.warning(f"⚠️ Ошибка оценки рисков: {e}")
    
    async def _execute_trades(self):
        """Выполнение торговых операций"""
        trading_service = self.services.get("trading")
        if not trading_service:
            return
        
        executed_trades = 0
        
        for signal_key, signal_data in self.signals_cache.items():
            signal = signal_data.get("signal")
            if not signal or not signal_data.get("risk_validated", False):
                continue
            
            try:
                symbol = signal_data["symbol"]
                action = signal["action"]
                confidence = signal.get("confidence", 0.5)
                
                # Выполняем торговую операцию только при высокой уверенности
                if confidence > 0.6:
                    trade_result = await trading_service.create_order(
                        symbol=symbol,
                        side=action,
                        amount=1000,  # базовая сумма
                        order_type="market"
                    )
                    
                    if trade_result.get("status") == "created":
                        executed_trades += 1
                        self.performance_metrics["total_trades"] += 1
                        
                        # Обновляем производительность стратегии
                        strategy_name = signal_data["strategy"]
                        for strategy in self.strategies:
                            if strategy["name"] == strategy_name:
                                strategy["performance"]["trades"] += 1
                                break
                        
                        logger.info(f"✅ Ордер выполнен: {symbol} {action} (уверенность: {confidence:.2f})")
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка выполнения торговой операции: {e}")
        
        if executed_trades > 0:
            logger.info(f"📊 Выполнено торговых операций: {executed_trades}")
    
    async def _monitor_positions(self):
        """Мониторинг позиций"""
        trading_service = self.services.get("trading")
        if not trading_service:
            return
        
        try:
            # Получаем активные позиции
            positions = await trading_service.get_active_positions()
            
            if positions:
                logger.info(f"📋 Активных позиций: {len(positions)}")
                
                # Мониторим каждую позицию
                for position in positions:
                    # Здесь могла бы быть логика trailing stop, take profit и т.д.
                    pass
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка мониторинга позиций: {e}")
    
    async def _update_performance_metrics(self):
        """Обновление метрик производительности"""
        try:
            trading_service = self.services.get("trading")
            if trading_service:
                # Получаем статистику торговли
                stats = await trading_service.get_trading_statistics()
                
                if stats:
                    self.performance_metrics.update({
                        "successful_trades": stats.get("winning_trades", 0),
                        "win_rate": stats.get("win_rate", 0.0),
                        "profit_factor": stats.get("profit_factor", 0.0)
                    })
                
                # Получаем портфолио
                portfolio = await trading_service.get_portfolio_summary()
                if portfolio:
                    self.performance_metrics["total_value"] = portfolio.get("total_value", 0)
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка обновления метрик: {e}")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для graceful shutdown"""
        logger.info(f"📡 Получен сигнал {signum}, начинаю остановку...")
        self.running = False
    
    async def shutdown(self):
        """Graceful shutdown системы"""
        logger.info("🛑 Остановка торговой системы...")
        
        try:
            # Закрываем все позиции (опционально)
            # await self._close_all_positions()
            
            # Сохраняем состояние
            await self._save_state()
            
            # Останавливаем сервисы
            for service_name, service in self.services.items():
                try:
                    if hasattr(service, 'stop'):
                        await service.stop()
                    elif hasattr(service, 'shutdown'):
                        await service.shutdown()
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка остановки сервиса {service_name}: {e}")
            
            logger.info("✅ Система остановлена")
            
        except Exception as e:
            logger.error(f"❌ Ошибка при остановке: {e}")
    
    async def _save_state(self):
        """Сохранение состояния системы"""
        try:
            state = {
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": self.performance_metrics,
                "strategies_performance": [s["performance"] for s in self.strategies],
                "monitored_symbols": self.monitored_symbols
            }
            
            # В реальной системе сохраняли бы в БД или файл
            logger.info(f"💾 Состояние сохранено: {state}")
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка сохранения состояния: {e}")
    
    async def _check_system_health(self):
        """Проверка здоровья системы"""
        health_checker = self.services.get("health_checker")
        if health_checker and hasattr(health_checker, 'check_health'):
            try:
                health_status = await health_checker.check_health()
                if not health_status.get("healthy", True):
                    logger.warning(f"⚠️ Проблемы со здоровьем системы: {health_status}")
            except Exception as e:
                logger.warning(f"⚠️ Ошибка проверки здоровья: {e}")
    
    async def _analyze_sessions_and_context(self):
        """Анализ сессий и контекста"""
        try:
            session_analyzer = self.services.get("session_analyzer")
            agent_context = self.services.get("agent_context")
            
            if session_analyzer and hasattr(session_analyzer, 'analyze_current_session'):
                session_analysis = await session_analyzer.analyze_current_session()
                if session_analysis:
                    logger.debug(f"📊 Анализ сессии: {session_analysis}")
            
            if agent_context and hasattr(agent_context, 'update_context'):
                await agent_context.update_context(self.market_data_cache)
                logger.debug("🤖 Agent Context обновлен")
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка анализа сессий: {e}")
    
    async def _evolve_strategies(self):
        """Эволюция торговых стратегий"""
        try:
            strategy_optimizer = self.services.get("strategy_optimizer")
            
            if strategy_optimizer and hasattr(strategy_optimizer, 'optimize_strategies'):
                # Передаем данные о производительности стратегий
                strategy_performance = [s["performance"] for s in self.strategies]
                optimization_result = await strategy_optimizer.optimize_strategies(strategy_performance)
                
                if optimization_result:
                    logger.debug(f"🧬 Стратегии оптимизированы: {optimization_result}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Ошибка эволюции стратегий: {e}")
    
    async def _assess_risks_with_protection(self):
        """Оценка рисков с Circuit Breaker защитой"""
        circuit_breaker = self.services.get("circuit_breaker")
        
        try:
            # Используем Circuit Breaker для защиты от сбоев
            if circuit_breaker and hasattr(circuit_breaker, 'call'):
                await circuit_breaker.call(self._assess_risks)
            else:
                await self._assess_risks()
                
        except Exception as e:
            logger.warning(f"⚠️ Circuit Breaker сработал при оценке рисков: {e}")
            
            # Используем Fallback механизм
            fallback_handler = self.services.get("fallback_handler")
            if fallback_handler and hasattr(fallback_handler, 'handle_risk_failure'):
                await fallback_handler.handle_risk_failure()
    
    async def _execute_trades_with_agents(self):
        """Выполнение торговых операций через агентную систему"""
        try:
            agent_context = self.services.get("agent_context")
            
            # Если агентный контекст доступен, используем его для координации
            if agent_context and hasattr(agent_context, 'execute_trading_decisions'):
                trading_decisions = await agent_context.execute_trading_decisions(self.signals_cache)
                
                if trading_decisions:
                    logger.debug(f"🤖 Агентные торговые решения: {len(trading_decisions)}")
                    
                    # Выполняем решения через обычную торговую логику
                    await self._execute_agent_decisions(trading_decisions)
            else:
                # Fallback к стандартной торговле
                await self._execute_trades()
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка выполнения торговли через агентов: {e}")
            # Fallback к стандартной торговле
            await self._execute_trades()
    
    async def _execute_agent_decisions(self, decisions):
        """Выполнение решений агентов"""
        trading_service = self.services.get("trading")
        if not trading_service:
            return
        
        executed_trades = 0
        
        for decision in decisions:
            try:
                if decision.get("action") and decision.get("symbol"):
                    trade_result = await trading_service.create_order(
                        symbol=decision["symbol"],
                        side=decision["action"],
                        amount=decision.get("amount", 1000),
                        order_type=decision.get("order_type", "market")
                    )
                    
                    if trade_result.get("status") == "created":
                        executed_trades += 1
                        self.performance_metrics["total_trades"] += 1
                        
            except Exception as e:
                logger.warning(f"⚠️ Ошибка выполнения агентного решения: {e}")
        
        if executed_trades > 0:
            logger.info(f"🤖 Выполнено агентных торговых операций: {executed_trades}")
    
    async def _process_events(self):
        """Обработка событий через Event Bus"""
        try:
            event_bus = self.services.get("event_bus")
            
            if event_bus and hasattr(event_bus, 'process_pending_events'):
                await event_bus.process_pending_events()
                
            # Отправляем событие о завершении цикла
            if event_bus and hasattr(event_bus, 'publish'):
                await event_bus.publish("trading_cycle_completed", {
                    "timestamp": datetime.now(),
                    "symbols_processed": len(self.monitored_symbols),
                    "signals_generated": len(self.signals_cache)
                })
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка обработки событий: {e}")
    
    async def _send_error_event(self, error):
        """Отправка события об ошибке"""
        try:
            event_bus = self.services.get("event_bus")
            
            if event_bus and hasattr(event_bus, 'publish'):
                await event_bus.publish("system_error", {
                    "timestamp": datetime.now(),
                    "error": str(error),
                    "error_type": type(error).__name__
                })
                
        except Exception as e:
            logger.warning(f"⚠️ Ошибка отправки события об ошибке: {e}")


async def main():
    """Главная функция запуска системы"""
    logger.info("🚀 ATB Integrated Trading System")
    logger.info("=" * 60)
    
    system = IntegratedTradingSystem()
    
    try:
        # Инициализация системы
        if await system.initialize():
            logger.info("🎯 Система готова к торговле!")
            
            # Запуск торгового цикла
            await system.start_trading()
        else:
            logger.error("❌ Не удалось инициализировать систему")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⌨️ Получен сигнал прерывания")
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        logger.exception("Детали ошибки:")
        sys.exit(1)
    finally:
        logger.info("🏁 Завершение работы")


if __name__ == "__main__":
    print("🚀 ATB Integrated Trading System")
    print("=" * 60)
    print("Интегрированная система автономной торговли")
    print("Включает: торговлю, риск-менеджмент, ML, мониторинг")
    print("=" * 60)
    
    asyncio.run(main())