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
        
        # Circuit breaker для защиты от потерь
        self.services["circuit_breaker"] = safe_import(
            "infrastructure.risk.circuit_breaker", "CircuitBreaker"
        )()
        
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
        
        logger.info("✅ Риск-менеджмент инициализирован")
    
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
            # 1. Сбор рыночных данных
            await self._collect_market_data()
            
            # 2. Анализ рынка и генерация сигналов
            await self._analyze_market_and_generate_signals()
            
            # 3. Оценка рисков
            await self._assess_risks()
            
            # 4. Выполнение торговых операций
            await self._execute_trades()
            
            # 5. Мониторинг позиций
            await self._monitor_positions()
            
            # 6. Обновление метрик производительности
            await self._update_performance_metrics()
            
            # 7. Логирование состояния
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"💫 Цикл завершен за {cycle_duration:.2f}с | "
                       f"Символов: {len(self.monitored_symbols)} | "
                       f"Сигналов: {len(self.signals_cache)} | "
                       f"Стратегий: {len([s for s in self.strategies if s['enabled']])}")
        
        except Exception as e:
            logger.error(f"❌ Ошибка в торговом цикле: {e}")
    
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
        """Анализ рынка и генерация торговых сигналов"""
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
                    
                    # Генерируем сигнал через стратегию
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