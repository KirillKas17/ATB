#!/usr/bin/env python3
"""
Пример интеграции DOASS в основной цикл системы ATB.

Этот пример демонстрирует, как модуль DynamicOpportunityAwareSymbolSelector (DOASS)
интегрируется в торговый цикл системы для динамического выбора торговых пар.
"""

import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any
from decimal import Decimal

from loguru import logger

# Импорты ATB системы
from application.di_container import get_container, ContainerConfig
from application.symbol_selection.opportunity_selector import DynamicOpportunityAwareSymbolSelector, DOASSConfig
from domain.symbols.market_phase_classifier import MarketPhaseClassifier
from domain.symbols.opportunity_score import OpportunityScoreCalculator
from infrastructure.data.symbol_metrics_provider import SymbolMetricsProvider
from infrastructure.agents.agent_context_refactored import AgentContext, AgentContextManager
from unittest.mock import Mock


class DOASSIntegrationExample:
    """Пример интеграции DOASS в торговый цикл."""

    def __init__(self):
        """Инициализация примера."""
        self.container = None
        self.doass = None
        self.agent_context_manager = AgentContextManager()
        self.trading_orchestrator = None
        
        # Конфигурация для примера
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
            "BNBUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT", "EOSUSDT"
        ]
        
        logger.info("DOASS Integration Example initialized")

    async def setup_system(self):
        """Настройка системы с интеграцией DOASS."""
        try:
            logger.info("Setting up ATB system with DOASS integration...")
            
            # Создаем конфигурацию контейнера с включенным DOASS
            config = ContainerConfig(
                doass_enabled=True,
                cache_enabled=True,
                risk_management_enabled=True,
                technical_analysis_enabled=True
            )
            
            # Получаем DI контейнер
            self.container = get_container(config)
            
            # Получаем DOASS из контейнера
            self.doass = self.container.get("doass")
            
            if not self.doass:
                logger.error("Failed to get DOASS from container")
                return False
                
            logger.info("DOASS successfully loaded from DI container")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up system: {e}")
            return False

    async def demonstrate_symbol_selection(self):
        """Демонстрация выбора символов через DOASS."""
        try:
            logger.info("Demonstrating DOASS symbol selection...")
            
            # Выполняем анализ и выбор символов
            result = await self.doass.select_opportunity_symbols(
                symbols=self.symbols,
                max_symbols=5,
                min_opportunity_score=0.7
            )
            
            logger.info(f"DOASS Analysis Results:")
            logger.info(f"  Total symbols analyzed: {result.total_symbols_analyzed}")
            logger.info(f"  Selected symbols: {result.selected_symbols}")
            logger.info(f"  Processing time: {result.processing_time_ms:.2f}ms")
            logger.info(f"  Cache hit rate: {result.cache_hit_rate:.2f}")
            
            # Показываем детальную информацию по выбранным символам
            for symbol in result.selected_symbols[:3]:  # Показываем топ-3
                if symbol in result.detailed_profiles:
                    profile = result.detailed_profiles[symbol]
                    logger.info(f"\nSymbol: {symbol}")
                    logger.info(f"  Opportunity Score: {profile.opportunity_score:.3f}")
                    logger.info(f"  Confidence: {profile.confidence:.3f}")
                    logger.info(f"  Market Phase: {profile.market_phase.value}")
                    logger.info(f"  Volume: {profile.volume:.2f}")
                    logger.info(f"  Spread: {profile.spread:.4f}")
                    logger.info(f"  Volatility: {profile.volatility:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in symbol selection demonstration: {e}")
            return None

    async def demonstrate_agent_context_integration(self):
        """Демонстрация интеграции с AgentContext."""
        try:
            logger.info("Demonstrating DOASS integration with AgentContext...")
            
            # Создаем контекст агента для BTCUSDT
            symbol = "BTCUSDT"
            agent_context = AgentContext(symbol=symbol)
            
            # Выполняем анализ DOASS
            doass_result = await self.doass.select_opportunity_symbols(
                symbols=[symbol],
                max_symbols=1,
                min_opportunity_score=0.5
            )
            
            # Обновляем контекст агента
            agent_context.doass_result = doass_result
            self.agent_context_manager.update_context(symbol, agent_context)
            
            # Применяем модификаторы DOASS
            agent_context.apply_doass_modifier()
            
            # Получаем статус DOASS
            doass_status = agent_context.get_doass_status()
            
            logger.info(f"AgentContext Integration Results for {symbol}:")
            logger.info(f"  Is analyzed: {doass_status['is_analyzed']}")
            logger.info(f"  Status: {doass_status['status']}")
            
            if doass_status['is_analyzed']:
                logger.info(f"  Opportunity Score: {doass_status['opportunity_score']:.3f}")
                logger.info(f"  Confidence: {doass_status['confidence']:.3f}")
                logger.info(f"  Market Phase: {doass_status['market_phase']}")
            
            # Показываем модификаторы стратегий
            modifiers = agent_context.strategy_modifiers
            logger.info(f"\nStrategy Modifiers after DOASS:")
            logger.info(f"  Order Aggressiveness: {modifiers.order_aggressiveness:.3f}")
            logger.info(f"  Position Size Multiplier: {modifiers.position_size_multiplier:.3f}")
            logger.info(f"  Confidence Multiplier: {modifiers.confidence_multiplier:.3f}")
            logger.info(f"  Risk Multiplier: {modifiers.risk_multiplier:.3f}")
            
            return agent_context
            
        except Exception as e:
            logger.error(f"Error in AgentContext integration demonstration: {e}")
            return None

    async def demonstrate_trading_cycle_integration(self):
        """Демонстрация интеграции в торговый цикл."""
        try:
            logger.info("Demonstrating DOASS integration in trading cycle...")
            
            # Симулируем торговый цикл
            for symbol in ["BTCUSDT", "ETHUSDT"]:
                logger.info(f"\nProcessing trading cycle for {symbol}...")
                
                # 1. Анализ DOASS
                doass_result = await self.doass.select_opportunity_symbols(
                    symbols=[symbol],
                    max_symbols=1,
                    min_opportunity_score=0.5
                )
                
                # 2. Создание контекста агента
                agent_context = AgentContext(symbol=symbol)
                agent_context.doass_result = doass_result
                
                # 3. Применение модификаторов
                agent_context.apply_doass_modifier()
                
                # 4. Симуляция торгового решения
                if doass_result.selected_symbols and symbol in doass_result.selected_symbols:
                    profile = doass_result.detailed_profiles.get(symbol)
                    if profile and profile.opportunity_score > 0.8:
                        logger.info(f"  HIGH OPPORTUNITY: {symbol} - Score: {profile.opportunity_score:.3f}")
                        logger.info(f"  Recommended action: Increase position size and aggressiveness")
                    elif profile and profile.opportunity_score > 0.6:
                        logger.info(f"  MODERATE OPPORTUNITY: {symbol} - Score: {profile.opportunity_score:.3f}")
                        logger.info(f"  Recommended action: Normal trading with moderate position size")
                    else:
                        logger.info(f"  LOW OPPORTUNITY: {symbol} - Score: {profile.opportunity_score:.3f}")
                        logger.info(f"  Recommended action: Reduce position size or skip")
                else:
                    logger.info(f"  NO OPPORTUNITY: {symbol} not selected by DOASS")
                    logger.info(f"  Recommended action: Skip trading")
                
                # 5. Показываем финальные модификаторы
                modifiers = agent_context.strategy_modifiers
                logger.info(f"  Final Strategy Modifiers:")
                logger.info(f"    Order Aggressiveness: {modifiers.order_aggressiveness:.3f}")
                logger.info(f"    Position Size: {modifiers.position_size_multiplier:.3f}")
                logger.info(f"    Confidence: {modifiers.confidence_multiplier:.3f}")
                logger.info(f"    Risk: {modifiers.risk_multiplier:.3f}")
            
        except Exception as e:
            logger.error(f"Error in trading cycle integration demonstration: {e}")

    async def demonstrate_performance_monitoring(self):
        """Демонстрация мониторинга производительности DOASS."""
        try:
            logger.info("Demonstrating DOASS performance monitoring...")
            
            # Выполняем несколько анализов для демонстрации производительности
            start_time = datetime.now()
            
            for i in range(3):
                logger.info(f"Performance test iteration {i+1}/3...")
                
                result = await self.doass.select_opportunity_symbols(
                    symbols=self.symbols,
                    max_symbols=5,
                    min_opportunity_score=0.6
                )
                
                logger.info(f"  Iteration {i+1} results:")
                logger.info(f"    Processing time: {result.processing_time_ms:.2f}ms")
                logger.info(f"    Cache hit rate: {result.cache_hit_rate:.2f}")
                logger.info(f"    Symbols selected: {len(result.selected_symbols)}")
                
                # Небольшая пауза между итерациями
                await asyncio.sleep(0.1)
            
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"\nPerformance Summary:")
            logger.info(f"  Total execution time: {total_time:.2f}s")
            logger.info(f"  Average per iteration: {total_time/3:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in performance monitoring demonstration: {e}")

    async def demonstrate_advanced_features(self):
        """Демонстрация продвинутых возможностей DOASS."""
        try:
            logger.info("Demonstrating DOASS advanced features...")
            
            # 1. Анализ с различными порогами
            logger.info("\n1. Analysis with different thresholds:")
            
            thresholds = [0.5, 0.7, 0.9]
            for threshold in thresholds:
                result = await self.doass.select_opportunity_symbols(
                    symbols=self.symbols,
                    max_symbols=10,
                    min_opportunity_score=threshold
                )
                logger.info(f"  Threshold {threshold}: {len(result.selected_symbols)} symbols selected")
            
            # 2. Анализ с различными максимальными количествами символов
            logger.info("\n2. Analysis with different max symbols:")
            
            max_symbols_list = [3, 5, 8]
            for max_symbols in max_symbols_list:
                result = await self.doass.select_opportunity_symbols(
                    symbols=self.symbols,
                    max_symbols=max_symbols,
                    min_opportunity_score=0.6
                )
                logger.info(f"  Max symbols {max_symbols}: {len(result.selected_symbols)} symbols selected")
            
            # 3. Показываем корреляционную матрицу
            logger.info("\n3. Correlation matrix analysis:")
            result = await self.doass.select_opportunity_symbols(
                symbols=self.symbols,
                max_symbols=5,
                min_opportunity_score=0.6
            )
            
            if hasattr(result, 'correlation_matrix') and not result.correlation_matrix.empty:
                logger.info(f"  Correlation matrix shape: {result.correlation_matrix.shape}")
                logger.info(f"  High correlations (>0.8): {len(result.correlation_matrix[result.correlation_matrix > 0.8])}")
            
            # 4. Показываем группы энтанглмента
            if hasattr(result, 'entanglement_groups') and result.entanglement_groups:
                logger.info(f"\n4. Entanglement groups:")
                for i, group in enumerate(result.entanglement_groups):
                    logger.info(f"  Group {i+1}: {group}")
            
        except Exception as e:
            logger.error(f"Error in advanced features demonstration: {e}")

    async def run_full_demonstration(self):
        """Запуск полной демонстрации интеграции DOASS."""
        try:
            logger.info("=" * 60)
            logger.info("DOASS INTEGRATION DEMONSTRATION")
            logger.info("=" * 60)
            
            # 1. Настройка системы
            if not await self.setup_system():
                logger.error("Failed to setup system")
                return
            
            # 2. Демонстрация выбора символов
            await self.demonstrate_symbol_selection()
            
            # 3. Демонстрация интеграции с AgentContext
            await self.demonstrate_agent_context_integration()
            
            # 4. Демонстрация интеграции в торговый цикл
            await self.demonstrate_trading_cycle_integration()
            
            # 5. Демонстрация мониторинга производительности
            await self.demonstrate_performance_monitoring()
            
            # 6. Демонстрация продвинутых возможностей
            await self.demonstrate_advanced_features()
            
            logger.info("\n" + "=" * 60)
            logger.info("DOASS INTEGRATION DEMONSTRATION COMPLETED")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error in full demonstration: {e}")


async def main():
    """Главная функция для запуска примера."""
    try:
        # Настройка логирования
        logging.basicConfig(level=logging.INFO)
        logger.info("Starting DOASS Integration Example")
        
        # Создание и запуск примера
        example = DOASSIntegrationExample()
        await example.run_full_demonstration()
        
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    # Запуск асинхронного примера
    asyncio.run(main()) 