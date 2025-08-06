#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Пример использования интеграции новостной аналитики в торговой системе.

Демонстрирует как новостная аналитика и анализ социальных медиа
интегрируются в принятие торговых решений.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

from application.di_container_refactored import get_service_locator
from application.use_cases.trading_orchestrator import (
    ExecuteStrategyRequest,
    ProcessSignalRequest,
    PortfolioRebalanceRequest
)
from domain.entities.strategy import Signal, SignalType
from domain.entities.trading import OrderSide, OrderType
from shared.config import get_config


class EnhancedTradingExample:
    """Пример использования расширенной торговой системы с новостной аналитикой."""

    def __init__(self) -> None:
        self.container = get_service_locator()
        self.orchestrator = self.container.get("trading_orchestrator_use_case")
        self.enhanced_trading_service = self.container.get("enhanced_trading_service")
        self.logger = logging.getLogger(__name__)

    async def run_comprehensive_example(self) -> None:
        """Запуск комплексного примера с новостной аналитикой."""
        try:
            self.logger.info("🚀 Запуск примера интеграции новостной аналитики")
            
            # 1. Анализ рыночного сентимента
            await self._demonstrate_sentiment_analysis()
            
            # 2. Выполнение стратегии с учетом сентимента
            await self._demonstrate_strategy_execution()
            
            # 3. Обработка сигналов с новостной аналитикой
            await self._demonstrate_signal_processing()
            
            # 4. Ребалансировка портфеля с учетом сентимента
            await self._demonstrate_portfolio_rebalancing()
            
            # 5. Мониторинг и отчетность
            await self._demonstrate_monitoring()
            
            self.logger.info("✅ Пример успешно завершен")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в примере: {e}")
            raise

    async def _demonstrate_sentiment_analysis(self) -> None:
        """Демонстрация анализа рыночного сентимента."""
        self.logger.info("📊 Анализ рыночного сентимента...")
        
        trading_pairs = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        
        for pair in trading_pairs:
            try:
                # Получаем анализ сентимента
                sentiment_analysis = await self.enhanced_trading_service.get_market_sentiment_analysis(
                    trading_pair=pair,
                    include_news=True,
                    include_social=True
                )
                
                if sentiment_analysis.get('error'):
                    self.logger.warning(f"⚠️ Ошибка анализа для {pair}: {sentiment_analysis['error']}")
                    continue
                
                # Выводим результаты
                self.logger.info(f"📈 {pair}:")
                self.logger.info(f"   Общий сентимент: {sentiment_analysis.get('overall_sentiment', 0):.3f}")
                self.logger.info(f"   Индекс страха/жадности: {sentiment_analysis.get('fear_greed_index', 50):.1f}")
                self.logger.info(f"   Новостной сентимент: {sentiment_analysis.get('news_sentiment', 0):.3f}")
                self.logger.info(f"   Социальный сентимент: {sentiment_analysis.get('social_sentiment', 0):.3f}")
                self.logger.info(f"   Количество новостей: {sentiment_analysis.get('news_count', 0)}")
                self.logger.info(f"   Количество постов: {sentiment_analysis.get('social_posts_count', 0)}")
                
                # Показываем рекомендации
                recommendations = sentiment_analysis.get('recommendations', [])
                if recommendations:
                    self.logger.info(f"   💡 Рекомендации:")
                    for rec in recommendations:
                        self.logger.info(f"      - {rec}")
                
                # Показываем трендовые темы
                trending_topics = sentiment_analysis.get('trending_topics', [])
                if trending_topics:
                    self.logger.info(f"   🔥 Трендовые темы: {', '.join(trending_topics[:5])}")
                
                self.logger.info("")
                
            except Exception as e:
                self.logger.error(f"❌ Ошибка анализа сентимента для {pair}: {e}")

    async def _demonstrate_strategy_execution(self) -> None:
        """Демонстрация выполнения стратегии с учетом сентимента."""
        self.logger.info("🎯 Выполнение стратегии с новостной аналитикой...")
        
        # Создаем запрос на выполнение стратегии
        request = ExecuteStrategyRequest(
            strategy_id="sentiment_aware_strategy",
            portfolio_id="demo_portfolio",
            symbol="BTC/USDT",
            amount=Decimal("0.001"),
            risk_level="moderate",
            use_sentiment_analysis=True  # Включаем анализ сентимента
        )
        
        try:
            # Выполняем стратегию
            response = await self.orchestrator.execute_strategy(request)
            
            self.logger.info(f"📊 Результат выполнения стратегии:")
            self.logger.info(f"   Выполнено: {response.executed}")
            self.logger.info(f"   Создано ордеров: {len(response.orders_created)}")
            self.logger.info(f"   Сгенерировано сигналов: {len(response.signals_generated)}")
            self.logger.info(f"   Сообщение: {response.message}")
            
            # Показываем детали анализа сентимента
            if response.sentiment_analysis:
                sentiment = response.sentiment_analysis
                self.logger.info(f"   📈 Анализ сентимента:")
                self.logger.info(f"      Общий сентимент: {sentiment.get('overall_sentiment', 0):.3f}")
                self.logger.info(f"      Индекс страха/жадности: {sentiment.get('fear_greed_index', 50):.1f}")
                self.logger.info(f"      Уверенность: {sentiment.get('confidence', 0):.3f}")
            
            # Показываем созданные ордера
            for i, order in enumerate(response.orders_created):
                self.logger.info(f"   📋 Ордер {i+1}:")
                self.logger.info(f"      ID: {order.id}")
                self.logger.info(f"      Пара: {order.symbol}")
                self.logger.info(f"      Сторона: {order.side}")
                self.logger.info(f"      Размер: {order.amount}")
                self.logger.info(f"      Статус: {order.status}")
                
                # Показываем метаданные сентимента
                if hasattr(order, 'metadata') and order.metadata:
                    sentiment_meta = order.metadata.get('sentiment_score')
                    if sentiment_meta is not None:
                        self.logger.info(f"      Сентимент: {sentiment_meta:.3f}")
                
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка выполнения стратегии: {e}")

    async def _demonstrate_signal_processing(self) -> None:
        """Демонстрация обработки сигналов с новостной аналитикой."""
        self.logger.info("📡 Обработка торговых сигналов с новостной аналитикой...")
        
        # Создаем тестовый сигнал
        test_signal = Signal(
            signal_type=SignalType.BUY,
            symbol="ETH/USDT",
            strength=0.7,
            confidence=0.8,
            timestamp=datetime.now(),
            metadata={
                'technical_sentiment': 0.6,
                'market_volatility': 0.3,
                'source': 'technical_analysis'
            }
        )
        
        # Создаем запрос на обработку сигнала
        request = ProcessSignalRequest(
            signal=test_signal,
            portfolio_id="demo_portfolio",
            auto_execute=True,
            use_sentiment_analysis=True  # Включаем анализ сентимента
        )
        
        try:
            # Обрабатываем сигнал
            response = await self.orchestrator.process_signal(request)
            
            self.logger.info(f"📊 Результат обработки сигнала:")
            self.logger.info(f"   Обработан: {response.processed}")
            self.logger.info(f"   Создано ордеров: {len(response.orders_created)}")
            self.logger.info(f"   Сообщение: {response.message}")
            
            # Показываем детали анализа сентимента
            if response.sentiment_analysis:
                sentiment = response.sentiment_analysis
                self.logger.info(f"   📈 Анализ сентимента для сигнала:")
                self.logger.info(f"      Общий сентимент: {sentiment.get('overall_sentiment', 0):.3f}")
                self.logger.info(f"      Индекс страха/жадности: {sentiment.get('fear_greed_index', 50):.1f}")
                
                # Показываем рекомендации
                recommendations = sentiment.get('recommendations', [])
                if recommendations:
                    self.logger.info(f"      💡 Рекомендации:")
                    for rec in recommendations:
                        self.logger.info(f"         - {rec}")
            
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки сигнала: {e}")

    async def _demonstrate_portfolio_rebalancing(self) -> None:
        """Демонстрация ребалансировки портфеля с учетом сентимента."""
        self.logger.info("⚖️ Ребалансировка портфеля с новостной аналитикой...")
        
        # Создаем целевые веса портфеля
        target_weights = {
            "BTC/USDT": Decimal("0.5"),
            "ETH/USDT": Decimal("0.3"),
            "ADA/USDT": Decimal("0.2")
        }
        
        # Создаем запрос на ребалансировку
        request = PortfolioRebalanceRequest(
            portfolio_id="demo_portfolio",
            target_weights=target_weights,
            tolerance=Decimal("0.05"),  # 5% толерантность
            use_sentiment_analysis=True  # Включаем анализ сентимента
        )
        
        try:
            # Выполняем ребалансировку
            response = await self.orchestrator.rebalance_portfolio(request)
            
            self.logger.info(f"📊 Результат ребалансировки портфеля:")
            self.logger.info(f"   Ребалансирован: {response.rebalanced}")
            self.logger.info(f"   Создано ордеров: {len(response.orders_created)}")
            self.logger.info(f"   Сообщение: {response.message}")
            
            # Показываем текущие и целевые веса
            self.logger.info(f"   📈 Веса портфеля:")
            for symbol in target_weights.keys():
                current_weight = response.current_weights.get(symbol, Decimal("0"))
                target_weight = response.target_weights.get(symbol, Decimal("0"))
                self.logger.info(f"      {symbol}: {current_weight:.3f} → {target_weight:.3f}")
            
            # Показываем анализ сентимента для каждой пары
            if response.sentiment_analysis:
                self.logger.info(f"   📊 Анализ сентимента по парам:")
                for symbol, sentiment in response.sentiment_analysis.items():
                    if not sentiment.get('error'):
                        overall_sentiment = sentiment.get('overall_sentiment', 0)
                        fear_greed = sentiment.get('fear_greed_index', 50)
                        self.logger.info(f"      {symbol}: сентимент={overall_sentiment:.3f}, F/G={fear_greed:.1f}")
            
            self.logger.info("")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка ребалансировки портфеля: {e}")

    async def _demonstrate_monitoring(self) -> None:
        """Демонстрация мониторинга и отчетности."""
        self.logger.info("📊 Мониторинг и отчетность...")
        
        # Получаем активные сессии
        active_sessions = list(self.orchestrator.active_sessions.values())
        
        self.logger.info(f"   Активных сессий: {len(active_sessions)}")
        
        for session in active_sessions:
            self.logger.info(f"   📋 Сессия {session.session_id}:")
            self.logger.info(f"      Портфель: {session.portfolio_id}")
            self.logger.info(f"      Статус: {session.status}")
            self.logger.info(f"      Начало: {session.start_time}")
            self.logger.info(f"      Создано ордеров: {len(session.orders_created)}")
            self.logger.info(f"      Выполнено сделок: {len(session.trades_executed)}")
            self.logger.info(f"      P&L: {session.pnl}")
        
        # Получаем статистику по ордерам
        try:
            active_orders = await self.orchestrator.get_active_orders()
            self.logger.info(f"   📋 Активных ордеров: {len(active_orders)}")
            
            for order in active_orders:
                self.logger.info(f"      {order.symbol}: {order.side} {order.amount} @ {order.status}")
                
                # Показываем метаданные сентимента если есть
                if hasattr(order, 'metadata') and order.metadata:
                    sentiment_score = order.metadata.get('sentiment_score')
                    if sentiment_score is not None:
                        self.logger.info(f"         Сентимент: {sentiment_score:.3f}")
        
        except Exception as e:
            self.logger.warning(f"⚠️ Не удалось получить активные ордера: {e}")
        
        self.logger.info("")

    async def run_simple_example(self) -> None:
        """Запуск простого примера для быстрого тестирования."""
        self.logger.info("🚀 Запуск простого примера...")
        
        try:
            # Простой анализ сентимента
            sentiment = await self.enhanced_trading_service.get_market_sentiment_analysis(
                trading_pair="BTC/USDT",
                include_news=True,
                include_social=True
            )
            
            self.logger.info(f"📊 Анализ сентимента BTC/USDT:")
            self.logger.info(f"   Общий сентимент: {sentiment.get('overall_sentiment', 0):.3f}")
            self.logger.info(f"   Индекс страха/жадности: {sentiment.get('fear_greed_index', 50):.1f}")
            
            if sentiment.get('recommendations'):
                self.logger.info(f"   Рекомендации: {sentiment['recommendations'][0]}")
            
            self.logger.info("✅ Простой пример завершен")
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка в простом примере: {e}")


async def main() -> None:
    """Основная функция."""
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Создание и запуск примера
    example = EnhancedTradingExample()
    
    # Запуск простого примера для быстрого тестирования
    await example.run_simple_example()
    
    # Раскомментируйте для запуска полного примера
    # await example.run_comprehensive_example()


if __name__ == "__main__":
    asyncio.run(main()) 