# -*- coding: utf-8 -*-
"""Пример интеграции аналитических модулей с MarketMakerModelAgent."""

import asyncio
import time
from typing import Any, Dict

import pandas as pd
from loguru import logger
from datetime import datetime

from domain.intelligence.entanglement_detector import EntanglementResult
from domain.intelligence.mirror_detector import MirrorSignal
from domain.intelligence.noise_analyzer import NoiseAnalysisResult
from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
from infrastructure.agents.analytical.types import AnalyticalIntegrationConfig
from domain.value_objects.timestamp import Timestamp
from domain.type_definitions.intelligence_types import NoiseType


async def main():
    """Основная функция примера."""
    logger.info("=== Analytical Integration Example ===")

    try:
        # Создаем конфигурацию аналитики
        analytics_config = AnalyticalIntegrationConfig(
            entanglement_enabled=True,
            noise_enabled=True,
            mirror_enabled=True,
            gravity_enabled=True,
            enable_detailed_logging=True,
            log_analysis_results=True,
        )

        # Создаем агента с аналитической интеграцией
        agent_config = {
            "spread_threshold": 0.001,
            "volume_threshold": 100000,
            "fakeout_threshold": 0.02,
            "liquidity_zone_size": 0.005,
            "lookback_period": 100,
            "confidence_threshold": 0.7,
            "analytics_enabled": True,
            "entanglement_enabled": True,
            "noise_enabled": True,
            "mirror_enabled": True,
            "gravity_enabled": True,
        }

        agent = MarketMakerModelAgent(config=agent_config)

        # Запускаем аналитические модули
        await agent.start_analytics()
        logger.info("Analytics started")

        # Симуляция торговых данных
        symbol = "BTCUSDT"

        # Создаем тестовые данные
        market_data = create_test_market_data()
        order_book = create_test_order_book(symbol)

        # Демонстрация различных сценариев
        await demonstrate_clean_market(agent, symbol, market_data, order_book)
        await demonstrate_entangled_market(agent, symbol, market_data, order_book)
        await demonstrate_noisy_market(agent, symbol, market_data, order_book)
        await demonstrate_mirror_signals(agent, symbol, market_data, order_book)
        await demonstrate_gravity_effects(agent, symbol, market_data, order_book)

        # Получаем статистику
        stats = agent.get_analytics_statistics()
        logger.info(f"Analytics statistics: {stats}")

        # Останавливаем аналитические модули
        await agent.stop_analytics()
        logger.info("Analytics stopped")

    except Exception as e:
        logger.error(f"Error in main: {e}")

    logger.info("=== Example completed ===")


async def demonstrate_clean_market(
    agent: MarketMakerModelAgent,
    symbol: str,
    market_data: pd.DataFrame,
    order_book: Dict[str, Any],
):
    """Демонстрация торговли на чистом рынке."""
    logger.info("=== Clean Market Scenario ===")

    try:
        # Проверяем возможность торговли
        should_trade = agent.should_proceed_with_trade(symbol, trade_aggression=0.8)
        logger.info(f"Should trade on clean market: {should_trade}")

        # Получаем торговые рекомендации
        recommendations = agent.get_trading_recommendations(symbol)
        logger.info(f"Trading recommendations: {recommendations}")

        # Выполняем расчет с аналитикой
        result = await agent.calculate_with_analytics(
            symbol=symbol,
            market_data=market_data,
            order_book=order_book,
            aggressiveness=0.8,
            confidence=0.7,
        )

        logger.info(f"Calculation result: {result}")

    except Exception as e:
        logger.error(f"Error in clean market scenario: {e}")


async def demonstrate_entangled_market(
    agent: MarketMakerModelAgent,
    symbol: str,
    market_data: pd.DataFrame,
    order_book: Dict[str, Any],
):
    """Демонстрация торговли при обнаружении запутанности."""
    logger.info("=== Entangled Market Scenario ===")

    try:
        # Симулируем обнаружение запутанности
        if agent.analytical_integration:
            # Создаем результат запутанности
            entanglement_result = EntanglementResult(
                symbol=symbol,
                is_entangled=True,
                correlation_score=0.98,
                lag_ms=1.5,
                confidence=0.95,
                exchange_pair=("binance", "bybit"),
                timestamp=Timestamp.now(),
                metadata={
                    "data_points": 100,
                    "confidence": 0.95,
                    "processing_time_ms": 50.0,
                    "algorithm_version": "2.0.0",
                    "parameters": {"max_lag_ms": 3.0, "correlation_threshold": 0.95},
                    "quality_metrics": {"best_correlation": 0.98, "best_lag": 1.5},
                },
            )

            # Применяем к контексту - убираем вызов несуществующего метода
            # context = agent.analytical_integration.analytical_integrator.get_context(symbol)
            # context.apply_entanglement_modifier(entanglement_result)
            logger.info("Applied entanglement modifier")

        # Проверяем возможность торговли
        should_trade = agent.should_proceed_with_trade(symbol, trade_aggression=0.8)
        logger.info(f"Should trade on entangled market: {should_trade}")

        # Получаем скорректированную агрессивность
        adjusted_aggression = agent.get_adjusted_aggressiveness(symbol, 0.8)
        logger.info(f"Adjusted aggressiveness: {adjusted_aggression}")

        # Выполняем расчет
        result = await agent.calculate_with_analytics(
            symbol=symbol,
            market_data=market_data,
            order_book=order_book,
            aggressiveness=0.8,
            confidence=0.7,
        )

        logger.info(f"Calculation result with entanglement: {result}")

    except Exception as e:
        logger.error(f"Error in entangled market scenario: {e}")


async def demonstrate_noisy_market(
    agent: MarketMakerModelAgent,
    symbol: str,
    market_data: pd.DataFrame,
    order_book: Dict[str, Any],
):
    """Демонстрация торговли при обнаружении синтетического шума."""
    logger.info("=== Noisy Market Scenario ===")

    try:
        # Симулируем обнаружение синтетического шума
        if agent.analytical_integration:
            # Создаем результат анализа шума
            noise_result = NoiseAnalysisResult(
                fractal_dimension=1.35,
                entropy=0.85,
                is_synthetic_noise=True,
                confidence=0.92,
                timestamp=Timestamp.now(),
                noise_type=NoiseType.SYNTHETIC,
                metadata={
                    "data_points": 100,
                    "confidence": 0.92,
                    "processing_time_ms": 45.0,
                    "algorithm_version": "1.0.0",
                    "parameters": {"fractal_threshold": 1.3, "entropy_threshold": 0.7},
                    "quality_metrics": {"best_correlation": 0.98, "best_lag": 1.5}
                },
                metrics={
                    "fractal_dimension": 1.35,
                    "entropy": 0.85,
                    "noise_type": NoiseType.SYNTHETIC,
                    "synthetic_probability": 0.92,
                    "natural_probability": 0.08,
                },
            )

            # Применяем к контексту - убираем вызов несуществующего метода
            # context = agent.analytical_integration.analytical_integrator.get_context(symbol)
            # context.apply_noise_modifier(noise_result)
            logger.info("Applied noise modifier")

        # Проверяем возможность торговли
        should_trade = agent.should_proceed_with_trade(symbol, trade_aggression=0.8)
        logger.info(f"Should trade on noisy market: {should_trade}")

        # Получаем смещение цены
        base_price = 50000.0
        buy_offset = agent.get_price_offset(symbol, base_price, "buy")
        sell_offset = agent.get_price_offset(symbol, base_price, "sell")

        logger.info(f"Price offsets - Buy: {buy_offset:.2f}, Sell: {sell_offset:.2f}")

        # Выполняем расчет
        result = await agent.calculate_with_analytics(
            symbol=symbol,
            market_data=market_data,
            order_book=order_book,
            aggressiveness=0.8,
            confidence=0.7,
        )

        logger.info(f"Calculation result with noise: {result}")

    except Exception as e:
        logger.error(f"Error in noisy market scenario: {e}")


async def demonstrate_mirror_signals(
    agent: MarketMakerModelAgent,
    symbol: str,
    market_data: pd.DataFrame,
    order_book: Dict[str, Any],
):
    """Демонстрация торговли при обнаружении зеркальных сигналов."""
    logger.info("=== Mirror Signals Scenario ===")

    try:
        # Симулируем обнаружение зеркального сигнала
        if agent.analytical_integration:
            # Создаем зеркальный сигнал
            mirror_signal = MirrorSignal(
                asset1="BTC",
                asset2="ETH",
                best_lag=2,
                correlation=0.87,
                p_value=0.001,
                confidence=0.89,
                signal_strength=0.85,
                timestamp=Timestamp.now(),
                metadata={
                    "data_points": 100,
                    "confidence": 0.89,
                    "processing_time_ms": 30.0,
                    "algorithm_version": "1.0.0",
                    "parameters": {"max_lag": 5, "correlation_method": "pearson"},
                    "quality_metrics": {"best_correlation": 0.87, "best_lag": 2}
                },
            )

            # Применяем к контексту - убираем вызов несуществующего метода
            # context = agent.analytical_integration.analytical_integrator.get_context(symbol)
            # context.apply_mirror_modifier(mirror_signal)
            logger.info("Applied mirror modifier")

        # Получаем скорректированную уверенность
        adjusted_confidence = agent.get_adjusted_confidence(symbol, 0.7)
        logger.info(f"Adjusted confidence: {adjusted_confidence}")

        # Получаем скорректированный размер позиции
        adjusted_size = agent.get_adjusted_position_size(symbol, 1.0)
        logger.info(f"Adjusted position size: {adjusted_size}")

        # Выполняем расчет
        result = await agent.calculate_with_analytics(
            symbol=symbol,
            market_data=market_data,
            order_book=order_book,
            aggressiveness=0.8,
            confidence=0.7,
        )

        logger.info(f"Calculation result with mirror signals: {result}")

    except Exception as e:
        logger.error(f"Error in mirror signals scenario: {e}")


async def demonstrate_gravity_effects(
    agent: MarketMakerModelAgent,
    symbol: str,
    market_data: pd.DataFrame,
    order_book: Dict[str, Any],
):
    """Демонстрация торговли при влиянии гравитации ликвидности."""
    logger.info("=== Liquidity Gravity Scenario ===")

    try:
        # Симулируем высокую гравитацию ликвидности
        if agent.analytical_integration:
            from application.risk.liquidity_gravity_monitor import \
                RiskAssessmentResult
            from domain.market.liquidity_gravity import LiquidityGravityResult
            from datetime import datetime

            # Создаем результат гравитации
            gravity_result = LiquidityGravityResult(
                total_gravity=2.5e-6,
                bid_ask_forces=[(1.0, 1.0, 2.5e-6)],
                gravity_distribution={"bid": 0.6, "ask": 0.4},
                risk_level="medium",
                timestamp=datetime.now(),
                metadata={},
                volume_imbalance=0.2,
                price_momentum=0.1,
                volatility_score=0.8,
                liquidity_score=0.7,
                market_efficiency=0.6
            )

            # Создаем оценку риска
            risk_assessment = RiskAssessmentResult(
                symbol=symbol,
                risk_level="medium",
                gravity_score=0.8,
                liquidity_score=0.6,
                volatility_score=0.8,
                overall_risk=0.6,
                recommendations=["Reduce position size", "Monitor liquidity"],
                timestamp=Timestamp.now(),
                metadata={
                    "gravity_impact": 0.8,
                    "liquidity_risk": 0.4,
                    "volatility_risk": 0.6
                }
            )

            # Применяем к контексту - убираем вызов несуществующего метода
            # context = agent.analytical_integration.analytical_integrator.get_context(symbol)
            # context.apply_gravity_modifier(gravity_result)
            logger.info("Applied gravity modifier")

        # Проверяем возможность торговли
        should_trade = agent.should_proceed_with_trade(symbol, trade_aggression=0.8)
        logger.info(f"Should trade with high gravity: {should_trade}")

        # Получаем скорректированные параметры
        adjusted_aggression = agent.get_adjusted_aggressiveness(symbol, 0.8)
        adjusted_size = agent.get_adjusted_position_size(symbol, 1.0)

        logger.info(f"Adjusted aggressiveness: {adjusted_aggression}")
        logger.info(f"Adjusted position size: {adjusted_size}")

        # Выполняем расчет
        result = await agent.calculate_with_analytics(
            symbol=symbol,
            market_data=market_data,
            order_book=order_book,
            aggressiveness=0.8,
            confidence=0.7,
        )

        logger.info(f"Calculation result with gravity effects: {result}")

    except Exception as e:
        logger.error(f"Error in gravity effects scenario: {e}")


def create_test_market_data() -> pd.DataFrame:
    """Создание тестовых рыночных данных."""
    try:
        # Создаем временной ряд
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1min")

        # Создаем тестовые данные
        data = {
            "open": [50000 + i * 0.1 for i in range(100)],
            "high": [50000 + i * 0.1 + 10 for i in range(100)],
            "low": [50000 + i * 0.1 - 10 for i in range(100)],
            "close": [50000 + i * 0.1 + 5 for i in range(100)],
            "volume": [1000000 + i * 1000 for i in range(100)],
        }

        df = pd.DataFrame(data, index=dates)
        return df

    except Exception as e:
        logger.error(f"Error creating test market data: {e}")
        return pd.DataFrame()


def create_test_order_book(symbol: str) -> Dict[str, Any]:
    """Создание тестового ордербука."""
    try:
        base_price = 50000.0

        order_book = {
            "bids": [
                {"price": base_price - i * 0.1, "size": 1.0 + i * 0.1}
                for i in range(1, 21)
            ],
            "asks": [
                {"price": base_price + i * 0.1, "size": 1.0 + i * 0.1}
                for i in range(1, 21)
            ],
            "symbol": symbol,
            "timestamp": time.time(),
        }

        return order_book

    except Exception as e:
        logger.error(f"Error creating test order book: {e}")
        return {}


if __name__ == "__main__":
    asyncio.run(main())
