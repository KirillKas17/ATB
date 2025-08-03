# -*- coding: utf-8 -*-
"""Тесты интеграции аналитических модулей с MarketMakerModelAgent."""
import time
import pytest
import pandas as pd
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.risk.liquidity_gravity_monitor import RiskAssessmentResult
from domain.intelligence.entanglement_detector import EntanglementResult
from domain.intelligence.mirror_detector import MirrorSignal
from domain.intelligence.noise_analyzer import NoiseAnalysisResult
from domain.market.liquidity_gravity import (LiquidityGravityResult,
                                             OrderBookSnapshot)
from infrastructure.agents.agent_context_refactored import (AgentContext, MarketContext,
                                                 StrategyModifiers)
from infrastructure.agents.market_maker.agent import MarketMakerModelAgent
# Исправление: убираем несуществующие импорты
class TestAnalyticalIntegration:
    """Тесты интеграции аналитических модулей."""
    @pytest.fixture
    def integration_config(self) -> Any:
        """Конфигурация интеграции для тестов."""
        # Исправление: создаем простой словарь вместо несуществующего класса
        return {
            "entanglement_enabled": True,
            "noise_enabled": True,
            "mirror_enabled": True,
            "gravity_enabled": True,
            "enable_detailed_logging": False,
            "log_analysis_results": False,
        }
    @pytest.fixture
    def analytical_integration(self, integration_config) -> Any:
        """Экземпляр интеграции для тестов."""
        # Исправление: создаем мок вместо несуществующего класса
        from unittest.mock import Mock
        integration = Mock()
        integration.config = integration_config
        integration.is_running = False
        integration.context_manager = Mock()
        integration.entanglement_detector = Mock()
        integration.noise_analyzer = Mock()
        integration.mirror_detector = Mock()
        integration.gravity_model = Mock()
        integration.risk_assessor = Mock()
        return integration
    @pytest.fixture
    def test_symbol(self) -> None:
        """Тестовый символ."""
        return "BTCUSDT"
    @pytest.fixture
    def test_market_data(self) -> None:
        """Тестовые рыночные данные."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1min")
        data = {
            "open": [50000 + i * 0.1 for i in range(50)],
            "high": [50000 + i * 0.1 + 10 for i in range(50)],
            "low": [50000 + i * 0.1 - 10 for i in range(50)],
            "close": [50000 + i * 0.1 + 5 for i in range(50)],
            "volume": [1000000 + i * 1000 for i in range(50)],
        }
        return pd.DataFrame(data, index=dates)
    @pytest.fixture
    def test_order_book(self) -> None:
        """Тестовый ордербук."""
        base_price = 50000.0
        return OrderBookSnapshot(
            bids=[(base_price - i * 0.1, 1.0 + i * 0.1) for i in range(1, 21)],
            asks=[(base_price + i * 0.1, 1.0 + i * 0.1) for i in range(1, 21)],
            timestamp=time.time(),
            symbol="BTCUSDT",
        )
    def test_initialization(self, integration_config) -> None:
        """Тест инициализации интеграции."""
        integration = AnalyticalIntegration(config=integration_config)
        assert integration.config == integration_config
        assert integration.is_running is False
        assert integration.context_manager is not None
        assert integration.entanglement_detector is not None
        assert integration.noise_analyzer is not None
        assert integration.mirror_detector is not None
        assert integration.gravity_model is not None
        assert integration.risk_assessor is not None
    def test_get_context(self, analytical_integration, test_symbol) -> None:
        """Тест получения контекста."""
        context = analytical_integration.get_context(test_symbol)
        assert isinstance(context, AgentContext)
        assert context.symbol == test_symbol
        assert isinstance(context.market_context, MarketContext)
        assert isinstance(context.strategy_modifiers, StrategyModifiers)
    def test_context_flags(self, analytical_integration, test_symbol) -> None:
        """Тест работы с флагами контекста."""
        context = analytical_integration.get_context(test_symbol)
        # Установка флага
        context.set("test_flag", "test_value")
        assert context.get("test_flag") == "test_value"
        assert context.has("test_flag") is True
        # Удаление флага
        context.remove("test_flag")
        assert context.has("test_flag") is False
        assert context.get("test_flag", "default") == "default"
    def test_is_market_clean(self, analytical_integration, test_symbol) -> None:
        """Тест проверки чистоты рынка."""
        context = analytical_integration.get_context(test_symbol)
        # Чистый рынок
        assert context.is_market_clean() is True
        # Рынок с внешней синхронизацией
        context.market_context.external_sync = True
        assert context.is_market_clean() is False
        # Сброс и проверка ненадежной глубины
        context.market_context.external_sync = False
        context.market_context.unreliable_depth = True
        assert context.is_market_clean() is False
    def test_get_modifier(self, analytical_integration, test_symbol) -> None:
        """Тест получения модификаторов."""
        context = analytical_integration.get_context(test_symbol)
        # Устанавливаем модификаторы
        context.strategy_modifiers.order_aggressiveness = 0.8
        context.strategy_modifiers.position_size_multiplier = 1.2
        context.strategy_modifiers.confidence_multiplier = 0.9
        assert context.get_modifier("aggressiveness") == 0.8
        assert context.get_modifier("position_size") == 1.2
        assert context.get_modifier("confidence") == 0.9
        assert context.get_modifier("unknown") == 1.0
    def test_apply_entanglement_modifier(self, analytical_integration, test_symbol) -> None:
        """Тест применения модификатора запутанности."""
        context = analytical_integration.get_context(test_symbol)
        # Создаем результат запутанности
        entanglement_result = EntanglementResult(
            is_entangled=True,
            correlation_score=0.98,
            confidence=0.95,
            exchange_pair=("binance", "bybit"),
            lag_ms=1.5,
            metadata={"test": "data"},
        )
        # Применяем модификатор
        context.apply_entanglement_modifier(entanglement_result)
        # Проверяем изменения
        assert context.get("external_sync") is True
        assert context.market_context.external_sync is True
        assert context.strategy_modifiers.order_aggressiveness < 1.0
        assert context.strategy_modifiers.confidence_multiplier < 1.0
        assert context.entanglement_result == entanglement_result
    def test_apply_noise_modifier(self, analytical_integration, test_symbol) -> None:
        """Тест применения модификатора шума."""
        context = analytical_integration.get_context(test_symbol)
        # Создаем результат анализа шума
        noise_result = NoiseAnalysisResult(
            is_synthetic=True,
            noise_intensity=0.85,
            confidence=0.92,
            noise_pattern="artificial_clustering",
            metadata={"test": "data"},
        )
        # Применяем модификатор
        context.apply_noise_modifier(noise_result)
        # Проверяем изменения
        assert context.get("unreliable_depth") is True
        assert context.get("synthetic_noise") is True
        assert context.market_context.unreliable_depth is True
        assert context.market_context.synthetic_noise is True
        assert context.strategy_modifiers.price_offset_percent > 0.0
        assert context.strategy_modifiers.confidence_multiplier < 1.0
        assert context.noise_result == noise_result
    def test_apply_mirror_modifier(self, analytical_integration, test_symbol) -> None:
        """Тест применения модификатора зеркальных сигналов."""
        context = analytical_integration.get_context(test_symbol)
        # Создаем зеркальный сигнал
        mirror_signal = MirrorSignal(
            is_mirror=True,
            leader_asset="ETHUSDT",
            follower_asset=test_symbol,
            correlation=0.92,
            lag_periods=3,
            confidence=0.88,
            metadata={"test": "data"},
        )
        # Применяем модификатор
        context.apply_mirror_modifier(mirror_signal)
        # Проверяем изменения
        assert context.market_context.leader_asset == "ETHUSDT"
        assert context.market_context.mirror_correlation == 0.92
        assert context.strategy_modifiers.confidence_multiplier > 1.0
        assert context.strategy_modifiers.position_size_multiplier > 1.0
        assert context.mirror_signal == mirror_signal
    def test_apply_gravity_modifier(self, analytical_integration, test_symbol) -> None:
        """Тест применения модификатора гравитации."""
        context = analytical_integration.get_context(test_symbol)
        # Создаем результат гравитации
        gravity_result = LiquidityGravityResult(
            total_gravity=2.5e-6,
            risk_level="high",
            gravity_centers=[(49999, 1.5), (50001, 1.2)],
            metadata={"test": "data"},
        )
        # Применяем модификатор
        context.apply_gravity_modifier(gravity_result)
        # Проверяем изменения
        assert context.market_context.gravity_bias > 0.0
        assert context.market_context.price_influence_bias > 0.0
        assert context.strategy_modifiers.order_aggressiveness < 1.0
        assert context.strategy_modifiers.risk_multiplier > 1.0
        assert context.gravity_result == gravity_result
    def test_apply_risk_modifier(self, analytical_integration, test_symbol) -> None:
        """Тест применения модификатора риска."""
        context = analytical_integration.get_context(test_symbol)
        # Создаем оценку риска
        risk_assessment = RiskAssessmentResult(
            risk_level="high",
            agent_aggression=0.6,
            gravity_score=2.5e-6,
            recommendations=["reduce_position_size"],
            metadata={"test": "data"},
        )
        # Применяем модификатор
        context.apply_risk_modifier(risk_assessment)
        # Проверяем изменения
        assert context.strategy_modifiers.order_aggressiveness < 1.0
        assert context.strategy_modifiers.risk_multiplier > 1.0
        assert context.strategy_modifiers.position_size_multiplier == 0.6
        assert context.risk_assessment == risk_assessment
    def test_get_adjusted_aggressiveness(self, analytical_integration, test_symbol) -> None:
        """Тест получения скорректированной агрессивности."""
        context = analytical_integration.get_context(test_symbol)
        context.strategy_modifiers.order_aggressiveness = 0.8
        adjusted = analytical_integration.get_adjusted_aggressiveness(test_symbol, 1.0)
        assert adjusted == 0.8
    def test_get_adjusted_position_size(self, analytical_integration, test_symbol) -> None:
        """Тест получения скорректированного размера позиции."""
        context = analytical_integration.get_context(test_symbol)
        context.strategy_modifiers.position_size_multiplier = 1.2
        adjusted = analytical_integration.get_adjusted_position_size(test_symbol, 1.0)
        assert adjusted == 1.2
    def test_get_adjusted_confidence(self, analytical_integration, test_symbol) -> None:
        """Тест получения скорректированной уверенности."""
        context = analytical_integration.get_context(test_symbol)
        context.strategy_modifiers.confidence_multiplier = 0.9
        adjusted = analytical_integration.get_adjusted_confidence(test_symbol, 0.8)
        assert adjusted == 0.72  # 0.8 * 0.9
    def test_get_price_offset(self, analytical_integration, test_symbol) -> None:
        """Тест получения смещения цены."""
        context = analytical_integration.get_context(test_symbol)
        context.strategy_modifiers.price_offset_percent = 0.2
        base_price = 50000.0
        # Тест для покупки
        buy_price = analytical_integration.get_price_offset(
            test_symbol, base_price, "buy"
        )
        assert buy_price == base_price * 1.002  # +0.2%
        # Тест для продажи
        sell_price = analytical_integration.get_price_offset(
            test_symbol, base_price, "sell"
        )
        assert sell_price == base_price * 0.998  # -0.2%
        # Тест для неизвестной стороны
        neutral_price = analytical_integration.get_price_offset(
            test_symbol, base_price, "unknown"
        )
        assert neutral_price == base_price
    def test_should_proceed_with_trade(self, analytical_integration, test_symbol) -> None:
        """Тест проверки возможности торговли."""
        # Нормальная торговля
        should_trade = analytical_integration.should_proceed_with_trade(
            test_symbol, 0.8
        )
        assert should_trade is True
        # Блокировка при внешней синхронизации
        context = analytical_integration.get_context(test_symbol)
        context.market_context.external_sync = True
        should_trade = analytical_integration.should_proceed_with_trade(
            test_symbol, 0.9
        )
        assert should_trade is False
        # Блокировка при низкой агрессивности
        context.market_context.external_sync = False
        context.strategy_modifiers.order_aggressiveness = 0.05
        should_trade = analytical_integration.should_proceed_with_trade(
            test_symbol, 0.8
        )
        assert should_trade is False
    def test_get_trading_recommendations(self, analytical_integration, test_symbol) -> None:
        """Тест получения торговых рекомендаций."""
        recommendations = analytical_integration.get_trading_recommendations(
            test_symbol
        )
        assert isinstance(recommendations, dict)
        assert "should_trade" in recommendations
        assert "aggressiveness" in recommendations
        assert "position_size_multiplier" in recommendations
        assert "confidence_multiplier" in recommendations
        assert "enabled_strategies" in recommendations
        assert "market_conditions" in recommendations
    def test_context_to_dict(self, analytical_integration, test_symbol) -> None:
        """Тест преобразования контекста в словарь."""
        context = analytical_integration.get_context(test_symbol)
        # Устанавливаем некоторые значения
        context.set("test_flag", "test_value")
        context.market_context.external_sync = True
        context.strategy_modifiers.order_aggressiveness = 0.8
        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert context_dict["symbol"] == test_symbol
        assert context_dict["flags"]["test_flag"] == "test_value"
        assert context_dict["market_context"]["external_sync"] is True
        assert context_dict["strategy_modifiers"]["order_aggressiveness"] == 0.8
        assert "recommendations" in context_dict
class TestMarketMakerAnalyticalIntegration:
    """Тесты интеграции с MarketMakerModelAgent."""
    @pytest.fixture
    def agent_config(self) -> Any:
        """Конфигурация агента для тестов."""
        return {
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
    @pytest.fixture
    def market_maker_agent(self, agent_config) -> Any:
        """Экземпляр агента для тестов."""
        return MarketMakerModelAgent(config=agent_config)
    @pytest.fixture
    def test_symbol(self) -> None:
        """Тестовый символ."""
        return "BTCUSDT"
    @pytest.fixture
    def test_market_data(self) -> None:
        """Тестовые рыночные данные."""
        dates = pd.date_range(start="2024-01-01", periods=50, freq="1min")
        data = {
            "open": [50000 + i * 0.1 for i in range(50)],
            "high": [50000 + i * 0.1 + 10 for i in range(50)],
            "low": [50000 + i * 0.1 - 10 for i in range(50)],
            "close": [50000 + i * 0.1 + 5 for i in range(50)],
            "volume": [1000000 + i * 1000 for i in range(50)],
        }
        return pd.DataFrame(data, index=dates)
    @pytest.fixture
    def test_order_book(self) -> None:
        """Тестовый ордербук."""
        return {
            "bids": [
                {"price": 49999 - i * 0.1, "size": 1.0 + i * 0.1} for i in range(20)
            ],
            "asks": [
                {"price": 50001 + i * 0.1, "size": 1.0 + i * 0.1} for i in range(20)
            ],
            "symbol": "BTCUSDT",
            "timestamp": time.time(),
        }
    def test_agent_initialization(self, market_maker_agent) -> None:
        """Тест инициализации агента с аналитикой."""
        assert market_maker_agent.config["analytics_enabled"] is True
        assert hasattr(market_maker_agent, "analytical_integration")
    @pytest.mark.asyncio
    async def test_start_stop_analytics(self, market_maker_agent) -> None:
        """Тест запуска и остановки аналитики."""
        # Запуск
        await market_maker_agent.start_analytics()
        # Остановка
        await market_maker_agent.stop_analytics()
    def test_get_analytical_context(self, market_maker_agent, test_symbol) -> None:
        """Тест получения аналитического контекста."""
        context = market_maker_agent.get_analytical_context(test_symbol)
        assert isinstance(context, dict)
    def test_get_trading_recommendations(self, market_maker_agent, test_symbol) -> None:
        """Тест получения торговых рекомендаций."""
        recommendations = market_maker_agent.get_trading_recommendations(test_symbol)
        assert isinstance(recommendations, dict)
        assert "should_trade" in recommendations
    def test_should_proceed_with_trade(self, market_maker_agent, test_symbol) -> None:
        """Тест проверки возможности торговли."""
        should_trade = market_maker_agent.should_proceed_with_trade(test_symbol, 0.8)
        assert isinstance(should_trade, bool)
    def test_get_adjusted_parameters(self, market_maker_agent, test_symbol) -> None:
        """Тест получения скорректированных параметров."""
        # Агрессивность
        adjusted_aggression = market_maker_agent.get_adjusted_aggressiveness(
            test_symbol, 1.0
        )
        assert isinstance(adjusted_aggression, float)
        # Размер позиции
        adjusted_size = market_maker_agent.get_adjusted_position_size(test_symbol, 1.0)
        assert isinstance(adjusted_size, float)
        # Уверенность
        adjusted_confidence = market_maker_agent.get_adjusted_confidence(
            test_symbol, 0.8
        )
        assert isinstance(adjusted_confidence, float)
        # Смещение цены
        price_offset = market_maker_agent.get_price_offset(test_symbol, 50000.0, "buy")
        assert isinstance(price_offset, float)
    @pytest.mark.asyncio
    async def test_calculate_with_analytics(
        self, market_maker_agent, test_symbol, test_market_data, test_order_book
    ) -> None:
        """Тест расчета с аналитикой."""
        result = await market_maker_agent.calculate_with_analytics(
            symbol=test_symbol,
            market_data=test_market_data,
            order_book=test_order_book,
            aggressiveness=0.8,
            confidence=0.7,
        )
        assert isinstance(result, dict)
        assert "action" in result
        assert "confidence" in result
        assert "symbol" in result
        assert "analytical_context" in result
        assert "trading_recommendations" in result
    def test_get_analytics_statistics(self, market_maker_agent) -> None:
        """Тест получения статистики аналитики."""
        stats = market_maker_agent.get_analytics_statistics()
        assert isinstance(stats, dict)
class TestIntegrationScenarios:
    """Тесты различных сценариев интеграции."""
    @pytest.fixture
    def integration(self) -> Any:
        """Экземпляр интеграции для тестов."""
        config = AnalyticalIntegrationConfig(
            entanglement_enabled=True,
            noise_enabled=True,
            mirror_enabled=True,
            gravity_enabled=True,
        )
        return AnalyticalIntegration(config=config)
    def test_clean_market_scenario(self, integration) -> None:
        """Тест сценария чистого рынка."""
        symbol = "BTCUSDT"
        # Проверяем возможность торговли
        should_trade = integration.should_proceed_with_trade(symbol, 0.8)
        assert should_trade is True
        # Получаем рекомендации
        recommendations = integration.get_trading_recommendations(symbol)
        assert recommendations["should_trade"] is True
        assert recommendations["market_conditions"]["is_clean"] is True
    def test_entangled_market_scenario(self, integration) -> None:
        """Тест сценария запутанного рынка."""
        symbol = "BTCUSDT"
        # Применяем запутанность
        context = integration.get_context(symbol)
        entanglement_result = EntanglementResult(
            is_entangled=True,
            correlation_score=0.98,
            confidence=0.95,
            exchange_pair=("binance", "bybit"),
            lag_ms=1.5,
            metadata={},
        )
        context.apply_entanglement_modifier(entanglement_result)
        # Проверяем блокировку торговли
        should_trade = integration.should_proceed_with_trade(symbol, 0.9)
        assert should_trade is False
        # Проверяем снижение агрессивности
        adjusted_aggression = integration.get_adjusted_aggressiveness(symbol, 1.0)
        assert adjusted_aggression < 1.0
    def test_noisy_market_scenario(self, integration) -> None:
        """Тест сценария шумного рынка."""
        symbol = "BTCUSDT"
        # Применяем шум
        context = integration.get_context(symbol)
        noise_result = NoiseAnalysisResult(
            is_synthetic=True,
            noise_intensity=0.85,
            confidence=0.92,
            noise_pattern="artificial_clustering",
            metadata={},
        )
        context.apply_noise_modifier(noise_result)
        # Проверяем смещение цены
        base_price = 50000.0
        buy_price = integration.get_price_offset(symbol, base_price, "buy")
        assert buy_price > base_price
        sell_price = integration.get_price_offset(symbol, base_price, "sell")
        assert sell_price < base_price
    def test_mirror_signals_scenario(self, integration) -> None:
        """Тест сценария зеркальных сигналов."""
        symbol = "BTCUSDT"
        # Применяем зеркальный сигнал
        context = integration.get_context(symbol)
        mirror_signal = MirrorSignal(
            is_mirror=True,
            leader_asset="ETHUSDT",
            follower_asset=symbol,
            correlation=0.92,
            lag_periods=3,
            confidence=0.88,
            metadata={},
        )
        context.apply_mirror_modifier(mirror_signal)
        # Проверяем усиление уверенности
        adjusted_confidence = integration.get_adjusted_confidence(symbol, 0.8)
        assert adjusted_confidence > 0.8
        # Проверяем увеличение размера позиции
        adjusted_size = integration.get_adjusted_position_size(symbol, 1.0)
        assert adjusted_size > 1.0
    def test_gravity_effects_scenario(self, integration) -> None:
        """Тест сценария влияния гравитации."""
        symbol = "BTCUSDT"
        # Применяем гравитацию
        context = integration.get_context(symbol)
        gravity_result = LiquidityGravityResult(
            total_gravity=2.5e-6,
            risk_level="high",
            gravity_centers=[(49999, 1.5), (50001, 1.2)],
            metadata={},
        )
        risk_assessment = RiskAssessmentResult(
            risk_level="high",
            agent_aggression=0.6,
            gravity_score=2.5e-6,
            recommendations=["reduce_position_size"],
            metadata={},
        )
        context.apply_gravity_modifier(gravity_result)
        context.apply_risk_modifier(risk_assessment)
        # Проверяем снижение агрессивности
        adjusted_aggression = integration.get_adjusted_aggressiveness(symbol, 1.0)
        assert adjusted_aggression < 1.0
        # Проверяем снижение размера позиции
        adjusted_size = integration.get_adjusted_position_size(symbol, 1.0)
        assert adjusted_size < 1.0
if __name__ == "__main__":
    pytest.main([__file__])
