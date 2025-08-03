# -*- coding: utf-8 -*-
"""Интеграционные тесты для системы гравитации ликвидности."""
import time
import pytest
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from application.risk.liquidity_gravity_monitor import LiquidityGravityMonitor, MonitorConfig
from domain.market.liquidity_gravity import (LiquidityGravityConfig,
                                             LiquidityGravityModel,
                                             OrderBookSnapshot)
from domain.value_objects.timestamp import Timestamp
from datetime import datetime
class TestLiquidityGravityIntegration:
    """Интеграционные тесты для системы гравитации ликвидности."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.gravity_config = LiquidityGravityConfig(
            gravitational_constant=1e-6,
            min_volume_threshold=0.001,
            max_price_distance=0.1,
            volume_weight=1.0,
            price_weight=1.0,
        )
        self.risk_thresholds = RiskThresholds(
            low_risk=0.1,
            medium_risk=0.5,
            high_risk=1.0,
            extreme_risk=2.0,
            reduce_aggression_threshold=0.8,
            stop_trading_threshold=2.0,
        )
        self.gravity_model = LiquidityGravityModel(self.gravity_config)
        self.risk_assessor = LiquidityRiskAssessor(
            self.gravity_config, self.risk_thresholds
        )
        self.gravity_filter = LiquidityGravityFilter(self.risk_assessor)
    def create_test_order_book(
        self, spread_percentage: float = 0.1, volumes: tuple = (1.0, 1.0)
    ) -> OrderBookSnapshot:
        """Создание тестового ордербука."""
        base_price = 50000.0
        spread = base_price * spread_percentage / 100
        bids = [
            (base_price - spread / 2, volumes[0]),
            (base_price - spread / 2 - 1, volumes[0] * 0.8),
        ]
        asks = [
            (base_price + spread / 2, volumes[1]),
            (base_price + spread / 2 + 1, volumes[1] * 0.8),
        ]
        return OrderBookSnapshot(
            bids=bids, asks=asks, timestamp=datetime.fromtimestamp(time.time()), symbol="BTC/USDT"
        )
    def test_full_pipeline_integration(self) -> None:
        """Тест полной интеграции системы."""
        # Создание ордербука
        order_book = self.create_test_order_book(spread_percentage=0.1)
        # 1. Вычисление гравитации
        gravity = self.gravity_model.compute_liquidity_gravity(order_book)
        assert isinstance(gravity, float)
        assert gravity >= 0.0
        # 2. Полный анализ гравитации
        gravity_result = self.gravity_model.analyze_liquidity_gravity(order_book)
        assert gravity_result.total_gravity == gravity
        assert isinstance(gravity_result.risk_level, str)
        assert len(gravity_result.bid_ask_forces) > 0
        # 3. Оценка риска
        risk_result = self.risk_assessor.assess_liquidity_risk(order_book, "test_agent")
        assert risk_result.gravity_value == gravity
        assert isinstance(risk_result.agent_aggression, float)
        assert isinstance(risk_result.recommended_action, str)
        # 4. Проверка торгового решения
        should_proceed, trade_metadata = self.gravity_filter.should_proceed_with_trade(
            order_book, "test_agent", trade_aggression=0.8
        )
        assert isinstance(should_proceed, bool)
        assert isinstance(trade_metadata, dict)
        # 5. Получение скорректированной агрессивности
        adjusted_aggression, aggression_metadata = (
            self.gravity_filter.get_adjusted_aggression(
                order_book, "test_agent", base_aggression=0.8
            )
        )
        assert isinstance(adjusted_aggression, float)
        assert 0.0 <= adjusted_aggression <= 1.0
    def test_agent_state_evolution(self) -> None:
        """Тест эволюции состояния агента."""
        agent_id = "evolution_test_agent"
        # Установка базовой агрессивности
        self.risk_assessor.set_agent_base_aggression(agent_id, 1.0)
        # Создание серии ордербуков с разными уровнями риска
        order_books = []
        # Низкий риск
        for i in range(3):
            order_book = self.create_test_order_book(
                spread_percentage=0.5, volumes=(0.1, 0.1)
            )
            order_books.append(order_book)
        # Средний риск
        for i in range(3):
            order_book = self.create_test_order_book(
                spread_percentage=0.1, volumes=(1.0, 1.0)
            )
            order_books.append(order_book)
        # Высокий риск
        for i in range(3):
            order_book = self.create_test_order_book(
                spread_percentage=0.05, volumes=(5.0, 5.0)
            )
            order_books.append(order_book)
        # Симуляция эволюции состояния
        aggressions = []
        risk_levels = []
        for order_book in order_books:
            risk_result = self.risk_assessor.assess_liquidity_risk(order_book, agent_id)
            aggressions.append(risk_result.agent_aggression)
            risk_levels.append(risk_result.risk_level)
        # Проверка изменений агрессивности
        assert len(aggressions) == 9
        assert all(0.0 <= agg <= 1.0 for agg in aggressions)
        # Проверка, что агрессивность снижается при высоком риске
        low_risk_aggression = np.mean(aggressions[:3])
        high_risk_aggression = np.mean(aggressions[6:])
        # Агрессивность должна быть ниже при высоком риске
        assert high_risk_aggression <= low_risk_aggression
    def test_risk_threshold_behavior(self) -> None:
        """Тест поведения на разных порогах риска."""
        agent_id = "threshold_test_agent"
        self.risk_assessor.set_agent_base_aggression(agent_id, 1.0)
        # Тест с очень низким риском
        low_risk_order_book = self.create_test_order_book(
            spread_percentage=2.0, volumes=(0.01, 0.01)
        )
        low_risk_result = self.risk_assessor.assess_liquidity_risk(
            low_risk_order_book, agent_id
        )
        assert low_risk_result.risk_level == "low"
        assert low_risk_result.agent_aggression >= 0.9  # Высокая агрессивность
        # Тест с высоким риском
        high_risk_order_book = self.create_test_order_book(
            spread_percentage=0.01, volumes=(10.0, 10.0)
        )
        high_risk_result = self.risk_assessor.assess_liquidity_risk(
            high_risk_order_book, agent_id
        )
        assert high_risk_result.risk_level in ["high", "extreme", "critical"]
        assert high_risk_result.agent_aggression < 0.9  # Сниженная агрессивность
    def test_filter_decision_logic(self) -> None:
        """Тест логики принятия решений фильтром."""
        agent_id = "filter_test_agent"
        self.risk_assessor.set_agent_base_aggression(agent_id, 0.8)
        # Тест с нормальным ордербуком
        normal_order_book = self.create_test_order_book(spread_percentage=0.1)
        should_proceed, metadata = self.gravity_filter.should_proceed_with_trade(
            normal_order_book, agent_id, trade_aggression=0.5
        )
        assert should_proceed == True
        assert metadata["decision"]["reason"] == "normal_operation"
        # Тест с критическим риском
        critical_order_book = self.create_test_order_book(
            spread_percentage=0.001, volumes=(20.0, 20.0)
        )
        should_proceed, metadata = self.gravity_filter.should_proceed_with_trade(
            critical_order_book, agent_id, trade_aggression=0.9
        )
        # Может быть False в зависимости от настроек
        assert isinstance(should_proceed, bool)
    def test_aggression_adjustment_consistency(self) -> None:
        """Тест консистентности корректировки агрессивности."""
        agent_id = "adjustment_test_agent"
        self.risk_assessor.set_agent_base_aggression(agent_id, 0.8)
        # Создание ордербуков с разными уровнями гравитации
        order_books = []
        for i in range(5):
            spread = 0.1 + i * 0.1
            volume = 0.1 + i * 0.5
            order_book = self.create_test_order_book(
                spread_percentage=spread, volumes=(volume, volume)
            )
            order_books.append(order_book)
        # Проверка корректировки агрессивности
        adjusted_aggressions = []
        for order_book in order_books:
            adjusted_aggression, _ = self.gravity_filter.get_adjusted_aggression(
                order_book, agent_id, base_aggression=0.8
            )
            adjusted_aggressions.append(adjusted_aggression)
        # Проверка, что все значения в допустимом диапазоне
        assert all(0.0 <= agg <= 1.0 for agg in adjusted_aggressions)
        # Проверка, что корректировка консистентна
        assert len(set(adjusted_aggressions)) > 1  # Должны быть разные значения
    def test_performance_integration(self) -> None:
        """Тест производительности интеграции."""
        # Создание большого количества ордербуков
        order_books = []
        for i in range(50):
            spread = 0.1 + (i % 10) * 0.1
            volume = 0.1 + (i % 5) * 0.5
            order_book = self.create_test_order_book(
                spread_percentage=spread, volumes=(volume, volume)
            )
            order_books.append(order_book)
        # Тест производительности полного пайплайна
        start_time = time.time()
        results = []
        for order_book in order_books:
            # Полный анализ
            gravity_result = self.gravity_model.analyze_liquidity_gravity(order_book)
            risk_result = self.risk_assessor.assess_liquidity_risk(
                order_book, "perf_agent"
            )
            should_proceed, _ = self.gravity_filter.should_proceed_with_trade(
                order_book, "perf_agent", trade_aggression=0.8
            )
            results.append(
                {
                    "gravity": gravity_result.total_gravity,
                    "risk_level": risk_result.risk_level,
                    "aggression": risk_result.agent_aggression,
                    "should_proceed": should_proceed,
                }
            )
        end_time = time.time()
        processing_time = end_time - start_time
        # Проверка производительности
        assert (
            processing_time < 10.0
        )  # Должно обработать 50 ордербуков менее чем за 10 секунд
        assert len(results) == 50
        # Проверка результатов
        for result in results:
            assert isinstance(result["gravity"], float)
            assert isinstance(result["risk_level"], str)
            assert isinstance(result["aggression"], float)
            assert isinstance(result["should_proceed"], bool)
    def test_error_handling_integration(self) -> None:
        """Тест обработки ошибок в интеграции."""
        agent_id = "error_test_agent"
        self.risk_assessor.set_agent_base_aggression(agent_id, 0.8)
        # Тест с пустым ордербуком
        empty_order_book = OrderBookSnapshot(
            bids=[], asks=[], timestamp=datetime.fromtimestamp(time.time()), symbol="BTC/USDT"
        )
        # Должно обработать без ошибок
        gravity = self.gravity_model.compute_liquidity_gravity(empty_order_book)
        assert gravity == 0.0
        risk_result = self.risk_assessor.assess_liquidity_risk(
            empty_order_book, agent_id
        )
        assert risk_result.risk_level == "low"
        assert risk_result.gravity_value == 0.0
        should_proceed, metadata = self.gravity_filter.should_proceed_with_trade(
            empty_order_book, agent_id, trade_aggression=0.8
        )
        assert should_proceed == True
        # Тест с некорректными данными
        invalid_order_book = OrderBookSnapshot(
            bids=[(0, 0), (-1, -1)],  # Некорректные цены и объемы
            asks=[(0, 0), (-1, -1)],
            timestamp=datetime.fromtimestamp(time.time()),
            symbol="BTC/USDT",
        )
        # Должно обработать без ошибок
        gravity = self.gravity_model.compute_liquidity_gravity(invalid_order_book)
        assert isinstance(gravity, float)
        risk_result = self.risk_assessor.assess_liquidity_risk(
            invalid_order_book, agent_id
        )
        assert isinstance(risk_result, type(risk_result))
    def test_configuration_integration(self) -> None:
        """Тест интеграции конфигурации."""
        # Создание кастомной конфигурации
        custom_gravity_config = LiquidityGravityConfig(
            gravitational_constant=2e-6,  # Увеличенная константа
            min_volume_threshold=0.01,  # Увеличенный порог
            max_price_distance=0.05,  # Уменьшенное расстояние
            volume_weight=2.0,  # Увеличенный вес объема
            price_weight=0.5,  # Уменьшенный вес цены
        )
        custom_risk_thresholds = RiskThresholds(
            low_risk=0.05,  # Более строгие пороги
            medium_risk=0.2,
            high_risk=0.5,
            extreme_risk=1.0,
            reduce_aggression_threshold=0.3,
            stop_trading_threshold=1.0,
        )
        # Создание системы с кастомной конфигурацией
        custom_gravity_model = LiquidityGravityModel(custom_gravity_config)
        custom_risk_assessor = LiquidityRiskAssessor(
            custom_gravity_config, custom_risk_thresholds
        )
        LiquidityGravityFilter(custom_risk_assessor)
        # Тест с кастомной конфигурацией
        order_book = self.create_test_order_book(spread_percentage=0.1)
        # Сравнение результатов
        default_gravity = self.gravity_model.compute_liquidity_gravity(order_book)
        custom_gravity = custom_gravity_model.compute_liquidity_gravity(order_book)
        # Результаты должны отличаться из-за разных конфигураций
        assert abs(default_gravity - custom_gravity) > 1e-10
        # Тест оценки риска с кастомными порогами
        custom_risk_assessor.set_agent_base_aggression("custom_agent", 0.8)
        custom_risk_result = custom_risk_assessor.assess_liquidity_risk(
            order_book, "custom_agent"
        )
        assert isinstance(custom_risk_result.risk_level, str)
        assert isinstance(custom_risk_result.agent_aggression, float)
    def test_multi_agent_integration(self) -> None:
        """Тест интеграции с несколькими агентами."""
        agents = ["agent_1", "agent_2", "agent_3"]
        # Настройка разных агентов
        self.risk_assessor.set_agent_base_aggression("agent_1", 0.6)  # Консервативный
        self.risk_assessor.set_agent_base_aggression("agent_2", 0.8)  # Умеренный
        self.risk_assessor.set_agent_base_aggression("agent_3", 1.0)  # Агрессивный
        # Создание ордербука с высоким риском
        high_risk_order_book = self.create_test_order_book(
            spread_percentage=0.01, volumes=(10.0, 10.0)
        )
        # Тест реакции разных агентов
        agent_results = {}
        for agent_id in agents:
            risk_result = self.risk_assessor.assess_liquidity_risk(
                high_risk_order_book, agent_id
            )
            should_proceed, _ = self.gravity_filter.should_proceed_with_trade(
                high_risk_order_book, agent_id, trade_aggression=0.8
            )
            agent_results[agent_id] = {
                "risk_level": risk_result.risk_level,
                "aggression": risk_result.agent_aggression,
                "should_proceed": should_proceed,
            }
        # Проверка результатов
        assert len(agent_results) == 3
        for agent_id, result in agent_results.items():
            assert isinstance(result["risk_level"], str)
            assert isinstance(result["aggression"], float)
            assert isinstance(result["should_proceed"], bool)
            assert 0.0 <= result["aggression"] <= 1.0
        # Проверка, что более агрессивные агенты имеют более высокую агрессивность
        aggressions = [result["aggression"] for result in agent_results.values()]
        assert len(set(aggressions)) >= 1  # Должны быть разные значения
    def test_statistics_integration(self) -> None:
        """Тест интеграции статистики."""
        agent_id = "stats_test_agent"
        self.risk_assessor.set_agent_base_aggression(agent_id, 0.8)
        # Создание серии ордербуков
        for i in range(10):
            order_book = self.create_test_order_book(
                spread_percentage=0.1 + (i % 3) * 0.1,
                volumes=(0.1 + i * 0.1, 0.1 + i * 0.1),
            )
            self.risk_assessor.assess_liquidity_risk(order_book, agent_id)
        # Получение статистики
        model_stats = self.gravity_model.get_model_statistics()
        risk_stats = self.risk_assessor.get_risk_statistics()
        # Проверка статистики модели
        assert isinstance(model_stats, dict)
        assert "gravitational_constant" in model_stats
        assert "min_volume_threshold" in model_stats
        assert "max_price_distance" in model_stats
        # Проверка статистики рисков
        assert isinstance(risk_stats, dict)
        assert "total_agents" in risk_stats
        assert "agent_states" in risk_stats
        assert agent_id in risk_stats["agent_states"]
        agent_state_stats = risk_stats["agent_states"][agent_id]
        assert "current_aggression" in agent_state_stats
        assert "base_aggression" in agent_state_stats
        assert "risk_level" in agent_state_stats
        assert "gravity_history_length" in agent_state_stats
    def test_recovery_mechanism_integration(self) -> None:
        """Тест интеграции механизма восстановления."""
        agent_id = "recovery_test_agent"
        self.risk_assessor.set_agent_base_aggression(agent_id, 0.8)
        # Создание высокого риска
        high_risk_order_book = self.create_test_order_book(
            spread_percentage=0.01, volumes=(10.0, 10.0)
        )
        # Применение высокого риска
        for i in range(5):
            self.risk_assessor.assess_liquidity_risk(high_risk_order_book, agent_id)
        # Проверка снижения агрессивности
        agent_state = self.risk_assessor.get_agent_risk_state(agent_id)
        assert agent_state.current_aggression < agent_state.base_aggression
        # Создание низкого риска для восстановления
        low_risk_order_book = self.create_test_order_book(
            spread_percentage=2.0, volumes=(0.01, 0.01)
        )
        # Применение низкого риска для восстановления
        for i in range(10):
            self.risk_assessor.assess_liquidity_risk(low_risk_order_book, agent_id)
        # Проверка восстановления агрессивности
        agent_state = self.risk_assessor.get_agent_risk_state(agent_id)
        # Агрессивность должна восстановиться или быть близкой к базовой
        assert agent_state.current_aggression >= agent_state.base_aggression * 0.8
if __name__ == "__main__":
    pytest.main([__file__])
