#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Комплексные тесты для Quantum Arbitrage Strategy.
"""

import pytest
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from domain.strategies.quantum_arbitrage_strategy import (
    QuantumArbitrageStrategy, QuantumState, ArbitrageOpportunity,
    calculate_quantum_probability, detect_arbitrage_opportunities
)
from domain.value_objects.price import Price
from domain.value_objects.currency import Currency
from domain.entities.market_data import MarketData, OrderBook
from domain.exceptions import StrategyError, ValidationError


class TestQuantumArbitrageStrategy:
    """Тесты для Quantum Arbitrage Strategy."""

    @pytest.fixture
    def usd_currency(self) -> Currency:
        """Фикстура USD валюты."""
        return Currency("USD")

    @pytest.fixture
    def btc_currency(self) -> Currency:
        """Фикстура BTC валюты."""
        return Currency("BTC")

    @pytest.fixture
    def strategy_config(self) -> Dict[str, Any]:
        """Фикстура конфигурации стратегии."""
        return {
            "min_arbitrage_threshold": Decimal("0.001"),  # 0.1%
            "max_position_size": Decimal("1.0"),
            "quantum_coherence_threshold": 0.85,
            "entanglement_sensitivity": 0.75,
            "max_execution_latency": 50,  # milliseconds
            "risk_multiplier": Decimal("2.0"),
            "exchanges": ["binance", "bybit", "okx"],
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "quantum_states": 16
        }

    @pytest.fixture
    def sample_market_data(self, usd_currency: Currency) -> Dict[str, MarketData]:
        """Фикстура рыночных данных с разных бирж."""
        return {
            "binance": MarketData(
                symbol="BTCUSDT",
                price=Price(Decimal("45000.00"), usd_currency),
                volume=Decimal("1000.5"),
                timestamp=1640995200000,
                bid=Price(Decimal("44999.50"), usd_currency),
                ask=Price(Decimal("45000.50"), usd_currency)
            ),
            "bybit": MarketData(
                symbol="BTCUSDT",
                price=Price(Decimal("45025.00"), usd_currency),
                volume=Decimal("850.3"),
                timestamp=1640995200500,
                bid=Price(Decimal("45024.50"), usd_currency),
                ask=Price(Decimal("45025.50"), usd_currency)
            ),
            "okx": MarketData(
                symbol="BTCUSDT",
                price=Price(Decimal("45010.00"), usd_currency),
                volume=Decimal("920.8"),
                timestamp=1640995200200,
                bid=Price(Decimal("45009.50"), usd_currency),
                ask=Price(Decimal("45010.50"), usd_currency)
            )
        }

    @pytest.fixture
    def quantum_strategy(self, strategy_config: Dict[str, Any]) -> QuantumArbitrageStrategy:
        """Фикстура квантовой арбитражной стратегии."""
        return QuantumArbitrageStrategy(**strategy_config)

    def test_strategy_initialization(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест инициализации стратегии."""
        assert quantum_strategy.min_arbitrage_threshold == Decimal("0.001")
        assert quantum_strategy.max_position_size == Decimal("1.0")
        assert quantum_strategy.quantum_coherence_threshold == 0.85
        assert quantum_strategy.entanglement_sensitivity == 0.75
        assert len(quantum_strategy.exchanges) == 3
        assert len(quantum_strategy.symbols) == 2
        assert quantum_strategy.quantum_states == 16

    def test_quantum_state_creation(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест создания квантового состояния."""
        amplitudes = np.random.random(16) + 1j * np.random.random(16)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        quantum_state = QuantumState(amplitudes=amplitudes, coherence=0.9)
        
        assert len(quantum_state.amplitudes) == 16
        assert abs(np.sum(np.abs(quantum_state.amplitudes)**2) - 1.0) < 1e-10
        assert quantum_state.coherence == 0.9
        assert quantum_state.is_coherent() is True

    def test_quantum_probability_calculation(self) -> None:
        """Тест расчета квантовой вероятности."""
        # Создаем простое квантовое состояние
        amplitudes = np.array([0.6, 0.8j, 0.0, 0.0])
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        quantum_state = QuantumState(amplitudes=amplitudes, coherence=0.95)
        
        # Вероятность измерения в состоянии 0
        prob_0 = calculate_quantum_probability(quantum_state, 0)
        assert abs(prob_0 - 0.36) < 1e-10  # |0.6|^2 = 0.36
        
        # Вероятность измерения в состоянии 1
        prob_1 = calculate_quantum_probability(quantum_state, 1)
        assert abs(prob_1 - 0.64) < 1e-10  # |0.8|^2 = 0.64

    def test_arbitrage_opportunity_detection(
        self, 
        quantum_strategy: QuantumArbitrageStrategy,
        sample_market_data: Dict[str, MarketData]
    ) -> None:
        """Тест обнаружения арбитражных возможностей."""
        opportunities = detect_arbitrage_opportunities(
            market_data=sample_market_data,
            min_threshold=quantum_strategy.min_arbitrage_threshold
        )
        
        # Должны найти возможности между биржами
        assert len(opportunities) > 0
        
        # Проверяем структуру возможности
        opportunity = opportunities[0]
        assert isinstance(opportunity, ArbitrageOpportunity)
        assert opportunity.buy_exchange in quantum_strategy.exchanges
        assert opportunity.sell_exchange in quantum_strategy.exchanges
        assert opportunity.profit_percentage > quantum_strategy.min_arbitrage_threshold
        assert opportunity.buy_price < opportunity.sell_price

    def test_quantum_entanglement_detection(
        self, 
        quantum_strategy: QuantumArbitrageStrategy,
        sample_market_data: Dict[str, MarketData]
    ) -> None:
        """Тест обнаружения квантовой запутанности между биржами."""
        entanglement_matrix = quantum_strategy.calculate_market_entanglement(sample_market_data)
        
        # Матрица должна быть симметричной
        assert entanglement_matrix.shape == (3, 3)  # 3 биржи
        assert np.allclose(entanglement_matrix, entanglement_matrix.T)
        
        # Диагональные элементы должны быть 1 (биржа запутана сама с собой)
        assert np.allclose(np.diag(entanglement_matrix), 1.0)
        
        # Недиагональные элементы должны быть в диапазоне [0, 1]
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert 0 <= entanglement_matrix[i, j] <= 1

    def test_quantum_coherence_measurement(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест измерения квантовой когерентности."""
        # Полностью когерентное состояние
        coherent_amplitudes = np.array([1.0, 0.0, 0.0, 0.0])
        coherent_state = QuantumState(amplitudes=coherent_amplitudes, coherence=1.0)
        
        coherence = quantum_strategy.measure_coherence(coherent_state)
        assert coherence >= 0.99  # Практически полная когерентность
        
        # Частично декогерентное состояние
        mixed_amplitudes = np.array([0.5, 0.5, 0.5, 0.5])
        mixed_amplitudes = mixed_amplitudes / np.linalg.norm(mixed_amplitudes)
        mixed_state = QuantumState(amplitudes=mixed_amplitudes, coherence=0.6)
        
        mixed_coherence = quantum_strategy.measure_coherence(mixed_state)
        assert 0.5 <= mixed_coherence <= 0.7

    def test_quantum_superposition_arbitrage(
        self, 
        quantum_strategy: QuantumArbitrageStrategy,
        sample_market_data: Dict[str, MarketData]
    ) -> None:
        """Тест арбитража в квантовой суперпозиции."""
        # Создаем суперпозицию арбитражных состояний
        superposition_state = quantum_strategy.create_arbitrage_superposition(sample_market_data)
        
        assert len(superposition_state.amplitudes) == quantum_strategy.quantum_states
        assert superposition_state.is_coherent()
        
        # Измеряем арбитражные возможности из суперпозиции
        measured_opportunities = quantum_strategy.measure_arbitrage_from_superposition(
            superposition_state, sample_market_data
        )
        
        assert len(measured_opportunities) > 0
        assert all(opp.profit_percentage > 0 for opp in measured_opportunities)

    def test_quantum_tunneling_execution(
        self, 
        quantum_strategy: QuantumArbitrageStrategy
    ) -> None:
        """Тест исполнения через квантовое туннелирование."""
        # Создаем арбитражную возможность с высоким барьером
        opportunity = ArbitrageOpportunity(
            symbol="BTCUSDT",
            buy_exchange="binance",
            sell_exchange="bybit",
            buy_price=Price(Decimal("45000.00"), Currency("USD")),
            sell_price=Price(Decimal("45050.00"), Currency("USD")),
            profit_percentage=Decimal("0.0011"),
            execution_window=100,  # ms
            barrier_height=0.8  # Высокий барьер
        )
        
        # Рассчитываем вероятность туннелирования
        tunneling_probability = quantum_strategy.calculate_tunneling_probability(opportunity)
        
        assert 0 < tunneling_probability < 1
        assert tunneling_probability < 0.5  # Низкая вероятность для высокого барьера

    def test_quantum_interference_optimization(
        self, 
        quantum_strategy: QuantumArbitrageStrategy,
        sample_market_data: Dict[str, MarketData]
    ) -> None:
        """Тест оптимизации через квантовую интерференцию."""
        opportunities = detect_arbitrage_opportunities(
            sample_market_data, quantum_strategy.min_arbitrage_threshold
        )
        
        # Применяем квантовую интерференцию для оптимизации
        optimized_opportunities = quantum_strategy.apply_quantum_interference(opportunities)
        
        assert len(optimized_opportunities) <= len(opportunities)
        
        # Оптимизированные возможности должны иметь лучшие параметры
        if optimized_opportunities:
            best_optimized = max(optimized_opportunities, key=lambda x: x.profit_percentage)
            best_original = max(opportunities, key=lambda x: x.profit_percentage)
            
            assert best_optimized.profit_percentage >= best_original.profit_percentage

    def test_quantum_error_correction(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест квантовой коррекции ошибок."""
        # Создаем состояние с ошибками
        noisy_amplitudes = np.random.random(16) + 1j * np.random.random(16)
        noisy_amplitudes = noisy_amplitudes / np.linalg.norm(noisy_amplitudes)
        
        # Добавляем шум
        noise = 0.1 * (np.random.random(16) + 1j * np.random.random(16))
        corrupted_amplitudes = noisy_amplitudes + noise
        corrupted_amplitudes = corrupted_amplitudes / np.linalg.norm(corrupted_amplitudes)
        
        corrupted_state = QuantumState(amplitudes=corrupted_amplitudes, coherence=0.7)
        
        # Применяем коррекцию ошибок
        corrected_state = quantum_strategy.apply_error_correction(corrupted_state)
        
        assert corrected_state.coherence > corrupted_state.coherence
        assert corrected_state.is_coherent()

    def test_multi_exchange_arbitrage_execution(
        self, 
        quantum_strategy: QuantumArbitrageStrategy,
        sample_market_data: Dict[str, MarketData]
    ) -> None:
        """Тест исполнения арбитража на множественных биржах."""
        opportunities = detect_arbitrage_opportunities(
            sample_market_data, quantum_strategy.min_arbitrage_threshold
        )
        
        # Группируем возможности по символам
        grouped_opportunities = quantum_strategy.group_opportunities_by_symbol(opportunities)
        
        assert "BTCUSDT" in grouped_opportunities
        assert len(grouped_opportunities["BTCUSDT"]) > 0
        
        # Выполняем мультибиржевое исполнение
        execution_plan = quantum_strategy.create_multi_exchange_execution_plan(
            grouped_opportunities["BTCUSDT"]
        )
        
        assert execution_plan.total_profit > 0
        assert len(execution_plan.trades) > 0
        assert execution_plan.execution_time < quantum_strategy.max_execution_latency

    def test_quantum_risk_management(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест квантового риск-менеджмента."""
        # Создаем портфель позиций
        portfolio_positions = {
            "BTCUSDT": Decimal("0.5"),
            "ETHUSDT": Decimal("2.0")
        }
        
        # Рассчитываем квантовый риск
        quantum_risk = quantum_strategy.calculate_quantum_risk(portfolio_positions)
        
        assert quantum_risk.var_95 > 0  # Value at Risk
        assert quantum_risk.expected_shortfall > quantum_risk.var_95
        assert 0 <= quantum_risk.coherence_risk <= 1
        assert quantum_risk.entanglement_risk >= 0

    def test_adaptive_quantum_parameters(
        self, 
        quantum_strategy: QuantumArbitrageStrategy,
        sample_market_data: Dict[str, MarketData]
    ) -> None:
        """Тест адаптивной настройки квантовых параметров."""
        original_coherence = quantum_strategy.quantum_coherence_threshold
        original_sensitivity = quantum_strategy.entanglement_sensitivity
        
        # Симулируем изменение рыночных условий
        volatile_market_data = sample_market_data.copy()
        for exchange, data in volatile_market_data.items():
            # Увеличиваем волатильность
            data.price = Price(
                data.price.value * Decimal("1.05"), 
                data.price.currency
            )
        
        # Адаптируем параметры
        quantum_strategy.adapt_parameters(volatile_market_data)
        
        # Параметры должны измениться в ответ на волатильность
        assert quantum_strategy.quantum_coherence_threshold != original_coherence
        assert quantum_strategy.entanglement_sensitivity != original_sensitivity

    def test_quantum_algorithm_performance(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест производительности квантового алгоритма."""
        import time
        
        # Создаем большой набор тестовых данных
        large_market_data = {}
        for i in range(10):
            large_market_data[f"exchange_{i}"] = MarketData(
                symbol="BTCUSDT",
                price=Price(Decimal(f"45000.{i:02d}"), Currency("USD")),
                volume=Decimal(f"100{i}.5"),
                timestamp=1640995200000 + i * 1000,
                bid=Price(Decimal(f"44999.{i:02d}"), Currency("USD")),
                ask=Price(Decimal(f"45001.{i:02d}"), Currency("USD"))
            )
        
        start_time = time.time()
        
        opportunities = detect_arbitrage_opportunities(
            large_market_data, quantum_strategy.min_arbitrage_threshold
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Алгоритм должен работать быстро даже с большим объемом данных
        assert execution_time < 1.0  # Менее 1 секунды
        assert len(opportunities) > 0

    def test_quantum_state_persistence(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест сохранения квантового состояния."""
        # Создаем квантовое состояние
        amplitudes = np.random.random(16) + 1j * np.random.random(16)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        original_state = QuantumState(amplitudes=amplitudes, coherence=0.9)
        
        # Сериализуем состояние
        state_data = quantum_strategy.serialize_quantum_state(original_state)
        
        # Десериализуем состояние
        restored_state = quantum_strategy.deserialize_quantum_state(state_data)
        
        # Проверяем сохранность
        assert np.allclose(original_state.amplitudes, restored_state.amplitudes)
        assert original_state.coherence == restored_state.coherence

    def test_quantum_strategy_validation(self, strategy_config: Dict[str, Any]) -> None:
        """Тест валидации параметров квантовой стратегии."""
        # Валидная конфигурация
        valid_strategy = QuantumArbitrageStrategy(**strategy_config)
        assert valid_strategy.validate_configuration() is True
        
        # Невалидная конфигурация - отрицательный порог
        invalid_config = strategy_config.copy()
        invalid_config["min_arbitrage_threshold"] = Decimal("-0.001")
        
        with pytest.raises(ValidationError, match="Arbitrage threshold must be positive"):
            QuantumArbitrageStrategy(**invalid_config)
        
        # Невалидная конфигурация - слишком высокая когерентность
        invalid_config = strategy_config.copy()
        invalid_config["quantum_coherence_threshold"] = 1.5
        
        with pytest.raises(ValidationError, match="Coherence threshold must be between 0 and 1"):
            QuantumArbitrageStrategy(**invalid_config)

    def test_quantum_measurement_collapse(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест коллапса квантового состояния при измерении."""
        # Создаем суперпозицию состояний
        amplitudes = np.array([0.6, 0.8, 0.0, 0.0] + [0.0] * 12)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        superposition_state = QuantumState(amplitudes=amplitudes, coherence=0.95)
        
        # Измеряем состояние
        measured_state, measurement_result = quantum_strategy.measure_quantum_state(
            superposition_state
        )
        
        # После измерения состояние должно коллапсировать
        assert measurement_result in [0, 1]  # Возможные результаты измерения
        assert measured_state.coherence < superposition_state.coherence
        
        # Амплитуда измеренного состояния должна быть максимальной
        max_amplitude_index = np.argmax(np.abs(measured_state.amplitudes))
        assert max_amplitude_index == measurement_result

    def test_quantum_arbitrage_edge_cases(self, quantum_strategy: QuantumArbitrageStrategy) -> None:
        """Тест граничных случаев квантового арбитража."""
        # Случай с одинаковыми ценами на всех биржах
        identical_market_data = {
            "binance": MarketData(
                symbol="BTCUSDT",
                price=Price(Decimal("45000.00"), Currency("USD")),
                volume=Decimal("1000.0"),
                timestamp=1640995200000,
                bid=Price(Decimal("44999.50"), Currency("USD")),
                ask=Price(Decimal("45000.50"), Currency("USD"))
            ),
            "bybit": MarketData(
                symbol="BTCUSDT",
                price=Price(Decimal("45000.00"), Currency("USD")),
                volume=Decimal("1000.0"),
                timestamp=1640995200000,
                bid=Price(Decimal("44999.50"), Currency("USD")),
                ask=Price(Decimal("45000.50"), Currency("USD"))
            )
        }
        
        opportunities = detect_arbitrage_opportunities(
            identical_market_data, quantum_strategy.min_arbitrage_threshold
        )
        
        # Не должно быть арбитражных возможностей
        assert len(opportunities) == 0

    def test_quantum_strategy_memory_efficiency(
        self, 
        quantum_strategy: QuantumArbitrageStrategy
    ) -> None:
        """Тест эффективности использования памяти."""
        import sys
        
        # Создаем множество квантовых состояний
        states = []
        for _ in range(100):
            amplitudes = np.random.random(16) + 1j * np.random.random(16)
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
            states.append(QuantumState(amplitudes=amplitudes, coherence=0.9))
        
        # Проверяем использование памяти
        total_memory = sum(sys.getsizeof(state) for state in states)
        average_memory = total_memory / len(states)
        
        # Каждое состояние должно занимать разумное количество памяти
        assert average_memory < 5000  # Менее 5KB на состояние
        
        # Все состояния должны быть валидными
        assert all(state.is_coherent() for state in states)