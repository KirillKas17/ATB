"""
Тесты для доменного сервиса обнаружения паттернов.
"""
import pytest
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from domain.services.pattern_discovery import PatternDiscovery, IPatternDiscovery
from domain.types.ml_types import PatternResult, PatternType, PatternConfidence
class TestPatternDiscovery:
    """Тесты для сервиса обнаружения паттернов."""
    @pytest.fixture
    def pattern_discovery(self) -> Any:
        """Фикстура сервиса обнаружения паттернов."""
        return PatternDiscovery()
    @pytest.fixture
    def sample_market_data(self) -> Any:
        """Фикстура с примерными рыночными данными."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)
        return pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(51000, 52000, 100),
            'low': np.random.uniform(49000, 50000, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'rsi': np.random.uniform(20, 80, 100),
            'macd': np.random.normal(0, 10, 100),
            'bollinger_upper': np.random.uniform(51000, 52000, 100),
            'bollinger_lower': np.random.uniform(49000, 50000, 100)
        }, index=dates)
    @pytest.fixture
    def sample_pattern_data(self) -> Any:
        """Фикстура с данными, содержащими паттерны."""
        dates = pd.date_range('2024-01-01', periods=50, freq='1H')
        np.random.seed(42)
        # Создаем данные с явными паттернами
        base_price = 50000
        trend = np.cumsum(np.random.normal(0, 5, 50))
        return pd.DataFrame({
            'open': base_price + trend + np.random.normal(0, 10, 50),
            'high': base_price + trend + np.random.uniform(0, 20, 50),
            'low': base_price + trend - np.random.uniform(0, 20, 50),
            'close': base_price + trend + np.random.normal(0, 10, 50),
            'volume': np.random.uniform(1000, 5000, 50),
            'rsi': np.random.uniform(30, 70, 50),
            'macd': np.random.normal(0, 5, 50)
        }, index=dates)
    def test_pattern_discovery_initialization(self, pattern_discovery) -> None:
        """Тест инициализации сервиса."""
        assert pattern_discovery is not None
        assert isinstance(pattern_discovery, IPatternDiscovery)
        assert hasattr(pattern_discovery, 'config')
        assert isinstance(pattern_discovery.config, dict)
    def test_pattern_discovery_config_defaults(self, pattern_discovery) -> None:
        """Тест конфигурации по умолчанию."""
        config = pattern_discovery.config
        assert "pattern_threshold" in config
        assert "min_pattern_length" in config
        assert "max_pattern_length" in config
        assert "confidence_threshold" in config
        assert isinstance(config["pattern_threshold"], float)
        assert isinstance(config["min_pattern_length"], int)
    def test_discover_price_patterns(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест обнаружения ценовых паттернов."""
        patterns = pattern_discovery.discover_price_patterns(sample_pattern_data)
        assert isinstance(patterns, list)
        # Проверяем, что все паттерны имеют правильную структуру
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert "pattern_type" in pattern
            assert "start_time" in pattern
            assert "end_time" in pattern
            assert "confidence" in pattern
            assert "strength" in pattern
            assert isinstance(pattern["pattern_type"], str)
            assert isinstance(pattern["confidence"], str)
            assert isinstance(pattern["strength"], float)
            assert pattern["pattern_type"] in ["trend", "reversal", "consolidation", "breakout"]
            assert pattern["confidence"] in ["high", "medium", "low"]
            assert pattern["strength"] >= 0.0 and pattern["strength"] <= 1.0
    def test_discover_price_patterns_empty_data(self, pattern_discovery) -> None:
        """Тест обнаружения ценовых паттернов с пустыми данными."""
        empty_data = pd.DataFrame()
        patterns = pattern_discovery.discover_price_patterns(empty_data)
        assert isinstance(patterns, list)
        assert len(patterns) == 0
    def test_discover_price_patterns_insufficient_data(self, pattern_discovery) -> None:
        """Тест обнаружения ценовых паттернов с недостаточными данными."""
        insufficient_data = pd.DataFrame({
            'close': [50000, 50001, 50002],
            'volume': [1000, 1001, 1002]
        })
        patterns = pattern_discovery.discover_price_patterns(insufficient_data)
        assert isinstance(patterns, list)
        # При недостатке данных должно быть мало паттернов или их не должно быть
        assert len(patterns) <= 1
    def test_discover_volume_patterns(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест обнаружения паттернов объема."""
        patterns = pattern_discovery.discover_volume_patterns(sample_pattern_data)
        assert isinstance(patterns, list)
        # Проверяем, что все паттерны имеют правильную структуру
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert "pattern_type" in pattern
            assert "start_time" in pattern
            assert "end_time" in pattern
            assert "confidence" in pattern
            assert "strength" in pattern
            assert isinstance(pattern["pattern_type"], str)
            assert isinstance(pattern["confidence"], str)
            assert isinstance(pattern["strength"], float)
            assert pattern["pattern_type"] in ["volume_spike", "volume_trend", "volume_divergence"]
            assert pattern["confidence"] in ["high", "medium", "low"]
            assert pattern["strength"] >= 0.0 and pattern["strength"] <= 1.0
    def test_discover_volume_patterns_empty_data(self, pattern_discovery) -> None:
        """Тест обнаружения паттернов объема с пустыми данными."""
        empty_data = pd.DataFrame()
        patterns = pattern_discovery.discover_volume_patterns(empty_data)
        assert isinstance(patterns, list)
        assert len(patterns) == 0
    def test_discover_technical_patterns(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест обнаружения технических паттернов."""
        patterns = pattern_discovery.discover_technical_patterns(sample_pattern_data)
        assert isinstance(patterns, list)
        # Проверяем, что все паттерны имеют правильную структуру
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert "pattern_type" in pattern
            assert "start_time" in pattern
            assert "end_time" in pattern
            assert "confidence" in pattern
            assert "strength" in pattern
            assert isinstance(pattern["pattern_type"], str)
            assert isinstance(pattern["confidence"], str)
            assert isinstance(pattern["strength"], float)
            assert pattern["pattern_type"] in ["rsi_divergence", "macd_crossover", "bollinger_squeeze"]
            assert pattern["confidence"] in ["high", "medium", "low"]
            assert pattern["strength"] >= 0.0 and pattern["strength"] <= 1.0
    def test_discover_technical_patterns_empty_data(self, pattern_discovery) -> None:
        """Тест обнаружения технических паттернов с пустыми данными."""
        empty_data = pd.DataFrame()
        patterns = pattern_discovery.discover_technical_patterns(empty_data)
        assert isinstance(patterns, list)
        assert len(patterns) == 0
    def test_discover_candlestick_patterns(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест обнаружения паттернов свечей."""
        patterns = pattern_discovery.discover_candlestick_patterns(sample_pattern_data)
        assert isinstance(patterns, list)
        # Проверяем, что все паттерны имеют правильную структуру
        for pattern in patterns:
            assert isinstance(pattern, dict)
            assert "pattern_type" in pattern
            assert "start_time" in pattern
            assert "end_time" in pattern
            assert "confidence" in pattern
            assert "strength" in pattern
            assert isinstance(pattern["pattern_type"], str)
            assert isinstance(pattern["confidence"], str)
            assert isinstance(pattern["strength"], float)
            assert pattern["pattern_type"] in ["doji", "hammer", "shooting_star", "engulfing"]
            assert pattern["confidence"] in ["high", "medium", "low"]
            assert pattern["strength"] >= 0.0 and pattern["strength"] <= 1.0
    def test_discover_candlestick_patterns_empty_data(self, pattern_discovery) -> None:
        """Тест обнаружения паттернов свечей с пустыми данными."""
        empty_data = pd.DataFrame()
        patterns = pattern_discovery.discover_candlestick_patterns(empty_data)
        assert isinstance(patterns, list)
        assert len(patterns) == 0
    def test_discover_all_patterns(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест обнаружения всех типов паттернов."""
        all_patterns = pattern_discovery.discover_all_patterns(sample_pattern_data)
        assert isinstance(all_patterns, dict)
        assert "price_patterns" in all_patterns
        assert "volume_patterns" in all_patterns
        assert "technical_patterns" in all_patterns
        assert "candlestick_patterns" in all_patterns
        assert "summary" in all_patterns
        assert isinstance(all_patterns["price_patterns"], list)
        assert isinstance(all_patterns["volume_patterns"], list)
        assert isinstance(all_patterns["technical_patterns"], list)
        assert isinstance(all_patterns["candlestick_patterns"], list)
        assert isinstance(all_patterns["summary"], dict)
    def test_discover_all_patterns_empty_data(self, pattern_discovery) -> None:
        """Тест обнаружения всех типов паттернов с пустыми данными."""
        empty_data = pd.DataFrame()
        all_patterns = pattern_discovery.discover_all_patterns(empty_data)
        assert isinstance(all_patterns, dict)
        assert len(all_patterns["price_patterns"]) == 0
        assert len(all_patterns["volume_patterns"]) == 0
        assert len(all_patterns["technical_patterns"]) == 0
        assert len(all_patterns["candlestick_patterns"]) == 0
    def test_validate_pattern(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест валидации паттерна."""
        # Создаем тестовый паттерн
        test_pattern = PatternResult(
            pattern_type="trend",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T13:00:00Z",
            confidence="medium",
            strength=0.7
        )
        is_valid = pattern_discovery.validate_pattern(test_pattern, sample_pattern_data)
        assert isinstance(is_valid, bool)
    def test_validate_pattern_empty_data(self, pattern_discovery) -> None:
        """Тест валидации паттерна с пустыми данными."""
        test_pattern = PatternResult(
            pattern_type="trend",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T13:00:00Z",
            confidence="medium",
            strength=0.7
        )
        empty_data = pd.DataFrame()
        is_valid = pattern_discovery.validate_pattern(test_pattern, empty_data)
        assert isinstance(is_valid, bool)
        assert is_valid == False  # Должен быть невалидным при отсутствии данных
    def test_calculate_pattern_confidence(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест расчета уверенности паттерна."""
        # Создаем тестовый паттерн
        test_pattern = PatternResult(
            pattern_type="trend",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T13:00:00Z",
            confidence="medium",
            strength=0.7
        )
        confidence = pattern_discovery.calculate_pattern_confidence(test_pattern, sample_pattern_data)
        assert isinstance(confidence, float)
        assert confidence >= 0.0 and confidence <= 1.0
    def test_calculate_pattern_confidence_empty_data(self, pattern_discovery) -> None:
        """Тест расчета уверенности паттерна с пустыми данными."""
        test_pattern = PatternResult(
            pattern_type="trend",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T13:00:00Z",
            confidence="medium",
            strength=0.7
        )
        empty_data = pd.DataFrame()
        confidence = pattern_discovery.calculate_pattern_confidence(test_pattern, empty_data)
        assert isinstance(confidence, float)
        assert confidence == 0.0  # Должен вернуть 0 при отсутствии данных
    def test_get_pattern_statistics(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест получения статистики паттернов."""
        # Сначала обнаруживаем паттерны
        all_patterns = pattern_discovery.discover_all_patterns(sample_pattern_data)
        # Получаем статистику
        stats = pattern_discovery.get_pattern_statistics(all_patterns)
        assert isinstance(stats, dict)
        assert "total_patterns" in stats
        assert "pattern_distribution" in stats
        assert "confidence_distribution" in stats
        assert "average_strength" in stats
        assert isinstance(stats["total_patterns"], int)
        assert isinstance(stats["pattern_distribution"], dict)
        assert isinstance(stats["confidence_distribution"], dict)
        assert isinstance(stats["average_strength"], float)
        assert stats["total_patterns"] >= 0
        assert stats["average_strength"] >= 0.0 and stats["average_strength"] <= 1.0
    def test_get_pattern_statistics_empty_patterns(self, pattern_discovery) -> None:
        """Тест получения статистики паттернов с пустыми паттернами."""
        empty_patterns = {
            "price_patterns": [],
            "volume_patterns": [],
            "technical_patterns": [],
            "candlestick_patterns": [],
            "summary": {}
        }
        stats = pattern_discovery.get_pattern_statistics(empty_patterns)
        assert isinstance(stats, dict)
        assert stats["total_patterns"] == 0
        assert stats["average_strength"] == 0.0
    def test_predict_pattern_outcomes(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест предсказания исходов паттернов."""
        # Сначала обнаруживаем паттерны
        patterns = pattern_discovery.discover_price_patterns(sample_pattern_data)
        if patterns:
            pattern = patterns[0]
            # Предсказываем исход
            outcome = pattern_discovery.predict_pattern_outcomes(pattern, sample_pattern_data)
            assert isinstance(outcome, dict)
            assert "predicted_direction" in outcome
            assert "confidence" in outcome
            assert "time_horizon" in outcome
            assert "expected_movement" in outcome
            assert isinstance(outcome["predicted_direction"], str)
            assert isinstance(outcome["confidence"], float)
            assert isinstance(outcome["time_horizon"], int)
            assert isinstance(outcome["expected_movement"], float)
            assert outcome["predicted_direction"] in ["up", "down", "sideways"]
            assert outcome["confidence"] >= 0.0 and outcome["confidence"] <= 1.0
            assert outcome["time_horizon"] >= 0
    def test_predict_pattern_outcomes_empty_data(self, pattern_discovery) -> None:
        """Тест предсказания исходов паттернов с пустыми данными."""
        test_pattern = PatternResult(
            pattern_type="trend",
            start_time="2024-01-01T12:00:00Z",
            end_time="2024-01-01T13:00:00Z",
            confidence="medium",
            strength=0.7
        )
        empty_data = pd.DataFrame()
        outcome = pattern_discovery.predict_pattern_outcomes(test_pattern, empty_data)
        assert isinstance(outcome, dict)
        assert outcome["predicted_direction"] == "sideways"
        assert outcome["confidence"] == 0.0
        assert outcome["time_horizon"] == 0
        assert outcome["expected_movement"] == 0.0
    def test_pattern_discovery_error_handling(self, pattern_discovery) -> None:
        """Тест обработки ошибок в сервисе."""
        # Тест с None данными
        with pytest.raises(Exception):
            pattern_discovery.discover_price_patterns(None)
        # Тест с невалидным типом данных
        with pytest.raises(Exception):
            pattern_discovery.discover_volume_patterns("invalid_data")
        # Тест с невалидным паттерном
        with pytest.raises(Exception):
            pattern_discovery.validate_pattern("invalid_pattern", pd.DataFrame())
    def test_pattern_discovery_performance(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест производительности сервиса."""
        import time
        start_time = time.time()
        for _ in range(5):
            pattern_discovery.discover_all_patterns(sample_pattern_data)
        end_time = time.time()
        # Проверяем, что 5 операций выполняются менее чем за 3 секунды
        assert (end_time - start_time) < 3.0
    def test_pattern_discovery_thread_safety(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест потокобезопасности сервиса."""
        import threading
        import queue
        results = queue.Queue()
        def discover_patterns() -> Any:
            try:
                result = pattern_discovery.discover_price_patterns(sample_pattern_data)
                results.put(result)
            except Exception as e:
                results.put(e)
        # Запускаем несколько потоков одновременно
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=discover_patterns)
            threads.append(thread)
            thread.start()
        # Ждем завершения всех потоков
        for thread in threads:
            thread.join()
        # Проверяем, что все результаты корректны
        for _ in range(3):
            result = results.get()
            assert isinstance(result, list)
    def test_pattern_discovery_config_customization(self) -> None:
        """Тест кастомизации конфигурации сервиса."""
        custom_config = {
            "pattern_threshold": 0.8,
            "min_pattern_length": 5,
            "max_pattern_length": 50,
            "confidence_threshold": 0.7
        }
        service = PatternDiscovery(custom_config)
        assert service.config["pattern_threshold"] == 0.8
        assert service.config["min_pattern_length"] == 5
        assert service.config["max_pattern_length"] == 50
        assert service.config["confidence_threshold"] == 0.7
    def test_pattern_discovery_integration_with_different_patterns(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест интеграции с различными типами паттернов."""
        # Тестируем все типы паттернов
        price_patterns = pattern_discovery.discover_price_patterns(sample_pattern_data)
        volume_patterns = pattern_discovery.discover_volume_patterns(sample_pattern_data)
        technical_patterns = pattern_discovery.discover_technical_patterns(sample_pattern_data)
        candlestick_patterns = pattern_discovery.discover_candlestick_patterns(sample_pattern_data)
        
        # Проверяем, что все результаты корректны
        assert isinstance(price_patterns, list)
        assert isinstance(volume_patterns, list)
        assert isinstance(technical_patterns, list)
        assert isinstance(candlestick_patterns, list)
        
        # Проверяем, что все паттерны имеют правильную структуру
        for pattern in price_patterns + volume_patterns + technical_patterns + candlestick_patterns:
            # Исправляем использование isinstance с TypedDict на проверку типов
            assert hasattr(pattern, 'pattern_type')
            assert hasattr(pattern, 'confidence')
            assert hasattr(pattern, 'strength')
            assert isinstance(pattern.pattern_type, str)
            assert isinstance(pattern.confidence, str)
            assert isinstance(pattern.strength, (int, float))
    def test_pattern_discovery_data_consistency(self, pattern_discovery, sample_pattern_data) -> None:
        """Тест согласованности данных."""
        # Выполняем обнаружение паттернов несколько раз с одинаковыми данными
        results = []
        for _ in range(3):
            patterns = pattern_discovery.discover_price_patterns(sample_pattern_data)
            results.append(patterns)
        # Проверяем, что результаты согласованны
        for i in range(1, len(results)):
            assert len(results[i]) == len(results[0])
            # Проверяем, что типы паттернов согласованны
            for j in range(min(len(results[i]), len(results[0]))):
                assert results[i][j]["pattern_type"] == results[0][j]["pattern_type"] 
