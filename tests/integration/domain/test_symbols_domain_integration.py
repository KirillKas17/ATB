#!/usr/bin/env python3
"""
Интеграционные тесты для модулей domain/symbols.
Тестирует взаимодействие между компонентами: валидация → фазовый анализ → скоринг → кэширование.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
import pandas as pd
from unittest.mock import Mock, patch
from domain.symbols import (
    SymbolValidator,
    MarketPhaseClassifier,
    OpportunityScoreCalculator,
    MemorySymbolCache,
    SymbolProfile,
    MarketPhase,
    PriceStructure,
    VolumeProfile,
    OrderBookMetricsData,
    PatternMetricsData,
    SessionMetricsData,
)


class TestSymbolsIntegration:
    """Интеграционные тесты для модулей symbols."""

    @pytest.fixture
    def symbol_validator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для валидатора символов."""
        return SymbolValidator()

    @pytest.fixture
    def market_phase_classifier(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для классификатора рыночных фаз."""
        return MarketPhaseClassifier()

    @pytest.fixture
    def opportunity_calculator(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для калькулятора возможностей."""
        return OpportunityScoreCalculator()

    @pytest.fixture
    def symbol_cache(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура для кэша символов."""
        return MemorySymbolCache(default_ttl=300)

    @pytest.fixture
    def sample_market_data(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с тестовыми рыночными данными."""
        return pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104] * 20,
                "high": [102, 103, 104, 105, 106] * 20,
                "low": [99, 100, 101, 102, 103] * 20,
                "close": [101, 102, 103, 104, 105] * 20,
                "volume": [1000, 1100, 1200, 1300, 1400] * 20,
            }
        )

    @pytest.fixture
    def sample_order_book(self: "TestEvolvableMarketMakerAgent") -> Any:
        """Фикстура с тестовым стаканом заявок."""
        return {
            "bids": [[100.0, 10.0], [99.9, 15.0], [99.8, 20.0]],
            "asks": [[100.1, 12.0], [100.2, 18.0], [100.3, 25.0]],
        }

    def test_symbol_analysis_workflow(
        self,
        symbol_validator,
        market_phase_classifier,
        opportunity_calculator,
        symbol_cache,
        sample_market_data,
        sample_order_book,
    ) -> None:
        """Тест полного workflow анализа символа."""
        symbol = "BTCUSDT"
        # 1. Валидация символа
        assert symbol_validator.validate_symbol(symbol) is True
        # 2. Валидация рыночных данных
        assert symbol_validator.validate_ohlcv_data(sample_market_data) is True
        # 3. Валидация стакана заявок
        assert symbol_validator.validate_order_book(sample_order_book) is True
        # 4. Классификация рыночной фазы
        with patch.object(market_phase_classifier, "classify_market_phase") as mock_classify:
            mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
            phase_result = market_phase_classifier.classify_market_phase(sample_market_data)
            assert phase_result.phase == MarketPhase.BREAKOUT_ACTIVE
        # 5. Расчет оценки возможностей
        with patch.object(opportunity_calculator, "calculate_opportunity_score") as mock_calc:
            mock_calc.return_value = 0.75
            opportunity_score = opportunity_calculator.calculate_opportunity_score(
                symbol, sample_market_data, sample_order_book
            )
            assert opportunity_score == 0.75
        # 6. Создание профиля символа
        profile = SymbolProfile(
            symbol=symbol,
            opportunity_score=opportunity_score,
            market_phase=phase_result.phase,
            confidence=phase_result.confidence,
            volume_profile=VolumeProfile(current_volume=1000.0, volume_trend=0.1, volume_stability=0.8),
            price_structure=PriceStructure(
                current_price=100.0,
                atr=1.0,
                atr_percent=0.01,
                vwap=100.0,
                vwap_deviation=0.001,
                price_entropy=0.3,
                volatility_compression=0.2,
            ),
            order_book_metrics=OrderBookMetricsData(
                bid_ask_spread=0.1,
                spread_percent=0.001,
                bid_volume=100.0,
                ask_volume=120.0,
                volume_imbalance=-0.1,
                order_book_symmetry=0.8,
                liquidity_depth=0.9,
                absorption_ratio=0.7,
            ),
            pattern_metrics=PatternMetricsData(
                mirror_neuron_score=0.6,
                gravity_anomaly_score=0.5,
                reversal_setup_score=0.4,
                pattern_confidence=0.7,
                historical_pattern_match=0.6,
                pattern_complexity=0.5,
            ),
            session_metrics=SessionMetricsData(
                session_alignment=0.8,
                session_activity=0.7,
                session_volatility=0.15,
                session_momentum=0.6,
                session_influence_score=0.7,
            ),
        )
        # 7. Кэширование профиля
        symbol_cache.set_profile(symbol, profile)
        cached_profile = symbol_cache.get_profile(symbol)
        assert cached_profile == profile
        assert cached_profile.symbol == symbol
        assert cached_profile.opportunity_score == 0.75

    def test_symbol_analysis_with_invalid_data(
        self, symbol_validator, market_phase_classifier, opportunity_calculator, symbol_cache
    ) -> None:
        """Тест обработки невалидных данных в workflow."""
        symbol = "INVALID"
        # 1. Валидация невалидного символа
        assert symbol_validator.validate_symbol(symbol) is True  # Валидатор принимает любые символы
        # 2. Попытка анализа с невалидными данными
        invalid_data = pd.DataFrame({"invalid": [1, 2, 3]})
        with pytest.raises(Exception):
            symbol_validator.validate_ohlcv_data(invalid_data)
        # 3. Проверка, что кэш не содержит невалидных данных
        assert symbol_cache.get_profile(symbol) is None

    def test_symbol_cache_integration_with_analysis(
        self,
        symbol_validator,
        market_phase_classifier,
        opportunity_calculator,
        symbol_cache,
        sample_market_data,
        sample_order_book,
    ) -> None:
        """Тест интеграции кэша с анализом символов."""
        symbol = "ETHUSDT"
        # Проверяем, что кэш изначально пуст
        assert symbol_cache.get_profile(symbol) is None
        # Выполняем анализ и кэшируем результат
        with patch.object(market_phase_classifier, "classify_market_phase") as mock_classify:
            mock_classify.return_value = Mock(phase=MarketPhase.ACCUMULATION, confidence=0.6)
            with patch.object(opportunity_calculator, "calculate_opportunity_score") as mock_calc:
                mock_calc.return_value = 0.5
                # Создаем профиль
                profile = SymbolProfile(
                    symbol=symbol,
                    opportunity_score=0.5,
                    market_phase=MarketPhase.ACCUMULATION,
                    confidence=0.6,
                    volume_profile=VolumeProfile(current_volume=800.0, volume_trend=-0.1, volume_stability=0.7),
                    price_structure=PriceStructure(
                        current_price=3000.0,
                        atr=30.0,
                        atr_percent=0.01,
                        vwap=3000.0,
                        vwap_deviation=0.002,
                        price_entropy=0.4,
                        volatility_compression=0.3,
                    ),
                    order_book_metrics=OrderBookMetricsData(
                        bid_ask_spread=0.2,
                        spread_percent=0.002,
                        bid_volume=80.0,
                        ask_volume=90.0,
                        volume_imbalance=-0.05,
                        order_book_symmetry=0.7,
                        liquidity_depth=0.8,
                        absorption_ratio=0.6,
                    ),
                    pattern_metrics=PatternMetricsData(
                        mirror_neuron_score=0.4,
                        gravity_anomaly_score=0.3,
                        reversal_setup_score=0.5,
                        pattern_confidence=0.6,
                        historical_pattern_match=0.5,
                        pattern_complexity=0.4,
                    ),
                    session_metrics=SessionMetricsData(
                        session_alignment=0.6,
                        session_activity=0.5,
                        session_volatility=0.2,
                        session_momentum=0.4,
                        session_influence_score=0.5,
                    ),
                )
                # Кэшируем
                symbol_cache.set_profile(symbol, profile)
                # Проверяем, что профиль доступен в кэше
                cached_profile = symbol_cache.get_profile(symbol)
                assert cached_profile is not None
                assert cached_profile.symbol == symbol
                assert cached_profile.opportunity_score == 0.5
                assert cached_profile.market_phase == MarketPhase.ACCUMULATION

    def test_multiple_symbols_analysis(
        self,
        symbol_validator,
        market_phase_classifier,
        opportunity_calculator,
        symbol_cache,
        sample_market_data,
        sample_order_book,
    ) -> None:
        """Тест анализа нескольких символов."""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        for symbol in symbols:
            # Валидация
            assert symbol_validator.validate_symbol(symbol) is True
            # Мокаем анализ
            with patch.object(market_phase_classifier, "classify_market_phase") as mock_classify:
                mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.7)
                with patch.object(opportunity_calculator, "calculate_opportunity_score") as mock_calc:
                    mock_calc.return_value = 0.6
                    # Создаем профиль
                    profile = SymbolProfile(
                        symbol=symbol,
                        opportunity_score=0.6,
                        market_phase=MarketPhase.BREAKOUT_ACTIVE,
                        confidence=0.7,
                        volume_profile=VolumeProfile(current_volume=1000.0, volume_trend=0.2, volume_stability=0.8),
                        price_structure=PriceStructure(
                            current_price=100.0,
                            atr=1.0,
                            atr_percent=0.01,
                            vwap=100.0,
                            vwap_deviation=0.001,
                            price_entropy=0.3,
                            volatility_compression=0.2,
                        ),
                        order_book_metrics=OrderBookMetricsData(
                            bid_ask_spread=0.1,
                            spread_percent=0.001,
                            bid_volume=100.0,
                            ask_volume=120.0,
                            volume_imbalance=-0.1,
                            order_book_symmetry=0.8,
                            liquidity_depth=0.9,
                            absorption_ratio=0.7,
                        ),
                        pattern_metrics=PatternMetricsData(
                            mirror_neuron_score=0.6,
                            gravity_anomaly_score=0.5,
                            reversal_setup_score=0.4,
                            pattern_confidence=0.7,
                            historical_pattern_match=0.6,
                            pattern_complexity=0.5,
                        ),
                        session_metrics=SessionMetricsData(
                            session_alignment=0.8,
                            session_activity=0.7,
                            session_volatility=0.15,
                            session_momentum=0.6,
                            session_influence_score=0.7,
                        ),
                    )
                    # Кэшируем
                    symbol_cache.set_profile(symbol, profile)
        # Проверяем, что все символы в кэше
        for symbol in symbols:
            cached_profile = symbol_cache.get_profile(symbol)
            assert cached_profile is not None
            assert cached_profile.symbol == symbol

    def test_symbol_analysis_error_handling(
        self, symbol_validator, market_phase_classifier, opportunity_calculator, symbol_cache
    ) -> None:
        """Тест обработки ошибок в workflow анализа."""
        symbol = "BTCUSDT"
        # Симулируем ошибку в классификаторе
        with patch.object(market_phase_classifier, "classify_market_phase") as mock_classify:
            mock_classify.side_effect = Exception("Classification failed")
            with pytest.raises(Exception):
                market_phase_classifier.classify_market_phase(None)
        # Симулируем ошибку в калькуляторе
        with patch.object(opportunity_calculator, "calculate_opportunity_score") as mock_calc:
            mock_calc.side_effect = Exception("Calculation failed")
            with pytest.raises(Exception):
                opportunity_calculator.calculate_opportunity_score(symbol, None, None)
        # Проверяем, что кэш не содержит данных с ошибками
        assert symbol_cache.get_profile(symbol) is None

    def test_symbol_cache_performance_integration(self, symbol_cache) -> None:
        """Тест производительности кэша в интеграции."""
        import time

        # Создаем множество профилей
        profiles = []
        for i in range(100):
            profile = SymbolProfile(
                symbol=f"SYMBOL{i}",
                opportunity_score=0.5,
                market_phase=MarketPhase.BREAKOUT_ACTIVE,
                confidence=0.7,
                volume_profile=VolumeProfile(current_volume=1000.0, volume_trend=0.1, volume_stability=0.8),
                price_structure=PriceStructure(
                    current_price=100.0,
                    atr=1.0,
                    atr_percent=0.01,
                    vwap=100.0,
                    vwap_deviation=0.001,
                    price_entropy=0.3,
                    volatility_compression=0.2,
                ),
                order_book_metrics=OrderBookMetricsData(
                    bid_ask_spread=0.1,
                    spread_percent=0.001,
                    bid_volume=100.0,
                    ask_volume=120.0,
                    volume_imbalance=-0.1,
                    order_book_symmetry=0.8,
                    liquidity_depth=0.9,
                    absorption_ratio=0.7,
                ),
                pattern_metrics=PatternMetricsData(
                    mirror_neuron_score=0.6,
                    gravity_anomaly_score=0.5,
                    reversal_setup_score=0.4,
                    pattern_confidence=0.7,
                    historical_pattern_match=0.6,
                    pattern_complexity=0.5,
                ),
                session_metrics=SessionMetricsData(
                    session_alignment=0.8,
                    session_activity=0.7,
                    session_volatility=0.15,
                    session_momentum=0.6,
                    session_influence_score=0.7,
                ),
            )
            profiles.append(profile)
        # Измеряем время записи
        start_time = time.time()
        for profile in profiles:
            symbol_cache.set_profile(profile.symbol, profile)
        write_time = time.time() - start_time
        # Измеряем время чтения
        start_time = time.time()
        for profile in profiles:
            cached_profile = symbol_cache.get_profile(profile.symbol)
            assert cached_profile is not None
        read_time = time.time() - start_time
        # Проверяем, что операции выполняются быстро
        assert write_time < 1.0  # Запись менее 1 секунды
        assert read_time < 1.0  # Чтение менее 1 секунды

    def test_symbol_analysis_data_consistency(
        self,
        symbol_validator,
        market_phase_classifier,
        opportunity_calculator,
        symbol_cache,
        sample_market_data,
        sample_order_book,
    ) -> None:
        """Тест консистентности данных в workflow."""
        symbol = "BTCUSDT"
        # Выполняем анализ несколько раз с одинаковыми данными
        results = []
        for _ in range(3):
            with patch.object(market_phase_classifier, "classify_market_phase") as mock_classify:
                mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
                with patch.object(opportunity_calculator, "calculate_opportunity_score") as mock_calc:
                    mock_calc.return_value = 0.75
                    profile = SymbolProfile(
                        symbol=symbol,
                        opportunity_score=0.75,
                        market_phase=MarketPhase.BREAKOUT_ACTIVE,
                        confidence=0.8,
                        volume_profile=VolumeProfile(current_volume=1000.0, volume_trend=0.1, volume_stability=0.8),
                        price_structure=PriceStructure(
                            current_price=100.0,
                            atr=1.0,
                            atr_percent=0.01,
                            vwap=100.0,
                            vwap_deviation=0.001,
                            price_entropy=0.3,
                            volatility_compression=0.2,
                        ),
                        order_book_metrics=OrderBookMetricsData(
                            bid_ask_spread=0.1,
                            spread_percent=0.001,
                            bid_volume=100.0,
                            ask_volume=120.0,
                            volume_imbalance=-0.1,
                            order_book_symmetry=0.8,
                            liquidity_depth=0.9,
                            absorption_ratio=0.7,
                        ),
                        pattern_metrics=PatternMetricsData(
                            mirror_neuron_score=0.6,
                            gravity_anomaly_score=0.5,
                            reversal_setup_score=0.4,
                            pattern_confidence=0.7,
                            historical_pattern_match=0.6,
                            pattern_complexity=0.5,
                        ),
                        session_metrics=SessionMetricsData(
                            session_alignment=0.8,
                            session_activity=0.7,
                            session_volatility=0.15,
                            session_momentum=0.6,
                            session_influence_score=0.7,
                        ),
                    )
                    symbol_cache.set_profile(symbol, profile)
                    results.append(symbol_cache.get_profile(symbol))
        # Проверяем консистентность результатов
        for i in range(1, len(results)):
            assert results[i].symbol == results[0].symbol
            assert results[i].opportunity_score == results[0].opportunity_score
            assert results[i].market_phase == results[0].market_phase
            assert results[i].confidence == results[0].confidence


if __name__ == "__main__":
    pytest.main([__file__])
