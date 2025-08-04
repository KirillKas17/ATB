#!/usr/bin/env python3
"""
Интеграционные тесты для рабочего процесса символов.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from shared.numpy_utils import np
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal

from domain.symbols import (
    OrderBookMetricsData,
    PatternMetricsData,
    SessionMetricsData,
    SymbolProfile,
    MarketPhase,
)
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from application.symbol_selection.opportunity_selector import DynamicOpportunityAwareSymbolSelector
class TestSymbolsWorkflowIntegration:
    """Интеграционные тесты полного workflow анализа символов."""
    @pytest.fixture
    def workflow_components(self) -> Any:
        """Фикстура со всеми компонентами workflow."""
        return {
            'validator': SymbolValidator(),
            'classifier': MarketPhaseClassifier(),
            'calculator': OpportunityScoreCalculator(),
            'cache': MemorySymbolCache(default_ttl=300),
            'doass': DynamicOpportunityAwareSymbolSelector()
        }
    @pytest.fixture
    def realistic_market_data(self) -> Any:
        """Фикстура с реалистичными рыночными данными."""
        np.random.seed(42)
        n_periods = 100
        # Генерируем реалистичные OHLCV данные
        base_price = 50000
        returns = np.random.normal(0, 0.02, n_periods)  # 2% волатильность
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        data = []
        for i, price in enumerate(prices):
            # Генерируем OHLC на основе цены
            volatility = 0.01  # 1% внутридневная волатильность
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = price * (1 + np.random.normal(0, volatility/2))
            close_price = price * (1 + np.random.normal(0, volatility/2))
            # Генерируем объем
            base_volume = 1000
            volume = base_volume * (1 + np.random.normal(0, 0.3))
            data.append({
                'open': max(open_price, 0),
                'high': max(high, open_price, close_price),
                'low': max(low, 0),
                'close': max(close_price, 0),
                'volume': max(volume, 0)
            })
        return pd.DataFrame(data)
    @pytest.fixture
    def realistic_order_book(self) -> Any:
        """Фикстура с реалистичным стаканом заявок."""
        base_price = 50000
        spread = 0.001  # 0.1% спред
        bids = []
        asks = []
        for i in range(10):
            bid_price = base_price * (1 - spread/2 - i * 0.0001)
            ask_price = base_price * (1 + spread/2 + i * 0.0001)
            bid_volume = 100 * (1 + np.random.normal(0, 0.2))
            ask_volume = 100 * (1 + np.random.normal(0, 0.2))
            bids.append([bid_price, max(bid_volume, 0)])
            asks.append([ask_price, max(ask_volume, 0)])
        return {
            'bids': sorted(bids, key=lambda x: x[0], reverse=True),
            'asks': sorted(asks, key=lambda x: x[0])
        }
    def test_complete_symbol_analysis_workflow(self, workflow_components, 
                                             realistic_market_data, 
                                             realistic_order_book) -> None:
        """Тест полного workflow анализа символа."""
        symbol = "BTCUSDT"
        components = workflow_components
        # Этап 1: Валидация входных данных
        assert components['validator'].validate_symbol(symbol) is True
        assert components['validator'].validate_ohlcv_data(realistic_market_data) is True
        assert components['validator'].validate_order_book(realistic_order_book) is True
        # Этап 2: Классификация рыночной фазы
        with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
            mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
            phase_result = components['classifier'].classify_market_phase(realistic_market_data)
            assert phase_result.phase == MarketPhase.BREAKOUT_ACTIVE
            assert phase_result.confidence == 0.8
        # Этап 3: Расчет оценки возможностей
        with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
            mock_calc.return_value = 0.75
            opportunity_score = components['calculator'].calculate_opportunity_score(
                symbol, realistic_market_data, realistic_order_book
            )
            assert opportunity_score == 0.75
        # Этап 4: Создание профиля символа
        profile = SymbolProfile(
            symbol=symbol,
            opportunity_score=opportunity_score,
            market_phase=phase_result.phase,
            confidence=ConfidenceValue(phase_result.confidence),
            volume_profile=VolumeProfile(current_volume=VolumeValue(realistic_market_data['volume'].iloc[-1]), volume_trend=VolumeValue(0.1), volume_stability=ConfidenceValue(0.8)),
            price_structure=PriceStructure(current_price=PriceValue(realistic_market_data['close'].iloc[-1]), atr=ATRValue(1.0), atr_percent=0.01, vwap=VWAPValue(realistic_market_data['close'].mean()), vwap_deviation=0.001, price_entropy=EntropyValue(0.3), volatility_compression=VolatilityValue(0.2)),
            order_book_metrics=OrderBookMetrics(bid_ask_spread=SpreadValue(0.1), spread_percent=0.001, bid_volume=VolumeValue(100.0), ask_volume=VolumeValue(120.0), volume_imbalance=-0.1, order_book_symmetry=0.8, liquidity_depth=0.9, absorption_ratio=0.7),
            pattern_metrics=PatternMetrics(mirror_neuron_score=0.6, gravity_anomaly_score=0.5, reversal_setup_score=0.4, pattern_confidence=PatternConfidenceValue(0.7), historical_pattern_match=0.6, pattern_complexity=0.5),
            session_metrics=SessionMetrics(session_alignment=SessionAlignmentValue(0.8), session_activity=0.7, session_volatility=VolatilityValue(0.15), session_momentum=MomentumValue(0.6), session_influence_score=0.7)
        )
        # Этап 5: Кэширование профиля
        components['cache'].set_profile(symbol, profile)
        cached_profile = components['cache'].get_profile(symbol)
        assert cached_profile == profile
        # Этап 6: Использование профиля в DOASS
        with patch.object(components['doass'], 'get_detailed_analysis') as mock_doass:
            mock_doass.return_value = Mock(
                selected_symbols=[symbol],
                detailed_profiles={symbol: profile},
                total_symbols_analyzed=1,
                processing_time_ms=100.0,
                cache_hit_rate=0.8
            )
            result = components['doass'].get_detailed_analysis(limit=5)
            assert symbol in result.selected_symbols
            assert result.detailed_profiles[symbol] == profile
    def test_symbol_analysis_with_market_regime_changes(self, workflow_components,
                                                      realistic_market_data,
                                                      realistic_order_book) -> None:
        """Тест анализа символа при смене рыночного режима."""
        symbol = "ETHUSDT"
        components = workflow_components
        # Симулируем разные рыночные фазы
        market_phases = [MarketPhase.BREAKOUT_ACTIVE, MarketPhase.EXHAUSTION, MarketPhase.ACCUMULATION]
        for phase in market_phases:
            with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
                mock_classify.return_value = Mock(phase=phase, confidence=0.7)
                with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                    # Разные оценки возможностей для разных фаз
                    if phase == MarketPhase.BREAKOUT_ACTIVE:
                        mock_calc.return_value = 0.8
                    elif phase == MarketPhase.EXHAUSTION:
                        mock_calc.return_value = 0.3
                    else:
                        mock_calc.return_value = 0.5
                    profile = SymbolProfile(
                        symbol=symbol,
                        opportunity_score=mock_calc.return_value,
                        market_phase=phase,
                        confidence=0.7,
                        volume=1000.0,
                        spread=0.001,
                        volatility=0.15,
                        price_structure=PriceStructure(),
                        volume_profile=VolumeProfile(),
                        orderbook_metrics=OrderBookMetrics(),
                        pattern_metrics=PatternMetrics(),
                        session_metrics=SessionMetrics()
                    )
                    # Кэшируем профиль
                    components['cache'].set_profile(symbol, profile)
                    cached_profile = components['cache'].get_profile(symbol)
                    # Проверяем соответствие фазы и оценки
                    assert cached_profile.market_phase == phase
                    if phase == MarketPhase.BREAKOUT_ACTIVE:
                        assert cached_profile.opportunity_score > 0.7
                    elif phase == MarketPhase.EXHAUSTION:
                        assert cached_profile.opportunity_score < 0.5
    def test_symbol_analysis_performance_under_load(self, workflow_components) -> None:
        """Тест производительности анализа символов под нагрузкой."""
        components = workflow_components
        symbols = [f"SYMBOL{i}" for i in range(50)]
        import time
        start_time = time.time()
        # Анализируем множество символов
        for symbol in symbols:
            with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
                mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.7)
                with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                    mock_calc.return_value = 0.6
                    profile = SymbolProfile(
                        symbol=symbol,
                        opportunity_score=0.6,
                        market_phase=MarketPhase.BREAKOUT_ACTIVE,
                        confidence=0.7,
                        volume=1000.0,
                        spread=0.001,
                        volatility=0.15,
                        price_structure=PriceStructure(),
                        volume_profile=VolumeProfile(),
                        orderbook_metrics=OrderBookMetrics(),
                        pattern_metrics=PatternMetrics(),
                        session_metrics=SessionMetrics()
                    )
                    components['cache'].set_profile(symbol, profile)
        processing_time = time.time() - start_time
        # Проверяем производительность
        assert processing_time < 5.0  # Обработка 50 символов менее 5 секунд
        # Проверяем, что все символы в кэше
        for symbol in symbols:
            assert components['cache'].get_profile(symbol) is not None
    def test_symbol_analysis_error_recovery(self, workflow_components,
                                          realistic_market_data,
                                          realistic_order_book) -> None:
        """Тест восстановления после ошибок в workflow."""
        symbol = "BTCUSDT"
        components = workflow_components
        # Симулируем ошибку в классификаторе
        with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
            mock_classify.side_effect = Exception("Classification failed")
            # Проверяем, что ошибка обрабатывается корректно
            with pytest.raises(Exception):
                components['classifier'].classify_market_phase(realistic_market_data)
        # Симулируем ошибку в калькуляторе
        with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
            mock_calc.side_effect = Exception("Calculation failed")
            with pytest.raises(Exception):
                components['calculator'].calculate_opportunity_score(
                    symbol, realistic_market_data, realistic_order_book
                )
        # Проверяем, что кэш остается стабильным
        assert components['cache'].get_profile(symbol) is None
    def test_symbol_analysis_data_persistence(self, workflow_components,
                                            realistic_market_data,
                                            realistic_order_book) -> None:
        """Тест персистентности данных в workflow."""
        symbol = "BTCUSDT"
        components = workflow_components
        # Создаем профиль
        with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
            mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
            with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                mock_calc.return_value = 0.75
                profile = SymbolProfile(
                    symbol=symbol,
                    opportunity_score=0.75,
                    market_phase=MarketPhase.BREAKOUT_ACTIVE,
                    confidence=0.8,
                    volume=1000.0,
                    spread=0.001,
                    volatility=0.15,
                    price_structure=PriceStructure(),
                    volume_profile=VolumeProfile(),
                    orderbook_metrics=OrderBookMetrics(),
                    pattern_metrics=PatternMetrics(),
                    session_metrics=SessionMetrics()
                )
                # Кэшируем профиль
                components['cache'].set_profile(symbol, profile)
                # Проверяем персистентность
                cached_profile = components['cache'].get_profile(symbol)
                assert cached_profile.symbol == symbol
                assert cached_profile.opportunity_score == 0.75
                assert cached_profile.market_phase == MarketPhase.BREAKOUT_ACTIVE
                assert cached_profile.confidence == 0.8
                # Проверяем, что данные не изменяются при повторном чтении
                cached_profile2 = components['cache'].get_profile(symbol)
                assert cached_profile2 == cached_profile
    def test_symbol_analysis_integration_with_external_systems(self, workflow_components) -> None:
        """Тест интеграции с внешними системами."""
        symbol = "BTCUSDT"
        components = workflow_components
        # Симулируем интеграцию с внешней системой данных
        external_data_provider = Mock()
        external_data_provider.get_market_data.return_value = pd.DataFrame({
            'open': [50000, 50100, 50200],
            'high': [50200, 50300, 50400],
            'low': [49900, 50000, 50100],
            'close': [50100, 50200, 50300],
            'volume': [1000, 1100, 1200]
        })
        external_data_provider.get_order_book.return_value = {
            'bids': [[50000, 10], [49999, 15]],
            'asks': [[50001, 12], [50002, 18]]
        }
        # Получаем данные из внешней системы
        market_data = external_data_provider.get_market_data(symbol)
        order_book = external_data_provider.get_order_book(symbol)
        # Валидируем данные
        assert components['validator'].validate_ohlcv_data(market_data) is True
        assert components['validator'].validate_order_book(order_book) is True
        # Анализируем данные
        with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
            mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
            with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                mock_calc.return_value = 0.75
                profile = SymbolProfile(
                    symbol=symbol,
                    opportunity_score=0.75,
                    market_phase=MarketPhase.BREAKOUT_ACTIVE,
                    confidence=0.8,
                    volume=1200.0,
                    spread=0.0002,
                    volatility=0.15,
                    price_structure=PriceStructure(),
                    volume_profile=VolumeProfile(),
                    orderbook_metrics=OrderBookMetrics(),
                    pattern_metrics=PatternMetrics(),
                    session_metrics=SessionMetrics()
                )
                # Кэшируем результат
                components['cache'].set_profile(symbol, profile)
                # Проверяем интеграцию
                cached_profile = components['cache'].get_profile(symbol)
                assert cached_profile is not None
                assert cached_profile.symbol == symbol
if __name__ == "__main__":
    pytest.main([__file__]) 
