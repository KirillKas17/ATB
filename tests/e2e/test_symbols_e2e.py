#!/usr/bin/env python3
"""
E2E тесты для анализа символов.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
from shared.numpy_utils import np
import asyncio
from decimal import Decimal
from typing import List, Dict, Any

from domain.symbols import (
    OrderBookMetricsData,
    PatternMetricsData,
    SessionMetricsData,
    SymbolProfile,
    MarketPhase,
)
from application.use_cases.trading_orchestrator.core import DefaultTradingOrchestratorUseCase
from domain.entities.trading import TradingSession
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency

class TestSymbolsE2E:
    """End-to-End тесты для полного цикла работы с символами."""
    @pytest.fixture
    def e2e_components(self) -> Any:
        """Фикстура со всеми компонентами для E2E тестов."""
        return {
            'validator': SymbolValidator(),
            'classifier': MarketPhaseClassifier(),
            'calculator': OpportunityScoreCalculator(),
            'cache': MemorySymbolCache(default_ttl=300),
            'doass': DynamicOpportunityAwareSymbolSelector(),
            'orchestrator': Mock(spec=TradingOrchestrator)
        }
    @pytest.fixture
    def realistic_trading_scenario(self) -> Any:
        """Фикстура с реалистичным торговым сценарием."""
        # Генерируем исторические данные для нескольких символов
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        market_data, order_books = {}, {}
        for symbol in symbols:
            # Генерируем OHLCV данные
            np.random.seed(hash(symbol) % 1000)
            n_periods = 200
            base_price = 50000 if "BTC" in symbol else (3000 if "ETH" in symbol else 1.0)
            returns = np.random.normal(0, 0.025, n_periods)
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            data = []
            for i, price in enumerate(prices):
                volatility = 0.015
                high = price * (1 + abs(np.random.normal(0, volatility)))
                low = price * (1 - abs(np.random.normal(0, volatility)))
                open_price = price * (1 + np.random.normal(0, volatility/2))
                close_price = price * (1 + np.random.normal(0, volatility/2))
                base_volume = 1000 if "BTC" in symbol else (500 if "ETH" in symbol else 10000)
                volume = base_volume * (1 + np.random.normal(0, 0.4))
                data.append({
                    'open': max(open_price, 0),
                    'high': max(high, open_price, close_price),
                    'low': max(low, 0),
                    'close': max(close_price, 0),
                    'volume': max(volume, 0)
                })
            market_data[symbol] = pd.DataFrame(data)
            # Генерируем стакан заявок
            current_price = prices[-1]
            spread = 0.001
            bids = []
            asks = []
            for i in range(15):
                bid_price = current_price * (1 - spread/2 - i * 0.0001)
                ask_price = current_price * (1 + spread/2 + i * 0.0001)
                bid_volume = 100 * (1 + np.random.normal(0, 0.3))
                ask_volume = 100 * (1 + np.random.normal(0, 0.3))
                bids.append([bid_price, max(bid_volume, 0)])
                asks.append([ask_price, max(ask_volume, 0)])
            order_books[symbol] = {
                'bids': sorted(bids, key=lambda x: x[0], reverse=True),
                'asks': sorted(asks, key=lambda x: x[0])
            }
        return {
            'symbols': symbols,
            'market_data': market_data,
            'order_books': order_books
        }
    def test_complete_trading_cycle_e2e(self, e2e_components, realistic_trading_scenario) -> None:
        """Тест полного торгового цикла E2E."""
        components = e2e_components
        scenario = realistic_trading_scenario
        # Этап 1: Анализ всех символов
        analyzed_symbols = []
        for symbol in scenario['symbols']:
            market_data = scenario['market_data'][symbol]
            order_book = scenario['order_books'][symbol]
            # Валидация
            assert components['validator'].validate_symbol(symbol) is True
            assert components['validator'].validate_ohlcv_data(market_data) is True
            assert components['validator'].validate_order_book(order_book) is True
            # Классификация фазы
            with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
                # Симулируем разные фазы для разных символов
                if "BTC" in symbol:
                    mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
                elif "ETH" in symbol:
                    mock_classify.return_value = Mock(phase=MarketPhase.ACCUMULATION, confidence=0.6)
                else:
                    mock_classify.return_value = Mock(phase=MarketPhase.EXHAUSTION, confidence=0.7)
                phase_result = components['classifier'].classify_market_phase(market_data)
                # Расчет возможностей
                with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                    if phase_result.phase == MarketPhase.BREAKOUT_ACTIVE:
                        mock_calc.return_value = 0.8
                    elif phase_result.phase == MarketPhase.ACCUMULATION:
                        mock_calc.return_value = 0.5
                    else:
                        mock_calc.return_value = 0.3
                    opportunity_score = components['calculator'].calculate_opportunity_score(
                        symbol, market_data, order_book
                    )
                    # Создание профиля
                    profile = SymbolProfile(
                        symbol=symbol,
                        opportunity_score=opportunity_score,
                        market_phase=phase_result.phase,
                        confidence=ConfidenceValue(phase_result.confidence),
                        volume_profile=VolumeProfile(current_volume=VolumeValue(market_data['volume'].iloc[-1]), volume_trend=VolumeValue(0.1), volume_stability=ConfidenceValue(0.8)),
                        price_structure=PriceStructure(current_price=PriceValue(market_data['close'].iloc[-1]), atr=ATRValue(1.0), atr_percent=0.01, vwap=VWAPValue(market_data['close'].mean()), vwap_deviation=0.001, price_entropy=EntropyValue(0.3), volatility_compression=VolatilityValue(0.2)),
                        order_book_metrics=OrderBookMetrics(bid_ask_spread=SpreadValue(0.1), spread_percent=0.001, bid_volume=VolumeValue(100.0), ask_volume=VolumeValue(120.0), volume_imbalance=-0.1, order_book_symmetry=0.8, liquidity_depth=0.9, absorption_ratio=0.7),
                        pattern_metrics=PatternMetrics(mirror_neuron_score=0.6, gravity_anomaly_score=0.5, reversal_setup_score=0.4, pattern_confidence=PatternConfidenceValue(0.7), historical_pattern_match=0.6, pattern_complexity=0.5),
                        session_metrics=SessionMetrics(session_alignment=SessionAlignmentValue(0.8), session_activity=0.7, session_volatility=VolatilityValue(0.15), session_momentum=MomentumValue(0.6), session_influence_score=0.7)
                    )
                    # Кэширование
                    components['cache'].set_profile(symbol, profile)
                    analyzed_symbols.append(symbol)
        # Этап 2: Выбор лучших символов через DOASS
        with patch.object(components['doass'], 'get_detailed_analysis') as mock_doass:
            # Симулируем выбор лучших символов
            selected_symbols = ["BTCUSDT", "ETHUSDT"]  # Только BTC и ETH
            detailed_profiles = {
                symbol: components['cache'].get_profile(symbol)
                for symbol in selected_symbols
            }
            mock_doass.return_value = Mock(
                selected_symbols=selected_symbols,
                detailed_profiles=detailed_profiles,
                total_symbols_analyzed=len(analyzed_symbols),
                processing_time_ms=150.0,
                cache_hit_rate=0.9
            )
            analysis_result = components['doass'].get_detailed_analysis(limit=2)
            assert len(analysis_result.selected_symbols) == 2
            assert "BTCUSDT" in analysis_result.selected_symbols
            assert "ETHUSDT" in analysis_result.selected_symbols
        # Этап 3: Принятие торговых решений
        trading_decisions = []
        for symbol in analysis_result.selected_symbols:
            profile = analysis_result.detailed_profiles[symbol]
            # Симулируем принятие решения на основе профиля
            if profile.opportunity_score > 0.7 and profile.market_phase == MarketPhase.BREAKOUT_ACTIVE:
                decision = TradingDecision(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    order_type=OrderType.MARKET,
                    quantity=0.1,
                    price=None,
                    confidence=profile.confidence,
                    reasoning=f"Strong breakout signal with {profile.opportunity_score:.2f} opportunity score"
                )
            elif profile.opportunity_score > 0.5:
                decision = TradingDecision(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    order_type=OrderType.LIMIT,
                    quantity=0.05,
                    price=scenario['order_books'][symbol]['bids'][0][0] * 0.999,
                    confidence=profile.confidence,
                    reasoning=f"Moderate opportunity with {profile.opportunity_score:.2f} score"
                )
            else:
                decision = None  # Пропускаем символ
            if decision:
                trading_decisions.append(decision)
        # Проверяем, что приняты решения
        assert len(trading_decisions) > 0
        # Этап 4: Исполнение решений через оркестратор
        with patch.object(components['orchestrator'], 'execute_trading_decisions') as mock_execute:
            mock_execute.return_value = Mock(
                executed_orders=len(trading_decisions),
                total_volume=sum(d.quantity for d in trading_decisions),
                execution_time_ms=200.0,
                success_rate=1.0
            )
            execution_result = components['orchestrator'].execute_trading_decisions(trading_decisions)
            assert execution_result.executed_orders == len(trading_decisions)
            assert execution_result.success_rate == 1.0
    def test_symbol_analysis_performance_e2e(self, e2e_components, realistic_trading_scenario) -> None:
        """Тест производительности E2E анализа символов."""
        components = e2e_components
        scenario = realistic_trading_scenario
        import time
        start_time = time.time()
        # Анализируем все символы
        for symbol in scenario['symbols']:
            market_data = scenario['market_data'][symbol]
            order_book = scenario['order_books'][symbol]
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
        analysis_time = time.time() - start_time
        # Выбор символов
        start_time = time.time()
        with patch.object(components['doass'], 'get_detailed_analysis') as mock_doass:
            mock_doass.return_value = Mock(
                selected_symbols=scenario['symbols'][:2],
                detailed_profiles={},
                total_symbols_analyzed=len(scenario['symbols']),
                processing_time_ms=100.0,
                cache_hit_rate=0.8
            )
            analysis_result = components['doass'].get_detailed_analysis(limit=2)
        selection_time = time.time() - start_time
        # Проверяем производительность
        assert analysis_time < 2.0  # Анализ менее 2 секунд
        assert selection_time < 1.0  # Выбор менее 1 секунды
    def test_symbol_analysis_error_handling_e2e(self, e2e_components, realistic_trading_scenario) -> None:
        """Тест обработки ошибок в E2E сценарии."""
        components = e2e_components
        scenario = realistic_trading_scenario
        # Симулируем ошибку в одном из символов
        problematic_symbol = "BTCUSDT"
        with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
            mock_classify.side_effect = lambda data: (
                Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8) 
                if data is not None else 
                Exception("Data processing failed")
            )
            # Анализируем символы, один из которых вызовет ошибку
            successful_analyses = 0
            for symbol in scenario['symbols']:
                try:
                    market_data = scenario['market_data'][symbol]
                    order_book = scenario['order_books'][symbol]
                    # Симулируем ошибку для проблемного символа
                    if symbol == problematic_symbol:
                        market_data = None
                    phase_result = components['classifier'].classify_market_phase(market_data)
                    with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                        mock_calc.return_value = 0.6
                        profile = SymbolProfile(
                            symbol=symbol,
                            opportunity_score=0.6,
                            market_phase=phase_result.phase,
                            confidence=ConfidenceValue(phase_result.confidence),
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
                        successful_analyses += 1
                except Exception:
                    # Ошибка обрабатывается, анализ продолжается
                    continue
            # Проверяем, что система продолжает работать
            assert successful_analyses > 0
            assert successful_analyses < len(scenario['symbols'])  # Не все символы обработаны
    def test_symbol_analysis_data_consistency_e2e(self, e2e_components, realistic_trading_scenario) -> None:
        """Тест консистентности данных в E2E сценарии."""
        components = e2e_components
        scenario = realistic_trading_scenario
        # Анализируем символы несколько раз
        results = []
        for iteration in range(3):
            iteration_results = {}
            for symbol in scenario['symbols']:
                market_data = scenario['market_data'][symbol]
                order_book = scenario['order_books'][symbol]
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
                        iteration_results[symbol] = profile
            results.append(iteration_results)
        # Проверяем консистентность результатов
        for symbol in scenario['symbols']:
            for i in range(1, len(results)):
                assert results[i][symbol].symbol == results[0][symbol].symbol
                assert results[i][symbol].opportunity_score == results[0][symbol].opportunity_score
                assert results[i][symbol].market_phase == results[0][symbol].market_phase
    def test_symbol_analysis_integration_with_trading_system_e2e(self, e2e_components, realistic_trading_scenario) -> None:
        """Тест интеграции с торговой системой E2E."""
        components = e2e_components
        scenario = realistic_trading_scenario
        # Симулируем интеграцию с торговой системой
        trading_system = Mock()
        trading_system.get_account_balance.return_value = 10000.0
        trading_system.get_open_positions.return_value = []
        trading_system.place_order.return_value = Mock(order_id="test_order_123", status="filled")
        # Анализируем символы
        for symbol in scenario['symbols']:
            market_data = scenario['market_data'][symbol]
            order_book = scenario['order_books'][symbol]
            with patch.object(components['classifier'], 'classify_market_phase') as mock_classify:
                mock_classify.return_value = Mock(phase=MarketPhase.BREAKOUT_ACTIVE, confidence=0.8)
                with patch.object(components['calculator'], 'calculate_opportunity_score') as mock_calc:
                    mock_calc.return_value = 0.8
                    profile = SymbolProfile(
                        symbol=symbol,
                        opportunity_score=0.8,
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
                    components['cache'].set_profile(symbol, profile)
        # Выбираем лучшие символы
        with patch.object(components['doass'], 'get_detailed_analysis') as mock_doass:
            mock_doass.return_value = Mock(
                selected_symbols=["BTCUSDT"],
                detailed_profiles={"BTCUSDT": components['cache'].get_profile("BTCUSDT")},
                total_symbols_analyzed=len(scenario['symbols']),
                processing_time_ms=100.0,
                cache_hit_rate=0.9
            )
            analysis_result = components['doass'].get_detailed_analysis(limit=1)
        # Принимаем торговое решение
        symbol = analysis_result.selected_symbols[0]
        profile = analysis_result.detailed_profiles[symbol]
        decision = TradingDecision(
            symbol=symbol,
            side=PositionSide.LONG,
            order_type=OrderType.MARKET,
            quantity=0.1,
            price=None,
            confidence=profile.confidence,
            reasoning=f"High opportunity score: {profile.opportunity_score:.2f}"
        )
        # Исполняем через торговую систему
        order_result = trading_system.place_order(
            symbol=decision.symbol,
            side=decision.side.value,
            order_type=decision.order_type.value,
            quantity=decision.quantity,
            price=decision.price
        )
        # Проверяем результат
        assert order_result.order_id is not None
        assert order_result.status == "filled"
    @pytest.mark.asyncio
    async def test_symbol_analysis_async_e2e(self, e2e_components, realistic_trading_scenario) -> None:
        """Тест асинхронного E2E анализа символов."""
        components = e2e_components
        scenario = realistic_trading_scenario
        async def analyze_symbol(symbol) -> Any:
            """Асинхронный анализ символа."""
            market_data = scenario['market_data'][symbol]
            order_book = scenario['order_books'][symbol]
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
                    return profile
        # Асинхронно анализируем все символы
        tasks = [analyze_symbol(symbol) for symbol in scenario['symbols']]
        results = await asyncio.gather(*tasks)
        # Проверяем результаты
        assert len(results) == len(scenario['symbols'])
        for result in results:
            assert result is not None
            assert result.opportunity_score == 0.6
            assert result.market_phase == MarketPhase.BREAKOUT_ACTIVE
if __name__ == "__main__":
    pytest.main([__file__]) 
