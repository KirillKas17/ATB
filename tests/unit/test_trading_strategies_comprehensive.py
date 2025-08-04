"""
Комплексные unit тесты для торговых стратегий.
Включает стресс-тесты, тесты граничных случаев и безопасности.
"""

import pytest
import pandas as pd
from shared.numpy_utils import np
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any
import asyncio
from datetime import datetime, timedelta

from domain.strategies import StrategyFactory, StrategyRegistry
from domain.value_objects.position import Position, PositionSide
from domain.value_objects.order import Order, OrderType, OrderSide
from domain.value_objects.symbol import Symbol
from domain.value_objects.price import Price
from domain.value_objects.quantity import Quantity
from domain.entities.market import Market
from domain.exceptions import DomainException


class TestStrategyFactory:
    """Unit тесты для StrategyFactory."""

    @pytest.fixture
    def strategy_factory(self) -> StrategyFactory:
        """Создает StrategyFactory с моками."""
        return StrategyFactory()

    @pytest.fixture
    def mock_market_data(self) -> pd.DataFrame:
        """Создает моковые рыночные данные."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        np.random.seed(42)
        
        return pd.DataFrame({
            'open': np.random.uniform(99, 101, 100),
            'high': np.random.uniform(100, 102, 100),
            'low': np.random.uniform(98, 100, 100),
            'close': np.random.uniform(99, 101, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }, index=dates)

    def test_create_trend_strategy(self, strategy_factory: StrategyFactory):
        """Тест создания трендовой стратегии."""
        config = {
            'strategy_type': 'trend',
            'ema_fast': 12,
            'ema_slow': 26,
            'risk_percentage': 0.02
        }
        
        strategy = strategy_factory.create_strategy(config)
        
        assert strategy is not None
        assert hasattr(strategy, 'generate_signals')
        assert hasattr(strategy, 'calculate_position_size')

    def test_create_mean_reversion_strategy(self, strategy_factory: StrategyFactory):
        """Тест создания стратегии возврата к среднему."""
        config = {
            'strategy_type': 'mean_reversion',
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14
        }
        
        strategy = strategy_factory.create_strategy(config)
        
        assert strategy is not None
        assert hasattr(strategy, 'generate_signals')

    def test_create_invalid_strategy_type(self, strategy_factory: StrategyFactory):
        """Тест создания стратегии с неверным типом."""
        config = {
            'strategy_type': 'nonexistent_strategy'
        }
        
        with pytest.raises((ValueError, KeyError)):
            strategy_factory.create_strategy(config)

    def test_create_strategy_missing_config(self, strategy_factory: StrategyFactory):
        """Тест создания стратегии с неполной конфигурацией."""
        config = {
            'strategy_type': 'trend'
            # Отсутствуют обязательные параметры
        }
        
        # Должна либо создать стратегию с параметрами по умолчанию, либо выбросить исключение
        try:
            strategy = strategy_factory.create_strategy(config)
            assert strategy is not None
        except (ValueError, KeyError):
            pass  # Ожидаемое поведение


class TestTrendStrategy:
    """Unit тесты для трендовой стратегии."""

    @pytest.fixture
    def trend_strategy(self):
        """Создает экземпляр трендовой стратегии."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {
            'ema_fast': 12,
            'ema_slow': 26,
            'risk_percentage': 0.02,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
        
        return TrendStrategy(config)

    @pytest.fixture
    def trending_up_data(self) -> pd.DataFrame:
        """Создает данные восходящего тренда."""
        dates = pd.date_range('2023-01-01', periods=50, freq='1h')
        
        # Создаем восходящий тренд
        base_price = 100
        prices = []
        for i in range(50):
            base_price += np.random.normal(0.1, 0.5)  # Небольшой восходящий тренд
            prices.append(max(base_price, 0.01))
        
        return pd.DataFrame({
            'open': [p + np.random.normal(0, 0.1) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 3000, 50)
        }, index=dates)

    @pytest.fixture
    def trending_down_data(self) -> pd.DataFrame:
        """Создает данные нисходящего тренда."""
        dates = pd.date_range('2023-01-01', periods=50, freq='1h')
        
        # Создаем нисходящий тренд
        base_price = 100
        prices = []
        for i in range(50):
            base_price -= np.random.normal(0.1, 0.5)  # Нисходящий тренд
            prices.append(max(base_price, 0.01))
        
        return pd.DataFrame({
            'open': [p + np.random.normal(0, 0.1) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.2)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.2)) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 3000, 50)
        }, index=dates)

    def test_generate_signals_uptrend(self, trend_strategy, trending_up_data):
        """Тест генерации сигналов на восходящем тренде."""
        signals = trend_strategy.generate_signals(trending_up_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert len(signals) > 0
        
        # В восходящем тренде должны быть сигналы на покупку
        buy_signals = signals[signals['signal'] > 0]
        assert len(buy_signals) > 0

    def test_generate_signals_downtrend(self, trend_strategy, trending_down_data):
        """Тест генерации сигналов на нисходящем тренде."""
        signals = trend_strategy.generate_signals(trending_down_data)
        
        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        
        # В нисходящем тренде должны быть сигналы на продажу
        sell_signals = signals[signals['signal'] < 0]
        assert len(sell_signals) > 0

    def test_calculate_position_size(self, trend_strategy):
        """Тест расчета размера позиции."""
        balance = Decimal('10000')
        price = Price(Decimal('100'))
        risk_amount = Decimal('200')  # 2% от баланса
        
        position_size = trend_strategy.calculate_position_size(
            balance=balance, 
            price=price, 
            risk_amount=risk_amount
        )
        
        assert isinstance(position_size, Quantity)
        assert position_size.value > 0
        assert position_size.value <= balance / price.value  # Не превышает доступные средства

    def test_validate_signal_strength(self, trend_strategy, trending_up_data):
        """Тест валидации силы сигнала."""
        signals = trend_strategy.generate_signals(trending_up_data)
        
        # Проверяем, что сила сигналов в разумных пределах
        assert all(-1 <= signal <= 1 for signal in signals['signal'])

    def test_strategy_with_insufficient_data(self, trend_strategy):
        """Тест стратегии с недостаточным количеством данных."""
        insufficient_data = pd.DataFrame({
            'open': [100],
            'high': [101],
            'low': [99],
            'close': [100.5],
            'volume': [1000]
        })
        
        # Стратегия должна корректно обработать недостаток данных
        result = trend_strategy.generate_signals(insufficient_data)
        
        assert isinstance(result, pd.DataFrame)
        # Может быть пустым или содержать нулевые сигналы


class TestRiskManagement:
    """Тесты управления рисками в стратегиях."""

    @pytest.fixture
    def risk_manager(self):
        """Создает менеджер рисков."""
        from application.services.risk_service import RiskService
        
        # Мокаем зависимости
        config_repo = Mock()
        position_repo = Mock()
        order_repo = Mock()
        
        return RiskService(config_repo, position_repo, order_repo)

    @pytest.fixture
    def sample_position(self) -> Position:
        """Создает образец позиции."""
        return Position(
            symbol=Symbol("BTCUSDT"),
            side=PositionSide.LONG,
            quantity=Quantity(Decimal('1.5')),
            entry_price=Price(Decimal('50000')),
            current_price=Price(Decimal('51000')),
            unrealized_pnl=Decimal('1500')
        )

    def test_position_size_limits(self, risk_manager, sample_position):
        """Тест ограничений размера позиции."""
        # Тестируем максимальный размер позиции
        max_position_size = Decimal('100000')  # $100k
        
        is_valid = risk_manager.validate_position_size(
            position=sample_position,
            max_size=max_position_size
        )
        
        assert isinstance(is_valid, bool)

    def test_stop_loss_calculation(self, risk_manager, sample_position):
        """Тест расчета стоп-лосса."""
        stop_loss_pct = Decimal('0.03')  # 3%
        
        stop_loss_price = risk_manager.calculate_stop_loss(
            position=sample_position,
            stop_loss_percentage=stop_loss_pct
        )
        
        assert isinstance(stop_loss_price, Price)
        assert stop_loss_price.value < sample_position.entry_price.value

    def test_portfolio_risk_exposure(self, risk_manager):
        """Тест расчета рискового воздействия портфеля."""
        positions = [
            Mock(unrealized_pnl=Decimal('1000')),
            Mock(unrealized_pnl=Decimal('-500')),
            Mock(unrealized_pnl=Decimal('200'))
        ]
        
        total_exposure = risk_manager.calculate_portfolio_exposure(positions)
        
        assert isinstance(total_exposure, dict)
        assert 'total_pnl' in total_exposure
        assert 'risk_exposure' in total_exposure

    def test_drawdown_protection(self, risk_manager):
        """Тест защиты от просадок."""
        account_balance = Decimal('100000')
        current_drawdown = Decimal('0.15')  # 15% просадка
        max_drawdown = Decimal('0.20')  # Максимальная допустимая просадка 20%
        
        should_stop_trading = risk_manager.check_drawdown_limit(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown
        )
        
        assert isinstance(should_stop_trading, bool)
        assert not should_stop_trading  # 15% < 20%, торговля продолжается

    def test_maximum_drawdown_breach(self, risk_manager):
        """Тест превышения максимальной просадки."""
        current_drawdown = Decimal('0.25')  # 25% просадка
        max_drawdown = Decimal('0.20')  # Максимальная допустимая просадка 20%
        
        should_stop_trading = risk_manager.check_drawdown_limit(
            current_drawdown=current_drawdown,
            max_drawdown=max_drawdown
        )
        
        assert should_stop_trading  # Должна остановить торговлю


class TestStrategyPerformance:
    """Тесты производительности стратегий."""

    @pytest.fixture
    def performance_data(self) -> pd.DataFrame:
        """Создает данные для тестирования производительности."""
        dates = pd.date_range('2023-01-01', periods=10000, freq='1min')
        np.random.seed(42)
        
        return pd.DataFrame({
            'open': np.random.uniform(99, 101, 10000),
            'high': np.random.uniform(100, 102, 10000),
            'low': np.random.uniform(98, 100, 10000),
            'close': np.random.uniform(99, 101, 10000),
            'volume': np.random.uniform(100, 1000, 10000)
        }, index=dates)

    def test_strategy_execution_time(self, performance_data):
        """Тест времени выполнения стратегии."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        import time
        start_time = time.time()
        
        signals = strategy.generate_signals(performance_data)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Стратегия должна обрабатывать 10к записей за разумное время
        assert execution_time < 5.0, f"Слишком долгое выполнение: {execution_time}s"
        assert isinstance(signals, pd.DataFrame)

    def test_memory_usage_large_dataset(self, performance_data):
        """Тест использования памяти на больших датасетах."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Тестируем обработку большого датасета
        try:
            signals = strategy.generate_signals(performance_data)
            assert isinstance(signals, pd.DataFrame)
            assert len(signals) > 0
        except MemoryError:
            pytest.fail("MemoryError при обработке больших данных")

    def test_concurrent_strategy_execution(self):
        """Тест одновременного выполнения нескольких стратегий."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        # Создаем несколько стратегий
        strategies = []
        for i in range(5):
            config = {'ema_fast': 10 + i, 'ema_slow': 20 + i*2}
            strategies.append(TrendStrategy(config))
        
        # Создаем тестовые данные
        data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 1000),
            'high': np.random.uniform(100, 102, 1000),
            'low': np.random.uniform(98, 100, 1000),
            'close': np.random.uniform(99, 101, 1000),
            'volume': np.random.uniform(100, 1000, 1000)
        })
        
        # Запускаем стратегии одновременно
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(strategy.generate_signals, data) for strategy in strategies]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Проверяем, что все стратегии выполнились успешно
        assert len(results) == 5
        assert all(isinstance(result, pd.DataFrame) for result in results)


class TestStrategyStress:
    """Стресс тесты для стратегий."""

    def test_extreme_volatility_handling(self):
        """Тест обработки экстремальной волатильности."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Создаем данные с экстремальной волатильностью
        extreme_data = pd.DataFrame({
            'open': [100, 50, 150, 25, 200],
            'high': [110, 60, 160, 35, 210],
            'low': [90, 40, 140, 15, 190],
            'close': [105, 55, 155, 30, 205],
            'volume': [1000, 5000, 500, 10000, 200]
        })
        
        # Стратегия должна корректно обработать экстремальные данные
        signals = strategy.generate_signals(extreme_data)
        
        assert isinstance(signals, pd.DataFrame)
        # Сигналы должны быть в разумных пределах даже при экстремальных данных
        assert all(-1 <= signal <= 1 for signal in signals['signal'] if not pd.isna(signal))

    def test_missing_data_handling(self):
        """Тест обработки пропущенных данных."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Создаем данные с пропусками
        data_with_gaps = pd.DataFrame({
            'open': [100, np.nan, 102, 103, np.nan],
            'high': [101, 103, np.nan, 104, 105],
            'low': [99, 100, 101, np.nan, 103],
            'close': [100.5, 102.5, 102.2, 103.5, 104.2],
            'volume': [1000, np.nan, 1200, 1100, np.nan]
        })
        
        # Стратегия должна корректно обработать пропуски
        try:
            signals = strategy.generate_signals(data_with_gaps)
            assert isinstance(signals, pd.DataFrame)
        except Exception as e:
            # Если стратегия не может обработать пропуски, должна выбросить понятное исключение
            assert isinstance(e, (ValueError, KeyError))

    def test_zero_volume_handling(self):
        """Тест обработки нулевых объемов."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Данные с нулевыми объемами
        zero_volume_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [0, 0, 0]
        })
        
        signals = strategy.generate_signals(zero_volume_data)
        
        assert isinstance(signals, pd.DataFrame)

    def test_identical_prices_handling(self):
        """Тест обработки идентичных цен (отсутствие движения)."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Данные без движения цены
        flat_data = pd.DataFrame({
            'open': [100] * 50,
            'high': [100] * 50,
            'low': [100] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        })
        
        signals = strategy.generate_signals(flat_data)
        
        assert isinstance(signals, pd.DataFrame)
        # При отсутствии движения сигналы должны быть нулевыми или близкими к нулю
        assert all(abs(signal) < 0.1 for signal in signals['signal'] if not pd.isna(signal))


class TestStrategyIntegration:
    """Интеграционные тесты стратегий."""

    @pytest.fixture
    def mock_exchange(self):
        """Создает мок биржи."""
        exchange = Mock()
        exchange.get_account_balance = AsyncMock(return_value=Decimal('10000'))
        exchange.place_order = AsyncMock(return_value={'order_id': '123', 'status': 'filled'})
        exchange.get_market_data = AsyncMock()
        return exchange

    @pytest.fixture
    def mock_risk_manager(self):
        """Создает мок риск-менеджера."""
        risk_manager = Mock()
        risk_manager.validate_order = Mock(return_value=True)
        risk_manager.calculate_position_size = Mock(return_value=Quantity(Decimal('1.0')))
        return risk_manager

    @pytest.mark.asyncio
    async def test_strategy_with_exchange_integration(self, mock_exchange, mock_risk_manager):
        """Тест интеграции стратегии с биржей."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Мокаем рыночные данные
        market_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        mock_exchange.get_market_data.return_value = market_data
        
        # Генерируем сигналы
        signals = strategy.generate_signals(market_data)
        
        # Проверяем, что сигналы сгенерированы
        assert isinstance(signals, pd.DataFrame)
        assert len(signals) > 0
        
        # Эмулируем выполнение сделки на основе сигнала
        if len(signals[signals['signal'] > 0]) > 0:
            # Есть сигнал на покупку
            balance = await mock_exchange.get_account_balance()
            assert balance > 0
            
            # Размещаем ордер
            order_result = await mock_exchange.place_order()
            assert order_result['status'] == 'filled'


class TestMockDependencies:
    """Тесты с полностью мокнутыми зависимостями."""

    def test_strategy_with_mocked_market_data_provider(self):
        """Тест стратегии с мокнутым провайдером рыночных данных."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        # Мокаем провайдер данных
        with patch('infrastructure.market_data.base_connector.BaseExchangeConnector') as mock_connector:
            mock_connector.get_historical_data.return_value = pd.DataFrame({
                'open': [100, 101, 102],
                'high': [101, 102, 103],
                'low': [99, 100, 101],
                'close': [100.5, 101.5, 102.5],
                'volume': [1000, 1100, 1200]
            })
            
            config = {'ema_fast': 12, 'ema_slow': 26}
            strategy = TrendStrategy(config)
            
            # Стратегия должна работать с мокнутыми данными
            data = mock_connector.get_historical_data.return_value
            signals = strategy.generate_signals(data)
            
            assert isinstance(signals, pd.DataFrame)

    def test_strategy_error_handling_with_mocks(self):
        """Тест обработки ошибок стратегии с моками."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Мокаем провайдер данных, который возвращает ошибку
        with patch.object(strategy, '_calculate_indicators', side_effect=Exception("Mocked error")):
            
            data = pd.DataFrame({
                'open': [100], 'high': [101], 'low': [99], 
                'close': [100.5], 'volume': [1000]
            })
            
            # Стратегия должна корректно обработать ошибку
            with pytest.raises(Exception):
                strategy.generate_signals(data)


# Тесты безопасности
class TestStrategySecurity:
    """Тесты безопасности стратегий."""

    def test_input_validation_sql_injection_attempt(self):
        """Тест защиты от попыток SQL инъекций."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        # Попытка передать вредоносные данные
        malicious_config = {
            'ema_fast': "'; DROP TABLE orders; --",
            'ema_slow': 26
        }
        
        # Конфигурация должна быть валидирована
        with pytest.raises((ValueError, TypeError)):
            TrendStrategy(malicious_config)

    def test_parameter_bounds_validation(self):
        """Тест валидации границ параметров."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        # Тестируем различные некорректные параметры
        invalid_configs = [
            {'ema_fast': -1, 'ema_slow': 26},  # Отрицательное значение
            {'ema_fast': 0, 'ema_slow': 26},   # Нулевое значение
            {'ema_fast': 26, 'ema_slow': 12},  # Быстрая EMA больше медленной
            {'ema_fast': 1000000, 'ema_slow': 26},  # Слишком большое значение
        ]
        
        for config in invalid_configs:
            with pytest.raises((ValueError, AssertionError)):
                strategy = TrendStrategy(config)
                # Или стратегия создается, но валидирует параметры при использовании
                if hasattr(strategy, 'validate_config'):
                    strategy.validate_config()

    def test_output_sanitization(self):
        """Тест санитизации выходных данных."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        signals = strategy.generate_signals(data)
        
        # Проверяем, что выходные данные не содержат вредоносного контента
        assert isinstance(signals, pd.DataFrame)
        
        # Все числовые значения должны быть конечными
        numeric_cols = signals.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert all(np.isfinite(val) or pd.isna(val) for val in signals[col])

    def test_resource_limits(self):
        """Тест ограничений ресурсов."""
        from infrastructure.strategies.trend_strategies import TrendStrategy
        
        config = {'ema_fast': 12, 'ema_slow': 26}
        strategy = TrendStrategy(config)
        
        # Тестируем с очень большим датасетом
        large_data = pd.DataFrame({
            'open': np.random.uniform(99, 101, 1000000),  # 1M записей
            'high': np.random.uniform(100, 102, 1000000),
            'low': np.random.uniform(98, 100, 1000000),
            'close': np.random.uniform(99, 101, 1000000),
            'volume': np.random.uniform(1000, 5000, 1000000)
        })
        
        # Должно либо успешно обработать, либо выбросить ошибку ресурсов
        try:
            import time
            start_time = time.time()
            
            signals = strategy.generate_signals(large_data)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Не должно занимать более 2 минут
            assert execution_time < 120, f"Слишком долгое выполнение: {execution_time}s"
            assert isinstance(signals, pd.DataFrame)
            
        except (MemoryError, TimeoutError):
            # Ожидаемое поведение для слишком больших данных
            pass