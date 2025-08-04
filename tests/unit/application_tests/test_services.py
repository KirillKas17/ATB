"""
Тесты для сервисов application слоя.
"""
import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from decimal import Decimal
from datetime import datetime
from domain.entities.market import MarketData
from domain.entities.portfolio import Portfolio
from domain.entities.strategy import Strategy
from domain.entities.trading import Trade
from domain.value_objects.money import Money
from domain.type_definitions import PortfolioId, Symbol, StrategyId
from application.types import (
    CreateOrderRequest, CreateOrderResponse, PortfolioSummary, StrategyPerformance
)
from application.services.implementations.market_service_impl import MarketServiceImpl
from application.services.implementations.ml_service_impl import MLServiceImpl
from application.services.implementations.trading_service_impl import TradingServiceImpl
from application.services.implementations.strategy_service_impl import StrategyServiceImpl
from application.services.implementations.portfolio_service_impl import PortfolioServiceImpl
from application.services.implementations.risk_service_impl import RiskServiceImpl
from application.services.implementations.cache_service_impl import CacheServiceImpl
try:
    from application.services.implementations.notification_service_impl import NotificationServiceImpl
except ImportError:
    NotificationServiceImpl = object
class TestMarketServiceImpl:
    """Тесты для MarketServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'cache_enabled': True,
            'update_interval': 60,
            'max_retries': 3
        }
        self.service = MarketServiceImpl(self.config)
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_get_market_data(self) -> None:
        """Тест получения рыночных данных."""
        symbol = Symbol("BTCUSDT")
        with patch.object(self.service, '_fetch_market_data') as mock_fetch:
            mock_fetch.return_value = MarketData(
                symbol=symbol,
                price=Money(Decimal('50000'), 'USDT'),
                volume=Decimal('1000'),
                timestamp=datetime.now()
            )
            result = await self.service.get_market_data(symbol)
            assert result is not None
            assert result.symbol == symbol
            mock_fetch.assert_called_once_with(symbol)
    @pytest.mark.asyncio
    async def test_get_orderbook(self) -> None:
        """Тест получения ордербука."""
        symbol = Symbol("BTCUSDT")
        with patch.object(self.service, '_fetch_orderbook') as mock_fetch:
            mock_orderbook = {
                'bids': [[50000, 1.0], [49999, 2.0]],
                'asks': [[50001, 1.5], [50002, 2.5]]
            }
            mock_fetch.return_value = mock_orderbook
            result = await self.service.get_orderbook(symbol)
            assert result == mock_orderbook
            mock_fetch.assert_called_once_with(symbol)
    @pytest.mark.asyncio
    async def test_calculate_spread(self) -> None:
        """Тест расчета спреда."""
        symbol = Symbol("BTCUSDT")
        with patch.object(self.service, 'get_orderbook') as mock_get_orderbook:
            mock_get_orderbook.return_value = {
                'bids': [[50000, 1.0]],
                'asks': [[50001, 1.0]]
            }
            result = await self.service.calculate_spread(symbol)
            assert result == Decimal('1')
            mock_get_orderbook.assert_called_once_with(symbol)
    @pytest.mark.asyncio
    async def test_get_market_statistics(self) -> None:
        """Тест получения статистики рынка."""
        result = await self.service.get_market_statistics()
        assert isinstance(result, dict)
        assert 'total_requests' in result
        assert 'cache_hits' in result
        assert 'cache_misses' in result
class TestMLServiceImpl:
    """Тесты для MLServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'model_path': '/path/to/model',
            'prediction_threshold': 0.7,
            'batch_size': 32
        }
        self.service = MLServiceImpl(self.config)
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_predict_price_movement(self) -> None:
        """Тест предсказания движения цены."""
        symbol = Symbol("BTCUSDT")
        historical_data = [Decimal('50000'), Decimal('50100'), Decimal('50200')]
        with patch.object(self.service, '_load_model') as mock_load:
            with patch.object(self.service, '_preprocess_data') as mock_preprocess:
                with patch.object(self.service, '_make_prediction') as mock_predict:
                    mock_preprocess.return_value = [0.1, 0.2, 0.3]
                    mock_predict.return_value = {'direction': 'up', 'confidence': 0.8}
                    result = await self.service.predict_price_movement(symbol, historical_data)
                    assert result['direction'] == 'up'
                    assert result['confidence'] == 0.8
                    mock_load.assert_called_once()
                    mock_preprocess.assert_called_once_with(historical_data)
                    mock_predict.assert_called_once()
    @pytest.mark.asyncio
    async def test_get_model_performance(self) -> None:
        """Тест получения производительности модели."""
        result = await self.service.get_model_performance()
        assert isinstance(result, dict)
        assert 'accuracy' in result
        assert 'precision' in result
        assert 'recall' in result
class TestTradingServiceImpl:
    """Тесты для TradingServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'max_orders': 100,
            'order_timeout': 30,
            'retry_attempts': 3
        }
        self.service = TradingServiceImpl(self.config)
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_create_order(self) -> None:
        """Тест создания ордера."""
        request = CreateOrderRequest(
            symbol=Symbol("BTCUSDT"),
            side="BUY",
            amount=Money(Decimal('0.1'), 'BTC'),
            price=Money(Decimal('50000'), 'USDT'),
            order_type="LIMIT"
        )
        with patch.object(self.service, '_validate_order') as mock_validate:
            with patch.object(self.service, '_execute_order') as mock_execute:
                mock_validate.return_value = True
                mock_execute.return_value = CreateOrderResponse(
                    order_id="order_123",
                    status="PENDING",
                    timestamp=datetime.now()
                )
                result = await self.service.create_order(request)
                assert result.order_id == "order_123"
                assert result.status == "PENDING"
                mock_validate.assert_called_once_with(request)
                mock_execute.assert_called_once_with(request)
    @pytest.mark.asyncio
    async def test_get_order_status(self) -> None:
        """Тест получения статуса ордера."""
        order_id = "order_123"
        with patch.object(self.service, '_fetch_order_status') as mock_fetch:
            mock_fetch.return_value = "FILLED"
            result = await self.service.get_order_status(order_id)
            assert result == "FILLED"
            mock_fetch.assert_called_once_with(order_id)
    @pytest.mark.asyncio
    async def test_cancel_order(self) -> None:
        """Тест отмены ордера."""
        order_id = "order_123"
        with patch.object(self.service, '_execute_cancel') as mock_cancel:
            mock_cancel.return_value = True
            result = await self.service.cancel_order(order_id)
            assert result is True
            mock_cancel.assert_called_once_with(order_id)
class TestStrategyServiceImpl:
    """Тесты для StrategyServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'max_strategies': 50,
            'strategy_timeout': 60,
            'backtest_enabled': True
        }
        self.service = StrategyServiceImpl(self.config)
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_create_strategy(self) -> None:
        """Тест создания стратегии."""
        config = {
            'name': "Test Strategy",
            'type': "TREND_FOLLOWING",
            'parameters': {'period': 20, 'threshold': 0.1}
        }
        with patch.object(self.service, '_validate_strategy_config') as mock_validate:
            with patch.object(self.service, '_create_strategy_instance') as mock_create:
                mock_validate.return_value = True
                mock_create.return_value = Strategy(
                    id=StrategyId("strategy_123"),
                    name="Test Strategy",
                    config=config,
                    created_at=datetime.now()
                )
                result = await self.service.create_strategy(config)
                assert result.name == "Test Strategy"
                assert result.config == config
                mock_validate.assert_called_once_with(config)
                mock_create.assert_called_once_with(config)
    @pytest.mark.asyncio
    async def test_get_strategy(self) -> None:
        """Тест получения стратегии."""
        strategy_id = StrategyId("strategy_123")
        with patch.object(self.service, '_fetch_strategy') as mock_fetch:
            mock_strategy = Strategy(
                id=strategy_id,
                name="Test Strategy",
                config={'name': "Test", 'type': "TREND_FOLLOWING"},
                created_at=datetime.now()
            )
            mock_fetch.return_value = mock_strategy
            result = await self.service.get_strategy(strategy_id)
            assert result == mock_strategy
            mock_fetch.assert_called_once_with(strategy_id)
    @pytest.mark.asyncio
    async def test_update_strategy(self) -> None:
        """Тест обновления стратегии."""
        strategy_id = StrategyId("strategy_123")
        config = {
            'name': "Updated Strategy",
            'type': "MEAN_REVERSION",
            'parameters': {'period': 30}
        }
        with patch.object(self.service, '_validate_strategy_config') as mock_validate:
            with patch.object(self.service, '_update_strategy_instance') as mock_update:
                mock_validate.return_value = True
                mock_update.return_value = True
                result = await self.service.update_strategy(strategy_id, config)
                assert result is True
                mock_validate.assert_called_once_with(config)
                mock_update.assert_called_once_with(strategy_id, config)
class TestPortfolioServiceImpl:
    """Тесты для PortfolioServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'max_portfolios': 10,
            'default_currency': 'USDT',
            'rebalancing_enabled': True
        }
        self.service = PortfolioServiceImpl(self.config)
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_create_portfolio(self) -> None:
        """Тест создания портфеля."""
        config = {
            'name': "Test Portfolio",
            'initial_balance': Money(Decimal('10000'), 'USDT'),
            'risk_level': "MEDIUM"
        }
        with patch.object(self.service, '_validate_portfolio_config') as mock_validate:
            with patch.object(self.service, '_create_portfolio_instance') as mock_create:
                mock_validate.return_value = True
                mock_portfolio = Portfolio(
                    id=PortfolioId("portfolio_123"),
                    name="Test Portfolio",
                    balance=Money(Decimal('10000'), 'USDT'),
                    created_at=datetime.now()
                )
                mock_create.return_value = mock_portfolio
                result = await self.service.create_portfolio(config)
                assert result.name == "Test Portfolio"
                assert result.balance.value == Decimal('10000')
                mock_validate.assert_called_once_with(config)
                mock_create.assert_called_once_with(config)
    @pytest.mark.asyncio
    async def test_get_portfolio(self) -> None:
        """Тест получения портфеля."""
        portfolio_id = PortfolioId("portfolio_123")
        with patch.object(self.service, '_fetch_portfolio') as mock_fetch:
            mock_portfolio = Portfolio(
                id=portfolio_id,
                name="Test Portfolio",
                balance=Money(Decimal('10000'), 'USDT'),
                created_at=datetime.now()
            )
            mock_fetch.return_value = mock_portfolio
            result = await self.service.get_portfolio(portfolio_id)
            assert result == mock_portfolio
            mock_fetch.assert_called_once_with(portfolio_id)
    @pytest.mark.asyncio
    async def test_get_portfolio_performance(self) -> None:
        """Тест получения производительности портфеля."""
        portfolio_id = PortfolioId("portfolio_123")
        with patch.object(self.service, '_calculate_performance') as mock_calc:
            mock_performance = PortfolioSummary(
                portfolio_id=portfolio_id,
                current_balance=Money(Decimal('10500'), 'USDT'),
                total_return=Decimal('0.05'),
                daily_return=Decimal('0.02'),
                volatility=Decimal('0.1'),
                sharpe_ratio=Decimal('1.5'),
                max_drawdown=Decimal('0.03'),
                win_rate=Decimal('0.6'),
                total_trades=10,
                timestamp=datetime.now()
            )
            mock_calc.return_value = mock_performance
            result = await self.service.get_portfolio_performance(portfolio_id)
            assert result == mock_performance
            mock_calc.assert_called_once_with(portfolio_id)
class TestRiskServiceImpl:
    """Тесты для RiskServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'risk_cache_ttl': 300,
            'var_confidence_levels': [0.95, 0.99],
            'max_position_size': Decimal('10000'),
            'max_portfolio_concentration': Decimal('0.2'),
            'max_daily_loss': Decimal('1000')
        }
        self.risk_repository = Mock()
        self.risk_calculator = Mock()
        self.service = RiskServiceImpl(
            risk_repository=self.risk_repository,
            risk_calculator=self.risk_calculator,
            config=self.config
        )
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_assess_portfolio_risk(self) -> None:
        """Тест оценки риска портфеля."""
        portfolio_id = PortfolioId("portfolio_123")
        with patch.object(self.service, '_get_market_data_for_positions') as mock_market:
            with patch.object(self.service, '_calculate_position_risk') as mock_pos_risk:
                with patch.object(self.service, 'calculate_var') as mock_var:
                    with patch.object(self.service, 'calculate_max_drawdown') as mock_drawdown:
                        mock_market.return_value = {}
                        mock_pos_risk.return_value = {
                            'exposure': Decimal('5000'),
                            'unrealized_pnl': Decimal('100'),
                            'risk_score': Decimal('0.1')
                        }
                        mock_var.return_value = Money(Decimal('500'), 'USDT')
                        mock_drawdown.return_value = Decimal('0.05')
                        result = await self.service.assess_portfolio_risk(portfolio_id)
                        assert result.portfolio_id == portfolio_id
                        assert result.var_95.value == Decimal('500')
                        assert result.max_drawdown == Decimal('0.05')
    @pytest.mark.asyncio
    async def test_calculate_var(self) -> None:
        """Тест расчета VaR."""
        portfolio_id = PortfolioId("portfolio_123")
        confidence_level = Decimal('0.95')
        with patch.object(self.service.risk_repository, 'get_portfolio_returns') as mock_returns:
            with patch.object(self.service.risk_calculator, 'calculate_var') as mock_calc:
                mock_returns.return_value = [0.01, -0.02, 0.03]
                mock_calc.return_value = Money(Decimal('500'), 'USDT')
                result = await self.service.calculate_var(portfolio_id, confidence_level)
                assert result.value == Decimal('500')
                mock_returns.assert_called_once_with(portfolio_id, days=252)
                mock_calc.assert_called_once()
    @pytest.mark.asyncio
    async def test_validate_risk_limits(self) -> None:
        """Тест валидации лимитов риска."""
        portfolio_id = PortfolioId("portfolio_123")
        order_request = CreateOrderRequest(
            symbol=Symbol("BTCUSDT"),
            side="BUY",
            amount=Money(Decimal('0.1'), 'BTC'),
            price=Money(Decimal('50000'), 'USDT'),
            order_type="LIMIT"
        )
        with patch.object(self.service, 'assess_portfolio_risk') as mock_assess:
            with patch.object(self.service, '_calculate_symbol_concentration') as mock_conc:
                with patch.object(self.service, '_calculate_daily_pnl') as mock_pnl:
                    mock_assess.return_value = {
                        'portfolio_id': portfolio_id,
                        'total_value': Money(Decimal('100000'), 'USDT'),
                        'var_95': Money(Decimal('1000'), 'USDT'),
                        'max_drawdown': Decimal('0.05'),
                        'volatility': Decimal('0.1'),
                        'beta': Decimal('1.0'),
                        'sharpe_ratio': Decimal('1.5'),
                        'sortino_ratio': Decimal('1.8'),
                        'calmar_ratio': Decimal('2.0'),
                        'position_count': 5,
                        'risk_score': Decimal('0.3'),
                        'last_updated': datetime.now(),
                        'metadata': {}
                    }
                    mock_conc.return_value = Decimal('0.1')
                    mock_pnl.return_value = Decimal('100')
                    result, errors = await self.service.validate_risk_limits(portfolio_id, order_request)
                    assert result is True
                    assert len(errors) == 0
class TestCacheServiceImpl:
    """Тесты для CacheServiceImpl."""
    def setup_method(self) -> Any:
        """Настройка перед каждым тестом."""
        self.config = {
            'default_ttl': 300,
            'max_size': 1000,
            'cleanup_interval': 60,
            'eviction_policy': 'LRU'
        }
        self.service = CacheServiceImpl(self.config)
    @pytest.mark.asyncio
    async def test_initialize(self) -> None:
        """Тест инициализации сервиса."""
        await self.service.initialize()
        assert self.service.is_running
        assert self.service.is_initialized
    @pytest.mark.asyncio
    async def test_validate_config(self) -> None:
        """Тест валидации конфигурации."""
        result = await self.service.validate_config()
        assert result is True
    @pytest.mark.asyncio
    async def test_set_and_get(self) -> None:
        """Тест установки и получения значения."""
        key = "test_key"
        value = "test_value"
        # Устанавливаем значение
        result = await self.service.set(key, value)
        assert result is True
        # Получаем значение
        retrieved = await self.service.get(key)
        assert retrieved == value
    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        """Тест удаления значения."""
        key = "test_key"
        value = "test_value"
        # Устанавливаем значение
        await self.service.set(key, value)
        # Удаляем значение
        result = await self.service.delete(key)
        assert result is True
        # Проверяем что значение удалено
        retrieved = await self.service.get(key)
        assert retrieved is None
    @pytest.mark.asyncio
    async def test_exists(self) -> None:
        """Тест проверки существования ключа."""
        key = "test_key"
        value = "test_value"
        # Проверяем несуществующий ключ
        exists = await self.service.exists(key)
        assert exists is False
        # Устанавливаем значение
        await self.service.set(key, value)
        # Проверяем существующий ключ
        exists = await self.service.exists(key)
        assert exists is True
    @pytest.mark.asyncio
    async def test_get_multi(self) -> None:
        """Тест получения нескольких значений."""
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        # Устанавливаем значения
        for key, value in data.items():
            await self.service.set(key, value)
        # Получаем несколько значений
        result = await self.service.get_multi(list(data.keys()))
        assert result == data
    @pytest.mark.asyncio
    async def test_set_multi(self) -> None:
        """Тест установки нескольких значений."""
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        # Устанавливаем несколько значений
        result = await self.service.set_multi(data)
        assert result is True
        # Проверяем что все значения установлены
        for key, value in data.items():
            retrieved = await self.service.get(key)
            assert retrieved == value
    @pytest.mark.asyncio
    async def test_get_stats(self) -> None:
        """Тест получения статистики."""
        # Устанавливаем несколько значений
        await self.service.set("key1", "value1")
        await self.service.set("key2", "value2")
        await self.service.get("key1")  # Hit
        await self.service.get("key3")  # Miss
        stats = await self.service.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['sets'] == 2
        assert stats['total_entries'] == 2
if NotificationServiceImpl is not object:
    class TestNotificationServiceImpl:
        """Тесты для NotificationServiceImpl."""
        def setup_method(self) -> Any:
            """Настройка перед каждым тестом."""
            self.config = {
                'default_level': 'info',
                'retry_attempts': 3,
                'retry_delay': 5,
                'batch_size': 10,
                'batch_timeout': 30,
                'email': {
                    'enabled': False,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'username': 'test@example.com',
                    'password': 'password',
                    'from_email': 'test@example.com',
                    'to_emails': ['admin@example.com']
                },
                'webhook': {
                    'enabled': False,
                    'url': 'https://webhook.example.com',
                    'headers': {},
                    'timeout': 10
                },
                'telegram': {
                    'enabled': False,
                    'bot_token': 'bot_token',
                    'chat_id': 'chat_id',
                    'api_url': 'https://api.telegram.org/bot'
                }
            }
            self.service = NotificationServiceImpl(self.config)
        @pytest.mark.asyncio
        async def test_initialize(self) -> None:
            """Тест инициализации сервиса."""
            await self.service.initialize()
            assert self.service.is_running
            assert self.service.is_initialized
        @pytest.mark.asyncio
        async def test_validate_config(self) -> None:
            """Тест валидации конфигурации."""
            result = await self.service.validate_config()
            assert result is True
        @pytest.mark.asyncio
        async def test_send_notification(self) -> None:
            """Тест отправки уведомления."""
            message = "Test notification"
            level = "info"
            result = await self.service.send_notification(message, level)
            assert result is True
        @pytest.mark.asyncio
        async def test_send_alert(self) -> None:
            """Тест отправки алерта."""
            alert_type = "risk"
            data = {"portfolio_id": "portfolio_123", "risk_level": "high"}
            result = await self.service.send_alert(alert_type, data)
            assert result is True
        @pytest.mark.asyncio
        async def test_subscribe_to_alerts(self) -> None:
            """Тест подписки на алерты."""
            user_id = "user_123"
            alert_types = ["risk", "performance"]
            result = await self.service.subscribe_to_alerts(user_id, alert_types)
            assert result is True
        @pytest.mark.asyncio
        async def test_send_trade_notification(self) -> None:
            """Тест отправки уведомления о сделке."""
            trade = Trade(
                id="trade_123",
                symbol=Symbol("BTCUSDT"),
                side="BUY",
                amount=Money(Decimal('0.1'), 'BTC'),
                price=Money(Decimal('50000'), 'USDT'),
                timestamp=datetime.now()
            )
            result = await self.service.send_trade_notification(trade)
            assert result is True
        @pytest.mark.asyncio
        async def test_send_risk_alert(self) -> None:
            """Тест отправки рискового алерта."""
            portfolio_id = PortfolioId("portfolio_123")
            risk_level = "high"
            details = {"var": 1000, "drawdown": 0.1}
            result = await self.service.send_risk_alert(portfolio_id, risk_level, details)
            assert result is True
        @pytest.mark.asyncio
        async def test_send_performance_report(self) -> None:
            """Тест отправки отчета о производительности."""
            portfolio_id = PortfolioId("portfolio_123")
            metrics = PortfolioSummary(
                portfolio_id=portfolio_id,
                current_balance=Money(Decimal('10500'), 'USDT'),
                total_return=Decimal('0.05'),
                daily_return=Decimal('0.02'),
                volatility=Decimal('0.1'),
                sharpe_ratio=Decimal('1.5'),
                max_drawdown=Decimal('0.03'),
                win_rate=Decimal('0.6'),
                total_trades=10,
                timestamp=datetime.now()
            )
            result = await self.service.send_performance_report(portfolio_id, metrics)
            assert result is True
        @pytest.mark.asyncio
        async def test_get_notification_statistics(self) -> None:
            """Тест получения статистики уведомлений."""
            # Отправляем несколько уведомлений
            await self.service.send_notification("Test 1")
            await self.service.send_notification("Test 2")
            await self.service.send_alert("risk", {"level": "high"})
            stats = await self.service.get_notification_statistics()
            assert stats['total_notifications'] == 3
            assert 'success_rate' in stats
            assert 'queue_size' in stats 
