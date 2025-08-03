"""
Unit тесты для StrategyProtocol.

Покрывает:
- Протоколы стратегий
- Интерфейсы стратегий
- Валидацию протоколов
- Бизнес-логику протоколов стратегий
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from decimal import Decimal

from domain.protocols.strategy_protocol import StrategyProtocol
from domain.entities.strategy import Strategy
from domain.value_objects.money import Money
from domain.exceptions.base_exceptions import ValidationError


class TestStrategyProtocol:
    """Тесты для StrategyProtocol."""
    
    @pytest.fixture
    def strategy_protocol(self):
        """Создание экземпляра StrategyProtocol."""
        return StrategyProtocol()
    
    @pytest.fixture
    def sample_strategy_data(self) -> Dict[str, Any]:
        """Тестовые данные стратегии."""
        return {
            "strategy_id": "strategy_001",
            "name": "Test Strategy",
            "description": "Test strategy description",
            "type": "trend_following",
            "status": "active",
            "parameters": {
                "lookback_period": 20,
                "threshold": 0.02,
                "stop_loss": 0.05,
                "take_profit": 0.10
            },
            "symbols": ["BTC/USDT", "ETH/USDT"],
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:00Z"
        }
    
    @pytest.fixture
    def sample_market_data(self) -> Dict[str, Any]:
        """Тестовые рыночные данные."""
        return {
            "symbol": "BTC/USDT",
            "price": "50000.00",
            "volume": "1000.50",
            "timestamp": "2024-01-01T12:00:00Z",
            "bid": "49999.00",
            "ask": "50001.00",
            "spread": "2.00"
        }
    
    def test_strategy_protocol_creation(self, strategy_protocol):
        """Тест создания протокола стратегии."""
        assert strategy_protocol is not None
        assert isinstance(strategy_protocol, StrategyProtocol)
    
    def test_validate_strategy_parameters(self, strategy_protocol, sample_strategy_data):
        """Тест валидации параметров стратегии."""
        parameters = sample_strategy_data["parameters"]
        
        is_valid = strategy_protocol.validate_strategy_parameters(parameters)
        
        assert is_valid is True
    
    def test_validate_strategy_parameters_invalid_lookback(self, strategy_protocol):
        """Тест валидации неверного периода lookback."""
        invalid_parameters = {
            "lookback_period": 0,  # Должен быть > 0
            "threshold": 0.02,
            "stop_loss": 0.05,
            "take_profit": 0.10
        }
        
        is_valid = strategy_protocol.validate_strategy_parameters(invalid_parameters)
        
        assert is_valid is False
    
    def test_validate_strategy_parameters_invalid_threshold(self, strategy_protocol):
        """Тест валидации неверного порога."""
        invalid_parameters = {
            "lookback_period": 20,
            "threshold": -0.02,  # Должен быть >= 0
            "stop_loss": 0.05,
            "take_profit": 0.10
        }
        
        is_valid = strategy_protocol.validate_strategy_parameters(invalid_parameters)
        
        assert is_valid is False
    
    def test_validate_strategy_parameters_invalid_stop_loss(self, strategy_protocol):
        """Тест валидации неверного стоп-лосса."""
        invalid_parameters = {
            "lookback_period": 20,
            "threshold": 0.02,
            "stop_loss": 1.5,  # Должен быть <= 1
            "take_profit": 0.10
        }
        
        is_valid = strategy_protocol.validate_strategy_parameters(invalid_parameters)
        
        assert is_valid is False
    
    def test_validate_strategy_parameters_invalid_take_profit(self, strategy_protocol):
        """Тест валидации неверного тейк-профита."""
        invalid_parameters = {
            "lookback_period": 20,
            "threshold": 0.02,
            "stop_loss": 0.05,
            "take_profit": -0.10  # Должен быть >= 0
        }
        
        is_valid = strategy_protocol.validate_strategy_parameters(invalid_parameters)
        
        assert is_valid is False
    
    def test_validate_strategy_symbols(self, strategy_protocol, sample_strategy_data):
        """Тест валидации символов стратегии."""
        symbols = sample_strategy_data["symbols"]
        
        is_valid = strategy_protocol.validate_strategy_symbols(symbols)
        
        assert is_valid is True
    
    def test_validate_strategy_symbols_empty(self, strategy_protocol):
        """Тест валидации пустого списка символов."""
        empty_symbols = []
        
        is_valid = strategy_protocol.validate_strategy_symbols(empty_symbols)
        
        assert is_valid is False
    
    def test_validate_strategy_symbols_invalid_format(self, strategy_protocol):
        """Тест валидации неверного формата символов."""
        invalid_symbols = ["BTCUSDT", "ETH/USDT"]  # Первый без разделителя
        
        is_valid = strategy_protocol.validate_strategy_symbols(invalid_symbols)
        
        assert is_valid is False
    
    def test_validate_strategy_type(self, strategy_protocol, sample_strategy_data):
        """Тест валидации типа стратегии."""
        strategy_type = sample_strategy_data["type"]
        
        is_valid = strategy_protocol.validate_strategy_type(strategy_type)
        
        assert is_valid is True
    
    def test_validate_strategy_type_invalid(self, strategy_protocol):
        """Тест валидации неверного типа стратегии."""
        invalid_type = "invalid_strategy_type"
        
        is_valid = strategy_protocol.validate_strategy_type(invalid_type)
        
        assert is_valid is False
    
    def test_validate_strategy_status(self, strategy_protocol, sample_strategy_data):
        """Тест валидации статуса стратегии."""
        status = sample_strategy_data["status"]
        
        is_valid = strategy_protocol.validate_strategy_status(status)
        
        assert is_valid is True
    
    def test_validate_strategy_status_invalid(self, strategy_protocol):
        """Тест валидации неверного статуса стратегии."""
        invalid_status = "invalid_status"
        
        is_valid = strategy_protocol.validate_strategy_status(invalid_status)
        
        assert is_valid is False
    
    def test_calculate_strategy_metrics(self, strategy_protocol, sample_strategy_data):
        """Тест расчета метрик стратегии."""
        strategy = Strategy(**sample_strategy_data)
        
        metrics = strategy_protocol.calculate_strategy_metrics(strategy)
        
        assert isinstance(metrics, dict)
        assert "total_trades" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
    
    def test_generate_trading_signals(self, strategy_protocol, sample_strategy_data, sample_market_data):
        """Тест генерации торговых сигналов."""
        strategy = Strategy(**sample_strategy_data)
        
        signals = strategy_protocol.generate_trading_signals(strategy, sample_market_data)
        
        assert isinstance(signals, list)
    
    def test_analyze_market_conditions(self, strategy_protocol, sample_market_data):
        """Тест анализа рыночных условий."""
        market_analysis = strategy_protocol.analyze_market_conditions(sample_market_data)
        
        assert isinstance(market_analysis, dict)
        assert "trend" in market_analysis
        assert "volatility" in market_analysis
        assert "liquidity" in market_analysis
        assert "support_resistance" in market_analysis
    
    def test_calculate_position_size(self, strategy_protocol, sample_strategy_data):
        """Тест расчета размера позиции."""
        strategy = Strategy(**sample_strategy_data)
        account_balance = Money(amount="10000.00", currency="USDT")
        risk_per_trade = Decimal("0.02")  # 2% риска на сделку
        
        position_size = strategy_protocol.calculate_position_size(strategy, account_balance, risk_per_trade)
        
        assert isinstance(position_size, Decimal)
        assert position_size > 0
    
    def test_calculate_position_size_zero_balance(self, strategy_protocol, sample_strategy_data):
        """Тест расчета размера позиции с нулевым балансом."""
        strategy = Strategy(**sample_strategy_data)
        account_balance = Money(amount="0.00", currency="USDT")
        risk_per_trade = Decimal("0.02")
        
        position_size = strategy_protocol.calculate_position_size(strategy, account_balance, risk_per_trade)
        
        assert position_size == Decimal("0")
    
    def test_calculate_position_size_zero_risk(self, strategy_protocol, sample_strategy_data):
        """Тест расчета размера позиции с нулевым риском."""
        strategy = Strategy(**sample_strategy_data)
        account_balance = Money(amount="10000.00", currency="USDT")
        risk_per_trade = Decimal("0.00")
        
        position_size = strategy_protocol.calculate_position_size(strategy, account_balance, risk_per_trade)
        
        assert position_size == Decimal("0")
    
    def test_validate_risk_management(self, strategy_protocol, sample_strategy_data):
        """Тест валидации управления рисками."""
        strategy = Strategy(**sample_strategy_data)
        
        is_valid = strategy_protocol.validate_risk_management(strategy)
        
        assert is_valid is True
    
    def test_validate_risk_management_no_stop_loss(self, strategy_protocol):
        """Тест валидации управления рисками без стоп-лосса."""
        strategy_data = {
            "strategy_id": "strategy_001",
            "name": "Test Strategy",
            "type": "trend_following",
            "status": "active",
            "parameters": {
                "lookback_period": 20,
                "threshold": 0.02,
                # Нет stop_loss
                "take_profit": 0.10
            },
            "symbols": ["BTC/USDT"]
        }
        
        strategy = Strategy(**strategy_data)
        
        is_valid = strategy_protocol.validate_risk_management(strategy)
        
        assert is_valid is False
    
    def test_calculate_performance_metrics(self, strategy_protocol, sample_strategy_data):
        """Тест расчета метрик производительности."""
        strategy = Strategy(**sample_strategy_data)
        
        performance_metrics = strategy_protocol.calculate_performance_metrics(strategy)
        
        assert isinstance(performance_metrics, dict)
        assert "total_return" in performance_metrics
        assert "annualized_return" in performance_metrics
        assert "volatility" in performance_metrics
        assert "sharpe_ratio" in performance_metrics
        assert "sortino_ratio" in performance_metrics
        assert "calmar_ratio" in performance_metrics
    
    def test_generate_strategy_report(self, strategy_protocol, sample_strategy_data):
        """Тест генерации отчета по стратегии."""
        strategy = Strategy(**sample_strategy_data)
        
        report = strategy_protocol.generate_strategy_report(strategy)
        
        assert isinstance(report, dict)
        assert "strategy_info" in report
        assert "performance_metrics" in report
        assert "risk_metrics" in report
        assert "trading_history" in report
        assert "recommendations" in report
    
    def test_validate_strategy_configuration(self, strategy_protocol, sample_strategy_data):
        """Тест валидации конфигурации стратегии."""
        strategy = Strategy(**sample_strategy_data)
        
        is_valid = strategy_protocol.validate_strategy_configuration(strategy)
        
        assert is_valid is True
    
    def test_validate_strategy_configuration_missing_required_fields(self, strategy_protocol):
        """Тест валидации конфигурации с отсутствующими обязательными полями."""
        incomplete_data = {
            "strategy_id": "strategy_001",
            # Отсутствует name
            "type": "trend_following",
            "status": "active",
            "parameters": {},
            "symbols": []
        }
        
        strategy = Strategy(**incomplete_data)
        
        is_valid = strategy_protocol.validate_strategy_configuration(strategy)
        
        assert is_valid is False
    
    def test_calculate_optimal_parameters(self, strategy_protocol, sample_strategy_data):
        """Тест расчета оптимальных параметров."""
        strategy = Strategy(**sample_strategy_data)
        historical_data = [
            {"price": "50000", "volume": "1000", "timestamp": "2024-01-01T00:00:00Z"},
            {"price": "51000", "volume": "1100", "timestamp": "2024-01-01T01:00:00Z"},
            {"price": "52000", "volume": "1200", "timestamp": "2024-01-01T02:00:00Z"}
        ]
        
        optimal_params = strategy_protocol.calculate_optimal_parameters(strategy, historical_data)
        
        assert isinstance(optimal_params, dict)
        assert "lookback_period" in optimal_params
        assert "threshold" in optimal_params
        assert "stop_loss" in optimal_params
        assert "take_profit" in optimal_params
    
    def test_backtest_strategy(self, strategy_protocol, sample_strategy_data):
        """Тест бэктестинга стратегии."""
        strategy = Strategy(**sample_strategy_data)
        historical_data = [
            {"price": "50000", "volume": "1000", "timestamp": "2024-01-01T00:00:00Z"},
            {"price": "51000", "volume": "1100", "timestamp": "2024-01-01T01:00:00Z"},
            {"price": "52000", "volume": "1200", "timestamp": "2024-01-01T02:00:00Z"}
        ]
        
        backtest_results = strategy_protocol.backtest_strategy(strategy, historical_data)
        
        assert isinstance(backtest_results, dict)
        assert "total_trades" in backtest_results
        assert "win_rate" in backtest_results
        assert "profit_factor" in backtest_results
        assert "total_return" in backtest_results
        assert "max_drawdown" in backtest_results
        assert "trades" in backtest_results 