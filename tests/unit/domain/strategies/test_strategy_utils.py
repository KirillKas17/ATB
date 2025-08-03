import pytest
from typing import Any, Dict, List, Optional, Union, AsyncGenerator
from decimal import Decimal
from datetime import datetime, timedelta
from domain.strategies.utils import StrategyUtils
from domain.entities.market import MarketData, OrderBook, Trade
class TestStrategyUtils:
    @pytest.fixture
    def utils(self) -> Any:
        return StrategyUtils()
    @pytest.fixture
    def sample_market_data(self) -> Any:
        """Создает тестовые рыночные данные."""
        data = []
        base_price = Decimal("50000")
        for i in range(100):
            timestamp = datetime.now() + timedelta(minutes=i)
            price = base_price + Decimal(str(i * 10))
            market_data = MarketData(
                symbol="BTC/USDT",
                timestamp=timestamp,
                price=price,
                volume=Decimal("1000"),
                high=price + Decimal("100"),
                low=price - Decimal("100"),
                open_price=price - Decimal("50"),
                close_price=price,
                order_book=OrderBook(
                    symbol="BTC/USDT",
                    timestamp=timestamp,
                    bids=[{"price": price - Decimal("1"), "size": Decimal("1.0")}],
                    asks=[{"price": price + Decimal("1"), "size": Decimal("1.0")}]
                ),
                trades=[
                    Trade(
                        id=f"trade_{i}",
                        symbol="BTC/USDT",
                        price=price,
                        size=Decimal("0.1"),
                        side="buy",
                        timestamp=timestamp
                    )
                ]
            )
            data.append(market_data)
        return data
    def test_calculate_sharpe_ratio(self, utils, sample_market_data) -> None:
        """Тест расчета коэффициента Шарпа."""
        # Создаем серию доходностей
        returns = [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), Decimal("0.03"), Decimal("0.01")]
        sharpe_ratio = utils.calculate_sharpe_ratio(returns, risk_free_rate=Decimal("0.02"))
        assert isinstance(sharpe_ratio, Decimal)
        assert sharpe_ratio > Decimal("0")  # Положительный Sharpe ratio для положительных доходностей
    def test_calculate_max_drawdown(self, utils) -> None:
        """Тест расчета максимальной просадки."""
        # Создаем серию цен
        prices = [
            Decimal("100"), Decimal("110"), Decimal("105"), 
            Decimal("95"), Decimal("90"), Decimal("100")
        ]
        max_dd = utils.calculate_max_drawdown(prices)
        assert isinstance(max_dd, Decimal)
        assert max_dd >= Decimal("0")
        assert max_dd <= Decimal("1")  # Просадка не может быть больше 100%
    def test_calculate_var(self, utils) -> None:
        """Тест расчета Value at Risk."""
        returns = [
            Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), 
            Decimal("0.03"), Decimal("-0.02"), Decimal("0.01")
        ]
        var_95 = utils.calculate_var(returns, confidence_level=Decimal("0.95"))
        var_99 = utils.calculate_var(returns, confidence_level=Decimal("0.99"))
        assert isinstance(var_95, Decimal)
        assert isinstance(var_99, Decimal)
        assert var_99 <= var_95  # VaR 99% должен быть меньше или равен VaR 95%
    def test_optimize_parameters(self, utils) -> None:
        """Тест оптимизации параметров стратегии."""
        # Определяем функцию для оптимизации (заглушка)
        def objective_function(params) -> Any:
            return float(params.get('param1', 0) + params.get('param2', 0))
        # Параметры для оптимизации
        param_ranges = {
            'param1': (1, 10),
            'param2': (5, 15)
        }
        best_params = utils.optimize_parameters(
            objective_function=objective_function,
            param_ranges=param_ranges,
            optimization_method="grid_search",
            max_iterations=10
        )
        assert isinstance(best_params, dict)
        assert 'param1' in best_params
        assert 'param2' in best_params
        assert best_params['param1'] >= 1 and best_params['param1'] <= 10
        assert best_params['param2'] >= 5 and best_params['param2'] <= 15
    def test_calculate_position_size(self, utils) -> None:
        """Тест расчета размера позиции."""
        account_balance = Decimal("10000")
        risk_per_trade = Decimal("0.02")  # 2% риска на сделку
        stop_loss_pct = Decimal("0.05")   # 5% стоп-лосс
        position_size = utils.calculate_position_size(
            account_balance=account_balance,
            risk_per_trade=risk_per_trade,
            stop_loss_pct=stop_loss_pct
        )
        assert isinstance(position_size, Decimal)
        assert position_size > Decimal("0")
        assert position_size <= account_balance
    def test_calculate_risk_metrics(self, utils) -> None:
        """Тест расчета метрик риска."""
        returns = [
            Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), 
            Decimal("0.03"), Decimal("-0.02"), Decimal("0.01")
        ]
        risk_metrics = utils.calculate_risk_metrics(returns)
        assert isinstance(risk_metrics, dict)
        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'var_99' in risk_metrics
        assert risk_metrics['volatility'] >= Decimal("0")
        assert risk_metrics['max_drawdown'] >= Decimal("0")
        assert risk_metrics['max_drawdown'] <= Decimal("1")
    def test_backtest_strategy(self, utils, sample_market_data) -> None:
        """Тест бэктестинга стратегии."""
        # Простая стратегия (заглушка)
        def strategy_logic(market_data, params) -> Any:
            # Простая логика: покупаем если цена растет
            if len(market_data) < 2:
                return None
            current_price = market_data[-1].price
            previous_price = market_data[-2].price
            if current_price > previous_price:
                return {
                    'action': 'buy',
                    'confidence': Decimal("0.7"),
                    'price': current_price
                }
            elif current_price < previous_price:
                return {
                    'action': 'sell',
                    'confidence': Decimal("0.6"),
                    'price': current_price
                }
            return None
        params = {'threshold': Decimal("0.01")}
        results = utils.backtest_strategy(
            strategy_logic=strategy_logic,
            market_data=sample_market_data,
            parameters=params,
            initial_capital=Decimal("10000")
        )
        assert isinstance(results, dict)
        assert 'total_return' in results
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'total_trades' in results
        assert 'win_rate' in results
    def test_analyze_market_regime(self, utils, sample_market_data) -> None:
        """Тест анализа рыночного режима."""
        market_regime = utils.analyze_market_regime(sample_market_data)
        assert isinstance(market_regime, dict)
        assert 'regime_type' in market_regime
        assert 'volatility' in market_regime
        assert 'trend_strength' in market_regime
        assert 'correlation' in market_regime
        assert market_regime['regime_type'] in ['trending', 'ranging', 'volatile']
        assert market_regime['volatility'] >= Decimal("0")
        assert market_regime['trend_strength'] >= Decimal("0")
        assert market_regime['trend_strength'] <= Decimal("1")
    def test_calculate_correlation_matrix(self, utils) -> None:
        """Тест расчета корреляционной матрицы."""
        # Создаем данные для нескольких активов
        asset_returns = {
            'BTC': [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01")],
            'ETH': [Decimal("0.015"), Decimal("0.025"), Decimal("-0.015")],
            'ADA': [Decimal("0.02"), Decimal("0.03"), Decimal("-0.02")]
        }
        correlation_matrix = utils.calculate_correlation_matrix(asset_returns)
        assert isinstance(correlation_matrix, dict)
        assert 'BTC' in correlation_matrix
        assert 'ETH' in correlation_matrix
        assert 'ADA' in correlation_matrix
        # Проверяем, что корреляция с самим собой равна 1
        assert correlation_matrix['BTC']['BTC'] == Decimal("1.0")
        assert correlation_matrix['ETH']['ETH'] == Decimal("1.0")
        assert correlation_matrix['ADA']['ADA'] == Decimal("1.0")
    def test_optimize_portfolio_weights(self, utils) -> None:
        """Тест оптимизации весов портфеля."""
        # Данные о доходностях активов
        asset_returns = {
            'BTC': [Decimal("0.01"), Decimal("0.02"), Decimal("-0.01")],
            'ETH': [Decimal("0.015"), Decimal("0.025"), Decimal("-0.015")],
            'ADA': [Decimal("0.02"), Decimal("0.03"), Decimal("-0.02")]
        }
        target_return = Decimal("0.15")  # 15% годовой доходности
        optimal_weights = utils.optimize_portfolio_weights(
            asset_returns=asset_returns,
            target_return=target_return,
            optimization_method="efficient_frontier"
        )
        assert isinstance(optimal_weights, dict)
        assert 'BTC' in optimal_weights
        assert 'ETH' in optimal_weights
        assert 'ADA' in optimal_weights
        # Проверяем, что веса в сумме дают 1
        total_weight = sum(optimal_weights.values())
        assert abs(total_weight - Decimal("1.0")) < Decimal("0.01")
        # Проверяем, что все веса неотрицательны
        for weight in optimal_weights.values():
            assert weight >= Decimal("0")
    def test_calculate_risk_adjusted_metrics(self, utils) -> None:
        """Тест расчета риск-скорректированных метрик."""
        returns = [
            Decimal("0.01"), Decimal("0.02"), Decimal("-0.01"), 
            Decimal("0.03"), Decimal("-0.02"), Decimal("0.01")
        ]
        metrics = utils.calculate_risk_adjusted_metrics(returns)
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'sortino_ratio' in metrics
        assert 'calmar_ratio' in metrics
        assert 'information_ratio' in metrics
        # Проверяем, что все метрики являются числами
        for metric_name, metric_value in metrics.items():
            assert isinstance(metric_value, (Decimal, float)) or metric_value is None 
