#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты всех математических операций финансовой системы.
Критически важно для финансовой системы - каждая формула должна быть точной.
"""

import pytest
import math
import numpy as np
from decimal import Decimal, getcontext
from typing import List, Tuple, Dict, Any
from scipy import stats
from scipy.optimize import minimize
import statistics

from domain.value_objects.money import Money
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage
from domain.mathematics.financial_calculations import (
    FinancialCalculator, RiskCalculator, PortfolioOptimizer,
    OptionsPricingModel, DerivativesCalculator, VolatilityCalculator
)
from domain.mathematics.statistical_models import (
    StatisticalAnalyzer, TimeSeriesAnalyzer, CorrelationAnalyzer,
    RegressionAnalyzer, MonteCarloSimulator
)
from domain.mathematics.quantum_math import (
    QuantumCalculator, QuantumPortfolioOptimizer, QuantumRiskModel
)


class TestAdvancedMathematicalOperations:
    """Comprehensive тесты математических операций."""

    def setup_method(self):
        """Настройка для каждого теста."""
        getcontext().prec = 50  # Высокая точность для математических операций

    @pytest.fixture
    def financial_calculator(self) -> FinancialCalculator:
        """Фикстура финансового калькулятора."""
        return FinancialCalculator(precision=Decimal('0.00000001'))

    @pytest.fixture
    def risk_calculator(self) -> RiskCalculator:
        """Фикстура риск калькулятора."""
        return RiskCalculator(
            confidence_levels=[0.95, 0.99, 0.999],
            time_horizons=[1, 5, 10, 30],  # дни
            monte_carlo_simulations=100000
        )

    @pytest.fixture
    def portfolio_optimizer(self) -> PortfolioOptimizer:
        """Фикстура оптимизатора портфеля."""
        return PortfolioOptimizer(
            optimization_method='SLSQP',
            risk_free_rate=Decimal('0.02'),
            max_iterations=1000,
            tolerance=Decimal('1e-8')
        )

    @pytest.fixture
    def sample_price_series(self) -> List[Decimal]:
        """Фикстура ценовой серии для тестов."""
        # Реалистичная ценовая серия BTC
        base_price = Decimal('45000')
        prices = [base_price]
        
        # Симулируем случайные изменения цены
        for i in range(1000):
            change_percent = Decimal(str(np.random.normal(0, 0.02)))  # 2% дневная волатильность
            new_price = prices[-1] * (Decimal('1') + change_percent)
            prices.append(max(new_price, Decimal('1000')))  # Минимум $1000
        
        return prices

    def test_present_value_calculations(
        self, 
        financial_calculator: FinancialCalculator
    ) -> None:
        """Тест расчетов приведенной стоимости."""
        
        # Test случаи для NPV
        cash_flows = [
            Decimal('-100000'),  # Начальная инвестиция
            Decimal('30000'),    # Год 1
            Decimal('40000'),    # Год 2
            Decimal('50000'),    # Год 3
            Decimal('20000')     # Год 4
        ]
        
        discount_rates = [
            Decimal('0.05'),   # 5%
            Decimal('0.10'),   # 10%
            Decimal('0.15'),   # 15%
            Decimal('0.20')    # 20%
        ]
        
        for discount_rate in discount_rates:
            npv = financial_calculator.calculate_npv(cash_flows, discount_rate)
            
            # NPV должна уменьшаться с ростом ставки дисконтирования
            if discount_rate == Decimal('0.05'):
                assert npv > Decimal('0')  # Проект прибыльный при 5%
            elif discount_rate == Decimal('0.20'):
                assert npv < Decimal('0')  # Проект убыточный при 20%
        
        # Test IRR (внутренняя норма доходности)
        irr = financial_calculator.calculate_irr(cash_flows)
        assert Decimal('0.10') < irr < Decimal('0.25')  # IRR между 10% и 25%
        
        # Проверка: NPV при IRR должна быть близка к 0
        npv_at_irr = financial_calculator.calculate_npv(cash_flows, irr)
        assert abs(npv_at_irr) < Decimal('1')  # Погрешность менее $1

    def test_compound_interest_precision(
        self, 
        financial_calculator: FinancialCalculator
    ) -> None:
        """Тест точности сложных процентов."""
        
        principal = Decimal('100000.00')
        annual_rate = Decimal('0.05')
        years = 30
        
        # Различные периоды капитализации
        compounding_frequencies = {
            'annually': 1,
            'semi_annually': 2,
            'quarterly': 4,
            'monthly': 12,
            'weekly': 52,
            'daily': 365,
            'continuous': float('inf')
        }
        
        results = {}
        
        for frequency_name, n in compounding_frequencies.items():
            if frequency_name == 'continuous':
                # Непрерывная капитализация: A = Pe^(rt)
                import math
                final_amount = principal * Decimal(str(math.exp(float(annual_rate * years))))
            else:
                # Дискретная капитализация: A = P(1 + r/n)^(nt)
                final_amount = financial_calculator.calculate_compound_interest(
                    principal, annual_rate, years, n
                )
            
            results[frequency_name] = final_amount
        
        # Проверяем что более частая капитализация дает больший результат
        assert results['annually'] < results['monthly']
        assert results['monthly'] < results['daily']
        assert results['daily'] < results['continuous']
        
        # Проверяем разумные пределы
        assert Decimal('400000') < results['continuous'] < Decimal('500000')

    def test_options_pricing_black_scholes(
        self, 
        financial_calculator: FinancialCalculator
    ) -> None:
        """Тест модели Блэка-Шоулза для оценки опционов."""
        
        # Параметры опциона
        spot_price = Decimal('100.00')
        strike_price = Decimal('105.00')
        time_to_expiry = Decimal('0.25')  # 3 месяца
        risk_free_rate = Decimal('0.05')  # 5%
        volatility = Decimal('0.20')      # 20%
        
        # Call опцион
        call_price = financial_calculator.black_scholes_call(
            spot_price, strike_price, time_to_expiry, 
            risk_free_rate, volatility
        )
        
        # Put опцион
        put_price = financial_calculator.black_scholes_put(
            spot_price, strike_price, time_to_expiry,
            risk_free_rate, volatility
        )
        
        # Проверяем Put-Call паритет
        # C - P = S - Ke^(-rT)
        discount_factor = financial_calculator.calculate_discount_factor(
            risk_free_rate, time_to_expiry
        )
        theoretical_diff = spot_price - strike_price * discount_factor
        actual_diff = call_price - put_price
        
        assert abs(actual_diff - theoretical_diff) < Decimal('0.01')
        
        # Проверяем Greeks (чувствительности)
        greeks = financial_calculator.calculate_option_greeks(
            spot_price, strike_price, time_to_expiry,
            risk_free_rate, volatility, option_type='call'
        )
        
        # Delta должна быть между 0 и 1 для call опциона
        assert Decimal('0') < greeks['delta'] < Decimal('1')
        
        # Gamma должна быть положительной
        assert greeks['gamma'] > Decimal('0')
        
        # Vega должна быть положительной (опцион дорожает с ростом волатильности)
        assert greeks['vega'] > Decimal('0')
        
        # Theta обычно отрицательная (временное затухание)
        assert greeks['theta'] < Decimal('0')

    def test_value_at_risk_calculations(
        self, 
        risk_calculator: RiskCalculator,
        sample_price_series: List[Decimal]
    ) -> None:
        """Тест расчетов Value at Risk."""
        
        # Рассчитываем дневные доходности
        returns = []
        for i in range(1, len(sample_price_series)):
            daily_return = (sample_price_series[i] - sample_price_series[i-1]) / sample_price_series[i-1]
            returns.append(daily_return)
        
        portfolio_value = Decimal('1000000')  # $1M портфель
        
        # Различные методы расчета VaR
        var_methods = ['historical', 'parametric', 'monte_carlo']
        confidence_levels = [Decimal('0.95'), Decimal('0.99'), Decimal('0.999')]
        
        var_results = {}
        
        for method in var_methods:
            var_results[method] = {}
            for confidence in confidence_levels:
                var = risk_calculator.calculate_var(
                    returns, portfolio_value, confidence, method
                )
                var_results[method][confidence] = var
                
                # VaR должна увеличиваться с ростом confidence level
                assert var > Decimal('0')
                
                if confidence == Decimal('0.99'):
                    assert var > var_results[method].get(Decimal('0.95'), Decimal('0'))
        
        # Expected Shortfall (CVaR) должна быть больше VaR
        for confidence in confidence_levels:
            var_95 = var_results['historical'][confidence]
            cvar = risk_calculator.calculate_expected_shortfall(
                returns, portfolio_value, confidence
            )
            assert cvar > var_95
        
        # Проверяем backtesting VaR
        backtest_results = risk_calculator.backtest_var(
            returns[-250:],  # Последние 250 дней
            var_results['historical'][Decimal('0.95')],
            confidence_level=Decimal('0.95')
        )
        
        # Количество нарушений должно быть близко к expected (5% для 95% VaR)
        expected_violations = int(250 * 0.05)
        assert abs(backtest_results['violations'] - expected_violations) <= 5

    def test_portfolio_optimization_markowitz(
        self, 
        portfolio_optimizer: PortfolioOptimizer
    ) -> None:
        """Тест оптимизации портфеля по Марковицу."""
        
        # Исторические доходности для 4 активов
        asset_returns = {
            'BTC': [Decimal(str(x)) for x in [0.02, -0.01, 0.03, -0.02, 0.04, 0.01, -0.03, 0.05]],
            'ETH': [Decimal(str(x)) for x in [0.015, -0.008, 0.025, -0.015, 0.035, 0.008, -0.025, 0.04]],
            'STOCKS': [Decimal(str(x)) for x in [0.008, 0.005, 0.01, -0.005, 0.012, 0.007, 0.003, 0.009]],
            'BONDS': [Decimal(str(x)) for x in [0.002, 0.003, 0.0025, 0.004, 0.0015, 0.003, 0.0035, 0.002]]
        }
        
        # Рассчитываем ковариационную матрицу
        covariance_matrix = portfolio_optimizer.calculate_covariance_matrix(asset_returns)
        expected_returns = portfolio_optimizer.calculate_expected_returns(asset_returns)
        
        # Оптимизация для максимального Sharpe ratio
        optimal_weights = portfolio_optimizer.optimize_sharpe_ratio(
            expected_returns, covariance_matrix
        )
        
        # Сумма весов должна равняться 1
        weights_sum = sum(optimal_weights.values())
        assert abs(weights_sum - Decimal('1')) < Decimal('0.0001')
        
        # Все веса должны быть неотрицательными (long-only constraint)
        assert all(weight >= Decimal('0') for weight in optimal_weights.values())
        
        # Рассчитываем метрики оптимального портфеля
        portfolio_return = portfolio_optimizer.calculate_portfolio_return(
            optimal_weights, expected_returns
        )
        portfolio_risk = portfolio_optimizer.calculate_portfolio_risk(
            optimal_weights, covariance_matrix
        )
        sharpe_ratio = portfolio_optimizer.calculate_sharpe_ratio(
            portfolio_return, portfolio_risk
        )
        
        # Sharpe ratio должен быть положительным и разумным
        assert Decimal('0') < sharpe_ratio < Decimal('5')
        
        # Эффективная граница
        target_returns = [Decimal('0.005'), Decimal('0.01'), Decimal('0.015'), Decimal('0.02')]
        efficient_frontier = []
        
        for target_return in target_returns:
            min_var_weights = portfolio_optimizer.optimize_minimum_variance(
                expected_returns, covariance_matrix, target_return
            )
            min_risk = portfolio_optimizer.calculate_portfolio_risk(
                min_var_weights, covariance_matrix
            )
            efficient_frontier.append((target_return, min_risk))
        
        # Риск должен увеличиваться с ростом ожидаемой доходности
        for i in range(1, len(efficient_frontier)):
            assert efficient_frontier[i][1] >= efficient_frontier[i-1][1]

    def test_correlation_and_regression_analysis(
        self, 
        financial_calculator: FinancialCalculator
    ) -> None:
        """Тест корреляционного и регрессионного анализа."""
        
        # Генерируем данные с известной корреляцией
        np.random.seed(42)  # Для воспроизводимости
        
        x = np.random.normal(0, 1, 1000)
        y = 0.7 * x + 0.3 * np.random.normal(0, 1, 1000)  # Корреляция ~0.7
        z = np.random.normal(0, 1, 1000)  # Независимая переменная
        
        x_decimal = [Decimal(str(val)) for val in x]
        y_decimal = [Decimal(str(val)) for val in y]
        z_decimal = [Decimal(str(val)) for val in z]
        
        # Корреляционный анализ
        correlation_xy = financial_calculator.calculate_correlation(x_decimal, y_decimal)
        correlation_xz = financial_calculator.calculate_correlation(x_decimal, z_decimal)
        
        # Корреляция x и y должна быть высокой
        assert Decimal('0.6') < abs(correlation_xy) < Decimal('0.8')
        
        # Корреляция x и z должна быть низкой
        assert abs(correlation_xz) < Decimal('0.1')
        
        # Линейная регрессия y = a + b*x
        regression_result = financial_calculator.linear_regression(x_decimal, y_decimal)
        
        # Коэффициент наклона должен быть близок к 0.7
        assert Decimal('0.6') < regression_result['slope'] < Decimal('0.8')
        
        # R-squared должен быть высоким для хорошей подгонки
        assert regression_result['r_squared'] > Decimal('0.4')
        
        # Проверяем статистическую значимость
        assert regression_result['p_value_slope'] < Decimal('0.01')  # Значимо на 1% уровне
        
        # Полиномиальная регрессия
        poly_result = financial_calculator.polynomial_regression(
            x_decimal, y_decimal, degree=2
        )
        
        # Полиномиальная регрессия должна дать лучшую подгонку или такую же
        assert poly_result['r_squared'] >= regression_result['r_squared']

    def test_time_series_analysis(
        self, 
        sample_price_series: List[Decimal]
    ) -> None:
        """Тест анализа временных рядов."""
        
        time_series_analyzer = TimeSeriesAnalyzer()
        
        # Тест на стационарность (ADF test)
        prices = sample_price_series[:500]  # Первые 500 наблюдений
        returns = []
        for i in range(1, len(prices)):
            ret = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(ret)
        
        # Цены обычно нестационарны
        price_stationarity = time_series_analyzer.adf_test(prices)
        assert price_stationarity['is_stationary'] is False
        
        # Доходности обычно стационарны
        returns_stationarity = time_series_analyzer.adf_test(returns)
        assert returns_stationarity['is_stationary'] is True
        
        # Автокорреляционная функция
        autocorr = time_series_analyzer.calculate_autocorrelation(returns, max_lags=20)
        
        # Первый лаг автокорреляции доходностей должен быть близок к 0
        assert abs(autocorr[1]) < Decimal('0.1')
        
        # ARCH эффекты (кластеризация волатильности)
        squared_returns = [ret ** 2 for ret in returns]
        arch_test = time_series_analyzer.arch_test(squared_returns)
        
        # Для реалистичных финансовых данных часто есть ARCH эффекты
        if arch_test['p_value'] < Decimal('0.05'):
            assert arch_test['arch_effects'] is True
        
        # GARCH модель для моделирования волатильности
        garch_params = time_series_analyzer.fit_garch(returns, p=1, q=1)
        
        # Параметры GARCH должны быть положительными
        assert garch_params['omega'] > Decimal('0')
        assert garch_params['alpha'] > Decimal('0')
        assert garch_params['beta'] > Decimal('0')
        
        # Условие стационарности: alpha + beta < 1
        assert garch_params['alpha'] + garch_params['beta'] < Decimal('1')

    def test_monte_carlo_simulations(
        self, 
        risk_calculator: RiskCalculator
    ) -> None:
        """Тест Монте-Карло симуляций."""
        
        # Параметры для симуляции
        initial_price = Decimal('45000')
        drift = Decimal('0.0005')  # Дневной drift
        volatility = Decimal('0.02')  # Дневная волатильность
        time_steps = 252  # Торговых дней в году
        num_simulations = 10000
        
        # Геометрическое броуновское движение
        simulated_paths = risk_calculator.monte_carlo_gbm(
            initial_price, drift, volatility, time_steps, num_simulations
        )
        
        assert len(simulated_paths) == num_simulations
        assert len(simulated_paths[0]) == time_steps + 1  # +1 для начальной цены
        
        # Все цены должны быть положительными
        for path in simulated_paths[:100]:  # Проверяем первые 100 путей
            assert all(price > Decimal('0') for price in path)
        
        # Финальные цены
        final_prices = [path[-1] for path in simulated_paths]
        
        # Проверяем что распределение разумно
        mean_final = sum(final_prices) / len(final_prices)
        std_final = Decimal(str(np.std([float(p) for p in final_prices])))
        
        # Среднее должно быть близко к expected value
        expected_final = initial_price * (Decimal('1') + drift) ** time_steps
        relative_error = abs(mean_final - expected_final) / expected_final
        assert relative_error < Decimal('0.05')  # 5% погрешность
        
        # VaR из симуляции
        var_95 = risk_calculator.calculate_var_from_simulation(
            initial_price, final_prices, Decimal('0.95')
        )
        
        assert var_95 > Decimal('0')
        assert var_95 < initial_price  # VaR не может быть больше начальной стоимости

    def test_quantum_portfolio_optimization(self) -> None:
        """Тест квантовой оптимизации портфеля."""
        
        quantum_optimizer = QuantumPortfolioOptimizer(
            num_qubits=4,  # 4 актива
            num_layers=3,
            backend='simulator'
        )
        
        # Данные для оптимизации
        expected_returns = {
            'asset_1': Decimal('0.12'),
            'asset_2': Decimal('0.10'),
            'asset_3': Decimal('0.08'),
            'asset_4': Decimal('0.06')
        }
        
        risk_matrix = [
            [Decimal('0.04'), Decimal('0.01'), Decimal('0.005'), Decimal('0.002')],
            [Decimal('0.01'), Decimal('0.03'), Decimal('0.008'), Decimal('0.003')],
            [Decimal('0.005'), Decimal('0.008'), Decimal('0.02'), Decimal('0.004')],
            [Decimal('0.002'), Decimal('0.003'), Decimal('0.004'), Decimal('0.015')]
        ]
        
        # Квантовая оптимизация
        quantum_weights = quantum_optimizer.optimize_portfolio(
            expected_returns, risk_matrix, risk_aversion=Decimal('2.0')
        )
        
        # Сумма весов должна равняться 1
        weights_sum = sum(quantum_weights.values())
        assert abs(weights_sum - Decimal('1')) < Decimal('0.01')
        
        # Веса должны быть неотрицательными
        assert all(weight >= Decimal('0') for weight in quantum_weights.values())
        
        # Сравнение с классической оптимизацией
        classical_optimizer = PortfolioOptimizer()
        classical_weights = classical_optimizer.optimize_mean_variance(
            expected_returns, risk_matrix, risk_aversion=Decimal('2.0')
        )
        
        # Квантовое решение должно быть близко к классическому
        for asset in expected_returns.keys():
            weight_diff = abs(quantum_weights[asset] - classical_weights[asset])
            assert weight_diff < Decimal('0.1')  # Максимум 10% отличие

    def test_advanced_derivatives_pricing(
        self, 
        financial_calculator: FinancialCalculator
    ) -> None:
        """Тест оценки сложных деривативов."""
        
        derivatives_calculator = DerivativesCalculator()
        
        # Барьерный опцион
        barrier_params = {
            'spot_price': Decimal('100'),
            'strike_price': Decimal('105'),
            'barrier_level': Decimal('120'),
            'time_to_expiry': Decimal('0.25'),
            'risk_free_rate': Decimal('0.05'),
            'volatility': Decimal('0.20'),
            'barrier_type': 'up_and_out'
        }
        
        barrier_price = derivatives_calculator.price_barrier_option(**barrier_params)
        
        # Барьерный опцион должен стоить меньше обычного
        regular_call = financial_calculator.black_scholes_call(
            barrier_params['spot_price'],
            barrier_params['strike_price'],
            barrier_params['time_to_expiry'],
            barrier_params['risk_free_rate'],
            barrier_params['volatility']
        )
        
        assert barrier_price < regular_call
        assert barrier_price > Decimal('0')
        
        # Азиатский опцион
        asian_params = {
            'spot_price': Decimal('100'),
            'strike_price': Decimal('100'),
            'time_to_expiry': Decimal('0.25'),
            'risk_free_rate': Decimal('0.05'),
            'volatility': Decimal('0.20'),
            'num_observations': 63  # Ежедневные наблюдения за квартал
        }
        
        asian_call = derivatives_calculator.price_asian_option(**asian_params)
        
        # Азиатский опцион должен стоить меньше европейского
        european_call = financial_calculator.black_scholes_call(
            asian_params['spot_price'],
            asian_params['strike_price'],
            asian_params['time_to_expiry'],
            asian_params['risk_free_rate'],
            asian_params['volatility']
        )
        
        assert asian_call < european_call
        assert asian_call > Decimal('0')

    def test_volatility_surface_modeling(self) -> None:
        """Тест моделирования поверхности волатильности."""
        
        vol_calculator = VolatilityCalculator()
        
        # Данные implied volatility для разных strikes и expiries
        volatility_data = {
            (Decimal('0.25'), Decimal('90')): Decimal('0.25'),   # 3M, 90 strike
            (Decimal('0.25'), Decimal('100')): Decimal('0.20'),  # 3M, 100 strike (ATM)
            (Decimal('0.25'), Decimal('110')): Decimal('0.23'),  # 3M, 110 strike
            (Decimal('0.5'), Decimal('90')): Decimal('0.23'),    # 6M, 90 strike
            (Decimal('0.5'), Decimal('100')): Decimal('0.19'),   # 6M, 100 strike
            (Decimal('0.5'), Decimal('110')): Decimal('0.21'),   # 6M, 110 strike
            (Decimal('1.0'), Decimal('90')): Decimal('0.21'),    # 1Y, 90 strike
            (Decimal('1.0'), Decimal('100')): Decimal('0.18'),   # 1Y, 100 strike
            (Decimal('1.0'), Decimal('110')): Decimal('0.19'),   # 1Y, 110 strike
        }
        
        # Построение поверхности волатильности
        vol_surface = vol_calculator.build_volatility_surface(volatility_data)
        
        # Интерполяция волатильности
        interpolated_vol = vol_surface.get_volatility(
            time_to_expiry=Decimal('0.75'),  # 9 месяцев
            strike=Decimal('95')             # Strike между точками
        )
        
        assert Decimal('0.15') < interpolated_vol < Decimal('0.30')
        
        # Проверяем volatility smile/skew
        strikes = [Decimal('90'), Decimal('95'), Decimal('100'), Decimal('105'), Decimal('110')]
        smile_vols = []
        
        for strike in strikes:
            vol = vol_surface.get_volatility(Decimal('0.25'), strike)
            smile_vols.append(vol)
        
        # ATM volatility должна быть минимальной (smile эффект)
        atm_vol = vol_surface.get_volatility(Decimal('0.25'), Decimal('100'))
        assert all(vol >= atm_vol for vol in smile_vols)

    def test_risk_attribution_analysis(
        self, 
        risk_calculator: RiskCalculator
    ) -> None:
        """Тест анализа риск-атрибуции."""
        
        # Портфель с весами и факторами риска
        portfolio_weights = {
            'TECH_STOCKS': Decimal('0.4'),
            'FINANCIAL_STOCKS': Decimal('0.3'),
            'CRYPTO': Decimal('0.2'),
            'BONDS': Decimal('0.1')
        }
        
        # Факторные нагрузки (betas)
        factor_loadings = {
            'TECH_STOCKS': {
                'market': Decimal('1.2'),
                'tech_factor': Decimal('1.0'),
                'volatility_factor': Decimal('0.8')
            },
            'FINANCIAL_STOCKS': {
                'market': Decimal('1.1'),
                'interest_rate_factor': Decimal('1.5'),
                'volatility_factor': Decimal('0.6')
            },
            'CRYPTO': {
                'market': Decimal('0.3'),
                'crypto_factor': Decimal('1.0'),
                'volatility_factor': Decimal('2.0')
            },
            'BONDS': {
                'market': Decimal('-0.2'),
                'interest_rate_factor': Decimal('-1.0'),
                'volatility_factor': Decimal('-0.3')
            }
        }
        
        # Факторные волатильности
        factor_volatilities = {
            'market': Decimal('0.16'),
            'tech_factor': Decimal('0.25'),
            'interest_rate_factor': Decimal('0.12'),
            'crypto_factor': Decimal('0.80'),
            'volatility_factor': Decimal('0.30')
        }
        
        # Факторные корреляции
        factor_correlations = {
            ('market', 'tech_factor'): Decimal('0.6'),
            ('market', 'interest_rate_factor'): Decimal('-0.3'),
            ('market', 'crypto_factor'): Decimal('0.2'),
            ('market', 'volatility_factor'): Decimal('0.4'),
            ('tech_factor', 'crypto_factor'): Decimal('0.3'),
            # ... остальные корреляции
        }
        
        # Анализ риск-атрибуции
        risk_attribution = risk_calculator.calculate_risk_attribution(
            portfolio_weights, factor_loadings, factor_volatilities, factor_correlations
        )
        
        # Общий риск портфеля
        total_risk = risk_attribution['total_portfolio_risk']
        assert total_risk > Decimal('0')
        
        # Компонентные риски
        component_risks = risk_attribution['component_risks']
        
        # Сумма компонентных рисков должна равняться общему риску
        sum_component_risks = sum(component_risks.values())
        assert abs(sum_component_risks - total_risk) < Decimal('0.001')
        
        # Факторные вклады
        factor_contributions = risk_attribution['factor_contributions']
        
        # Crypto factor должен вносить значительный вклад из-за высокой волатильности
        assert factor_contributions['crypto_factor'] > factor_contributions['interest_rate_factor']
        
        # Диверсификационный эффект
        diversification_ratio = risk_attribution['diversification_ratio']
        assert Decimal('0') < diversification_ratio < Decimal('1')  # Портфель диверсифицирован

    def test_stress_testing_scenarios(
        self, 
        risk_calculator: RiskCalculator
    ) -> None:
        """Тест стресс-тестирования сценариев."""
        
        # Базовый портфель
        portfolio = {
            'positions': {
                'BTC': {'quantity': Decimal('10'), 'current_price': Decimal('45000')},
                'ETH': {'quantity': Decimal('100'), 'current_price': Decimal('3200')},
                'STOCKS': {'value': Decimal('500000')},
                'BONDS': {'value': Decimal('200000')}
            }
        }
        
        # Стресс-сценарии
        stress_scenarios = {
            'market_crash_2008': {
                'BTC': Decimal('-0.50'),    # -50%
                'ETH': Decimal('-0.45'),    # -45%
                'STOCKS': Decimal('-0.37'), # -37%
                'BONDS': Decimal('0.05')    # +5%
            },
            'crypto_winter': {
                'BTC': Decimal('-0.80'),    # -80%
                'ETH': Decimal('-0.75'),    # -75%
                'STOCKS': Decimal('-0.10'), # -10%
                'BONDS': Decimal('0.02')    # +2%
            },
            'inflation_shock': {
                'BTC': Decimal('0.20'),     # +20%
                'ETH': Decimal('0.15'),     # +15%
                'STOCKS': Decimal('-0.20'), # -20%
                'BONDS': Decimal('-0.15')   # -15%
            },
            'tech_bubble_burst': {
                'BTC': Decimal('-0.30'),    # -30%
                'ETH': Decimal('-0.35'),    # -35%
                'STOCKS': Decimal('-0.50'), # -50%
                'BONDS': Decimal('0.10')    # +10%
            }
        }
        
        stress_results = {}
        
        for scenario_name, shocks in stress_scenarios.items():
            scenario_pnl = risk_calculator.calculate_stress_test_pnl(
                portfolio, shocks
            )
            stress_results[scenario_name] = scenario_pnl
        
        # Проверяем результаты
        assert stress_results['market_crash_2008'] < Decimal('0')  # Убыток
        assert stress_results['crypto_winter'] < stress_results['market_crash_2008']  # Больший убыток
        
        # Inflation shock может быть прибыльным из-за crypto hedge
        inflation_result = stress_results['inflation_shock']
        
        # Worst case scenario
        worst_case_pnl = min(stress_results.values())
        
        # Максимальный убыток не должен превышать разумных пределов
        portfolio_value = risk_calculator.calculate_portfolio_value(portfolio)
        max_loss_ratio = abs(worst_case_pnl) / portfolio_value
        
        assert max_loss_ratio < Decimal('0.90')  # Максимум 90% потерь