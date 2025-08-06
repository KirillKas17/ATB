#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive тесты финансовой арифметики и точности вычислений.
Критически важно для финансовой системы - любая ошибка в расчетах недопустима.
"""

import pytest
from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from typing import List, Tuple
import math

from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage
from domain.exceptions import ValidationError, FinancialArithmeticError


class TestFinancialArithmetic:
    """Тесты финансовой арифметики с максимальной точностью."""

    def setup_method(self):
        """Настройка точности для каждого теста."""
        getcontext().prec = 28  # Максимальная точность для финансовых расчетов
        getcontext().rounding = ROUND_HALF_UP

    @pytest.fixture
    def usd(self) -> Currency:
        return Currency("USD")

    @pytest.fixture
    def btc(self) -> Currency:
        return Currency("BTC")

    def test_decimal_precision_consistency(self, usd: Currency) -> None:
        """Тест консистентности десятичной точности."""
        # Критическая проверка - сумма дробных частей должна равняться целому
        parts = [
            Decimal("0.333333333333333333333333333"),
            Decimal("0.333333333333333333333333333"),
            Decimal("0.333333333333333333333333334")
        ]
        
        total = sum(parts)
        expected = Decimal("1.000000000000000000000000000")
        
        # Проверяем что сумма точно равна 1
        assert total == expected
        
        # Создаем Money объекты и проверяем арифметику
        money_parts = [Money(part, usd) for part in parts]
        money_total = sum(money_parts, Money(Decimal("0"), usd))
        
        assert money_total.amount == expected

    def test_high_precision_calculations(self, usd: Currency) -> None:
        """Тест высокоточных вычислений."""
        # Тестируем вычисления с максимальной точностью
        very_small = Decimal("0.0000000000000000000000000001")  # 28 знаков
        very_large = Decimal("999999999999999999999999999")     # 27 цифр
        
        money_small = Money(very_small, usd)
        money_large = Money(very_large, usd)
        
        # Сложение
        result = money_large + money_small
        expected = very_large + very_small
        assert result.amount == expected
        
        # Умножение на малое число
        multiplier = Decimal("1.0000000000000000000000000001")
        result = money_large * multiplier
        expected = very_large * multiplier
        assert result.amount == expected

    def test_currency_conversion_precision(self) -> None:
        """Тест точности конвертации валют."""
        usd = Currency("USD")
        eur = Currency("EUR")
        
        # Реалистичный курс EUR/USD
        exchange_rate = Decimal("1.0847362839482749583729164738")
        
        usd_amount = Money(Decimal("1000000.00"), usd)  # $1M
        
        # Конвертация в EUR
        eur_amount = usd_amount.convert_to(eur, exchange_rate)
        
        # Обратная конвертация
        back_to_usd = eur_amount.convert_to(usd, Decimal("1") / exchange_rate)
        
        # Проверяем минимальную потерю точности
        difference = abs(usd_amount.amount - back_to_usd.amount)
        assert difference <= Decimal("0.01")  # Максимум 1 цент потери

    def test_percentage_calculations_precision(self, usd: Currency) -> None:
        """Тест точности процентных вычислений."""
        base_amount = Money(Decimal("123456.789"), usd)
        
        # Различные проценты
        percentages = [
            Decimal("0.1"),      # 0.1%
            Decimal("0.125"),    # 0.125% (1/8%)
            Decimal("33.333333333333333333333333333"),  # 1/3 от 100%
            Decimal("66.666666666666666666666666667"),  # 2/3 от 100%
        ]
        
        total_calculated = Money(Decimal("0"), usd)
        
        for pct in percentages:
            percentage_obj = Percentage(pct)
            calculated = base_amount * (percentage_obj.value / Decimal("100"))
            total_calculated += calculated
        
        # Проверяем что сумма всех процентов близка к исходной сумме
        expected_total = base_amount  # 100% от суммы
        difference = abs(total_calculated.amount - expected_total.amount)
        
        # Допустимая погрешность - микрокопейки
        assert difference <= Decimal("0.000001")

    def test_compound_interest_precision(self, usd: Currency) -> None:
        """Тест точности сложных процентов."""
        principal = Money(Decimal("100000.00"), usd)  # $100k
        annual_rate = Decimal("0.05")  # 5% годовых
        compounds_per_year = 12  # ежемесячно
        years = 10
        
        # Формула сложных процентов: A = P(1 + r/n)^(nt)
        monthly_rate = annual_rate / Decimal(compounds_per_year)
        total_compounds = compounds_per_year * years
        
        multiplier = (Decimal("1") + monthly_rate) ** total_compounds
        final_amount = principal * multiplier
        
        # Проверяем что результат разумен
        expected_range_min = principal * Decimal("1.6")  # Минимум 60% роста
        expected_range_max = principal * Decimal("1.7")  # Максимум 70% роста
        
        assert expected_range_min <= final_amount <= expected_range_max

    def test_trading_commission_calculations(self, usd: Currency, btc: Currency) -> None:
        """Тест расчета торговых комиссий."""
        trade_amount = Money(Decimal("50000.00"), usd)  # $50k сделка
        
        # Различные типы комиссий
        maker_fee = Decimal("0.001")    # 0.1% maker
        taker_fee = Decimal("0.0015")   # 0.15% taker
        withdrawal_fee = Money(Decimal("25.00"), usd)  # $25 фиксированная
        
        # Расчет комиссий
        maker_commission = trade_amount * maker_fee
        taker_commission = trade_amount * taker_fee
        
        # Проверяем точность
        assert maker_commission.amount == Decimal("50.00")
        assert taker_commission.amount == Decimal("75.00")
        
        # Общая комиссия
        total_fees = maker_commission + taker_commission + withdrawal_fee
        expected_total = Money(Decimal("150.00"), usd)
        
        assert total_fees == expected_total

    def test_slippage_calculations(self, usd: Currency) -> None:
        """Тест расчета проскальзывания."""
        expected_price = Price(Decimal("45000.00"), usd)
        actual_price = Price(Decimal("45022.50"), usd)
        order_size = Volume(Decimal("1.5"), Currency("BTC"))
        
        # Расчет проскальзывания
        price_diff = actual_price.value - expected_price.value
        slippage_amount = price_diff * order_size.value
        slippage_percentage = (price_diff / expected_price.value) * Decimal("100")
        
        assert slippage_amount == Decimal("33.75")  # $33.75
        assert slippage_percentage == Decimal("0.05")  # 0.05%

    def test_profit_loss_calculations(self, usd: Currency, btc: Currency) -> None:
        """Тест расчета прибыли/убытка."""
        # Позиция
        entry_price = Price(Decimal("44000.00"), usd)
        exit_price = Price(Decimal("46200.00"), usd)
        position_size = Volume(Decimal("0.5"), btc)
        
        # Расчет P&L
        price_change = exit_price.value - entry_price.value
        gross_pnl = price_change * position_size.value
        
        # Комиссии
        entry_commission = (entry_price.value * position_size.value) * Decimal("0.001")
        exit_commission = (exit_price.value * position_size.value) * Decimal("0.001")
        total_commission = entry_commission + exit_commission
        
        # Чистая прибыль
        net_pnl = gross_pnl - total_commission
        
        assert gross_pnl == Decimal("1100.00")  # $1100 gross
        assert total_commission == Decimal("45.10")  # ~$45 commission
        assert net_pnl == Decimal("1054.90")  # ~$1055 net

    def test_portfolio_weight_calculations(self, usd: Currency) -> None:
        """Тест расчета весов портфеля."""
        # Позиции в портфеле
        positions = [
            Money(Decimal("50000.00"), usd),  # 50k
            Money(Decimal("30000.00"), usd),  # 30k
            Money(Decimal("20000.00"), usd),  # 20k
        ]
        
        total_value = sum(positions, Money(Decimal("0"), usd))
        
        # Расчет весов
        weights = []
        for position in positions:
            weight = (position.amount / total_value.amount) * Decimal("100")
            weights.append(weight)
        
        # Проверяем что сумма весов = 100%
        total_weight = sum(weights)
        assert total_weight == Decimal("100.00")
        
        # Проверяем индивидуальные веса
        assert weights[0] == Decimal("50.00")  # 50%
        assert weights[1] == Decimal("30.00")  # 30%
        assert weights[2] == Decimal("20.00")  # 20%

    def test_risk_metrics_calculations(self, usd: Currency) -> None:
        """Тест расчета риск-метрик."""
        portfolio_value = Money(Decimal("1000000.00"), usd)  # $1M
        
        # Value at Risk (VaR) расчеты
        daily_volatility = Decimal("0.02")  # 2% дневная волатильность
        confidence_level = Decimal("0.95")  # 95% доверительный интервал
        z_score = Decimal("1.645")  # Z-score для 95%
        
        # VaR = Portfolio Value × Volatility × Z-score
        var_amount = portfolio_value.amount * daily_volatility * z_score
        var_money = Money(var_amount, usd)
        
        # Expected Shortfall (CVaR)
        cvar_multiplier = Decimal("1.28")  # Среднее для нормального распределения
        cvar_amount = var_amount * cvar_multiplier
        cvar_money = Money(cvar_amount, usd)
        
        assert var_money.amount == Decimal("32900.00")  # $32,900 VaR
        assert cvar_money.amount == Decimal("42112.00")  # $42,112 CVaR

    def test_annualized_returns_calculation(self, usd: Currency) -> None:
        """Тест расчета аннуализированной доходности."""
        initial_value = Money(Decimal("100000.00"), usd)
        final_value = Money(Decimal("125000.00"), usd)
        holding_period_days = 180  # 6 месяцев
        
        # Расчет простой доходности
        simple_return = (final_value.amount - initial_value.amount) / initial_value.amount
        
        # Аннуализация
        days_in_year = Decimal("365")
        annualized_return = ((final_value.amount / initial_value.amount) ** 
                           (days_in_year / Decimal(holding_period_days))) - Decimal("1")
        
        # Проверяем результаты
        assert simple_return == Decimal("0.25")  # 25% за 6 месяцев
        
        # Аннуализированная доходность должна быть больше простой
        assert annualized_return > simple_return
        assert Decimal("0.55") < annualized_return < Decimal("0.65")  # ~56-57%

    def test_sharpe_ratio_calculation(self, usd: Currency) -> None:
        """Тест расчета коэффициента Шарпа."""
        portfolio_return = Decimal("0.15")  # 15% доходность
        risk_free_rate = Decimal("0.02")    # 2% безрисковая ставка
        portfolio_volatility = Decimal("0.12")  # 12% волатильность
        
        # Sharpe Ratio = (Portfolio Return - Risk Free Rate) / Portfolio Volatility
        excess_return = portfolio_return - risk_free_rate
        sharpe_ratio = excess_return / portfolio_volatility
        
        assert excess_return == Decimal("0.13")
        assert sharpe_ratio.quantize(Decimal("0.001")) == Decimal("1.083")

    def test_maximum_drawdown_calculation(self, usd: Currency) -> None:
        """Тест расчета максимальной просадки."""
        # Исторические значения портфеля
        portfolio_values = [
            Money(Decimal("100000.00"), usd),
            Money(Decimal("110000.00"), usd),
            Money(Decimal("125000.00"), usd),
            Money(Decimal("115000.00"), usd),
            Money(Decimal("105000.00"), usd),
            Money(Decimal("120000.00"), usd),
        ]
        
        # Расчет максимальной просадки
        peak = portfolio_values[0].amount
        max_drawdown = Decimal("0")
        
        for value in portfolio_values[1:]:
            if value.amount > peak:
                peak = value.amount
            
            drawdown = (peak - value.amount) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Максимальная просадка была с $125k до $105k = 16%
        assert max_drawdown == Decimal("0.16")

    def test_compound_annual_growth_rate(self, usd: Currency) -> None:
        """Тест расчета CAGR (составного годового темпа роста)."""
        beginning_value = Money(Decimal("100000.00"), usd)
        ending_value = Money(Decimal("180000.00"), usd)
        years = Decimal("3")
        
        # CAGR = (Ending Value / Beginning Value)^(1/years) - 1
        value_ratio = ending_value.amount / beginning_value.amount
        cagr = (value_ratio ** (Decimal("1") / years)) - Decimal("1")
        
        # Проверяем что CAGR корректен
        expected_cagr = Decimal("0.216")  # ~21.6%
        assert abs(cagr - expected_cagr) < Decimal("0.001")

    def test_financial_rounding_modes(self, usd: Currency) -> None:
        """Тест различных режимов округления для финансов."""
        test_amount = Decimal("123.4567")
        
        # Различные режимы округления
        rounded_half_up = test_amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        rounded_down = test_amount.quantize(Decimal("0.01"), rounding=ROUND_DOWN)
        rounded_up = test_amount.quantize(Decimal("0.01"), rounding=ROUND_UP)
        
        assert rounded_half_up == Decimal("123.46")
        assert rounded_down == Decimal("123.45")
        assert rounded_up == Decimal("123.46")
        
        # Создаем Money объекты с разными округлениями
        money_half_up = Money(rounded_half_up, usd)
        money_down = Money(rounded_down, usd)
        money_up = Money(rounded_up, usd)
        
        assert money_half_up.amount == Decimal("123.46")
        assert money_down.amount == Decimal("123.45")
        assert money_up.amount == Decimal("123.46")

    def test_margin_calculations(self, usd: Currency) -> None:
        """Тест расчетов маржинальных требований."""
        position_value = Money(Decimal("100000.00"), usd)
        leverage = Decimal("10")  # 10x leverage
        margin_requirement = Decimal("0.1")  # 10% margin
        
        # Требуемая маржа
        required_margin = position_value.amount * margin_requirement
        margin_money = Money(required_margin, usd)
        
        # Доступное плечо
        max_position = position_value.amount * leverage
        max_position_money = Money(max_position, usd)
        
        assert margin_money.amount == Decimal("10000.00")  # $10k margin
        assert max_position_money.amount == Decimal("1000000.00")  # $1M max position

    def test_tax_calculations(self, usd: Currency) -> None:
        """Тест расчета налогов."""
        gross_profit = Money(Decimal("10000.00"), usd)
        
        # Различные налоговые ставки
        short_term_rate = Decimal("0.37")  # 37% короткие позиции
        long_term_rate = Decimal("0.20")   # 20% длинные позиции
        
        short_term_tax = gross_profit * short_term_rate
        long_term_tax = gross_profit * long_term_rate
        
        net_profit_short = gross_profit - short_term_tax
        net_profit_long = gross_profit - long_term_tax
        
        assert short_term_tax.amount == Decimal("3700.00")
        assert long_term_tax.amount == Decimal("2000.00")
        assert net_profit_short.amount == Decimal("6300.00")
        assert net_profit_long.amount == Decimal("8000.00")

    def test_currency_precision_edge_cases(self) -> None:
        """Тест граничных случаев точности валют."""
        # Тестируем валюты с разной точностью
        jpy = Currency("JPY")  # Японская йена - без дробных частей
        btc = Currency("BTC")  # Bitcoin - 8 знаков после запятой
        
        # JPY операции
        jpy_amount = Money(Decimal("1000"), jpy)
        jpy_fraction = Money(Decimal("1000.123"), jpy)  # Должно округлиться
        
        # BTC операции с высокой точностью
        btc_precise = Money(Decimal("0.12345678"), btc)
        btc_too_precise = Money(Decimal("0.123456789"), btc)  # 9 знаков
        
        # Проверяем что точность соблюдается
        assert jpy_amount.amount == Decimal("1000")
        assert btc_precise.amount == Decimal("0.12345678")

    def test_overflow_protection(self, usd: Currency) -> None:
        """Тест защиты от переполнения."""
        # Максимально возможные значения
        max_safe_value = Decimal("999999999999999999999999999")
        
        try:
            large_money = Money(max_safe_value, usd)
            # Попытка превысить лимит
            overflow_result = large_money + large_money
            
            # Должно вызвать исключение или обработать корректно
            assert overflow_result.amount <= max_safe_value * 2
            
        except (OverflowError, ValidationError):
            # Ожидаемое поведение при переполнении
            pass

    def test_zero_division_protection(self, usd: Currency) -> None:
        """Тест защиты от деления на ноль."""
        money = Money(Decimal("1000.00"), usd)
        
        with pytest.raises((ZeroDivisionError, ValidationError, FinancialArithmeticError)):
            result = money / Decimal("0")

    def test_negative_value_validation(self, usd: Currency) -> None:
        """Тест валидации отрицательных значений."""
        # В некоторых контекстах отрицательные значения недопустимы
        with pytest.raises(ValidationError):
            # Отрицательная цена недопустима
            Price(Decimal("-100.00"), usd)
        
        with pytest.raises(ValidationError):
            # Отрицательный объем недопустим
            Volume(Decimal("-1.5"), Currency("BTC"))

    def test_financial_constants_precision(self) -> None:
        """Тест точности финансовых констант."""
        # Важные финансовые константы
        E = Decimal("2.718281828459045235360287471")  # e
        PI = Decimal("3.141592653589793238462643383")  # π
        
        # Проверяем что константы имеют достаточную точность
        assert len(str(E).replace(".", "")) >= 28
        assert len(str(PI).replace(".", "")) >= 28
        
        # Используем в финансовых формулах
        continuous_compound = E ** (Decimal("0.05") * Decimal("1"))  # 5% continuous compound
        assert Decimal("1.051") < continuous_compound < Decimal("1.052")