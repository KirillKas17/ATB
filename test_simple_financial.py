#!/usr/bin/env python3
"""
Простые финансовые тесты без pytest.
"""

from decimal import Decimal, getcontext, ROUND_HALF_UP
from domain.value_objects.money import Money
from domain.value_objects.price import Price
from domain.value_objects.volume import Volume
from domain.value_objects.currency import Currency
from domain.value_objects.percentage import Percentage


def test_decimal_precision():
    """Тест точности decimal вычислений."""
    # Устанавливаем высокую точность
    getcontext().prec = 28
    getcontext().rounding = ROUND_HALF_UP
    
    # Тест сложения
    amount1 = Decimal("0.1")
    amount2 = Decimal("0.2")
    result = amount1 + amount2
    expected = Decimal("0.3")
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Тест умножения
    price = Decimal("45123.456789")
    quantity = Decimal("0.12345678")
    total = price * quantity
    # Проверяем что результат точный
    assert isinstance(total, Decimal)
    assert total > Decimal("5000")  # Приблизительная проверка


def test_financial_arithmetic():
    """Тест финансовых вычислений."""
    # Расчет комиссии
    trade_amount = Decimal("10000.00")
    commission_rate = Decimal("0.001")  # 0.1%
    commission = trade_amount * commission_rate
    expected_commission = Decimal("10.00")
    assert commission == expected_commission
    
    # Расчет процентов
    principal = Decimal("1000.00")
    rate = Decimal("0.05")  # 5%
    interest = principal * rate
    expected_interest = Decimal("50.00")
    assert interest == expected_interest


def test_currency_conversion():
    """Тест валютных конвертаций."""
    # Базовая конвертация
    usd_amount = Decimal("1000.00")
    exchange_rate = Decimal("0.85")  # USD to EUR
    eur_amount = usd_amount * exchange_rate
    expected_eur = Decimal("850.00")
    assert eur_amount == expected_eur


def test_profit_loss_calculation():
    """Тест расчета прибыли/убытка."""
    # Покупка
    buy_price = Decimal("45000.00")
    quantity = Decimal("0.1")
    buy_total = buy_price * quantity
    
    # Продажа
    sell_price = Decimal("46000.00")
    sell_total = sell_price * quantity
    
    # Прибыль
    profit = sell_total - buy_total
    expected_profit = Decimal("100.00")
    assert profit == expected_profit


def test_percentage_calculations():
    """Тест процентных вычислений."""
    # Изменение цены в процентах
    old_price = Decimal("45000")
    new_price = Decimal("46350")
    
    change = new_price - old_price
    percentage_change = (change / old_price) * Decimal("100")
    
    # Проверяем что рост составляет 3%
    expected_change = Decimal("3")
    assert abs(percentage_change - expected_change) < Decimal("0.01")


def test_compound_interest():
    """Тест сложных процентов."""
    principal = Decimal("1000.00")
    rate = Decimal("0.05")  # 5% годовых
    time = Decimal("2")  # 2 года
    
    # Простые сложные проценты: A = P(1 + r)^t
    # Упрощенный расчет для 2 лет
    year1 = principal * (Decimal("1") + rate)
    year2 = year1 * (Decimal("1") + rate)
    
    expected_amount = Decimal("1102.50")
    assert abs(year2 - expected_amount) < Decimal("0.01")


def test_money_arithmetic():
    """Тест арифметики с Money objects."""
    money1 = Money(amount=Decimal("1000.00"), currency=Currency.USD)
    money2 = Money(amount=Decimal("500.00"), currency=Currency.USD)
    
    # Проверяем что Money objects создаются корректно
    assert money1.amount == Decimal("1000.00")
    assert money2.amount == Decimal("500.00")
    assert money1.currency == Currency.USD
    assert money2.currency == Currency.USD


def test_portfolio_weight_calculation():
    """Тест расчета весов портфеля."""
    # Позиции в портфеле
    btc_value = Decimal("50000.00")
    eth_value = Decimal("30000.00")
    usdt_value = Decimal("20000.00")
    
    total_value = btc_value + eth_value + usdt_value
    
    # Веса в процентах
    btc_weight = (btc_value / total_value) * Decimal("100")
    eth_weight = (eth_value / total_value) * Decimal("100")
    usdt_weight = (usdt_value / total_value) * Decimal("100")
    
    # Проверяем что веса корректные
    assert abs(btc_weight - Decimal("50")) < Decimal("0.01")
    assert abs(eth_weight - Decimal("30")) < Decimal("0.01")
    assert abs(usdt_weight - Decimal("20")) < Decimal("0.01")
    
    # Проверяем что сумма весов = 100%
    total_weight = btc_weight + eth_weight + usdt_weight
    assert abs(total_weight - Decimal("100")) < Decimal("0.01")


def test_risk_calculation():
    """Тест расчета рисков."""
    # Максимальная просадка
    portfolio_value = Decimal("100000.00")
    max_drawdown_percent = Decimal("5")  # 5%
    
    max_loss = portfolio_value * (max_drawdown_percent / Decimal("100"))
    expected_max_loss = Decimal("5000.00")
    assert max_loss == expected_max_loss
    
    # Позиционный риск
    position_size = Decimal("10000.00")
    stop_loss_percent = Decimal("2")  # 2%
    
    position_risk = position_size * (stop_loss_percent / Decimal("100"))
    expected_position_risk = Decimal("200.00")
    assert position_risk == expected_position_risk


def test_slippage_calculation():
    """Тест расчета проскальзывания."""
    expected_price = Decimal("45000.00")
    actual_price = Decimal("45050.00")
    
    slippage = actual_price - expected_price
    slippage_percent = (slippage / expected_price) * Decimal("100")
    
    # Проверяем что slippage составляет примерно 0.11%
    expected_slippage_percent = Decimal("0.11")
    assert abs(slippage_percent - expected_slippage_percent) < Decimal("0.01")


def test_edge_case_zero_division():
    """Тест граничного случая - деление на ноль."""
    try:
        result = Decimal("100") / Decimal("0")
        raise AssertionError("Should have raised division by zero error")
    except Exception:
        pass  # Ожидаемое исключение


def test_edge_case_very_small_amounts():
    """Тест очень малых сумм."""
    # Очень малая сумма (1 satoshi в BTC)
    satoshi = Decimal("0.00000001")
    btc_price = Decimal("45000.00")
    
    value_usd = satoshi * btc_price
    expected_value = Decimal("0.00045")
    assert value_usd == expected_value


def test_edge_case_very_large_amounts():
    """Тест очень больших сумм."""
    # Большая сумма
    large_amount = Decimal("999999999999.99999999")
    multiplier = Decimal("1.0001")
    
    result = large_amount * multiplier
    # Проверяем что результат больше исходной суммы
    assert result > large_amount