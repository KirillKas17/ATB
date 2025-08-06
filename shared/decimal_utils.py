"""
Утилиты для работы с Decimal в торговых операциях.
Обеспечивает точность финансовых расчетов.
"""

from decimal import Decimal, getcontext, ROUND_HALF_UP, ROUND_DOWN, ROUND_UP
from typing import Union, Optional
import pandas as pd

# Устанавливаем точность для финансовых операций
getcontext().prec = 28  # 28 знаков точности


class TradingDecimal:
    """Утилиты для работы с Decimal в торговых операциях"""
    
    # Стандартные точности для разных активов
    CRYPTO_PRECISION = 8  # BTC, ETH и т.д.
    FOREX_PRECISION = 5   # USD, EUR и т.д.
    STOCK_PRECISION = 2   # Акции
    
    @staticmethod
    def to_decimal(value: Union[int, float, str, Decimal], precision: Optional[int] = None) -> Decimal:
        """
        Безопасное преобразование в Decimal с контролем точности.
        
        Args:
            value: Значение для преобразования
            precision: Количество знаков после запятой (по умолчанию полная точность)
            
        Returns:
            Decimal: Точное десятичное число
        """
        if isinstance(value, Decimal):
            result = value
        elif isinstance(value, str):
            result = Decimal(value)
        elif isinstance(value, (int, float)):
            # Для float преобразуем через строку для избежания проблем с точностью
            result = Decimal(str(value))
        else:
            raise ValueError(f"Неподдерживаемый тип для преобразования в Decimal: {type(value)}")
            
        # Применяем точность если указана
        if precision is not None:
            result = result.quantize(Decimal('0.1') ** precision, rounding=ROUND_HALF_UP)
            
        return result
    
    @staticmethod
    def safe_divide(dividend: Union[Decimal, float, int], 
                   divisor: Union[Decimal, float, int], 
                   default: Decimal = Decimal('0')) -> Decimal:
        """
        Безопасное деление с защитой от деления на ноль.
        
        Args:
            dividend: Делимое
            divisor: Делитель
            default: Значение по умолчанию при делении на ноль
            
        Returns:
            Decimal: Результат деления или значение по умолчанию
        """
        dividend_dec = TradingDecimal.to_decimal(dividend)
        divisor_dec = TradingDecimal.to_decimal(divisor)
        
        if divisor_dec == 0:
            return default
            
        return dividend_dec / divisor_dec
    
    @staticmethod
    def calculate_percentage(value: Union[Decimal, float, int], 
                           percentage: Union[Decimal, float, int]) -> Decimal:
        """
        Точный расчет процента от значения.
        
        Args:
            value: Базовое значение
            percentage: Процент (например, 2.5 для 2.5%)
            
        Returns:
            Decimal: Значение процента
        """
        value_dec = TradingDecimal.to_decimal(value)
        percentage_dec = TradingDecimal.to_decimal(percentage)
        
        return value_dec * percentage_dec / Decimal('100')
    
    @staticmethod
    def calculate_stop_loss(entry_price: Union[Decimal, float, int],
                          direction: str,
                          stop_percentage: Union[Decimal, float, int]) -> Decimal:
        """
        Точный расчет стоп-лосса.
        
        Args:
            entry_price: Цена входа
            direction: Направление ("long", "short", "buy", "sell")
            stop_percentage: Процент стопа (например, 2.0 для 2%)
            
        Returns:
            Decimal: Цена стоп-лосса
        """
        entry_dec = TradingDecimal.to_decimal(entry_price)
        stop_pct_dec = TradingDecimal.to_decimal(stop_percentage)
        
        stop_amount = TradingDecimal.calculate_percentage(entry_dec, stop_pct_dec)
        
        if direction.lower() in ["long", "buy"]:
            return entry_dec - stop_amount
        elif direction.lower() in ["short", "sell"]:
            return entry_dec + stop_amount
        else:
            raise ValueError(f"Неподдерживаемое направление: {direction}")
    
    @staticmethod
    def calculate_take_profit(entry_price: Union[Decimal, float, int],
                            direction: str,
                            profit_percentage: Union[Decimal, float, int]) -> Decimal:
        """
        Точный расчет тейк-профита.
        
        Args:
            entry_price: Цена входа
            direction: Направление ("long", "short", "buy", "sell")
            profit_percentage: Процент профита (например, 5.0 для 5%)
            
        Returns:
            Decimal: Цена тейк-профита
        """
        entry_dec = TradingDecimal.to_decimal(entry_price)
        profit_pct_dec = TradingDecimal.to_decimal(profit_percentage)
        
        profit_amount = TradingDecimal.calculate_percentage(entry_dec, profit_pct_dec)
        
        if direction.lower() in ["long", "buy"]:
            return entry_dec + profit_amount
        elif direction.lower() in ["short", "sell"]:
            return entry_dec - profit_amount
        else:
            raise ValueError(f"Неподдерживаемое направление: {direction}")
    
    @staticmethod
    def calculate_position_size(account_balance: Union[Decimal, float, int],
                              risk_percentage: Union[Decimal, float, int],
                              entry_price: Union[Decimal, float, int],
                              stop_loss: Union[Decimal, float, int]) -> Decimal:
        """
        Точный расчет размера позиции на основе риска.
        
        Args:
            account_balance: Баланс счета
            risk_percentage: Процент риска от баланса (например, 1.0 для 1%)
            entry_price: Цена входа
            stop_loss: Цена стоп-лосса
            
        Returns:
            Decimal: Размер позиции
        """
        balance_dec = TradingDecimal.to_decimal(account_balance)
        risk_pct_dec = TradingDecimal.to_decimal(risk_percentage)
        entry_dec = TradingDecimal.to_decimal(entry_price)
        stop_dec = TradingDecimal.to_decimal(stop_loss)
        
        # Сумма риска в валюте счета
        risk_amount = TradingDecimal.calculate_percentage(balance_dec, risk_pct_dec)
        
        # Риск на единицу (расстояние до стопа)
        risk_per_unit = abs(entry_dec - stop_dec)
        
        # Размер позиции = Сумма риска / Риск на единицу
        if risk_per_unit == 0:
            return Decimal('0')
            
        return TradingDecimal.safe_divide(risk_amount, risk_per_unit)
    
    @staticmethod
    def calculate_pnl(entry_price: Union[Decimal, float, int],
                     exit_price: Union[Decimal, float, int],
                     position_size: Union[Decimal, float, int],
                     direction: str) -> Decimal:
        """
        Точный расчет PnL (прибыли/убытка).
        
        Args:
            entry_price: Цена входа
            exit_price: Цена выхода
            position_size: Размер позиции
            direction: Направление ("long", "short", "buy", "sell")
            
        Returns:
            Decimal: PnL (положительное значение = прибыль, отрицательное = убыток)
        """
        entry_dec = TradingDecimal.to_decimal(entry_price)
        exit_dec = TradingDecimal.to_decimal(exit_price)
        size_dec = TradingDecimal.to_decimal(position_size)
        
        if direction.lower() in ["long", "buy"]:
            # Для длинной позиции: (цена_выхода - цена_входа) * размер
            return (exit_dec - entry_dec) * size_dec
        elif direction.lower() in ["short", "sell"]:
            # Для короткой позиции: (цена_входа - цена_выхода) * размер
            return (entry_dec - exit_dec) * size_dec
        else:
            raise ValueError(f"Неподдерживаемое направление: {direction}")
    
    @staticmethod
    def round_to_tick_size(price: Union[Decimal, float, int], 
                          tick_size: Union[Decimal, float, int]) -> Decimal:
        """
        Округление цены до размера тика биржи.
        
        Args:
            price: Цена для округления
            tick_size: Размер тика (например, 0.01 для центов)
            
        Returns:
            Decimal: Округленная цена
        """
        price_dec = TradingDecimal.to_decimal(price)
        tick_dec = TradingDecimal.to_decimal(tick_size)
        
        if tick_dec == 0:
            return price_dec
            
        # Округляем до ближайшего тика
        return (price_dec / tick_dec).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_dec
    
    @staticmethod
    def validate_price_levels(entry_price: Union[Decimal, float, int],
                            stop_loss: Union[Decimal, float, int],
                            take_profit: Union[Decimal, float, int],
                            direction: str) -> bool:
        """
        Валидация логики уровней цен.
        
        Args:
            entry_price: Цена входа
            stop_loss: Стоп-лосс
            take_profit: Тейк-профит
            direction: Направление
            
        Returns:
            bool: True если логика корректна
        """
        entry_dec = TradingDecimal.to_decimal(entry_price)
        stop_dec = TradingDecimal.to_decimal(stop_loss)
        profit_dec = TradingDecimal.to_decimal(take_profit)
        
        if direction.lower() in ["long", "buy"]:
            # Для LONG: stop_loss < entry_price < take_profit
            return stop_dec < entry_dec < profit_dec
        elif direction.lower() in ["short", "sell"]:
            # Для SHORT: take_profit < entry_price < stop_loss
            return profit_dec < entry_dec < stop_dec
        else:
            return False


# Удобные функции для быстрого использования
def to_trading_decimal(value: Union[int, float, str, Decimal], precision: int = 8) -> Decimal:
    """Быстрое преобразование в торговый Decimal с точностью 8 знаков"""
    return TradingDecimal.to_decimal(value, precision)


def safe_percentage(value: Union[Decimal, float, int], 
                   percentage: Union[Decimal, float, int]) -> Decimal:
    """Быстрый расчет процента"""
    return TradingDecimal.calculate_percentage(value, percentage)


def calculate_risk_size(balance: Union[Decimal, float, int],
                       risk_pct: Union[Decimal, float, int],
                       entry: Union[Decimal, float, int],
                       stop: Union[Decimal, float, int]) -> Decimal:
    """Быстрый расчет размера позиции по риску"""
    return TradingDecimal.calculate_position_size(balance, risk_pct, entry, stop)


# Константы для общих значений
ZERO = Decimal('0')
ONE = Decimal('1')
HUNDRED = Decimal('100')

# Стандартные точности
CRYPTO_PRECISION = 8
FOREX_PRECISION = 5
STOCK_PRECISION = 2