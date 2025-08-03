# -*- coding: utf-8 -*-
"""
Валидаторы для торговых данных.
"""

import logging
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Tuple

from domain.entities.trading import OrderSide, OrderType


class TradingDataValidator:
    """Валидатор торговых данных."""

    def __init__(self) -> None:
        # Регулярные выражения для валидации
        self.symbol_pattern = re.compile(r"^[A-Z0-9]{2,20}$")
        self.currency_pattern = re.compile(r"^[A-Z]{3,10}$")
        self.email_pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
        
        # Лимиты
        self.min_quantity = Decimal("0.000001")
        self.max_quantity = Decimal("1000000")
        self.min_price = Decimal("0.000001")
        self.max_price = Decimal("1000000")

    def validate_order_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация данных ордера."""
        errors: List[str] = []
        
        # Обязательные поля
        required_fields = ["trading_pair", "side", "order_type", "quantity"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Валидация отдельных полей
        errors.extend(self._validate_trading_pair(data.get("trading_pair")))
        errors.extend(self._validate_order_side(data.get("side")))
        errors.extend(self._validate_order_type(data.get("order_type")))
        errors.extend(self._validate_quantity(data.get("quantity")))
        
        # Валидация цены (обязательна для лимитных ордеров)
        if data.get("order_type") in ["limit", "stop_limit", "take_profit"]:
            if "price" not in data:
                errors.append("Price is required for limit orders")
            else:
                errors.extend(self._validate_price(data.get("price")))
        
        # Валидация stop_price (для стоп-ордеров)
        if data.get("order_type") in ["stop", "stop_limit"]:
            if "stop_price" not in data:
                errors.append("Stop price is required for stop orders")
            else:
                errors.extend(self._validate_price(data.get("stop_price")))
        
        # Валидация time_in_force
        if "time_in_force" in data:
            errors.extend(self._validate_time_in_force(data["time_in_force"]))
        
        return len(errors) == 0, errors

    def validate_position_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация данных позиции."""
        errors: List[str] = []
        
        # Обязательные поля
        required_fields = ["symbol", "side", "quantity", "entry_price"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Валидация отдельных полей
        symbol = data.get("symbol")
        if symbol is not None:
            errors.extend(self._validate_symbol(symbol))
        errors.extend(self._validate_order_side(data.get("side")))
        errors.extend(self._validate_quantity(data.get("quantity")))
        errors.extend(self._validate_price(data.get("entry_price")))
        
        # Валидация текущей цены
        if "current_price" in data:
            errors.extend(self._validate_price(data["current_price"]))
        
        # Валидация stop_loss и take_profit
        if "stop_loss" in data and data["stop_loss"]:
            errors.extend(self._validate_price(data["stop_loss"]))
        
        if "take_profit" in data and data["take_profit"]:
            errors.extend(self._validate_price(data["take_profit"]))
        
        return len(errors) == 0, errors

    def validate_trading_pair_data(
        self, data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Валидация данных торговой пары."""
        errors: List[str] = []
        
        # Обязательные поля
        required_fields = ["symbol", "base_currency", "quote_currency"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Валидация отдельных полей
        symbol = data.get("symbol")
        if symbol is not None:
            errors.extend(self._validate_symbol(symbol))
        
        base_currency = data.get("base_currency")
        if base_currency is not None:
            errors.extend(self._validate_currency(base_currency))
        
        quote_currency = data.get("quote_currency")
        if quote_currency is not None:
            errors.extend(self._validate_currency(quote_currency))
        
        # Валидация точности
        if "price_precision" in data:
            errors.extend(self._validate_precision(data["price_precision"]))
        
        if "quantity_precision" in data:
            errors.extend(self._validate_precision(data["quantity_precision"]))
        
        # Валидация лимитов
        if "min_quantity" in data:
            errors.extend(self._validate_quantity(data["min_quantity"]))
        
        if "max_quantity" in data:
            errors.extend(self._validate_quantity(data["max_quantity"]))
        
        if "min_price" in data:
            errors.extend(self._validate_price(data["min_price"]))
        
        if "max_price" in data:
            errors.extend(self._validate_price(data["max_price"]))
        
        return len(errors) == 0, errors

    def validate_account_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Валидация данных аккаунта."""
        errors: List[str] = []
        
        # Обязательные поля
        required_fields = ["name", "email"]
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Валидация отдельных полей
        name = data.get("name")
        if name is not None:
            errors.extend(self._validate_name(name))
        
        email = data.get("email")
        if email is not None:
            errors.extend(self._validate_email(email))
        
        # Валидация комиссии
        if "commission" in data:
            errors.extend(self._validate_commission(data["commission"]))
        
        # Валидация кредитного плеча
        if "leverage" in data:
            errors.extend(self._validate_leverage(data["leverage"]))
        
        return len(errors) == 0, errors

    def _validate_trading_pair(self, trading_pair: Any) -> List[str]:
        """Валидация торговой пары."""
        errors = []
        if not trading_pair:
            errors.append("Trading pair cannot be empty")
        elif not isinstance(trading_pair, str):
            errors.append("Trading pair must be a string")
        elif not self.symbol_pattern.match(trading_pair):
            errors.append("Trading pair must be 2-20 uppercase alphanumeric characters")
        return errors

    def _validate_symbol(self, symbol: str) -> List[str]:
        """Валидация символа."""
        errors: List[str] = []
        if not symbol:
            errors.append("Symbol cannot be empty")
        elif not isinstance(symbol, str):
            errors.append("Symbol must be a string")
        elif not self.symbol_pattern.match(symbol):
            errors.append("Symbol must be 2-20 uppercase alphanumeric characters")
        return errors

    def _validate_currency(self, currency: str) -> List[str]:
        """Валидация валюты."""
        errors: List[str] = []
        if not currency:
            errors.append("Currency cannot be empty")
        elif not isinstance(currency, str):
            errors.append("Currency must be a string")
        elif not self.currency_pattern.match(currency):
            errors.append("Currency must be 3-10 uppercase letters")
        return errors

    def _validate_order_side(self, side: Any) -> List[str]:
        """Валидация стороны ордера."""
        errors: List[str] = []
        if not side:
            errors.append("Order side cannot be empty")
            return errors  # type: ignore[unreachable]
        else:
            try:
                if isinstance(side, str):
                    OrderSide(side)
                elif isinstance(side, OrderSide):
                    pass
                else:
                    errors.append("Invalid order side")
            except ValueError:
                errors.append("Invalid order side value")
            return errors

    def _validate_order_type(self, order_type: Any) -> List[str]:
        """Валидация типа ордера."""
        errors: List[str] = []
        if not order_type:
            errors.append("Order type cannot be empty")
            return errors  # type: ignore[unreachable]
        else:
            try:
                if isinstance(order_type, str):
                    OrderType(order_type)
                elif isinstance(order_type, OrderType):
                    pass
                else:
                    errors.append("Invalid order type")
            except ValueError:
                errors.append("Invalid order type value")
            return errors

    def _validate_quantity(self, quantity: Any) -> List[str]:
        """Валидация количества."""
        errors: List[str] = []
        if quantity is None:
            errors.append("Quantity cannot be null")
            return errors
        else:
            try:
                if isinstance(quantity, str):
                    qty = Decimal(quantity)
                elif isinstance(quantity, (int, float, Decimal)):
                    qty = Decimal(str(quantity))
                else:
                    errors.append("Quantity must be a number")
                    return errors
                if qty <= 0:
                    errors.append("Quantity must be positive")
                elif qty < self.min_quantity:
                    errors.append(f"Quantity must be at least {self.min_quantity}")
                elif qty > self.max_quantity:
                    errors.append(f"Quantity must be at most {self.max_quantity}")
            except (ValueError, InvalidOperation):
                errors.append("Invalid quantity format")
            return errors

    def _validate_price(self, price: Any) -> List[str]:
        """Валидация цены."""
        errors: List[str] = []
        if price is None:
            errors.append("Price cannot be null")
            return errors
        else:
            try:
                if isinstance(price, str):
                    prc = Decimal(price)
                elif isinstance(price, (int, float, Decimal)):
                    prc = Decimal(str(price))
                else:
                    errors.append("Price must be a number")
                    return errors
                if prc <= 0:
                    errors.append("Price must be positive")
                elif prc < self.min_price:
                    errors.append(f"Price must be at least {self.min_price}")
                elif prc > self.max_price:
                    errors.append(f"Price must be at most {self.max_price}")
            except (ValueError, InvalidOperation):
                errors.append("Invalid price format")
            return errors

    def _validate_time_in_force(self, time_in_force: str) -> List[str]:
        """Валидация времени действия ордера."""
        errors = []
        valid_values = ["GTC", "IOC", "FOK"]
        if time_in_force not in valid_values:
            errors.append(f"Time in force must be one of: {valid_values}")
        return errors

    def _validate_leverage(self, leverage: Any) -> List[str]:
        """Валидация кредитного плеча."""
        errors: List[str] = []
        if leverage is None:
            errors.append("Leverage cannot be null")
            return errors
        try:
            if isinstance(leverage, str):
                lev = Decimal(leverage)
            elif isinstance(leverage, (int, float, Decimal)):
                lev = Decimal(str(leverage))
            else:
                errors.append("Leverage must be a number")
                return errors
            if lev <= 0:
                errors.append("Leverage must be positive")
            elif lev > 100:
                errors.append("Leverage cannot exceed 100")
        except (ValueError, InvalidOperation):
            errors.append("Invalid leverage format")
        return errors

    def _validate_precision(self, precision: int) -> List[str]:
        """Валидация точности."""
        errors: List[str] = []
        if not isinstance(precision, int):
            errors.append("Precision must be an integer")
            return errors  # type: ignore[unreachable]
        if precision < 0:
            errors.append("Precision cannot be negative")
            return errors  # type: ignore[unreachable]
        elif precision > 20:
            errors.append("Precision cannot exceed 20")
            return errors  # type: ignore[unreachable]
        return errors

    def _validate_name(self, name: str) -> List[str]:
        """Валидация имени."""
        errors: List[str] = []
        if not name:
            errors.append("Name cannot be empty")
        elif not isinstance(name, str):
            errors.append("Name must be a string")
        elif len(name) < 1:
            errors.append("Name must be at least 1 character")
        elif len(name) > 100:
            errors.append("Name must be at most 100 characters")
        return errors

    def _validate_email(self, email: str) -> List[str]:
        """Валидация email."""
        errors: List[str] = []
        if not email:
            errors.append("Email cannot be empty")
        elif not isinstance(email, str):
            errors.append("Email must be a string")
        elif not self.email_pattern.match(email):
            errors.append("Invalid email format")
        return errors

    def _validate_commission(self, commission: Any) -> List[str]:
        """Валидация комиссии."""
        errors: List[str] = []
        if commission is None:
            errors.append("Commission cannot be null")
            return errors
        try:
            if isinstance(commission, str):
                comm = Decimal(commission)
            elif isinstance(commission, (int, float, Decimal)):
                comm = Decimal(str(commission))
            else:
                errors.append("Commission must be a number")
                return errors
            if comm < 0:
                errors.append("Commission cannot be negative")
                return errors
            elif comm > 1:
                errors.append("Commission cannot exceed 100%")
                return errors
        except (ValueError, InvalidOperation):
            errors.append("Invalid commission format")
        return errors


class TradingBusinessRuleValidator:
    """Валидатор бизнес-правил торговли."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    def validate_order_business_rules(
        self, order_data: Dict[str, Any], account_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Валидация бизнес-правил для ордера."""
        errors: list[str] = []
        # Проверка достаточности средств
        if "quantity" in order_data and "price" in order_data and order_data["price"]:
            balance_errors = self._validate_sufficient_balance(order_data, account_data)
            errors.extend(balance_errors)
        # Проверка минимального размера ордера
        if "quantity" in order_data:
            min_size_errors = self._validate_minimum_order_size(order_data)
            errors.extend(min_size_errors)
        # Проверка максимального размера ордера
        if "quantity" in order_data:
            max_size_errors = self._validate_maximum_order_size(order_data)
            errors.extend(max_size_errors)
        # Проверка лимитов на количество ордеров
        order_count_errors = self._validate_order_count_limits(account_data)
        errors.extend(order_count_errors)
        return len(errors) == 0, errors

    def validate_position_business_rules(
        self, position_data: Dict[str, Any], account_data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Валидация бизнес-правил для позиции."""
        errors: List[str] = []
        # Проверка лимитов на позиции
        position_limit_errors = self._validate_position_limits(
            position_data, account_data
        )
        errors.extend(position_limit_errors)
        # Проверка рисков позиции
        risk_errors = self._validate_position_risk(position_data, account_data)
        errors.extend(risk_errors)
        return len(errors) == 0, errors

    def _validate_sufficient_balance(
        self, order_data: Dict[str, Any], account_data: Dict[str, Any]
    ) -> List[str]:
        """Проверка достаточности средств."""
        errors = []
        try:
            quantity = Decimal(str(order_data["quantity"]))
            price = Decimal(str(order_data["price"]))
            required_amount = quantity * price
            
            # Получаем баланс из данных аккаунта
            balance = Decimal(str(account_data.get("balance", 0)))
            
            if required_amount > balance:
                errors.append(
                    f"Insufficient balance. Required: {required_amount}, Available: {balance}"
                )
        except (ValueError, KeyError, TypeError) as e:
            self.logger.warning(f"Error validating balance: {e}")
            errors.append("Error validating balance")
        return errors

    def _validate_minimum_order_size(self, order_data: Dict[str, Any]) -> List[str]:
        """Проверка минимального размера ордера."""
        errors = []
        try:
            quantity = Decimal(str(order_data["quantity"]))
            min_size = Decimal("0.001")  # Минимальный размер ордера
            
            if quantity < min_size:
                errors.append(f"Order size must be at least {min_size}")
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error validating minimum order size: {e}")
            errors.append("Error validating minimum order size")
        return errors

    def _validate_maximum_order_size(self, order_data: Dict[str, Any]) -> List[str]:
        """Проверка максимального размера ордера."""
        errors = []
        try:
            quantity = Decimal(str(order_data["quantity"]))
            max_size = Decimal("1000000")  # Максимальный размер ордера
            
            if quantity > max_size:
                errors.append(f"Order size cannot exceed {max_size}")
        except (ValueError, KeyError) as e:
            self.logger.warning(f"Error validating maximum order size: {e}")
            errors.append("Error validating maximum order size")
        return errors

    def _validate_order_count_limits(self, account_data: Dict[str, Any]) -> List[str]:
        """Проверка лимитов на количество ордеров."""
        errors: List[str] = []
        # Здесь можно добавить логику проверки лимитов на количество ордеров
        return errors

    def _validate_position_limits(
        self, position_data: Dict[str, Any], account_data: Dict[str, Any]
    ) -> List[str]:
        """Проверка лимитов на позиции."""
        errors: List[str] = []
        # Здесь можно добавить логику проверки лимитов на позиции
        return errors

    def _validate_position_risk(
        self, position_data: Dict[str, Any], account_data: Dict[str, Any]
    ) -> List[str]:
        """Проверка рисков позиции."""
        errors: List[str] = []
        # Здесь можно добавить логику проверки рисков позиции
        return errors
