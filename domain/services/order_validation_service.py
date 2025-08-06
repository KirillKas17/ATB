"""
Сервис валидации ордеров.
Переносит бизнес-логику валидации из конфигурации в domain слой.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Protocol

from shared.config import TradingConfig, RiskConfig


class OrderValidationService(Protocol):
    """Протокол для сервиса валидации ордеров."""
    
    def validate_order_size(self, exchange: str, symbol: str, quantity: Decimal) -> bool:
        """Валидация размера ордера."""
        ...
    
    def validate_position_size(self, account_balance: Decimal, price: Decimal, 
                             quantity: Decimal, confidence: Decimal) -> bool:
        """Валидация размера позиции с учетом риска."""
        ...
    
    def calculate_position_size(self, account_balance: Decimal, price: Decimal, 
                              confidence: Decimal) -> Decimal:
        """Расчет размера позиции с учетом риска."""
        ...


class DefaultOrderValidationService:
    """Реализация сервиса валидации ордеров."""
    
    def __init__(self, trading_config: TradingConfig, risk_config: RiskConfig):
        self.trading_config = trading_config
        self.risk_config = risk_config
    
    def validate_order_size(self, exchange: str, symbol: str, quantity: Decimal) -> bool:
        """Валидация размера ордера согласно требованиям биржи."""
        min_size = self.trading_config.get_min_order_size(exchange, symbol)
        return quantity >= min_size
    
    def validate_position_size(self, account_balance: Decimal, price: Decimal, 
                             quantity: Decimal, confidence: Decimal = Decimal("1.0")) -> bool:
        """Валидация размера позиции с учетом риска."""
        max_quantity = self.calculate_position_size(account_balance, price, confidence)
        return quantity <= max_quantity
    
    def calculate_position_size(self, account_balance: Decimal, price: Decimal, 
                              confidence: Decimal = Decimal("1.0")) -> Decimal:
        """Расчет размера позиции с учетом риска и уверенности."""
        risk_amount = self.risk_config.get_position_size_risk_amount(account_balance, confidence)
        return risk_amount / price if price > 0 else Decimal("0")


class RiskAdjustedOrderValidationService(DefaultOrderValidationService):
    """Расширенный сервис валидации с учетом волатильности."""
    
    def calculate_position_size(self, account_balance: Decimal, price: Decimal, 
                              confidence: Decimal = Decimal("1.0"), 
                              current_volatility: Decimal = Decimal("0")) -> Decimal:
        """Расчет размера позиции с учетом волатильности."""
        # Получаем коэффициент корректировки на основе волатильности
        volatility_factor = self.risk_config.get_volatility_adjustment_factor(current_volatility)
        
        # Корректируем риск с учетом волатильности
        adjusted_risk_amount = self.risk_config.get_position_size_risk_amount(
            account_balance, confidence
        ) * volatility_factor
        
        return adjusted_risk_amount / price if price > 0 else Decimal("0")
    
    def validate_position_size(self, account_balance: Decimal, price: Decimal, 
                             quantity: Decimal, confidence: Decimal = Decimal("1.0"),
                             current_volatility: Decimal = Decimal("0")) -> bool:
        """Валидация размера позиции с учетом волатильности."""
        max_quantity = self.calculate_position_size(
            account_balance, price, confidence, current_volatility
        )
        return quantity <= max_quantity
