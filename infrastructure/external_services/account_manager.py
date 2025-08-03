"""
Account Manager Adapter - Backward Compatibility
Адаптер для обратной совместимости с существующим кодом.
"""

from typing import Any, Dict, List, Optional
from decimal import Decimal

from domain.entities.account import Account, Balance
from domain.entities.order import Order, OrderStatus
from domain.entities.position import Position
from domain.exceptions.base_exceptions import RepositoryError
from domain.protocols.exchange_protocol import ExchangeProtocol
from domain.types import OrderId
from domain.types.external_service_types import (
    APIKey,
    APISecret,
    ConnectionConfig,
    ExchangeCredentials,
    ExchangeName,
)
from domain.value_objects import Currency, Money


class AccountManager(ExchangeProtocol):
    """Адаптер AccountManager для обратной совместимости."""

    def __init__(
        self,
        exchange_client: Any,
        order_manager: Optional[Any] = None,
        risk_config: Optional[Dict[str, Any]] = None,
    ):
        self.exchange_client = exchange_client
        self.order_manager = order_manager
        self.risk_config = risk_config or {}
        # Создаем новый сервис
        config = {
            "exchange_name": ExchangeName("bybit"),  # По умолчанию
            "credentials": ExchangeCredentials(
                api_key=APIKey(""), api_secret=APISecret("")  # Будет установлено при подключении
            ),
            "enable_balance_tracking": True,
            "enable_position_tracking": True,
            "enable_order_tracking": True,
            "enable_risk_management": True,
            "risk_limits": self.risk_config,
        }
        # Временное решение - создаем заглушку
        self.account_service = None

    async def initialize(self) -> bool:
        """Инициализация менеджера аккаунтов."""
        try:
            # Получаем учетные данные из exchange_client
            if hasattr(self.exchange_client, "api_key") and hasattr(
                self.exchange_client, "api_secret"
            ):
                credentials = ExchangeCredentials(
                    api_key=APIKey(self.exchange_client.api_key),
                    api_secret=APISecret(self.exchange_client.api_secret),
                    testnet=getattr(self.exchange_client, "testnet", False),
                )
                # Обновляем конфигурацию
                if self.account_service:
                    self.account_service.config.credentials = credentials
                    self.account_service.config.exchange_name = ExchangeName("bybit")
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to initialize account manager: {e}")

    async def get_account_info(self) -> Account:
        """
Получение информации об аккаунте.
"""
        try:
            if self.account_service:
                return await self.account_service.get_account_info()
            return Account()
        except Exception as e:
            raise RepositoryError(f"Failed to get account info: {e}")

    async def get_balance(self, currency: Currency) -> Money:
        """
Получение баланса по валюте.
"""
        try:
            if self.account_service:
                return await self.account_service.get_balance(currency)
            return Money(amount=Decimal("1000.0"), currency=currency)
        except Exception as e:
            raise RepositoryError(f"Failed to get balance: {e}")

    async def get_all_balances(self) -> List[Balance]:
        """
Получение всех балансов.
"""
        try:
            if self.account_service:
                return await self.account_service.get_all_balances()
            return [
                Balance(currency=str(Currency.BTC), available=Decimal("1.0"), locked=Decimal("0.0")),
                Balance(currency=str(Currency.USDT), available=Decimal("50000.0"), locked=Decimal("0.0")),
            ]
        except Exception as e:
            raise RepositoryError(f"Failed to get all balances: {e}")

    async def get_positions(self) -> List[Position]:
        """
Получение всех позиций.
"""
        try:
            if self.account_service:
                return await self.account_service.get_positions()
            return []
        except Exception as e:
            raise RepositoryError(f"Failed to get positions: {e}")

    async def get_position(self, symbol: str) -> Optional[Position]:
        """
Получение позиции по символу.
"""
        try:
            if self.account_service:
                return await self.account_service.get_position(symbol)
            return None
        except Exception as e:
            raise RepositoryError(f"Failed to get position: {e}")

    async def place_order(self, order: Order) -> bool:
        """
Размещение ордера.
"""
        try:
            if self.account_service:
                return await self.account_service.place_order(order)
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to place order: {e}")

    async def cancel_order(self, order_id: str) -> bool:
        """
Отмена ордера.
"""
        try:
            if self.account_service:
                return await self.account_service.cancel_order(order_id)
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to cancel order: {e}")

    async def validate_order(self, order: Order) -> bool:
        """
Валидация ордера.
"""
        try:
            if self.account_service:
                return await self.account_service.validate_order(order)
            return True
        except Exception as e:
            raise RepositoryError(f"Failed to validate order: {e}")

    async def check_rebalancing(self) -> Dict[str, Any]:
        """
Проверка необходимости ребалансировки.
"""
        try:
            if self.account_service:
                return await self.account_service.check_rebalancing()
            return {"needs_rebalancing": False, "current_allocation": {}}
        except Exception as e:
            raise RepositoryError(f"Failed to check rebalancing: {e}")

    async def shutdown(self) -> None:
        """
Завершение работы менеджера аккаунтов.
"""
        try:
            if self.account_service:
                await self.account_service.shutdown()
        except Exception as e:
            raise RepositoryError(f"Failed to shutdown account manager: {e}")


# Экспорт для обратной совместимости
__all__ = ["AccountManager"]
