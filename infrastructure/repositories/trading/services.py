"""
Сервисы для торгового репозитория.
"""

import logging
from typing import Any, Dict, List, Optional

from domain.entities.account import Account
from domain.entities.order import Order
from domain.entities.position import Position


class TradingRepositoryServices:
    """Сервисы для торгового репозитория."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

    async def calculate_order_metrics(self, order: Order) -> Dict[str, Any]:
        """Расчет метрик ордера."""
        try:
            quantity = float(order.quantity)
            price = float(order.price.amount) if order.price else 0
            value = quantity * price if price > 0 else 0
            return {
                "order_id": str(order.id),
                "symbol": str(order.trading_pair),
                "quantity": quantity,
                "price": price,
                "value": value,
                "side": order.side.value,
                "type": order.order_type.value,
                "status": order.status.value,
                "created_at": str(order.created_at),
                "updated_at": str(order.updated_at) if order.updated_at else None,
            }
        except Exception as e:
            self.logger.error(f"Error calculating order metrics: {e}")
            return {"error": str(e)}

    async def calculate_position_metrics(self, position: Position) -> Dict[str, Any]:
        """Расчет метрик позиции."""
        try:
            quantity = float(position.volume.to_decimal())
            avg_price = float(position.entry_price.amount)
            unrealized_pnl = (
                float(position.unrealized_pnl.amount) if position.unrealized_pnl else 0
            )
            realized_pnl = float(position.realized_pnl.amount) if position.realized_pnl else 0
            return {
                "position_id": str(position.id),
                "symbol": str(position.trading_pair),
                "quantity": quantity,
                "average_price": avg_price,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": realized_pnl,
                "total_pnl": unrealized_pnl + realized_pnl,
                "side": position.side.value,
                "created_at": str(position.created_at),
                "updated_at": str(position.updated_at) if position.updated_at else None,
            }
        except Exception as e:
            self.logger.error(f"Error calculating position metrics: {e}")
            return {"error": str(e)}

    async def calculate_account_metrics(self, account: Account) -> Dict[str, Any]:
        """Расчет метрик аккаунта."""
        try:
            total_balance = sum(float(balance.total) for balance in account.balances)
            available_balance = sum(
                float(balance.total)
                for balance in account.balances
                if balance.currency in ["USDT", "USD", "BTC", "ETH"]
            )
            return {
                "account_id": account.id,
                "total_balance": total_balance,
                "available_balance": available_balance,
                "currency": "USDT",
                "created_at": account.created_at.isoformat(),
                "updated_at": (
                    account.updated_at.isoformat() if account.updated_at else None
                ),
            }
        except Exception as e:
            self.logger.error(f"Error calculating account metrics: {e}")
            return {"error": str(e)}

    async def validate_order_data(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация данных ордера."""
        errors = []
        required_fields = ["trading_pair", "side", "order_type", "quantity"]
        for field in required_fields:
            if field not in order_data:
                errors.append(f"Missing required field: {field}")
        if "quantity" in order_data:
            try:
                quantity = float(order_data["quantity"])
                if quantity <= 0:
                    errors.append("Quantity must be positive")
            except (ValueError, TypeError):
                errors.append("Invalid quantity format")
        if "price" in order_data and order_data["price"] is not None:
            try:
                price = float(order_data["price"])
                if price <= 0:
                    errors.append("Price must be positive")
            except (ValueError, TypeError):
                errors.append("Invalid price format")
        return {"valid": len(errors) == 0, "errors": errors}

    async def validate_position_data(
        self, position_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Валидация данных позиции."""
        errors = []
        required_fields = ["trading_pair", "side", "quantity"]
        for field in required_fields:
            if field not in position_data:
                errors.append(f"Missing required field: {field}")
        if "quantity" in position_data:
            try:
                quantity = float(position_data["quantity"])
                if quantity == 0:
                    errors.append("Quantity cannot be zero")
            except (ValueError, TypeError):
                errors.append("Invalid quantity format")
        return {"valid": len(errors) == 0, "errors": errors}

    async def format_order_for_storage(self, order: Order) -> Dict[str, Any]:
        """Форматирование ордера для хранения."""
        trading_pair_id = order.trading_pair.symbol if hasattr(order.trading_pair, 'symbol') else str(order.trading_pair)
        return {
            "id": str(order.id),
            "trading_pair_id": trading_pair_id,
            "portfolio_id": str(order.portfolio_id),
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": str(order.quantity),
            "price": str(order.price.amount) if order.price else None,
            "status": order.status.value,
            "created_at": str(order.created_at),
            "updated_at": str(order.updated_at) if order.updated_at else None,
        }

    async def format_position_for_storage(self, position: Position) -> Dict[str, Any]:
        """Форматирование позиции для хранения."""
        trading_pair_id = position.trading_pair.symbol if hasattr(position.trading_pair, 'symbol') else str(position.trading_pair)
        return {
            "id": str(position.id),
            "trading_pair_id": trading_pair_id,
            "portfolio_id": str(position.portfolio_id),
            "side": position.side.value,
            "quantity": str(position.volume.to_decimal()),
            "average_price": str(position.entry_price.amount),
            "unrealized_pnl": (
                str(position.unrealized_pnl.amount) if position.unrealized_pnl else None
            ),
            "realized_pnl": (
                str(position.realized_pnl.amount) if position.realized_pnl else None
            ),
            "created_at": str(position.created_at),
            "updated_at": str(position.updated_at) if position.updated_at else None,
        }
